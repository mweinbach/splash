#!/usr/bin/env python3
"""Strict R4 actual-AIR per-row symbolic SSA and metadata address audit.

This interpreter follows actual control flow and phi chronology, expands actual
outlined calls, and preserves ordered FP operation operands and attributes.
Loads have typed pointer provenance and addresses. It never reads GPU operands.
Only a finite integer-mask multiplicand or same-address += chunk boundary may
commute; each occurrence is counted.
No reassociation, general commutation, opcode census or SSA operand erasure exists.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import re
import sys
import build as source

HERE = Path(__file__).resolve().parent
FORMATS = source.FORMATS

def split(text):
    result = []; depth = 0; start = 0
    for i, c in enumerate(text):
        depth += c in '([{'; depth -= c in ')]}'
        if c == ',' and depth == 0: result.append(text[start:i].strip()); start = i + 1
    result.append(text[start:].strip()); return result

class Module:
    def __init__(self, path):
        self.path = Path(path); self.text = self.path.read_text(); self.functions = {}
        self.attributes = dict(re.findall(r'^attributes #(\d+) = (\{.*\})$', self.text, re.M))
        for header, name, params, content in re.findall(r'^(define[^\n]*?@([^ (]+)\(([^\n]*)\)[^\n]*\{)\n(.*?)^}', self.text, re.M | re.S):
            args = [re.findall(r'%\d+', p)[-1] for p in split(params)]
            blocks = {}; label = str(len(args)); blocks[label] = []
            for line in content.splitlines():
                line = line.split(';', 1)[0].strip()
                if not line: continue
                if blocks[label] and blocks[label][-1].startswith('switch ') and not blocks[label][-1].endswith(']'):
                    blocks[label][-1] += ' ' + line; continue
                if re.fullmatch(r'\d+:', line): label = line[:-1]; blocks[label] = []; continue
                blocks[label].append(line)
            self.functions[name] = (args, blocks, header)
    def find(self, part):
        matches = [n for n in self.functions if part in n]
        if len(matches) != 1: raise ValueError(f'unique actual function required: {part}: {matches}')
        return matches[0]
    def policy(self):
        pattern = r'"(?:unsafe-fp-math|denormal-fp-math|no-infs-fp-math|no-nans-fp-math|no-signed-zeros-fp-math|approx-func-fp-math)"="[^"]+"'
        policies = set()
        for _, blocks, header in self.functions.values():
            instructions = '\n'.join(line for lines in blocks.values() for line in lines)
            if not re.search(r'\b(?:fadd|fmul|fpext|fptrunc)\b|call.*@(?:air\.simd_sum|llvm\.fmuladd)', instructions): continue
            number = re.search(r'#(\d+) \{$', header)[1]
            policies.update(re.findall(pattern, self.attributes[number]))
        policies.update(re.findall(r'fast_math_(?:enable|disable)|denorms_(?:enable|disable)', self.text))
        return sorted(policies)
    def intrinsicMetadata(self):
        result = {}
        for name in ('air.convert.f.f32.s.i32', 'llvm.fmuladd.f32', 'air.simd_sum.f32'):
            matches = re.findall(r'^declare[^\n]*@' + re.escape(name) + r'\([^\n]*$', self.text, re.M)
            if len(matches) != 1: raise ValueError('unique actual FP intrinsic declaration required')
            declaration = matches[0]
            declaration = re.sub(r'#(\d+)', lambda m: self.attributes[m[1]], declaration)
            result[name] = declaration
        return result

class Bits:
    def __init__(self, bits): self.bits = tuple(bits)
    def __repr__(self): return 'Bits' + repr(self.bits)

class Executor:
    def __init__(self, module, params):
        self.module = module; self.params = params; self.memory = {}; self.loads = []; self.stores = []
        self.allocations = 0; self.steps = 0; self.maskCommutes = 0; self.chunkCommutes = 0; self.callFlags = set()
        self.reductions = []
    def value(self, token, values):
        token = token.strip()
        if token.startswith('%'): return values[token]
        if token == 'true': return True
        if token == 'false': return False
        if token == 'null': return ('ptr', 'Null', 0)
        if re.fullmatch(r'-?\d+', token): return int(token)
        if re.fullmatch(r'(?:0x[0-9A-Fa-f]+|-?\d+\.\d+e[+-]\d+)', token): return ('constant', token)
        raise ValueError('unsupported actual value: ' + token)
    def operand(self, text, values):
        return self.value(text.rsplit(' ', 1)[-1], values)
    def fp(self, operation, flags, operands):
        # Exact integer-to-F32 conversion of masks is always finite, never NaN.
        # This explicit local exception is the sole commutation accepted here.
        if operation in ('fmul', 'llvm.fmuladd.f32'):
            a, b = operands[:2]
            is_mask = lambda x: isinstance(x, tuple) and x[0] == 'finite-mask-convert'
            if is_mask(a) and not is_mask(b):
                operands = (b, a, *operands[2:]); self.maskCommutes += 1
        result = ('FP', operation, flags, *operands)
        if operation == 'air.simd_sum.f32': self.reductions.append(result)
        return result
    def load(self, typ, ptr):
        _, kind, offset = ptr; self.loads.append((kind, offset, typ))
        if kind == 'Params':
            index = offset // 4 if offset < 32 else 8 + (offset - 32) // 8
            return self.params[index]
        if kind == 'W':
            if typ != 'i8': raise ValueError('packed integer load type drift')
            return Bits(('W', offset, i) for i in range(8))
        if kind in ('S', 'B'): return ('coefficient-BF16', kind, offset)
        if kind == 'Input': return ('activation-BF16', offset)
        if ptr in self.memory: return self.memory[ptr]
        raise ValueError('unwritten actual SSA private memory load: ' + repr(ptr))
    def store(self, typ, ptr, value):
        # Explicit += boundary only: SSA must feed the exact prior value of
        # this same private result address to one fadd, with the current dot
        # as the other operand. Retain this chronology; record orientation.
        # Other fadds (XSUM/qdot fragments) are never commuted or reassociated.
        previous = self.memory.get(ptr)
        if typ == 'float' and ptr[1].startswith('private-') and previous is not None and isinstance(value, tuple) and value[:2] == ('FP', 'fadd'):
            a, b = value[3:]
            if a == previous: value = ('FP', 'chunk-fadd', value[2], previous, b)
            elif b == previous:
                value = ('FP', 'chunk-fadd', value[2], previous, a); self.chunkCommutes += 1
        self.memory[ptr] = value; self.stores.append((ptr, typ, value))
    def run(self, name, arguments):
        names, blocks, _ = self.module.functions[name]
        if len(names) != len(arguments): raise ValueError('outlined ABI argument mismatch')
        values = dict(zip(names, arguments)); block = str(len(names)); previous = None
        while True:
            lines = blocks[block]; pending = {}
            for line in lines:
                match = re.match(r'(%\d+) = phi (.*?) (\[.*)', line)
                if not match: continue
                choices = re.findall(r'\[ (.*?), %(\d+) \]', match[3])
                selected = [v for v, pred in choices if pred == previous]
                if len(selected) != 1: raise ValueError('actual phi predecessor not unique')
                pending[match[1]] = self.value(selected[0], values)
            values.update(pending)
            for line in lines:
                self.steps += 1
                if self.steps > 3000000: raise ValueError('bounded SSA instruction limit exceeded')
                target = None; instruction = line
                match = re.match(r'(%\d+) = (.*)', line)
                if match: target, instruction = match.groups()
                if instruction.startswith('phi '): continue
                if instruction.startswith('ret '):
                    return None if instruction == 'ret void' else self.operand(instruction[4:], values)
                if instruction.startswith('br '):
                    labels = re.findall(r'label %(\d+)', instruction)
                    if len(labels) == 1: next_block = labels[0]
                    else: next_block = labels[0 if self.operand(split(instruction[3:])[0], values) else 1]
                    previous, block = block, next_block; break
                if instruction.startswith('switch '):
                    head, cases = instruction[7:].split(' [', 1)
                    value_text, default = split(head); chosen = self.operand(value_text, values)
                    next_block = re.search(r'label %(\d+)', default)[1]
                    for value, label in re.findall(r'i\d+ (-?\d+), label %(\d+)', cases):
                        if chosen == int(value): next_block = label
                    previous, block = block, next_block; break
                if instruction.startswith('unreachable'): raise ValueError('actual unreachable path')
                if instruction.startswith('alloca '):
                    result = ('ptr', 'private-' + str(self.allocations), 0); self.allocations += 1
                elif instruction.startswith('getelementptr '):
                    parts = split(instruction.removeprefix('getelementptr ').removeprefix('inbounds '))
                    ptr = self.operand(parts[1], values); indices = [self.operand(p, values) for p in parts[2:]]
                    typ = parts[0]
                    if typ == '%struct.FlashAffineParams':
                        if indices[0] != 0: raise ValueError('parameter pointer stride drift')
                        i = indices[1]; delta = i * 4 if i < 8 else 32 + (i - 8) * 8
                    elif typ.startswith('['):
                        count = int(re.match(r'\[(\d+) x', typ)[1]); delta = indices[0] * count * 4 + indices[1] * 4
                    elif typ == '%"struct.metal::_atomic"': delta = 0
                    else: delta = indices[0] * {'i8': 1, 'bfloat': 2, 'float': 4, 'i32': 4}[typ]
                    result = ('ptr', ptr[1], ptr[2] + delta)
                elif instruction.startswith('load '):
                    parts = split(instruction[5:]); result = self.load(parts[0], self.operand(parts[1], values))
                elif instruction.startswith('store '):
                    parts = split(instruction[6:]); self.store(parts[0].split()[0], self.operand(parts[1], values), self.operand(parts[0], values)); continue
                elif instruction.startswith('extractelement '):
                    parts = split(instruction[15:]); result = self.operand(parts[0], values)[self.operand(parts[1], values)]
                elif instruction.startswith('bitcast '):
                    expr = instruction[8:].rsplit(' to ', 1)[0]; result = self.operand(expr, values)
                    # Finite-path dependency proof only. This does not claim
                    # GPU NaN/Inf/sticky qualification; those remain Root gates.
                    if not (isinstance(result, tuple) and result[0] == 'ptr'): result = 0
                elif instruction.startswith(('zext ', 'sext ', 'trunc ')):
                    op, rest = instruction.split(' ', 1); left, typ = rest.rsplit(' to ', 1)
                    value = self.operand(left, values); width = int(typ[1:])
                    if isinstance(value, Bits): result = Bits((value.bits + (False,) * width)[:width])
                    else: result = value & ((1 << width) - 1)
                elif instruction.startswith(('fpext ', 'fptrunc ')):
                    op, rest = instruction.split(' ', 1); left, typ = rest.rsplit(' to ', 1)
                    result = self.fp(op, left.split()[0] + '->' + typ, (self.operand(left, values),))
                elif re.match(r'f(?:add|mul|sub|div)\b', instruction):
                    match = re.match(r'(f\w+) (.*?) (?:float|bfloat) (.*)', instruction)
                    if not match: match = re.match(r'(f\w+) (float|bfloat) (.*)', instruction); op, _, args = match.groups(); flags = ''
                    else: op, flags, args = match.groups()
                    result = self.fp(op, flags, tuple(self.value(p, values) for p in split(args)))
                elif re.match(r'(?:tail )?call\b', instruction):
                    match = re.search(r'call (.*?) @([^ (]+)\((.*)\)(?: #\d+)?$', instruction)
                    if not match: raise ValueError('unsupported actual call: ' + instruction)
                    call_type, called, arg_text = match.groups(); args = [self.operand(p, values) for p in split(arg_text)] if arg_text else []
                    flags = 'fast' if call_type.startswith('fast ') else ''
                    if called.startswith('llvm.lifetime.'): continue
                    if called.startswith('llvm.memset.'):
                        if args[1] != 0: raise ValueError('nonzero actual result initialization')
                        ptr = args[0]
                        for off in range(0, args[2], 4): self.memory[('ptr', ptr[1], ptr[2] + off)] = ('constant', '0.000000e+00')
                        continue
                    if called in self.module.functions:
                        self.callFlags.add((called.split('mlx_qmv_f32xsum_v1')[-1], flags))
                        result = self.run(called, args)
                    elif called == 'air.convert.f.f32.s.i32':
                        if not isinstance(args[0], Bits) or any(x is not False for x in args[0].bits[16:]):
                            raise ValueError('integer mask conversion finite/range proof failed')
                        result = ('finite-mask-convert', tuple(args[0].bits))
                    elif called in ('llvm.fmuladd.f32', 'air.simd_sum.f32'): result = self.fp(called, flags, tuple(args))
                    elif called.startswith('air.atomic.global.or.'): continue
                    else: raise ValueError('unsupported actual external call: ' + called)
                elif instruction.startswith('icmp '):
                    pred, rest = instruction[5:].split(' ', 1); typ, args = rest.split(' ', 1); a, b = [self.value(p, values) for p in split(args)]
                    result = {'eq': lambda: a == b, 'ne': lambda: a != b, 'ult': lambda: a < b,
                              'ule': lambda: a <= b, 'ugt': lambda: a > b, 'uge': lambda: a >= b,
                              'sgt': lambda: a > b, 'slt': lambda: a < b}[pred]()
                elif instruction.startswith('select '):
                    parts = split(instruction[7:]); result = self.operand(parts[1 if self.operand(parts[0], values) else 2], values)
                elif re.match(r'(?:add|sub|mul|shl|lshr|ashr|and|or|xor|udiv|sdiv)\b', instruction):
                    match = re.match(r'(\w+) (?:(?:nuw|nsw|exact) )*(i\d+) (.*)', instruction)
                    op, typ, args = match.groups(); width = int(typ[1:]); a, b = [self.value(p, values) for p in split(args)]
                    if isinstance(a, Bits):
                        if not isinstance(b, int): raise ValueError('unsupported symbolic packed integer operator')
                        if op == 'and': result = Bits(x if (b >> i) & 1 else False for i, x in enumerate(a.bits))
                        elif op == 'shl': result = Bits(((False,) * b + a.bits)[:width])
                        else: raise ValueError('packed integer transform outside strict audit: ' + op)
                    else:
                        functions = {'add': lambda: a+b, 'sub': lambda: a-b, 'mul': lambda: a*b,
                                     'shl': lambda: a<<b, 'lshr': lambda: a>>b, 'ashr': lambda: a>>b,
                                     'and': lambda: a&b, 'or': lambda: a|b, 'xor': lambda: a^b,
                                     'udiv': lambda: a//b, 'sdiv': lambda: a//b}
                        result = functions[op]() & ((1 << width) - 1)
                else: raise ValueError('unsupported actual SSA instruction: ' + instruction)
                if target is not None: values[target] = result
            else: raise ValueError('actual basic block has no terminator')

def pointer(kind): return ('ptr', kind, 0)
def params(bits, group, k, n, padding=0):
    ws = k * bits // 8 + padding; ps = k // group * 2 + 2 * padding
    return [4, 1, k, n, 1, bits, group, 0, ws, 0, ps, 0]
def shipping(module, bits, group, k, n, row, lane, out_row, padding=0, tap=False):
    e = Executor(module, params(bits, group, k, n, padding))
    name = module.find(f'project_mathILt{bits}ELt{group}ELb1')
    args = [('ptr', 'Input', row*k*2), pointer('W'), pointer('S'), pointer('B'), pointer('Output'), pointer('Diagnostics')]
    if tap: args.append(pointer('Raw'))
    e.run(name, args + [pointer('Params'), out_row, 0, row, lane])
    return e
def candidate(module, bits, group, k, n, y, lane, out_row, padding=0, tap=False):
    e = Executor(module, params(bits, group, k, n, padding))
    name = module.find(f'project_pairILt{bits}ELt{group}EE')
    args = [pointer('Input'), pointer('W'), pointer('S'), pointer('B'), pointer('Output'), pointer('Diagnostics')]
    if tap: args.append(pointer('Raw'))
    e.run(name, args + [pointer('Params'), (out_row//8, y, 0), (out_row%8)//4, lane])
    return e
def outputs(e): return {ptr[2]: value for ptr, typ, value in e.stores if ptr[1] == 'Output' and typ == 'bfloat'}
def raw_outputs(e): return {ptr[2]//4*2: value for ptr, typ, value in e.stores if ptr[1] == 'Raw' and typ == 'float'}
def check_tap(e):
    expected = {offset: value[3] for offset, value in outputs(e).items()}
    if raw_outputs(e) != expected: raise ValueError('actual raw tap is not the same SSA BF16 conversion source')
def check_addresses(e, bits, group, k, n, padding):
    p = params(bits, group, k, n, padding)
    extents = {'Input': 4*k*2, 'W': (n-1)*p[8] + k*bits//8,
               'S': (n-1)*p[10] + k//group*2, 'B': (n-1)*p[10] + k//group*2}
    for kind, offset, typ in e.loads:
        if kind in extents:
            size = {'i8': 1, 'bfloat': 2}[typ]
            if offset < 0 or offset + size > extents[kind] or (typ == 'bfloat' and offset % 2):
                raise ValueError('typed actual pointer extent/alignment violation')

def mutationNegatives(native, candidate_module):
    results = {}
    old = [shipping(native, 4, 64, 2560, 10240, r, 0, 0) for r in (0, 1)]
    expected = [e.reductions[column] for column in range(4) for e in old]
    qdot_name = candidate_module.find('qdot_pairILt4ELt16EE')
    load_name = candidate_module.find('load_vectorIDF16bfLi16ELi4EE')
    main_name = candidate_module.find('project_pairILt4ELt64EE')
    for label in ('cross_row_input_pointer_same_opcode', 'integer_mask_same_opcode', 'prescale_constant_same_opcode', 'paired_output_address_same_opcode'):
        changed = copy.deepcopy(candidate_module)
        fn = qdot_name if label in ('cross_row_input_pointer_same_opcode','integer_mask_same_opcode') else load_name if label == 'prescale_constant_same_opcode' else main_name
        args, blocks, header = changed.functions[fn]; edits = 0
        first_output_index = None
        for block, lines in blocks.items():
            for i, line in enumerate(lines):
                after = line
                if label == 'cross_row_input_pointer_same_opcode':
                    after = re.sub(r'%2\b', '%1', line)
                elif label == 'integer_mask_same_opcode' and not edits and re.search(r'and i8 %\d+, 15$', line):
                    after = line[:-2] + '14'
                elif label == 'prescale_constant_same_opcode' and not edits and '6.250000e-02' in line:
                    after = line.replace('6.250000e-02','1.250000e-01')
                elif label == 'paired_output_address_same_opcode':
                    match = re.search(r'getelementptr inbounds bfloat, bfloat addrspace\(1\)\* %4, i64 (%\d+)',line)
                    if match:
                        if first_output_index is None: first_output_index = match[1]
                        elif not edits: after = line[:match.start(1)] + first_output_index + line[match.end(1):]
                if after != line: lines[i] = after; edits += 1
        if not edits: raise ValueError('nonvacuous actual-IR mutation anchor required: ' + label)
        actual = candidate(changed, 4, 64, 2560, 10240, 0, 0, 0)
        expected_outputs = {offset:value for e in old for offset,value in outputs(e).items()}
        rejected = outputs(actual) != expected_outputs if label == 'paired_output_address_same_opcode' else actual.reductions != expected
        if not rejected: raise ValueError('actual dependency/address audit failed its negative: ' + label)
        results[label] = True
    changed = copy.deepcopy(candidate_module)
    attr = re.search(r'#(\d+) \{$', changed.functions[qdot_name][2])[1]
    changed.attributes[attr] = changed.attributes[attr].replace('"unsafe-fp-math"="false"', '"unsafe-fp-math"="true"')
    if changed.policy() == native.policy(): raise ValueError('arithmetic attribute mutation was not rejected')
    results['arithmetic_helper_attribute_drift'] = True
    return results

def main():
    p = argparse.ArgumentParser(); p.add_argument('--build', type=Path, default=HERE/'_cpu_build_v1'); a = p.parse_args()
    b = a.build.resolve()
    if not b.is_relative_to(HERE): raise ValueError('audit artifacts must remain under the owned kernel directory')
    native = Module(b/'native-qmv.ll'); new = Module(b/'candidate.ll')
    control_tap = Module(b/'control_probe.ll'); candidate_tap = Module(b/'candidate_probe.ll')
    if source.sha(b/'native-qmv.air') != source.AIR: raise ValueError('immutable actual control AIR differs')
    if any(m.policy() != native.policy() for m in (new, control_tap, candidate_tap)):
        raise ValueError('actual arithmetic-helper FP math policy drift')
    if any(m.intrinsicMetadata() != native.intrinsicMetadata() for m in (new, control_tap, candidate_tap)):
        raise ValueError('actual FP intrinsic ABI/attributes drift')
    negatives = mutationNegatives(native, new)
    records = []; total_rows = 0; commutations = 0; chunk_commutations = 0; tap_rows = 0
    original_mask_orders = 0; original_chunk_orders = 0
    for bits, group, k, n in source.OBSERVED:
        for padding in (0, 1):
            for lane in (0, 31):
                for out_row in (0, n-4):
                    for y in (0, 1):
                        paired = candidate(new, bits, group, k, n, y, lane, out_row, padding)
                        paired_tap = candidate(candidate_tap, bits, group, k, n, y, lane, out_row, padding, True)
                        if outputs(paired_tap) != outputs(paired): raise ValueError('actual candidate shipping/tap arithmetic dependency drift')
                        if paired_tap.reductions != paired.reductions: raise ValueError('actual candidate shipping/tap lane reduction dependency drift')
                        check_tap(paired_tap)
                        rows = (2*y, 2*y+1)
                        expected = {}; expected_loads = []; expected_reductions_by_row = []
                        for row in rows:
                            original = shipping(native, bits, group, k, n, row, lane, out_row, padding)
                            original_tap = shipping(control_tap, bits, group, k, n, row, lane, out_row, padding, True)
                            if outputs(original_tap) != outputs(original): raise ValueError('actual current shipping/control tap arithmetic dependency drift')
                            if original_tap.reductions != original.reductions: raise ValueError('actual original shipping/control tap lane reduction dependency drift')
                            check_tap(original_tap); tap_rows += 1
                            if len(original.reductions) != 4: raise ValueError('vacuous/incomplete original per-row lane reduction proof')
                            expected_reductions_by_row.append(original.reductions)
                            expected.update(outputs(original)); expected_loads.extend(original.loads)
                            check_addresses(original, bits, group, k, n, padding); total_rows += 1
                            original_mask_orders += original.maskCommutes; original_chunk_orders += original.chunkCommutes
                        if outputs(paired) != expected: raise ValueError(f'ordered full-K actual FP dependency mismatch: {bits}/{group} K{k} N{n} y{y} lane{lane} out{out_row} pad{padding}')
                        expected_reductions = [r[column] for column in range(4) for r in expected_reductions_by_row]
                        if paired.reductions != expected_reductions:
                            raise ValueError('actual nonvacuous per-row lane reduction dependency mismatch')
                        check_addresses(paired, bits, group, k, n, padding)
                        old_reads = [(kind, off, typ) for kind, off, typ in expected_loads if kind in ('W', 'S', 'B', 'Input')]
                        new_reads = [(kind, off, typ) for kind, off, typ in paired.loads if kind in ('W', 'S', 'B', 'Input')]
                        if set(new_reads) != set(old_reads): raise ValueError('actual typed readonly load pointer set drift')
                        commutations += paired.maskCommutes
                        chunk_commutations += paired.chunkCommutes
        # Complete physical-row/output-column mapping is checked independently
        # of the finite-path SSA proof, without any model or GPU operand data.
        addresses = []
        for x in range(n//8):
            for sg in (0, 1):
                for y in (0, 1):
                    for row in (2*y, 2*y+1):
                        addresses.extend(row*n + x*8 + sg*4 + c for c in range(4))
        if len(addresses) != 4*n or set(addresses) != set(range(4*n)):
            raise ValueError('R4 complete/disjoint row mapping proof failed')
        records.append({'bits': bits, 'group': group, 'K': k, 'N': n, 'complete_rows': 4,
                        'complete_disjoint_output_words': 4*n, 'partial_grid_host_admission_required': True})
    report = {'schema': 'R4-raw-large-guard-pair-actual-SSA-per-row-dependency-load-audit-v1',
              'pass': True, 'actual_native_control_AIR_sha256': source.AIR,
              'actual_candidate_AIR_sha256': source.sha(b/'candidate.air'),
              'arithmetic_helper_FP_policy_equal': True,
              'actual_FP_intrinsic_ABI_and_attributes_equal': True,
              'nonarithmetic_immutable_macroentry_has_unsafe_attrs_new_plainentry_safe': True,
              'per_row_full_K_FP_chronology_and_typed_dependencies_equal_with_recorded_commutations': True,
              'actual_typed_readonly_pointer_sets_equal': True, 'per_row_proof_cases': total_rows,
              'finite_integer_mask_multiplicand_commutations_recorded': commutations,
              'same_address_chunk_accumulator_operand_commutations_recorded': chunk_commutations,
              'original_mask_operand_orders_canonicalized': original_mask_orders,
              'original_chunk_operand_orders_canonicalized': original_chunk_orders,
              'same_opcode_actual_IR_dependency_address_attribute_mutation_negatives': negatives,
              'reassociation_or_blanket_DAG_normalization': False,
              'scope': 'Actual SSA executed at lanes0/31, first/last output SIMD, both paired row groups/all4rows, compact/odd padded strides, fullK; arbitrary finite symbolic BF16 inputs. No reassociation; only explicit finite integer-mask multiplicand and same-address += chunk operand commutations accepted/recorded. No current-trained-model or nonfinite bit parity claim.',
              'actual_GPU_raw_bits_NaN_Inf_sticky_register_spill_or_performance_proved': False,
              'shipping_tap_coupling_actual_SSA_audit_complete': True, 'shipping_tap_per_row_cases': tap_rows,
              'authorized_natural_shape_format_combinations': records,
              '13_synthetic_input_pattern_GPU_qualification_complete': False,
              'current_trained_R4_inputs_or_113_R5_call_evidence_inherited': False,
              'GPU_work': False, 'operand_payload_reads': 0}
    (b/'strict-SSA-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('pass','per_row_proof_cases','finite_integer_mask_multiplicand_commutations_recorded','GPU_work')}))

if __name__ == '__main__': main()
