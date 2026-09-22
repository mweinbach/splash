#!/usr/bin/env python3
"""Fail-closed actual AIR SSA/CFG/attribute audit; no model or GPU access.

The compared object is a graph, not an opcode census. Distinct SSA definitions
stay distinct, and argument/alloca identity, typed operands, sharing, predicates,
FP flags, call attributes, descriptor stores and opaque call order are retained.
Diagnostic stores alone may be removed from the shipping projection. Every FP
load or operation remains a root, even if it only feeds a diagnostic store.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys


LOCAL = re.compile(r'%(?:"[^"\n]+"|[-A-Za-z$._0-9]+)')
GLOBAL = re.compile(r'@(?:"[^"\n]+"|[-A-Za-z$._0-9]+)')
FP = re.compile(r'^(?:fadd|fsub|fmul|fdiv|frem|fneg|fptrunc|fpext|fcmp|fptosi|fptoui|sitofp|uitofp)\b')
FP_LOAD = re.compile(r'^load\s+(?:(?:atomic|volatile)\s+)*(?:float|bfloat|half|double|<\d+ x (?:float|bfloat|half|double)>)\s*,')
TERMINATOR = re.compile(r'^(?:br|ret|switch|unreachable|indirectbr|invoke|resume)\b')
HELPERS = ('finite', 'error', 'nan', 'sigmoid', 'rank', 'scan', 'execute')


class Refusal(RuntimeError):
    pass


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def split_top(text, separator=','):
    result, begin, stack, quoted, escaped = [], 0, [], False, False
    for position, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in '([{<':
            stack.append(char)
        elif char in ')]}>':
            if not stack:
                raise Refusal('Unbalanced LLVM syntax: ' + text[:160])
            stack.pop()
        elif char == separator and not stack:
            result.append(text[begin:position].strip())
            begin = position + 1
    if stack or quoted:
        raise Refusal('Unterminated LLVM syntax: ' + text[:160])
    return result + [text[begin:].strip()]


def helper_alias(text):
    # Itanium names encode the exact source identifier length. Keep the rest of
    # the mangled type/signature intact instead of erasing whole call names.
    def replace(match):
        length = int(match.group(1)) - len('r1_finite_summary_private_')
        name = match.group(2)
        if name.endswith('_tap'):
            name, length = name[:-4], length - 4
        if length != len(name):
            raise Refusal('Unexpected private helper mangling: ' + match.group(0))
        return str(length) + name
    text = re.sub(r'(\d+)r1_finite_summary_private_(gathered_mpp_(?:' + '|'.join(HELPERS) + r')(?:_consumer)?(?:_tap)?)', replace, text)
    for name in HELPERS:
        text = text.replace('r1_finite_summary_private_gathered_mpp_' + name,
                            'gathered_mpp_' + name)
    if 'r1_finite_summary_private_' in text:
        raise Refusal('Unknown private helper cannot be normalized')
    return text


class Function:
    def __init__(self, module, header, body):
        self.module, self.header, self.body = module, header, body
        match = re.search(r'@([^ (]+)\(', header)
        if not match:
            raise Refusal('Unrecognized function header')
        self.name = match.group(1)
        self.definition_prefix = module.normalize(header[:match.start()].strip())
        start, depth, finish = match.end(), 1, None
        for i in range(start, len(header)):
            depth += (header[i] == '(') - (header[i] == ')')
            if not depth:
                finish = i
                break
        if finish is None:
            raise Refusal('Unterminated function arguments')
        args = split_top(header[start:finish]) if header[start:finish].strip() else []
        self.args, self.arg_types = [], []
        for arg in args:
            tokens = LOCAL.findall(arg)
            if not tokens:
                raise Refusal('Unnamed/unrecognized definition argument: ' + arg)
            self.args.append(tokens[-1])
            self.arg_types.append(arg[:arg.rfind(tokens[-1])].strip())
        self.attributes = module.normalize(header[finish + 1:].replace('{', '').strip())
        labels = re.findall(r'^([-A-Za-z$._0-9]+):', body, re.M)
        self.blocks = {str(len(self.args)): 'B0'}
        self.blocks.update({label: 'B' + str(i + 1) for i, label in enumerate(labels)})
        self.defs, self.instructions, self.allocas = {}, [], {}
        block = 'B0'
        for raw in body.splitlines():
            line = raw.split(';', 1)[0].strip()
            if not line:
                continue
            if line.endswith(':'):
                block = self.blocks[line[:-1]]
                continue
            if not raw[:1].isspace():
                raise Refusal('Unsupported multiline LLVM instruction: ' + raw[:160])
            definition = re.match(r'(%(?:"[^"\n]+"|[-A-Za-z$._0-9]+)) = (.*)', line)
            lhs, rhs = (definition.group(1), definition.group(2)) if definition else (None, line)
            record = {'lhs': lhs, 'rhs': rhs, 'block': block}
            self.instructions.append(record)
            if lhs:
                if lhs in self.defs:
                    raise Refusal('Duplicate SSA definition: ' + lhs)
                self.defs[lhs] = record
                if rhs.startswith('alloca '):
                    self.allocas[lhs] = 'L' + str(len(self.allocas))

    def value_refs(self, text):
        return [x for x in LOCAL.findall(text) if x in self.defs or x in self.args or x[1:] in self.blocks]

    def pointer_base(self, value, seen=None):
        if value in self.args:
            return self.args.index(value)
        seen = set() if seen is None else seen
        if value in seen or value not in self.defs:
            return None
        seen.add(value)
        rhs = self.defs[value]['rhs']
        if not rhs.startswith(('getelementptr ', 'bitcast ', 'addrspacecast ')):
            return None
        refs = self.value_refs(rhs)
        return self.pointer_base(refs[0], seen) if refs else None

    def stores(self):
        result = []
        for record in self.instructions:
            if not record['rhs'].startswith('store '):
                continue
            fields = split_top(record['rhs'][6:])
            refs = self.value_refs(fields[1])
            if len(refs) != 1:
                raise Refusal('Unrecognized store pointer: ' + record['rhs'])
            result.append((record, fields[0], refs[0], self.pointer_base(refs[0])))
        return result

    def graph(self, base_args=None):
        self.module.begin_metadata_context()
        base_args = len(self.args) if base_args is None else base_args
        if base_args > len(self.args):
            raise Refusal('Missing original function arguments')
        diagnostic = {id(record) for record, value, ptr, base in self.stores() if base is not None and base >= base_args}
        categories = {'FP': [], 'calls': [], 'stores': [], 'control': [], 'observable_order': []}
        for record in self.instructions:
            rhs = record['rhs']
            if FP.match(rhs) or FP_LOAD.match(rhs) or re.match(r'^(?:select|phi|bitcast)\b.*\b(?:float|bfloat|half|double)\b', rhs) and not rhs.endswith('*'):
                categories['FP'].append(record)
            if re.search(r'\bcall\b', rhs):
                categories['calls'].append(record)
            if rhs.startswith('store ') and id(record) not in diagnostic:
                categories['stores'].append(record)
            if TERMINATOR.match(rhs):
                categories['control'].append(record)
            # An SSA address graph alone does not establish what a constructor
            # reads. Preserve chronological memory/call/load/control order,
            # including field initialization before opaque constructor calls.
            if rhs.startswith('load ') or re.search(r'\bcall\b', rhs) or TERMINATOR.match(rhs) or rhs.startswith('store ') and id(record) not in diagnostic:
                categories['observable_order'].append(record)
        nodes, node_ids = [], {}

        def value(token):
            if token in self.args:
                index = self.args.index(token)
                if index >= base_args:
                    raise Refusal('Diagnostic argument reaches shipping graph: ' + token)
                return 'A' + str(index)
            if token[1:] in self.blocks:
                return '%' + self.blocks[token[1:]]
            if token not in self.defs:
                return token  # Named type: preserve its exact spelling/layout.
            if token in node_ids:
                return node_ids[token]
            record = self.defs[token]
            name = 'L:' + self.allocas[token] if token in self.allocas else 'V' + str(len(nodes))
            node_ids[token] = name
            slot = len(nodes)
            nodes.append(None)  # Preserve actual cycles and sharing, e.g. phi.
            normalized = LOCAL.sub(lambda m: value(m.group(0)), self.module.normalize(record['rhs']))
            observable = FP.match(record['rhs']) or FP_LOAD.match(record['rhs']) or re.search(r'\bcall\b', record['rhs']) or record['rhs'].startswith(('load ', 'phi '))
            nodes[slot] = {'id': name, 'definition': normalized, 'control_block': record['block'] if observable else None}
            return name

        def instruction(record):
            if record['lhs']:
                return {'block': record['block'], 'node': value(record['lhs'])}
            return {'block': record['block'], 'instruction': LOCAL.sub(lambda m: value(m.group(0)), self.module.normalize(record['rhs']))}

        roots = {key: [instruction(record) for record in records] for key, records in categories.items()}
        return {'definition_prefix': self.definition_prefix,
                'arguments': [self.module.normalize(x) for x in self.arg_types[:base_args]], 'attributes': self.attributes,
                'blocks': list(dict.fromkeys(self.blocks.values())), 'roots': roots, 'nodes': nodes,
                'metadata_identity_graph': list(self.module._meta_nodes)}, node_ids, diagnostic

    def full(self):
        graph, _, diagnostic = self.graph()
        self.module.begin_metadata_context()
        if diagnostic:
            raise Refusal('Unexpected diagnostic argument in complete helper')
        # Include all definitions, including integer instructions not otherwise
        # observable. Complete shared helper equivalence permits no omissions.
        ids = {arg: 'A' + str(i) for i, arg in enumerate(self.args)}
        ids.update({lhs: 'V' + str(i) for i, lhs in enumerate(self.defs)})
        def local(match):
            token = match.group(0)
            return '%' + self.blocks[token[1:]] if token[1:] in self.blocks else ids.get(token, token)
        lines = [{'block': r['block'], 'lhs': ids.get(r['lhs']), 'rhs': LOCAL.sub(local, self.module.normalize(r['rhs']))} for r in self.instructions]
        return {'definition_prefix': self.definition_prefix, 'arguments': graph['arguments'],
                'attributes': self.attributes, 'instructions': lines,
                'metadata_identity_graph': list(self.module._meta_nodes)}


class Module:
    def __init__(self, path):
        self.path, self.data = path, path.read_bytes()
        self.text = self.data.decode()
        self.attributes = dict(re.findall(r'^attributes #([0-9]+) = (.*)$', self.text, re.M))
        self.metadata = dict(re.findall(r'^!([0-9]+) = (.*)$', self.text, re.M))
        self.begin_metadata_context()
        self.functions = {}
        for header, body in re.findall(r'^(define [^\n]+)\n(.*?)^}', self.text, re.M | re.S):
            function = Function(self, header, body)
            self.functions[function.name] = function
        self.declarations = {}
        for line in self.text.splitlines():
            if line.startswith('declare '):
                self.declarations[re.search(r'@([^ (]+)\(', line).group(1)] = line
        self.globals = {line.split(' = ', 1)[0]: line for line in self.text.splitlines() if line.startswith(('@', '%')) and ' = ' in line}
    def begin_metadata_context(self):
        self._meta_names, self._meta_nodes = {}, []

    def metadata_ref(self, identifier):
        names, nodes = self._meta_names, self._meta_nodes
        def visit(key):
            if key in names:
                return names[key]
            if key not in self.metadata:
                raise Refusal('Missing referenced metadata !' + key)
            name = 'M' + str(len(nodes)); names[key] = name
            position = len(nodes); nodes.append(None)
            nodes[position] = helper_alias(re.sub(r'!([0-9]+)', lambda m: visit(m.group(1)), self.metadata[key]))
            return name
        return visit(identifier)

    def normalize(self, text):
        text = helper_alias(text)
        def attribute(match):
            if match.group(1) not in self.attributes:
                raise Refusal('Missing attribute group #' + match.group(1))
            return 'ATTR' + helper_alias(self.attributes[match.group(1)])
        text = re.sub(r'#([0-9]+)', attribute, text)
        text = re.sub(r'!([0-9]+)', lambda m: '!' + self.metadata_ref(m.group(1)), text)
        return text

    def execute(self, gate, consumer=False):
        names = [name for name in self.functions if 'gathered_mpp_execute' in name
                 and ('ILb1' if gate else 'ILb0') in name
                 and ('execute_consumer' in name) == consumer]
        if len(names) != 1:
            raise Refusal('Expected one outlined execute helper: ' + str(names))
        return self.functions[names[0]]


def difference(a, b):
    if a == b:
        return None
    aa, bb = json.dumps(a, sort_keys=True, indent=2).splitlines(), json.dumps(b, sort_keys=True, indent=2).splitlines()
    for i, (left, right) in enumerate(zip(aa, bb)):
        if left != right:
            return {'line': i + 1, 'original': left[:400], 'other': right[:400]}
    return {'original_lines': len(aa), 'other_lines': len(bb)}


def compare_helpers(left, right, tapping=False):
    results = []
    for gate in (True, False):
        original, other = left.execute(gate), right.execute(gate)
        graph_a, _, _ = original.graph()
        graph_b, ids_b, diagnostic = other.graph(len(original.args))
        discrepancy = difference(graph_a, graph_b)
        record = {'role': 'gate_up' if gate else 'down', 'original': original.name, 'other': other.name,
                  'actual_SSA_graph_equal': discrepancy is None, 'first_difference': discrepancy,
                  'original_graph_sha256': sha(json.dumps(graph_a, sort_keys=True)), 'other_graph_sha256': sha(json.dumps(graph_b, sort_keys=True)),
                  'root_counts': {key: len(value) for key, value in graph_a['roots'].items()},
                  'original_graph': graph_a, 'other_graph': graph_b}
        if tapping:
            tap_stores = []
            stages = producer_stages(other, gate)
            expected = {15: 'raw_g', 16: 'scaled_g', 19: 'bf_g'}
            if gate:
                expected.update({17: 'raw_u', 18: 'scaled_u', 20: 'bf_u', 21: 'silu', 22: 'combined'})
            shipping_outputs = [(inst, stored, ptr) for inst, stored, ptr, base in other.stores()
                                if base == 7 and len(other.value_refs(stored)) == 1]
            if len(shipping_outputs) != 1:
                raise Refusal('Cannot establish one actual shipping output-store address')
            def element_index(ptr):
                if ptr not in other.defs or not other.defs[ptr]['rhs'].startswith('getelementptr '):
                    raise Refusal('Cannot establish actual tap/output GEP element index')
                fields = split_top(other.defs[ptr]['rhs'])
                if len(fields) != 3:
                    raise Refusal('Unexpected tap/output GEP index shape')
                return fields[2]
            shipping_index = element_index(shipping_outputs[0][2])
            for inst, stored, ptr, base in other.stores():
                if id(inst) not in diagnostic:
                    continue
                refs = other.value_refs(stored)
                coupled = len(refs) == 1 and refs[0] in ids_b
                role = expected.get(base)
                tap_stores.append({'parameter_index': base, 'stored_SSA': refs[0] if refs else stored,
                                   'shipping_node': ids_b.get(refs[0]) if refs else None, 'same_shipping_SSA_definition': coupled,
                                   'expected_producer_role': role, 'expected_producer_SSA': stages.get(role),
                                   'exact_producer_role_SSA_match': bool(role) and len(refs) == 1 and refs[0] == stages[role],
                                   'logical_element_index': element_index(ptr),
                                   'same_actual_shipping_element_index_SSA': element_index(ptr) == shipping_index})
            record['diagnostic_stores'] = tap_stores
            record['tap_values_share_shipping_SSA'] = len(tap_stores) == len(expected) and set(x['parameter_index'] for x in tap_stores) == set(expected) and all(x['same_shipping_SSA_definition'] and x['same_actual_shipping_element_index_SSA'] and x['exact_producer_role_SSA_match'] for x in tap_stores)
        else:
            record['complete_outlined_execute_SSA_CFG_equal'] = original.full() == other.full()
            record['actual_SSA_graph_equal'] = record['actual_SSA_graph_equal'] and record['complete_outlined_execute_SSA_CFG_equal']
        results.append(record)
    return results


def producer_stages(function, gate):
    stages = {}
    for lhs, record in function.defs.items():
        if not record['rhs'].startswith('fmul float '):
            continue
        refs = function.value_refs(record['rhs'])
        if len(refs) != 2:
            raise Refusal('Cannot establish native F32 multiply operand tree')
        for scale, raw in (refs, refs[::-1]):
            if scale not in function.defs or raw not in function.defs:
                continue
            scale_load, raw_load = function.defs[scale]['rhs'], function.defs[raw]['rhs']
            if not FP_LOAD.match(scale_load) or not FP_LOAD.match(raw_load):
                continue
            pointers = function.value_refs(scale_load)
            base = function.pointer_base(pointers[0]) if len(pointers) == 1 else None
            if base not in (2, 4):
                continue
            suffix = 'g' if base == 2 else 'u'
            if 'raw_' + suffix in stages:
                raise Refusal('Ambiguous raw/scaled producer role')
            stages['raw_' + suffix], stages['scaled_' + suffix] = raw, lhs
    for suffix in ('g', 'u') if gate else ('g',):
        if 'scaled_' + suffix not in stages:
            raise Refusal('Missing late row-scale producer role')
        matches = [lhs for lhs, r in function.defs.items() if r['rhs'].startswith('fptrunc float ') and function.value_refs(r['rhs']) == [stages['scaled_' + suffix]]]
        if len(matches) != 1:
            raise Refusal('Ambiguous BF16 projection boundary')
        stages['bf_' + suffix] = matches[0]
    if gate:
        matches = [lhs for lhs, r in function.defs.items() if r['rhs'].startswith('fmul bfloat ') and stages['bf_g'] in function.value_refs(r['rhs']) and stages['bf_u'] not in function.value_refs(r['rhs'])]
        if len(matches) != 1:
            raise Refusal('Ambiguous native BF16 SiLU producer')
        stages['silu'] = matches[0]
        matches = [lhs for lhs, r in function.defs.items() if r['rhs'].startswith('fmul bfloat ') and set(function.value_refs(r['rhs'])) == {stages['silu'], stages['bf_u']}]
        if len(matches) != 1:
            raise Refusal('Ambiguous native BF16 combined producer')
        stages['combined'] = matches[0]
    return stages


def call_arguments(record):
    rhs = record['rhs']; match = re.search(r'@([^ (]+)\(', rhs)
    if not match:
        raise Refusal('Unrecognized direct call: ' + rhs)
    start, depth = match.end(), 1
    for i in range(start, len(rhs)):
        depth += (rhs[i] == '(') - (rhs[i] == ')')
        if not depth:
            return match.group(1), split_top(rhs[start:i]), rhs[:match.start()], rhs[i + 1:]
    raise Refusal('Unterminated direct call')


def wrapper_storage(module, name):
    result = {}
    for suffix in ('safe_a', 'nonfinite'):
        matches = [(symbol, line) for symbol, line in module.globals.items()
                   if name in symbol and symbol.endswith('E' + str(len(suffix)) + suffix)]
        if len(matches) != 1:
            raise Refusal('Expected one wrapper-local storage global: ' + name + '/' + suffix)
        symbol, line = matches[0]
        declaration = line.split(' = ', 1)[1]
        if not declaration.startswith('internal addrspace(3) global '):
            raise Refusal('Wrapper-local threadgroup declaration differs: ' + symbol)
        result[suffix] = {'symbol': symbol, 'declaration': module.normalize(declaration)}
    return result


def wrapper_call_graph(module, name, gate, swapped, tapping, storage, consumer=False):
    if name not in module.functions:
        raise Refusal('Missing wrapper: ' + name)
    function = module.functions[name]
    if function.graph()[0]['roots']['FP']:
        raise Refusal('Unexpected wrapper FP load/arithmetic: ' + name)
    calls = [r for r in function.instructions if 'gathered_mpp_execute' in r['rhs']]
    if len(calls) != 1:
        raise Refusal('Expected one outlined producer call: ' + name)
    callee, args, prefix, attrs = call_arguments(calls[0])
    if callee != module.execute(gate, consumer).name:
        raise Refusal('Wrong outlined producer specialization: ' + name)
    count = 10 if gate else 8
    group_index = ((19 if gate else 12) if tapping else count) + (2 if consumer else 0)
    private = name.startswith('r1_finite_summary_')
    thread_index, tid_index = group_index + (2 if private else 1), group_index + (3 if private else 2)
    if len(function.args) <= tid_index:
        raise Refusal('Wrapper builtin argument extent differs')
    replacements = {function.args[i]: 'A' + str(i) for i in range(count)}
    replacements.update({function.args[group_index]: 'LOGICAL_GROUP', function.args[thread_index]: 'THREADS', function.args[tid_index]: 'TID'})
    globals_ = {r['symbol']: '@' + suffix for suffix, r in storage.items()}
    if swapped:
        refs = function.value_refs(args[10])
        if len(refs) != 1 or refs[0] not in function.defs:
            raise Refusal('Swapped logical group has no actual SSA definition')
        shuffle = function.defs[refs[0]]['rhs']
        expected = 'shufflevector <3 x i32> ' + function.args[group_index] + ', <3 x i32> '
        if not shuffle.startswith(expected) or not re.search(r'<i32 2, i32 1, i32 0>$', shuffle):
            raise Refusal('Actual axis-swap vector/mask differs: ' + shuffle)
        replacements[refs[0]] = 'LOGICAL_GROUP'
    nodes, names = [], {}

    def local(match):
        token = match.group(0)
        if token in replacements:
            return replacements[token]
        if token not in function.defs:
            if token in function.args:
                raise Refusal('Unexpected wrapper argument reaches original producer: ' + token)
            return token
        if token in names:
            return names[token]
        label = 'V' + str(len(nodes)); names[token] = label
        position = len(nodes); nodes.append(None)
        rhs = module.normalize(function.defs[token]['rhs'])
        rhs = GLOBAL.sub(lambda m: globals_.get(m.group(0), m.group(0)), rhs)
        nodes[position] = LOCAL.sub(local, rhs)
        return label

    if len(args) != (23 if tapping else 15) + consumer:
        raise Refusal('Outlined helper argument count differs')
    if consumer:
        references = function.value_refs(args[15])
        if len(references) != 1 or references[0] not in function.defs:
            raise Refusal('Consumer scan bool is not one actual validated SSA result')
        seen, leaves = set(), set()
        def visit(value):
            if value in seen:
                return
            seen.add(value)
            if value in function.args:
                leaves.add(function.args.index(value)); return
            if value in function.defs:
                for operand in function.value_refs(function.defs[value]['rhs']):
                    visit(operand)
        visit(references[0])
        if not leaves <= {count, count + 1, group_index}:
            raise Refusal('Consumer scan bool depends on nonuniform or unrelated input arguments')
    if tapping:
        shift = 2 if consumer else 0
        expected = [10+shift,11+shift,12+shift,13+shift,14+shift,15+shift,16+shift,17+shift] if gate else [8+shift,9+shift,None,None,10+shift,None,None,None]
        for argument, index in zip(args[15+consumer:], expected):
            if index is None:
                if not argument.endswith(' null'):
                    raise Refusal('Unused down tap argument is not literal null')
            elif function.value_refs(argument) != [function.args[index]]:
                raise Refusal('Typed diagnostic argument/buffer binding drift: ' + name + ': ' + argument)
    roots = [LOCAL.sub(local, GLOBAL.sub(lambda m: globals_.get(m.group(0), m.group(0)), module.normalize(arg))) for arg in args[:15]]
    return {'leading_parameter_types': [module.normalize(x) for x in function.arg_types[:count]],
            'callee_role': 'gate_up' if gate else 'down', 'call_prefix': module.normalize(prefix),
            'call_attributes': module.normalize(attrs), 'roots': roots, 'nodes': nodes}, function.attributes


def compare_wrappers(original, candidate, taps):
    records, original_local_globals = [], set()
    for gate in (True, False):
        native = 'flash_gathered_mpp_' + ('gate_up' if gate else 'down') + '_m16_n64_sg4'
        base_storage = wrapper_storage(original, native)
        original_local_globals.update(x['symbol'] for x in base_storage.values())
        base_graph, base_attrs = wrapper_call_graph(original, native, gate, False, False, base_storage)
        for module, tapping in ((candidate, False), (taps, True)):
            for swapped in (False, True):
                name = 'r1_traversal_' + ('gu_' if gate else 'down_') + ('swapped' if swapped else 'native') + ('_tap' if tapping else '')
                storage = wrapper_storage(module, name)
                storage_match = {suffix: storage[suffix]['declaration'] == base_storage[suffix]['declaration'] for suffix in base_storage}
                graph, attrs = wrapper_call_graph(module, name, gate, swapped, tapping, storage)
                discrepancy = difference(base_graph, graph)
                records.append({'module': module.path.name, 'wrapper': name, 'storage': storage,
                                'original_storage': base_storage, 'exact_threadgroup_declarations_equal': all(storage_match.values()),
                                'original_producer_argument_dependency_graph_equal': discrepancy is None,
                                'first_difference': discrepancy, 'original_call_graph': base_graph, 'private_call_graph': graph,
                                'contains_FP_load_or_arithmetic': False,
                                'whole_entry_attributes_equal': base_attrs == attrs,
                                'original_entry_attributes': base_attrs, 'private_entry_attributes': attrs,
                                'entry_attribute_scope': 'Entry wrappers contain no FP load/arithmetic; all producer/scan/SDK and callsite attributes are separately required to match exactly.'})
    return records, original_local_globals


def foundation_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args(); build = args.build.resolve()
    report = {'schema': 'native-r1-finite-summary-actual-AIR-SSA-arithmetic-audit-v1', 'candidate_native_FP_tree_match': False,
              'taps_shipping_FP_tree_match': False, 'finite_scan_predicate_equivalence_proved': False, 'GPU_work': False, 'model_or_activation_payload_reads': False,
              'normalization': ['exact private helper identifier prefix and matching Itanium identifier length', 'SSA alpha names with distinct definitions and sharing retained', 'metadata numbering with referenced graph payloads retained', 'attribute numbering resolved to exact attribute sets'],
              'failures': [], 'actual_GPU_bit_exactness_proved': False, 'opaque_MPP_internal_arithmetic_proved': False}
    try:
        modules = [Module(build / (name + '.ll')) for name in ('original-gathered', 'candidate', 'taps')]
        original, candidate, taps = modules
        report['inputs'] = [{'path': m.path.name, 'bytes': len(m.data), 'sha256': sha(m.data)} for m in modules]
        recipe_path = build / 'source/dev/benchmarks/expert_r1_finite_summary_sep22/kernel/original_recipe.json'
        recipe = json.loads(recipe_path.read_text())
        if sha((build / 'original-gathered.air').read_bytes()) != recipe['immutable_performance_control']['sha256']:
            raise Refusal('Immutable original native AIR hash differs')
        native = compare_helpers(original, candidate)
        tapped = compare_helpers(candidate, taps, True)
        report['candidate_native_helpers'] = native
        report['taps_shipping_helpers'] = tapped
        wrappers, original_local_globals = compare_wrappers(original, candidate, taps)
        report['wrapper_storage_and_actual_typed_call_provenance'] = wrappers
        shared = []
        original_names = {helper_alias(name): name for name in original.functions if 'gathered_mpp_execute' not in name and not name.startswith('flash_gathered_mpp_')}
        for module in (candidate, taps):
            normalized = {helper_alias(name): name for name in module.functions}
            for name, original_name in original_names.items():
                if name not in normalized:
                    raise Refusal('Missing shared/control helper in ' + module.path.name + ': ' + name)
                discrepancy = difference(original.functions[original_name].full(), module.functions[normalized[name]].full())
                shared.append({'module': module.path.name, 'function': name, 'complete_SSA_CFG_attributes_equal': discrepancy is None, 'first_difference': discrepancy})
            for name, declaration in original.declarations.items():
                if name not in module.declarations or original.normalize(declaration) != module.normalize(module.declarations[name]):
                    raise Refusal('Shared external declaration/attributes differ: ' + name)
            for name, declaration in original.globals.items():
                if name in original_local_globals:
                    continue  # Explicit per-wrapper declaration/binding proof above.
                if name not in module.globals or original.normalize(declaration) != module.normalize(module.globals[name]):
                    raise Refusal('Named type/global MPP descriptor differs: ' + name)
        report['shared_control_and_SDK_helpers'] = shared
        report['shared_external_declarations_and_global_descriptors_equal'] = True
        wrapper_match = all(x['exact_threadgroup_declarations_equal'] and x['original_producer_argument_dependency_graph_equal'] for x in wrappers)
        report['candidate_native_FP_tree_match'] = wrapper_match and all(x['actual_SSA_graph_equal'] for x in native) and all(x['complete_SSA_CFG_attributes_equal'] for x in shared)
        report['taps_shipping_FP_tree_match'] = wrapper_match and all(x['actual_SSA_graph_equal'] and x['tap_values_share_shipping_SSA'] for x in tapped) and all(x['complete_SSA_CFG_attributes_equal'] for x in shared)
        for record in wrappers:
            if not record['exact_threadgroup_declarations_equal'] or not record['original_producer_argument_dependency_graph_equal']:
                report['failures'].append({'reason': 'Wrapper threadgroup declaration/producer argument provenance drift', 'wrapper': record['wrapper'], 'difference': record['first_difference']})
        for group in (native, tapped):
            for record in group:
                if not record['actual_SSA_graph_equal']:
                    report['failures'].append({'reason': 'Actual SSA/CFG/FP/load/call/store graph drift', 'role': record['role'], 'function': record['other'], 'difference': record['first_difference']})
                if record.get('tap_values_share_shipping_SSA') is False:
                    report['failures'].append({'reason': 'Tap does not directly reuse each shipping producer SSA value', 'role': record['role']})
        for record in shared:
            if not record['complete_SSA_CFG_attributes_equal']:
                report['failures'].append({'reason': 'Shared/control helper full graph or attributes drift', **record})
        for module in modules:
            if module.path.read_bytes() != module.data:
                raise Refusal('Input IR changed during audit: ' + module.path.name)
    except Exception as error:
        report['candidate_native_FP_tree_match'] = report['taps_shipping_FP_tree_match'] = False
        report['failures'].append({'reason': 'Graph could not be established', 'error': str(error), 'exception': type(error).__name__})
    output = build / 'arithmetic-audit.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    passed = report['candidate_native_FP_tree_match'] and report['taps_shipping_FP_tree_match']
    print(json.dumps({'pass': passed, 'candidate_native_FP_tree_match': report['candidate_native_FP_tree_match'],
                      'taps_shipping_FP_tree_match': report['taps_shipping_FP_tree_match'], 'report': str(output), 'failures': report['failures']}))
    return 0 if passed else 1


def receiver_specialization_proof(original, private):
    names = [name for name in original.functions
             if 'tensorIU9MTLdevicea' in name and '_clIJLm0ELm1' in name]
    if len(names) != 1:
        raise Refusal('Cannot identify exactly one original signed-I8 constructor')
    generic = names[0]; specialized = generic + '.specialized.p0'
    if specialized not in private.functions:
        if generic not in private.functions:
            raise Refusal('Original or private receiver constructor is missing')
        if original.functions[generic].full() != private.functions[generic].full():
            raise Refusal('Unspecialized constructor body/attributes differ')
        return {'generic': generic, 'specialized': generic, 'receiver_specialization': False,
                'full_body_attributes_memory_order_equal': True}
    old, new = original.functions[generic], private.functions[specialized]
    if old.arg_types != ['%class.anon.15 addrspace(8)* noundef'] or new.arg_types != ['%class.anon.15* noundef']:
        raise Refusal('Receiver specialization changed an unexpected formal argument')
    if old.definition_prefix != 'define linkonce_odr void' or new.definition_prefix != 'define internal void':
        raise Refusal('Receiver specialization linkage/return type is not the expected private clone')
    derived = {old.args[0]}
    changed = []
    for record in old.instructions:
        rhs = record['rhs']
        if 'addrspace(8)' not in rhs:
            continue
        if not rhs.startswith(('bitcast ', 'getelementptr ', 'load ')):
            raise Refusal('Generic receiver escapes into an unsupported operation')
        refs = old.value_refs(rhs)
        if not refs or refs[0] not in derived:
            raise Refusal('Address-space8 operand is not derived from the closure receiver')
        if record['lhs'] and rhs.startswith(('bitcast ', 'getelementptr ')):
            derived.add(record['lhs'])
        changed.append(rhs)
    projected = copy.deepcopy(old)
    projected.arg_types[0] = new.arg_types[0]
    projected.definition_prefix = new.definition_prefix
    for record in projected.instructions:
        if 'addrspace(8)' in record['rhs']:
            record['rhs'] = record['rhs'].replace(' addrspace(8)', '')
    expected, actual = projected.full(), new.full()
    discrepancy = difference(expected, actual)
    if discrepancy:
        raise Refusal('Bounded private receiver constructor full graph/attributes drift: ' + str(discrepancy))
    if any(FP.match(r['rhs']) or FP_LOAD.match(r['rhs']) for r in old.instructions + new.instructions):
        raise Refusal('Receiver specialization unexpectedly contains numeric FP work')
    init = [r['rhs'] for r in old.instructions if '@air.init_strided_private_tensor.i32.global(' in r['rhs']]
    if len(init) != 1:
        raise Refusal('Expected one unchanged signed-I8 strided tensor initializer')
    return {'generic': generic, 'specialized': specialized, 'receiver_specialization': True,
            'receiver_only_address_space8_to_private0': True,
            'device_coefficient_address_space1_unchanged': True,
            'changed_receiver_field_operations': changed,
            'full_body_attributes_memory_order_equal': True,
            'original_specialized_body': expected, 'actual_private_body': actual,
            'equivalent_body_sha256': sha(json.dumps(actual, sort_keys=True)),
            'linkage_change': 'Original linkonce_odr symbol remains in immutable control AIR; private replacement is a distinct internal symbol and cannot override it.'}


def canonical_constructor_calls(function, proof, gate):
    projected = copy.deepcopy(function)
    count = 0; removed = set(); instances = []
    for record in projected.instructions:
        if '@' + proof['generic'] + '(' not in record['rhs'] and '@' + proof['specialized'] + '(' not in record['rhs']:
            continue
        callee, arguments, prefix, attrs = call_arguments(record)
        if len(arguments) != 1:
            raise Refusal('Signed-I8 constructor actual argument count differs')
        refs = projected.value_refs(arguments[0])
        if len(refs) != 1:
            raise Refusal('Signed-I8 constructor receiver is not one actual SSA object')
        receiver = refs[0]
        if callee == proof['generic'] and proof['receiver_specialization']:
            if receiver not in projected.defs:
                raise Refusal('Original generic receiver has no private cast definition')
            cast = projected.defs[receiver]['rhs']
            match = re.fullmatch(r'addrspacecast %class\.anon\.15\* (%[0-9]+) to %class\.anon\.15 addrspace\(8\)\*', cast)
            if not match:
                raise Refusal('Constructor receiver is not the exact known private0-to-generic8 cast')
            actual = match.group(1)
            users = [r for r in projected.instructions if r is not projected.defs[receiver] and receiver in projected.value_refs(r['rhs'])]
            if users != [record]:
                raise Refusal('Receiver cast has additional uses; normalization is not bounded')
            removed.add(receiver)
        else:
            actual = receiver
        if actual not in projected.allocas or not projected.defs[actual]['rhs'].startswith('alloca %class.anon.15,'):
            raise Refusal('Specialized receiver is not the original distinct allocated private closure')
        instances.append({'actual_receiver_SSA': actual, 'ordered_allocation': projected.allocas[actual],
                          'original_cast_SSA': receiver if receiver != actual else None,
                          'nonnull_proved_by_private_alloca': True})
        record['rhs'] = prefix + '@' + proof['specialized'] + '(%class.anon.15* noundef nonnull ' + actual + ')' + attrs
        count += 1
    if count != (2 if gate else 1):
        raise Refusal('Expected exactly GU two/DOWN one signed-I8 receiver instances')
    projected.instructions = [r for r in projected.instructions if r['lhs'] not in removed]
    for name in removed:
        del projected.defs[name]
    return projected, instances


def scan_boundary_projection(function, gate, consumer, finite=False):
    """Project only the recognized scan branch, preserving all other SSA/CFG.

    fallback=True is the native scan path. finite=True omits it only under the
    separately checked finite-certificate premise. No numeric operand, pointer
    index, metadata payload, attribute or other branch is normalized here.
    """
    projected = copy.deepcopy(function)
    scan = [r for r in projected.instructions if 'gathered_mpp_scanILt' in r['rhs'] and re.search(r'\bcall\b', r['rhs'])]
    if len(scan) != 1:
        raise Refusal('Cannot identify one literal native scan call')
    call = scan[0]; aliases = {}; removed = set(); merged_blocks = {}
    if consumer:
        if len(projected.args) not in (16, 24) or projected.arg_types[15] != 'i1 noundef zeroext':
            raise Refusal('Authorized needsOriginalScan formal bool differs')
        need = projected.args[15]
        users = [r for r in projected.instructions if need in projected.value_refs(r['rhs'])]
        if gate:
            if len(users) != 1 or not users[0]['rhs'].startswith('br i1 ' + need + ','):
                raise Refusal('GU authorized bool reaches operations outside the scan boundary')
            branch = users[0]
            labels = re.findall(r'label (%[0-9]+)', branch['rhs'])
            if len(labels) != 2 or projected.blocks[labels[0][1:]] != call['block']:
                raise Refusal('GU bool no longer selects exactly the literal scan block')
            pre, scan_block, merge = branch['block'], call['block'], projected.blocks[labels[1][1:]]
            block_records = [r for r in projected.instructions if r['block'] == scan_block]
            if len(block_records) != 2 or block_records[0] is not call or block_records[1]['rhs'] != 'br label ' + labels[1]:
                raise Refusal('GU optional scan block contains additional operations')
            phis = [r for r in projected.instructions if r['block'] == merge and r['rhs'].startswith('phi i1 ')]
            expected = 'phi i1 [ ' + call['lhs'] + ', ' + labels[0] + ' ], [ false, %' + next(k for k,v in projected.blocks.items() if v==pre) + ' ]'
            if len(phis) != 1 or phis[0]['rhs'] != expected:
                raise Refusal('GU bad merge is not exactly old scan result versus false')
            aliases[phis[0]['lhs']] = 'false' if finite else call['lhs']
            removed.update({id(branch), id(block_records[1]), id(phis[0])})
            merged_blocks = {scan_block: pre, merge: pre}
        else:
            if len(users) != 1 or users[0]['rhs'] != 'xor i1 ' + need + ', true':
                raise Refusal('DOWN bool is not exclusively the expected scan-skip inversion')
            invert = users[0]
            ors = [r for r in projected.instructions if r['rhs'].startswith('or i1 ') and invert['lhs'] in projected.value_refs(r['rhs'])]
            if len(ors) != 1:
                raise Refusal('DOWN scan eligibility combines unexpected values')
            refs = projected.value_refs(ors[0]['rhs']); rank = next(x for x in refs if x != invert['lhs'])
            if rank not in projected.defs or not projected.defs[rank]['rhs'].startswith('icmp eq i32 ') or not projected.defs[rank]['rhs'].endswith(', -1'):
                raise Refusal('DOWN original invalid-rank outer guard changed')
            aliases[ors[0]['lhs']] = 'true' if finite else rank
            removed.update({id(invert), id(ors[0])})
        projected.args.pop(15); projected.arg_types.pop(15)
    if finite:
        aliases[call['lhs']] = 'false'; removed.add(id(call))
    def resolve(match):
        token = match.group(0)
        return aliases.get(token, token)
    survivors = [r for r in projected.instructions if id(r) not in removed]
    removed_values = {r['lhs'] for r in projected.instructions if id(r) in removed and r['lhs']}
    for record in survivors:
        record['rhs'] = LOCAL.sub(resolve, record['rhs'])
        record['block'] = merged_blocks.get(record['block'], record['block'])
    if merged_blocks:
        old_labels = {v:k for k,v in projected.blocks.items()}
        for record in survivors:
            record['rhs'] = LOCAL.sub(lambda m: '%' + old_labels[merged_blocks[projected.blocks[m.group(0)[1:]]]] if m.group(0)[1:] in projected.blocks and projected.blocks[m.group(0)[1:]] in merged_blocks else m.group(0), record['rhs'])
        remaining = [v for v in projected.blocks.values() if v not in merged_blocks]
        compact = {value:'B'+str(i) for i,value in enumerate(remaining)}
        projected.blocks = {key:compact[merged_blocks.get(value,value)] for key,value in projected.blocks.items()}
        for record in survivors:
            record['block'] = compact[record['block']]
    projected.instructions = survivors
    projected.defs = {r['lhs']:r for r in survivors if r['lhs']}
    return projected, {'consumer':consumer,'gate':gate,'finite_continuation':finite,
                       'literal_scan_call':call['rhs'],'explicit_value_aliases':aliases,
                       'only_declared_scan_blocks_collapsed':merged_blocks,
                       'removed_boundary_instructions':len(removed)}


def finite_source_certificate(build, recipe):
    folder = build / 'source/dev/benchmarks/expert_r1_finite_summary_sep22/kernel'
    source_path = Path(recipe['authenticated_original_sources'][0]['path'])
    original = source_path.read_text()
    if sha(original) != recipe['authenticated_original_sources'][0]['sha256']:
        raise Refusal('Authenticated native source receipt differs')
    end = original.index('kernel void flash_gathered_mpp_gate_up_m16_n64_sg4(')
    prefix = original[:end]
    if (folder/'native_helper.metal').read_text() != prefix:
        raise Refusal('Native helper prefix/pragmas changed')
    begin = prefix.index('template <bool Gate>\ninline void gathered_mpp_execute(')
    execute = prefix[begin:]
    consumer = (folder/'consumer_helper.metalh').read_text()
    old = '  const bool bad = (Gate || rank != UINT_MAX) ? gathered_mpp_scan<K>(x, safe_a, nonfinite, tid, diag) : false;'
    new = '  const bool bad = (Gate || rank != UINT_MAX) ? (needsOriginalScan ? gathered_mpp_scan<K>(x, safe_a, nonfinite, tid, diag) : false) : false;'
    if consumer.count(new) != 1:
        raise Refusal('Consumer does not retain the exact declared conditional scan predicate')
    restored = consumer.replace(new, old).replace('r1_finite_summary_private_gathered_mpp_execute_consumer', 'gathered_mpp_execute').replace('threadgroup atomic_uint *nonfinite, bool needsOriginalScan)', 'threadgroup atomic_uint *nonfinite)')
    if restored != execute:
        raise Refusal('Consumer changes extend beyond the declared scan boundary')
    summary = (folder/'summary.metalh').read_text()
    required = ['(bits[k] & 0x7f80u) == 0x7f80u', 'for (uint k = lane; k < 2560; k += 32)',
                'for (uint k = lane; k < 640; k += 32)', 'const bool bad = simd_any(local_bad);',
                'id >= 0 && id < 512 && ranks[uint(id)] < 512',
                '!r1_finite_summary_packet_valid(packet, v, role, k) || packet[route] != 0',
                'packet[10] != kR1FiniteSummaryMagic', 'packet[11] != v.epoch', 'packet[12] != role',
                'packet[13] != 1', 'packet[14] != k', 'packet[15]', 'packet[slot] > 1',
                'role == kR1FiniteSummaryGURole && slot && packet[slot]',
                'v.epoch != 0 && v.role == role && !v.reserved0 && !v.reserved1']
    if any(token not in summary for token in required):
        raise Refusal('Summary packet/classifier source certificate changed')
    if summary.count(required[0]) != 2 or prefix.count('(bits[k] & 0x7f80u) == 0x7f80u') != 1:
        raise Refusal('Summary is not the exact original BF16 exponent-bit classifier')
    for name in ('gu_flags', 'down_flags'):
        start = summary.index('kernel void r1_finite_summary_' + name + '(')
        section = summary[start:]
        guard = section[:section.index('r1_finite_summary_reset(packet, lane);')]
        width = re.search(r'(?:uint|ushort) (\w+) \[\[threads_per_simdgroup\]\]', guard)
        if not width or not re.search(r'\b' + width.group(1) + r' != 32\b', guard):
            raise Refusal('Summary does not guard actual SIMD width32 before collectives')
        if 'total.x != 1' not in guard or 'threads.x != 32' not in guard:
            raise Refusal('Summary topology guard no longer precedes reset')
    return {'native_helper_prefix_sha256':sha(prefix), 'consumer_reverse_boundary_byte_exact':True,
            'same_integer_BF16_classifier':True, 'coverage':{'GU':{'K':2560,'lanes':32,'stride':32},'down':{'K':640,'lanes':32,'stride':32,'routes':10}},
            'actual_SIMD_width32_guard_before_collectives':True,
            'proof': 'On one complete SIMD32, lane+32*j partitions every native-scanned word. simd_any computes the same OR as the original threadgroup atomic OR. Trusted zero therefore equals old scan false. Excluded DOWN routes use the unchanged outer no-read guard. Nonzero/unrecognized/stale structural metadata selects literal original scan.',
            'premises':['Fixed ordered trusted summary→matching consumer submission', 'No mutation of H/activation/IDs/ranks/packet between matching producer and consumer', 'GU summary ignores ID validity; DOWN exclusion classification stays unchanged', 'Same-role/current nonzero epoch and matching packet metadata; no arbitrary forged-current zero'],
            'content_authentication_of_arbitrary_current_packet_proved':False}


def finite_main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--build',type=Path,required=True)
    args=parser.parse_args(); build=args.build.resolve()
    report={'schema':'native-r1-finite-summary-bounded-actual-AIR-arithmetic-audit-v1',
            'candidate_native_FP_tree_match':False,'taps_shipping_FP_tree_match':False,
            'finite_scan_predicate_equivalence_proved':False,'failures':[], 'GPU_work':False,
            'model_or_activation_payload_reads':False,'actual_GPU_bit_exactness_proved':False,
            'normalization':['Exact private helper prefix/tap identifier and SSA/metadata alpha identity', 'Explicit certified signed-I8 receiver8→known private0 specialization only', 'Only the declared conditional scan bool cone/block contraction']}
    try:
        original,candidate,taps=[Module(build/(name+'.ll'))for name in ('original-gathered','candidate','taps')]
        modules=(original,candidate,taps)
        report['inputs']=[{'path':m.path.name,'bytes':len(m.data),'sha256':sha(m.data)}for m in modules]
        recipe=json.loads((build/'source/dev/benchmarks/expert_r1_finite_summary_sep22/kernel/original_recipe.json').read_text())
        if sha((build/'original-gathered.air').read_bytes())!=recipe['immutable_performance_control']['sha256']:
            raise Refusal('Immutable original AIR differs')
        report['finite_scan_boundary_source_certificate']=finite_source_certificate(build,recipe)
        constructors=[]; matches=[]; wrappers=[]; shared=[]; local_globals=set()
        for private in (candidate,taps):
            proof=receiver_specialization_proof(original,private);constructors.append({'module':private.path.name,**proof})
            for gate in (True,False):
                reference,reference_instances=canonical_constructor_calls(original.execute(gate),proof,gate)
                reference_graph=reference.graph()[0]
                original_wrapper='flash_gathered_mpp_'+('gate_up'if gate else 'down')+'_m16_n64_sg4'
                original_storage=wrapper_storage(original,original_wrapper)
                local_globals.update(x['symbol']for x in original_storage.values())
                original_call,original_attrs=wrapper_call_graph(original,original_wrapper,gate,False,False,original_storage)
                for consumer in (False,True):
                    actual,instances=canonical_constructor_calls(private.execute(gate,consumer),proof,gate)
                    boundary=None
                    if consumer: actual,boundary=scan_boundary_projection(actual,gate,True)
                    graph,node_ids,diagnostic=actual.graph(15)
                    discrepancy=difference(reference_graph,graph)
                    record={'module':private.path.name,'role':'GU'if gate else 'down','consumer':consumer,
                            'actual_downstream_SSA_load_FP_pointer_memory_order_attributes_equal':discrepancy is None,
                            'first_difference':discrepancy,'reference_graph_sha256':sha(json.dumps(reference_graph,sort_keys=True)),
                            'actual_graph_sha256':sha(json.dumps(graph,sort_keys=True)),'graph':graph,
                            'constructor_instances':instances,'original_constructor_instances':reference_instances,
                            'declared_scan_boundary':boundary}
                    if private is taps:
                        stages=producer_stages(actual,gate);expected={15:'raw_g',16:'scaled_g',19:'bf_g'}
                        if gate:expected.update({17:'raw_u',18:'scaled_u',20:'bf_u',21:'silu',22:'combined'})
                        stores=[(r,val,ptr,base)for r,val,ptr,base in actual.stores()if id(r)in diagnostic]
                        output=[ptr for r,val,ptr,base in actual.stores()if base==7 and len(actual.value_refs(val))==1]
                        if len(output)!=1:raise Refusal('Missing one actual shipping output-store address')
                        def index(ptr):
                            fields=split_top(actual.defs[ptr]['rhs'])
                            if not fields[0].startswith('getelementptr ')or len(fields)!=3:raise Refusal('Unexpected tap/output index tree')
                            return fields[2]
                        coupled=len(stores)==len(expected)and set(base for r,val,ptr,base in stores)==set(expected)
                        details=[]
                        for r,val,ptr,base in stores:
                            refs=actual.value_refs(val);ok=base in expected and refs==[stages[expected[base]]]and index(ptr)==index(output[0])
                            details.append({'parameter':base,'role':expected.get(base),'same_actual_producer_SSA_and_output_index':ok});coupled=coupled and ok
                        record['tap_stage_and_address_coupling']=details;record['tap_coupling_match']=coupled
                    matches.append(record)
                    name='r1_finite_summary_'+('gu_'if gate else 'down_')+('consumer'if consumer else 'native')+('_tap'if private is taps else '')
                    storage=wrapper_storage(private,name)
                    call,attrs=wrapper_call_graph(private,name,gate,False,private is taps,storage,consumer)
                    storage_equal=all(storage[s]['declaration']==original_storage[s]['declaration']for s in storage)
                    wrappers.append({'module':private.path.name,'wrapper':name,'exact_threadgroup_storage_triple_equal':storage_equal,
                                     'typed_original_helper_receiver_and_pointer_bindings_equal':call==original_call,
                                     'first_difference':difference(original_call,call),'storage':storage,
                                     'whole_entry_attributes_equal':attrs==original_attrs,'entry_contains_FP_load_or_arithmetic':False})
            for name,function in original.functions.items():
                if 'gathered_mpp_execute' in name or name.startswith('flash_gathered_mpp_')or name==proof['generic']:continue
                normalized={helper_alias(n):n for n in private.functions}
                target=normalized.get(helper_alias(name))
                if target is None:raise Refusal('Missing original control/SDK helper: '+name)
                other=private.functions[target];equal=function.full()==other.full()
                shared.append({'module':private.path.name,'function':name,'complete_body_return_argument_attributes_equal':equal})
                if not equal:raise Refusal('Original control/SDK helper drift: '+name)
            for name,declaration in original.declarations.items():
                if name not in private.declarations or original.normalize(declaration)!=private.normalize(private.declarations[name]):raise Refusal('External typed call/attributes drift: '+name)
            for name,line in original.globals.items():
                if name in local_globals:continue
                if name not in private.globals or original.normalize(line)!=private.normalize(private.globals[name]):raise Refusal('Named storage/type/MPP descriptor drift: '+name)
        report['bounded_private_receiver_constructor_proofs']=constructors
        report['actual_native_consumer_and_tap_graphs']=matches
        report['typed_wrapper_storage_and_call_proofs']=wrappers
        report['original_control_scan_and_SDK_helpers']=shared
        math_equal=all(x['actual_downstream_SSA_load_FP_pointer_memory_order_attributes_equal']for x in matches)
        wrapper_equal=all(x['exact_threadgroup_storage_triple_equal']and x['typed_original_helper_receiver_and_pointer_bindings_equal']for x in wrappers)
        report['candidate_native_FP_tree_match']=math_equal and wrapper_equal
        report['taps_shipping_FP_tree_match']=math_equal and wrapper_equal and all(x.get('tap_coupling_match',True)for x in matches)
        report['finite_scan_predicate_equivalence_proved']=report['candidate_native_FP_tree_match']and report['taps_shipping_FP_tree_match']
        for x in matches:
            if not x['actual_downstream_SSA_load_FP_pointer_memory_order_attributes_equal']:report['failures'].append({'reason':'Actual downstream graph drift','module':x['module'],'role':x['role'],'consumer':x['consumer'],'difference':x['first_difference']})
            if x.get('tap_coupling_match')is False:report['failures'].append({'reason':'Actual raw/scaled/BF16 tap stage/index SSA drift','role':x['role'],'consumer':x['consumer']})
        for x in wrappers:
            if not x['exact_threadgroup_storage_triple_equal']or not x['typed_original_helper_receiver_and_pointer_bindings_equal']:report['failures'].append({'reason':'Threadgroup storage/typed wrapper binding drift','wrapper':x['wrapper'],'difference':x['first_difference']})
        for m in modules:
            if m.path.read_bytes()!=m.data:raise Refusal('Actual IR changed during certificate construction')
    except Exception as error:
        report['candidate_native_FP_tree_match']=report['taps_shipping_FP_tree_match']=report['finite_scan_predicate_equivalence_proved']=False
        report['failures'].append({'reason':'Bounded actual-byte proof could not be established','error':str(error),'exception':type(error).__name__})
    output=build/'arithmetic-audit.json';output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    passed=all(report[k]for k in ('candidate_native_FP_tree_match','taps_shipping_FP_tree_match','finite_scan_predicate_equivalence_proved'))
    print(json.dumps({'pass':passed,'candidate_native_FP_tree_match':report['candidate_native_FP_tree_match'],'taps_shipping_FP_tree_match':report['taps_shipping_FP_tree_match'],'finite_scan_predicate_equivalence_proved':report['finite_scan_predicate_equivalence_proved'],'failures':report['failures'],'report':str(output)}))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(finite_main())
