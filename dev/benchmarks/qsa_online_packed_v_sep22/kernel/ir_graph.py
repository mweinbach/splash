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
    # Exact one-to-one private source identifiers and their Itanium lengths.
    mappings = [
        ('bulk_qsa_mpp_failure',), ('bulk_qsa_mpp_visible',), ('bulk_qsa_mpp_token',),
        ('bulk_qsa_online_partition',), ('bulk_rowtile_error',), ('BulkQueryStorage',),
        ('bulk_rowtile_online',), ('BulkOnlineScratch',), ('BulkTemporalScratch',),
        ('BulkAttentionScratch',), ('bulk_temporal_sg8_error',),
        ('BulkTemporalSG8QueryStorage',), ('bulk_temporal_sg8_online',), ('bulk_qsa_fast_gate',),
    ]
    for role in ('native', 'candidate'):
        for tap in ('', '_tap'):
            prefix = 'qsa_online_packed_v_' + role + tap + '_'
            for (old,) in mappings:
                private = prefix + old
                def adjust_identifier(match):
                    if int(match.group(1)) != len(private):
                        raise Refusal('Unexpected private Itanium identifier length: ' + match.group(0))
                    return str(len(old)) + old
                text = re.sub(r'(\d+)' + re.escape(private),
                              adjust_identifier, text)
                text = re.sub(r'\b' + re.escape(private) + r'\b', old, text)
        for phase, original in (('early','flash_qsa_mpp_prefill_bulk_early_2048'),
                                ('temporal','flash_qsa_mpp_prefill_bulk_temporal_sg8_2048'),
                                ('reduce','flash_qsa_fast_prefill_bulk_reduce_2048')):
            for tap in ('_tap',''):
                private = 'qsa_online_packed_v_' + role + '_' + phase + tap
                def adjust_entry_identifier(match):
                    if int(match.group(1)) != len(private):
                        raise Refusal('Unexpected entry Itanium identifier length: ' + match.group(0))
                    return str(len(original)) + original
                text = re.sub(r'(\d+)' + re.escape(private), adjust_entry_identifier, text)
                text = re.sub(r'\b' + re.escape(private) + r'\b', original, text)
    if 'qsa_online_packed_v_' in text.replace('qsa_online_packed_v_pack', 'PACK_ENTRY'):
        raise Refusal('Unknown private identifier cannot be normalized: ' + text[:200])
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

    def execute(self, gate, candidate=False):
        names = [name for name in self.functions if 'gathered_mpp_execute' in name
                 and ('ILb1' if gate else 'ILb0') in name and ('_candidate_' in name)==candidate]
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
