#!/usr/bin/env python3
"""Raw-Q4-only increment after every admitted original composite/22-case gate."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = 'dev/benchmarks/raw_q4_verify_worker_sep22'
PARENT_PRIVATE = 'dev/benchmarks/guard_hc_fast_composite_sep22'
SOURCE = '162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a'
PARENT_SOURCE = '38756086d4862cc9a87db70705928cb2a497450e1268bcb251bb0c0b10d8baba'
EXE = '663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438'
LIB = '7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8'
AIR = '4a671caf5641a451f5d60d5d2295b9064e4d78bf23effa74c5b86eaeeaec1003'
RECEIPT = '4d4ce0cee4e9c40351b443ec069162eb5f0098d0e2a392809bba0dd06d4c8818'
SECTION = 'raw_q4_rowpair_verify'
SCHEMA = 'mainGDNQKV-rawQ4-VerifyR4-rowpair-v1'
SCOPE = 'singleton VerifyR4 only; authenticated main GDN QKV Q4/G64/K2560/N10240; selected F32 cache and PLE excluded'
FAMILY = ';private-rawQ4-rowpair-mainGDNQKV-VerifyR4-sourceSha256='
COUNTER_SCOPE = 'process graph construction, not GPU completion'
PROGRAMS = ('policy.hpp', 'overlay.py', 'policy_cpu.cpp', 'prepare.py')
OWNERSHIP = tuple(SECTION + '.' + key for key in ('schema', 'requested', 'source_identity_sha256',
    'scope', 'qualified_candidate_AIR_sha256', 'expected_main_roles_per_VerifyR4',
    'GPU_allocation_bytes_added', 'counter_scope', 'whole_state_qualified'))

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a, b): return type(a) is type(b) and a == b
def u64(value): return type(value) is int and 0 <= value < 2 ** 64

def make_hooks(http, expected):
    if type(expected) is not bool: raise ValueError('Expected rowpair must be Boolean')
    def status(value, plan, store, execution_mode='mtp3'):
        p = value.get(SECTION) if isinstance(value, dict) else None
        if not isinstance(p, dict): return ['Raw Q4 registered profile missing']
        errors = []
        wanted = {'schema': SCHEMA, 'requested': expected, 'source_identity_sha256': SOURCE,
            'scope': SCOPE, 'qualified_candidate_AIR_sha256': AIR,
            'expected_main_roles_per_VerifyR4': 26, 'GPU_allocation_bytes_added': 0,
            'counter_scope': COUNTER_SCOPE, 'whole_state_qualified': False}
        for key, target in wanted.items():
            if not same(p.get(key), target): errors.append('Raw Q4 profile differs: ' + key)
        routes = http.get_path(value, 'identity.kernel_routes')
        if not isinstance(routes, str) or routes.count(FAMILY) != int(expected) or routes.count(FAMILY + SOURCE) != int(expected):
            errors.append('Raw Q4 enabled/source execution marker mismatch')
        calls, rows = p.get('graph_calls'), p.get('graph_rows')
        if not u64(calls) or not u64(rows) or rows != 4 * calls: errors.append('Raw Q4 cumulative counters invalid')
        elif not expected and (calls or rows): errors.append('Disabled raw Q4 recorded graph work')
        if expected and u64(calls):
            guard = http.get_path(value, 'compact_r4_preflight.guard_preflight_graph_calls')
            if not u64(guard) or calls * 48 != guard * 26: errors.append('Raw Q4 26 and guard48 cumulative counts disagree')
        return errors
    def coverage(before, after, case, require_counters=True, execution_mode='mtp3'):
        errors = []
        for side, value in (('before', before), ('after', after)):
            errors.extend(side + ': ' + error for error in status(value, None, None, execution_mode))
        for key in OWNERSHIP:
            if not http.same_json(http.get_path(before, key), http.get_path(after, key)):
                errors.append('Raw Q4 provenance changed: ' + key)
        a, b = (http.get_path(value, 'mtp.completed_cycles_by_proposed_depth') for value in (before, after))
        if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 16
                and all(u64(x) for x in a + b) and all(y >= x for x, y in zip(a, b))):
            return {}, errors + ['Raw Q4 actual-depth histogram invalid/decreasing']
        cycles = b[3] - a[3]
        caller = execution_mode == 'mtp3' and case.get('compact_scope', 'singleton-main') == 'singleton-main' and case.get('body', {}).get('response_format', {}).get('type') != 'json_schema'
        wanted = 26 * cycles if expected and caller else 0
        if not caller and cycles: errors.append('Excluded raw Q4 caller advanced singleton R4 cycles')
        deltas = {}
        for key in ('graph_calls', 'graph_rows'):
            old, new = (http.get_path(value, SECTION + '.' + key) for value in (before, after))
            if not u64(old) or not u64(new) or new < old: errors.append('Raw Q4 counter invalid/decreasing: ' + key)
            else: deltas[key] = new - old
        if len(deltas) == 2 and (deltas['graph_calls'] != wanted or deltas['graph_rows'] != 4 * wanted):
            errors.append('Raw Q4 delta differs from26 actual singleton H3 cycles')
        return {'raw_Q4_graph_counter_deltas': deltas, 'raw_Q4_expected_calls': wanted,
            'raw_Q4_actual_H3_cycle_delta': cycles, 'raw_Q4_requested': expected}, errors
    def ownership(value): return {key: http.get_path(value, key) for key in OWNERSHIP}
    return status, coverage, ownership

def authenticate(build, require_state=False):
    build = Path(build).resolve(); m = json.loads((build / 'overlay-manifest.json').read_text())
    if (m.get('schema') != 'rawQ4-GDN26-VerifyR4-narrow-worker-source-v1'
            or m.get('source_identity_sha256') != SOURCE or m.get('parent_source_identity') != PARENT_SOURCE
            or m.get('changed_paths') != ['runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm']):
        raise ValueError('Unknown raw Q4 compiled profile')
    parts = m.get('identity_parts')
    if not isinstance(parts, dict) or hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != SOURCE:
        raise ValueError('Raw Q4 source identity tuple mismatch')
    records = {r['path']: r['sha256'] for r in m['files']}
    for name in PROGRAMS:
        path = PRIVATE + '/' + name
        if digest(build / 'source' / path) != records[path] or records[path] != parts[name]:
            raise ValueError('Raw Q4 compiled program drift: ' + name)
    header = (build / 'source' / PRIVATE / 'source_identity.hpp').read_text()
    if not re.search(r'kSourceIdentitySha256\[\]\s*=\s*"' + SOURCE + '"', header):
        raise ValueError('Raw Q4 compiled identity header mismatch')
    if digest(build / 'splash-flash') != EXE or digest(build / 'splash.metallib') != LIB or digest(build / 'rawQ4-qualified.air') != AIR:
        raise ValueError('Raw Q4 exact executable/library/AIR drift')
    origin = Path(m['base']).resolve()
    if digest(origin / 'compiled-cpu-seal.json') != parts['parent_compiled_seal_sha256']:
        raise ValueError('Raw Q4 sealed composite parent drift')
    if require_state:
        path = build / 'Root-rawQ4-native-qualified.json'
        if digest(path) != RECEIPT: raise ValueError('Fresh Root raw Q4 receipt digest mismatch')
        r = json.loads(path.read_text())
        wanted = {'schema': 'rawQ4-mainGDN26-VerifyR4-current-whole-native-proof-v1', 'pass': True,
            'qualification_complete': True, 'Root_GPU_executed': True, 'source_identity_sha256': SOURCE,
            'exe_sha256': EXE, 'newlib_sha256': LIB, 'qualified_candidate_AIR_sha256': AIR,
            'frames': 26, 'repeated_frames': 54, 'bytes_compared': 10699174139, 'planes_compared': 9891,
            'backend_destroyed': True, 'rawQ4_calls': 130, 'rawQ4_rows': 520, 'guardCalls': 240,
            'guardRows': 960, 'HCcalls': 485, 'HCrows': 1940, 'pads': 485, 'actual_owned_buffer_guard_cases': 8,
            'root_report_sha256': '4b2f5726f83f045205ad09ec0d277a4b29782c28cd0e827bc0f6d801b6d53112',
            'oracle_sha256': '2feffceaadba8883e52553657212b8a746b8859a68a991c2eaac9d6f2409986d',
            'ORIGINAL22_or_normal_performance_qualified': False}
        if any(not same(r.get(key), value) for key, value in wanted.items()) or not isinstance(r.get('root_report_path'), str) or not r['root_report_path'].strip():
            raise ValueError('Fresh Root receipt does not bind the qualified raw Q4 whole worker')
    return build, origin

def load(build, expected=False, require_state=False):
    build, origin = authenticate(build, require_state)
    path = origin / 'source' / PARENT_PRIVATE / 'semantic_quality.py'
    spec = importlib.util.spec_from_file_location('_rawQ4_registered_composite', path)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    # Parent admission and every original active composite/b32/compact/base gate
    # remain mandatory. Its receipt admits its own origin, never this new worker.
    runner = parent.load(origin, require_state=require_state)
    return parent.install(runner, make_hooks(runner.http, expected))

def runtime_args(rest, build):
    explicit = False
    for i, token in enumerate(rest):
        option = token.split('=', 1)[0]
        if len(option) > 2 and option.startswith('--') and '--runtime-build'.startswith(option) and option != '--runtime-build':
            raise ValueError('Measured runtime selection requires complete --runtime-build')
        if token == '--runtime-build':
            if i + 1 == len(rest): raise ValueError('Missing measured runtime-build value')
            value = rest[i + 1]
        elif token.startswith('--runtime-build='): value = token.split('=', 1)[1]
        else: continue
        explicit = True
        if not value or Path(value).resolve() != build: raise ValueError('Every measured runtime-build must equal authenticated build')
    return rest if explicit else rest + ['--runtime-build', str(build)]

def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--expected-rowpair', choices=('0', '1'), required=True)
    args, rest = parser.parse_known_args(argv)
    measured = bool(rest and rest[0] == 'measure')
    if measured: rest = runtime_args(rest, args.build.resolve())
    return load(args.build, expected=args.expected_rowpair == '1', require_state=measured).main(rest)

if __name__ == '__main__': raise SystemExit(main())
