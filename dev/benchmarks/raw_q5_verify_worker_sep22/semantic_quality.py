#!/usr/bin/env python3
"""Private Q5 hooks after the sealed Q4/composite/original22 quality gates.

This adapter reads only program/source files and small qualification receipts.
It never opens the receipt's state-report path or any model/capture payload.
Unregistered child artifacts and missing fresh combined proof fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = 'dev/benchmarks/raw_q5_verify_worker_sep22'
# This private registry admits only the sealed combined worker.
SOURCE = '90d42b5ce8edd6c7254b6f9286628a7ffa7a6eede53fa3058eba8b519e9a64d6'
EXE = '613e4dfe6a9b429cabf8fbd2270b5857dbc601cde6d1b3af10c46fec1b8095c0'
LIB = '8fffe24fd4d99174cc522dcef6090b5b2e72e36f26e5c78afe9df46a118d4246'
Q4_HELPER_SHA256 = 'be9f0a262eebd1ea3a67e9b6ace88571f5cbeab3dc10c5dfb36f3405f67d772e'
Q4_HELPER_SEAL_SHA256 = '5ef801ae5714622705d910454b8320ea74c242e4d6322f485deadc2b9a901be2'
Q4_HELPER_PRIVATE = 'dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py'
PARENT_SOURCE = '162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a'
PARENT_EXE = '663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438'
PARENT_LIB = '7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8'
PARENT_RECEIPT = '4d4ce0cee4e9c40351b443ec069162eb5f0098d0e2a392809bba0dd06d4c8818'
Q4_AIR = '4a671caf5641a451f5d60d5d2295b9064e4d78bf23effa74c5b86eaeeaec1003'
AIR = '99d57fd78957cfcc68ecc30d426c754b0dd702e42a32acef0008155234482a69'
SECTION = 'raw_q5_rowpair_verify'
SCHEMA = 'mainGDNout-rawQ5-VerifyR4-rowpair-v1'
SCOPE = 'singleton VerifyR4 only; authenticated main GDN output Q5/G128/K6144/N2560; selected F32 cache and all other roles excluded'
FAMILY = ';private-rawQ5-rowpair-mainGDNout-VerifyR4-sourceSha256='
COUNTER_SCOPE = 'process graph construction, not GPU completion'
MANIFEST_SCHEMA = 'rawQ4Q5-GDN36-VerifyR4-narrow-worker-source-v1'
SOURCE_SCOPE = 'MAIN GDN output36 only; singleton VerifyR4 actualNULLtileRAW Q5G128 K6144N2560; allotherpaths unchanged; originalbacking'
PROOF_FILE = 'Root-rawQ4Q5-native-qualified.json'
PROOF_SCHEMA = 'rawQ4Q5-mainGDN26-output36-VerifyR4-current-whole-native-proof-v1'
PROGRAMS = ('policy.hpp', 'overlay.py', 'policy_cpu.cpp', 'prepare.py', 'PLAN.json')
COUNTERS = ('graph_calls', 'graph_rows')
OWNERSHIP = tuple(SECTION + '.' + key for key in (
    'schema', 'requested', 'source_identity_sha256', 'scope',
    'qualified_candidate_AIR_sha256', 'expected_main_roles_per_VerifyR4',
    'GPU_allocation_bytes_added', 'counter_scope', 'whole_state_qualified'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    return type(a) is type(b) and a == b


def u64(value):
    return type(value) is int and 0 <= value < 2 ** 64


def sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def document(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Qualification metadata must be a JSON object')
    return value


def registered():
    if not all(sha256(value) for value in (SOURCE, EXE, LIB, Q4_HELPER_SHA256)):
        raise ValueError('Combined raw Q4/Q5 compiled profile is not registered')


def make_hooks(http, expected, source=None):
    if type(expected) is not bool:
        raise ValueError('Expected Q5 rowpair must be Boolean')
    source = SOURCE if source is None else source
    if not sha256(source):
        raise ValueError('Required combined Q5 source identity is invalid')

    def status(status, plan, store, execution_mode='mtp3'):
        p = status.get(SECTION) if isinstance(status, dict) else None
        if not isinstance(p, dict):
            return ['Raw Q5 registered profile missing']
        errors = []
        wanted = {
            'schema': SCHEMA, 'requested': expected, 'source_identity_sha256': source,
            'scope': SCOPE, 'qualified_candidate_AIR_sha256': AIR,
            'expected_main_roles_per_VerifyR4': 36, 'GPU_allocation_bytes_added': 0,
            'counter_scope': COUNTER_SCOPE, 'whole_state_qualified': False}
        for key, target in wanted.items():
            if not same(p.get(key), target):
                errors.append('Raw Q5 profile differs: ' + key)
        routes = http.get_path(status, 'identity.kernel_routes')
        markers = re.findall(re.escape(FAMILY) + r'([0-9a-f]{64})(?=;|$)', routes) if isinstance(routes, str) else []
        if (not isinstance(routes, str) or routes.count(FAMILY) != int(expected)
                or markers != ([source] if expected else [])):
            errors.append('Raw Q5 enabled/source execution marker mismatch')
        identity = status.get('identity')
        if isinstance(identity, dict) and SECTION in identity:
            errors.append('Raw Q5 process counters/profile must remain outside identity')
        # Every combined comparison retains the admitted active Q4 profile.
        q4 = status.get('raw_q4_rowpair_verify')
        if (not isinstance(q4, dict) or q4.get('requested') is not True
                or q4.get('source_identity_sha256') != PARENT_SOURCE):
            errors.append('Raw Q5 comparison requires the registered active Q4 parent')
        calls, rows = p.get('graph_calls'), p.get('graph_rows')
        if not u64(calls) or not u64(rows) or rows != 4 * calls:
            errors.append('Raw Q5 cumulative counters invalid')
        elif not expected and (calls or rows):
            errors.append('Disabled raw Q5 recorded graph work')
        elif expected:
            q4_calls = http.get_path(status, 'raw_q4_rowpair_verify.graph_calls')
            q4_rows = http.get_path(status, 'raw_q4_rowpair_verify.graph_rows')
            guard = http.get_path(status, 'compact_r4_preflight.guard_preflight_graph_calls')
            guard_rows = http.get_path(status, 'compact_r4_preflight.guard_preflight_graph_rows')
            if (not all(u64(x) for x in (q4_calls, q4_rows, guard, guard_rows))
                    or q4_rows != 4 * q4_calls or guard_rows != 4 * guard
                    or calls * 26 != q4_calls * 36 or calls * 48 != guard * 36):
                errors.append('Raw Q5 36/Q4 26/guard48 cumulative counts disagree')
        return errors

    def coverage(before, after, case, require_counters=True, execution_mode='mtp3'):
        errors = []
        for side, value in (('before', before), ('after', after)):
            errors.extend(side + ': ' + error for error in status(value, None, None, execution_mode))
        for key in OWNERSHIP:
            if not http.same_json(http.get_path(before, key), http.get_path(after, key)):
                errors.append('Raw Q5 provenance changed: ' + key)
        a, b = (http.get_path(value, 'mtp.completed_cycles_by_proposed_depth') for value in (before, after))
        if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 16
                and all(u64(x) for x in a + b) and all(y >= x for x, y in zip(a, b))):
            return {}, errors + ['Raw Q5 actual-depth histogram invalid/decreasing']
        cycles = b[3] - a[3]
        if not isinstance(case, dict) or not isinstance(case.get('body', {}), dict):
            return {}, errors + ['Raw Q5 case/body metadata invalid']
        response = case.get('body', {}).get('response_format', {})
        if not isinstance(response, dict):
            return {}, errors + ['Raw Q5 response-format metadata invalid']
        caller = (execution_mode == 'mtp3'
            and case.get('compact_scope', 'singleton-main') == 'singleton-main'
            and response.get('type') != 'json_schema')
        wanted = 36 * cycles if expected and caller else 0
        if not caller and cycles:
            errors.append('Excluded raw Q5 caller advanced singleton R4 cycles')
        deltas = {}
        for key in COUNTERS:
            old, new = (http.get_path(value, SECTION + '.' + key) for value in (before, after))
            if not u64(old) or not u64(new) or new < old:
                errors.append('Raw Q5 counter invalid/decreasing: ' + key)
            else:
                deltas[key] = new - old
        if len(deltas) == 2 and (deltas['graph_calls'] != wanted or deltas['graph_rows'] != 4 * wanted):
            errors.append('Raw Q5 delta differs from36 actual singleton H3 cycles')
        return {
            'raw_Q5_graph_counter_deltas': deltas, 'raw_Q5_expected_calls': wanted,
            'raw_Q5_actual_H3_cycle_delta': cycles, 'raw_Q5_requested': expected}, errors

    def ownership(status):
        return {key: http.get_path(status, key) for key in OWNERSHIP}

    return status, coverage, ownership


def install(runner, hooks):
    # Retain every old gate and forward its original positional/keyword API.
    old_status, old_coverage = runner.gate_status, runner.coverage
    old_ownership = runner.ownership_policy
    status_hook, coverage_hook, ownership_hook = hooks

    def gate_status(status, plan, store, execution_mode='mtp3'):
        # The frozen Q4 hook calls its first argument "value". Bind the public
        # keyword API here and pass identical values positionally through it.
        return (old_status(status, plan, store, execution_mode=execution_mode)
                + status_hook(status, plan, store, execution_mode=execution_mode))

    def coverage(*args, **kwargs):
        details, errors = old_coverage(*args, **kwargs)
        extra, added = coverage_hook(*args, **kwargs)
        return {**details, **extra}, errors + added

    def ownership_policy(*args, **kwargs):
        return {**old_ownership(*args, **kwargs), **ownership_hook(*args, **kwargs)}

    runner.gate_status, runner.coverage = gate_status, coverage
    runner.ownership_policy = ownership_policy
    return runner


def require_native_state(build):
    registered()
    proof = document(Path(build) / PROOF_FILE)
    wanted = {
        'schema': PROOF_SCHEMA, 'pass': True, 'qualification_complete': True,
        'Root_GPU_executed': True, 'source_identity_sha256': SOURCE,
        'exe_sha256': EXE, 'newlib_sha256': LIB,
        'qualified_candidate_AIR_sha256': AIR, 'rawQ4_qualified_candidate_AIR_sha256': Q4_AIR,
        'frames': 26, 'repeated_frames': 54, 'backend_destroyed': True,
        'rawQ4_calls': 130, 'rawQ4_rows': 520, 'rawQ5_calls': 180, 'rawQ5_rows': 720,
        'guardCalls': 240, 'guardRows': 960, 'HCcalls': 485, 'HCrows': 1940,
        'pads': 485, 'actual_owned_buffer_guard_cases': 8,
        'allocation_guard_violations': 0, 'allocation_guard_axes_passed': 6,
        'ORIGINAL22_or_normal_performance_qualified': False}
    if any(not same(proof.get(key), value) for key, value in wanted.items()):
        raise ValueError('Fresh Root combined receipt does not bind the current Q4/Q5 whole worker')
    if (not isinstance(proof.get('root_report_path'), str) or not proof['root_report_path'].strip()
            or not sha256(proof.get('root_report_sha256')) or not sha256(proof.get('oracle_sha256'))):
        raise ValueError('Fresh Root combined receipt provenance is missing or malformed')


def authenticate(build, require_state=False):
    registered()
    build = Path(build).resolve()
    manifest = document(build / 'overlay-manifest.json')
    wanted = {
        'schema': MANIFEST_SCHEMA, 'source_identity_sha256': SOURCE,
        'parent_source_identity': PARENT_SOURCE, 'qualified_candidate_AIR_sha256': AIR,
        'eligible_main_GDN_output_roles': 36, 'eligible_parent_main_GDN_QKV_roles': 26,
        'changed_paths': ['runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm']}
    if any(not same(manifest.get(key), value) for key, value in wanted.items()):
        raise ValueError('Unknown combined raw Q4/Q5 compiled profile')
    parts = manifest.get('identity_parts')
    keys = set(PROGRAMS) | {
        'parent_source_identity', 'parent_compiled_seal_sha256',
        'parent_current_native_receipt_sha256', 'qualified_candidate_AIR_sha256', 'scope'}
    if (not isinstance(parts, dict) or set(parts) != keys
            or parts['parent_source_identity'] != PARENT_SOURCE
            or parts['parent_current_native_receipt_sha256'] != PARENT_RECEIPT
            or parts['qualified_candidate_AIR_sha256'] != AIR or parts['scope'] != SOURCE_SCOPE
            or not all(sha256(parts[key]) for key in keys - {'scope'})
            or hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != SOURCE):
        raise ValueError('Combined raw Q4/Q5 source identity tuple mismatch')
    files = manifest.get('files')
    if not isinstance(files, list):
        raise ValueError('Combined source file registry is missing')
    records = {}
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get('path'), str) or not sha256(record.get('sha256')):
            raise ValueError('Combined source file registry is malformed')
        name = record['path']
        if not name or Path(name).is_absolute() or '..' in Path(name).parts or name in records:
            raise ValueError('Combined source registry has unsafe/duplicate paths')
        records[name] = record['sha256']
    for name in PROGRAMS:
        path = PRIVATE + '/' + name
        if records.get(path) != parts[name] or digest(build / 'source' / path) != parts[name]:
            raise ValueError('Combined compiled program drift: ' + name)
    header = (build / 'source' / PRIVATE / 'source_identity.hpp').read_text()
    if not re.search(r'kSourceIdentitySha256\[\]\s*=\s*"' + SOURCE + '"', header):
        raise ValueError('Combined compiled identity header mismatch')
    if digest(build / 'splash-flash') != EXE or digest(build / 'splash.metallib') != LIB or digest(build / 'rawQ5-qualified.air') != AIR:
        raise ValueError('Combined exact executable/library/Q5 AIR drift')
    if not isinstance(manifest.get('base'), str) or not manifest['base'].strip():
        raise ValueError('Combined Q4 parent path missing')
    origin = Path(manifest['base']).resolve()
    if (digest(origin / 'compiled-cpu-seal.json') != parts['parent_compiled_seal_sha256']
            or digest(origin / 'Root-rawQ4-native-qualified.json') != PARENT_RECEIPT
            or digest(origin / 'splash-flash') != PARENT_EXE or digest(origin / 'splash.metallib') != PARENT_LIB):
        raise ValueError('Combined sealed Q4 parent drift')
    if require_state:
        require_native_state(build)
    return build, origin


def load_q4_helper(origin):
    path = Path(origin) / 'source' / Q4_HELPER_PRIVATE
    if (digest(path) != Q4_HELPER_SHA256
            or digest(Path(origin) / 'rawQ4-semantic-source-seal.json') != Q4_HELPER_SEAL_SHA256):
        raise ValueError('Refreshed sealed Q4 quality adapter drift')
    spec = importlib.util.spec_from_file_location('_rawQ5_sealed_registered_Q4', path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    return parent


def load(build, expected=False, require_state=False):
    if type(expected) is not bool:
        raise ValueError('Expected Q5 rowpair must be Boolean')
    build, origin = authenticate(build, require_state=require_state)
    parent = load_q4_helper(origin)
    runner = parent.load(origin, expected=True, require_state=True)
    return install(runner, make_hooks(runner.http, expected))


def runtime_args(rest, build):
    explicit = False
    for i, token in enumerate(rest):
        option = token.split('=', 1)[0]
        if len(option) > 2 and option.startswith('--') and '--runtime-build'.startswith(option) and option != '--runtime-build':
            raise ValueError('Measured runtime selection requires complete --runtime-build')
        if token == '--runtime-build':
            if i + 1 == len(rest):
                raise ValueError('Missing measured runtime-build value')
            value = rest[i + 1]
        elif token.startswith('--runtime-build='):
            value = token.split('=', 1)[1]
        else:
            continue
        explicit = True
        if not value or Path(value).resolve() != build:
            raise ValueError('Every measured runtime-build must equal authenticated combined build')
    return rest if explicit else rest + ['--runtime-build', str(build)]


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--expected-rowpair', choices=('0', '1'), required=True)
    args, rest = parser.parse_known_args(argv)
    measured = bool(rest and rest[0] == 'measure')
    if measured:
        rest = runtime_args(rest, args.build.resolve())
    return load(args.build, expected=args.expected_rowpair == '1', require_state=measured).main(rest)


if __name__ == '__main__':
    raise SystemExit(main())
