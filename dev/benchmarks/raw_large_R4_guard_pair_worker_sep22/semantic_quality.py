#!/usr/bin/env python3
"""Private R4 guard-pair source admission and strict original22 gate extension.

CPU source authentication is separate from trained-state admission. Child
performance remains blocked until Root registers the real compiled artifacts
and a fresh trained-third-R4 receipt. No tensor/report path is opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

# Source admission imports frozen parent programs without creating bytecode
# cache files in their trees.
sys.dont_write_bytecode = True

# Both the live source and its sealed extras copy use this explicitly known
# workspace. A sealed helper's parents[3] is the snapshot's source directory.
ROOT = Path('/Users/mweinbach/Projects/splash')
PRIVATE = 'dev/benchmarks/raw_large_R4_guard_pair_worker_sep22'
PARENT_SOURCE = '162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a'
PARENT_EXE = '663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438'
PARENT_LIB = '7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8'
PARENT_RECEIPT = '4d4ce0cee4e9c40351b443ec069162eb5f0098d0e2a392809bba0dd06d4c8818'
PARENT_HELPER_SHA = 'be9f0a262eebd1ea3a67e9b6ace88571f5cbeab3dc10c5dfb36f3405f67d772e'
PARENT_HELPER_SEAL_SHA = '5ef801ae5714622705d910454b8320ea74c242e4d6322f485deadc2b9a901be2'
PARENT_HELPER_PRIVATE = 'dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py'
MODEL = 'ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e'
SECTION = 'raw_large_R4_guard_pair'
SCHEMA = 'main-RAW-large-VerifyR4-guard-pair-v1'
SCOPE = 'singleton main VerifyR4; cache-member selective NULL RAW; constructor-authenticated GDN qkv/z/out, QSA q/o and first PLE key; legacy GDN26 Q4 takes precedence'
FAMILY = ';private-raw-large-R4-guard-pair-87-main-roles-sourceSha256='
COUNTER_SCOPE = 'successful process graph construction, not GPU completion; exclusions count refused add attempts'
MANIFEST_SCHEMA = 'raw-large-R4-private-worker-CPU-v1'
PROGRAMS = ('policy.hpp', 'overlay.py', 'policy_cpu.cpp', 'prepare.py')
AIR_FILE = 'raw-large-R4.air'
# Root fills these with genuine sealed program/proof metadata. None is not an
# alternate admitted profile, nor a fabricated positive qualification receipt.
SOURCE = '31ea922e53327f5b62f8c6f3b9263c4052700f406d15734e5a23003efa6a78bb'
EXE = 'd47cb10f578adabbfa9540f3634f8c792a5b3e2234edb49afe70e1e32891364a'
LIB = 'b39c1b41213f53d4302cd532c6e368ed72df344a453723b9dfc4896540396998'
AIR = 'fd3ccf64ddc6427aae052d10f4620ec741838ed139538eba0768b8f439a8a64e'
PROOF_FILE = PROOF_SCHEMA = PROOF_SHA256 = None
PROOF_REQUIRED = None
STATIC = {
    'schema': SCHEMA, 'scope': SCOPE, 'expected_RAW_roles_per_VerifyR4': 113,
    'legacy_Q4_precedence_roles': 26, 'expected_new_calls_per_actual_H3': 87,
    'physical_rows': 4, 'GPU_allocation_bytes_added': 0,
    'new_residency_or_backing': False, 'numerical_policy_changed': False,
    'whole_state_qualified': False, 'counter_scope': COUNTER_SCOPE}
COUNTERS = ('graph_calls', 'graph_rows', 'excluded_context_graph_calls')
OWNERSHIP = tuple(SECTION + '.' + key for key in (
    *STATIC, 'requested', 'source_identity_sha256', 'authenticated_RAW_roles', 'authenticated_new_roles'))
IDENTITY = ('identity.source', 'identity.loaded_model_layout_sha256',
    'identity.target_numerical_derivative_sha256', 'identity.kernel_routes')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def u64(value):
    return type(value) is int and 0 <= value < 2 ** 64


def sha256(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def document(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Source/qualification metadata must be an object')
    return value


def make_hooks(http, expected, source=None):
    if type(expected) is not bool:
        raise ValueError('Expected guard-pair selection must be Boolean')
    source = SOURCE if source is None else source
    if not sha256(source):
        raise ValueError('Required compiled guard-pair source identity is invalid')

    def status(status, plan, store, execution_mode='mtp3'):
        profile = status.get(SECTION) if isinstance(status, dict) else None
        if not isinstance(profile, dict):
            return ['R4 guard-pair registered profile missing']
        wanted = {**STATIC, 'requested': expected, 'source_identity_sha256': source,
            'authenticated_RAW_roles': 113 if expected else 0,
            'authenticated_new_roles': 87 if expected else 0}
        errors = ['R4 guard-pair profile differs: ' + key for key, value in wanted.items()
            if not same(profile.get(key), value)]
        routes = http.get_path(status, 'identity.kernel_routes')
        markers = re.findall(re.escape(FAMILY) + r'([0-9a-f]{64})(?=;|$)', routes) if isinstance(routes, str) else []
        if not isinstance(routes, str) or routes.count(FAMILY) != int(expected) or markers != ([source] if expected else []):
            errors.append('R4 guard-pair enabled/own-source marker mismatch')
        identity = status.get('identity')
        if isinstance(identity, dict) and SECTION in identity:
            errors.append('R4 guard-pair process counters/profile must remain outside identity')
        if http.get_path(status, 'identity.source') != MODEL:
            errors.append('R4 guard-pair original mathematical model source differs')
        for key in IDENTITY[1:3]:
            if not sha256(http.get_path(status, key)):
                errors.append('R4 guard-pair actual numeric identity unavailable: ' + key)
        if (http.get_path(status, 'raw_q4_rowpair_verify.requested') is not True
                or http.get_path(status, 'raw_q4_rowpair_verify.source_identity_sha256') != PARENT_SOURCE):
            errors.append('R4 guard-pair requires the registered active legacy Q4 parent')
        counters = {key: profile.get(key) for key in COUNTERS}
        if not all(u64(value) for value in counters.values()):
            errors.append('R4 guard-pair counters missing/invalid')
        else:
            calls, rows = counters['graph_calls'], counters['graph_rows']
            if rows != 4 * calls or counters['excluded_context_graph_calls']:
                errors.append('R4 guard-pair physical rows/refused-context counters differ')
            if not expected and (calls or rows):
                errors.append('Disabled R4 guard-pair constructed graph work')
            if expected:
                old = http.get_path(status, 'raw_q4_rowpair_verify.graph_calls')
                old_rows = http.get_path(status, 'raw_q4_rowpair_verify.graph_rows')
                guard = http.get_path(status, 'compact_r4_preflight.guard_preflight_graph_calls')
                guard_rows = http.get_path(status, 'compact_r4_preflight.guard_preflight_graph_rows')
                if (not all(u64(value) for value in (old, old_rows, guard, guard_rows))
                        or old_rows != 4 * old or guard_rows != 4 * guard
                        or calls * 26 != old * 87 or calls * 48 != guard * 87):
                    errors.append('R4 guard-pair 87/legacy26/guard48 graph census disagrees')
        return errors

    def coverage(before, after, case, require_counters=True, execution_mode='mtp3'):
        errors = []
        for side, value in (('before', before), ('after', after)):
            errors.extend(side + ': ' + error for error in status(value, None, None, execution_mode))
        for key in (*OWNERSHIP, *IDENTITY):
            if not http.same_json(http.get_path(before, key), http.get_path(after, key)):
                errors.append('R4 guard-pair actual static/numeric identity changed: ' + key)
        a, b = (http.get_path(value, 'mtp.completed_cycles_by_proposed_depth') for value in (before, after))
        if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 16
                and all(u64(value) for value in a + b) and all(y >= x for x, y in zip(a, b))):
            return {}, errors + ['R4 guard-pair actual-depth histogram invalid/decreasing']
        if not isinstance(case, dict) or not isinstance(case.get('body', {}), dict):
            return {}, errors + ['R4 guard-pair case/body metadata invalid']
        response = case.get('body', {}).get('response_format', {})
        if not isinstance(response, dict):
            return {}, errors + ['R4 guard-pair response-format metadata invalid']
        cycles = b[3] - a[3]
        caller = (execution_mode == 'mtp3' and case.get('compact_scope', 'singleton-main') == 'singleton-main'
            and response.get('type') != 'json_schema')
        wanted = 87 * cycles if expected and caller else 0
        if not caller and cycles:
            errors.append('Excluded R4 guard-pair context advanced singleton H3 completions')
        deltas = {}
        for key in COUNTERS:
            old, new = (http.get_path(value, SECTION + '.' + key) for value in (before, after))
            if not u64(old) or not u64(new) or new < old:
                errors.append('R4 guard-pair graph counter invalid/decreasing: ' + key)
            else:
                deltas[key] = new - old
        if len(deltas) == 3 and (deltas['graph_calls'] != wanted or deltas['graph_rows'] != 4 * wanted or deltas['excluded_context_graph_calls']):
            errors.append('R4 guard-pair delta differs from87 actual eligible completed H3 cycles')
        return {'raw_large_R4_graph_counter_deltas': deltas,
            'raw_large_R4_expected_calls': wanted, 'raw_large_R4_actual_H3_cycle_delta': cycles,
            'raw_large_R4_identity_relationship': {
                'original_model_source_identity_sha256': MODEL,
                'new_kernel_route_source_identity_sha256': source,
                'numerical_policy_changed': False,
                'identity_addition': 'conditional own-source marker in kernel_routes; inherited numeric identity retained'},
            'raw_large_R4_counter_scope': COUNTER_SCOPE}, errors

    def ownership(status):
        return {key: http.get_path(status, key) for key in (*OWNERSHIP, *IDENTITY)}

    return status, coverage, ownership


def install(runner, hooks):
    old_status, old_coverage, old_ownership = runner.gate_status, runner.coverage, runner.ownership_policy
    status_hook, coverage_hook, ownership_hook = hooks
    def gate_status(status, plan, store, execution_mode='mtp3'):
        # Normalize the public keyword API before the frozen Q4 value-named hook.
        return old_status(status, plan, store, execution_mode=execution_mode) + status_hook(status, plan, store, execution_mode=execution_mode)
    def coverage(*args, **kwargs):
        details, errors = old_coverage(*args, **kwargs)
        extra, added = coverage_hook(*args, **kwargs)
        return {**details, **extra}, errors + added
    def ownership_policy(*args, **kwargs):
        return {**old_ownership(*args, **kwargs), **ownership_hook(*args, **kwargs)}
    runner.gate_status, runner.coverage, runner.ownership_policy = gate_status, coverage, ownership_policy
    return runner


def records(items):
    if not isinstance(items, list):
        raise ValueError('Compiled file/artifact registry missing')
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str) or not sha256(item.get('sha256')):
            raise ValueError('Compiled file/artifact registry malformed')
        path = item['path']
        if not path or Path(path).is_absolute() or '..' in Path(path).parts or path in result:
            raise ValueError('Compiled file/artifact registry path unsafe/duplicate')
        result[path] = item['sha256']
    return result


def registered(binding):
    expected = {'source_identity_sha256': SOURCE, 'exe_sha256': EXE, 'lib_sha256': LIB, 'candidate_AIR_sha256': AIR}
    if not all(sha256(value) for value in expected.values()) or any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError('Current trained-R4 child program profile is not registered')
    if (not isinstance(PROOF_FILE, str) or not PROOF_FILE or Path(PROOF_FILE).name != PROOF_FILE
            or not isinstance(PROOF_SCHEMA, str) or not PROOF_SCHEMA or not sha256(PROOF_SHA256)
            or not isinstance(PROOF_REQUIRED, dict) or not PROOF_REQUIRED):
        raise ValueError('Fresh trained-third-R4 proof contract is not registered')


def require_native_state(build, binding):
    # Provisional graph-construction counts can lead completed H3 cycles. The
    # trained oracle owner supplies its actual receipt fields; none are inferred
    # from histogram/layer multiplications or an inherited ordinaryVerify proof.
    registered(binding)
    path = Path(build) / PROOF_FILE
    if digest(path) != PROOF_SHA256:
        raise ValueError('Fresh trained-third-R4 Root receipt digest differs')
    proof = document(path)
    wanted = {**PROOF_REQUIRED, 'schema': PROOF_SCHEMA, 'pass': True,
        'qualification_complete': True, 'Root_GPU_executed': True,
        'source_identity_sha256': SOURCE, 'exe_sha256': EXE, 'newlib_sha256': LIB,
        'candidate_AIR_sha256': AIR}
    if any(not same(proof.get(key), value) for key, value in wanted.items()):
        raise ValueError('Fresh trained-third-R4 Root receipt does not bind this child/state scope')
    if (not isinstance(proof.get('root_report_path'), str) or not proof['root_report_path'].strip()
            or not sha256(proof.get('root_report_sha256')) or not sha256(proof.get('oracle_sha256'))):
        raise ValueError('Fresh trained-third-R4 proof provenance missing/malformed')


def authenticate(build, require_state=False):
    build = Path(build).resolve()
    manifest = document(build / 'overlay-manifest.json')
    wanted = {'schema': MANIFEST_SCHEMA, 'pass': True,
        'private_header_consumers': ['FlashForward', 'FlashWorker'],
        'other48_host_and_Core4_unchanged': True, 'public_headers_changed': False,
        'original_ordered_AIR_relink_exact754': True, 'new_shipping_AIR_count': 1,
        'old_qualified_26Q4_precedence_retained': True, 'new_potential_main_roles': 87,
        'new_weight_cache_or_GPU_owner_bytes': 0, 'Metal_math_compiles': 0,
        'GPU_work': False, 'model_token_tensor_capture_or_generation_payload_reads': 0,
        'actual_trained_state_or_performance_or_Original22_qualified': False}
    if any(not same(manifest.get(key), value) for key, value in wanted.items()):
        raise ValueError('Unknown R4 guard-pair compiled source profile')
    parts = manifest.get('identity_parts')
    keys = set(PROGRAMS) | {'parent_source_identity', 'parent_compiled_seal_sha256', 'candidate_AIR_sha256', 'component_math_audit_sha256'}
    if (not isinstance(parts, dict) or set(parts) != keys or parts.get('parent_source_identity') != PARENT_SOURCE
            or not all(sha256(parts[key]) for key in keys)):
        raise ValueError('R4 guard-pair source identity tuple malformed/unknown parent')
    identity = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if manifest.get('source_identity_sha256') != identity or (SOURCE is not None and identity != SOURCE):
        raise ValueError('R4 guard-pair compiled source identity mismatch')
    files = records(manifest.get('files'))
    for name in PROGRAMS:
        path = PRIVATE + '/' + name
        if files.get(path) != parts[name] or digest(build / 'source' / path) != parts[name]:
            raise ValueError('R4 guard-pair compiled program drift: ' + name)
    header = (build / 'source' / PRIVATE / 'source_identity.hpp').read_text()
    if not re.search(r'kSourceIdentitySha256\[\]\s*=\s*"' + identity + '"', header):
        raise ValueError('R4 guard-pair compiled identity header differs')
    artifacts = records(manifest.get('artifacts'))
    if set(artifacts) != {'splash-flash', 'splash.metallib', 'policy-CPU'}:
        raise ValueError('R4 guard-pair executable/library/CPU closure missing')
    for path, value in artifacts.items():
        if digest(build / path) != value:
            raise ValueError('R4 guard-pair program artifact drift: ' + path)
    if digest(build / AIR_FILE) != parts['candidate_AIR_sha256']:
        raise ValueError('R4 guard-pair current shipping AIR drift')
    if not isinstance(manifest.get('base'), str) or not manifest['base'].strip():
        raise ValueError('R4 guard-pair immutable parent path missing')
    origin = Path(manifest['base']).resolve()
    if (digest(origin / 'compiled-cpu-seal.json') != parts['parent_compiled_seal_sha256']
            or digest(origin / 'splash-flash') != PARENT_EXE or digest(origin / 'splash.metallib') != PARENT_LIB
            or digest(origin / 'Root-rawQ4-native-qualified.json') != PARENT_RECEIPT):
        raise ValueError('R4 guard-pair sealed Q4 parent drift')
    binding = {'source_identity_sha256': identity, 'exe_sha256': artifacts['splash-flash'],
        'lib_sha256': artifacts['splash.metallib'], 'candidate_AIR_sha256': parts['candidate_AIR_sha256']}
    for key, expected in (('exe_sha256', EXE), ('lib_sha256', LIB), ('candidate_AIR_sha256', AIR)):
        if expected is not None and binding[key] != expected:
            raise ValueError('Unknown R4 guard-pair registered artifact: ' + key)
    if require_state:
        require_native_state(build, binding)
    return build, origin, binding


def load_parent(origin):
    path = Path(origin) / 'source' / PARENT_HELPER_PRIVATE
    if digest(path) != PARENT_HELPER_SHA or digest(Path(origin) / 'rawQ4-semantic-source-seal.json') != PARENT_HELPER_SEAL_SHA:
        raise ValueError('R4 guard-pair sealed Q4 semantic adapter drift')
    spec = importlib.util.spec_from_file_location('_raw_large_R4_sealed_Q4', path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    return parent


def load(build, expected=False, require_state=False):
    if type(expected) is not bool:
        raise ValueError('Expected guard-pair selection must be Boolean')
    build, origin, binding = authenticate(build, require_state=require_state)
    parent = load_parent(origin)
    runner = parent.load(origin, expected=True, require_state=True)
    return install(runner, make_hooks(runner.http, expected, binding['source_identity_sha256']))


def runtime_args(rest, build):
    explicit = False
    for index, token in enumerate(rest):
        option = token.split('=', 1)[0]
        if len(option) > 2 and option.startswith('--') and '--runtime-build'.startswith(option) and option != '--runtime-build':
            raise ValueError('Measured runtime selection requires complete --runtime-build')
        if token == '--runtime-build':
            if index + 1 == len(rest):
                raise ValueError('Missing measured runtime-build value')
            value = rest[index + 1]
        elif token.startswith('--runtime-build='):
            value = token.split('=', 1)[1]
        else:
            continue
        explicit = True
        if not value or Path(value).resolve() != build:
            raise ValueError('Every measured runtime-build must equal authenticated R4 child')
    return rest if explicit else rest + ['--runtime-build', str(build)]


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--expected-guard-pair', choices=('0', '1'), required=True)
    args, rest = parser.parse_known_args(argv)
    measured = bool(rest and rest[0] == 'measure')
    if measured:
        rest = runtime_args(rest, args.build.resolve())
    return load(args.build, expected=args.expected_guard_pair == '1', require_state=measured).main(rest)


if __name__ == '__main__':
    raise SystemExit(main())
