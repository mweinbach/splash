#!/usr/bin/env python3
"""Strict HC-fast hooks AFTER the admitted b32 guard/compact22 gates."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = 'dev/benchmarks/guard_hc_fast_composite_sep22'
GUARD_PRIVATE = 'dev/benchmarks/expert_r4_preflight_bundle_sep22'
GUARD_ID = 'b32c030bff181fb65e546685282650e1db5636a5dfa466dcf8a745c66931fad5'
HC_LEAF = '92680570764ecfe797fe7db566146d1be1f6e51246a73b197a423a722d1feb61'
HC_AIR = '7cc3d642e22bbf90fb95f6c524af232a75fe6c8d99d7385a5e51f7ee572eeefa'
HC_BRIDGE = 'c8a30c2c11a5cbf123c3273d89764ccd51258edaaaff85ca8af7c7b76eeba983'
HC_OVERLAY = '805292b689095939db02b0af2d56b50f83bc75d0b1c43bc76a5e27dafe3f50c9'
HC_POLICY = 'exact-main-VerifyR4-HCdown-SG4-positive-zero-inactive4to7-unchanged-HCup-F32-M8N32SG4-parent-v1'
HC_CERTIFICATE = '97roles-rawDownF32-BF16-Silu-gates-UpF32-BF16-mix-padded8-guards-exact-sep22-hc-pad-producer-r4-97chain-v2'
HC_MARKER = ';private-hc-pad-verify-r4-down-producer-unchanged-up-v1'
HC_FAMILY = ';private-hc-pad-verify-r4'
HC_SCOPE = 'main VerifyR4 graph construction; notGPU completion'
SECTION = 'guard_hc_fast_composite'
SCHEMA = 'C1-Bundle1-HCfastV8-VerifyR4-registered-composite-v1'
SCOPE = 'singleton VerifyR4 only; original guard13views and HC97roles retained'
SOURCE_SCOPE = 'singleton-VerifyR4-C1-Bundle1-HCfastV8-original-guard-epoch-and-H97-semantic-retained-v1'
MARKER = ';private-guardBundle-C1-HCfastV8-down7cc3-R4-only-sourceSha256='
FAMILY = ';private-guardBundle-C1-HCfastV8'
HC_FIELDS = ('identity.hc_pad_verify_r4_enabled', 'identity.hc_pad_verify_r4_policy', 'identity.hc_pad_verify_r4_source_certificate')
COUNTERS = ('graph_calls', 'graph_rows', 'padding_dispatches_saved')
OWNERSHIP = HC_FIELDS + tuple(SECTION + '.' + key for key in ('schema', 'requested', 'source_identity_sha256', 'scope',
    'HC_fast_leaf_manifest_sha256', 'HC_down_AIR_sha256', 'new_GPU_allocation_bytes', 'whole_composite_state_qualified'))
PROGRAMS = ('policy.hpp', 'overlay.py', 'prepare.py', 'worker.mk', 'witness.py')
PROOF_SCHEMA = 'guard-C1-Bundle1-HCfastV8-new-composite-native-proof-v1'


def u64(value): return type(value) is int and 0 <= value < 2 ** 64
def same(a, b): return type(a) is type(b) and a == b


def present(status, http):
    if not isinstance(status, dict): return False
    routes = http.get_path(status, 'identity.kernel_routes')
    identity = status.get('identity')
    # The registered b32 origin has none of this HC/composite context. Partial
    # new profiles must not fall back to that legacy path, even with HC off.
    return (SECTION in status or 'hc_pad_verify_r4_route_counters' in status
        or isinstance(identity, dict) and any(field.split('.', 1)[1] in identity for field in HC_FIELDS)
        or isinstance(routes, str) and (FAMILY in routes or HC_FAMILY in routes))


def make_hooks(http, source, require_active=False):
    if not isinstance(source, str) or not re.fullmatch('[0-9a-f]{64}', source): raise ValueError('Invalid compiled composite identity')
    marker = MARKER + source

    def status(status, plan, store, execution_mode='mtp3'):
        if not present(status, http):
            return ['Measurement requires the active registered guard/HC composite profile'] if require_active else []
        errors = []
        p = status.get(SECTION)
        if not isinstance(p, dict): return ['Registered guard/HC composite profile missing']
        requested = p.get('requested')
        if type(requested) is not bool: errors.append('Composite requested flag is not Boolean')
        if require_active and requested is not True: errors.append('Measurement requires composite requested=True')
        for key, expected in {'schema': SCHEMA, 'scope': SCOPE, 'source_identity_sha256': source,
                'HC_fast_leaf_manifest_sha256': HC_LEAF, 'HC_down_AIR_sha256': HC_AIR,
                'new_GPU_allocation_bytes': 0, 'whole_composite_state_qualified': False}.items():
            if not same(p.get(key), expected): errors.append('Registered composite profile differs: ' + key)
        routes = http.get_path(status, 'identity.kernel_routes')
        exact = routes.count(marker) if isinstance(routes, str) else 0
        family = routes.count(FAMILY) if isinstance(routes, str) else 0
        if (requested is True or require_active) and (exact != 1 or family != 1): errors.append('Composite marker/source mismatch')
        if requested is False and family: errors.append('Disabled composite changes execution identity')
        # A composed worker must never silently drop b32 profile validation by
        # looking like a legacy compact-only worker. Guard gates run before us.
        guard = status.get('compact_r4_preflight')
        if not isinstance(guard, dict) or guard.get('source_identity_sha256') != GUARD_ID:
            errors.append('Required registered b32 guard profile missing/unknown')
        if requested is True or require_active:
            for prefix in ('compact_r4_preflight', 'compact_native_r4_verify'):
                for flag in ('requested', 'enabled'):
                    if http.get_path(status, prefix + '.' + flag) is not True: errors.append('Active composite dependency differs: ' + prefix + '.' + flag)
        hc = http.get_path(status, HC_FIELDS[0])
        if type(hc) is not bool or hc is not requested: errors.append('Composite flag and HC enable flag disagree')
        if require_active and hc is not True: errors.append('Measurement requires HC enabled=True')
        if http.get_path(status, HC_FIELDS[1]) != HC_POLICY or http.get_path(status, HC_FIELDS[2]) != HC_CERTIFICATE:
            errors.append('Registered HC design policy/certificate differs')
        hc_occurrences = routes.count(HC_MARKER) if isinstance(routes, str) else -1
        hc_family = routes.count(HC_FAMILY) if isinstance(routes, str) else -1
        if hc_occurrences != int(hc is True) or hc_family != int(hc is True): errors.append('HC marker and enable flag disagree')
        counters = status.get('hc_pad_verify_r4_route_counters')
        if not isinstance(counters, dict): return errors + ['HC counters missing']
        if counters.get('scope') != HC_SCOPE: errors.append('HC counter scope mismatch')
        values = {}
        for key in COUNTERS:
            value = counters.get(key)
            if not u64(value): errors.append('HC counter invalid: ' + key)
            else:
                values[key] = value
                if hc is False and value: errors.append('Disabled HC recorded work: ' + key)
        if len(values) == 3 and (values['graph_rows'] != 4 * values['graph_calls'] or values['padding_dispatches_saved'] != values['graph_calls']):
            errors.append('HC cumulative counters do not reconcile')
        if requested is True and len(values) == 3:
            calls = http.get_path(status, 'compact_r4_preflight.guard_preflight_graph_calls')
            if not u64(calls) or values['graph_calls'] * 48 != calls * 97:
                errors.append('HC97 and guard48 cumulative trial counts disagree')
        return errors

    def coverage(before, after, case, require_counters=True, execution_mode='mtp3'):
        if not require_active and not present(before, http) and not present(after, http): return {}, []
        errors = []
        for side, value in (('before', before), ('after', after)):
            if not present(value, http): errors.append(side + ': Composite profile disappeared during request')
            else: errors.extend(side + ': ' + error for error in status(value, None, None, execution_mode))
        for key in OWNERSHIP:
            if not http.same_json(http.get_path(before, key), http.get_path(after, key)): errors.append('Composite provenance changed: ' + key)
        a, b = (http.get_path(value, 'mtp.completed_cycles_by_proposed_depth') for value in (before, after))
        if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 16 and all(u64(x) for x in a + b)
                and all(y >= x for x, y in zip(a, b))): return {}, errors + ['Composite completed-depth histogram invalid/decreasing']
        cycles = b[3] - a[3]
        caller = execution_mode == 'mtp3' and case.get('compact_scope', 'singleton-main') == 'singleton-main' and case.get('body', {}).get('response_format', {}).get('type') != 'json_schema'
        active = http.get_path(after, SECTION + '.requested') is True
        expected = 97 * cycles if active and caller else 0
        if not caller and cycles: errors.append('Excluded composite caller advanced singleton R4 cycles')
        deltas = {}
        for key in COUNTERS:
            old, new = (http.get_path(value, 'hc_pad_verify_r4_route_counters.' + key) for value in (before, after))
            if not u64(old) or not u64(new) or new < old: errors.append('HC counter invalid/decreasing: ' + key)
            else: deltas[key] = new - old
        if len(deltas) == 3 and (deltas['graph_calls'] != expected or deltas['graph_rows'] != 4 * expected or deltas['padding_dispatches_saved'] != expected):
            errors.append('HC graph delta differs from97 actual singleton H3 cycles')
        if active:
            for prefix, call, row in (('compact_r4_preflight', 'guard_preflight_graph_calls', 'guard_preflight_graph_rows'),
                    ('compact_native_r4_verify', 'plan_graph_calls', 'plan_graph_rows')):
                old, new = (http.get_path(value, prefix + '.' + call) for value in (before, after))
                old_rows, new_rows = (http.get_path(value, prefix + '.' + row) for value in (before, after))
                wanted = 48 * cycles if caller else 0
                if not all(u64(x) for x in (old, new, old_rows, new_rows)) or new - old != wanted or new_rows - old_rows != 4 * wanted:
                    errors.append('Composite HC/guard/compact deltas disagree: ' + prefix)
        return {'HC_fast_graph_counter_deltas': deltas, 'HC_fast_expected_calls': expected, 'composite_actual_H3_cycle_delta': cycles}, errors

    def ownership(status):
        if not present(status, http): return {}
        return {key: http.get_path(status, key) for key in OWNERSHIP}
    return status, coverage, ownership


def install(runner, hooks):
    old_status, old_coverage, old_ownership = runner.gate_status, runner.coverage, runner.ownership_policy
    status_hook, coverage_hook, ownership_hook = hooks
    def gate_status(*args, **kwargs): return old_status(*args, **kwargs) + status_hook(*args, **kwargs)
    def coverage(*args, **kwargs):
        details, errors = old_coverage(*args, **kwargs); extra, added = coverage_hook(*args, **kwargs)
        return {**details, **extra}, errors + added
    def ownership(status): return {**old_ownership(status), **ownership_hook(status)}
    runner.gate_status, runner.coverage, runner.ownership_policy = gate_status, coverage, ownership
    return runner


def authenticate(build):
    build = Path(build).resolve(); m = json.loads((build / 'overlay-manifest.json').read_text())
    if (m.get('schema') != 'guard-C1-HCfastV8-VerifyR4-composite-source-v1' or m.get('guard_source_identity') != GUARD_ID
            or m.get('HC_fast_leaf_manifest_sha256') != HC_LEAF or m.get('HC_down_AIR_sha256') != HC_AIR
            or not same(m.get('new_GPU_allocation_bytes'), 0) or m.get('Root_OLD_HC_full_proof_not_reused_as_new_composite_proof') is not True):
        raise ValueError('Unknown registered composite source profile')
    parts = m.get('identity_parts'); keys = set(PROGRAMS) | {'guard_parent_manifest', 'guard_source_identity', 'HC_overlay', 'HC_bridge', 'HC_fast_leaf_manifest', 'HC_fast_down_AIR', 'scope'}
    if (not isinstance(parts, dict) or set(parts) != keys or parts['scope'] != SOURCE_SCOPE or parts['guard_source_identity'] != GUARD_ID
            or parts['HC_fast_leaf_manifest'] != HC_LEAF or parts['HC_fast_down_AIR'] != HC_AIR
            or parts['HC_bridge'] != HC_BRIDGE or parts['HC_overlay'] != HC_OVERLAY): raise ValueError('Composite identity tuple mismatch')
    header = (build / 'source' / PRIVATE / 'source_identity.hpp').read_text()
    match = re.search(r'kCompositeSourceIdentitySha256\[\]\s*=\s*"([0-9a-f]{64})"', header)
    identity = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not match or match.group(1) != identity or m.get('source_identity_sha256') != identity: raise ValueError('Compiled composite source identity mismatch')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    records = {r['path']: r['sha256'] for r in m['files']}
    for name in PROGRAMS:
        path = PRIVATE + '/' + name
        if digest(build / 'source' / path) != records[path] or records[path] != parts[name]: raise ValueError('Composite program source part drift: ' + name)
    origin = Path(m['base']).resolve(); hc = Path(m['HC_origin']).resolve()
    if digest(origin / 'overlay-manifest.json') != parts['guard_parent_manifest']: raise ValueError('Registered guard parent manifest drift')
    if json.loads((origin / 'overlay-manifest.json').read_text()).get('source_identity_sha256') != GUARD_ID:
        raise ValueError('Unknown guard policy origin')
    for name, field in (('overlay.py', 'HC_overlay'), ('worker_bridge.hpp', 'HC_bridge')):
        path = 'dev/benchmarks/hc_pad_verify_worker_sep22/' + name
        if digest(hc / 'source' / path) != parts[field] or digest(build / 'source' / path) != parts[field]: raise ValueError('Registered HC source drift: ' + name)
    if digest(build / 'hc-pad.air') != HC_AIR: raise ValueError('Registered fast HC down AIR mismatch')
    profile = json.loads((hc / 'HC-fast-source-profile.json').read_text()); leaf = Path(profile['active_leaf'])
    if (profile.get('qualified') is not True or profile.get('active_leaf_manifest_sha256') != HC_LEAF
            or profile.get('HC_down_AIR_sha256') != HC_AIR): raise ValueError('Registered fast HC leaf admission metadata missing')
    if digest(leaf / 'manifest.json') != HC_LEAF: raise ValueError('Registered fast leaf manifest mismatch')
    leafm = json.loads((leaf / 'manifest.json').read_text())
    if (leafm.get('private_math_recipe') != 'Metal4.1/O3/default-fast/original089-baseline-match'
            or leafm.get('separate_down_and_safe_up_probe_translation_units') is not True or leafm.get('candidate_AIR_sha256') != HC_AIR):
        raise ValueError('Unregistered fast HC compiler recipe')
    return build, m, identity, origin


def require_native_state(build, identity):
    # This is a small Root-authenticated metadata receipt, not the tensor/state
    # report. Old V3 evidence cannot admit a new composite executable.
    r = json.loads((build / 'Root-native-qualified.json').read_text())
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    expected = {'schema': PROOF_SCHEMA, 'pass': True, 'qualification_complete': True,
        'source_identity_sha256': identity, 'exe_sha256': digest(build / 'splash-flash'),
        'newlib_sha256': digest(build / 'splash.metallib'), 'HC_down_AIR_sha256': HC_AIR,
        'HC_fast_leaf_manifest_sha256': HC_LEAF, 'frames': 26, 'repeated_frames': 54,
        'backend_destroyed': True, 'guardCalls': 240, 'guardRows': 960,
        'HCcalls': 485, 'HCrows': 1940, 'pads': 485}
    if (any(not same(r.get(key), value) for key, value in expected.items())
            or not isinstance(r.get('root_report_path'), str) or not r['root_report_path'].strip()):
        raise ValueError('Fresh Root composite state receipt does not bind this source/executable/library')


def load(build, require_state=False):
    build, manifest, identity, origin = authenticate(build)
    if require_state: require_native_state(build, identity)
    path = origin / 'source' / GUARD_PRIVATE / 'semantic_quality.py'
    spec = importlib.util.spec_from_file_location('_composite_registered_guard', path)
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    runner = guard.load(origin)
    return install(runner, make_hooks(runner.http, identity, require_active=require_state))


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False); parser.add_argument('--build', type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    measured = bool(rest and rest[0] == 'measure')
    if measured:
        build = args.build.resolve(); explicit = False
        for i, token in enumerate(rest):
            option = token.split('=', 1)[0]
            if len(option) > 2 and option.startswith('--') and '--runtime-build'.startswith(option) and option != '--runtime-build':
                raise ValueError('Measured runtime selection requires the complete --runtime-build option')
            if token == '--runtime-build':
                if i + 1 == len(rest): raise ValueError('Missing measured --runtime-build value')
                value = rest[i + 1]
            elif token.startswith('--runtime-build='): value = token.split('=', 1)[1]
            else: continue
            explicit = True
            if not value or Path(value).resolve() != build:
                raise ValueError('Every measured --runtime-build must equal the authenticated --build')
        if not explicit: rest += ['--runtime-build', str(build)]
    runner = load(args.build, require_state=measured)
    return runner.main(rest)


if __name__ == '__main__': raise SystemExit(main())
