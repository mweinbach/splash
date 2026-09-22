#!/usr/bin/env python3
"""Private CPU-preflight provenance hook over the unchanged compact22 runner."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = 'dev/benchmarks/expert_r4_preflight_bundle_sep22'
ORIGIN_PRIVATE = 'dev/benchmarks/expert_r4_compact_verify_worker_sep22'
ORIGIN_SOURCE = '8ed20bad23072a34a50bb0516b058e991ae396c7e8b88900193d487034735a79'
SECTION = 'compact_r4_preflight'
SCHEMA = 'CPU-only-unique13view-StoreLayerGraph-stack-bundle-v1'
SCOPE = 'singleton physicalR4 verification; graph construction not completion'
MARKER_PREFIX = ';private-CPU-only-R4-unique13view-preflight-localStoreLayerGraphBundle-original6stage-unchanged-public-guards-sourceSha256='
MARKER_FAMILY = ';private-CPU-only-R4-unique13view-preflight'
COUNTERS = ('guard_preflight_graph_calls', 'guard_preflight_graph_rows')
OWNERSHIP_FIELDS = tuple(SECTION + '.' + field for field in ('schema', 'scope', 'requested', 'enabled',
    'source_identity_sha256', 'unique_logical_views', 'immutable_spans', 'GPU_allocation_bytes_added',
    'shader_math_changed', 'original_public_native_API_guards_unchanged', 'whole_state_qualified'))
MANIFEST_SCHEMA = 'CPU-only-compactR4-complete-local-preflight-source-v1'
SOURCE_SCOPE = 'CPU-only-one13view-preflight-R4-native6stage-private-noncopyable-singleconsume-StoreLayerGraph-local-bundle-v1'
PROGRAM_PARTS = ('policy.hpp', 'guard.hpp', 'overlay.py', 'prepare.py', 'worker.mk', 'witness.py',
    'PLAN.md', 'baseline_extract.py', 'guard_decision_cpu.cpp', 'policy_cpu.cpp')


def _u64(value):
    return type(value) is int and 0 <= value < 2 ** 64


def _same(a, b):
    return type(a) is type(b) and a == b


def present(status, http):
    if not isinstance(status, dict): return False
    routes = http.get_path(status, 'identity.kernel_routes')
    return SECTION in status or isinstance(routes, str) and MARKER_FAMILY in routes


def make_hooks(http, source):
    if not isinstance(source, str) or not re.fullmatch('[0-9a-f]{64}', source):
        raise ValueError('Required compiled preflight source identity is invalid')
    marker = MARKER_PREFIX + source

    def status(status, plan, store, execution_mode='mtp3'):
        if not present(status, http): return []
        errors = []
        p = status.get(SECTION)
        if not isinstance(p, dict):
            return ['CPU-preflight profile/counters are unavailable']
        active, requested = p.get('enabled'), p.get('requested')
        if type(active) is not bool or type(requested) is not bool or active != requested:
            errors.append('CPU-preflight requested/enabled flag differs or is not Boolean')
        expected = {'schema': SCHEMA, 'scope': SCOPE, 'source_identity_sha256': source,
            'unique_logical_views': 13, 'immutable_spans': 96, 'GPU_allocation_bytes_added': 0,
            'shader_math_changed': False, 'original_public_native_API_guards_unchanged': True,
            'whole_state_qualified': False}
        for field, value in expected.items():
            if not _same(p.get(field), value):
                errors.append('Strict CPU-preflight profile differs: ' + field)
        routes = http.get_path(status, 'identity.kernel_routes')
        exact = routes.count(marker) if isinstance(routes, str) else 0
        family = routes.count(MARKER_FAMILY) if isinstance(routes, str) else 0
        if active is True and (exact != 1 or family != 1):
            errors.append('CPU-preflight execution marker/source does not match compiled profile')
        if active is False and (exact or family):
            errors.append('Disabled CPU-preflight changes execution identity')
        values = {}
        for field in COUNTERS:
            value = p.get(field)
            if not _u64(value):
                errors.append('CPU-preflight graph counter missing/invalid: ' + field)
            else:
                values[field] = value
                if active is False and value:
                    errors.append('Disabled CPU-preflight encoded guard bundles: ' + field)
        if len(values) == 2 and values[COUNTERS[1]] != 4 * values[COUNTERS[0]]:
            errors.append('CPU-preflight cumulative rows do not equal physicalR4 calls')
        if active is True:
            for field in ('enabled', 'requested'):
                if http.get_path(status, 'compact_native_r4_verify.' + field) is not True:
                    errors.append('CPU-preflight requires original compact verification: ' + field)
            old_calls = http.get_path(status, 'compact_native_r4_verify.plan_graph_calls')
            old_rows = http.get_path(status, 'compact_native_r4_verify.plan_graph_rows')
            if not _u64(old_calls) or not _u64(old_rows):
                errors.append('Original compact cumulative plan counters are unavailable')
            elif len(values) == 2 and (values[COUNTERS[0]] != old_calls or values[COUNTERS[1]] != old_rows):
                errors.append('CPU-preflight cumulative bundle count differs from original compact plans')
        return errors

    def coverage(before, after, case, require_counters=True, execution_mode='mtp3'):
        if not present(before, http) and not present(after, http): return {}, []
        errors = []
        for side, value in (('before', before), ('after', after)):
            if not present(value, http): errors.append(side + ': CPU-preflight profile disappeared during request')
            else: errors.extend(side + ': ' + error for error in status(value, None, None, execution_mode))
        for field in OWNERSHIP_FIELDS:
            if not http.same_json(http.get_path(before, field), http.get_path(after, field)):
                errors.append('CPU-preflight provenance changed during request: ' + field)
        deltas = {}
        for field in COUNTERS:
            old, new = (http.get_path(value, SECTION + '.' + field) for value in (before, after))
            if not _u64(old) or not _u64(new) or new < old:
                errors.append('CPU-preflight graph counter invalid/decreasing: ' + field)
            else:
                deltas[field] = new - old
        original = {}
        for field in ('plan_graph_calls', 'plan_graph_rows'):
            old, new = (http.get_path(value, 'compact_native_r4_verify.' + field) for value in (before, after))
            if not _u64(old) or not _u64(new) or new < old:
                errors.append('Original compact plan counter invalid/decreasing: ' + field)
            else:
                original[field] = new - old
        active = http.get_path(after, SECTION + '.enabled') is True
        expected_calls = original.get('plan_graph_calls', 0) if active else 0
        expected_rows = original.get('plan_graph_rows', 0) if active else 0
        if len(deltas) == 2 and (deltas[COUNTERS[0]] != expected_calls or deltas[COUNTERS[1]] != expected_rows):
            errors.append('CPU-preflight bundle delta differs from original compact plan calls/rows')
        # The inherited compact hook independently validates actual H3 and
        # excluded caller scope; validate that same strict histogram here too.
        old_hist, new_hist = (http.get_path(value, 'mtp.completed_cycles_by_proposed_depth') for value in (before, after))
        if (not isinstance(old_hist, list) or not isinstance(new_hist, list) or len(old_hist) != 16 or len(new_hist) != 16
                or not all(_u64(value) for value in old_hist + new_hist)
                or any(new < old for old, new in zip(old_hist, new_hist))):
            errors.append('CPU-preflight completed-depth histogram missing/invalid/decreasing')
        else:
            caller = execution_mode == 'mtp3' and case.get('compact_scope', 'singleton-main') == 'singleton-main' and case.get('body', {}).get('response_format', {}).get('type') != 'json_schema'
            expected_h3 = 48 * (new_hist[3] - old_hist[3]) if active and caller else 0
            if expected_calls != expected_h3 or expected_rows != 4 * expected_h3:
                errors.append('CPU-preflight plan coverage differs from48 actual singleton H3 cycles')
        return {'guard_preflight_graph_counter_deltas': deltas, 'guard_preflight_expected_calls': expected_calls,
            'guard_preflight_expected_rows': expected_rows, 'guard_preflight_source_identity_sha256': source}, errors

    def ownership(status):
        if not present(status, http): return {}
        return {field: http.get_path(status, field) for field in OWNERSHIP_FIELDS}
    return status, coverage, ownership


def load(build):
    build = Path(build).resolve()
    private = build / 'source' / PRIVATE
    manifest = json.loads((build / 'overlay-manifest.json').read_text())
    source_header = (private / 'source_identity.hpp').read_text()
    match = re.search(r'kPreflightSourceIdentitySha256\[\]\s*=\s*"([0-9a-f]{64})"', source_header)
    if (not match or manifest.get('schema') != MANIFEST_SCHEMA
            or manifest.get('original_compact_source_identity') != ORIGIN_SOURCE
            or manifest.get('source_identity_sha256') != match.group(1)
            or manifest.get('all_original_public_native_producers_fullvalidation_unchanged') is not True
            or manifest.get('shader_math_workspace_changed') is not False
            or not _same(manifest.get('additional_GPU_allocation_bytes'), 0)):
        raise ValueError('Required new compiled preflight header/manifest identity differs')
    parts = manifest.get('identity_parts')
    expected_keys = set(PROGRAM_PARTS) | {'parent_manifest', 'parent_semantic_seal', 'original_compact_source_identity', 'original_metallib', 'scope'}
    if (not isinstance(parts, dict) or set(parts) != expected_keys or parts.get('scope') != SOURCE_SCOPE
            or parts.get('original_compact_source_identity') != ORIGIN_SOURCE
            or parts.get('parent_manifest') != manifest.get('parent_manifest_sha256')
            or parts.get('parent_semantic_seal') != manifest.get('parent_semantic_seal_sha256')):
        raise ValueError('Unregistered preflight source scope/parent identity tuple')
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != match.group(1):
        raise ValueError('Canonical preflight source identity tuple differs')
    records = {entry['path']: entry.get('sha256', entry.get('overlay_sha256')) for entry in manifest['files']}
    for name in PROGRAM_PARTS:
        relative = PRIVATE + '/' + name
        if digest(build / 'source' / relative) != records.get(relative) or records.get(relative) != parts.get(name):
            raise ValueError('Compiled complete-preflight source differs: ' + name)
    origin = Path(manifest['base']).resolve()
    if (digest(origin / 'overlay-manifest.json') != parts['parent_manifest']
            or digest(origin / 'semantic-source-seal.json') != parts['parent_semantic_seal']
            or digest(origin / 'READY.json') != manifest.get('parent_READY_sha256')
            or digest(origin / 'splash.metallib') != parts['original_metallib']
            or digest(build / 'splash.metallib') != parts['original_metallib']):
        raise ValueError('Original compact parent source/semantic/READY/library pins differ')
    original = origin / 'source' / ORIGIN_PRIVATE / 'semantic_quality.py'
    spec = importlib.util.spec_from_file_location('_preflight_original_compact', original)
    compact = importlib.util.module_from_spec(spec); spec.loader.exec_module(compact)
    compact.configure_build(origin)
    if compact.compiled_identity() != ORIGIN_SOURCE:
        raise ValueError('Unregistered original compact policy origin')
    hooks = make_hooks(compact.http, match.group(1))
    compact.install_hooks(status=hooks[0], coverage=hooks[1], ownership=hooks[2])
    return compact.load()


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--build', type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    runner = load(args.build)
    explicit = any(token == '--runtime-build' or token.startswith('--runtime-build=') for token in rest)
    if rest and rest[0] == 'measure' and not explicit:
        rest += ['--runtime-build', str(args.build.resolve())]
    return runner.main(rest)


if __name__ == '__main__': raise SystemExit(main())
