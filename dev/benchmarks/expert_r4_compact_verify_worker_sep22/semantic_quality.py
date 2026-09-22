#!/usr/bin/env python3
"""Private compact-R4 verifier provenance/coverage adapter; original22 tasks intact.

Planning and CPU tests do not measure a model. Root retains the original
--run-root-gpu requirement for measurement. The shared quality module is never
edited or monkey-patched: this adapter patches only an isolated module instance.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT / 'dev/benchmarks/prefill4k_attribution_quality.py'
_spec = importlib.util.spec_from_file_location('_compact_verify_original_semantic', ORIGINAL)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
http = _base.http
SECTION = 'compact_native_r4_verify'
PROFILE_SCHEMA = 'parallel-integer-original-M16-six-stage-R4-verify-v1'
PROFILE_SCOPE = 'singleton main physical R4 verification only; graph construction not GPU completion'
REGISTERED_COMPONENT = '90008567b2c0caaf6a7326a9d9010f4e8430271334e8e6c044baa873311a6a0f'
REGISTERED_SOURCE_SCOPE = 'singleton-main-verification-physicalR4-Full512-cap4-S10-nativeM16-step16-sixstage-v1'
MARKER_SCOPE = ';private-physicalR4-singleton-main-verification-only'
COUNTERS = ('plan_graph_calls', 'plan_graph_rows', 'gate_graph_calls', 'gate_graph_rows', 'down_graph_calls', 'down_graph_rows')
OWNERSHIP_FIELDS = tuple(SECTION + '.' + name for name in ('schema', 'scope', 'enabled', 'requested', 'source_identity_sha256',
    'dispatches_per_layer', 'base_gather_dispatches_per_layer', 'planner_threadgroup_bytes',
    'additional_gpu_allocation_bytes', 'full_model_quality_qualified'))
DEFAULT_BUILD = ROOT.parent if ROOT.name == 'source' and (ROOT.parent / 'overlay-manifest.json').is_file() else ROOT / 'build/compact-native-r4-verify-teacher-sep22-worker-v1b'
_build = DEFAULT_BUILD
_source_identity = None
_marker_prefix = None


def configure_build(build):
    """Require the exact generated identity, not a SHA copied from HTTP status."""
    global _build, _source_identity, _marker_prefix
    _build = Path(build).resolve()
    private = _build / 'source/dev/benchmarks/expert_r4_compact_verify_worker_sep22'
    text = (private / 'source_identity.hpp').read_text()
    match = re.search(r'kSourceIdentitySha256\[\]\s*=\s*"([0-9a-f]{64})"\s*;', text)
    if not match:
        raise ValueError('Compiled private source identity is unavailable or malformed')
    manifest = http.strict_json((_build / 'overlay-manifest.json').read_text())
    if (manifest.get('schema') != 'compact-native-R4-singleton-verify-teacher-worker-source-v1'
            or manifest.get('source_identity_sha256') != match.group(1)
            or manifest.get('all_noncore_host_consumers_rebuilt') is not True
            or type(manifest.get('additional_gpu_allocation_bytes')) is not int
            or manifest.get('additional_gpu_allocation_bytes') != 0):
        raise ValueError('Compiled private manifest/source identity or closure differs')
    parts = manifest.get('identity_parts')
    if (not isinstance(parts, dict) or set(parts) != {'qualified_plan', 'qualified_component_source_identity',
            'bridge', 'worker_transform', 'parent_manifest', 'scope'}
            or parts.get('scope') != REGISTERED_SOURCE_SCOPE
            or parts.get('qualified_component_source_identity') != REGISTERED_COMPONENT
            or manifest.get('qualified_parallel_source_identity') != REGISTERED_COMPONENT
            or parts.get('parent_manifest') != manifest.get('parent_manifest_sha256')):
        raise ValueError('Unknown compact source scope/component or identity tuple')
    canonical = json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != match.group(1):
        raise ValueError('Compact canonical source identity tuple differs')
    # Source-only authentication: these are small compiled program files, not
    # model tensors, captured activations, generated responses or task plans.
    for name, field in (('bridge.hpp', 'bridge'), ('worker_prepare.py', 'worker_transform'), ('plan.metal', 'qualified_plan')):
        if hashlib.sha256((private / name).read_bytes()).hexdigest() != parts.get(field):
            raise ValueError('Compiled compact source part differs: ' + name)
    bridge = (private / 'bridge.hpp').read_text()
    prefix = re.search(r'kSemantics\s*=\s*((?:"(?:[^"\\]|\\.)*"\s*)+);', bridge)
    if not prefix:
        raise ValueError('Compiled execution marker is unavailable')
    pieces = re.findall(r'"(?:[^"\\]|\\.)*"', prefix.group(1))
    _marker_prefix = ''.join(json.loads(piece) for piece in pieces)
    if not _marker_prefix.startswith(MARKER_SCOPE) or not _marker_prefix.endswith(';sourceSha256='):
        raise ValueError('Compiled execution marker scope/source binding differs')
    _source_identity = match.group(1)
    return _source_identity


def compiled_identity():
    if _source_identity is None:
        configure_build(_build)
    return _source_identity


def compact_present(status):
    if not isinstance(status, dict):
        return False
    routes = http.get_path(status, 'identity.kernel_routes')
    return SECTION in status or isinstance(routes, str) and any(piece in routes for piece in
        (MARKER_SCOPE, ';INTEGERONLY-compact-native-M16', ';planner=expert_r4_compact_native_sep22_plan'))


def _u64(value):
    return type(value) is int and 0 <= value < 2 ** 64


def _histogram(status):
    value = http.get_path(status, 'mtp.completed_cycles_by_proposed_depth')
    return value if isinstance(value, list) and len(value) == 16 and all(_u64(item) for item in value) else None


def compact_status_errors(status):
    if not compact_present(status):
        return []  # Legacy statuses retain their original acceptance behavior.
    errors = []
    source = compiled_identity()
    profile = status.get(SECTION)
    if not isinstance(profile, dict):
        return ['Compact verifier profile/counters are unavailable']
    active = profile.get('enabled')
    requested = profile.get('requested')
    if type(active) is not bool or type(requested) is not bool or active != requested:
        errors.append('Compact verifier requested/enabled flag differs or is not Boolean')
    requirements = {'schema': PROFILE_SCHEMA, 'scope': PROFILE_SCOPE, 'source_identity_sha256': source,
        'dispatches_per_layer': 6, 'base_gather_dispatches_per_layer': 2, 'planner_threadgroup_bytes': 2432,
        'additional_gpu_allocation_bytes': 0, 'full_model_quality_qualified': False}
    for field, expected in requirements.items():
        if not http.same_json(profile.get(field), expected):
            errors.append('Strict compact verifier profile differs: ' + field)
    routes = http.get_path(status, 'identity.kernel_routes')
    marker = _marker_prefix + source
    occurrences = routes.count(marker) if isinstance(routes, str) else 0
    scope_occurrences = routes.count(MARKER_SCOPE) if isinstance(routes, str) else 0
    planner_occurrences = routes.count(';planner=expert_r4_compact_native_sep22_plan') if isinstance(routes, str) else 0
    if active is True and (occurrences != 1 or scope_occurrences != 1 or planner_occurrences != 1):
        errors.append('Enabled compact verifier lacks the exact compiled execution marker/source')
    if active is False and (occurrences or scope_occurrences or planner_occurrences):
        errors.append('Disabled compact verifier unexpectedly changes execution identity')
    if active is True:
        for field, expected in {'identity.target_all_rows_full512': True,
                'identity.target_gathered_mpp_enabled': True, 'identity.target_gathered_mpp_max_physical_rows': 4,
                'persisted_experts.enabled': True, 'persisted_experts.expert_count': 24576}.items():
            if not http.same_json(http.get_path(status, field), expected):
                errors.append('Enabled compact verifier base dependency differs: ' + field)
    values = {}
    for field in COUNTERS:
        value = profile.get(field)
        if not _u64(value):
            errors.append('Compact graph counter is unavailable or invalid: ' + field)
        else:
            values[field] = value
            if active is False and value != 0:
                errors.append('Disabled compact verifier unexpectedly recorded work: ' + field)
    if len(values) == len(COUNTERS):
        calls = [values[field] for field in ('plan_graph_calls', 'gate_graph_calls', 'down_graph_calls')]
        if len(set(calls)) != 1:
            errors.append('Compact cumulative plan/gate/down calls do not reconcile')
        for stage in ('plan', 'gate', 'down'):
            if values[stage + '_graph_rows'] != 4 * values[stage + '_graph_calls']:
                errors.append('Compact cumulative rows differ from physicalR4: ' + stage)
    if _histogram(status) is None:
        errors.append('Compact verifier requires the strict sixteen-U64 completed-depth histogram')
    return errors


def compact_coverage(before, after, case, execution_mode='mtp3', expected_eligible=None):
    if not compact_present(before) and not compact_present(after):
        return {}, []
    errors = []
    for side, status in (('before', before), ('after', after)):
        if not compact_present(status):
            errors.append(side + ': Compact profile vanished during the request')
        else:
            errors.extend(side + ': ' + error for error in compact_status_errors(status))
    for field in OWNERSHIP_FIELDS:
        if not http.same_json(http.get_path(before, field), http.get_path(after, field)):
            errors.append('Compact provenance changed during the request: ' + field)
    values = {}
    for field in COUNTERS:
        old, new = (http.get_path(status, SECTION + '.' + field) for status in (before, after))
        if not _u64(old) or not _u64(new) or new < old:
            errors.append('Compact graph counter is invalid or decreasing: ' + field)
        else:
            values[field] = new - old
    old_hist, new_hist = _histogram(before), _histogram(after)
    histogram = None
    if old_hist is not None and new_hist is not None:
        histogram = [new - old for old, new in zip(old_hist, new_hist, strict=True)]
        if any(value < 0 for value in histogram):
            errors.append('Completed proposed-depth histogram decreased')
            histogram = None
    if expected_eligible is None:
        expected_eligible = 0 if execution_mode == 'standard' or case.get('body', {}).get('response_format', {}).get('type') == 'json_schema' else 1
    scope = case.get('compact_scope', 'singleton-main')
    permitted_scopes = {'singleton-main', 'prefill-only', 'autoregressive', 'trained-head', 'batch'}
    if scope not in permitted_scopes:
        errors.append('Unknown compact coverage caller scope')
    active = http.get_path(after, SECTION + '.enabled') is True
    caller_allowed = execution_mode == 'mtp3' and expected_eligible == 1 and scope == 'singleton-main'
    allowed = active and caller_allowed
    expected_calls = 48 * histogram[3] if allowed and histogram is not None else 0
    if not caller_allowed and histogram is not None and any(histogram):
        errors.append('Excluded compact caller advanced singleton completed-depth cycles')
    if len(values) == len(COUNTERS):
        for stage in ('plan', 'gate', 'down'):
            if values[stage + '_graph_calls'] != expected_calls:
                errors.append('Compact call delta differs from actual depth3/R4 singleton cycles: ' + stage)
            if values[stage + '_graph_rows'] != 4 * expected_calls:
                errors.append('Compact row delta differs from actual physicalR4 cycles: ' + stage)
    return {'compact_native_r4_verify_counter_deltas': values,
        'compact_completed_proposed_depth_histogram_delta': histogram,
        'compact_expected_stage_calls': expected_calls, 'compact_expected_stage_rows': 4 * expected_calls,
        'compact_expected_graph_dispatches': 6 * expected_calls,
        'compact_base_gather_graph_dispatches': 2 * expected_calls,
        'compact_graph_scope': PROFILE_SCOPE, 'compact_source_identity_sha256': compiled_identity()}, errors


_original_gate = _base.gate_status
_original_coverage = _base.coverage
_original_ownership = _base.ownership_policy


def gate_status(status, plan, store, execution_mode='mtp3'):
    return _original_gate(status, plan, store, execution_mode) + compact_status_errors(status)


def coverage(before, after, case, require_counters=True, execution_mode='mtp3'):
    details, errors = _original_coverage(before, after, case, require_counters, execution_mode)
    compact, compact_errors = compact_coverage(before, after, case, execution_mode, details.get('expected_eligible_requests'))
    return {**details, **compact}, errors + compact_errors


def ownership_policy(status):
    policy = _original_ownership(status)
    if compact_present(status):
        policy.update({field: http.get_path(status, field) for field in OWNERSHIP_FIELDS})
    return policy


# Only this freshly loaded module instance receives the private instrumentation.
_base.gate_status = gate_status
_base.coverage = coverage
_base.ownership_policy = ownership_policy
specs = _base.specs
grade_record = _base.grade_record
execution_policy = _base.execution_policy
standard_mtp_errors = _base.standard_mtp_errors
make_plan = _base.make_plan
read_plan = _base.read_plan
measure = _base.measure
compare = _base.compare
MODEL = _base.MODEL
SOURCE = _base.SOURCE


def load(build=None):
    """Return the isolated original22 runner with compact gates installed."""
    if build is not None:
        configure_build(build)
    return _base


def install_hooks(*, status=None, coverage=None, ownership=None):
    """Chain later private gates without replacing tasks or execution policy.

    status(status,plan,store,mode)->errors; coverage(before,after,case,required,
    mode)->(details,errors); ownership(status)->extra fields. Only the isolated
    runner is patched. Existing compact/base gates execute before each hook.
    """
    if status is not None:
        previous = _base.gate_status
        def next_status(*args, _previous=previous, **kwargs):
            return _previous(*args, **kwargs) + status(*args, **kwargs)
        _base.gate_status = next_status
    if coverage is not None:
        previous = _base.coverage
        def next_coverage(*args, _previous=previous, **kwargs):
            details, errors = _previous(*args, **kwargs)
            extra, extra_errors = coverage(*args, **kwargs)
            return {**details, **extra}, errors + extra_errors
        _base.coverage = next_coverage
    if ownership is not None:
        previous = _base.ownership_policy
        def next_ownership(*args, _previous=previous, **kwargs):
            return {**_previous(*args, **kwargs), **ownership(*args, **kwargs)}
        _base.ownership_policy = next_ownership
    return _base


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--build', type=Path, default=DEFAULT_BUILD)
    args, rest = p.parse_known_args(argv)
    configure_build(args.build)
    explicit_runtime = any(token == '--runtime-build' or token.startswith('--runtime-build=') for token in rest)
    if rest and rest[0] == 'measure' and not explicit_runtime:
        rest += ['--runtime-build', str(args.build)]
    return _base.main(rest)


if __name__ == '__main__':
    raise SystemExit(main())
