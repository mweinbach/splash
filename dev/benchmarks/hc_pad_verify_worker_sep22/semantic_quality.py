#!/usr/bin/env python3
"""Add strict HC schedule provenance to an isolated unchanged22-task runner."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIELDS = ('identity.hc_pad_verify_r4_enabled', 'identity.hc_pad_verify_r4_policy',
          'identity.hc_pad_verify_r4_source_certificate')
COUNTERS = ('graph_calls', 'graph_rows', 'padding_dispatches_saved')
POLICY = 'exact-main-VerifyR4-HCdown-SG4-positive-zero-inactive4to7-unchanged-HCup-F32-M8N32SG4-parent-v1'
CERTIFICATE = '97roles-rawDownF32-BF16-Silu-gates-UpF32-BF16-mix-padded8-guards-exact-sep22-hc-pad-producer-r4-97chain-v2'
MARKER = ';private-hc-pad-verify-r4-down-producer-unchanged-up-v1'
REGISTERED_FAST_LEAF_MANIFEST = '92680570764ecfe797fe7db566146d1be1f6e51246a73b197a423a722d1feb61'
REGISTERED_FAST_DOWN_AIR = '7cc3d642e22bbf90fb95f6c524af232a75fe6c8d99d7385a5e51f7ee572eeefa'
REGISTERED_FAST_WORKER_LIBRARY = '390e67bb04e81c1aa291013f2de16fbca27bbf7771819b17af95aa7be4c88177'


def load(build):
    build = Path(build).resolve()
    manifest = json.loads((build / 'overlay-manifest.json').read_text())
    if (manifest.get('schema') != 'hc-pad-compact-r4-verify-teacher-source-v1'
            or manifest.get('public_headers_changed_by_HC') is not False
            or manifest.get('new_GPU_buffers_or_cache_bytes') != 0
            or manifest.get('numerical_derivative_changed_by_HC') is not False):
        raise ValueError('Unknown combined HC/compact source profile')
    files = {entry['path']: entry['sha256'] for entry in manifest['files']}
    for name in ['worker_bridge.hpp', 'overlay.py']:
        relative = 'dev/benchmarks/hc_pad_verify_worker_sep22/' + name
        if hashlib.sha256((build / 'source' / relative).read_bytes()).hexdigest() != files[relative]:
            raise ValueError('Compiled HC source profile drift: ' + name)
    if manifest.get('HC_compiler_policy_update_only_CPP_header_Source_unchanged') is not True:
        raise ValueError('Only registered V3 baseline-matched HC artifact profile is admitted')
    if manifest.get('HC_compiler_policy_update_only_CPP_header_Source_unchanged') is True:
        # The inherited C++ certificate describes source design, not the
        # changed compiled artifact. Admit only the fresh baseline-matched leaf
        # and the real combined full-state proof for this exact library.
        profile = json.loads((build / 'HC-fast-source-profile.json').read_text())
        digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if profile.get('qualified') is not True or profile.get('legacy_CPP_certificate_name_is_design_only_not_newartifact_admission') is not True:
            raise ValueError('HC fast compiled-artifact admission missing')
        leaf = Path(profile['active_leaf'])
        if (profile.get('active_leaf_manifest_sha256') != REGISTERED_FAST_LEAF_MANIFEST
                or profile.get('HC_down_AIR_sha256') != REGISTERED_FAST_DOWN_AIR
                or profile.get('new_library_sha256') != REGISTERED_FAST_WORKER_LIBRARY):
            raise ValueError('Unknown HC fast leaf/down/library profile; registered V8/V3 required')
        if digest(leaf / 'manifest.json') != profile['active_leaf_manifest_sha256'] or manifest.get('HC_component_manifest_sha256') != profile['active_leaf_manifest_sha256']:
            raise ValueError('HC active leaf manifest differs')
        leaf_manifest = json.loads((leaf / 'manifest.json').read_text())
        if (leaf_manifest.get('private_math_recipe') != 'Metal4.1/O3/default-fast/original089-baseline-match'
                or leaf_manifest.get('separate_down_and_safe_up_probe_translation_units') is not True
                or digest(build / 'hc-pad.air') != profile['HC_down_AIR_sha256']
                or profile['HC_down_AIR_sha256'] != leaf_manifest.get('candidate_AIR_sha256')):
            raise ValueError('HC private compiler recipe/down AIR differs from active fast leaf')
        if digest(build / 'splash.metallib') != profile['new_library_sha256'] or digest(build / 'splash-flash') != profile['actual_worker_sha256']:
            raise ValueError('HC active worker/library artifact differs')
        component_path = Path(profile['Root_97role_fast_report'])
        full_path = Path(profile['Root_full_trunk_fast_report'])
        if digest(component_path) != profile['Root_97role_fast_report_sha256'] or digest(full_path) != profile['Root_full_trunk_fast_report_sha256']:
            raise ValueError('HC fresh component/full-state proof digest differs')
        component = json.loads(component_path.read_text())
        full = json.loads(full_path.read_text())
        if (component.get('pass') is not True or component.get('cases') != 97
                or component.get('library_sha256') != leaf_manifest.get('metallib_sha256')
                or full.get('pass') is not True or full.get('qualification_complete') is not True
                or full.get('backend_destroyed') is not True or full.get('frames') != 26
                or full.get('repeated_frames') != 54
                or full.get('producer', {}).get('worker', {}).get('metallib_sha256') != profile['new_library_sha256']):
            raise ValueError('HC new fast artifact lacks exact97/full26-frame54-repeat qualification')
    # The registered compact origin keeps its own exact source-policy decoder.
    origin = Path(manifest['base'])
    adapter_path = origin / 'source/dev/benchmarks/expert_r4_compact_verify_worker_sep22/semantic_quality.py'
    if not adapter_path.exists():
        adapter_path = ROOT / 'dev/benchmarks/expert_r4_compact_verify_worker_sep22/semantic_quality.py'
    spec = importlib.util.spec_from_file_location('_hc_compact_origin', adapter_path)
    compact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compact)
    runner = compact.load(build=origin)
    http = compact.http

    def u64(value):
        return type(value) is int and 0 <= value < 2 ** 64

    def status(status, plan, store, execution_mode='mtp3'):
        status_value = status
        errors = []
        enabled = http.get_path(status_value, FIELDS[0])
        if type(enabled) is not bool:
            errors.append('HC VerifyR4 enable flag must be Boolean')
        if http.get_path(status_value, FIELDS[1]) != POLICY or http.get_path(status_value, FIELDS[2]) != CERTIFICATE:
            errors.append('HC VerifyR4 compiled policy/certificate differs')
        routes = http.get_path(status_value, 'identity.kernel_routes')
        occurrences = routes.count(MARKER) if isinstance(routes, str) else -1
        if occurrences != int(enabled is True):
            errors.append('HC VerifyR4 route marker differs from frozen enable flag')
        counters = http.get_path(status_value, 'hc_pad_verify_r4_route_counters')
        if not isinstance(counters, dict):
            return errors + ['HC VerifyR4 graph counters missing']
        if counters.get('scope') != 'main VerifyR4 graph construction; notGPU completion':
            errors.append('HC VerifyR4 counter scope differs')
        for name in COUNTERS:
            if not u64(counters.get(name)):
                errors.append('HC graph counter invalid: ' + name)
            elif enabled is False and counters[name]:
                errors.append('Disabled HC padding recorded work: ' + name)
        if all(u64(counters.get(name)) for name in COUNTERS):
            if counters['graph_rows'] != 4 * counters['graph_calls'] or counters['padding_dispatches_saved'] != counters['graph_calls']:
                errors.append('HC padding graph counters do not reconcile')
        return errors

    def coverage(before, after, case, require_counters=True, execution_mode='mtp3'):
        # The histogram contains actual completed singleton cycles, never a
        # guessed verification total. Original gates execute before this hook.
        added = []
        for side, value in [('before', before), ('after', after)]:
            added.extend(side + ': ' + error for error in status(value, None, None, execution_mode))
        for field in FIELDS:
            if not http.same_json(http.get_path(before, field), http.get_path(after, field)):
                added.append('HC ownership identity changed: ' + field)
        old_hist, new_hist = (http.get_path(value, 'mtp.completed_cycles_by_proposed_depth') for value in (before, after))
        if not (isinstance(old_hist, list) and isinstance(new_hist, list) and len(old_hist) == len(new_hist) == 16
                and all(u64(x) for x in old_hist + new_hist) and all(b >= a for a, b in zip(old_hist, new_hist))):
            return {'hc_graph_counter_deltas': None}, ['HC completed-depth histogram missing/decreasing']
        cycles4 = new_hist[3] - old_hist[3]
        enabled = http.get_path(after, FIELDS[0]) is True
        caller_allowed = (execution_mode == 'mtp3' and case.get('compact_scope', 'singleton-main') == 'singleton-main'
                          and case.get('body', {}).get('response_format', {}).get('type') != 'json_schema')
        expected = 97 * cycles4 if enabled and caller_allowed else 0
        if not caller_allowed and cycles4:
            added.append('Excluded HC caller advanced completedR4 singleton cycles')
        deltas = {}
        for name in COUNTERS:
            a, b = (http.get_path(value, 'hc_pad_verify_r4_route_counters.' + name) for value in (before, after))
            if not u64(a) or not u64(b) or b < a:
                added.append('HC counter missing/decreasing: ' + name)
            else:
                deltas[name] = b - a
        if len(deltas) == 3 and (deltas['graph_calls'] != expected or deltas['graph_rows'] != 4 * expected or deltas['padding_dispatches_saved'] != expected):
            added.append('HC graph deltas differ from97 qualifiedHC calls per actual depth3/R4 cycle')
        return {'hc_graph_counter_deltas': deltas, 'hc_expected_calls': expected,
                'hc_completed_depth3_cycles_delta': cycles4}, added

    def ownership(status):
        status_value = status
        return {field: http.get_path(status_value, field) for field in FIELDS}

    compact.install_hooks(status=status, coverage=coverage, ownership=ownership)
    return runner
