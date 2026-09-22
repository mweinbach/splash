"""CPU audits for the exact private Prefill-I8/Decode-Q4 phase profile.

These are graph-construction witnesses. Completed native requests and idle
snapshots remain separate requirements in the semantic runner.
"""
import re

from dev.benchmarks import qualify_flash_http as http

PHASE_SCHEMA = 'prefill-full512-i8-allrows-originalq4-decode-verify-v1'
PHASE_POLICY = ('all singleton Prefill rows Full512-I8; explicit Decode and every '
                'singleton Verify original packed Q4; original trained MTP')
PREFILL_F32_SELECTOR_POLICY = ('parent508 membership retained; missing null-policy coefficients '
    'select original RAW;296 qualified backing maps unchanged')
COUNTER_SCOPE = 'graph construction, not GPU completion'
LARGE_SCOPE = 'physical rows>=256; graph construction; excludes tiny prefill/decode/verifier'
STAGES = ('gate_up', 'down')
PHASES = ('prefill', 'decode', 'verify')
CACHE_COUNTERS = ('gate_up_graph_calls', 'gate_up_graph_rows', 'down_graph_calls',
                  'down_graph_rows', 'encoded_hit_dispatches', 'encoded_miss_dispatches',
                  'full_inventory_graph_calls')
PHASE_COUNTERS = tuple(f'{route}_{stage}_graph_{unit}'
                       for route in ('i8', 'q4') for stage in STAGES
                       for unit in ('calls', 'rows'))
F32_OWNERSHIP_FIELDS = (
    'identity.phase_f32_backing_count', 'identity.phase_f32_backing_bytes',
    'identity.phase_prefill_f32_selector_membership_count',
    'identity.phase_prefill_f32_selector_policy',
    'persisted_operands.f32_tensors', 'persisted_operands.f32_mapped_payload_bytes',
    'phase_f32_persistent_residency.retained_owner_count',
    'phase_f32_persistent_residency.retained_owner_bytes',
    'phase_f32_persistent_residency.transient_only_owner_count',
    'phase_f32_persistent_residency.transient_only_owner_bytes',
)


def phase_present(status):
    identity = status.get('identity', {}) if isinstance(status, dict) else {}
    return isinstance(identity, dict) and any(name in identity for name in (
        'target_phase_policy_schema', 'target_phase_policy'))


def phase_recognized(status):
    return (http.get_path(status, 'identity.target_phase_policy_schema') == PHASE_SCHEMA
            and http.get_path(status, 'identity.target_phase_policy') == PHASE_POLICY)


def _profile_errors(status):
    errors = []
    from dev.benchmarks import phase_saved_only_resource_quality
    errors.extend(phase_saved_only_resource_quality.status_errors(status))
    expected = {
        'identity.target_phase_policy_schema': PHASE_SCHEMA,
        'identity.target_phase_policy': PHASE_POLICY,
        'identity.target_hybrid_phase': True,
        'identity.target_all_rows_full512': False,
        'identity.original_target_gpu_omitted': False,
        'target_phase_graph_counters.enabled': True,
        'target_phase_graph_counters.scope': COUNTER_SCOPE,
    }
    for path, value in expected.items():
        if not http.same_json(http.get_path(status, path), value):
            errors.append('Explicit Prefill-I8/Decode-Q4 phase profile differs: ' + path)
    return errors


def _counter(value):
    return type(value) is int and 0 <= value < 2 ** 64


def phase_status_errors(status, store):
    """Bind a supplied composed derivative and its immutable resource inventory."""
    errors = _profile_errors(status)
    derivative = store.get('target_numerical_derivative_sha256')
    if (not isinstance(derivative, str) or not re.fullmatch(r'[0-9a-f]{64}', derivative)
            or http.get_path(status, 'identity.target_numerical_derivative_sha256') != derivative):
        errors.append('Explicit phase target numerical derivative identity differs or is unavailable')
    requirements = {
        'persisted_experts.expert_count': 24576,
        'identity.target_gathered_mpp_enabled': True,
        'identity.target_gathered_mpp_max_physical_rows': 4,
        'scheduler.maximum_prefill_rows': 2048,
        'scheduler.maximum_batch_prefill_rows_per_lane': 0,
        'capabilities.decode_batching': False,
        'capabilities.prefill_batching': False,
        'capabilities.batch_mtp': False,
        'capabilities.batch_mtp_prefill': False,
        'ple_storage.gpu_mapped_original_bytes': 74317889536,
        'identity.phase_f32_backing_count': 296,
        'identity.phase_f32_backing_bytes': 12097945600,
        'identity.phase_prefill_f32_selector_membership_count': 508,
        'identity.phase_prefill_f32_selector_policy': PREFILL_F32_SELECTOR_POLICY,
        'persisted_operands.f32_tensors': 296,
        'persisted_operands.f32_mapped_payload_bytes': 12097945600,
        'phase_f32_persistent_residency.retained_owner_count': 118,
        'phase_f32_persistent_residency.retained_owner_bytes': 3247964160,
        'phase_f32_persistent_residency.transient_only_owner_count': 178,
        'phase_f32_persistent_residency.transient_only_owner_bytes': 8849981440,
    }
    if store.get('inventory_per_layer') != 512 or store.get('expert_count') != 24576:
        errors.append('Explicit phase target requires the complete Full512 store witness')
    capacity = http.get_path(status, 'maximum_context_tokens')
    if type(capacity) is not int or not 0 < capacity <= 16384:
        errors.append('Explicit phase target exceeds its bounded maximum context')
    for path, value in requirements.items():
        if not http.same_json(http.get_path(status, path), value):
            errors.append('Explicit phase inventory/resource policy differs: ' + path)
    for phase in PHASES:
        for name in PHASE_COUNTERS:
            value = http.get_path(status, f'target_phase_graph_counters.{phase}.{name}')
            if not _counter(value):
                errors.append('Explicit phase graph counter is unavailable or invalid: ' + phase + '.' + name)
            elif name.startswith('q4_' if phase == 'prefill' else 'i8_') and value != 0:
                errors.append('Explicit phase encoded a forbidden expert route: ' + phase + '.' + name)
    return errors


def _deltas(before, after, prefix, names, errors):
    values = {}
    for name in names:
        path = prefix + '.' + name
        old, new = [http.get_path(status, path) for status in (before, after)]
        if not all(_counter(value) for value in (old, new)) or new < old:
            errors.append('Explicit phase graph counter is unavailable, invalid or decreased: ' + path)
        else:
            values[name] = new - old
    return values


def _expected_i8(windows):
    calls, rows = 48 * len(windows), 48 * sum(windows)
    return {'gate_up_graph_calls': calls, 'gate_up_graph_rows': rows,
            'down_graph_calls': calls, 'down_graph_rows': rows,
            'encoded_hit_dispatches': 2 * calls, 'encoded_miss_dispatches': 0,
            'full_inventory_graph_calls': 2 * calls}


def phase_coverage(before, after, windows):
    """Reconcile all I8 construction with Prefill and all Q4 with native AR/MTP."""
    errors = [f'{side}: {error}' for side, status in (('before', before), ('after', after))
              for error in _profile_errors(status)]
    for status in (before, after):
        if http.get_path(status, 'persisted_experts.expert_count') != 24576:
            errors.append('Explicit phase coverage lacks the complete Full512 inventory')
        if http.get_path(status, 'persisted_experts.graph_counters.large_row_counter_scope') != LARGE_SCOPE:
            errors.append('Explicit phase large-prefill graph counter scope differs')
    general = _deltas(before, after, 'persisted_experts.graph_counters', CACHE_COUNTERS, errors)
    large_names = tuple('large_row_' + name for name in CACHE_COUNTERS)
    large = {name.removeprefix('large_row_'): value for name, value in _deltas(
        before, after, 'persisted_experts.graph_counters', large_names, errors).items()}
    tiny = {name: general[name] - large[name] for name in CACHE_COUNTERS
            if name in general and name in large}
    tiny_windows = [rows for rows in windows if rows < 256]
    expected = _expected_i8(windows)
    expected_large = _expected_i8([rows for rows in windows if rows >= 256])
    expected_tiny = _expected_i8(tiny_windows)
    for label, actual, required in (('general Prefill', general, expected),
                                     ('large Prefill', large, expected_large),
                                     ('tiny Prefill general-minus-large', tiny, expected_tiny)):
        for name, value in required.items():
            if actual.get(name) != value:
                errors.append('Explicit phase ' + label + ' expert counter differs: ' + name)
    phases = {phase: _deltas(before, after, 'target_phase_graph_counters.' + phase,
                            PHASE_COUNTERS, errors) for phase in PHASES}
    native = _deltas(before, after, 'scheduler', ('decode_batches',), errors)
    mtp = _deltas(before, after, 'mtp', ('verification_cycles',), errors)
    batches, cycles = native.get('decode_batches'), mtp.get('verification_cycles')
    expected_q4 = None
    if batches is not None and cycles is not None:
        if cycles > batches:
            errors.append('Explicit phase verification cycles exceed native decode batches')
        else:
            expected_q4 = {'decode': 48 * (batches - cycles), 'verify': 48 * cycles}
    for phase, values in phases.items():
        for stage in STAGES:
            for unit in ('calls', 'rows'):
                forbidden = ('q4' if phase == 'prefill' else 'i8') + '_' + stage + '_graph_' + unit
                if values.get(forbidden) != 0:
                    errors.append('Explicit phase encoded a forbidden expert route: ' + phase + '.' + forbidden)
            if phase == 'prefill':
                for unit in ('calls', 'rows'):
                    name = stage + '_graph_' + unit
                    if values.get('i8_' + name) != expected[name] or values.get('i8_' + name) != general.get(name):
                        errors.append('Explicit phase Prefill-I8 counter differs from Store and frozen windows: ' + name)
            elif expected_q4 is not None:
                calls = values.get('q4_' + stage + '_graph_calls')
                rows = values.get('q4_' + stage + '_graph_rows')
                if calls != expected_q4[phase]:
                    errors.append('Explicit phase Q4 stage count differs from native AR/MTP commands: ' + phase + '.' + stage)
                if rows is not None and calls is not None:
                    if rows % 48 or (rows != calls if phase == 'decode' else not calls <= rows <= 4 * calls):
                        errors.append('Explicit phase Q4 stage row count differs from its bounded phase: ' + phase + '.' + stage)
        if values.get('q4_gate_up_graph_rows') != values.get('q4_down_graph_rows'):
            errors.append('Explicit phase Q4 paired stage rows differ: ' + phase)
    # Under this profile gathered MPP is exclusively tiny Prefill, cap4.
    gathered = _deltas(before, after, 'persisted_experts.graph_counters', tuple(
        'gathered_mpp_' + stage + '_graph_' + unit for stage in STAGES
        for unit in ('calls', 'rows')), errors)
    gathered_windows = [rows for rows in windows if rows <= 4]
    expected_gathered = _expected_i8(gathered_windows)
    for stage in STAGES:
        for unit in ('calls', 'rows'):
            name = stage + '_graph_' + unit
            if gathered.get('gathered_mpp_' + name) != expected_gathered[name]:
                errors.append('Explicit phase tiny-Prefill gathered-MPP counter differs: ' + name)
    return {
        'expert_graph_counter_deltas': large,
        'expert_graph_counters_available': len(general) == len(large) == len(CACHE_COUNTERS)
            and all(len(values) == len(PHASE_COUNTERS) for values in phases.values())
            and len(gathered) == 4 and batches is not None and cycles is not None,
        'target_all_rows_full512': False,
        'target_phase_policy_schema': PHASE_SCHEMA,
        'general_expert_graph_counter_deltas': general,
        'small_row_target_graph_counter_deltas': tiny,
        'tiny_prefill_graph_counter_deltas': tiny,
        'target_phase_graph_counter_deltas': phases,
        'gathered_prefill_graph_counter_deltas': gathered,
        'small_row_scope': 'tiny Prefill-I8 only; Decode/Verify use original Q4; graph construction, not GPU completion',
        'counter_scope': 'graph construction, not GPU completion; fresh idle/completed request counters establish completion separately',
    }, errors
