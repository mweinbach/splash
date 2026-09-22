"""CPU phase-policy audit fixtures; no phase runtime or model qualification."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from dev.benchmarks import prefill4k_attribution_quality as quality
from dev.benchmarks import prefill_decode_phase_quality as phase


PHASE_SCHEMA = 'prefill-full512-i8-allrows-originalq4-decode-verify-v1'
PHASE_POLICY = ('all singleton Prefill rows Full512-I8; explicit Decode and every singleton Verify '
    'original packed Q4; original trained MTP')
DERIVATIVE = 'f' * 64
PLAN_SHA256 = 'a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac'
LARGE_SCOPE = 'physical rows>=256; graph construction; excludes tiny prefill/decode/verifier'
COUNTER_SCOPE = 'graph construction, not GPU completion'
PHASE_FIELDS = ['i8_gate_up_graph_calls', 'i8_gate_up_graph_rows',
    'i8_down_graph_calls', 'i8_down_graph_rows', 'q4_gate_up_graph_calls',
    'q4_gate_up_graph_rows', 'q4_down_graph_calls', 'q4_down_graph_rows']


def set_stage(values, route, calls, rows):
    for stage in ['gate_up', 'down']:
        values[route + '_' + stage + '_graph_calls'] = calls
        values[route + '_' + stage + '_graph_rows'] = rows


def cache_counts(calls, rows):
    return {'gate_up_graph_calls': calls, 'gate_up_graph_rows': rows,
        'down_graph_calls': calls, 'down_graph_rows': rows,
        'encoded_hit_dispatches': 2 * calls, 'encoded_miss_dispatches': 0,
        'full_inventory_graph_calls': 2 * calls}


def phase_status(saved=None):
    status = copy.deepcopy(saved) if saved is not None else {
        'identity': {}, 'scheduler': {'maximum_prefill_rows': 2048, 'decode_batches': 0},
        'mtp': {'teacher_cache_only_priming_calls': 0, 'eligible_requests': 0,
            'verification_cycles': 0},
        'persisted_experts': {'expert_count': 24576}, 'persisted_operands': {}}
    status['identity'].update(target_phase_policy_schema=PHASE_SCHEMA,
        target_phase_policy=PHASE_POLICY, target_hybrid_phase=True,
        target_all_rows_full512=False, original_target_gpu_omitted=False,
        target_numerical_derivative_sha256=DERIVATIVE,
        target_gathered_mpp_enabled=True, target_gathered_mpp_max_physical_rows=4,
        phase_f32_backing_count=296, phase_f32_backing_bytes=12097945600,
        phase_prefill_f32_selector_membership_count=508,
        phase_prefill_f32_selector_policy=('parent508 membership retained; missing null-policy coefficients '
            'select original RAW;296 qualified backing maps unchanged'))
    status['maximum_context_tokens'] = 16384
    status['scheduler']['maximum_batch_prefill_rows_per_lane'] = 0
    status.setdefault('capabilities', {}).update(decode_batching=False,
        prefill_batching=False, batch_mtp=False, batch_mtp_prefill=False)
    status.setdefault('ple_storage', {})['gpu_mapped_original_bytes'] = 74317889536
    status['persisted_operands'].update(f32_tensors=296,
        f32_mapped_payload_bytes=12097945600)
    status['phase_f32_persistent_residency'] = {'retained_owner_count': 118,
        'retained_owner_bytes': 3247964160, 'transient_only_owner_count': 178,
        'transient_only_owner_bytes': 8849981440}
    status['target_phase_graph_counters'] = {'enabled': True, 'scope': COUNTER_SCOPE,
        **{name: {field: 0 for field in PHASE_FIELDS}
            for name in ['prefill', 'decode', 'verify']}}
    counters = {'scope': COUNTER_SCOPE, 'large_row_counter_scope': LARGE_SCOPE,
        **cache_counts(0, 0),
        **{'large_row_' + name: value for name, value in cache_counts(0, 0).items()},
        **{'gathered_mpp_' + stage + '_graph_' + unit: 0
            for stage in ['gate_up', 'down'] for unit in ['calls', 'rows']}}
    status['persisted_experts']['graph_counters'] = counters
    return status


def add_prefill(status, windows):
    large = [rows for rows in windows if rows >= 256]
    counters = status['persisted_experts']['graph_counters']
    counters.update(cache_counts(48 * len(windows), 48 * sum(windows)))
    counters.update({'large_row_' + name: value
        for name, value in cache_counts(48 * len(large), 48 * sum(large)).items()})
    set_stage(status['target_phase_graph_counters']['prefill'], 'i8',
        48 * len(windows), 48 * sum(windows))
    gathered = [rows for rows in windows if rows <= 4]
    for stage in ['gate_up', 'down']:
        counters['gathered_mpp_' + stage + '_graph_calls'] = 48 * len(gathered)
        counters['gathered_mpp_' + stage + '_graph_rows'] = 48 * sum(gathered)


def coverage_fixture(prompt=2049, constrained=False):
    before = phase_status()
    after = phase_status()
    windows = [min(2048, prompt - start) for start in range(0, prompt, 2048)]
    add_prefill(after, windows)
    after['scheduler']['decode_batches'] = 3
    after['mtp']['verification_cycles'] = 0 if constrained else 2
    set_stage(after['target_phase_graph_counters']['decode'], 'q4',
        144 if constrained else 48, 144 if constrained else 48)
    set_stage(after['target_phase_graph_counters']['verify'], 'q4',
        0 if constrained else 96, 0 if constrained else 384)
    if not constrained:
        after['mtp']['eligible_requests'] = 1
        after['mtp']['teacher_cache_only_priming_calls'] = sum(
            (rows - int(index == len(windows) - 1) + 127) // 128
            for index, rows in enumerate(windows))
    body = {'temperature': 0}
    if constrained:
        body['response_format'] = {'type': 'json_schema'}
    return before, after, {'prompt_token_count': prompt, 'body': body}, windows


class PhaseQualityFixtureTests(unittest.TestCase):
    def test_exact_native_phase_contract(self):
        self.assertEqual(phase.PHASE_SCHEMA, PHASE_SCHEMA)
        self.assertEqual(phase.PHASE_POLICY, PHASE_POLICY)
        store = {'inventory_per_layer': 512, 'expert_count': 24576,
            'target_numerical_derivative_sha256': DERIVATIVE}
        status = phase_status()
        self.assertEqual(phase.phase_status_errors(status, store), [])
        policy = quality.execution_policy(status)
        self.assertEqual(policy['identity.target_phase_policy_schema'], PHASE_SCHEMA)
        self.assertEqual(policy['identity.target_phase_policy'], PHASE_POLICY)
        self.assertNotIn('identity.target_phase_policy_schema', quality.execution_policy({'identity': {}}))
        ownership = quality.ownership_policy(status)
        self.assertEqual(ownership['identity.phase_f32_backing_count'], 296)
        self.assertEqual(ownership['phase_f32_persistent_residency.retained_owner_count'], 118)
        self.assertEqual(ownership['phase_f32_persistent_residency.transient_only_owner_count'], 178)
        self.assertFalse(phase.phase_present({'identity': {'target_hybrid_phase': True}}))
        for field in ['target_phase_policy_schema', 'target_phase_policy']:
            status = phase_status()
            status['identity'][field] = 'unknown phase policy'
            self.assertTrue(phase.phase_present(status))
            self.assertTrue(phase.phase_status_errors(status, store), field)

    def test_derivative_inventory_and_f32_scopes_are_explicit(self):
        store = {'inventory_per_layer': 512, 'expert_count': 24576,
            'target_numerical_derivative_sha256': DERIVATIVE}
        mutations = [('identity', 'target_hybrid_phase', False),
            ('identity', 'target_all_rows_full512', True),
            ('identity', 'original_target_gpu_omitted', True),
            ('identity', 'target_numerical_derivative_sha256', 'a' * 64),
            ('identity', 'target_gathered_mpp_enabled', False),
            ('identity', 'target_gathered_mpp_max_physical_rows', 8),
            ('scheduler', 'maximum_prefill_rows', 1024),
            ('scheduler', 'maximum_batch_prefill_rows_per_lane', 2048),
            ('capabilities', 'decode_batching', True),
            ('capabilities', 'prefill_batching', True),
            ('capabilities', 'batch_mtp', True),
            ('capabilities', 'batch_mtp_prefill', True),
            ('ple_storage', 'gpu_mapped_original_bytes', 6370164736),
            ('identity', 'phase_f32_backing_count', 118),
            ('identity', 'phase_f32_backing_bytes', 3247964160),
            ('identity', 'phase_prefill_f32_selector_membership_count', 296),
            ('identity', 'phase_prefill_f32_selector_policy', 'only296 policy-qualified prefixes'),
            ('persisted_operands', 'f32_tensors', 118),
            ('persisted_operands', 'f32_mapped_payload_bytes', 3247964160),
            ('phase_f32_persistent_residency', 'retained_owner_count', 296),
            ('phase_f32_persistent_residency', 'retained_owner_bytes', 12097945600),
            ('phase_f32_persistent_residency', 'transient_only_owner_count', 0),
            ('phase_f32_persistent_residency', 'transient_only_owner_bytes', 0)]
        for group, field, value in mutations:
            with self.subTest(group=group, field=field):
                status = phase_status()
                status[group][field] = value
                self.assertTrue(phase.phase_status_errors(status, store))
        for changed in [{}, {'inventory_per_layer': 512},
            {'inventory_per_layer': 256, 'target_numerical_derivative_sha256': DERIVATIVE}]:
            self.assertTrue(phase.phase_status_errors(phase_status(), changed))
        legacy_phase = phase_status()
        legacy_phase['identity'].pop('phase_prefill_f32_selector_membership_count')
        legacy_phase['identity'].pop('phase_prefill_f32_selector_policy')
        self.assertTrue(phase.phase_status_errors(legacy_phase, store))

    def test_all_prefill_windows_keep_i8_and_continuation_uses_q4(self):
        # These are CPU audit inputs, not additions to the frozen22 task plan.
        tails = [1, 2, 3, 4, 8, *range(9, 17), 32, 64, 128, 256, 512, 1024]
        for prompt in [2048, 4096, 8192, *[2048 + tail for tail in tails]]:
            with self.subTest(prompt=prompt):
                before, after, case, windows = coverage_fixture(prompt)
                details, errors = quality.coverage(before, after, case)
                self.assertEqual(errors, [])
                self.assertEqual(details['actual_main_row_windows'], windows)
                self.assertEqual(details['target_phase_policy_schema'], PHASE_SCHEMA)
                self.assertEqual(details['expert_graph_counter_deltas']['gate_up_graph_calls'],
                    48 * len([rows for rows in windows if rows >= 256]))
                self.assertEqual(details['general_expert_graph_counter_deltas']['gate_up_graph_rows'],
                    48 * prompt)

    def test_phase_counter_tampering_cannot_be_hidden_by_general_totals(self):
        before, after, case, windows = coverage_fixture()
        changes = [('decode', 'i8_gate_up_graph_calls', 48),
            ('verify', 'i8_down_graph_rows', 48),
            ('prefill', 'q4_gate_up_graph_calls', 48),
            ('prefill', 'i8_gate_up_graph_calls', 48),
            ('decode', 'q4_down_graph_calls', 0),
            ('decode', 'q4_down_graph_rows', 96),
            ('verify', 'q4_gate_up_graph_calls', 48),
            ('verify', 'q4_down_graph_rows', 385)]
        for group, field, value in changes:
            with self.subTest(group=group, field=field):
                broken = copy.deepcopy(after)
                broken['target_phase_graph_counters'][group][field] = value
                self.assertTrue(phase.phase_coverage(before, broken, windows)[1])
                self.assertTrue(quality.coverage(before, broken, case)[1])
        for group, field in [('scheduler', 'decode_batches'), ('mtp', 'verification_cycles')]:
            broken = copy.deepcopy(after)
            broken[group][field] += 1
            self.assertTrue(phase.phase_coverage(before, broken, windows)[1])

    def test_missing_malformed_and_negative_phase_evidence_fails_closed(self):
        before, after, case, windows = coverage_fixture()
        for value in [None, True, 96.0, -1, 2 ** 64]:
            with self.subTest(value=value):
                broken = copy.deepcopy(after)
                broken['target_phase_graph_counters']['prefill']['i8_gate_up_graph_calls'] = value
                self.assertTrue(phase.phase_coverage(before, broken, windows)[1])
        for field in ['enabled', 'scope', 'prefill', 'decode', 'verify']:
            broken = copy.deepcopy(after)
            broken['target_phase_graph_counters'].pop(field)
            self.assertTrue(phase.phase_coverage(before, broken, windows)[1])
        for group, field in [('target_phase_graph_counters', 'scope'),
            ('persisted_experts', 'graph_counters')]:
            broken = copy.deepcopy(after)
            broken[group].pop(field)
            self.assertTrue(quality.coverage(before, broken, case, require_counters=False)[1])
        broken = copy.deepcopy(before)
        broken['target_phase_graph_counters']['verify']['q4_gate_up_graph_calls'] = 100
        self.assertTrue(phase.phase_coverage(broken, after, windows)[1])

    def test_verify_rows_cover_the_fixed_depth_bounds(self):
        before, after, _, windows = coverage_fixture()
        for rows in range(96, 385, 48):
            changed = copy.deepcopy(after)
            set_stage(changed['target_phase_graph_counters']['verify'], 'q4', 96, rows)
            self.assertEqual(phase.phase_coverage(before, changed, windows)[1], [], rows)
        for rows in [48, 432]:
            changed = copy.deepcopy(after)
            set_stage(changed['target_phase_graph_counters']['verify'], 'q4', 96, rows)
            self.assertTrue(phase.phase_coverage(before, changed, windows)[1], rows)
        after['scheduler']['decode_batches'] = after['mtp']['verification_cycles'] = 0
        for name in ['decode', 'verify']:
            set_stage(after['target_phase_graph_counters'][name], 'q4', 0, 0)
        self.assertEqual(phase.phase_coverage(before, after, windows)[1], [])

    def test_general_minus_large_proves_the_r1_prefill_tail(self):
        before, after, case, windows = coverage_fixture()
        broken = copy.deepcopy(after)
        counters = broken['persisted_experts']['graph_counters']
        counters.update(cache_counts(48, 48 * 2048))
        self.assertTrue(phase.phase_coverage(before, broken, windows)[1])
        for field, value in [('encoded_hit_dispatches', 191),
            ('encoded_miss_dispatches', 1), ('full_inventory_graph_calls', 191)]:
            broken = copy.deepcopy(after)
            broken['persisted_experts']['graph_counters'][field] = value
            self.assertTrue(phase.phase_coverage(before, broken, windows)[1])
        broken = copy.deepcopy(after)
        broken['persisted_experts']['graph_counters']['large_row_counter_scope'] = 'all rows'
        self.assertTrue(quality.coverage(before, broken, case)[1])

    def test_frozen_teacher_and_constrained_gates_still_apply(self):
        before, after, case, _ = coverage_fixture()
        self.assertEqual(after['mtp']['teacher_cache_only_priming_calls'], 16)
        after['mtp']['teacher_cache_only_priming_calls'] = 15
        self.assertTrue(quality.coverage(before, after, case)[1])
        before, after, case, _ = coverage_fixture(constrained=True)
        self.assertEqual(quality.coverage(before, after, case)[1], [])
        after['mtp']['teacher_cache_only_priming_calls'] = 1
        self.assertTrue(quality.coverage(before, after, case)[1])
        before, after, case, windows = coverage_fixture(constrained=True)
        after['mtp']['verification_cycles'] = 1
        set_stage(after['target_phase_graph_counters']['decode'], 'q4', 96, 96)
        set_stage(after['target_phase_graph_counters']['verify'], 'q4', 48, 48)
        self.assertEqual(phase.phase_coverage(before, after, windows)[1], [])
        self.assertTrue(quality.coverage(before, after, case)[1])

    def test_standard_requires_the_saved_absent_head_and_zero_mtp_work(self):
        report_path = quality.ROOT / ('build/release/flash/'
            'sep21-gdn-ab-qsa-standard-mtp3-model-and-quality-v1-standard.semantic.json')
        if not report_path.is_file():
            self.skipTest('Saved standard semantic fixture is unavailable')
        saved = json.loads(report_path.read_text())['initial_status']
        before, after = phase_status(saved), phase_status(saved)
        add_prefill(after, [2048, 1])
        after['scheduler']['decode_batches'] += 1
        set_stage(after['target_phase_graph_counters']['decode'], 'q4', 48, 48)
        case = {'prompt_token_count': 2049, 'body': {'temperature': 0}}
        self.assertEqual(quality.coverage(before, after, case, execution_mode='standard')[1], [])
        self.assertTrue(quality.coverage(before, after, case)[1])
        for group, field, value in [('mtp', 'teacher_cache_only_priming_calls', 1),
            ('mtp', 'verification_cycles', 1), ('capabilities', 'mtp', True)]:
            broken = copy.deepcopy(after)
            broken[group][field] = value
            self.assertTrue(quality.coverage(before, broken, case, execution_mode='standard')[1])


class SavedPhaseComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report_path = quality.ROOT / ('build/release/flash/'
            'sep21-prefill-hc-pure-dense-w8a8-model-and-quality-v2-3.semantic.json')
        if not report_path.is_file():
            raise unittest.SkipTest('Saved HC actual22 generation fixture is unavailable')
        cls.baseline = json.loads(report_path.read_text())
        cls.plan = quality.read_plan(Path(cls.baseline['plan']))
        cls.frozen = {case['id']: case for case in cls.plan['cases']}

    def synthetic_phase_report(self):
        report = copy.deepcopy(self.baseline)
        report['label'] = 'synthetic CPU phase audit fixture'
        report['runtime_witness_scope'] = ('saved HC actual22 task responses; synthetic phase statuses; '
            'no phase runtime or numerical qualification')
        report['runtime_file_sha256'] = {'splash-flash': '0' * 64, 'splash.metallib': '1' * 64}
        report['store_witness']['target_numerical_derivative_sha256'] = DERIVATIVE
        for field in ['initial_status', 'final_status']:
            report[field] = phase_status(report[field])
        report['actual_execution_policy'] = quality.execution_policy(report['initial_status'])
        for row in report['cases']:
            before, after = [phase_status(row[field]) for field in ['status_before', 'status_after']]
            prompt = self.frozen[row['id']]['prompt_token_count']
            windows = [min(2048, prompt - start) for start in range(0, prompt, 2048)]
            add_prefill(after, windows)
            cycles = after['mtp']['verification_cycles'] - before['mtp']['verification_cycles']
            decode_batches = after['scheduler']['decode_batches'] - before['scheduler']['decode_batches']
            set_stage(after['target_phase_graph_counters']['decode'], 'q4',
                48 * (decode_batches - cycles), 48 * (decode_batches - cycles))
            set_stage(after['target_phase_graph_counters']['verify'], 'q4', 48 * cycles, 48 * cycles)
            row['status_before'], row['status_after'] = before, after
        return report

    def compare(self, candidate, allow_runtime_change=True):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ['baseline.json', 'candidate.json']]
            for path, report in zip(paths, [self.synthetic_phase_report(), candidate]):
                path.write_text(json.dumps(report, allow_nan=False))
            output = Path(directory) / 'comparison.json'
            with contextlib.redirect_stdout(io.StringIO()):
                status = quality.compare(SimpleNamespace(reports=paths, output=output,
                    allow_runtime_change=allow_runtime_change))
            return status, json.loads(output.read_text())

    def test_saved_actual22_outputs_regrade_without_redefining_tasks(self):
        self.assertEqual(self.plan['content_sha256'], PLAN_SHA256)
        self.assertEqual(len(self.plan['cases']), 22)
        candidate = self.synthetic_phase_report()
        status, audit = self.compare(candidate)
        self.assertEqual(status, 0)
        self.assertTrue(audit['valid'])
        self.assertTrue(audit['no_new_task_regressions'])
        self.assertFalse(audit['all_tasks_pass'])
        self.assertEqual(audit['comparisons'][0]['candidate_task_passed_cases'], 20)
        self.assertEqual(audit['comparisons'][0]['generation_differences'], 0)
        for row in candidate['cases']:
            self.assertEqual(row['records'][0]['request_body'],
                next(old for old in self.baseline['cases'] if old['id'] == row['id'])['records'][0]['request_body'])

    def test_saved_native_evidence_tampering_is_recomputed(self):
        def phase_decode(status):
            status['target_phase_graph_counters']['decode']['i8_gate_up_graph_calls'] = 48
        def no_r1(status):
            status['persisted_experts']['graph_counters'].update(cache_counts(48, 48 * 2048))
        def no_completed(status):
            status['requests']['completed'] -= 1
        def stale(status):
            status['transport']['status_stale'] = True
        for mutate in [phase_decode, no_r1, no_completed, stale]:
            with self.subTest(mutation=mutate.__name__):
                candidate = self.synthetic_phase_report()
                row = next(row for row in candidate['cases'] if row['id'] == 'json_nested_copy')
                mutate(row['status_after'])
                candidate['valid'] = True
                row['errors'] = []
                row['status'] = 'passed'
                row['cache_use_coverage'] = {'expert_graph_counters_available': True}
                status, audit = self.compare(candidate)
                self.assertEqual(status, 1)
                self.assertFalse(audit['valid'])
                self.assertFalse(audit['recomputed_report_audits'][1]['evidence_valid'])
        candidate = self.synthetic_phase_report()
        row = candidate['cases'][0]
        row['status_after']['status_snapshot']['steady_seconds'] = row['status_before']['status_snapshot']['steady_seconds']
        self.assertFalse(self.compare(candidate)[1]['valid'])

    def test_changed_actual_request_cannot_use_a_frozen_hash_summary(self):
        candidate = self.synthetic_phase_report()
        candidate['cases'][0]['records'][0]['request_body']['max_completion_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'Actual task request differs from frozen body'):
            self.compare(candidate)

    def test_runtime_change_still_requires_the_existing_explicit_gate(self):
        candidate = self.synthetic_phase_report()
        candidate['runtime_file_sha256']['splash-flash'] = '2' * 64
        with self.assertRaisesRegex(ValueError, 'requires explicit --allow-runtime-change'):
            self.compare(candidate, allow_runtime_change=False)
        self.assertEqual(self.compare(candidate)[0], 0)

    def test_forbidden_merge_attribute_remains_a_new_semantic_failure(self):
        candidate = self.synthetic_phase_report()
        row = next(row for row in candidate['cases'] if row['id'] == 'python_merge_counts')
        record = row['records'][0]
        record['text'] = ('def merge_counts(pairs):\n    result = {}\n    for key, amount in pairs:\n'
            '        result[key] = result.get(key, 0) + amount\n    return result')
        record['task_errors'] = []
        row['task_passed'] = True
        candidate['task_passed_cases'] = 22
        errors, details = quality.grade_record(record, self.frozen[row['id']])
        self.assertTrue(errors)
        self.assertFalse(details['execution_started'])
        self.assertEqual(details['passed_tests'], 0)
        status, audit = self.compare(candidate)
        self.assertEqual(status, 0)  # Evidence validity is independent of task success.
        self.assertTrue(audit['valid'])
        self.assertFalse(audit['no_new_task_regressions'])
        self.assertEqual(audit['comparisons'][0]['new_task_regression_ids'], ['python_merge_counts'])

    def test_different_valid_merge_text_is_only_a_generation_diagnostic(self):
        candidate = self.synthetic_phase_report()
        row = next(row for row in candidate['cases'] if row['id'] == 'python_merge_counts')
        row['records'][0]['text'] = ('def merge_counts(pairs):\n'
            '    return {key: sum(value for item, value in pairs if item == key) '
            'for key in set(item for item, value in pairs)}')
        status, audit = self.compare(candidate)
        self.assertEqual(status, 0)
        self.assertTrue(audit['no_new_task_regressions'])
        self.assertEqual(audit['comparisons'][0]['generation_differences'], 1)


if __name__ == '__main__':
    unittest.main()
