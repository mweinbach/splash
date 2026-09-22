"""CPU accounting fixtures for teacher bulk; no new native/GPU qualification."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from dev.benchmarks import prefill4k_attribution_quality as quality
from dev.benchmarks import singleton_teacher_bulk_quality as teacher
from dev.tests.flash.test_prefill_decode_phase_quality import coverage_fixture as phase_coverage_fixture


BULK_SCHEMA = 'singleton-teacher-original-cache-bulk2048-prefix128-v1'
BULK_SOURCE = '99b7f20ae3d78ad1b81a4fe47cef2a04c0ac327eea9732174e1d6e86c1bb7e1a'
WORKSPACE_BYTES = 310181888
PLAN_SHA256 = 'a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac'
WORK_COUNTERS = ['completed_teacher_commands', 'completed_pairs',
    'completed_original128_cache_prefix_windows', 'completed_original128_cache_prefix_rows',
    'completed_bulk_commands', 'completed_bulk_pairs', 'completed_tail_commands',
    'completed_tail_pairs']
# prompt, pairs, bulk commands/pairs, tail commands/pairs, prefix windows, API commands
VECTORS = [(1, 0, 0, 0, 0, 0, 0, 0), (128, 127, 0, 0, 1, 127, 1, 1),
    (129, 128, 1, 128, 0, 0, 1, 1), (256, 255, 1, 128, 1, 127, 2, 2),
    (257, 256, 1, 256, 0, 0, 2, 1), (2048, 2047, 1, 1920, 1, 127, 16, 2),
    (2049, 2048, 1, 2048, 0, 0, 16, 1), (2050, 2049, 1, 2048, 1, 1, 17, 2),
    (2176, 2175, 1, 2048, 1, 127, 17, 2), (2304, 2303, 2, 2176, 1, 127, 18, 3),
    (4096, 4095, 2, 3968, 1, 127, 32, 3), (8192, 8191, 4, 8064, 1, 127, 64, 5)]


def teacher_status(enabled=True, saved=None):
    status = copy.deepcopy(saved) if saved is not None else {'identity': {},
        'scheduler': {'maximum_prefill_rows': 2048},
        'mtp': {'teacher_cache_only_priming_calls': 0, 'eligible_requests': 0}}
    status['identity'].update(singleton_teacher_bulk_enabled=enabled,
        singleton_teacher_bulk_schema=BULK_SCHEMA if enabled else None,
        singleton_teacher_bulk_source_sha256=BULK_SOURCE if enabled else None)
    status['identity'].setdefault('target_numerical_derivative_sha256', 'c' * 64)
    if saved is None or enabled:
        status['mtp']['teacher_cache_only_requested'] = enabled
    status['mtp']['teacher_cache_only_priming_calls'] = 0
    status['mtp']['singleton_teacher_bulk'] = {'requested': enabled,
        'maximum_bulk_rows': 2048 if enabled else 0,
        'planned_workspace_bytes': WORKSPACE_BYTES if enabled else 0,
        'allocated_workspace_bytes': WORKSPACE_BYTES if enabled else 0,
        'legacy_teacher_call_scope': ('successful actual arena/API invocations; '
            'original128 logical cache prefixes counted separately'),
        'numeric_derivative_changed': False,
        'cache_math': ('original trained head global10240 RMS then four fc_hidden streams; '
            'unchanged M32/M16 descriptors; original chronological128 cache prefixes'),
        **{name: 0 for name in WORK_COUNTERS}}
    return status


def vector_counts(vector):
    _, pairs, bulk_commands, bulk_pairs, tail_commands, tail_pairs, prefixes, commands = vector
    return {'completed_teacher_commands': commands, 'completed_pairs': pairs,
        'completed_original128_cache_prefix_windows': prefixes,
        'completed_original128_cache_prefix_rows': pairs,
        'completed_bulk_commands': bulk_commands, 'completed_bulk_pairs': bulk_pairs,
        'completed_tail_commands': tail_commands, 'completed_tail_pairs': tail_pairs}


def windows_for(prompt):
    return [min(2048, prompt - start) for start in range(0, prompt, 2048)]


def teacher_fixture(vector, eligible=1):
    before, after = teacher_status(), teacher_status()
    if eligible:
        after['mtp']['singleton_teacher_bulk'].update(vector_counts(vector))
        after['mtp']['teacher_cache_only_priming_calls'] = vector[-1]
        after['mtp']['eligible_requests'] = 1
    return before, after, windows_for(vector[0])


class TeacherBulkAccountingTests(unittest.TestCase):
    def test_active_source_arena_and_workspace_are_bound(self):
        self.assertEqual(teacher.TEACHER_SCHEMA, BULK_SCHEMA)
        self.assertEqual(teacher.TEACHER_SOURCE_SHA, BULK_SOURCE)
        self.assertEqual(teacher.WORKSPACE_BYTES, WORKSPACE_BYTES)
        status = teacher_status()
        self.assertTrue(teacher.teacher_bulk_active(status))
        self.assertEqual(teacher.teacher_bulk_status_errors(status), [])
        store = {'target_numerical_derivative_sha256': 'c' * 64}
        self.assertEqual(teacher.teacher_bulk_status_errors(status, store), [])
        for changed in [{}, {'target_numerical_derivative_sha256': 'd' * 64},
            {'target_numerical_derivative_sha256': 'z' * 64}]:
            self.assertTrue(teacher.teacher_bulk_status_errors(status, changed))
        changes = [('identity', 'singleton_teacher_bulk_schema', 'unknown bulk schema'),
            ('identity', 'singleton_teacher_bulk_source_sha256', 'a' * 64),
            ('identity', 'singleton_teacher_bulk_enabled', False),
            ('mtp', 'requested', False), ('mtp', 'maximum_bulk_rows', 128),
            ('mtp', 'planned_workspace_bytes', WORKSPACE_BYTES - 1),
            ('mtp', 'allocated_workspace_bytes', WORKSPACE_BYTES + 1)]
        for group, field, value in changes:
            with self.subTest(group=group, field=field):
                broken = copy.deepcopy(status)
                target = broken['identity'] if group == 'identity' else broken['mtp']['singleton_teacher_bulk']
                target[field] = value
                self.assertTrue(teacher.teacher_bulk_active(broken))
                self.assertTrue(teacher.teacher_bulk_status_errors(broken))

    def test_boundary_vectors_count_real_commands_and_prefix_windows(self):
        for vector in VECTORS:
            with self.subTest(prompt=vector[0]):
                before, after, windows = teacher_fixture(vector)
                details, errors = teacher.teacher_bulk_coverage(before, after, windows, vector[0], 1)
                self.assertEqual(errors, [])
                self.assertEqual(details['expected_teacher_commands'], vector[-1])
                counts = after['mtp']['singleton_teacher_bulk']
                self.assertEqual(counts['completed_pairs'], vector[0] - 1)
                self.assertEqual(counts['completed_bulk_pairs'] + counts['completed_tail_pairs'], vector[0] - 1)

    def test_old_prefix_counts_cannot_be_reported_as_actual_api_calls(self):
        for prompt, fake_calls in [(2048, 16), (2049, 16), (4096, 32), (8192, 64)]:
            vector = next(vector for vector in VECTORS if vector[0] == prompt)
            before, after, windows = teacher_fixture(vector)
            after['mtp']['teacher_cache_only_priming_calls'] = fake_calls
            self.assertTrue(teacher.teacher_bulk_coverage(before, after, windows, prompt, 1)[1])

    def test_each_native_work_counter_is_recomputed(self):
        vector = next(vector for vector in VECTORS if vector[0] == 2048)
        before, after, windows = teacher_fixture(vector)
        for name in WORK_COUNTERS:
            with self.subTest(counter=name):
                broken = copy.deepcopy(after)
                broken['mtp']['singleton_teacher_bulk'][name] += 1
                self.assertTrue(teacher.teacher_bulk_coverage(before, broken, windows, 2048, 1)[1])

    def test_missing_malformed_decreasing_and_overflow_work_fails(self):
        vector = next(vector for vector in VECTORS if vector[0] == 2048)
        before, after, windows = teacher_fixture(vector)
        for value in [None, True, 2.0, -1, 2 ** 64]:
            broken = copy.deepcopy(after)
            broken['mtp']['singleton_teacher_bulk']['completed_teacher_commands'] = value
            self.assertTrue(teacher.teacher_bulk_coverage(before, broken, windows, 2048, 1)[1])
        for name in WORK_COUNTERS:
            broken = copy.deepcopy(after)
            broken['mtp']['singleton_teacher_bulk'].pop(name)
            self.assertTrue(teacher.teacher_bulk_coverage(before, broken, windows, 2048, 1)[1])
        broken = copy.deepcopy(before)
        broken['mtp']['singleton_teacher_bulk']['completed_pairs'] = 2048
        self.assertTrue(teacher.teacher_bulk_coverage(broken, after, windows, 2048, 1)[1])

    def test_constrained_request_requires_zero_new_teacher_work(self):
        vector = next(vector for vector in VECTORS if vector[0] == 2048)
        before, after, windows = teacher_fixture(vector, eligible=0)
        self.assertEqual(teacher.teacher_bulk_coverage(before, after, windows, 2048, 0)[1], [])
        # Prior greedy requests can have completed work; this request's delta must be zero.
        for status in [before, after]:
            status['mtp']['singleton_teacher_bulk'].update(vector_counts(vector))
            status['mtp']['teacher_cache_only_priming_calls'] = 2
        self.assertEqual(teacher.teacher_bulk_coverage(before, after, windows, 2048, 0)[1], [])
        for name in WORK_COUNTERS:
            broken = copy.deepcopy(after)
            broken['mtp']['singleton_teacher_bulk'][name] += 1
            self.assertTrue(teacher.teacher_bulk_coverage(before, broken, windows, 2048, 0)[1])

    def test_inactive_standard_fields_prove_zero_workspace_and_bulk_work(self):
        status = teacher_status(enabled=False)
        self.assertFalse(teacher.teacher_bulk_active(status))
        self.assertEqual(teacher.teacher_bulk_status_errors(status), [])
        self.assertFalse(teacher.teacher_bulk_active({'identity': {}, 'mtp': {}}))
        for name in ['maximum_bulk_rows', 'planned_workspace_bytes', 'allocated_workspace_bytes', *WORK_COUNTERS]:
            with self.subTest(field=name):
                broken = copy.deepcopy(status)
                broken['mtp']['singleton_teacher_bulk'][name] = 1
                self.assertTrue(teacher.teacher_bulk_status_errors(broken))

    def test_saved_standard_keeps_flag_zero_and_all_bulk_work_zero(self):
        path = quality.ROOT / ('build/release/flash/'
            'sep21-gdn-ab-qsa-standard-mtp3-model-and-quality-v1-standard.semantic.json')
        if not path.is_file():
            self.skipTest('Saved standard semantic fixture is unavailable')
        report = json.loads(path.read_text())
        plan = quality.read_plan(Path(report['plan']))
        case = next(case for case in plan['cases'] if case['id'] == 'json_nested_copy')
        row = next(row for row in report['cases'] if row['id'] == case['id'])
        before, after = [teacher_status(False, row[field])
            for field in ['status_before', 'status_after']]
        self.assertEqual(quality.coverage(before, after, case, execution_mode='standard')[1], [])
        for field in ['allocated_workspace_bytes', 'completed_teacher_commands',
            'completed_original128_cache_prefix_rows']:
            broken = copy.deepcopy(after)
            broken['mtp']['singleton_teacher_bulk'][field] = 1
            self.assertTrue(quality.coverage(before, broken, case, execution_mode='standard')[1])

    def test_teacher_bulk_and_prefill_i8_decode_q4_profiles_compose(self):
        for prompt, commands in [(2048, 2), (2049, 1), (8192, 5)]:
            with self.subTest(prompt=prompt):
                before, after, case, _ = phase_coverage_fixture(prompt)
                before, after = teacher_status(saved=before), teacher_status(saved=after)
                vector = next(vector for vector in VECTORS if vector[0] == prompt)
                after['mtp']['singleton_teacher_bulk'].update(vector_counts(vector))
                after['mtp']['teacher_cache_only_priming_calls'] = commands
                details, errors = quality.coverage(before, after, case)
                self.assertEqual(errors, [])
                self.assertEqual(details['teacher_actual_api_commands'], commands)
                self.assertEqual(details['teacher_successful_calls'], commands)
                self.assertEqual(details['tiny_prefill_graph_counter_deltas']['gate_up_graph_calls'],
                    48 if prompt == 2049 else 0)
                phases = details['target_phase_graph_counter_deltas']
                self.assertEqual(phases['prefill']['i8_gate_up_graph_rows'], 48 * prompt)
                self.assertEqual(phases['decode']['q4_gate_up_graph_calls'], 48)
                self.assertEqual(phases['verify']['q4_down_graph_calls'], 96)


class SavedTeacherBulkAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = quality.ROOT / ('build/release/flash/'
            'sep21-prefill-hc-pure-dense-w8a8-model-and-quality-v2-3.semantic.json')
        if not path.is_file():
            raise unittest.SkipTest('Saved HC actual22 generation fixture is unavailable')
        cls.saved = json.loads(path.read_text())
        cls.plan = quality.read_plan(Path(cls.saved['plan']))
        cls.frozen = {case['id']: case for case in cls.plan['cases']}

    def synthetic_report(self, enabled=True):
        report = copy.deepcopy(self.saved)
        report['label'] = 'synthetic CPU teacher bulk audit fixture'
        report['runtime_witness_scope'] = ('saved HC actual22 outputs; synthetic teacher statuses; '
            'no new native/GPU teacher qualification')
        for field in ['initial_status', 'final_status']:
            report[field] = teacher_status(enabled, report[field])
        report['actual_execution_policy'] = quality.execution_policy(report['initial_status'])
        for row in report['cases']:
            before, after = [teacher_status(enabled, row[field]) for field in ['status_before', 'status_after']]
            case = self.frozen[row['id']]
            eligible = case['body'].get('response_format', {}).get('type') != 'json_schema'
            if eligible:
                vector = next(vector for vector in VECTORS if vector[0] == case['prompt_token_count'])
                if enabled:
                    after['mtp']['singleton_teacher_bulk'].update(vector_counts(vector))
                    after['mtp']['teacher_cache_only_priming_calls'] = vector[-1]
                else:
                    after['mtp']['teacher_cache_only_priming_calls'] = vector[-2]
            row['status_before'], row['status_after'] = before, after
        return report

    def compare(self, candidate):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ['baseline.json', 'candidate.json']]
            for path, report in zip(paths, [self.synthetic_report(), candidate]):
                path.write_text(json.dumps(report, allow_nan=False))
            output = Path(directory) / 'comparison.json'
            with contextlib.redirect_stdout(io.StringIO()):
                status = quality.compare(SimpleNamespace(reports=paths, output=output,
                    allow_runtime_change=True))
            return status, json.loads(output.read_text())

    def test_frozen22_accounting_is_46_real_commands_and_432_prefixes(self):
        self.assertEqual(self.plan['content_sha256'], PLAN_SHA256)
        report = self.synthetic_report()
        totals = {name: 0 for name in WORK_COUNTERS}
        for row in report['cases']:
            case = self.frozen[row['id']]
            details, errors = quality.coverage(row['status_before'], row['status_after'], case)
            self.assertEqual(errors, [], row['id'])
            counters = row['status_after']['mtp']['singleton_teacher_bulk']
            for name in WORK_COUNTERS:
                totals[name] += counters[name]
        self.assertEqual(totals['completed_teacher_commands'], 46)
        self.assertEqual(totals['completed_original128_cache_prefix_windows'], 432)
        self.assertEqual(totals['completed_pairs'], 55277)
        status, audit = self.compare(report)
        self.assertEqual(status, 0)
        self.assertTrue(audit['valid'])
        self.assertTrue(audit['no_new_task_regressions'])
        self.assertEqual(audit['comparisons'][0]['generation_differences'], 0)
        self.assertEqual(audit['comparisons'][0]['candidate_task_passed_cases'], 20)

    def test_flag_zero_keeps_existing_128_command_coverage(self):
        report = self.synthetic_report(enabled=False)
        for row in report['cases']:
            self.assertEqual(quality.coverage(row['status_before'], row['status_after'], self.frozen[row['id']])[1], [], row['id'])
        row = next(row for row in report['cases'] if row['id'] == 'arithmetic_multiply')
        self.assertEqual(row['status_after']['mtp']['teacher_cache_only_priming_calls'], 16)
        row['status_after']['mtp']['teacher_cache_only_priming_calls'] = 2
        self.assertTrue(quality.coverage(row['status_before'], row['status_after'], self.frozen[row['id']])[1])

    def test_saved_command_prefix_row_and_constrained_tampering_is_rejected(self):
        for case_id, field, value in [('arithmetic_multiply', 'completed_teacher_commands', 16),
            ('json_nested_copy', 'completed_original128_cache_prefix_rows', 2047),
            ('json_arithmetic_schema', 'completed_tail_pairs', 1)]:
            with self.subTest(case=case_id, counter=field):
                report = self.synthetic_report()
                row = next(row for row in report['cases'] if row['id'] == case_id)
                row['status_after']['mtp']['singleton_teacher_bulk'][field] = value
                row['cache_use_coverage'] = {'expert_graph_counters_available': True}
                row['errors'] = []
                row['status'] = 'passed'
                report['valid'] = True
                status, audit = self.compare(report)
                self.assertEqual(status, 1)
                self.assertFalse(audit['valid'])


if __name__ == '__main__':
    unittest.main()
