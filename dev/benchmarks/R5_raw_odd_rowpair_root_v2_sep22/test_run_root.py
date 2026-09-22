"""CPU-only rejection tests using in-memory commands/reports and mocked groups.

Importing the sibling runner never invokes main(). These tests do not launch a
process or read model, token, capture, or actual Root report payloads.
"""

import errno
import importlib.util
import pathlib
import unittest
from unittest import mock


_SPEC = importlib.util.spec_from_file_location(
    'r5_raw_host_v2_runner_under_test',
    pathlib.Path(__file__).with_name('run_root.py'))
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('Cannot import the sibling CPU validator')
r = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(r)


TEARDOWN_KEYS = (
    'backend_destroyed', 'owned_handles_zero', 'sparse_resources_zero',
    'reserved_zero', 'denials_zero', 'backend_healthy_before_destroy',
    'backend_drained_before_destroy', 'sampled_process_cache_within_budget',
    'resource_gate_pass',
)


def command():
    build = r.ROOT / 'build/CPU-only-R5-HostV2-fixture'
    report = build / 'not-created-CPU-report.json'
    return {
        'schema': r.SCHEMA, 'cwd': str(r.ROOT), 'build': str(build),
        'report': str(report),
        'argv': [str(build / 'oracle'), '--gpu',
                 str(build / 'component.metallib'),
                 str(r.ROOT / 'install/local-models/Flash-Next-oQ4e-mtp-v1'),
                 str(report), '--shape', 'all', '--synthetic-cases'],
        'Root_only_GUV_bytes': 256 << 20,
        'selected_native_coefficients_per_case_max_bytes': 64 << 20,
        'input_scope': r.INPUT_SCOPE,
        'host_accounting_scope': 'owned-zero; measured-process-cache-bounded',
        'baseline_dispatches': 1, 'candidate_dispatches': 1,
        'minimum_warm_GPU_ms_each': 150, 'balanced_pairs': 18,
    }


def report():
    return {
        'pass': True, 'completed': True, 'completed_shape_cases': 7,
        'cases': [{'synthetic_case': n} for n in range(7)],
        'worker_integration': False, 'whole_model_qualified': False,
        'Gov_budget_bytes': 256 << 20,
        'teardown': dict.fromkeys(TEARDOWN_KEYS, True),
    }


class Tests(unittest.TestCase):
    def reject_command(self, value):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            r.validate_command(value)

    def reject_report(self, value):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            r.validate_report(value)

    def test_canonical_command(self):
        value = command()
        argv, path = r.validate_command(value)
        self.assertEqual(argv, value['argv'])
        self.assertEqual(path, pathlib.Path(value['report']))

    def test_repository_root_is_fixed_literal(self):
        self.assertEqual(str(r.ROOT), '/Users/mweinbach/Projects/splash')

    def test_each_command_field_is_required(self):
        for key in command():
            with self.subTest(field=key):
                value = command()
                del value[key]
                self.reject_command(value)

    def test_command_scope_and_protocol_changes_are_rejected(self):
        changes = {
            'schema': 'other', 'cwd': '/tmp',
            'Root_only_GUV_bytes': 512 << 20,
            'selected_native_coefficients_per_case_max_bytes': 65 << 20,
            'input_scope': 'actual model activations',
            'host_accounting_scope': 'owned-zero-only',
            'baseline_dispatches': 2, 'candidate_dispatches': 2,
            'minimum_warm_GPU_ms_each': 149, 'balanced_pairs': 16,
        }
        for key, replacement in changes.items():
            with self.subTest(field=key):
                value = command()
                value[key] = replacement
                self.reject_command(value)
        value = command()
        value['argv'][-2] = '6'
        self.reject_command(value)

    def test_command_build_outside_repository_is_rejected(self):
        value = command()
        old = value['build']
        value['build'] = '/tmp/CPU-only-R5-fixture'
        value['report'] = value['report'].replace(old, value['build'])
        value['argv'] = [part.replace(old, value['build'])
                         for part in value['argv']]
        self.reject_command(value)

    def test_command_boolean_dispatch_counts_are_rejected(self):
        for key in ('baseline_dispatches', 'candidate_dispatches'):
            with self.subTest(field=key):
                value = command()
                value[key] = True
                self.reject_command(value)

    def test_complete_all_seven_report(self):
        self.assertIsNone(r.validate_report(report()))

    def test_each_report_field_is_required(self):
        for key in report():
            with self.subTest(field=key):
                value = report()
                del value[key]
                self.reject_report(value)

    def test_report_pass_and_completion_require_true_boolean(self):
        for key in ('pass', 'completed'):
            for replacement in (False, None, 0, 1, 'true'):
                with self.subTest(field=key, replacement=replacement):
                    value = report()
                    value[key] = replacement
                    self.reject_report(value)

    def test_incomplete_six_and_wrong_case_counts_are_rejected(self):
        for count, cases in ((6, 6), (6, 7), (7, 6), (8, 7), (7, 8)):
            with self.subTest(count=count, cases=cases):
                value = report()
                value['completed_shape_cases'] = count
                value['cases'] = [{} for _ in range(cases)]
                self.reject_report(value)

    def test_report_count_requires_integer_not_boolean(self):
        for replacement in (True, False, 7.0, '7', None):
            with self.subTest(replacement=replacement):
                value = report()
                value['completed_shape_cases'] = replacement
                self.reject_report(value)

    def test_report_case_container_and_scope_are_exact(self):
        for key, replacement in (
                ('cases', tuple({} for _ in range(7))), ('cases', None),
                ('worker_integration', True), ('whole_model_qualified', True),
                ('worker_integration', 0), ('whole_model_qualified', 0),
                ('Gov_budget_bytes', 512 << 20)):
            with self.subTest(field=key):
                value = report()
                value[key] = replacement
                self.reject_report(value)

    def test_every_teardown_flag_is_required_and_true_boolean(self):
        for key in TEARDOWN_KEYS:
            with self.subTest(field=key, missing=True):
                value = report()
                del value['teardown'][key]
                self.reject_report(value)
            for replacement in (False, None, 0, 1, 'true'):
                with self.subTest(field=key, replacement=replacement):
                    value = report()
                    value['teardown'][key] = replacement
                    self.reject_report(value)
        for replacement in (None, [], True):
            value = report()
            value['teardown'] = replacement
            self.reject_report(value)

    def test_group_gone_requires_esrch(self):
        with mock.patch.object(r.os, 'killpg',
                               side_effect=ProcessLookupError(errno.ESRCH,
                                                              'fake ESRCH')) as call:
            self.assertIs(r.group_gone(12345), True)
            call.assert_called_once_with(12345, 0)

    def test_still_live_process_group_is_false(self):
        with mock.patch.object(r.os, 'killpg', return_value=None) as call:
            self.assertIs(r.group_gone(12345), False)
            call.assert_called_once_with(12345, 0)

    def test_group_permission_failure_cannot_prove_esrch(self):
        with mock.patch.object(r.os, 'killpg',
                               side_effect=PermissionError(errno.EPERM,
                                                           'fake EPERM')):
            with self.assertRaises(PermissionError):
                r.group_gone(12345)


if __name__ == '__main__':
    unittest.main()
