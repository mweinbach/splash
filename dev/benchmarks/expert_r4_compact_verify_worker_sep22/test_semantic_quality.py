"""Focused CPU-only private adapter checks; no model/saved-generation fixtures."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

from dev.benchmarks import prefill4k_attribution_quality as original
from dev.benchmarks.expert_r4_compact_verify_worker_sep22 import semantic_quality as quality
from dev.tests.flash import test_prefill4k_semantic_quality as original_tests


class CompactVerifySemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = Path(os.environ.get('COMPACT_R4_WORKER_BUILD', quality.DEFAULT_BUILD)).resolve()
        cls.source = quality.configure_build(cls.build)
        cls.marker = quality._marker_prefix + cls.source

    def status(self, enabled=True, histogram=None, compact_calls=0, pref_calls=0, pref_rows=0,
               prime=0, eligible=0):
        histogram = [0] * 16 if histogram is None else list(histogram)
        verify_calls = 48 * sum(histogram)
        verify_rows = 48 * sum((depth + 1) * count for depth, count in enumerate(histogram))
        legacy = original_tests.SemanticQualityTests().status(24576, pref_rows + verify_rows,
            pref_calls + verify_calls, prime, eligible)
        legacy['persisted_experts']['enabled'] = True
        counters = legacy['persisted_experts']['graph_counters']
        large = original_tests.SemanticQualityTests().status(24576, pref_rows, pref_calls, prime, eligible)['persisted_experts']['graph_counters']
        counters.update({'large_row_' + key: value for key, value in large.items()})
        counters['large_row_counter_scope'] = 'physical rows>=256; graph construction; excludes tiny prefill/decode/verifier'
        legacy['identity'] = {'target_all_rows_full512': True, 'target_gathered_mpp_enabled': True,
            'target_gathered_mpp_max_physical_rows': 4, 'kernel_routes': 'parent-frozen-kernels' + (self.marker if enabled else '')}
        legacy['mtp']['completed_cycles_by_proposed_depth'] = histogram
        legacy['mtp']['verification_cycles'] = sum(histogram)
        legacy[quality.SECTION] = {'schema': quality.PROFILE_SCHEMA, 'scope': quality.PROFILE_SCOPE,
            'requested': enabled, 'enabled': enabled, 'source_identity_sha256': self.source,
            'dispatches_per_layer': 6, 'base_gather_dispatches_per_layer': 2,
            'planner_threadgroup_bytes': 2432, 'additional_gpu_allocation_bytes': 0,
            'full_model_quality_qualified': False,
            **{stage + '_graph_' + kind: compact_calls * (4 if kind == 'rows' else 1)
               for stage in ('plan', 'gate', 'down') for kind in ('calls', 'rows')}}
        return legacy

    def mixed(self, enabled=True):
        hist = [0] * 16
        hist[1], hist[2], hist[3] = 1, 1, 3
        return self.status(enabled), self.status(enabled, hist, 144 if enabled else 0,
            pref_calls=96, pref_rows=48 * 4096, prime=32, eligible=1)

    def test_all22_specs_bodies_graders_and_shared_module_unchanged(self):
        self.assertEqual(quality.specs(), original.specs())
        self.assertEqual(len(quality.specs()), 22)
        self.assertIsNot(quality._base, original)
        self.assertIsNot(quality._base.gate_status, original.gate_status)
        self.assertEqual(quality._base.EXECUTION_POLICY_FIELDS, original.EXECUTION_POLICY_FIELDS)
        # Existing pure base acceptance/grading/cache tests run unchanged.
        # Their two published/saved-artifact methods remain in the original
        # class, and are deliberately not opened by this CPU-only component.
        methods = [name for name in unittest.defaultTestLoader.getTestCaseNames(original_tests.SemanticQualityTests)
            if name not in {'test_published_top256_witness_and_partial_projection_policy',
                'test_real_top64_failures_remain_visible_in_complete_comparison'}]
        suite = unittest.TestSuite(original_tests.SemanticQualityTests(name) for name in methods)
        result = unittest.TestResult(); suite.run(result)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])

    def test_exact_compiled_active_and_disabled_status(self):
        for enabled in (False, True):
            self.assertEqual(quality.compact_status_errors(self.status(enabled)), [])
        active = self.status(True, [0, 0, 0, 2] + [0] * 12, 96)
        self.assertEqual(quality.compact_status_errors(active), [])

    def test_compiled_source_identity_tuple_and_registered_component_fail_closed(self):
        manifest_path = self.build / 'overlay-manifest.json'
        original_manifest = manifest_path.read_text()
        read_text = Path.read_text
        for field, value in (('source_identity_sha256', 'f' * 64), ('qualified_parallel_source_identity', 'f' * 64)):
            manifest = quality.http.strict_json(original_manifest); manifest[field] = value
            with mock.patch.object(Path, 'read_text', autospec=True) as reader:
                reader.side_effect = lambda path, *args, **kwargs: __import__('json').dumps(manifest) if path == manifest_path else read_text(path, *args, **kwargs)
                with self.assertRaises(ValueError): quality.configure_build(self.build)
        with mock.patch.object(Path, 'read_bytes', return_value=b'corrupt compiled program source'):
            with self.assertRaises(ValueError): quality.configure_build(self.build)
        quality.configure_build(self.build)

    def test_identity_profile_and_requested_counter_agreement_fail_closed(self):
        valid = self.status()
        edits = [(quality.SECTION + '.source_identity_sha256', 'f' * 64),
            (quality.SECTION + '.schema', 'unknown-v2'), (quality.SECTION + '.scope', 'all callers'),
            (quality.SECTION + '.requested', False), (quality.SECTION + '.requested', 1),
            (quality.SECTION + '.enabled', 1), (quality.SECTION + '.dispatches_per_layer', 2),
            (quality.SECTION + '.planner_threadgroup_bytes', 0),
            (quality.SECTION + '.additional_gpu_allocation_bytes', True),
            (quality.SECTION + '.full_model_quality_qualified', True),
            ('identity.kernel_routes', 'unknown-prefix'),
            ('identity.kernel_routes', valid['identity']['kernel_routes'].replace(self.source, 'f' * 64)),
            ('identity.kernel_routes', valid['identity']['kernel_routes'] + self.marker)]
        for path, value in edits:
            bad = copy.deepcopy(valid); parent = bad
            parts = path.split('.')
            for key in parts[:-1]: parent = parent[key]
            parent[parts[-1]] = value
            self.assertTrue(quality.compact_status_errors(bad), path)
        for key in ('requested', 'source_identity_sha256', 'plan_graph_calls'):
            bad = copy.deepcopy(valid); del bad[quality.SECTION][key]
            self.assertTrue(quality.compact_status_errors(bad), key)
        bad = copy.deepcopy(valid); del bad[quality.SECTION]
        self.assertTrue(quality.compact_status_errors(bad))
        bad = self.status(False); bad['identity']['kernel_routes'] += self.marker
        self.assertTrue(quality.compact_status_errors(bad))

    def test_each_counter_rejects_bad_types_values_missing_and_decrease(self):
        for field in quality.COUNTERS:
            for invalid in (True, False, 0., -1, 2 ** 64, None):
                bad = self.status(); bad[quality.SECTION][field] = invalid
                self.assertTrue(quality.compact_status_errors(bad), (field, invalid))
            bad = self.status(); del bad[quality.SECTION][field]
            self.assertTrue(quality.compact_status_errors(bad), field)
            before, after = self.status(True, compact_calls=1), self.status(True, compact_calls=1)
            after[quality.SECTION][field] -= 1
            self.assertTrue(quality.compact_coverage(before, after, {'body': {}})[1], field)

    def test_actual_mixed_depth_histogram_not_total_verification_cycles(self):
        before, after = self.mixed()
        details, errors = quality.coverage(before, after, {'prompt_token_count': 4096, 'body': {'temperature': 0}})
        self.assertEqual(errors, [])
        self.assertEqual(details['compact_expected_stage_calls'], 144)
        self.assertEqual(details['compact_expected_stage_rows'], 576)
        self.assertEqual(details['compact_expected_graph_dispatches'], 864)
        self.assertEqual(details['compact_base_gather_graph_dispatches'], 288)
        self.assertEqual(details['expert_graph_counter_deltas']['encoded_hit_dispatches'], 192)
        self.assertEqual(details['small_row_target_graph_counter_deltas']['gate_up_graph_calls'], 240)
        for field in quality.COUNTERS:
            for delta in (-1, 1):
                bad = copy.deepcopy(after); bad[quality.SECTION][field] += delta
                self.assertTrue(quality.compact_coverage(before, bad, {'body': {}})[1], (field, delta))

    def test_disabled_greedy_cycles_are_old_route_not_compact_work(self):
        before, after = self.mixed(False)
        details, errors = quality.coverage(before, after, {'prompt_token_count': 4096, 'body': {'temperature': 0}})
        self.assertEqual(errors, [])
        self.assertEqual(details['compact_expected_stage_calls'], 0)
        self.assertTrue(all(value == 0 for value in details['compact_native_r4_verify_counter_deltas'].values()))

    def test_enabled_no_depth3_cycles_does_not_require_positive_new_route(self):
        before = self.status()
        hist = [0] * 16; hist[1] = 2; hist[2] = 3
        after = self.status(True, hist, compact_calls=0)
        details, errors = quality.compact_coverage(before, after, {'body': {}})
        self.assertEqual(errors, [])
        self.assertEqual(details['compact_expected_stage_calls'], 0)

    def test_strict_histogram_and_provenance_monotonicity(self):
        before, after = self.mixed()
        for invalid in (None, [0] * 15, [0] * 17, [True] + [0] * 15,
                [-1] + [0] * 15, [0.] + [0] * 15, [2 ** 64] + [0] * 15):
            bad = copy.deepcopy(after); bad['mtp']['completed_cycles_by_proposed_depth'] = invalid
            self.assertTrue(quality.compact_coverage(before, bad, {'body': {}})[1])
        old = copy.deepcopy(after); new = copy.deepcopy(after)
        new['mtp']['completed_cycles_by_proposed_depth'][3] -= 1
        self.assertTrue(quality.compact_coverage(old, new, {'body': {}})[1])
        bad = self.status(False)
        self.assertTrue(quality.compact_coverage(before, bad, {'body': {}})[1])

    def test_standard_constrained_prefill_ar_head_batch_zero_compact(self):
        for scope in ('prefill-only', 'autoregressive', 'trained-head', 'batch'):
            before, after = self.status(), self.status(pref_calls=48, pref_rows=48 * 2048)
            after['mtp']['joint_target_verifiers_by_width'] = [0, 0, 0, 3]
            details, errors = quality.compact_coverage(before, after, {'body': {}, 'compact_scope': scope})
            self.assertEqual(errors, [], scope)
            self.assertEqual(details['compact_expected_stage_calls'], 0)
            self.assertGreater(after['persisted_experts']['graph_counters']['gate_up_graph_calls'], 0)
            bad = copy.deepcopy(after); bad[quality.SECTION]['plan_graph_calls'] = 1
            self.assertTrue(quality.compact_coverage(before, bad, {'body': {}, 'compact_scope': scope})[1])
        for mode, body in (('standard', {}), ('mtp3', {'response_format': {'type': 'json_schema'}})):
            self.assertEqual(quality.compact_coverage(self.status(), self.status(), {'body': body}, mode)[1], [])
            before, after = self.mixed()
            self.assertTrue(quality.compact_coverage(before, after, {'body': body}, mode)[1])

    def test_both_original_coverage_return_paths_keep_existing_errors(self):
        before, after = self.mixed()
        case = {'prompt_token_count': 4096, 'body': {'temperature': 0}}
        broken = copy.deepcopy(after); broken['persisted_experts']['graph_counters']['large_row_encoded_hit_dispatches'] = 864
        self.assertTrue(quality.coverage(before, broken, case)[1])
        with mock.patch.object(quality._base.phase_quality, 'phase_present', return_value=True), \
                mock.patch.object(quality._base.phase_quality, 'phase_coverage', return_value=({'target_phase_graph_counter_deltas': {'verify': {'calls': 1}}}, ['base phase failure'])):
            details, errors = quality.coverage(before, after, case)
            self.assertIn('base phase failure', errors)
            self.assertEqual(details['compact_expected_stage_calls'], 144)

    def test_legacy_no_compact_and_execution_policy_unchanged(self):
        before = original_tests.SemanticQualityTests().status(24576, 0, 0, 0, 0)
        after = original_tests.SemanticQualityTests().status(24576, 48 * 4096, 96, 32, 1)
        case = {'prompt_token_count': 4096}
        self.assertEqual(quality.coverage(before, after, case), original.coverage(before, after, case))
        self.assertEqual(quality.compact_status_errors(before), [])
        self.assertEqual(quality.ownership_policy(before), original.ownership_policy(before))
        active, disabled = self.status(), self.status(False)
        self.assertEqual(quality.execution_policy(active), quality.execution_policy(disabled))
        self.assertNotEqual(quality.ownership_policy(active), quality.ownership_policy(disabled))

    def test_extension_hooks_only_isolated_gates_and_ownership(self):
        spec = importlib.util.spec_from_file_location('_compact_adapter_extension_test', Path(quality.__file__))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        runner = module.load(self.build)
        old_specs, old_policy = runner.specs(), runner.execution_policy(self.status())
        module.install_hooks(status=lambda *args, **kwargs: ['extension status'], coverage=lambda *args, **kwargs: ({'extension': True}, ['extension coverage']),
            ownership=lambda status: {'extension.owner': True})
        self.assertEqual(runner.specs(), old_specs)
        self.assertEqual(runner.execution_policy(self.status()), old_policy)
        self.assertTrue(runner.ownership_policy(self.status())['extension.owner'])
        before, after = self.mixed()
        details, errors = runner.coverage(before, after, {'prompt_token_count': 4096, 'body': {}})
        self.assertTrue(details['extension']); self.assertIn('extension coverage', errors)
        details, errors = runner.coverage(before=before, after=after, case={'prompt_token_count': 4096, 'body': {}},
            require_counters=True, execution_mode='mtp3')
        self.assertTrue(details['extension']); self.assertIn('extension coverage', errors)
        self.assertTrue(runner.ownership_policy(status=self.status())['extension.owner'])
        self.assertIsNot(runner.gate_status, original.gate_status)

    def test_extension_status_hook_preserves_original_keyword_signature(self):
        spec = importlib.util.spec_from_file_location('_compact_adapter_status_keyword_test', Path(quality.__file__))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        runner = module.load(self.build)
        with mock.patch.object(runner, 'gate_status', return_value=['original status']) as original_status:
            module.install_hooks(status=lambda *args, **kwargs: ['extension status'])
            self.assertEqual(runner.gate_status(status={}, plan={}, store={}, execution_mode='mtp3'), ['original status', 'extension status'])
            original_status.assert_called_once_with(status={}, plan={}, store={}, execution_mode='mtp3')

    def test_cli_preserves_explicit_runtime_build_both_argparse_forms(self):
        for runtime in (['--runtime-build', '/explicit/baseline'], ['--runtime-build=/explicit/baseline']):
            with mock.patch.object(quality._base, 'main', side_effect=lambda argv: argv):
                self.assertEqual(quality.main(['--build', str(self.build), 'measure', *runtime]), ['measure', *runtime])
        with mock.patch.object(quality._base, 'main', side_effect=lambda argv: argv):
            self.assertEqual(quality.main(['--build', str(self.build), 'measure']), ['measure', '--runtime-build', str(self.build)])


if __name__ == '__main__':
    unittest.main()
