"""CPU-only synthetic metadata tests; no GPU/model/capture/state reports."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks import prefill4k_attribution_quality as original
from dev.benchmarks.raw_q5_verify_worker_sep22 import semantic_quality as q
from dev.benchmarks.guard_hc_fast_composite_sep22.semantic_quality import install as frozen_parent_install
from dev.benchmarks.guard_hc_fast_composite_sep22 import semantic_quality as hc
from dev.benchmarks.guard_hc_fast_composite_sep22 import test_semantic_quality as hc_tests


class CombinedQ5Tests(unittest.TestCase):
    source = 'a' * 64

    def setUp(self):
        self.hooks = q.make_hooks(http, True, self.source)
        self.status_hook, self.coverage_hook, self.ownership_hook = self.hooks

    def status(self, active=True, cycles=0):
        histogram = [0] * 16
        histogram[3] = cycles
        q4, bundle, calls = 26 * cycles, 48 * cycles, 36 * cycles if active else 0
        return {
            'identity': {'kernel_routes': 'unchanged-Q4-composite-original22' + (q.FAMILY + self.source if active else '')},
            q.SECTION: {
                'schema': q.SCHEMA, 'requested': active, 'source_identity_sha256': self.source,
                'scope': q.SCOPE, 'qualified_candidate_AIR_sha256': q.AIR,
                'graph_calls': calls, 'graph_rows': 4 * calls,
                'expected_main_roles_per_VerifyR4': 36, 'GPU_allocation_bytes_added': 0,
                'counter_scope': q.COUNTER_SCOPE, 'whole_state_qualified': False},
            'raw_q4_rowpair_verify': {'requested': True, 'source_identity_sha256': q.PARENT_SOURCE,
                'graph_calls': q4, 'graph_rows': 4 * q4},
            'compact_r4_preflight': {'guard_preflight_graph_calls': bundle, 'guard_preflight_graph_rows': 4 * bundle},
            'mtp': {'completed_cycles_by_proposed_depth': histogram}}

    def test_strict_active_and_disabled_profiles(self):
        for active in (False, True):
            status, coverage, _ = q.make_hooks(http, active, self.source)
            self.assertEqual(status(self.status(active, 3), None, None), [])
            self.assertEqual(coverage(self.status(active), self.status(active, 5), {'body': {}})[1], [])
        for value in (None, False, {}, {'identity': {'kernel_routes': ''}}, {q.SECTION: False}):
            self.assertTrue(self.status_hook(value, None, None))
        for value in (None, 0, 1, '1', 0.):
            with self.assertRaises(ValueError):
                q.make_hooks(http, value, self.source)
        for source in (None, '', 'A' * 64, 'f' * 63, False):
            with mock.patch.object(q, 'SOURCE', None), self.assertRaises(ValueError):
                q.make_hooks(http, True, source)

    def test_all_static_fields_flags_and_marker_fail_closed(self):
        good = self.status()
        for field in (path.split('.', 1)[1] for path in q.OWNERSHIP):
            for wrong in (None, False, True, 'unknown'):
                if q.same(good[q.SECTION][field], wrong):
                    continue
                value = copy.deepcopy(good)
                value[q.SECTION][field] = wrong
                self.assertTrue(self.status_hook(value, None, None), (field, wrong))
            value = copy.deepcopy(good)
            del value[q.SECTION][field]
            self.assertTrue(self.status_hook(value, None, None), field)
        for marker in ('base', q.FAMILY + 'f' * 64, q.FAMILY + self.source + 'suffix', q.FAMILY + self.source + q.FAMILY + self.source, False):
            value = copy.deepcopy(good)
            value['identity']['kernel_routes'] = marker
            self.assertTrue(self.status_hook(value, None, None))
        value = copy.deepcopy(good)
        value['identity'][q.SECTION] = dict(value[q.SECTION])
        self.assertTrue(self.status_hook(value, None, None))

    def test_all_counters_strict_u64_and_parent_dependencies(self):
        for field in q.COUNTERS:
            for wrong in (None, True, False, 0., -1, 2 ** 64):
                value = self.status()
                value[q.SECTION][field] = wrong
                self.assertTrue(self.status_hook(value, None, None), (field, wrong))
        for prefix, fields in (
                ('raw_q4_rowpair_verify', ('requested', 'source_identity_sha256', 'graph_calls', 'graph_rows')),
                ('compact_r4_preflight', ('guard_preflight_graph_calls', 'guard_preflight_graph_rows'))):
            for field in fields:
                value = self.status(cycles=5)
                del value[prefix][field]
                self.assertTrue(self.status_hook(value, None, None), (prefix, field))
            value = self.status(cycles=5)
            del value[prefix]
            self.assertTrue(self.status_hook(value, None, None), prefix)
        value = self.status()
        value['raw_q4_rowpair_verify']['requested'] = False
        self.assertTrue(self.status_hook(value, None, None))
        disabled, _, _ = q.make_hooks(http, False, self.source)
        value = self.status(False)
        value[q.SECTION]['graph_calls'], value[q.SECTION]['graph_rows'] = 1, 4
        self.assertTrue(disabled(value, None, None))

    def test_actual_depth3_not_total_or_proposed_depth(self):
        before, after = self.status(), self.status(cycles=5)
        after['mtp']['completed_cycles_by_proposed_depth'][1] = 9
        after['mtp']['verification_cycles'] = 14
        details, errors = self.coverage_hook(before=before, after=after, case={'body': {}}, require_counters=False, execution_mode='mtp3')
        self.assertEqual(errors, [])
        self.assertEqual(details['raw_Q5_expected_calls'], 180)
        self.assertEqual(details['raw_Q5_graph_counter_deltas'], {'graph_calls': 180, 'graph_rows': 720})
        for field in q.COUNTERS:
            for delta in (-1, 1):
                changed = copy.deepcopy(after)
                changed[q.SECTION][field] += delta
                self.assertTrue(self.coverage_hook(before, changed, {'body': {}})[1], field)
        self.assertTrue(self.coverage_hook(self.status(cycles=2), self.status(cycles=1), {'body': {}})[1])

    def test_both_snapshots_and_provenance_checked(self):
        for side in (0, 1):
            for section in (q.SECTION, 'raw_q4_rowpair_verify', 'compact_r4_preflight'):
                values = [self.status(), self.status()]
                del values[side][section]
                self.assertTrue(self.coverage_hook(*values, {'body': {}})[1], (side, section))
            for field in (path.split('.', 1)[1] for path in q.OWNERSHIP):
                values = [self.status(), self.status()]
                values[side][q.SECTION][field] = 'changed'
                self.assertTrue(self.coverage_hook(*values, {'body': {}})[1], (side, field))
        for bad in (None, {}, False, {'identity': {'kernel_routes': 'old-Q4-only'}}):
            self.assertTrue(self.coverage_hook(bad, bad, {'body': {}})[1])

    def test_histogram_malformed_decreasing_and_excluded_callers(self):
        for bad in (None, [0] * 15, [0] * 17, [False] * 16, [0.] * 16, [-1] * 16, [2 ** 64] * 16):
            for side in (0, 1):
                values = [self.status(), self.status()]
                values[side]['mtp']['completed_cycles_by_proposed_depth'] = bad
                self.assertTrue(self.coverage_hook(*values, {'body': {}})[1])
        cases = [{'body': {}, 'compact_scope': scope} for scope in ('batch', 'prefill-only', 'autoregressive', 'trained-head')]
        cases += [{'body': {'response_format': {'type': 'json_schema'}}}]
        for case in cases:
            self.assertEqual(self.coverage_hook(self.status(), self.status(), case)[1], [])
            self.assertTrue(self.coverage_hook(self.status(), self.status(cycles=1), case)[1])
        for mode in ('ar', 'mtp1', 'batch'):
            self.assertEqual(self.coverage_hook(self.status(), self.status(), {'body': {}}, execution_mode=mode)[1], [])
            self.assertTrue(self.coverage_hook(self.status(), self.status(cycles=1), {'body': {}}, execution_mode=mode)[1])
        for case in (None, [], {'body': False}, {'body': {'response_format': False}}):
            self.assertTrue(self.coverage_hook(self.status(), self.status(), case)[1])

    def runner(self):
        class Runner:
            http = http
            specs = staticmethod(original.specs)
            execution_policy = staticmethod(original.execution_policy)
            grade = object()

            def gate_status(self, *args, **kwargs): return ['old Q4/guard/HC failure']
            def coverage(self, *args, **kwargs): return {'old.guard': True}, ['old coverage failure']
            def ownership_policy(self, *args, **kwargs): return {'old.owner': True}
            def main(self, argv): return argv
        return Runner()

    def test_install_preserves_old_gates_specs_graders_execution_and_kwargs(self):
        runner = self.runner()
        original_grade = runner.grade
        runner = q.install(runner, self.hooks)
        errors = runner.gate_status(status=self.status(), plan=None, store=None, execution_mode='mtp3')
        self.assertEqual(errors, ['old Q4/guard/HC failure'])
        details, errors = runner.coverage(before=self.status(), after=self.status(cycles=5), case={'body': {}}, require_counters=True, execution_mode='mtp3')
        self.assertTrue(details['old.guard'])
        self.assertIn('old coverage failure', errors)
        self.assertTrue(runner.ownership_policy(status=self.status())['old.owner'])
        self.assertEqual(runner.specs(), original.specs())
        self.assertEqual(len(runner.specs()), 22)
        self.assertIs(runner.grade, original_grade)
        self.assertEqual(runner.execution_policy({}), original.execution_policy({}))

    def build(self):
        if q.ROOT.name == 'source' and (q.ROOT.parent / 'overlay-manifest.json').exists():
            return q.ROOT.parent
        return q.ROOT / 'build/rawQ4Q5-GDN-VerifyR4-composite-sep22-worker-v1'

    def test_real_sealed_q4_hook_outer_keyword_and_positional_verdicts_match(self):
        _, origin = q.authenticate(self.build())
        parent = q.load_q4_helper(origin)
        value = self.status()
        value['raw_q4_rowpair_verify'].update({
            'schema': parent.SCHEMA, 'scope': parent.SCOPE,
            'qualified_candidate_AIR_sha256': parent.AIR,
            'expected_main_roles_per_VerifyR4': 26, 'GPU_allocation_bytes_added': 0,
            'counter_scope': parent.COUNTER_SCOPE, 'whole_state_qualified': False})
        value['identity']['kernel_routes'] = parent.FAMILY + parent.SOURCE + ';base' + q.FAMILY + self.source
        runner = frozen_parent_install(self.runner(), parent.make_hooks(http, True))
        runner = q.install(runner, self.hooks)
        positional = runner.gate_status(value, None, None, execution_mode='mtp3')
        keyword = runner.gate_status(status=value, plan=None, store=None, execution_mode='mtp3')
        self.assertEqual(positional, ['old Q4/guard/HC failure'])
        self.assertEqual(positional, keyword)
        for args, kwargs in (((), {}), ((value,), {}), ((value, None, None), {'status': value}),
                ((value, None, None), {'unknown': True})):
            with self.assertRaises(TypeError): runner.gate_status(*args, **kwargs)

    def test_status_binding_preserves_execution_mode_keyword(self):
        runner = self.runner()
        runner.gate_status = mock.Mock(return_value=['old failure'])
        old = runner.gate_status
        runner = q.install(runner, self.hooks)
        value = self.status()
        runner.gate_status(status=value, plan={'plan': True}, store={'store': True}, execution_mode='standard')
        old.assert_called_once_with(value, {'plan': True}, {'store': True}, execution_mode='standard')

    def test_combined_parent_active_flags_and_all_parent_counter_gates_retained(self):
        _, origin = q.authenticate(self.build())
        parent = q.load_q4_helper(origin)
        def snapshot(cycles=0):
            value = hc_tests.CompositeTests().status(True, cycles)
            child = self.status(cycles=cycles)
            value[q.SECTION] = child[q.SECTION]
            value['raw_q4_rowpair_verify'] = child['raw_q4_rowpair_verify']
            value['raw_q4_rowpair_verify'].update({
                'schema': parent.SCHEMA, 'scope': parent.SCOPE,
                'qualified_candidate_AIR_sha256': parent.AIR,
                'expected_main_roles_per_VerifyR4': 26, 'GPU_allocation_bytes_added': 0,
                'counter_scope': parent.COUNTER_SCOPE, 'whole_state_qualified': False})
            value['identity']['kernel_routes'] += parent.FAMILY + parent.SOURCE + q.FAMILY + self.source
            return value
        runner = frozen_parent_install(self.runner(), hc.make_hooks(http, hc_tests.CompositeTests.source, require_active=True))
        runner = frozen_parent_install(runner, parent.make_hooks(http, True))
        runner = q.install(runner, self.hooks)
        baseline = runner.gate_status(snapshot(), None, None)
        self.assertEqual(baseline, ['old Q4/guard/HC failure'])
        paths = [('raw_q4_rowpair_verify', 'requested'), (q.SECTION, 'requested'),
            ('compact_r4_preflight', 'requested'), ('compact_r4_preflight', 'enabled'),
            ('compact_native_r4_verify', 'requested'), ('compact_native_r4_verify', 'enabled'),
            (hc.SECTION, 'requested'), ('identity', 'hc_pad_verify_r4_enabled')]
        paths += [('raw_q4_rowpair_verify', field) for field in q.COUNTERS]
        paths += [('compact_r4_preflight', field) for field in ('guard_preflight_graph_calls', 'guard_preflight_graph_rows')]
        paths += [('compact_native_r4_verify', field) for field in ('plan_graph_calls', 'plan_graph_rows')]
        paths += [('hc_pad_verify_r4_route_counters', field) for field in hc.COUNTERS]
        for section, field in paths:
            value = snapshot(cycles=5)
            value[section][field] = False
            if section != 'compact_native_r4_verify' or field in ('requested', 'enabled'):
                self.assertGreater(len(runner.gate_status(status=value, plan=None, store=None)), len(baseline), (section, field))
            for side in (0, 1):
                values = [snapshot(), snapshot()]
                values[side][section][field] = False
                errors = runner.coverage(before=values[0], after=values[1], case={'body': {}})[1]
                self.assertGreater(len(errors), 1, (side, section, field))

    def proof(self):
        return {
            'schema': q.PROOF_SCHEMA, 'pass': True, 'qualification_complete': True, 'Root_GPU_executed': True,
            'source_identity_sha256': self.source, 'exe_sha256': 'b' * 64, 'newlib_sha256': 'c' * 64,
            'qualified_candidate_AIR_sha256': q.AIR, 'rawQ4_qualified_candidate_AIR_sha256': q.Q4_AIR,
            'frames': 26, 'repeated_frames': 54, 'backend_destroyed': True,
            'rawQ4_calls': 130, 'rawQ4_rows': 520, 'rawQ5_calls': 180, 'rawQ5_rows': 720,
            'guardCalls': 240, 'guardRows': 960, 'HCcalls': 485, 'HCrows': 1940, 'pads': 485,
            'actual_owned_buffer_guard_cases': 8, 'allocation_guard_violations': 0, 'allocation_guard_axes_passed': 6,
            'ORIGINAL22_or_normal_performance_qualified': False,
            'root_report_path': '/Root/unopened-state-report.json', 'root_report_sha256': 'd' * 64,
            'oracle_sha256': 'e' * 64}

    def registry(self):
        return mock.patch.multiple(q, SOURCE=self.source, EXE='b' * 64, LIB='c' * 64, Q4_HELPER_SHA256='f' * 64)

    def test_combined_receipt_missing_old_proof_unknown_profile_and_all_fields(self):
        with tempfile.TemporaryDirectory() as tmp, self.registry():
            build = Path(tmp)
            path = build / q.PROOF_FILE
            with self.assertRaises(FileNotFoundError): q.require_native_state(build)
            (build / 'Root-rawQ4-native-qualified.json').write_text(json.dumps(self.proof()))
            with self.assertRaises(FileNotFoundError): q.require_native_state(build)
            proof = self.proof()
            path.write_text(json.dumps(proof))
            q.require_native_state(build)
            for key in proof:
                changed = copy.deepcopy(proof)
                del changed[key]
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError, msg=key): q.require_native_state(build)
            for key, value in proof.items():
                if isinstance(value, dict): continue
                wrong = '' if key == 'root_report_path' else False if value is True else True if type(value) is int else 'unknown'
                changed = copy.deepcopy(proof)
                changed[key] = wrong
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError, msg=key): q.require_native_state(build)
            for key in ('allocation_guard_violations', 'allocation_guard_axes_passed'):
                for wrong in (False, True, None, proof[key] - 1, proof[key] + 1):
                    changed = copy.deepcopy(proof)
                    changed[key] = wrong
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError, msg=key): q.require_native_state(build)
            for malformed in ('{', 'null', '[]', 'false'):
                path.write_text(malformed)
                with self.assertRaises(ValueError): q.require_native_state(build)
        with mock.patch.object(q, 'SOURCE', None), self.assertRaises(ValueError):
            q.authenticate('/unregistered/profile')

    def test_load_chains_strict_q4_and_child_state_admission(self):
        build, origin = Path('/combined'), Path('/immutable-Q4-v2')
        parent = mock.Mock()
        parent.load.return_value = self.runner()
        with self.registry(), mock.patch.object(q, 'authenticate', return_value=(build, origin)) as auth, mock.patch.object(q, 'load_q4_helper', return_value=parent):
            for active in (False, True):
                for state in (False, True):
                    runner = q.load(build, expected=active, require_state=state)
                    auth.assert_called_with(build, require_state=state)
                    parent.load.assert_called_with(origin, expected=True, require_state=True)
                    self.assertEqual(runner.specs(), original.specs())
        with mock.patch.object(q, 'authenticate') as auth, self.assertRaises(ValueError):
            q.load(build, expected=1)
        auth.assert_not_called()

    def test_real_source_only_load_preserves_original22_and_newproof_required(self):
        build = self.build()
        with mock.patch.object(q, 'require_native_state', side_effect=AssertionError('Do not read pending child proof')):
            runner = q.load(build, expected=False, require_state=False)
        self.assertEqual(runner.specs(), original.specs())
        self.assertEqual(runner.execution_policy({}), original.execution_policy({}))
        self.assertEqual(inspect.getsource(runner.grade_record), inspect.getsource(original.grade_record))
        _, origin = q.authenticate(build)
        with self.assertRaises(ValueError): q.authenticate(origin)
        with mock.patch.object(q, 'require_native_state', side_effect=ValueError('Fresh combined proof missing')) as proof:
            with self.assertRaises(ValueError): q.load(build, expected=True, require_state=True)
        proof.assert_called_once_with(build.resolve())

    def test_real_registry_malformed_profiles_source_artifact_and_parent_drift(self):
        build, origin = q.authenticate(self.build())
        manifest_path = build / 'overlay-manifest.json'
        manifest = q.document(manifest_path)
        original_document, original_digest = q.document, q.digest
        for key in ('schema', 'source_identity_sha256', 'parent_source_identity',
                'qualified_candidate_AIR_sha256', 'eligible_main_GDN_output_roles',
                'eligible_parent_main_GDN_QKV_roles', 'changed_paths', 'identity_parts', 'files'):
            changed = copy.deepcopy(manifest)
            changed[key] = None
            def read(path):
                return changed if Path(path) == manifest_path else original_document(path)
            with mock.patch.object(q, 'document', side_effect=read), self.assertRaises(ValueError, msg=key):
                q.authenticate(build)
        for part in manifest['identity_parts']:
            changed = copy.deepcopy(manifest)
            changed['identity_parts'][part] = 'unknown'
            with mock.patch.object(q, 'document', side_effect=read), self.assertRaises(ValueError, msg=part):
                q.authenticate(build)
        for record in ({'path': '../outside', 'sha256': 'f' * 64},
                {'path': '/absolute', 'sha256': 'f' * 64}, manifest['files'][0], {'path': 'invalid', 'sha256': False}):
            changed = copy.deepcopy(manifest)
            changed['files'].append(record)
            with mock.patch.object(q, 'document', side_effect=read), self.assertRaises(ValueError):
                q.authenticate(build)
        paths = [build / 'splash-flash', build / 'splash.metallib', build / 'rawQ5-qualified.air',
            origin / 'compiled-cpu-seal.json', origin / 'Root-rawQ4-native-qualified.json',
            origin / 'splash-flash', origin / 'splash.metallib']
        paths += [build / 'source' / q.PRIVATE / name for name in q.PROGRAMS]
        for changed_path in paths:
            def altered(path):
                return 'f' * 64 if Path(path) == changed_path else original_digest(path)
            with mock.patch.object(q, 'digest', side_effect=altered), self.assertRaises(ValueError, msg=str(changed_path)):
                q.authenticate(build)
        for changed_path in (origin / 'source' / q.Q4_HELPER_PRIVATE, origin / 'rawQ4-semantic-source-seal.json'):
            with mock.patch.object(q, 'digest', side_effect=altered), self.assertRaises(ValueError):
                q.load_q4_helper(origin)

    def test_cli_runtime_both_forms_all_occurrences_and_abbreviations(self):
        build = Path('/private/combined')
        for expected in ('0', '1'):
            with mock.patch.object(q, 'load', return_value=self.runner()) as loader:
                for runtime in (['--runtime-build', str(build)], ['--runtime-build=' + str(build)],
                        ['--runtime-build', str(build), '--runtime-build=' + str(build)]):
                    self.assertEqual(q.main(['--build', str(build), '--expected-rowpair', expected, 'measure', *runtime]), ['measure', *runtime])
                    loader.assert_called_with(build, expected=expected == '1', require_state=True)
                self.assertEqual(q.main(['--build', str(build), '--expected-rowpair', expected, 'measure']), ['measure', '--runtime-build', str(build)])
                for runtime in (['--runtime-build', '/old/Q4'], ['--runtime-build=/old/Q4'],
                        ['--runtime-build', str(build), '--runtime-build=/old/Q4'],
                        ['--runtime-build=/old/Q4', '--runtime-build', str(build)],
                        ['--runtime-b=' + str(build)], ['--runt', str(build)],
                        ['--runtime-build='], ['--runtime-build']):
                    loader.reset_mock()
                    with self.assertRaises(ValueError, msg=runtime):
                        q.main(['--build', str(build), '--expected-rowpair', expected, 'measure', *runtime])
                    loader.assert_not_called()
                q.main(['--build', str(build), '--expected-rowpair', expected, 'source-only'])
                self.assertFalse(loader.call_args.kwargs['require_state'])


if __name__ == '__main__':
    unittest.main()
