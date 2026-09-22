"""Five synthetic metadata tests. No actual reports, outputs or GPU reads."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks.raw_q5_verify_worker_sep22 import semantic_quality as q5
from dev.benchmarks.raw_q4q5_postrun_sep22 import postrun_compare as q


class PostrunTests(unittest.TestCase):
    def fixture(self):
        plan = {'content_sha256': 'a'*64, 'cases': [{'id': str(i), 'body': {}} for i in range(22)]}
        execution = {'fixed-request-policy': 'original22'}
        q4 = q5.load_q4_helper(q.REGISTRY['parent']['build'])
        q4_hooks = q4.make_hooks(http, True)
        q5_hooks = q5.make_hooks(http, True)

        def snapshot(name, cycles):
            hist = [0]*16; hist[3] = cycles
            status = {'identity': {'kernel_routes': q4.FAMILY + q4.SOURCE},
                'raw_q4_rowpair_verify': {'schema': q4.SCHEMA, 'requested': True,
                    'source_identity_sha256': q4.SOURCE, 'scope': q4.SCOPE,
                    'qualified_candidate_AIR_sha256': q4.AIR, 'graph_calls': 26*cycles,
                    'graph_rows': 104*cycles, 'expected_main_roles_per_VerifyR4': 26,
                    'GPU_allocation_bytes_added': 0, 'counter_scope': q4.COUNTER_SCOPE,
                    'whole_state_qualified': False},
                'compact_r4_preflight': {'guard_preflight_graph_calls': 48*cycles,
                    'guard_preflight_graph_rows': 192*cycles},
                'mtp': {'completed_cycles_by_proposed_depth': hist}, 'idle': True}
            if name == 'combined':
                status['identity']['kernel_routes'] += q5.FAMILY + q5.SOURCE
                status[q5.SECTION] = {'schema': q5.SCHEMA, 'requested': True,
                    'source_identity_sha256': q5.SOURCE, 'scope': q5.SCOPE,
                    'qualified_candidate_AIR_sha256': q5.AIR, 'graph_calls': 36*cycles,
                    'graph_rows': 144*cycles, 'expected_main_roles_per_VerifyR4': 36,
                    'GPU_allocation_bytes_added': 0, 'counter_scope': q5.COUNTER_SCOPE,
                    'whole_state_qualified': False}
            return status

        profiles, reports = {}, []
        for name in ('parent', 'combined'):
            hooks = [q4_hooks] + ([q5_hooks] if name == 'combined' else [])
            def status(value, plan, store, mode='mtp3', hooks=hooks):
                return sum((hook[0](value, plan, store, mode) for hook in hooks), [])
            def coverage(before, after, case, required=True, mode='mtp3', hooks=hooks):
                errors = sum((hook[1](before, after, case, required, mode)[1] for hook in hooks), [])
                return {}, errors
            runner = SimpleNamespace(gate_status=status, coverage=coverage,
                ownership_policy=lambda value: {'full_old_owner_gate': True}, execution_policy=lambda value: execution)
            profiles[name] = q.Profile(name, runner)
            reports.append({'schema': 'splash-prefill4k-semantic-report-v1', 'execution_mode': 'mtp3',
                'completed': True, 'full_plan_coverage': True, 'strict_cache_graph_coverage_required': True,
                'runtime_file_sha256': dict(q.REGISTRY[name]['runtime']),
                'plan_content_sha256': plan['content_sha256'], 'actual_execution_policy': execution,
                'initial_status': snapshot(name, 0), 'final_status': snapshot(name, 22), 'store_witness': {},
                'cases': [{'id': str(i), 'records': [{'synthetic_metadata_only': True}], 'status_before': snapshot(name, i), 'status_after': snapshot(name, i+1)} for i in range(22)]})
        fake_http = SimpleNamespace(get_path=http.get_path, idle=lambda value: value.get('idle') is True)
        return plan, reports, profiles, fake_http

    def dispatcher(self):
        plan, reports, profiles, client = self.fixture()
        return q.RecordedProfiles(profiles, client).bind_reports(reports, plan), plan, reports

    def test_two_specific_profiles_and_all46_contexts_keyword_api(self):
        dispatch, plan, reports = self.dispatcher()
        for name, report in zip(('parent', 'combined'), reports):
            self.assertEqual(dispatch.profile_for(report['initial_status']).name, name)
            self.assertEqual(dispatch.status(status=report['initial_status'], plan=plan, store={}), [])
            for row, case in zip(report['cases'], plan['cases']):
                self.assertEqual(dispatch.coverage(before=row['status_before'], after=row['status_after'], case=case)[1], [])
            self.assertTrue(dispatch.ownership(report['final_status'])['full_old_owner_gate'])

    def test_specific_runtime_parent_no_q5_and_unknown_mixed_contexts(self):
        for mutation in ('runtime', 'parentQ5', 'parentIdentityQ5', 'parentMarkerOnly', 'candidateMissingQ5', 'candidateMixedSource', 'parentFlag'):
            plan, reports, profiles, client = self.fixture()
            if mutation == 'runtime': reports[1]['runtime_file_sha256'] = reports[0]['runtime_file_sha256']
            if mutation == 'parentQ5': reports[0]['initial_status'][q5.SECTION] = dict(reports[1]['initial_status'][q5.SECTION])
            if mutation == 'parentIdentityQ5': reports[0]['initial_status']['identity'][q5.SECTION] = dict(reports[1]['initial_status'][q5.SECTION])
            if mutation == 'parentMarkerOnly':
                values=[reports[0]['initial_status'], reports[0]['final_status']]+[status for row in reports[0]['cases'] for status in (row['status_before'],row['status_after'])]
                for status in values:status['identity']['kernel_routes']+=q5.FAMILY+q5.SOURCE
            if mutation == 'candidateMissingQ5': del reports[1]['cases'][4]['status_before'][q5.SECTION]
            if mutation == 'candidateMixedSource': reports[1]['cases'][4]['status_before'][q5.SECTION]['source_identity_sha256'] = 'f'*64
            if mutation == 'parentFlag': reports[0]['cases'][2]['status_after']['raw_q4_rowpair_verify']['requested'] = False
            with self.assertRaises(ValueError): q.RecordedProfiles(profiles, client).bind_reports(reports, plan)
        dispatch, plan, reports = self.dispatcher()
        self.assertTrue(dispatch.coverage(reports[0]['initial_status'], reports[1]['final_status'], plan['cases'][0])[1])
        status = copy.deepcopy(reports[1]['initial_status']); status['unknown-extra'] = True
        self.assertTrue(dispatch.status(status, plan, {}))
        with self.assertRaises(ValueError): dispatch.ownership(status)

    def test_complete_original22_and_fixed_policy_counters_flags_are_mandatory(self):
        for mutation in ('partial', 'order', 'zeroRecords', 'twoRecords', 'completed', 'schema', 'mode', 'plan', 'policy', 'counter', 'marker', 'idle'):
            plan, reports, profiles, client = self.fixture(); report = reports[1]
            if mutation == 'partial': report['cases'].pop()
            if mutation == 'order': report['cases'].reverse()
            if mutation == 'zeroRecords': report['cases'][2]['records'] = []
            if mutation == 'twoRecords': report['cases'][2]['records'] *= 2
            if mutation == 'completed': report['completed'] = False
            if mutation == 'schema': report['schema'] = 'unknown'
            if mutation == 'mode': report['execution_mode'] = 'standard'
            if mutation == 'plan': report['plan_content_sha256'] = 'f'*64
            if mutation == 'policy': report['actual_execution_policy'] = {'different': True}
            if mutation == 'counter': report['cases'][5]['status_after'][q5.SECTION]['graph_calls'] += 36
            if mutation == 'marker': report['final_status']['identity']['kernel_routes'] = 'old-parent'
            if mutation == 'idle': report['cases'][1]['status_before']['idle'] = False
            with self.assertRaises(ValueError): q.RecordedProfiles(profiles, client).bind_reports(reports, plan)

    def test_external_report_shas_and_synthetic_metadata_only_file_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp)/str(i) for i in range(2)]
            for path in paths: path.write_text('{"synthetic_metadata_only":true}')
            pins = [q.file_digest(path) for path in paths]; q.check_report_pins(paths, pins)
            for wrong in ([pins[0]], ['f'*64, pins[1]], [True, pins[1]], ['invalid', pins[1]]):
                with self.assertRaises(ValueError): q.check_report_pins(paths, wrong)
            paths[0].write_text('{"changed_synthetic_metadata":true}')
            with self.assertRaises(ValueError): q.check_report_pins(paths, pins)

    def test_real_registered_metadata_load_preserves_original_functions_and_old_gates(self):
        profiles = q.load_profiles()
        parent, child = profiles['parent'].runner, profiles['combined'].runner
        self.assertIsNot(parent.compare.__globals__, child.compare.__globals__)
        self.assertEqual(parent.specs(), child.specs()); self.assertEqual(len(parent.specs()), 22)
        original_compare, original_grade, original_specs, original_execution = parent.compare, parent.grade_record, parent.specs, parent.execution_policy
        dispatch, plan, reports = self.dispatcher()
        dispatch.profiles['parent'].status = lambda *args, **kwargs: ['ALL-original22-Q4-HC-guard-compact-old-gates']
        self.assertIn('ALL-original22-Q4-HC-guard-compact-old-gates', dispatch.status(reports[0]['initial_status'], plan, {}))
        q.install_dispatch(parent, dispatch)
        self.assertIs(parent.compare, original_compare); self.assertIs(parent.grade_record, original_grade)
        self.assertIs(parent.specs, original_specs); self.assertIs(parent.execution_policy, original_execution)


if __name__ == '__main__': unittest.main()
