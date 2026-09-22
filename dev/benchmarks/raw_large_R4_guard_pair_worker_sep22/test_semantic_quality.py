"""CPU synthetic metadata and sealed-program tests, never tensor/Root reports."""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks import prefill4k_attribution_quality as original
from dev.benchmarks.guard_hc_fast_composite_sep22 import semantic_quality as hc
from dev.benchmarks.guard_hc_fast_composite_sep22 import test_semantic_quality as hc_tests
from dev.benchmarks.raw_large_R4_guard_pair_worker_sep22 import semantic_quality as q


class GuardPairTests(unittest.TestCase):
    source = 'a' * 64

    def setUp(self):
        self.hooks = q.make_hooks(http, True, self.source)
        self.gate, self.coverage, self.ownership = self.hooks

    def status(self, active=True, graphs=0, completed=None):
        hist = [0] * 16
        hist[3] = graphs if completed is None else completed
        calls = 87 * graphs if active else 0
        return {
            'identity': {'source': q.MODEL, 'loaded_model_layout_sha256': 'b' * 64,
                'target_numerical_derivative_sha256': 'c' * 64,
                'kernel_routes': 'unchanged-full-Q4-HC-guard-compact' + (q.FAMILY + self.source if active else '')},
            q.SECTION: {**q.STATIC, 'requested': active, 'source_identity_sha256': self.source,
                'authenticated_RAW_roles': 113 if active else 0, 'authenticated_new_roles': 87 if active else 0,
                'graph_calls': calls, 'graph_rows': 4 * calls, 'excluded_context_graph_calls': 0},
            'raw_q4_rowpair_verify': {'requested': True, 'source_identity_sha256': q.PARENT_SOURCE,
                'graph_calls': 26 * graphs, 'graph_rows': 104 * graphs},
            'compact_r4_preflight': {'guard_preflight_graph_calls': 48 * graphs, 'guard_preflight_graph_rows': 192 * graphs},
            'mtp': {'completed_cycles_by_proposed_depth': hist}}

    def build(self):
        if q.ROOT.name == 'source' and (q.ROOT.parent / 'overlay-manifest.json').exists():
            return q.ROOT.parent
        return q.ROOT / 'build/raw-large-R4-guard-pair-currentQ4-sep22-worker-v1'

    def test_enabled_disabled_missing_false_and_static_profile_fields(self):
        for active in (False, True):
            gate, coverage, _ = q.make_hooks(http, active, self.source)
            self.assertEqual(gate(self.status(active, 3), None, None), [])
            self.assertEqual(coverage(self.status(active), self.status(active, 3), {'body': {}})[1], [])
            good = self.status(active, 3)
            for field in (*q.STATIC, 'requested', 'source_identity_sha256', 'authenticated_RAW_roles', 'authenticated_new_roles'):
                value = copy.deepcopy(good)
                del value[q.SECTION][field]
                self.assertTrue(gate(value, None, None), field)
                for wrong in (None, True, False, -1, 1., 'unknown', 2 ** 64):
                    if q.same(good[q.SECTION][field], wrong): continue
                    value = copy.deepcopy(good)
                    value[q.SECTION][field] = wrong
                    self.assertTrue(gate(value, None, None), (active, field, wrong))
        for status in (None, False, {}, {q.SECTION: None}, {q.SECTION: False}):
            self.assertTrue(self.gate(status, None, None))
        for expected in (None, 0, 1, '1', 0.):
            with self.assertRaises(ValueError): q.make_hooks(http, expected, self.source)
        for source in ('', 'f' * 63, 'A' * 64, False):
            with self.assertRaises(ValueError): q.make_hooks(http, True, source)

    def test_numeric_source_relationship_and_markers_are_explicit(self):
        for routes in ('base', q.FAMILY + 'f' * 64, q.FAMILY + self.source + 'suffix',
                q.FAMILY + self.source + q.FAMILY + self.source, False):
            value = self.status()
            value['identity']['kernel_routes'] = routes
            self.assertTrue(self.gate(value, None, None))
        for field in ('source', 'loaded_model_layout_sha256', 'target_numerical_derivative_sha256'):
            for wrong in (None, False, 'unknown'):
                value = self.status()
                value['identity'][field] = wrong
                self.assertTrue(self.gate(value, None, None), field)
            a, b = self.status(), self.status()
            b['identity'][field] = 'f' * 64
            self.assertTrue(self.coverage(a, b, {'body': {}})[1], field)
        value = self.status()
        value['identity'][q.SECTION] = dict(value[q.SECTION])
        self.assertTrue(self.gate(value, None, None))
        details, errors = self.coverage(self.status(), self.status(graphs=1), {'body': {}})
        self.assertEqual(errors, [])
        relationship = details['raw_large_R4_identity_relationship']
        self.assertEqual(relationship['original_model_source_identity_sha256'], q.MODEL)
        self.assertEqual(relationship['new_kernel_route_source_identity_sha256'], self.source)
        self.assertIs(relationship['numerical_policy_changed'], False)

    def test_all_u64_counter_fields_and_87_26_48_census(self):
        for field in q.COUNTERS:
            for wrong in (None, True, False, 0., -1, 2 ** 64):
                value = self.status(graphs=2)
                value[q.SECTION][field] = wrong
                self.assertTrue(self.gate(value, None, None), (field, wrong))
        for section, fields in (('raw_q4_rowpair_verify', ('requested', 'source_identity_sha256', 'graph_calls', 'graph_rows')),
                ('compact_r4_preflight', ('guard_preflight_graph_calls', 'guard_preflight_graph_rows'))):
            for field in fields:
                value = self.status(graphs=3)
                del value[section][field]
                self.assertTrue(self.gate(value, None, None), (section, field))
        value = self.status(graphs=1)
        value[q.SECTION]['excluded_context_graph_calls'] = 1
        self.assertTrue(self.gate(value, None, None))
        gate, _, _ = q.make_hooks(http, False, self.source)
        value = self.status(False)
        value[q.SECTION]['graph_calls'], value[q.SECTION]['graph_rows'] = 1, 4
        self.assertTrue(gate(value, None, None))

    def test_actual_h3_completed_cycles_and_truthful_pending_graphs(self):
        before, after = self.status(graphs=2), self.status(graphs=5)
        after['mtp']['completed_cycles_by_proposed_depth'][1] = 11
        after['mtp']['completed_cycles_by_proposed_depth'][4] = 8
        after['mtp']['verification_cycles'] = 24
        details, errors = self.coverage(before=before, after=after, case={'body': {}}, require_counters=False, execution_mode='mtp3')
        self.assertEqual(errors, [])
        self.assertEqual(details['raw_large_R4_expected_calls'], 261)
        self.assertEqual(details['raw_large_R4_graph_counter_deltas']['graph_rows'], 1044)
        # Third provisional graph is constructed while successful H3 count is
        # still two. A snapshot is truthful; it is not terminal-cycle coverage.
        pending = self.status(graphs=3, completed=2)
        self.assertEqual(self.gate(pending, None, None), [])
        self.assertTrue(self.coverage(self.status(graphs=2), pending, {'body': {}})[1])
        self.assertTrue(self.coverage(after, before, {'body': {}})[1])
        for field in q.COUNTERS:
            changed = copy.deepcopy(after)
            changed[q.SECTION][field] += 1
            self.assertTrue(self.coverage(before, changed, {'body': {}})[1], field)

    def test_both_snapshots_identity_missing_mixed_and_histograms(self):
        for side in (0, 1):
            for section in (q.SECTION, 'raw_q4_rowpair_verify', 'compact_r4_preflight'):
                values = [self.status(), self.status()]
                del values[side][section]
                self.assertTrue(self.coverage(*values, {'body': {}})[1], (side, section))
            for field in (*q.STATIC, 'requested', 'source_identity_sha256', 'authenticated_RAW_roles', 'authenticated_new_roles'):
                values = [self.status(), self.status()]
                values[side][q.SECTION][field] = 'mixed'
                self.assertTrue(self.coverage(*values, {'body': {}})[1], (side, field))
            for hist in (None, [], [0] * 15, [0] * 17, [False] * 16, [0.] * 16, [-1] * 16, [2 ** 64] * 16):
                values = [self.status(), self.status()]
                values[side]['mtp']['completed_cycles_by_proposed_depth'] = hist
                self.assertTrue(self.coverage(*values, {'body': {}})[1])

    def test_excluded_prefill_ar_r5_head_batch_contexts_zero(self):
        for scope in ('prefill-only', 'autoregressive', 'physical-r5', 'trained-head', 'batch'):
            case = {'body': {}, 'compact_scope': scope}
            self.assertEqual(self.coverage(self.status(), self.status(), case)[1], [])
            self.assertTrue(self.coverage(self.status(), self.status(graphs=1), case)[1])
        for mode in ('standard', 'mtp1', 'batch'):
            self.assertEqual(self.coverage(self.status(), self.status(), {'body': {}}, execution_mode=mode)[1], [])
            self.assertTrue(self.coverage(self.status(), self.status(graphs=1), {'body': {}}, execution_mode=mode)[1])
        a, b = self.status(), self.status()
        b['mtp']['completed_cycles_by_proposed_depth'][4] = 9
        self.assertEqual(self.coverage(a, b, {'body': {}, 'compact_scope': 'physical-r5'})[1], [])
        for case in (None, [], {'body': False}, {'body': {'response_format': False}}):
            self.assertTrue(self.coverage(a, a, case)[1])
        self.assertTrue(self.coverage(a, self.status(graphs=1), {'body': {'response_format': {'type': 'json_schema'}}})[1])

    def runner(self):
        class Runner:
            http = http
            specs = staticmethod(original.specs)
            grade_record = staticmethod(original.grade_record)
            execution_policy = staticmethod(original.execution_policy)
            compare = object()
            def gate_status(self, *args, **kwargs): return ['old-original22-Q4-HC-guard-compact']
            def coverage(self, *args, **kwargs): return {'old_detail': True}, ['old_coverage']
            def ownership_policy(self, *args, **kwargs): return {'old_owner': True}
            def main(self, argv): return argv
        return Runner()

    def test_install_preserves_all_old_gates_bodies_graders_and_keyword_api(self):
        runner = self.runner()
        methods = [runner.specs, runner.grade_record, runner.execution_policy, runner.compare]
        old_status = mock.Mock(wraps=runner.gate_status)
        runner.gate_status = old_status
        runner = q.install(runner, self.hooks)
        value = self.status()
        self.assertEqual(runner.gate_status(status=value, plan=None, store=None, execution_mode='mtp3'), ['old-original22-Q4-HC-guard-compact'])
        old_status.assert_called_with(value, None, None, execution_mode='mtp3')
        self.assertEqual(runner.gate_status(value, None, None), ['old-original22-Q4-HC-guard-compact'])
        details, errors = runner.coverage(before=value, after=value, case={'body': {}}, execution_mode='mtp3')
        self.assertTrue(details['old_detail']); self.assertIn('old_coverage', errors)
        self.assertTrue(runner.ownership_policy(status=value)['old_owner'])
        self.assertEqual(methods, [runner.specs, runner.grade_record, runner.execution_policy, runner.compare])
        self.assertEqual(runner.specs(), original.specs()); self.assertEqual(len(runner.specs()), 22)
        for args, kwargs in (((), {}), ((value,), {}), ((value, None, None), {'status': value}), ((value, None, None), {'unknown': True})):
            with self.assertRaises(TypeError): runner.gate_status(*args, **kwargs)

    def test_real_source_only_admission_chains_registered_q4_and_never_inherits_state(self):
        build, origin, binding = q.authenticate(self.build())
        self.assertEqual(binding['source_identity_sha256'], q.SOURCE)
        self.assertEqual(binding['candidate_AIR_sha256'], q.AIR)
        with mock.patch.object(q, 'require_native_state', side_effect=AssertionError('Source-only admission must not open future proof')):
            runner = q.load(build, expected=False, require_state=False)
        self.assertEqual(runner.specs(), original.specs())
        self.assertEqual(inspect.getsource(runner.grade_record), inspect.getsource(original.grade_record))
        self.assertEqual(runner.execution_policy({}), original.execution_policy({}))
        with self.assertRaises(ValueError): q.authenticate(origin)
        with mock.patch.object(q, 'document', side_effect=AssertionError('Unregistered future proof must not be opened')):
            with self.assertRaises(ValueError): q.require_native_state(build, binding)
        with self.assertRaises(ValueError): q.load(build, expected=True, require_state=True)
        parent = mock.Mock(); parent.load.return_value = self.runner()
        with mock.patch.object(q, 'authenticate', return_value=(build, origin, binding)), mock.patch.object(q, 'load_parent', return_value=parent):
            q.load(build, expected=True, require_state=False)
        parent.load.assert_called_once_with(origin, expected=True, require_state=True)

    def test_copied_helper_under_optimized_no_bytecode_python(self):
        # A fresh source-depth copy reproduces sealed extras placement without
        # modifying v1 extras. Root may additionally select the actual v2 sealed
        # helper after copying it, and the same test checks its exact source hash.
        with tempfile.TemporaryDirectory() as temp:
            selected = os.environ.get('RAW_LARGE_R4_SEALED_HELPER')
            if selected:
                copied_path = Path(selected).resolve()
            else:
                copied_path = Path(temp) / 'worker/source' / q.PRIVATE / 'semantic_quality.py'
                copied_path.parent.mkdir(parents=True)
                shutil.copyfile(Path(q.__file__), copied_path)
            expected_sha = q.digest(Path(q.__file__))
            code = '''
import hashlib, importlib.util, json
from pathlib import Path
import sys
path = Path(sys.argv[1]).resolve()
expected_sha = sys.argv[2]
if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
    raise RuntimeError('Copied helper differs from this v2 source')
spec = importlib.util.spec_from_file_location('_sealed_guard_pair_v2', path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
if helper.ROOT != Path('/Users/mweinbach/Projects/splash'):
    raise RuntimeError('Copied helper does not retain the explicit known workspace')
build = helper.ROOT / 'build/raw-large-R4-guard-pair-currentQ4-sep22-worker-v1'
authenticated, origin, binding = helper.authenticate(build, require_state=False)
runner = helper.load(build, expected=False, require_state=False)
if binding['source_identity_sha256'] != helper.SOURCE or len(runner.specs()) != 22:
    raise RuntimeError('Copied helper failed the actual registered source/parent chain')
if origin != helper.ROOT / 'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2':
    raise RuntimeError('Copied helper selected an unsupported parent origin')
try:
    helper.load(build, expected=True, require_state=True)
except ValueError:
    pass
else:
    raise RuntimeError('Copied source-only helper admitted unregistered trained proof')
if sys.flags.optimize != 1 or not sys.dont_write_bytecode:
    raise RuntimeError('Regression must execute under Python -B -O')
print(json.dumps({'copied_source_only_parent_chain': True, 'trained_admission_blocked': True,
    'optimize': sys.flags.optimize, 'bytecode_writing': False}))
'''
            result = subprocess.run([sys.executable, '-B', '-O', '-c', code, str(copied_path), expected_sha],
                cwd=q.ROOT, check=True, capture_output=True, text=True)
            receipt = json.loads(result.stdout)
            self.assertIs(receipt['copied_source_only_parent_chain'], True)
            self.assertIs(receipt['trained_admission_blocked'], True)
            self.assertEqual(receipt['optimize'], 1)
            self.assertIs(receipt['bytecode_writing'], False)
            self.assertFalse((copied_path.parent / '__pycache__').exists())

    def test_real_frozen_parent_keyword_compatibility_and_parent_flags_counters(self):
        _, origin, _ = q.authenticate(self.build())
        parent = q.load_parent(origin)
        def snapshot(graphs=0):
            value = hc_tests.CompositeTests().status(True, graphs)
            child = self.status(graphs=graphs)
            value[q.SECTION] = child[q.SECTION]
            value['raw_q4_rowpair_verify'] = child['raw_q4_rowpair_verify']
            value['raw_q4_rowpair_verify'].update({'schema': parent.SCHEMA, 'scope': parent.SCOPE,
                'qualified_candidate_AIR_sha256': parent.AIR, 'expected_main_roles_per_VerifyR4': 26,
                'GPU_allocation_bytes_added': 0, 'counter_scope': parent.COUNTER_SCOPE, 'whole_state_qualified': False})
            value['identity'].update({key: item for key, item in child['identity'].items() if key != 'kernel_routes'})
            value['identity']['kernel_routes'] += parent.FAMILY + parent.SOURCE + q.FAMILY + self.source
            return value
        runner = hc.install(self.runner(), hc.make_hooks(http, hc_tests.CompositeTests.source, require_active=True))
        runner = hc.install(runner, parent.make_hooks(http, True))
        runner = q.install(runner, self.hooks)
        value = snapshot()
        self.assertEqual(runner.gate_status(value, None, None), runner.gate_status(status=value, plan=None, store=None))
        self.assertEqual(runner.gate_status(value, None, None), ['old-original22-Q4-HC-guard-compact'])
        paths = [('raw_q4_rowpair_verify', 'requested'), ('compact_r4_preflight', 'requested'), ('compact_r4_preflight', 'enabled'),
            ('compact_native_r4_verify', 'requested'), ('compact_native_r4_verify', 'enabled'), (hc.SECTION, 'requested'), ('identity', 'hc_pad_verify_r4_enabled')]
        paths += [('raw_q4_rowpair_verify', key) for key in ('graph_calls', 'graph_rows')]
        paths += [('compact_r4_preflight', key) for key in ('guard_preflight_graph_calls', 'guard_preflight_graph_rows')]
        paths += [('hc_pad_verify_r4_route_counters', key) for key in hc.COUNTERS]
        for section, key in paths:
            for side in (0, 1):
                values = [snapshot(), snapshot()]
                values[side][section][key] = False
                self.assertGreater(len(runner.coverage(before=values[0], after=values[1], case={'body': {}})[1]), 1, (side, section, key))

    def test_unknown_manifest_parts_registry_program_air_and_parent_drift(self):
        build, origin, _ = q.authenticate(self.build())
        path = build / 'overlay-manifest.json'
        manifest, real_document, real_digest = q.document(path), q.document, q.digest
        def read(p): return changed if Path(p) == path else real_document(p)
        for key in manifest['identity_parts']:
            changed = copy.deepcopy(manifest); changed['identity_parts'][key] = 'unknown'
            with mock.patch.object(q, 'document', side_effect=read), self.assertRaises(ValueError, msg=key): q.authenticate(build)
        for key in ('schema', 'pass', 'source_identity_sha256', 'files', 'artifacts', 'private_header_consumers', 'new_potential_main_roles', 'new_weight_cache_or_GPU_owner_bytes'):
            changed = copy.deepcopy(manifest); changed[key] = None
            with mock.patch.object(q, 'document', side_effect=read), self.assertRaises(ValueError, msg=key): q.authenticate(build)
        for record in ({'path': '../outside', 'sha256': 'f'*64}, {'path': '/absolute', 'sha256': 'f'*64}, manifest['files'][0]):
            changed = copy.deepcopy(manifest); changed['files'].append(record)
            with mock.patch.object(q, 'document', side_effect=read), self.assertRaises(ValueError): q.authenticate(build)
        paths = [build / 'splash-flash', build / 'splash.metallib', build / q.AIR_FILE,
            origin / 'splash-flash', origin / 'splash.metallib', origin / 'compiled-cpu-seal.json', origin / 'Root-rawQ4-native-qualified.json']
        paths += [build / 'source' / q.PRIVATE / name for name in q.PROGRAMS]
        for changed_path in paths:
            def changed_digest(p): return 'f'*64 if Path(p) == changed_path else real_digest(p)
            with mock.patch.object(q, 'digest', side_effect=changed_digest), self.assertRaises(ValueError, msg=str(changed_path)): q.authenticate(build)

    def test_future_proof_registration_is_explicit_metadata_not_an_old_receipt(self):
        binding = {'source_identity_sha256': self.source, 'exe_sha256': 'b'*64, 'lib_sha256': 'c'*64, 'candidate_AIR_sha256': 'd'*64}
        # Clearly synthetic protocol fields exercise registration/type handling;
        # they do not supply fabricated real trained-state counts or admission.
        required = {'synthetic_trained_boundary_metadata': True, 'synthetic_processes_destroyed': True,
            'synthetic_selection': {'request_id': 1, 'generation_id': 1, 'physical_rows': 4}}
        proof = {**required, 'schema': 'synthetic-third-R4-test-only', 'pass': True, 'qualification_complete': True,
            'Root_GPU_executed': True, 'source_identity_sha256': self.source, 'exe_sha256': 'b'*64, 'newlib_sha256': 'c'*64,
            'candidate_AIR_sha256': 'd'*64, 'root_report_path': '/synthetic/unopened-report.json', 'root_report_sha256': 'e'*64, 'oracle_sha256': 'f'*64}
        with tempfile.TemporaryDirectory() as temp:
            build = Path(temp); name = 'synthetic-trained-proof-test-only.json'; path = build / name
            (build / 'Root-rawQ4-native-qualified.json').write_text(json.dumps(proof))
            with mock.patch.multiple(q, SOURCE=self.source, EXE='b'*64, LIB='c'*64, AIR='d'*64,
                    PROOF_FILE=name, PROOF_SCHEMA=proof['schema'], PROOF_REQUIRED=required, PROOF_SHA256='0'*64):
                with self.assertRaises(FileNotFoundError): q.require_native_state(build, binding)
                path.write_text(json.dumps(proof))
                with self.assertRaises(ValueError): q.require_native_state(build, binding)
                with mock.patch.object(q, 'PROOF_SHA256', q.digest(path)): q.require_native_state(build, binding)
                for key in proof:
                    changed = dict(proof); del changed[key]; path.write_text(json.dumps(changed))
                    with mock.patch.object(q, 'PROOF_SHA256', q.digest(path)), self.assertRaises(ValueError, msg=key): q.require_native_state(build, binding)
                for key in ('schema', 'source_identity_sha256', 'exe_sha256', 'newlib_sha256', 'candidate_AIR_sha256', 'Root_GPU_executed', 'qualification_complete'):
                    changed = dict(proof); changed[key] = False; path.write_text(json.dumps(changed))
                    with mock.patch.object(q, 'PROOF_SHA256', q.digest(path)), self.assertRaises(ValueError, msg=key): q.require_native_state(build, binding)
                for key in binding:
                    changed = dict(binding); changed[key] = '0'*64
                    with self.assertRaises(ValueError, msg=key): q.require_native_state(build, changed)
                changed = copy.deepcopy(proof); changed['synthetic_selection']['request_id'] = True
                path.write_text(json.dumps(changed))
                with mock.patch.object(q, 'PROOF_SHA256', q.digest(path)), self.assertRaises(ValueError): q.require_native_state(build, binding)

    def test_cli_measured_requires_freshproof_and_runtime_bothforms_everyoccurrence(self):
        build = Path('/private/current-R4')
        for expected in ('0', '1'):
            with mock.patch.object(q, 'load', return_value=self.runner()) as loader:
                for runtime in (['--runtime-build', str(build)], ['--runtime-build='+str(build)], ['--runtime-build', str(build), '--runtime-build='+str(build)]):
                    result = q.main(['--build', str(build), '--expected-guard-pair', expected, 'measure', *runtime])
                    self.assertEqual(result, ['measure', *runtime]); loader.assert_called_with(build, expected=expected=='1', require_state=True)
                self.assertEqual(q.main(['--build', str(build), '--expected-guard-pair', expected, 'measure']), ['measure', '--runtime-build', str(build)])
                for runtime in (['--runtime-build'], ['--runtime-build='], ['--runtime-b', str(build)], ['--runt='+str(build)],
                        ['--runtime-build', '/old/R5'], ['--runtime-build=/old/ordinaryVerify'], ['--runtime-build', str(build), '--runtime-build=/old/Q4']):
                    loader.reset_mock()
                    with self.assertRaises(ValueError): q.main(['--build', str(build), '--expected-guard-pair', expected, 'measure', *runtime])
                    loader.assert_not_called()
                q.main(['--build', str(build), '--expected-guard-pair', expected, 'source-only'])
                self.assertFalse(loader.call_args.kwargs['require_state'])


if __name__ == '__main__':
    unittest.main()
