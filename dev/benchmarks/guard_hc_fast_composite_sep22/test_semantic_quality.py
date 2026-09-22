"""CPU metadata hooks, source admission and no-old-proof shortcut fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks import prefill4k_attribution_quality as original
from dev.benchmarks.guard_hc_fast_composite_sep22 import semantic_quality as q


class CompositeTests(unittest.TestCase):
    source = '38756086d4862cc9a87db70705928cb2a497450e1268bcb251bb0c0b10d8baba'
    def setUp(self): self.status_hook, self.coverage_hook, self.ownership_hook = q.make_hooks(http, self.source)
    def status(self, enabled=True, cycles=0):
        hist = [0] * 16; hist[3] = cycles
        calls = 48 * cycles; hc = 97 * cycles if enabled else 0
        return {'identity': {'kernel_routes': 'base-guard-compact' + (q.MARKER + self.source + q.HC_MARKER if enabled else ''),
            'hc_pad_verify_r4_enabled': enabled, 'hc_pad_verify_r4_policy': q.HC_POLICY, 'hc_pad_verify_r4_source_certificate': q.HC_CERTIFICATE},
            q.SECTION: {'schema': q.SCHEMA, 'requested': enabled, 'source_identity_sha256': self.source, 'scope': q.SCOPE,
                'HC_fast_leaf_manifest_sha256': q.HC_LEAF, 'HC_down_AIR_sha256': q.HC_AIR, 'new_GPU_allocation_bytes': 0, 'whole_composite_state_qualified': False},
            'compact_r4_preflight': {'source_identity_sha256': q.GUARD_ID, 'requested': True, 'enabled': True,
                'guard_preflight_graph_calls': calls, 'guard_preflight_graph_rows': 4 * calls},
            'compact_native_r4_verify': {'requested': True, 'enabled': True, 'plan_graph_calls': calls, 'plan_graph_rows': 4 * calls},
            'hc_pad_verify_r4_route_counters': {'scope': q.HC_SCOPE, 'graph_calls': hc, 'graph_rows': 4 * hc, 'padding_dispatches_saved': hc},
            'mtp': {'completed_cycles_by_proposed_depth': hist}}
    def test_active_inactive_strict_snapshot_status(self):
        for active in (False, True): self.assertEqual(self.status_hook(self.status(active, 3), None, None), [])
        for section in (q.SECTION, 'compact_r4_preflight', 'hc_pad_verify_r4_route_counters'):
            value = self.status(); del value[section]; self.assertTrue(self.status_hook(value, None, None))
    def test_flags_source_registry_marker_and_dependency(self):
        for key, value in (('requested', 1), ('requested', False), ('source_identity_sha256', 'f' * 64),
                ('HC_fast_leaf_manifest_sha256', 'f' * 64), ('HC_down_AIR_sha256', 'f' * 64), ('new_GPU_allocation_bytes', True), ('whole_composite_state_qualified', True)):
            s = self.status(); s[q.SECTION][key] = value; self.assertTrue(self.status_hook(s, None, None), key)
        for prefix in ('compact_r4_preflight', 'compact_native_r4_verify'):
            s = self.status(); s[prefix]['enabled'] = False; self.assertTrue(self.status_hook(s, None, None))
        s = self.status(); s['compact_r4_preflight']['source_identity_sha256'] = 'f' * 64; self.assertTrue(self.status_hook(s, None, None))
    def test_actual97_and48_h3_counts_mixed_depths(self):
        before, after = self.status(), self.status(True, 5)
        after['mtp']['completed_cycles_by_proposed_depth'][1] = 4
        after['mtp']['verification_cycles'] = 9
        details, errors = self.coverage_hook(before=before, after=after, case={'body': {}}, require_counters=True, execution_mode='mtp3')
        self.assertEqual(errors, []); self.assertEqual(details['HC_fast_expected_calls'], 485)
        for key in q.COUNTERS:
            for delta in (-1, 1):
                changed = copy.deepcopy(after); changed['hc_pad_verify_r4_route_counters'][key] += delta
                self.assertTrue(self.coverage_hook(before, changed, {'body': {}})[1], key)
        changed = copy.deepcopy(after); changed['compact_r4_preflight']['guard_preflight_graph_calls'] += 1
        self.assertTrue(self.coverage_hook(before, changed, {'body': {}})[1])
    def test_u64_histogram_monotonic_and_excluded_callers(self):
        for key in q.COUNTERS:
            for value in (None, True, False, 0., -1, 2 ** 64):
                s = self.status(); s['hc_pad_verify_r4_route_counters'][key] = value; self.assertTrue(self.status_hook(s, None, None))
        for scope in ('prefill-only', 'autoregressive', 'trained-head', 'batch'):
            self.assertEqual(self.coverage_hook(self.status(), self.status(), {'body': {}, 'compact_scope': scope})[1], [])
            self.assertTrue(self.coverage_hook(self.status(), self.status(True, 1), {'body': {}, 'compact_scope': scope})[1])
        self.assertTrue(self.coverage_hook(self.status(True, 2), self.status(True, 1), {'body': {}})[1])
    def test_both_missing_hc_context_and_unknown_bundle_fail_closed(self):
        a, b = self.status(), self.status()
        for value in (a, b): del value['identity']['hc_pad_verify_r4_enabled']
        self.assertTrue(self.coverage_hook(a, b, {'body': {}})[1])
        for value in (a, b): value['compact_r4_preflight'] = {'source_identity_sha256': 'unknown'}
        self.assertTrue(self.coverage_hook(a, b, {'body': {}})[1])
        legacy = {'identity': {'kernel_routes': 'original-b32-guard'}}
        self.assertEqual(self.status_hook(legacy, None, None), [])
        self.assertEqual(self.coverage_hook(legacy, legacy, {'body': {}}), ({}, []))
        self.assertEqual(self.ownership_hook(legacy), {})
        for partial in ({'identity': {'kernel_routes': q.FAMILY}},
                {'identity': {'hc_pad_verify_r4_enabled': False}},
                {'identity': {'hc_pad_verify_r4_enabled': None}},
                {'identity': {'kernel_routes': q.HC_FAMILY}}, {q.SECTION: None}):
            self.assertTrue(self.status_hook(partial, None, None))
            self.assertTrue(self.coverage_hook(legacy, partial, {'body': {}})[1])
            self.assertTrue(self.coverage_hook(partial, legacy, {'body': {}})[1])
    def test_install_preserves_previous_guard_gates_keywords_and_execution_policy(self):
        class Runner:
            def gate_status(self, *args, **kwargs): return ['original guard failure']
            def coverage(self, *args, **kwargs): return {'original.guard': True}, ['original guard failure']
            def ownership_policy(self, status): return {'original.guard': True}
            specs = staticmethod(original.specs)
            execution_policy = staticmethod(original.execution_policy)
        runner = q.install(Runner(), (self.status_hook, self.coverage_hook, self.ownership_hook))
        self.assertIn('original guard failure', runner.gate_status(status=self.status(), plan=None, store=None, execution_mode='mtp3'))
        details, errors = runner.coverage(before=self.status(), after=self.status(), case={'body': {}}, execution_mode='mtp3')
        self.assertTrue(details['original.guard']); self.assertIn('original guard failure', errors)
        self.assertTrue(runner.ownership_policy(status=self.status())['original.guard'])
        self.assertEqual(runner.specs(), original.specs()); self.assertEqual(runner.execution_policy(self.status()), original.execution_policy(self.status()))
    def build(self):
        if os.environ.get('GUARD_HC_FAST_COMPOSITE_BUILD'):
            return Path(os.environ['GUARD_HC_FAST_COMPOSITE_BUILD']).resolve()
        if q.ROOT.name == 'source' and (q.ROOT.parent / 'overlay-manifest.json').exists(): return q.ROOT.parent
        return q.ROOT / 'build/guard-HC-fast-composite-sep22-worker-v1'
    def test_real_prepared_authentication_and_source_only_load(self):
        build = self.build()
        _, _, identity, origin = q.authenticate(build)
        self.assertEqual(identity, self.source); self.assertEqual(origin.name, 'compact-r4-preflight-bundle-sep22-worker-v1')
        with mock.patch.object(q, 'require_native_state', side_effect=AssertionError('Source-only load must not read Root state receipt')):
            runner = q.load(build, require_state=False)
        self.assertEqual(runner.specs(), original.specs()); self.assertEqual(len(runner.specs()), 22)
        self.assertEqual(runner.execution_policy({}), original.execution_policy({}))
    def test_fresh_receipt_exact_newartifact_source_counts_and_missing_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp); (build / 'splash-flash').write_bytes(b'new composite executable')
            (build / 'splash.metallib').write_bytes(b'registered fast AIR closure')
            proof = {'schema': q.PROOF_SCHEMA, 'pass': True, 'qualification_complete': True,
                'source_identity_sha256': self.source, 'exe_sha256': hashlib.sha256((build / 'splash-flash').read_bytes()).hexdigest(),
                'newlib_sha256': hashlib.sha256((build / 'splash.metallib').read_bytes()).hexdigest(),
                'HC_down_AIR_sha256': q.HC_AIR, 'HC_fast_leaf_manifest_sha256': q.HC_LEAF,
                'frames': 26, 'repeated_frames': 54, 'backend_destroyed': True,
                'guardCalls': 240, 'guardRows': 960, 'HCcalls': 485, 'HCrows': 1940, 'pads': 485,
                'root_report_path': '/Root/unopened-full-state-report.json'}
            path = build / 'Root-native-qualified.json'
            with self.assertRaises(FileNotFoundError): q.require_native_state(build, self.source)
            path.write_text(json.dumps(proof)); q.require_native_state(build, self.source)
            for key in proof:
                changed = dict(proof); del changed[key]; path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError, msg=key): q.require_native_state(build, self.source)
            for key in ('source_identity_sha256', 'exe_sha256', 'newlib_sha256', 'HC_down_AIR_sha256', 'HC_fast_leaf_manifest_sha256'):
                changed = dict(proof); changed[key] = '0' * 64; path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError, msg=key): q.require_native_state(build, self.source)
            for key in ('frames', 'repeated_frames', 'guardCalls', 'guardRows', 'HCcalls', 'HCrows', 'pads'):
                for value in (True, proof[key] - 1, proof[key] + 1):
                    changed = dict(proof); changed[key] = value; path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError, msg=key): q.require_native_state(build, self.source)
    def test_source_registry_rejects_unregistered_bridge_overlay_and_schema(self):
        build = self.build(); path = build / 'overlay-manifest.json'; manifest = json.loads(path.read_text())
        original_read = Path.read_text
        for part in ('HC_bridge', 'HC_overlay', 'guard_source_identity', 'HC_fast_leaf_manifest', 'HC_fast_down_AIR', 'scope'):
            changed = copy.deepcopy(manifest); changed['identity_parts'][part] = 'unknown'
            def read(p, *args, **kwargs):
                return json.dumps(changed) if p == path else original_read(p, *args, **kwargs)
            with mock.patch.object(Path, 'read_text', read), self.assertRaises(ValueError, msg=part): q.authenticate(build)
        for key in ('schema', 'guard_source_identity', 'HC_fast_leaf_manifest_sha256', 'HC_down_AIR_sha256'):
            changed = copy.deepcopy(manifest); changed[key] = 'unknown'
            with mock.patch.object(Path, 'read_text', read), self.assertRaises(ValueError, msg=key): q.authenticate(build)
    def test_cli_measure_requires_newstate_runtime_build_bothforms(self):
        class Runner:
            def main(self, argv): return argv
        build = Path('/private/composite')
        with mock.patch.object(q, 'load', return_value=Runner()) as loader:
            for runtime in (['--runtime-build', str(build)], ['--runtime-build=' + str(build)],
                    ['--runtime-build', str(build), '--runtime-build=' + str(build), '--runtime-build', str(build / '../composite')]):
                self.assertEqual(q.main(['--build', str(build), 'measure', *runtime]), ['measure', *runtime])
                self.assertTrue(loader.call_args.kwargs['require_state'])
            self.assertEqual(q.main(['--build', str(build), 'measure']), ['measure', '--runtime-build', str(build)])
            for runtime in (['--runtime-build', '/old/guard'], ['--runtime-build=/old/guard'],
                    ['--runtime-build', str(build), '--runtime-build=/old/guard'],
                    ['--runtime-build=/old/guard', '--runtime-build', str(build)],
                    ['--runtime-build', str(build), '--runtime-b=/old/guard'],
                    ['--runtime-build', str(build), '--runt', '/old/guard'],
                    ['--runtime-build='], ['--runtime-build']):
                loader.reset_mock()
                with self.assertRaises(ValueError, msg=runtime): q.main(['--build', str(build), 'measure', *runtime])
                loader.assert_not_called()
    def test_strict_measured_hooks_reject_legacy_inactive_or_missing_context(self):
        status, coverage, _ = q.make_hooks(http, self.source, require_active=True)
        self.assertEqual(status(self.status(), None, None), [])
        self.assertEqual(coverage(self.status(), self.status(True, 5), {'body': {}})[1], [])
        legacy = {'identity': {'kernel_routes': 'original-b32-guard'}}
        self.assertTrue(status(legacy, None, None)); self.assertTrue(coverage(legacy, legacy, {'body': {}})[1])
        self.assertTrue(status(self.status(False), None, None))
        for before, after in ((legacy, self.status()), (self.status(), legacy), (self.status(False), self.status(False))):
            self.assertTrue(coverage(before=before, after=after, case={'body': {}})[1])
        for prefix in ('compact_r4_preflight', 'compact_native_r4_verify'):
            for flag in ('requested', 'enabled'):
                changed = self.status(); changed[prefix][flag] = False
                self.assertTrue(status(changed, None, None))
        for field in ('hc_pad_verify_r4_enabled', 'kernel_routes'):
            changed = self.status(); changed['identity'][field] = False if field.endswith('enabled') else 'original-b32-guard'
            self.assertTrue(status(changed, None, None))
    def test_actual_loader_freshproof_mode_installs_strict_status_and_coverage(self):
        with mock.patch.object(q, 'require_native_state') as proof:
            q.load(self.build(), require_state=True)
            proof.assert_called_once_with(self.build().resolve(), self.source)
        with mock.patch.object(q, 'make_hooks', wraps=q.make_hooks) as hooks, mock.patch.object(q, 'require_native_state'):
            q.load(self.build(), require_state=True)
            self.assertTrue(hooks.call_args.kwargs['require_active'])
    def test_main_validproof_stub_cannot_measure_oldguard_context(self):
        class Runner:
            http = http
            def gate_status(self, *args, **kwargs): return []
            def coverage(self, *args, **kwargs): return {}, []
            def ownership_policy(self, status): return {}
            def main(self, argv):
                snapshot = {'identity': {'kernel_routes': 'original-b32-guard'}}
                return self.gate_status(snapshot, None, None), self.coverage(snapshot, snapshot, {'body': {}})[1]
        def qualified_loader(build, require_state=False):
            self.assertTrue(require_state)  # Represents a valid fresh receipt.
            return q.install(Runner(), q.make_hooks(http, self.source, require_active=require_state))
        build = Path('/private/composite')
        with mock.patch.object(q, 'load', side_effect=qualified_loader):
            status_errors, coverage_errors = q.main(['--build', str(build), 'measure', '--runtime-build', str(build)])
        self.assertTrue(status_errors); self.assertTrue(coverage_errors)


if __name__ == '__main__': unittest.main()
