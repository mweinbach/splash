"""Pure metadata fixtures for new preflight hooks; no model/capture data."""
import copy
from pathlib import Path
import unittest
from unittest import mock
from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks import prefill4k_attribution_quality as original
from dev.benchmarks.expert_r4_preflight_bundle_sep22 import semantic_quality as quality


class PreflightSemanticTests(unittest.TestCase):
    source = quality.ORIGIN_SOURCE  # Synthetic fixture ID; never a generated header.

    def setUp(self):
        self.status_hook, self.coverage_hook, self.ownership_hook = quality.make_hooks(http, self.source)

    def status(self, enabled=True, calls=0, rows=None, hist=None, compact_calls=None):
        if rows is None: rows = 4 * calls
        if compact_calls is None: compact_calls = calls
        return {'identity': {'kernel_routes': 'original-compact-policy' + (quality.MARKER_PREFIX + self.source if enabled else '')},
            quality.SECTION: {'schema': quality.SCHEMA, 'scope': quality.SCOPE, 'requested': enabled, 'enabled': enabled,
                'source_identity_sha256': self.source, 'unique_logical_views': 13, 'immutable_spans': 96,
                'GPU_allocation_bytes_added': 0, 'shader_math_changed': False, 'original_public_native_API_guards_unchanged': True,
                'whole_state_qualified': False, 'guard_preflight_graph_calls': calls, 'guard_preflight_graph_rows': rows},
            'compact_native_r4_verify': {'enabled': True, 'requested': True, 'plan_graph_calls': compact_calls, 'plan_graph_rows': 4 * compact_calls},
            'mtp': {'completed_cycles_by_proposed_depth': [0] * 16 if hist is None else hist}}

    def test_status_enabled_and_disabled(self):
        for enabled in (False, True): self.assertEqual(self.status_hook(self.status(enabled), None, None), [])
        self.assertEqual(self.status_hook(self.status(True, 144), None, None), [])

    def test_exact_profile_flags_marker_dependency_and_source(self):
        for field, value in (('schema', 'other'), ('scope', 'other'), ('enabled', 1), ('requested', False),
                ('source_identity_sha256', 'f' * 64), ('unique_logical_views', 12), ('immutable_spans', 95),
                ('GPU_allocation_bytes_added', True), ('shader_math_changed', True),
                ('original_public_native_API_guards_unchanged', False), ('whole_state_qualified', True)):
            value_status = self.status(); value_status[quality.SECTION][field] = value
            self.assertTrue(self.status_hook(value_status, None, None), field)
        for field in ('schema', 'requested', 'source_identity_sha256'):
            value_status = self.status(); del value_status[quality.SECTION][field]
            self.assertTrue(self.status_hook(value_status, None, None), field)
        for enabled in (False, True):
            value_status = self.status(enabled); value_status['identity']['kernel_routes'] = quality.MARKER_PREFIX + 'f' * 64
            self.assertTrue(self.status_hook(value_status, None, None))
        value_status = self.status(); value_status['compact_native_r4_verify']['requested'] = False
        self.assertTrue(self.status_hook(value_status, None, None))

    def test_strict_u64_missing_bool_float_negative_overflow_and_monotonic(self):
        for field in quality.COUNTERS:
            for bad in (None, True, False, 0., -1, 2 ** 64):
                value_status = self.status(); value_status[quality.SECTION][field] = bad
                self.assertTrue(self.status_hook(value_status, None, None), (field, bad))
            value_status = self.status(); del value_status[quality.SECTION][field]
            self.assertTrue(self.status_hook(value_status, None, None))
        old, new = self.status(True, 2), self.status(True, 1)
        self.assertTrue(self.coverage_hook(old, new, {'body': {}})[1])

    def test_actual_h3_48_calls_and_same_parent_plan_deltas(self):
        hist = [0] * 16; hist[1], hist[2], hist[3] = 1, 1, 3
        before, after = self.status(), self.status(True, 144, hist=hist)
        details, errors = self.coverage_hook(before=before, after=after, case={'body': {}}, require_counters=True, execution_mode='mtp3')
        self.assertEqual(errors, []); self.assertEqual(details['guard_preflight_expected_calls'], 144)
        self.assertEqual(details['guard_preflight_expected_rows'], 576)
        for field in quality.COUNTERS:
            for delta in (-1, 1):
                changed = copy.deepcopy(after); changed[quality.SECTION][field] += delta
                self.assertTrue(self.coverage_hook(before, changed, {'body': {}})[1])
        changed = copy.deepcopy(after); changed['compact_native_r4_verify']['plan_graph_calls'] += 1
        self.assertTrue(self.coverage_hook(before, changed, {'body': {}})[1])

    def test_disabled_original_compact_cycles_allowed_new_counters_zero(self):
        hist = [0] * 16; hist[3] = 2
        before, after = self.status(False), self.status(False, hist=hist, compact_calls=96)
        self.assertEqual(self.coverage_hook(before, after, {'body': {}})[1], [])
        changed = self.status(False, 96, hist=hist)
        self.assertTrue(self.coverage_hook(before, changed, {'body': {}})[1])

    def test_missing_context_both_sides_and_excluded_context_fail_closed(self):
        a, b = self.status(), self.status(); del a[quality.SECTION]; del b[quality.SECTION]
        self.assertTrue(self.coverage_hook(a, b, {'body': {}})[1])
        # Original admitted compact statuses have no new profile AND no new
        # marker: legacy comparison retains the original behavior unchanged.
        a['identity']['kernel_routes'] = b['identity']['kernel_routes'] = 'original-compact-policy'
        self.assertEqual(self.status_hook(a, None, None), [])
        self.assertEqual(self.coverage_hook(a, b, {'body': {}}), ({}, []))
        self.assertEqual(self.ownership_hook(a), {})
        changed = self.status()
        self.assertTrue(self.coverage_hook(a, changed, {'body': {}})[1])
        for partial in (quality.MARKER_FAMILY, quality.MARKER_FAMILY + '-localStoreLayerGraphBundle'):
            first, second = copy.deepcopy(a), copy.deepcopy(b)
            first['identity']['kernel_routes'] = second['identity']['kernel_routes'] = partial
            self.assertTrue(self.status_hook(first, None, None))
            self.assertTrue(self.coverage_hook(first, second, {'body': {}})[1])
        hist = [0] * 16; hist[3] = 1
        for scope in ('prefill-only', 'autoregressive', 'trained-head', 'batch'):
            self.assertEqual(self.coverage_hook(self.status(), self.status(), {'body': {}, 'compact_scope': scope})[1], [])
            self.assertTrue(self.coverage_hook(self.status(), self.status(True, 48, hist=hist), {'body': {}, 'compact_scope': scope})[1])
        self.assertTrue(self.coverage_hook(self.status(), self.status(True, 48, hist=hist), {'body': {}}, execution_mode='standard')[1])
        self.assertTrue(self.coverage_hook(self.status(), self.status(True, 48, hist=hist), {'body': {'response_format': {'type': 'json_schema'}}})[1])

    def test_strict_histogram_and_cumulative_rows(self):
        for hist in (None, [0] * 15, [True] + [0] * 15, [-1] + [0] * 15, [0.] + [0] * 15, [2 ** 64] + [0] * 15):
            changed = self.status(); changed['mtp']['completed_cycles_by_proposed_depth'] = hist
            self.assertTrue(self.coverage_hook(self.status(), changed, {'body': {}})[1])
        self.assertTrue(self.status_hook(self.status(True, 1, rows=3), None, None))

    def test_ownership_only_original_execution_policy_and22_preserved(self):
        value = self.status()
        self.assertEqual(set(self.ownership_hook(status=value)), set(quality.OWNERSHIP_FIELDS))
        self.assertEqual(len(original.specs()), 22)
        spec_before = copy.deepcopy(original.specs()); policy_before = original.execution_policy(value)
        self.ownership_hook(value); self.status_hook(value, None, None)
        self.assertEqual(original.specs(), spec_before); self.assertEqual(original.execution_policy(value), policy_before)

    def test_cli_explicit_runtime_build_bothforms_and_default(self):
        class Runner:
            def main(self, argv): return argv
        build = Path('/private/prepared/worker')
        with mock.patch.object(quality, 'load', return_value=Runner()):
            for args in (['--runtime-build', '/explicit/original'], ['--runtime-build=/explicit/original']):
                self.assertEqual(quality.main(['--build', str(build), 'measure', *args]), ['measure', *args])
            self.assertEqual(quality.main(['--build', str(build), 'measure']), ['measure', '--runtime-build', str(build)])


if __name__ == '__main__': unittest.main()
