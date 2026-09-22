"""Pure CPU admission/counter tests; no model or actual response data."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks.immutable_interval_worker_sep22 import semantic_quality as q

class IntervalTests(unittest.TestCase):
    def snapshot(self, active, cycles=0, extra=0):
        hist = [0] * 16
        hist[3] = cycles
        work = 624 * cycles + extra
        return {q.SECTION: {
            "schema": q.SCHEMA, "requested": active, "source_policy_sha256": q.SOURCE,
            "scope": q.SCOPE, "GPU_allocation_bytes_added": 0,
            "FP_graph_public_guards_unchanged": True, "counter_scope": q.COUNTER_SCOPE,
            "finalized_tables": int(active), "finalized_spans": 96 * int(active),
            "indexable_tables": int(active), "indexed_accepts": work if active else 0,
            "original_callbacks": 0 if active else work},
            "mtp": {"completed_cycles_by_proposed_depth": hist}}
    def test_both_flags_static_types_and_finalized_actual_census(self):
        for active in (False, True):
            gate = q.index_hooks(http, active)[0]
            self.assertEqual(gate(self.snapshot(active, 3), None, None), [])
            for key in q.STATIC:
                for bad in (None, not active, 1.0, "wrong"):
                    s = self.snapshot(active, 3)
                    if q.same(s[q.SECTION][key], bad):
                        continue
                    s[q.SECTION][key] = bad
                    self.assertTrue(gate(s, None, None), (active, key, bad))
            for key in q.COUNTERS:
                for bad in (True, -1, 1.0, 2 ** 64):
                    s = self.snapshot(active, 3)
                    s[q.SECTION][key] = bad
                    self.assertTrue(gate(s, None, None))
            for key in q.COUNTERS[:3]:
                s = self.snapshot(active, 3)
                s[q.SECTION][key] += 1
                self.assertTrue(gate(s, None, None))
    def test_real_H3_minimum_and_extra_prefill_AR_queries_not_rigid_total(self):
        for active in (False, True):
            coverage = q.index_hooks(http, active)[1]
            a, b = self.snapshot(active, 2, 10), self.snapshot(active, 5, 999)
            details, errors = coverage(a, b, {"body": {}})
            self.assertEqual(errors, [])
            self.assertEqual(details["immutable96_minimum_VerifyR4_queries"], 3 * 624)
            bad = self.snapshot(active, 5, 9)
            key = "indexed_accepts" if active else "original_callbacks"
            bad[q.SECTION][key] = a[q.SECTION][key] + 3 * 624 - 1
            self.assertTrue(coverage(a, bad, {"body": {}})[1])
            self.assertTrue(coverage(b, a, {"body": {}})[1])
            standard_a, standard_b = self.snapshot(active), self.snapshot(active, extra=1200)
            self.assertEqual(coverage(standard_a, standard_b, {}, execution_mode="standard")[1], [])
            self.assertTrue(coverage(standard_a, standard_a, {}, execution_mode="standard")[1])
            # A fallback is a legitimate exact original callback for an indexed
            # unknown/overlapping query; counters are not faked as all hits.
            if active:
                standard_b[q.SECTION]["original_callbacks"] = 2
                self.assertEqual(coverage(standard_a, standard_b, {}, execution_mode="standard")[1], [])
    def test_original_errors_and_ownership_remain_installed(self):
        class Runner:
            http = http
            def gate_status(self, *args, **kwargs):
                return ["all original rawQ4/22/cache/numeric errors"]
            def coverage(self, *args, **kwargs):
                return {"old": True}, ["old error"]
            def ownership_policy(self, value):
                return {"old_owner": True}
        parent = SimpleNamespace(load=lambda *args, **kwargs: Runner())
        with mock.patch.object(q, "authenticate"), mock.patch.object(q, "parent_module", return_value=parent):
            runner = q.load(q.BUILD, True)
            self.assertEqual(runner.gate_status(self.snapshot(True), None, None), ["all original rawQ4/22/cache/numeric errors"])
            details, errors = runner.coverage(self.snapshot(True), self.snapshot(True, extra=100), {})
            self.assertTrue(details["old"])
            self.assertIn("old error", errors)
            self.assertTrue(runner.ownership_policy(self.snapshot(True))["old_owner"])
            changed = self.snapshot(True, extra=100)
            changed[q.SECTION]["source_policy_sha256"] = "f" * 64
            self.assertTrue(runner.coverage(self.snapshot(True), changed, {})[1])
    def test_exact_source_metadata_and_missing_fresh_receipt(self):
        q.authenticate(q.BUILD, require_state=False)
        with self.assertRaises(ValueError):
            q.authenticate(q.BUILD, require_state=True)
        with self.assertRaises(ValueError):
            q.authenticate(Path("/tmp/unknown-worker"))
        actual = q.digest
        for suffix in ("compiled-cpu-seal.json", "splash-flash", "splash.metallib"):
            with mock.patch.object(q, "digest", side_effect=lambda p, s=suffix: "f" * 64 if str(p).endswith(s) else actual(p)):
                with self.assertRaises(ValueError):
                    q.authenticate(q.BUILD)
    def test_actual_parent_admission_origin_retained(self):
        parent = q.parent_module()
        for active in (False, True):
            runner = q.load(q.BUILD, active, require_state=False)
            self.assertTrue(callable(runner.gate_status))
        # Existing complete runtime selector check remains unmodified.
        with self.assertRaises(ValueError):
            parent.runtime_args(["measure", "--runtime-build", str(q.BUILD), "--runtime-build=/tmp/other"], q.BUILD)
    def test_fresh_child_native_receipt_tamper_and_external_pin(self):
        # Synthetic metadata tests admission logic only, never GPU proof.
        wanted = {"schema": "immutable96-index-bounded-current-native-admission-v1", "pass": True,
                  "Root_GPU_executed": True, "source_policy_sha256": q.SOURCE, "worker_sha256": q.EXE,
                  "worker_seal_sha256": q.SEAL, "metallib_sha256": q.LIB, "oracle_sha256": q.QA_EXE,
                  "CPU_READY_sha256": q.QA_READY, "frames": 10, "repeated_frames": 0,
                  "all134_state_and216_physical_tapes_compared": True,
                  "original_unused_PLE_count_tails_known_initialized_QA_only": True,
                  "VerifyR4_indexed_accepts": 624, "VerifyR4_original_callbacks": 0,
                  "actual_owned_alias_guard_cases": 8, "all8_errors_exact_category_and_text": True,
                  "actual_final_zero_governor_reservations": True, "backend_destroyed": True,
                  "original22_or_whole_performance_qualified": False, "bytes_compared": 1000,
                  "CPU_SYNTHETIC_test_not_actual_admission": True}
        import json
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "synthetic.json"
            p.write_text(json.dumps(wanted))
            q.authenticate(q.BUILD, p, q.digest(p), True)
            with self.assertRaises(ValueError):
                q.authenticate(q.BUILD, p, "f" * 64, True)
            for field in ("source_policy_sha256", "worker_sha256", "CPU_READY_sha256",
                          "all134_state_and216_physical_tapes_compared",
                          "original_unused_PLE_count_tails_known_initialized_QA_only",
                          "VerifyR4_indexed_accepts", "backend_destroyed"):
                bad = dict(wanted)
                bad[field] = None
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    q.authenticate(q.BUILD, p, q.digest(p), True)

if __name__ == "__main__":
    unittest.main()
