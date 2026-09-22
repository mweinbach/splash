"""Synthetic CPU dispatch tests; no actual reports or generation data."""
import copy
import unittest
from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks.immutable_interval_postrun_sep22 import compare as c
from dev.benchmarks.immutable_interval_worker_sep22 import semantic_quality as q

def snapshot(active, cycles):
    hist = [0] * 16
    hist[3] = cycles
    work = 624 * cycles + 100 * cycles
    return {"identity": {"engine_instance_id": int(active) + 100, "source": "synthetic"},
            q.SECTION: {"schema": q.SCHEMA, "requested": active, "source_policy_sha256": q.SOURCE,
                        "scope": q.SCOPE, "GPU_allocation_bytes_added": 0,
                        "FP_graph_public_guards_unchanged": True, "counter_scope": q.COUNTER_SCOPE,
                        "finalized_tables": int(active), "finalized_spans": 96 * int(active),
                        "indexable_tables": int(active), "indexed_accepts": work if active else 0,
                        "original_callbacks": 0 if active else work},
            "mtp": {"completed_cycles_by_proposed_depth": hist}}

def fixture(active):
    plan = {"content_sha256": c.PLAN_SHA, "cases": [{"id": str(i), "body": {}} for i in range(22)]}
    report = {"schema": "splash-prefill4k-semantic-report-v1", "execution_mode": "mtp3",
              "completed": True, "full_plan_coverage": True, "strict_cache_graph_coverage_required": True,
              "runtime_file_sha256": {"splash-flash": q.EXE, "splash.metallib": q.LIB},
              "plan_content_sha256": c.PLAN_SHA, "initial_status": snapshot(active, 0),
              "final_status": snapshot(active, 22), "store_witness": {},
              "cases": [{"id": str(i), "status_before": snapshot(active, i),
                         "status_after": snapshot(active, i + 1)} for i in range(22)]}
    return plan, report

class Tests(unittest.TestCase):
    def bound(self):
        p, a = fixture(False)
        _, b = fixture(True)
        return c.RecordedIndexFlags(q, http).bind([a, b], p), p, a, b
    def test_exact_both_flags_and_distinct_perrun_engine_ids(self):
        d, p, a, b = self.bound()
        for flag, report in ((False, a), (True, b)):
            self.assertIs(d.flag_for(report["initial_status"]), flag)
            self.assertEqual(d.status(status=report["initial_status"], plan=p, store={}), [])
            for case in report["cases"]:
                self.assertEqual(d.coverage(case["status_before"], case["status_after"], {})[1], [])
    def test_mixed_or_unknown_complete_fingerprints_rejected(self):
        d, p, a, b = self.bound()
        changed = copy.deepcopy(b["initial_status"])
        changed["arbitrary_extra_metadata"] = 1
        self.assertTrue(d.status(changed, p, {}))
        self.assertTrue(d.coverage(a["cases"][0]["status_before"], b["cases"][0]["status_after"], {})[1])
        with self.assertRaises(ValueError):
            d.ownership(changed)
    def test_source_requested_types_counter_minimum_and_identity_drift(self):
        for group, field, value in ((q.SECTION, "source_policy_sha256", "f" * 64),
                                    (q.SECTION, "requested", False),
                                    (q.SECTION, "indexed_accepts", True),
                                    ("identity", "engine_instance_id", 999)):
            p, a = fixture(False)
            _, b = fixture(True)
            b["cases"][4]["status_after"][group][field] = value
            with self.assertRaises(ValueError):
                c.RecordedIndexFlags(q, http).bind([a, b], p)
        p, a = fixture(False)
        _, b = fixture(True)
        b["cases"][4]["status_after"][q.SECTION]["indexed_accepts"] = b["cases"][4]["status_before"][q.SECTION]["indexed_accepts"] + 623
        with self.assertRaises(ValueError):
            c.RecordedIndexFlags(q, http).bind([a, b], p)
    def test_complete22_runtime_and_external_plan_are_mandatory(self):
        for field, bad in (("completed", False), ("full_plan_coverage", False),
                           ("strict_cache_graph_coverage_required", False),
                           ("plan_content_sha256", "f" * 64),
                           ("runtime_file_sha256", {"splash-flash": "f" * 64, "splash.metallib": q.LIB})):
            p, a = fixture(False)
            _, b = fixture(True)
            b[field] = bad
            with self.assertRaises(ValueError):
                c.RecordedIndexFlags(q, http).bind([a, b], p)
        p, a = fixture(False)
        _, b = fixture(True)
        b["cases"].pop()
        with self.assertRaises(ValueError):
            c.RecordedIndexFlags(q, http).bind([a, b], p)
    def test_original_errors_comparator_and_grader_are_retained(self):
        d, p, a, b = self.bound()
        class Runner:
            def gate_status(self, *args, **kwargs):
                return ["old raw/base/compact/HC error"]
            def coverage(self, *args, **kwargs):
                return {"old": True}, ["old coverage error"]
            def ownership_policy(self, status):
                return {"old_owner": True}
            def compare(self, args):
                return 123
            def grade_record(self, *args):
                return "unchanged"
        r = Runner()
        compare, grade = r.compare.__func__, r.grade_record.__func__
        c.install(r, d)
        self.assertIn("old raw/base/compact/HC error", r.gate_status(b["initial_status"], p, {}))
        details, errors = r.coverage(b["cases"][0]["status_before"], b["cases"][0]["status_after"], {})
        self.assertTrue(details["old"])
        self.assertIn("old coverage error", errors)
        self.assertTrue(r.ownership_policy(b["initial_status"])["old_owner"])
        self.assertIs(r.compare.__func__, compare)
        self.assertIs(r.grade_record.__func__, grade)

if __name__ == "__main__":
    unittest.main()
