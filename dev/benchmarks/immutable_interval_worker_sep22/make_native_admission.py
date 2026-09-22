#!/usr/bin/env python3
"""Root-only tiny admission after the actual bounded flag0/flag1 pair succeeds."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
spec = importlib.util.spec_from_file_location("_immutable96_same_source_admission", Path(__file__).with_name("semantic_quality.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--control-report", type=Path, required=True)
    p.add_argument("--control-report-sha256", required=True)
    p.add_argument("--compare-report", type=Path, required=True)
    p.add_argument("--compare-report-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-root-owned", action="store_true", required=True)
    a = p.parse_args()
    require(not a.output.exists() and not Path(str(a.output) + ".writing").exists(), "fresh tiny admission required")
    build = Path("/Users/mweinbach/Projects/splash/build/immutable96-index-Q4-bounded-native-QA-sep22-v3")
    require(sha(build / "oracle") == q.QA_EXE and sha(build / "CPU_READY.json") == q.QA_READY,
            "exact current independently reviewed bounded QA source/executable required")
    require(sha(a.control_report) == a.control_report_sha256 and sha(a.compare_report) == a.compare_report_sha256,
            "externally typed actual Root report pins required")
    control = json.loads(a.control_report.read_text())
    candidate = json.loads(a.compare_report.read_text())
    for r, role, indexed, original in ((control, "export", 0, 624), (candidate, "compare", 624, 0)):
        expected = {"pass": True, "role": role, "backend_destroyed": True, "frames": 10, "repeated_frames": 0,
                    "teacher_head_proved": False, "worker_residency_proved": False}
        require(all(q.same(r.get(k), v) for k, v in expected.items()), "healthy exact bounded actual Root report required")
        producer = r["producer"]
        expected_producer = {"immutable_index_requested": role == "compare", "immutable_index_source_policy_sha256": q.SOURCE,
                            "VerifyR4_indexed_accepts": indexed, "VerifyR4_original_callbacks": original,
                            "known_initialized_original_unused_snapshot_control_tail_scope": True,
                            "samebinary_flag0_control_has_getenv_counter_overhead": True,
                            "qualified_worker_executable_sha256": q.EXE, "qualified_rowpair_library_sha256": q.LIB,
                            "actual_owned_buffer_guard_cases": 8, "actual_bundle_verify_calls": 48, "actual_bundle_verify_rows": 192,
                            "actual_rawQ4_rowpair_calls": 26, "actual_rawQ4_rowpair_rows": 104,
                            "actual_HC_VerifyR4_calls": 97, "actual_HC_VerifyR4_rows": 388, "actual_HC_padding_dispatches_saved": 97}
        require(all(q.same(producer.get(k), v) for k, v in expected_producer.items()), "actual graph/source/counter ownership differs")
        require(producer["worker"]["worker_sha256"] == q.EXE and producer["worker"]["worker_seal_sha256"] == q.SEAL and producer["worker"]["source_policy"] == q.SOURCE, "new worker source tuple differs")
        allocation = r["allocation"]
        require(all(v is True for v in allocation["guards"].values()), "actual native category/guard ledger failed")
        require(allocation["planned"]["held_separate_HostRAM_admission"] == 2 << 30, "host admission missing")
        gov = allocation["governor_snapshot"]
        require(gov["reserved_bytes"] == 0 and gov["denied_reservations"] == 0 and gov["host_measurement_valid"] is True and gov["growth_allowed"] is True, "actual zero healthy final governor required")
    require(candidate["bounded_host_state_comparison_complete"] is True, "completed bounded state comparison required")
    require(control["common"] == candidate["common"], "actual source/input/numerical controls differ")
    require(q.u64(candidate["bytes_compared"]) and 0 < candidate["bytes_compared"] < 4 << 30, "complete bounded actual bytes required")
    require(candidate["planes_compared"] == 1208, "all four134 state / three221 tape / three3 output inventories required")
    require(control["spill_bytes"] == candidate["bytes_compared"], "entire exact control spill comparison required")
    receipt = {"schema": "immutable96-index-bounded-current-native-admission-v1", "pass": True,
               "Root_GPU_executed": True, "source_policy_sha256": q.SOURCE, "worker_sha256": q.EXE,
               "worker_seal_sha256": q.SEAL, "metallib_sha256": q.LIB, "oracle_sha256": q.QA_EXE,
               "CPU_READY_sha256": q.QA_READY, "frames": 10, "repeated_frames": 0,
               "all134_state_and216_physical_tapes_compared": True,
               "original_unused_PLE_count_tails_known_initialized_QA_only": True,
               "VerifyR4_indexed_accepts": 624, "VerifyR4_original_callbacks": 0,
               "actual_owned_alias_guard_cases": 8, "all8_errors_exact_category_and_text": True,
               "actual_final_zero_governor_reservations": True, "backend_destroyed": True,
               "bytes_compared": candidate["bytes_compared"], "planes_compared": candidate["planes_compared"],
               "control_report": str(a.control_report.resolve()), "control_report_sha256": a.control_report_sha256,
               "compare_report": str(a.compare_report.resolve()), "compare_report_sha256": a.compare_report_sha256,
               "samebinary_flag0_shared_getenv_counter_cost_included": True,
               "production_uninitialized_tail_parity_claimed": False,
               "original22_or_whole_performance_qualified": False}
    Path(str(a.output) + ".writing").write_text(json.dumps(receipt, indent=2) + "\n")
    Path(str(a.output) + ".writing").rename(a.output)
    print(json.dumps({"pass": True, "path": str(a.output), "sha256": sha(a.output)}))

if __name__ == "__main__":
    main()
