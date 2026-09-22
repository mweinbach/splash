"""Refresh decode scenarios from existing JSON evidence; no inference."""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "build/release/flash"
PINS = {}


def read(name):
    path = EVIDENCE / name
    data = path.read_bytes()
    PINS[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def batch_rates(report):
    assert report["valid"] and report["native_common_span_available"]
    result = {}
    for width in (2, 4):
        waves = [w for w in report["waves"] if w["width"] == width and not w["warmup"]]
        assert len(waves) == 3 and all(w["valid"] for w in waves)
        result[width] = median(w["exact_native_common_span"]["native_aggregate_decode_tokens_per_second"] for w in waves)
    return result


def main():
    old = read("sep21-decode-current-route-roofs-v2.json")
    current = read("sep22-4k-prefill-current-acceptance-decode-scenarios-v1.json")
    sensitivity = read("sep22-qualified-rawQ4-current-verifier-decode-sensitivities-v1.json")
    q4 = read("sep22-rawQ4-GDN26-matched-actual-performance-comparison-v1.json")
    standard = batch_rates(read("sep22-restored-batch-bqsa-teacher-bulk0-standard-B4-B2-v1.native-aggregate-audit.json"))
    mtp = batch_rates(read("sep22-batchinteger-target-only-bulk0-MTP3-B4-B2-v3.native-aggregate-audit.json"))
    assert q4["pass"] and sensitivity["actualCycles"] == 105 and sensitivity["actualAcceptedDrafts"] == 150
    assert not old["physical_DRAM_transaction_counters_collected"]
    scenarios = {r["requested_batch"]: r for r in old["shared_native_cohort_scenarios"]}
    rows = []
    for width in (1, 2, 4):
        s = scenarios[width]
        rows.append({
            "native_lanes": width,
            "standard_optimistic_payload_ceiling_tps": s["standard_unique_operand_traffic_ceiling_aggregate_tps"],
            "standard_streaming_reference_tps": s["standard_no_interrow_cache_reuse_streaming_reference_aggregate_tps"],
            "mtp_optimistic_payload_ceiling_tps": current["mtp_current_acceptance_unique_traffic_ceiling_tps"] if width == 1 else s["mtp_unique_operand_traffic_ceiling_aggregate_tps"],
            "mtp_streaming_reference_tps": current["mtp_current_acceptance_streaming_reference_tps"] if width == 1 else s["mtp_no_interrow_cache_reuse_streaming_reference_aggregate_tps"],
            "mtp_perfect_four_unique_ceiling_tps": s["mtp_perfect_prefix4_unique_operand_traffic_ceiling_aggregate_tps"],
            "mtp_observed_prefix_per_lane_cycle": sensitivity["prefixPerCycle"] if width == 1 else s["mtp_committed_prefix_per_lane_cycle"],
            "standard_observed_native_tps": None if width == 1 else standard[width],
            "mtp_observed_native_tps": sensitivity["actual_rate_tps"] if width == 1 else mtp[width],
            "observation_scope": "Q4 singleton: original22 qualified, current prefill >4K median" if width == 1 else "separate restored standard and integer-verifier MTP paths; batch original22 qualification and restored composition pending",
        })
    out = {
        "schema": "splash-current-decode-roofs-and-observations-sep22-v1",
        "GPU_executed": False, "model_or_operand_payload_read": False,
        "physical_DRAM_traffic_measured": False, "attainable_peak_proven": False,
        "effective_resident_read_payload_GBps": old["measured_large_resident_read_payload_GBps"],
        "rows": rows, "current_B1_verifier_sensitivities": sensitivity["conditionalTargets"],
        "canonical_controls": "uncached 2048 input / 256 output, greedy, 1 warmup / 3 measured; native emitted post-first decode",
        "limitations": [
            "Ceiling = effective payload bandwidth times useful tokens / modeled operand-state bytes; compute, transactions, dispatch and host costs omitted.",
            "Inventory and sharing assumptions are inherited from the source-backed route report; these are not new physical traffic measurements.",
            "Perfect-four acceptance is unobserved. Conditional verifier speedups hold other costs and acceptance fixed.",
            "B2/B4 figures are aggregate, not per-lane. Common graphs above four lanes are unsupported.",
            "No current qualified standard B1 benchmark is substituted from an older numerical policy.",
        ],
        "provenance_sha256": PINS,
    }
    path = EVIDENCE / "sep22-current-decode-roofs-and-observations-v1.json"
    assert not path.exists(), "Preserve earlier reports"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"report": str(path), "rows": rows}))


if __name__ == "__main__":
    main()
