"""CPU-only matched HTTP throughput audit for maintenance ON/OFF reports."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from audit import maintenance_delta, number, require, timing_delta

ACCEPTANCE_COUNTERS = ("verification_cycles", "drafted_tokens", "accepted_committed_drafts",
                       "matched_proposals", "emitted_accepted_proposals",
                       "joint_head_vocabulary_register_commands", "joint_head_vocabulary_register_rows",
                       "singleton_committed_fold_calls")


def phase_calls(before, after):
    result = {}
    for tree in ("model_timing", "mtp"):
        for key, value in after[tree].items():
            if isinstance(value, dict) and "total_gpu_ms" in value:
                start = before[tree][key]
                result[f"{tree}.{key}"] = value["host_command_subphases"]["timed_commands"] \
                    - start["host_command_subphases"]["timed_commands"]
    return result


def load(path):
    report = json.loads(path.read_text())
    require(report["valid"], "invalid or incomplete HTTP performance report")
    return report


def audit_condition(report, label):
    before, after = report["runtime_before"], report["runtime_after"]
    require(before["identity"]["engine_instance_id"] == after["identity"]["engine_instance_id"],
            "engine restarted during performance benchmark")
    require(after["ready"] and after["metal"]["healthy"] and not after["cache"]["reused_tokens"],
            "unhealthy backend or KV reuse")
    groups = {}
    for wave in report["waves"]:
        require(wave["valid"], "invalid measured wave")
        require(wave["completion_tokens"] == wave["width"] * 128,
                "wrong measured completion count")
        require(len(wave["records"]) == wave["width"], "wrong measured lane count")
        for record in wave["records"]:
            require(record["http_status"] == 200 and record["done"] and not record["errors"],
                    "invalid HTTP result")
            usage = record["usage"]
            require(usage["prompt_tokens"] == wave["context"] and usage["completion_tokens"] == 128
                    and usage["prompt_tokens_details"]["cached_tokens"] == 0,
                    "wrong workload context/output/KV usage")
        score = number(wave["full_wave_output_tokens_per_second"], "wave output throughput")
        require(score > 0, "nonpositive wave throughput")
        key = f"context{wave['context']}_width{wave['width']}"
        groups.setdefault(key, []).append(score)
    require(set(groups) == {"context128_width1", "context128_width4",
                            "context2048_width1", "context2048_width4"},
            "unexpected matched workload grid")
    require(all(len(scores) == 2 for scores in groups.values()), "expected two measurements per workload")
    elapsed = after["status_snapshot"]["steady_seconds"] - before["status_snapshot"]["steady_seconds"]
    return {"label": label, "plan_sha256": report["plan_sha256"],
            "instrumentation_on": before["request_command_trace"]["enabled"],
            "diagnostic_traced_results_not_untraced_performance_score": before["request_command_trace"]["enabled"],
            "workloads": {key: {"samples_tokens_per_second": values,
                                "median_tokens_per_second": statistics.median(values)}
                          for key, values in groups.items()},
            "maintenance": maintenance_delta(before, after, elapsed),
            "user_phase_timing_delta": timing_delta(before, after),
            "user_phase_command_counts": phase_calls(before, after),
            "mtp_acceptance_counter_delta": {key: after["mtp"][key] - before["mtp"][key]
                                             for key in ACCEPTANCE_COUNTERS},
            "mtp_accepted_prefix_histogram_delta": [x - y for x, y in zip(
                after["mtp"]["accepted_prefix_histogram"], before["mtp"]["accepted_prefix_histogram"], strict=True)],
            "mtp_cycles_by_depth_delta": [x - y for x, y in zip(
                after["mtp"]["completed_cycles_by_proposed_depth"], before["mtp"]["completed_cycles_by_proposed_depth"], strict=True)],
            "runtime_after_active_requests": after["scheduler"]["active_requests"],
            "runtime_after_command_in_flight": after["scheduler"]["command_in_flight"],
            "final_drain_must_be_verified_by_separate_status": True,
            "admission_denials_delta": after["admission"]["denied_state_reservations"]
                - before["admission"]["denied_state_reservations"],
            "requests_failed_delta": after["requests"]["failed"] - before["requests"]["failed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("on", "off", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "fresh output required")
    on, off = load(args.on), load(args.off)
    require(on["plan_sha256"] == off["plan_sha256"], "different benchmark plans")
    require(on["native_mtp_depth_policy"] == off["native_mtp_depth_policy"], "different MTP policy")
    on_result, off_result = audit_condition(on, "on"), audit_condition(off, "off")
    require(on_result["instrumentation_on"] == off_result["instrumentation_on"], "different tracing mode")
    for key in ("mtp_acceptance_counter_delta", "mtp_accepted_prefix_histogram_delta",
                "mtp_cycles_by_depth_delta", "user_phase_command_counts"):
        require(on_result[key] == off_result[key], f"changed control behavior: {key}")
    for left, right in zip(on["waves"], off["waves"], strict=True):
        require((left["sample"], left["context"], left["width"]) ==
                (right["sample"], right["context"], right["width"]), "different workload wave sequence")
        for a, b in zip(left["records"], right["records"], strict=True):
            require(a["request_body"] == b["request_body"], "different lane request bodies")
            require(a["text"] == b["text"] and a["usage"] == b["usage"],
                    "changed exact outputs/usage in throughput comparison")
    changes = {key: {"off_median_tokens_per_second": values["median_tokens_per_second"],
                     "on_median_tokens_per_second": on_result["workloads"][key]["median_tokens_per_second"],
                     "relative_change": on_result["workloads"][key]["median_tokens_per_second"]
                         / values["median_tokens_per_second"] - 1}
               for key, values in off_result["workloads"].items()}
    result = {"schema": "splash-idle-maintenance-v11-matched-http-cpu-audit-v1",
              "valid": True, "gpu_work_in_this_audit": False,
              "exact_outputs_usage_requests_and_plan_unchanged": True,
              "mtp_acceptance_depth_histogram_and_user_phase_call_counts_unchanged": True,
              "no_maintenance_commands_during_active_matched_benchmark":
                  on_result["maintenance"]["counter_delta"]["completed_commands"] == 0,
              "conditions": {"on": on_result, "off": off_result}, "workload_changes": changes,
              "artifact_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (args.on, args.off)},
              "limitations": ["Two measurements per workload with sequential process order do not establish precise small differences.",
                  "Both conditions use identical request command tracing; values are a diagnostic throughput control when tracing is enabled.",
                  "Runtime-after may precede final bookkeeping; a separate healthy drained status is required."]}
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({"valid": True, "workload_changes": changes}))


if __name__ == "__main__":
    main()
