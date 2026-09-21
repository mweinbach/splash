"""CPU-only audit of completed maintenance ON/OFF idle HTTP reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

MAINT_KEYS = ("private_idle_residency_maintenance", "idle_residency_maintenance")
COUNTERS = ("completed_commands", "pressure_suspensions", "maintenance_failures",
            "busy_safe_point_skips", "cold_misses_wall_at_least_interval", "gpu_ms", "wall_ms")
IDENTITY_KEYS = ("source", "loaded_model_layout_sha256", "forward_semantics",
                 "mtp_semantics", "mtp_attention_route", "joint_head_attention_route",
                 "joint_head_vocabulary_route", "joint_verifier_semantics")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value), f"nonfinite/missing {label}")
    return value


def load(path):
    value = json.loads(path.read_text())
    require(value.get("valid") is True, f"incomplete or invalid report: {path}")
    return value


def maintenance_status(status):
    present = [key for key in MAINT_KEYS if key in status]
    require(len(present) <= 1, "ambiguous maintenance status keys")
    if not present:
        return None, None
    return present[0], status[present[0]]


def status_idle(value, context):
    require(value["ready"] and value["metal"]["healthy"], f"unhealthy/not ready {context}")
    scheduler = value["scheduler"]
    require(not scheduler["active_requests"] and not scheduler["queued"]
            and not scheduler["command_in_flight"], f"model not drained {context}")
    require(not value["admission"]["waiting"] and not value["frontend"]["active"]
            and not value["frontend"]["waiting"] and not value["transport"]["pending"],
            f"frontend/admission/transport not drained {context}")
    require(not value["cache"]["enabled"] and not value["cache"]["reused_tokens"],
            f"KV reuse enabled in {context}")


def maintenance_delta(before, after, elapsed_seconds):
    start_key, start = maintenance_status(before)
    end_key, end = maintenance_status(after)
    if start is None or end is None:
        require(start is None and end is None, "maintenance status appeared/disappeared during probe")
        return None
    require(start_key == end_key, "maintenance status key changed during probe")
    require(start["requested"] == end["requested"], "maintenance request changed during probe")
    require(start["interval_ms"] == end["interval_ms"], "maintenance interval changed")
    delta = {key: number(end[key], key) - number(start[key], key) for key in COUNTERS}
    require(all(value >= 0 for value in delta.values()), "maintenance counter regressed")
    require(delta["maintenance_failures"] == 0, "maintenance failure during probe")
    require(end["added_weight_backing_bytes"] == 0, "maintenance added weight backing")
    if end["requested"]:
        require(end.get("available", True), "requested maintenance unavailable in qualified ON probe")
        require(not end.get("failure_reason", ""), "maintenance failure reason in qualified ON probe")
        require(end["immutable_owner_count"] == 1134
                and end["immutable_owner_bytes"] == 144326852608,
                "unexpected maintenance source union")
        require(end["gpu_bytes_read_per_command"] == 4536, "unexpected maintenance source reads")
        require(0 < end["added_diagnostic_allocation_bytes"] <= 32768,
                "maintenance diagnostics exceeded conservative32 KiB admission")
    else:
        require(all(value == 0 for value in delta.values()), "disabled maintenance counters advanced")
    elapsed_seconds = number(elapsed_seconds, "counter interval seconds")
    require(elapsed_seconds > 0, "nonpositive counter interval")
    return {"status_key": end_key, "requested": end["requested"],
            "available": end.get("available"), "failure_reason": end.get("failure_reason"),
            "interval_ms": end["interval_ms"],
            "elapsed_seconds": elapsed_seconds, "counter_delta": delta,
            "actual_gpu_duty_fraction": delta["gpu_ms"] / (1000 * elapsed_seconds),
            "maintenance_wall_fraction": delta["wall_ms"] / (1000 * elapsed_seconds),
            "commands_per_second": delta["completed_commands"] / elapsed_seconds,
            "mean_gpu_ms": delta["gpu_ms"] / delta["completed_commands"]
                if delta["completed_commands"] else None,
            "mean_wall_ms": delta["wall_ms"] / delta["completed_commands"]
                if delta["completed_commands"] else None,
            "last_gpu_ms": end["last_gpu_ms"], "last_wall_ms": end["last_wall_ms"],
            "maximum_wall_ms": end["maximum_wall_ms"],
            "immutable_owner_count": end["immutable_owner_count"],
            "immutable_owner_bytes": end["immutable_owner_bytes"],
            "added_diagnostic_allocation_bytes": end["added_diagnostic_allocation_bytes"],
            "physical_pinning_guaranteed": end["physical_pinning_guaranteed"],
            "request_arrival_can_overlap_one_maintenance_command":
                end["request_arrival_can_overlap_one_maintenance_command"],
            "maintenance_command_in_flight_at_end_snapshot": end["maintenance_command_in_flight"]}


def summarize(report, label):
    before, after = report["runtime_before"], report["runtime_after"]
    status_idle(after, f"{label} final")
    require(before["identity"]["engine_instance_id"] == after["identity"]["engine_instance_id"],
            f"engine instance changed in {label}")
    rows = []
    for entry in report["records"]:
        response, budget = entry["record"], entry.get("output_budget", 16)
        usage, metrics = response["usage"], response["metrics"]
        require(entry["valid"] and response["http_status"] == 200 and response["done"]
                and not response["errors"] and not response["error_frames"],
                f"invalid HTTP response in {label}:{entry['label']}")
        require(usage["prompt_tokens"] == 128 and usage["completion_tokens"] == budget
                and usage["total_tokens"] == 128 + budget
                and usage["prompt_tokens_details"]["cached_tokens"] == 0
                and usage["completion_tokens_details"]["reasoning_tokens"] == 0,
                f"wrong usage/cache in {label}:{entry['label']}")
        require(response["finish_reason"] == "length" and not response["tool_calls"]
                and not response["reasoning_text"] and response["stream"],
                f"wrong protocol result in {label}:{entry['label']}")
        require(metrics["cache"]["matched_tokens"] == 0 and metrics["cache"]["status"] == "miss",
                f"cached request in {label}:{entry['label']}")
        latency = metrics["request_latency"]
        first = number(response["first_content_ms"], "HTTP first content")
        queue = number(latency["queue_to_start_ms"], "queue to start")
        require(first > 0 and queue >= -0.01, "invalid request latency")
        rows.append({"label": entry["label"], "output_budget": budget,
                     "requested_idle_seconds": entry["idle_seconds"],
                     "http_first_content_ms": first, "http_total_ms": response["http_total_ms"],
                     "http_headers_ms": response["http_headers_ms"],
                     "server_ttft_ms": latency["ttft_ms"],
                     "queue_to_start_ms": queue,
                     "start_to_first_token_ms": latency["start_to_first_token_ms"],
                     "response_text": response["text"], "response_sha256": response["response_sha256"],
                     "request_body": response["request_body"]})
    elapsed = after["status_snapshot"]["steady_seconds"] - before["status_snapshot"]["steady_seconds"]
    require(elapsed > 0, "invalid status elapsed time")
    request_delta = {key: after["requests"][key] - before["requests"][key]
                     for key in ("submitted", "completed", "cancelled", "failed")}
    require(request_delta == {"submitted": len(rows), "completed": len(rows),
                              "cancelled": 0, "failed": 0},
            "request count delta does not match completed HTTP fixture count")
    idle = [row for row in rows if row["requested_idle_seconds"] > 0]
    immediate = [row for row in rows if row["label"].startswith("immediate")]
    return {"label": label, "engine_instance_id": after["identity"]["engine_instance_id"],
            "records": rows, "final_healthy_drained": True,
            "mean_idle_http_ttft_ms": statistics.mean(r["http_first_content_ms"] for r in idle),
            "mean_idle_queue_ms": statistics.mean(r["queue_to_start_ms"] for r in idle),
            "mean_immediate_http_ttft_ms": statistics.mean(r["http_first_content_ms"] for r in immediate),
            "maintenance": maintenance_delta(before, after, elapsed),
            "model_timing_delta": timing_delta(before, after),
            "elapsed_status_seconds": elapsed,
            "status_snapshot_age_ms": after["status_read"]["safe_point_snapshot_age_ms"],
            "memory_actual": after["memory_actual"],
            "admission_counter_delta": {key: after["admission"][key] - before["admission"][key]
                for key in ("denied_state_reservations", "admitted_after_allocation_queue_recheck")},
            "request_count_delta": request_delta}


def timing_delta(before, after):
    result = {}
    def visit(start, end, path):
        if not isinstance(end, dict):
            return
        if "total_gpu_ms" in end:
            result[path] = {key: end[key] - start[key]
                            for key in ("total_gpu_ms", "total_wall_ms", "forward_host_wall_ms")}
            require(all(v >= 0 for v in result[path].values()), f"model timing regressed {path}")
        else:
            for key, value in end.items():
                if isinstance(value, dict):
                    visit(start[key], value, f"{path}.{key}" if path else key)
    visit(before["model_timing"], after["model_timing"], "model_timing")
    visit(before["mtp"], after["mtp"], "mtp")
    return result


def compare_exact_reports(reference, candidate):
    require(len(reference["records"]) == len(candidate["records"]), "different request count")
    for left, right in zip(reference["records"], candidate["records"], strict=True):
        require(left["label"] == right["label"] and left["output_budget"] == right["output_budget"],
                "different HTTP fixture sequence")
        require(left["request_body"] == right["request_body"], "different HTTP request body")
        require(left["response_text"] == right["response_text"]
                and left["response_sha256"] == right["response_sha256"], "changed exact response")


def profile_summary(path, report):
    if path is None:
        return None
    commands = [json.loads(line) for line in path.read_text().splitlines() if line]
    instance = report["runtime_after"]["identity"]["engine_instance_id"]
    commands = [row for row in commands if row["instance_id"] == instance]
    targets = [row for row in commands if row.get("phase") == "prefill" and row.get("role") == "target_trunk"]
    maintenance = [row for row in commands if row.get("phase") == "idle_immutable_resource_maintenance"]
    require(len(targets) >= len(report["records"]), "too few target profiles")
    selected = []
    used = set()
    for entry in report["records"]:
        response = entry["record"]
        began, ended = response["client_started_monotonic"], response["client_ended_monotonic"]
        matches = [row for row in targets
                   if row["clock_bridges"]["commit"]["valid"]
                   and began - .001 <= row["host_submit_entry_steady_seconds"]
                       - row["clock_bridges"]["commit"]["steady_minus_mach_seconds"] <= ended + .001]
        require(len(matches) == 1, f"target profile temporal match ambiguous: {entry['label']}")
        row = matches[0]
        require(row["command_sequence"] not in used, "duplicate target match")
        used.add(row["command_sequence"])
        require(row["profile_present"] and row["profile_status"] == "complete"
                and row["cross_clock_valid"] and row["actual_rows"] == 128
                and row["lanes"] == 1 and not row["encoder_boundaries_altered"],
                "invalid target command profile")
        selected.append({"label": entry["label"], "command_sequence": row["command_sequence"],
                         "dispatch_count": row["dispatch_count"],
                         "target_commit_to_gpu_ms": row["commit_end_to_gpu_start_seconds"] * 1000,
                         "target_submit_entry_to_gpu_ms": row["backend_submit_entry_to_gpu_start_seconds"] * 1000,
                         "target_gpu_ms": row["gpu_seconds"] * 1000,
                         "target_command_wall_ms": row["command_wall_seconds"] * 1000})
    first = report["runtime_before"]["status_snapshot"]["steady_seconds"]
    last = report["runtime_after"]["status_snapshot"]["steady_seconds"]
    maintenance = [row for row in maintenance
                   if first <= row["host_submit_entry_steady_seconds"] <= last]
    for row in maintenance:
        require(row["role"] == "immutable_resource_maintenance" and row["actual_rows"] == 0
                and row["lanes"] == 0 and not row["requests"], "maintenance attributed to user requests")
        require(row["profile_present"] and row["profile_status"] == "complete"
                and row["cross_clock_valid"] and row["dispatch_count"] == 1
                and not row["encoder_boundaries_altered"] and not row["sampling_barriers"],
                "invalid maintenance command profile")
    gpu_sum = sum(row["gpu_seconds"] for row in maintenance) * 1000
    wall_sum = sum(row["command_wall_seconds"] for row in maintenance) * 1000
    delta = maintenance_delta(report["runtime_before"], report["runtime_after"], last - first)
    if delta is not None:
        require(delta["counter_delta"]["completed_commands"] == len(maintenance),
                "maintenance profile/status count differs")
        require(math.isclose(delta["counter_delta"]["gpu_ms"], gpu_sum, abs_tol=1e-6)
                and math.isclose(delta["counter_delta"]["wall_ms"], wall_sum, abs_tol=1e-6),
                "maintenance profile/status timing differs")
    intervals = [right["host_submit_entry_steady_seconds"] - left["host_submit_entry_steady_seconds"]
                 for left, right in zip(maintenance, maintenance[1:])]
    return {"targets": selected, "maintenance_profile_count": len(maintenance),
            "maintenance_unattributed_to_user_requests": True,
            "maintenance_profile_status_counter_sums_match": True,
            "maintenance_profile_gpu_ms": gpu_sum, "maintenance_profile_wall_ms": wall_sum,
            "maintenance_inter_command_seconds": intervals,
            "target_dispatches_unchanged_across_probe": len({row['dispatch_count'] for row in selected}) == 1}


def long_idle_summary(report):
    require(report["vm_scope"] == "systemwide", "unexpected VM scope")
    record = next(row for row in report["records"] if row["label"] == "after_long_idle")
    samples = record["snapshots"]
    require(len(samples) == 3, "expected three long-idle VM snapshots")
    intervals = []
    for left, right in zip(samples, samples[1:]):
        status_idle(left["runtime"], "long-idle snapshot")
        before, after = left["runtime"], right["runtime"]
        timings = timing_delta(before, after)
        require(all(value == 0 for phase in timings.values() for value in phase.values()),
                "maintenance leaked into user model_timing during idle")
        elapsed = after["status_snapshot"]["steady_seconds"] - before["status_snapshot"]["steady_seconds"]
        intervals.append(maintenance_delta(before, after, elapsed))
    status_idle(samples[-1]["runtime"], "last long-idle snapshot")
    wired = [number(sample["wired_bytes"], "system wired bytes") for sample in samples]
    return {"system_vm_scope": "global, temporal corroboration; no per-resource causal join",
            "requested_idle_seconds": record["idle_seconds"],
            "snapshot_elapsed_seconds": [sample["elapsed"] for sample in samples],
            "system_wired_bytes": wired, "system_wired_gib": [value / 2**30 for value in wired],
            "system_wired_spread_bytes": max(wired) - min(wired),
            "system_wired_minimum_bytes": min(wired),
            "user_model_timing_unchanged_during_measured_idle_intervals": True,
            "measured_idle_maintenance_intervals": intervals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "on", "off", "long", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--on-trace", type=Path)
    parser.add_argument("--off-trace", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "fresh output required")
    reports = {name: load(getattr(args, name)) for name in ("baseline", "on", "off", "long")}
    summaries = {name: summarize(report, name) for name, report in reports.items()}
    compare_exact_reports(summaries["baseline"], summaries["on"])
    compare_exact_reports(summaries["baseline"], summaries["off"])
    baseline_identity = reports["baseline"]["runtime_after"]["identity"]
    for name, report in reports.items():
        for key in IDENTITY_KEYS:
            require(report["runtime_after"]["identity"][key] == baseline_identity[key],
                    f"changed model/kernel identity {name}:{key}")
    require(summaries["on"]["maintenance"]["requested"]
            and not summaries["off"]["maintenance"]["requested"], "ON/OFF routes not demonstrated")
    require(summaries["on"]["maintenance"]["counter_delta"]["completed_commands"] > 0,
            "ON route executed no maintenance")
    long = summaries["long"]
    require(long["maintenance"]["requested"], "long-idle maintenance not enabled")
    require(len({row["response_text"] for row in long["records"]}) == 1, "long-idle output changed")
    profiles = {"on": profile_summary(args.on_trace, reports["on"]),
                "off": profile_summary(args.off_trace, reports["off"]),
                "long": profile_summary(args.on_trace, reports["long"])}
    gain_ms = summaries["off"]["mean_idle_http_ttft_ms"] - summaries["on"]["mean_idle_http_ttft_ms"]
    result = {"schema": "splash-private-idle-maintenance-v11-cpu-http-audit-v1",
              "valid": True, "gpu_work_in_this_audit": False,
              "diagnostic_idle_probe_not_ordinary_http_performance_score": True,
              "conditions": summaries, "profiles": profiles,
              "long_idle_vm_and_duty": long_idle_summary(reports["long"]),
              "exact_output_and_usage_unchanged_on_off": True,
              "model_source_layout_math_identity_unchanged": True,
              "comparison": {"off_minus_on_idle_http_ttft_ms": gain_ms,
                             "on_idle_http_ttft_reduction_fraction": gain_ms / summaries["off"]["mean_idle_http_ttft_ms"],
                             "off_minus_on_idle_queue_ms": summaries["off"]["mean_idle_queue_ms"] - summaries["on"]["mean_idle_queue_ms"],
                             "idle_delay_removed_in_observed_samples": summaries["on"]["mean_idle_http_ttft_ms"] < 600
                                 and summaries["off"]["mean_idle_http_ttft_ms"] > 1000,
                             "queue_delay_not_substituted_in_observed_samples": summaries["on"]["mean_idle_queue_ms"] < 100},
              "artifact_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (args.baseline, args.on, args.off, args.long, args.on_trace, args.off_trace) if path},
              "limitations": ["Sequential process order and system activity prevent attribution of small timing differences.",
                  "Maintenance can retain roughly144 GB of GPU accessibility under normal pressure; system wired counters are global.",
                  "Request arrival may overlap one synchronous maintenance command, whose observed maximum is not a hard latency bound.",
                  "First user request still pays initial wiring; readiness and maintenance do not relocate it to startup.",
                  "Idle snapshots prove measured VM stability and timing isolation, not guaranteed physical pinning or undocumented collector policy.",
                  "Full service quality/lifecycle and ordinary single/concurrent performance qualification are separate evidence."]}
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({"valid": True, "comparison": result["comparison"],
                      "on_maintenance": summaries["on"]["maintenance"],
                      "long_idle_vm_and_duty": result["long_idle_vm_and_duty"]}))


if __name__ == "__main__":
    main()
