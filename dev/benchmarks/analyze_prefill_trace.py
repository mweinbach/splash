#!/usr/bin/env python3
"""Summarize metadata-only prefill-trace documents and validate attribution.

Counter attribution is descriptive; performance comes from the surrounding
unprofiled runs, because stage sampling introduces encoder boundaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def analyze_native(path, records):
    errors, statuses, by_rows = [], defaultdict(int), defaultdict(lambda: defaultdict(list))
    events = [record for record in records if record.get("event") == "prefill_completed"]
    modes = set()
    for event in events:
        modes.add(event["gpu_mode_requested"])
        rows = event["batch"]["input_rows"]
        values = by_rows[rows]
        values["gpu_ms"].append(event["batch"]["fused_gpu_seconds"] * 1000)
        values["command_wall_ms"].append(event["batch"]["fused_wall_seconds"] * 1000)
        if event.get("command_profiles_missing") or event["model_profiles_dropped"]:
            errors.append(f"event {event['event_index']}: missing instrumentation records")
        phases = defaultdict(dict)
        for phase in event["phases"]:
            phases[phase["command_sequence"]][phase["stage"]] = phase
        if len(phases) != 1 or any(set(spans) != {"graph_build", "ticket_wait", "completion_host"} for spans in phases.values()):
            errors.append(f"event {event['event_index']}: incomplete prefill phases")
        for stage in ("graph_build", "completion_host"):
            values[stage + "_ms"].append(sum(spans.get(stage, {}).get("wall_seconds", 0) for spans in phases.values()) * 1000)
        for command in event["command_profiles"]:
            statuses[command["status"]] += 1
            if command.get("dispatches_truncated") or command.get("dropped_profiles_before"):
                errors.append(f"event {event['event_index']}: incomplete dispatch metadata")
            if command["encoder_boundaries_altered"] or command["sampling_barriers"]:
                values["profile_perturbs_encoding"].append(1)
            values["scheduled_callback_lag_ms"].append((command["host_scheduled_seconds"] - command["host_commit_end_seconds"]) * 1000)
            for key in ("host_preparation_seconds", "host_encoding_seconds"):
                values[key.replace("_seconds", "_ms")].append(command[key] * 1000)
            bridges = command.get("clock_bridges", {})
            before, after = bridges.get("commit", {}), bridges.get("completed", {})
            if before.get("valid") and after.get("valid"):
                offset = before["steady_minus_mach_seconds"]
                drift = abs(after["steady_minus_mach_seconds"] - offset)
                # A sleep/clock discontinuity invalidates cross-axis inference.
                tolerance = before["uncertainty_seconds"] + after["uncertainty_seconds"] + 0.000005
                if drift <= tolerance:
                    values["commit_to_gpu_start_ms"].append((command["command_gpu_start_seconds"] + offset - command["host_commit_end_seconds"]) * 1000)
                    values["gpu_end_to_completed_callback_ms"].append((command["host_completed_seconds"] - command["command_gpu_end_seconds"] - offset) * 1000)
                    if command.get("command_kernel_timing_valid"):
                        values["driver_scheduling_ms"].append((command["command_kernel_end_seconds"] - command["command_kernel_start_seconds"]) * 1000)
                        values["driver_end_to_gpu_start_ms"].append((command["command_gpu_start_seconds"] - command["command_kernel_end_seconds"]) * 1000)
                else:
                    values["clock_bridge_discontinuities"].append(1)
            for name, sample in command.get("device_memory_samples", {}).items():
                if sample["valid"]:
                    values["memory_read_" + name + "_ms"].append(sample["seconds"] * 1000)
    if not events:
        errors.append("no completed prefill events")
    return {"source": str(path.resolve()), "trace_type": "native_jsonl", "valid": not errors,
            "validation_errors": errors, "prefill_events": len(events), "modes": sorted(modes),
            "command_status_counts": dict(statuses),
            "rows": [{"rows": rows, "events": len(values["gpu_ms"]),
                      "medians": {key: median(items) for key, items in values.items()}}
                     for rows, values in sorted(by_rows.items())],
            "timing_notes": "Callback lag is descriptive. Actual commit-to-GPU delay requires both per-command clock bridges. Ticket-consumption spans overlap command/GPU work. No performance baseline is inferred from JSONL alone."}


def analyze(path):
    contents = path.read_text()
    try:
        document = json.loads(contents)
    except json.JSONDecodeError:
        return analyze_native(path, [json.loads(line) for line in contents.splitlines() if line.strip()])
    if "event" in document:
        return analyze_native(path, [document])
    errors, attributed = [], defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    off, traced, cpu = [], [], defaultdict(list)
    statuses = defaultdict(int)
    traces = 0
    for run in document["runs"]:
        if run["label"] != "trace":
            off.append(run["prefill_gpu_seconds"])
            if run["phases"] or run["commands"]:
                errors.append("unprofiled run retained instrumentation records")
            continue
        traces += 1
        traced.append(run["prefill_gpu_seconds"])
        phases = defaultdict(dict)
        for phase in run["phases"]:
            sequence, stage = phase["command_sequence"], phase["stage"]
            if stage in phases[sequence]:
                errors.append(f"duplicate {stage} for command {sequence}")
            phases[sequence][stage] = phase
            if not finite_nonnegative(phase["wall_seconds"]) or phase["ended_steady_seconds"] < phase["began_steady_seconds"]:
                errors.append(f"invalid host span for command {sequence}")
        if len(phases) != len(run["chunks"]) or run["dropped_phases"]:
            errors.append("missing/truncated model command phases")
        for sequence, spans in phases.items():
            if set(spans) != {"graph_build", "ticket_wait", "completion_host"}:
                errors.append(f"missing phase for command {sequence}")
        for stage in ("graph_build", "completion_host"):
            cpu[stage].append(sum(spans.get(stage, {}).get("wall_seconds", 0) for spans in phases.values()))
        for command in run["commands"]:
            statuses[command["status"]] += 1
            if command.get("dispatches_truncated") or command.get("dispatch_metadata_truncated") or command["dropped_profiles_before"]:
                errors.append("missing/truncated command metadata")
            graph = phases.get(command["sequence"], {}).get("graph_build")
            if graph is None:
                # Auxiliary copies may be profiled separately. Do not count
                # them as a target-prefill chunk or guess a row count.
                continue
            rows = graph["rows"]
            if len(command["dispatches"]) != graph["dispatches"]:
                errors.append(f"dispatch count mismatch for command {command['sequence']}")
            if command["status"] != "complete":
                errors.append(f"command {command['sequence']}: {command['status']}: {command['reason']}")
                continue
            if command["mode"] == "command":
                if command["encoder_boundaries_altered"] or command["sampling_barriers"]:
                    errors.append("command profiling unexpectedly changed encoder topology")
                continue
            for dispatch in command["dispatches"]:
                if not dispatch["timestamps_valid"] or not finite_nonnegative(dispatch["gpu_seconds"]):
                    errors.append(f"unavailable timestamp in command {command['sequence']}")
                    continue
                entry = attributed[rows][dispatch["pipeline"]]
                entry[0] += 1
                entry[1] += dispatch["gpu_seconds"]
        for key in ("host_preparation_seconds", "host_encoding_seconds", "sparse_dependency_wait_seconds"):
            cpu[key].append(sum(command[key] for command in run["commands"]))
    by_rows = []
    for rows, pipelines in sorted(attributed.items()):
        total = sum(value[1] for value in pipelines.values())
        by_rows.append({"rows": rows, "counter_gpu_ms_per_trace": total * 1000 / traces,
            "pipelines": [{"pipeline": name, "dispatches_per_trace": value[0] / traces,
                           "gpu_ms_per_trace": value[1] * 1000 / traces,
                           "share": value[1] / total if total else None}
                          for name, value in sorted(pipelines.items(), key=lambda pair: pair[1][1], reverse=True)]})
    return {"source": str(path.resolve()), "valid": not errors, "validation_errors": errors,
            "device": document["device"], "prompt_tokens": document["prompt_tokens"],
            "width": document["width"], "mode": document["gpu_profile_mode"],
            "unprofiled_gpu_median_ms": median(off) * 1000,
            "profiled_gpu_median_ms": median(traced) * 1000,
            "observed_profile_gpu_overhead_percent": (median(traced) / median(off) - 1) * 100,
            "command_status_counts": dict(statuses),
            "host_median_ms_per_trace": {stage: median(values) * 1000 for stage, values in cpu.items()},
            "attribution_by_rows": by_rows,
            "scope": document["scope"],
            "timing_notes": "Ticket waits overlap GPU work. Metal calibration and host steady timestamps have separate origins. Counter-stage times include profiling perturbation; use unprofiled measurements for speed comparisons."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [analyze(path) for path in args.traces]
    result = {"valid": all(report["valid"] for report in reports), "reports": reports}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for report in reports:
        if report.get("trace_type") == "native_jsonl":
            print(f"{Path(report['source']).name}: {report['prefill_events']} prefill events; valid={report['valid']}")
            for row in report["rows"]:
                metrics = row["medians"]
                print(f"  rows={row['rows']}: GPU {metrics['gpu_ms']:.2f} ms, "
                      f"wall {metrics['command_wall_ms']:.2f} ms, "
                      f"scheduled callback lag {metrics.get('scheduled_callback_lag_ms', float('nan')):.2f} ms")
            continue
        print(f"{Path(report['source']).name}: GPU {report['unprofiled_gpu_median_ms']:.2f} ms; "
              f"profile overhead {report['observed_profile_gpu_overhead_percent']:+.2f}%; valid={report['valid']}")
        for row in report["attribution_by_rows"]:
            print(f"  rows={row['rows']}: " + "; ".join(
                f"{pipeline['pipeline']} {pipeline['gpu_ms_per_trace']:.2f} ms"
                for pipeline in row["pipelines"][:5]))
        for error in report["validation_errors"]:
            print("  ERROR:", error)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
