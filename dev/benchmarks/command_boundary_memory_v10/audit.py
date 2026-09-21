"""CPU-only audit of existing v9 memory-query timing and driver-wire evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ["v9-idle-residency-refresh-saved-screen.json.commands.jsonl",
             "v9-persistent-state-idle-control.json.commands.jsonl"]
    rows = []
    for name in names:
        for line in (PROJECT / "build/release/flash" / name).read_text().splitlines():
            record = json.loads(line)
            command = record["command"]
            bridge = command["clock_bridges"]["commit"]
            assert bridge["valid"]
            offset = bridge["steady_minus_mach_seconds"]
            start = command["command_kernel_start_seconds"] + offset
            end = command["command_kernel_end_seconds"] + offset
            assert command["command_kernel_timing_valid"] and end >= start
            spans = {}
            overlap_sum = 0.0
            for name_span, span in command["device_memory_samples"].items():
                assert span["valid"]
                began, ended = span["began_steady_seconds"], span["ended_steady_seconds"]
                assert ended >= began
                overlap = max(0.0, min(ended, end) - max(began, start))
                overlap_sum += overlap
                spans[name_span] = {
                    "duration_us": span["seconds"] * 1e6,
                    "driver_interval_overlap_us": overlap * 1e6,
                    "began_minus_driver_start_ms": (began - start) * 1e3,
                    "ended_minus_driver_end_ms": (ended - end) * 1e3,
                }
            rows.append({
                "input": name,
                "sample": record["sample"],
                "segment": record["segment"],
                "refresh_enabled": record.get("refresh_enabled"),
                "kernel_driver_interval_ms": (end - start) * 1e3,
                "memory_queries": spans,
                "query_overlap_sum_us": overlap_sum * 1e6,
                "clock_bridge_uncertainty_us": bridge["uncertainty_seconds"] * 1e6,
            })
    driver = json.loads((PROJECT / "build/release/flash/v9-idle-driver-cpu-audit.json").read_text())
    windows = []
    for window in driver["windows"]:
        creation, submission, completion = (window[key] for key in
            ["creation_s", "submission_end_s", "completion_s"])
        wire = window["wire_memory"]
        windows.append({
            "sample": window["sample"],
            "cpu_encode_and_submit_ms": (submission - creation) * 1e3,
            "post_submit_completion_ms": (completion - submission) * 1e3,
            "wire_active_union_ms": wire["union_ms"],
            "wire_envelope_ms": (wire["last_end_s"] - wire["first_start_s"]) * 1e3,
            "between_wire_events_no_wire_interval_ms": (
                (wire["last_end_s"] - wire["first_start_s"]) * 1e3 - wire["union_ms"]),
            "last_wire_to_command_completion_ms": (completion - wire["last_end_s"]) * 1e3,
            "wired_bytes": wire["requested_bytes"],
            "wired_resources": wire["count"],
            "original_shard_wire_union_ms": window["shard_sized_wire_memory"]["union_ms"],
        })
    result = {
        "schema": "splash-private-boundary-memory-existing-evidence-v10",
        "cpu_only": True,
        "existing_whole_graph_samples": rows,
        "largest_query_duration_us": max(span["duration_us"] for row in rows
                                          for span in row["memory_queries"].values()),
        "driver_trace_windows": windows,
        "interpretation": [
            "Existing getter spans are too short to explain the second-scale delay as a blocking CPU read.",
            "Pre/postcommit getter spans precede the driver interval; scheduled/completed spans occur at or after its end.",
            "Wire Memory duration is substantial, but gaps between events remain unclassified without CPU call-stack profiling.",
            "Kernel-driver timestamps do not identify a specific internal driver operation.",
            "The separate traces are not direct cross-run joins, and absent resource/connection IDs prohibit exact resource causality joins.",
        ],
        "next_discriminating_measurement": "Time Profiler on native worker during one matched immediate/idle request pair, resolving stacks on the native Wire Memory thread and Metal scheduling threads.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"valid": True, "cpu_only": True, "samples": len(rows),
                      "largest_query_duration_us": result["largest_query_duration_us"]}))


if __name__ == "__main__":
    main()
