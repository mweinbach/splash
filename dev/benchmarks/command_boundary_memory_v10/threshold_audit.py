"""CPU-only temporal join of Root's HTTP idle-threshold probes and command trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--stacks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    http = json.loads(args.http.read_text())
    commands = [json.loads(line) for line in args.trace.read_text().splitlines()]
    candidates = [c for c in commands if c["role"] == "target_trunk"
                  and c["phase"] == "prefill" and c["actual_rows"] == 128]
    rows, sequences = [], set()
    for probe in http["records"]:
        record = probe["record"]
        start = record["client_started_monotonic"]
        end = record["client_ended_monotonic"]
        matched = []
        for command in candidates:
            bridge = command["clock_bridges"]["commit"]
            assert bridge["valid"]
            offset = bridge["steady_minus_mach_seconds"]
            submit = command["host_submit_entry_steady_seconds"] - offset
            if start <= submit <= end:
                matched.append(command)
        if len(matched) != 1:
            raise AssertionError(f"Expected one temporal match for probe {probe['index']}; got {len(matched)}")
        command = matched[0]
        assert command["command_sequence"] not in sequences
        sequences.add(command["command_sequence"])
        assert probe["valid"] and command["cross_clock_valid"]
        assert command["driver_kernel_timing_valid"] and command["hardware_timestamps_valid"]
        assert not command["encoder_boundaries_altered"] and not command["sampling_barriers"]
        assert record["text"] == "1" and record["usage"]["prompt_tokens"] == 128
        assert record["usage"]["completion_tokens"] == 1
        assert record["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        row = {
            "index": probe["index"],
            "requested_idle_seconds": probe["requested_idle_seconds"],
            "elapsed_idle_seconds": probe["elapsed_idle_seconds"],
            "http_request_id": record["request_ids"][0],
            "native_request": command["requests"],
            "command_sequence": command["command_sequence"],
            "ttft_ms": record["metrics"]["request_latency"]["ttft_ms"],
            "native_start_to_first_token_ms": record["metrics"]["request_latency"]["start_to_first_token_ms"],
            "commit_end_to_gpu_start_ms": command["commit_end_to_gpu_start_seconds"] * 1e3,
            "driver_processing_ms": command["driver_kernel_processing_seconds"] * 1e3,
            "actual_gpu_ms": command["gpu_seconds"] * 1e3,
            "cross_clock_uncertainty_us": command["cross_clock_uncertainty_seconds"] * 1e6,
            "join_method": "unique native submit Mach time inside client monotonic interval; no shared HTTP/native request identifier",
        }
        rows.append(row)
    warmed = [row for row in rows if row["index"] > 0]
    fast = [row for row in warmed if row["commit_end_to_gpu_start_ms"] < 100]
    slow = [row for row in warmed if row["commit_end_to_gpu_start_ms"] >= 100]
    result = {
        "schema": "splash-idle-threshold-cpu-audit-v10",
        "cpu_only": True,
        "all_http_outputs_and_native_command_traces_valid": True,
        "matched_probes": len(rows),
        "rows": rows,
        "initial_probe_uncontrolled_prior_idle_excluded_from_threshold": True,
        "largest_fast_observed_idle_seconds": max(row["elapsed_idle_seconds"] for row in fast),
        "smallest_slow_observed_idle_seconds": min(row["elapsed_idle_seconds"] for row in slow),
        "fast_median_commit_to_gpu_ms": median(row["commit_end_to_gpu_start_ms"] for row in fast),
        "slow_median_commit_to_gpu_ms": median(row["commit_end_to_gpu_start_ms"] for row in slow),
        "fast_median_actual_gpu_ms": median(row["actual_gpu_ms"] for row in fast),
        "slow_median_actual_gpu_ms": median(row["actual_gpu_ms"] for row in slow),
        "interpretation": "Idle sensitivity is inside the command's driver-processing interval, with little change in actual GPU execution. A one-to-two-second observed threshold suggests a resource or process expiry policy; the exact policy is not identified.",
        "limits": [
            "The threshold is a coarse bracket from one ascending/descending sweep; it is not a measured exact timeout.",
            "Elapsed idle is client-side after the prior response, not an authoritative kernel resource last-use timestamp.",
            "Command-level profiling is enabled; no stage/counter boundaries changed.",
            "These command traces do not contain kernel stacks or resource-to-command driver IDs.",
        ],
    }
    if args.stacks:
        text = args.stacks.read_text()
        stacks = []
        for signature in ["CommandTicket::wait()", "launchMappingThread", "IOGPUCommandQueueSubmitCommandBuffers", "currentAllocatedSize"]:
            hits = []
            for line in text.splitlines():
                if signature in line:
                    match = re.search(r"(?:^|\s)(\d+)\s+" + re.escape(signature), line)
                    # ObjC/mangled symbols may prefix the signature; first count is still the sample tree's inclusive count.
                    if not match:
                        match = re.search(r"(?:^|[ +!:|])\s*(\d+)\s+", line)
                    hits.append({"line": line.strip(), "inclusive_samples": int(match[1]) if match else None})
            stacks.append({"signature": signature, "hits": hits})
        result["user_stack_sample"] = {
            "input": str(args.stacks), "signature_hits": stacks,
            "limits": [
                "Inclusive sample counts are not durations and sampling coverage does not span the complete request.",
                "The launchMappingThread IPC wait may be a permanent listener; it is not evidence of active page mapping by itself.",
                "The submission trap shows a driver boundary but cannot identify kernel wire/map/cache work.",
            ],
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["cpu_only", "matched_probes", "largest_fast_observed_idle_seconds", "smallest_slow_observed_idle_seconds", "fast_median_commit_to_gpu_ms", "slow_median_commit_to_gpu_ms", "fast_median_actual_gpu_ms", "slow_median_actual_gpu_ms"]}))


if __name__ == "__main__":
    main()
