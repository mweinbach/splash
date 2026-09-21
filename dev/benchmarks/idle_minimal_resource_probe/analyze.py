#!/usr/bin/env python3
"""CPU-only summary of completed private minimal resource probe reports."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def summarize(paths: list[Path]) -> dict:
    cases = []
    for path in paths:
        data = json.loads(path.read_text())
        if not data.get("valid") or not data.get("execution_complete"):
            raise ValueError(f"incomplete or invalid probe report: {path}")
        if data.get("command_timing_abi_bytes") != 200:
            raise ValueError(f"incorrect CommandTiming ABI: {path}")
        groups = defaultdict(list)
        geometries = {item["name"]: item for item in data["geometries"]}
        for sample_index, sample in enumerate(data["samples"]):
            if not sample.get("exact_words_and_guards"):
                raise ValueError(f"invalid exact word/guard result: {path}")
            wake_target = None
            if sample["phase"] == "idle_tiny_wake" and sample_index + 1 < len(data["samples"]):
                following = data["samples"][sample_index + 1]
                if following["phase"] == "after_tiny_wake":
                    wake_target = following["geometry"]
            groups[(sample["geometry"], sample["phase"], sample["requested_idle_seconds"], wake_target)].append(sample)
        for (geometry, phase, idle, wake_target), samples in groups.items():
            def values(field: str):
                result = [s[field] for s in samples if s.get(field) is not None]
                if any(not math.isfinite(v) for v in result):
                    raise ValueError(f"nonfinite timing field {field}: {path}")
                return result

            wait = values("commit_end_to_hardware_gpu_start_seconds")
            os_wait = values("commit_end_to_os_kernel_start_seconds")
            gpu = [s["command"]["gpu_seconds"] for s in samples]
            wall = values("end_to_end_seconds")
            actual_idle = values("previous_hardware_gpu_end_to_commit_begin_seconds")
            case = {
                "report": str(path),
                "mode": data["mode"],
                "weights_loaded": data["weights_loaded"],
                "geometry": geometry,
                "phase": phase,
                "tiny_wake_target_geometry": wake_target,
                "requested_idle_seconds": idle,
                "source_base_count": geometries[geometry]["source_base_count"],
                "native_source_bytes": geometries[geometry]["native_source_bytes"],
                "gpu_source_words_read": geometries[geometry]["gpu_source_words_read"],
                "sample_count": len(samples),
                "hardware_gpu_wait_median_ms": statistics.median(wait) * 1000,
                "hardware_gpu_wait_ms": [v * 1000 for v in wait],
                "os_kernel_start_wait_median_ms": statistics.median(os_wait) * 1000 if os_wait else None,
                "gpu_execution_median_us": statistics.median(gpu) * 1e6,
                "gpu_execution_us": [v * 1e6 for v in gpu],
                "end_to_end_median_ms": statistics.median(wall) * 1000,
                "actual_gpu_idle_median_seconds": statistics.median(actual_idle) if actual_idle else None,
                "memory_query_spans": [s["command"]["device_memory_samples"] for s in samples],
                "host_preparation_ms": [s["command"]["host_preparation_seconds"] * 1000 for s in samples],
                "host_encoding_ms": [s["command"]["host_encoding_seconds"] * 1000 for s in samples],
                "host_commit_ms": [s["command"]["host_commit_seconds"] * 1000 for s in samples],
            }
            cases.append(case)
    return {
        "schema": "splash-private-idle-minimal-resource-probe-cpu-summary-v1",
        "gpu_commands_executed_by_analyzer": 0,
        "valid": True,
        "cases": cases,
        "interpretation_limits": [
            "kernel clock concerns the OS kernel; hardware GPU execution uses GPUStartTime/GPUEndTime",
            "native resource count and total declared bytes are confounded in original1/4/21",
            "anonymous4g does not assume all pages have become physically committed",
            "profiling and ordinary backend memory queries remain present; their spans may overlap GPU work",
            "tiny wake ruling out a global wake delay does not alone establish the precise driver VM mechanism",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a fresh CPU summary output")
    result = summarize(args.reports)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        print(f'{case["mode"]} {case["geometry"]} {case["phase"]} idle={case["requested_idle_seconds"]} '
              f'wait={case["hardware_gpu_wait_median_ms"]:.3f}ms '
              f'gpu={case["gpu_execution_median_us"]:.2f}us '
              f'wall={case["end_to_end_median_ms"]:.3f}ms')


if __name__ == "__main__":
    main()
