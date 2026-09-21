"""CPU-only independent consolidation of idle wired-page, command and minimal-probe evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
REPORTS = PROJECT / "build/release/flash"
GIB = 1024 ** 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = [REPORTS / name for name in [
        "v10-idle-wired-pages.json", "v10-idle-threshold-cpu-audit.json",
        "v9-idle-driver-cpu-audit.json", "v10-resource-root-cause-cpu-summary-v2.json",
        "v10-resource-model-scaling.json", "v10-resource-tiny-only.json",
    ]]
    raw, threshold, driver, minimal, model_probe, tiny_probe = [
        json.loads(path.read_text()) for path in inputs]
    assert raw["systemwide_not_process_scoped"]
    record = raw["record"]
    assert record["text"] == "1" and record["done"] and not record["errors"]
    assert record["usage"]["prompt_tokens"] == 128
    assert record["usage"]["completion_tokens"] == 1
    assert record["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert threshold["all_http_outputs_and_native_command_traces_valid"]
    for probe in [model_probe, tiny_probe]:
        assert probe["valid"] and probe["execution_complete"]
        assert all(sample["exact_words_and_guards"] for sample in probe["samples"])
    samples = raw["samples"]
    before = samples[0]
    after = [sample for sample in samples if sample["phase"] == "after_request"]
    initial, final = after[0], after[-1]
    page_size = initial["vm"]["page_size"]
    assert page_size == 16384
    assert all(sample["vm"]["page_size"] == page_size for sample in samples)
    assert all(sample["vm"]["wired_bytes"] ==
               sample["vm"]["pages"]["Pages wired down"] * page_size for sample in samples)

    def page_delta(first: dict, last: dict, key: str) -> int:
        return last["vm"]["pages"][key] - first["vm"]["pages"][key]

    wired_loss = initial["vm"]["wired_bytes"] - final["vm"]["wired_bytes"]
    expected_driver_bytes = driver["windows"][0]["wire_memory"]["requested_bytes"]
    assert all(window["wire_memory"]["requested_bytes"] == expected_driver_bytes
               for window in driver["windows"])
    state_bytes = driver["between_captured_requests"]["source_state_attribution"]["expected_bytes"]
    assert driver["between_captured_requests"]["source_state_attribution"][
        "exact_match_to_observed_unwire_size_multiset"]
    timeline = []
    for sample in after:
        timeline.append({
            "requested_delay_s": sample["requested_delay"],
            "elapsed_delay_s": sample["elapsed_delay"],
            "sample_time_minus_first_vm_sample_s": sample["sample_monotonic"] - initial["sample_monotonic"],
            "wired_bytes": sample["vm"]["wired_bytes"],
            "wired_gib": sample["vm"]["wired_bytes"] / GIB,
            "wired_delta_from_initial_bytes": sample["vm"]["wired_bytes"] - initial["vm"]["wired_bytes"],
            "file_backed_delta_from_initial_bytes": page_delta(initial, sample, "File-backed pages") * page_size,
            "pageins_since_request_completion": page_delta(initial, sample, "Pageins"),
            "pageouts_since_request_completion": page_delta(initial, sample, "Pageouts"),
            "swapins_since_request_completion": page_delta(initial, sample, "Swapins"),
            "swapouts_since_request_completion": page_delta(initial, sample, "Swapouts"),
        })
    # Source geometry determines exactly which original native buffers are declared.
    geometries = {g["name"]: g for g in model_probe["geometries"]}
    selected_minimal = []
    for sample in model_probe["samples"]:
        if sample["phase"] not in ["idle_direct", "idle_tiny_wake", "after_tiny_wake",
                                   "idle_immediate_followup", "wake_immediate_followup"]:
            continue
        g = geometries[sample["geometry"]]
        c = sample["command"]
        selected_minimal.append({
            "geometry": sample["geometry"], "phase": sample["phase"],
            "declared_native_source_count": g["source_base_count"],
            "declared_native_source_bytes": g["native_source_bytes"],
            "gpu_source_words_read": g["gpu_source_words_read"],
            "requested_idle_s": sample["requested_idle_seconds"],
            "actual_previous_gpu_end_to_commit_begin_s": sample["previous_hardware_gpu_end_to_commit_begin_seconds"],
            "commit_end_to_actual_gpu_ms": sample["commit_end_to_hardware_gpu_start_seconds"] * 1e3,
            "gpu_execution_us": c["gpu_seconds"] * 1e6,
            "driver_kernel_processing_ms": (c["command_kernel_end_seconds"] - c["command_kernel_start_seconds"]) * 1e3,
            "exact_words_and_guards": sample["exact_words_and_guards"],
        })
    result = {
        "schema": "splash-idle-wired-page-reclaim-cpu-audit-v10",
        "valid": True, "cpu_only": True, "gpu_commands_executed_by_auditor": 0,
        "input_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in inputs},
        "system_vm_scope": "Systemwide VM counters; not PID-scoped and not joined to Metal resource IDs",
        "workload": {"prompt_tokens": 128, "output_tokens": 1, "cached_tokens": 0,
                     "output": "1", "request_completed_successfully": True},
        "timeline": timeline,
        "byte_reversal": {
            "request_wired_growth_bytes": initial["vm"]["wired_bytes"] - before["vm"]["wired_bytes"],
            "request_file_backed_reduction_bytes": -page_delta(before, initial, "File-backed pages") * page_size,
            "idle_wired_loss_bytes": wired_loss,
            "idle_file_backed_growth_bytes": page_delta(initial, final, "File-backed pages") * page_size,
            "idle_free_growth_bytes": page_delta(initial, final, "Pages free") * page_size,
            "idle_active_growth_bytes": page_delta(initial, final, "Pages active") * page_size,
            "idle_inactive_growth_bytes": page_delta(initial, final, "Pages inactive") * page_size,
            "idle_anonymous_growth_bytes": page_delta(initial, final, "Anonymous pages") * page_size,
            "idle_compressor_occupied_delta_bytes": page_delta(initial, final, "Pages occupied by compressor") * page_size,
            "previous_native_driver_wired_requested_bytes": expected_driver_bytes,
            "vm_wired_loss_minus_previous_driver_bytes": wired_loss - expected_driver_bytes,
            "relative_byte_difference_percent": 100 * (wired_loss - expected_driver_bytes) / expected_driver_bytes,
        },
        "idle_io_deltas": {key: {"pages": page_delta(initial, final, key),
                                  "bytes": page_delta(initial, final, key) * page_size}
                           for key in ["Pageins", "Pageouts", "Swapins", "Swapouts"]},
        "state_destruction_scale": {
            "known_trunk_state_release_bytes": state_bytes,
            "known_state_release_percent_of_vm_idle_wired_loss": 100 * state_bytes / wired_loss,
            "initial_to_0_25s_vm_wired_loss_bytes": initial["vm"]["wired_bytes"] - after[2]["vm"]["wired_bytes"],
            "interpretation": "A 349MB request-state release can explain the small immediate decline within global counter noise; it cannot explain the later 145.9GB decline.",
        },
        "threshold_cross_run_corroboration": {key: threshold[key] for key in [
            "largest_fast_observed_idle_seconds", "smallest_slow_observed_idle_seconds",
            "fast_median_commit_to_gpu_ms", "slow_median_commit_to_gpu_ms",
            "fast_median_actual_gpu_ms", "slow_median_actual_gpu_ms"]},
        "minimal_resource_probes": selected_minimal,
        "tiny_only_idle_control": [
            {"requested_idle_s": sample["requested_idle_seconds"],
             "commit_end_to_actual_gpu_ms": sample["commit_end_to_hardware_gpu_start_seconds"] * 1e3,
             "gpu_execution_us": sample["command"]["gpu_seconds"] * 1e6,
             "exact_words_and_guards": sample["exact_words_and_guards"]}
            for sample in tiny_probe["samples"] if sample["phase"] == "idle_direct"
        ],
        "proven_observations": [
            "A successful model request coincides with system wired memory growing by 145.88GB and file-backed memory shrinking by 144.33GB.",
            "After completion, wired memory remains near 160.4GB through the 1.25s sample, then falls sharply by the 1.5s and 1.75s samples and returns near its pre-request level by 3s.",
            "The five-second wired loss is 145,926,045,696 bytes, within 0.0123 percent of the 145,943,855,104 bytes requested by native Wire Memory events in a separate earlier trace.",
            "The released wired pages mostly reappear in file-backed/active page accounting; free memory changes by only about 620MB.",
            "Idle IO counter deltas are 49 pageins (802,816 bytes), zero pageouts, zero swapins and zero swapouts.",
            "An independent tiny-wake command does not remove the large resource command delay even though GPU idle since the tiny command is less than 0.25ms.",
        ],
        "supported_root_cause_inference": "The major delay is repeated OS/Metal driver preparation of previously resident model pages after their wired/pinned status expires during idle. Full native resources are prepared despite tiny shader reads. This combines native PID-scoped wiring traces, systemwide page-state reversal, command-driver timing and minimal-resource wake controls.",
        "unsupported_or_unresolved_claims": [
            "The proprietary collector's exact timeout, method name, budget policy and trigger are not identified.",
            "The VM samples do not directly identify a PID, specific weight resource, residency set or command buffer.",
            "Separate driver and VM runs cannot be treated as an exact resource-level causal join.",
            "Minimal probes confound declared source bytes and resource count; they do not establish linear per-byte cost.",
            "The short screen is not broad statistical repeatability evidence or a demonstrated production fix.",
        ],
        "primary_alternatives_constrained": {
            "model_reload_or_disk_faulting": "Not a primary explanation: almost no pageins, no pageouts/swap, and resident file-backed accounting reverses the wired change while native model owners remain alive.",
            "gpu_global_power_wake": "Not sufficient: tiny-only wakes in about 7–10ms, while a just-woken GPU still waits about 855ms for all21 original resources.",
            "gpu_math_or_thermal_throttling": "Not a primary explanation: threshold sweep added about1.29s driver wait while actual GPU work changed about10ms; minimal shader work is only microseconds.",
            "application_memory_getter_or_encoder": "Not a primary explanation in captured samples: getter spans are microseconds and application prepare/encode/commit is submillisecond in minimal probes.",
            "request_state_release": "Not sufficient: known349MB state is only0.239 percent of the145.9GB idle wired decline.",
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"valid": True, "cpu_only": True, "wired_loss_bytes": wired_loss,
                      "vm_vs_driver_difference_percent": result["byte_reversal"]["relative_byte_difference_percent"],
                      "timeline_samples": len(timeline), "minimal_probe_samples": len(selected_minimal)}))


if __name__ == "__main__":
    main()
