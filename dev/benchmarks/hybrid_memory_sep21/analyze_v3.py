#!/usr/bin/env python3
"""CPU-only admission and command-boundary analysis of the saved hybrid run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]


def analyze(path: Path) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    assert report["completed"] and all(w["valid"] for w in report["waves"])
    initial = report["server_runs"][0]["initial_status"]
    waves = []
    for w in report["waves"]:
        delta = w["native_counter_delta"]
        record = w["records"][0]
        measurement = record["measurement"]
        phases = {}
        for name in ("model_timing.prefill", "mtp.head_priming", "mtp.head_decode", "mtp.target_verify", "mtp.prefix_restore"):
            phases[name] = dict(gpu_ms=delta[name + ".total_gpu_ms"],
                                command_wall_ms=delta[name + ".total_wall_ms"],
                                forward_host_wall_ms=delta[name + ".forward_host_wall_ms"],
                                commit_to_scheduled_callback_ms=delta[name + ".host_command_subphases.commit_to_scheduled_callback_ms"],
                                preparation_ms=delta[name + ".host_command_subphases.preparation_ms"],
                                encoding_ms=delta[name + ".host_command_subphases.encoding_ms"])
        gpu = phases["model_timing.prefill"]["gpu_ms"]
        forward = phases["model_timing.prefill"]["forward_host_wall_ms"]
        steady_gpu = sum(phases[p]["gpu_ms"] for p in ("mtp.head_decode", "mtp.target_verify", "mtp.prefix_restore"))
        governor = w["status_after"]["memory_governor"]
        waves.append(dict(trial=w["trial"], warmup=w["warmup"], phases=phases,
                          native_post_first_emission_tokens=measurement["native_post_first_emission_tokens"],
                          native_exact_decode_tps=measurement["native_exact_decode_tokens_per_second"],
                          prefill_gpu_tps=w["prompt_tokens"] * 1000 / gpu,
                          prefill_inclusive_forward_tps=w["prompt_tokens"] * 1000 / forward,
                          unchanged_observed_decode_gpu_only_tps=measurement["native_post_first_emission_tokens"] * 1000 / steady_gpu,
                          initial_emission_to_done_ms=measurement["native_first_emission_to_done_ms"],
                          request_queue_to_start_ms=record["metrics"]["request_latency"]["queue_to_start_ms"],
                          host_available_bytes=governor["host_available_bytes"],
                          host_headroom_above_reserve_bytes=governor["host_available_bytes"]-governor["host_reserve_bytes"],
                          growth_allowed=governor["growth_allowed"],
                          denied_reservations=governor["denied_reservations"],
                          actual_memory_bytes=w["status_after"]["memory_actual"]["current_bytes"],
                          state_idle=w["post_wave_idle_pending"] is False,
                          verification_cycles=w["mtp_counters"]["mtp.verification_cycles"],
                          accepted_committed_drafts=w["mtp_counters"]["mtp.accepted_committed_drafts"]))
    warm = [w for w in waves if not w["warmup"]]
    median_gpu = statistics.median(w["phases"]["model_timing.prefill"]["gpu_ms"] for w in warm)
    last = warm[-1]
    fixed_other = last["phases"]["model_timing.prefill"]["forward_host_wall_ms"] - last["phases"]["model_timing.prefill"]["gpu_ms"]
    initial_host = initial["memory_governor"]
    first_host_drop = initial_host["host_available_bytes"]-waves[0]["host_available_bytes"]
    return dict(schema="splash-hybrid-v3-admission-command-boundary-cpu-analysis-v1",
                gpu_execution=False, model_payload_bytes_read=0,
                report_complete=True, server_unloaded=report["server_runs"][0]["unloaded"],
                initial_memory=initial["memory_actual"], initial_governor=initial_host,
                saved_residency=initial["saved_operands_residency"],
                original_text_residency=initial["original_text_residency"], waves=waves,
                warm_exact_decode_median_tps=statistics.median(w["native_exact_decode_tps"] for w in warm),
                warm_prefill_gpu_median_ms=median_gpu,
                warm_prefill_gpu_median_tps=2048*1000/median_gpu,
                warm_decode_observed_gpu_only_median_tps=statistics.median(w["unchanged_observed_decode_gpu_only_tps"] for w in warm),
                first_wave_host_available_drop_bytes=first_host_drop,
                existing_original_mapped_bytes=initial["ple_storage"]["gpu_mapped_original_bytes"],
                physical_wired_page_or_dram_counter_evidence_collected=False,
                performance_hypothesis="Original Q4 experts are excluded from the saved-only residency union. The first GPU wave consumes about73GB more host-estimated memory and large pre-GPU command-boundary gaps occur without GPU-compute changes. Phase-dependent driver rewiring is plausible, not proven.",
                admission_finding="After the first wave, host headroom is near/below the unchanged 1GiB growth threshold; repeated reservation refusals add1.4-1.7s queue delay to subsequent TTFT. Engine ledger headroom alone does not permit growth.",
                prefill_4000_target=dict(maximum_inclusive_ms=512,
                                         minimum_gpu_duration_reduction_fraction_ignoring_other_costs=1-512/median_gpu,
                                         last_warm_non_gpu_forward_ms=fixed_other,
                                         required_gpu_ms_if_last_other_costs_preserved=512-fixed_other,
                                         gpu_duration_reduction_fraction_if_last_other_costs_preserved=1-(512-fixed_other)/median_gpu,
                                         conclusion="Residency stabilization may remove cold boundary gaps but unchanged611ms GPU work cannot achieve4000tok/s."),
                provenance=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,default=ROOT/"build/release/flash/sep21-hybrid-q4-decode-i8-prefill-b1-v3.json")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.report.resolve())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("warm_exact_decode_median_tps","warm_prefill_gpu_median_ms","warm_prefill_gpu_median_tps","warm_decode_observed_gpu_only_median_tps","first_wave_host_available_drop_bytes","prefill_4000_target")}))


if __name__=="__main__":
    main()
