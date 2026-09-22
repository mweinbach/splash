#!/usr/bin/env python3
"""CPU-only combined Full512/wide ledger estimate and live host-stage check."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT = 16384


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/prefill4k-wide-fullcache")
    parser.add_argument("--wide-evidence", type=Path, default=ROOT / "build/release/flash/prefill4k-wide-chunk-grid.json")
    parser.add_argument("--fullcache-preflight", type=Path, default=ROOT / "build/release/flash/prefill4k-fullcache-preflight512.json")
    parser.add_argument("--profile", type=Path, default=ROOT / ".splash-local-profile.json")
    parser.add_argument("--requests", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--rows", type=int, choices=[4096, 8192], default=8192)
    parser.add_argument("--engine-limit-bytes", type=int, default=247390116250)
    parser.add_argument("--margin-bytes", type=int, default=2 << 30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a fresh combined ledger report")
    require(args.engine_limit_bytes > 0 and args.margin_bytes >= 0, "Invalid ledger bounds")
    evidence = json.loads(args.wide_evidence.read_text())
    full = json.loads(args.fullcache_preflight.read_text())
    profile = json.loads(args.profile.read_text())
    require(evidence["completed"], "Wide geometry evidence is incomplete")
    initial = evidence["initial_status"][str(args.rows)]
    final = evidence["final_status"][str(args.rows)]
    capacity = final["maximum_context_tokens"]
    require(final["scheduler"]["maximum_prefill_rows"] == args.rows, "Wide arena evidence differs")
    require(initial["maximum_context_tokens"] == capacity and initial["scheduler"]["maximum_prefill_rows"] == args.rows, "Initial/final wide geometry differs")
    require(initial["memory_audit"]["valid"] and final["memory_audit"]["valid"], "Wide reference allocation ledger is invalid")
    for key in ["identity", "persisted_experts", "persisted_operands"]:
        for field in ({"identity": ["source", "loaded_model_layout_sha256", "kernel_routes"],
                       "persisted_experts": ["enabled", "store_manifest_sha256", "selection_plan_sha256", "mapped_bytes", "expert_count"],
                       "persisted_operands": ["store_manifest_sha256", "bf16_mapped_payload_bytes", "f32_mapped_payload_bytes"]}[key]):
            require(initial[key][field] == final[key][field], f"Initial/final reference identity differs: {key}.{field}")
    require(final["ple_storage"]["ssd_streaming_enabled"], "Full512 extrapolation requires PLE SSD storage")
    require(final["scheduler"]["active_requests"] == 0 and not final["scheduler"]["command_in_flight"], "Wide evidence must end idle")
    old = final["persisted_experts"]
    require(old["enabled"] and old["expert_count"] == 48 * 64, "Reference must load exactly Top64")
    require(old["store_manifest_sha256"] == full["preserved_top64_manifest_sha256"], "Reference cache identity differs")
    require(final["identity"]["source"] == full["source_identity_sha256"] == profile["source_identity_sha256"], "Source identity differs")
    require(full["planned_allocation_bytes"] == 121174228992 and full["total_bytes"] == 121173442560, "Full512 geometry differs")
    flags = profile["environment"]
    require(flags.get("SPLASH_FLASH_MTP_DRAFT_DEPTH") == "3", "Recalculate verification/fold ledger for changed MTP depth")
    require(flags.get("SPLASH_FLASH_BATCH_PREFILL_ROWS") == "2048", "Recalculate batch geometry ledger for changed lane rows")
    for flag in [
        "SPLASH_FLASH_DENSE_CACHE", "SPLASH_FLASH_FLOAT_DENSE_CACHE",
        "SPLASH_FLASH_INT8_HEAD", "SPLASH_FLASH_QSA_F32", "SPLASH_FLASH_QSA_MPP",
        "SPLASH_FLASH_BLOCKED_MOE", "SPLASH_FLASH_MOE_DIRECT_A",
        "SPLASH_FLASH_PLE_SSD_STREAMING", "SPLASH_FLASH_GDN_LAZY_ROLLBACK",
        "SPLASH_FLASH_GPU_GREEDY",
        "SPLASH_FLASH_MTP", "SPLASH_FLASH_BATCH", "SPLASH_FLASH_BATCH_PREFILL",
        "SPLASH_FLASH_BATCH_MTP", "SPLASH_FLASH_BATCH_MTP_PREFILL",
        "SPLASH_FLASH_MTP_QSA_F32", "SPLASH_FLASH_MTP_QSA_MPP",
    ]:
        require(flags.get(flag) == "1", f"Recalculate ledger for changed flag {flag}")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SPLASH_FLASH_")}
    environment.update(flags)
    probe_path = args.build / "memory-cpu"
    probe = json.loads(subprocess.check_output([str(probe_path), str(capacity), str(args.rows)], env=environment, text=True, cwd=ROOT))
    require(probe["gpu_work"] is False and probe["model_loaded"] is False, "Host probe must remain CPU only")
    require(probe["host_measurement_valid"], "Missing live host measurement")
    require(probe["engine_limit_bytes"] >= args.engine_limit_bytes, "Requested engine bound exceeds physical/reserve policy")
    operands = final["persisted_operands"]
    require(operands["bf16_tensors"] == 509 and operands["f32_tensors"] == 508, "Recalculate immutable operand inventory")
    old_planned = old["mapped_bytes"] + 48 * ALIGNMENT
    replacement_delta = full["planned_allocation_bytes"] - old_planned
    measured_request_and_transient = max(0, final["memory_actual"]["peak_bytes"] - initial["memory_actual"]["current_bytes"])
    audit = final["memory_audit"]
    require(probe["target_request_state_bytes"] == audit["state_bytes_per_request"], "Compiled target state plan differs from wide evidence")
    require(probe["mtp_request_state_bytes"] + probe["fixed_depth3_mtp_committed_hidden_bytes"] == audit["mtp_extra_state_bytes_per_eligible_request"], "Compiled MTP state/fold plan differs from wide evidence")
    request_bytes = audit["state_bytes_per_request"] + audit["mtp_extra_state_bytes_per_eligible_request"]
    per_request_upper = max(request_bytes, measured_request_and_transient)
    projected_startup = initial["memory_actual"]["current_bytes"] + replacement_delta
    projected_peak = projected_startup + args.requests * per_request_upper
    # Match current Worker plannedTrunk: zero-copy Q8 head reuses F32 padding,
    # each saved BF16/F32 cache has one 16KiB diagnostic allocation.
    trunk_terms = {
        "compiled_forward_row_upper_bytes": probe["trunk_row_workspace_planned_bytes"],
        "full512_payload_and_rank_bytes": full["planned_allocation_bytes"],
        "saved_f32_cache_planned_bytes": operands["f32_mapped_payload_bytes"] + ALIGNMENT,
        "saved_bf16_cache_planned_bytes": operands["bf16_mapped_payload_bytes"] + ALIGNMENT,
        "zero_copy_q8_head_extra_bytes": 2 * ALIGNMENT,
        "worker_extra_qsa_upper_bytes": min(args.rows, 128) * 24 * 4 * (256 + 2) * 4 + 2 * ALIGNMENT,
        "compiled_blocked_moe_upper_bytes": probe["blocked_moe_workspace_planned_bytes"],
    }
    planned_trunk = sum(trunk_terms.values())
    complete_static_terms = {
        "original_weight_maps_bytes": audit["weight_bytes"],
        "planned_trunk_bytes": planned_trunk,
        "sequential_mtp_workspace_bytes": probe["sequential_mtp_workspace_planned_bytes"],
        "sequential_mtp_generated_bf16_cache_bytes": 178274304,
        "batch_decode_workspace_bytes": probe["batch_decode_workspace_planned_bytes"],
        "batch_prefill_workspace_bytes": probe["batch_prefill_workspace_planned_bytes"],
        "batch_prefill_owned_target_hidden_bytes": 4 * 2048 * 10240 * 2,
        "batch_teacher_prime_workspace_bytes": probe["batch_teacher_prime_workspace_planned_bytes"],
        "batch_teacher_prime_owned_input_bytes": 4 * 128 * 10240 * 2,
        "joint_verifier_workspace_bytes": probe["joint_verifier_workspace_planned_bytes"],
        "joint_head_shared_vocabulary_workspace_bytes": probe["joint_head_workspace_planned_bytes_shared_vocabulary"],
        "joint_head_owned_target_hidden_bytes": 4 * 4 * 10240 * 2,
    }
    static_plan = sum(complete_static_terms.values())
    complete_plan_with_states = static_plan + args.requests * request_bytes
    assessed_peak = max(projected_peak, complete_plan_with_states)
    engine_fits = assessed_peak + args.margin_bytes <= args.engine_limit_bytes
    # This is the largest pending stage tested against the CURRENT unloaded
    # sampler. Original weight mapping changes availability before this stage;
    # private Worker then records a fresh snapshot and tryReserve enforces it.
    current_stage_host_fits = planned_trunk + args.margin_bytes <= probe["host_headroom_bytes"]
    full_residency_bound_fits = assessed_peak + args.margin_bytes <= probe["host_headroom_bytes"]
    result = {
        "schema": 1, "gpu_work": False, "model_loaded": False,
        "arithmetic_change": True, "capacity": capacity, "maximum_singleton_rows": args.rows,
        "maximum_request_states": args.requests, "engine_limit_bytes": args.engine_limit_bytes,
        "experimental_margin_bytes": args.margin_bytes, "old_top64_planned_bytes": old_planned,
        "full512_planned_bytes": full["planned_allocation_bytes"], "cache_replacement_delta_bytes": replacement_delta,
        "measured_wide_top64_startup_bytes": initial["memory_actual"]["current_bytes"],
        "measured_wide_top64_peak_bytes": final["memory_actual"]["peak_bytes"],
        "request_state_bytes_per_lane": request_bytes, "per_request_state_and_observed_transient_allowance_bytes": per_request_upper,
        "request_peak_scope": "exact state bytes plus measured historical transient allowance; workload-specific estimate, not a guaranteed peak bound",
        "candidate_startup_ledger_estimate_bytes": projected_startup,
        "candidate_peak_ledger_estimate_bytes": projected_peak,
        "estimated_engine_headroom_bytes": args.engine_limit_bytes - projected_peak,
        "complete_static_planned_terms": complete_static_terms,
        "complete_static_planned_bytes": static_plan,
        "complete_planned_bytes_with_request_states": complete_plan_with_states,
        "assessed_engine_bytes_before_margin": assessed_peak,
        "engine_headroom_after_complete_plan_bytes": args.engine_limit_bytes - complete_plan_with_states,
        "complete_plan_scope": "compiled static planners plus source-qualified generated teacher-cache geometry; excludes optional idle diagnostic allocation and driver/pipeline allowance",
        "idle_maintenance_requirement": "set SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=0 for combined paired tests",
        "planned_trunk_terms": trunk_terms, "planned_trunk_bytes": planned_trunk,
        "current_live_host_probe": probe, "current_live_host_fits_largest_stage_with_margin": current_stage_host_fits,
        "physical_full_residency_upper_bound_fits_current_host_headroom": full_residency_bound_fits,
        "full_residency_bound_scope": "conservative whole-ledger bound; does not imply residency, pinning, or actual phased host admission",
        "stage_scope": "current unloaded host sample only; original mappings precede trunk reservation and require a fresh private Worker snapshot",
        "runtime_required": "set SPLASH_FLASH_PRIVATE_ADMISSION_REPORT to a fresh path; require actual startup tryReserve and later state reservations to pass",
        "residency_union_double_charged": False, "disk_minimum_free_bytes_used_as_ram": False,
        "engine_bound_fits_with_margin": engine_fits,
        "ready_for_runtime_admission_attempt": engine_fits and current_stage_host_fits,
        "evidence_sha256": {str(path): sha(path) for path in [args.wide_evidence, args.fullcache_preflight, args.profile, probe_path]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ["candidate_peak_ledger_estimate_bytes", "estimated_engine_headroom_bytes", "planned_trunk_bytes", "current_live_host_fits_largest_stage_with_margin", "ready_for_runtime_admission_attempt"]}))
    require(engine_fits, "Combined candidate exceeds the engine budget/margin; reject before startup")
    require(current_stage_host_fits, "Combined pending trunk stage exceeds current measured host reserve/margin; reject before startup")


if __name__ == "__main__":
    main()
