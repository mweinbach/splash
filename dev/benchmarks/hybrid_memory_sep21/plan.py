#!/usr/bin/env python3
"""Metadata-only Original-Q4 decode + Full512-I8 prefill feasibility ledger.

No model payload is opened, hashed, mapped, or loaded. This is a proposed
fixed-singleton/depth-3 cache selection, not a runtime or admission proof.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


ALIGNMENT = 16384


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded(n: int) -> int:
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def build(root: Path, source: Path) -> dict:
    model_path = root / "install/local-models/Flash-Next-oQ4e-mtp-v1/manifest.json"
    config_path = model_path.parent / "config.json"
    saved_path = root / "install/local-models/Flash-Next-operands-v1/manifest.json"
    full_path = root / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1/manifest.json"
    model = json.loads(model_path.read_text())
    config = json.loads(config_path.read_text())["text_config"]
    saved = json.loads(saved_path.read_text())
    full = json.loads(full_path.read_text())
    assert model["source_identity_sha256"] == saved["source_identity_sha256"] == full["source_identity_sha256"]
    assert config["num_hidden_layers"] == 48 and config["ple_layer_ids"] == [2]
    policy_path = source / "runtime/flash/FlashFloatDenseCache.cpp"
    forward_path = source / "runtime/flash/FlashForward.cpp"
    hc_path = source / "runtime/flash/FlashHCFused.cpp"
    descriptor_path = source / "runtime/flash/FlashDescriptor.mm"
    policy_text = policy_path.read_text()
    assert "rows < 4 || rows > 16" in hc_path.read_text()
    assert "must contain zero-based layer 1 only" in descriptor_path.read_text()
    forward_text = forward_path.read_text()
    assert 'layer == impl_->descriptor.pleLayerIndices.front()' in forward_text
    assert "const auto tile = flashFloatDenseSmallRowsPolicy" in forward_text
    assert "defaultPrefixes(weights, !" in forward_text
    pattern = r'\{"([^"\n]+)",\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\}'
    policy = {}
    for m in re.finditer(pattern, policy_text):
        role, *values = m.groups()
        n, k, bits, group, *tiles = map(int, values)
        policy[role, n, k, bits, group] = tiles
    assert len(policy) == 29
    entries = {e["projection"]: e for e in saved["entries"] if e["format"] == "F32"}
    assert len(entries) == 508 and "language_model.lm_head" not in entries
    all_bytes = sum(e["allocated_bytes"] for e in entries.values())
    assert all_bytes == 14391705600

    def inventory(rows: int) -> dict:
        selected = []
        column = 0 if rows < 8 else 1 if rows == 8 else 2 if rows == 16 else 3
        for name, e in sorted(entries.items()):
            p = e["source"]
            role = re.sub(r"^language_model\.model\.layers\.\d+\.", "", name)
            if name.endswith("input_mix_weight_up"):
                role = "hc_up"
            if role.startswith("ple.") and not name.startswith("language_model.model.layers.1."):
                continue
            tile = policy.get((role, p["output_size"], p["input_size"], p["bits"], p["group_size"]))
            if not tile or tile[column] < 0:
                continue
            q = model["quantization"].get(name, model["quantization"])
            assert (q["bits"], q["group_size"]) == (p["bits"], p["group_size"])
            t = model["tensors"]
            assert t[name + ".weight"]["shape"][-2] == p["output_size"]
            assert t[name + ".scales"]["shape"][-1] * p["group_size"] == p["input_size"]
            assert e["allocated_bytes"] == rounded(p["output_size"] * p["input_size"] * 4)
            kind = "hc_up" if role == "hc_up" else "qsa_output" if role == "self_attn.o_proj" else "generic_projection"
            selected.append(dict(projection=name, role=role, route=kind,
                                 allocated_bytes=e["allocated_bytes"], payload_file=e["file"],
                                 qsa_output_n32_eligible=role == "self_attn.o_proj" and p["bits"] in (5, 6) and p["group_size"] == 64,
                                 source=p))
        total = sum(e["allocated_bytes"] for e in selected)
        groups = {}
        for kind in ("hc_up", "qsa_output", "generic_projection"):
            group = [e for e in selected if e["route"] == kind]
            groups[kind] = dict(count=len(group), payload_bytes=sum(e["allocated_bytes"] for e in group))
        return dict(physical_rows=rows, count=len(selected), payload_bytes=total,
                    planned_bytes_including_persistent_diagnostics=total + ALIGNMENT,
                    pruned_count=len(entries) - len(selected), pruned_bytes=all_bytes - total,
                    by_route=groups, by_role=dict(collections.Counter(e["role"] for e in selected)),
                    selected=selected)

    inventories = {str(r): inventory(r) for r in (4, 8, 16)}
    assert inventories["4"]["by_route"]["generic_projection"]["count"] == 14
    assert inventories["4"]["by_route"]["hc_up"]["count"] == 97
    assert inventories["4"]["by_route"]["qsa_output"]["count"] == 7
    assert sum(e["qsa_output_n32_eligible"] for e in inventories["4"]["selected"]) == 5
    assert inventories["4"]["payload_bytes"] == 3247964160
    for lower, upper in (("4", "8"), ("8", "16")):
        assert {e["projection"] for e in inventories[lower]["selected"]} <= {e["projection"] for e in inventories[upper]["selected"]}
    physical = 274877906944
    reserve = max(16 << 30, physical // 10)
    limit = physical - reserve
    # These constants are independently source-derived and reconciled in the
    # retained CPU ledger; the workload peak is explicitly an extrapolation.
    baseline_peak = 120129617920
    top64_planned = 15147466752
    full_planned = full["total_bytes"] + 48 * ALIGNMENT
    assert full_planned == 121174228992
    full_peak_proxy = baseline_peak - top64_planned + full_planned
    wide_static_planner = 228729896960
    planned_wide_increase = 3176333312
    full_static_planner = wide_static_planner - planned_wide_increase
    request_bytes = 621969408
    historical_batch_planners = dict(
        batch_decode=18857984, batch_prefill=4351918080,
        owned_batch_prefill_hidden=167772160, batch_mtp_prime=288964608,
        owned_prime_input=10485760, joint_verify=522518528,
        joint_head=17154048, owned_joint_hidden=327680,
    )
    disabled_batch_planned = sum(historical_batch_planners.values())
    exact_sg8_bulk_planned = 234356736
    phase_policy = dict(
        implemented=False, target="private phase-dependent numerical derivative",
        source_identity_sha256=model["source_identity_sha256"],
        weights_manifest_fingerprint=saved["weights_manifest_fingerprint"],
        full512_manifest_sha256=sha(full_path),
        singleton_mtp_depth=3, singleton_target_verify_maximum_rows=4,
        main_nonverification_prefill="Full512 signed I8 / late F32 row scale only when rows >= 256; source BF16 activation and final rounding policies retained",
        main_small_prefill_and_standard_decode="original packed Q4 target experts at rows < 256",
        singleton_verification="original packed Q4 target experts, original selective F32 dense routes, all 97 original-coefficient F32 HC-up bridges at physical rows 4",
        batched_standard_and_verification="original packed Q4 experts; joint MTP lanes > 1 unsupported by the fixed-R4 cache policy",
        trained_mtp="original trained bank and existing projection policies",
        vocabulary="existing original Q8-code head policy; no new F32 vocabulary map",
        bf16_large_prefill_coefficients="existing saved BF16 cache retained",
        fixed_r4_f32_selection_sha256=hashlib.sha256("\n".join(e["projection"] for e in inventories["4"]["selected"]).encode()).hexdigest(),
        prefix_caching=False,
        parity_claim="none: the prefill state is derived from I8 coefficients and subsequent Q4 verification does not reproduce a universally original-Q4 model",
    )
    scenarios = {}
    for key, inv in inventories.items():
        pruned = inv["pruned_bytes"]
        scenarios[key] = dict(
            workload_peak_extrapolation_bytes=full_peak_proxy - pruned,
            conservative_static_planner_bytes=full_static_planner - pruned,
            conservative_one_eligible_request_bytes=full_static_planner - pruned + request_bytes,
            conservative_four_eligible_request_bytes=full_static_planner - pruned + 4 * request_bytes,
            four_request_engine_limit_headroom_before_driver_and_host_overhead=limit - (full_static_planner - pruned + 4 * request_bytes),
            fixed_no_batch_experiment_static_planner_including_exact_sg8_bulk=full_static_planner - pruned - disabled_batch_planned + exact_sg8_bulk_planned,
            fixed_no_batch_experiment_one_eligible_request_including_exact_sg8_bulk=full_static_planner - pruned - disabled_batch_planned + exact_sg8_bulk_planned + request_bytes,
            fixed_no_batch_experiment_four_cooperative_eligible_requests_including_exact_sg8_bulk=full_static_planner - pruned - disabled_batch_planned + exact_sg8_bulk_planned + 4 * request_bytes,
        )
    return dict(schema="splash-original-q4-decode-full512-prefill-f32-pruning-cpu-plan-v1",
                gpu_execution=False, model_loaded=False, model_payload_bytes_read=0,
                proposed_phase_policy=phase_policy,
                proposed_phase_policy_sha256=hashlib.sha256(json.dumps(phase_policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                physical_bytes=physical, host_reserve_bytes=reserve, engine_limit_bytes=limit,
                all_default_f32_count=len(entries), all_default_f32_payload_bytes=all_bytes,
                full512_payload_and_rank_bytes=full_planned,
                original_mapped_weight_bytes=74317889536,
                retained_bf16_payload_and_diagnostics_bytes=8467267584,
                retained_sequential_mtp_bf16_cache_and_diagnostics_bytes=178274304,
                persistent_f32_padding_bytes=1048576,
                inventories=inventories, scenarios=scenarios,
                peak_provenance=dict(retained_top64_peak_bytes=baseline_peak,
                                     removed_top64_planned_bytes=top64_planned,
                                     original_plus_full512_workload_peak_extrapolation_bytes=full_peak_proxy,
                                     conservative_full512_static_planner_at_2048_rows=full_static_planner,
                                     eligible_request_bytes=request_bytes),
                fixed_experiment_planner_adjustments=dict(disabled_historical_batch_categories=historical_batch_planners,
                                                          disabled_batch_planned_bytes=disabled_batch_planned,
                                                          added_exact_sg8_bulk_planned_bytes=exact_sg8_bulk_planned,
                                                          claim="source-derived arithmetic; Root compiled CPU planner remains authoritative"),
                host_admission=dict(required_normal_pressure=True,
                                    next_stage_requires_headroom_after_reserve_bytes=1 << 30,
                                    previous_failure_available_bytes_approximate=19760000000,
                                    fixed_r4_projected_available_bytes_if_every_pruned_byte_releases_host_used_memory=19760000000 + inventories["4"]["pruned_bytes"],
                                    claim="arithmetic plausibility only; file-backed reclaim accounting and driver/CPU memory make fresh phased governor checks mandatory"),
                alias_and_constructor_accounting=[
                    "Every F32 selected mmap is verified in place and becomes its final no-copy Metal backing; no second tensor-sized copy.",
                    "F32 diagnostics and padding stay permanent and are included in the cache/Forward planner respectively.",
                    "Unused saved metadata entries may remain; changing constructor names and matching planner names is what removes maps.",
                    "Full512 plane views and rank/source aliases do not add backing; original Q4 weights remain mapped once for decode.",
                    "All sequential/batch/joint arenas and owned feature staging in the retained planner remain charged.",
                    "Saved residency registration refers to already charged selected backing and is not an extra allocation.",
                    "Pipeline/driver and CPU metadata overhead remain outside exact coefficient arithmetic.",
                ],
                provenance=[dict(path=str(p), sha256=sha(p)) for p in (model_path, config_path, saved_path, full_path, policy_path, forward_path, hc_path, descriptor_path)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source", type=Path, default=Path("build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1/source"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = args.source if args.source.is_absolute() else root / args.source
    result = build(root, source.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(out=str(args.out), gpu_execution=False, model_payload_bytes_read=0,
                          fixed_r4_payload_bytes=result["inventories"]["4"]["payload_bytes"],
                          pruned_bytes=result["inventories"]["4"]["pruned_bytes"],
                          fixed_r4_scenarios=result["scenarios"]["4"])))


if __name__ == "__main__":
    main()
