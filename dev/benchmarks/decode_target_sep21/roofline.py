#!/usr/bin/env python3
"""CPU-only Flash-Next logical-payload roofs and measured-phase targets.

Reads JSON metadata, source policy, and retained small reports. It never opens
model payload files, constructs a model, or creates a Metal device. Decimal GB/s.
These are traffic-model ceilings, not a promise of attainable generation speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(repo: Path, bandwidth: float, temporal_unique: float | None) -> dict:
    package = repo / "install/local-models/Flash-Next-oQ4e-mtp-v1"
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((package / "config.json").read_text())["text_config"]
    tensors = manifest["tensors"]
    experts, topk = config["num_experts"], config["num_experts_per_tok"]
    quant = manifest["quantization"]
    category = {k: 0.0 for k in (
        "target_dense", "target_experts_all", "target_vocab",
        "mtp_dense", "mtp_experts_all", "ple_table", "embedding", "unused")}
    full512_path = repo / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1/manifest.json"
    full512 = json.loads(full512_path.read_text())
    full512_planes = {}
    for layer in full512["layers"]:
        for role, projection in layer["projections"].items():
            full512_planes[role] = full512_planes.get(role, 0) + projection["codes"]["length"] + projection["scales"]["length"]
    full512_expert_bytes = sum(full512_planes.values())
    weights = {}
    for name, tensor in tensors.items():
        size = tensor["length"]
        if name.startswith("language_model.lm_head."):
            category["target_vocab"] += size
        elif name.startswith("language_model.model.embed_tokens."):
            category["embedding"] += size
        elif "ngram_embedding" in name:
            category["ple_table"] += size
        elif name.startswith("language_model.model.layers.") or name.startswith("language_model.model.hyper_connection_mixer."):
            key = "target_experts_all" if ".switch_mlp." in name else "target_dense"
            category[key] += size
        elif name.startswith("mtp."):
            key = "mtp_experts_all" if ".switch_mlp." in name else "mtp_dense"
            category[key] += size
        else:
            category["unused"] += size
        if name.endswith(".weight") and tensor["dtype"] == "U32" and "ngram_embedding" not in name:
            prefix = name[:-7]
            q = quant.get(prefix, quant)
            scales = tensors[prefix + ".scales"]
            outputs = tensor["shape"][-2]
            inputs = scales["shape"][-1] * q["group_size"]
            count = tensor["shape"][0] if len(tensor["shape"]) == 3 else 1
            packed = tensor["length"] + scales["length"] + tensors[prefix + ".biases"]["length"]
            weights[prefix] = dict(outputs=outputs, inputs=inputs, count=count,
                                   bits=q["bits"], group=q["group_size"], packed=packed)

    # Extract the production selective F32 matrix policy, not model payloads.
    source_path = repo / "runtime/flash/FlashFloatDenseCache.cpp"
    source = source_path.read_text()
    policy = {}
    pattern = r'\{"([^"\n]+)",\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\}'
    for match in re.finditer(pattern, source):
        role, *values = match.groups()
        n, k, bits, group, r4, r8, r16, middle = map(int, values)
        policy[role, n, k, bits, group] = r4, r8, r16, middle

    def expanded_delta(rows: int, include_hc_up: bool = True) -> int:
        # A logical unique operand footprint for the existing <=16-row graph.
        # It omits repeated shader loads, cache hits, transactions, and scratch.
        delta = 0
        for name, p in weights.items():
            if not name.startswith("language_model.model.") or p["count"] != 1:
                continue
            role = re.sub(r"^language_model\.model\.layers\.\d+\.", "", name)
            if name.endswith("input_mix_weight_up"):
                if not include_hc_up:
                    continue  # Stock BatchForward fused HC uses original rawup.
                role = "hc_up"
            entry = policy.get((role, p["outputs"], p["inputs"], p["bits"], p["group"]))
            if entry and 4 <= rows <= 16:
                idx = 0 if rows < 8 else 1 if rows == 8 else 2 if rows == 16 else 3
                if entry[idx] >= 0:
                    delta += 4 * p["outputs"] * p["inputs"] - p["packed"]
        return delta

    route_path = repo / "build/release/flash/v6-verify-ctx2048-singleton-R16-stage.json.routes.jsonl"
    route = json.loads(route_path.read_text().splitlines()[0])["routes"]
    unions, overlaps = [], []
    for layer in route["layers"]:
        ids = layer["row_major_expert_ids"]
        rows = [set(ids[i:i+topk]) for i in range(0, len(ids), topk)]
        unions.extend(len(set.union(*rows[i:i+4])) for i in range(0, len(rows)-3, 4))
        overlaps.extend(len(a & b) / topk for a, b in zip(rows, rows[1:]))
    observed_temporal_unique = statistics.mean(unions)
    temporal_unique = temporal_unique if temporal_unique is not None else observed_temporal_unique
    temporal_by_rows = {}
    for count in range(1, 17):
        observed = []
        for layer in route["layers"]:
            ids = layer["row_major_expert_ids"]
            row_sets = [set(ids[i:i+topk]) for i in range(0, len(ids), topk)]
            observed.extend(len(set.union(*row_sets[i:i+count])) for i in range(len(row_sets)-count+1))
        temporal_by_rows[count] = statistics.mean(observed)

    # Standard state read+write floor. One recurrent state per sequence is
    # 36*48*128*128*4 bytes. QSA keys+values up to selected 2048 tokens per layer.
    recurrent = 36 * 48 * 128 * 128 * 4
    qsa = 12 * min(2048, config["indexer_budget"]) * 512 * 2 * 2
    ar_state = 2 * recurrent + qsa
    mtp_state = 4 * recurrent + qsa  # includes lazy rollback initial snapshot
    standard_single = category["target_dense"] + category["target_vocab"] + category["target_experts_all"] * topk / experts
    head_single = category["mtp_dense"] + category["target_vocab"] + category["mtp_experts_all"] * topk / experts
    per_expert = category["target_experts_all"] / experts
    deep_scalar = []
    for depth in range(1, 16):
        rows = depth + 1
        # Full commitment creates rows true fold pairs, in chunks <=8. Only
        # the final fold chunk computes vocabulary logits. Consecutive head
        # folds optimistically share ten selected experts within each chunk.
        fold_chunks = (rows + 7) // 8
        head_traffic = depth * head_single + (fold_chunks - 1) * (head_single - category["target_vocab"])
        verify_q4 = category["target_dense"] + category["target_vocab"] + per_expert * temporal_by_rows[rows]
        verify_i8 = category["target_dense"] + category["target_vocab"] + full512_expert_bytes / experts * temporal_by_rows[rows]
        original_current_bytes = verify_q4 + head_traffic + mtp_state + expanded_delta(rows)
        full_current_bytes = verify_i8 + head_traffic + mtp_state + expanded_delta(rows)
        deep_scalar.append(dict(depth=depth, ideal_committed_tokens=rows, fold_chunks=fold_chunks,
            observed_single_case_target_unique_experts=temporal_by_rows[rows],
            original_q4_existing_cache_perfect_prefix_proxy_tps=rows * bandwidth * 1e9 / original_current_bytes,
            full512_i8_existing_cache_perfect_prefix_proxy_tps=rows * bandwidth * 1e9 / full_current_bytes,
            full512_i8_existing_cache_perfect_prefix_spec_tps=rows * 1200e9 / full_current_bytes,
            full512_i8_cycle_GB=full_current_bytes / 1e9))

    def expert_union(lanes: int, unique_per_lane: float) -> float:
        # Independent lanes with uniform expert frequency; every lane's own
        # temporal expert set may have measured correlation.
        return experts * (1 - (1 - unique_per_lane / experts) ** lanes)

    matrix = []
    for batch in (1, 2, 4, 8, 16):
        ar_weights = category["target_dense"] + category["target_vocab"] + per_expert * expert_union(batch, topk)
        ar_bytes = ar_weights + batch * ar_state
        verify_weights = category["target_dense"] + category["target_vocab"] + per_expert * expert_union(batch, temporal_unique)
        head_weights = category["mtp_dense"] + category["target_vocab"] + category["mtp_experts_all"] / experts * expert_union(batch, topk)
        cycle_bytes = verify_weights + 3 * head_weights + batch * mtp_state
        # Native executor has <=4 lanes. Independent groups do not share one
        # guaranteed weight-reuse tile. Actual caches could retain some weights.
        group = min(batch, 4)
        group_ar = category["target_dense"] + category["target_vocab"] + per_expert * expert_union(group, topk) + group * ar_state
        group_verify = category["target_dense"] + category["target_vocab"] + per_expert * expert_union(group, temporal_unique)
        group_head = category["mtp_dense"] + category["target_vocab"] + category["mtp_experts_all"] / experts * expert_union(group, topk)
        group_mtp = group_verify + 3 * group_head + group * mtp_state
        full_ar_bytes = category["target_dense"] + category["target_vocab"] + full512_expert_bytes / experts * expert_union(batch, topk) + batch * ar_state
        full_group_ar = category["target_dense"] + category["target_vocab"] + full512_expert_bytes / experts * expert_union(group, topk) + group * ar_state + expanded_delta(group, include_hc_up=False)
        full_verify = category["target_dense"] + category["target_vocab"] + full512_expert_bytes / experts * expert_union(group, temporal_unique)
        full_group_mtp = full_verify + 3 * group_head + group * mtp_state + expanded_delta(group * 4)
        matrix.append(dict(batch=batch,
            standard_cycle_GB=ar_bytes / 1e9,
            mtp_depth3_cycle_GB=cycle_bytes / 1e9,
            standard_spec_aggregate_tps=batch * 1200e9 / ar_bytes,
            standard_proxy_aggregate_tps=batch * bandwidth * 1e9 / ar_bytes,
            mtp_depth3_coding_prefix3_spec_aggregate_tps=batch * 3 * 1200e9 / cycle_bytes,
            mtp_depth3_coding_prefix3_proxy_aggregate_tps=batch * 3 * bandwidth * 1e9 / cycle_bytes,
            mtp_depth3_full_prefix4_spec_aggregate_tps=batch * 4 * 1200e9 / cycle_bytes,
            mtp_depth3_full_prefix4_proxy_aggregate_tps=batch * 4 * bandwidth * 1e9 / cycle_bytes,
            native_four_lane_group_standard_proxy_tps=group * bandwidth * 1e9 / group_ar,
            native_four_lane_group_mtp_coding_proxy_tps=group * 3 * bandwidth * 1e9 / group_mtp,
            target_existing_f32_cache_delta_GB=expanded_delta(min(batch * 4, 16)) / 1e9,
            native_four_lane_group_existing_cache_mtp_coding_proxy_tps=group * 3 * bandwidth * 1e9 / (group_mtp + expanded_delta(group * 4)),
            native_four_lane_group_existing_cache_mtp_full_proxy_tps=group * 4 * bandwidth * 1e9 / (group_mtp + expanded_delta(group * 4)),
            full512_standard_spec_ideal_aggregate_tps=batch * 1200e9 / full_ar_bytes,
            full512_standard_proxy_ideal_aggregate_tps=batch * bandwidth * 1e9 / full_ar_bytes,
            full512_native_group4_existing_cache_standard_proxy_tps=group * bandwidth * 1e9 / full_group_ar,
            full512_native_group4_existing_cache_mtp_coding_prefix3_1875_proxy_tps=group * 3.1875 * bandwidth * 1e9 / full_group_mtp,
            full512_native_group4_existing_cache_mtp_full_proxy_tps=group * 4 * bandwidth * 1e9 / full_group_mtp,
            full512_native_group4_mtp_existing_cycle_GB=full_group_mtp / 1e9,
        ))

    trace_path = repo / "build/release/flash/ultra-decode-cpu-audit.json"
    trace = json.loads(trace_path.read_text())["baseline_trace"]
    phases = {p["phase"]: p for p in trace["phases"]}
    cycles = phases["target_verify"]["calls"]
    verify_ms = phases["target_verify"]["gpu_ms"] / cycles
    other_ms = sum(phases[k]["gpu_ms"] for k in ("committed_head_fold", "draft_head_chain", "target_prefix_restore")) / cycles
    wall_boundary_ms = sum(phases[k]["command_wall_ms"] - phases[k]["gpu_ms"] for k in ("committed_head_fold", "draft_head_chain", "target_verify", "target_prefix_restore")) / cycles
    normal_path = repo / "build/release/flash/ultra-locality-status-qsa.json"
    normal = json.loads(normal_path.read_text())
    tasks = {}
    for name in ("code", "count", "prose"):
        records = [r for r in normal["records"] if r["name"] == name and not r["warmup"]]
        tasks[name] = dict(
            native_stream_median_tps=statistics.median(r["record"]["native_request_metrics"]["request_latency"]["stream_tokens_per_second"] for r in records),
            mean_committed_tokens_per_cycle=statistics.mean((r["mtp"]["verification_cycles"] + r["mtp"]["accepted_committed_drafts"]) / r["mtp"]["verification_cycles"] for r in records),
            aggregate_proposal_acceptance=statistics.mean(r["mtp"]["accepted_committed_drafts"] / r["mtp"]["drafted_tokens"] for r in records))
    measured_targets = []
    # Include current measured boundary floor; untraced end-to-end status,
    # encoding, transport, and restore patterns will still affect actual speed.
    for speedup in (1, 1.5, 2, 3, 4):
        cycle_ms = verify_ms / speedup + other_ms + wall_boundary_ms
        measured_targets.append(dict(verifier_speedup=speedup,
            coding_prefix3_tps=3000 / cycle_ms, full_prefix4_tps=4000 / cycle_ms,
            target_verify_gpu_ms=verify_ms / speedup, cycle_ms=cycle_ms))

    primitives = []
    primitive_paths = [repo / "build/release/flash" / name for name in (
        "prefill4k-direct-gathered-mpp-r1-mixed-v1.json",
        "prefill4k-direct-gathered-mpp-r2-first.json",
        "prefill4k-direct-gathered-mpp-r4-repeated-v1.json",
        "prefill4k-direct-gathered-mpp-r16-repeated-v1.json")]
    for path in primitive_paths:
        primitive = json.loads(path.read_text())
        rows = primitive["rows"]
        flop = rows * topk * 3 * 2 * config["hidden_size"] * config["moe_intermediate_size"]
        medians = {t["variant"]: statistics.median(s["gpu_ms"] for s in t["samples"]) for t in primitive["timings"]}
        primitives.append(dict(path=str(path.relative_to(repo)), rows=rows,
            route_pattern=primitive["route_pattern"], input_scope=primitive["hidden_policy"],
            exact_old_mpp_parity_pass=primitive["exact_old_mpp_parity_pass"],
            frozen_F64_strict_pass=primitive["candidate_f64_strict_pass"],
            layer0_gpu_median_ms=medians,
            quantized_linear_equivalent_GFLOP=flop / 1e9,
            effective_chain_TFLOPs={k: flop / (v * 1e9) for k,v in medians.items()},
            all48_layers_same_timing_moe_only_ms={k: 48*v for k,v in medians.items()},
            caveat="one warm synthetic layer; family-equivalent FLOPs only; cannot infer GPU-wide BF16 tensor peak or actual whole-model ceiling"))
    full_normal_path = repo / "build/release/flash/ultra-locality-prefill4k-allrows-full512-first.json"
    full_normal = json.loads(full_normal_path.read_text())
    full_records = [r for r in full_normal["records"] if not r["warmup"]]
    full_cycle_ms = statistics.median(r["native_delta"]["metrics"]["decode_wall_ms"] / r["mtp"]["verification_cycles"] for r in full_records)
    full_prefix = statistics.mean((r["mtp"]["verification_cycles"] + r["mtp"]["accepted_committed_drafts"]) / r["mtp"]["verification_cycles"] for r in full_records)
    r4 = next(p for p in primitives if p["rows"] == 4)
    moe_saving = 48 * (r4["layer0_gpu_median_ms"]["oldMPP"] - r4["layer0_gpu_median_ms"]["gatheredMPP"])

    return dict(schema="splash-flash-next-decode-payload-roofline-sep21-v1",
        gpu_execution=False, model_payload_files_opened=False,
        apple_spec=dict(memory_bandwidth_GBps=1200, url="https://www.apple.com/mac-studio/specs/", checked="2026-09-21"),
        bandwidth_proxy=dict(effective_operand_payload_GBps=bandwidth,
            physical_sustained_DRAM_bandwidth_measured=False,
            source="dev/benchmarks/flash-head-q8-r1-v8.md",
            definition="675430400 original Q8 code+scale+bias bytes / 0.8 ms retained vocabulary primitive; warm caches and repeated transactions unresolved"),
        source_identity=manifest["source_identity_sha256"],
        metadata=dict(layers=config["num_hidden_layers"], experts=experts, selected_experts=topk,
            manifest_total_logical_GB=sum(t["length"] for t in tensors.values()) / 1e9,
            packed_categories_GB={k:v/1e9 for k,v in category.items()},
            target_single_active_packed_GB=standard_single/1e9, head_single_active_packed_GB=head_single/1e9,
            target_single_weight_only_standard_spec_tps=1200e9/standard_single,
            target_single_weight_only_standard_proxy_tps=bandwidth*1e9/standard_single,
            standard_minimum_state_GB=ar_state/1e9, mtp_depth3_minimum_state_GB=mtp_state/1e9,
            target_active_quantized_projection_GFLOP=sum(2*p["outputs"]*p["inputs"]*(topk if p["count"]>1 else 1) for n,p in weights.items() if (n.startswith("language_model.model.") and "embed_tokens" not in n) or n=="language_model.lm_head") / 1e9,
            selected_F32_cache_deltas_GB={str(r):expanded_delta(r)/1e9 for r in (1,2,4,8,16)}),
        full512_metadata=dict(logical_store_GB=full512_expert_bytes / 1e9,
            logical_projection_planes_GB={k:v/1e9 for k,v in full512_planes.items()},
            selected10_all48layers_GB=full512_expert_bytes * topk / experts / 1e9,
            target_single_active_compact_GB=(category["target_dense"] + category["target_vocab"] + full512_expert_bytes * topk / experts) / 1e9,
            original_trained_mtp_unchanged=True,
            route_scope="Full512 I8 numerical derivative, existing target selective F32 dense routes inventory separate from packed Q4 ideal"),
        retained_route_calibration=dict(scope="one diagnostic 16-row speculative window, not a normal depth3 route population",
            layer_four_row_windows=len(unions), observed_unique_experts_per_four_rows=observed_temporal_unique,
            observed_neighbor_overlap_fraction=statistics.mean(overlaps), modeled_unique_experts_per_four_rows=temporal_unique),
        traffic_roofs=matrix,
        scalar_depth1_to15_perfect_acceptance_scenarios=deep_scalar,
        measured_normal_depth3=tasks,
        measured_trace_cycle=dict(scope="one diagnostic warmed coding trace; GPU times distinct from untraced HTTP",
            committed_tokens_per_cycle=3, verify_gpu_ms=verify_ms, head_and_restore_gpu_ms=other_ms,
            command_boundary_ms=wall_boundary_ms, gpu_verify_fraction=verify_ms/(verify_ms+other_ms)),
        measured_phase_targets=measured_targets,
        measured_moe_primitive_rates=primitives,
        full512_measured_conditional_target=dict(
            source_report=str(full_normal_path.relative_to(repo)),
            median_decode_tps=statistics.median(r["native_delta"]["derived"]["decode_tokens_per_second"] for r in full_records),
            committed_prefix_per_cycle=full_prefix, measured_cycle_ms=full_cycle_ms,
            extrapolated_layer0_r4_chain_saving_all48layers_ms=moe_saving,
            prediction_if_every_layer_reproduces_synthetic_r4_gain_tps=full_prefix * 1000 / (full_cycle_ms - moe_saving),
            caveat="conditional measured-kernel extrapolation; mixed actual routes, late-layer cache state and all other kernels can lower gain; requires whole-worker measurement"),
        assumptions=[
            "All rates are aggregate tokens/s. Per-lane rate divides aggregate by active batch.",
            "Ideal compact route retains original packed quantized matrices and reuses each dense/vocabulary operand once per graph; existing shader work can reread them.",
            "Experts are independently uniformly distributed across lanes. Four-row temporal union comes from one retained diagnostic route population and can be overridden.",
            "Depth3 uses one four-row target verification plus three head calls per cycle: one committed-feature fold and two additional proposals.",
            "MTP output numerator uses actual prefix length, not aggregate independent acceptance: coding3 and ideal full4 tokens/cycle. Prose observed2.44.",
            "Standard recurrent state has read+write once; MTP also has an initial lazy rollback snapshot. Cache/state transaction amplification and all activations, SSD preparation, partial recurrence replay, dispatch, host and transport costs are omitted.",
            "Native batch executors currently cap four lanes. Batch8/16 ideal common-graph roofs need architecture work; grouped4 roofs assume no guaranteed reuse across separate groups.",
            "Existing F32 cache adds logical operand bytes. Delta is source-policy inventory, not measured physical traffic or exact shader reread count.",
            "Apple does not publish a directly useful BF16/affine kernel throughput spec here; no unsupported absolute compute ceiling is invented. A caller can add a measured compute constraint.",
            "No traffic roof is a demonstrated attainable generation target. Measured phase targets state explicit verifier speedup requirements and preserve observed acceptance.",
            "Scalar depths1..15 scenarios assume perfect acceptance, speculative-window target expert unions and optimistic within-fold trained-head expert sharing. Deep target verification numerical behavior is unqualified; old depth15/depth3 generations differed. Peers cap depth3.",
        ],
        provenance=[dict(path=str(p.relative_to(repo)),sha256=digest(p)) for p in (manifest_path,source_path,route_path,trace_path,normal_path,full512_path,full_normal_path,*primitive_paths)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--payload-gbps", type=float, default=844.288)
    parser.add_argument("--bandwidth-report", type=Path,
                        help="validated standalone resident large-buffer report; use lower of8/16GiB read medians")
    parser.add_argument("--temporal-unique", type=float)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.payload_gbps <= 0 or args.temporal_unique is not None and not 10 <= args.temporal_unique <= 40:
        parser.error("positive bandwidth and temporal unique in10..40 required")
    calibration = None
    if args.bandwidth_report:
        calibration = json.loads(args.bandwidth_report.read_text())
        if not (calibration.get("valid") and calibration.get("execution_complete")
                and calibration.get("source_patterns_unchanged_full_word_validation")
                and calibration.get("record_guards_pass")
                and calibration.get("actual_allocated_gpu_bytes", 32*(1<<30)) < 32*(1<<30)):
            parser.error("bandwidth report did not complete full validation below32GiB")
        reads = [c for c in calibration["cases"] if c["mode"].startswith("read_")]
        if len(reads) != 2 or not all(c["all_cta_checksums_counts_fingerprints_pass"] for c in reads):
            parser.error("bandwidth report requires two validated large-buffer read cases")
        args.payload_gbps = min(c["median_data_payload_GBps"] for c in reads)
    report = build(args.repo.resolve(), args.payload_gbps, args.temporal_unique)
    if calibration:
        report["bandwidth_proxy"] = dict(
            effective_operand_payload_GBps=args.payload_gbps,
            physical_sustained_DRAM_bandwidth_measured=False,
            source=str(args.bandwidth_report),
            definition="lower of two standalone resident large-buffer read GPU-time medians; everyCTA exact64 checksum/count/addressfingerprint checked; explicitbytes/time, not physicalDRAMcounter",
            read_cases=[dict(mode=c["mode"], data_bytes_read=c["explicit_data_bytes_read"],
                            median_gpu_seconds=c["median_gpu_seconds"],
                            effective_payload_GBps=c["median_data_payload_GBps"]) for c in reads],
            copy_cases=[dict(mode=c["mode"], effective_read_plus_write_payload_GBps=c["median_data_payload_GBps"])
                        for c in calibration["cases"] if c["mode"].startswith("copy_")])
        report["provenance"].append(dict(path=str(args.bandwidth_report),sha256=digest(args.bandwidth_report)))
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
