#!/usr/bin/env python3
"""CPU-only stock Full512 kernel-route traffic scenarios and measured targets.

Only JSON metadata/source/reports are read. Unique operands give an optimistic
cache-sharing footprint. Row-streaming scenarios count explicit matrix reuse
in MPP tiles but assume no cache reuse of independently loaded raw rows/routes.
Neither scenario measures physical DRAM transactions or proves attainability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import roofline


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(repo: Path, matrix_path: Path, bandwidth_path: Path) -> dict:
    matrix = json.loads(matrix_path.read_text())
    bandwidth = json.loads(bandwidth_path.read_text())
    assert matrix["completed"] and bandwidth["valid"] and bandwidth["execution_complete"]
    read_cases = [c for c in bandwidth["cases"] if c["mode"].startswith("read_")]
    gbps = min(c["median_data_payload_GBps"] for c in read_cases)
    assert math.isfinite(gbps) and gbps > 0
    generic = roofline.build(repo, gbps, None)
    metadata = generic["metadata"]
    cat = {k: v*1e9 for k,v in metadata["packed_categories_GB"].items()}
    bank = generic["full512_metadata"]["logical_store_GB"]*1e9
    state_ar = metadata["standard_minimum_state_GB"]*1e9
    state_mtp = metadata["mtp_depth3_minimum_state_GB"]*1e9
    count, topk = metadata["experts"], metadata["selected_experts"]
    binary = Path(matrix["provenance"]["files"]["binary"]["path"])
    assert sha(binary) == matrix["provenance"]["files"]["binary"]["sha256"]
    source = binary.parent / "source/runtime/flash"
    hc_source = source / "FlashHCFused.cpp"
    batch_source = source / "FlashBatchForward.cpp"
    policy_path = source / "FlashFloatDenseCache.cpp"
    worker_source = source / "FlashWorker.mm"
    # Fail on policy drift rather than attributing an unmeasured new route.
    assert "rows < 4 || rows > 16" in hc_source.read_text()
    assert "g.rows < 4 || g.rows > 16" in hc_source.read_text()
    assert "trunk.batchHCFusedUpMixF32" not in batch_source.read_text()
    assert "addHCFusedUpMix(graph" in batch_source.read_text()
    assert 'environmentSwitch("SPLASH_FLASH_FLOAT_DENSE_CACHE")' in worker_source.read_text()
    for setting in ("standard","3"):
        env = matrix["environments"][setting]["resolved_flash_environment"]
        assert env["SPLASH_FLASH_FLOAT_DENSE_CACHE"] == "1"
        assert env["SPLASH_FLASH_FLOAT_DENSE_SELECTIVE"] == "1"
        assert env["SPLASH_FLASH_FUSE_HC"] == "1"
        assert env["SPLASH_FLASH_HC_UP_F32_MPP"] == "1"
    manifest_path = repo / "install/local-models/Flash-Next-oQ4e-mtp-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tensors, quant = manifest["tensors"], manifest["quantization"]
    matrices = {}
    for name, tensor in tensors.items():
        if not (name.endswith(".weight") and tensor["dtype"] == "U32" and "ngram_embedding" not in name):
            continue
        prefix = name[:-7]
        scale, bias = tensors[prefix+".scales"], tensors[prefix+".biases"]
        q = quant.get(prefix, quant)
        matrices[prefix] = dict(
            n=tensor["shape"][-2], k=scale["shape"][-1]*q["group_size"],
            bits=q["bits"], group=q["group_size"],
            packed=tensor["length"]+scale["length"]+bias["length"],
            experts=tensor["shape"][0] if len(tensor["shape"]) == 3 else 1)
    policy = {}
    pattern = r'\{"([^"\n]+)",\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\}'
    for match in re.finditer(pattern, policy_path.read_text()):
        role, *values = match.groups()
        n,k,bits,group,r4,r8,r16,middle = map(int,values)
        policy[role,n,k,bits,group] = r4,r8,r16,middle
    hc_matrices = [p for n,p in matrices.items() if n.startswith("language_model.model.") and n.endswith("input_mix_weight_up")]
    assert len(hc_matrices) == 97
    hc_packed = sum(p["packed"] for p in hc_matrices)
    hc_f32 = sum(p["n"]*p["k"]*4 for p in hc_matrices)

    def route_inventory(rows: int, executor: str) -> dict:
        selected = []
        for name,p in matrices.items():
            if not name.startswith("language_model.model.") or p["experts"] != 1:
                continue
            role = re.sub(r"^language_model\.model\.layers\.\d+\.", "", name)
            if name.endswith("input_mix_weight_up"):
                role = "hc_up"
                if executor == "batch_ar":
                    continue  # Source and actual counters prove raw fusedup.
            entry = policy.get((role,p["n"],p["k"],p["bits"],p["group"]))
            if not entry or not 4 <= rows <= 16:
                continue
            idx = 0 if rows < 8 else 1 if rows == 8 else 2 if rows == 16 else 3
            if entry[idx] < 0:
                continue
            tile_rows = 8 if entry[idx] == 0 else 16
            # Both specialized fusedHC-up and qualified QSA N32 use M8.
            if role == "hc_up" or (role == "self_attn.o_proj" and p["bits"] in (5,6) and p["group"] == 64):
                tile_rows = 8
            selected.append(dict(name=name, role=role, packed_bytes=p["packed"],
                                 f32_bytes=p["n"]*p["k"]*4, tile_rows=tile_rows,
                                 row_tiles=(rows+tile_rows-1)//tile_rows))
        packed_selected = sum(p["packed_bytes"] for p in selected)
        expanded_unique = sum(p["f32_bytes"] for p in selected)
        expanded_streamed = sum(p["f32_bytes"]*p["row_tiles"] for p in selected)
        return dict(physical_rows=rows, executor=executor,
                    f32_matrix_count=len(selected),
                    f32_hc_up_count=sum(p["role"] == "hc_up" for p in selected),
                    selected_packed_GB=packed_selected/1e9,
                    expanded_unique_GB=expanded_unique/1e9,
                    expanded_with_actual_row_tiles_GB=expanded_streamed/1e9,
                    unique_operand_delta_GB=(expanded_unique-packed_selected)/1e9,
                    raw_remaining_dense_GB=(cat["target_dense"]-packed_selected)/1e9,
                    selected=selected)

    inventories = {str(r)+":"+e: route_inventory(r,e) for r,e in (
        (1,"singleton_ar"),(2,"batch_ar"),(4,"batch_ar"),
        (4,"singleton_verify"),(8,"batch_verify"),(16,"batch_verify"))}
    evidence = []
    results = []
    head_nonvocab = cat["mtp_dense"] + cat["mtp_experts_all"]*topk/count
    # fc_hidden applies the same original matrix independently to four streams.
    head_nonvocab_streamed = head_nonvocab + 3*matrices["mtp.fc_hidden"]["packed"]
    temporal_unique = generic["retained_route_calibration"]["modeled_unique_experts_per_four_rows"]
    for batch in (1,2,4,8,16):
        lanes = min(batch,4)
        unsupported = batch > 4
        ar_waves = [w for w in matrix["waves"] if w["mtp_setting"] == "standard" and w["http_width"] == lanes and not w["warmup"]]
        mtp_waves = [w for w in matrix["waves"] if w["mtp_setting"] == "3" and w["http_width"] == lanes and not w["warmup"]]
        prefixes = [(lanes*w["mtp_counters"]["mtp.verification_cycles"] + w["mtp_counters"]["mtp.accepted_committed_drafts"])/(lanes*w["mtp_counters"]["mtp.verification_cycles"]) for w in mtp_waves]
        prefix = statistics.mean(prefixes)
        ar = inventories[str(lanes)+":"+("singleton_ar" if lanes == 1 else "batch_ar")]
        ver = inventories[str(4*lanes)+":"+("singleton_verify" if lanes == 1 else "batch_verify")]
        union = lambda n,s: count*(1-(1-s/count)**n)
        # Best cache-sharing footprints: source matrices only once per cohort.
        ar_unique = (cat["target_dense"]+ar["unique_operand_delta_GB"]*1e9+cat["target_vocab"] + bank/count*union(lanes,topk)+lanes*state_ar)
        ver_unique = (cat["target_dense"]+ver["unique_operand_delta_GB"]*1e9+cat["target_vocab"] + bank/count*union(lanes,temporal_unique)+lanes*state_mtp)
        head_unique = 3*(cat["mtp_dense"]+cat["target_vocab"]+cat["mtp_experts_all"]/count*union(lanes,topk))
        # Streaming raw input rows independently. Current gathered expert MPP
        # intentionally has validRows1; no explicit cross-route tile reuse.
        ar_streamed = (lanes*ar["raw_remaining_dense_GB"]*1e9 + ar["expanded_with_actual_row_tiles_GB"]*1e9 +cat["target_vocab"] +lanes*bank*topk/count +lanes*state_ar)
        ver_streamed = (4*lanes*ver["raw_remaining_dense_GB"]*1e9+ver["expanded_with_actual_row_tiles_GB"]*1e9+cat["target_vocab"] +4*lanes*bank*topk/count+lanes*state_mtp)
        # Fold has observed true prefix pairs perlane; two further scalar draft
        # calls follow. The vocabulary MPP explicitly shares each cohort call.
        head_streamed = (prefix+2)*lanes*head_nonvocab_streamed+3*cat["target_vocab"]
        head_full_prefix_streamed = 6*lanes*head_nonvocab_streamed+3*cat["target_vocab"]
        ar_tps = statistics.median(w["client_wave_post_first_emission_tokens_per_second"] for w in ar_waves)
        mtp_tps = statistics.median(w["client_wave_post_first_emission_tokens_per_second"] for w in mtp_waves)
        verify_ms = statistics.median(w["mtp_counters"]["mtp.target_verify.total_gpu_ms"] /w["mtp_counters"]["mtp.verification_cycles"] for w in mtp_waves)
        # Whole-wave actual decode interval anchors all remaining kernel/host
        # costs. Conditional verifier gains do not assume those costs disappear.
        actual_cycle_ms = prefix*lanes*1000/mtp_tps
        conditional = [dict(verifier_speedup=s, predicted_aggregate_tps=prefix*lanes*1000/(actual_cycle_ms-verify_ms+verify_ms/s)) for s in (1.5,2,3,4)]
        results.append(dict(requested_batch=batch, supported_common_graph=not unsupported,
            modeled_native_cohort_lanes=lanes, mtp_committed_prefix_per_lane_cycle=prefix,
            standard_unique_operand_and_state_GB=ar_unique/1e9,
            standard_unique_operand_traffic_ceiling_aggregate_tps=lanes*gbps*1e9/ar_unique,
            standard_no_interrow_cache_reuse_requested_stream_GB=ar_streamed/1e9,
            standard_no_interrow_cache_reuse_streaming_reference_aggregate_tps=lanes*gbps*1e9/ar_streamed,
            mtp_unique_operand_and_state_cycle_GB=(ver_unique+head_unique)/1e9,
            mtp_unique_operand_traffic_ceiling_aggregate_tps=prefix*lanes*gbps*1e9/(ver_unique+head_unique),
            mtp_perfect_prefix4_unique_operand_traffic_ceiling_aggregate_tps=4*lanes*gbps*1e9/(ver_unique+head_unique),
            mtp_no_interrow_cache_reuse_requested_stream_cycle_GB=(ver_streamed+head_streamed)/1e9,
            mtp_no_interrow_cache_reuse_streaming_reference_aggregate_tps=prefix*lanes*gbps*1e9/(ver_streamed+head_streamed),
            mtp_perfect_prefix4_no_interrow_cache_reuse_streaming_reference_aggregate_tps=4*lanes*gbps*1e9/(ver_streamed+head_full_prefix_streamed),
            measured_standard_aggregate_tps=ar_tps, measured_mtp_aggregate_tps=mtp_tps,
            measured_mtp_verify_gpu_ms_per_cohort_cycle=verify_ms,
            whole_wave_observed_mtp_cycle_ms=actual_cycle_ms,
            measured_phase_conditional_targets=conditional))
        if batch <= 4:
            evidence.append(dict(batch=batch, standard_cached_hc_up_calls=[w["native_counter_delta"]["hc_up_route_counters.cached_encoded_calls"] for w in ar_waves],
                standard_hc_up_eligible_calls=[w["native_counter_delta"]["hc_up_route_counters.geometry_eligible_attempts"] for w in ar_waves],
                standard_hc_up_unsupported_geometry_calls=[w["native_counter_delta"]["hc_up_route_counters.skipped_unsupported_geometry"] for w in ar_waves],
                mtp_cached_hc_up_calls=[w["native_counter_delta"]["hc_up_route_counters.cached_encoded_calls"] for w in mtp_waves]))

    singleton = results[0]
    independent = [dict(batch=b, shared_native_common_graph_supported=b<=4,
        model="serial independent scalar executions; no crossrequest matrixcache reuse; aggregate doesnotmultiply bybatch",
        standard_unique_operand_traffic_ceiling_aggregate_tps=singleton["standard_unique_operand_traffic_ceiling_aggregate_tps"],
        mtp_unique_operand_traffic_ceiling_aggregate_tps=singleton["mtp_unique_operand_traffic_ceiling_aggregate_tps"],
        standard_streaming_reference_aggregate_tps=singleton["standard_no_interrow_cache_reuse_streaming_reference_aggregate_tps"],
        mtp_streaming_reference_aggregate_tps=singleton["mtp_no_interrow_cache_reuse_streaming_reference_aggregate_tps"]) for b in (1,2,4,8,16)]
    return dict(schema="splash-stock-full512-current-route-traffic-scenarios-v2",
        gpu_execution=False, model_payload_files_opened=False,
        measured_large_resident_read_payload_GBps=gbps,
        physical_DRAM_transaction_counters_collected=False,
        runtime_binary_sha256=matrix["provenance"]["files"]["binary"]["sha256"],
        actual_hc_up_route=dict(stock_main_minimum_rows=4, stock_main_maximum_rows=16,
            batch_ar_fused_hc_up="raw original affine; never cachedHC-up bridge",
            hc_up_matrix_count=97, original_packed_all97_GB=hc_packed/1e9,
            hypothetical_f32_all97_GB=hc_f32/1e9, hypothetical_forced_r1_delta_GB=(hc_f32-hc_packed)/1e9,
            hypothetical_forced_r1_standard_traffic_ceiling_tps=gbps*1e9/(singleton["standard_unique_operand_and_state_GB"]*1e9+hc_f32-hc_packed),
            actual_counter_evidence=evidence),
        current_route_inventories=inventories,
        shared_native_cohort_scenarios=results,
        independent_scalar_lane_scenarios=independent,
        caveats=[
            "Unique operands are an optimistic cache-sharing footprint, not a demonstrated physical traffic or attainable decode ceiling.",
            "Streaming references repeat raw dense coefficients perinputrow and Full512 I8 expert operands pervalid route; explicitMPP rowtiles reuse F32 and vocabulary operands. PhysicalGPU caches can retain independently requested operands.",
            "Cached F32 R16 HC-up/QSA-output and some dense tiles have two M8 tiles, so their source payload is requested twice; uniqueoperand inventory alone hides that.",
            "Standard batchHC-up is raw even when overallphysicalrows4; singleton and verifier cachedHC-up require4..16rows. Source flags alone do notselect impossiblegeometry.",
            "MTP head fold uses actual true prefix pairs; fc_hidden uses four independent streams, two further draftcalls follow, and vocabulary is shared percohort call.",
            "Scratch, padding, gathered device activation reloads, SSD preparation, partialrollback replay, transaction amplification, compute, dispatch and hostwork are omitted from traffic scenarios.",
            "Batch8/16 common graphs are unsupported. Current output models serial4-lane cohorts; scalar-independent aggregate rate stays flat withoutguaranteedcrossrequestreuse.",
            "Uniform independentlane expert unions and temporalunion from one older16row diagnostic source are optimistic assumptions; currentFull512router distributions can differ.",
            "Conditional speed targets preserve observed wholewave remaining costs and measured actual acceptance. No prediction isclaimed achieved or provenattainable.",
        ],
        provenance=[dict(path=str(p),sha256=sha(p)) for p in (matrix_path,bandwidth_path,manifest_path,hc_source,batch_source,policy_path,worker_source)])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument("--matrix",type=Path,default=Path("build/release/flash/sep21-gathered-standard-mtp3-batch-matrix-v1.json"))
    parser.add_argument("--bandwidth",type=Path,default=Path("build/release/flash/sep21-sustained-memory-payload-v1.json"))
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=build(args.repo.resolve(),args.matrix.resolve(),args.bandwidth.resolve())
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
