"""CPU-only acceptance/scope refresh. Preserve every earlier evidence report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "build/release/flash"


def calculate() -> dict:
    pins = {}

    def read(name: str) -> dict:
        path = EVIDENCE / name
        data = path.read_bytes()
        pins[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    old = read("sep21-decode-current-route-roofs-v2.json")
    metadata = read("sep21-decode-roofline-measured-payload-v1.json")
    isolation = read("sep22-context-matched-prefill-policy-isolation-native-results-v1.json")
    qualified_b1 = read("sep22-qualified-fixed4-R5-vs-fixed3-actual-performance-v1.json")
    qualified_batch = read("sep22-qualified-current-BQSA4-integer-matched-actual-performance-v1.json")
    batch_audit_name = "sep22-current-BQSA4-integer-normal-new-MTP3-B4-B2-v1.native-audit.json"
    batch = read(batch_audit_name)
    assert pins["build/release/flash/"+batch_audit_name] == qualified_batch["provenance_sha256"][batch_audit_name]
    assert isolation["pass"] and qualified_b1["pass"] and qualified_batch["pass"] and batch["valid"]
    assert batch["native_common_span_available"] and not batch["payload_reads"]
    assert not old["physical_DRAM_transaction_counters_collected"]
    bw = old["measured_large_resident_read_payload_GBps"]
    assert math.isfinite(bw) and bw > 0
    m = metadata["metadata"]
    c = m["packed_categories_GB"]
    experts, topk = m["experts"], m["selected_experts"]
    target_dense, vocab = c["target_dense"], c["target_vocab"]
    bank = metadata["full512_metadata"]["logical_store_GB"]
    ar_state, mtp_state = m["standard_minimum_state_GB"], m["mtp_depth3_minimum_state_GB"]
    temporal = metadata["retained_route_calibration"]["modeled_unique_experts_per_four_rows"]
    inventory = old["current_route_inventories"]

    # Small JSON metadata only; no .bin/.safetensors payload is opened.
    manifest_path = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1/manifest.json"
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    pins[str(manifest_path.relative_to(ROOT))] = hashlib.sha256(manifest_data).hexdigest()
    assert manifest["source_identity_sha256"] == metadata["source_identity"]
    fc_hidden = sum(manifest["tensors"]["mtp.fc_hidden."+suffix]["length"] for suffix in ("weight","scales","biases"))/1e9
    head_nonvocab = c["mtp_dense"] + c["mtp_experts_all"]*topk/experts
    head_streamed_nonvocab = head_nonvocab + 3*fc_hidden
    union = lambda lanes, unique: experts*(1-(1-unique/experts)**lanes)

    def denominators(lanes: int, useful_per_lane: float) -> dict:
        ar = inventory[f"{lanes}:"+("singleton_ar" if lanes == 1 else "batch_ar")]
        ver = inventory[f"{4*lanes}:"+("singleton_verify" if lanes == 1 else "batch_verify")]
        standard_unique = target_dense+ar["unique_operand_delta_GB"]+vocab+bank/experts*union(lanes,topk)+lanes*ar_state
        standard_stream = lanes*ar["raw_remaining_dense_GB"]+ar["expanded_with_actual_row_tiles_GB"]+vocab+lanes*bank*topk/experts+lanes*ar_state
        target_unique = target_dense+ver["unique_operand_delta_GB"]+vocab+bank/experts*union(lanes,temporal)+lanes*mtp_state
        head_unique = 3*(c["mtp_dense"]+vocab+c["mtp_experts_all"]/experts*union(lanes,topk))
        target_stream = 4*lanes*ver["raw_remaining_dense_GB"]+ver["expanded_with_actual_row_tiles_GB"]+vocab+4*lanes*bank*topk/experts+lanes*mtp_state
        head_stream = (useful_per_lane+2)*lanes*head_streamed_nonvocab+3*vocab
        return dict(standard_unique_GB=standard_unique, standard_streaming_GB=standard_stream,
                    mtp_unique_GB=target_unique+head_unique, mtp_streaming_GB=target_stream+head_stream,
                    head_fold_streaming_slope_GB_per_useful_token=lanes*head_streamed_nonvocab)

    standard_rows = []
    for lanes in (1,2,4):
        d = denominators(lanes,3)
        standard_rows.append(dict(native_lanes=lanes,
            optimistic_unique_operand_scenario_tps=bw*lanes/d["standard_unique_GB"],
            no_interrow_cache_reuse_streaming_reference_tps=bw*lanes/d["standard_streaming_GB"],
            unique_operand_state_GB=d["standard_unique_GB"], streaming_operand_state_GB=d["standard_streaming_GB"],
            current_same_policy_qualified_observation_tps=None,
            scope="historical source-backed standard operand model; no current matched qualified standard observation substituted"))

    mtp_rows = []
    wanted = ("sep22-initial3148-gathered-B1-MTP3-context16k-control-v1.json",
              "sep22-currentQ4-prefill-only-fma-MTP3-diagnostic-v1.json")
    for name in wanted:
        source = next(r for r in isolation["rows"] if r["report"] == name)
        useful = (source["cycles"]+source["accepted_drafts"])/source["cycles"]
        d = denominators(1,useful)
        mtp_rows.append(dict(native_lanes=1, configuration="context_matched_original_control" if name == wanted[0] else "FMA_only_acceptance_preserving_diagnostic",
            proposed_depth=3, target_verify_physical_rows=4,
            actual_cycles=source["cycles"], accepted_committed_drafts=source["accepted_drafts"],
            actual_postfirst_useful_tokens=source["cycles"]+source["accepted_drafts"],
            useful_tokens_per_lane_cycle=useful,
            optimistic_unique_operand_scenario_tps=bw*useful/d["mtp_unique_GB"],
            no_interrow_cache_reuse_streaming_reference_tps=bw*useful/d["mtp_streaming_GB"],
            unique_operand_state_cycle_GB=d["mtp_unique_GB"], streaming_operand_state_cycle_GB=d["mtp_streaming_GB"],
            observed_native_decode_median_tps=source["native_decode_median"], observed_native_prefill_median_tps=source["prefill_median"],
            equivalent_native_cycle_ms=1000*useful/source["native_decode_median"],
            full_original22_qualified_for_this_new_setting=False,
            scope="context matched 16K; original control worker differs from current decoder; FMA-only setting is causal performance diagnostic, not newly qualified deployment evidence",
            source_report=name, source_report_sha256=source["sha256"]))

    for lanes in (2,4):
        waves = [w for w in batch["waves"] if w["width"] == lanes and not w["warmup"]]
        assert len(waves) == 3 and all(w["valid"] for w in waves)
        samples = []
        for wave in waves:
            graphs = wave["actual_native_decode_graph_counts_by_width"]
            assert all(value == 0 for key,value in graphs.items() if key != f"b{lanes}")
            cycles = graphs[f"b{lanes}"]
            tokens = wave["actual_native_post_first_emission_tokens"]
            samples.append((cycles,tokens,tokens/(lanes*cycles)))
        assert len(set(samples)) == 1, "Do not silently average distinct acceptance trajectories"
        cycles,tokens,useful = samples[0]
        d = denominators(lanes,useful)
        observed = next(r["new"] for r in qualified_batch["rows"] if r["width"] == lanes)
        assert observed["exact_native_common_span_median_tokens_per_second"] == median(w["exact_native_common_span"]["native_aggregate_decode_tokens_per_second"] for w in waves)
        mtp_rows.append(dict(native_lanes=lanes, configuration="qualified_current_BQSA4_integer_batch",
            proposed_depth=3,target_verify_physical_rows=4*lanes,
            actual_cycles=cycles,actual_postfirst_useful_tokens=tokens,
            accepted_drafts_equivalent_from_emitted_count=tokens-lanes*cycles,
            useful_tokens_per_lane_cycle=useful,
            optimistic_unique_operand_scenario_tps=bw*lanes*useful/d["mtp_unique_GB"],
            no_interrow_cache_reuse_streaming_reference_tps=bw*lanes*useful/d["mtp_streaming_GB"],
            unique_operand_state_cycle_GB=d["mtp_unique_GB"],streaming_operand_state_cycle_GB=d["mtp_streaming_GB"],
            observed_native_decode_median_tps=observed["exact_native_common_span_median_tokens_per_second"],
            observed_native_prefill_median_tps=observed["native_prefill_median_tokens_per_second"],
            original22_no_new_regressions=qualified_batch["original22_no_new_regressions_at_both_widths"],
            all_prefill_trials_above4000=qualified_batch["B4_new_prefill_all3_above4000"] if lanes == 4 else qualified_batch["B2_prefill_over4000"],
            scope="current native common-span aggregate; same compiled worker integer0/1 matched; original22 qualified at both widths; footprint/sharing denominator remains inherited model, not new physical traffic"))

    perfect = []
    for lanes in (1,2,4):
        d = denominators(lanes,4)
        perfect.append(dict(native_lanes=lanes, proposed_depth=3, useful_tokens_per_lane_cycle=4,
            optimistic_unique_operand_scenario_tps=bw*4*lanes/d["mtp_unique_GB"],
            no_interrow_cache_reuse_streaming_reference_tps=bw*4*lanes/d["mtp_streaming_GB"],
            all_four_accepted_evidence_available=False))
    q = qualified_b1["new_summary"]
    graph_constraints = []
    phase_sources = [
        (wanted[0],1,3,mtp_rows[0]["observed_native_decode_median_tps"],mtp_rows[0]["useful_tokens_per_lane_cycle"]),
        (wanted[1],1,3,mtp_rows[1]["observed_native_decode_median_tps"],mtp_rows[1]["useful_tokens_per_lane_cycle"]),
        ("sep22-fixed4-R5-2K256-original22-v1.json",1,4,q["native_exact_request_decode_median_tokens_per_second"],255/79),
    ]
    batch_normal_name = "sep22-current-BQSA4-integer-normal-new-MTP3-B4-B2-v1.json"
    for row in mtp_rows[2:]:
        phase_sources.append((batch_normal_name,row["native_lanes"],3,row["observed_native_decode_median_tps"],row["useful_tokens_per_lane_cycle"]))
    for name,lanes,depth,rate,useful in phase_sources:
        normal = read(name)
        assert normal["completed"]
        waves = [w for w in normal["waves"] if not w["warmup"] and w["http_width"] == lanes]
        assert len(waves) == 3
        phases = {}
        for key,label in (("mtp.target_verify.total_gpu_ms","verify"),("mtp.head_decode.total_gpu_ms","head"),("mtp.prefix_restore.total_gpu_ms","restore")):
            phases[label] = median(w["mtp_counters"][key]/w["mtp_counters"]["mtp.verification_cycles"] for w in waves)
        for w in waves:
            counters = w["mtp_counters"]
            measured_useful = (lanes*counters["mtp.verification_cycles"]+counters["mtp.accepted_committed_drafts"])/(lanes*counters["mtp.verification_cycles"])
            assert math.isclose(measured_useful,useful,rel_tol=1e-12)
        equivalent_cycle = 1000*lanes*useful/rate
        gpu_sum = sum(phases.values())
        assert equivalent_cycle > gpu_sum > 0
        graph_constraints.append(dict(source_report=name,native_lanes=lanes,proposed_depth=depth,
            observed_native_tps=rate,useful_tokens_per_lane_cycle=useful,
            equivalent_native_cycle_ms=equivalent_cycle,median_gpu_phase_ms_per_cycle=phases,
            total_median_gpu_phases_ms=gpu_sum,remaining_native_clock_cost_ms=equivalent_cycle-gpu_sum,
            conditional_rate_if_all_nonGPU_costs_removed_tps=1000*lanes*useful/gpu_sum,
            conditional_verifier_speedups=[dict(verifier_speedup=s,conditional_tps=1000*lanes*useful/(equivalent_cycle-phases["verify"]+phases["verify"]/s)) for s in (1.5,2,3)],
            scope="current normal whole-model phase medians; preserve observed acceptance; hypothetical zerohost or verifier gains are not attained silicon peaks"))

    stage = read("sep22-current-fixed4-R5-target-stage-profile-v2.json")
    assert stage["diagnostic_valid"] and stage["physical_rows"] == 5 and stage["draft_depth"] == 4
    assert not stage["throughput_baseline"] and stage["tensor_payload_reads"] == 0
    measured = {d["index"]:d for d in stage["raw_command_profile"]["dispatches"]}
    shapes = defaultdict(lambda:dict(calls=0,gpu_ms=0.,useful_linear_GFLOP=0.))
    expert_ms = hc_ms = f32_ms = hc_padded_gf = f32_padded_gf = 0.
    for dispatch in stage["Forward_graph"]["dispatches"]:
        timestamp = measured[dispatch["index"]]
        assert timestamp["pipeline"] == dispatch["pipeline"] and timestamp["timestamps_valid"]
        ms = timestamp["gpu_seconds"]*1000
        name = dispatch["pipeline"]
        bindings = {b["index"]:b["size_bytes"] for b in dispatch["bindings"]}
        if name.startswith("flash_affine_mlx_qmv_f32xsum"):
            # Source ABI: BF16 input slot0, BF16 output slot5, true rows5.
            k,n = bindings[0]//10,bindings[5]//10
            if k >= 1024 and n >= 1024:
                item=shapes[k,n]
                item["calls"]+=1;item["gpu_ms"]+=ms;item["useful_linear_GFLOP"]+=2*5*k*n/1e9
        if name in ("flash_int8_expert_store_gate_up_m16_n64","flash_int8_expert_store_down_scatter_m16_n64"):
            expert_ms+=ms
        if name == "flash_hc_up_f32_mpp_m8_n32_s4":
            hc_ms+=ms;hc_padded_gf+=2*8*(bindings[1]//4)/1e9
        if name.startswith("flash_float_dense_small_rows_m") or name.startswith("flash_qsa_out_f32"):
            p=dispatch["parameters"]
            assert p["decoded"]
            f32_ms+=ms;f32_padded_gf+=2*p["padded_rows"]*p["K"]*p["output_count"]/1e9
    assert sum(s["calls"] for s in shapes.values()) == 113
    expert_gf=2*3*5*48*10*2560*640/1e9
    raw_gf=sum(s["useful_linear_GFLOP"] for s in shapes.values())
    raw_ms=sum(s["gpu_ms"] for s in shapes.values())
    families = [dict(family="Full512 expert GU+down",useful_linear_GFLOP=expert_gf,gpu_ms=expert_ms,linear_equivalent_TFps=expert_gf/expert_ms),
        dict(family="raw large original affine113calls",useful_linear_GFLOP=raw_gf,gpu_ms=raw_ms,linear_equivalent_TFps=raw_gf/raw_ms),
        dict(family="F32 cached dense plus HC-up",padded_linear_GFLOP=f32_padded_gf+hc_padded_gf,gpu_ms=f32_ms+hc_ms,
             padded_linear_equivalent_TFps=(f32_padded_gf+hc_padded_gf)/(f32_ms+hc_ms))]
    micro = []
    for name in ("sep22-expert-batch-r8-compact-native-u40-v2.json","sep22-expert-batch-r16-compact-native-u10-v2.json","sep22-expert-batch-r16-compact-native-oldmixed-v2.json"):
        probe=read(name)
        assert probe["exact_old_native_all_stage_pass"] and probe["timing_pass"]
        timing=next(t for t in probe["timings"] if t["name"] == "compactNativeM16")
        ms=median(s["gpu_ms"] for s in timing["samples"])
        gf=2*3*probe["rows"]*10*2560*640/1e9
        micro.append(dict(source_report=name,layer=probe["layer"],rows=probe["rows"],route_pattern=probe["case"],
            gpu_chain_median_ms=ms,useful_linear_GFLOP=gf,useful_linear_equivalent_TFps=gf/ms,
            scope="one synthetic normalized layer with warm route reuse; attainable component evidence, not wholemodel or silicon compute peak"))

    fma_hc=read("sep22-currentQ4-prefill-fma-hcnorm-MTP3-diagnostic-v1.json")
    fma_hc_summary=fma_hc["summary"][0]
    std_name="sep22-current-standard-B1-small-native-metadata-v1.json"
    std=read(std_name)
    assert pins["build/release/flash/"+std_name] == "9a125a8c15db90fea06311261562333f0a40000e384434346595ee00738be262"
    assert std["completed"] and len(std["trials"]) == 3
    assert all(t["valid"] and t["actual_native_batches"] == {"b1":255,"b2":0,"b3":0,"b4":0} for t in std["trials"])
    std_gpu_rate=median(t["gpu_decode_tps"] for t in std["trials"])
    std_rate=std["summary"]["native_exact_request_decode_median_tokens_per_second"]
    standard_rows[0].update(current_observed_native_tps=std_rate,
        current_observed_native_prefill_median_tps=std["summary"]["native_prefill_median_tokens_per_second"],
        current_prefill_all_three_above4000=all(t["prefill_tps"] > 4000 for t in std["trials"]),
        current_configuration_quality_qualified=std["quality_qualified_for_new_configuration"],
        scope="current scalar performance metadata,255real AR1 commands/trial; new setting not original22 qualified; oldercontext-matched standard control still pending")
    output = dict(schema="splash-context-matched-decode-scenarios-and-qualified-observations-sep22-v4",
        GPU_executed=False, model_or_operand_payload_read=False, physical_DRAM_traffic_measured=False, attainable_peak_proven=False,
        effective_resident_array_read_payload_GBps=bw,
        bandwidth_scope="validated large resident arrays explicit payload/time; not physical DRAM transaction measurement and not attainable affine/MPP peak",
        standard_operand_scenarios=standard_rows, mtp_depth3_current_acceptance_scenarios=mtp_rows,
        qualified_B1_4K_observation=dict(proposed_depth=4,target_verify_physical_rows=5,
            actual_cycles=79,accepted_committed_drafts=176,useful_tokens_per_lane_cycle=255/79,
            observed_native_decode_median_tps=q["native_exact_request_decode_median_tokens_per_second"],
            observed_native_prefill_median_tps=q["native_prefill_median_tokens_per_second"],
            all_three_prefill_trials_above4000=qualified_b1["new_prefill_all3_above4000"],
            original22_no_new_regressions=qualified_b1["original22_no_new_regressions"],
            depth3_operand_denominator_applied=False,
            scope="qualified fixed4/R5 numerical execution geometry; direct observation only; do not substitute fixed3/R4 acceptance or traffic denominator"),
        additional_completed_diagnostic=dict(configuration="FMA plus HC prefill, fixed3",observed_native_decode_median_tps=fma_hc_summary["native_exact_request_decode_median_tokens_per_second"],
            observed_native_prefill_median_tps=fma_hc_summary["native_prefill_median_tokens_per_second"],full_original22_qualified_for_setting=False),
        compute_and_execution_constraints=dict(silicon_compute_peak_calibrated=False,
            current_standard_B1_graph_model=dict(observed_native_tps=std_rate,
                conditional_rate_with_current_GPU_costs_and_zero_nonGPU_costs_tps=std_gpu_rate,
                median_GPU_ms_per_token=1000/std_gpu_rate,median_native_ms_per_token=1000/std_rate,
                GPU_decode_rate_trials=[t["gpu_decode_tps"] for t in std["trials"]],
                quality_qualified_for_new_configuration=std["quality_qualified_for_new_configuration"],
                scope="actual255AR1 commands/trial; GPUrate numerator equalsreal generated ARtoken count; fixedGPUcost zerohost sensitivity, nothardwarepeak"),
            normal_whole_model_graph_models=graph_constraints,
            qualified_R5_stage=dict(command_gpu_ms=stage["raw_command_profile"]["gpu_seconds"]*1000,
                command_wall_ms=stage["raw_command_profile"]["wall_seconds"]*1000,
                timed_dispatches=len(measured),scope="third realR5 verify after2unprofiled warm calls; stage instrumentation perturbs scheduling; not canonical throughput",
                measured_family_equivalent_rates=families,
                raw_large_shapes=[dict(K=k,N=n,**value) for (k,n),value in shapes.items()]),
            favorable_and_mixed_component_microbenchmarks=micro,
            scalar_R5_M16_expert_padding=dict(routes=50,unique_experts_minimum=10,unique_experts_maximum=50,
                maximum_valid_row_fraction=5/16,minimum_valid_row_fraction=1/16,
                formula="50/(16*unique_experts); each expert hasatmost5true rows in scalarR5; exact extra instruction and physical cache costs are not measured"),
            missing_for_defensible_compute_peak=[
                "Shape-matched standalone hardware tensor/ALU saturation measurements with exact dtype, padding and activejob counts; linear-equivalent rates do not count unpack/scale/activation work.",
                "Actual perlayer expert overlap/job occupancy for the current trajectory; favorable R16u10 fulltiles cannot be imposed on singletonR5.",
                "Measured per-family speedups integrated into the unchanged acceptance/quality path, with wholeworker timing and overlapping bandwidth/compute constraints verified.",
            ]),
        perfect_acceptance_depth3_scenarios=perfect,
        inventory_assumptions=dict(source="Sep21 stock Full512 current-route inventory, inherited storage/sharing model; current shader instruction/transaction traffic not remeasured",
            target_dense_original_GB=target_dense,vocabulary_original_GB=vocab,full512_I8_bank_GB=bank,
            target_selected10_all48_layers_GB=bank*topk/experts,
            standard_state_floor_per_lane_GB=ar_state,mtp_state_floor_per_lane_GB=mtp_state,
            expert_count=experts,selected_experts_per_token=topk,
            target_temporal_unique_experts_per_four_rows=temporal,
            temporal_union_scope="192 four-row windows from one older16-row speculative diagnostic; not current routing population",
            head_fc_hidden_original_packed_GB=fc_hidden,head_nonvocabulary_row_stream_GB=head_streamed_nonvocab,
            hc_up_original97_packed_GB=old["actual_hc_up_route"]["original_packed_all97_GB"],
            hc_up_F32_eligible_main_rows="4..16; standard batch fusedHC-up remains raw",
            verifier_F32_inventory_GB={str(rows):dict(unique=inventory[f"{rows}:"+("singleton_verify" if rows == 4 else "batch_verify")]["expanded_unique_GB"],
                explicit_row_tiles=inventory[f"{rows}:"+("singleton_verify" if rows == 4 else "batch_verify")]["expanded_with_actual_row_tiles_GB"]) for rows in (4,8,16)}),
        formulas=dict(useful_prefix="(native_cohort_cycles*lanes+committed_drafts)/(native_cohort_cycles*lanes), equivalently actual postfirst emitted tokens/(cycles*lanes)",
            expert_union="512*(1-(1-s/512)^lanes); independent uniform lanes",
            optimistic_traffic_scenario="effective resident-array GBps*lanes*useful_prefix/unique_operand_state_cycle_GB",
            streaming_head="(useful_prefix+2)*lanes*head_nonvocabulary_row_stream_GB+3*vocabulary_GB; true fold plus two draft calls",
            standard_unique="dense+selected_F32_expansion+vocabulary+I8_bank/512*expert_union(lanes,10)+lanes*standard_state",
            mtp_unique="four-row target dense/F32/vocabulary/expert union/state +3 trained-head calls, vocabulary shared per cohort"),
        scope_corrections=[
            "Prior B1 53.226/105cycles150drafts is historical qualified fixed3 with altered prefill trajectory; current matched original control is68.434/80cycles175drafts.",
            "Best acceptance-preserving FMA-only diagnostic is70.857/79cycles176drafts and3528.617prefill; it is not original22 qualified for the new setting.",
            "Current qualified fixed4/R5 is61.544/4006.925prefill all3>4K; its traffic denominator is not the fixed3/R4 denominator.",
            "Current qualified B4 has82cycles1020postfirst emittedtokens =>3.109756useful/lane/cycle, not older78cycles708drafts=>3.269231.",
            "Current B2/B4 matched native observations are104.420/151.000 with original22 noNewRegressions at both widths; B4prefill4094.042 all3>4K, B2prefill3619.698 below4K.",
            "Old fixed3 verifier sensitivity times are not reapplied to the matched controls or qualified fixed4/R5; current matching phase timings are required.",
        ],
        limitations=[
            "These conditional bandwidth scenarios are not demonstrated hardware peaks. Compute, dispatch, host, SSD, padding/scratch and transaction amplification are omitted.",
            "Unique source-operand sharing is optimistic; streaming references count historical explicit row tiles/raw rows and can change with newer producers. Cache reuse was not physically measured.",
            "No current qualified same-policy standard B1/B2/B4 observations are substituted from older numerical routes.",
            "Current standard B1 performance35.821/4043.737prefill is complete but new setting notoriginal22 qualified; matchedoldstandard comparison remains pending.",
            "Native common graphs support at most4lanes. Batch8/16 and perfect acceptance are unmeasured hypothetical extensions.",
            "Original22 noNewRegressions is a scoped comparison with retained baseline failures, not universal accuracy or public promotion qualification.",
        ], provenance_sha256=pins)
    for row in standard_rows:
        assert row["optimistic_unique_operand_scenario_tps"] >= row["no_interrow_cache_reuse_streaming_reference_tps"] > 0
    for row in mtp_rows:
        assert 1 <= row["useful_tokens_per_lane_cycle"] <= 4
        assert row["optimistic_unique_operand_scenario_tps"] >= row["no_interrow_cache_reuse_streaming_reference_tps"] > 0
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=EVIDENCE/"sep22-context-matched-decode-roofs-and-current-observations-v4.json")
    args=parser.parse_args()
    if args.out.exists():
        raise SystemExit("Preserve earlier reports; select a fresh --out path")
    out=calculate()
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(args.out)


if __name__ == "__main__":
    main()
