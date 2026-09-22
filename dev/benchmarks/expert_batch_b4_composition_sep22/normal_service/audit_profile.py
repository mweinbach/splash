#!/usr/bin/env python3
"""Post-unload source/counter/native-emission audit. Root reads actual reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from launch_common import ROOT,HERE,load_pinned,validate_binding,require,get,module,same,sha

ARENA_ROUTE=";private-batch-lane2k-begin0-exact-bulk-sg8-one-fully-consumed-reused-workspace-v1"
BQSA_ROUTE=";private-batch-real4-MTP3-allfresh2048-existing-packedV-twopass-v2"
TEACHER_ROUTE="mtp-batch-teacher-cache-only-compact-real-lanes-original-qsa-cache-prefix-no-attention-or-mlp-v1"

def delta(wave,path):
    before,after=[get(wave.get(side,{}),path) for side in ("status_before","status_after")]
    require(type(before) is int and type(after) is int and 0<=before<=after<2**64,"Invalid/decreasing native counter: "+path)
    require(same(wave.get("native_counter_delta",{}).get(path),after-before),"Recorded native delta differs from actual snapshots: "+path)
    return after-before

def helpers(plan):
    binding=json.loads(Path(plan["composition_binding"]).read_text())
    require(sha(plan["composition_binding"])==plan["composition_binding_sha256"],"Current composition identity binding drift")
    adapter=module(binding["service_adapter_path"],"_frozen_current_composition_service_for_normal_audit")
    quality=module(plan["frozen_quality_core"],"_frozen_current_composition_identity_for_normal_audit")
    return binding,adapter,quality

def status_errors(status,plan,role,helper=None):
    binding,adapter,quality=helpers(plan) if helper is None else helper
    errors=adapter.status_errors(status,binding,"mtp3",True,role=="new")
    errors+=quality.composition_identity_errors(status,role,binding)
    fixed={"identity.target_gathered_mpp_max_physical_rows":4,"maximum_context_tokens":16384,
        "memory_audit.valid":True,"memory_audit.batch_prefill_workspace_bytes":plan["expected_native_batch_prefill_ctor_bytes"],
        "native_lifecycle_timestamps_sep22.enabled":True,"identity.mtp_batch_teacher_priming_route":TEACHER_ROUTE}
    for path,expected in fixed.items():
        if not same(get(status,path),expected):errors.append("Actual current constructor/clock/teacher source differs: "+path)
    routes=get(status,"identity.batch_prefill_kernel_routes")
    if not isinstance(routes,str) or any(marker not in routes for marker in (ARENA_ROUTE,BQSA_ROUTE)):
        errors.append("Actual constructed legacy fallback and new BQSA4 routes required")
    available,reserve=[get(status,"memory_governor."+key) for key in ("host_available_bytes","host_reserve_bytes")]
    if type(available) is not int or type(reserve) is not int or not 0<=reserve<=available:
        errors.append("Actual governor host headroom unavailable/invalid")
    return errors

def wave_errors(wave,plan,role,helper=None):
    errors=[];width=wave.get("http_width");actual={}
    if width not in (4,2):return {},["Unqualified native wave width"]
    if wave.get("mtp_setting")!="3" or wave.get("prompt_tokens")!=2048 or wave.get("output_budget_tokens")!=256:
        errors.append("Frozen MTP3/2048/256 geometry differs")
    for side in ("status_before","status_after"):
        errors += [side+": "+error for error in status_errors(wave.get(side,{}),plan,role,helper)]
    if wave.get("status_before",{}).get("identity")!=wave.get("status_after",{}).get("identity"):
        errors.append("Worker/raw/numerical/source identity changed within the measured wave")
    expected={f"scheduler.prefill_batches_by_width.b{n}":int(n==width) for n in range(1,5)}
    expected.update({"batch_prefill_twopass_counters.encoded_QSA_lane_calls":48 if width==4 else 0,
        "batch_prefill_twopass_counters.encoded_QSA_lane_layer_calls":48 if width==4 else 0,
        "batch_prefill_twopass_counters.completed_native_forwards":int(width==4),
        "mtp.batch_teacher_cache_only_priming_calls":15,"mtp.batch_teacher_cache_only_completed_lanes":15*width,
        "mtp.batch_teacher_cache_only_completed_real_pairs":1920*width,"mtp.teacher_cache_only_priming_calls":width,
        "batch_prefill.true_target_hidden_copied_bytes":width*2048*10240*2})
    expected.update({f"scheduler.batch_mtp_priming_batches_by_width.b{n}":15 if n==width else 0 for n in range(1,5)})
    try:
        for path,value in expected.items():
            actual[path]=delta(wave,path)
            if actual[path]!=value:errors.append("Actual true cohort/QSA/teacher prefix/tail differs: "+path)
        integer={}
        for rows in (8,16):
            calls=[]
            for stage in ("plan","gate","down"):
                path=f"compact_native_batch_verify.r{rows}.{stage}_graph_calls";count=delta(wave,path);actual[path]=count;calls.append(count)
                path=f"compact_native_batch_verify.r{rows}.{stage}_graph_rows";physical=delta(wave,path);actual[path]=physical
                if count%48 or physical!=count*rows:errors.append("Actual integer48layer/physical row delta differs: "+path)
            if len(set(calls))!=1:errors.append(f"Actual R{rows} integer plan/gate/down totals differ")
            integer[rows]=calls[0]
        if role=="old" and any(integer.values()):errors.append("Integer0 control encoded integer work")
        if width==2 and integer[16]:errors.append("PhysicalR16 cannot arise from a B2-only measured cohort")
        if role=="new":
            # Every eligible integer graph is one rows4 target verifier cycle.
            # Width4 may later become width2; widths1/3 and rows1..3 fall back.
            for rows,native_width in ((8,2),(16,4)):
                cycles=delta(wave,f"scheduler.decode_batches_by_width.b{native_width}")
                if integer[rows]//48>cycles:errors.append("Integer rows4 graph constructions exceed actual native cohort cycles")
    except ValueError as error:errors.append(str(error))
    return {"width":width,"trial":wave.get("trial"),"warmup":wave.get("warmup"),"valid":not errors,"errors":errors,
        "actual_native_source_and_priming_counter_deltas":actual,
        "new_BQSA4_lane_layer_calls_expected":48 if width==4 else 0,
        "scope":"Encoded counters count graph construction; healthy requests/native DONE and Stop footer are checked separately.",
        "source_target_rows_per_lane":4,"legitimate_B4_to_R8_after_peer_finish_allowed":width==4,
        "prepared_predictions_used_as_output_numerator":False},errors

def aggregate_integer_errors(waves,role):
    """Actual route work must occur in measured trials, never inferred from warmup."""
    errors=[];totals={}
    for width,rows in ((4,16),(2,8)):
        selected=[wave for wave in waves if wave.get("width")==width and wave.get("warmup") is False]
        if len(selected)!=3:errors.append("Complete three measured trials required for integer exposure: B"+str(width))
        total=0
        for wave in selected:
            path=f"compact_native_batch_verify.r{rows}.plan_graph_calls"
            n=wave.get("actual_native_source_and_priming_counter_deltas",{}).get(path)
            if type(n) is not int or n<0 or n%48:errors.append("Missing/invalid actual measured integer exposure: "+path)
            else:total+=n
        totals["b"+str(width)+"_r"+str(rows)+"_measured_graph_calls"]=total
        if role=="new" and total<=0:errors.append("Selected R16/R8 integer work must occur across each width's actual measured trials")
        if role=="old" and total:errors.append("Integer0 control cannot have measured integer exposure")
    return {"valid":not errors,"errors":errors,"actual_measured_route_totals":totals,
        "scope":"Actual graph construction across measured trials; warmup exposure does not qualify; per-wave zero fallback is allowed."},errors

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role",choices=("old","new"),required=True)
    parser.add_argument("--launch-witness-sha256",required=True)
    parser.add_argument("--proof-binding-sha256",required=True)
    parser.add_argument("--run-root-reports",action="store_true")
    args=parser.parse_args();require(args.run_root_reports,"Root-exclusive actual report access required")
    plan=load_pinned(args.launch_witness_sha256);proof=validate_binding(plan,args.launch_witness_sha256,args.proof_binding_sha256)
    selected=plan["profiles"][args.role];output=Path(selected["profile_audit"])
    require(not output.exists(),"Fresh source/native profile audit required")
    report=json.loads(Path(selected["report"]).read_text());native=module(HERE/"audit_native.py","_unchanged_native_common_span_for_combined_audit")
    groups=native.trace_groups(Path(selected["trace"]),[w["http_width"] for w in report.get("waves",[])])
    result=native.audit(report,groups);errors=list(result["errors"])
    for name,digest in (("binary",plan["worker_sha256"]),("metallib",plan["metallib_sha256"])):
        if get(report,"provenance.files."+name+".sha256")!=digest:errors.append("Actual measured runtime differs: "+name)
    for key,value in selected["explicit_flags"].items():
        if get(report,"environments.3.resolved_flash_environment."+key)!=value:errors.append("Actual resolved canonical flag differs: "+key)
    helper=helpers(plan);waves=[]
    for index,wave in enumerate(report.get("waves",[])):
        row,issues=wave_errors(wave,plan,args.role,helper);waves.append(row);errors += [f"wave{index}: "+error for error in issues]
    exposure,issues=aggregate_integer_errors(waves,args.role);errors+=issues
    evidence=report.get("server_runs",[])
    if len(evidence)!=1 or any(get(run,"unload_evidence.returncode")!=0 or get(run,"unload_evidence.post_parent_exit_sigkill_required") is not False for run in evidence):
        errors.append("Actual graceful frontend/worker exit0 without SIGKILL required")
    published={"schema":"current-BQSA4-integer-MTP3-normal-native-commonspan-source-audit-v1","valid":not errors,"errors":errors,
        "role":args.role,"current_worker_sha256":plan["worker_sha256"],"compiled_seal_sha256":plan["worker_seal_sha256"],
        "native_and_mode_task_proof_binding":proof,"additional_BQSA4_arena_bytes":509607936,"legacy_BQSA_arena_retained_bytes":234356736,
        "expected_actual_native_constructor_charge":plan["expected_native_batch_prefill_ctor_bytes"],
        "primary_decode_metric":"exact_native_common_span_median_tokens_per_second",
        "primary_numerator":"actual emitted postfirst tokens, excluding each lane's first emitted bundle",
        "native_clock_provenance":"unchanged nativeclockv5 first-emission/DONE steady timestamps; same worker clock domain; buffered Stop flush; no measured file write",
        "old_extra_EVERYROW_equivalence":False,"old_extra_failed_rows":116,"standard_qualified":False,
        "worker_cancel_deadline_or_public_promotion_qualified":False,"GPU_executed_by_auditor":False,
        "model_or_operand_tensor_payload_read_or_hashed":False,"original_measured_driver_unchanged":True,
        "summary":result["summary"],"waves":waves,"actual_measured_integer_exposure":exposure,"native_audit":result}
    output.write_text(json.dumps(published,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"valid":published["valid"],"output":str(output),"summary":published["summary"]}))
    return 0 if published["valid"] else 1

if __name__=="__main__":raise SystemExit(main())
