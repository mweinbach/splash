#!/usr/bin/env python3
"""Root-only native ORIGINAL22 batch tasks; CPU comparison uses original graders."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT=Path("/Users/mweinbach/Projects/splash")
sys.path.insert(0,str(ROOT))
from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks import prefill4k_attribution_quality as original

PLAN="a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac"
SOURCE="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
LAYOUT="edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0"
LIB="1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf"
POLICY="d87b4a84305ab37d690bf44a742bfbd9fb8737234419dc0edc1326e552a1516a"
SCHEMA="batch-real2or4-allfresh2048-existing-packedV-twopass-v1"
SHADER="6bb23ded512839feda31e04a3812802d6aab5eef6d4b8e33474b794a8247a131"
HOST="1f9f56f5ac77bedaa054fb51d0d3ed8b5d6e12d5eb1d5e8688a5b0fd9fde735e"
TARGET="b87448342df3b3a9ae1642379b8aab513bb208ff234e552bcf70efc26a82c09d"
MIXED_IDS=("arithmetic_multiply","extract_shipment","json_counts_unconstrained","python_sum_even")
ARENA=509607936
TEXT_POLICY="whole real2/4 cohort fresh2048 only; existing singleton QSA packedV floating tree on identical batch projections; original B1/other batch/head arithmetic unchanged; dedicated509607936 arena, legacy234356736 fallback retained"
MARKER=";private-batch-real2or4-allfresh2048-existing-packedV-twopass-v1"
ROLES={
 "old":("build/batch-prefill-restored-teacher-clock-sep22-worker-v1",
        "2a9f674a04b32e0e21a40f95add0cc8c17c313c39c4760342b1b419ae0e43309",
        "4a6fb14387ccfd5897b3cdc86fca0a89bbcf30ca8e42d616a60a40c933e374fc"),
 "new":("build/batch-prefill-twopass-restored-sep22-worker-v3",
        "640c4af9d23aa048a5b3e0d8b1fd1acc02a925d6b32accd09ae2849ac09701fa",
        "50f4a427e9847428ca0ee26af3e7b091a39f4d124ae736e518b4b619991cc291"),
}
COUNTER_SCOPE="arena and encoded fields count graph construction; completed_native_forwards counts healthy API completion after diagnostics/state publication"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def same(a,b):return type(a) is type(b) and a==b
def u64(x):return type(x) is int and 0<=x<2**64
def delta(a,b,path,errors):
    old,new=http.get_path(a,path),http.get_path(b,path)
    if not u64(old) or not u64(new) or new<old:
        errors.append("Missing/invalid/decreasing native counter: "+path);return None
    return new-old
def require(value,message):
    if not value:raise ValueError(message)
def frozen_plan(path):
    plan=original.read_plan(Path(path))
    require(plan["content_sha256"]==PLAN,"Only the unchanged ORIGINAL22 frozen plan is admitted")
    require(len(plan["cases"])==22 and len({c["id"] for c in plan["cases"]})==22,"All22 unique original cases required")
    for c in plan["cases"]:
        require(original.digest({k:v for k,v in c["body"].items() if k!="model"})==c["request_body_sha256_without_model"],"Frozen request body digest differs")
    return plan
def authenticate(role,build,receipt,receipt_sha):
    require(role in ROLES,"Unknown explicit numerical role");expected=ROLES[role];build=Path(build).resolve()
    require(build==(ROOT/expected[0]).resolve(),"Measured runtime must equal the registered role build")
    require(sha(build/"splash-flash")==expected[1] and sha(build/"splash.metallib")==LIB,"Exact registered executable/library required")
    require(sha(build/"compiled-cpu-seal.json")==expected[2],"Registered compiled source closure drift")
    receipt=Path(receipt);require(re.fullmatch("[0-9a-f]{64}",receipt_sha or "") and sha(receipt)==receipt_sha,"Root supplied exact small state receipt pin required")
    r=http.strict_json(receipt.read_text())
    wanted={"schema":"batch-twoPass-intended-state-admission-v1","pass":True,
        "Root_GPU_executed":True,"candidate_source_policy_sha256":POLICY,
        "candidate_worker_sha256":ROLES["new"][1],"metallib_sha256":LIB,
        "proof_capacity":4096,"original22_or_service16K_qualified":False,
        "old_extra_EVERYROW_failed_rows":116}
    require(all(same(r.get(k),v) for k,v in wanted.items()),"Fresh independent intended-state receipt cannot admit this task run")
    for width in (2,4):
        p=r.get("B"+str(width));require(isinstance(p,dict) and p.get("pass") is True and
            p.get("full134state_four_checkpoints_exact") is True and p.get("genuine_greedy_AR3_exact") is True and
            p.get("backend_destroyed") is True and isinstance(p.get("root_report_sha256"),str) and
            re.fullmatch("[0-9a-f]{64}",p["root_report_sha256"]),"Both actual B2/B4 full intended-state proofs are mandatory")
    return {"role":role,"runtime_build":str(build),"worker_sha256":expected[1],"metallib_sha256":LIB,
            "compiled_seal_sha256":expected[2],"Root_state_receipt":str(receipt),"Root_state_receipt_sha256":receipt_sha,
            "numerical_alternative":role=="new","old_EXTRA_EVERYROW_equivalence":False,"prior_extra_rows_failed":116}
def batch_identity(status,role):
    errors=[];identity=status.get("identity",{}) if isinstance(status,dict) else {}
    enabled=role=="new"
    for path,value in {"identity.source":SOURCE,"identity.loaded_model_layout_sha256":LAYOUT,
        "batch_prefill.enabled":True,"batch_prefill.maximum_lanes":4,
        "batch_prefill.maximum_real_rows_per_lane":2048,"maximum_context_tokens":16384,
        "scheduler.maximum_prefill_rows":2048,"scheduler.maximum_batch_prefill_rows_per_lane":2048}.items():
        if not same(http.get_path(status,path),value):errors.append("Registered batch policy differs: "+path)
    fields={"batch_prefill_twopass_requested":enabled,
        "batch_prefill_twopass_schema":SCHEMA if enabled else None,
        "batch_prefill_twopass_policy":TEXT_POLICY if enabled else None,
        "batch_prefill_twopass_source_sha256":POLICY if enabled else None,
        "batch_prefill_twopass_shader_sha256":SHADER if enabled else None,
        "batch_prefill_twopass_host_sha256":HOST if enabled else None,
        "batch_prefill_twopass_arena_plan_bytes":ARENA if enabled else 0}
    if enabled:
        for key,value in fields.items():
            if not same(identity.get(key),value):errors.append("New batch source/numerical scope differs: "+key)
        material="\n".join((identity.get("kernel_routes",""),SCHEMA,TEXT_POLICY,POLICY,SHADER,HOST))
        if identity.get("batch_prefill_twopass_numerical_identity")!=hashlib.sha256(material.encode()).hexdigest():
            errors.append("New batch numerical identity mismatch")
        c=status.get("batch_prefill_twopass_counters")
        if not isinstance(c,dict):return errors+["New batch cumulative counters unavailable"]
        for k,v in {"scope":COUNTER_SCOPE,"constructed_arenas":1,"constructed_arena_bytes":ARENA}.items():
            if not same(c.get(k),v):errors.append("New arena ownership/counter scope differs: "+k)
        for k in ("encoded_QSA_lane_calls","encoded_QSA_lane_layer_calls","completed_native_forwards"):
            if not u64(c.get(k)):errors.append("New cumulative counter invalid: "+k)
        if u64(c.get("encoded_QSA_lane_calls")) and c.get("encoded_QSA_lane_layer_calls")!=c["encoded_QSA_lane_calls"]:
            errors.append("New lane/layer cumulative counts disagree")
        if all(u64(c.get(k)) for k in ("encoded_QSA_lane_calls","completed_native_forwards")):
            calls,forwards=c["encoded_QSA_lane_calls"],c["completed_native_forwards"]
            if calls%24 or not 24*forwards<=calls<=48*forwards:
                errors.append("New cumulative eligible2/4 cohort counts disagree")
    else:
        # Old compiled worker predates the additive fields; any partial/new profile is refused.
        if any(k in identity for k in fields) or "batch_prefill_twopass_counters" in status:
            errors.append("Old baseline unexpectedly carries a new numerical profile")
    routes=identity.get("batch_prefill_kernel_routes")
    if not isinstance(routes,str) or routes.count(MARKER)!=int(enabled):
        errors.append("Batch numerical route marker mismatch")
    return errors
def status_errors(status,role,plan,store,mode):
    if not isinstance(status,dict):return ["Native status must be a complete object"]
    return original.gate_status(status,plan,store,mode)+batch_identity(status,role)
def schedule(case,width,mode):
    prompt=case["prompt_token_count"];windows=[min(2048,prompt-start) for start in range(0,prompt,2048)]
    eligible=mode=="mtp3" and case["body"].get("response_format",{}).get("type")!="json_schema"
    pair_windows=[min(n,prompt-start-1) for start,n in zip(range(0,prompt,2048),windows,strict=True)] if eligible else [0]*len(windows)
    # Actual selectedHeadPrimePositions groups equal128 or tiny1..4, never127.
    grouped=sum(n//128 for n in pair_windows)+sum(0<n%128<=4 for n in pair_windows)
    scalar=sum(n%128>4 for n in pair_windows)*width
    grouped_pairs=width*sum((n//128)*128+(n%128 if 0<n%128<=4 else 0) for n in pair_windows)
    return {"windows":windows,"first_window_new":windows[0]==2048,"eligible":width if eligible else 0,
        "grouped_calls":grouped,"grouped_lanes":grouped*width,"grouped_pairs":grouped_pairs,
        "scalar_calls":scalar,"true_pairs":width*(prompt-1) if eligible else 0,
        "hidden_bytes":width*prompt*10240*2 if eligible else 0}
def coverage(before,after,case,width,role,mode):
    errors=[];expected=schedule(case,width,mode);observed={}
    wanted={"requests.submitted":width,"requests.completed":width,"requests.cancelled":0,"requests.failed":0,
        "metrics.prefill_input_tokens":width*case["prompt_token_count"],
        "scheduler.prefill_rows":width*case["prompt_token_count"],
        "scheduler.prefill_batches":len(expected["windows"]),"batch_prefill.members_dropped_at_control_boundaries":0,
        "mtp.eligible_requests":expected["eligible"],"mtp.teacher_cache_only_priming_calls":expected["scalar_calls"],
        "mtp.batch_teacher_cache_only_priming_calls":expected["grouped_calls"],
        "mtp.batch_teacher_cache_only_completed_lanes":expected["grouped_lanes"],
        "mtp.batch_teacher_cache_only_completed_real_pairs":expected["grouped_pairs"],
        "scheduler.batch_mtp_priming_batches":expected["grouped_calls"],
        "scheduler.batch_mtp_priming_real_rows":expected["grouped_pairs"],
        "batch_prefill.true_target_hidden_copied_bytes":expected["hidden_bytes"]}
    for w in (1,2,3,4):
        wanted["scheduler.prefill_batches_by_width.b"+str(w)]=len(expected["windows"]) if w==width else 0
        wanted["scheduler.batch_mtp_priming_batches_by_width.b"+str(w)]=expected["grouped_calls"] if w==width else 0
    large=[n for n in expected["windows"] if width*n>=256]
    calls=48*len(large);physical=48*width*sum(large)
    for key,value in {"gate_up_graph_calls":calls,"down_graph_calls":calls,"gate_up_graph_rows":physical,
        "down_graph_rows":physical,"encoded_hit_dispatches":2*calls,"encoded_miss_dispatches":0,
        "full_inventory_graph_calls":2*calls}.items():
        wanted["persisted_experts.graph_counters.large_row_"+key]=value
    if role=="new":
        wanted.update({"batch_prefill_twopass_counters.encoded_QSA_lane_calls":12*width if expected["first_window_new"] else 0,
            "batch_prefill_twopass_counters.encoded_QSA_lane_layer_calls":12*width if expected["first_window_new"] else 0,
            "batch_prefill_twopass_counters.completed_native_forwards":1 if expected["first_window_new"] else 0})
    for path,value in wanted.items():
        got=delta(before,after,path,errors);observed[path]=got
        if got is not None and got!=value:errors.append(f"Real native wave counter differs: {path}: {got} != {value}")
    for path in ("metrics.metal_failures",):
        got=delta(before,after,path,errors);observed[path]=got
        if got:errors.append("Native metal failure during task wave")
    for field in original.EXECUTION_POLICY_FIELDS:
        if not http.same_json(http.get_path(before,field),http.get_path(after,field)):errors.append("Execution policy changed: "+field)
    if before.get("identity")!=after.get("identity"):errors.append("Runtime/source/ownership identity changed during wave")
    return {"expected":expected,"native_counter_deltas":observed,"actual_requested_width_proved":not errors},errors
def exact_record(record,case,model):
    body=record.get("request_body")
    require(isinstance(body,dict) and original.digest({k:v for k,v in body.items() if k!="model"})==case["request_body_sha256_without_model"],"Saved lane body differs from ORIGINAL22")
    require(body.get("model")==model,"Saved requested model differs")
    require(record.get("prompt_u32le_sha256")==case["prompt_u32le_sha256"],"Frozen prompt identity differs")
    return original.grade_record(record,case)
def snapshot_errors(initial,before,after):
    errors=[]
    for side,value in (("before",before),("after",after)):
        if not isinstance(value,dict):errors.append(side+": missing saved native snapshot");continue
        if value.get("identity")!=initial.get("identity"):errors.append(side+": runtime/source identity differs from initial")
        if original.execution_policy(value)!=original.execution_policy(initial):errors.append(side+": execution policy differs from initial")
        for path in ("transport.restarts",):
            if not u64(http.get_path(value,path)) or http.get_path(value,path)!=http.get_path(initial,path):errors.append(side+": restart identity changed or unavailable")
    a,b=(http.get_path(s,"status_snapshot.steady_seconds") for s in (before,after))
    if type(a) not in (int,float) or type(b) not in (int,float) or not (0<=a<b<1e15):errors.append("Saved after snapshot is not fresh")
    return errors
def response_ids_errors(records,width):
    ids=[r.get("request_ids") for r in records]
    if any(not isinstance(x,list) or len(x)!=1 or not isinstance(x[0],str) or not x[0] for x in ids):
        return ["Every real lane needs one unambiguous HTTP response ID"]
    if len({x[0] for x in ids})!=width:return ["Real lane HTTP response IDs must be distinct"]
    return []
def parent_identity(status):
    value=copy.deepcopy(status["identity"])
    # A fresh worker has a new process instance. Per-run binding remains exact.
    value.pop("engine_instance_id",None)
    for key in ("batch_prefill_twopass_requested","batch_prefill_twopass_schema","batch_prefill_twopass_policy",
        "batch_prefill_twopass_source_sha256","batch_prefill_twopass_shader_sha256","batch_prefill_twopass_host_sha256",
        "batch_prefill_twopass_numerical_identity","batch_prefill_twopass_arena_plan_bytes"):
        value.pop(key,None)
    routes=value.get("batch_prefill_kernel_routes")
    if isinstance(routes,str):value["batch_prefill_kernel_routes"]=routes.replace(MARKER,"")
    return value
def send_wave(args,runner,lane_cases,plan,store):
    require(len(lane_cases)==args.width and len({c["prompt_token_count"] for c in lane_cases})==1,"Real equal-row case cohort required")
    require(len({c["body"].get("response_format",{}).get("type")=="json_schema" for c in lane_cases})==1,"Mixed wave must share original MTP eligibility")
    before=runner.wait_idle();wave={"id":lane_cases[0]["id"],"lane_case_ids":[c["id"] for c in lane_cases],
        "status_before":before,"records":[],"errors":[]}
    errors=status_errors(before,args.role,plan,store,args.execution_mode);barrier=threading.Barrier(args.width)
    def lane(index):
        case=lane_cases[index];body=copy.deepcopy(case["body"]);body["model"]=args.model
        client=http.HTTPClient(args);barrier.wait(timeout=30);record=client.send("POST","/v1/chat/completions",body)
        record.update(expected_model=args.model,cache_disabled=runner.cache_disabled,request_body=body,
            prompt_u32le_sha256=case["prompt_u32le_sha256"],
            request_body_sha256_without_model=case["request_body_sha256_without_model"],real_lane=index,original_case_id=case["id"])
        task_errors,details=exact_record(record,case,args.model);record["task_errors"]=task_errors
        if details is not None:record["python_grading"]=details
        return record
    with ThreadPoolExecutor(max_workers=args.width) as pool:wave["records"]=list(pool.map(lane,range(args.width)))
    after=runner.wait_idle(after_snapshot=http.get_path(before,"status_snapshot.steady_seconds"));wave["status_after"]=after
    errors.extend(status_errors(after,args.role,plan,store,args.execution_mode));errors.extend(snapshot_errors(runner.initial,before,after))
    detail,more=coverage(before,after,lane_cases[0],args.width,args.role,args.execution_mode);errors.extend(more)
    errors.extend(response_ids_errors(wave["records"],args.width));wave["batch_coverage"]=detail
    wave["errors"]=errors;wave["task_passed_lanes"]=sum(not r["task_errors"] for r in wave["records"])
    wave["evidence_valid"]=not errors;wave["status"]="passed" if not errors and wave["task_passed_lanes"]==args.width else "failed"
    return wave
def measure(args):
    require(args.run_root_gpu,"Model requests require explicit Root-exclusive --run-root-gpu")
    endpoint=urlsplit(args.base_url);require(endpoint.scheme=="http" and endpoint.hostname in ("127.0.0.1","localhost","::1") and endpoint.port and endpoint.port!=8000,"Dedicated loopback quality endpoint")
    require(not args.output.exists(),"Fresh quality report required");plan=frozen_plan(args.plan)
    admission=authenticate(args.role,args.runtime_build,args.state_receipt,args.state_receipt_sha256)
    store=original.store_witness(args.expert_store,512);require(args.target_derivative_sha256==TARGET,"Exact unchanged registered target numerical derivative required")
    store["target_numerical_derivative_sha256"]=args.target_derivative_sha256
    report={"schema":"original22-native-batch-numerical-alternative-report-v1","completed":False,"valid":False,
        "role":args.role,"width":args.width,"execution_mode":args.execution_mode,"plan":str(args.plan.resolve()),
        "plan_content_sha256":plan["content_sha256"],"model":args.model,"runtime_admission":admission,"store_witness":store,
        "cases":[],"errors":[],"frozen_plan_common_policy":copy.deepcopy(plan["common_policy"]),
        "batch_wave_policy":{"real_request_concurrency":args.width,"same_unchanged_case_body_per_lane":True,"no_dummy_lanes":True},
        "old_extra_EVERYROW_gate":{"passed":False,"failed_rows":116,"scope":"prior actual B4 layer3 lane0 captured QSA; oldSG8 comparison"},
        "quality_scope":plan["quality_scope"],"rate_or_general_quality_claimed":False}
    original.write_new(args.output,report)
    try:
        runner=http.Qualification(args,report);report["actual_execution_policy"]=original.execution_policy(runner.initial)
        require(not status_errors(runner.initial,args.role,plan,store,args.execution_mode),"Initial native role/source policy admission failed")
        for case in plan["cases"]:
            wave=send_wave(args,runner,[case]*args.width,plan,store)
            report["cases"].append(wave);original.checkpoint(args.output,report)
            print(json.dumps({"id":case["id"],"role":args.role,"width":args.width,"task_passed_lanes":wave["task_passed_lanes"],"coverage_errors":wave["errors"]}),flush=True)
            require(not wave["errors"],"Native width/source/cache/head coverage failed; do not qualify silent fallback")
        selected={c["id"]:c for c in plan["cases"]}
        report["mixed_task_isolation"]=send_wave(args,runner,[selected[i] for i in MIXED_IDS[:args.width]],plan,store)
        require(not report["mixed_task_isolation"]["errors"],"Mixed original-task lane isolation coverage failed")
        report["final_status"]=runner.wait_idle();report["errors"].extend(status_errors(report["final_status"],args.role,plan,store,args.execution_mode));report["completed"]=True
    except Exception as e:report["errors"].append(type(e).__name__+": "+str(e))
    report["full_plan_coverage"]=report["completed"] and len(report["cases"])==22 and all(len(x["records"])==args.width for x in report["cases"])
    report["evidence_valid"]=report["full_plan_coverage"] and not report["errors"] and all(x["evidence_valid"] for x in report["cases"])
    report["task_passed_lanes"]=sum(x["task_passed_lanes"] for x in report["cases"])
    report["valid"]=report["evidence_valid"] and report["task_passed_lanes"]==22*args.width
    original.checkpoint(args.output,report);print(json.dumps({"report":str(args.output),"evidence_valid":report["evidence_valid"],"task_passed_lanes":report["task_passed_lanes"]}))
    return 0 if report["evidence_valid"] else 1
def audit(report,plan):
    errors=list(report.get("errors",[]));width=report.get("width");role=report.get("role");mode=report.get("execution_mode");model=report.get("model")
    require(width in (2,4) and role in ROLES and mode in ("standard","mtp3"),"Unknown saved batch context")
    require(report["plan_content_sha256"]==plan["content_sha256"],"Saved original plan drift")
    rows=report.get("cases",[]);frozen={c["id"]:c for c in plan["cases"]}
    require(len(rows)==22 and len({x["id"] for x in rows})==22 and {x["id"] for x in rows}==set(frozen),"Every unique original task required")
    initial=report.get("initial_status");require(isinstance(initial,dict),"Saved initial native status required")
    require(original.execution_policy(initial)==report["actual_execution_policy"],"Execution summary differs from actual saved initial policy")
    outcomes={};texts={}
    for key in ("initial_status","final_status"):
        value=report.get(key);errors.extend(status_errors(value,role,plan,report["store_witness"],mode))
        if not http.idle(value):errors.append("Saved terminal status not idle")
        if not isinstance(value,dict) or value.get("identity")!=initial["identity"]:errors.append("Saved terminal identity differs from initial")
        if isinstance(value,dict) and (original.execution_policy(value)!=report["actual_execution_policy"] or
            http.get_path(value,"transport.restarts")!=http.get_path(initial,"transport.restarts")):
            errors.append("Saved terminal policy/restart differs from initial")
    for wave in rows:
        case=frozen[wave["id"]];records=wave.get("records",[]);require(len(records)==width,"All real saved lane responses required")
        errors.extend(wave.get("errors",[]));before,after=wave["status_before"],wave["status_after"]
        for value in (before,after):
            errors.extend(status_errors(value,role,plan,report["store_witness"],mode))
            if not http.idle(value):errors.append("Saved wave status not idle")
        _,more=coverage(before,after,case,width,role,mode);errors.extend(more)
        errors.extend(snapshot_errors(initial,before,after));errors.extend(response_ids_errors(records,width))
        require(sorted(r.get("real_lane") for r in records)==list(range(width)),"Each real lane appears exactly once")
        require(wave.get("lane_case_ids")==[case["id"]]*width and all(r.get("original_case_id")==case["id"] for r in records),"Saved real lane case association differs")
        outcomes[case["id"]]=[not exact_record(r,case,model)[0] for r in records];texts[case["id"]]=[r.get("text","") for r in records]
    mixed=report.get("mixed_task_isolation");require(isinstance(mixed,dict),"Bounded mixed original-task wave required")
    selected=[frozen[i] for i in MIXED_IDS[:width]];require(mixed.get("lane_case_ids")==[c["id"] for c in selected],"Mixed task lane association differs")
    records=mixed.get("records",[]);require(len(records)==width,"All mixed real lane responses required")
    errors.extend(mixed.get("errors",[]));errors.extend(snapshot_errors(initial,mixed["status_before"],mixed["status_after"]))
    errors.extend(response_ids_errors(records,width))
    for value in (mixed["status_before"],mixed["status_after"]):
        errors.extend(status_errors(value,role,plan,report["store_witness"],mode))
        if not http.idle(value):errors.append("Mixed saved snapshots not idle")
    _,more=coverage(mixed["status_before"],mixed["status_after"],selected[0],width,role,mode);errors.extend(more)
    mixed_outcomes=[];mixed_texts=[]
    for lane,(r,c) in enumerate(zip(records,selected,strict=True)):
        require(r.get("real_lane")==lane and r.get("original_case_id")==c["id"],"Mixed record/case ordering differs")
        mixed_outcomes.append(not exact_record(r,c,model)[0]);mixed_texts.append(r.get("text",""))
    return {"valid":report.get("completed") is True and not errors,"errors":errors,"outcomes":outcomes,"texts":texts,
            "mixed_outcomes":mixed_outcomes,"mixed_texts":mixed_texts}
def compare(args):
    require(not args.output.exists(),"Fresh comparison report");plan=frozen_plan(args.plan)
    reports=[http.strict_json(p.read_text()) for p in (args.old_report,args.new_report)]
    require([r["role"] for r in reports]==["old","new"],"Explicit matched old/new admission required")
    require(reports[0]["width"]==reports[1]["width"] and reports[0]["execution_mode"]==reports[1]["execution_mode"],"Matched width/mode required")
    require(reports[0]["actual_execution_policy"]==reports[1]["actual_execution_policy"],"Context/chunk/head/sampler execution policy differs")
    require(reports[0]["store_witness"]==reports[1]["store_witness"],"Underlying coefficient/store/target derivative differs")
    require(parent_identity(reports[0]["initial_status"])==parent_identity(reports[1]["initial_status"]),"Unchanged parent target/head/kernel identity differs beyond documented batch alternative")
    for r in reports:
        a=r["runtime_admission"];authenticate(r["role"],a["runtime_build"],a["Root_state_receipt"],a["Root_state_receipt_sha256"])
    audits=[audit(r,plan) for r in reports];regressions=[];differences=[]
    for case in plan["cases"]:
        name=case["id"]
        for lane,(old,new) in enumerate(zip(audits[0]["outcomes"][name],audits[1]["outcomes"][name],strict=True)):
            if old and not new:regressions.append({"id":name,"lane":lane})
            if audits[0]["texts"][name][lane]!=audits[1]["texts"][name][lane]:differences.append({"id":name,"lane":lane})
    for lane,(old,new) in enumerate(zip(audits[0]["mixed_outcomes"],audits[1]["mixed_outcomes"],strict=True)):
        if old and not new:regressions.append({"id":MIXED_IDS[lane],"lane":lane,"scope":"mixed_original_task_wave"})
        if audits[0]["mixed_texts"][lane]!=audits[1]["mixed_texts"][lane]:differences.append({"id":MIXED_IDS[lane],"lane":lane,"scope":"mixed_original_task_wave"})
    result={"schema":"original22-native-batch-paired-comparison-v1","valid":all(a["valid"] for a in audits),
        "width":reports[0]["width"],"execution_mode":reports[0]["execution_mode"],"original_plan_content_sha256":PLAN,
        "separate_numerical_profiles_admitted":True,"shared_numeric_profile_parity_claimed":False,
        "no_new_task_regressions":not regressions,"new_task_regressions":regressions,"generation_differences":differences,
        "baseline_failed_lanes":[{"id":k,"lane":i} for k,v in audits[0]["outcomes"].items() for i,passed in enumerate(v) if not passed],
        "candidate_failed_lanes":[{"id":k,"lane":i} for k,v in audits[1]["outcomes"].items() for i,passed in enumerate(v) if not passed],
        "evidence_audits":[{"valid":a["valid"],"errors":a["errors"]} for a in audits],
        "old_extra_EVERYROW_gate_passed":False,"old_extra_failed_rows":116,
        "performance_or_general_quality_claimed":False,"GPU_executed_by_comparator":False}
    original.write_new(args.output,result);print(json.dumps({"valid":result["valid"],"no_new_task_regressions":result["no_new_task_regressions"],"new_task_regressions":regressions}))
    return 0 if result["valid"] and result["no_new_task_regressions"] else 1
def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False);sub=p.add_subparsers(dest="action",required=True)
    m=sub.add_parser("measure",allow_abbrev=False)
    for name in ("plan","runtime-build","expert-store","state-receipt","output"):m.add_argument("--"+name,type=Path,required=True)
    m.add_argument("--role",choices=ROLES,required=True);m.add_argument("--width",type=int,choices=(2,4),required=True)
    m.add_argument("--execution-mode",choices=("standard","mtp3"),required=True);m.add_argument("--state-receipt-sha256",required=True)
    m.add_argument("--target-derivative-sha256",required=True);m.add_argument("--base-url",required=True)
    m.add_argument("--model",default=original.MODEL);m.add_argument("--timeout",type=float,default=600)
    m.add_argument("--cleanup-timeout",type=float,default=60);m.add_argument("--run-root-gpu",action="store_true")
    c=sub.add_parser("compare",allow_abbrev=False)
    for name in ("plan","old-report","new-report","output"):c.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args(argv);return measure(args) if args.action=="measure" else compare(args)
if __name__=="__main__":raise SystemExit(main())
