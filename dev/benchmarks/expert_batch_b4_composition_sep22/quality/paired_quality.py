#!/usr/bin/env python3
"""Private paired ORIGINAL22 harness: same composed Worker, integer flag0/1.

Root supplies an externally pinned fresh composition binding. This module
executes the unchanged ORIGINAL22 gates and actual composition adapter gates;
no global monkeypatch or original body, budget, grader, or fixture mutation.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
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
TARGET="b87448342df3b3a9ae1642379b8aab513bb208ff234e552bcf70efc26a82c09d"
MIXED_IDS=("arithmetic_multiply","extract_shipment","json_counts_unconstrained","python_sum_even")
RUNTIME=ROOT/"build/integer-b4-twopass-composed-sep22-worker-v2"
POLICY="b191ef4f7636d7700560c46789cdbec1322f93524ae63c93c3fef34d4ed25d07"
INTEGER_SOURCE="edb3b1496296dad42908eb98fec4dc965ddfaaf98316eeadad1390cb5babcf94"
ROLES=("old","new")
INTEGER_MARKER=(
    ";private-physicalR8R16-batch-target-verification-rows4-only"
    ";INTEGERONLY-compact-native-M16-exact-original-float-kernels-six-stages"
    ";planner=expert_batch_r8r16_compact_native_sep22_plan-parallel-SIMDcount"
    ";native-pack-GU-excludedPoison-prepareDown-down-unchanged"
    ";actual-active-lanes2or4-partial3or1-and-rows1to3-original"
    ";component-exact-native-stages-no-model-quality-claim"
    ";sourceSha256="+INTEGER_SOURCE)

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



def is_sha(value):
    return isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None

def load_service(binding):
    path=Path(binding["service_adapter_path"]).resolve()
    require(is_sha(binding.get("service_adapter_sha256")) and sha(path)==binding["service_adapter_sha256"],"Pinned supplemental composition source drift")
    spec=importlib.util.spec_from_file_location("_pinned_original22_composition_service",path)
    require(spec is not None and spec.loader is not None,"Supplemental adapter loader unavailable")
    service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
    require(all(callable(getattr(service,name,None)) for name in ("load_inner","status_errors","coverage")),"Supplemental adapter API differs")
    return service

def load_binding(path,expected_sha):
    path=Path(path).resolve()
    require(is_sha(expected_sha) and sha(path)==expected_sha,"External Root composition binding digest differs")
    binding=http.strict_json(path.read_text())
    require(isinstance(binding,dict),"Complete external composition binding required")
    require(Path(binding.get("runtime_build","")).resolve()==RUNTIME.resolve(),"Same fresh composed Worker required for both roles")
    require(binding.get("policy_source_sha256")==POLICY and binding.get("integer_source_identity_sha256")==INTEGER_SOURCE,"Current B4/integer source identities required")
    require(binding.get("target_numeric_parent_sha256")==TARGET,"Fully wrapped unchanged numerical parent required")
    for key in ("inner_adapter_sha256","target_base_numeric_parent_sha256","target_execution_base_child_sha256","target_execution_child_sha256"):
        require(is_sha(binding.get(key)),"Fresh source/raw/wrapped identity binding missing: "+key)
    require(Path(binding.get("inner_adapter_path","")).resolve()==RUNTIME/"source/dev/benchmarks/expert_batch_b4_composition_sep22/adapter.py","Immutable v2 inner adapter required")
    require(binding.get("frozen_original22_common_status_and_coverage_required") is True,"Actual frozen common gates required")
    service=load_service(binding);inner=service.load_inner(binding)
    require(inner.load_binding(path,expected_sha)==binding,"Fresh immutable source/runtime binding changed during validation")
    return binding

def target_derivative(binding,role):
    require(role in ROLES,"Unknown integer execution role")
    return TARGET if role=="old" else binding["target_execution_child_sha256"]

def target_base_derivative(binding,role):
    require(role in ROLES,"Unknown integer execution role")
    return binding["target_base_numeric_parent_sha256"] if role=="old" else binding["target_execution_base_child_sha256"]

def authenticate(role,binding_path,binding_sha):
    binding=load_binding(binding_path,binding_sha);require(role in ROLES,"Unknown explicit integer execution role")
    return {"role":role,"runtime_build":str(RUNTIME.resolve()),"worker_sha256":binding["worker_sha256"],
        "metallib_sha256":binding["metallib_sha256"],"compiled_seal_sha256":binding["compiled_seal_sha256"],
        "overlay_manifest_sha256":binding["overlay_manifest_sha256"],
        "Root_binding":str(Path(binding_path).resolve()),"Root_binding_sha256":binding_sha,
        "integer_enabled":role=="new","batch_prefill_twopass_enabled":True,
        "target_numerical_derivative_sha256":target_derivative(binding,role),
        "target_base_numerical_derivative_sha256":target_base_derivative(binding,role),
        "no_state_task_or_performance_qualification_inherited":True,
        "old_EXTRA_EVERYROW_equivalence":False,"prior_extra_rows_failed":116}

def composition_identity_errors(status,role,binding):
    if not isinstance(status,dict):return ["Native status must be a complete object"]
    require(role in ROLES,"Unknown integer execution role")
    errors=[];identity=status.get("identity",{});raw=identity.get("batch_prefill_twopass_numerical_parent_routes")
    expected_routes=raw+(INTEGER_MARKER if role=="new" else "") if isinstance(raw,str) and raw else None
    if expected_routes is None or identity.get("kernel_routes")!=expected_routes:
        errors.append("Displayed integer kernel child must wrap the exact raw BQSA Forward parent routes")
    wanted={"identity.target_numerical_derivative_sha256":target_derivative(binding,role),
        "identity.target_base_numerical_derivative_sha256":target_base_derivative(binding,role),
        "compact_native_batch_verify.target_numeric_parent_sha256":TARGET,
        "compact_native_batch_verify.target_base_numeric_parent_sha256":binding["target_base_numeric_parent_sha256"]}
    for path,value in wanted.items():
        if not same(http.get_path(status,path),value):errors.append("Raw/wrapped role execution identity differs: "+path)
    return errors

def status_errors(status,role,plan,store,mode,binding):
    if not isinstance(status,dict):return ["Native status must be a complete object"]
    require(mode=="mtp3" and role in ROLES,"This paired composition admits MTP3 integer0/1 only")
    expected=target_derivative(binding,role)
    errors=[]
    if store.get("target_numerical_derivative_sha256")!=expected:
        errors.append("Common ORIGINAL22 witness must use the actual role's fully wrapped execution derivative")
    # Execute the original common status gate; a declaration cannot replace it.
    errors += original.gate_status(status,plan,store,mode)
    errors += composition_identity_errors(status,role,binding)
    errors += load_service(binding).status_errors(status,binding,mode,True,role=="new")
    return errors

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


def common_coverage(before,after,case,width,role,mode):
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



def coverage(before,after,case,width,role,mode,binding):
    require(mode=="mtp3" and role in ROLES,"This paired composition admits MTP3 integer0/1 only")
    # BYTE-preserved V5 common ledger, with its obsolete B2/B4 new-profile arm
    # skipped locally. The immutable adapter supplies fresh B4/R8/R16 evidence.
    record,errors=common_coverage(before,after,case,width,"old",mode)
    composition,more=load_service(binding).coverage(before,after,case,width,mode,binding,True,role=="new")
    errors += more
    record["composition_route_coverage"]=composition
    record["original22_common_coverage_invoked"]=True
    record["actual_requested_width_proved"]=not errors
    return record,errors

def aggregate_integer_coverage(initial,final,width,role,binding):
    """Require actual campaign construction, allowing zero-work short waves."""
    require(width in (2,4) and role in ROLES,"Unknown aggregate integer width/role")
    errors=[];deltas={};counts={}
    inner=load_service(binding).load_inner(binding)
    for side,status in (("initial",initial),("final",final)):
        errors += [side+": "+error for error in inner.integer_errors(status,binding,role=="new")]
    for rows in (8,16):
        calls=[]
        for stage in ("plan","gate","down"):
            path=f"compact_native_batch_verify.r{rows}.{stage}_graph_calls"
            count=inner.counter_delta(initial,final,path,errors);deltas[path]=count;calls.append(count)
            path=f"compact_native_batch_verify.r{rows}.{stage}_graph_rows"
            physical=inner.counter_delta(initial,final,path,errors);deltas[path]=physical
            if count is not None and physical is not None and (count%48 or physical!=count*rows):
                errors.append("Aggregate integer physical-row/48layer delta differs: "+path)
        if all(count is not None for count in calls):
            if len(set(calls))!=1:errors.append(f"Aggregate R{rows} integer plan/gate/down totals differ")
            counts[rows]=calls[0]
        else:counts[rows]=None
    if role=="old":
        if any(count for count in deltas.values() if count is not None):
            errors.append("Integer0 control campaign encoded integer graph work")
    elif width==2:
        if counts[8] is None or counts[8]<=0:
            errors.append("Integer1 B2 campaign must execute actual physicalR8 integer graphs")
        if any(value for path,value in deltas.items() if ".r16." in path and value is not None):
            errors.append("Integer1 B2 campaign cannot execute physicalR16 integer graphs")
    elif counts[16] is None or counts[16]<=0:
        errors.append("Integer1 B4 campaign must execute actual physicalR16 integer graphs; R8 alone is insufficient")
    return {"valid":not errors,"role":role,"width":width,
        "required_positive_physical_rows":None if role=="old" else 8 if width==2 else 16,
        "actual_integer_graph_calls_by_physical_rows":{"r8":counts[8],"r16":counts[16]},
        "actual_initial_to_final_counter_deltas":deltas,
        "counter_scope":"graph construction across the complete original22 plus mixed campaign; healthy native terminals remain required by common gates",
        "zero_integer_work_per_short_or_schema_wave_allowed":True,"errors":errors},errors

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



def parent_identity(status,role,binding):
    require(not composition_identity_errors(status,role,binding),"Cannot normalize an unbound role execution identity")
    value=copy.deepcopy(status["identity"])
    value.pop("engine_instance_id",None)
    value["kernel_routes"]=value["batch_prefill_twopass_numerical_parent_routes"]
    value["target_numerical_derivative_sha256"]=TARGET
    value["target_base_numerical_derivative_sha256"]=binding["target_base_numeric_parent_sha256"]
    # Both arms retain every BQSA policy/arena/raw-parent identity field.
    return value

def coefficient_store(store,role,binding):
    require(store.get("target_numerical_derivative_sha256")==target_derivative(binding,role),"Saved role execution derivative differs")
    result=copy.deepcopy(store);result.pop("target_numerical_derivative_sha256")
    return result

def send_wave(args,runner,lane_cases,plan,store,binding):
    require(len(lane_cases)==args.width and len({c["prompt_token_count"] for c in lane_cases})==1,"Real equal-row case cohort required")
    require(len({c["body"].get("response_format",{}).get("type")=="json_schema" for c in lane_cases})==1,"Mixed wave must share original MTP eligibility")
    before=runner.wait_idle();wave={"id":lane_cases[0]["id"],"lane_case_ids":[c["id"] for c in lane_cases],
        "status_before":before,"records":[],"errors":[]}
    errors=status_errors(before,args.role,plan,store,args.execution_mode,binding);barrier=threading.Barrier(args.width)
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
    errors.extend(status_errors(after,args.role,plan,store,args.execution_mode,binding));errors.extend(snapshot_errors(runner.initial,before,after))
    detail,more=coverage(before,after,lane_cases[0],args.width,args.role,args.execution_mode,binding);errors.extend(more)
    errors.extend(response_ids_errors(wave["records"],args.width));wave["batch_coverage"]=detail
    wave["errors"]=errors;wave["task_passed_lanes"]=sum(not r["task_errors"] for r in wave["records"])
    wave["evidence_valid"]=not errors;wave["status"]="passed" if not errors and wave["task_passed_lanes"]==args.width else "failed"
    return wave


def measure(args):
    require(args.run_root_gpu,"Model requests require explicit Root-exclusive --run-root-gpu")
    endpoint=urlsplit(args.base_url);require(endpoint.scheme=="http" and endpoint.hostname in ("127.0.0.1","localhost","::1") and endpoint.port and endpoint.port!=8000,"Dedicated loopback quality endpoint")
    require(not args.output.exists(),"Fresh quality report required");plan=frozen_plan(args.plan)
    require(args.execution_mode=="mtp3","This paired composition admits MTP3 only")
    binding=load_binding(args.binding,args.binding_sha256)
    admission=authenticate(args.role,args.binding,args.binding_sha256)
    store=original.store_witness(args.expert_store,512)
    store["target_numerical_derivative_sha256"]=target_derivative(binding,args.role)
    report={"schema":"original22-paired-integer-B4-composition-report-v1","completed":False,"valid":False,
        "role":args.role,"width":args.width,"execution_mode":args.execution_mode,"plan":str(args.plan.resolve()),
        "plan_content_sha256":plan["content_sha256"],"model":args.model,"runtime_admission":admission,"store_witness":store,
        "cases":[],"errors":[],"frozen_plan_common_policy":copy.deepcopy(plan["common_policy"]),
        "batch_wave_policy":{"real_request_concurrency":args.width,"same_unchanged_case_body_per_lane":True,"no_dummy_lanes":True},
        "old_extra_EVERYROW_gate":{"passed":False,"failed_rows":116,"scope":"prior actual B4 layer3 lane0 captured QSA; oldSG8 comparison"},
        "quality_scope":plan["quality_scope"],"rate_or_general_quality_claimed":False}
    original.write_new(args.output,report)
    try:
        runner=http.Qualification(args,report);report["actual_execution_policy"]=original.execution_policy(runner.initial)
        require(not status_errors(runner.initial,args.role,plan,store,args.execution_mode,binding),"Initial native role/source policy admission failed")
        for case in plan["cases"]:
            wave=send_wave(args,runner,[case]*args.width,plan,store,binding)
            report["cases"].append(wave);original.checkpoint(args.output,report)
            print(json.dumps({"id":case["id"],"role":args.role,"width":args.width,"task_passed_lanes":wave["task_passed_lanes"],"coverage_errors":wave["errors"]}),flush=True)
            require(not wave["errors"],"Native width/source/cache/head coverage failed; do not qualify silent fallback")
        selected={c["id"]:c for c in plan["cases"]}
        report["mixed_task_isolation"]=send_wave(args,runner,[selected[i] for i in MIXED_IDS[:args.width]],plan,store,binding)
        require(not report["mixed_task_isolation"]["errors"],"Mixed original-task lane isolation coverage failed")
        report["final_status"]=runner.wait_idle();report["errors"].extend(status_errors(report["final_status"],args.role,plan,store,args.execution_mode,binding))
        aggregate,more=aggregate_integer_coverage(runner.initial,report["final_status"],args.width,args.role,binding)
        report["aggregate_integer_coverage"]=aggregate;report["errors"].extend(more);report["completed"]=True
    except Exception as e:report["errors"].append(type(e).__name__+": "+str(e))
    report["full_plan_coverage"]=report["completed"] and len(report["cases"])==22 and all(len(x["records"])==args.width for x in report["cases"])
    report["evidence_valid"]=report["full_plan_coverage"] and not report["errors"] and all(x["evidence_valid"] for x in report["cases"])
    report["task_passed_lanes"]=sum(x["task_passed_lanes"] for x in report["cases"])
    report["valid"]=report["evidence_valid"] and report["task_passed_lanes"]==22*args.width
    original.checkpoint(args.output,report);print(json.dumps({"report":str(args.output),"evidence_valid":report["evidence_valid"],"task_passed_lanes":report["task_passed_lanes"]}))
    return 0 if report["evidence_valid"] else 1


def audit(report,plan):
    errors=list(report.get("errors",[]));width=report.get("width");role=report.get("role");mode=report.get("execution_mode");model=report.get("model")
    require(width in (2,4) and role in ROLES and mode=="mtp3","Unknown saved composition batch context")
    admission=report["runtime_admission"]
    require(admission==authenticate(role,admission["Root_binding"],admission["Root_binding_sha256"]),"Saved fresh role source/runtime admission differs")
    binding=load_binding(admission["Root_binding"],admission["Root_binding_sha256"])
    require(report["plan_content_sha256"]==plan["content_sha256"],"Saved original plan drift")
    rows=report.get("cases",[]);frozen={c["id"]:c for c in plan["cases"]}
    require(len(rows)==22 and len({x["id"] for x in rows})==22 and {x["id"] for x in rows}==set(frozen),"Every unique original task required")
    initial=report.get("initial_status");require(isinstance(initial,dict),"Saved initial native status required")
    require(original.execution_policy(initial)==report["actual_execution_policy"],"Execution summary differs from actual saved initial policy")
    outcomes={};texts={}
    for key in ("initial_status","final_status"):
        value=report.get(key);errors.extend(status_errors(value,role,plan,report["store_witness"],mode,binding))
        if not http.idle(value):errors.append("Saved terminal status not idle")
        if not isinstance(value,dict) or value.get("identity")!=initial["identity"]:errors.append("Saved terminal identity differs from initial")
        if isinstance(value,dict) and (original.execution_policy(value)!=report["actual_execution_policy"] or
            http.get_path(value,"transport.restarts")!=http.get_path(initial,"transport.restarts")):
            errors.append("Saved terminal policy/restart differs from initial")
    for wave in rows:
        case=frozen[wave["id"]];records=wave.get("records",[]);require(len(records)==width,"All real saved lane responses required")
        errors.extend(wave.get("errors",[]));before,after=wave["status_before"],wave["status_after"]
        for value in (before,after):
            errors.extend(status_errors(value,role,plan,report["store_witness"],mode,binding))
            if not http.idle(value):errors.append("Saved wave status not idle")
        _,more=coverage(before,after,case,width,role,mode,binding);errors.extend(more)
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
        errors.extend(status_errors(value,role,plan,report["store_witness"],mode,binding))
        if not http.idle(value):errors.append("Mixed saved snapshots not idle")
    _,more=coverage(mixed["status_before"],mixed["status_after"],selected[0],width,role,mode,binding);errors.extend(more)
    mixed_outcomes=[];mixed_texts=[]
    for lane,(r,c) in enumerate(zip(records,selected,strict=True)):
        require(r.get("real_lane")==lane and r.get("original_case_id")==c["id"],"Mixed record/case ordering differs")
        mixed_outcomes.append(not exact_record(r,c,model)[0]);mixed_texts.append(r.get("text",""))
    aggregate,more=aggregate_integer_coverage(initial,report["final_status"],width,role,binding);errors.extend(more)
    if report.get("aggregate_integer_coverage")!=aggregate:errors.append("Saved aggregate integer audit differs from actual initial-to-final counters")
    return {"valid":report.get("completed") is True and not errors,"errors":errors,"outcomes":outcomes,"texts":texts,
            "mixed_outcomes":mixed_outcomes,"mixed_texts":mixed_texts,"aggregate_integer_coverage":aggregate}


def compare(args):
    require(not args.output.exists(),"Fresh comparison report");plan=frozen_plan(args.plan)
    reports=[http.strict_json(p.read_text()) for p in (args.old_report,args.new_report)]
    require([r["role"] for r in reports]==["old","new"],"Explicit matched old/new admission required")
    require(reports[0]["width"]==reports[1]["width"] and reports[0]["execution_mode"]==reports[1]["execution_mode"],"Matched width/mode required")
    require(reports[0]["actual_execution_policy"]==reports[1]["actual_execution_policy"],"Context/chunk/head/sampler execution policy differs")
    bindings=[]
    for r in reports:
        a=r["runtime_admission"];require(a==authenticate(r["role"],a["Root_binding"],a["Root_binding_sha256"]),"Saved fresh role source/runtime admission differs")
        bindings.append(load_binding(a["Root_binding"],a["Root_binding_sha256"]))
    require(bindings[0]==bindings[1],"Both roles must use the same fresh composition closure")
    require(reports[0]["model"]==reports[1]["model"],"Matched requested model required")
    require(coefficient_store(reports[0]["store_witness"],"old",bindings[0])==coefficient_store(reports[1]["store_witness"],"new",bindings[1]),"Underlying coefficient/store differs")
    require(parent_identity(reports[0]["initial_status"],"old",bindings[0])==parent_identity(reports[1]["initial_status"],"new",bindings[1]),"Unchanged parent target/head/BQSA identity differs beyond the bound integer execution child")
    audits=[audit(r,plan) for r in reports];regressions=[];differences=[]
    for case in plan["cases"]:
        name=case["id"]
        for lane,(old,new) in enumerate(zip(audits[0]["outcomes"][name],audits[1]["outcomes"][name],strict=True)):
            if old and not new:regressions.append({"id":name,"lane":lane})
            if audits[0]["texts"][name][lane]!=audits[1]["texts"][name][lane]:differences.append({"id":name,"lane":lane})
    for lane,(old,new) in enumerate(zip(audits[0]["mixed_outcomes"],audits[1]["mixed_outcomes"],strict=True)):
        if old and not new:regressions.append({"id":MIXED_IDS[lane],"lane":lane,"scope":"mixed_original_task_wave"})
        if audits[0]["mixed_texts"][lane]!=audits[1]["mixed_texts"][lane]:differences.append({"id":MIXED_IDS[lane],"lane":lane,"scope":"mixed_original_task_wave"})
    result={"schema":"original22-paired-integer-B4-composition-comparison-v1","valid":all(a["valid"] for a in audits),
        "width":reports[0]["width"],"execution_mode":reports[0]["execution_mode"],"original_plan_content_sha256":PLAN,
        "separate_numerical_profiles_admitted":True,"shared_numeric_profile_parity_claimed":False,
        "no_new_task_regressions":not regressions,"new_task_regressions":regressions,"generation_differences":differences,
        "baseline_failed_lanes":[{"id":k,"lane":i} for k,v in audits[0]["outcomes"].items() for i,passed in enumerate(v) if not passed],
        "candidate_failed_lanes":[{"id":k,"lane":i} for k,v in audits[1]["outcomes"].items() for i,passed in enumerate(v) if not passed],
        "evidence_audits":[{"valid":a["valid"],"errors":a["errors"],"aggregate_integer_coverage":a["aggregate_integer_coverage"]} for a in audits],
        "old_extra_EVERYROW_gate_passed":False,"old_extra_failed_rows":116,
        "performance_or_general_quality_claimed":False,"GPU_executed_by_comparator":False}
    original.write_new(args.output,result);print(json.dumps({"valid":result["valid"],"no_new_task_regressions":result["no_new_task_regressions"],"new_task_regressions":regressions}))
    return 0 if result["valid"] and result["no_new_task_regressions"] else 1



def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False);sub=p.add_subparsers(dest="action",required=True)
    m=sub.add_parser("measure",allow_abbrev=False)
    for name in ("plan","binding","expert-store","output"):m.add_argument("--"+name,type=Path,required=True)
    m.add_argument("--binding-sha256",required=True)
    m.add_argument("--role",choices=ROLES,required=True);m.add_argument("--width",type=int,choices=(2,4),required=True)
    m.add_argument("--execution-mode",choices=("mtp3",),required=True)
    m.add_argument("--base-url",required=True);m.add_argument("--model",default=original.MODEL)
    m.add_argument("--timeout",type=float,default=600);m.add_argument("--cleanup-timeout",type=float,default=60)
    m.add_argument("--run-root-gpu",action="store_true")
    c=sub.add_parser("compare",allow_abbrev=False)
    for name in ("plan","old-report","new-report","output"):c.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args(argv);return measure(args) if args.action=="measure" else compare(args)
if __name__=="__main__":raise SystemExit(main())
