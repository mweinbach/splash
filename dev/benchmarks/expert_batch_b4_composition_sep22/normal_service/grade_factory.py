#!/usr/bin/env python3
"""Root-only actual-report regrade; emits one tiny task admission, never inference.

Importing this module reads nothing. Actual reports, fixtures, proof receipts,
and their digests are accessed only after explicit --run-root-reports.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT=Path("/Users/mweinbach/Projects/splash")
TASK=ROOT/"build/current-BQSA4-integer-paired-original22-MTP3-sep22-v2"
SUMMARY=ROOT/"build/release/flash/sep22-current-BQSA4-integer-paired-original22-MTP3-suite-v2.json"
SCHEMA="current-BQSA4-integer-MTP3-B2-B4-actual-original22-grade-admission-v1"
PINS_SCHEMA="current-BQSA4-integer-taskV2-Root-actual-report-pins-v1"
WORKER="ca7bc795bfb42b50949f1eaf1f035620ec3bf1b861ab397ed9a166ccc3202e68"
LIB="74af1228995f38890df035555b2000018899783fd46ad3a12f8a835c833231c8"
SOURCE="edb3b1496296dad42908eb98fec4dc965ddfaaf98316eeadad1390cb5babcf94"
POLICY="b191ef4f7636d7700560c46789cdbec1322f93524ae63c93c3fef34d4ed25d07"
SEAL="32338e764d23627d0bfb74f12393c9aa84a0e4aaa37b2ec1da520c3c381f7461"
STAGE="7f82c4592d08d7d317b1b7a3d3fa21e7dbb84e61f8e84b1ca68beae629a327cd"
NATIVE_SEAL="80d7618e4b002a077ddfc2e9f801e6f5944e5bf40d5bfb61f77b29a5162b55d5"
PLAN="a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac"
TASK_COMMAND="20e4ebad4dea7e2c1bf5f6bba1746a91b1af82e4a76184e23aa74451980f1f30"
TASK_READY="de99d0c9ffc93a01961691dfd63d2d10a16c21fb18a89d7ab6897c8a477b1fd6"
TASK_CONFIG="def2607122e3b2752e8671c8762296de305ef052d00ea3a9488e379e8cf5b0a3"
TASK_BINDING="dff527c9c8ea60dbaa64b759ccf2328feda0e0f9aba78d4c189ba03d75d13d82"
QUALITY_SOURCE="0b8890e28f4cdde5b5e472c04c4c39c52119bfa8f240107d98ad2309e40e474e"
LAUNCHER_SOURCE="f0043ab8823ae644b168e6b51d9bd50e949ac54780b9039469a172ade47e13b5"
RAW_PARENT="04b42a9f509d23dc0de5087ba7853812497dcd15c20bcf2df0352eedc9bb1a80"
WRAPPED_PARENT="b87448342df3b3a9ae1642379b8aab513bb208ff234e552bcf70efc26a82c09d"
RAW_CHILD="b40b095ec17517f1b014738a72dc0d37a2d11dd0499a40875e645f71d8e7764f"
WRAPPED_CHILD="37b89da17689fd1b7f3b61ed3db0c14f743270e16c4dbe7f2803ea918878dad0"
INTEGER_FLAG="SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22"
BQSA_FLAG="SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22"

def require(value,message):
    if not value:raise ValueError(message)
def same(a,b):return type(a) is type(b) and a==b
def is_sha(value):return isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def get(value,path):
    for key in path.split("."):
        if not isinstance(value,dict):return None
        value=value.get(key)
    return value
def exact(value,fields,message):
    require(isinstance(value,dict) and all(same(value.get(k),v) for k,v in fields.items()),message)
def canonical_path(value):
    require(isinstance(value,str) and value,"Actual artifact reference path required")
    path=Path(value)
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,"Frozen source module unavailable")
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value

def artifact_paths():
    return {"summary":SUMMARY,
        **{f"comparison.B{width}":SUMMARY.with_name(SUMMARY.stem+f"-mtp3-B{width}.comparison.json") for width in (4,2)},
        **{f"report.{role}.B{width}":SUMMARY.with_name(SUMMARY.stem+f"-{role}-mtp3-B{width}.json") for role in ("old","new") for width in (4,2)}}

def validate_actual_pins(pins,summary_path,summary_sha):
    require(Path(summary_path).resolve()==SUMMARY,"Only the active canonical taskV2 campaign admitted")
    exact(pins,{"schema":PINS_SCHEMA,"task_root_command_sha256":TASK_COMMAND,
        "task_CPU_READY_sha256":TASK_READY,"task_config_sha256":TASK_CONFIG,
        "task_binding_sha256":TASK_BINDING},"Current externally pinned taskV2 manifest required")
    files=pins.get("files");expected={str(p) for p in artifact_paths().values()}
    require(isinstance(files,dict) and set(files)==expected and all(is_sha(s) for s in files.values()),"Exact summary, both comparisons, and four reports required")
    require(is_sha(summary_sha) and files[str(SUMMARY)]==summary_sha,"External summary SHA differs from actual artifact manifest")
    return files

def validate_binding(binding):
    exact(binding,{"worker_sha256":WORKER,"metallib_sha256":LIB,"compiled_seal_sha256":SEAL,
        "integer_source_identity_sha256":SOURCE,"policy_source_sha256":POLICY,
        "target_base_numeric_parent_sha256":RAW_PARENT,"target_numeric_parent_sha256":WRAPPED_PARENT,
        "target_execution_base_child_sha256":RAW_CHILD,"target_execution_child_sha256":WRAPPED_CHILD},
        "Current source-derived raw/wrapped execution identities required")

def validate_summary(summary,config,launcher,files):
    exact(summary,{"schema":"current-BQSA4-integer-MTP3-paired-original22-summary-v1",
        "completed":True,"valid":True,"GPU_executed":True,"execution_modes":["mtp3"],
        "standard_qualified":False,"native_context":16384,"binding_sha256":TASK_BINDING,
        "native_stage_receipt_sha256":STAGE,"same_worker_integer_only_control_delta":True,
        "original_plan_content_sha256":PLAN,"performance_qualified":False,
        "all_actual_evidence_valid":True,"no_new_task_regressions_at_both_widths":True,
        "old_extra_EVERYROW_gate_passed":False,"old_extra_failed_rows":116,"errors":[]},
        "Completed current paired MTP3 summary required; no standalone/standard/old proof inheritance")
    runs=summary.get("server_runs")
    require(isinstance(runs,list) and [r.get("role") for r in runs]==["old","new"],"Both ordered actual integer0/1 workers required")
    resolved=[]
    build=Path(config["runtime_build"]) if "runtime_build" in config else ROOT/"build/integer-b4-twopass-composed-sep22-worker-v2"
    for role,run in zip(("old","new"),runs,strict=True):
        exact(run,{"role":role,"execution_mode":"mtp3","integer_enabled":role=="new","unloaded":True},"Wrong actual worker role/mode/unload")
        require(run.get("command")==launcher.server_command(config,build,role),"Actual server command differs from canonical taskV2 role")
        evidence=run.get("unload_evidence")
        exact(evidence,{"process_group_gone":True,"returncode":0,"post_parent_exit_sigkill_required":False},"Both workers must exit0 without SIGKILL and leave no process group")
        require(type(run.get("pid")) is int and evidence.get("process_group")==run["pid"],"Unload must belong to the actual server process group")
        env=get(run,"environment.resolved_flash_environment")
        require(isinstance(env,dict) and all(type(k) is str and type(v) is str for k,v in env.items()),"Literal actual resolved flash flags required")
        expected=launcher.role_environment(config,role)
        require(all(env.get(k)==v for k,v in expected.items()),"Actual resolved task flags differ from canonical role flags")
        require(env.get(BQSA_FLAG)=="1" and env.get(INTEGER_FLAG)==("1" if role=="new" else "0"),"Both BQSA1 and only integer0/1 required")
        entries=run.get("quality_reports")
        require(isinstance(entries,list) and [r.get("width") for r in entries]==[4,2],"Both actual native report widths required")
        for width,row in zip((4,2),entries,strict=True):
            path=artifact_paths()[f"report.{role}.B{width}"]
            require(canonical_path(row.get("path"))==path and row.get("sha256")==files[str(path)],"Actual role report reference/SHA differs")
        resolved.append(env)
    changed={k for k in set(resolved[0])|set(resolved[1]) if resolved[0].get(k)!=resolved[1].get(k)}
    require(changed=={INTEGER_FLAG},"Integer flag must be the ONLY actual resolved role difference")
    comparisons=summary.get("comparisons")
    require(isinstance(comparisons,list) and [r.get("width") for r in comparisons]==[4,2],"Both ordered actual paired comparisons required")
    for width,row in zip((4,2),comparisons,strict=True):
        path=artifact_paths()[f"comparison.B{width}"]
        exact(row,{"width":width,"execution_mode":"mtp3","valid":True,"no_new_task_regressions":True,
            "sha256":files[str(path)]},"Actual comparison reference/SHA/no-regression differs")
        require(canonical_path(row.get("path"))==path,"Actual comparison path differs from canonical current campaign")
    return runs

def validate_aggregate(audit,width,role):
    require(audit.get("valid") is True and audit.get("errors")==[],"Frozen ORIGINAL22/common/aggregate audit failed")
    aggregate=audit.get("aggregate_integer_coverage")
    exact(aggregate,{"valid":True,"width":width,"role":role,"errors":[]},"Actual campaign aggregate required")
    deltas=aggregate.get("actual_initial_to_final_counter_deltas");require(isinstance(deltas,dict),"Actual aggregate deltas required")
    totals={}
    for rows in (8,16):
        counts=[]
        for stage in ("plan","gate","down"):
            calls=deltas.get(f"compact_native_batch_verify.r{rows}.{stage}_graph_calls")
            physical=deltas.get(f"compact_native_batch_verify.r{rows}.{stage}_graph_rows")
            require(type(calls) is int and type(physical) is int and 0<=calls<2**64 and 0<=physical<2**64 and calls%48==0 and physical==calls*rows,"Actual aggregate integer 48layer/physical-row evidence invalid")
            counts.append(calls)
        require(len(set(counts))==1,"Actual aggregate plan/gate/down totals differ");totals[rows]=counts[0]
    require(aggregate.get("actual_integer_graph_calls_by_physical_rows")=={"r8":totals[8],"r16":totals[16]},"Actual aggregate summary differs from counters")
    if role=="old":require(totals=={8:0,16:0},"Integer0 control encoded integer work")
    elif width==2:require(totals[8]>0 and totals[16]==0,"Actual new B2 requires positive R8 and zero R16")
    else:require(totals[16]>0,"Actual new B4 requires positive R16; R8 alone is insufficient")
    return totals

def paired_outcomes(quality,plan,left,right):
    regressions=[];differences=[]
    for case in plan["cases"]:
        name=case["id"]
        for lane,(old,new) in enumerate(zip(left["outcomes"][name],right["outcomes"][name],strict=True)):
            if old and not new:regressions.append({"id":name,"lane":lane})
            if left["texts"][name][lane]!=right["texts"][name][lane]:differences.append({"id":name,"lane":lane})
    for lane,(old,new) in enumerate(zip(left["mixed_outcomes"],right["mixed_outcomes"],strict=True)):
        if old and not new:regressions.append({"id":quality.MIXED_IDS[lane],"lane":lane,"scope":"mixed_original_task_wave"})
        if left["mixed_texts"][lane]!=right["mixed_texts"][lane]:differences.append({"id":quality.MIXED_IDS[lane],"lane":lane,"scope":"mixed_original_task_wave"})
    return regressions,differences

def reaudit_pairs(quality,plan,binding,reports,comparisons,files):
    result={}
    for width in (4,2):
        left,right=[reports[(role,width)] for role in ("old","new")]
        for role,report in zip(("old","new"),(left,right),strict=True):
            exact(report,{"schema":"original22-paired-integer-B4-composition-report-v1","completed":True,
                "role":role,"width":width,"execution_mode":"mtp3","plan_content_sha256":PLAN},"Actual current role/width task report required")
            require(report["runtime_admission"]==quality.authenticate(role,TASK/"paired-composition-binding.json",TASK_BINDING),"Actual task runtime admission differs from current source closure")
        require(left["model"]==right["model"] and left["actual_execution_policy"]==right["actual_execution_policy"],"Actual request/model/execution policy differs")
        require(quality.coefficient_store(left["store_witness"],"old",binding)==quality.coefficient_store(right["store_witness"],"new",binding),"Actual coefficient/store mismatch")
        require(quality.parent_identity(left["initial_status"],"old",binding)==quality.parent_identity(right["initial_status"],"new",binding),"Actual unchanged parent/BQSA policy mismatch")
        # These immutable calls regrade ORIGINAL22 plus mixed responses and rerun
        # original common status/coverage, per-run identity, and actual counters.
        audits=[quality.audit(report,plan) for report in (left,right)]
        old,new=[validate_aggregate(audit,width,role) for role,audit in zip(("old","new"),audits,strict=True)]
        regressions,differences=paired_outcomes(quality,plan,*audits)
        require(not regressions,"Regraded actual task responses have a new regression")
        comparison=comparisons[width]
        exact(comparison,{"schema":"original22-paired-integer-B4-composition-comparison-v1","valid":True,
            "width":width,"execution_mode":"mtp3","original_plan_content_sha256":PLAN,
            "no_new_task_regressions":True,"new_task_regressions":[],"generation_differences":differences},
            "Actual comparison differs from fresh frozen regrade")
        baseline=[{"id":k,"lane":i} for k,v in audits[0]["outcomes"].items() for i,passed in enumerate(v) if not passed]
        candidate=[{"id":k,"lane":i} for k,v in audits[1]["outcomes"].items() for i,passed in enumerate(v) if not passed]
        require(comparison.get("baseline_failed_lanes")==baseline and comparison.get("candidate_failed_lanes")==candidate,"Actual persistent task failures were not preserved")
        expected=[{"valid":a["valid"],"errors":a["errors"],"aggregate_integer_coverage":a["aggregate_integer_coverage"]} for a in audits]
        require(comparison.get("evidence_audits")==expected,"Saved comparison audit differs from recomputed actual evidence")
        rows=16 if width==4 else 8
        result["B"+str(width)]={"valid":True,"no_new_task_regressions":True,
            "all22_original_cases_all_real_lanes_graded":True,"actual_native_requested_width_proved":True,
            "all_mixed_real_lanes_graded":True,"graceful_worker_unload":True,
            "positive_physical_integer_rows":rows,"actual_new_integer_graph_calls":new[rows],"old_integer_graph_calls":sum(old.values()),
            "comparison_report_sha256":files[str(artifact_paths()[f"comparison.B{width}"])],
            "old_report_sha256":files[str(artifact_paths()[f"report.old.B{width}"])],
            "new_report_sha256":files[str(artifact_paths()[f"report.new.B{width}"])],
            "baseline_failed_lanes":baseline,"candidate_failed_lanes":candidate}
    return result

def run_root_reports(args):
    require(args.run_root_reports,"Only Root may explicitly read actual reports with --run-root-reports")
    require(not args.output.exists(),"Fresh actual task admission required")
    require(is_sha(args.factory_source_sha256) and sha(Path(__file__))==args.factory_source_sha256,"Externally pinned grade factory source required")
    for path,pin in ((TASK/"CPU_READY.json",TASK_READY),(TASK/"root-command.json",TASK_COMMAND),
        (TASK/"root-paired-MTP3-quality-config.json",TASK_CONFIG),(TASK/"paired-composition-binding.json",TASK_BINDING),
        (TASK/"source/paired_quality.py",QUALITY_SOURCE),(TASK/"source/run_paired_quality.py",LAUNCHER_SOURCE)):
        require(sha(path)==pin,"Frozen active taskV2 source/plan binding drift: "+str(path))
    require(is_sha(args.actual_artifact_pins_sha256) and sha(args.actual_artifact_pins)==args.actual_artifact_pins_sha256,"External actual artifact manifest digest differs")
    pins=json.loads(args.actual_artifact_pins.read_text());files=validate_actual_pins(pins,args.summary,args.summary_sha256)
    config=json.loads((TASK/"root-paired-MTP3-quality-config.json").read_text())
    launcher=module(TASK/"source/run_paired_quality.py","_frozen_taskV2_launcher_for_actual_grade")
    quality,binding=launcher.validate(config,True);validate_binding(binding)
    require(config["quality_driver"]==str(TASK/"source/paired_quality.py"),"Exact frozen taskV2 grader required")
    plan=quality.frozen_plan(Path(config["original_plan"]))
    documents={}
    for name,path in artifact_paths().items():
        require(sha(path)==files[str(path)],"Root-pinned actual report digest differs: "+name)
        documents[name]=quality.http.strict_json(path.read_text())
    validate_summary(documents["summary"],config,launcher,files)
    reports={(role,width):documents[f"report.{role}.B{width}"] for role in ("old","new") for width in (4,2)}
    comparisons={width:documents[f"comparison.B{width}"] for width in (4,2)}
    widths=reaudit_pairs(quality,plan,binding,reports,comparisons,files)
    result={"schema":SCHEMA,"pass":True,"execution_mode":"mtp3","qualified_native_widths":[4,2],"maximum_context_tokens":16384,
        "worker_sha256":WORKER,"metallib_sha256":LIB,"source_identity_sha256":SOURCE,"BQSA4_policy_sha256":POLICY,
        "compiled_worker_seal_sha256":SEAL,"native_stage_receipt_sha256":STAGE,"native_QA_seal_sha256":NATIVE_SEAL,
        "original_plan_content_sha256":PLAN,"task_root_command_sha256":TASK_COMMAND,"task_CPU_READY_sha256":TASK_READY,
        "task_config_sha256":TASK_CONFIG,"binding_sha256":TASK_BINDING,"frozen_quality_source_sha256":QUALITY_SOURCE,
        "frozen_launcher_source_sha256":LAUNCHER_SOURCE,"grade_factory_source_sha256":args.factory_source_sha256,
        "actual_artifact_pins_sha256":args.actual_artifact_pins_sha256,"actual_summary_sha256":args.summary_sha256,
        "externally_pinned_actual_artifact_sha256":files,"target_base_numeric_parent_sha256":RAW_PARENT,
        "target_numeric_parent_sha256":WRAPPED_PARENT,"target_execution_base_child_sha256":RAW_CHILD,
        "target_execution_child_sha256":WRAPPED_CHILD,"frozen_common_status_and_coverage_reaudited":True,
        "actual_responses_regraded_from_frozen22_bodies":True,"both_roles_gracefully_unloaded":True,
        "same_worker_BQSA1_only_integer_flag_delta_reaudited":True,"performance_qualified":False,
        "standard_qualified":False,"public_promotion_qualified":False,"old_extra_EVERYROW_equivalence":False,
        "old_extra_failed_rows":116,"lifecycle_qualified":False,"all18_fullphysical_qualified":False,
        "GPU_executed_by_factory":False,**widths}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as output:json.dump(result,output,indent=2,allow_nan=False);output.write("\n")
    print(json.dumps({"pass":True,"output":str(args.output),"qualified_native_widths":[4,2],"GPU_executed_by_factory":False}))
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    for name in ("summary","actual-artifact-pins","output"):parser.add_argument("--"+name,type=Path,required=True)
    for name in ("summary-sha256","actual-artifact-pins-sha256","factory-source-sha256"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--run-root-reports",action="store_true")
    args=parser.parse_args(argv)
    if not args.run_root_reports:parser.error("Actual reports remain unread until Root explicitly supplies --run-root-reports")
    return run_root_reports(args)
if __name__=="__main__":raise SystemExit(main())
