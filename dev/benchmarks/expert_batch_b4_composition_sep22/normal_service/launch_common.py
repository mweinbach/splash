"""Fail-closed source and actual proof binding for the combined batch matrix."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path("/Users/mweinbach/Projects/splash")
HERE=Path(__file__).resolve().parent
PLAN_SCHEMA="current-BQSA4-integer-MTP3-native-normal-plan-v1"
GRADE_SCHEMA="current-BQSA4-integer-MTP3-original22-bothwidth-grade-admission-v1"
GRADE_SHA="413384e51eeae597e5111f350c6f59815d56d93d23f70fce1354c04521449683"
STAGE_SCHEMA="current-BQSA4-plus-integer-batchverify-selected7-all18logical-Root-native-admission-v1"
STAGE_SHA="7f82c4592d08d7d317b1b7a3d3fa21e7dbb84e61f8e84b1ca68beae629a327cd"
WORKER="ca7bc795bfb42b50949f1eaf1f035620ec3bf1b861ab397ed9a166ccc3202e68"
LIB="74af1228995f38890df035555b2000018899783fd46ad3a12f8a835c833231c8"
SOURCE="edb3b1496296dad42908eb98fec4dc965ddfaaf98316eeadad1390cb5babcf94"
POLICY="b191ef4f7636d7700560c46789cdbec1322f93524ae63c93c3fef34d4ed25d07"
SEAL="32338e764d23627d0bfb74f12393c9aa84a0e4aaa37b2ec1da520c3c381f7461"
PLAN_CONTENT="a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac"
TARGET_PARENT="b87448342df3b3a9ae1642379b8aab513bb208ff234e552bcf70efc26a82c09d"
TARGET_CHILD="37b89da17689fd1b7f3b61ed3db0c14f743270e16c4dbe7f2803ea918878dad0"
TARGET_BASE_PARENT="04b42a9f509d23dc0de5087ba7853812497dcd15c20bcf2df0352eedc9bb1a80"
TARGET_BASE_CHILD="b40b095ec17517f1b014738a72dc0d37a2d11dd0499a40875e645f71d8e7764f"
INTEGER_FLAG="SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22"
BQSA_FLAG="SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22"
TRACE_FLAG="SPLASH_FLASH_NATIVE_LIFECYCLE_TRACE_SEP22"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def same(a,b):return type(a) is type(b) and a==b
def is_sha(value):return type(value) is str and re.fullmatch("[0-9a-f]{64}",value) is not None
def get(value,path):
    for key in path.split("."):
        if not isinstance(value,dict):return None
        value=value.get(key)
    return value
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,"Pinned source loader unavailable")
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def load_pinned(expected):
    path=HERE/"launch-witness.json"
    require(is_sha(expected) and sha(path)==expected,"External canonical source witness digest differs")
    witness=json.loads(path.read_text())
    require(witness.get("schema")=="current-BQSA4-integer-normal-source-witness-v1","Registered source witness required")
    for record in witness["files"]:
        require(sha(record["path"])==record["sha256"],"Frozen normal source/runtime/helper drift: "+record["path"])
    plan=json.loads((HERE/"service-plan.json").read_text())
    require(plan.get("schema")==PLAN_SCHEMA and plan.get("roles")==["old","new"] and plan.get("allowed_modes")==["3"],"Only paired current-worker MTP3 plan admitted")
    return plan

def canonical(plan,role):
    require(role in ("old","new"),"Only registered integer0/1 normal roles admitted")
    require(plan.get("allowed_native_widths")==[4,2] and plan.get("allowed_modes")==["3"],"Exact MTP3 B4 then B2 matrix required")
    baseline=plan["frozen_restored_baseline"]["modes"]["3"]
    selected=plan["profiles"][role]
    argv=list(baseline["argv"])
    argv[argv.index("--binary")+1]=str(Path(plan["runtime"])/"splash-flash")
    argv[argv.index("--output")+1]=selected["report"]
    found=0
    for index,token in enumerate(argv):
        if token=="--env" and argv[index+1].startswith(TRACE_FLAG+"="):
            argv[index+1]=TRACE_FLAG+"="+selected["trace"];found+=1
    require(found==1,"Exactly one inherited native trace destination required")
    argv += ["--env",BQSA_FLAG+"=1","--env",INTEGER_FLAG+"="+("1" if role=="new" else "0"),"--port",str(selected["port"])]
    require(argv==selected["argv"],"Canonical original driver/64 flags plus BQSA1/integer0or1 required")
    require(argv[argv.index("--batches")+1]=="4,2" and argv[argv.index("--mtp")+1]=="3","No other mode or width order allowed")
    for flag,value in (("--contexts","2048"),("--output-tokens","256"),("--warmup","1"),("--trials","3"),
                       ("--max-context","16384"),("--lane-variation","shared"),("--workloads","coding")):
        require(argv[argv.index(flag)+1]==value,"Frozen native workload differs: "+flag)
    require("--semantic-plan" not in argv and "--dry-run" not in argv,"No task suite or dry preparation in measured worker")
    flags={argv[i+1].split("=",1)[0]:argv[i+1].split("=",1)[1] for i,t in enumerate(argv) if t=="--env"}
    require(len(flags)==66 and flags==selected["explicit_flags"],"Exact 66 flag set required")
    for key,value in (("SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS","4"),("SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21","0"),
        ("SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS","0"),("SPLASH_FLASH_NATIVE_LIFECYCLE_TIMESTAMPS_SEP22","1")):
        require(flags.get(key)==value,"Frozen batch/source/timestamp policy differs: "+key)
    return argv

def validate_stage(plan,stage):
    fixed={"schema":STAGE_SCHEMA,"pass":True,"Root_GPU_executed":True,"candidate_worker_sha256":WORKER,
        "candidate_source_identity_sha256":SOURCE,"BQSA4_policy_sha256":POLICY,"metallib_sha256":LIB,
        "compiled_worker_seal_sha256":SEAL,"proof_capacity":4096,"all_four_pairs_complete":True,
        "selected7_fullphysical_qualified":True,"all18_logical_output_max16_MoE_rejections_owners_qualified":True,
        "all18_fullphysical_qualified":False,"full_numeric_flags_count":47,"original22_or_service16K_qualified":False,
        "trained_head_qualified":False,"worker_cancel_deadline_qualified":False,"performance_qualified":False,"public_promotion_qualified":False}
    require(isinstance(stage,dict) and all(same(stage.get(k),v) for k,v in fixed.items()),"Exact actual current stage contract required; no old or widened proof")
    require(plan["native_stage_receipt_sha256"]==STAGE_SHA,"Only canonical Root current-stage digest admitted")
    require(len(stage.get("parts",[]))==4 and all(p.get("pass") is True and p.get("backend_destroyed") is True for p in stage["parts"]),"All four current native role pairs must be complete and destroyed")

def validate_grade(plan,grade):
    fixed={"schema":GRADE_SCHEMA,"pass":True,"execution_mode":"mtp3","qualified_native_widths":[4,2],
        "maximum_context_tokens":16384,"worker_sha256":WORKER,"metallib_sha256":LIB,"source_identity_sha256":SOURCE,
        "BQSA4_policy_sha256":POLICY,"compiled_worker_seal_sha256":SEAL,"native_stage_receipt_sha256":STAGE_SHA,
        "original_plan_content_sha256":PLAN_CONTENT,"performance_qualified":False,"standard_qualified":False,
        "old_extra_EVERYROW_equivalence":False,"both_roles_gracefully_unloaded":True,
        "binding_sha256":"dff527c9c8ea60dbaa64b759ccf2328feda0e0f9aba78d4c189ba03d75d13d82"}
    require(isinstance(grade,dict) and all(same(grade.get(k),v) for k,v in fixed.items()),"Fresh mode-specific BOTH width actual task receipt required")
    comparisons=grade.get("comparisons")
    require(isinstance(comparisons,list) and len(comparisons)==2,"Both actual width task comparisons required")
    for result,width,digest in zip(comparisons,(4,2),(
        "2a9d09a924fa4a4895cc436ea683232e042dd0eaa8cceb140b75a1df23ef215c",
        "d945d6bbe0a2b87eaf5262df44818296c488b352ff93c8673f2632efd6c04920"),strict=True):
        require(isinstance(result,dict) and same(result.get("width"),width) and result.get("execution_mode")=="mtp3",
            "Ordered actual MTP3 width comparisons required")
        require(result.get("valid") is True and result.get("no_new_task_regressions") is True and result.get("sha256")==digest,
            "Exact Root actual task comparison including original22/mixed/native exposure re-audit required")
    require(plan.get("actual_grade_receipt_sha256")==GRADE_SHA,"Canonical Root actual bothwidth grade digest only")

def validate_binding(plan,expected_witness,expected_binding):
    path=HERE/"Root-actual-proof-binding.json"
    require(is_sha(expected_binding) and sha(path)==expected_binding,"External actual proof binding digest differs")
    binding=json.loads(path.read_text())
    require(binding.get("schema")=="current-BQSA4-integer-Root-native-and-task-normal-binding-v1" and binding.get("pass") is True and binding.get("launch_witness_sha256")==expected_witness,"Actual task/native proof must bind this exact canonical normal source")
    stage=Path(plan["native_stage_receipt"]);grade=Path(binding["grade_receipt"])
    require(sha(stage)==binding["native_stage_receipt_sha256"]==STAGE_SHA,"Canonical actual native receipt drift")
    require(sha(grade)==binding["grade_receipt_sha256"]==GRADE_SHA,"Actual Root bothwidth canonical grade receipt changed after external binding")
    validate_stage(plan,json.loads(stage.read_text()));validate_grade(plan,json.loads(grade.read_text()))
    return binding
