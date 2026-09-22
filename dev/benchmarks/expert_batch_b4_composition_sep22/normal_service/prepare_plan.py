#!/usr/bin/env python3
"""CPU/source-only canonical combined benchmark preparation, no payload access."""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from launch_common import ROOT,HERE,PLAN_SCHEMA,WORKER,LIB,SOURCE,POLICY,SEAL,GRADE_SHA,STAGE_SHA,sha,require,canonical

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args();destination=args.destination.resolve();require(not destination.exists(),"Fresh normal source directory required")
    require(destination.is_relative_to(ROOT/"build"),"Private build-only metadata directory")
    destination.mkdir(parents=True)
    names=("launch_common.py","run_root_service.py","prepare_plan.py","bind_root_proofs.py","audit_profile.py","tests_normal.py")
    for name in names:ast.parse((HERE/name).read_text());shutil.copy2(HERE/name,destination/name)
    shutil.copy2(ROOT/"dev/benchmarks/current_batch_sep22/audit_report.py",destination/"audit_native.py")
    baseline_path=ROOT/"build/restored-batch-service-plan-sep22-v1/service-plan.json"
    baseline=json.loads(baseline_path.read_text());require(len(baseline["modes"]["3"]["explicit_flags"])==64,"Exact original restored64flags required")
    runtime=ROOT/"build/integer-b4-twopass-composed-sep22-worker-v2"
    require(sha(runtime/"splash-flash")==WORKER and sha(runtime/"splash.metallib")==LIB and sha(runtime/"compiled-cpu-seal.json")==SEAL,"Current source-qualified combined runtime only")
    quality=ROOT/"build/current-BQSA4-integer-paired-original22-MTP3-sep22-v2"
    plan={"schema":PLAN_SCHEMA,"CPU_preparation_GPU_executed":False,"model_fixture_response_operand_capture_payload_read_or_hashed":False,
        "roles":["old","new"],"allowed_modes":["3"],"allowed_native_widths":[4,2],"runtime":str(runtime),
        "worker_sha256":WORKER,"metallib_sha256":LIB,"worker_seal_sha256":SEAL,"source_identity_sha256":SOURCE,"BQSA4_policy_sha256":POLICY,
        "frozen_restored_baseline":{"source":str(baseline_path),"sha256":sha(baseline_path),"modes":{"3":baseline["modes"]["3"]},"controls":baseline["controls"]},
        "composition_binding":str(quality/"paired-composition-binding.json"),"composition_binding_sha256":"dff527c9c8ea60dbaa64b759ccf2328feda0e0f9aba78d4c189ba03d75d13d82",
        "frozen_quality_core":str(quality/"source/paired_quality.py"),
        "native_stage_receipt":str(ROOT/"build/release/flash/sep22-current-BQSA4-integer-Root-native-stage-admission-v1.json"),"native_stage_receipt_sha256":STAGE_SHA,
        "actual_grade_receipt":str(ROOT/"build/release/flash/sep22-current-BQSA4-integer-paired-original22-MTP3-suite-v2-mtp3.grade-admission.json"),"actual_grade_receipt_sha256":GRADE_SHA,
        "grade_producer_command_sha256":"20e4ebad4dea7e2c1bf5f6bba1746a91b1af82e4a76184e23aa74451980f1f30",
        "grade_producer_CPU_READY_sha256":"de99d0c9ffc93a01961691dfd63d2d10a16c21fb18a89d7ab6897c8a477b1fd6",
        "expected_native_batch_prefill_ctor_bytes":baseline["actual_proof_max4_BQSA_arena_bytes"]+509607936,
        "primary_metric":"actual native postfirst emitted tokens / common earliestfirst→latestDONE native span",
        "old_extra_EVERYROW_equivalence":False,"old_extra_failed_rows":116,"standard_or_B1_performance_qualified":False,
        "worker_cancel_deadline_or_public_promotion_qualified":False,"profiles":{}}
    for role,port in (("old",8064),("new",8065)):
        old=baseline["modes"]["3"];argv=list(old["argv"])
        stem="sep22-current-BQSA4-integer-normal-"+role+"-MTP3-B4-B2-v1"
        report=ROOT/"build/release/flash"/(stem+".json");trace=report.with_suffix(".jsonl")
        argv[argv.index("--binary")+1]=str(runtime/"splash-flash");argv[argv.index("--output")+1]=str(report)
        for i,token in enumerate(argv):
            if token=="--env" and argv[i+1].startswith("SPLASH_FLASH_NATIVE_LIFECYCLE_TRACE_SEP22="):
                argv[i+1]="SPLASH_FLASH_NATIVE_LIFECYCLE_TRACE_SEP22="+str(trace)
        argv += ["--env","SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22=1","--env","SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22="+("1" if role=="new" else "0"),"--port",str(port)]
        flags={argv[i+1].split("=",1)[0]:argv[i+1].split("=",1)[1] for i,t in enumerate(argv) if t=="--env"}
        require(len(flags)==66,"Original64flags +BQSA1 +integer roleflag only")
        plan["profiles"][role]={"argv":argv,"explicit_flags":flags,"port":port,"report":str(report),"trace":str(trace),
            "native_audit":str(report.with_name(stem+".native-audit.json")),"profile_audit":str(report.with_name(stem+".profile-audit.json"))}
        (destination/f"root-{role}-original-driver-review-only.txt").write_text(shlex.join(argv)+"\n")
        audit=[str(ROOT/".venv/bin/python"),"-B",str(destination/"audit_native.py"),"--report",str(report),"--native-lifecycle-trace",str(trace),"--output",plan["profiles"][role]["native_audit"]]
        (destination/f"root-{role}-postflush-native-audit-command.txt").write_text(shlex.join(audit)+"\n")
    for role in plan["roles"]:canonical(plan,role)
    old,new=[plan["profiles"][role]["explicit_flags"] for role in ("old","new")]
    require([k for k in old if old[k]!=new[k] and k!="SPLASH_FLASH_NATIVE_LIFECYCLE_TRACE_SEP22"]==["SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22"],"Only arithmetic flag difference is integer0/1")
    (destination/"service-plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    pins={}
    def pin(path):pins[str(path.resolve())]=sha(path)
    for path in destination.glob("*.py"):pin(path)
    pin(destination/"service-plan.json");pin(baseline_path)
    metadata=json.loads((quality/"CPU_READY.json").read_text())
    for record in metadata["source_files"]:
        path=Path(record["path"])
        # All registered task inputs here are source/runtime/small metadata;
        # the actual task fixture is separately declared and never opened.
        require(sha(path)==record["sha256"],"Unchanged qualified source/runtime dependency drift: "+str(path));pin(path)
    for path in (quality/"CPU_READY.json",quality/"root-command.json",quality/"root-paired-MTP3-quality-config.json",
        ROOT/"build/release/flash/sep22-current-BQSA4-integer-paired-original22-v2-independent-source-review-v1.json"):
        pin(path)
    witness={"schema":"current-BQSA4-integer-normal-source-witness-v1","files":[{"path":path,"sha256":digest} for path,digest in pins.items()]}
    witness_path=destination/"launch-witness.json";witness_path.write_text(json.dumps(witness,indent=2)+"\n")
    command=[str(ROOT/".venv/bin/python"),"-B",str(destination/"bind_root_proofs.py"),"--launch-witness-sha256",sha(witness_path),"--grade-sha256",GRADE_SHA]
    (destination/"root-bind-actual-proofs-command.txt").write_text(shlex.join(command)+"\n")
    tests=subprocess.run([str(ROOT/".venv/bin/python"),"-B",str(destination/"tests_normal.py")],cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT)},text=True,capture_output=True)
    require(tests.returncode==0,"Synthetic source tests failed: "+tests.stdout+tests.stderr)
    ready={"schema":"current-BQSA4-integer-normal-CPU-source-ready-v1","pass":True,"GPU_executed":False,
        "model_fixture_generation_response_operand_payload_read_or_hashed":False,"source_files":witness["files"],
        "launch_witness_sha256":sha(witness_path),"service_plan_sha256":sha(destination/"service-plan.json"),
        "canonical_actual_grade_receipt_sha256":GRADE_SHA,"canonical_native_stage_receipt_sha256":STAGE_SHA,
        "explicit_flags_perrole":66,"original_driver_or_measured_sampling_code_changes":0,"CPU_tests":{"pass":True,"stdout":tests.stdout,"stderr":tests.stderr},
        "actual_qualified_task_proofs_bound_by_Root_next":True,"performance_qualified":False,"independent_source_review_pending":True,
        "optional_backup_factory_used_or_required":False}
    (destination/"CPU_READY.json").write_text(json.dumps(ready,indent=2)+"\n")
    print(json.dumps({"directory":str(destination),"CPU_READY_sha256":sha(destination/"CPU_READY.json"),"witness_sha256":sha(witness_path),"flags":66,"GPU_executed":False}))
    return 0

if __name__=="__main__":raise SystemExit(main())
