#!/usr/bin/env python3
"""Bind the already completed canonical small Root receipts to sealed commands."""
import argparse
import json
from pathlib import Path
import shlex
from launch_common import HERE,ROOT,GRADE_SHA,STAGE_SHA,load_pinned,canonical,validate_grade,validate_stage,sha,require

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--launch-witness-sha256",required=True)
    parser.add_argument("--grade-sha256",required=True)
    args=parser.parse_args();plan=load_pinned(args.launch_witness_sha256)
    path=HERE/"Root-actual-proof-binding.json";require(not path.exists(),"Fresh immutable actual proof binding required")
    require(args.grade_sha256==GRADE_SHA==sha(plan["actual_grade_receipt"]),"Exact canonical actual Root bothwidth grade4133 required")
    require(sha(plan["native_stage_receipt"])==STAGE_SHA,"Canonical current native stage7f82 required")
    validate_grade(plan,json.loads(Path(plan["actual_grade_receipt"]).read_text()))
    validate_stage(plan,json.loads(Path(plan["native_stage_receipt"]).read_text()))
    binding={"schema":"current-BQSA4-integer-Root-native-and-task-normal-binding-v1","pass":True,
        "launch_witness_sha256":args.launch_witness_sha256,"grade_receipt":plan["actual_grade_receipt"],
        "grade_receipt_sha256":GRADE_SHA,"native_stage_receipt_sha256":STAGE_SHA,
        "GPU_or_model_work_by_binding":False,"actual_task_scope":"fresh same-worker integer0/1 Original22+mixed B4/B2, including positive R16/R8 campaign exposure and healthy unloads",
        "worker_lifecycle_or_performance_qualification_inherited":False}
    path.write_text(json.dumps(binding,indent=2)+"\n");digest=sha(path)
    for role in ("old","new"):
        canonical(plan,role)
        argv=[str(ROOT/".venv/bin/python"),"-B",str(HERE/"run_root_service.py"),"--role",role,
            "--launch-witness-sha256",args.launch_witness_sha256,"--proof-binding-sha256",digest,"--run-root-gpu"]
        (HERE/f"root-{role}-bound-command.txt").write_text(shlex.join(argv)+"\n")
        audit=[str(ROOT/".venv/bin/python"),"-B",str(HERE/"audit_profile.py"),"--role",role,
            "--launch-witness-sha256",args.launch_witness_sha256,"--proof-binding-sha256",digest,"--run-root-reports"]
        (HERE/f"root-{role}-postflush-profile-audit-command.txt").write_text(shlex.join(audit)+"\n")
    print(json.dumps({"binding_sha256":digest,"GPU_executed":False,"canonical_grade_sha256":GRADE_SHA}))
    return 0

if __name__=="__main__":raise SystemExit(main())
