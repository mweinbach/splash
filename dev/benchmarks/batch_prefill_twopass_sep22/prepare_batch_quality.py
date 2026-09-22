#!/usr/bin/env python3
"""Seal CPU-only four-profile native task launcher; no inference/tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path("/Users/mweinbach/Projects/splash")
sys.path.insert(0,str(ROOT))
from dev.benchmarks.batch_prefill_twopass_sep22 import batch_quality as q
HERE=Path(__file__).resolve().parent
RECEIPT=ROOT/"build/release/flash/sep22-batch-twoPass-B2-B4-intended-state-admission-v1.json"
RECEIPT_SHA="23f15fc2edbeb3b23962178f61ddd6b4e91fd8944ef1dbc2d0d2676df8a4e27d"
PLAN=ROOT/"build/release/flash/prefill4k-semantic-plan-v1.json"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--build",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port",type=int,default=8058)
    args=parser.parse_args();build=args.build.resolve()
    if build.exists():raise ValueError("Fresh quality seal directory required")
    plan=q.frozen_plan(PLAN)
    for role in ("old","new"):q.authenticate(role,ROOT/q.ROLES[role][0],RECEIPT,RECEIPT_SHA)
    build.mkdir(parents=True);source=build/"source";source.mkdir()
    owned=["batch_quality.py","run_batch_quality.py","test_batch_quality.py","prepare_batch_quality.py","ALTERNATIVE_QUALIFICATION.md"]
    for name in owned:shutil.copy2(HERE/name,source/name)
    original=ROOT/"dev/benchmarks/splash_tuning_sep21.py";shutil.copy2(original,source/"splash_tuning_sep21.py")
    parent=ROOT/q.ROLES["old"][0];oldcommand=json.loads((parent/"root-main-proof-command.json").read_text())
    scripts=[source/name for name in owned]+[HERE/name for name in owned]+[source/"splash_tuning_sep21.py",original,
        ROOT/"server/server.py",ROOT/"install/launcher.py",RECEIPT,PLAN]
    # Freeze every original grader/status/helper Python source dependency that
    # can be imported by the unchanged original quality and launcher functions.
    names=["qualify_flash_http.py","prefill4k_attribution_quality.py","prefill4k_attribution_quality_python.py",
           "flash_precision_quality.py","prefill_decode_phase_quality.py","singleton_teacher_bulk_quality.py",
           "teacher_singleton_lease_quality.py","phase_saved_only_resource_quality.py",
           "flash_http_performance.py","flash_context_grid.py"]
    scripts += [ROOT/"dev/benchmarks"/name for name in names]
    for role in ("old","new"):
        runtime=ROOT/q.ROLES[role][0]
        scripts += [runtime/"splash-flash",runtime/"splash.metallib",runtime/"compiled-cpu-seal.json"]
    config={"schema":"batch-twoPass-original22-four-profile-Root-plan-v1","width_order":[4,2],"modes":["standard","mtp3"],
        "target_derivative_sha256":q.TARGET,"original_plan":str(PLAN),"quality_driver":str(source/"batch_quality.py"),
        "original_driver_snapshot":str(source/"splash_tuning_sep21.py"),
        "state_receipt":str(RECEIPT),"state_receipt_sha256":RECEIPT_SHA,
        "package":oldcommand["command"][2],"tokenizer":plan["source_tokenizer"],"model":q.original.MODEL,
        "expert_store":oldcommand["environment"]["SPLASH_FLASH_INT8_EXPERT_STORE"],
        "base_environment":oldcommand["environment"],"port":args.port,"startup_timeout":600,
        "artifact_pins":{str(p):sha(p) for p in scripts},
        "task_bodies_and_budgets_unchanged":True,"old_extra_EVERYROW_gate_passed":False,"old_extra_failed_rows":116,
        "native_requested_width_proof_from_counters":True,"HTTP_response_ids_are_not_native_call_IDs":True,
        "quality_only_no_rate_or_perf_floor":True}
    path=build/"root-four-profile-quality-config.json";path.write_text(json.dumps(config,indent=2)+"\n")
    tests=subprocess.run([sys.executable,"-B","-m","unittest","dev.benchmarks.batch_prefill_twopass_sep22.test_batch_quality"],
                         cwd=ROOT,check=True,capture_output=True,text=True)
    summary=args.output.resolve()
    if summary.exists():raise ValueError("Fresh four-profile summary required; preserve previous attempts")
    command=[sys.executable,"-B",str(source/"run_batch_quality.py"),"--config",str(path),"--config-sha256",sha(path),"--output",str(summary)]
    dry=subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True)
    dryvalue=json.loads(dry.stdout)
    if not dryvalue["CPU_dry_valid"] or dryvalue["GPU_executed"] or dryvalue["tokenizer_or_model_payload_loaded"]:
        raise ValueError("Metadata-only launcher dry path failed")
    command += ["--run-root-gpu"]
    rootcmd={"schema":"batch-twoPass-original22-four-profile-command-v1","Root_GPU_only":True,
             "command":command,"cwd":str(ROOT),"summary":str(summary),"config_sha256":sha(path),
             "state_receipt_sha256":RECEIPT_SHA,"source_pins":config["artifact_pins"],
             "required_independent_source_review":True,"actual_model_requests_executed_by_preparer":False}
    (build/"root-command.json").write_text(json.dumps(rootcmd,indent=2)+"\n")
    ready={"schema":"batch-twoPass-original22-four-profile-CPU-source-seal-v1","pass":True,
           "GPU_executed":False,"model_capture_fixture_response_operand_payload_read_or_hashed":False,
           "original_plan_content_sha256":q.PLAN,"original_grader_body_or_budget_changes":0,
           "source_files":[{"path":str(p),"sha256":sha(p)}for p in scripts],
           "config_sha256":sha(path),"root_command_sha256":sha(build/"root-command.json"),
           "actual_B2_B4_state_receipt_sha256":RECEIPT_SHA,
           "CPU_tests":{"pass":True,"tests":9,"output":tests.stdout+tests.stderr},
           "metadata_only_dry":dryvalue,"profiles":4,"width_order":[4,2],"cases_per_width":22,
           "extra_mixed_wave_original_case_ids":q.MIXED_IDS,
           "independent_review_pending":True,"actual_quality_or_performance_qualified":False}
    (build/"CPU_READY.json").write_text(json.dumps(ready,indent=2)+"\n")
    print(json.dumps({"CPU_READY":str(build/"CPU_READY.json"),"CPU_READY_sha256":sha(build/"CPU_READY.json"),
                     "root_command":str(build/"root-command.json"),"root_command_sha256":sha(build/"root-command.json"),
                     "config_sha256":sha(path),"dry":dryvalue}))
if __name__=="__main__":main()
