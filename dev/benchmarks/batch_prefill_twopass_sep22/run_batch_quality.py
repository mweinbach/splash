#!/usr/bin/env python3
"""Root-owned four-profile launcher; each loaded profile measures B4 then B2."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path("/Users/mweinbach/Projects/splash")
sys.path.insert(0,str(ROOT))
from dev.benchmarks.batch_prefill_twopass_sep22 import batch_quality as quality
from dev.benchmarks.qualify_flash_http import HTTPClient

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(value,message):
    if not value:raise ValueError(message)
def publish(path,value):
    temporary=path.with_name(path.name+".writing")
    temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)
def server_command(config,build):
    # --local-package selects its bundled tokenizer in the actual server CLI.
    return [sys.executable,"-u",str(ROOT/"server/server.py"),"--local-package",config["package"],
            "--model",config["model"],"--binary",str(build/"splash-flash"),
            "--port",str(config["port"]),"--max-memory","auto","--max-context","16384","--no-webui"]
def validate(config):
    check(config["schema"]=="batch-twoPass-original22-four-profile-Root-plan-v1","Unregistered quality launcher schema")
    check(config["target_derivative_sha256"]==quality.TARGET,"Registered unchanged target derivative")
    check(config["width_order"]==[4,2] and config["modes"]==["standard","mtp3"],"Complete paired width/mode matrix")
    quality.frozen_plan(Path(config["original_plan"]))
    for path,pin in config["artifact_pins"].items():check(sha(path)==pin,"Quality code/receipt/artifact drift:"+path)
    for role in ("old","new"):quality.authenticate(role,ROOT/quality.ROLES[role][0],config["state_receipt"],config["state_receipt_sha256"])
    endpoint=config["port"];check(type(endpoint)is int and 1024<endpoint<65536 and endpoint!=8000,"Private quality port")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-root-gpu",action="store_true")
    args=parser.parse_args();check(sha(args.config)==args.config_sha256,"Exact Root quality command pin")
    config=json.loads(args.config.read_text());validate(config)
    if not args.run_root_gpu:
        print(json.dumps({"CPU_dry_valid":True,"GPU_executed":False,"tokenizer_or_model_payload_loaded":False,
                          "profiles":4,"widths":[4,2],"original_cases":22,"unchanged_all_real_lane_bodies_budgets_graders":True}));return 0
    check(not args.output.exists(),"Fresh four-profile summary")
    driver=Path(config["original_driver_snapshot"])
    spec=importlib.util.spec_from_file_location("_unchanged_original_tuning_for_quality",driver)
    tuning=importlib.util.module_from_spec(spec);spec.loader.exec_module(tuning)
    report={"schema":"batch-twoPass-original22-four-profile-summary-v1","completed":False,"GPU_executed":True,
            "source_policy":quality.POLICY,"native_context":16384,"original_plan_content_sha256":quality.PLAN,
            "server_runs":[],"comparisons":[],"errors":[],"performance_qualified":False,
            "old_extra_EVERYROW_gate_passed":False,"old_extra_failed_rows":116}
    publish(args.output,report);allreports={};unloads={}
    lockpath=ROOT/"build/splash-tuning-gpu.lock";lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for mode in config["modes"]:
                for role in ("old","new"):
                    build=ROOT/quality.ROLES[role][0]
                    explicit=dict(config["base_environment"])
                    explicit["SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22"]="0" if role=="old" else "1"
                    explicit["SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21"]="0"
                    explicit["SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22"]="0";explicit["SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22"]="0"
                    if mode=="standard":
                        explicit["SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY"]="0"
                    envargs=SimpleNamespace(package=Path(config["package"]),environment_overrides=explicit)
                    environment,envproof=tuning.environment_for(envargs,"standard" if mode=="standard" else "3")
                    with socket.socket() as sock:check(sock.connect_ex(("127.0.0.1",config["port"]))!=0,"Private port already occupied")
                    logpath=args.output.with_name(args.output.stem+f"-{role}-{mode}.server.log")
                    command=server_command(config,build)
                    run={"role":role,"execution_mode":mode,"command":command,"environment":envproof,
                         "log":str(logpath),"quality_reports":[],"unloaded":False}
                    report["server_runs"].append(run);publish(args.output,report)
                    with logpath.open("x") as log:
                        server=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        run["pid"]=server.pid
                        try:
                            client=HTTPClient(SimpleNamespace(base_url=f"http://127.0.0.1:{config['port']}",timeout=10,model=config["model"]))
                            waitargs=SimpleNamespace(idle_timeout=30,timeout=10,status_timeout=3,model=config["model"],max_context=16384)
                            deadline=time.monotonic()+config.get("startup_timeout",600)
                            while True:
                                check(server.poll() is None,"Server exited during startup")
                                try:initial=tuning.wait_idle(client,waitargs,timeout=min(30,max(.01,deadline-time.monotonic())));break
                                except (OSError,ValueError,TimeoutError):
                                    check(time.monotonic()<deadline,"Native quality server startup timeout");time.sleep(.5)
                            run["initial_status"]=initial;publish(args.output,report)
                            for width in config["width_order"]:
                                path=args.output.with_name(args.output.stem+f"-{role}-{mode}-B{width}.json")
                                argv=[sys.executable,"-B",config["quality_driver"],"measure","--role",role,"--width",str(width),
                                      "--execution-mode",mode,"--plan",config["original_plan"],"--runtime-build",str(build),
                                      "--expert-store",config["expert_store"],"--state-receipt",config["state_receipt"],
                                      "--state-receipt-sha256",config["state_receipt_sha256"],"--target-derivative-sha256",quality.TARGET,
                                      "--base-url",f"http://127.0.0.1:{config['port']}","--model",config["model"],"--output",str(path),"--run-root-gpu"]
                                subprocess.run(argv,cwd=ROOT,env=environment,check=True)
                                allreports[(role,mode,width)]=path;run["quality_reports"].append({"width":width,"path":str(path),"sha256":sha(path)})
                                publish(args.output,report)
                            run["final_status"]=tuning.wait_idle(client,waitargs,identity=initial["identity"],timeout=60)
                        finally:
                            evidence=tuning.unload(server);run["unload_evidence"]=evidence
                            run["unloaded"]=evidence.get("process_group_gone") is True
                            unloads[(role,mode)]=run["unloaded"] and evidence.get("returncode")==0 and evidence.get("post_parent_exit_sigkill_required") is False
                            publish(args.output,report)
                        check(unloads[(role,mode)],"Native worker/frontend did not unload cleanly")
                for width in config["width_order"]:
                    path=args.output.with_name(args.output.stem+f"-{mode}-B{width}.comparison.json")
                    argv=[sys.executable,"-B",config["quality_driver"],"compare","--plan",config["original_plan"],
                          "--old-report",str(allreports[("old",mode,width)]),"--new-report",str(allreports[("new",mode,width)]),"--output",str(path)]
                    subprocess.run(argv,cwd=ROOT,check=True);comparison=json.loads(path.read_text())
                    check(comparison["valid"] and comparison["no_new_task_regressions"],"Fresh mode/width task regression or invalid evidence")
                    report["comparisons"].append({"execution_mode":mode,"width":width,"path":str(path),"sha256":sha(path)})
                grade={"schema":"batch-twoPass-original22-mode-grade-admission-v1","pass":True,"source_identity":quality.POLICY,
                       "original_plan_content_sha256":quality.PLAN,"execution_mode":mode,"maximum_context_tokens":16384,
                       "candidate_worker_sha256":quality.ROLES["new"][1],"candidate_seal_sha256":quality.ROLES["new"][2],
                       "metallib_sha256":quality.LIB,"state_receipt_sha256":config["state_receipt_sha256"],
                       "performance_qualified":False,"old_extra_EVERYROW_equivalence":False}
                for width in (2,4):
                    result=next(r for r in report["comparisons"] if r["execution_mode"]==mode and r["width"]==width)
                    grade["B"+str(width)]={"valid":True,"no_new_task_regressions":True,"all22_original_cases_all_real_lanes_graded":True,
                         "actual_native_requested_width_proved":True,"graceful_worker_unload":unloads[("old",mode)] and unloads[("new",mode)],
                         "comparison_report_sha256":result["sha256"],"comparison_report_path":result["path"]}
                gradepath=args.output.with_name(args.output.stem+f"-{mode}.grade-admission.json")
                check(not gradepath.exists(),"Fresh mode grade receipt");publish(gradepath,grade)
                report.setdefault("mode_grade_receipts",[]).append({"execution_mode":mode,"path":str(gradepath),"sha256":sha(gradepath)})
                publish(args.output,report)
            report["completed"]=True;report["valid"]=True
        except Exception as e:
            report["errors"].append(type(e).__name__+": "+str(e));report["valid"]=False
        publish(args.output,report)
    print(json.dumps({"completed":report["completed"],"valid":report.get("valid",False),"summary":str(args.output)}))
    return 0 if report.get("valid") else 1
if __name__=="__main__":raise SystemExit(main())
