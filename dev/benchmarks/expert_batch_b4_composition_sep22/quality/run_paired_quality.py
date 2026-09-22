#!/usr/bin/env python3
"""Root-only ORIGINAL22 pair on one composed worker, integer flag0 then1."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/Users/mweinbach/Projects/splash")
SCHEMA = "current-BQSA4-integer-MTP3-paired-original22-Root-plan-v1"
STAGE_SCHEMA = "current-BQSA4-plus-integer-batchverify-selected7-all18logical-Root-native-admission-v1"
STAGE_SHA = "7f82c4592d08d7d317b1b7a3d3fa21e7dbb84e61f8e84b1ca68beae629a327cd"
WORKER = "ca7bc795bfb42b50949f1eaf1f035620ec3bf1b861ab397ed9a166ccc3202e68"
LIB = "74af1228995f38890df035555b2000018899783fd46ad3a12f8a835c833231c8"
SOURCE = "edb3b1496296dad42908eb98fec4dc965ddfaaf98316eeadad1390cb5babcf94"
POLICY = "b191ef4f7636d7700560c46789cdbec1322f93524ae63c93c3fef34d4ed25d07"
SEAL = "32338e764d23627d0bfb74f12393c9aa84a0e4aaa37b2ec1da520c3c381f7461"
NATIVE_SEAL = "80d7618e4b002a077ddfc2e9f801e6f5944e5bf40d5bfb61f77b29a5162b55d5"
INTEGER_FLAG = "SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22"
BQSA_FLAG = "SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22"
DEPS = ("SPLASH_FLASH_BATCH_PREFILL", "SPLASH_FLASH_BATCH_QSA_BULK_PREFILL",
        "SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21", "SPLASH_FLASH_QSA_F32", "SPLASH_FLASH_QSA_MPP",
        "SPLASH_FLASH_QSA_ROW_TILES", "SPLASH_FLASH_QSA_BULK_PREFILL", "SPLASH_FLASH_QSA_BULK_PREFILL_SG8",
        "SPLASH_FLASH_MTP", "SPLASH_FLASH_BATCH_MTP", "SPLASH_FLASH_BATCH_MTP_PREFILL",
        "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY", "SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check(value, message):
    if not value:
        raise ValueError(message)

def publish(path, value):
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    check(spec is not None and spec.loader is not None, "Pinned source module unavailable")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def same(a, b):
    return type(a) is type(b) and a == b

def validate_stage(stage):
    """Only Root's actual current native-stage contract; no task inheritance."""
    check(isinstance(stage, dict) and stage.get("schema") == STAGE_SCHEMA, "Exact current Root stage schema required")
    fixed = {"pass": True, "Root_GPU_executed": True,
        "candidate_worker_sha256": WORKER, "candidate_source_identity_sha256": SOURCE,
        "BQSA4_policy_sha256": POLICY, "metallib_sha256": LIB, "compiled_worker_seal_sha256": SEAL,
        "native_QA_seal_sha256": NATIVE_SEAL, "proof_capacity": 4096,
        "all_four_pairs_complete": True, "selected7_fullphysical_qualified": True,
        "all18_logical_output_max16_MoE_rejections_owners_qualified": True,
        "all18_fullphysical_qualified": False, "full_numeric_flags_count": 47,
        "actual_verify_commands_per_role": 17, "actual_commit_commands_per_role": 16,
        "actual_rejection_checks_per_role": 13, "BQSA4_actual_fresh_grouped_prefill_layer_calls": 48,
        "B2_oldSG8_BQSA_new_layer_calls": 0, "trained_head_qualified": False,
        "worker_cancel_deadline_qualified": False, "original22_or_service16K_qualified": False,
        "performance_qualified": False, "public_promotion_qualified": False,
        "snapshot_reserved_zero_claimed": False}
    for key, expected in fixed.items():
        check(same(stage.get(key), expected), "Current native stage field differs: " + key)
    parts = stage.get("parts")
    check(isinstance(parts, list) and len(parts) == 4, "All four actual native pairs required")
    for index, (part, frames) in enumerate(zip(parts, (158,158,158,157), strict=True), 1):
        check(isinstance(part, dict) and same(part.get("ordinal"), index), "Ordered actual native pair required")
        check(part.get("pass") is True and part.get("backend_destroyed") is True, "Actual role pair teardown required")
        check(same(part.get("frames"), frames), "Exact current native inventory required")

def role_environment(config, role):
    check(role in ("old", "new"), "Only integer0/1 roles admitted")
    explicit = dict(config["base_environment"])
    check(explicit.get(BQSA_FLAG) == "1", "Same BQSA4 math required in both roles")
    check(explicit.get(INTEGER_FLAG) == "0", "Registered control starts at integer0")
    check(all(explicit.get(key) == "1" for key in DEPS), "Exact BQSA4 thirteen prerequisite flags required")
    check(explicit.get("SPLASH_FLASH_MTP_DRAFT_DEPTH") == "3", "Only native MTP3 admitted")
    check(explicit.get("SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS") == "4", "Exact gathered cap4 required")
    check(explicit.get("SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21") == "0" and
          explicit.get("SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS") == "0", "Batch teacher bulk0/QA0 required")
    explicit[INTEGER_FLAG] = "1" if role == "new" else "0"
    return explicit

def server_command(config, build, role):
    # The actual local-package CLI owns its bundled tokenizer.
    return [sys.executable, "-u", str(ROOT/"server/server.py"), "--local-package", config["package"],
            "--model", config["model"], "--binary", str(build/"splash-flash"),
            "--port", str(config["ports"][role]), "--max-memory", "auto", "--max-context", "16384", "--no-webui"]

def validate(config, payload_allowed=False):
    check(config.get("schema") == SCHEMA, "Exact paired composition plan required")
    check(config.get("modes") == ["mtp3"] and config.get("width_order") == [4,2], "Only paired MTP3 B4 then B2 admitted")
    check(config.get("same_worker_integer_only_control_delta") is True, "One worker and one integer flag delta required")
    numeric = config.get("native_stage_numeric_flags")
    check(isinstance(numeric, dict) and len(numeric) == 47 and all(type(k) is str and type(v) is str for k,v in numeric.items()), "Complete literal native-stage 47-flag tuple required")
    check(all(config["base_environment"].get(k) == v for k,v in numeric.items()), "Task numerical tuple differs from the current qualified native stage")
    for role in ("old", "new"):
        port = config["ports"][role]
        check(type(port) is int and 1024 < port < 65536 and port != 8000, "Dedicated private task port")
        role_environment(config, role)
    check(config["ports"]["old"] != config["ports"]["new"], "Separate registered role ports required")
    check(config.get("native_stage_receipt_sha256") == STAGE_SHA, "Only actual Root current-stage external digest admitted")
    check(sha(config["native_stage_receipt"]) == STAGE_SHA, "Actual Root native-stage receipt drift")
    validate_stage(json.loads(Path(config["native_stage_receipt"]).read_text()))
    for path, pin in config["artifact_pins"].items():
        check(sha(path) == pin, "Pinned code/runtime/source metadata drift: " + path)
    quality = module(config["quality_driver"], "_current_composed_paired_original22")
    binding = quality.load_binding(config["binding"], config["binding_sha256"])
    for role in ("old", "new"):
        quality.authenticate(role, config["binding"], config["binding_sha256"])
    check(binding["worker_sha256"] == WORKER and binding["metallib_sha256"] == LIB, "Same current composed worker/library")
    if payload_allowed:
        check(sha(config["original_plan"]) == config["original_plan_file_sha256"], "Exact frozen original fixture file required")
        quality.frozen_plan(Path(config["original_plan"]))
    return quality, binding

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root-gpu", action="store_true")
    args = parser.parse_args()
    check(sha(args.config) == args.config_sha256, "External registered Root command config digest differs")
    config = json.loads(args.config.read_text())
    quality, binding = validate(config, args.run_root_gpu)
    if not args.run_root_gpu:
        print(json.dumps({"CPU_source_admission_valid": True, "GPU_executed": False,
            "fixture_model_tokenizer_or_response_payload_read": False, "profiles": 2, "width_order": [4,2],
            "integer_only_flag_delta": True, "actual_task_or_performance_qualified": False}))
        return 0
    check(not args.output.exists(), "Fresh paired task summary required")
    sys.path.insert(0, str(ROOT))
    from dev.benchmarks.qualify_flash_http import HTTPClient
    tuning = module(config["original_driver_snapshot"], "_unchanged_original_tuning_for_composed_quality")
    report = {"schema": "current-BQSA4-integer-MTP3-paired-original22-summary-v1",
        "completed": False, "valid": False, "GPU_executed": True, "execution_modes": ["mtp3"],
        "all_modes_completed": False, "standard_qualified": False, "native_context": 16384,
        "binding_sha256": config["binding_sha256"], "native_stage_receipt_sha256": STAGE_SHA,
        "same_worker_integer_only_control_delta": True, "original_plan_content_sha256": quality.PLAN,
        "server_runs": [], "comparisons": [], "errors": [], "performance_qualified": False,
        "old_extra_EVERYROW_gate_passed": False, "old_extra_failed_rows": 116,
        "prior_standalone_STD_and_B2_MTP_regressions_retained": True}
    publish(args.output, report)
    allreports = {}; unloads = {}; no_reg = True
    lockpath = ROOT/"build/splash-tuning-gpu.lock"
    with lockpath.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for role in ("old", "new"):
                # Recheck frozen code/runtime/current proof immediately before each load.
                quality, binding = validate(config, True)
                build = Path(binding["runtime_build"])
                explicit = role_environment(config, role)
                environment, envproof = tuning.environment_for(SimpleNamespace(
                    package=Path(config["package"]), environment_overrides=explicit), "3")
                resolved = envproof["resolved_flash_environment"]
                check(all(resolved.get(key) == value for key,value in explicit.items()), "Launcher changed a registered explicit task flag")
                port = config["ports"][role]
                with socket.socket() as sock:
                    check(sock.connect_ex(("127.0.0.1",port)) != 0, "Registered private role port already occupied")
                command = server_command(config, build, role)
                logpath = args.output.with_name(args.output.stem+f"-{role}-mtp3.server.log")
                run = {"role":role,"execution_mode":"mtp3","integer_enabled":role=="new",
                    "command":command,"environment":envproof,"log":str(logpath),"quality_reports":[],"unloaded":False}
                report["server_runs"].append(run); publish(args.output,report)
                with logpath.open("x") as log:
                    server = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    run["pid"] = server.pid
                    try:
                        client = HTTPClient(SimpleNamespace(base_url=f"http://127.0.0.1:{port}",timeout=10,model=config["model"]))
                        waitargs = SimpleNamespace(idle_timeout=30,timeout=10,status_timeout=3,model=config["model"],max_context=16384)
                        deadline = time.monotonic()+config["startup_timeout"]
                        while True:
                            check(server.poll() is None,"Server exited during native task startup")
                            try:
                                initial = tuning.wait_idle(client,waitargs,timeout=min(30,max(.01,deadline-time.monotonic())))
                                break
                            except (OSError,ValueError,TimeoutError):
                                check(time.monotonic()<deadline,"Native task server startup timeout");time.sleep(.5)
                        run["initial_status"] = initial; publish(args.output,report)
                        for width in config["width_order"]:
                            path = args.output.with_name(args.output.stem+f"-{role}-mtp3-B{width}.json")
                            argv = [sys.executable,"-B",config["quality_driver"],"measure","--binding",config["binding"],
                                "--binding-sha256",config["binding_sha256"],"--role",role,"--width",str(width),
                                "--execution-mode","mtp3","--plan",config["original_plan"],"--expert-store",config["expert_store"],
                                "--base-url",f"http://127.0.0.1:{port}","--model",config["model"],"--output",str(path),"--run-root-gpu"]
                            subprocess.run(argv,cwd=ROOT,env=environment,check=True)
                            allreports[(role,width)] = path
                            run["quality_reports"].append({"width":width,"path":str(path),"sha256":sha(path)})
                            publish(args.output,report)
                        run["final_status"] = tuning.wait_idle(client,waitargs,identity=initial["identity"],timeout=60)
                    finally:
                        evidence = tuning.unload(server);run["unload_evidence"] = evidence
                        run["unloaded"] = evidence.get("process_group_gone") is True
                        unloads[role] = run["unloaded"] and evidence.get("returncode") == 0 and evidence.get("post_parent_exit_sigkill_required") is False
                        publish(args.output,report)
                    check(unloads[role],"Native worker/frontend did not unload cleanly")
            for width in config["width_order"]:
                path = args.output.with_name(args.output.stem+f"-mtp3-B{width}.comparison.json")
                argv = [sys.executable,"-B",config["quality_driver"],"compare","--plan",config["original_plan"],
                    "--old-report",str(allreports[("old",width)]),"--new-report",str(allreports[("new",width)]),"--output",str(path)]
                completed = subprocess.run(argv,cwd=ROOT,check=False)
                check(completed.returncode in (0,1) and path.exists(),"Comparator did not publish actual paired evidence")
                comparison = json.loads(path.read_text())
                check(comparison.get("valid") is True and comparison.get("width") == width and comparison.get("execution_mode") == "mtp3", "Invalid actual paired task evidence")
                no_reg = no_reg and comparison.get("no_new_task_regressions") is True
                report["comparisons"].append({"width":width,"execution_mode":"mtp3","path":str(path),"sha256":sha(path),
                    "valid":True,"no_new_task_regressions":comparison.get("no_new_task_regressions") is True})
                publish(args.output,report)
            report["completed"] = True;report["valid"] = no_reg;report["all_actual_evidence_valid"] = True
            report["no_new_task_regressions_at_both_widths"] = no_reg
            # A successful admission is emitted only after both actual paired comparisons and clean unloads.
            if no_reg:
                grade = {"schema":"current-BQSA4-integer-MTP3-original22-bothwidth-grade-admission-v1", "pass":True,
                    "execution_mode":"mtp3","qualified_native_widths":[4,2],"maximum_context_tokens":16384,
                    "worker_sha256":WORKER,"metallib_sha256":LIB,"source_identity_sha256":SOURCE,"BQSA4_policy_sha256":POLICY,
                    "compiled_worker_seal_sha256":SEAL,"binding_sha256":config["binding_sha256"],
                    "native_stage_receipt_sha256":STAGE_SHA,"original_plan_content_sha256":quality.PLAN,
                    "performance_qualified":False,"standard_qualified":False,"old_extra_EVERYROW_equivalence":False,
                    "comparisons":report["comparisons"],"both_roles_gracefully_unloaded":all(unloads.values())}
                gradepath = args.output.with_name(args.output.stem+"-mtp3.grade-admission.json")
                check(not gradepath.exists(),"Fresh actual task admission only")
                publish(gradepath,grade);report["mode_grade_receipt"] = {"path":str(gradepath),"sha256":sha(gradepath)}
        except Exception as error:
            report["errors"].append(type(error).__name__+": "+str(error));report["valid"] = False
        publish(args.output,report)
    print(json.dumps({"completed":report["completed"],"valid":report["valid"],"summary":str(args.output)}))
    return 0 if report["valid"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
