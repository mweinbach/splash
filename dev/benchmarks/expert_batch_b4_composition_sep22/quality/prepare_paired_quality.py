#!/usr/bin/env python3
"""CPU/source-only preparation; never opens model, fixture, or response data."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path("/Users/mweinbach/Projects/splash")
CODE = ROOT/"dev/benchmarks/expert_batch_b4_composition_sep22/quality"
BUILD = ROOT/"build/integer-b4-twopass-composed-sep22-worker-v2"
PARENT = ROOT/"build/batch-twoPass-original22-quality-sep22-root-v5"
SUPPLEMENT = ROOT/"build/b4-composition-service-adapter-sep22-v1"
IDENTITIES = ROOT/"build/bqsa4-integer-source-derived-identities-sep22-v1"
STAGE = ROOT/"build/release/flash/sep22-current-BQSA4-integer-Root-native-stage-admission-v1.json"
FLAGS = ROOT/"build/bqsa4-integer-native-inventory-observer-sep22-v3/expected.json"
PLAN_SHA = "3fd2bbccf372dd78378929015db04ea347c22d4394c1eea55f39cb5f59a8400c"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

def check(value, message):
    if not value:
        raise ValueError(message)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    check(not destination.exists() and not args.output.exists(), "Fresh source tree and unused Root output required")
    check(destination.is_relative_to(ROOT/"build"), "Private build metadata directory required")
    source = destination/"source"
    source.mkdir(parents=True)
    for name in ("run_paired_quality.py", "paired_quality.py", "tests_paired_quality.py", "tests_launcher.py", "prepare_paired_quality.py"):
        shutil.copy2(CODE/name, source/name)
    shutil.copy2(PARENT/"source/splash_tuning_sep21.py",source/"splash_tuning_sep21.py")
    launcher = load(source/"run_paired_quality.py", "_prepared_source_launcher")
    parent_config = json.loads((PARENT/"root-four-profile-quality-config.json").read_text())
    parent_ready = json.loads((PARENT/"CPU_READY.json").read_text())
    declared_plan = next(x for x in parent_ready["source_files"] if x["path"] == parent_config["original_plan"])
    check(declared_plan["sha256"] == PLAN_SHA, "Frozen parent's declared ORIGINAL22 file digest changed")
    # Read only source recipes and explicitly authorized small metadata.
    check(sha(STAGE) == launcher.STAGE_SHA, "Root actual native stage receipt external digest changed")
    launcher.validate_stage(json.loads(STAGE.read_text()))
    check(sha(IDENTITIES/"SOURCE_READY.json") == "13e3b754ed6bfdace8f5fa25a6b062e8f77b840672b68258d341c1ee4e72bd6b", "Exact source-derived identity receipt required")
    identity_ready = json.loads((IDENTITIES/"SOURCE_READY.json").read_text())
    identities = identity_ready["identities"]
    policy_path = BUILD/"source/dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp"
    policy = policy_path.read_text()
    constants = {key:re.search(r'inline constexpr const char \*'+key+r' = "([^"]+)";',policy).group(1)
                 for key in ("schema","policy","sourceSHA","shaderSHA","hostSHA")}
    inner = BUILD/"source/dev/benchmarks/expert_batch_b4_composition_sep22/adapter.py"
    supplemental = SUPPLEMENT/"source/service_adapter.py"
    binding = {"schema":"integer-B4-twopass-adapter-binding-v1", "runtime_build":str(BUILD),
        "overlay_manifest_sha256":sha(BUILD/"overlay-manifest.json"), "compiled_seal_sha256":sha(BUILD/"compiled-cpu-seal.json"),
        "worker_sha256":sha(BUILD/"splash-flash"), "metallib_sha256":sha(BUILD/"splash.metallib"),
        "policy_source_sha256":constants["sourceSHA"], "policy_text":constants["policy"],
        "policy_shader_sha256":constants["shaderSHA"], "policy_host_sha256":constants["hostSHA"],"policy_schema":constants["schema"],
        "integer_source_identity_sha256":launcher.SOURCE,
        "original22_plan_content_sha256":"a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac",
        "target_numeric_parent_sha256":identities["wrapped_target_numeric_parent_SHA"],
        "target_base_numeric_parent_sha256":identities["raw_Store_numeric_parent_SHA"],
        "target_execution_base_child_sha256":identities["raw_Store_execution_child_SHA"],
        "target_execution_child_sha256":identities["wrapped_target_execution_child_SHA"],
        "qualified_shapes_or_scores_inherited":False, "inner_adapter_path":str(inner), "inner_adapter_sha256":sha(inner),
        "service_adapter_path":str(supplemental), "service_adapter_sha256":sha(supplemental),
        "frozen_original22_common_status_and_coverage_required":True,
        "native_stage_receipt":str(STAGE), "native_stage_receipt_sha256":launcher.STAGE_SHA,
        "source_derived_identity_receipt_sha256":sha(IDENTITIES/"SOURCE_READY.json")}
    binding_path = destination/"paired-composition-binding.json"
    write(binding_path,binding)
    base = dict(parent_config["base_environment"])
    base.update({launcher.INTEGER_FLAG:"0", launcher.BQSA_FLAG:"1",
        "SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21":"0",
        "SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22":"0", "SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22":"0"})
    pins = {}
    def pin(path):
        pins[str(path.resolve())] = sha(path)
    for path in source.iterdir():
        if path.is_file():pin(path)
    # Code dependencies are copied-source or live source; fixture digest is
    # declared separately and opened only by Root's explicit run path.
    for record in parent_ready["source_files"]:
        path = Path(record["path"])
        if path.suffix == ".py" and (path.is_relative_to(ROOT/"dev") or path.is_relative_to(ROOT/"server") or path.is_relative_to(ROOT/"install")):
            check(sha(path) == record["sha256"], "Unchanged original code dependency drift: "+str(path));pin(path)
    for path in (BUILD/"splash-flash", BUILD/"splash.metallib", BUILD/"overlay-manifest.json", BUILD/"compiled-cpu-seal.json",
        policy_path,inner,supplemental,SUPPLEMENT/"SOURCE_READY.json",
        ROOT/"dev/benchmarks/expert_batch_b4_composition_sep22/service_adapter.py",
        ROOT/"build/release/flash/sep22-B4-composition-service-supplement-independent-source-review-v1.json",
        IDENTITIES/"SOURCE_READY.json",FLAGS,binding_path,STAGE,
        PARENT/"source/batch_quality.py",PARENT/"CPU_READY.json"):
        pin(path)
    for record in identity_ready["inputs"]:
        path = IDENTITIES/record["path"]
        check(sha(path) == record["sha256"], "Source identity probe input drift");pin(path)
    config = {"schema":launcher.SCHEMA, "width_order":[4,2], "modes":["mtp3"],
        "same_worker_integer_only_control_delta":True, "original_plan":parent_config["original_plan"],
        "original_plan_file_sha256":PLAN_SHA, "quality_driver":str(source/"paired_quality.py"),
        "original_driver_snapshot":str(source/"splash_tuning_sep21.py"),"binding":str(binding_path),"binding_sha256":sha(binding_path),
        "native_stage_receipt":str(STAGE),"native_stage_receipt_sha256":launcher.STAGE_SHA,
        "native_stage_numeric_flags":json.loads(FLAGS.read_text())["numeric_flags"],
        "package":parent_config["package"],"model":parent_config["model"],"expert_store":parent_config["expert_store"],
        "base_environment":base,"ports":{"old":8058,"new":8059},"startup_timeout":600,"artifact_pins":pins,
        "unchanged_original22_bodies_budgets_graders":True,"all_original22_and_mixed_real_lanes_required":True,
        "prior_extra_EVERYROW_failed_rows":116,"prior_standalone_Std_B2MTP_regressions_preserved":True,
        "new_common_gates_are_executed_not_a_boolean_contract":True,"task_or_performance_inherited":False}
    config_path = destination/"root-paired-MTP3-quality-config.json"
    write(config_path,config)
    launcher.validate(config,False)  # Source/metadata admission; no fixture/tokenizer/model access.
    old = launcher.role_environment(config,"old");new = launcher.role_environment(config,"new")
    check([key for key in old if old[key]!=new[key]] == [launcher.INTEGER_FLAG], "Exactly one integer toggle")
    python = ROOT/".venv/bin/python"
    test_results = []
    for path in (source/"tests_paired_quality.py",source/"tests_launcher.py"):
        result = subprocess.run([str(python),"-B",str(path)],cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT)},text=True,capture_output=True)
        check(result.returncode == 0, "Source-only CPU tests failed: "+result.stdout+result.stderr)
        test_results.append({"source":str(path),"pass":True,"stdout":result.stdout,"stderr":result.stderr})
    dry = subprocess.run([str(python),"-B",str(source/"run_paired_quality.py"),"--config",str(config_path),
        "--config-sha256",sha(config_path),"--output",str(args.output)],cwd=ROOT,text=True,capture_output=True)
    check(dry.returncode == 0, "Source admission failed: "+dry.stdout+dry.stderr)
    argv = [str(python),"-B",str(source/"run_paired_quality.py"),"--config",str(config_path),
        "--config-sha256",sha(config_path),"--output",str(args.output),"--run-root-gpu"]
    command_path = destination/"root-command.txt"
    command_path.write_text(shlex.join(argv)+"\n")
    command_json = destination/"root-command.json"
    write(command_json,{"schema":"current-BQSA4-integer-original22-MTP3-pinned-invocation-v1", "argv":argv,
        "cwd":str(ROOT),"config_sha256":sha(config_path),"launcher_sha256":sha(source/"run_paired_quality.py"),
        "binding_sha256":sha(binding_path),"output":str(args.output),"only_Root_executes_model_or_service":True})
    ready = {"schema":"current-BQSA4-integer-paired-original22-MTP3-CPU-source-seal-v1","pass":True,
        "GPU_executed":False,"model_fixture_token_capture_response_operand_payload_read_or_hashed":False,
        "source_files":[{"path":path,"sha256":digest} for path,digest in pins.items()],
        "config_sha256":sha(config_path),"root_command_sha256":sha(command_json),"binding_sha256":sha(binding_path),
        "current_Root_native_stage_receipt_sha256":launcher.STAGE_SHA,"source_derived_identity_receipt_sha256":sha(IDENTITIES/"SOURCE_READY.json"),
        "profiles":2,"modes":["mtp3"],"width_order":[4,2],"original_cases_perwidth":22,
        "original_body_budget_grader_changes":0,"CPU_tests":test_results,"source_only_admission":json.loads(dry.stdout),
        "core_tests_execute_sibling_frozen_quality_source":True,
        "all_actual_test_and_runtime_live_import_sources_pinned":True,
        "actual_integer_exposure_required_new_B4_R16_new_B2_R8":True,
        "actual_task_or_performance_qualified":False,"independent_source_review_pending":True}
    write(destination/"CPU_READY.json",ready)
    print(json.dumps({"CPU_READY":str(destination/"CPU_READY.json"),"root_command":str(command_json),
        "binding_sha256":sha(binding_path),"config_sha256":sha(config_path),"GPU_executed":False}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
