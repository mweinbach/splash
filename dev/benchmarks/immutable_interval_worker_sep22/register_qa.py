#!/usr/bin/env python3
"""Register fresh bounded Root commands; source/artifact metadata only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path("/Users/mweinbach/Projects/splash")
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "build/trunkverify-rawQ4-GDN26-VerifyR4-sep22-v1/Root-rawQ4-native-command.json"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path, required=True)
    a = p.parse_args()
    build = a.build.resolve()
    require = lambda ok, message: None if ok else (_ for _ in ()).throw(ValueError(message))
    ready_path = build / "CPU_READY.json"
    ready = json.loads(ready_path.read_text())
    require(ready["pass"] and len(ready["objects"]) == 53, "complete current53-object CPU closure required")
    native = PARENT.parent
    original = native / "source/dev/benchmarks/trunk_verify_exact_sep22/oracle.mm"
    require(sha(original) == "40011a8f59b7fff6013b5c73ac5588f5d5c3e497b7b74521565d14f882cacb04", "authenticated original native harness source drift")
    manifest = json.loads((native / "manifest.json").read_text())
    require(manifest["native_harness_source_sha256"] == sha(original), "sealed original native source witness drift")
    for h in manifest["headers"]:
        require(sha(h["path"]) == h["sha256"], "authenticated original friend/inspector header drift")
    require(not (build / "root-export-command.json").exists(), "fresh registration required")
    runner = build / "run-root-qa.py"
    shutil.copy2(HERE / "run_qa.py", runner)
    parent = json.loads(PARENT.read_text())
    basename = "sep22-immutable96-index-Q4-bounded-native-v1"
    reports = {role: ROOT / "build/release/flash" / (basename + "-" + role + ".json") for role in ("export", "compare")}
    spill = ROOT / "build/release/flash" / (basename + "-spill")
    provenance = json.loads((build / "provenance.json").read_text())
    commands = []
    for role in ("export", "compare"):
        env = dict(parent["environment"])
        env["SPLASH_FLASH_IMMUTABLE_INTERVAL_INDEX_SEP22"] = "0" if role == "export" else "1"
        c = {
            "schema": "immutable96-bounded-samebinary-Root-command-v1",
            "Root_GPU_only": True,
            "role": role,
            "cwd": parent["cwd"],
            "argv": [str(build / "oracle"), "--gpu", role, str(build / "splash.metallib"), *parent["argv"][4:6], str(reports[role]), str(spill)],
            "environment": env,
            "source_policy_sha256": provenance["source_policy"],
            "oracle_sha256": ready["oracle_sha256"],
            "metallib_sha256": ready["metallib_sha256"],
            "CPU_READY_path": str(ready_path),
            "CPU_READY_sha256": sha(ready_path),
            "runner_source_sha256": sha(runner),
            "parent_command_path": str(PARENT),
            "parent_command_sha256": sha(PARENT),
            "control_report": str(reports["export"]),
            "scope": "one Prefill2K Verify4 keep2 genuine correctionR1; all134 state and216 physical tapes; original unused PLE/count tails known initialized in QA only; actual8 owned alias errors",
            "samebinary_flag0_has_getenv_counter_overhead": True,
            "performance_teacher_head_worker_original22_qualified": False,
            "model_fixture_capture_response_operand_export_payload_read_or_hashed": False,
        }
        path = build / ("root-" + role + "-command.json")
        path.write_text(json.dumps(c, indent=2) + "\n")
        commands.append({"role": role, "path": str(path), "sha256": sha(path),
                         "launch_argv": [str(ROOT / ".venv/bin/python"), "-B", str(runner), "--command", str(path), "--expected-sha256", sha(path)]})
    dry = subprocess.run([*commands[0]["launch_argv"], "--validate-only"], cwd=ROOT, check=True, capture_output=True, text=True)
    registration = {
        "schema": "immutable96-bounded-native-source-and-Root-registration-v1",
        "pass": True, "GPU_executed": False,
        "original_harness_path": str(original), "original_harness_sha256": sha(original),
        "original_header_source_pins_verified": True,
        "CPU_READY_path": str(ready_path), "CPU_READY_sha256": sha(ready_path),
        "program_sources": [{"path": str(path), "sha256": sha(path)} for path in (HERE / "prepare_qa.py", HERE / "register_qa.py", HERE / "run_qa.py", runner)],
        "commands": commands, "export_source_dryrun": json.loads(dry.stdout),
        "model_fixture_capture_response_operand_export_payload_read_or_hashed": False,
    }
    (build / "Root-registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    print(json.dumps({"pass": True, "registration": str(build / "Root-registration.json"), "registration_sha256": sha(build / "Root-registration.json"), "commands": commands}))

if __name__ == "__main__":
    main()
