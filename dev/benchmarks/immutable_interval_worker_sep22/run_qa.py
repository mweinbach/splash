#!/usr/bin/env python3
"""Strict Root-only execution of a preregistered source/metadata QA command."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def validate(command_path, expected_sha):
    require(sha(command_path) == expected_sha, "externally pinned command metadata drift")
    c = json.loads(command_path.read_text())
    require(c["schema"] == "immutable96-bounded-samebinary-Root-command-v1" and c["Root_GPU_only"], "registered Root-only command required")
    require(sha(__file__) == c["runner_source_sha256"], "runner source drift")
    ready_path = Path(c["CPU_READY_path"])
    require(sha(ready_path) == c["CPU_READY_sha256"], "CPU closure metadata drift")
    ready = json.loads(ready_path.read_text())
    require(ready["pass"] and not ready["GPU_executed"] and ready["bounded_expected_unique_frames"] == 10 and ready["bounded_expected_repeated_frames"] == 0, "complete bounded CPU closure required")
    require(ready["all8_actual_owned_guard_probes_exact_error_category_and_text"], "actual owned alias error coverage required")
    require(ready["separate_held_host_admission_bytes"] == 2 << 30 and ready["source_campaign_spill_upper_bound_bytes"] < 4 << 30, "whole host/spill admission source proof required")
    require(sha(c["argv"][0]) == ready["oracle_sha256"] == c["oracle_sha256"], "native QA executable drift")
    require(sha(c["argv"][3]) == ready["metallib_sha256"] == c["metallib_sha256"], "exact current library drift")
    for r in ready["objects"] + ready["headers"] + ready["source_files"]:
        require(sha(r["path"]) == r["sha256"], "actual QA source/header/object drift: " + r["path"])
    require(sha(Path(ready["worker"]) / "compiled-cpu-seal.json") == ready["worker_seal_sha256"], "current worker seal drift")
    parent_path = Path(c["parent_command_path"])
    require(sha(parent_path) == c["parent_command_sha256"], "original current policy command drift")
    parent = json.loads(parent_path.read_text())
    expected_env = dict(parent["environment"])
    expected_env["SPLASH_FLASH_IMMUTABLE_INTERVAL_INDEX_SEP22"] = "0" if c["role"] == "export" else "1"
    require(c["environment"] == expected_env, "only immutable-index flag may differ from registered current numerical policy")
    require(c["role"] in ("export", "compare") and c["argv"][1:3] == ["--gpu", c["role"]], "samebinary native role required")
    require(c["argv"][4:6] == parent["argv"][4:6] and c["cwd"] == parent["cwd"], "model/fixture/cwd identity drift")
    require(c["environment"]["SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22"] == "1", "qualified current rawQ4 route required")
    report = Path(c["argv"][6])
    for suffix in ("", ".partial", ".failure.json", ".writing"):
        require(not Path(str(report) + suffix).exists(), "fresh report path required")
    spill = Path(c["argv"][7])
    if c["role"] == "export":
        require(not spill.exists(), "fresh bounded spill path required")
    else:
        # Root-only execution reads the completed CONTROL metadata here.
        previous = json.loads(Path(c["control_report"]).read_text())
        require(previous["pass"] and previous["role"] == "export" and previous["backend_destroyed"] and previous["frames"] == 10 and previous["repeated_frames"] == 0, "healthy completed separate flag0 process required")
        require(previous["producer"]["immutable_index_source_policy_sha256"] == c["source_policy_sha256"] and not previous["producer"]["immutable_index_requested"], "matching original-loop control policy required")
        require(previous["producer"]["VerifyR4_indexed_accepts"] == 0 and previous["producer"]["VerifyR4_original_callbacks"] == 624, "real original-loop Verify624 control required")
        require(previous["allocation"]["governor_snapshot"]["reserved_bytes"] == 0, "actual control zero reservations required")
        require((spill / "complete.json").exists(), "completed bounded control spill required")
    return c

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--command", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    c = validate(a.command.resolve(), a.expected_sha256)
    if a.validate_only:
        print(json.dumps({"pass": True, "GPU_executed": False, "model_fixture_operand_export_payload_read_or_hashed": False, "role": c["role"]}))
        return 0
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPLASH_FLASH_")}
    env.update(c["environment"])
    r = subprocess.run(c["argv"], cwd=c["cwd"], env=env)
    return r.returncode if r.returncode >= 0 else 128 - r.returncode

if __name__ == "__main__":
    raise SystemExit(main())
