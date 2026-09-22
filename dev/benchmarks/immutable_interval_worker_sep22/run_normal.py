#!/usr/bin/env python3
"""Strict Root launcher; real serving occurs only after source/receipt admission."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--command", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    require(sha(a.command) == a.expected_sha256, "externally typed Root command pin required")
    c = json.loads(a.command.read_text())
    require(c["schema"] == "immutable96-current-MTP3-singleton-normal-Root-command-v1" and c["Root_GPU_only"] and c["flag"] in ("0", "1"), "known current singletonMTP3 normal command required")
    require(sha(__file__) == c["runner_source_sha256"], "normal runner drift")
    build = Path(c["source_build"])
    ready_path = build / "CPU_READY.json"
    require(sha(ready_path) == c["source_ready_sha256"], "Python source closure drift")
    ready = json.loads(ready_path.read_text())
    for r in ready["files"]:
        require(sha(r["path"]) == r["sha256"], "frozen adapter/driver/test source drift")
    parent_path = Path(c["parent_command_path"])
    require(sha(parent_path) == c["parent_command_sha256"], "original command controls drift")
    parent = json.loads(parent_path.read_text())
    expected = list(parent["argv"])
    expected[2] = str(build / "tuning.py")
    expected[expected.index("--binary") + 1] = str(Path(c["worker"]) / "splash-flash")
    expected[expected.index("--output") + 1] = str(Path(c["cwd"]) / "build/release/flash" / ("sep22-immutable96-index-Q4-matched-flag" + c["flag"] + "-model-and-quality-v1.json"))
    expected[expected.index("--port") + 1] = "8047"
    expected.extend(("--env", "SPLASH_FLASH_IMMUTABLE_INTERVAL_INDEX_SEP22=" + c["flag"],
                     "--native-receipt", c["native_receipt"], "--native-receipt-sha256", c["native_receipt_sha256"]))
    require(c["argv"] == expected and c["cwd"] == parent["cwd"], "only source/runtime/output/port and documented index admission may change")
    output = Path(expected[expected.index("--output") + 1])
    require(not output.exists(), "fresh normal report required")
    spec = importlib.util.spec_from_file_location("_immutable96_live_normal_admission", build / "semantic_quality.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.load(c["worker"], c["flag"] == "1", True, c["native_receipt"], c["native_receipt_sha256"])
    require((c["worker_sha256"], c["worker_seal_sha256"], c["source_policy_sha256"], c["metallib_sha256"]) ==
            (adapter.EXE, adapter.SEAL, adapter.SOURCE, adapter.LIB), "exact child source/artifact profile required")
    if a.validate_only:
        print(json.dumps({"pass": True, "GPU_executed": False, "model_fixture_tokenizer_response_operand_payload_read_or_hashed": False, "flag": c["flag"]}))
        return 0
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SPLASH_FLASH_", "FLASH_"))}
    r = subprocess.run(c["argv"], cwd=c["cwd"], env=env)
    return r.returncode if r.returncode >= 0 else 128 - r.returncode

if __name__ == "__main__":
    raise SystemExit(main())
