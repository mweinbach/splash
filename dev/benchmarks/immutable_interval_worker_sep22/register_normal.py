#!/usr/bin/env python3
"""Bind normal commands only after an externally pinned real Root admission."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path("/Users/mweinbach/Projects/splash")
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "build/rawQ4-GDN26-matched-model-sep22-root-v2/root-flag1-command.json"
WORKER = ROOT / "build/immutable96-index-Q4-sep22-worker-v1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--source-build", type=Path, required=True)
    p.add_argument("--native-receipt", type=Path, required=True)
    p.add_argument("--native-receipt-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build, out = a.source_build.resolve(), a.output.resolve()
    require(not out.exists(), "fresh normal command registration required")
    ready_path = build / "CPU_READY.json"
    ready = json.loads(ready_path.read_text())
    require(ready["pass"] and ready["literal_inverse_source_journal_pass"], "frozen original-driver inverse source witness required")
    for r in ready["files"]:
        require(sha(r["path"]) == r["sha256"], "Python admission/driver source drift")
    spec = importlib.util.spec_from_file_location("_immutable96_bound_normal", build / "semantic_quality.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.authenticate(WORKER, a.native_receipt.resolve(), a.native_receipt_sha256, True)
    parent = json.loads(PARENT.read_text())
    old = parent["argv"]
    def option(name):
        return old[old.index(name) + 1]
    require(all(option(key) == value for key, value in {
        "--mtp": "3", "--contexts": "2048", "--batches": "1", "--workloads": "coding",
        "--output-tokens": "256", "--warmup": "1", "--trials": "3", "--max-context": "16384"}.items()),
        "exact canonical2K256/warm1/three trials/16K singleton MTP3 controls required")
    out.mkdir(parents=True)
    runner = out / "run-root-normal.py"
    shutil.copy2(HERE / "run_normal.py", runner)
    commands = []
    for flag in ("0", "1"):
        argv = list(old)
        argv[2] = str(build / "tuning.py")
        argv[argv.index("--binary") + 1] = str(WORKER / "splash-flash")
        argv[argv.index("--output") + 1] = str(ROOT / "build/release/flash" / ("sep22-immutable96-index-Q4-matched-flag" + flag + "-model-and-quality-v1.json"))
        argv[argv.index("--port") + 1] = "8047"
        argv.extend(("--env", "SPLASH_FLASH_IMMUTABLE_INTERVAL_INDEX_SEP22=" + flag,
                     "--native-receipt", str(a.native_receipt.resolve()),
                     "--native-receipt-sha256", a.native_receipt_sha256))
        c = {"schema": "immutable96-current-MTP3-singleton-normal-Root-command-v1", "Root_GPU_only": True,
             "flag": flag, "cwd": parent["cwd"], "argv": argv,
             "source_build": str(build), "source_ready_sha256": sha(ready_path),
             "parent_command_path": str(PARENT), "parent_command_sha256": sha(PARENT),
             "native_receipt": str(a.native_receipt.resolve()), "native_receipt_sha256": a.native_receipt_sha256,
             "worker": str(WORKER), "worker_sha256": adapter.EXE, "worker_seal_sha256": adapter.SEAL,
             "source_policy_sha256": adapter.SOURCE, "metallib_sha256": adapter.LIB,
             "runner_source_sha256": sha(runner), "samebinary_flag0_shared_getenv_counter_overhead": True,
             "best_parent53_226_improvement_not_inferred_from_instrumented0_gain": True,
             "original22_grade_plan_body_budget_and_normal_rate_math_unchanged": True}
        path = out / ("root-flag" + flag + "-command.json")
        path.write_text(json.dumps(c, indent=2) + "\n")
        commands.append({"flag": flag, "path": str(path), "sha256": sha(path),
                         "launch_argv": [str(ROOT / ".venv/bin/python"), "-B", str(runner),
                                         "--command", str(path), "--expected-sha256", sha(path)]})
    witness = {"schema": "immutable96-current-normal-externally-bound-registration-v1", "pass": True,
               "GPU_executed": False, "native_receipt_sha256": a.native_receipt_sha256,
               "source_ready_sha256": sha(ready_path), "commands": commands,
               "programs": [{"path": str(p), "sha256": sha(p)} for p in (HERE / "register_normal.py", HERE / "run_normal.py", runner)]}
    (out / "binding.json").write_text(json.dumps(witness, indent=2) + "\n")
    print(json.dumps({"pass": True, "binding": str(out / "binding.json"), "binding_sha256": sha(out / "binding.json"), "commands": commands}))

if __name__ == "__main__":
    main()
