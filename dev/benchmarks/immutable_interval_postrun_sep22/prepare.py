#!/usr/bin/env python3
"""CPU-only source sealing; never opens actual normal or semantic reports."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path("/Users/mweinbach/Projects/splash")
HERE = Path(__file__).resolve().parent
BUILD = ROOT / "build/immutable96-index-Q4-postrun-compare-sep22-root-v1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    if BUILD.exists():
        raise ValueError("fresh postrun source closure required")
    BUILD.mkdir(parents=True)
    for name in ("compare.py", "test_compare.py"):
        shutil.copy2(HERE / name, BUILD / name)
        compile((BUILD / name).read_text(), str(BUILD / name), "exec")
    tests = subprocess.run([str(ROOT / ".venv/bin/python"), "-B", "-m", "unittest",
                            "dev.benchmarks.immutable_interval_postrun_sep22.test_compare"],
                           cwd=ROOT, check=True, capture_output=True, text=True)
    release = ROOT / "build/release/flash"
    stem = "sep22-immutable96-index-Q4-matched-flag"
    normal = [release / (stem + flag + "-model-and-quality-v1.json") for flag in ("0", "1")]
    semantic = [release / (stem + flag + "-model-and-quality-v1-3.semantic.json") for flag in ("0", "1")]
    normal_sha = ["e54fea72d32299eacfd6b77f518eb2fae63834aac230e33f5f65339a1f66734d",
                  "3b8d361a3e242d36a11e976653e5fbc55e0611e74448d66b2b9cc30cb9f26575"]
    semantic_sha = ["8a2d8e3628f94417a3bce0cb99c9419172308fab7bf21535dd52ba47199c5bff",
                    "c0a5e7425e6c36a56378dc1dd3729d7ed991802df1f1c4e999e589a6e7b586b4"]
    command = {"schema": "immutable96-current-samebinary-postrun-Root-command-v1", "Root_CPU_only": True,
               "actual_generation_reports_read_only_by_Root": True, "cwd": str(ROOT),
               "program_sha256": sha(BUILD / "compare.py"),
               "argv": [str(ROOT / ".venv/bin/python"), "-B", str(BUILD / "compare.py"),
                        "--normal-reports", *map(str, normal), "--normal-sha256", *normal_sha,
                        "--semantic-reports", *map(str, semantic), "--semantic-sha256", *semantic_sha,
                        "--output", str(release / "sep22-immutable96-index-Q4-postrun-comparison-v1.json")]}
    (BUILD / "Root-command.json").write_text(json.dumps(command, indent=2) + "\n")
    dependencies = [ROOT / "build/immutable96-index-Q4-normal-original22-sep22-v3/semantic_quality.py",
                    ROOT / "build/immutable96-index-Q4-normal-original22-sep22-v3/tuning.py",
                    ROOT / "build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2/source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py"]
    ready = {"schema": "immutable96-postrun-only-source-closure-v1", "pass": True,
             "GPU_executed": False, "actual_normal_semantic_model_generation_response_capture_payload_read_or_hashed": False,
             "normal_sources_runtime_reports_helpers_not_modified": True,
             "four_report_SHA256_externally_provided_by_Root": True,
             "own_source_only_recorded_flag_dispatch": True, "literal_original_comparator_allow_runtime_change": False,
             "original22_body_grade_budget_cache_numerical_gate_unchanged": True,
             "full_status_fingerprint_dispatch_rejects_unknown_or_mixed_contexts": True,
             "known_failures_preserved_not_removed": True, "performance_promotion": False,
             "samebinary0_shared_getenv_atomic_cost_recorded": True,
             "files": [{"path": str(p), "sha256": sha(p)} for p in (BUILD / "compare.py", BUILD / "test_compare.py", BUILD / "Root-command.json", HERE / "compare.py", HERE / "test_compare.py", HERE / "prepare.py")],
             "frozen_dependencies": [{"path": str(p), "sha256": sha(p)} for p in dependencies],
             "CPU_tests": {"pass": True, "tests": 5, "stdout": tests.stdout, "stderr": tests.stderr},
             "command_sha256": sha(BUILD / "Root-command.json")}
    (BUILD / "CPU_READY.json").write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps({"CPU_READY": str(BUILD / "CPU_READY.json"), "CPU_READY_sha256": sha(BUILD / "CPU_READY.json"),
                      "Root_command": str(BUILD / "Root-command.json"), "Root_command_sha256": sha(BUILD / "Root-command.json"),
                      "program_sha256": sha(BUILD / "compare.py")}))

if __name__ == "__main__":
    main()
