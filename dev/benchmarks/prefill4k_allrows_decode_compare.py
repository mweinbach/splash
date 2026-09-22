#!/usr/bin/env python3
"""Freeze matched MPP/C2 decode cohort flags and bodies; never run inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "local/Qwen3.8-Flash-Next-oQ4e-mtp"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/prefill4k-allrows-qmv-c2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh matched cohort plan")
    performance = ROOT / "build/release/flash/ultra-locality-prefill4k-allrows-full512-first.json"
    baseline = json.loads(performance.read_text())
    code = next(row for row in baseline["plan"] if row["name"] == "code")
    assert code["prompt_tokens"] == 2048
    body = {"model": MODEL, "messages": [{"role": "user", "content": code["content"]}],
            "max_completion_tokens": 256, "temperature": 0, "seed": 0,
            "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
    semantic_path = ROOT / "build/release/flash/prefill4k-semantic-plan-v1.json"
    semantic = json.loads(semantic_path.read_text())
    cases = semantic["cases"]
    baseline_semantic = ROOT / "build/release/flash/prefill4k-semantic-prefill4k-allrows-full512-first.json"
    frozen_quality = json.loads(baseline_semantic.read_text())
    failed_ids = [row["id"] for row in frozen_quality["cases"] if not row["task_passed"]]
    assert len(cases) == 22 and sorted(failed_ids) == ["arithmetic_inventory", "arithmetic_signed"]
    cohorts = []
    for label, enabled, columns, derivative in [
        ("same-build-mpp-control", "0", "1", "2e858faa201554642a443d48d38b7302159fe1a811c028a895454cc8ce5c073d"),
        ("same-build-c2-decode", "1", "2", "ce70bf75e8c89f97e5f6bc37a540ae2be93ad85fba5fce5598dac759b0739601"),
    ]:
        flags = {"SPLASH_FLASH_ALLROWS_FULL512_TARGET": "1",
                 "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV": enabled,
                 "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV_COLUMNS": columns,
                 "SPLASH_FLASH_INT8_EXPERT_STORE": str(ROOT / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1"),
                 "SPLASH_FLASH_PREFILL_ROWS": "2048", "SPLASH_FLASH_MTP_DRAFT_DEPTH": "3",
                 "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY": "1", "SPLASH_FLASH_PREFILL_DENSE_TILES": "1",
                 "SPLASH_FLASH_PLE_SSD_STREAMING": "1", "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE": "0",
                 "SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT": "0"}
        command = [str(ROOT / ".venv/bin/python"), str(ROOT / "dev/benchmarks/prefill4k_allrows_http.py"),
                   "--build", str(args.build.resolve()), "--rows", "2048", "--port", "8011",
                   "--admission-report", f"build/release/flash/CHOOSE-FRESH-{label}-admission.json",
                   "--witness", f"build/release/flash/CHOOSE-FRESH-{label}-launch.json"]
        if enabled == "1":
            command += ["--gathered-qmv", "--qmv-columns", "2"]
        command += ["--run"]
        cohorts.append({"label": label, "binary": str(args.build.resolve() / "splash-flash"),
                        "policy_overrides": flags, "target_numerical_derivative_sha256": derivative,
                        "launch_command_root_only": command, "warmup_requests": 1,
                        "measured_requests": 3, "body": body})
    report = {"schema": 1, "gpu_work": False, "model_loaded": False,
              "same_binary_for_both_cohorts": True, "coefficient_store_unchanged": True,
              "large_prefill_producer_unchanged": True,
              "small_row_scope": "physical rows1..16 target decode/verifier and tiny prefill tails; coefficient bytes and late row scale unchanged, reduction policy differs",
              "phase_route_scope": "executor API has no semantic phase argument; row restrictions deliberately cover tiny tails",
              "require_before_root_http": "C2 one-layer frozen F64/cancellation/sign/BF16/ownership gates must pass",
              "request_match": "identical existing nonce/content,2K uncached/256 full output,reasoning none,temp0,seed0; compare3 warmed repeats",
              "require_actual_status": {"identity.target_all_rows_full512": True,
                                        "ple_storage.target_original_disk_tensor_count": 432,
                                        "ple_storage.gpu_mapped_original_bytes": 6370164736,
                                        "persisted_experts.expert_count": 24576},
              "compare_counters": ["large_row_* prefills equal across cohorts", "gathered_qmv_* zero incontrol/nonzero incandidate", "general expert counters and small-row differences", "MTP drafted/accepted", "native and stream timing separately", "fresh idle/no failures/denials"],
              "frozen_quality_case_ids": [case["id"] for case in cases],
              "baseline_task_passed_cases": frozen_quality["task_passed_cases"], "baseline_existing_failure_ids": failed_ids,
              "default_promotion_allowed": False, "cohorts": cohorts,
              "source_sha256": {str(path): sha(path) for path in [performance, semantic_path, baseline_semantic]},
              "runtime_sha256": {name: sha(args.build / name) for name in ["splash-flash", "splash.metallib"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    print(json.dumps({"gpu_work": False, "cohorts": len(cohorts), "frozen_cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
