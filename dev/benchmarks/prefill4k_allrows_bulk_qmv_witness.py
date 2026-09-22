#!/usr/bin/env python3
"""Bounded combined snapshot/source/admission witness; never load weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from prefill4k_allrows_overlay import ROOT, module


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-allrows-qmv-c2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh witness path")
    build = args.build.resolve()
    manifest = json.loads((build / "overlay-manifest.json").read_text())
    base = Path(manifest["bulk_qmv_base_build"])
    base_path = base / "overlay-manifest.json"
    assert sha(base_path.read_bytes()) == manifest["bulk_qmv_base_manifest_sha256"]
    transforms = [module(name).transform for name in manifest["bulk_qmv_transform_order"]]
    extras = module("allrows_qmv").extra_files()
    extras.update(module("allrows_qmv_c2").extra_files())
    mismatch = []
    for record in manifest["files"]:
        copied = (build / "source" / record["path"]).read_bytes()
        if "base_overlay_sha256" in record:
            original = (base / "source" / record["path"]).read_bytes()
            assert sha(original) == record["base_overlay_sha256"]
            expected = original.decode()
            for transform in transforms:
                expected = transform(record["path"], expected)
            if record["path"] == "runtime/flash/FlashWorker.mm":
                expected = expected.replace("      (void)gathered_i8_qmv::requested();", "      (void)gathered_i8_qmv::requestedColumns();\n      (void)gathered_i8_qmv::requested();")
            data = expected.encode()
        else:
            data = extras[record["path"]].encode()
        if copied != data or sha(copied) != record["overlay_sha256"]:
            mismatch.append(record["path"])
    forward = (build / "source/runtime/flash/FlashForward.cpp").read_text()
    worker = (build / "source/runtime/flash/FlashWorker.mm").read_text()
    checks = {
        "sources_fresh": not mismatch,
        "bulk_five_plane_admission_preserved": "total += prefill4k::bulkExactPlannedBytes()" in forward,
        "bulk_allocation_actual_check_preserved": "bulkPlaneBytes != prefill4k::bulkExactPlannedBytes()" in forward,
        "bulk_original_execution_id_preserved": "private-qsa-bulk-prefill-begin0-r2048-w128-m16p1-p4-m32p4-v2" in forward,
        "bulk_sg8_execution_id_preserved": "private-qsa-bulk-prefill-temporal-sg8-w4to15-m32-p4-t256-v1" in forward,
        "c2_execution_policy_recorded": "gathered_i8_qmv::numericalPolicy" in forward,
        "phase_scope_recorded": "expert_kernel_phase_attribution" in worker,
        "qmv_only_small_rows": "rows <= 16" in forward,
    }
    profile = json.loads((ROOT / ".splash-local-profile.json").read_text())
    probes = {}
    for enabled in ["0", "1"]:
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SPLASH_FLASH_")}
        environment.update(profile["environment"])
        environment.update({"SPLASH_FLASH_ALLROWS_FULL512_TARGET": "1",
                            "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV": "1",
                            "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV_COLUMNS": "2",
                            "SPLASH_FLASH_QSA_BULK_PREFILL": enabled,
                            "SPLASH_FLASH_QSA_BULK_PREFILL_SG8": enabled,
                            "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE": "0"})
        probes[enabled] = json.loads(subprocess.check_output([str(build / "memory-cpu"), "16384", "2048"], env=environment, text=True))
    delta = probes["1"]["trunk_row_workspace_planned_bytes"] - probes["0"]["trunk_row_workspace_planned_bytes"]
    checks["compiled_bulk_planning_delta_matches_frozen_223_5_mib"] = delta == 234356736 == manifest["qsa_bulk_workspace_extra_bytes"]
    result = {"schema": 1, "gpu_work": False, "model_loaded": False, "payload_bytes_read": 0,
              "files_checked": len(manifest["files"]), "base_manifest_sha256": manifest["bulk_qmv_base_manifest_sha256"],
              "source_mismatch": mismatch, "checks": checks, "compiled_planner_extra_bytes": delta,
              "compiled_cpu_planners": probes,
              "expected_target_numerical_derivative_sha256": "ce70bf75e8c89f97e5f6bc37a540ae2be93ad85fba5fce5598dac759b0739601",
              "expected_original_target_gpu_omitted_bytes": 67947724800,
              "runtime_sha256": {name: sha((build / name).read_bytes()) for name in ["splash-flash", "splash.metallib"]},
              "pass": all(checks.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"], "files_checked": len(manifest["files"]), "planner_extra_bytes": delta, "gpu_work": False}))
    if not result["pass"]:
        raise ValueError("Combined snapshot/source admission witness failed")


if __name__ == "__main__":
    main()
