#!/usr/bin/env python3
"""CPU-only source, sealed-artifact and admission checks for a private bulk build."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import bulk_overlay


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(build: Path) -> dict:
    build = build.resolve()
    manifest_bytes = (build / "overlay-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    base = Path(manifest["bulk_source_build"])
    base_manifest_bytes = (base / "overlay-manifest.json").read_bytes()
    assert sha(base_manifest_bytes) == manifest["bulk_input_manifest_sha256"]
    base_manifest = json.loads(base_manifest_bytes)
    before = {r["path"]: r for r in base_manifest["files"]}
    changed, added = [], []
    for record in manifest["files"]:
        relative = record["path"]
        data = (build / "source" / relative).read_bytes()
        assert sha(data) == record["overlay_sha256"], f"Private source drift: {relative}"
        if relative not in before:
            added.append(relative)
            continue
        original = (base / "source" / relative).read_bytes()
        assert sha(original) == before[relative]["overlay_sha256"], f"Base source drift: {relative}"
        if data != original:
            changed.append(relative)
    assert changed == ["runtime/flash/FlashForward.cpp"]
    assert len(added) == 5
    base_forward = (base / "source/runtime/flash/FlashForward.cpp").read_text()
    expected, edits = bulk_overlay.transform(base_forward)
    assert expected == (build / "source/runtime/flash/FlashForward.cpp").read_text()
    assert bulk_overlay.restore(expected, edits) == base_forward
    # These fields describe the coefficient inventory and numerical derivative;
    # composing byte-exact QSA does not modify any of them.
    preserved = ["omitted_original_target_tensor_count", "omitted_original_target_gpu_bytes",
                 "remaining_original_gpu_bytes", "trained_mtp_source_changed"]
    assert all(manifest.get(key) == base_manifest.get(key) for key in preserved)
    common = os.environ.copy()
    common.update({"SPLASH_FLASH_QSA_F32": "1", "SPLASH_FLASH_QSA_MPP": "1",
                   "SPLASH_FLASH_QSA_ROW_TILES": "1"})
    plans = {}
    maximum = re.search(r"kFlashGDNMaximumRows\s*=\s*(\d+)",
                        (build / "source/runtime/flash/FlashGDN.hpp").read_text())
    assert maximum
    maximum_rows = int(maximum.group(1))
    row_cases = [rows for rows in (128, 2048, 8192) if rows <= maximum_rows]
    for rows in row_cases:
        for mode, bulk, sg8 in (("off", "0", "0"), ("sg4", "1", "0"), ("sg8", "1", "1")):
            env = dict(common, SPLASH_FLASH_QSA_BULK_PREFILL=bulk,
                       SPLASH_FLASH_QSA_BULK_PREFILL_SG8=sg8)
            result = subprocess.run([str(build / "memory-cpu"), "262144", str(rows)], env=env,
                                    check=True, capture_output=True, text=True)
            value = json.loads(result.stdout)
            assert value["gpu_work"] is False and value["model_loaded"] is False
            plans[f"r{rows}_{mode}"] = value
        baseline = plans[f"r{rows}_off"]
        for mode in ("sg4", "sg8"):
            candidate = plans[f"r{rows}_{mode}"]
            expected_bytes = bulk_overlay.PLANNED_BYTES if rows >= 2048 else 0
            assert candidate["trunk_row_workspace_planned_bytes"] - baseline["trunk_row_workspace_planned_bytes"] == expected_bytes
            for key, value in baseline.items():
                if key.endswith("bytes") and not key.startswith(("host_", "trunk_row_")):
                    assert candidate[key] == value, f"Unrelated plan changed: {key}"
    invalid = dict(common, SPLASH_FLASH_QSA_BULK_PREFILL="0",
                   SPLASH_FLASH_QSA_BULK_PREFILL_SG8="1")
    result = subprocess.run([str(build / "memory-cpu"), "262144", "2048"], env=invalid,
                            capture_output=True, text=True)
    assert result.returncode != 0 and "requires SPLASH_FLASH_QSA_BULK_PREFILL=1" in result.stderr
    unsupported_rows = []
    if maximum_rows < 8192:
        for bulk, sg8 in (("0", "0"), ("1", "0"), ("1", "1")):
            env = dict(common, SPLASH_FLASH_QSA_BULK_PREFILL=bulk,
                       SPLASH_FLASH_QSA_BULK_PREFILL_SG8=sg8)
            result = subprocess.run([str(build / "memory-cpu"), "262144", "8192"], env=env,
                                    capture_output=True, text=True)
            assert result.returncode != 0
            unsupported_rows.append({"rows": 8192, "bulk": bulk, "sg8": sg8,
                                     "rejected": True, "reason": result.stderr.strip()})
    artifacts = {name: sha((build / name).read_bytes())
                 for name in ("splash-flash", "splash.metallib", "prefill4k-attribution", "memory-cpu")}
    return {"pass": True, "gpu_executed": False, "models_loaded": False,
            "payload_bytes_read": 0, "build": str(build), "base": str(base),
            "overlay_manifest_sha256": sha(manifest_bytes), "source_files_checked": len(manifest["files"]),
            "existing_source_changes": changed, "new_private_files": added,
            "base_forward_restored_byte_exact": True,
            "coefficient_loader_store_worker_and_derivative_inputs_byte_identical": True,
            "allocation_plan_extra_bytes": bulk_overlay.PLANNED_BYTES,
            "allocation_plan_extra_mib": bulk_overlay.PLANNED_BYTES / (1 << 20),
            "admission_modes_checked": sorted(plans), "invalid_sg8_without_bulk_rejected": True,
            "original_maximum_rows": maximum_rows, "unsupported_rows_rejected": unsupported_rows,
            "artifact_sha256": artifacts, "whole_model_qualification": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=bulk_overlay.ROOT / "build/prefill4k-qsa-bulk-allrows-full512")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.build)
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("Choose a fresh witness output")
        args.output.write_text(data)
    print(data, end="")


if __name__ == "__main__":
    main()
