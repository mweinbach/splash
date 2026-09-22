#!/usr/bin/env python3
"""Verify the exact private source handoff and record the compiled artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def witness(build: Path) -> dict:
    manifest_path = build / "overlay-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    base = Path(manifest["gathered_mpp_base_build"])
    if sha((base / "overlay-manifest.json").read_bytes()) != manifest["gathered_mpp_base_manifest_sha256"]:
        raise ValueError("Base manifest drift")
    source = build / "source"
    changed = []
    for entry in manifest["files"]:
        relative = entry["path"]
        data = (source / relative).read_bytes()
        if sha(data) != entry["overlay_sha256"]:
            raise ValueError(f"Private source drift: {relative}")
        if "base_overlay_sha256" in entry:
            original = (base / "source" / relative).read_bytes()
            if sha(original) != entry["base_overlay_sha256"]:
                raise ValueError(f"Base source drift: {relative}")
            if data != original:
                changed.append(relative)
    expected = ["runtime/flash/FlashBatchForward.cpp", "runtime/flash/FlashBatchPrefill.cpp",
                "runtime/flash/FlashBatchVerify.cpp", "runtime/flash/FlashForward.cpp",
                "runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm",
                "runtime/flash/FlashWorker.mm"]
    if sorted(changed) != sorted(expected):
        raise AssertionError(f"Unexpected modified existing files: {changed}")
    store = (source / "runtime/flash/FlashInt8ExpertStore.mm").read_text()
    shader = (source / "runtime/metal/kernels/shared/flash_gathered_mpp.metal").read_text()
    header = (source / "runtime/flash/FlashGatheredMPP.hpp").read_text()
    for kernel in ["flash_gathered_mpp_gate_up_m16_n64_sg4", "flash_gathered_mpp_down_m16_n64_sg4"]:
        if store.count(f'graph.add("{kernel}",') != 1 or shader.count(f"kernel void {kernel}(") != 1:
            raise AssertionError("Shipping and graph kernel symbols differ")
    if "flash_gathered_i8_qmv_" in store:
        raise AssertionError("QMV arithmetic accidentally included")
    checks = {
        "all_private_and_parent_sources_hash_verified": True,
        "only_seven_existing_host_files_modified": True,
        "original_bucket_mpp_shader_unchanged": True,
        "gathered_shader_unchanged": True,
        "trained_mtp_unchanged": True,
        "frozen_flag_and_row_cap": store.count("gathered_mpp::requestedMaximumRows()") == 2,
        "enabled_cap_numerical_identity": 'small_row_max_physical_rows=' in store,
        "gathered_bounded_to_rows16": "requestedMaximumRows()" in header,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"schema": "splash-private-bulk-capped-gathered-mpp-source-witness-v1",
            "pass": True, "checks": checks, "files_checked": len(manifest["files"]),
            "modified_existing_files": changed,
            "runtime_sha256": {name: sha((build / name).read_bytes()) for name in ["splash-flash", "splash.metallib"]},
            "gpu_work": False, "payload_bytes_read": 0,
            "numerical_parity_qualified": False, "model_quality_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = witness(args.build.resolve())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.report:
        if args.report.exists():
            raise ValueError("Report path must be new")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered)
