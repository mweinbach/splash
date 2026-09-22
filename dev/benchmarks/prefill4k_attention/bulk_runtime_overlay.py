#!/usr/bin/env python3
"""Snapshot a certified private runtime and compose the exact bulk QSA overlay.

CPU-only metadata/source work. The source build and normal production files are
never modified. Every handed-off source file is hash checked before copying;
the I8 loader/store/worker and numerical derivative remain byte identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import bulk_overlay

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/prefill4k_attention")
EXTRA_FILES = ("bulk.hpp", "bulk.cpp", "coalesced.hpp", "coalesced.cpp", "bulk_attention_sg8.metal")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(source_build: Path, output: Path) -> dict:
    source_build, output = source_build.resolve(), output.resolve()
    if ROOT / "build" not in output.parents or output == source_build:
        raise ValueError("Choose a fresh, separate output beneath repository build")
    if output.exists():
        raise ValueError("Runtime overlay refuses an existing output directory")
    base_manifest_bytes = (source_build / "overlay-manifest.json").read_bytes()
    base_manifest = json.loads(base_manifest_bytes)
    records, bytes_by_path = [], {}
    for record in base_manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source-manifest path")
        data = (source_build / "source" / relative).read_bytes()
        if digest(data) != record["overlay_sha256"]:
            raise ValueError(f"Source handoff is stale: {relative}")
        bytes_by_path[relative] = data
        records.append(dict(record))
    relative_forward = Path("runtime/flash/FlashForward.cpp")
    base_forward = bytes_by_path[relative_forward]
    modified, edits = bulk_overlay.transform(base_forward.decode())
    if bulk_overlay.restore(modified, edits).encode() != base_forward:
        raise AssertionError("Bulk edit journal did not restore the certified base")
    bytes_by_path[relative_forward] = modified.encode()
    for record in records:
        if record["path"] == str(relative_forward):
            record["bulk_input_sha256"] = record["overlay_sha256"]
            record["overlay_sha256"] = digest(modified.encode())
            record["patched"] = True
    for name in EXTRA_FILES:
        relative = PRIVATE / name
        data = (ROOT / relative).read_bytes()
        bytes_by_path[relative] = data
        records.append({"path": str(relative), "new_private_file": True,
                        "patched": True, "overlay_sha256": digest(data)})
    manifest = dict(base_manifest)
    manifest.update({
        "bulk_source_build": str(source_build),
        "bulk_input_manifest_sha256": digest(base_manifest_bytes),
        "qsa_bulk_composed": True,
        "qsa_bulk_sg8_available": True,
        "qsa_bulk_source_generator_sha256": digest(Path(__file__).read_bytes()),
        "qsa_bulk_transform_sha256": digest((ROOT / PRIVATE / "bulk_overlay.py").read_bytes()),
        "qsa_bulk_workspace_extra_bytes": bulk_overlay.PLANNED_BYTES,
        "qsa_bulk_required_environment": "SPLASH_FLASH_QSA_BULK_PREFILL=1",
        "qsa_bulk_eligible_call": "main nonverification begin0 rows2048 only",
        "qsa_bulk_trained_coefficients_changed": False,
        "qsa_bulk_gpu_qualified": None,
        "payload_bytes_read": 0,
        "normal_sources_modified": False,
        "files": records,
    })
    # Stage and then rename, so consumers never observe a partial manifest/tree.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="private-bulk-runtime-", dir=output.parent) as temp:
        staged = Path(temp) / "output"
        for relative, data in bytes_by_path.items():
            destination = staged / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        (staged / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        witness = bulk_overlay.manifest(source_build / "source" / relative_forward,
                                        base_forward.decode(), modified, edits)
        (staged / "bulk-forward-manifest.json").write_text(json.dumps(witness, indent=2) + "\n")
        staged.rename(output)
    return {"prepared": str(output), "gpu_executed": False, "payload_bytes_read": 0,
            "source_files": len(records), "base_forward_restored_byte_exact": True,
            "trained_coefficients_and_derivative_unchanged": True,
            "extra_workspace_bytes": bulk_overlay.PLANNED_BYTES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-build", type=Path, default=ROOT / "build/prefill4k-allrows-full512")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-allrows-full512")
    args = parser.parse_args()
    print(json.dumps(generate(args.source_build, args.output)))


if __name__ == "__main__":
    main()
