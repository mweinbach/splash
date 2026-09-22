#!/usr/bin/env python3
"""Compose C2 decoding onto the sealed exact-bulk Full512 source snapshot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from prefill4k_allrows_overlay import ROOT, module


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-allrows-full512")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-allrows-qmv-c2")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if ROOT / "build" not in output.parents or output == base:
        raise ValueError("Combined output must be a distinct private build path")
    base_path = base / "overlay-manifest.json"
    original_manifest = json.loads(base_path.read_text())
    if not original_manifest.get("qsa_bulk_composed") or not original_manifest.get("qsa_bulk_sg8_available"):
        raise ValueError("Use the sealed exact bulk SG8 source snapshot")
    names = ["allrows_qmv", "allrows_qmv_routes", "allrows_qmv_c2", "allrows_qmv_status"]
    transforms = [module(name).transform for name in names]
    manifest = copy.deepcopy(original_manifest)
    manifest.update({"route": "private-allrows-full512-exact-bulk-sg8-and-c2-decode-v1",
                     "bulk_qmv_base_build": str(base), "bulk_qmv_base_manifest_sha256": sha(base_path.read_bytes()),
                     "bulk_qmv_transform_order": names, "gathered_qmv_composed": True,
                     "gathered_qmv_c2_composed": True, "payload_bytes_read": 0,
                     "gpu_executed": False})
    manifest["transform_sha256"].update({name: sha((ROOT / "dev/benchmarks" / f"prefill4k_{name}.py").read_bytes()) for name in names})
    manifest["files"] = []
    for record in original_manifest["files"]:
        source = base / "source" / record["path"]
        original = source.read_bytes()
        if sha(original) != record["overlay_sha256"]:
            raise ValueError(f"Sealed source drift: {record['path']}")
        text = original.decode()
        for transform in transforms:
            text = transform(record["path"], text)
        if record["path"] == "runtime/flash/FlashWorker.mm":
            text = text.replace("      (void)gathered_i8_qmv::requested();", "      (void)gathered_i8_qmv::requestedColumns();\n      (void)gathered_i8_qmv::requested();")
        data = text.encode()
        destination = output / "source" / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.read_bytes() != data:
            destination.write_bytes(data)
        manifest["files"].append({**record, "base_overlay_sha256": sha(original),
                                  "combined_changed": data != original, "overlay_sha256": sha(data)})
    extra = module("allrows_qmv").extra_files()
    extra.update(module("allrows_qmv_c2").extra_files())
    for relative, text in extra.items():
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode()
        if not destination.exists() or destination.read_bytes() != data:
            destination.write_bytes(data)
        manifest["files"].append({"path": relative, "new_private_file": True,
                                  "patched": True, "overlay_sha256": sha(data)})
    destination = output / "overlay-manifest.json"
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    if not destination.exists() or destination.read_bytes() != data:
        destination.write_bytes(data)
    print(json.dumps({"prepared": str(output), "files": len(manifest["files"]),
                      "base_source_files_checked": len(original_manifest["files"]),
                      "qsa_bulk_workspace_extra_bytes": manifest["qsa_bulk_workspace_extra_bytes"],
                      "payload_bytes_read": 0, "gpu_executed": False}))


if __name__ == "__main__":
    main()
