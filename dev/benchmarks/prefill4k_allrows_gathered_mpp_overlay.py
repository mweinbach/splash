#!/usr/bin/env python3
"""Independent direct-MPP source build, preserving rejected C2 artifacts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from prefill4k_allrows_overlay import ROOT, module


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-allrows-full512")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-allrows-gathered-mpp")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if ROOT / "build" not in output.parents or output == base:
        raise ValueError("Direct gathered MPP requires a distinct private output")
    base_path = base / "overlay-manifest.json"
    parent = json.loads(base_path.read_text())
    if parent.get("gathered_qmv_composed") or parent.get("gathered_qmv_c2_composed"):
        raise ValueError("Use the original all-row MPP control")
    names = ["allrows_gathered_mpp", "allrows_gathered_mpp_routes"]
    transforms = [module(name).transform for name in names]
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-allrows-full512-direct-gathered-mpp-v1",
                     "gathered_mpp_composed": True, "gathered_mpp_base_build": str(base),
                     "gathered_mpp_base_manifest_sha256": digest(base_path.read_bytes()),
                     "gathered_mpp_transform_order": names, "gpu_executed": False, "payload_bytes_read": 0})
    manifest["transform_sha256"].update({name: digest((ROOT / "dev/benchmarks" / f"prefill4k_{name}.py").read_bytes()) for name in names})
    manifest["files"] = []
    for record in parent["files"]:
        original = (base / "source" / record["path"]).read_bytes()
        if digest(original) != record["overlay_sha256"]:
            raise ValueError(f"Original control source drift: {record['path']}")
        text = original.decode()
        for transform in transforms:
            text = transform(record["path"], text)
        data = text.encode()
        path = output / "source" / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
        manifest["files"].append({**record, "base_overlay_sha256": digest(original),
                                  "combined_changed": data != original, "overlay_sha256": digest(data)})
    for relative, text in module("allrows_gathered_mpp").extra_files().items():
        data = text.encode()
        path = output / "source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
        manifest["files"].append({"path": relative, "new_private_file": True,
                                  "patched": True, "overlay_sha256": digest(data)})
    path = output / "overlay-manifest.json"
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    print(json.dumps({"prepared": str(output), "files": len(manifest["files"]),
                      "base_source_files_checked": len(parent["files"]), "gpu_executed": False, "payload_bytes_read": 0}))


if __name__ == "__main__":
    main()
