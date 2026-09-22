#!/usr/bin/env python3
"""Compose a PRIVATE no-original-target-GPU all-row Full512 candidate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def module(name: str):
    path = ROOT / "dev/benchmarks" / f"prefill4k_{name}.py"
    spec = importlib.util.spec_from_file_location(f"private_{name}", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load transform {path}")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qmv", action="store_true", help="Compose the separately identified gathered small-row reduction")
    parser.add_argument("--qmv-c2", action="store_true", help="Also compile the optional paired columns2 variant")
    args = parser.parse_args()
    args.qmv = args.qmv or args.qmv_c2
    default_build = "build/prefill4k-allrows-qmv-c2" if args.qmv_c2 else "build/prefill4k-allrows-qmv" if args.qmv else "build/prefill4k-allrows-full512"
    output = (args.output or ROOT / default_build).resolve()
    if ROOT / "build" not in output.parents:
        raise ValueError("Private output must remain beneath repository build")
    names = ["wide_overlay", "fullcache_overlay", "allrows_loader", "allrows_routes", "allrows_store", "allrows_worker"]
    if args.qmv:
        if output == ROOT / "build/prefill4k-allrows-full512":
            raise ValueError("Gathered QMV must not overwrite the MPP control build")
        names += ["allrows_qmv", "allrows_qmv_routes"]
        if args.qmv_c2:
            names += ["allrows_qmv_c2", "allrows_qmv_status"]
    transforms = [module(name).transform for name in names]
    relatives = [str(path.relative_to(ROOT)) for path in sorted((ROOT / "runtime/flash").glob("*")) if path.is_file()]
    relatives += [f"runtime/metal/kernels/shared/{name}.metal" for name in ["flash_gdn", "flash_gdn_fused", "flash_gdn_staged", "flash_int8_expert_store"]]
    relatives += [
        "dev/benchmarks/flash_int8_expert_store_metadata_cpu.mm",
        "dev/benchmarks/flash_int8_expert_store_oracle.mm",
        "dev/benchmarks/flash_expert_int8_bucket_reference.hpp",
    ]
    manifest = {
        "schema": 1, "route": "private-allrows-full512-target-original-gpu-omitted-v1",
        "normal_sources_modified": False, "arithmetic_change": True,
        "required_environment_flag": "SPLASH_FLASH_ALLROWS_FULL512_TARGET=1",
        "omitted_original_target_tensor_count": 432,
        "omitted_original_target_gpu_bytes": 67947724800,
        "remaining_original_gpu_bytes": 6370164736,
        "trained_mtp_source_changed": False,
        "gathered_qmv_composed": args.qmv,
        "gathered_qmv_c2_composed": args.qmv_c2,
        "transform_sha256": {name: hashlib.sha256((ROOT / "dev/benchmarks" / f"prefill4k_{name}.py").read_bytes()).hexdigest() for name in names},
        "files": [],
    }
    for relative in relatives:
        original = (ROOT / relative).read_bytes()
        text = original.decode()
        for transform in transforms:
            text = transform(relative, text)
        if args.qmv_c2 and relative == "runtime/flash/FlashWorker.mm":
            text = text.replace("      (void)gathered_i8_qmv::requested();", "      (void)gathered_i8_qmv::requestedColumns();\n      (void)gathered_i8_qmv::requested();")
        modified = text.encode()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.read_bytes() != modified:
            destination.write_bytes(modified)
        manifest["files"].append({"path": relative, "patched": original != modified,
            "original_sha256": hashlib.sha256(original).hexdigest(), "overlay_sha256": hashlib.sha256(modified).hexdigest()})
    if args.qmv:
        private_extra = module("allrows_qmv").extra_files()
        if args.qmv_c2:
            private_extra.update(module("allrows_qmv_c2").extra_files())
        for relative, text in private_extra.items():
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = text.encode()
            if not destination.exists() or destination.read_bytes() != data:
                destination.write_bytes(data)
            manifest["files"].append({"path": relative, "new_private_file": True,
                "patched": True, "overlay_sha256": hashlib.sha256(data).hexdigest()})
    path = output / "overlay-manifest.json"
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    print(json.dumps({"prepared": str(output), "gpu_work": False, "files": len(manifest["files"]), "patched": sum(item["patched"] for item in manifest["files"])}))


if __name__ == "__main__":
    main()
