#!/usr/bin/env python3
"""Clone the root-built v2 closure and repair its CPU-only startup dependency.

The sole source edit removes an unreachable batch-ILP required-one condition
and rejects alternative BF16 small-row cache selection before Metal. All model
math sources, F32 selections, kernels, and their numerical identity stay intact.
Only Worker is recompiled; all other root-built objects/library are hash checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def copy_checked(source: Path, destination: Path, expected: str | None = None) -> str:
    data = source.read_bytes()
    value = sha(data)
    if expected is not None and value != expected:
        raise ValueError(f"Root-built source/artifact changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    if source.stat().st_mode & 0o111:
        destination.chmod(source.stat().st_mode & 0o777)
    return value


def repair(base: Path, output: Path) -> dict:
    if output.exists() or ROOT / "build" not in output.resolve().parents:
        raise ValueError("Require a fresh private v3 snapshot")
    parent_data = (base / "overlay-manifest.json").read_bytes()
    parent = json.loads(parent_data)
    output.mkdir(parents=True)
    records = []
    worker_rel = "runtime/flash/FlashWorker.mm"
    for e in parent["files"]:
        rel = e["path"]
        data = (base / "source" / rel).read_bytes()
        if sha(data) != e["overlay_sha256"]:
            raise ValueError(f"Root-built source changed: {rel}")
        changed = data
        if rel == worker_rel:
            text = data.decode()
            token = '"SPLASH_FLASH_GDN_BATCH_ILP", '
            if text.count(token) != 1:
                raise ValueError("Expected exactly one unused batch-ILP required-one token")
            text = text.replace(token, "", 1)
            anchor = '      if (environmentSwitch("SPLASH_FLASH_ALLROWS_FULL512_TARGET") ||'
            if text.count(anchor) != 1:
                raise ValueError("Missing hybrid pre-backend alternative policy guard")
            text = text.replace(anchor, '''      if (environmentSwitch("SPLASH_FLASH_DENSE_SMALL_ROWS"))
        throw std::invalid_argument("private fixed-R4 hybrid requires DENSE_SMALL_ROWS=0 for original selective F32 coefficients");
''' + anchor, 1)
            changed = text.encode()
        destination = output / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(changed)
        records.append({**e, "root_built_v2_source_sha256": sha(data),
                        "startup_guard_repaired": changed != data,
                        "overlay_sha256": sha(changed)})
    frozen = []
    for e in parent["frozen_link_inputs"]:
        value = copy_checked(base / e["private_path"], output / e["private_path"], e["sha256"])
        frozen.append({**e, "root_built_v2_path": str(base / e["private_path"]), "sha256": value})
    host_objects = []
    for path in sorted((base / "host").glob("*.o")):
        if path.name == "FlashWorker.o":
            continue
        rel = Path("host") / path.name
        value = copy_checked(path, output / rel)
        host_objects.append(dict(path=str(rel), root_built_v2_path=str(path), sha256=value))
    reused_binaries = []
    for name in ("splash.metallib", "prefill4k-attribution", "policy-cpu"):
        value = copy_checked(base / name, output / name)
        reused_binaries.append(dict(path=name, root_built_v2_path=str(base / name), sha256=value))
    copy_checked(base / "link-inputs.mk", output / "link-inputs.mk")
    profile = json.loads((base / "environment.json").read_text())
    profile.update({"SPLASH_FLASH_GDN_BATCH_ILP": "0", "SPLASH_FLASH_DENSE_SMALL_ROWS": "0",
                    "SPLASH_FLASH_ALLROWS_FULL512_TARGET": "0", "SPLASH_FLASH_ALLROWS_GATHERED_MPP": "0",
                    "SPLASH_FLASH_MTP_ADAPTIVE": "0"})
    (output / "environment.json").write_text(json.dumps(profile, indent=2) + "\n")
    witness = json.loads((base / "cpu-source-witness.json").read_text())
    witness["cpu_startup_dependency_repair"] = dict(only_source_changed=worker_rel,
                                                  GDN_BATCH_ILP=0, alternative_small_bf16_cache_rejected_before_backend=True,
                                                  numerical_model_policy_changed=False,
                                                  target_numerical_derivative_sha256=parent["target_numerical_derivative_sha256"])
    (output / "cpu-source-witness.json").write_text(json.dumps(witness, indent=2) + "\n")
    manifest = {**parent, "route": parent["route"] + "-startup-dependency-repaired-v3",
                "root_built_v2_parent": str(base), "root_built_v2_parent_manifest_sha256": sha(parent_data),
                "fixed_profile": profile, "files": records, "frozen_link_inputs": frozen,
                "hash_matched_reused_host_objects": host_objects, "hash_matched_reused_binaries": reused_binaries,
                "startup_profile_correction": dict(disabled_flag="SPLASH_FLASH_GDN_BATCH_ILP",
                                                   reason="batch-only parser requires disabled BATCH_PREFILL",
                                                   only_recompiled_source=worker_rel,
                                                   numerical_kernel_policy_changed=False)}
    (output / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(prepared=str(output), source_files=len(records), changed_source_files=1,
                reused_host_objects=len(host_objects), reused_binaries=len(reused_binaries),
                target_numerical_derivative_sha256=parent["target_numerical_derivative_sha256"],
                model_payload_bytes_read=0, gpu_execution=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/hybrid-q4-i8-fixed-r4-sep21-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "build/hybrid-q4-i8-fixed-r4-sep21-v3")
    args = parser.parse_args()
    print(json.dumps(repair(args.base.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
