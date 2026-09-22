#!/usr/bin/env python3
"""Replace an interim proof receipt without recompiling any host or shader."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/expert_r1_traversal_sep22")
AIR_SHA = "a0cd35e03daf13324d0308c8b4d8cee1d6d932989be6cbdc0429e6ec471d05c2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--kernel-build", type=Path, required=True)
    args = parser.parse_args()
    build, kernel = args.build.resolve(), args.kernel_build.resolve()
    if (build / "CPU_READY.json").exists() or (build / "root-registration.json").exists():
        raise ValueError("Proof refresh requires withdrawn admission and no Root registration")
    interim = json.loads((build / "CPU_INTERIM.json").read_text())
    for record in interim["sources"]:
        if sha(build / "source" / record["path"]) != record["sha256"]:
            raise ValueError("The interim frozen source drifted")
    for record in interim["reused53"] + interim["artifacts"]:
        if sha(build / record["path"]) != record["sha256"]:
            raise ValueError("The interim compiled artifact or proof drifted")
    final = json.loads((kernel / "CPU_READY.json").read_text())
    if (final.get("pass") is not True or final.get("GPU_work") is not False
            or final.get("original_AIR_sha256") != AIR_SHA
            or final.get("candidate_native_FP_tree_match") is not True
            or final.get("taps_shipping_FP_tree_match") is not True):
        raise ValueError("The final native proof must admit the unchanged AIR pair")
    changed = []
    for record in final["sources"]:
        source = kernel / record["path"]
        relative = Path(record["program_path"])
        if relative.is_absolute() or ".." in relative.parts or sha(source) != record["sha256"]:
            raise ValueError("Invalid final proof source")
        destination = build / "source" / relative
        if sha(destination) != record["sha256"]:
            if relative.parent != PRIVATE / "kernel" or relative.name not in ("audit.py", "build.py"):
                raise ValueError("A compiled shader/header input changed; metadata refresh is forbidden")
            shutil.copyfile(source, destination)
            changed.append(str(relative))
    for record in final["artifacts"]:
        source = kernel / record["path"]
        if sha(source) != record["sha256"]:
            raise ValueError("Final kernel proof artifact drift")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid proof artifact path")
        if relative.suffix in (".air", ".ll"):
            if sha(build / "kernel-proof" / relative) != record["sha256"]:
                raise ValueError("Shader AIR/IR changed; metadata refresh is forbidden")
        elif relative.name != "arithmetic-audit.json":
            raise ValueError("Unexpected proof-only artifact")
        shutil.copyfile(source, build / "kernel-proof" / relative)
    for name in ("candidate.air", "taps.air"):
        if sha(build / name) != sha(kernel / name):
            raise ValueError("The executable's shader pair differs from the final audited pair")
    shutil.copyfile(kernel / "CPU_READY.json", build / "kernel-CPU_READY.json")
    updater = build / "source" / PRIVATE / "reopen_proof.py"
    shutil.copyfile(Path(__file__), updater)
    sources = [{"path": str(path.relative_to(build / "source")), "sha256": sha(path)}
               for path in sorted((build / "source").rglob("*")) if path.is_file()]
    artifacts = [{"path": record["path"], "sha256": sha(build / record["path"])}
                 for record in interim["artifacts"]]
    receipt = dict(interim)
    parts = dict(receipt["identity_parts"])
    parts["kernel_receipt_sha256"] = sha(kernel / "CPU_READY.json")
    parts["proof_only_updater_sha256"] = sha(updater)
    identity = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt.update({"sources": sources, "artifacts": artifacts, "identity_parts": parts,
                    "source_identity_sha256": identity, "kernel_build": str(kernel),
                    "kernel_receipt_sha256": sha(kernel / "CPU_READY.json"),
                    "proof_refresh": {"shader_and_host_recompiles": 0,
                                      "shader_AIR_and_IR_bytes_unchanged": True,
                                      "compiled_host_and_library_bytes_unchanged": True,
                                      "changed_proof_program_paths": changed,
                                      "interim_receipt_sha256": sha(build / "CPU_INTERIM.json")}})
    (build / "CPU_READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"build": str(build), "CPU_READY_sha256": sha(build / "CPU_READY.json"),
                      "source_identity_sha256": identity, "recompiled_math": False, "GPU_work": False}))


if __name__ == "__main__":
    main()
