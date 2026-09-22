#!/usr/bin/env python3
"""Seal qualified pointwise/FMA choices over the exact v2 batch teacher source."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import shlex
import subprocess
from worker_overlay import ROOT, PRIVATE, QUALIFIED_KERNEL_SHA256, sha, write, transform as pointwise_transform

FMA_PRIVATE = Path("dev/benchmarks/gdn_chunk_sep21")
FMA_KERNEL_SHA = "166381991f5444585e7db3634f41432322efdd66bf08a8c402d5fed49e58c699"
CHANGED_NAMES = {"FlashExpertDenseCache", "FlashForward", "FlashInt8ExpertStore", "FlashMoE",
                 "FlashMoEBlocked", "FlashWorker", "FlashGDNStaged", "FlashBatchPrefill"}


def fma_module():
    path = ROOT / FMA_PRIVATE / "worker_overlay.py"
    spec = importlib.util.spec_from_file_location("sealed_batch_fma_transform", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform(relative, text):
    return fma_module().transform(relative, pointwise_transform(relative, text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-batch-teacher-gathered-sep21-v2")
    parser.add_argument("--fma-base", type=Path, default=ROOT / "build/gdn-prefill-fma-sep21-worker-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-batch-pointwise-fma-sep21-v1")
    args = parser.parse_args()
    base, fma_base, output = args.base.resolve(), args.fma_base.resolve(), args.output.resolve()
    if ROOT / "build" not in output.parents or output in (base, fma_base):
        raise ValueError("Combined batch worker needs a distinct private build")
    parent_path, fma_parent_path = base / "overlay-manifest.json", fma_base / "overlay-manifest.json"
    parent, fma_parent = json.loads(parent_path.read_text()), json.loads(fma_parent_path.read_text())
    audit_path = base / "host-build-audit.json"
    audit = json.loads(audit_path.read_text())
    if not parent.get("batch_teacher_composed") or parent.get("batch_teacher_oracle_version") != 2 or not parent.get("batch_bulk_composed") or not parent.get("gathered_mpp_composed"):
        raise ValueError("The qualified v2 teacher + batch bulk + gathered source is required")
    if sha(parent_path.read_bytes()) != audit["manifest_sha256"] or sha((base / "splash-flash").read_bytes()) != audit["binary_sha256"]:
        raise ValueError("Qualified batch v2 audit drift")
    if sha((ROOT / FMA_PRIVATE / "worker_overlay.py").read_bytes()) != fma_parent["gdn_fma_transform_sha256"]:
        raise ValueError("Qualified FMA transform drift")
    fma_records = {record["path"]: record for record in fma_parent["files"]}
    for relative in (FMA_PRIVATE / "worker_bridge.hpp", FMA_PRIVATE / "worker_policy_cpu.cpp", FMA_PRIVATE / "scalar_fma.metal"):
        if sha((fma_base / "source" / relative).read_bytes()) != fma_records[str(relative)]["overlay_sha256"]:
            raise ValueError(f"Qualified FMA source drift: {relative}")
    if sha((fma_base / "source" / FMA_PRIVATE / "scalar_fma.metal").read_bytes()) != FMA_KERNEL_SHA:
        raise ValueError("FMA kernel differs from qualified source")
    if sha((ROOT / PRIVATE / "candidate.metal").read_bytes()) != QUALIFIED_KERNEL_SHA256:
        raise ValueError("Pointwise kernel differs from qualified source")
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-v2-batchteacher-bulk-gathered-pointwise-and-optional-prefill-fma-sep21-v1",
                     "batch_pointwise_fma_composed": True, "pointwise_composed": True, "gdn_fma_composed": True,
                     "batch_pointwise_fma_base_build": str(base), "batch_pointwise_fma_base_manifest_sha256": sha(parent_path.read_bytes()),
                     "batch_pointwise_fma_base_audit_sha256": sha(audit_path.read_bytes()),
                     "batch_pointwise_fma_qualified_fma_build": str(fma_base), "batch_pointwise_fma_fma_manifest_sha256": sha(fma_parent_path.read_bytes()),
                     "batch_pointwise_fma_prepare_sha256": sha(Path(__file__).read_bytes()),
                     "batch_pointwise_fma_pointwise_transform_sha256": sha((ROOT / PRIVATE / "worker_overlay.py").read_bytes()),
                     "batch_pointwise_fma_fma_transform_sha256": fma_parent["gdn_fma_transform_sha256"],
                     "pointwise_kernel_source_sha256": QUALIFIED_KERNEL_SHA256, "gdn_fma_kernel_source_sha256": FMA_KERNEL_SHA,
                     "independent_flags": ["SPLASH_FLASH_MOE_POINTWISE_SEP21", "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21"],
                     "both_flags0_original_execution": True, "additional_gpu_workspace_bytes": 0,
                     "pointwise_changes_numerical_derivative": False, "gdn_fma_changes_numerical_derivative": True,
                     "gdn_fma_scope": fma_parent["gdn_fma_scope"], "no_sg2_tail_added": True,
                     "source_model_fidelity_qualification": "inherited root-qualified batch teacher v2 plus independently qualified pointwise and FMA; combined actual model run remains pending",
                     "gpu_executed": False, "model_loaded": False, "payload_bytes_read": 0, "files": [],
                     "combined_link_inputs": [], "original_batch_object_inventory": []})
    changed = []
    for record in parent["files"]:
        relative = record["path"]
        original = (base / "source" / relative).read_bytes()
        if sha(original) != record["overlay_sha256"]:
            raise ValueError(f"Sealed v2 source drift: {relative}")
        data = transform(relative, original.decode()).encode()
        write(output / "source" / relative, data)
        if data != original:
            changed.append(relative)
        manifest["files"].append({**record, "combined_changed": data != original,
                                  "combined_base_sha256": sha(original), "overlay_sha256": sha(data)})
    expected_changed = {"runtime/flash/" + name + (".mm" if name in ("FlashWorker", "FlashInt8ExpertStore") else ".cpp") for name in CHANGED_NAMES}
    expected_changed.add("dev/benchmarks/prefill4k_attribution.mm")
    if set(changed) != expected_changed:
        raise ValueError(f"Expected only eight host module routes + attribution, got {changed}")
    extras = [(ROOT / PRIVATE / name, PRIVATE / name) for name in ("bridge.hpp", "candidate.metal", "policy_cpu.cpp")]
    extras += [(fma_base / "source" / FMA_PRIVATE / name, FMA_PRIVATE / name) for name in ("worker_bridge.hpp", "worker_policy_cpu.cpp", "scalar_fma.metal")]
    memory_source = ROOT / "dev/benchmarks/prefill4k_wide_memory.mm"
    if sha(memory_source.read_bytes()) != audit["linked_artifacts"]["memory-cpu"]["source_sha256"]:
        raise ValueError("Original CPU memory helper source drift")
    extras.append((memory_source, Path("dev/benchmarks/prefill4k_wide_memory.mm")))
    for source, relative in extras:
        data = source.read_bytes()
        write(output / "source" / relative, data)
        manifest["files"].append({"path": str(relative), "combined_support_source": str(source), "overlay_sha256": sha(data)})
    already = {record["path"] for record in manifest["files"]}
    for record in fma_parent["files"]:
        relative = record["path"]
        if relative in already or not relative.startswith("runtime/") or Path(relative).suffix not in (".h", ".hpp"):
            continue
        data = (fma_base / "source" / relative).read_bytes()
        if sha(data) != record["overlay_sha256"]:
            raise ValueError(f"Qualified transitive header drift: {relative}")
        write(output / "source" / relative, data)
        manifest["files"].append({"path": relative, "combined_support_source": str(fma_base / "source" / relative), "overlay_sha256": sha(data)})
    inputs = {"REUSED": [], "CORE": [], "AIRS": []}
    def freeze_input(source, relative, category, expected):
        data = source.read_bytes()
        if sha(data) != expected:
            raise ValueError(f"Authenticated original input drift: {source}")
        write(output / relative, data)
        inputs[category].append(relative.as_posix())
        manifest["combined_link_inputs"].append({"source_path": str(source), "private_path": relative.as_posix(), "category": category, "sha256": expected})
    # The v2 authenticated inventory is authoritative; never infer objects from
    # host-directory wildcards or inherit the older FMA worker's teacher objects.
    names = set()
    for record in audit["inputs"]:
        source = Path(record["output"])
        if sha(source.read_bytes()) != record["sha256"]:
            raise ValueError(f"Qualified v2 original object drift: {source}")
        manifest["original_batch_object_inventory"].append({"path": str(source), "sha256": record["sha256"]})
        if source.suffix != ".o":
            continue
        if not source.is_relative_to(base) or source.name in names:
            raise ValueError(f"Unauthenticated/duplicate v2 object: {source}")
        names.add(source.name)
        if source.stem in CHANGED_NAMES:
            continue
        category = "CORE" if source.parent.name == "core" else "REUSED"
        freeze_input(source, Path("reused/v2") / source.relative_to(base), category, record["sha256"])
    if not {"FlashMTP.o", "FlashBatchMTPForward.o", "MetalBackend.o"} <= names:
        raise ValueError("Correct original teacher owner/batch/core objects missing")
    # Authenticate the exact batch shader closure by reconstructing the v2
    # audited metallib, including its M128 dense-prefill replacement. The older
    # FMA worker's baseline AIR set predates that replacement and is unsuitable.
    batch_parent = Path(parent["batch_teacher_base_build"])
    build_log = batch_parent / "build.log"
    commands = [shlex.split(line.split(" -- ", 1)[-1]) for line in build_log.read_text().splitlines()
                if "xcrun -sdk macosx metallib " in line]
    commands = [command for command in commands if Path(command[command.index("-o") + 1]).name == "splash.metallib"]
    if len(commands) != 1:
        raise ValueError("Expected one explicit original batch metallib invocation")
    command = commands[0]
    original_airs = [Path(value) for value in command[4:command.index("-o")]]
    if len(original_airs) != 68 or len(set(original_airs)) != 68:
        raise ValueError("Expected exactly 68 original batch AIR inputs")
    if Path("build/flash-next/metal/shared/flash_dense_cache_prefill.air") in original_airs or Path("build/flash-next/metal/shared/flash_qsa_bulk.air") in original_airs:
        raise ValueError("Batch shader closure must exclude both original dense-prefill and QSA-bulk AIRs")
    control_airs = []
    for path in original_airs:
        source = path if path.is_absolute() else ROOT / path
        relative = Path("reused/original-batch") / source.relative_to(ROOT)
        freeze_input(source, relative, "AIRS", sha(source.read_bytes()))
        control_airs.append(output / relative)
    control_library = output / "original-control.metallib"
    subprocess.run(["xcrun", "-sdk", "macosx", "metallib", *map(str, control_airs), "-o", str(control_library)], cwd=ROOT, check=True)
    control_sha = sha(control_library.read_bytes())
    if control_sha != audit["metallib_sha256"]:
        raise ValueError("Original 68-AIR reconstruction differs from qualified v2 metallib bytes")
    manifest["original_batch_air_log_sha256"] = sha(build_log.read_bytes())
    manifest["original_batch_air_inventory"] = [str(path) for path in original_airs]
    manifest["reconstructed_control_metallib_sha256"] = control_sha
    pointwise_air_records = [record for record in fma_parent["gdn_fma_link_inputs"]
                             if record["category"] == "AIRS" and Path(record["private_path"]).name == "pointwise.air"]
    if len(pointwise_air_records) != 1:
        raise ValueError("Qualified pointwise AIR missing or duplicated")
    pointwise_air = pointwise_air_records[0]
    freeze_input(fma_base / pointwise_air["private_path"], Path("reused/qualified-fma/pointwise.air"), "AIRS", pointwise_air["sha256"])
    fma_air = fma_base / "gdn-prefill-fma.air"
    # The FMA parent manifest seals inherited AIRs, while its source witness
    # authenticates the final library. Bind the one newly compiled FMA AIR by
    # replaying that exact link order and requiring identical qualified bytes.
    fma_witness_path = fma_base / "source-witness.json"
    fma_witness = json.loads(fma_witness_path.read_text())
    if not fma_witness["source_integrity_pass"] or sha((fma_base / "splash.metallib").read_bytes()) != fma_witness["artifact_sha256"]["splash.metallib"]:
        raise ValueError("Qualified FMA artifact witness drift")
    fma_link_inputs = [fma_base / record["private_path"] for record in fma_parent["gdn_fma_link_inputs"] if record["category"] == "AIRS"]
    fma_control = output / "qualified-fma-control.metallib"
    subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(fma_air), *map(str, fma_link_inputs), "-o", str(fma_control)], cwd=ROOT, check=True)
    fma_control_sha = sha(fma_control.read_bytes())
    if fma_control_sha != fma_witness["artifact_sha256"]["splash.metallib"]:
        raise ValueError("Imported FMA AIR differs from the qualified FMA library artifact")
    manifest["qualified_fma_artifact_witness_sha256"] = sha(fma_witness_path.read_bytes())
    manifest["qualified_fma_reconstructed_metallib_sha256"] = fma_control_sha
    freeze_input(fma_air, Path("reused/qualified-fma/gdn-prefill-fma.air"), "AIRS", sha(fma_air.read_bytes()))
    linked_make = "\n".join(f"{key} := " + " ".join("$(BUILD)/" + path for path in paths) for key, paths in inputs.items()) + "\n"
    write(output / "link-inputs.mk", linked_make.encode())
    manifest["combined_link_make_sha256"] = sha(linked_make.encode())
    manifest["combined_changed_files"] = changed
    write(output / "overlay-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"prepared": str(output), "sources": len(manifest["files"]), "frozen_inputs": len(manifest["combined_link_inputs"]),
                      "changed_sources": changed, "gpu_work": False, "payload_bytes_read": 0}))


if __name__ == "__main__":
    main()
