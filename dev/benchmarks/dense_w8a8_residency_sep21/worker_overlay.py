#!/usr/bin/env python3
"""Freeze persistent-lease pruning without changing backing, math or GPU code."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/dense_w8a8_residency_sep21")
DEFAULT_BASE = ROOT / "build/dense-w8a8-hybrid-sg2-tail-fma-sep21-worker-v2"
DEFAULT_BUILD = ROOT / "build/dense-w8a8-residency-prune-sep21-worker-v1"
FLAG = "SPLASH_FLASH_DENSE_W8A8_RESIDENCY_PRUNE_SEP21"
DENSE_FLAG = "SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21"
REPLACED_NAMES = {"FlashForward", "FlashWorker"}
EXPECTED_CHANGED = {"runtime/flash/FlashForward.cpp", "runtime/flash/FlashWorker.mm",
                    "dev/benchmarks/prefill4k_attribution.mm"}
PRIVATE_NAMES = ("worker_bridge.hpp", "worker_policy_cpu.cpp")
TOOL_NAMES = ("worker_overlay.py", "worker_transform.py", "worker_witness.py", "worker.mk", "README.md")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def transform(relative, text, path=None):
    return module(path or Path(__file__).with_name("worker_transform.py"),
                  "frozen_dense_residency_transform").transform(relative, text)


def parent_helper(base, manifest, override=None):
    record = next(r for r in manifest["dense_w8a8_tools"] if Path(r["private_path"]).name == "worker_overlay.py")
    path = override or base / record["private_path"]
    if sha(path.read_bytes()) != record["sha256"]:
        raise ValueError("Archived parent dense generator changed")
    return module(path, "frozen_parent_dense_overlay"), record


def census_cases(hybrid):
    return {
        "scope": "persistent residency lease only; all backing and transient bindings retained",
        "hybrid_strict_union": hybrid,
        "coefficient_i8_owner_count": 168,
        "coefficient_i8_owner_bytes": 1890975744,
        "selected_bf16_source_owner_count": 84,
        "selected_bf16_source_owner_bytes": 3774873600,
        "selected_bf16_logical_tensor_bytes": 3774873600,
        "maximum_rows_ge2048_hybrid_cases": [
            {"prune": 0, "dense": 0, "owners": 916, "bytes": 204143722496},
            {"prune": 0, "dense": 1, "owners": 916, "bytes": 204143722496},
            {"prune": 1, "dense": 0, "owners": 748, "bytes": 202252746752},
            {"prune": 1, "dense": 1, "owners": 832, "bytes": 200368848896},
        ] if hybrid else [],
        "below_2048_cache_scope": "inherited; no derived dense W8A8 cache",
        "math_changed": False, "numerical_identity_changed": False,
        "backing_or_workspace_plan_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-make", type=Path)
    parser.add_argument("--contracts", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_BUILD)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if output == base or ROOT / "build" not in output.parents or (output / "overlay-manifest.json").exists():
        raise ValueError("Choose a fresh private residency overlay under build/")
    parent_path = base / "overlay-manifest.json"
    parent = json.loads(parent_path.read_text())
    if parent.get("dense_w8a8_composed") is not True:
        raise ValueError("Residency overlay requires the sealed dense W8A8 parent")
    helper, helper_record = parent_helper(base, parent)
    base_make = args.base_make.resolve() if args.base_make else base / "machinery/worker.mk"
    make_record = next(r for r in parent["dense_w8a8_tools"] if Path(r["private_path"]).name == "worker.mk")
    if sha(base_make.read_bytes()) != make_record["sha256"]:
        raise ValueError("Use the exact archived parent worker Make rules")
    closure = helper.effective_closure(base, base_make)
    seals, owned_inputs = helper.parent_input_seals(base, parent, closure)
    prior_witness = helper.parent_artifact_witness(base)
    source_records = {r["path"]: r for r in parent["files"]}
    if len(source_records) != len(parent["files"]):
        raise ValueError("Parent frozen sources have duplicate paths")
    dependencies = helper.frozen_dependency_closure(base, closure["objects"], source_records)
    archive_seals = {r["private_path"]: r["sha256"] for r in parent["dense_w8a8_tools"]}
    archive_seals.update({r["private_path"]: r["sha256"] for r in parent["dense_w8a8_qualified_inputs"]})
    for evidence in parent["dense_w8a8_component_certificates"]:
        archive_seals["qualification/" + Path(evidence["path"]).name] = evidence["report_sha256"]
    for relative, expected in archive_seals.items():
        if sha((base / relative).read_bytes()) != expected:
            raise ValueError(f"Authenticated inherited source/contract archive drift: {relative}")
    backend_source = next(r for r in parent["dense_w8a8_qualified_inputs"] if r["path"] == "runtime/metal/MetalBackend.mm")
    backend_object = next(r for r in parent["dense_w8a8_link_inputs"] if r["category"] == "CORE" and Path(r["private_path"]).name == "MetalBackend.o")
    transformed = []
    changed = []
    for relative, record in source_records.items():
        original = (base / "source" / helper.safe_relative(relative)).read_bytes()
        if sha(original) != record["overlay_sha256"]:
            raise ValueError(f"Frozen parent source drift: {relative}")
        data = transform(relative, original.decode()).encode()
        transformed.append((relative, record, original, data))
        if data != original:
            changed.append(relative)
    if set(changed) != EXPECTED_CHANGED:
        raise ValueError(f"Expected persistent selection plus prebackend/status changes only: {changed}")
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-dense-w8a8-persistent-residency-prune-sep21-v1",
                     "dense_w8a8_residency_composed": True,
                     "dense_w8a8_residency_base_build": str(base),
                     "dense_w8a8_residency_base_manifest_sha256": sha(parent_path.read_bytes()),
                     "dense_w8a8_residency_base_make_path": str(base_make),
                     "dense_w8a8_residency_base_make_sha256": sha(base_make.read_bytes()),
                     "dense_w8a8_residency_required_environment": f"{FLAG}=0|1",
                     "dense_w8a8_residency_default": 0,
                     "dense_w8a8_residency_parent_input_seals": seals,
                     "dense_w8a8_residency_owned_inputs_without_prior_individual_seal": owned_inputs,
                     "dense_w8a8_residency_parent_artifact_witness": prior_witness,
                     "dense_w8a8_residency_census": census_cases(parent.get("dense_w8a8_hybrid_parent") is True),
                     "dense_w8a8_residency_resource_plan_sha256": sha(json.dumps(parent["dense_w8a8_resource_plan"], sort_keys=True).encode()),
                     "dense_w8a8_residency_math_changed": False,
                     "dense_w8a8_residency_numerical_identity_changed": False,
                     "dense_w8a8_residency_backing_and_planners_changed": False,
                     "dense_w8a8_residency_gpu_code_changed": False,
                     "dense_w8a8_residency_metallib_sha256": prior_witness["artifacts"]["splash.metallib"],
                     "dense_w8a8_residency_parent_helper_sha256": helper_record["sha256"],
                     "dense_w8a8_residency_archive_parent_seals": archive_seals,
                     "dense_w8a8_residency_contract_source_pins": {
                         "backend_source_sha256": backend_source["sha256"],
                         "backend_source_private_path": "inherited-archives/" + backend_source["private_path"],
                         "backend_object_sha256": backend_object["sha256"],
                         "operand_store_source_path": "runtime/flash/FlashOperandStore.mm",
                         "operand_store_source_sha256": source_records["runtime/flash/FlashOperandStore.mm"]["overlay_sha256"],
                     },
                     "dense_w8a8_residency_changed_files": changed,
                     "dense_w8a8_residency_effective_parent_objects": [str(p) for p in closure["objects"]],
                     "dense_w8a8_residency_effective_parent_airs": [str(p) for p in closure["airs"]],
                     "dense_w8a8_residency_parent_dependencies": dependencies,
                     "dense_w8a8_residency_link_inputs": [], "dense_w8a8_residency_tools": [],
                     "dense_w8a8_residency_inherited_archives": [], "dense_w8a8_residency_contracts": [],
                     "files": [], "gpu_executed": False, "payload_bytes_read": 0})
    for relative, record, original, data in transformed:
        write(output / "source" / relative, data)
        manifest["files"].append({**record, "dense_w8a8_residency_base_overlay_sha256": sha(original),
                                  "dense_w8a8_residency_changed": data != original, "overlay_sha256": sha(data)})
    for name in PRIVATE_NAMES:
        relative = PRIVATE / name
        data = (ROOT / relative).read_bytes()
        write(output / "source" / relative, data)
        manifest["files"].append({"path": relative.as_posix(), "new_dense_w8a8_residency_file": True,
                                  "repository_sha256": sha(data), "overlay_sha256": sha(data)})
    inputs = {"REUSED": [], "CORE": [], "AIRS": []}
    def freeze(path, category):
        if base not in path.parents:
            raise ValueError(f"Compiler input escapes frozen parent: {path}")
        relative = Path("reused/base") / path.relative_to(base)
        data = path.read_bytes()
        write(output / relative, data)
        inputs[category].append(relative.as_posix())
        manifest["dense_w8a8_residency_link_inputs"].append({"source_path": str(path), "private_path": relative.as_posix(),
                                                           "category": category, "sha256": sha(data)})
    for path in closure["objects"]:
        if path.stem not in REPLACED_NAMES:
            freeze(path, "CORE" if path in closure["core"] else "REUSED")
    for path in closure["airs"]:
        freeze(path, "AIRS")
    for record in dependencies:
        if record["dependency_metadata_present"]:
            path = Path(record["source_path"])
            relative = Path("residency-qualification/dependencies") / path.relative_to(base)
            write(output / relative, path.read_bytes())
            record["private_path"] = relative.as_posix()
    for directory in ("qualification", "machinery"):
        for path in sorted((base / directory).rglob("*")):
            if not path.is_file():
                continue
            relative = Path("inherited-archives") / path.relative_to(base)
            data = path.read_bytes()
            write(output / relative, data)
            manifest["dense_w8a8_residency_inherited_archives"].append({"source_path": str(path), "private_path": relative.as_posix(), "sha256": sha(data)})
    for path in args.contracts or [ROOT / PRIVATE / "contract.md"]:
        path = path.resolve()
        if path.suffix not in (".json", ".md", ".txt"):
            raise ValueError("Contract evidence must be source/URL audit metadata")
        data = path.read_bytes()
        relative = Path("residency-contracts") / path.name
        if (output / relative).exists():
            raise ValueError("Contract evidence basenames must be unique")
        write(output / relative, data)
        manifest["dense_w8a8_residency_contracts"].append({"source_path": str(path), "private_path": relative.as_posix(), "sha256": sha(data)})
    make = "\n".join(f"{key} := " + " ".join("$(BUILD)/" + p for p in paths) for key, paths in inputs.items()) + "\n"
    write(output / "link-inputs.mk", make.encode())
    manifest["dense_w8a8_residency_link_make_sha256"] = sha(make.encode())
    for name in TOOL_NAMES:
        path = ROOT / PRIVATE / name
        relative = Path("machinery") / name
        data = path.read_bytes()
        write(output / relative, data)
        manifest["dense_w8a8_residency_tools"].append({"source_path": str(path), "private_path": relative.as_posix(), "sha256": sha(data)})
    write(output / "machinery/parent_dense_worker.py", (base / helper_record["private_path"]).read_bytes())
    for path, relative in (
        (parent_path, "residency-qualification/base-overlay-manifest.json"),
        (base_make, "residency-qualification/base-worker.mk"),
        (base / "link-inputs.mk", "residency-qualification/base-link-inputs.mk"),
        (Path(prior_witness["source_path"]), "residency-qualification/base-cpu-witness-v1.json"),
    ):
        write(output / relative, path.read_bytes())
    library = (base / "splash.metallib").read_bytes()
    write(output / "reused/base/splash.metallib", library)
    write(output / "splash.metallib", library)
    for source, target in (("policy-cpu", "inherited-dense-policy-cpu"), ("cache-policy-cpu", "inherited-cache-policy-cpu")):
        path = base / source
        data = path.read_bytes()
        write(output / target, data)
        (output / target).chmod(path.stat().st_mode & 0o777)
        manifest["dense_w8a8_residency_inherited_archives"].append({"source_path": str(path), "private_path": target, "sha256": sha(data)})
    for name in ("splash-flash.config", "splash.metallib.config"):
        data = (base / name).read_bytes().rstrip(b"\n") if (base / name).exists() else parent["route"].encode()
        write(output / name, data + b"-persistent-residency-prune-sep21-v1\n")
    write(output / "overlay-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"prepared": str(output), "frozen_sources": len(manifest["files"]), "changed_sources": changed,
                      "effective_parent_objects": len(closure["objects"]), "effective_parent_airs": len(closure["airs"]),
                      "frozen_inputs": len(manifest["dense_w8a8_residency_link_inputs"]),
                      "metallib_byte_identical": True, "backing_math_and_plans_unchanged": True,
                      "gpu_work": False, "payload_bytes_read": 0}))


if __name__ == "__main__":
    main()
