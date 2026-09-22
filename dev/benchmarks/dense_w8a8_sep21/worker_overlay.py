#!/usr/bin/env python3
"""Freeze a private dense W8A8 worker using the actual parent link closure.

This program reads source, object, AIR and report metadata only. It does not
instantiate a backend/cache or open a model/captured operand payload.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/dense_w8a8_sep21")
DEFAULT_BASE = ROOT / "build/adaptive-expert-tail-sg2k128-fma-sep21-worker-v1"
DEFAULT_BUILD = ROOT / "build/dense-w8a8-sep21-worker-v3"
QUALIFIED_KERNEL_SHA256 = "32806c52af23cee704b4e64864fa6befc3b38f81fc097f05f5d8d3c5fc1df555"
FLAG = "SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21"
REPLACED_NAMES = {"FlashForward", "FlashWorker"}
EXPECTED_CHANGED = {"runtime/flash/FlashForward.cpp", "runtime/flash/FlashWorker.mm",
                    "dev/benchmarks/prefill4k_attribution.mm"}
PRIVATE_NAMES = ("candidate.metal", "abi.hpp", "quantization.hpp", "worker_cache.hpp",
                 "worker_cache.cpp", "worker_bridge.hpp", "worker_policy_cpu.cpp", "cache_policy_cpu.cpp")
TOOL_NAMES = ("worker_overlay.py", "worker_transform.py", "worker_witness.py", "worker.mk", "worker_README.md")
ROLE_SPECS = {
    "qkv": ("linear_attn.in_proj_qkv", 2560, 10240, 36),
    "z": ("linear_attn.in_proj_z", 2560, 6144, 36),
    "qsa_q": ("self_attn.q_proj", 2560, 12288, 12),
}
SHADER_HEADERS = ("runtime/metal/abi/FlashDenseCache.h", "runtime/metal/abi/FlashAffine.h",
                  "runtime/metal/kernels/common/flash_dense_traversal.h")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def transform(relative, text, transform_path=None):
    path = transform_path or Path(__file__).with_name("worker_transform.py")
    spec = importlib.util.spec_from_file_location("dense_w8a8_worker_transform", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.transform(relative, text)


def resolve_input(path):
    return path if path.is_absolute() else ROOT / path


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe frozen source path: {value}")
    return path


def infer_base_make(base, manifest):
    candidates = [
        base / "worker.mk",
        base / "source/dev/benchmarks/adaptive_expert_tail_sep21/hybrid_sg2/worker.mk",
        base / "source/dev/benchmarks/adaptive_expert_tail_sep21/combined/worker.mk",
        base / "source/dev/benchmarks/moe_pointwise_sep21/worker.mk",
    ]
    for path in candidates:
        if path.exists():
            return path
    if manifest.get("hybrid_combined_composed"):
        return ROOT / "dev/benchmarks/adaptive_expert_tail_sep21/combined/hybrid.mk"
    if manifest.get("combined_tail_composed"):
        return ROOT / "dev/benchmarks/adaptive_expert_tail_sep21/combined/worker.mk"
    if manifest.get("pointwise_composed") and not manifest.get("fixed_sg2_scope"):
        return ROOT / "dev/benchmarks/moe_pointwise_sep21/worker.mk"
    raise ValueError("Provide --base-make for this parent worker; closure must use its actual build rules")


def unique_paths(tokens, suffix, label):
    if any(Path(t).suffix != suffix for t in tokens):
        raise ValueError(f"Unexpected {label} prerequisite: {tokens}")
    paths = [resolve_input(Path(t)).resolve() for t in tokens]
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate effective {label} inputs")
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(f"Incomplete effective {label} inputs")
    return paths


def effective_closure(base, base_make):
    """Use expanded target prerequisites, not an inherited manifest's input list."""
    result = subprocess.run(["make", "-pqn", "-rR", "-f", str(base_make), f"BUILD={base}",
                             str(base / "splash-flash"), str(base / "splash.metallib")],
                            cwd=ROOT, text=True, capture_output=True, timeout=60)
    if result.returncode not in (0, 1):
        raise ValueError(f"Parent make closure failed: {result.stderr.strip()}")
    database = result.stdout
    def prerequisites(target):
        prefix = str(base / target) + ":"
        lines = [line for line in database.splitlines() if line.startswith(prefix)]
        if len(lines) != 1:
            raise ValueError(f"Cannot determine actual parent {target} prerequisites")
        return shlex.split(lines[0].split(":", 1)[1])
    objects = unique_paths(prerequisites("splash-flash"), ".o", "host/core")
    if len({p.name for p in objects}) != len(objects):
        raise ValueError("Duplicate host/core object basenames in the actual parent link")
    airs = unique_paths(prerequisites("splash.metallib"), ".air", "AIR")
    core_lines = re.findall(r"^CORE\s*:=\s*(.*)$", database, re.MULTILINE)
    if len(core_lines) != 1:
        raise ValueError("Cannot determine the parent CORE object classification")
    core = unique_paths(shlex.split(core_lines[0]), ".o", "core")
    if not set(core).issubset(objects):
        raise ValueError("Parent CORE objects are absent from the actual host link")
    if {p.stem for p in objects if p.stem in REPLACED_NAMES} != REPLACED_NAMES:
        raise ValueError("Actual parent link must contain Forward and Worker exactly once")
    if any(sum(p.stem == name for p in objects) != 1 for name in REPLACED_NAMES):
        raise ValueError("Ambiguous actual parent Forward/Worker object")
    return {"objects": objects, "core": core, "airs": airs}


def parent_input_seals(base, manifest, closure):
    effective = set(closure["objects"] + closure["airs"])
    known = {}
    for key, value in manifest.items():
        if not key.endswith("link_inputs") or not isinstance(value, list):
            continue
        for record in value:
            if not isinstance(record, dict) or "private_path" not in record or "sha256" not in record:
                continue
            path = (base / safe_relative(record["private_path"])).resolve()
            if path not in effective:
                continue
            if path in known and known[path]["sha256"] != record["sha256"]:
                raise ValueError(f"Conflicting parent compiler input seals: {path}")
            if sha(path.read_bytes()) != record["sha256"]:
                raise ValueError(f"Authenticated parent compiler input drift: {path}")
            known[path] = {"manifest_key": key, "sha256": record["sha256"]}
    unsealed = effective - set(known)
    if any(path.parent != base / "host" and not (path.parent == base and path.suffix == ".air") for path in unsealed):
        raise ValueError(f"Inherited effective input has no matching parent seal: {sorted(map(str, unsealed))}")
    return {str(path): value for path, value in known.items()}, [str(p) for p in sorted(unsealed)]


def parent_artifact_witness(base):
    path = base / "cpu-witness-v1.json"
    witness = json.loads(path.read_text())
    if witness.get("pass") is not True:
        raise ValueError("Parent worker has no passing prior CPU/source artifact witness")
    artifacts = witness["runtime_sha256"]
    for name in ("splash-flash", "splash.metallib", "prefill4k-attribution"):
        if sha((base / name).read_bytes()) != artifacts[name]:
            raise ValueError(f"Parent published artifact differs from prior CPU/source witness: {name}")
    return {"source_path": str(path), "sha256": sha(path.read_bytes()), "artifacts": artifacts,
            "owned_object_provenance": "prior published artifacts pass; .d sources/headers and owned inputs independently pinned here; no prior individual owned-object seal available"}


def dep_tokens(data):
    first = data.decode().replace("\\\n", " ").splitlines()[0]
    if ":" not in first:
        raise ValueError("Malformed compiler dependency metadata")
    return shlex.split(first.split(":", 1)[1])


def frozen_dependency_closure(base, objects, file_records):
    """Pin available .d files and bind parent-owned includes to frozen sources."""
    records = []
    for obj in objects:
        dep = obj.with_suffix(".d")
        own_object = obj.parent == base / "host"
        if own_object and not dep.is_file():
            raise ValueError(f"Missing dependency metadata for rebuilt parent object: {obj}")
        if not dep.is_file():
            records.append({"object_source_path": str(obj), "dependency_metadata_present": False,
                            "provenance": "inherited frozen object; parent source/header closure retained in files"})
            continue
        data = dep.read_bytes()
        dependencies = []
        for token in dep_tokens(data):
            path = resolve_input(Path(token)).resolve()
            if base / "source" in path.parents:
                relative = path.relative_to(base / "source").as_posix()
                if relative not in file_records:
                    raise ValueError(f"Parent .d includes an unmanifested frozen source: {relative}")
                if sha(path.read_bytes()) != file_records[relative]["overlay_sha256"]:
                    raise ValueError(f"Parent header dependency drift: {relative}")
                dependencies.append({"source_path": str(path), "frozen_source_path": relative,
                                     "sha256": sha(path.read_bytes())})
            elif own_object and ROOT / "runtime" in path.parents:
                raise ValueError(f"Rebuilt parent object used a live runtime dependency: {token}")
            elif path.is_file():
                dependencies.append({"source_path": str(path), "external_source_sha256": sha(path.read_bytes())})
        records.append({"object_source_path": str(obj), "dependency_metadata_present": True,
                        "source_path": str(dep), "sha256": sha(data), "dependencies": dependencies})
    return records


def verify_qualified_manifest(path):
    manifest = json.loads(path.read_text())
    verified = []
    for category in ("sources", "artifacts"):
        for record in manifest[category]:
            original = resolve_input(Path(record["path"]))
            data = original.read_bytes()
            if sha(data) != record["sha256"] or len(data) != record["bytes"]:
                raise ValueError(f"Qualified dense compiler input/artifact drift: {original}")
            verified.append({"category": category, **record})
    candidate = next(r for r in manifest["sources"] if r["path"] == (PRIVATE / "candidate.metal").as_posix())
    if candidate["sha256"] != QUALIFIED_KERNEL_SHA256:
        raise ValueError("Qualified W8A8 kernel does not match the root-certified shader")
    return verified


def qualify_role(path, role):
    report = json.loads(path.read_text())
    suffix, width, outputs, layers = ROLE_SPECS[role]
    if not report["projection"].endswith(suffix) or (report["rows"], report["input_size"], report["output_size"]) != (2048, width, outputs):
        raise ValueError(f"Unexpected W8A8 component report geometry: {path}")
    if (not report["numerical_alternative"] or report["full_model_quality_qualified"] or
            not report["captured_control_exact"] or not report["guards_passed"] or
            not report["immutable_coefficients_passed"] or not report["probe_immutable_during_timing"] or
            not report["activation_converter_in_every_timed_projection"] or report["probe_i32_writes_in_timing"] or
            report["cpu_operand_output_access_in_warm_or_timing"]):
        raise ValueError(f"Incomplete W8A8 component certificate: {path}")
    activation = report["activation_certificate"]
    if any(activation[k] != 0 for k in ("scale_bit_mismatches", "rne_code_mismatches", "forbidden_minus128_codes")):
        raise ValueError(f"Activation conversion certificate failed: {path}")
    variants = [v for v in report["variants"] if v["w8a8"]]
    selected = next(v for v in variants if v["simdgroups"] == 4)
    for variant in variants:
        error = variant["full_baseline_error"]
        if (not variant["preregistered_component_quality_passed"] or not variant["numerical_alternative"] or
                variant["integer_dot_mismatches"] or variant["full_late_scale_identity_mismatches"] or
                error["nonfinite"] or error["relative_l2"] > 0.02 or error["cosine"] < 0.9998 or
                variant["warm_candidate_gpu_ms"] < 150 or variant["warm_matched_control_gpu_ms"] < 150 or
                variant["timed_ab_pairs"] != variant["timed_ba_pairs"]):
            raise ValueError(f"W8A8 component variant qualification failed: {path}")
    return {"role": role, "path": str(path), "report_sha256": sha(path.read_bytes()),
            "projection": report["projection"], "rows": 2048, "input_size": width,
            "output_size": outputs, "role_count": layers, "selected_simdgroups": 4,
            "source_weight_sha256": report["source_weight_sha256"],
            "source_input_sha256": report["source_input_sha256"],
            "full_baseline_error": selected["full_baseline_error"],
            "activation_quantization_error": activation["source_fp64_dequantization_error"],
            "weight_quantization_error": report["weight_quantization_error"],
            "paired_median_speedup": selected["paired_median_speedup"],
            "numerical_alternative": True, "full_model_quality_qualified": False}


def resource_plan():
    def rounded(bytes):
        return (bytes + 16383) // 16384 * 16384
    roles = []
    for role, (_, width, outputs, count) in ROLE_SPECS.items():
        roles.append({"role": role, "input_size": width, "output_size": outputs, "count": count,
                      "i8_coefficient_bytes": width * outputs * count,
                      "f32_coefficient_scale_bytes": 4 * outputs * count,
                      "rounded_planned_bytes": count * (rounded(width * outputs) + rounded(4 * outputs))})
    return {"role_count": 84, "coefficient_buffer_count": 168, "roles": roles,
            "coefficient_logical_bytes": sum(r["i8_coefficient_bytes"] + r["f32_coefficient_scale_bytes"] for r in roles),
            "coefficient_planned_bytes": sum(r["rounded_planned_bytes"] for r in roles),
            "activation_workspace_planned_bytes": rounded(2048 * 6144) + rounded(2048 * 4) + rounded(4),
            "activation_workspace_logical_bytes": [2048 * 6144, 2048 * 4, 4],
            "activation_workspace_buffer_count": 3, "maximum_rows_guard": 2048,
            "allocation_alignment": 16384,
            "coefficient_source": "verified cached original saved BF16 coefficient rows",
            "coefficient_quantization": "symmetric signed I8 [-127,127], one F32 scale per output row",
            "activation_quantization": "GPU BF16 row max, F32 scale, precise RNE signed I8",
            "timed_route": "main singleton nonverification exactly R2048; SG4 M128/N64",
            "excluded_roles": ["linear_attn.out_proj"],
            "planned_for_both_flags": True, "cache_constructed_by_this_freezer": False,
            "resource_allocation_guard": "MemoryGovernor reservation before private workspace/coefficient cache construction"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-make", type=Path)
    parser.add_argument("--qualified", type=Path, default=ROOT / "build/dense-w8a8-sep21")
    parser.add_argument("--qkv-report", type=Path)
    parser.add_argument("--z-report", type=Path)
    parser.add_argument("--qsa-q-report", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUILD)
    args = parser.parse_args()
    base, qualified, output = (p.resolve() for p in (args.base, args.qualified, args.output))
    if output in (base, qualified) or ROOT / "build" not in output.parents:
        raise ValueError("Dense W8A8 overlay requires a distinct private build directory")
    if (output / "overlay-manifest.json").exists():
        raise ValueError("Choose a fresh private worker output; frozen workers are immutable")
    parent_path = base / "overlay-manifest.json"
    parent = json.loads(parent_path.read_text())
    full512 = parent.get("pointwise_composed") and parent.get("omitted_original_target_tensor_count") == 432
    hybrid = parent.get("hybrid_combined_composed") is True
    if not (full512 or hybrid):
        raise ValueError("Expected a sealed private pointwise Full512 or composed resident hybrid parent worker")
    base_make = args.base_make.resolve() if args.base_make else infer_base_make(base, parent)
    closure = effective_closure(base, base_make)
    known_input_seals, unsealed_owned_inputs = parent_input_seals(base, parent, closure)
    parent_witness = parent_artifact_witness(base)
    records = {r["path"]: r for r in parent["files"]}
    if len(records) != len(parent["files"]):
        raise ValueError("Parent source manifest has duplicate paths")
    deps = frozen_dependency_closure(base, closure["objects"], records)
    qualified_path = qualified / "source-manifest.json"
    qualified_verified = verify_qualified_manifest(qualified_path)
    qualified_sources = {r["path"]: r for r in qualified_verified if r["category"] == "sources"}
    shader_cross_pins = []
    for relative in SHADER_HEADERS:
        if records[relative]["overlay_sha256"] != qualified_sources[relative]["sha256"]:
            raise ValueError(f"Frozen parent shader header differs from root-qualified source: {relative}")
        shader_cross_pins.append({"path": relative, "sha256": qualified_sources[relative]["sha256"]})
    report_paths = {"qkv": args.qkv_report or qualified / "qkv-root-v1.json",
                    "z": args.z_report or qualified / "z-root-v1.json",
                    "qsa_q": args.qsa_q_report or qualified / "qsa-q-root-v1.json"}
    evidence = [qualify_role(path.resolve(), role) for role, path in report_paths.items()]
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-inherited-dense-w8a8-prefill-sep21-v1",
                     "dense_w8a8_parent_route": parent["route"], "dense_w8a8_hybrid_parent": hybrid,
                     "dense_w8a8_composed": True, "dense_w8a8_numerical_alternative": True,
                     "dense_w8a8_full_model_quality_qualified": False,
                     "dense_w8a8_required_environment": f"{FLAG}=0|1", "dense_w8a8_default": 0,
                     "dense_w8a8_base_build": str(base), "dense_w8a8_base_manifest_sha256": sha(parent_path.read_bytes()),
                     "dense_w8a8_base_make_path": str(base_make), "dense_w8a8_base_make_sha256": sha(base_make.read_bytes()),
                     "dense_w8a8_kernel_source_sha256": QUALIFIED_KERNEL_SHA256,
                     "dense_w8a8_qualified_manifest_sha256": sha(qualified_path.read_bytes()),
                     "dense_w8a8_qualified_manifest_path": str(qualified_path),
                     "dense_w8a8_qualified_inputs": qualified_verified,
                     "dense_w8a8_component_certificates": evidence, "dense_w8a8_resource_plan": resource_plan(),
                     "dense_w8a8_controls_independent": True, "dense_w8a8_public_headers_unchanged": True,
                     "dense_w8a8_scope": "main singleton nonverification R2048 QKV/Z/QSA-Q only; batch/decode/verify inherited",
                     "dense_w8a8_whole_model_qualification": "pending root single-driver model quality/performance qualification",
                     "dense_w8a8_effective_parent_objects": [str(p) for p in closure["objects"]],
                     "dense_w8a8_effective_parent_airs": [str(p) for p in closure["airs"]],
                     "dense_w8a8_parent_dependencies": deps,
                     "dense_w8a8_parent_input_seals": known_input_seals,
                     "dense_w8a8_parent_owned_inputs_without_prior_individual_seal": unsealed_owned_inputs,
                     "dense_w8a8_parent_artifact_witness": parent_witness,
                     "dense_w8a8_shader_header_cross_pins": shader_cross_pins,
                     "dense_w8a8_link_inputs": [], "dense_w8a8_tools": [], "files": [],
                     "normal_sources_modified": False, "arithmetic_change": True,
                     "gpu_executed": False, "payload_bytes_read": 0})
    changed = []
    for relative, record in records.items():
        original = (base / "source" / safe_relative(relative)).read_bytes()
        if sha(original) != record["overlay_sha256"]:
            raise ValueError(f"Parent frozen source drift: {relative}")
        data = transform(relative, original.decode()).encode()
        if data != original:
            changed.append(relative)
        write(output / "source" / relative, data)
        manifest["files"].append({**record, "dense_w8a8_changed": data != original,
                                  "dense_w8a8_base_overlay_sha256": sha(original), "overlay_sha256": sha(data)})
    if set(changed) != EXPECTED_CHANGED:
        raise ValueError(f"Unexpected dense W8A8 transformed sources: {changed}")
    for name in PRIVATE_NAMES:
        relative = PRIVATE / name
        data = (ROOT / relative).read_bytes()
        if name == "candidate.metal" and sha(data) != QUALIFIED_KERNEL_SHA256:
            raise ValueError("Dense W8A8 shader differs from the root-qualified source")
        write(output / "source" / relative, data)
        manifest["files"].append({"path": relative.as_posix(), "new_dense_w8a8_file": True,
                                  "repository_sha256": sha(data), "overlay_sha256": sha(data)})
    inputs = {"REUSED": [], "CORE": [], "AIRS": []}
    def freeze(path, category):
        if base not in path.parents:
            raise ValueError(f"Effective parent input must be private under its frozen build: {path}")
        relative = Path("reused/base") / path.relative_to(base)
        data = path.read_bytes()
        write(output / relative, data)
        inputs[category].append(relative.as_posix())
        manifest["dense_w8a8_link_inputs"].append({"source_path": str(path), "private_path": relative.as_posix(),
                                                 "category": category, "sha256": sha(data)})
    for path in closure["objects"]:
        if path.stem not in REPLACED_NAMES:
            freeze(path, "CORE" if path in closure["core"] else "REUSED")
    for path in closure["airs"]:
        freeze(path, "AIRS")
    for record in deps:
        if record["dependency_metadata_present"]:
            path = Path(record["source_path"])
            relative = Path("qualification/parent-dependencies") / path.relative_to(base)
            write(output / relative, path.read_bytes())
            record["private_path"] = relative.as_posix()
    make = "\n".join(f"{key} := " + " ".join("$(BUILD)/" + p for p in paths) for key, paths in inputs.items()) + "\n"
    write(output / "link-inputs.mk", make.encode())
    manifest["dense_w8a8_link_make_sha256"] = sha(make.encode())
    manifest["dense_w8a8_changed_files"] = changed
    for name in TOOL_NAMES:
        path = ROOT / PRIVATE / name
        relative = Path("machinery") / name
        data = path.read_bytes()
        write(output / relative, data)
        manifest["dense_w8a8_tools"].append({"source_path": str(path), "private_path": relative.as_posix(), "sha256": sha(data)})
    for index, record in enumerate(manifest["dense_w8a8_qualified_inputs"]):
        relative = Path("qualification/standalone") / record["category"] / f"{index:03}-{Path(record['path']).name}"
        write(output / relative, resolve_input(Path(record["path"])).read_bytes())
        record["private_path"] = relative.as_posix()
    if hybrid:
        census = parent["expert_residency_metadata_census"]
        if (census["selected_owner_count"], census["selected_owner_bytes"], census["composite_owner_count"], census["composite_owner_bytes"]) != (25, 69363302400, 748, 202252746752):
            raise ValueError("Resident hybrid parent differs from its sealed original expert/union census")
        manifest["dense_w8a8_hybrid_residency_census"] = {
            "original_expert_owner_count": 25, "original_expert_owner_bytes": 69363302400,
            "inherited_union_owner_count": 748, "inherited_union_owner_bytes": 202252746752,
            "added_immutable_owner_count": 168, "added_immutable_owner_bytes": resource_plan()["coefficient_planned_bytes"],
            "maximum_rows_ge2048_union_owner_count": 916,
            "maximum_rows_ge2048_union_owner_bytes": 202252746752 + resource_plan()["coefficient_planned_bytes"],
            "workspace_added_to_immutable_union": False,
            "inherited_original_expert_headroom_and_owner_guard_unchanged": True,
            "all_census_evidence_is_metadata_only": True}
    for path in [parent_path, qualified_path, base_make, Path(parent_witness["source_path"]), *[Path(e["path"]) for e in evidence]]:
        relative = Path("qualification") / ("base-overlay-manifest.json" if path == parent_path else
            "base-worker.mk" if path == base_make else
            "base-cpu-witness-v1.json" if path == Path(parent_witness["source_path"]) else path.name)
        write(output / relative, path.read_bytes())
    manifest["dense_w8a8_base_link_make_sha256"] = sha((base / "link-inputs.mk").read_bytes())
    write(output / "qualification/base-link-inputs.mk", (base / "link-inputs.mk").read_bytes())
    for name in ("splash-flash.config", "splash.metallib.config"):
        inherited = (base / name).read_bytes().rstrip(b"\n") if (base / name).is_file() else ("sealed-parent-route:" + parent["route"]).encode()
        write(output / name, inherited + b"-dense-w8a8-prefill-sep21-v1\n")
    write(output / "overlay-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    write(output / "base-build.txt", (str(base) + "\n").encode())
    print(json.dumps({"prepared": str(output), "changed_sources": changed, "frozen_sources": len(manifest["files"]),
                      "effective_parent_objects": len(closure["objects"]), "effective_parent_airs": len(closure["airs"]),
                      "frozen_link_inputs": len(manifest["dense_w8a8_link_inputs"]), "gpu_work": False,
                      "payload_bytes_read": 0, "numerical_alternative": True,
                      "resource_plan": manifest["dense_w8a8_resource_plan"]}))


if __name__ == "__main__":
    main()
