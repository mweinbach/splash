#!/usr/bin/env python3
"""CPU/source witness for persistent hints; no backend, cache or GPU execution."""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
from worker_overlay import (ROOT, PRIVATE, DEFAULT_BUILD, FLAG, DENSE_FLAG, REPLACED_NAMES,
                            EXPECTED_CHANGED, sha, transform, parent_helper, census_cases)


def section(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


def clean_env():
    env = dict(os.environ)
    env.update({FLAG: "0", DENSE_FLAG: "0", "SPLASH_FLASH_MOE_POINTWISE_SEP21": "0",
                "SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21": "0",
                "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21": "0", "SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT": "0"})
    return env


def run_cpu(args, env=None):
    result = subprocess.run([str(arg) for arg in args], cwd=ROOT, env=env or clean_env(),
                            capture_output=True, text=True, timeout=60)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh persistent-residency CPU witness output")
    build = args.build.resolve()
    manifest = json.loads((build / "overlay-manifest.json").read_text())
    base = Path(manifest["dense_w8a8_residency_base_build"])
    parent = json.loads((build / "residency-qualification/base-overlay-manifest.json").read_text())
    helper, _ = parent_helper(base, parent, build / "machinery/parent_dense_worker.py")
    base_make = Path(manifest["dense_w8a8_residency_base_make_path"])
    checks = {
        "parent_manifest_fresh": sha((base / "overlay-manifest.json").read_bytes()) == manifest["dense_w8a8_residency_base_manifest_sha256"],
        "parent_manifest_snapshot_fresh": sha((build / "residency-qualification/base-overlay-manifest.json").read_bytes()) == manifest["dense_w8a8_residency_base_manifest_sha256"],
        "archived_parent_make_fresh": sha(base_make.read_bytes()) == manifest["dense_w8a8_residency_base_make_sha256"],
        "archived_parent_make_snapshot_fresh": sha((build / "residency-qualification/base-worker.mk").read_bytes()) == manifest["dense_w8a8_residency_base_make_sha256"],
        "link_make_fresh": sha((build / "link-inputs.mk").read_bytes()) == manifest["dense_w8a8_residency_link_make_sha256"],
        "archived_parent_generator_fresh": sha((build / "machinery/parent_dense_worker.py").read_bytes()) == manifest["dense_w8a8_residency_parent_helper_sha256"],
        "only_three_expected_host_sources_changed": set(manifest["dense_w8a8_residency_changed_files"]) == EXPECTED_CHANGED,
        "fixed_resource_plan_metadata_unchanged": manifest["dense_w8a8_resource_plan"] == parent["dense_w8a8_resource_plan"] and sha(json.dumps(manifest["dense_w8a8_resource_plan"], sort_keys=True).encode()) == manifest["dense_w8a8_residency_resource_plan_sha256"],
        "census_metadata_matches_corrected_fixed_cases": manifest["dense_w8a8_residency_census"] == census_cases(parent.get("dense_w8a8_hybrid_parent") is True),
        "numerical_math_backing_and_gpu_code_changes_labeled_false": all(manifest[k] is False for k in ("dense_w8a8_residency_math_changed", "dense_w8a8_residency_numerical_identity_changed", "dense_w8a8_residency_backing_and_planners_changed", "dense_w8a8_residency_gpu_code_changed")),
    }
    tools = manifest["dense_w8a8_residency_tools"]
    checks["all_archived_residency_machinery_fresh"] = all(sha((build / r["private_path"]).read_bytes()) == r["sha256"] for r in tools)
    tools_by_name = {Path(r["private_path"]).name: r for r in tools}
    checks["executed_witness_and_generator_match_archived_machinery"] = sha(Path(__file__).read_bytes()) == tools_by_name["worker_witness.py"]["sha256"] and sha(Path(transform.__code__.co_filename).read_bytes()) == tools_by_name["worker_overlay.py"]["sha256"]
    source_mismatch = []
    records = {r["path"]: r for r in manifest["files"]}
    checks["all_source_paths_unique"] = len(records) == len(manifest["files"])
    for relative, record in records.items():
        actual = (build / "source" / relative).read_bytes()
        if record.get("new_dense_w8a8_residency_file"):
            expected = actual
            fresh = sha(actual) == record["repository_sha256"]
        else:
            original = (base / "source" / relative).read_bytes()
            fresh = sha(original) == record["dense_w8a8_residency_base_overlay_sha256"]
            expected = transform(relative, original.decode(), build / "machinery/worker_transform.py").encode()
        if not fresh or actual != expected or sha(actual) != record["overlay_sha256"]:
            source_mismatch.append(relative)
    checks["all_parent_and_frozen_sources_fresh"] = not source_mismatch
    closure = helper.effective_closure(base, base_make)
    checks["actual_parent_object_and_air_closure_unchanged"] = [str(p) for p in closure["objects"]] == manifest["dense_w8a8_residency_effective_parent_objects"] and [str(p) for p in closure["airs"]] == manifest["dense_w8a8_residency_effective_parent_airs"]
    inputs = manifest["dense_w8a8_residency_link_inputs"]
    checks["every_parent_object_retained_except_forward_worker"] = {r["source_path"] for r in inputs if r["category"] in ("REUSED", "CORE")} == {str(p) for p in closure["objects"] if p.stem not in REPLACED_NAMES}
    checks["worker_cache_object_reused_exactly"] = len([r for r in inputs if Path(r["source_path"]).name == "worker_cache.o"]) == 1
    checks["all_parent_air_objects_retained_once"] = [r["source_path"] for r in inputs if r["category"] == "AIRS"] == [str(p) for p in closure["airs"]]
    input_mismatch = [r["private_path"] for r in inputs if sha((build / r["private_path"]).read_bytes()) != r["sha256"] or sha(Path(r["source_path"]).read_bytes()) != r["sha256"]]
    checks["all_reused_compiler_inputs_fresh"] = not input_mismatch
    checks["matching_parent_link_input_seals_authenticated"] = all(sha(Path(path).read_bytes()) == r["sha256"] for path, r in manifest["dense_w8a8_residency_parent_input_seals"].items())
    library_sha = manifest["dense_w8a8_residency_metallib_sha256"]
    checks["metallib_and_gpu_code_byte_identical_to_parent"] = all(sha(path.read_bytes()) == library_sha for path in (build / "splash.metallib", build / "reused/base/splash.metallib", base / "splash.metallib"))
    prior = manifest["dense_w8a8_residency_parent_artifact_witness"]
    checks["prior_parent_published_artifact_witness_fresh"] = sha((build / "residency-qualification/base-cpu-witness-v1.json").read_bytes()) == prior["sha256"] and all(sha((base / name).read_bytes()) == digest for name, digest in prior["artifacts"].items())
    archive_mismatch = [r["private_path"] for r in manifest["dense_w8a8_residency_inherited_archives"] if sha((build / r["private_path"]).read_bytes()) != r["sha256"]]
    checks["all_inherited_dense_qualification_and_machinery_archives_fresh"] = not archive_mismatch
    checks["inherited_source_and_contract_archives_match_prior_parent_seals"] = all(sha((build / "inherited-archives" / relative).read_bytes()) == expected for relative, expected in manifest["dense_w8a8_residency_archive_parent_seals"].items())
    checks["contract_url_and_source_audit_metadata_archived"] = bool(manifest["dense_w8a8_residency_contracts"]) and all(sha((build / r["private_path"]).read_bytes()) == r["sha256"] for r in manifest["dense_w8a8_residency_contracts"])
    contract_text = "\n".join((build / r["private_path"]).read_text() for r in manifest["dense_w8a8_residency_contracts"])
    checks["legacy_transient_contract_inference_and_limits_labeled"] = "supported inference" in contract_text and "setBuffer" in contract_text and "commandBuffer" in contract_text and "does not prove idle pinning" in contract_text and "https://developer.apple.com/" in contract_text
    pins = manifest["dense_w8a8_residency_contract_source_pins"]
    backend = (build / pins["backend_source_private_path"]).read_text()
    operand_store = (build / "source" / pins["operand_store_source_path"]).read_text()
    checks["contract_actual_backend_source_and_core_object_hashes_bound"] = sha(backend.encode()) == pins["backend_source_sha256"] and pins["backend_source_sha256"] in contract_text and pins["backend_object_sha256"] in contract_text and any(r["category"] == "CORE" and Path(r["private_path"]).name == "MetalBackend.o" and r["sha256"] == pins["backend_object_sha256"] for r in inputs)
    checks["ordinary_legacy_direct_bindings_and_strong_command_retention_source_preserved"] = all(token in backend for token in ("[impl_->queue commandBuffer]", "[command computeCommandEncoder]", "[encoder setBuffer:buffer.allocation->buffer", "[encoder useResources:item.readOnlyIndirectResources.data()", "ticketState->retainedAllocations.push_back(allocation)"))
    mapper = section(operand_store, "FlashTensor FlashOperandStore::mapTensor(", "const std::string &FlashOperandStore::identitySha256()")
    checks["saved_source_full_owner_mapper_unchanged_and_hash_bound"] = sha(operand_store.encode()) == pins["operand_store_source_sha256"] and "backend.wrapSharedMemory(mapping->base(), entry.allocated, mapping," in mapper and "result.logicalBytes = entry.logical;" in mapper and "backend.view(" not in mapper
    worker = (build / "source/runtime/flash/FlashWorker.mm").read_text()
    forward = (build / "source/runtime/flash/FlashForward.cpp").read_text()
    attribution = (build / "source/dev/benchmarks/prefill4k_attribution.mm").read_text()
    base_worker = (base / "source/runtime/flash/FlashWorker.mm").read_text()
    base_forward = (base / "source/runtime/flash/FlashForward.cpp").read_text()
    base_attribution = (base / "source/dev/benchmarks/prefill4k_attribution.mm").read_text()
    bridge = (build / "source" / PRIVATE / "worker_bridge.hpp").read_text()
    include = f'#include "{PRIVATE.as_posix()}/worker_bridge.hpp"\n'
    normal_forward = forward.removeprefix(include) if hasattr(str, "removeprefix") else forward[len(include):]
    extra_member = "  const bool denseW8ResidencyPrune = dense_w8a8_residency_sep21::requested();\n"
    checks["only_one_new_frozen_residency_state_member"] = normal_forward.count(extra_member) == 1
    normal_forward = normal_forward.replace(extra_member, "", 1)
    getter_begin = "std::vector<metal::MetalBuffer> FlashForward::cachedOperandsOnly() const {"
    getter_end = "metal::MetalBuffer FlashForward::capturedExpertIDs("
    new_getter = section(normal_forward, getter_begin, getter_end)
    old_getter = section(base_forward, getter_begin, getter_end)
    restored_forward = normal_forward.replace(new_getter, old_getter, 1)
    checks["entire_forward_math_lifetime_backing_planners_and_other_getters_unchanged"] = restored_forward == base_forward
    checks["persistent_selector_filters_saved_views_and_conditional_derived_only"] = "bf16PersistentOperands(" in new_getter and "includeDerived(" in new_getter and "sourceMetadataMatches(" in bridge and "sameView(source.buffer)" in bridge and "matches != 1 || omitted[match]" in bridge
    checks["persistent_selector_has_exact_corrected84_source_census"] = "kBF16SourceBytes = 3774873600ULL" in bridge and "count != kBF16SourceCount || bytes != kBF16SourceBytes" in bridge
    checks["persistent_selector_does_not_free_backing_allocate_or_change_graph"] = all(token not in new_getter + bridge for token in ("allocateBuffer(", "graph.add(", "tryReserve(", "std::make_unique<", ".reset(", ".erase(", ".clear("))
    target_begin = '      << R"(,"target_numerical_derivative_sha256":)" << '
    target_end = '\n      << R"(,"dense_w8a8_prefill_enabled":)"'
    checks["target_numerical_identity_expression_byte_identical"] = section(worker, target_begin, target_end) == section(base_worker, target_begin, target_end)
    admission_begin = "      uint64_t plannedTrunk = FlashForward::workspacePlannedBytes("
    admission_end = "      SavedOperandsResidencyStatus savedResidency;"
    checks["entire_trunk_cache_governor_allocation_and_admission_unchanged"] = section(worker, admission_begin, admission_end) == section(base_worker, admission_begin, admission_end)
    checks["public_runtime_headers_and_cache_implementation_unchanged"] = all((build / "source" / r).read_bytes() == (base / "source" / r).read_bytes() for r in records if (r.startswith("runtime/") and r.endswith((".h", ".hpp"))) or r.endswith("dense_w8a8_sep21/worker_cache.cpp"))
    checks["all_metal_sources_and_abi_headers_unchanged"] = all((build / "source" / r).read_bytes() == (base / "source" / r).read_bytes() for r in records if r.endswith(".metal") or "/metal/abi/" in r)
    checks["prune_strict_selector_frozen_before_path_metadata_backend"] = worker.index("(void)dense_w8a8_residency_sep21::requested();") < worker.index("std::filesystem::canonical(argv[2])") < worker.index("metal::MetalBackend backend(")
    checks["attribution_prune_frozen_before_backend_and_math_unchanged"] = attribution.index("(void)dense_w8a8_residency_sep21::requested();") < attribution.index("metal::MetalBackend backend(") and attribution.replace(include, "", 1).replace("      (void)dense_w8a8_residency_sep21::requested(); // Freeze strict persistent hint policy before backend/model.\n", "", 1) == base_attribution
    checks["status_labels_persistent_hint_and_limits_separately"] = all(token in worker for token in ("dense_w8a8_persistent_residency", "omitted_bf16_source_owner_count", "omitted_unused_i8_owner_count", '"backing_freed":false', '"governor_bypassed":false', '"reclamation_verified":false'))
    if parent.get("dense_w8a8_hybrid_parent"):
        original_begin = "        if (hybridExpertResidencyRequested) {"
        original_end = "        if (originalTextResidencyRequested) {"
        checks["hybrid_original25_expert_and_host_headroom_guard_unchanged"] = section(worker, original_begin, original_end) == section(base_worker, original_begin, original_end)
        checks["hybrid_strict_union_uses_actual_policy_cases"] = "savedResidencyLease.bufferCount() != dense_w8a8_residency_sep21::expectedHybridOwners(" in worker and "savedResidencyLease.byteCount() != dense_w8a8_residency_sep21::expectedHybridBytes(" in worker
    parent_dep_mismatch = []
    for record in manifest["dense_w8a8_residency_parent_dependencies"]:
        if not record["dependency_metadata_present"]:
            continue
        if sha((build / record["private_path"]).read_bytes()) != record["sha256"] or sha(Path(record["source_path"]).read_bytes()) != record["sha256"]:
            parent_dep_mismatch.append(record["source_path"])
        for dep in record["dependencies"]:
            if sha(Path(dep["source_path"]).read_bytes()) != dep.get("sha256", dep.get("external_source_sha256")):
                parent_dep_mismatch.append(dep["source_path"])
    checks["parent_dependency_and_header_closure_fresh"] = not parent_dep_mismatch
    live_dependencies, dependency_mismatch, compiled_dependencies = [], [], []
    for dep_path in (build / "host/FlashForward.d", build / "host/FlashWorker.d", build / "policy-cpu.d", build / "prefill4k-attribution.d"):
        if not dep_path.exists():
            dependency_mismatch.append(str(dep_path)); continue
        for token in helper.dep_tokens(dep_path.read_bytes()):
            path = helper.resolve_input(Path(token)).resolve()
            if build / "source" in path.parents:
                relative = path.relative_to(build / "source").as_posix()
                if relative not in records or sha(path.read_bytes()) != records[relative]["overlay_sha256"]:
                    dependency_mismatch.append(relative)
                else:
                    compiled_dependencies.append({"path": relative, "sha256": records[relative]["overlay_sha256"]})
            elif ROOT / "runtime" in path.parents or ROOT / "dev" in path.parents:
                live_dependencies.append(str(path))
    checks["new_compiled_hosts_and_executables_have_private_header_closure"] = not live_dependencies and not dependency_mismatch
    probes = {mode or "default": run_cpu([build / "policy-cpu", *([mode] if mode else [])]) for mode in ("", "--freeze0", "--freeze1", "--invalid")}
    checks["compiled_residency_policy_and_strict_freeze_pass"] = all(r["returncode"] == 0 for r in probes.values())
    policy = json.loads(probes["default"]["stdout"]) if probes["default"]["returncode"] == 0 else {}
    checks["compiled_residency_policy_census_matches_frozen_metadata"] = policy.get("persistent_residency_cpu_policy") == "passed" and policy.get("omitted_source_count") == 84 and policy.get("omitted_source_bytes") == 3774873600 and policy.get("gpu_work") is False and policy.get("payload_reads") is False
    inherited_probes = {name: run_cpu([build / name]) for name in ("inherited-dense-policy-cpu", "inherited-cache-policy-cpu")}
    checks["inherited_dense_and_cache_cpu_policies_pass"] = all(r["returncode"] == 0 for r in inherited_probes.values())
    worker_probe = run_cpu([build / "splash-flash", "--cpu-self-test"])
    checks["main_worker_cpu_self_test_pass"] = worker_probe["returncode"] == 0
    rejected = {}
    for dense in ("0", "1"):
        for value in ("", "2", "true", "01", " 1", "1 ", "-1"):
            env = clean_env(); env[FLAG] = value; env[DENSE_FLAG] = dense
            result = run_cpu([build / "splash-flash", "serve-flash-native", "/persistent-prune-does-not-exist", "16384", "auto"], env)
            key = f"dense={dense},prune={value!r}"; rejected[key] = result
            checks[f"invalid_prune_rejected_before_paths_backend:{key}"] = result["returncode"] != 0 and FLAG in result["stderr"] and "must be 0 or 1" in result["stderr"]
    witness = {"schema": "splash-dense-w8a8-persistent-residency-prune-cpu-witness-v1", "pass": all(checks.values()),
               "gpu_work": False, "payload_bytes_read": 0, "model_loaded": False, "cache_constructed": False,
               "checks": checks, "source_mismatch": source_mismatch, "link_input_mismatch": input_mismatch,
               "archive_mismatch": archive_mismatch, "parent_dependency_mismatch": parent_dep_mismatch,
               "dependency_mismatch": dependency_mismatch, "live_project_dependencies": live_dependencies,
               "compiled_dependencies": list({r["path"]: r for r in compiled_dependencies}.values()),
               "compiled_residency_policy": probes, "inherited_dense_cpu_policies": inherited_probes,
               "main_worker_cpu_self_test": worker_probe, "invalid_prune_probes": rejected,
               "persistent_census": manifest["dense_w8a8_residency_census"], "resource_plan": manifest["dense_w8a8_resource_plan"],
               "contract_evidence": manifest["dense_w8a8_residency_contracts"],
               "frozen_source_count": len(records), "frozen_link_input_count": len(inputs),
               "runtime_sha256": {name: sha((build / name).read_bytes()) for name in ("splash-flash", "splash.metallib", "prefill4k-attribution")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(witness, indent=2) + "\n")
    print(json.dumps({"pass": witness["pass"], "failed_checks": [k for k, v in checks.items() if not v],
                      "frozen_sources": len(records), "frozen_link_inputs": len(inputs),
                      "metallib_byte_identical": checks["metallib_and_gpu_code_byte_identical_to_parent"],
                      "gpu_work": False, "payload_bytes_read": 0}))
    if not witness["pass"]:
        raise ValueError("Persistent residency prune CPU/source witness failed")


if __name__ == "__main__":
    main()
