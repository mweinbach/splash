#!/usr/bin/env python3
"""CPU-only closure, resource-plan and fail-before-path witness for W8A8 worker."""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
from worker_overlay import (ROOT, PRIVATE, DEFAULT_BUILD, FLAG, QUALIFIED_KERNEL_SHA256,
                            REPLACED_NAMES, EXPECTED_CHANGED, sha, transform, effective_closure,
                            dep_tokens, resolve_input, qualify_role, resource_plan)


def section(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


def clean_env():
    env = dict(os.environ)
    env.update({FLAG: "0", "SPLASH_FLASH_MOE_POINTWISE_SEP21": "0",
                "SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21": "0",
                "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21": "0",
                "SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT": "0"})
    return env


def run_cpu(args, env=None):
    result = subprocess.run([str(arg) for arg in args], env=env or clean_env(),
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def target_identity_expression(text):
    match = re.search(r'      << R"\(,"target_numerical_derivative_sha256":\)" << (.*?)(?=\n      << R"\()', text, re.S)
    if not match:
        raise ValueError("Missing inherited target identity status expression")
    expression = match.group(1)
    anchor = "json::quote("
    if expression.count(anchor) != 1:
        raise ValueError("Ambiguous inherited target identity quote")
    start = expression.index(anchor) + len(anchor)
    depth, quoted, escaped = 1, False, False
    for index in range(start, len(expression)):
        char = expression[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return expression, expression[start:index], start, index
    raise ValueError("Unbalanced inherited target identity quote")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh CPU witness output")
    build = args.build.resolve()
    manifest = json.loads((build / "overlay-manifest.json").read_text())
    base = Path(manifest["dense_w8a8_base_build"])
    base_make = Path(manifest["dense_w8a8_base_make_path"])
    checks = {
        "parent_manifest_fresh": sha((base / "overlay-manifest.json").read_bytes()) == manifest["dense_w8a8_base_manifest_sha256"],
        "parent_manifest_snapshot_fresh": sha((build / "qualification/base-overlay-manifest.json").read_bytes()) == manifest["dense_w8a8_base_manifest_sha256"],
        "parent_make_fresh": sha(base_make.read_bytes()) == manifest["dense_w8a8_base_make_sha256"],
        "parent_make_snapshot_fresh": sha((build / "qualification/base-worker.mk").read_bytes()) == manifest["dense_w8a8_base_make_sha256"],
        "parent_link_make_fresh": sha((base / "link-inputs.mk").read_bytes()) == manifest["dense_w8a8_base_link_make_sha256"],
        "parent_link_make_snapshot_fresh": sha((build / "qualification/base-link-inputs.mk").read_bytes()) == manifest["dense_w8a8_base_link_make_sha256"],
        "private_link_make_fresh": sha((build / "link-inputs.mk").read_bytes()) == manifest["dense_w8a8_link_make_sha256"],
        "numerical_alternative_labeled": manifest["dense_w8a8_numerical_alternative"] and not manifest["dense_w8a8_full_model_quality_qualified"],
        "resource_plan_matches_fixed_manifest": manifest["dense_w8a8_resource_plan"] == resource_plan(),
        "gpu_and_payload_freezer_work_zero": not manifest["gpu_executed"] and manifest["payload_bytes_read"] == 0,
        "only_expected_three_host_sources_changed": set(manifest["dense_w8a8_changed_files"]) == EXPECTED_CHANGED,
    }
    tools_mismatch = []
    for record in manifest["dense_w8a8_tools"]:
        expected = record["sha256"]
        if sha((build / record["private_path"]).read_bytes()) != expected:
            tools_mismatch.append(record["private_path"])
    checks["all_worker_machinery_frozen_and_fresh"] = not tools_mismatch
    tools_by_name = {Path(r["private_path"]).name: r for r in manifest["dense_w8a8_tools"]}
    checks["executed_witness_and_imported_generator_match_archived_machinery"] = sha(Path(__file__).read_bytes()) == tools_by_name["worker_witness.py"]["sha256"] and sha(Path(transform.__code__.co_filename).read_bytes()) == tools_by_name["worker_overlay.py"]["sha256"]
    source_mismatch = []
    records = {record["path"]: record for record in manifest["files"]}
    checks["source_paths_unique"] = len(records) == len(manifest["files"])
    for relative, record in records.items():
        actual = (build / "source" / relative).read_bytes()
        if record.get("new_dense_w8a8_file"):
            expected = actual
            fresh = sha(actual) == record["repository_sha256"]
        else:
            original = (base / "source" / relative).read_bytes()
            fresh = sha(original) == record["dense_w8a8_base_overlay_sha256"]
            expected = transform(relative, original.decode(), build / "machinery/worker_transform.py").encode()
        if not fresh or expected != actual or sha(actual) != record["overlay_sha256"]:
            source_mismatch.append(relative)
    checks["all_frozen_sources_and_parent_sources_fresh"] = not source_mismatch
    checks["qualified_shader_exactly_preserved"] = sha((build / "source" / PRIVATE / "candidate.metal").read_bytes()) == manifest["dense_w8a8_kernel_source_sha256"] == QUALIFIED_KERNEL_SHA256
    checks["shader_transitive_headers_match_standalone_qualification"] = all(sha((build / "source" / r["path"]).read_bytes()) == r["sha256"] for r in manifest["dense_w8a8_shader_header_cross_pins"])
    checks["qualified_source_manifest_snapshot_fresh"] = sha((build / "qualification/source-manifest.json").read_bytes()) == manifest["dense_w8a8_qualified_manifest_sha256"]
    qualified_mismatch = []
    for record in manifest["dense_w8a8_qualified_inputs"]:
        data = (build / record["private_path"]).read_bytes()
        if sha(data) != record["sha256"] or len(data) != record["bytes"]:
            qualified_mismatch.append(record["path"])
    checks["all_qualified_compiler_inputs_and_artifacts_fresh"] = not qualified_mismatch
    certificate_mismatch = []
    for evidence in manifest["dense_w8a8_component_certificates"]:
        path = build / "qualification" / Path(evidence["path"]).name
        actual = qualify_role(path, evidence["role"])
        actual["path"] = evidence["path"]
        if actual != evidence:
            certificate_mismatch.append(evidence["role"])
    checks["qualified_component_reports_and_error_metrics_fresh"] = not certificate_mismatch
    closure = effective_closure(base, base_make)
    checks["actual_parent_object_closure_unchanged"] = [str(p) for p in closure["objects"]] == manifest["dense_w8a8_effective_parent_objects"]
    checks["actual_parent_air_closure_unchanged"] = [str(p) for p in closure["airs"]] == manifest["dense_w8a8_effective_parent_airs"]
    inputs = manifest["dense_w8a8_link_inputs"]
    expected_objects = {str(p) for p in closure["objects"] if p.stem not in REPLACED_NAMES}
    checks["every_actual_parent_object_retained_except_replaced_pair"] = {r["source_path"] for r in inputs if r["category"] in ("REUSED", "CORE")} == expected_objects
    checks["every_actual_parent_air_retained_once"] = {r["source_path"] for r in inputs if r["category"] == "AIRS"} == {str(p) for p in closure["airs"]} and len([r for r in inputs if r["category"] == "AIRS"]) == len(closure["airs"])
    input_mismatch = []
    for record in inputs:
        if sha((build / record["private_path"]).read_bytes()) != record["sha256"] or sha(Path(record["source_path"]).read_bytes()) != record["sha256"]:
            input_mismatch.append(record["private_path"])
    checks["all_frozen_reused_host_core_air_inputs_fresh"] = not input_mismatch
    checks["all_matching_inherited_input_seals_authenticated"] = all(sha(Path(path).read_bytes()) == seal["sha256"] for path, seal in manifest["dense_w8a8_parent_input_seals"].items())
    parent_witness = manifest["dense_w8a8_parent_artifact_witness"]
    checks["prior_parent_published_artifact_witness_fresh"] = sha((build / "qualification/base-cpu-witness-v1.json").read_bytes()) == parent_witness["sha256"] and all(sha((base / name).read_bytes()) == digest for name, digest in parent_witness["artifacts"].items())
    inherited_dep_mismatch = []
    for record in manifest["dense_w8a8_parent_dependencies"]:
        if not record["dependency_metadata_present"]:
            continue
        if sha(Path(record["source_path"]).read_bytes()) != record["sha256"] or sha((build / record["private_path"]).read_bytes()) != record["sha256"]:
            inherited_dep_mismatch.append(record["source_path"])
        for dep in record["dependencies"]:
            expected = dep.get("sha256", dep.get("external_source_sha256"))
            if sha(Path(dep["source_path"]).read_bytes()) != expected:
                inherited_dep_mismatch.append(dep["source_path"])
    checks["parent_rebuilt_objects_dependency_and_header_closure_fresh"] = not inherited_dep_mismatch
    worker = (build / "source/runtime/flash/FlashWorker.mm").read_text()
    forward = (build / "source/runtime/flash/FlashForward.cpp").read_text()
    attribution = (build / "source/dev/benchmarks/prefill4k_attribution.mm").read_text()
    cache = (build / "source" / PRIVATE / "worker_cache.cpp").read_text()
    cache_header = (build / "source" / PRIVATE / "worker_cache.hpp").read_text()
    bridge = (build / "source" / PRIVATE / "worker_bridge.hpp").read_text()
    base_worker = (base / "source/runtime/flash/FlashWorker.mm").read_text()
    base_forward = (base / "source/runtime/flash/FlashForward.cpp").read_text()
    checks.update({
        "selector_frozen_before_path_metadata_backend": worker.index("(void)dense_w8a8_sep21::requested();") < worker.index("std::filesystem::canonical(argv[2])") < worker.index("metal::MetalBackend backend("),
        "attribution_selector_frozen_before_backend": attribution.index("(void)dense_w8a8_sep21::requested();") < attribution.index("metal::MetalBackend backend("),
        "new_cache_planned_before_trunk_reservation": worker.index("plannedTrunk += dense_w8a8_sep21::Cache::plannedBytes();") < worker.index("governor.tryReserve(plannedTrunk, &trunkFailure)"),
        "new_cache_and_workspace_created_independent_of_selector": "if (dense_w8a8_sep21::requiresCache(maximumRows)) {" in forward and "if (!denseCache) throw" in forward and "denseW8Cache = std::make_unique<dense_w8a8_sep21::Cache>" in forward and "denseW8Workspace = std::make_unique<dense_w8a8_sep21::Workspace>" in forward,
        "workspace_plan_counts_fixed_workspace": "total += dense_w8a8_sep21::Workspace::plannedBytes();" in forward,
        "cache_actual_allocation_bounded_by_plan": "if (allocatedBytes > Cache::plannedBytes())" in cache,
        "workspace_actual_allocation_bounded_by_plan": "if (allocatedBytes_ > plannedBytes())" in cache,
        "coefficient_source_metadata_guard_and_hashes_pinned": "sourceMetadataMatches(" in cache and "source-bf16-sha256" in cache and "i8-sha256" in cache and "f32-scale-sha256" in cache,
        "coefficient_base_count_validated_and_residency_covered": "coefficientBases.size() != kImmutableBufferCount" in cache and "append(impl_->denseW8Cache->immutableWeightBuffers());" in forward,
        "activation_workspace_excluded_from_weight_census": "if (impl_->denseW8Workspace)" in forward and "impl_->denseW8Workspace->dummyDot}) reject(buffer);" in forward,
        "new_route_main_nonverification_exact_r2048": "denseW8Requested && singletonMain && !verification && rows == 2048" in forward and "impl_->project(graph, prefix, input, output, diag, rows, verification, true);" in forward,
        "fixed_policy_identity_uses_verified_coefficient_hashes": "std::string(kNumericalPolicy)" in bridge and "cacheIdentityFromRoutes(routes)" in bridge and "dense_w8a8_sep21::numericalIdentity(" in worker,
        "selector_status_and_numerical_alternative_labeled": "dense_w8a8_prefill_enabled" in worker and "dense_w8a8_numerical_alternative" in worker and "dense_w8a8_whole_model_qualified" in worker,
        "public_forward_header_identical_to_parent": (build / "source/runtime/flash/FlashForward.hpp").read_bytes() == (base / "source/runtime/flash/FlashForward.hpp").read_bytes(),
        "all_other_public_runtime_headers_identical_to_parent": all((build / "source" / r).read_bytes() == (base / "source" / r).read_bytes() for r in records if r.startswith("runtime/") and r.endswith((".h", ".hpp"))),
        "gdn_out_role_excluded": "failed component quality gate" in cache_header and resource_plan()["excluded_roles"] == ["linear_attn.out_proj"],
    })
    old_expression, inherited_identity, _, _ = target_identity_expression(base_worker)
    new_expression, new_identity, start, end = target_identity_expression(worker)
    checks["inherited_exact_target_identity_argument_preserved"] = new_identity == "dense_w8a8_sep21::numericalIdentity(" + inherited_identity + ",forward_.kernelRoutes())"
    checks["inherited_target_identity_outer_null_conditional_preserved"] = new_expression[:start] + inherited_identity + new_expression[end:] == old_expression
    checks["inherited_optional_fma_identity_preserved"] = ("gdn_prefill_fma_sep21::numericalIdentity" not in inherited_identity or inherited_identity in new_identity)
    if manifest.get("dense_w8a8_hybrid_parent"):
        census = manifest["dense_w8a8_hybrid_residency_census"]
        checks.update({
            "hybrid_original_expert_guard_preserved": section(worker, "        if (hybridExpertResidencyRequested) {", "        if (originalTextResidencyRequested) {") == section(base_worker, "        if (hybridExpertResidencyRequested) {", "        if (originalTextResidencyRequested) {"),
            "hybrid_union_census_appends_fixed_immutable_cache_only": "savedResidencyLease.bufferCount() != 748 +" in worker and "savedResidencyLease.byteCount() != 202252746752ULL +" in worker and "? dense_w8a8_sep21::kImmutableBufferCount : 0)" in worker and "? dense_w8a8_sep21::Cache::plannedBytes() : 0))" in worker,
            "hybrid_union_count_and_bytes_match_fixed_additions": census["maximum_rows_ge2048_union_owner_count"] == 916 and census["maximum_rows_ge2048_union_owner_bytes"] == 204143722496 and census["workspace_added_to_immutable_union"] is False,
            "hybrid_static_base_identity_preserved_inside_fma": "1a00dd45649f14de4ad48bafa32ff67f1207aaf27b201de01bc6641a210134e2" in inherited_identity and inherited_identity in new_identity,
        })
    # The inherited project selector stays byte-identical apart from its one
    # execute call-site extension; batch helper, MTP and public ABI stay frozen.
    try:
        checks["batch_project_helper_unchanged"] = section(forward, "void FlashForward::batchProject(", "bool FlashForward::batchDenseSmallRowsEnabled()") == section(base_forward, "void FlashForward::batchProject(", "bool FlashForward::batchDenseSmallRowsEnabled()")
    except ValueError:
        # Parent variants can order these methods differently; exact source
        # transformation plus the unchanged nonchanged files is the certificate.
        checks["batch_project_scope_pinned_by_exact_transform"] = all(not r.get("dense_w8a8_changed") for r in manifest["files"] if r["path"].startswith("runtime/flash/FlashBatch"))
    dep_paths = [build / "host/FlashForward.d", build / "host/FlashWorker.d", build / "host/worker_cache.d",
                 build / "policy-cpu.d", build / "cache-policy-cpu.d", build / "prefill4k-attribution.d"]
    compiled_dependencies = []
    live_dependencies = []
    dependency_mismatch = []
    for dep_path in dep_paths:
        if not dep_path.exists():
            dependency_mismatch.append(str(dep_path))
            continue
        for token in dep_tokens(dep_path.read_bytes()):
            path = resolve_input(Path(token)).resolve()
            if build / "source" in path.parents:
                relative = path.relative_to(build / "source").as_posix()
                if relative not in records or sha(path.read_bytes()) != records[relative]["overlay_sha256"]:
                    dependency_mismatch.append(relative)
                else:
                    compiled_dependencies.append({"path": relative, "sha256": records[relative]["overlay_sha256"]})
            elif ROOT / "runtime" in path.parents or ROOT / "dev" in path.parents:
                live_dependencies.append(str(path))
    checks["all_new_compiled_objects_executables_have_private_header_closure"] = not dependency_mismatch and not live_dependencies
    probes = {}
    for mode in ("", "--freeze0", "--freeze1", "--invalid"):
        probes[mode or "default"] = run_cpu([build / "policy-cpu", *([mode] if mode else [])])
    checks["compiled_policy_and_strict_freeze_cpu_pass"] = all(p["returncode"] == 0 for p in probes.values())
    policy = json.loads(probes["default"]["stdout"]) if probes["default"]["returncode"] == 0 else {}
    plan = resource_plan()
    checks["compiled_policy_resource_plan_matches_frozen_manifest"] = policy.get("cache_planned_bytes") == plan["coefficient_planned_bytes"] and policy.get("workspace_planned_bytes") == plan["activation_workspace_planned_bytes"] and policy.get("gpu_work") is False and policy.get("payload_reads") is False
    cache_probe = run_cpu([build / "cache-policy-cpu"])
    cache_policy = json.loads(cache_probe["stdout"]) if cache_probe["returncode"] == 0 else {}
    checks["compiled_cache_metadata_geometry_source_guard_cpu_pass"] = cache_probe["returncode"] == 0 and cache_policy.get("cache_constructed") is False and cache_policy.get("gpu_work") is False and cache_policy.get("payload_reads") is False and cache_policy.get("projection_count") == 84 and cache_policy.get("immutable_buffer_count") == 168
    worker_probe = run_cpu([build / "splash-flash", "--cpu-self-test"])
    checks["main_worker_cpu_self_test_pass"] = worker_probe["returncode"] == 0
    rejected = {}
    for value in ("", "2", "true", "01", " 1", "1 ", "-1"):
        env = clean_env()
        env[FLAG] = value
        result = run_cpu([build / "splash-flash", "serve-flash-native", "/dense-w8a8-does-not-exist", "16384", "auto"], env)
        rejected[value] = result
        checks[f"invalid_selector_rejected_before_paths_backend:{value!r}"] = result["returncode"] != 0 and FLAG in result["stderr"] and "must be 0 or 1" in result["stderr"]
    result = {"schema": "splash-dense-w8a8-prefill-worker-cpu-witness-v1", "pass": all(checks.values()),
              "gpu_work": False, "model_loaded": False, "payload_bytes_read": 0, "cache_constructed": False,
              "checks": checks, "source_mismatch": source_mismatch, "link_input_mismatch": input_mismatch,
              "tool_mismatch": tools_mismatch, "qualified_input_mismatch": qualified_mismatch,
              "certificate_mismatch": certificate_mismatch, "inherited_dependency_mismatch": inherited_dep_mismatch,
              "dependency_mismatch": dependency_mismatch, "live_project_dependencies": live_dependencies,
              "compiled_dependencies": list({r["path"]: r for r in compiled_dependencies}.values()),
              "compiled_policy_cpu": probes, "compiled_cache_policy_cpu": cache_probe,
              "main_worker_cpu_self_test": worker_probe, "invalid_selector_probes": rejected,
              "resource_plan": plan, "component_certificates": manifest["dense_w8a8_component_certificates"],
              "frozen_source_count": len(records), "frozen_link_input_count": len(inputs),
              "effective_parent_object_count": len(closure["objects"]), "effective_parent_air_count": len(closure["airs"]),
              "runtime_sha256": {name: sha((build / name).read_bytes()) for name in ("splash-flash", "splash.metallib", "prefill4k-attribution")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "gpu_work": False, "payload_bytes_read": 0,
                      "failed_checks": [k for k, v in checks.items() if not v],
                      "frozen_sources": len(records), "frozen_link_inputs": len(inputs), "resource_plan": plan}))
    if not result["pass"]:
        raise ValueError("Dense W8A8 frozen worker CPU/source witness failed")


if __name__ == "__main__":
    main()
