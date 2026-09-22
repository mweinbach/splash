#!/usr/bin/env python3
"""Prove sealed v2 teacher, batch shader closure and independent pointwise/FMA policies."""
from pathlib import Path
import argparse
import json
import os
import subprocess
from batch_worker_prepare import ROOT, PRIVATE, FMA_PRIVATE, FMA_KERNEL_SHA, QUALIFIED_KERNEL_SHA256, CHANGED_NAMES, sha, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/prefill4k-batch-pointwise-fma-sep21-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh source witness path")
    build = args.build.resolve()
    manifest = json.loads((build / "overlay-manifest.json").read_text())
    base = Path(manifest["batch_pointwise_fma_base_build"])
    fma_base = Path(manifest["batch_pointwise_fma_qualified_fma_build"])
    audit = json.loads((base / "host-build-audit.json").read_text())
    errors = []
    checks = {
        "base_manifest_and_authenticated_object_audit_fresh": sha((base / "overlay-manifest.json").read_bytes()) == manifest["batch_pointwise_fma_base_manifest_sha256"] and sha((base / "host-build-audit.json").read_bytes()) == manifest["batch_pointwise_fma_base_audit_sha256"],
        "qualified_fma_manifest_fresh": sha((fma_base / "overlay-manifest.json").read_bytes()) == manifest["batch_pointwise_fma_fma_manifest_sha256"],
        "prepare_and_both_transformers_fresh": sha((ROOT / PRIVATE / "batch_worker_prepare.py").read_bytes()) == manifest["batch_pointwise_fma_prepare_sha256"] and sha((ROOT / PRIVATE / "worker_overlay.py").read_bytes()) == manifest["batch_pointwise_fma_pointwise_transform_sha256"] and sha((ROOT / FMA_PRIVATE / "worker_overlay.py").read_bytes()) == manifest["batch_pointwise_fma_fma_transform_sha256"],
    }
    for record in manifest["files"]:
        relative = record["path"]
        if "combined_support_source" in record:
            expected = Path(record["combined_support_source"]).read_bytes()
        else:
            original = (base / "source" / relative).read_bytes()
            if sha(original) != record["combined_base_sha256"]:
                errors.append("parent source " + relative)
            expected = transform(relative, original.decode()).encode()
        actual = (build / "source" / relative).read_bytes()
        if expected != actual or sha(actual) != record["overlay_sha256"]:
            errors.append("output source " + relative)
    source_paths = {record["path"] for record in manifest["files"]}
    checks["all_sources_authenticated_with_exact_scoped_transforms"] = not errors
    checks["source_tree_contains_only_manifested_sources"] = source_paths == {str(path.relative_to(build / "source")) for path in (build / "source").rglob("*") if path.is_file()}
    checks["pointwise_and_fma_kernels_match_independent_qualified_pins"] = sha((build / "source" / PRIVATE / "candidate.metal").read_bytes()) == QUALIFIED_KERNEL_SHA256 and sha((build / "source" / FMA_PRIVATE / "scalar_fma.metal").read_bytes()) == FMA_KERNEL_SHA
    input_errors = [record["private_path"] for record in manifest["combined_link_inputs"] if sha((build / record["private_path"]).read_bytes()) != record["sha256"]]
    checks["all_frozen_link_inputs_fresh"] = not input_errors
    checks["sealed_link_list_digest_fresh"] = sha((build / "link-inputs.mk").read_bytes()) == manifest["combined_link_make_sha256"]
    checks["original68_air_control_reconstructs_qualified_v2_metallib_bytes"] = sha((build / "original-control.metallib").read_bytes()) == manifest["reconstructed_control_metallib_sha256"] == audit["metallib_sha256"]
    fma_artifact_witness = json.loads((fma_base / "source-witness.json").read_text())
    checks["imported_fma_air_bound_to_authenticated_qualified_library_bytes"] = sha((fma_base / "source-witness.json").read_bytes()) == manifest["qualified_fma_artifact_witness_sha256"] and sha((build / "qualified-fma-control.metallib").read_bytes()) == manifest["qualified_fma_reconstructed_metallib_sha256"] == fma_artifact_witness["artifact_sha256"]["splash.metallib"]
    original_airs = manifest["original_batch_air_inventory"]
    checks["batch_m128_dense_prefill_replacement_retained_core_counterpart_excluded"] = len(original_airs) == 68 and any("prefill4k-batch-bulk-gathered-sep21-v1/metal/flash_dense_cache_prefill.air" in path for path in original_airs) and "build/flash-next/metal/shared/flash_dense_cache_prefill.air" not in original_airs and "build/flash-next/metal/shared/flash_qsa_bulk.air" not in original_airs
    protected_names = {"FlashMTP", "FlashBatchMTPForward", "FlashBatchForward", "FlashBatchVerify", "FlashBatchVerifyGDN", "FlashGDNLazyRollback"}
    for name in protected_names:
        record = next(record for record in manifest["combined_link_inputs"] if Path(record["private_path"]).name == name + ".o")
        original_record = next(record for record in audit["inputs"] if Path(record["output"]).name == name + ".o")
        checks["correct_v2_original_object_retained_" + name] = record["source_path"] == original_record["output"] and record["sha256"] == original_record["sha256"]
        checks["protected_execution_source_identical_" + name] = (build / "source/runtime/flash" / (name + ".cpp")).read_bytes() == (base / "source/runtime/flash" / (name + ".cpp")).read_bytes()
    for name in ("FlashMTP", "FlashBatchMTPForward"):
        actual = (build / "source/runtime/flash" / (name + ".hpp")).read_bytes()
        checks["correct_teacher_api_header_retained_" + name] = actual == (base / "source/runtime/flash" / (name + ".hpp")).read_bytes() and b"primeTeacherCache" in actual
    worker = (build / "source/runtime/flash/FlashWorker.mm").read_text()
    checks["both_flags_frozen_before_paths_and_backend"] = worker.index("(void)gdn_prefill_fma_sep21::requested();") < worker.index("std::filesystem::canonical(argv[2])") and worker.index("(void)pointwise_sep21::requested();") < worker.index("std::filesystem::canonical(argv[2])") < worker.index("metal::MetalBackend backend(")
    checks["teacher_worker_cache_prime_api_and_accounting_retained"] = "batchPrimeHead_->primeTeacherCache(headStates" in worker and "mtp_batch_teacher_priming_route" in worker
    checks["no_new_gpu_allocation_calls"] = all((build / "source" / record["path"]).read_text().count("allocateBuffer(") == (base / "source" / record["path"]).read_text().count("allocateBuffer(") for record in manifest["files"] if "combined_base_sha256" in record)
    checks["only_expected_eight_host_modules_plus_attribution_changed"] = len(manifest["combined_changed_files"]) == 9
    checks["sg2_tail_not_added"] = manifest["no_sg2_tail_added"] and not any("sg2" in record["private_path"] for record in manifest["combined_link_inputs"])
    live_deps = []
    for dep in (build / "host").glob("*.d"):
        first = dep.read_text().replace("\\\n", " ").splitlines()[0]
        live_deps += [token for token in first.split(": ", 1)[1].split() if token.startswith("runtime/") and token.endswith((".h", ".hpp"))]
    checks["compiled_changed_hosts_resolve_only_sealed_runtime_headers"] = not live_deps
    probes = {}
    for name, modes in (("pointwise-policy-cpu", ([], ["--freeze0"], ["--freeze1"])), ("fma-policy-cpu", ([], ["--freeze0"], ["--freeze1"])), ("teacher-policy-cpu", ([],))):
        for mode in modes:
            result = json.loads(subprocess.check_output([str(build / name), *mode], text=True))
            probes[name + " " + " ".join(mode)] = result
    checks["compiled_policy_and_freeze_checks_pass"] = all(result.get("pass", result.get("valid", False)) for result in probes.values())
    native = json.loads(subprocess.check_output([str(build / "splash-flash"), "--cpu-self-test"], text=True))
    checks["native_worker_cpu_selftest_pass"] = native["valid"] and not native["gpu_work"]
    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("SPLASH_FLASH_")}
    clean_env.update({"SPLASH_FLASH_GDN_STAGED": "1", "SPLASH_FLASH_ALLROWS_FULL512_TARGET": "1", "SPLASH_FLASH_BATCH_QSA_BULK_PREFILL": "1", "SPLASH_FLASH_BATCH_PREFILL": "1", "SPLASH_FLASH_BATCH_MTP_PREFILL": "1", "SPLASH_FLASH_MTP": "1", "SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY": "1", "SPLASH_FLASH_QSA_BULK_PREFILL": "1", "SPLASH_FLASH_QSA_BULK_PREFILL_SG8": "1", "SPLASH_FLASH_QSA_F32": "1", "SPLASH_FLASH_QSA_MPP": "1", "SPLASH_FLASH_QSA_ROW_TILES": "1"})
    planners = {}
    for pointwise in ("0", "1"):
        for fma in ("0", "1"):
            env = {**clean_env, "SPLASH_FLASH_MOE_POINTWISE_SEP21": pointwise, "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21": fma}
            key = "pointwise" + pointwise + "_fma" + fma
            plan = json.loads(subprocess.check_output([str(build / "memory-cpu"), "16384", "2048"], env=env, text=True))
            planners[key] = {name: value for name, value in plan.items() if "planned_bytes" in name or "workspace_planned_bytes" in name or "request_state" in name}
            run = subprocess.run([str(build / "splash-flash"), "serve-flash-native", "/batch-pointwise-no-such-path", "16384", "auto"], env=env, text=True, capture_output=True)
            checks["independent_flags_parse_before_path_" + key] = run.returncode != 0 and "canonical" in run.stderr and "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21" not in run.stderr and "SPLASH_FLASH_MOE_POINTWISE_SEP21" not in run.stderr
    checks["all_four_flag_combinations_have_identical_cpu_memory_plans"] = all(plan == next(iter(planners.values())) for plan in planners.values())
    malformed = {}
    for flag in ("SPLASH_FLASH_MOE_POINTWISE_SEP21", "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21"):
        for value in ("", "2", "true", "01", " 1", "1 "):
            env = {**clean_env, "SPLASH_FLASH_MOE_POINTWISE_SEP21": "0", "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21": "0", flag: value}
            run = subprocess.run([str(build / "splash-flash"), "serve-flash-native", "/batch-pointwise-no-such-path", "16384", "auto"], env=env, text=True, capture_output=True)
            malformed[flag + "=" + repr(value)] = run.stderr
            checks["malformed_rejected_before_path_" + flag + "=" + repr(value)] = run.returncode != 0 and flag in run.stderr and "canonical" not in run.stderr
    result = {"schema": "splash-v2-batch-pointwise-fma-source-cpu-witness-v1", "pass": all(checks.values()),
              "source_integrity_pass": all(checks.values()), "combined_actual_model_fidelity_pass": None,
              "gpu_work": False, "model_loaded": False, "payload_bytes_read": 0, "files_checked": len(manifest["files"]),
              "frozen_inputs_checked": len(manifest["combined_link_inputs"]), "original_batch_airs": 68,
              "checks": checks, "source_errors": errors, "input_errors": input_errors, "live_runtime_header_dependencies": live_deps,
              "compiled_policy_cpu": probes, "native_cpu_selftest": native, "four_flag_planners": planners, "malformed_flag_probes": malformed,
              "artifact_sha256": {name: sha((build / name).read_bytes()) for name in ("splash-flash", "splash.metallib", "prefill4k-attribution", "batch-teacher-oracle", "original-control.metallib")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"], "files_checked": result["files_checked"], "frozen_inputs": result["frozen_inputs_checked"], "failed_checks": [name for name, ok in checks.items() if not ok], "gpu_work": False}))
    if not result["pass"]:
        raise ValueError("Combined v2 batch pointwise/FMA CPU source witness failed")


if __name__ == "__main__":
    main()
