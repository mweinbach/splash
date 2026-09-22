#!/usr/bin/env python3
"""Device-free sealed source, policy and fail-before-backend witness."""
from pathlib import Path
import argparse
import json
import os
import subprocess
from worker_overlay import ROOT, PRIVATE, QUALIFIED_KERNEL_SHA256, sha, transform


def extract(text, begin, end):
    start = text.index(begin)
    return text[start:text.index(end, start)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/moe-pointwise-sep21-worker-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh witness output")
    build = args.build.resolve()
    manifest = json.loads((build / "overlay-manifest.json").read_text())
    base = Path(manifest["pointwise_base_build"])
    assert sha((base / "overlay-manifest.json").read_bytes()) == manifest["pointwise_base_manifest_sha256"]
    mismatches = []
    for record in manifest["files"]:
        relative = record["path"]
        if record.get("new_pointwise_file"):
            source = (ROOT / relative).read_bytes()
            assert sha(source) == record["pointwise_repository_sha256"]
        else:
            source = (base / "source" / relative).read_bytes()
            assert sha(source) == record["pointwise_base_overlay_sha256"]
        expected = transform(relative, source.decode()).encode()
        actual = (build / "source" / relative).read_bytes()
        if expected != actual or sha(actual) != record["overlay_sha256"]:
            mismatches.append(relative)
    worker = (build / "source/runtime/flash/FlashWorker.mm").read_text()
    forward = (build / "source/runtime/flash/FlashForward.cpp").read_text()
    base_forward = (base / "source/runtime/flash/FlashForward.cpp").read_text()
    store = (build / "source/runtime/flash/FlashInt8ExpertStore.mm").read_text()
    base_store = (base / "source/runtime/flash/FlashInt8ExpertStore.mm").read_text()
    kernel = (build / "source" / PRIVATE / "candidate.metal").read_text()
    input_mismatch = [record["private_path"] for record in manifest["pointwise_link_inputs"]
                      if sha((build / record["private_path"]).read_bytes()) != record["sha256"]]
    checks = {
        "sources_fresh": not mismatches,
        "kernel_identical_to_root_qualified_source": sha(kernel.encode()) == manifest["pointwise_kernel_source_sha256"] == QUALIFIED_KERNEL_SHA256,
        "parse_frozen_before_paths_metadata_and_backend": worker.index("(void)pointwise_sep21::requested();") < worker.index("std::filesystem::canonical(argv[2])") < worker.index("metal::MetalBackend backend("),
        "forward_marker_only_kernel_routes": "std::string(pointwise_sep21::marker(pointwise_sep21::requested()))" in forward,
        "workspace_planner_identical_to_base": extract(forward, "uint64_t FlashForward::workspacePlannedBytes", "std::string FlashForward::kernelRoutes") == extract(base_forward, "uint64_t FlashForward::workspacePlannedBytes", "std::string FlashForward::kernelRoutes"),
        "existing_numerical_derivative_identical_to_base": extract(store, "    std::string derivative =", "    numericalIdentity =") == extract(base_store, "    std::string derivative =", "    numericalIdentity ="),
        "no_additional_gpu_allocation_calls": all((build / "source" / record["path"]).read_text().count("allocateBuffer(") == (base / "source" / record["path"]).read_text().count("allocateBuffer(") for record in manifest["files"] if not record.get("new_pointwise_file")),
        "small_and_bulk_arithmetic_preserves_bf16_reduction": "bfloat partials[8]" in kernel and "slot += 8" in kernel and "routed = routed + partials[y]" in kernel,
        "precise_sigmoid_and_finite_poisoning_preserved": "metal::precise::exp" in kernel and "!metal::isfinite(float(weighted))" in kernel and "pointwise_nan()" in kernel,
        "three_poison_sites_composed": sum((build / "source" / relative).read_text().count("pointwise_sep21::addPoison(graph") for relative in ("runtime/flash/FlashInt8ExpertStore.mm", "runtime/flash/FlashMoEBlocked.cpp", "runtime/flash/FlashExpertDenseCache.cpp")) == 3,
        "additional_planned_and_actual_buffers_zero": manifest["pointwise_added_allocations_bytes"] == 0,
        "frozen_reused_host_core_and_air_inputs_fresh": not input_mismatch,
        "frozen_link_make_list_fresh": sha((build / "link-inputs.mk").read_bytes()) == manifest["pointwise_link_inputs_make_sha256"],
    }
    live_runtime_dependencies = []
    for dep in (build / "host").glob("*.d"):
        first = dep.read_text().replace("\\\n", " ").splitlines()[0]
        for token in first.split(": ", 1)[1].split():
            if token.startswith("runtime/") and token.endswith((".h", ".hpp")):
                live_runtime_dependencies.append(token)
    checks["compiled_changed_hosts_have_no_live_runtime_header_dependencies"] = not live_runtime_dependencies
    probes = {}
    for mode in ([], ["--freeze0"], ["--freeze1"]):
        probes[" ".join(mode) or "default"] = json.loads(subprocess.check_output([str(build / "policy-cpu"), *mode], text=True))
    checks["compiled_policy_and_freeze_cpu_checks_pass"] = all(probe["pass"] and not probe["gpu_work"] for probe in probes.values())
    rejected = {}
    for value in ("", "2", "true", "01", " 1", "1 "):
        env = dict(os.environ)
        env["SPLASH_FLASH_MOE_POINTWISE_SEP21"] = value
        run = subprocess.run([str(build / "splash-flash"), "serve-flash-native", "/pointwise-does-not-exist", "16384", "auto"], env=env, text=True, capture_output=True)
        rejected[value] = {"returncode": run.returncode, "stderr": run.stderr}
        checks[f"invalid_switch_rejected_before_path_{value!r}"] = run.returncode != 0 and "SPLASH_FLASH_MOE_POINTWISE_SEP21 must be0 or1" in run.stderr
    result = {"schema": "splash-moe-pointwise-worker-cpu-witness-v1", "pass": all(checks.values()),
              "gpu_work": False, "model_loaded": False, "payload_bytes_read": 0, "files_checked": len(manifest["files"]),
              "source_mismatch": mismatches, "checks": checks, "compiled_policy_cpu": probes, "invalid_switch_probes": rejected,
              "frozen_link_inputs_checked": len(manifest["pointwise_link_inputs"]), "link_input_mismatch": input_mismatch,
              "live_runtime_header_dependencies": live_runtime_dependencies,
              "kernel_source_sha256": manifest["pointwise_kernel_source_sha256"],
              "base_manifest_sha256": manifest["pointwise_base_manifest_sha256"],
              "runtime_sha256": {name: sha((build / name).read_bytes()) for name in ("splash-flash", "splash.metallib", "prefill4k-attribution")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"], "files_checked": len(manifest["files"]), "gpu_work": False, "checks": checks}))
    if not result["pass"]:
        raise ValueError("Pointwise sealed source/CPU witness failed")


if __name__ == "__main__":
    main()
