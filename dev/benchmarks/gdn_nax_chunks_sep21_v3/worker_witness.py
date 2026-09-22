#!/usr/bin/env python3
"""CPU/source/artifact closure witness. Runs only the independent CPU policy.

Never invokes splash-flash, an attribution/oracle binary, a Metal device, or a
model/payload loader. Hashing linked objects/AIR/metallib is artifact inspection.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/gdn_nax_chunks_sep21_v3")
DEFAULT_BUILD = ROOT / "build/gdn-wy-dense-prefill-sep21-worker-v1"
FLAG = "SPLASH_FLASH_GDN_PREFILL_WY_SEP21"
STAGED_FLAG = "SPLASH_FLASH_GDN_STAGED"
EXPECTED_CHANGED = {"runtime/flash/FlashForward.cpp", "runtime/flash/FlashForward.hpp",
                    "runtime/flash/FlashWorker.mm", "runtime/flash/FlashBatchPrefill.cpp",
                    "dev/benchmarks/prefill4k_attribution.mm"}
REPLACED_NAMES = {"FlashForward", "FlashWorker", "FlashBatchPrefill", "MetalBackend"}
KERNELS = {"candidate.metal", "native_fallback.metal", "snapshot.metal"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    result = (root / path).resolve()
    if root.resolve() not in result.parents:
        raise ValueError(f"Manifest path escapes frozen root: {relative}")
    return result


def section(text, begin, end):
    if begin is None and end is None:
        return text
    if not begin or not end or text.count(begin) != 1:
        raise ValueError("Protected region requires one unambiguous begin anchor")
    start = text.index(begin)
    stop = text.index(end, start + len(begin))
    return text[start:stop]


def resolve_input(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def dep_tokens(data):
    line = data.decode().replace("\\\n", " ").splitlines()[0]
    if ":" not in line:
        raise ValueError("Malformed dependency metadata")
    return shlex.split(line.split(":", 1)[1])


def effective_closure(build, make_path):
    # -q -n prints expanded prerequisites without executing recipes.
    result = subprocess.run(["make", "-pqn", "-rR", "-f", str(make_path), f"BUILD={build}",
                             str(build / "splash-flash"), str(build / "splash.metallib")],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    if result.returncode not in (0, 1):
        raise ValueError(f"make prerequisite inspection failed: {result.stderr}")
    def inputs(name, suffix):
        lines = [line for line in result.stdout.splitlines() if line.startswith(str(build / name) + ":")]
        if len(lines) != 1:
            raise ValueError(f"Ambiguous actual prerequisite list: {name}")
        paths = [resolve_input(t) for t in shlex.split(lines[0].split(":", 1)[1])]
        if not paths or len(paths) != len(set(paths)) or any(p.suffix != suffix or not p.is_file() for p in paths):
            raise ValueError(f"Missing, duplicate or unexpected actual link input: {name}")
        return paths
    return {"objects": inputs("splash-flash", ".o"), "airs": inputs("splash.metallib", ".air")}


def run_policy(path, mode):
    env = dict(os.environ)
    env.update({FLAG: "0", STAGED_FLAG: "1", "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21": "0"})
    result = subprocess.run([str(path), *([mode] if mode else [])], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to replace an existing CPU witness")
    build = args.build.resolve()
    manifest_path = build / "overlay-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    base = Path(manifest["gdn_wy_base_build"]).resolve()
    parent = json.loads((base / "overlay-manifest.json").read_text())
    base_make = Path(manifest["gdn_wy_base_make_path"])
    checks, errors, details = {}, {}, {}

    def check(name, fn):
        try:
            checks[name] = bool(fn())
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
            checks[name] = False
            errors[name] = f"{type(error).__name__}: {error}"

    check("parent_manifest_fresh", lambda: sha((base / "overlay-manifest.json").read_bytes()) == manifest["gdn_wy_base_manifest_sha256"])
    check("parent_manifest_snapshot_fresh", lambda: sha((build / "qualification/base-overlay-manifest.json").read_bytes()) == manifest["gdn_wy_base_manifest_sha256"])
    check("parent_make_fresh", lambda: sha(base_make.read_bytes()) == manifest["gdn_wy_base_make_sha256"])
    check("parent_make_snapshot_fresh", lambda: sha((build / "qualification/base-worker.mk").read_bytes()) == manifest["gdn_wy_base_make_sha256"])
    check("private_link_make_fresh", lambda: sha((build / "link-inputs.mk").read_bytes()) == manifest["gdn_wy_link_make_sha256"])
    check("pure_dense_parent", lambda: parent["dense_w8a8_hybrid_parent"] is False)
    check("freezer_gpu_and_payload_work_zero", lambda: manifest["gpu_executed"] is False and manifest["payload_bytes_read"] == 0)
    check("whole_model_qualification_not_claimed", lambda: manifest["gdn_wy_whole_model_qualified"] is False)

    tools = manifest["gdn_wy_tools"]
    tools_by_name = {Path(r["private_path"]).name: r for r in tools}
    check("all_archived_worker_machinery_fresh", lambda: len(tools_by_name) == len(tools) and all(sha(safe_path(build, r["private_path"]).read_bytes()) == r["sha256"] for r in tools))
    check("executed_witness_matches_archived_machinery", lambda: sha(Path(__file__).read_bytes()) == tools_by_name["worker_witness.py"]["sha256"])
    overlay = importlib.import_module("worker_overlay")
    check("imported_generator_matches_archived_machinery", lambda: sha(Path(overlay.__file__).read_bytes()) == tools_by_name["worker_overlay.py"]["sha256"])

    records = {r["path"]: r for r in manifest["files"]}
    parent_records = {r["path"]: r for r in parent["files"]}
    check("source_paths_unique_and_all_parent_files_retained", lambda: len(records) == len(manifest["files"]) and len(parent_records) == len(parent["files"]) and set(parent_records) <= set(records))
    source_mismatch, changed, added = [], [], []
    for relative, record in records.items():
        try:
            actual = safe_path(build / "source", relative).read_bytes()
            if sha(actual) != record["overlay_sha256"]:
                raise ValueError("Frozen source digest differs")
            if relative in parent_records:
                original = safe_path(base / "source", relative).read_bytes()
                expected_parent = parent_records[relative]["overlay_sha256"]
                if sha(original) != expected_parent or record["base_overlay_sha256"] != expected_parent:
                    raise ValueError("Inherited source hash differs from pure dense parent")
                if actual != original:
                    changed.append(relative)
                if actual != overlay.transform(relative, original.decode()).encode():
                    raise ValueError("Frozen source differs from exact authorized transform")
            else:
                added.append(relative)
                if relative != "runtime/metal/MetalBackend.mm" and Path(relative).parent != PRIVATE:
                    raise ValueError("New source outside private WY/kernel/backend scope")
                allowed_private = KERNELS | {"worker_bridge.hpp", "worker_policy_cpu.cpp", "worker_source_hashes.hpp"}
                if relative != "runtime/metal/MetalBackend.mm" and Path(relative).name not in allowed_private:
                    raise ValueError("Unapproved extra private source")
                if "repository_sha256" in record and sha(safe_path(ROOT, relative).read_bytes()) != record["repository_sha256"]:
                    raise ValueError("New repository source has drifted")
        except (OSError, ValueError, KeyError, UnicodeError) as error:
            source_mismatch.append(relative)
            errors[f"source:{relative}"] = f"{type(error).__name__}: {error}"
    details.update({"source_mismatch": source_mismatch, "changed_files": sorted(changed), "added_files": sorted(added)})
    check("all_parent_and_frozen_sources_fresh_and_exact_transforms", lambda: not source_mismatch)
    check("only_five_expected_inherited_sources_changed", lambda: set(changed) == EXPECTED_CHANGED and set(manifest["gdn_wy_changed_files"]) == EXPECTED_CHANGED)
    backend_reference = manifest["gdn_wy_backend_reference"]
    check("captured_backend_reference_fresh_and_adapter_exact", lambda:
          sha(Path(backend_reference["source_path"]).read_bytes()) == backend_reference["sha256"] ==
          sha(safe_path(build, backend_reference["private_path"]).read_bytes()) and
          (build / "source/runtime/metal/MetalBackend.mm").read_bytes() ==
          overlay.transform("runtime/metal/MetalBackend.mm", safe_path(build, backend_reference["private_path"]).read_text()).encode())
    protected = manifest["gdn_wy_protected_regions"]
    protected_mismatch = []
    for record in protected:
        try:
            old = section(safe_path(base / "source", record["path"]).read_text(), record.get("begin"), record.get("end"))
            new = section(safe_path(build / "source", record["path"]).read_text(), record.get("begin"), record.get("end"))
            if old != new or sha(old.encode()) != record["base_sha256"] or sha(new.encode()) != record["overlay_sha256"]:
                raise ValueError("Protected source section changed")
        except (OSError, ValueError, KeyError) as error:
            protected_mismatch.append(record["path"])
            errors[f"protected:{record['path']}:{record.get('begin')}"] = str(error)
    check("protected_verification_rollback_replay_and_fma_regions_unchanged", lambda: bool(protected) and not protected_mismatch)
    details["protected_region_mismatch"] = protected_mismatch

    kernel_sources = manifest["gdn_wy_kernel_sources"]
    check("three_original_v3_kernel_sources_exactly_pinned", lambda: {Path(r["path"]).name for r in kernel_sources} == KERNELS and len(kernel_sources) == 3 and all(sha(safe_path(build / "source", r["path"]).read_bytes()) == r["sha256"] == overlay.KERNEL_HASHES[Path(r["path"]).name] == sha(safe_path(ROOT, r["path"]).read_bytes()) for r in kernel_sources))
    bridge = safe_path(build / "source", PRIVATE / "worker_bridge.hpp").read_text()
    worker = (build / "source/runtime/flash/FlashWorker.mm").read_text()
    forward = (build / "source/runtime/flash/FlashForward.cpp").read_text()
    backend = (build / "source/runtime/metal/MetalBackend.mm").read_text()
    native = safe_path(build / "source", PRIVATE / "native_fallback.metal").read_bytes()
    canonical = (base / "source/runtime/metal/kernels/shared/flash_gdn_staged.metal").read_bytes()
    base_helper = canonical[:canonical.index(b"#define GDS_ENTRY")]
    native_reference = manifest["gdn_wy_native_reference"]
    reference = safe_path(build, native_reference["private_path"]).read_bytes()
    helper = reference[:reference.index(b"#define GDS_ENTRY")]
    check("accepted_frozen_native_reference_archived_and_fresh", lambda:
          sha(reference) == native_reference["sha256"] == sha(Path(native_reference["source_path"]).read_bytes()))
    check("native_reference_only_shrinks_row_guard_with_exact_recurrence_preserved", lambda:
          base_helper.count(b"p.rows <= 8192") == 1 and
          base_helper.replace(b"p.rows <= 8192", b"p.rows <= 2048", 1) == helper)
    check("native_fallback_uses_literal_original_nonfma_helper", lambda: native.startswith(helper) and b"gds_recurrence<16,16>" in native and b"fma(" not in helper and b"fp contract(off)" in helper and b"fp reassociate(off)" in helper)
    check("strict_selector_frozen_before_paths_backend_model", lambda: worker.index("(void)gdn_wy_sep21::requested();") < worker.index("std::filesystem::canonical(argv[2])") < worker.index("metal::MetalBackend backend("))
    check("physical_workspace_planned_before_governor_reserve", lambda: "gdn_wy_sep21::plannedBytes(" in forward and worker.index("FlashForward::workspacePlannedBytes(") < worker.index("governor.tryReserve(plannedTrunk"))
    check("singleton_main_dispatch_excludes_every_verification_window", lambda: "gdn_wy_sep21::eligible(rows,1,impl_->wyGDN,verification)" in re.sub(r"\s+", "", forward))
    check("identity_static_inputs_bound_and_counters_excluded", lambda: all(token in section(bridge, "inline std::string numericalIdentity(", "struct Counters") for token in ("kPolicy", "kMarker", "kCandidateSourceSHA256", "kNativeFallbackSourceSHA256", "kSnapshotSourceSHA256", "kGuardProofSHA256")) and all(token not in section(bridge, "inline std::string numericalIdentity(", "struct Counters") for token in ("encodedCalls", "encodedRows", "completedCalls", "lastLayerFlaggedHeads", "lastLayerGuardReasons", "contents()")))
    graph_names = ["flash_gdn_fused_prepare", "private_gdn_wy_snapshot", "private_gdn_wy_prepare_t32_sg8", "private_gdn_wy_v32_t32_sg8", "private_gdn_wy_restore", "private_gdn_wy_native_fallback", "flash_gdn_output", "flash_gdn_convolution_carry"]
    check("guarded_graph_dispatch_order_preserves_prepare_output_carry", lambda: [bridge.index('graph.add("' + name + '"') for name in graph_names] == sorted(bridge.index('graph.add("' + name + '"') for name in graph_names))
    check("normal_backend_explicit_wy_buffer_hazard_barriers", lambda: "MTLBarrierScopeBuffers" in backend and all(name in backend for name in graph_names[1:6]))
    check("worker_wy_status_source_scope_and_qualification_present", lambda: "gdn_wy_sep21::numericalIdentity(" in worker and "gdn_prefill_wy_whole_model_qualified" in worker and "gdn_wy_sep21::kScope" in worker)

    closure = effective_closure(base, base_make)
    check("actual_parent_object_closure_matches_frozen_metadata", lambda: [str(p) for p in closure["objects"]] == manifest["gdn_wy_effective_parent_objects"])
    check("actual_parent_air_closure_matches_frozen_metadata", lambda: [str(p) for p in closure["airs"]] == manifest["gdn_wy_effective_parent_airs"])
    inputs = manifest["gdn_wy_link_inputs"]
    reused = [r for r in inputs if r["category"] in ("REUSED", "CORE")]
    inherited_airs = [r for r in inputs if r["category"] == "AIRS"]
    check("every_actual_parent_object_retained_except_four_rebuilds", lambda: len(reused) == len({r["source_path"] for r in reused}) and {r["source_path"] for r in reused} == {str(p) for p in closure["objects"] if p.stem not in REPLACED_NAMES})
    check("every_actual_parent_air_retained_exactly_once", lambda: len(inherited_airs) == len(closure["airs"]) and {r["source_path"] for r in inherited_airs} == {str(p) for p in closure["airs"]})
    check("inherited_original_gdn_staged_object_retained", lambda: any(Path(r["source_path"]).name == "FlashGDNStaged.o" for r in reused))
    check("all_frozen_parent_object_air_inputs_fresh", lambda: all(sha(safe_path(build, r["private_path"]).read_bytes()) == r["sha256"] == sha(Path(r["source_path"]).read_bytes()) for r in inputs))
    check("matching_parent_link_input_seals_authenticated", lambda: all(sha(Path(path).read_bytes()) == record["sha256"] for path, record in manifest["gdn_wy_parent_input_seals"].items()))
    parent_witness = manifest["gdn_wy_parent_artifact_witness"]
    check("parent_published_artifacts_and_witness_fresh", lambda: sha((build / "qualification/base-cpu-witness-v1.json").read_bytes()) == parent_witness["sha256"] and all(sha((base / name).read_bytes()) == digest for name, digest in parent_witness["artifacts"].items()))
    parent_dep_mismatch = []
    for record in manifest["gdn_wy_parent_dependencies"]:
        if not record["dependency_metadata_present"]:
            continue
        try:
            if sha(safe_path(build, record["private_path"]).read_bytes()) != record["sha256"] or sha(Path(record["source_path"]).read_bytes()) != record["sha256"]:
                raise ValueError("Inherited .d has drifted")
            for dep in record["dependencies"]:
                if sha(Path(dep["source_path"]).read_bytes()) != dep.get("sha256", dep.get("external_source_sha256")):
                    raise ValueError("Inherited compiler dependency has drifted")
        except (OSError, ValueError, KeyError) as error:
            parent_dep_mismatch.append(record["source_path"])
            errors[f"parent-dependency:{record['source_path']}"] = str(error)
    check("parent_compiler_dependency_and_header_closure_fresh", lambda: not parent_dep_mismatch)

    dep_paths = [build / "host" / (name + ".d") for name in sorted(REPLACED_NAMES)] + [build / "policy-cpu.d", build / "prefill4k-attribution.d"]
    dependency_mismatch, live_dependencies, compiled_dependencies = [], [], []
    for dep_path in dep_paths:
        try:
            for token in dep_tokens(dep_path.read_bytes()):
                path = resolve_input(token)
                if build / "source" in path.parents:
                    relative = path.relative_to(build / "source").as_posix()
                    if relative not in records or sha(path.read_bytes()) != records[relative]["overlay_sha256"]:
                        dependency_mismatch.append(relative)
                    else:
                        compiled_dependencies.append({"path": relative, "sha256": records[relative]["overlay_sha256"]})
                elif ROOT / "runtime" in path.parents or ROOT / "dev" in path.parents:
                    live_dependencies.append(str(path))
        except (OSError, ValueError) as error:
            dependency_mismatch.append(str(dep_path))
            errors[f"dependency:{dep_path}"] = str(error)
    check("rebuilt_hosts_and_policy_use_private_manifest_header_closure", lambda: not dependency_mismatch and not live_dependencies)
    details.update({"parent_dependency_mismatch": parent_dep_mismatch, "dependency_mismatch": dependency_mismatch,
                    "live_project_dependencies": live_dependencies,
                    "compiled_dependencies": list({r["path"]: r for r in compiled_dependencies}.values())})

    own_make = build / "machinery/worker.mk"
    own_closure = effective_closure(build, own_make)
    expected_host = {safe_path(build, r["private_path"]) for r in reused} | {build / "host" / (name + ".o") for name in REPLACED_NAMES}
    expected_air = {safe_path(build, r["private_path"]) for r in inherited_airs} | {build / (name + ".air") for name in ("candidate", "native-fallback", "snapshot")}
    check("actual_private_host_link_has_exact_frozen_plus_rebuilt_inputs", lambda: set(own_closure["objects"]) == expected_host)
    check("actual_private_metallib_link_has_exact_inherited_plus_three_wy_airs", lambda: set(own_closure["airs"]) == expected_air)
    probes = {mode or "default": run_policy(build / "policy-cpu", mode) for mode in ("", "--freeze0", "--freeze1", "--invalid", "--missing-staged")}
    check("compiled_strict_policy_dependency_freeze_and_geometry_pass", lambda: all(r["returncode"] == 0 for r in probes.values()))
    policy = json.loads(probes["default"]["stdout"]) if probes["default"]["returncode"] == 0 else {}
    check("compiled_physical_resource_plan_and_cpu_only_contract_match", lambda: policy.get("gdn_wy_worker_cpu_policy") == "passed" and policy.get("workspace_planned_bytes") == 167133184 and policy.get("logical_bytes") == 167116992 and policy.get("workspace_rows") == 2048 and policy.get("workspace_lanes") == 1 and policy.get("gpu_work") is False and policy.get("payload_reads") is False and policy.get("worker_invoked") is False)
    hashes_text = safe_path(build / "source", manifest["gdn_wy_generated_hashes"]).read_text()
    pins = dict(re.findall(r'(k\w+SHA256)\s*=\s*"([0-9a-f]{64})"', hashes_text))
    expected_pins = {"kCandidateSourceSHA256": next(r["sha256"] for r in kernel_sources if Path(r["path"]).name == "candidate.metal"),
                     "kNativeFallbackSourceSHA256": next(r["sha256"] for r in kernel_sources if Path(r["path"]).name == "native_fallback.metal"),
                     "kSnapshotSourceSHA256": next(r["sha256"] for r in kernel_sources if Path(r["path"]).name == "snapshot.metal")}
    check("generated_identity_hashes_match_exact_linked_kernel_sources", lambda: all(pins.get(key) == value for key, value in expected_pins.items()) and bool(pins.get("kGuardProofSHA256")))
    guard_proof = manifest["gdn_wy_guard_proof"]
    check("static_guard_proof_hash_authenticates_archived_and_original_certificate", lambda:
          pins.get("kGuardProofSHA256") == guard_proof["sha256"] ==
          sha(Path(guard_proof["source_path"]).read_bytes()) ==
          sha(safe_path(build, guard_proof["private_path"]).read_bytes()))
    proof = json.loads(safe_path(build, guard_proof["private_path"]).read_text())
    check("native_helper_hash_matches_accepted_cpu_source_proof", lambda: sha(helper) == proof["native_helper_sha256"])
    check("fixed_resource_plan_matches_archived_guard_layout", lambda:
          proof.get("pass") is True and manifest["gdn_wy_resource_plan"] == {"rows": 2048, "lanes": 1, **proof["arena"]} and
          proof["arena"] == {"coefficients": 163971072, "snapshot": 3145728, "flags": 192,
                              "logical_bytes": 167116992, "rounded_physical_bytes": 167133184})
    policy_match = re.search(r'kPolicy\s*=\s*"([^"]+)"', bridge)
    marker_match = re.search(r'kMarker\s*=\s*"([^"]+)"', bridge)
    static_values = [policy_match.group(1), marker_match.group(1), *[pins.get(key, "") for key in ("kCandidateSourceSHA256", "kNativeFallbackSourceSHA256", "kSnapshotSourceSHA256", "kGuardProofSHA256")]]
    check("cpu_numerical_identity_independently_recomputed_from_static_inputs", lambda: policy.get("numerical_identity_enabled") == sha(("a" * 64 + "\n" + "\n".join(static_values)).encode()))
    workspace_base = "fixed-shared-arena:R2048:B1:T32:H48:stride53376:coeff0+163971072:snapshot163971072+3145728:flags167116800+192:physical167133184:align16384"
    check("cpu_workspace_identity_independently_recomputed_from_fixed_layout", lambda: policy.get("workspace_identity") == sha((workspace_base + "\n" + "\n".join(static_values)).encode()))
    artifact_names = ("splash-flash", "splash.metallib", "prefill4k-attribution", "policy-cpu")
    artifacts = {name: sha((build / name).read_bytes()) for name in artifact_names}
    result = {"schema": "splash.gdn-wy-private-worker-cpu-witness.v1", "pass": all(checks.values()),
              "gpu_work": False, "model_loaded": False, "payload_bytes_read": 0, "worker_invoked": False,
              "workspace_constructed": False, "checks": checks, "errors": errors, **details,
              "compiled_policy_cpu": probes, "resource_plan": {"rows": 2048, "lanes": 1,
                  "coefficients_bytes": 163971072, "snapshot_bytes": 3145728, "flags_bytes": 192,
                  "logical_bytes": 167116992, "physical_bytes": 167133184},
              "frozen_source_count": len(records), "frozen_link_input_count": len(inputs),
              "runtime_sha256": artifacts, "overlay_manifest_sha256": sha(manifest_path.read_bytes())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"], "failed_checks": [name for name, passed in checks.items() if not passed],
                      "gpu_work": False, "payload_bytes_read": 0, "worker_invoked": False,
                      "output": str(args.output), "source_count": len(records), "physical_bytes": 167133184}))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
