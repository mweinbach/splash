#!/usr/bin/env python3
"""Freeze a CPU-only experimental singleton WY worker over pure dense-W8 v3."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/gdn_wy_hc_worker_sep21")
DEFAULT_BASE = ROOT / "build/prefill-hc-inject-norm-sep21-worker-v3"
DEFAULT_BUILD = ROOT / "build/gdn-wy-hc-sep21-worker-v1"
INTEGRATION_PAUSED_REASON = (
    "KNOWN real-input v6 regression: 39/48 heads replayed; native about 1.90 ms "
    "versus WY 3.37 ms. Partial draft is not ready to freeze or compose."
)
REPLACED_NAMES = {"FlashForward", "FlashWorker", "FlashBatchPrefill", "MetalBackend"}
EXPECTED_CHANGED = {
    "runtime/flash/FlashForward.cpp", "runtime/flash/FlashForward.hpp",
    "runtime/flash/FlashWorker.mm", "runtime/flash/FlashBatchPrefill.cpp",
    "dev/benchmarks/prefill4k_attribution.mm",
}
KERNEL_HASHES = {
    "candidate.metal": "aef2719900b561b3b9ed8e45bcbda4b09a2db2d93bcaa6b6bb9ba305b95b161f",
    "snapshot.metal": "35cecd82752d29c72251655eb66c605642d94493299a3920d33c1e79b560dd1d",
    "native_fallback.metal": "fe2eef4e818f6380e005840bc33877590a100422de35067e0e8594471eb98087",
}
KERNEL_PRIVATE = Path("dev/benchmarks/gdn_nax_chunks_sep21_v6")
PRIVATE_NAMES = ("worker_bridge.hpp", "worker_policy_cpu.cpp", "candidate.metal", "snapshot.metal", "native_fallback.metal", "telemetry.hpp", "telemetry.metal")
TOOL_NAMES = ("worker_overlay.py", "worker_witness.py", "worker.mk", "worker_README.md")

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f"WY sealed source anchor drift: {before!r}")
    return text.replace(before, after)

def transform(relative, text):
    if relative not in EXPECTED_CHANGED and relative != "runtime/metal/MetalBackend.mm":
        return text
    include = '#include "dev/benchmarks/gdn_wy_hc_worker_sep21/worker_bridge.hpp"\n'
    if relative in {"runtime/flash/FlashForward.cpp", "runtime/flash/FlashWorker.mm", "dev/benchmarks/prefill4k_attribution.mm"}:
        text = include + text
    if relative == "runtime/flash/FlashForward.hpp":
        text = replace(text, "#pragma once\n", "#pragma once\n\n" + include)
        text = replace(text, "  [[nodiscard]] std::string kernelRoutes() const;", """  [[nodiscard]] std::string kernelRoutes() const;
  [[nodiscard]] bool gdnWYEnabled() const noexcept;
  [[nodiscard]] uint64_t gdnWYArenaPlannedBytes() const noexcept;
  [[nodiscard]] uint64_t gdnWYArenaActualBytes() const noexcept;
  [[nodiscard]] gdn_wy_hc_sep21::Counters gdnWYCounters() const noexcept;""")
    if relative == "runtime/flash/FlashForward.cpp":
        text = replace(text, '  const bool stagedGDN = fusionEnabled("SPLASH_FLASH_GDN_STAGED");',
            '  const bool stagedGDN = fusionEnabled("SPLASH_FLASH_GDN_STAGED");\n  const bool wyGDN = gdn_wy_hc_sep21::requested();')
        text = replace(text, '  std::array<metal::MetalBuffer, kScratchCount> scratch;',
            '  std::array<metal::MetalBuffer, kScratchCount> scratch;\n  std::unique_ptr<gdn_wy_hc_sep21::Workspace> wyGDNWorkspace;')
        text = replace(text, '    const uint64_t before = backend.memoryStats().allocatedBytes;',
            '''    const uint64_t before = backend.memoryStats().allocatedBytes;
    if (gdn_wy_hc_sep21::provisioning(maximumRows, wyGDN)) {
      wyGDNWorkspace = std::make_unique<gdn_wy_hc_sep21::Workspace>(backend);
      if (wyGDNWorkspace->allocatedBytes() != gdn_wy_hc_sep21::plannedBytes(maximumRows, wyGDN))
        throw std::logic_error("private GDN WY workspace ledger mismatch");
    }''')
        text = replace(text, '  if (dense_w8a8_sep21::requiresCache(maximumRows))\n    total += dense_w8a8_sep21::Workspace::plannedBytes();',
            '''  if (dense_w8a8_sep21::requiresCache(maximumRows))
    total += dense_w8a8_sep21::Workspace::plannedBytes();
  total += gdn_wy_hc_sep21::plannedBytes(maximumRows, gdn_wy_hc_sep21::requested());''')
        old = '''      (impl_->stagedGDN
          ? (gdn_prefill_fma_sep21::requested()
              ? std::string(gdn_prefill_fma_sep21::marker(true))
              : std::string(";gdn-prefill-staged-v16-t16"))
          : std::string{}) +'''
        new = '''      (impl_->stagedGDN
          ? (impl_->wyGDNWorkspace
              ? std::string(gdn_wy_hc_sep21::selectionMarker(true))
              : gdn_prefill_fma_sep21::requested()
              ? std::string(gdn_prefill_fma_sep21::marker(true))
              : std::string(";gdn-prefill-staged-v16-t16"))
          : std::string{}) +'''
        text = replace(text, old, new)
        text = replace(text, '  for (const auto &buffer : impl_->scratch) reject(buffer);',
            '''  for (const auto &buffer : impl_->scratch) reject(buffer);
  if (impl_->wyGDNWorkspace)
    for (const auto &buffer : impl_->wyGDNWorkspace->buffers()) reject(buffer);''')
        text = replace(text, '''      } else {
        if (impl_->stagedGDN && rows >= 64)
          addGDNStagedPrefill(graph, weights, buffers, state.gdn[layer], rows, 1,''',
            '''      } else {
        if (impl_->wyGDNWorkspace && gdn_wy_hc_sep21::eligible(rows, 1, impl_->wyGDN, verification))
          impl_->wyGDNWorkspace->addPrefill(graph, weights, buffers, state.gdn[layer], rows,
              static_cast<float>(impl_->descriptor.normEpsilon));
        else if (impl_->stagedGDN && rows >= 64)
          addGDNStagedPrefill(graph, weights, buffers, state.gdn[layer], rows, 1,''')
        text = replace(text, '  metal::CommandGraph graph;\n  struct LazyTrialGuard {',
            '''  const uint64_t wyCallsBefore = impl_->wyGDNWorkspace ? impl_->wyGDNWorkspace->counters().encodedCalls : 0;
  metal::CommandGraph graph;
  struct LazyTrialGuard {''')
        text = replace(text, '  state.length += rows;\n  impl_->capturedRows =',
            '''  if (impl_->wyGDNWorkspace)
    impl_->wyGDNWorkspace->complete(impl_->wyGDNWorkspace->counters().encodedCalls - wyCallsBefore);
  state.length += rows;
  impl_->capturedRows =''')
        text = replace(text, 'std::string FlashForward::kernelRoutes() const {',
            '''bool FlashForward::gdnWYEnabled() const noexcept { return impl_ && impl_->wyGDNWorkspace; }
uint64_t FlashForward::gdnWYArenaPlannedBytes() const noexcept {
  return impl_ ? gdn_wy_hc_sep21::plannedBytes(impl_->maximumRows, impl_->wyGDN) : 0;
}
uint64_t FlashForward::gdnWYArenaActualBytes() const noexcept {
  return impl_ && impl_->wyGDNWorkspace ? impl_->wyGDNWorkspace->allocatedBytes() : 0;
}
gdn_wy_hc_sep21::Counters FlashForward::gdnWYCounters() const noexcept {
  return impl_ && impl_->wyGDNWorkspace ? impl_->wyGDNWorkspace->counters() : gdn_wy_hc_sep21::Counters{};
}
std::string FlashForward::kernelRoutes() const {''')
    if relative == "runtime/flash/FlashBatchPrefill.cpp":
        # Only source-marker acceptance changes. The batch prefill recurrence
        # helper and lane/ILP selection remain exactly inherited FMA behavior.
        text = replace(text, '''            route(gdn_prefill_fma_sep21::marker(true))) ||''',
            '''            route(gdn_prefill_fma_sep21::marker(true)) ||
            route(gdn_wy_hc_sep21::selectionMarker(true))) ||''')
    if relative == "runtime/flash/FlashWorker.mm":
        text = replace(text, '      << R"(,"memory_pressure":)" << json::quote(pressureName(governor.pressure))',
            '      << R"(,"gdn_prefill_wy_counters":)" << gdn_wy_hc_sep21::countersJSON(forward_.gdnWYCounters())\n      << R"(,"memory_pressure":)" << json::quote(pressureName(governor.pressure))')
        text = replace(text, '      (void)gdn_prefill_fma_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.',
            '''      (void)gdn_prefill_fma_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.
      (void)gdn_wy_hc_sep21::requested(); // Freeze strict singleton WY selector before paths/metadata/backend/model.''')
        old = '''      << R"(,"target_numerical_derivative_sha256":)" << (persistedExperts ? json::quote(dense_w8a8_sep21::numericalIdentity(
          gdn_prefill_fma_sep21::numericalIdentity(persistedExperts->numericalIdentitySha256(),
              gdn_prefill_fma_sep21::requested()),forward_.kernelRoutes())) : "null")'''
        new = '''      << R"(,"target_numerical_derivative_sha256":)" << (persistedExperts ? json::quote(gdn_wy_hc_sep21::numericalIdentity(
          dense_w8a8_sep21::numericalIdentity(
              gdn_prefill_fma_sep21::numericalIdentity(persistedExperts->numericalIdentitySha256(),
                  gdn_prefill_fma_sep21::requested()),forward_.kernelRoutes()), forward_.gdnWYEnabled())) : "null")'''
        text = replace(text, old, new)
        text = replace(text, '      << R"(,"gdn_prefill_fma_enabled":)" << (gdn_prefill_fma_sep21::requested() ? "true" : "false")',
            '''      << R"(,"gdn_prefill_wy_requested":)" << (gdn_wy_hc_sep21::requested() ? "true" : "false")
      << R"(,"gdn_prefill_wy_effective_singleton":)" << (forward_.gdnWYEnabled() ? "true" : "false")
      << R"(,"gdn_prefill_wy_whole_model_qualified":false)"
      << R"(,"gdn_prefill_wy_gpu_replay_qualification":"Root v6 standalone resources/debug/equivalence/guard-policy passed; whole-worker qualification pending","gdn_prefill_wy_strict_f64_accuracy_proven":false)"
      << R"(,"gdn_prefill_wy_numerical_policy":)" << json::quote(gdn_wy_hc_sep21::kPolicy)
      << R"(,"gdn_prefill_wy_candidate_sha256":)" << json::quote(gdn_wy_hc_sep21::kCandidateSourceSHA256)
      << R"(,"gdn_prefill_wy_native_fallback_sha256":)" << json::quote(gdn_wy_hc_sep21::kNativeFallbackSourceSHA256)
      << R"(,"gdn_prefill_wy_snapshot_sha256":)" << json::quote(gdn_wy_hc_sep21::kSnapshotSourceSHA256)
      << R"(,"gdn_prefill_wy_guard_proof_sha256":)" << json::quote(gdn_wy_hc_sep21::kGuardProofSHA256)
      << R"(,"gdn_prefill_wy_workspace_identity_sha256":)" << json::quote(gdn_wy_hc_sep21::workspaceIdentity())
      << R"(,"gdn_prefill_wy_scope":)" << json::quote(gdn_wy_hc_sep21::kScope)
      << R"(,"gdn_prefill_wy_fixed_rows":2048,"gdn_prefill_wy_fixed_lanes":1)"
      << R"(,"gdn_prefill_wy_coefficients_logical_bytes":)" << (forward_.gdnWYEnabled() ? gdn_wy_hc_sep21::kCoefficientsBytes : 0)
      << R"(,"gdn_prefill_wy_snapshot_logical_bytes":)" << (forward_.gdnWYEnabled() ? gdn_wy_hc_sep21::kSnapshotBytes : 0)
      << R"(,"gdn_prefill_wy_flags_logical_bytes":)" << (forward_.gdnWYEnabled() ? gdn_wy_hc_sep21::kFlagsBytes : 0)
      << R"(,"gdn_prefill_wy_arena_planned_bytes":)" << forward_.gdnWYArenaPlannedBytes()
      << R"(,"gdn_prefill_wy_arena_actual_bytes":)" << forward_.gdnWYArenaActualBytes()
      << R"(,"gdn_prefill_fma_effective_in_wy_singleton_window":)" << (gdn_prefill_fma_sep21::requested() && !forward_.gdnWYEnabled() ? "true" : "false")
      << R"(,"gdn_prefill_fma_enabled":)" << (gdn_prefill_fma_sep21::requested() ? "true" : "false")''')
    if relative == "dev/benchmarks/prefill4k_attribution.mm":
        text = replace(text, '      << R"(,"memory_pressure":)" << json::quote(pressureName(governor.pressure))',
            '      << R"(,"gdn_prefill_wy_counters":)" << gdn_wy_hc_sep21::countersJSON(forward_.gdnWYCounters())\n      << R"(,"memory_pressure":)" << json::quote(pressureName(governor.pressure))')
        text = replace(text, '      (void)gdn_prefill_fma_sep21::requested(); // Freeze numerical prefill policy before backend creation.',
            '''      (void)gdn_prefill_fma_sep21::requested(); // Freeze numerical prefill policy before backend creation.
      (void)gdn_wy_hc_sep21::requested(); // Freeze strict WY prefill flag before backend/model creation.''')
    if relative == "runtime/metal/MetalBackend.mm":
        text = replace(text, '''            [encoder dispatchThreadgroups:item.groups
                     threadsPerThreadgroup:item.threads];''',
            '''            [encoder dispatchThreadgroups:item.groups
                     threadsPerThreadgroup:item.threads];
            // Private singleton WY producer/consumer ordering. No dispatch ABI
            // or inherited pipeline changes; five exact private names only.
            if (dispatch.pipelineName == "private_gdn_wy_snapshot" ||
                dispatch.pipelineName == "private_gdn_wy_prepare_t32_sg8" ||
                dispatch.pipelineName == "private_gdn_wy_v32_t32_sg8" ||
                dispatch.pipelineName == "private_gdn_wy_restore" ||
                dispatch.pipelineName == "private_gdn_wy_native_fallback" ||
                dispatch.pipelineName == "private_gdn_wy_telemetry_eligible" ||
                dispatch.pipelineName == "private_gdn_wy_telemetry_replay")
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];''')
    return text

def helper(base):
    path = base / "machinery/parent_overlay.py"
    spec = importlib.util.spec_from_file_location("frozen_dense_overlay_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def resolve_input(path):
    return path if path.is_absolute() else ROOT / path

def dep_tokens(data):
    import shlex
    return shlex.split(data.decode().replace("\\\n", " ").splitlines()[0].split(":", 1)[1])

def effective_closure(base, base_make):
    return helper(base).effective_closure(base, base_make)

def main():
    raise RuntimeError(INTEGRATION_PAUSED_REASON)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--source-proof", type=Path, default=ROOT / "build/gdn-nax-chunks-sep21-v6/replay-source-proof-v2.json")
    parser.add_argument("--refresh-unsealed", action="store_true", help="Refresh only this generator's private output before a passing worker witness exists")
    args = parser.parse_args()
    base, output, proof_path = (p.resolve() for p in (args.base, args.output, args.source_proof))
    witness_path = output / "cpu-witness-v1.json"
    prior_pass = witness_path.exists() and json.loads(witness_path.read_text()).get("pass") is True
    refreshable = args.refresh_unsealed and not prior_pass
    if refreshable and (output / "overlay-manifest.json").exists():
        existing = json.loads((output / "overlay-manifest.json").read_text())
        if existing.get("route") != "private-singleton-guarded-wy-v6-over-hc-dense-w8-sg2tail-fma-sep21-v1" or existing.get("gpu_executed") is not False:
            raise ValueError("Only this generator's unsealed private output can be refreshed")
    if base == output or output != DEFAULT_BUILD or ((output / "overlay-manifest.json").exists() and not refreshable):
        raise ValueError("WY requires the fresh named private experimental build directory")
    parent_path = base / "overlay-manifest.json"
    parent = json.loads(parent_path.read_text())
    if parent.get("dense_w8a8_hybrid_parent") is not False or not parent.get("prefill_hc_composed"):
        raise ValueError("WY requires sealed pure dense-W8 v3, not hybrid parent")
    dense = helper(base)
    base_make = base / "machinery/worker.mk"
    closure = dense.effective_closure(base, base_make)
    input_seals, unsealed = dense.parent_input_seals(base, parent, closure)
    parent_witness = dense.parent_artifact_witness(base)
    records = {r["path"]: r for r in parent["files"]}
    if len(records) != len(parent["files"]):
        raise ValueError("Duplicate frozen parent source records")
    deps = dense.frozen_dependency_closure(base, closure["objects"], records)
    proof_bytes = proof_path.read_bytes()
    proof = json.loads(proof_bytes)
    if proof.get("pass") is not True:
        raise ValueError("Accepted v6 CPU native-replay source proof is required")
    if proof["arena"]["rounded_physical_bytes"] != 167919616 or proof["policy"]["gpu_executed"]:
        raise ValueError("Unexpected WY physical/GPU source-proof scope")
    native_reference_path = ROOT / KERNEL_PRIVATE / "frozen/canonical.metal"
    native_reference_bytes = native_reference_path.read_bytes()
    native_helper = native_reference_bytes[:native_reference_bytes.index(b"#define GDS_ENTRY")]
    if sha(native_helper) != proof["native_helper_sha256"]:
        raise ValueError("Root accepted literal native helper reference drift")
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-singleton-guarded-wy-v6-over-hc-dense-w8-sg2tail-fma-sep21-v1",
        "gdn_wy_base_build": str(base), "gdn_wy_base_manifest_sha256": sha(parent_path.read_bytes()),
        "gdn_wy_base_make_path": str(base_make), "gdn_wy_base_make_sha256": sha(base_make.read_bytes()),
        "gdn_wy_effective_parent_objects": [str(p) for p in closure["objects"]],
        "gdn_wy_effective_parent_airs": [str(p) for p in closure["airs"]],
        "gdn_wy_parent_input_seals": input_seals, "gdn_wy_parent_owned_unsealed_inputs": unsealed,
        "gdn_wy_parent_artifact_witness": parent_witness, "gdn_wy_parent_dependencies": deps,
        "gdn_wy_guard_proof": {"source_path": str(proof_path), "private_path": "qualification/replay-source-proof-v1.json", "sha256": sha(proof_bytes)},
        "gdn_wy_native_reference": {"source_path": str(native_reference_path), "private_path": "qualification/native-reference.metal", "sha256": sha(native_reference_bytes)},
        "gdn_wy_resource_plan": {"rows": 2048, "lanes": 1, **proof["arena"]},
        "gdn_wy_replaced_names": sorted(REPLACED_NAMES), "gdn_wy_kernel_sources": [],
        "gdn_wy_link_inputs": [], "gdn_wy_tools": [], "gdn_wy_protected_regions": [], "files": [],
        "gdn_wy_scope": "main singleton nonverification R64..2048 only; batch and all speculative/state paths inherited",
        "gdn_wy_whole_model_qualified": False, "gdn_wy_gpu_replay_equivalence_proven": True,
        "gdn_wy_strict_f64_accuracy_proven": False, "normal_sources_modified": False,
        "gpu_executed": False, "payload_bytes_read": 0})
    changed = []
    texts = {}
    for relative, record in records.items():
        original = (base / "source" / relative).read_bytes()
        if sha(original) != record["overlay_sha256"]:
            raise ValueError(f"Frozen parent source drift: {relative}")
        data = transform(relative, original.decode()).encode()
        if data != original:
            changed.append(relative)
        write(output / "source" / relative, data)
        manifest["files"].append({**record, "gdn_wy_changed": data != original,
            "base_overlay_sha256": sha(original), "overlay_sha256": sha(data)})
        texts[relative] = (original.decode(), data.decode())
    if set(changed) != EXPECTED_CHANGED:
        raise ValueError(f"Unexpected WY changed inherited files: {changed}")
    backend_record = next(r for r in parent["dense_w8a8_qualified_inputs"]
        if r["category"] == "sources" and r["path"] == "runtime/metal/MetalBackend.mm")
    backend_path = base / backend_record["private_path"]
    if not backend_path.is_file():
        backend_path = Path(parent["prefill_hc_base_build"]) / backend_record["private_path"]
    backend_bytes = backend_path.read_bytes()
    if sha(backend_bytes) != backend_record["sha256"]:
        raise ValueError("Captured parent qualified backend source drift")
    write(output / "qualification/backend-reference.mm", backend_bytes)
    manifest["gdn_wy_backend_reference"] = {"source_path": str(backend_path),
        "private_path": "qualification/backend-reference.mm", "sha256": sha(backend_bytes)}
    backend_data = transform("runtime/metal/MetalBackend.mm", backend_bytes.decode()).encode()
    write(output / "source/runtime/metal/MetalBackend.mm", backend_data)
    manifest["files"].append({"path": "runtime/metal/MetalBackend.mm", "new_gdn_wy_file": True,
        "gdn_wy_changed": True, "base_private_path": backend_record["private_path"],
        "base_overlay_sha256": sha(backend_bytes), "overlay_sha256": sha(backend_data)})
    for relative in ("runtime/flash/FlashGDNStaged.cpp", "runtime/flash/FlashGDNStaged.hpp",
        "runtime/flash/FlashBatchForward.cpp", "runtime/flash/FlashBatchVerify.cpp",
        "runtime/flash/FlashBatchVerifyGDN.cpp", "runtime/flash/FlashBatchMTPForward.cpp",
        "runtime/flash/FlashGDNLazyRollback.cpp"):
        before, after = texts[relative]
        manifest["gdn_wy_protected_regions"].append({"path": relative, "begin": None, "end": None,
            "base_sha256": sha(before.encode()), "overlay_sha256": sha(after.encode())})
    forward = "runtime/flash/FlashForward.cpp"
    for begin, end in (("FlashForwardResult FlashForward::verify(", "FlashForwardResult FlashForward::forwardImpl("),
        ("      if (verification && rows > 1) {\n        if (impl_->lazyGDN) {", "\n      } else {\n"),
        ("metal::CommandTiming FlashForward::commitVerify(", "void FlashForward::abortVerify("),
        ("void FlashForward::abortVerify(", "} // namespace splash::flash")):
        def section(text):
            start = text.index(begin)
            return text[start:text.index(end, start) if end else len(text)]
        before, after = [section(t) for t in texts[forward]]
        if before != after:
            raise ValueError(f"Protected speculative path changed: {begin}")
        manifest["gdn_wy_protected_regions"].append({"path": forward, "begin": begin, "end": end,
            "base_sha256": sha(before.encode()), "overlay_sha256": sha(after.encode())})
    for name in PRIVATE_NAMES:
        relative = PRIVATE / name
        original_private = KERNEL_PRIVATE / name if name in KERNEL_HASHES else relative
        data = (ROOT / original_private).read_bytes()
        if name in KERNEL_HASHES and sha(data) != KERNEL_HASHES[name]:
            raise ValueError(f"Root-ready WY kernel source drift: {name}")
        write(output / "source" / relative, data)
        manifest["files"].append({"path": relative.as_posix(), "new_gdn_wy_file": True,
            "repository_path": original_private.as_posix(), "repository_sha256": sha(data), "overlay_sha256": sha(data)})
        if name in KERNEL_HASHES:
            manifest["gdn_wy_kernel_sources"].append({"path": relative.as_posix(), "sha256": sha(data)})
    generated = '#pragma once\nnamespace splash::flash::gdn_wy_hc_sep21 {\n'
    for symbol, value in (("kCandidateSourceSHA256", KERNEL_HASHES["candidate.metal"]),
        ("kNativeFallbackSourceSHA256", KERNEL_HASHES["native_fallback.metal"]),
        ("kSnapshotSourceSHA256", KERNEL_HASHES["snapshot.metal"]), ("kGuardProofSHA256", sha(proof_bytes)),
        ("kTelemetrySourceSHA256", sha((ROOT / PRIVATE / "telemetry.metal").read_bytes())),
        ("kTelemetryABISHA256", sha((ROOT / PRIVATE / "telemetry.hpp").read_bytes()))):
        generated += f'inline constexpr const char *{symbol} = "{value}";\n'
    generated += '} // namespace splash::flash::gdn_wy_hc_sep21\n'
    generated_path = PRIVATE / "worker_source_hashes.hpp"
    write(output / "source" / generated_path, generated.encode())
    manifest["files"].append({"path": generated_path.as_posix(), "new_gdn_wy_file": True,
        "generated_gdn_wy_hashes": True, "overlay_sha256": sha(generated.encode())})
    manifest["gdn_wy_generated_hashes"] = generated_path.as_posix()
    abi = (ROOT / KERNEL_PRIVATE / "frozen/metal/abi/FlashGDN.h").read_bytes()
    if sha(abi) != records["runtime/metal/abi/FlashGDN.h"]["overlay_sha256"]:
        raise ValueError("WY captured ABI differs from inherited worker ABI")
    inputs = {"REUSED": [], "CORE": [], "AIRS": []}
    def freeze(path, category):
        relative = Path("reused/base") / path.relative_to(base)
        data = path.read_bytes()
        write(output / relative, data)
        inputs[category].append(relative.as_posix())
        manifest["gdn_wy_link_inputs"].append({"source_path": str(path), "private_path": relative.as_posix(),
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
    manifest["gdn_wy_link_make_sha256"] = sha(make.encode())
    manifest["gdn_wy_changed_files"] = changed
    for name in TOOL_NAMES:
        data = (ROOT / PRIVATE / name).read_bytes()
        relative = Path("machinery") / name
        write(output / relative, data)
        manifest["gdn_wy_tools"].append({"source_path": str(ROOT / PRIVATE / name),
            "private_path": relative.as_posix(), "sha256": sha(data)})
    for path, name in ((parent_path, "base-overlay-manifest.json"), (base_make, "base-worker.mk"),
        (base / "link-inputs.mk", "base-link-inputs.mk"), (base / "cpu-witness-v1.json", "base-cpu-witness-v1.json"),
        (base / "machinery/parent_overlay.py", "base-worker-overlay.py"), (proof_path, "replay-source-proof-v1.json")):
        write(output / "qualification" / name, path.read_bytes())
    for name in ("splash-flash.config", "splash.metallib.config"):
        write(output / name, (base / name).read_bytes().rstrip(b"\n") + b"-private-gdn-wy-v6-singleton-guarded-over-hc-prefill-sep21-v1\n")
    write(output / "base-build.txt", (str(base) + "\n").encode())
    write(output / "qualification/native-reference.metal", native_reference_bytes)
    write(output / "overlay-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({"prepared": str(output), "changed_sources": changed, "parent_objects": len(closure["objects"]),
        "parent_airs": len(closure["airs"]), "arena_physical_bytes": 167919616, "gpu_executed": False, "payload_bytes_read": 0}))

if __name__ == "__main__":
    main()
