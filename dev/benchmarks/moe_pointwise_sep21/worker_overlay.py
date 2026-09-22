#!/usr/bin/env python3
"""Freeze exact pointwise dispatch over the qualified private gathered+bulk source."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/moe_pointwise_sep21")
QUALIFIED_KERNEL_SHA256 = "6651a9e9537b04cb337f38529078f42a9300af9f9f9d9b82b679e1955b765f23"
CHANGED_NAMES = {"FlashMoE", "FlashMoEBlocked", "FlashInt8ExpertStore", "FlashExpertDenseCache", "FlashForward", "FlashWorker"}
EXCLUDED_AIRS = {"flash_gdn", "flash_gdn_fused", "flash_gdn_staged", "flash_int8_expert_store", "flash_qsa_bulk"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f"Pointwise sealed-source anchor drift: {before!r}")
    return text.replace(before, after)


def transform(relative, text):
    include = '#include "dev/benchmarks/moe_pointwise_sep21/bridge.hpp"\n'
    if relative in ("runtime/flash/FlashMoE.cpp", "runtime/flash/FlashMoEBlocked.cpp",
                    "runtime/flash/FlashInt8ExpertStore.mm", "runtime/flash/FlashExpertDenseCache.cpp",
                    "runtime/flash/FlashForward.cpp", "runtime/flash/FlashWorker.mm"):
        text = include + text
    if relative == "runtime/flash/FlashMoE.cpp":
        text = replace(text, '  graph.add("flash_moe_combine",', '''  const auto selectedPointwise = pointwise_sep21::combineRoute(
      rows, width, experts, selections, pointwise_sep21::requested());
  const char *pointwisePipeline = selectedPointwise == pointwise_sep21::CombineRoute::WholeRow
      ? "private_moe_combine_cta_row" : selectedPointwise == pointwise_sep21::CombineRoute::SimdSlots
      ? "private_moe_combine_simd_slots" : "flash_moe_combine";
  graph.add(pointwisePipeline,''')
        text = replace(text, '            pointwiseGroups(rows, width, 1), {kPointwiseThreads, 1, 1});',
                       '''            selectedPointwise == pointwise_sep21::CombineRoute::WholeRow
                ? metal::DispatchSize{1, rows, 1} : pointwiseGroups(rows, width, 1),
            {kPointwiseThreads, 1, 1});''')
    if relative == "runtime/flash/FlashInt8ExpertStore.mm":
        text = replace(text, '''  graph.add("flash_moe_blocked_poison_excluded_routes",
      {s.buckets.canonicalToPacked, s.scatteredDown, diagnostics}, poison,
      {10, routes, 1}, {256, 1, 1});''', '''  pointwise_sep21::addPoison(graph,
      {s.buckets.canonicalToPacked, s.scatteredDown, diagnostics}, poison);''')
    if relative == "runtime/flash/FlashMoEBlocked.cpp":
        text = replace(text, '''  graph.add("flash_moe_blocked_poison_excluded_routes",
            {scratch.buckets.canonicalToPacked, scratch.scatteredDown, diagnostics},
            p, {10, p.route_capacity, 1}, {256, 1, 1});''', '''  pointwise_sep21::addPoison(graph,
            {scratch.buckets.canonicalToPacked, scratch.scatteredDown, diagnostics}, p);''')
    if relative == "runtime/flash/FlashExpertDenseCache.cpp":
        text = replace(text, '''  graph.add("flash_moe_blocked_poison_excluded_routes", {s.buckets.canonicalToPacked,
      s.scatteredDown, diagnostics}, params.blocked, {10, params.blocked.route_capacity, 1});''',
                       '''  pointwise_sep21::addPoison(graph, {s.buckets.canonicalToPacked,
      s.scatteredDown, diagnostics}, params.blocked);''')
    if relative == "runtime/flash/FlashForward.cpp":
        text = replace(text, '  return std::string(flashAffineSemantics()) +',
                       '''  return std::string(flashAffineSemantics()) +
      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +''')
    if relative == "runtime/flash/FlashWorker.mm":
        text = replace(text, '''      std::signal(SIGPIPE, SIG_IGN);''', '''      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.
      std::signal(SIGPIPE, SIG_IGN);''')
    if relative == "dev/benchmarks/prefill4k_attribution.mm":
        text = include + text
        text = replace(text, '    if (name == "flash_moe_combine") { insideMoE = shared = false; return "moe"; }',
                       '    if (name == "flash_moe_combine" || name.starts_with("private_moe_combine_")) { insideMoE = shared = false; return "moe"; }')
        text = replace(text, '      metal::MetalBackend backend(argv[1]);',
                       '''      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.
      metal::MetalBackend backend(argv[1]);''')
    return text


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "build/moe-pointwise-sep21-worker-v1")
    parser.add_argument("--corebase", type=Path, default=ROOT / "build/flash-next")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if output == base or ROOT / "build" not in output.parents:
        raise ValueError("Pointwise requires a distinct private build directory")
    parent_path = base / "overlay-manifest.json"
    parent = json.loads(parent_path.read_text())
    if parent.get("omitted_original_target_tensor_count") != 432 or not parent.get("gathered_mpp_composed") or not parent.get("qsa_bulk_composed"):
        raise ValueError("Expected sealed private Full512 gathered-MPP plus exact bulk SG8 source")
    kernel_sha = sha((ROOT / PRIVATE / "candidate.metal").read_bytes())
    if kernel_sha != QUALIFIED_KERNEL_SHA256:
        raise ValueError("Pointwise candidate differs from the root GPU qualified kernel source")
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-full512-bulk-gathered-exact-pointwise-sep21-v1",
                     "pointwise_composed": True, "pointwise_required_environment": "SPLASH_FLASH_MOE_POINTWISE_SEP21=0|1",
                     "pointwise_flag0_original_graphs": True, "pointwise_added_allocations_bytes": 0,
                     "pointwise_changes_numerical_derivative": False,
                     "pointwise_combine_scope": "canonical W2560/E512/K10; SIMDslots physical rows1..16; whole-rowCTA physical rows256..8192; others original",
                     "pointwise_poison_scope": "physical rows256..8192 only; rows1..255 original",
                     "pointwise_exact_qualification": "root GPU synthetic 110combine/26poison cases; zero BF16/diagnostic/input/canary byte differences; actual-model qualification remains pending",
                     "pointwise_base_build": str(base), "pointwise_base_manifest_sha256": sha(parent_path.read_bytes()),
                     "pointwise_transform_sha256": sha(Path(__file__).read_bytes()),
                     "gpu_executed": False, "payload_bytes_read": 0, "files": []})
    changed = []
    for record in parent["files"]:
        relative = record["path"]
        original = (base / "source" / relative).read_bytes()
        if sha(original) != record["overlay_sha256"]:
            raise ValueError(f"Sealed base source drift: {relative}")
        data = transform(relative, original.decode()).encode()
        write(output / "source" / relative, data)
        if data != original:
            changed.append(relative)
        manifest["files"].append({**record, "pointwise_changed": data != original,
                                  "pointwise_base_overlay_sha256": sha(original), "overlay_sha256": sha(data)})
    if len(changed) != 6:
        raise ValueError(f"Expected six host-only route changes, got {changed}")
    extras = [PRIVATE / name for name in ("bridge.hpp", "candidate.metal", "policy_cpu.cpp")]
    extras.append(Path("dev/benchmarks/prefill4k_attribution.mm"))
    for relative in extras:
        original = (ROOT / relative).read_bytes()
        data = transform(str(relative), original.decode()).encode()
        write(output / "source" / relative, data)
        manifest["files"].append({"path": str(relative), "new_pointwise_file": True,
                                  "pointwise_repository_sha256": sha(original), "overlay_sha256": sha(data)})
    # Preserve inherited private headers, then freeze every other runtime header
    # so transitive Metal/ABI/engine includes cannot fall back to the live tree.
    already = {record["path"] for record in manifest["files"]}
    for path in sorted((ROOT / "runtime").rglob("*")):
        relative = path.relative_to(ROOT)
        if path.suffix not in (".h", ".hpp") or str(relative) in already:
            continue
        data = path.read_bytes()
        write(output / "source" / relative, data)
        manifest["files"].append({"path": str(relative), "new_pointwise_file": True,
                                  "sealed_transitive_runtime_header": True,
                                  "pointwise_repository_sha256": sha(data), "overlay_sha256": sha(data)})
    corebase = args.corebase.resolve()
    manifest["pointwise_link_inputs"] = []
    inputs = {"REUSED": [], "CORE": [], "AIRS": []}
    def freeze_input(path, relative, category):
        data = path.read_bytes()
        write(output / relative, data)
        inputs[category].append(relative.as_posix())
        manifest["pointwise_link_inputs"].append({"source_path": str(path), "private_path": relative.as_posix(),
                                                 "category": category, "sha256": sha(data)})
    for path in sorted((base / "host").glob("*.o")):
        if path.stem not in CHANGED_NAMES:
            freeze_input(path, Path("reused/base-host") / path.name, "REUSED")
    for relative in ("engine/metal/MetalBackend.o", "engine/metal/DeviceCapabilities.o",
                     "engine/engine/Protocol.o", "engine/engine/MemoryGovernor.o"):
        freeze_input(corebase / relative, Path("reused/core") / relative, "CORE")
    for path in sorted((corebase / "metal").glob("*/*.air")):
        if path.parent.name == "shared" and path.stem in EXCLUDED_AIRS:
            continue
        freeze_input(path, Path("reused/core") / path.relative_to(corebase), "AIRS")
    for path in sorted((base / "metal").glob("*.air")):
        freeze_input(path, Path("reused/base-metal") / path.name, "AIRS")
    frozen_make = "\n".join(f"{key} := " + " ".join("$(BUILD)/" + path for path in paths)
                            for key, paths in inputs.items()) + "\n"
    write(output / "link-inputs.mk", frozen_make.encode())
    manifest["pointwise_link_inputs_make_sha256"] = sha(frozen_make.encode())
    manifest["pointwise_kernel_source_sha256"] = kernel_sha
    manifest["pointwise_changed_files"] = changed
    write(output / "overlay-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    write(output / "base-build.txt", (str(base) + "\n").encode())
    # Config markers identify this private linked artifact without changing numerical identity.
    for name in ("splash-flash.config", "splash.metallib.config"):
        write(output / name, (base / name).read_bytes().rstrip(b"\n") + b"-pointwise-exact-sep21-v1\n")
    print(json.dumps({"prepared": str(output), "files_checked": len(parent["files"]), "changed_files": changed,
                      "frozen_link_inputs": len(manifest["pointwise_link_inputs"]), "frozen_sources": len(manifest["files"]),
                      "added_allocation_bytes": 0, "gpu_executed": False, "payload_bytes_read": 0}))


if __name__ == "__main__":
    main()
