#!/usr/bin/env python3
"""Prepare a sealed private phase-dependent Q4/I8 singleton source experiment.

Source/metadata-only preparation. Compilation, admission, and GPU qualification
are left to Root. Production source, models, and prior builds are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path("dev/benchmarks/hybrid_memory_sep21")
CHANGED = {"FlashMoE", "FlashMoEBlocked", "FlashExpertDenseCache", "FlashInt8ExpertStore", "FlashFloatDenseCache", "FlashDenseCache", "FlashForward", "FlashWorker"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def replace(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"Private hybrid source anchor drift: {before!r}")
    return text.replace(before, after)


def worker_transform(text: str, policy_hash: str, required: list[str]) -> str:
    text = replace(text, '#include "flash/FlashInt8ExpertStore.hpp"',
                   '#include "flash/FlashInt8ExpertStore.hpp"\n#include "flash/FlashInt8ExpertStoreMetadata.hpp"')
    required_source = ", ".join(json.dumps(s) for s in required)
    guard = '''      // Private fixed-R4 hybrid: every dependency rejects before backend/probes.
      if (!flashHybridQ4I8FixedR4Enabled())
        throw std::invalid_argument("private hybrid requires SPLASH_FLASH_HYBRID_Q4_I8_FIXED_R4=1");
      for (const char *name : {REQUIRED_FLAGS})
        if (!environmentSwitch(name))
          throw std::invalid_argument(std::string("private hybrid requires ") + name + "=1");
      if (batchEnabled || batchPrefillEnabled || batchMTPEnabled || batchMTPPrefillEnabled ||
          gpuPrefillCopyEnabled || idleMaintenanceRequested || originalTextResidencyRequested)
        throw std::invalid_argument("private fixed-R4 hybrid forbids true batch routes, idle/original-text residency and batch GPU-copy routes");
      if (environmentSwitch("SPLASH_FLASH_DENSE_SMALL_ROWS"))
        throw std::invalid_argument("private fixed-R4 hybrid requires DENSE_SMALL_ROWS=0 for original selective F32 coefficients");
      if (environmentSwitch("SPLASH_FLASH_ALLROWS_FULL512_TARGET") ||
          environmentSwitch("SPLASH_FLASH_ALLROWS_GATHERED_MPP") ||
          environmentSwitch("SPLASH_FLASH_MTP_ADAPTIVE"))
        throw std::invalid_argument("private hybrid forbids all-row-I8 and adaptive/deeper target routes");
      if (prefillRows != 2048 || singletonMTP.maximumDepth != 3 || capacity > 16384)
        throw std::invalid_argument("private hybrid requires arena2048, fixed depth3 and context <= 16384");
      if (mtpEnabled && !teacherCacheOnlyEnabled)
        throw std::invalid_argument("private hybrid MTP requires original teacher cache-only priming");
      if (std::getenv("SPLASH_FLASH_HOT_EXPERT_PLAN"))
        throw std::invalid_argument("private hybrid forbids a second expert coefficient cache");
      const char *hybridStorePath = std::getenv("SPLASH_FLASH_INT8_EXPERT_STORE");
      const char *hybridDenseStorePath = std::getenv("SPLASH_FLASH_OPERAND_STORE");
      if (!hybridStorePath || !*hybridStorePath || !hybridDenseStorePath || !*hybridDenseStorePath)
        throw std::invalid_argument("private hybrid requires saved certified Full512 and original dense operands");
      const auto hybridStoreMetadata = loadFlashInt8ExpertStoreMetadata(hybridStorePath,
          "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
          "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",
          NormConvention::OnePlusWeight);
      if (hybridStoreMetadata.identitySha256 != "ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1" ||
          hybridStoreMetadata.plannedBytes != 121174228992ULL)
        throw std::invalid_argument("private hybrid requires exact certified Full512 metadata");
      for (const auto &layer : hybridStoreMetadata.layers) {
        if (layer.selectedIDs.size() != 512)
          throw std::invalid_argument("private hybrid requires all512 coefficients for large prefill");
        for (uint32_t expert = 0; expert < 512; ++expert)
          if (layer.selectedIDs[expert] != expert)
            throw std::invalid_argument("private hybrid Full512 rank inventory changed");
      }
'''.replace("REQUIRED_FLAGS", required_source)
    text = replace(text, '      metal::MetalBackend backend((executablePath().parent_path() / "splash.metallib").string());',
                   guard + '      metal::MetalBackend backend((executablePath().parent_path() / "splash.metallib").string());')
    identity = '''      << R"(,"target_numerical_derivative_sha256":)" << json::quote("POLICY_HASH")
      << R"(,"target_phase_policy":"main nonverification rows>=256 Full512-I8; rows<256 and all singleton verification original packed Q4; original selective F32/HC-up and Q8 vocabulary; original trained MTP","target_hybrid_phase":true,"target_all_rows_full512":false,"original_target_gpu_omitted":false,"fixed_r4_cache_policy":true,"true_batch_graph_supported":false,"universal_original_q4_parity_claim":false)"
'''.replace("POLICY_HASH", policy_hash)
    text = replace(text, '      << R"(,"loaded_model_layout_sha256":)" << json::quote(weights_.manifestFingerprint())\n      << R"(,"engine_instance_id":)" << instance_',
                   '      << R"(,"loaded_model_layout_sha256":)" << json::quote(weights_.manifestFingerprint())\n' + identity + '      << R"(,"engine_instance_id":)" << instance_')
    return text


def prepare(base: Path, output: Path, corebase: Path) -> dict:
    if output.exists() or ROOT / "build" not in output.resolve().parents or output.resolve() == base.resolve():
        raise ValueError("Choose a fresh private build output")
    plan = module(ROOT / PRIVATE / "plan.py", "hybrid_memory_plan_module")
    pruning = module(ROOT / PRIVATE / "fixed_r4_overlay.py", "hybrid_fixed_r4_overlay_module")
    bulk = module(ROOT / PRIVATE / "bulk_compose.py", "hybrid_bulk_compose_module")
    point = module(ROOT / "dev/benchmarks/moe_pointwise_sep21/worker_overlay.py", "hybrid_pointwise_transform_module")
    parent_path = base / "overlay-manifest.json"
    parent_bytes = parent_path.read_bytes()
    parent = json.loads(parent_bytes)
    original = {}
    for record in parent["files"]:
        rel = record["path"]
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe parent source path")
        data = (base / "source" / path).read_bytes()
        if digest(data) != record["overlay_sha256"]:
            raise ValueError(f"Sealed raw parent has changed: {rel}")
        original[rel] = data.decode()
    assert "allRowsInt8Target" not in original["runtime/flash/FlashForward.cpp"]
    assert "sourceTensors" in original["runtime/flash/FlashInt8ExpertStore.mm"]
    assert "targetOriginalDiskTensorCount" not in original["runtime/flash/FlashWeights.mm"]
    cpu_plan = plan.build(ROOT, ROOT / "build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1/source")
    profile = json.loads((ROOT / ".splash-local-profile.json").read_text())["environment"]
    overrides = {"SPLASH_FLASH_HYBRID_Q4_I8_FIXED_R4": "1", "SPLASH_FLASH_QSA_BULK_PREFILL": "1",
                 "SPLASH_FLASH_QSA_BULK_PREFILL_SG8": "1", "SPLASH_FLASH_MOE_POINTWISE_SEP21": "1",
                 "SPLASH_FLASH_PREFILL_ROWS": "2048", "SPLASH_FLASH_MTP_DRAFT_DEPTH": "3",
                 "SPLASH_FLASH_BATCH": "0", "SPLASH_FLASH_BATCH_PREFILL": "0", "SPLASH_FLASH_BATCH_MTP": "0",
                 "SPLASH_FLASH_BATCH_MTP_PREFILL": "0", "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE": "0",
                 "SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT": "0", "SPLASH_FLASH_GPU_PREFILL_COPY": "0",
                 "SPLASH_FLASH_GDN_BATCH_ILP": "0", "SPLASH_FLASH_DENSE_SMALL_ROWS": "0",
                 "SPLASH_FLASH_ALLROWS_FULL512_TARGET": "0", "SPLASH_FLASH_ALLROWS_GATHERED_MPP": "0",
                 "SPLASH_FLASH_MTP_ADAPTIVE": "0"}
    profile.update(overrides)
    required = sorted(k for k, v in profile.items() if v == "1" and k not in
                      {"SPLASH_FLASH_MTP", "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY", "SPLASH_FLASH_HYBRID_Q4_I8_FIXED_R4"})
    policy = dict(cpu_plan["proposed_phase_policy"],
                  true_batch_routes=False, cooperatively_scheduled_individual_requests=True,
                  maximum_context=16384, fixed_profile_required_one_flags=required,
                  pointwise_policy="existing exact qualified pointwise kernel, including raw-Q4 validator dispatch copying",
                  main_large_prefill_phase_guard="!verification && rows >= 256")
    policy_hash = digest(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode())
    modified = pruning.transform_sources(dict(original))
    modified, bulk_helpers, bulk_witness = bulk.compose_bulk(modified)
    modified.update(bulk_helpers)
    # Reuse exactly the role restriction qualified on the actual 2K route.
    dense_policy_rel = "runtime/flash/FlashPrefillDenseTiles.hpp"
    modified[dense_policy_rel] = (ROOT / "build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1/source" / dense_policy_rel).read_text()
    dense_rel = "runtime/flash/FlashDenseCache.cpp"
    modified[dense_rel] = replace(modified[dense_rel],
                                 "flashPrefillDenseTilePolicy(rows,uint32_t(weight.shape[0]),uint32_t(weight.shape[1]))",
                                 "flashPrefillDenseTilePolicy(prefix,rows,uint32_t(weight.shape[0]),uint32_t(weight.shape[1]))")
    modified, point_helpers, point_witness = bulk.compose_pointwise(modified)
    modified.update(point_helpers)
    forward_rel = "runtime/flash/FlashForward.cpp"
    modified[forward_rel] = replace(modified[forward_rel],
                                   "    const bool blocked = impl_->blockMoE && rows >= 256;",
                                   "    const bool blocked = impl_->blockMoE && !verification && rows >= 256;")
    modified["runtime/flash/FlashWorker.mm"] = worker_transform(modified["runtime/flash/FlashWorker.mm"], policy_hash, required)
    # Freeze missing ABI/engine/Metal headers and the existing projected dense
    # shader source; inherited parent Flash headers always take precedence.
    for path in sorted((ROOT / "runtime").rglob("*")):
        rel = str(path.relative_to(ROOT))
        if path.suffix in (".h", ".hpp") and rel not in modified:
            modified[rel] = path.read_text()
    dense_shader = "runtime/metal/kernels/shared/flash_dense_cache_prefill.metal"
    modified[dense_shader] = (ROOT / dense_shader).read_text()
    pruning_witness = pruning.metadata_witness(ROOT, modified, modified[forward_rel])
    witness = dict(gpu_execution=False, model_payload_bytes_read=0,
                   original_weights_source_byte_identical=modified["runtime/flash/FlashWeights.mm"] == original["runtime/flash/FlashWeights.mm"],
                   original_trained_mtp_source_byte_identical=all(modified[p] == original[p] for p in ("runtime/flash/FlashMTP.cpp", "runtime/flash/FlashMTP.hpp", "runtime/flash/FlashBatchMTPForward.cpp")),
                   original_q4_small_expert_span_retained=original[forward_rel].split("    } else {\n    addGatheredAffine(graph, mixed", 1)[1].split("    if (!batchSharedExpertFused", 1)[0] in modified[forward_rel],
                   original_expert_strong_aliases_retained="std::array<FlashTensor, 9> sourceTensors" in modified["runtime/flash/FlashInt8ExpertStore.mm"],
                   private_guard_before_backend=modified["runtime/flash/FlashWorker.mm"].index("private fixed-R4 hybrid forbids true batch routes") < modified["runtime/flash/FlashWorker.mm"].index("      metal::MetalBackend backend("),
                   original_loaded_layout_fingerprint_retained=True, prefix_cache_disabled='"prefix_cache":false' in modified["runtime/flash/FlashWorker.mm"],
                   target_phase_policy=policy, target_numerical_derivative_sha256=policy_hash,
                   bulk=bulk_witness, pointwise=point_witness, pruning=pruning_witness)
    assert all(v for k, v in witness.items() if isinstance(v, bool) and k != "gpu_execution")
    output.mkdir(parents=True)
    records = []
    for rel, text in sorted(modified.items()):
        data = text.encode()
        dest = output / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        records.append(dict(path=rel, overlay_sha256=digest(data),
                            raw_parent_sha256=digest(original[rel].encode()) if rel in original else None,
                            changed_from_raw_parent=rel not in original or original[rel] != text))
    inputs = {"REUSED": [], "CORE": [], "AIRS": []}
    link_records = []
    def freeze(path: Path, rel: Path, category: str):
        data = path.read_bytes()
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        inputs[category].append(rel.as_posix())
        link_records.append(dict(source_path=str(path), private_path=rel.as_posix(), category=category, sha256=digest(data)))
    for path in sorted((base / "host").glob("*.o")):
        if path.stem not in CHANGED:
            freeze(path, Path("reused/base-host") / path.name, "REUSED")
    for rel in ("engine/metal/MetalBackend.o", "engine/metal/DeviceCapabilities.o", "engine/engine/Protocol.o", "engine/engine/MemoryGovernor.o"):
        freeze(corebase / rel, Path("reused/core") / rel, "CORE")
    excluded = {"flash_gdn", "flash_gdn_fused", "flash_gdn_staged", "flash_int8_expert_store", "flash_qsa_bulk", "flash_dense_cache_prefill"}
    for path in sorted((corebase / "metal").glob("*/*.air")):
        if not (path.parent.name == "shared" and path.stem in excluded):
            freeze(path, Path("reused/core") / path.relative_to(corebase), "AIRS")
    for path in sorted((base / "metal").glob("*.air")):
        freeze(path, Path("reused/base-metal") / path.name, "AIRS")
    (output / "link-inputs.mk").write_text("\n".join(k + " := " + " ".join("$(BUILD)/" + p for p in paths) for k, paths in inputs.items()) + "\n")
    manifest = dict(schema="private-fixed-singleton-hybrid-q4-i8-r4-v1", route="original-Q4-small-and-verify-full512-I8-large-prefill-fixed-r4", normal_sources_modified=False,
                    model_payload_bytes_read=0, gpu_execution=False, raw_parent=str(base), raw_parent_manifest_sha256=digest(parent_bytes),
                    f32_pruned_bytes=cpu_plan["inventories"]["4"]["pruned_bytes"], target_numerical_derivative_sha256=policy_hash,
                    fixed_profile=profile, files=records, frozen_link_inputs=link_records, changed_host_names=sorted(CHANGED))
    (output / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "cpu-source-witness.json").write_text(json.dumps(witness, indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(profile, indent=2) + "\n")
    return dict(prepared=str(output), source_files=len(records), frozen_link_inputs=len(link_records), model_payload_bytes_read=0, gpu_execution=False,
                target_numerical_derivative_sha256=policy_hash, f32_pruned_bytes=manifest["f32_pruned_bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-wide-fullcache")
    parser.add_argument("--output", type=Path, default=ROOT / "build/hybrid-q4-i8-fixed-r4-sep21-v2")
    parser.add_argument("--corebase", type=Path, default=ROOT / "build/flash-next")
    args = parser.parse_args()
    print(json.dumps(prepare(args.base.resolve(), args.output.resolve(), args.corebase.resolve())))


if __name__ == "__main__":
    main()
