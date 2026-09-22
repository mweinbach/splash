"""Private candidate startup/identity/status instrumentation."""
from __future__ import annotations


def replace(text: str, before: str, after: str, count: int = 1) -> str:
    if text.count(before) != count:
        raise RuntimeError(f"All-row Worker source drift: {before!r}")
    return text.replace(before, after)


def transform(relative: str, text: str) -> str:
    if relative != "runtime/flash/FlashWorker.mm":
        return text
    text = replace(text, '#include "flash/FlashInt8ExpertStore.hpp"', '#include "flash/FlashInt8ExpertStore.hpp"\n#include "flash/FlashInt8ExpertStoreMetadata.hpp"')
    early = '''      // Fail the private policy/derivative before Metal backend/probe creation.
      if (!environmentSwitch("SPLASH_FLASH_ALLROWS_FULL512_TARGET") ||
          !environmentSwitch("SPLASH_FLASH_PLE_SSD_STREAMING") ||
          !environmentSwitch("SPLASH_FLASH_BLOCKED_MOE") ||
          !environmentSwitch("SPLASH_FLASH_MOE_DIRECT_A") ||
          !environmentSwitch("SPLASH_FLASH_MOE_Q4X8"))
        throw std::invalid_argument("private all-row Full512 requires explicit policy, PLE SSD, blocked and Direct-A routes");
      if (std::getenv("SPLASH_FLASH_HOT_EXPERT_PLAN"))
        throw std::invalid_argument("private all-row Full512 forbids a second hot-expert policy");
      if (environmentSwitch("SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE") ||
          environmentSwitch("SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT"))
        throw std::invalid_argument("private all-row Full512 requires idle/original-text residency disabled");
      const char *fullStorePath = std::getenv("SPLASH_FLASH_INT8_EXPERT_STORE");
      if (!fullStorePath || !*fullStorePath)
        throw std::invalid_argument("private all-row Full512 requires its certified store path");
      const auto earlyFull = loadFlashInt8ExpertStoreMetadata(fullStorePath,
          "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
          "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",
          NormConvention::OnePlusWeight);
      if (earlyFull.identitySha256 != "ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1")
        throw std::invalid_argument("private all-row target requires the certified Full512 derivative");
      for (const auto &layer : earlyFull.layers) {
        if (layer.selectedIDs.size() != 512)
          throw std::invalid_argument("private all-row target requires exactly Full512 in every layer");
        for (uint32_t id = 0; id < 512; ++id) if (layer.selectedIDs[id] != id)
          throw std::invalid_argument("private all-row target requires canonical complete expert IDs");
      }
'''
    text = replace(text, "      metal::MetalBackend backend((executablePath().parent_path() / \"splash.metallib\").string());", early + "      metal::MetalBackend backend((executablePath().parent_path() / \"splash.metallib\").string());")
    text = replace(text, '      << R"(,"loaded_model_layout_sha256":)" << json::quote(weights_.manifestFingerprint())', '      << R"(,"loaded_model_layout_sha256":)" << json::quote(weights_.manifestFingerprint())\n      << R"(,"target_numerical_derivative_sha256":)" << (persistedExperts ? json::quote(persistedExperts->numericalIdentitySha256()) : "null")\n      << R"(,"target_all_rows_full512":true,"original_target_gpu_omitted":true)"')
    text = replace(text, '"scope":"large-row target prefill; original Q4 misses, decode and trained MTP"', '"scope":"PRIVATE all target rows including decode/verifier; signed INT8 late row scaling; original trained MTP retained"')
    counters = '''      << R"(,"large_row_counter_scope":"physical rows>=256; graph construction; excludes tiny prefill/decode/verifier")"
      << R"(,"large_row_gate_up_graph_calls":)" << persistedExpertGraphs.large_row_gate_up_graph_calls
      << R"(,"large_row_gate_up_graph_rows":)" << persistedExpertGraphs.large_row_gate_up_graph_rows
      << R"(,"large_row_down_graph_calls":)" << persistedExpertGraphs.large_row_down_graph_calls
      << R"(,"large_row_down_graph_rows":)" << persistedExpertGraphs.large_row_down_graph_rows
      << R"(,"large_row_encoded_hit_dispatches":)" << persistedExpertGraphs.large_row_encoded_hit_dispatches
      << R"(,"large_row_encoded_miss_dispatches":)" << persistedExpertGraphs.large_row_encoded_miss_dispatches
      << R"(,"large_row_full_inventory_graph_calls":)" << persistedExpertGraphs.large_row_full_inventory_graph_calls
'''
    text = replace(text, '      << R"(,"full_inventory_graph_calls":)" << persistedExpertGraphs.full_inventory_graph_calls', '      << R"(,"full_inventory_graph_calls":)" << persistedExpertGraphs.full_inventory_graph_calls\n' + counters.rstrip())
    stats = '''      << R"(,"all_rows_full512_target":)" << (pleIO.allRowsFull512Target ? "true" : "false")
      << R"(,"target_original_disk_tensor_count":)" << pleIO.targetOriginalDiskTensorCount
      << R"(,"target_original_disk_projection_count":)" << pleIO.targetOriginalDiskProjectionCount
      << R"(,"target_original_disk_logical_bytes":)" << pleIO.targetOriginalDiskLogicalBytes
      << R"(,"target_original_disk_payload_bytes":)" << pleIO.targetOriginalDiskPayloadBytes
      << R"(,"target_full512_store_manifest_sha256":)" << json::quote(pleIO.targetFull512StoreManifestSha256)
'''
    # Loader storage facts are separate from host row-cache I/O statistics.
    stats = stats.replace("pleIO.", "pleStorage.")
    text = replace(text, '      << R"(,"gpu_mapped_original_bytes":)" << pleStorage.gpuMappedBytes', '      << R"(,"gpu_mapped_original_bytes":)" << pleStorage.gpuMappedBytes\n' + stats.rstrip())
    return text
