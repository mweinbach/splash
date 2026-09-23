#include "dev/benchmarks/expert_r4_preflight_bundle_sep22/policy.hpp"
#include "dev/benchmarks/expert_r4_compact_verify_worker_sep22/bridge.hpp"
#include "dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy.hpp"
// Independent private gathered MPP Store overlay v1.
#pragma once

#include "FlashMoEBlocked.hpp"
#include "FlashGatheredMPP.hpp"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace splash::flash {
// Private diagnostics: successful graph construction, not GPU completion.
struct FlashInt8ExpertStoreGraphCounters {
  uint64_t gate_up_graph_calls = 0, gate_up_graph_rows = 0;
  uint64_t down_graph_calls = 0, down_graph_rows = 0;
  uint64_t encoded_hit_dispatches = 0, encoded_miss_dispatches = 0;
  uint64_t full_inventory_graph_calls = 0;
  // Physical rows>=256 only, excluding tiny prefill/decode/verifier graphs.
  uint64_t large_row_gate_up_graph_calls = 0, large_row_gate_up_graph_rows = 0;
  uint64_t large_row_down_graph_calls = 0, large_row_down_graph_rows = 0;
  uint64_t large_row_encoded_hit_dispatches = 0, large_row_encoded_miss_dispatches = 0;
  uint64_t large_row_full_inventory_graph_calls = 0;
  // Graph construction, not GPU completion. gathered MPP only physical rows1..16.
  uint64_t gathered_mpp_gate_up_graph_calls = 0, gathered_mpp_gate_up_graph_rows = 0;
  uint64_t gathered_mpp_down_graph_calls = 0, gathered_mpp_down_graph_rows = 0;
};
inline constexpr const char *kFlashInt8ExpertStoreSemantics =
    "private-allrows-full512-signed-i8-f32-late-row-scale-bf16-dots-m16tight-small-m32m64-large-no-target-q4gpu-v1";

// A numerical alternative for large-row prefill only. The immutable signed
// INT8 coefficients are converted offline, checksum verified and mapped
// readonly. One store serves the target trunk and its batched prefills.
// Original Q4 coefficients remain the miss fallback; small-row decode and
// trained MTP weights are not selected by this class.
class FlashInt8ExpertStore final {
public:
  FlashInt8ExpertStore(metal::MetalBackend &backend, const FlashWeights &weights,
                      const std::filesystem::path &directory);
  ~FlashInt8ExpertStore();
  FlashInt8ExpertStore(const FlashInt8ExpertStore &) = delete;
  FlashInt8ExpertStore &operator=(const FlashInt8ExpertStore &) = delete;
  FlashInt8ExpertStore(FlashInt8ExpertStore &&) noexcept;
  FlashInt8ExpertStore &operator=(FlashInt8ExpertStore &&) noexcept;

  // CPU metadata, source geometry and file-size checks only; no payload read,
  // mapping or backend allocation occurs before the caller reserves this.
  [[nodiscard]] static uint64_t plannedBytes(const FlashWeights &weights,
      const std::filesystem::path &directory);
  [[nodiscard]] const std::string &identitySha256() const;
  [[nodiscard]] const std::string &numericalIdentitySha256() const;
  [[nodiscard]] const std::string &planSha256() const;
  [[nodiscard]] uint64_t mappedBytes() const noexcept;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] std::span<const uint32_t> selectedExpertIDs(uint32_t layer) const;
  [[nodiscard]] std::vector<metal::MetalBuffer> immutableWeightBuffers() const;

  [[nodiscard]] FlashInt8ExpertStoreGraphCounters graphCounters() const noexcept;

  void addGateUp(metal::CommandGraph &graph, uint32_t layer,
      const FlashMoEBlockedScratch &scratch, metal::MetalBuffer diagnostics,
      uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections = 10) const;
  void addDownScatter(metal::CommandGraph &graph, uint32_t layer,
      const FlashMoEBlockedScratch &scratch, metal::MetalBuffer diagnostics,
      uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections = 10) const;

  // Component-bit-exact qualified scheduling; actual-model equality pending.
  void addFixedSG2PrefillGateUp(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,
      uint32_t rows,FlashMoEBlockedTile tile,uint32_t selections=10) const;
  void addFixedSG2PrefillDownScatter(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,
      uint32_t rows,FlashMoEBlockedTile tile,uint32_t selections=10) const;
  [[nodiscard]] fixed_sg2_prefill_sep21::Counters fixedSG2PrefillCounters() const noexcept;

  // Store methods keep immutable base mappings and rank allocations private.
  // Dispatch-bound MetalBuffer copies retain their bases until graph disposal.
  [[nodiscard]] bool gatheredMPPEnabled() const;
  [[nodiscard]] uint32_t gatheredMPPMaximumRows() const;
  void addGatheredMPPGateUp(metal::CommandGraph &graph, uint32_t layer,
      metal::MetalBuffer input, metal::MetalBuffer originalExpertIDs,
      metal::MetalBuffer canonicalIntermediate, metal::MetalBuffer diagnostics,
      uint32_t rows, uint32_t selections = 10) const;
  void addGatheredMPPDown(metal::CommandGraph &graph, uint32_t layer,
      metal::MetalBuffer canonicalIntermediate, metal::MetalBuffer originalExpertIDs,
      metal::MetalBuffer canonicalExpertDown, metal::MetalBuffer diagnostics,
      uint32_t rows, uint32_t selections = 10) const;
  [[nodiscard]] bool compactNativeR4VerifyEnabled() const;
  [[nodiscard]] compact_native_r4_verify_sep22::Counters compactNativeR4VerifyCounters() const;
  void addCompactNativeR4VerifyPack(metal::CommandGraph &graph,uint32_t layer,
      metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &scratch,
      metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
  void addCompactNativeR4VerifyGateUp(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
  void addCompactNativeR4VerifyDown(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
  [[nodiscard]] compact_r4_preflight_sep22::Counters compactR4PreflightCounters() const;
  void addCompactNativeR4VerifyChain(metal::CommandGraph &graph,uint32_t layer,
      metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &scratch,
      metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace splash::flash
