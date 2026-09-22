#pragma once

// Private incremental execution strategy over the frozen pointwise worker.
// Original I8 weights, F32 late scales and all BF16 boundaries are retained.
// Whole-model equality remains a root-run qualification requirement.
#include "flash/FlashMoEBlocked.hpp"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace splash::flash::fixed_sg2_prefill_sep21 {

inline constexpr uint32_t kVariant=7,kEligibleRows=2048;
inline constexpr const char *kMarker=
    ";private-prefill-moe-sg2-k128-sep21-main-nonverification-r2048-m32-original-i8-f32lateScale-bf16-boundaries-v1";

[[nodiscard]] inline uint32_t parseSelection(const char *value) {
  if (!value || std::string_view(value)=="0") return 0;
  if (std::string_view(value)=="7") return kVariant;
  throw std::invalid_argument("SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT must be 0 or 7");
}

// Initial worker validation calls this before model admission/loading.
// Subsequent environment changes cannot alter dispatch selection.
[[nodiscard]] inline uint32_t selection() {
  static const uint32_t frozen=parseSelection(std::getenv("SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT"));
  return frozen;
}
[[nodiscard]] inline bool requested() {return selection()==kVariant;}

[[nodiscard]] constexpr bool eligibleFor(uint32_t rows,FlashMoEBlockedTile tile,
                                        bool verification,uint32_t selected) noexcept {
  return selected==kVariant && rows==kEligibleRows &&
      tile==FlashMoEBlockedTile::M32N64 && !verification;
}
[[nodiscard]] inline bool eligible(uint32_t rows,FlashMoEBlockedTile tile,bool verification) {
  return eligibleFor(rows,tile,verification,selection());
}
[[nodiscard]] constexpr const char *markerFor(uint32_t selected) noexcept {
  return selected==kVariant ? kMarker : "";
}
[[nodiscard]] inline const char *marker() {return markerFor(selection());}

[[nodiscard]] constexpr const char *producerNameFor(bool gate,uint32_t selected) noexcept {
  if (selected!=kVariant) return "";
  return gate ? "prefill_moe_sep21_memory_fixed_gate_up_m32_n64_k128_sg2" :
      "prefill_moe_sep21_memory_fixed_down_scatter_m32_n64_k128_sg2";
}
[[nodiscard]] inline const char *producerName(bool gate) {return producerNameFor(gate,selection());}
[[nodiscard]] constexpr uint32_t producerThreadsFor(uint32_t selected) noexcept {
  return selected==kVariant ? 64 : 0;
}
[[nodiscard]] inline uint32_t producerThreads() {return producerThreadsFor(selection());}

// Counts describe graph construction, rather than GPU completion. The store
// may snapshot its mutable atomic counters into this plain value for reports.
struct Counters final {
  bool enabled=false;
  uint64_t gateCalls=0,gateRows=0,downCalls=0,downRows=0;
};

} // namespace splash::flash::fixed_sg2_prefill_sep21
