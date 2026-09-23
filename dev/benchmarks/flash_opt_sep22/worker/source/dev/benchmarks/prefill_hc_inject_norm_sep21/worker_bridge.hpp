#pragma once
#include "bridge.hpp"
#include <atomic>
#include <cstdlib>
#include <string_view>

namespace splash::flash::prefill_hc_inject_norm_sep21 {
inline constexpr const char *kFlag = "SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21";
inline constexpr std::string_view kPolicy =
    "private-prefill-hc-inject-norm-r512to2048-main-nonverify-literal-bf16-f32-norm-v1";
[[nodiscard]] inline bool parse(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument(std::string(kFlag) + " must be 0 or 1");
}
[[nodiscard]] inline bool requested() {
  static const bool value = parse(std::getenv(kFlag));
  return value;
}
[[nodiscard]] constexpr bool mainEligible(uint32_t rows, bool verification,
    bool singletonMain, bool selected) noexcept {
  return selected && singletonMain && !verification && rows >= 512 && rows <= 2048;
}
[[nodiscard]] constexpr bool nextNormEligible(bool mainSelected, bool nextHasPLE,
    bool nextIsTerminalMixer) noexcept {
  return mainSelected && !nextHasPLE && !nextIsTerminalMixer;
}
[[nodiscard]] inline const char *selectionMarker(bool selected) noexcept {
  return selected ? ";private-prefill-hc-inject-norm-r512to2048-main-nonverify-selected1-v1"
                  : ";private-prefill-hc-inject-norm-selected0-basegraphs-v1";
}
// Encoding observations only. These counters never enter any numerical/cache
// identity or kernel route string, and allocate no host/GPU/residency plane.
struct EncodedCounters {
  std::atomic<uint64_t> forwards{0}, attentionToMlp{0}, mlpToNext{0}, pleExcluded{0}, terminalExcluded{0};
};
inline EncodedCounters &encodedCounters() {
  static EncodedCounters counters;
  return counters;
}
inline void recordForward() noexcept { encodedCounters().forwards.fetch_add(1,std::memory_order_relaxed); }
inline void recordAttentionToMlp() noexcept { encodedCounters().attentionToMlp.fetch_add(1,std::memory_order_relaxed); }
inline void recordMlpToNext() noexcept { encodedCounters().mlpToNext.fetch_add(1,std::memory_order_relaxed); }
inline void recordPLEExcluded() noexcept { encodedCounters().pleExcluded.fetch_add(1,std::memory_order_relaxed); }
inline void recordTerminalExcluded() noexcept { encodedCounters().terminalExcluded.fetch_add(1,std::memory_order_relaxed); }
} // namespace splash::flash::prefill_hc_inject_norm_sep21
