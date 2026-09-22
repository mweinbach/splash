#pragma once
// Standard singleton operation policy; the inherited vector arithmetic and its
// RN/RTZ/FTZ certificate remain independent of this stricter call-site scope.
#include "dev/benchmarks/gemv_decode_r1_worker_sep21/bridge.hpp"
#include <cstdint>
#include <string>
#include <string_view>

namespace splash::flash::gemv_r1_teacher_sep22 {
inline constexpr std::string_view kScopeSchema =
    ";scope=strict-standard-singleton-R1-no-mtpState-no-prefill-no-verify-no-head-no-batch-v1";
[[nodiscard]] constexpr bool selectedFor(bool standardDecode, uint32_t rows,
    bool verification, bool gatheredEligible, bool vectorEnabled) noexcept {
  return standardDecode && rows == 1 && !verification && gatheredEligible && vectorEnabled;
}
[[nodiscard]] inline bool selected(bool standardDecode, uint32_t rows,
    bool verification, bool gatheredEligible) {
  return selectedFor(standardDecode, rows, verification, gatheredEligible,
      gemv_decode_r1_sep21::requested());
}
[[nodiscard]] inline std::string markerFor(bool enabled) {
  if (!enabled) return {};
  return gemv_decode_r1_sep21::markerFor(true) + std::string(kScopeSchema);
}
[[nodiscard]] inline std::string implementationMarker() {
  return markerFor(gemv_decode_r1_sep21::requested());
}
} // namespace splash::flash::gemv_r1_teacher_sep22
