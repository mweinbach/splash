#pragma once

#include "flash/FlashOriginalResidency.hpp"

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace splash::flash::private_full_residency {
inline constexpr uint64_t kOriginalBytes = 106320429056ULL;
inline constexpr uint64_t kOriginalBases = 21;
inline constexpr uint64_t kMarginBytes = 2ULL << 30;
inline constexpr std::string_view kSource =
    "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
inline constexpr std::string_view kLayout =
    "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0";

inline bool parseSwitch(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_PRIVATE_FULL_ORIGINAL_RESIDENT must be 0 or 1");
}

inline void validateFlags(bool fullRequested, bool textRequested) {
  if (fullRequested && textRequested)
    throw std::invalid_argument("private full-original residency and original-text residency are mutually exclusive");
}

inline void validateGeometry(std::string_view source, std::string_view layout,
    uint64_t bases, uint64_t bytes, uint32_t layers, uint32_t experts,
    uint32_t hiddenSize) {
  if (source != kSource || layout != kLayout || bases != kOriginalBases ||
      bytes != kOriginalBytes || layers != 48 || experts != 512 || hiddenSize != 2560)
    throw std::invalid_argument("private full-original residency requires the qualified 21 original bases and source/layout geometry");
}

inline bool hostAllowed(bool valid, bool growth, bool normal,
    uint64_t hostHeadroom) noexcept {
  return flashOriginalResidencyHostAllowed(valid, growth, normal,
      hostHeadroom, kOriginalBytes);
}
} // namespace splash::flash::private_full_residency
