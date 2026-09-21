#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>

namespace splash::flash::idle_maintenance {

inline constexpr uint64_t kIntervalMilliseconds = 500;
inline constexpr uint64_t kMarginBytes = 2ULL << 30;
inline constexpr std::string_view kSource =
    "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
inline constexpr std::string_view kLayout =
    "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0";
inline constexpr uint64_t kOriginalBases = 21;
inline constexpr uint64_t kOriginalBytes = 106320429056ULL;
inline constexpr uint64_t kOwnerCount = 1134;
inline constexpr uint64_t kOwnerBytes = 144326852608ULL;
inline constexpr uint32_t kGuard = 0xd75a19c3;
inline constexpr uint32_t kUnwritten = 0xa823745e;
inline constexpr uint32_t kOutputWords = 4096;
inline constexpr uint32_t kOutputValueBegin = 16;
inline constexpr uint32_t kChecksumIndex = 1;
inline constexpr uint32_t kCountIndex = 2;

inline bool parseSwitch(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument(
      "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE must be 0 or 1");
}

inline uint64_t parseInterval(const char *value) {
  if (!value) return kIntervalMilliseconds;
  const std::string_view text(value);
  if (text.empty() || text[0] == '0' || text.size() > 4)
    throw std::invalid_argument(
        "SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS must be a canonical decimal from 100 to 1000");
  uint64_t interval = 0;
  for (char digit : text) {
    if (digit < '0' || digit > '9')
      throw std::invalid_argument(
          "SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS must be a canonical decimal from 100 to 1000");
    interval = interval * 10 + static_cast<unsigned>(digit - '0');
  }
  if (interval < 100 || interval > 1000)
    throw std::invalid_argument(
        "SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS must be a canonical decimal from 100 to 1000");
  return interval;
}

// This is a fresh host-availability check, not a promise to pin resources or
// an admission for new backing. The caller also checks zero reservations.
inline bool hostAllowed(bool valid, bool growth, bool effectiveNormal,
    bool systemNormal, uint64_t available, uint64_t reserve) noexcept {
  return valid && growth && effectiveNormal && systemNormal &&
      reserve <= std::numeric_limits<uint64_t>::max() - kMarginBytes &&
      available > reserve + kMarginBytes;
}

inline void validateGeometry(std::string_view source, std::string_view layout,
    uint64_t originalCount, uint64_t originalBytes, uint64_t ownerCount,
    uint64_t ownerBytes, uint32_t layers, uint32_t experts, uint32_t hidden) {
  if (source != kSource || layout != kLayout ||
      originalCount != kOriginalBases || originalBytes != kOriginalBytes ||
      ownerCount != kOwnerCount || ownerBytes != kOwnerBytes ||
      layers != 48 || experts != 512 || hidden != 2560)
    throw std::invalid_argument(
        "idle residency maintenance requires the qualified immutable source, "
        "layout, original bases and complete owner geometry");
}

// Armed means a real user request successfully completed after GPU work. Mask-blocked
// requests are active and therefore never qualify as idle. The final incoming
// check is queue-mutex protected; later arrivals can still overlap one command.
inline bool eligible(bool armed, bool allActiveEmpty, bool pendingEmpty,
    bool liveEmpty, bool inFlight, bool ticketsOutstanding, bool stopping,
    bool healthy, bool incomingEmpty) noexcept {
  return armed && allActiveEmpty && pendingEmpty && liveEmpty && !inFlight &&
      !ticketsOutstanding && !stopping && healthy && incomingEmpty;
}

inline uint32_t mixedWord(uint32_t sourceFirstWord, uint32_t ownerIndex) noexcept {
  return sourceFirstWord ^ (0x9e3779b9u * (ownerIndex + 1u));
}

// The helper checks the same ABI independently. This pure reference makes
// failures in every guard word, exact owner read, count and checksum testable.
inline bool validateOutput(std::span<const uint32_t> output,
    std::span<const uint32_t> expectedMixedWords) noexcept {
  if (output.size() != kOutputWords || expectedMixedWords.size() != kOwnerCount)
    return false;
  uint32_t checksum = 0;
  for (uint32_t value : expectedMixedWords) checksum ^= value;
  for (uint32_t index = 0; index != kOutputWords; ++index) {
    uint32_t expected = kGuard;
    if (index == kChecksumIndex) expected = checksum;
    else if (index == kCountIndex) expected = static_cast<uint32_t>(kOwnerCount);
    else if (index >= kOutputValueBegin &&
             index - kOutputValueBegin < expectedMixedWords.size())
      expected = expectedMixedWords[index - kOutputValueBegin];
    if (output[index] != expected) return false;
  }
  return true;
}

static_assert(kOutputValueBegin + kOwnerCount <= kOutputWords);
static_assert(kChecksumIndex < kOutputValueBegin &&
    kCountIndex < kOutputValueBegin && kChecksumIndex != kCountIndex);

} // namespace splash::flash::idle_maintenance
