#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace splash::model::experimental_int8 {

inline constexpr uint32_t kMaximumRows = 2048;
inline constexpr uint32_t kMaximumInputWidth = 25600;
inline constexpr uint64_t kCodeScratchBytes = 52428800;
inline constexpr uint64_t kScaleScratchBytes = 8192;
inline constexpr uint32_t kPartialChunkColumns = 512;
inline constexpr uint32_t kMaximumPartialRows = 128;
inline constexpr uint32_t kMaximumPartialChunks =
    (kMaximumInputWidth + kPartialChunkColumns - 1) / kPartialChunkColumns;
inline constexpr uint64_t kPartialPeakScratchBytes =
    uint64_t{kMaximumPartialRows} * kMaximumPartialChunks * sizeof(float);
inline constexpr uint64_t kPartialInvalidScratchBytes =
    uint64_t{kMaximumPartialRows} * kMaximumPartialChunks * sizeof(uint32_t);
static_assert(sizeof(float) == 4 && kMaximumPartialChunks == 50);
inline constexpr int kMaximumCode = 127;

[[nodiscard]] constexpr uint64_t convertedProjectionBytes(uint64_t outputSize,
                                                         uint64_t inputSize) {
  if (!outputSize || outputSize % 256 || !inputSize || inputSize % 256 ||
      inputSize > kMaximumInputWidth)
    throw std::invalid_argument("INT8 projection requires positive 256-aligned dimensions and K <= 25600");
  const uint64_t rowBytes = inputSize + 4;
  if (outputSize > std::numeric_limits<uint64_t>::max() / rowBytes)
    throw std::overflow_error("INT8 converted projection byte count overflows");
  return outputSize * rowBytes;
}

// Explicit ties-to-even rounding keeps the quantizer independent of the host's
// nearbyint rounding mode. Clamp before the integer cast, including finite
// source values outside the representable code range. Both zero signs encode 0.
[[nodiscard]] inline int8_t quantizeNormalized(float value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("INT8 quantization requires a finite source value");
  const float magnitude = std::fabs(value);
  if (magnitude >= static_cast<float>(kMaximumCode))
    return static_cast<int8_t>(value < 0.0f ? -kMaximumCode : kMaximumCode);
  const float lower = std::floor(magnitude);
  int code = static_cast<int>(lower);
  const float fraction = magnitude - lower;
  if (fraction > 0.5f || (fraction == 0.5f && (code & 1)))
    ++code;
  return static_cast<int8_t>(value < 0.0f ? -code : code);
}

[[nodiscard]] inline int8_t quantize(float value, float scale) {
  if (!std::isfinite(value))
    throw std::invalid_argument("INT8 quantization requires a finite source value");
  if (!std::isfinite(scale) || scale <= 0.0f)
    throw std::invalid_argument("INT8 quantization scale must be finite and positive");
  // Compare in double before division so finite value/scale overflow saturates
  // instead of creating infinity. Within range, use the FP32 normalized value.
  if (std::fabs(static_cast<double>(value)) >=
      static_cast<double>(kMaximumCode) * static_cast<double>(scale))
    return static_cast<int8_t>(value < 0.0f ? -kMaximumCode : kMaximumCode);
  return quantizeNormalized(value / scale);
}

[[nodiscard]] inline float selectRowScale(float maximumAbsolute) {
  if (!std::isfinite(maximumAbsolute) || maximumAbsolute < 0.0f)
    throw std::invalid_argument("INT8 row maximum must be finite and nonnegative");
  if (maximumAbsolute == 0.0f)
    return 1.0f;
  const float scale = maximumAbsolute / static_cast<float>(kMaximumCode);
  if (!std::isfinite(scale) || scale <= 0.0f)
    throw std::invalid_argument("INT8 row scale is not finite and positive");
  return scale;
}

[[nodiscard]] inline float selectRowScale(std::span<const float> row) {
  if (row.empty())
    throw std::invalid_argument("INT8 row scale selection requires a nonempty row");
  float maximumAbsolute = 0.0f;
  for (const float value : row) {
    if (!std::isfinite(value))
      throw std::invalid_argument("INT8 row contains a nonfinite source value");
    maximumAbsolute = std::max(maximumAbsolute, std::fabs(value));
  }
  return selectRowScale(maximumAbsolute);
}

} // namespace splash::model::experimental_int8
