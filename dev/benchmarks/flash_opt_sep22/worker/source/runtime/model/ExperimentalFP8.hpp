#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace splash::model::experimental_fp8 {

inline constexpr size_t kBlockElements = 32;
inline constexpr float kMaximumE4M3 = 448.0f;
inline constexpr uint32_t kMaximumInputWidth = 25600;
inline constexpr uint64_t kHalfScratchBytes = 104857600;

[[nodiscard]] constexpr uint64_t convertedScaleRowStride(uint64_t inputSize) {
  if (!inputSize || inputSize % 256 || inputSize > kMaximumInputWidth)
    throw std::invalid_argument("FP8 input width must be 256-aligned and at most 25600");
  // The SDK's auxiliary plane is tight along K/32. Only the primary FP8
  // tensor's row stride requires the documented 128-byte alignment.
  return inputSize / kBlockElements;
}

[[nodiscard]] constexpr uint64_t convertedProjectionBytes(uint64_t outputSize,
                                                         uint64_t inputSize) {
  if (!outputSize || outputSize % 256)
    throw std::invalid_argument("FP8 output width must be positive and 256-aligned");
  const uint64_t rowBytes = inputSize + convertedScaleRowStride(inputSize);
  if (outputSize > std::numeric_limits<uint64_t>::max() / rowBytes)
    throw std::overflow_error("FP8 converted projection byte count overflows");
  return outputSize * rowBytes;
}

// E4M3 has a bias of seven, subnormals at exponent zero, and finite values
// through exponent 15 / mantissa six. Codes 0x7f and 0xff represent NaN.
inline constexpr std::array<float, 127> kPositiveE4M3 = [] {
  std::array<float, 127> values{};
  for (size_t mantissa = 0; mantissa < 8; ++mantissa)
    values[mantissa] = static_cast<float>(mantissa) / 512.0f;
  float unit = 1.0f / 512.0f;
  for (size_t exponent = 1; exponent <= 15; ++exponent) {
    for (size_t mantissa = 0; mantissa < 8; ++mantissa) {
      const size_t code = exponent * 8 + mantissa;
      if (code < values.size())
        values[code] = static_cast<float>(8 + mantissa) * unit;
    }
    unit *= 2.0f;
  }
  return values;
}();

static_assert(kPositiveE4M3[1] == 1.0f / 512.0f);
static_assert(kPositiveE4M3[8] == 1.0f / 64.0f);
static_assert(kPositiveE4M3[126] == kMaximumE4M3);

[[nodiscard]] constexpr float decodePositiveE4M3(uint8_t code) {
  if (code >= kPositiveE4M3.size())
    throw std::invalid_argument("E4M3 positive code must be finite and unsigned");
  return kPositiveE4M3[code];
}

[[nodiscard]] constexpr float decodeE4M3(uint8_t code) {
  const float value = decodePositiveE4M3(static_cast<uint8_t>(code & 0x7f));
  return code & 0x80 ? -value : value;
}

// Round the normalized value to the nearest finite E4M3 value, with ties
// choosing an even mantissa bit. Finite overflow saturates; signed zero stays
// signed. Nonfinite source values are conversion errors, never NaN encodings.
[[nodiscard]] inline uint8_t encodeE4M3(float value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("E4M3 conversion requires a finite source value");
  const uint8_t sign = std::signbit(value) ? 0x80 : 0;
  const float magnitude = std::fabs(value);
  if (magnitude >= kMaximumE4M3)
    return static_cast<uint8_t>(126 | sign);
  const auto upper = std::lower_bound(kPositiveE4M3.begin(),
                                     kPositiveE4M3.end(), magnitude);
  size_t code = static_cast<size_t>(upper - kPositiveE4M3.begin());
  if (code && *upper != magnitude) {
    const float lowerDistance = magnitude - kPositiveE4M3[code - 1];
    const float upperDistance = *upper - magnitude;
    if (lowerDistance < upperDistance ||
        (lowerDistance == upperDistance && ((code - 1) & 1) == 0))
      --code;
  }
  return static_cast<uint8_t>(code | sign);
}

[[nodiscard]] inline float decodeUE8M0Scale(uint8_t code) {
  if (code == 255)
    throw std::invalid_argument("UE8M0 scale must be finite");
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

// Smallest representable power-of-two scale that contains the block maximum
// within E4M3's finite range. UE8M0 stores exponent + 127; zero blocks use one.
[[nodiscard]] inline uint8_t selectUE8M0Scale(float maximumAbsolute) {
  if (!std::isfinite(maximumAbsolute) || maximumAbsolute < 0.0f)
    throw std::invalid_argument("FP8 block maximum must be finite and nonnegative");
  if (maximumAbsolute == 0.0f)
    return 127;
  int exponent = 0;
  const float fraction = std::frexp(maximumAbsolute, &exponent);
  // 448 = 0.875 * 2^9. Comparing the exact binary fraction avoids rounding
  // log2 at an E4M3 range boundary, including subnormal source values.
  const int scaleExponent = std::clamp(exponent - 9 + (fraction > 0.875f),
                                     -127, 127);
  return static_cast<uint8_t>(scaleExponent + 127);
}

[[nodiscard]] inline uint8_t selectUE8M0Scale(std::span<const float> block) {
  if (block.size() != kBlockElements)
    throw std::invalid_argument("FP8 scale selection requires a 32-element block");
  float maximumAbsolute = 0.0f;
  for (const float value : block) {
    if (!std::isfinite(value))
      throw std::invalid_argument("FP8 block contains a nonfinite source value");
    maximumAbsolute = std::max(maximumAbsolute, std::fabs(value));
  }
  return selectUE8M0Scale(maximumAbsolute);
}

} // namespace splash::model::experimental_fp8
