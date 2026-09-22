#pragma once

// CPU-only coefficient conversion and FP64 error accounting. Numerical output
// acceptance belongs to the oracle, independently of coefficient quantization.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace splash::dense_w8a8 {

inline float bf16Number(uint16_t bits) {
  return std::bit_cast<float>(uint32_t(bits) << 16);
}

inline uint16_t bf16Bits(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    // Preserve infinities; preserve the sign and payload of NaNs while making
    // them quiet, including FP32 payloads that only occupy discarded bits.
    return uint16_t((bits >> 16) | ((bits & 0x007fffffu) ? 0x0040u : 0u));
  }
  return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline void validateScale(float scale) {
  if (!(scale > 0.0f) || !std::isfinite(scale))
    throw std::invalid_argument("W8A8 scale must be finite and positive");
}

inline bool symmetricClipped(float value, float scale) {
  if (!std::isfinite(value))
    throw std::invalid_argument("W8A8 coefficient must be finite");
  validateScale(scale);
  // At 127.5, ties-to-even produces 128 before clamping. Merely rounding a
  // quotient in (127,127.5) to 127 is not counted as clipping.
  return std::abs(value / scale) >= 127.5f;
}

inline int8_t symmetricCode(float value, float scale) {
  if (!std::isfinite(value))
    throw std::invalid_argument("W8A8 coefficient must be finite");
  validateScale(scale);
  // Deliberately divide in FP32, matching the intended device conversion.
  const float quotient = value / scale;
  // These checks also safely saturate a finite division that overflowed.
  if (quotient >= 127.0f) return int8_t(127);
  if (quotient <= -127.0f) return int8_t(-127);
  const float magnitude = std::abs(quotient);
  const float integral = std::floor(magnitude);
  const float fraction = magnitude - integral;
  int rounded = int(integral);  // Integral is bounded by 126 here.
  if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) ++rounded;
  return int8_t(std::signbit(quotient) ? -rounded : rounded);
}

inline float symmetricScale(const uint16_t *source, uint32_t k) {
  if (k && !source) throw std::invalid_argument("W8A8 source row is null");
  float maximum = 0.0f;
  for (uint32_t i = 0; i < k; ++i) {
    const float value = bf16Number(source[i]);
    if (!std::isfinite(value))
      throw std::invalid_argument("W8A8 source row contains a nonfinite BF16 coefficient");
    maximum = std::max(maximum, std::abs(value));
  }
  if (maximum == 0.0f) return 1.0f;
  const float scale = maximum / 127.0f;
  if (!(scale > 0.0f) || !std::isfinite(scale))
    throw std::invalid_argument("W8A8 source row scale underflowed or is nonfinite");
  return scale;
}

class QuantError {
 public:
  void add(double source, double dequantized, int8_t code, bool clipped = false) {
    addValues(source, dequantized, source != 0.0 && code == 0, clipped);
  }

  // A pair overload is useful for complete output comparisons. It uses FP64
  // values directly and counts a nonzero source mapped to an observed zero.
  void add(double source, double observed) {
    addValues(source, observed, source != 0.0 && observed == 0.0, false);
  }

  void addCoefficient(uint16_t source, int8_t code, float scale, bool clipped = false) {
    validateScale(scale);
    add(double(bf16Number(source)), double(scale) * double(code), code, clipped);
  }

  uint64_t elements() const { return elements_; }
  uint64_t nonfinite() const { return nonfinite_; }
  uint64_t sourceNonzeroQuantizedZero() const { return sourceNonzeroQuantizedZero_; }
  uint64_t clippedCount() const { return clipped_; }
  double maxAbsError() const { return maxAbs_; }
  double squaredError() const { return squaredError_; }
  double squaredReference() const { return squaredReference_; }
  double squaredActual() const { return squaredActual_; }
  double product() const { return product_; }
  double l2() const { return std::sqrt(squaredError_); }
  double referenceL2() const { return std::sqrt(squaredReference_); }
  double relativeL2() const {
    if (squaredReference_ == 0.0)
      return squaredError_ == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    return std::sqrt(squaredError_ / squaredReference_);
  }
  double cosine() const {
    if (squaredReference_ == 0.0 || squaredActual_ == 0.0)
      return squaredReference_ == 0.0 && squaredActual_ == 0.0 ? 1.0 : 0.0;
    return std::clamp(product_ / (std::sqrt(squaredReference_) * std::sqrt(squaredActual_)), -1.0, 1.0);
  }

  void writeJSON(std::ostream &out) const {
    const auto precision = out.precision();
    const auto flags = out.flags();
    out << std::dec << std::noshowpos << std::defaultfloat
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\"elements\":" << elements_ << ",\"nonfinite\":" << nonfinite_
        << ",\"source_nonzero_quantized_zero\":" << sourceNonzeroQuantizedZero_
        << ",\"clipped\":" << clipped_ << ",\"max_abs\":";
    writeNumber(out, maxAbs_);
    out << ",\"l2\":"; writeNumber(out, l2());
    out << ",\"relative_l2\":"; writeNumber(out, relativeL2());
    out << ",\"cosine\":"; writeNumber(out, cosine());
    out << '}';
    out.precision(precision);
    out.flags(flags);
  }

 private:
  static void writeNumber(std::ostream &out, double value) {
    if (std::isfinite(value)) out << value;
    else out << "null";
  }

  void addValues(double source, double actual, bool quantizedZero, bool clipped) {
    ++elements_;
    clipped_ += clipped;
    if (!std::isfinite(source) || !std::isfinite(actual)) { ++nonfinite_; return; }
    sourceNonzeroQuantizedZero_ += quantizedZero;
    const double difference = actual - source;
    maxAbs_ = std::max(maxAbs_, std::abs(difference));
    squaredError_ += difference * difference;
    squaredReference_ += source * source;
    squaredActual_ += actual * actual;
    product_ += source * actual;
  }

  uint64_t elements_ = 0, nonfinite_ = 0, sourceNonzeroQuantizedZero_ = 0, clipped_ = 0;
  double maxAbs_ = 0.0, squaredError_ = 0.0, squaredReference_ = 0.0, squaredActual_ = 0.0, product_ = 0.0;
};

inline void quantizeRow(const uint16_t *source, uint32_t k, int8_t *output,
                        float &scale, QuantError &error) {
  if (k && !output) throw std::invalid_argument("W8A8 output row is null");
  const float rowScale = symmetricScale(source, k);  // Validate the whole row before writing it.
  scale = rowScale;
  for (uint32_t i = 0; i < k; ++i) {
    const float value = bf16Number(source[i]);
    const int8_t code = symmetricCode(value, rowScale);
    output[i] = code;
    error.addCoefficient(source[i], code, rowScale, symmetricClipped(value, rowScale));
  }
}

// A symmetric signed INT8 dot has this magnitude bound, including adversarial
// operands. This is arithmetic certification, independent of output quality.
inline constexpr uint64_t int32DotMagnitudeBound(uint32_t k) {
  return uint64_t(k) * 127u * 127u;
}

}  // namespace splash::dense_w8a8
