#pragma once
// Independent CPU certificates for the private NUMERICAL W4A8 alternative.
// Source: original unsigned U4/G64 codes, original BF16 scale/bias and BF16 A.
// Candidate: XOR-centered signed I4, row-quantized signed I8 A, exact G64 I32
// dots/sums, staged F32 epilogue, ascending G64 accumulation and ONE late
// rowActivationScale multiply. No bit-parity
// or model-quality claim follows from these arithmetic certificates.
// SDK nuance: MTLTensor host packed strides require 128-byte alignment. Raw
// tensor_inline explicit 320-byte Down rows compile; runtime legality is not
// established. The candidate's 384-byte padded Down layout is conservative.
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::bench::w4a8 {
inline constexpr uint32_t kGroupSize = 64, kMaximumK = 2560;
inline constexpr int64_t kMaximumGroupDotMagnitude = 127 * 8 * kGroupSize;
inline constexpr int64_t kMaximumGroupSumMagnitude = 127 * kGroupSize;
inline constexpr double kMaximumProducerRelativeL2 = 0.03;
inline constexpr double kMinimumProducerCosine = 0.9995;
inline constexpr double kUnit32 = 0x1p-24, kUnit64 = 0x1p-53;
inline constexpr double kFlush32 = std::numeric_limits<float>::min();

inline float number(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }
inline uint16_t bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u)
    return uint16_t((bits >> 16) | ((bits & 0x007fffffu) ? 0x0040u : 0u));
  return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
inline bool sourceFinite(uint16_t bits) { return (bits & 0x7f80u) != 0x7f80u; }
inline bool positive(float value) { return std::isfinite(value) && value > 0; }
inline uint8_t centeredByte(uint8_t original) { return uint8_t(original ^ 0x88u); }
inline uint8_t originalByte(uint8_t centered) { return uint8_t(centered ^ 0x88u); }
inline int unsignedCode(std::span<const uint8_t> packed, uint32_t k) {
  return (packed[k / 2] >> ((k & 1u) * 4)) & 15;
}
inline int signedCode(std::span<const uint8_t> packed, uint32_t k) {
  const int nibble = unsignedCode(packed, k);
  return nibble < 8 ? nibble : nibble - 16;
}
inline float rowScale(float maximum) {
  if (!std::isfinite(maximum) || maximum < 0)
    throw std::invalid_argument("W4A8 row maximum is nonfinite/negative");
  return maximum > 0 ? std::max(maximum / 127.0f, std::numeric_limits<float>::min()) : 1.0f;
}
inline int roundEven(double value) {
  if (value >= 127) return 127;
  if (value <= -127) return -127;
  if (!std::isfinite(value)) return 0;
  const double magnitude = std::abs(value), base = std::floor(magnitude);
  const int integer = int(base);
  const double fraction = magnitude - base;
  const int result = integer + int(fraction > 0.5 || (fraction == 0.5 && (integer & 1)));
  return std::signbit(value) ? -result : result;
}
inline bool powerOfTwo(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value) & 0x7fffffffu;
  if (!bits || bits >= 0x7f800000u) return false;
  return bits >= 0x00800000u ? !(bits & 0x007fffffu) : std::has_single_bit(bits);
}
inline double ulp(float value) {
  if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
  const float magnitude = std::abs(value);
  if (!magnitude) return std::numeric_limits<float>::denorm_min();
  const float next = std::nextafter(magnitude, std::numeric_limits<float>::infinity());
  return std::isfinite(next) ? double(next) - magnitude
      : double(magnitude) - std::nextafter(magnitude, 0.0f);
}
inline float roundedMultiply(float a, float b) { volatile float result = a * b; return result; }
inline float roundedAdd(float a, float b) { volatile float result = a + b; return result; }
inline float roundedDivide(float a, float b) { volatile float result = a / b; return result; }
inline double upward(double value) { return std::nextafter(value, std::numeric_limits<double>::infinity()); }
inline double addUp(double a, double b) { return upward(a + b); }
inline double mulUp(double a, double b) { return upward(a * b); }
inline void writeNumber(std::ostream &out, double value) {
  if (std::isfinite(value)) out << std::setprecision(17) << value; else out << "null";
}

namespace detail {
// An independent exact dyadic accumulator. BF16 products are multiples of
// 2^-266; at K<=2560 all admitted BF16*BF16*U4 terms fit in 576 bits.
template <size_t Words> struct UInt {
  std::array<uint32_t, Words> words{};
  void addWord(size_t index, uint32_t value) {
    uint64_t carry = value;
    while (carry) {
      if (index >= Words) throw std::overflow_error("W4A8 exact dyadic accumulator overflow");
      const uint64_t total = uint64_t(words[index]) + carry;
      words[index++] = uint32_t(total); carry = total >> 32;
    }
  }
  void addShifted(uint32_t value, uint32_t shift) {
    if (!value) return;
    const uint64_t shifted = uint64_t(value) << (shift % 32);
    addWord(shift / 32, uint32_t(shifted));
    if (uint32_t(shifted >> 32)) addWord(shift / 32 + 1, uint32_t(shifted >> 32));
  }
  int compare(const UInt &other) const {
    for (size_t i = Words; i-- > 0;)
      if (words[i] != other.words[i]) return words[i] > other.words[i] ? 1 : -1;
    return 0;
  }
  UInt subtract(const UInt &other) const {
    UInt result; uint64_t borrow = 0;
    for (size_t i = 0; i < Words; ++i) {
      const uint64_t sub = uint64_t(other.words[i]) + borrow;
      result.words[i] = uint32_t(uint64_t(words[i]) - sub);
      borrow = uint64_t(words[i]) < sub;
    }
    if (borrow) throw std::logic_error("W4A8 unsigned dyadic subtraction underflow");
    return result;
  }
  bool bit(uint32_t index) const {
    return index / 32 < Words && ((words[index / 32] >> (index % 32)) & 1u);
  }
  bool anyBelow(uint32_t exclusive) const {
    for (uint32_t i = 0; i < exclusive / 32; ++i) if (words[i]) return true;
    const uint32_t remainder = exclusive % 32;
    return remainder && (words[exclusive / 32] & ((uint32_t(1) << remainder) - 1));
  }
  int highestBit() const {
    for (size_t i = Words; i-- > 0;)
      if (words[i]) return int(i * 32 + 31 - std::countl_zero(words[i]));
    return -1;
  }
};
struct Rounded { double value = 0, error = 0; };
template <size_t Words> inline Rounded convert(const UInt<Words> &value, int exponent) {
  const int highest = value.highestBit();
  if (highest < 0) return {};
  const uint32_t shift = highest > 52 ? uint32_t(highest - 52) : 0;
  uint64_t top = 0;
  for (uint32_t bit = 0; bit < 53; ++bit) if (value.bit(shift + bit)) top |= uint64_t(1) << bit;
  const bool discarded = value.anyBelow(shift);
  if (shift && value.bit(shift - 1) && (value.anyBelow(shift - 1) || (top & 1))) ++top;
  return {std::ldexp(double(top), exponent + int(shift)),
          discarded ? std::ldexp(1.0, exponent + int(shift) - 1) : 0.0};
}
struct BFParts { uint32_t mantissa = 0, shift = 0; bool negative = false; };
inline BFParts parts(uint16_t word) {
  if (!sourceFinite(word)) throw std::invalid_argument("W4A8 nonfinite source BF16 word");
  const uint32_t exponent = (word >> 7) & 255u;
  return {(word & 127u) + (exponent ? 128u : 0u), exponent ? exponent - 1 : 0u,
          bool(word & 0x8000u)};
}
template <size_t Words> struct SignedDyadic {
  UInt<Words> positive, negative;
  void addBF16(uint16_t word, int multiplier) {
    const auto p = parts(word);
    const uint32_t magnitude = p.mantissa * uint32_t(std::abs(multiplier));
    (p.negative != (multiplier < 0) ? negative : positive).addShifted(magnitude, p.shift);
  }
  void addProduct(uint16_t a, uint16_t b, int multiplier) {
    const auto x = parts(a), y = parts(b);
    const uint32_t magnitude = x.mantissa * y.mantissa * uint32_t(std::abs(multiplier));
    ((x.negative != y.negative) != (multiplier < 0) ? negative : positive)
        .addShifted(magnitude, x.shift + y.shift);
  }
  std::pair<bool, UInt<Words>> normalized() const {
    const bool sign = positive.compare(negative) < 0;
    return {sign, sign ? negative.subtract(positive) : positive.subtract(negative)};
  }
  bool equals(const SignedDyadic &other) const {
    const auto [sign, magnitude] = normalized();
    const auto [otherSign, otherMagnitude] = other.normalized();
    return sign == otherSign && magnitude.compare(otherMagnitude) == 0;
  }
  Rounded rounded(int exponent) const {
    const auto [sign, magnitude] = normalized();
    auto result = convert(magnitude, exponent);
    if (sign) result.value = -result.value;
    return result;
  }
};
inline Rounded affineCoefficient(uint16_t scale, uint16_t bias, int q) {
  SignedDyadic<10> exact; exact.addBF16(scale, q); exact.addBF16(bias, 1);
  return exact.rounded(-133);
}
inline bool centeredIdentity(uint16_t scale, uint16_t bias, int q) {
  SignedDyadic<10> original, centered;
  original.addBF16(scale, q); original.addBF16(bias, 1);
  centered.addBF16(scale, q - 8); centered.addBF16(scale, 8); centered.addBF16(bias, 1);
  return original.equals(centered);
}
inline Rounded correctedBiasResidual(float corrected, uint16_t scale, uint16_t bias) {
  // Common quantum2^-149 expresses F32 corrected bias and BF16 source terms
  // exactly. This catches a tiny original bias lost beside a large8*s, even
  // when an ordinary F64 subtraction would incorrectly report zero.
  if (!std::isfinite(corrected)) throw std::invalid_argument("W4A8 nonfinite corrected bias");
  SignedDyadic<10> exact;
  const uint32_t bits = std::bit_cast<uint32_t>(corrected), exponent = (bits >> 23) & 255u;
  const uint32_t mantissa = (bits & 0x007fffffu) + (exponent ? 0x00800000u : 0u);
  (bits & 0x80000000u ? exact.negative : exact.positive)
      .addShifted(mantissa, exponent ? exponent - 1 : 0u);
  for (const auto &[word, multiplier] : std::array<std::pair<uint16_t, int>, 2>{{{scale, -8}, {bias, -1}}}) {
    const auto p = parts(word);
    (p.negative != (multiplier < 0) ? exact.negative : exact.positive)
        .addShifted(p.mantissa * uint32_t(std::abs(multiplier)), p.shift + 16);
  }
  return exact.rounded(-149);
}
inline Rounded coefficientResidual(uint16_t rounded, uint16_t scale, uint16_t bias, int q) {
  SignedDyadic<10> exact;
  exact.addBF16(rounded, 1); exact.addBF16(scale, -q); exact.addBF16(bias, -1);
  return exact.rounded(-133);
}
inline double subtractionEnvelope(double a, double b) {
  return addUp(mulUp(kUnit64, addUp(std::abs(a), std::abs(b))),
      std::numeric_limits<double>::denorm_min());
}
// Value/error propagation models separate correctly rounded F32 stages with
// optional flushing of subnormal INPUTS and OUTPUTS at every stage. It includes
// outward-rounded F64 bookkeeping errors. All actual stages must remain finite.
struct Bounded { double value = 0, error = 0; };
inline double inputError(Bounded a) {
  const double magnitude = std::abs(a.value);
  if (!a.error && magnitude >= kFlush32) return 0;
  const double lower = std::nextafter(magnitude - a.error, -std::numeric_limits<double>::infinity());
  if (lower >= kFlush32) return a.error;
  return addUp(a.error, std::min(kFlush32, addUp(magnitude, a.error)));
}
inline Bounded multiply(Bounded a, Bounded b) {
  const double value = a.value * b.value, ea = inputError(a), eb = inputError(b);
  const double propagated = addUp(addUp(mulUp(std::abs(a.value), eb),
      mulUp(std::abs(b.value), ea)), mulUp(ea, eb));
  const double norm = mulUp(std::abs(a.value), std::abs(b.value));
  const double reference = addUp(mulUp(kUnit64, norm), std::numeric_limits<double>::denorm_min());
  const double rounding = addUp(mulUp(kUnit32, addUp(norm, propagated)), kFlush32);
  return {value, addUp(addUp(propagated, rounding), reference)};
}
inline Bounded add(Bounded a, Bounded b) {
  const double value = a.value + b.value;
  const double propagated = addUp(inputError(a), inputError(b));
  const double norm = addUp(std::abs(a.value), std::abs(b.value));
  const double reference = addUp(mulUp(kUnit64, norm), std::numeric_limits<double>::denorm_min());
  const double rounding = addUp(mulUp(kUnit32,
      addUp(std::abs(value), addUp(propagated, reference))), kFlush32);
  return {value, addUp(addUp(propagated, rounding), reference)};
}
} // namespace detail

struct QuantizationReport {
  uint64_t rows = 0, values = 0, sourceNonfinite = 0, zeroRows = 0;
  uint64_t codesOutsideRange = 0, scaleFaults = 0, scaleBitMismatches = 0, rneFaults = 0;
  uint64_t rneBoundaryAmbiguities = 0, subnormalScaleFlushes = 0, subnormalInputFlushes = 0;
  double maximumAbsoluteError = 0, maximumHalfStepViolation = 0;
  bool pass() const { return !sourceNonfinite && !codesOutsideRange && !scaleFaults && !rneFaults; }
  void write(std::ostream &out) const {
    out << "{\"pass\":" << (pass() ? "true" : "false") << ",\"rows\":" << rows
        << ",\"values\":" << values << ",\"source_nonfinite\":" << sourceNonfinite
        << ",\"codes_outside_range\":" << codesOutsideRange << ",\"scale_faults\":" << scaleFaults
        << ",\"scale_bit_mismatches\":" << scaleBitMismatches << ",\"rne_faults\":" << rneFaults
        << ",\"rne_boundary_ambiguities\":" << rneBoundaryAmbiguities
        << ",\"subnormal_scale_flushes\":" << subnormalScaleFlushes
        << ",\"subnormal_input_flushes\":" << subnormalInputFlushes
        << ",\"maximum_absolute_error\":"; writeNumber(out, maximumAbsoluteError);
    out << ",\"maximum_half_step_violation\":"; writeNumber(out, maximumHalfStepViolation);
    out << ",\"policy\":\"maxabs/127 F32 floor FLT_MIN; zero scale1; RNE clamp127; separately reported subnormal FTZ allowance\"}";
  }
};
inline QuantizationReport certifyQuantizedRows(const uint16_t *source, const int8_t *codes,
    const float *scales, uint32_t rows, uint32_t K) {
  if (!source || !codes || !scales || !rows || !K || K > kMaximumK)
    throw std::invalid_argument("W4A8 activation certificate geometry invalid");
  QuantizationReport r; r.rows = rows; r.values = uint64_t(rows) * K;
  for (uint32_t row = 0; row < rows; ++row) {
    float maximum = 0;
    for (uint32_t k = 0; k < K; ++k) {
      const auto word = source[uint64_t(row) * K + k];
      if (!sourceFinite(word)) ++r.sourceNonfinite;
      else maximum = std::max(maximum, std::abs(number(word)));
    }
    const float expected = rowScale(maximum), actualScale = scales[row];
    r.zeroRows += maximum == 0;
    if (!positive(actualScale)) { ++r.scaleFaults; continue; }
    const bool scaleFlush = maximum > 0 && maximum < std::numeric_limits<float>::min() && actualScale == 1;
    r.subnormalScaleFlushes += scaleFlush;
    r.scaleBitMismatches += std::bit_cast<uint32_t>(expected) != std::bit_cast<uint32_t>(actualScale);
    if ((!maximum && actualScale != 1) ||
        (!scaleFlush && std::abs(double(actualScale) - expected) > 3 * ulp(expected))) ++r.scaleFaults;
    for (uint32_t k = 0; k < K; ++k) {
      const auto at = uint64_t(row) * K + k; const int actualCode = codes[at];
      r.codesOutsideRange += actualCode < -127 || actualCode > 127;
      if (!sourceFinite(source[at])) continue;
      const float value = number(source[at]), ratio = roundedDivide(value, actualScale);
      const int expectedCode = roundEven(ratio);
      const bool inputFlush = value != 0 && std::abs(value) < std::numeric_limits<float>::min() && actualCode == 0;
      if (actualCode != expectedCode) {
        const double radius = powerOfTwo(actualScale) ? 0 : 3 * ulp(ratio);
        const int low = roundEven(double(ratio) - radius), high = roundEven(double(ratio) + radius);
        if (inputFlush) ++r.subnormalInputFlushes;
        else if (radius > 0 && actualCode >= std::min(low, high) && actualCode <= std::max(low, high))
          ++r.rneBoundaryAmbiguities;
        else ++r.rneFaults;
      }
      const double error = std::abs(double(value) - double(actualCode) * actualScale);
      r.maximumAbsoluteError = std::max(r.maximumAbsoluteError, error);
      const double violation = error - double(actualScale) * 0.5;
      r.maximumHalfStepViolation = std::max(r.maximumHalfStepViolation, violation);
      const double allowance = 4 * ulp(value) + 4 * ulp(actualScale) * 127 + (inputFlush ? kFlush32 : 0);
      if (violation > allowance) ++r.rneFaults;
    }
  }
  return r;
}

struct SourceRowView {
  std::span<const uint16_t> activationBf16;
  std::span<const uint8_t> originalPackedU4;
  std::span<const uint16_t> groupScaleBf16, groupBiasBf16;
  std::span<const uint8_t> centeredPackedI4{}; // Optional actual candidate bytes.
};
struct ObservedRowView {
  float activationScale = 0;
  std::span<const int8_t> activationI8;
  std::span<const int32_t> groupIntegerDot, groupIntegerSum;
  std::span<const float> correctedBiasF32{}; // Empty: expression bound, no capture claim.
  float linearOutputF32 = 0;
};
struct RowCertificate {
  uint32_t K = 0, groups = 0;
  uint64_t sourceNonfinite = 0, packedMismatches = 0, centeredIdentityMismatches = 0;
  uint64_t naiveFp64IdentityMismatches = 0, integerDotMismatches = 0, integerSumMismatches = 0;
  uint64_t correctedBiasFaults = 0, correctedBiasBitMismatches = 0, nonfiniteStages = 0;
  bool centeredPackedCaptured = false, correctedBiasCaptured = false;
  bool withinFloatingEnvelope = false, withinOriginalEnvelope = false;
  bool withinBf16CoefficientEnvelope = false, signReliable = false, signMatches = true;
  QuantizationReport quantization;
  double originalAffineF64 = 0, originalBf16CoefficientF64 = 0, centeredNominalF64 = 0;
  double activationQuantizationEnvelope = 0, biasAdjustmentEnvelope = 0;
  double floatingEnvelope = 0, referenceEnvelope = 0, totalEnvelope = 0;
  double coefficientBoundaryEnvelope = 0, bf16ReferenceEnvelope = 0;
  double absoluteErrorFromOriginal = 0, absoluteErrorFromNominal = 0, actualLinearF32 = 0;
  double expectedStagedLinearF32 = 0;
  bool stagedLinearBitsExact = false;
  bool pass() const {
    return quantization.pass() && !sourceNonfinite && !packedMismatches && !centeredIdentityMismatches
        && !integerDotMismatches && !integerSumMismatches && !correctedBiasFaults && !nonfiniteStages
        && withinFloatingEnvelope && withinOriginalEnvelope && withinBf16CoefficientEnvelope && signMatches;
  }
  void write(std::ostream &out) const {
    out << "{\"pass\":" << (pass() ? "true" : "false") << ",\"K\":" << K << ",\"groups\":" << groups
        << ",\"source_nonfinite\":" << sourceNonfinite << ",\"packed_mismatches\":" << packedMismatches
        << ",\"centered_identity_mismatches\":" << centeredIdentityMismatches
        << ",\"naive_fp64_identity_mismatches\":" << naiveFp64IdentityMismatches
        << ",\"integer_dot_mismatches\":" << integerDotMismatches << ",\"integer_sum_mismatches\":" << integerSumMismatches
        << ",\"corrected_bias_faults\":" << correctedBiasFaults
        << ",\"corrected_bias_bit_mismatches\":" << correctedBiasBitMismatches << ",\"nonfinite_stages\":" << nonfiniteStages
        << ",\"centered_packed_captured\":" << (centeredPackedCaptured ? "true" : "false")
        << ",\"corrected_bias_captured\":" << (correctedBiasCaptured ? "true" : "false")
        << ",\"within_floating_envelope\":" << (withinFloatingEnvelope ? "true" : "false")
        << ",\"within_original_envelope\":" << (withinOriginalEnvelope ? "true" : "false")
        << ",\"within_bf16_coefficient_envelope\":" << (withinBf16CoefficientEnvelope ? "true" : "false")
        << ",\"sign_reliable\":" << (signReliable ? "true" : "false") << ",\"sign_matches\":" << (signMatches ? "true" : "false");
    const auto field = [&](const char *name, double value) { out << ",\"" << name << "\":"; writeNumber(out, value); };
    field("original_affine_fp64", originalAffineF64); field("original_bf16_coefficient_fp64", originalBf16CoefficientF64);
    field("centered_nominal_fp64", centeredNominalF64); field("activation_quantization_envelope", activationQuantizationEnvelope);
    field("bias_adjustment_envelope", biasAdjustmentEnvelope); field("floating_envelope", floatingEnvelope);
    field("reference_envelope", referenceEnvelope); field("total_envelope", totalEnvelope);
    field("bf16_coefficient_boundary_envelope", coefficientBoundaryEnvelope); field("bf16_reference_envelope", bf16ReferenceEnvelope);
    field("absolute_error_from_original", absoluteErrorFromOriginal); field("absolute_error_from_nominal", absoluteErrorFromNominal);
    field("actual_linear_f32", actualLinearF32);
    field("expected_staged_linear_f32", expectedStagedLinearF32);
    out << ",\"staged_linear_bits_exact\":" << (stagedLinearBitsExact ? "true" : "false");
    out << ",\"quantization\":"; quantization.write(out);
    out << ",\"reference\":\"independent original BF16 dyadic affine dot with one FP64 RNE; absolute staged F32 input/output FTZ envelope\",\"model_quality_qualified\":false,\"bit_parity_claim\":false}";
  }
};

inline RowCertificate certifyRow(const SourceRowView &source, const ObservedRowView &observed) {
  const size_t K = source.activationBf16.size(), G = K / kGroupSize;
  if (!K || K > kMaximumK || K % kGroupSize || source.originalPackedU4.size() != K / 2
      || source.groupScaleBf16.size() != G || source.groupBiasBf16.size() != G
      || observed.activationI8.size() != K || observed.groupIntegerDot.size() != G
      || observed.groupIntegerSum.size() != G
      || (!source.centeredPackedI4.empty() && source.centeredPackedI4.size() != K / 2)
      || (!observed.correctedBiasF32.empty() && observed.correctedBiasF32.size() != G))
    throw std::invalid_argument("W4A8 row certificate view geometry invalid");
  RowCertificate r; r.K = uint32_t(K); r.groups = uint32_t(G); r.actualLinearF32 = observed.linearOutputF32;
  r.centeredPackedCaptured = !source.centeredPackedI4.empty(); r.correctedBiasCaptured = !observed.correctedBiasF32.empty();
  r.quantization = certifyQuantizedRows(source.activationBf16.data(), observed.activationI8.data(), &observed.activationScale, 1, uint32_t(K));
  for (size_t k = 0; k < K; ++k) r.sourceNonfinite += !sourceFinite(source.activationBf16[k]);
  for (size_t g = 0; g < G; ++g)
    r.sourceNonfinite += !sourceFinite(source.groupScaleBf16[g]) || !sourceFinite(source.groupBiasBf16[g]);
  if (r.centeredPackedCaptured)
    for (size_t byte = 0; byte < K / 2; ++byte)
      r.packedMismatches += source.centeredPackedI4[byte] != centeredByte(source.originalPackedU4[byte])
          || originalByte(source.centeredPackedI4[byte]) != source.originalPackedU4[byte];
  if (r.sourceNonfinite || !positive(observed.activationScale) || !std::isfinite(observed.linearOutputF32)) {
    r.nonfiniteStages += !std::isfinite(observed.linearOutputF32); return r;
  }
  detail::SignedDyadic<18> affineExact, bf16Exact;
  detail::Bounded accumulation{};
  float stagedUnscaled = 0;
  const double activationScale = observed.activationScale;
  for (uint32_t g = 0; g < G; ++g) {
    const uint16_t sWord = source.groupScaleBf16[g], bWord = source.groupBiasBf16[g];
    const float s = number(sWord), b = number(bWord);
    const auto expression = detail::add({double(b), 0}, detail::multiply({8, 0}, {double(s), 0}));
    const float eightScale = roundedMultiply(8, s), expectedBias = roundedAdd(b, eightScale);
    if (!std::isfinite(eightScale) || !std::isfinite(expectedBias)) { ++r.nonfiniteStages; continue; }
    const float correctedBias = r.correctedBiasCaptured ? observed.correctedBiasF32[g] : expectedBias;
    if (!std::isfinite(correctedBias)) { ++r.correctedBiasFaults; continue; }
    double biasUncertainty = 0;
    if (r.correctedBiasCaptured) {
      r.correctedBiasBitMismatches += std::bit_cast<uint32_t>(correctedBias) != std::bit_cast<uint32_t>(expectedBias);
      const double diagnostic = detail::subtractionEnvelope(double(correctedBias), expression.value);
      if (std::abs(double(correctedBias) - expression.value) > addUp(expression.error, diagnostic)) ++r.correctedBiasFaults;
    } else {
      // Missing probe: bound the source-declared expression, without asserting
      // that any particular GPU intermediate equals the host-computed value.
      biasUncertainty = addUp(expression.error,
          addUp(std::abs(double(expectedBias) - expression.value), detail::subtractionEnvelope(expectedBias, expression.value)));
    }
    int64_t dot = 0, sum = 0;
    for (uint32_t j = 0; j < kGroupSize; ++j) {
      const uint32_t k = g * kGroupSize + j;
      const int q = unsignedCode(source.originalPackedU4, k), centered = q - 8;
      const int a = observed.activationI8[k]; const double x = number(source.activationBf16[k]);
      dot += int64_t(a) * centered; sum += a;
      r.centeredIdentityMismatches += !detail::centeredIdentity(sWord, bWord, q);
      const auto coefficient = detail::affineCoefficient(sWord, bWord, q);
      const double naiveOriginal = double(q) * s + b;
      const double naiveCentered = double(centered) * s + (double(b) + 8 * double(s));
      r.naiveFp64IdentityMismatches += naiveOriginal != naiveCentered;
      affineExact.addProduct(source.activationBf16[k], sWord, q);
      affineExact.addProduct(source.activationBf16[k], bWord, 1);
      const double approximateA = double(a) * activationScale;
      const double errorA = addUp(std::abs(x - approximateA), detail::subtractionEnvelope(x, approximateA));
      const double coefficientMagnitude = addUp(std::abs(coefficient.value), coefficient.error);
      r.activationQuantizationEnvelope = addUp(r.activationQuantizationEnvelope, mulUp(errorA, coefficientMagnitude));
      const float coefficientProduct = roundedMultiply(float(q), s);
      const float reconstructed = roundedAdd(coefficientProduct, b);
      const uint16_t roundedWord = bf16(reconstructed);
      if (!std::isfinite(coefficientProduct) || !std::isfinite(reconstructed) || !sourceFinite(roundedWord)) {
        ++r.nonfiniteStages; continue;
      }
      bf16Exact.addProduct(source.activationBf16[k], roundedWord, 1);
      const auto exactResidual = detail::coefficientResidual(roundedWord, sWord, bWord, q);
      const double residual = addUp(std::abs(exactResidual.value), exactResidual.error);
      r.coefficientBoundaryEnvelope = addUp(r.coefficientBoundaryEnvelope, mulUp(std::abs(x), residual));
    }
    r.integerDotMismatches += std::abs(dot) > kMaximumGroupDotMagnitude || observed.groupIntegerDot[g] != dot;
    r.integerSumMismatches += std::abs(sum) > kMaximumGroupSumMagnitude || observed.groupIntegerSum[g] != sum;
    const auto exactResidual = detail::correctedBiasResidual(correctedBias, sWord, bWord);
    const double biasResidual = addUp(std::abs(exactResidual.value), exactResidual.error);
    r.biasAdjustmentEnvelope = addUp(r.biasAdjustmentEnvelope,
        mulUp(activationScale, mulUp(std::abs(double(sum)), biasResidual)));
    const auto scaleDot = detail::multiply({double(s), 0}, {double(dot), 0});
    const auto biasDot = detail::multiply({double(correctedBias), biasUncertainty}, {double(sum), 0});
    const auto group = detail::add(scaleDot, biasDot);
    accumulation = detail::add(accumulation, group);
    const float fScaleDot = roundedMultiply(s, float(dot));
    const float fBiasDot = roundedMultiply(correctedBias, float(sum));
    const float fGroup = roundedAdd(fScaleDot, fBiasDot);
    stagedUnscaled = roundedAdd(stagedUnscaled, fGroup);
    if (!std::isfinite(fScaleDot) || !std::isfinite(fBiasDot) || !std::isfinite(fGroup) || !std::isfinite(stagedUnscaled)) ++r.nonfiniteStages;
  }
  accumulation = detail::multiply(accumulation, {activationScale, 0});
  const float stagedScaled = roundedMultiply(stagedUnscaled, observed.activationScale);
  r.expectedStagedLinearF32 = stagedScaled;
  r.stagedLinearBitsExact = std::bit_cast<uint32_t>(stagedScaled) == std::bit_cast<uint32_t>(observed.linearOutputF32);
  r.nonfiniteStages += !std::isfinite(stagedScaled);
  const auto original = affineExact.rounded(-266), bf16Reference = bf16Exact.rounded(-266);
  r.originalAffineF64 = original.value; r.originalBf16CoefficientF64 = bf16Reference.value;
  r.referenceEnvelope = original.error; r.bf16ReferenceEnvelope = bf16Reference.error;
  r.centeredNominalF64 = accumulation.value; r.floatingEnvelope = accumulation.error;
  const double diagnostic = detail::subtractionEnvelope(r.actualLinearF32, r.originalAffineF64);
  r.totalEnvelope = addUp(addUp(r.activationQuantizationEnvelope, r.biasAdjustmentEnvelope),
      addUp(r.floatingEnvelope, addUp(r.referenceEnvelope, diagnostic)));
  r.absoluteErrorFromOriginal = std::abs(r.actualLinearF32 - r.originalAffineF64);
  r.absoluteErrorFromNominal = std::abs(r.actualLinearF32 - r.centeredNominalF64);
  r.withinFloatingEnvelope = std::isfinite(r.floatingEnvelope) && r.absoluteErrorFromNominal <=
      addUp(r.floatingEnvelope, detail::subtractionEnvelope(r.actualLinearF32, r.centeredNominalF64));
  r.withinOriginalEnvelope = std::isfinite(r.totalEnvelope) && r.absoluteErrorFromOriginal <= r.totalEnvelope;
  const double bf16Diagnostic = detail::subtractionEnvelope(r.actualLinearF32, r.originalBf16CoefficientF64);
  const double bf16Envelope = addUp(r.totalEnvelope,
      addUp(r.coefficientBoundaryEnvelope, addUp(r.bf16ReferenceEnvelope, bf16Diagnostic)));
  r.withinBf16CoefficientEnvelope = std::isfinite(bf16Envelope) &&
      std::abs(r.actualLinearF32 - r.originalBf16CoefficientF64) <= bf16Envelope;
  r.signReliable = std::abs(r.originalAffineF64) > r.totalEnvelope;
  r.signMatches = !r.signReliable || (r.actualLinearF32 != 0 && std::signbit(r.actualLinearF32) == std::signbit(r.originalAffineF64));
  return r;
}

struct OutputMetrics {
  uint64_t values = 0, nonfinite = 0, absoluteEnvelopeFaults = 0, reliableSignFaults = 0;
  double relativeL2 = 0, cosine = 1, referenceNorm = 0, errorNorm = 0, envelopeNorm = 0;
  bool nearZeroReference = false, cosineDefined = false;
  bool pass() const {
    return values && !nonfinite && !absoluteEnvelopeFaults && !reliableSignFaults &&
        (nearZeroReference || (cosineDefined && relativeL2 <= kMaximumProducerRelativeL2 && cosine >= kMinimumProducerCosine));
  }
  void write(std::ostream &out) const {
    out << "{\"pass\":" << (pass() ? "true" : "false") << ",\"values\":" << values
        << ",\"nonfinite\":" << nonfinite << ",\"absolute_envelope_faults\":" << absoluteEnvelopeFaults
        << ",\"reliable_sign_faults\":" << reliableSignFaults << ",\"relative_l2\":"; writeNumber(out, relativeL2);
    out << ",\"cosine\":"; writeNumber(out, cosine);
    out << ",\"reference_norm\":"; writeNumber(out, referenceNorm);
    out << ",\"error_norm\":"; writeNumber(out, errorNorm);
    out << ",\"envelope_norm\":"; writeNumber(out, envelopeNorm);
    out << ",\"near_zero_reference\":" << (nearZeroReference ? "true" : "false")
        << ",\"cosine_defined\":" << (cosineDefined ? "true" : "false")
        << ",\"preregistered_maximum_relative_l2\":0.03,\"preregistered_minimum_cosine\":0.9995,\"model_quality_qualified\":false}";
  }
};
inline OutputMetrics compareOutputs(std::span<const float> actual, std::span<const double> reference,
    std::span<const double> absoluteEnvelopes) {
  if (actual.empty() || actual.size() != reference.size() || actual.size() != absoluteEnvelopes.size())
    throw std::invalid_argument("W4A8 output metrics geometry invalid");
  OutputMetrics r; r.values = actual.size(); double dot = 0, actualNorm = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double a = actual[i], b = reference[i], envelope = absoluteEnvelopes[i];
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(envelope) || envelope < 0) { ++r.nonfinite; continue; }
    const double error = std::abs(a - b), allowance = addUp(envelope, detail::subtractionEnvelope(a, b));
    r.absoluteEnvelopeFaults += error > allowance;
    r.reliableSignFaults += std::abs(b) > allowance && (a == 0 || std::signbit(a) != std::signbit(b));
    r.referenceNorm = std::hypot(r.referenceNorm, b); r.errorNorm = std::hypot(r.errorNorm, error);
    r.envelopeNorm = std::hypot(r.envelopeNorm, envelope); actualNorm = std::hypot(actualNorm, a);
  }
  r.relativeL2 = r.referenceNorm ? r.errorNorm / r.referenceNorm
      : (r.errorNorm ? std::numeric_limits<double>::infinity() : 0);
  // Prespecified representational floor only. A large error envelope must not
  // turn an ordinary nonzero reference into a waived relative-metric case.
  r.nearZeroReference = r.referenceNorm <= std::sqrt(double(r.values)) * kFlush32;
  r.cosineDefined = r.referenceNorm > 0 && actualNorm > 0;
  if (r.cosineDefined) {
    for (size_t i = 0; i < actual.size(); ++i)
      if (std::isfinite(actual[i]) && std::isfinite(reference[i]))
        dot += (double(actual[i]) / actualNorm) * (reference[i] / r.referenceNorm);
    r.cosine = std::clamp(dot, -1.0, 1.0);
  } else r.cosine = r.nearZeroReference ? 1 : 0;
  return r;
}

inline void cpuSelfTest() {
  static_assert(kMaximumGroupDotMagnitude == 65024 && kMaximumGroupSumMagnitude == 8128);
  const auto require = [](bool ok) { if (!ok) throw std::logic_error("W4A8 independent CPU precision self-test failed"); };
  require(std::fegetround() == FE_TONEAREST);
  for (uint32_t byte = 0; byte < 256; ++byte) {
    const std::array<uint8_t, 1> original{uint8_t(byte)}, centered{centeredByte(uint8_t(byte))};
    require(originalByte(centered[0]) == original[0]);
    require(signedCode(centered, 0) == unsignedCode(original, 0) - 8);
    require(signedCode(centered, 1) == unsignedCode(original, 1) - 8);
  }
  for (const auto &[value, code] : std::array<std::pair<double, int>, 8>{{
      {.5, 0}, {1.5, 2}, {2.5, 2}, {3.5, 4}, {-.5, 0}, {-1.5, -2}, {-2.5, -2}, {-3.5, -4}}})
    require(roundEven(value) == code);
  require(rowScale(0) == 1 && rowScale(std::numeric_limits<float>::denorm_min()) == std::numeric_limits<float>::min());
  require(roundedAdd(1, 0x1p-24f) == 1 && roundedAdd(8, 0x1p-21f) == 8);
  require(std::bit_cast<uint32_t>(roundedAdd(8, 3 * 0x1p-21f)) == std::bit_cast<uint32_t>(8.0f) + 2);
  std::array<uint16_t, 8> ties{bf16(127), bf16(-127), bf16(.5f), bf16(1.5f), bf16(-.5f), bf16(-1.5f), 0, 0};
  std::array<int8_t, 8> tieCodes{127, -127, 0, 2, 0, -2, 0, 0}; float tieScale = 1;
  require(certifyQuantizedRows(ties.data(), tieCodes.data(), &tieScale, 1, 8).pass());
  tieCodes[2] = 1; require(!certifyQuantizedRows(ties.data(), tieCodes.data(), &tieScale, 1, 8).pass());
  std::array<uint16_t, 2> subnormal{uint16_t(1), uint16_t(0x807f)};
  std::array<int8_t, 2> subnormalCodes{0, -1}; float subnormalScale = std::numeric_limits<float>::min();
  require(certifyQuantizedRows(subnormal.data(), subnormalCodes.data(), &subnormalScale, 1, 2).pass());
  subnormalCodes[1] = 0;
  require(certifyQuantizedRows(subnormal.data(), subnormalCodes.data(), &subnormalScale, 1, 2).pass());
  float flushedScale = 1;
  require(certifyQuantizedRows(subnormal.data(), subnormalCodes.data(), &flushedScale, 1, 2).pass());
  uint32_t random = 0x01234567u;
  const auto next = [&]() { random = random * 1664525u + 1013904223u; return random; };
  const auto exercise = [&](uint32_t K, uint32_t pattern) {
    const uint32_t G = K / 64;
    std::vector<uint16_t> x(K), s(G), b(G); std::vector<uint8_t> packed(K / 2), centered(K / 2);
    std::vector<int8_t> a(K); std::vector<int32_t> dots(G), sums(G); std::vector<float> biases(G);
    for (uint32_t k = 0; k < K; ++k) {
      if (pattern == 0) x[k] = bf16(std::ldexp(float(int(next() % 255) - 127), -5));
      if (pattern == 1) x[k] = bf16(k & 1 ? -127.0f : 127.0f);
      if (pattern == 2) x[k] = 0;
      if (pattern == 3) x[k] = uint16_t((k & 1 ? 0x8000u : 0u) | (k % 127 + 1));
      if (pattern == 4) x[k] = bf16(1);
    }
    float maximum = 0; for (uint16_t word : x) maximum = std::max(maximum, std::abs(number(word)));
    const float scale = rowScale(maximum);
    for (uint32_t k = 0; k < K; ++k) a[k] = int8_t(roundEven(roundedDivide(number(x[k]), scale)));
    for (uint32_t byte = 0; byte < K / 2; ++byte) {
      packed[byte] = pattern == 4 ? 0 : uint8_t(next()); centered[byte] = centeredByte(packed[byte]);
    }
    float output = 0;
    for (uint32_t g = 0; g < G; ++g) {
      s[g] = pattern == 4 ? bf16(0x1p30f) : bf16(std::ldexp(float(int(next() % 255) - 127), -10));
      b[g] = pattern == 4 ? bf16(1) : bf16(std::ldexp(float(int(next() % 255) - 127), -8));
      biases[g] = roundedAdd(number(b[g]), roundedMultiply(8, number(s[g])));
      for (uint32_t j = 0; j < 64; ++j) {
        const uint32_t k = g * 64 + j; dots[g] += int(a[k]) * signedCode(centered, k); sums[g] += a[k];
      }
      const float term = roundedAdd(roundedMultiply(number(s[g]), float(dots[g])),
          roundedMultiply(biases[g], float(sums[g])));
      output = roundedAdd(output, term);
    }
    output = roundedMultiply(output, scale);
    const SourceRowView source{x, packed, s, b, centered};
    const ObservedRowView observed{scale, a, dots, sums, biases, output};
    const auto certificate = certifyRow(source, observed); require(certificate.pass());
    if (pattern == 4) require(certificate.naiveFp64IdentityMismatches == 0 && certificate.biasAdjustmentEnvelope > 0);
    auto missingBias = observed; missingBias.correctedBiasF32 = {};
    require(certifyRow(source, missingBias).pass() && !certifyRow(source, missingBias).correctedBiasCaptured);
    ++dots[0]; require(!certifyRow(source, observed).pass()); --dots[0];
    ++sums[0]; require(!certifyRow(source, observed).pass()); --sums[0];
    centered[0] ^= 1; require(!certifyRow(source, observed).pass()); centered[0] ^= 1;
    auto invalidOutput = observed;
    invalidOutput.linearOutputF32 = float(double(output) + std::max(1.0, 4 * certificate.totalEnvelope));
    require(!certifyRow(source, invalidOutput).pass());
  };
  for (uint32_t K : {64u, 640u, 2560u}) for (uint32_t pattern = 0; pattern < 5; ++pattern) exercise(K, pattern);
  require(detail::centeredIdentity(bf16(0x1p80f), bf16(1), 0));
  const auto tinyOriginal = detail::affineCoefficient(bf16(0x1p80f), bf16(1), 0);
  require(tinyOriginal.value == 1 && double(-8) * 0x1p80 + (1 + 8 * 0x1p80) == 0);
  const auto tinyBias = detail::correctedBiasResidual(8, bf16(1), bf16(0x1p-60f));
  require(tinyBias.value == -0x1p-60 && tinyBias.error == 0);
  // A mandatory raw-integer probe distinguishes output errors from centering,
  // dot/sum errors and source-byte faults. Extrema, S0/Dnonzero, and overflow
  // cases are independent of the random fixture generator above.
  std::array<uint16_t, 64> extremeA; extremeA.fill(bf16(127));
  std::array<int8_t, 64> extremeQ; extremeQ.fill(127);
  std::array<uint8_t, 32> extremePacked{}, extremeCentered; extremeCentered.fill(0x88);
  std::array<uint16_t, 1> extremeS{bf16(1)}, extremeB{bf16(-8)};
  std::array<int32_t, 1> extremeD{-65024}, extremeSum{8128};
  std::array<float, 1> extremeBias{0};
  SourceRowView extremeSource{extremeA, extremePacked, extremeS, extremeB, extremeCentered};
  ObservedRowView extremeObserved{1, extremeQ, extremeD, extremeSum, extremeBias, -65024};
  require(certifyRow(extremeSource, extremeObserved).pass());
  extremeA.fill(bf16(-127)); extremeQ.fill(-127); extremeD[0] = 65024; extremeSum[0] = -8128;
  extremeObserved.linearOutputF32 = 65024; require(certifyRow(extremeSource, extremeObserved).pass());
  for (uint32_t k = 0; k < 64; ++k) {
    extremeA[k] = bf16(k & 1 ? -127.0f : 127.0f); extremeQ[k] = k & 1 ? -127 : 127;
  }
  extremePacked.fill(0xf0); extremeCentered.fill(0x78); extremeS[0] = bf16(1); extremeB[0] = 0;
  extremeD[0] = -60960; extremeSum[0] = 0; extremeBias[0] = 8;
  extremeObserved.linearOutputF32 = -60960; require(certifyRow(extremeSource, extremeObserved).pass());
  // Subnormal source-scale DAZ stress: exact original result is normal, even
  // though the source-scale word itself is subnormal. The envelope propagates
  // this input flush through D and S rather than adding only one finalFLT_MIN.
  extremeA.fill(bf16(127)); extremeQ.fill(127); extremePacked.fill(0xff); extremeCentered.fill(0x77);
  extremeS[0] = 0x0040; extremeB[0] = 0; extremeD[0] = 56896; extremeSum[0] = 8128;
  extremeBias[0] = 0; extremeObserved.linearOutputF32 = 0;
  const auto daz = certifyRow(extremeSource, extremeObserved);
  require(daz.pass() && daz.originalAffineF64 > kFlush32 && daz.absoluteErrorFromOriginal > kFlush32);
  extremeS[0] = bf16(1); extremeB[0] = bf16(0x1p-60f); extremeA.fill(bf16(1));
  extremeQ.fill(127); extremePacked.fill(0); extremeCentered.fill(0x88);
  extremeD[0] = -65024; extremeSum[0] = 8128; extremeBias[0] = 8;
  extremeObserved.activationScale = rowScale(1); extremeObserved.linearOutputF32 = 0;
  const auto cancellation = certifyRow(extremeSource, extremeObserved);
  require(cancellation.pass() && cancellation.naiveFp64IdentityMismatches == 64
      && cancellation.originalAffineF64 == 0x1p-54 && cancellation.biasAdjustmentEnvelope > 0);
  extremeS[0] = 0x7f7f; extremeB[0] = 0;
  require(!certifyRow(extremeSource, extremeObserved).pass()); //8*s stage overflow.
  extremeS[0] = bf16(1); extremeBias[0] = std::numeric_limits<float>::infinity();
  require(!certifyRow(extremeSource, extremeObserved).pass());
  extremeBias[0] = 8; extremeObserved.linearOutputF32 = std::numeric_limits<float>::quiet_NaN();
  require(!certifyRow(extremeSource, extremeObserved).pass());
  extremeObserved.linearOutputF32 = 0; extremeA[0] = 0x7f80;
  require(!certifyRow(extremeSource, extremeObserved).pass());
  const std::array<double, 3> reference{1, -2, 3}, envelopes{.02, .02, .02};
  const std::array<float, 3> good{1.01f, -2.01f, 3.01f}, bad{-1, -2, 3};
  require(compareOutputs(good, reference, envelopes).pass()); require(!compareOutputs(bad, reference, envelopes).pass());
  const std::array<double, 2> zeroReference{}, zeroEnvelope{0x1p-120, 0x1p-120};
  const std::array<float, 2> zeroActual{}; require(compareOutputs(zeroActual, zeroReference, zeroEnvelope).pass());
  const std::array<double, 1> ordinaryReference{1}, excessivelyLooseEnvelope{1000};
  const std::array<float, 1> ordinaryBad{0};
  require(!compareOutputs(ordinaryBad, ordinaryReference, excessivelyLooseEnvelope).pass());
}
} // namespace splash::bench::w4a8
