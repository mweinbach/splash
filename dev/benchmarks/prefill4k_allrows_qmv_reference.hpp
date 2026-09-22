#pragma once
// Independent CPU F64 late-scale reference for Root's captured primitive dots.
// No Metal API. A numerical bound alone never qualifies the changed reducer.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace splash::flash::gathered_i8_qmv_reference {
inline float bf16Number(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }
inline uint16_t bf16FromF64(double value) {
  const uint64_t word = std::bit_cast<uint64_t>(value);
  const uint16_t sign = uint16_t(word >> 48) & 0x8000;
  if (std::isnan(value)) return sign | 0x7fc0;
  if (std::isinf(value)) return sign | 0x7f80;
  const double magnitude = std::abs(value);
  if (!magnitude) return sign;
  int exponent = 0;
  std::frexp(magnitude, &exponent);
  const int unbiased = exponent - 1;
  if (unbiased > 127) return sign | 0x7f80;
  const int quantum = unbiased < -126 ? -133 : unbiased - 7;
  const double units = std::ldexp(magnitude, -quantum);
  const double floorUnits = std::floor(units), remainder = units - floorUnits;
  uint32_t rounded = uint32_t(floorUnits);
  if (remainder > 0.5 || (remainder == 0.5 && (rounded & 1))) ++rounded;
  if (unbiased < -126) return sign | uint16_t(rounded);
  int biased = unbiased + 127;
  if (rounded == 256) { rounded = 128; ++biased; }
  if (biased >= 255) return sign | 0x7f80;
  return sign | uint16_t(biased << 7) | uint16_t(rounded - 128);
}
inline uint16_t orderedBF16(uint16_t bits) {
  return bits & 0x8000 ? uint16_t(~bits) : uint16_t(bits ^ 0x8000);
}
inline uint32_t bf16ULPs(uint16_t a, uint16_t b) {
  const uint16_t aa = orderedBF16(a), bb = orderedBF16(b);
  return aa > bb ? aa - bb : bb - aa;
}
inline uint32_t orderedF32(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  return word & 0x80000000 ? ~word : word ^ 0x80000000;
}
inline uint64_t f32ULPs(float a, float b) {
  const uint32_t aa = orderedF32(a), bb = orderedF32(b);
  return aa > bb ? uint64_t(aa) - bb : uint64_t(bb) - aa;
}
inline double gamma(uint32_t operations, double unitRoundoff) {
  const double nu = operations * unitRoundoff;
  if (nu >= 1) throw std::invalid_argument("accumulation gamma exceeds finite bound");
  return nu / (1 - nu);
}
struct CompensatedSum {
  double sum = 0, correction = 0;
  void add(double value) {
    const double next = sum + value;
    correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
    sum = next;
  }
  double value() const { return sum + correction; }
};
struct DotReference {
  double dot, sumAbsProducts, exactProductRoundingError, dotAbsoluteBound;
  double scaled, scaledAbsoluteBound;
  uint16_t bf16;
  uint32_t expectedStickyMinimum;
  bool nonfiniteInput, productOverflow, productSubnormal, scaledOverflow, scaledSubnormal;
  bool exceptional;
};
inline DotReference reference(std::span<const uint16_t> input,
                              std::span<const int8_t> codes, float rowScale,
                              uint32_t lanes = 32, bool opaqueMPPControl = false) {
  if (input.size() != codes.size() || (input.size() != 640 && input.size() != 2560) ||
      lanes != 32 || !std::isfinite(rowScale) || !(rowScale > 0))
    throw std::invalid_argument("F64 QMV reference requires K640/2560, lane32 and positive finite row scale");
  CompensatedSum dot, absolute, productError;
  bool nonfinite = false, overflow = false, subnormal = false;
  uint32_t diagnostics = 0;
  for (size_t k = 0; k < input.size(); ++k) {
    if (codes[k] == std::numeric_limits<int8_t>::min())
      throw std::invalid_argument("Full512 symmetric I8 reference excludes-128");
    float x = bf16Number(input[k]);
    if (!std::isfinite(x)) { x = 0; nonfinite = true; diagnostics |= 4; }
    // BF16 × symmetric I8 has <=15 significant bits and is normally exact F32.
    const double exactProduct = double(x) * double(codes[k]);
    const float roundedProduct = float(exactProduct);
    if (!std::isfinite(roundedProduct)) { overflow = true; diagnostics |= 4; }
    if (exactProduct && std::abs(exactProduct) < std::numeric_limits<float>::min()) subnormal = true;
    dot.add(exactProduct); absolute.add(std::abs(exactProduct));
    if (std::isfinite(roundedProduct)) productError.add(std::abs(double(roundedProduct) - exactProduct));
  }
  const double sumAbs = absolute.value(), exactDot = dot.value(), pe = productError.value();
  // Conservative without assuming the undocumented GPU simd_sum reduction tree:
  // ceil(K/32) lane additions plus at most31 reductions along a contribution.
  // The opaque MPP control does not inherit QMV's lane assignment. Its
  // separately reported conservative bound assumes at mostK F32 additions.
  const uint32_t operations = opaqueMPPControl ? uint32_t(input.size()) :
      uint32_t((input.size() + 31) / 32) + 31;
  constexpr double u32 = 0x1p-24, u64 = 0x1p-53;
  const double f64Uncertainty = gamma(uint32_t(input.size()) * 4, u64) * sumAbs;
  const double dotBound = pe + gamma(operations, u32) * (sumAbs + pe) + f64Uncertainty;
  const double scaled = exactDot * double(rowScale);
  const double scaledBound = double(rowScale) * dotBound +
      gamma(1, u32) * double(rowScale) * (std::abs(exactDot) + dotBound) +
      std::numeric_limits<float>::denorm_min();
  const bool scaledOverflow = std::abs(scaled) > std::numeric_limits<float>::max();
  const bool scaledSubnormal = scaled && std::abs(scaled) < std::numeric_limits<float>::min();
  if (scaledOverflow) diagnostics |= 4;
  return {exactDot, sumAbs, pe, dotBound, scaled, scaledBound, bf16FromF64(scaled), diagnostics,
      nonfinite, overflow, subnormal, scaledOverflow, scaledSubnormal,
      nonfinite || overflow || subnormal || scaledOverflow || scaledSubnormal};
}
struct CaptureReport {
  double dotAbsoluteError, scaledAbsoluteError, cancellationRatio;
  uint64_t f32DotULPs, f32ScaledULPs;
  uint32_t bf16ULPs;
  bool bf16Exact, signMatches, negativeZeroMatches;
  bool dotWithinBound, scaledWithinBound, strictSensitive, strictBF16Pass;
  bool diagnosticsCoverExpected, finiteCapture, regularFinitePrimitivePass;
};
inline CaptureReport assess(const DotReference &ref, float capturedDot,
                            float capturedScaled, uint16_t capturedBF16,
                            uint32_t diagnostics) {
  const bool finite = std::isfinite(capturedDot) && std::isfinite(capturedScaled) &&
      std::isfinite(bf16Number(capturedBF16));
  const double de = std::abs(double(capturedDot) - ref.dot), se = std::abs(double(capturedScaled) - ref.scaled);
  const bool dotBound = finite && de <= ref.dotAbsoluteBound;
  const bool scaledBound = finite && se <= ref.scaledAbsoluteBound;
  const bool sign = (capturedBF16 & 0x8000) == (ref.bf16 & 0x8000);
  const bool zeroSign = (capturedBF16 & 0x7fff) != 0 || (ref.bf16 & 0x7fff) != 0 || sign;
  const bool exact = capturedBF16 == ref.bf16;
  const bool sensitive = std::abs(ref.dot) <= 32 * ref.dotAbsoluteBound ||
      std::abs(ref.scaled) <= 32 * ref.scaledAbsoluteBound ||
      std::abs(ref.scaled) < std::numeric_limits<float>::min();
  const bool strict = exact && sign && zeroSign;
  const bool covered = (diagnostics & ref.expectedStickyMinimum) == ref.expectedStickyMinimum;
  // Exceptional fixtures always require explicitly paired diagnostic/bit review.
  // Even a regular finite primitive pass never declares model qualification.
  return {de, se, ref.sumAbsProducts ? std::abs(ref.dot) / ref.sumAbsProducts : 0,
      f32ULPs(capturedDot, float(ref.dot)), f32ULPs(capturedScaled, float(ref.scaled)),
      bf16ULPs(capturedBF16, ref.bf16), exact, sign, zeroSign, dotBound, scaledBound,
      sensitive, strict, covered, finite,
      !ref.exceptional && covered && dotBound && scaledBound && (!sensitive || strict)};
}
} // namespace splash::flash::gathered_i8_qmv_reference
