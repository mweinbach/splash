#pragma once

// Benchmark-only numerical evidence. This header never changes a runtime route.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace splash::flash::benchmark {
inline float number(uint16_t value) {
  return std::bit_cast<float>(uint32_t(value) << 16);
}
inline uint16_t bf16(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  if ((word & 0x7f800000u) == 0x7f800000u)
    return uint16_t((word >> 16) | ((word & 0x7fffffu) ? 0x40u : 0u));
  return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
inline uint32_t orderedBF16(uint16_t value) {
  return (value & 0x8000) ? uint32_t(0x8000 - (value & 0x7fff)) : uint32_t(0x8000 + value);
}
inline uint16_t fromOrderedBF16(uint32_t value) {
  return value < 0x8000 ? uint16_t(0x8000 | (0x8000 - value)) : uint16_t(value - 0x8000);
}
inline uint32_t bf16ULP(uint16_t a, uint16_t b) {
  const uint32_t x = orderedBF16(a), y = orderedBF16(b);
  return x > y ? x - y : y - x;
}
inline uint16_t bf16Double(double value) {
  if (std::isnan(value)) return 0x7fc0;
  if (value == 0) return std::signbit(value) ? 0x8000 : 0;
  const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  const double overflow = double(number(0x7f7f)) + std::ldexp(1.0, 119);
  if (std::abs(value) >= overflow) return uint16_t(sign | 0x7f80);
  // A F32 conversion only locates nearby candidates; choosing in double
  // avoids double-rounding a value very close to a BF16 midpoint.
  const uint32_t hint = std::clamp(orderedBF16(bf16(float(value))), 0x81u, 0xff7fu);
  uint16_t best = fromOrderedBF16(hint);
  double distance = std::abs(value - double(number(best)));
  for (int offset = -2; offset <= 2; ++offset) {
    const int64_t order = int64_t(hint) + offset;
    if (order < 0x81 || order > 0xff7f) continue;
    const uint16_t candidate = fromOrderedBF16(uint32_t(order));
    const double next = std::abs(value - double(number(candidate)));
    if (next < distance || (next == distance && !(candidate & 1u) && (best & 1u))) {
      best = candidate; distance = next;
    }
  }
  if ((best & 0x7fff) == 0) best = sign;
  return best;
}
inline double upward(double value) {
  return std::nextafter(value, std::numeric_limits<double>::infinity());
}
inline double gamma(uint32_t terms, int unitExponent) {
  const double nu = double(terms) * std::ldexp(1.0, unitExponent);
  if (!terms || !(nu < 1)) throw std::invalid_argument("invalid dot rounding bound domain");
  return upward(nu / (1 - nu));
}
inline double f32DotBound(uint32_t terms, double sumAbsoluteProducts,
                          double flushedInputProducts = 0) {
  if (!std::isfinite(sumAbsoluteProducts) || sumAbsoluteProducts < 0 ||
      !std::isfinite(flushedInputProducts) || flushedInputProducts < 0)
    throw std::invalid_argument("invalid dot reference product magnitude");
  const double f32Gamma = gamma(terms, -24), f64Gamma = gamma(terms, -53);
  // Standard deterministic gamma_K * sum|x*w| bound for arbitrary binary
  // F32 product/addition or FMA reduction. No probabilistic assumptions.
  // Reference L1 and double sum have their own gamma_K uncertainty. Include
  // conservative Metal F32 flush-to-zero input/intermediate uncertainty.
  // https://eprints.maths.manchester.ac.uk/2731/1/paper.pdf
  const double denominator = std::nextafter(1 - f64Gamma, 0.0);
  const double reduction = upward(upward(upward(f32Gamma + f64Gamma) *
      sumAbsoluteProducts) / denominator);
  const double inputFlush = upward(upward(1 + f32Gamma) * flushedInputProducts);
  const double underflow = upward(2 * double(terms) * std::numeric_limits<float>::min());
  return upward(upward(reduction + inputFlush) + underflow);
}
struct CellRelation final {
  uint32_t ulp = 0;
  double midpointDistance = 0;
  bool pass = false;
};
inline CellRelation cellRelation(uint16_t actual, uint16_t golden, double sum, double bound) {
  CellRelation result;
  result.ulp = bf16ULP(actual, golden);
  if (!std::isfinite(number(actual)) || !std::isfinite(number(golden)) ||
      !std::isfinite(sum) || !std::isfinite(bound) || bound < 0) return result;
  if (result.ulp == 0) { result.pass = true; return result; }
  if (result.ulp != 1) return result;
  const double sharedMidpoint = (double(number(actual)) + double(number(golden))) * .5;
  result.midpointDistance = std::abs(sum - sharedMidpoint);
  // This proves BF16 cell compatibility with a F32 reduction error bound,
  // not equality of hidden accumulator values or identical model generation.
  result.pass = result.midpointDistance <= bound;
  return result;
}
} // namespace splash::flash::benchmark
