#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace splash::flash::prefill4k_int8columns {
inline double upward(double x) { return std::nextafter(x, std::numeric_limits<double>::infinity()); }
inline double addUp(double a, double b) { return upward(a + b); }
inline double mulUp(double a, double b) { return upward(a * b); }
inline double gammaUp(uint32_t n, double unit) {
  const double product = mulUp(double(n), unit);
  return upward(product / std::nextafter(1 - product, -std::numeric_limits<double>::infinity()));
}
inline double underflowUp(uint32_t n, double unit, double eta) {
  const double product = mulUp(double(n), unit);
  return upward(mulUp(double(n), eta) /
      std::nextafter(1 - product, -std::numeric_limits<double>::infinity()));
}
inline double bfNumber(uint16_t word) { return double(std::bit_cast<float>(uint32_t(word) << 16)); }

namespace detail {
// BF16*I8 is an integer multiple of 2^-133. At K<=2560 its magnitude needs
// <=280 bits; multiplication by the stored F32 mantissa needs <=304 bits.
// Exact integer arithmetic therefore fits these ten base-2^32 words.
struct UInt320 {
  std::array<uint32_t, 10> words{};
  void addWord(size_t i, uint32_t value) {
    uint64_t carry = value;
    while (carry) {
      if (i == words.size()) throw std::overflow_error("INT8 certificate dyadic accumulator overflow");
      const uint64_t sum = uint64_t(words[i]) + carry;
      words[i++] = uint32_t(sum); carry = sum >> 32;
    }
  }
  void addShifted(uint32_t value, uint32_t shift) {
    const uint64_t shifted = uint64_t(value) << (shift % 32);
    addWord(shift / 32, uint32_t(shifted));
    addWord(shift / 32 + 1, uint32_t(shifted >> 32));
  }
  void add(const UInt320 &other) {
    uint64_t carry = 0;
    for (size_t i = 0; i < words.size(); ++i) {
      const uint64_t sum = uint64_t(words[i]) + other.words[i] + carry;
      words[i] = uint32_t(sum); carry = sum >> 32;
    }
    if (carry) throw std::overflow_error("INT8 certificate dyadic norm overflow");
  }
  void multiply(uint32_t value) {
    uint64_t carry = 0;
    for (auto &word : words) {
      const uint64_t product = uint64_t(word) * value + carry;
      word = uint32_t(product); carry = product >> 32;
    }
    if (carry) throw std::overflow_error("INT8 certificate scaled dyadic overflow");
  }
  int compare(const UInt320 &other) const {
    for (size_t i = words.size(); i-- > 0;)
      if (words[i] != other.words[i]) return words[i] > other.words[i] ? 1 : -1;
    return 0;
  }
  UInt320 subtract(const UInt320 &other) const {
    UInt320 result; uint64_t borrow = 0;
    for (size_t i = 0; i < words.size(); ++i) {
      const uint64_t sub = uint64_t(other.words[i]) + borrow;
      result.words[i] = uint32_t(uint64_t(words[i]) - sub);
      borrow = uint64_t(words[i]) < sub;
    }
    if (borrow) throw std::logic_error("INT8 certificate negative unsigned subtraction");
    return result;
  }
  bool bit(uint32_t i) const { return (words.at(i / 32) >> (i % 32)) & 1; }
  bool anyBelow(uint32_t exclusive) const {
    for (uint32_t i = 0; i < exclusive / 32; ++i) if (words[i]) return true;
    const uint32_t remainder = exclusive % 32;
    return remainder && (words[exclusive / 32] & ((uint32_t(1) << remainder) - 1));
  }
  int highestBit() const {
    for (size_t i = words.size(); i-- > 0;)
      if (words[i]) return int(i * 32 + 31 - std::countl_zero(words[i]));
    return -1;
  }
};
struct Rounded { double value = 0, error = 0; };
inline Rounded convert(const UInt320 &integer, int exponent, bool roundUp = false) {
  const int highest = integer.highestBit();
  if (highest < 0) return {};
  const uint32_t shift = highest > 52 ? uint32_t(highest - 52) : 0;
  uint64_t top = 0;
  for (uint32_t bit = 0; bit < 53; ++bit)
    if (integer.bit(shift + bit)) top |= uint64_t(1) << bit;
  const bool discarded = integer.anyBelow(shift);
  if (roundUp ? discarded : (shift && integer.bit(shift - 1) &&
      (integer.anyBelow(shift - 1) || (top & 1)))) ++top;
  // top (including a carry to 2^53) is exact in FP64; ldexp stays normal and
  // finite for every admitted BF16/I8/F32 input and legal K in this header.
  return {std::ldexp(double(top), exponent + int(shift)),
      !roundUp && discarded ? std::ldexp(1.0, exponent + int(shift) - 1) : 0.0};
}
} // namespace detail

struct Certificate {
  double dotScaled = 0;              // correctly RNE-rounded exact sum(x*code*storedScale)
  double norm = 0, dotNorm = 0;      // upward scaled and unscaled exact L1 norms
  double candidateEnvelope = 0;     // raw F32 output versus the exact-real representation
  double referenceEnvelope = 0;     // one FP64 reference conversion, <=half ULP (zero if exact)
  double diagnosticEnvelope = 0;    // reference plus modeled-output FP64 subtraction allowance
};

// Independent source-byte contract: signed I8 includes -128; BF16 words are
// promoted exactly; stored positive F32 row scale is never fitted/requantized.
// Bounds assume finite correctly rounded F32 accumulation/scaling with gradual
// underflow. This CPU certificate does not establish MPP's actual arithmetic.
inline Certificate certificate(const int8_t *codes, uint32_t K, float scale, const uint16_t *input) {
  if (!codes || !input || (K != 640 && K != 2560))
    throw std::invalid_argument("INT8 certificate requires K640/K2560 and readable inputs");
  if (!std::isfinite(scale) || !(scale > 0))
    throw std::invalid_argument("INT8 certificate requires a finite positive stored F32 scale");
  detail::UInt320 positive, negative;
  for (uint32_t k = 0; k < K; ++k) {
    const uint16_t word = input[k];
    const uint32_t exponent = (word >> 7) & 255, fraction = word & 127;
    if (exponent == 255) throw std::invalid_argument("INT8 certificate nonfinite BF16 input");
    const int code = codes[k];
    const uint32_t mantissa = exponent ? fraction + 128 : fraction;
    const uint32_t magnitude = mantissa * uint32_t(code < 0 ? -code : code);
    if (!magnitude) continue;
    auto &sum = (bool(word & 0x8000) != (code < 0)) ? negative : positive;
    sum.addShifted(magnitude, exponent ? exponent - 1 : 0);
  }
  const bool negativeDot = positive.compare(negative) < 0;
  auto dot = negativeDot ? negative.subtract(positive) : positive.subtract(negative);
  auto norm = positive; norm.add(negative);
  Certificate result; result.dotNorm = detail::convert(norm, -133, true).value;
  const uint32_t scaleBits = std::bit_cast<uint32_t>(scale);
  const uint32_t scaleExponent = (scaleBits >> 23) & 255;
  const uint32_t scaleMantissa = (scaleBits & 0x7fffff) + (scaleExponent ? 0x800000 : 0);
  const int dyadicExponent = -133 + (scaleExponent ? int(scaleExponent) - 150 : -149);
  dot.multiply(scaleMantissa); norm.multiply(scaleMantissa);
  const auto reference = detail::convert(dot, dyadicExponent);
  result.dotScaled = negativeDot ? -reference.value : reference.value;
  result.referenceEnvelope = reference.error;
  result.norm = detail::convert(norm, dyadicExponent, true).value;
  constexpr double u = 0x1p-24, eta = 0x1p-150, u64 = 0x1p-53;
  constexpr double eta64 = std::numeric_limits<double>::denorm_min();
  const double s = double(scale);
  // Conservative 2K covers separate F32 product/add as well as FMA.
  const double dotError = addUp(mulUp(gammaUp(2 * K, u), result.dotNorm),
      underflowUp(2 * K, u, eta));
  result.candidateEnvelope = addUp(addUp(mulUp(mulUp(addUp(1, u), s), dotError),
      mulUp(u, result.norm)), eta);
  const double subtraction = addUp(mulUp(u64, addUp(addUp(result.norm,
      result.candidateEnvelope), std::abs(result.dotScaled))), eta64);
  result.diagnosticEnvelope = addUp(result.referenceEnvelope, subtraction);
  return result;
}

// Prefer observed got for the diagnostic bound. Compare ordinary
// abs(double(got)-c.dotScaled) <= comparisonEnvelope(got,c), and reject any
// nonfinite got first. These envelopes concern raw F32 captures, before BF16.
inline double comparisonEnvelope(double got, const Certificate &c) {
  if (!std::isfinite(got)) throw std::invalid_argument("INT8 certificate nonfinite raw F32 output");
  constexpr double u64 = 0x1p-53, eta64 = std::numeric_limits<double>::denorm_min();
  const double subtraction = addUp(mulUp(u64, addUp(std::abs(got), std::abs(c.dotScaled))), eta64);
  return addUp(c.candidateEnvelope, addUp(c.referenceEnvelope, subtraction));
}

// Tiny standalone CPU gate; no Metal/framework/model dependency.
inline void selfTest() {
  std::array<int8_t, 640> codes{}; std::array<uint16_t, 640> input{};
  const auto bf = [](float x) { return uint16_t(std::bit_cast<uint32_t>(x) >> 16); };
  const auto require = [](bool pass) { if (!pass) throw std::runtime_error("INT8 dyadic certificate self-test failed"); };
  input[0] = bf(1.5f); codes[0] = -128;
  require(certificate(codes.data(), 640, 0.125f, input.data()).dotScaled == -24);
  input.fill(0); codes.fill(0);
  input[0] = bf(-0x1p93f); input[1] = bf(0x1p20f); input[2] = bf(0x1p-100f);
  input[3] = bf(0x1p100f); input[4] = bf(0x1p20f);
  codes[0] = -128; codes[1] = 1; codes[2] = 1; codes[3] = -1; codes[4] = -1;
  const auto cancelled = certificate(codes.data(), 640, 0x1p-20f, input.data());
  require(cancelled.dotScaled == 0x1p-120 && cancelled.referenceEnvelope == 0);
  input.fill(0); codes.fill(0); input[0] = bf(0x1p60f); input[1] = bf(128); codes[0] = codes[1] = 1;
  const auto tie = certificate(codes.data(), 640, 1, input.data());
  require(tie.dotScaled == 0x1p60 && tie.referenceEnvelope == 128 && tie.norm == 0x1p60 + 256);
  input.fill(0); codes.fill(0); input[0] = 1; codes[0] = 1;
  require(certificate(codes.data(), 640, std::numeric_limits<float>::denorm_min(), input.data()).dotScaled == 0x1p-282);
  input.fill(0); input[0] = 0x8000;
  const auto zero = certificate(codes.data(), 640, 1, input.data());
  require(zero.dotScaled == 0 && zero.norm == 0);
  bool rejected = false;
  try { (void)certificate(codes.data(), 640, 0, input.data()); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected);
}
} // namespace splash::flash::prefill4k_int8columns
