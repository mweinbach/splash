#pragma once
// CPU-only independent references and frozen BF16 quality gates for the private
// singleton balanced-char4 R4 oracle. A bound is not model qualification, and it
// does not override the separately retained strict-sensitive rejection.
#include "../prefill4k_allrows_qmv_reference.hpp"
#include <array>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <vector>

namespace expert_r4_quality {
namespace frozen = splash::flash::gathered_i8_qmv_reference;
using FrozenDotReference = frozen::DotReference;
using DotReference = FrozenDotReference;
using CaptureReport = frozen::CaptureReport;

inline constexpr double kMaximumRelativeL2 = 1e-4;
inline constexpr double kMinimumCosine = .999999;
// Generic MSL permits RN or RTZ for F32 arithmetic. Use the larger RTZ bound;
// the unchanged CPU RN simulator is a diagnostic, not native-RN qualification.
inline constexpr double kGenericF32UnitRoundoff = 0x1p-23;

inline void validate(std::span<const uint16_t> input,
                     std::span<const int8_t> codes, float rowScale) {
  if (input.size() != codes.size() ||
      (input.size() != 640 && input.size() != 2560) ||
      !std::isfinite(rowScale) || !(rowScale > 0))
    throw std::invalid_argument("GEMV reference requires K640/2560 and a positive finite row scale");
  if (std::find(codes.begin(), codes.end(), std::numeric_limits<int8_t>::min()) != codes.end())
    throw std::invalid_argument("GEMV symmetric I8 reference excludes -128");
}

inline uint32_t reducerDepth(size_t k, uint32_t lanes) {
  if ((k != 640 && k != 2560) || lanes != 32)
    throw std::invalid_argument("R4 reducer depth requires K640/2560 and lane32");
  // Two balanced char4 additions on a contribution path, one lane-accumulator
  // addition per chunk, and the fixed five-stage descending XOR32 tree.
  return uint32_t((k + 127) / 128) + 2 + 5;
}

inline uint32_t dependencyAdditionCount(size_t k, uint32_t lanes) {
  (void)reducerDepth(k, lanes);
  // Each char4 uses three balanced product additions and one lane-accumulator
  // addition: K logical additions total, then 31 in the output XOR tree.
  return uint32_t(k) + 31;
}

namespace detail {
// Every envelope operation is nonnegative. Advancing the round-to-nearest F64
// result by one representable value avoids understating the mathematical bound.
inline double upperAdd(double a, double b) {
  const double value = a + b;
  return value ? std::nextafter(value, std::numeric_limits<double>::infinity()) : 0;
}
inline double upperMultiply(double a, double b) {
  const double value = a * b;
  return a && b ? std::nextafter(value, std::numeric_limits<double>::infinity()) : 0;
}
inline double upperDivide(double a, double b) {
  const double value = a / b;
  return a ? std::nextafter(value, std::numeric_limits<double>::infinity()) : 0;
}
inline double lowerOneMinus(double value) {
  return std::nextafter(1 - value, -std::numeric_limits<double>::infinity());
}
inline double gammaUpper(uint32_t operations, double unitRoundoff) {
  const double nu = upperMultiply(double(operations), unitRoundoff);
  const double denominator = lowerOneMinus(nu);
  if (!(denominator > 0)) throw std::invalid_argument("accumulation gamma exceeds finite bound");
  return upperDivide(nu, denominator);
}
inline double positiveSumUpper(double sum, double uncertaintyGamma) {
  // |computed-true| <= gamma*true implies true <= computed/(1-gamma).
  // Use the same conservative 4K F64 operation count as the dot reference.
  return upperDivide(sum, lowerOneMinus(uncertaintyGamma));
}
inline double underflowDotAllowance(uint32_t depth, uint32_t additions) {
  // MSL 8.1 permits denormal operands and results to flush. For F32 add,
  // fl(z)=z(1+delta)+eta, |delta|<=u32; dropping two operands, relative
  // rounding, and flushing the result give |eta|<=(3+2u32)*lambda<=4lambda.
  // Each local eta crosses at most D later additions, lambda=minNormalF32.
  return upperMultiply(upperMultiply(double(additions), 4 * std::numeric_limits<float>::min()),
                       upperAdd(1, gammaUpper(depth, kGenericF32UnitRoundoff)));
}
inline DotReference independentReference(std::span<const uint16_t> input,
                                        std::span<const int8_t> codes,
                                        float rowScale, uint32_t depth, uint32_t additions) {
  validate(input, codes, rowScale);
  // This calculation does not use the F32 simulator or the captured GPU result.
  // BF16 x symmetric I8 needs at most 15 significant bits, so a finite normal
  // product is exact in F32. Explicitly retain any exceptional product error.
  frozen::CompensatedSum sum, absolute, productError;
  bool nonfinite = false, overflow = false, subnormal = false;
  uint32_t diagnostics = 0;
  for (size_t k = 0; k < input.size(); ++k) {
    // This private oracle preregisters every source BF16 subnormal as
    // exceptional, including code0 and products that become F32-normal.
    // DotReference's existing productSubnormal field therefore means source-
    // or product-subnormal here; it is intentionally stricter than the frozen
    // reference's product-only classification.
    if ((input[k] & 0x7f80u) == 0 && (input[k] & 0x007fu)) subnormal = true;
    float x = frozen::bf16Number(input[k]);
    if (!std::isfinite(x)) { x = 0; nonfinite = true; diagnostics |= 4; }
    const double product = double(x) * double(codes[k]);
    const float roundedProduct = float(product);
    if (!std::isfinite(roundedProduct)) { overflow = true; diagnostics |= 4; }
    if (product && std::abs(product) < std::numeric_limits<float>::min()) subnormal = true;
    sum.add(product);
    absolute.add(std::abs(product));
    if (std::isfinite(roundedProduct)) productError.add(std::abs(double(roundedProduct) - product));
  }
  const double dot = sum.value(), sumAbs = absolute.value(), pe = productError.value();
  constexpr double u32 = kGenericF32UnitRoundoff, u64 = 0x1p-53;
  const double uncertaintyGamma = gammaUpper(uint32_t(input.size()) * 4, u64);
  const double sumAbsUpper = positiveSumUpper(sumAbs, uncertaintyGamma);
  const double peUpper = positiveSumUpper(pe, uncertaintyGamma);
  const double referenceUncertainty = upperMultiply(uncertaintyGamma, sumAbsUpper);
  const double dotUF = underflowDotAllowance(depth, additions);
  const double rawBound = upperAdd(upperAdd(peUpper,
      upperMultiply(gammaUpper(depth, u32), upperAdd(sumAbsUpper, peUpper))),
      upperAdd(referenceUncertainty, dotUF));
  // The generic relative-error theorem needs nonoverflowing arithmetic. RTZ
  // may saturate an overflowing node to a finite value, so finite capture alone
  // cannot prove that precondition. This envelope covers every intermediate
  // subset/chain/char4/XOR unrounded-node magnitude. productOverflow means
  // source-product overflow or this conservative intermediate-overflow risk.
  const bool productOverflow = overflow ||
      upperAdd(upperAdd(sumAbsUpper, peUpper), rawBound) > std::numeric_limits<float>::max();
  const double scaled = dot * double(rowScale);
  // Late scaling may flush the captured raw-dot operand even with a normal
  // scale. Add lambda to dotBound before multiplying; add another lambda for
  // a flushed scaled result. Explicitly allow the F64 late reference-product
  // uncertainty. No BF16 error is folded into either F32 bound.
  const double rawScaleOperandBound = upperAdd(rawBound, std::numeric_limits<float>::min());
  const double lateReferenceUncertainty = upperMultiply(gammaUpper(1, u64),
      upperMultiply(double(rowScale), std::abs(dot)));
  const double scaledBound = upperAdd(upperAdd(upperAdd(upperMultiply(double(rowScale), rawScaleOperandBound),
      upperMultiply(upperMultiply(gammaUpper(1, u32), double(rowScale)),
                    upperAdd(std::abs(dot), rawScaleOperandBound))), std::numeric_limits<float>::min()),
      lateReferenceUncertainty);
  const bool referenceScaledOverflow = std::abs(scaled) > std::numeric_limits<float>::max();
  // scaledOverflow means reference-scaled overflow or conservative late-scale
  // preimage-envelope risk. Risk-only exceptions do not imply shader sticky4.
  const bool scaledOverflow = referenceScaledOverflow ||
      upperMultiply(double(rowScale), upperAdd(std::abs(dot), rawScaleOperandBound)) >
          std::numeric_limits<float>::max();
  // The existing scaledSubnormal field means scale-input or scaled-output
  // subnormal here. A subnormal scale stays exceptional even with normal y.
  const bool scaledSubnormal = rowScale < std::numeric_limits<float>::min() ||
      (scaled && std::abs(scaled) < std::numeric_limits<float>::min());
  if (referenceScaledOverflow) diagnostics |= 4;
  return {dot, sumAbs, pe, rawBound, scaled, scaledBound, frozen::bf16FromF64(scaled), diagnostics,
      nonfinite, productOverflow, subnormal, scaledOverflow, scaledSubnormal,
      nonfinite || productOverflow || subnormal || scaledOverflow || scaledSubnormal};
}

// Volatile stores deliberately make every simulator operation an F32 rounding
// point; they prevent a compiler from replacing a product/add pair with FMA.
inline float addF32(float a, float b) { volatile float result = a + b; return result; }
inline float multiplyF32(float a, float b) { volatile float result = a * b; return result; }
inline void jsonNumber(std::ostream &out, double value) {
  if (std::isfinite(value)) out << std::setprecision(17) << value;
  else out << "null";
}
} // namespace detail

inline DotReference vectorReference(std::span<const uint16_t> input,
                                    std::span<const int8_t> codes, float rowScale,
                                    uint32_t lanes = 32) {
  return detail::independentReference(input, codes, rowScale, reducerDepth(input.size(), lanes),
                                      dependencyAdditionCount(input.size(), lanes));
}

inline DotReference scalarReference(std::span<const uint16_t> input,
                                    std::span<const int8_t> codes, float rowScale) {
  return detail::independentReference(input, codes, rowScale, uint32_t(input.size()), uint32_t(input.size()));
}

struct Capture {
  float dot = 0, scaled = 0;
  uint16_t bf16 = 0;
  uint32_t diagnostics = 0;
};

inline Capture vectorCapture(std::span<const uint16_t> input,
                             std::span<const int8_t> codes, float rowScale,
                             uint32_t lanes = 32) {
  validate(input, codes, rowScale);
  (void)reducerDepth(input.size(), lanes);
  std::array<float, 32> partials{};
  uint32_t diagnostics = 0;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    for (size_t base = 4 * lane; base < input.size(); base += 4 * lanes) {
      std::array<float, 4> products{};
      for (size_t component = 0; component < 4; ++component) {
        const size_t k = base + component;
        if (k >= input.size()) continue;
        float x = frozen::bf16Number(input[k]);
        if (!std::isfinite(x)) { x = 0; diagnostics |= 4; }
        const float product = detail::multiplyF32(x, float(codes[k]));
        if (!std::isfinite(product)) diagnostics |= 4;
        products[component] = product;
      }
      const float chunk = detail::addF32(detail::addF32(products[0], products[1]),
                                          detail::addF32(products[2], products[3]));
      partials[lane] = detail::addF32(partials[lane], chunk);
    }
  }
  // Explicit descending XOR tree. All lanes read the previous stage, exactly as
  // a simultaneous shuffle does; an in-place CPU loop would be a different sum.
  for (uint32_t offset = lanes / 2; offset; offset >>= 1) {
    const auto previous = partials;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      partials[lane] = detail::addF32(previous[lane], previous[lane ^ offset]);
  }
  const float scaled = detail::multiplyF32(partials[0], rowScale);
  const uint16_t bf16 = frozen::bf16FromF64(double(scaled));
  if (!std::isfinite(partials[0]) || !std::isfinite(scaled) ||
      !std::isfinite(frozen::bf16Number(bf16))) diagnostics |= 4;
  return {partials[0], scaled, bf16, diagnostics};
}

struct CertificateReport {
  CaptureReport strict;
  bool certifiedBoundSignFinitePass = false;
  bool strictSensitiveFailure = false;
};

inline CertificateReport assess(const DotReference &reference, float capturedDot,
                                float capturedScaled, uint16_t capturedBF16,
                                uint32_t diagnostics) {
  const auto strict = frozen::assess(reference, capturedDot, capturedScaled, capturedBF16, diagnostics);
  const bool certified = !reference.exceptional && strict.diagnosticsCoverExpected &&
      strict.finiteCapture && strict.dotWithinBound && strict.scaledWithinBound &&
      strict.signMatches && strict.negativeZeroMatches;
  return {strict, certified, strict.strictSensitive && !strict.strictBF16Pass};
}

inline CertificateReport assess(const DotReference &reference, const Capture &capture) {
  return expert_r4_quality::assess(reference, capture.dot, capture.scaled, capture.bf16, capture.diagnostics);
}

struct RouteMetric {
  size_t elements = 0, mismatches = 0, nonfinite = 0;
  double relativeL2 = 0, cosine = 1, maxAbsoluteError = 0;
  bool pass() const {
    return elements && !nonfinite && std::isfinite(relativeL2) && std::isfinite(cosine) &&
        relativeL2 <= kMaximumRelativeL2 && cosine >= kMinimumCosine;
  }
  void json(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"mismatches\":" << mismatches
        << ",\"nonfinite\":" << nonfinite << ",\"relative_l2\":";
    detail::jsonNumber(out, relativeL2);
    out << ",\"cosine\":"; detail::jsonNumber(out, cosine);
    out << ",\"max_absolute_error\":"; detail::jsonNumber(out, maxAbsoluteError);
    out << ",\"pass\":" << (pass() ? "true" : "false") << '}';
  }
};

namespace detail {
inline RouteMetric metric(std::span<const uint16_t> baseline,
                          std::span<const uint16_t> candidate) {
  RouteMetric result;
  result.elements = baseline.size();
  frozen::CompensatedSum error, normA, normB, dot;
  for (size_t i = 0; i < baseline.size(); ++i) {
    if (baseline[i] != candidate[i]) ++result.mismatches;
    const double a = frozen::bf16Number(baseline[i]), b = frozen::bf16Number(candidate[i]);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++result.nonfinite; continue; }
    const double delta = a - b;
    result.maxAbsoluteError = std::max(result.maxAbsoluteError, std::abs(delta));
    error.add(delta * delta); normA.add(a * a); normB.add(b * b); dot.add(a * b);
  }
  if (result.nonfinite) {
    result.relativeL2 = std::numeric_limits<double>::infinity();
    result.cosine = std::numeric_limits<double>::quiet_NaN();
    return result;
  }
  const double squaredError = error.value(), a2 = normA.value(), b2 = normB.value();
  result.relativeL2 = a2 ? std::sqrt(squaredError / a2) :
      (squaredError ? std::numeric_limits<double>::infinity() : 0);
  // Two exactly-zero vectors are identical in quality, including signed zero.
  // Exactly one zero vector has no matching direction and fails cosine.
  result.cosine = a2 && b2 ? std::clamp((dot.value() / std::sqrt(a2)) / std::sqrt(b2), -1., 1.) :
      (a2 == b2 ? 1 : 0);
  return result;
}
} // namespace detail

struct MetricReport {
  uint32_t routeWidth = 0;
  RouteMetric aggregate;
  std::vector<RouteMetric> routes;
  bool pass() const {
    return aggregate.pass() && !routes.empty() &&
        std::all_of(routes.begin(), routes.end(), [](const auto &route) { return route.pass(); });
  }
  void json(std::ostream &out) const {
    out << "{\"maximum_relative_l2\":";
    detail::jsonNumber(out, kMaximumRelativeL2);
    out << ",\"minimum_cosine\":"; detail::jsonNumber(out, kMinimumCosine);
    out << ",\"route_width\":" << routeWidth << ",\"aggregate\":";
    aggregate.json(out);
    out << ",\"routes\":[";
    for (size_t i = 0; i < routes.size(); ++i) {
      if (i) out << ',';
      out << "{\"route\":" << i << ",\"metric\":";
      routes[i].json(out); out << '}';
    }
    out << "],\"pass\":" << (pass() ? "true" : "false") << '}';
  }
};

inline MetricReport metrics(std::span<const uint16_t> baseline,
                            std::span<const uint16_t> candidate, uint32_t routeWidth) {
  if (baseline.size() != candidate.size() || !routeWidth || baseline.empty() ||
      baseline.size() % routeWidth)
    throw std::invalid_argument("BF16 quality metrics require equal nonempty complete routes");
  MetricReport result;
  result.routeWidth = routeWidth;
  result.aggregate = detail::metric(baseline, candidate);
  result.routes.reserve(baseline.size() / routeWidth);
  for (size_t base = 0; base < baseline.size(); base += routeWidth)
    result.routes.push_back(detail::metric(baseline.subspan(base, routeWidth),
                                          candidate.subspan(base, routeWidth)));
  return result;
}

inline bool cpuSelfTest() {
  try {
    if (reducerDepth(640, 32) != 12 || reducerDepth(2560, 32) != 27 ||
        dependencyAdditionCount(640, 32) != 671 || dependencyAdditionCount(2560, 32) != 2591 ||
        !(detail::gammaUpper(12, kGenericF32UnitRoundoff) > frozen::gamma(12, 0x1p-24))) return false;
    bool rejected = false;
    try { (void)reducerDepth(640, 16); } catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) return false;
    const auto oneBoundFits = [](double actual, size_t k, uint32_t depth, uint32_t additions) {
      const double mathematical = frozen::gamma(depth, kGenericF32UnitRoundoff) * double(k) +
          frozen::gamma(uint32_t(k) * 4, 0x1p-53) * double(k) +
          double(additions) * 4 * std::numeric_limits<float>::min() *
              (1 + frozen::gamma(depth, kGenericF32UnitRoundoff));
      // The result must dominate the theorem with only the small additional
      // inflation needed for conservative F64 envelope evaluation.
      return actual >= mathematical && actual <= mathematical * (1 + 1e-10);
    };
    for (size_t k : {size_t(640), size_t(2560)}) {
      std::vector<uint16_t> x(k, 0x3f80);
      std::vector<int8_t> codes(k, 1);
      const auto scalar = scalarReference(x, codes, .125f);
      if (scalar.dot != double(k) || scalar.scaled != double(k) / 8 ||
          scalar.exactProductRoundingError || scalar.exceptional) return false;
      if (!oneBoundFits(scalar.dotAbsoluteBound, k, uint32_t(k), uint32_t(k))) return false;
      for (uint32_t lanes : {32u}) {
        const auto reference = vectorReference(x, codes, .125f, lanes);
        const auto captured = vectorCapture(x, codes, .125f, lanes);
        const auto report = assess(reference, captured);
        if (captured.dot != float(k) || captured.scaled != float(k) / 8 ||
            reference.dot != scalar.dot || reference.bf16 != scalar.bf16 ||
            !(reference.dotAbsoluteBound < scalar.dotAbsoluteBound) ||
            !report.certifiedBoundSignFinitePass || report.strictSensitiveFailure ||
            !report.strict.regularFinitePrimitivePass) return false;
        if (!oneBoundFits(reference.dotAbsoluteBound, k, reducerDepth(k, lanes),
                         dependencyAdditionCount(k, lanes))) return false;
        // Exact integer products cover every adjacent component and strided
        // chunk; the expected sum is accumulated independently in integer form.
        int64_t integerDot = 0;
        for (size_t i = 0; i < k; ++i) {
          const int value = int(i % 31) - 15;
          x[i] = frozen::bf16FromF64(value);
          codes[i] = int8_t(int(i % 253) - 126);
          integerDot += int64_t(value) * codes[i];
        }
        if (vectorReference(x, codes, 1, lanes).dot != double(integerDot) ||
            vectorCapture(x, codes, 1, lanes).dot != float(integerDot)) return false;
        std::fill(x.begin(), x.end(), 0x3f80);
        std::fill(codes.begin(), codes.end(), 1);
      }
    }
    std::vector<uint16_t> x(640, 0);
    std::vector<int8_t> codes(640, 1);
    x[0] = frozen::bf16FromF64(0x1p100); x[1] = 0x3f80;
    x[2] = frozen::bf16FromF64(-0x1p100);
    const auto cancelled = vectorReference(x, codes, 1, 32);
    const auto lostOne = assess(cancelled, vectorCapture(x, codes, 1, 32));
    if (cancelled.dot != 1 || !lostOne.certifiedBoundSignFinitePass ||
        !lostOne.strictSensitiveFailure || lostOne.strict.regularFinitePrimitivePass) return false;
    x[3] = 0x3f80;
    const auto associationReference = vectorReference(x, codes, 1, 32);
    const auto associationCapture = vectorCapture(x, codes, 1, 32);
    const float serialChar4 = detail::addF32(detail::addF32(detail::addF32(
        frozen::bf16Number(x[0]), frozen::bf16Number(x[1])), frozen::bf16Number(x[2])),
        frozen::bf16Number(x[3]));
    if (associationReference.dot != 2 || associationCapture.dot != 0 || serialChar4 != 1 ||
        !assess(associationReference, associationCapture).strictSensitiveFailure) return false;
    std::fill(x.begin(), x.end(), 0);
    x[0] = frozen::bf16FromF64(0x1p100); x[1] = frozen::bf16FromF64(-0x1p100); x[128] = 0x3f80;
    if (vectorReference(x, codes, 1, 32).dot != 1 || vectorCapture(x, codes, 1, 32).dot != 1) return false;
    // Descending XOR cancels lanes0/2 before adding lanes1/3. Ascending XOR
    // first pairs lanes0/1 and lanes2/3 and loses the four small units.
    std::fill(x.begin(), x.end(), 0);
    x[0] = frozen::bf16FromF64(0x1p100); x[4] = 0x3f80;
    x[8] = frozen::bf16FromF64(-0x1p100); x[12] = 0x4040;
    for (uint32_t lanes : {32u})
      if (vectorReference(x, codes, 1, lanes).dot != 4 ||
          vectorCapture(x, codes, 1, lanes).dot != 4) return false;
    std::fill(x.begin(), x.end(), 0);
    // All three source products are F32-normal, but the first char4 pair add
    // cancels to BF16's minimum quantum. Flushing that permitted intermediate
    // loses one BF16 ULP even though the eventual dot and scale are normal.
    x[0] = 0x0081; x[1] = 0x8080; x[2] = 0x0080;
    const auto ftzReference = vectorReference(x, codes, 1, 32);
    const auto ftzAdd = [](float a, float b) {
      const float lambda = std::numeric_limits<float>::min();
      if (std::abs(a) < lambda) a = std::copysign(0.f, a);
      if (std::abs(b) < lambda) b = std::copysign(0.f, b);
      const float result = detail::addF32(a, b);
      return std::abs(result) < lambda ? std::copysign(0.f, result) : result;
    };
    std::array<float, 32> ftzPartials{};
    const float ftzChunk = ftzAdd(ftzAdd(frozen::bf16Number(x[0]), frozen::bf16Number(x[1])),
                                 ftzAdd(frozen::bf16Number(x[2]), 0));
    ftzPartials[0] = ftzAdd(0, ftzChunk);
    for (uint32_t offset = 16; offset; offset >>= 1) {
      const auto previous = ftzPartials;
      for (uint32_t lane = 0; lane < 32; ++lane)
        ftzPartials[lane] = ftzAdd(previous[lane], previous[lane ^ offset]);
    }
    const float ftzDot = ftzPartials[0];
    const Capture ftzCapture{ftzDot, ftzDot, frozen::bf16FromF64(double(ftzDot)), 0};
    const auto ftzReport = assess(ftzReference, ftzCapture);
    const double oldNoUFBound = frozen::gamma(12, kGenericF32UnitRoundoff) * ftzReference.sumAbsProducts +
        frozen::gamma(640 * 4, 0x1p-53) * ftzReference.sumAbsProducts;
    if (ftzReference.exceptional || ftzReference.bf16 != 0x0081 || ftzCapture.bf16 != 0x0080 ||
        vectorCapture(x, codes, 1, 32).bf16 != 0x0081 ||
        !(ftzReport.strict.dotAbsoluteError > oldNoUFBound) ||
        ftzReport.strict.dotAbsoluteError > detail::underflowDotAllowance(12, 671) ||
        !ftzReport.strict.dotWithinBound || !ftzReport.strict.scaledWithinBound ||
        !ftzReport.certifiedBoundSignFinitePass || !ftzReport.strictSensitiveFailure ||
        ftzReport.strict.strictBF16Pass || ftzReport.strict.regularFinitePrimitivePass) return false;
    const std::array<uint16_t, 1> ftzExpected{ftzReference.bf16}, ftzObserved{ftzCapture.bf16};
    if (metrics(ftzExpected, ftzObserved, 1).pass()) return false;
    // A subnormal row-scale operand must be reviewed even when the mathematical
    // output is normal; generic multiplication may flush that operand to zero.
    std::fill(x.begin(), x.end(), 0);
    x[0] = frozen::bf16FromF64(0x1p100);
    const auto subnormalScale = vectorReference(x, codes, std::numeric_limits<float>::denorm_min(), 32);
    if (!subnormalScale.scaledSubnormal || !subnormalScale.exceptional ||
        !(std::abs(subnormalScale.scaled) >= std::numeric_limits<float>::min()) ||
        assess(subnormalScale, vectorCapture(x, codes, std::numeric_limits<float>::denorm_min(), 32))
            .certifiedBoundSignFinitePass) return false;
    std::fill(x.begin(), x.end(), 0);
    const auto zero = vectorReference(x, codes, 1, 32);
    if (!(zero.scaledAbsoluteBound >= zero.dotAbsoluteBound + 2 * std::numeric_limits<float>::min()))
      return false;
    if (expert_r4_quality::assess(zero, -0.f, -0.f, 0x8000, 0).certifiedBoundSignFinitePass) return false;
    x[0] = 0x0001;
    if (!vectorReference(x, codes, 1, 32).exceptional) return false;
    codes[0] = 0;
    const auto zeroSourceSubnormal = vectorReference(x, codes, 1, 32);
    if (zeroSourceSubnormal.dot != 0 || !zeroSourceSubnormal.productSubnormal ||
        !zeroSourceSubnormal.exceptional) return false;
    codes[0] = 127;
    if (!vectorReference(x, codes, 1, 32).exceptional) return false;
    x[0] = 0x007f;
    const auto normalProductSourceSubnormal = vectorReference(x, codes, 1, 32);
    if (std::abs(normalProductSourceSubnormal.dot) < std::numeric_limits<float>::min() ||
        !normalProductSourceSubnormal.productSubnormal ||
        !normalProductSourceSubnormal.exceptional) return false;
    codes[0] = 1;
    x[0] = 0x7fc0;
    const auto nonfinite = vectorReference(x, codes, 1, 32);
    if (!nonfinite.nonfiniteInput || nonfinite.expectedStickyMinimum != 4 ||
        assess(nonfinite, vectorCapture(x, codes, 1, 32)).certifiedBoundSignFinitePass) return false;
    x[0] = 0x7f7f; codes[0] = 127;
    if (!vectorReference(x, codes, 1, 32).productOverflow) return false;
    codes[0] = 1;
    for (size_t second : {size_t(1), size_t(128)}) {
      // Every source product is finite/normal, and the exact final dot is the
      // finite maximum BF16 value. The balanced-pair or chain sum can overflow,
      // including finite saturation under RTZ, before later cancellation.
      std::fill(x.begin(), x.end(), 0);
      x[0] = 0x7f7f; x[second] = 0x7f7f; x[second == 1 ? 2 : 256] = 0xff7f;
      const auto intermediateRisk = vectorReference(x, codes, 1, 32);
      if (intermediateRisk.dot != double(frozen::bf16Number(0x7f7f)) ||
          !std::isfinite(intermediateRisk.dot) || !std::isfinite(intermediateRisk.scaled) ||
          !intermediateRisk.productOverflow || !intermediateRisk.exceptional ||
          intermediateRisk.expectedStickyMinimum != 0) return false;
    }
    std::fill(x.begin(), x.end(), 0);
    x[0] = frozen::bf16FromF64(0x1p100); x[1] = frozen::bf16FromF64(-0x1p100); x[2] = 0x3f80;
    const auto scaledEnvelopeRisk = vectorReference(x, codes, 1e15f, 32);
    const auto finiteScaledCapture = vectorCapture(x, codes, 1e15f, 32);
    if (scaledEnvelopeRisk.dot != 1 || !std::isfinite(scaledEnvelopeRisk.scaled) ||
        scaledEnvelopeRisk.productOverflow || !scaledEnvelopeRisk.scaledOverflow ||
        !scaledEnvelopeRisk.exceptional || scaledEnvelopeRisk.expectedStickyMinimum != 0 ||
        finiteScaledCapture.diagnostics || !std::isfinite(finiteScaledCapture.scaled) ||
        assess(scaledEnvelopeRisk, finiteScaledCapture).certifiedBoundSignFinitePass) return false;
    std::fill(x.begin(), x.end(), 0);
    x[0] = 0x7f7f;
    codes[0] = 1;
    const auto bf16Overflow = vectorCapture(x, codes, 1.00390625f, 32);
    if (!std::isfinite(bf16Overflow.dot) || !std::isfinite(bf16Overflow.scaled) ||
        bf16Overflow.bf16 != 0x7f80 || bf16Overflow.diagnostics != 4) return false;
    codes[0] = -128;
    rejected = false;
    try { (void)vectorReference(x, codes, 1, 32); } catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) return false;

    std::array<uint16_t, 4> a{0, 0x8000, 0, 0}, b{0x8000, 0, 0, 0};
    if (!metrics(a, b, 2).pass()) return false;
    b[2] = 0x3f80;
    if (metrics(a, b, 2).pass()) return false;
    a = {0x3f80, 0x3f80, 0x3f80, 0x3f80}; b = a;
    if (!metrics(a, b, 2).pass()) return false;
    b[0] = 0x3f81;
    if (metrics(a, b, 2).pass()) return false;
    b = {0xbf80, 0xbf80, 0xbf80, 0xbf80};
    if (metrics(a, b, 2).aggregate.cosine != -1 || metrics(a, b, 2).pass()) return false;
    b = {0, 0, 0, 0};
    if (metrics(a, b, 2).aggregate.relativeL2 != 1 || metrics(a, b, 2).pass()) return false;
    b = a; b[0] = 0x7fc0;
    if (!metrics(a, b, 2).aggregate.nonfinite || metrics(a, b, 2).pass()) return false;
    // BF16 extremes still square safely in F64. Do not clamp the reference norm
    // to an arbitrary floor that would hide an error on tiny nonzero routes.
    a = {frozen::bf16FromF64(0x1p-80), 0, 0, 0};
    b = {0, frozen::bf16FromF64(0x1p-80), 0, 0};
    const auto tiny = metrics(a, b, 2);
    if (tiny.pass() || tiny.routes[0].relativeL2 != std::sqrt(2.) ||
        tiny.routes[0].cosine != 0) return false;
    a = {frozen::bf16FromF64(0x1p127), frozen::bf16FromF64(0x1p127), 0, 0};
    if (!metrics(a, a, 2).pass()) return false;
    a = {0x3f80, 0, 0, 0}; b = a;
    b[1] = frozen::bf16FromF64(0x1p-14);
    if (!metrics(a, b, 2).pass()) return false;
    b[1] = frozen::bf16FromF64(0x1p-13);
    if (metrics(a, b, 2).pass()) return false;
    // Aggregate error may be diluted by a large good route; each route must
    // independently retain the same frozen numerical thresholds.
    a = {0x4980, 0x4980, 0x3f80, 0x3f80}; b = a; b[2] = 0x3f81;
    const auto diluted = metrics(a, b, 2);
    if (!diluted.aggregate.pass() || diluted.routes[1].pass() || diluted.pass()) return false;
    rejected = false;
    try { (void)metrics(a, b, 3); } catch (const std::invalid_argument &) { rejected = true; }
    return rejected;
  } catch (...) { return false; }
}
} // namespace expert_r4_quality
