// Pure CPU self-test. No model payload files, Metal setup, or GPU execution.
#include "quantization.hpp"
#include <array>
#include <cfenv>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace splash::dense_w8a8;
uint64_t checks = 0;

void check(bool condition, const std::string &message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function function, const std::string &message) {
  bool threw = false;
  try { function(); } catch (const std::invalid_argument &) { threw = true; }
  check(threw, message);
}

bool near(double left, double right, double multiplier = 32.0) {
  return std::abs(left - right) <= multiplier * std::numeric_limits<double>::epsilon()
      * std::max({1.0, std::abs(left), std::abs(right)});
}

void bf16Conversion() {
  for (uint32_t bits = 0; bits < 65536; ++bits) {
    const uint16_t input = uint16_t(bits);
    const float value = bf16Number(input);
    check(std::bit_cast<uint32_t>(value) == bits << 16, "BF16 reconstruction must be exact");
    if (std::isfinite(value)) check(bf16Bits(value) == input, "Finite BF16 round trip");
    else if (std::isinf(value)) check(bf16Bits(value) == input, "Infinite BF16 round trip");
    else check(std::isnan(bf16Number(bf16Bits(value))), "NaN must remain NaN");
  }
  // Explicit discarded-mantissa ties exercise both parity directions and signs.
  const std::array<uint32_t, 8> lower = {
      0x00000000u, 0x00010000u, 0x007e0000u, 0x007f0000u,
      0x3f800000u, 0x3f810000u, 0x7f7e0000u, 0x7f7f0000u};
  for (uint32_t base : lower) for (uint32_t sign : {0u, 0x80000000u}) {
    const uint32_t bits = base | sign;
    check(bf16Bits(std::bit_cast<float>(bits | 0x7fffu)) == uint16_t(bits >> 16), "BF16 below-half rounding");
    check(bf16Bits(std::bit_cast<float>(bits | 0x8000u))
        == uint16_t((bits >> 16) + ((bits >> 16) & 1u)), "BF16 exact even tie");
    check(bf16Bits(std::bit_cast<float>(bits | 0x8001u)) == uint16_t((bits >> 16) + 1u), "BF16 above-half rounding");
  }
  check(std::isnan(bf16Number(bf16Bits(std::bit_cast<float>(0x7f800001u)))), "Low FP32 NaN payload survives conversion");
  check(std::signbit(bf16Number(bf16Bits(-0.0f))), "BF16 preserves negative zero");
}

void codeRounding() {
  check(std::fesetround(FE_TONEAREST) == 0, "Set independent nearbyint reference to RNE");
  const std::array<std::pair<float, int>, 18> explicitCases = {{{0.0f, 0}, {-0.0f, 0},
      {0.5f, 0}, {-0.5f, 0}, {1.5f, 2}, {-1.5f, -2}, {2.5f, 2}, {-2.5f, -2},
      {3.5f, 4}, {-3.5f, -4}, {125.5f, 126}, {-125.5f, -126}, {126.5f, 126},
      {-126.5f, -126}, {127.5f, 127}, {-127.5f, -127}, {500.0f, 127}, {-500.0f, -127}}};
  for (const auto &[value, expected] : explicitCases) {
    check(int(symmetricCode(value, 1.0f)) == expected, "Explicit signed RNE/clamp case");
    check(int(symmetricCode(value * 2.0f, 2.0f)) == expected, "RNE uses scaled quotient");
  }
  for (int twice = -260; twice <= 260; ++twice) {
    const float half = float(twice) / 2.0f;
    for (float value : {std::nextafter(half, -std::numeric_limits<float>::infinity()), half,
                        std::nextafter(half, std::numeric_limits<float>::infinity())}) {
      const int expected = int(std::clamp(std::nearbyint(double(value)), -127.0, 127.0));
      const int actual = int(symmetricCode(value, 1.0f));
      check(actual == expected, "Adjacent-to-tie result matches independent RNE");
      check(actual >= -127 && actual <= 127 && actual != -128, "Symmetric code never emits -128");
      check(symmetricClipped(value, 1.0f) == (std::abs(std::nearbyint(double(value))) > 127.0),
          "Clipping is measured after RNE");
    }
  }
  // Non-dyadic scales test the required FP32 division against an independent
  // FP64 nearbyint applied only after the quotient has rounded to FP32.
  for (float scale : {0.001f, 0.1f, 0.3f, 2.0f, 31.0f}) {
    for (int quarter = -2048; quarter <= 2048; ++quarter) {
      const float value = float(quarter) * 0.25f * scale;
      const float quotient = value / scale;
      const int expected = int(std::clamp(std::nearbyint(double(quotient)), -127.0, 127.0));
      check(int(symmetricCode(value, scale)) == expected, "FP32 division precedes independent integer RNE");
    }
  }
  check(symmetricCode(std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min()) == 127,
      "Overflowed finite quotient saturates safely");
  check(symmetricCode(-std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min()) == -127,
      "Negative overflowed quotient saturates safely");
  for (float scale : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    rejects([=] { (void)symmetricCode(1.0f, scale); }, "Reject invalid scale");
  for (float value : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()})
    rejects([=] { (void)symmetricCode(value, 1.0f); }, "Reject nonfinite coefficient");
}

void scaleAndRow() {
  const std::array<uint16_t, 3> zero = {0, 0x8000u, 0};
  check(symmetricScale(zero.data(), uint32_t(zero.size())) == 1.0f, "Zero row has unit scale");
  check(symmetricScale(nullptr, 0) == 1.0f, "Empty row has unit scale");
  const std::array<uint16_t, 4> values = {bf16Bits(1.0f), bf16Bits(-2.0f), 0, bf16Bits(0.25f)};
  check(symmetricScale(values.data(), uint32_t(values.size())) == 2.0f / 127.0f, "Scale is maximum absolute BF16 value divided by 127");
  const std::array<uint16_t, 1> minimum = {1};
  check(symmetricScale(minimum.data(), 1) > 0.0f, "Smallest BF16 row retains a positive FP32 scale");
  for (uint16_t value : {uint16_t(0x7f80u), uint16_t(0xff80u), uint16_t(0x7fc1u)}) {
    const std::array<uint16_t, 3> invalid = {bf16Bits(1.0f), value, bf16Bits(2.0f)};
    rejects([&] { (void)symmetricScale(invalid.data(), 3); }, "Reject nonfinite source row");
    std::array<int8_t, 3> output = {42, 42, 42};
    float scale = 42.0f;
    QuantError error;
    rejects([&] { quantizeRow(invalid.data(), 3, output.data(), scale, error); }, "Reject nonfinite row before writes");
    check(output == std::array<int8_t, 3>{42, 42, 42} && scale == 42.0f && error.elements() == 0,
        "Invalid row leaves outputs and certificate unchanged");
  }
  rejects([] { (void)symmetricScale(nullptr, 1); }, "Reject null nonempty source");
  QuantError error;
  float scale = -1.0f;
  quantizeRow(nullptr, 0, nullptr, scale, error);
  check(scale == 1.0f && error.elements() == 0, "Empty quantization writes unit scale only");
  rejects([&] { quantizeRow(values.data(), 4, nullptr, scale, error); }, "Reject null nonempty output");

  const std::array<float, 7> original = {127.0f, 0.5f, -0.5f, 1.5f, -1.5f, 0.0f, -0.0f};
  std::array<uint16_t, 7> source{};
  std::array<int8_t, 7> output{};
  for (size_t i = 0; i < source.size(); ++i) source[i] = bf16Bits(original[i]);
  QuantError rowError;
  quantizeRow(source.data(), uint32_t(source.size()), output.data(), scale, rowError);
  check(scale == 1.0f, "Fixture maximum makes exact unit scale");
  check(output == std::array<int8_t, 7>{127, 0, 0, 2, -2, 0, 0}, "Whole row uses signed even rounding");
  check(rowError.elements() == 7 && rowError.nonfinite() == 0 && rowError.sourceNonzeroQuantizedZero() == 2
      && rowError.clippedCount() == 0, "Full row certificate counts every coefficient and nonzero-to-zero result");
  check(rowError.maxAbsError() == 0.5 && rowError.squaredError() == 1.0, "FP64 coefficient error arithmetic");
  check(near(rowError.relativeL2(), std::sqrt(1.0 / 16134.0)), "Relative L2 certificate uses original BF16 norm");
  check(near(rowError.cosine(), 16135.0 / std::sqrt(16134.0 * 16137.0)), "Cosine certificate includes full original/dequantized vectors");
}

void errorCertificate() {
  QuantError exact;
  exact.add(2.0, 2.0);
  exact.add(-3.0, -3.0);
  check(exact.elements() == 2 && exact.relativeL2() == 0.0 && exact.cosine() == 1.0, "Exact vector certificate");
  QuantError allZero;
  allZero.add(0.0, 0.0);
  check(allZero.relativeL2() == 0.0 && allZero.cosine() == 1.0, "Zero versus zero certificate convention");
  QuantError destroyed;
  destroyed.add(2.0, 0.0, int8_t(0));
  check(destroyed.relativeL2() == 1.0 && destroyed.cosine() == 0.0
      && destroyed.sourceNonzeroQuantizedZero() == 1, "Nonzero versus zero certificate convention");
  QuantError zeroReference;
  zeroReference.add(0.0, 1.0);
  check(std::isinf(zeroReference.relativeL2()) && zeroReference.cosine() == 0.0, "Nonzero error versus zero source is unbounded");
  QuantError invalid;
  invalid.add(std::numeric_limits<double>::quiet_NaN(), 1.0, int8_t(1));
  invalid.add(1.0, std::numeric_limits<double>::infinity(), int8_t(1));
  invalid.add(200.0, 127.0, int8_t(127), true);
  check(invalid.elements() == 3 && invalid.nonfinite() == 2 && invalid.clippedCount() == 1,
      "Nonfinite and clipped counts are visible");
  std::ostringstream json;
  json.precision(4);
  json << std::hex << std::showpos << std::scientific;
  const auto originalFlags = json.flags();
  zeroReference.writeJSON(json);
  check(json.precision() == 4 && json.flags() == originalFlags, "JSON restores caller stream format");
  check(json.str().find("\"relative_l2\":null") != std::string::npos
      && json.str().find("\"elements\":1") != std::string::npos
      && json.str().find("\"source_nonzero_quantized_zero\":0") != std::string::npos,
      "Certificate JSON contains counts and valid JSON for unbounded relative error");
}

void int32BoundAndDecomposition() {
  for (uint32_t k : {2560u, 6144u, 32768u}) {
    const uint64_t bound = int32DotMagnitudeBound(k);
    check(bound == uint64_t(k) * 16129u, "I32 dot bound is K*16129");
    check(bound <= uint64_t(std::numeric_limits<int32_t>::max()), "Specified K is safely bounded in I32");
    int32_t positive = 0, negative = 0;
    for (uint32_t i = 0; i < k; ++i) { positive += 127 * 127; negative -= 127 * 127; }
    check(int64_t(positive) == int64_t(bound) && int64_t(negative) == -int64_t(bound),
        "Adversarial INT8 operands attain both certified dot limits without overflow");
  }
  check(int32DotMagnitudeBound(133144u) <= uint64_t(std::numeric_limits<int32_t>::max()), "Largest safe positive dot K");
  check(int32DotMagnitudeBound(133145u) > uint64_t(std::numeric_limits<int32_t>::max()), "Next dot K exceeds I32 bound");
  const std::array<int8_t, 12> a = {127, -127, 3, -7, 0, 1, 4, -9, 11, 64, -64, 2};
  const std::array<int8_t, 12> b = {-127, -127, -4, 8, 100, 2, 6, 5, -12, -32, -32, 3};
  int64_t integerDot = 0;
  for (size_t i = 0; i < a.size(); ++i) integerDot += int64_t(a[i]) * int64_t(b[i]);
  // An independent hand total certifies the exact integer accumulation itself.
  check(integerDot == -213, "Small-array integer dot exact hand total");
  for (float scaleA : {1.0f, 0.1f, 0.003f}) for (float scaleB : {2.0f, 0.3f, 31.0f}) {
    double dequantDot = 0.0, magnitudeSum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      const double term = (double(scaleA) * double(a[i])) * (double(scaleB) * double(b[i]));
      dequantDot += term;
      magnitudeSum += std::abs(term);
    }
    const double factored = double(integerDot) * double(scaleA) * double(scaleB);
    check(std::abs(dequantDot - factored) <= 32.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, magnitudeSum), "I32 dot times scales matches independently dequantized FP64 products");
  }
}
}  // namespace

int main() {
  try {
    bf16Conversion();
    codeRounding();
    scaleAndRow();
    errorCertificate();
    int32BoundAndDecomposition();
    std::cout << "CPU W8A8 quantization self-test passed: " << checks
              << " checks; no GPU execution or model payload reads\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CPU W8A8 quantization self-test failed after " << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}
