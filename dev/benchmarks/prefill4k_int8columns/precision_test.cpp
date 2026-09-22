#include "precision.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace cert = splash::flash::prefill4k_int8columns;
void require(bool pass, const char *message) { if (!pass) throw std::runtime_error(message); }
uint16_t bf(float x) {
  const uint32_t word = std::bit_cast<uint32_t>(x);
  require((word & 0x7f800000) != 0x7f800000, "test BF16 must be finite");
  require((word & 0xffff) == 0, "test input must be an exact BF16 word");
  return uint16_t(word >> 16);
}
template <class F> void rejected(F &&body) {
  bool did = false;
  try { body(); } catch (const std::invalid_argument &) { did = true; }
  require(did, "invalid certificate input accepted");
}
float simulated(const std::vector<int8_t> &codes, float scale, const std::vector<uint16_t> &input) {
  float result = 0;
  for (size_t k = 0; k < codes.size(); ++k) {
    volatile float product = float(cert::bfNumber(input[k])) * float(codes[k]);
    volatile float next = result + product; result = next;
  }
  volatile float scaled = result * scale; return scaled;
}
int main() {
  cert::selfTest();
  uint32_t tests = 0;
  for (uint32_t K : {640u, 2560u}) {
    std::vector<int8_t> codes(K); std::vector<uint16_t> input(K);
    const auto clear = [&] { std::fill(codes.begin(), codes.end(), 0); std::fill(input.begin(), input.end(), 0); };
    const auto check = [&](double expected, float scale = 1) {
      const auto c = cert::certificate(codes.data(), K, scale, input.data());
      require(c.dotScaled == expected, "exact-dyadic reference differs");
      require(std::isfinite(c.norm) && c.norm >= std::abs(expected), "scaled norm is not an upper bound");
      require(c.referenceEnvelope >= 0 && c.candidateEnvelope >= 0, "negative certificate envelope");
      const double got = simulated(codes, scale, input);
      if (std::isfinite(got)) require(std::abs(got - c.dotScaled) <= cert::comparisonEnvelope(got, c),
          "sequential F32 result exceeds certificate");
      ++tests; return c;
    };
    input[0] = bf(1.5f); codes[0] = -128; check(-24, 0.125f);
    clear(); input[0] = bf(2); input[1] = bf(-4); input[2] = bf(0.5f);
    codes[0] = -128; codes[1] = 127; codes[2] = -1; check(-191.125, 0.25f);
    clear(); input[0] = bf(0x1p80f); input[1] = bf(1); input[2] = bf(0x1p80f);
    codes[0] = 1; codes[1] = 1; codes[2] = -1; require(check(0.125, 0.125f).referenceEnvelope == 0,
        "simple cancellation reference was not exact");
    clear(); input[0] = bf(-0x1p93f); input[1] = bf(0x1p20f); input[2] = bf(0x1p-100f);
    input[3] = bf(0x1p100f); input[4] = bf(0x1p20f);
    codes[0] = -128; codes[1] = 1; codes[2] = 1; codes[3] = -1; codes[4] = -1;
    require(check(0x1p-120, 0x1p-20f).referenceEnvelope == 0, "nested cancellation residual lost");
    clear(); input[0] = bf(0x1p60f); input[1] = bf(128); codes[0] = codes[1] = 1;
    auto tie = check(0x1p60); require(tie.referenceEnvelope == 128 && tie.norm == 0x1p60 + 256,
        "FP64 even-lower tie or upward norm differs");
    input[2] = bf(256); codes[2] = 1; require(check(0x1p60 + 512).referenceEnvelope == 128,
        "FP64 even-upper tie differs");
    clear(); input[0] = bf(-0x1p60f); input[1] = bf(-128); codes[0] = codes[1] = 1;
    check(-0x1p60);
    clear(); input[0] = 0x8000; codes[0] = -128; input[1] = 0; codes[1] = 127;
    require(check(0).norm == 0, "signed-zero norm differs");
    clear(); input[0] = 1; codes[0] = 1; check(0x1p-133);
    check(0x1p-282, std::numeric_limits<float>::denorm_min());
    clear(); input[0] = bf(1); input[1] = bf(0x1p-24f); codes[0] = codes[1] = 1;
    check(1 + 0x1p-24);
    std::fill(input.begin(), input.end(), 0x007f); std::fill(codes.begin(), codes.end(), -128);
    const auto subnormal = check(-double(K) * 127 * 128 * 0x1p-133);
    require(std::abs(subnormal.dotScaled) > cert::comparisonEnvelope(0, subnormal),
        "certificate silently permits BF16 input flushing");
    std::fill(input.begin(), input.end(), 0x7f7f); std::fill(codes.begin(), codes.end(), 127);
    const auto large = cert::certificate(codes.data(), K, std::numeric_limits<float>::max(), input.data());
    require(std::isfinite(large.dotScaled) && std::isfinite(large.norm), "finite-source exact reference overflowed FP64");
    require(!std::isfinite(simulated(codes, 1, input)), "overflow fixture does not overflow F32");
    rejected([&] { cert::comparisonEnvelope(std::numeric_limits<double>::infinity(), large); }); ++tests;
    clear(); input[0] = 0x7f80; rejected([&] { cert::certificate(codes.data(), K, 1, input.data()); });
    input[0] = 0x7fc1; rejected([&] { cert::certificate(codes.data(), K, 1, input.data()); }); clear();
    for (float scale : {0.0f, -0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
      rejected([&] { cert::certificate(codes.data(), K, scale, input.data()); });
    rejected([&] { cert::certificate(codes.data(), 1, 1, input.data()); });
    rejected([&] { cert::certificate(nullptr, K, 1, input.data()); });
    rejected([&] { cert::certificate(codes.data(), K, 1, nullptr); }); tests += 10;
    std::mt19937 random(0x1c8u + K);
    for (uint32_t trial = 0; trial < 64; ++trial) {
      double exact = 0;
      for (uint32_t k = 0; k < K; ++k) {
        const int mantissa = int(random() % 511) - 255, exponent = int(random() % 25) - 12;
        input[k] = bf(std::ldexp(float(mantissa), exponent)); codes[k] = int8_t(int(random() % 256) - 128);
        // All terms lie on a common 2^-12 lattice and the total fits <=52
        // significant bits; this independently computed FP64 sum is exact.
        exact += cert::bfNumber(input[k]) * codes[k];
      }
      const auto c = check(exact * 0.03125, 0.03125f);
      require(c.referenceEnvelope == 0, "small-range random exact reference rounded");
    }
  }
  std::cout << "{\"pass\":true,\"gpu_commands\":0,\"model_payload_scans\":0,\"checks\":" << tests
      << ",\"reference\":\"exact signed BF16/I8 dyadic superaccumulator with one FP64 RNE\"}\n";
}
