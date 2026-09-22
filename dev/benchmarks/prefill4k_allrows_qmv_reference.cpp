// CPU test for the independent Root-run dot reference and rejection gate.
#include "prefill4k_allrows_qmv_reference.hpp"
#include <iostream>
#include <string_view>
#include <vector>
namespace {
namespace ref = splash::flash::gathered_i8_qmv_reference;
void require(bool pass, const char *reason) { if (!pass) throw std::runtime_error(reason); }
void tests() {
  require(ref::bf16FromF64(1.00390625) == 0x3f80 &&
      ref::bf16FromF64(1.01171875) == 0x3f82 && ref::bf16FromF64(-0.0) == 0x8000 &&
      ref::bf16FromF64(0x1p-134) == 0 && ref::bf16FromF64(0x1.8p-133) == 2,
      "independent F64-to-BF16 ties/zero/subnormal rounding differs");
  require(ref::bf16FromF64(std::numeric_limits<double>::infinity()) == 0x7f80 &&
      ref::bf16FromF64(std::numeric_limits<double>::quiet_NaN()) == 0x7fc0,
      "independent F64 exceptional rounding differs");
  std::vector<uint16_t> x(640, 0x3f80);
  std::vector<int8_t> codes(640, 1);
  auto normal = ref::reference(x, codes, 0.125f);
  require(normal.dot == 640 && normal.sumAbsProducts == 640 && normal.scaled == 80 &&
      normal.exactProductRoundingError == 0 && !normal.exceptional &&
      ref::assess(normal, 640, 80, ref::bf16FromF64(80), 0).regularFinitePrimitivePass,
      "regular exact dot/reference fixture differs");
  auto wrong = ref::assess(normal, 650, 81.25f, ref::bf16FromF64(81.25), 0);
  require(!wrong.dotWithinBound && !wrong.regularFinitePrimitivePass, "bad regular dot passed broad bound");
  std::fill(x.begin(), x.end(), 0);
  x[0] = ref::bf16FromF64(0x1p100); x[1] = 0x3f80; x[2] = ref::bf16FromF64(-0x1p100);
  auto cancelled = ref::reference(x, codes, 1);
  const auto lostOne = ref::assess(cancelled, 0, 0, 0, 0);
  require(cancelled.dot == 1 && lostOne.dotWithinBound && lostOne.scaledWithinBound &&
      lostOne.strictSensitive && !lostOne.strictBF16Pass && !lostOne.regularFinitePrimitivePass,
      "cancellation fixture passed on norm bound alone");
  std::fill(x.begin(), x.end(), 0);
  auto zero = ref::reference(x, codes, 1);
  auto wrongZero = ref::assess(zero, -0.0f, -0.0f, 0x8000, 0);
  require(wrongZero.dotWithinBound && !wrongZero.signMatches && !wrongZero.negativeZeroMatches &&
      !wrongZero.regularFinitePrimitivePass, "wrong signed zero passed absolute bound");
  x[0] = 0x0001;
  auto subnormal = ref::reference(x, codes, 1);
  require(subnormal.productSubnormal && subnormal.scaledSubnormal && subnormal.exceptional &&
      !ref::assess(subnormal, float(subnormal.dot), float(subnormal.scaled), subnormal.bf16, 0).regularFinitePrimitivePass,
      "subnormal fixture automatically qualified");
  x[0] = 0x7f7f; codes[0] = 127;
  auto overflow = ref::reference(x, codes, 1);
  require(overflow.productOverflow && overflow.exceptional && overflow.expectedStickyMinimum == 4 &&
      !ref::assess(overflow, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 0x7f80, 4).regularFinitePrimitivePass,
      "overflow fixture automatically qualified");
  x[0] = 0x7fc0; codes[0] = 1;
  auto nonfinite = ref::reference(x, codes, 1);
  require(nonfinite.dot == 0 && nonfinite.nonfiniteInput && nonfinite.expectedStickyMinimum == 4 &&
      !ref::assess(nonfinite, 0, 0, 0, 0).diagnosticsCoverExpected &&
      ref::assess(nonfinite, 0, 0, 0, 0x80000004u).diagnosticsCoverExpected,
      "nonfinite sanitizer diagnostic requirement differs");
}
}
int main(int argc, char **argv) {
  try {
    if (argc != 2 || std::string_view(argv[1]) != "--cpu-self-test")
      throw std::invalid_argument("CPU-only reference requires --cpu-self-test; no GPU mode exists");
    tests();
    std::cout << "{\"cpu_reference_checks\":\"passed\",\"gpu_work\":false,\"cancellation_norm_only_rejected\":true,\"numerical_qualification\":\"pending\"}\n";
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
