// CPU-only C1/C2 per-output order/byte expectation checks. No Metal backend.
#include "flash/FlashGatheredI8QMV.hpp"
#include <bit>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
namespace qmv = splash::flash::gathered_i8_qmv;
void require(bool value, const char *reason) { if (!value) throw std::runtime_error(reason); }
template<class F> void rejects(F &&f, const char *reason) {
  try { f(); } catch (const std::invalid_argument &) { return; }
  throw std::runtime_error(reason);
}
uint16_t bf16(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
float sanitized(uint16_t bits, uint32_t &diagnostics) {
  if ((bits & 0x7f80u) == 0x7f80u) { diagnostics |= 4; return 0.0f; }
  return number(bits);
}
float fixtureTree(std::array<float, 32> values) {
  // This fixed CPU reference tree does not claim Metal's simd_sum tree/order.
  for (uint32_t step = 16; step; step /= 2)
    for (uint32_t lane = 0; lane < step; ++lane) values[lane] += values[lane + step];
  return values[0];
}
std::array<float, 2> scalarC1(const std::vector<uint16_t> &x,
                            const std::array<std::vector<int8_t>, 2> &codes,
                            uint32_t &diagnostics) {
  std::array<float, 2> result{};
  for (uint32_t column = 0; column < 2; ++column) {
    std::array<float, 32> lanes{};
    for (uint32_t lane = 0; lane < 32; ++lane)
      for (uint32_t k = lane; k < x.size(); k += 32)
        lanes[lane] += float(codes[column][k]) * sanitized(x[k], diagnostics);
    result[column] = fixtureTree(lanes);
  }
  return result;
}
std::array<float, 2> sharedC2(const std::vector<uint16_t> &x,
                            const std::array<std::vector<int8_t>, 2> &codes,
                            uint32_t &diagnostics) {
  std::array<std::array<float, 32>, 2> lanes{};
  for (uint32_t lane = 0; lane < 32; ++lane)
    for (uint32_t k = lane; k < x.size(); k += 32) {
      const float activation = sanitized(x[k], diagnostics);
      for (uint32_t column = 0; column < 2; ++column)
        lanes[column][lane] += float(codes[column][k]) * activation;
    }
  return {fixtureTree(lanes[0]), fixtureTree(lanes[1])};
}
void checkColumns() {
  require(qmv::numericalPolicy(1) == qmv::kPolicy && qmv::numericalPolicy(2) == qmv::kC2Policy &&
      qmv::numericalPolicy(1) != qmv::numericalPolicy(2), "C1/C2 reduction identities not separate");
  rejects([] { (void)qmv::numericalPolicy(3); }, "unsupported columns policy admitted");
  for (uint32_t rows : {1u, 2u, 3u, 4u, 8u, 15u, 16u}) {
    const auto g = qmv::geometry(rows);
    require(g.gateColumnGroups / 2 == 80 && g.downColumnGroups / 2 == 320 &&
        (g.gateColumnGroups + g.downColumnGroups) / 2 * rows * 10 == 4000 * rows,
        "C2 exact half-grid launch differs");
  }
  uint64_t pairedColumns = 0;
  for (uint32_t width : {640u, 2560u})
    for (uint32_t fixture = 0; fixture < 16; ++fixture) {
      std::vector<uint16_t> x(width);
      std::array<std::vector<int8_t>, 2> codes{std::vector<int8_t>(width), std::vector<int8_t>(width)};
      for (uint32_t k = 0; k < width; ++k) {
        const int32_t value = int32_t((k * 173 + fixture * 31) % 211) - 105;
        x[k] = bf16(float(value) * (fixture % 2 ? 0.00390625f : 16.0f));
        for (uint32_t column = 0; column < 2; ++column)
          codes[column][k] = int8_t(int32_t((k * 53 + fixture * 73 + column * 31) % 255) - 127);
      }
      if (fixture == 0) { std::fill(x.begin(), x.end(), bf16(-0.0f)); }
      if (fixture == 1) { x[0] = 0x7fc0; x[32] = 0x7f80; x[63] = 0xff80; }
      if (fixture == 2) {
        // Cancellation in each SIMD lane, with different neighboring columns.
        for (uint32_t k = 0; k < width; ++k) {
          x[k] = bf16(k / 32 % 2 ? -128.0f : 128.0f);
          codes[0][k] = 127; codes[1][k] = -126;
        }
        x.back() = bf16(-127.5f);
      }
      uint32_t d1 = 0x80000000u, d2 = d1;
      const auto c1 = scalarC1(x, codes, d1), c2 = sharedC2(x, codes, d2);
      require(d1 == d2 && d1 == (fixture == 1 ? 0x80000004u : 0x80000000u),
          "C1/C2 finite operand/sticky diagnostic expectation differs");
      for (uint32_t column = 0; column < 2; ++column) {
        require(std::bit_cast<uint32_t>(c1[column]) == std::bit_cast<uint32_t>(c2[column]),
            "C2 changed per-output F32 laneK accumulation order");
        for (float scale : {0.003f, 0.125f, 1.0f, 31.75f})
          require(bf16(c1[column] * scale) == bf16(c2[column] * scale),
              "C1/C2 late-scale BF16 byte expectation differs");
        ++pairedColumns;
      }
    }
  const char *oldBase = std::getenv(qmv::kFlag), *oldColumns = std::getenv(qmv::kColumnsFlag);
  const std::string savedBase = oldBase ? oldBase : "", savedColumns = oldColumns ? oldColumns : "";
  const bool hadBase = oldBase != nullptr, hadColumns = oldColumns != nullptr;
  ::unsetenv(qmv::kColumnsFlag); ::setenv(qmv::kFlag, "0", 1);
  require(qmv::requestedColumns() == 1, "C1 default control changed");
  ::setenv(qmv::kColumnsFlag, "2", 1);
  rejects([] { (void)qmv::requestedColumns(); }, "C2 admitted with base QMV flag0");
  ::setenv(qmv::kFlag, "1", 1);
  require(qmv::requestedColumns() == 2, "explicit C2 flag rejected");
  ::setenv(qmv::kColumnsFlag, "1", 1);
  require(qmv::requestedColumns() == 1, "explicit C1 flag rejected");
  for (const char *bad : {"", "0", "3", "true", "02", " 2", "-1"}) {
    ::setenv(qmv::kColumnsFlag, bad, 1);
    rejects([] { (void)qmv::requestedColumns(); }, "malformed columns flag admitted");
  }
  if (hadBase) ::setenv(qmv::kFlag, savedBase.c_str(), 1); else ::unsetenv(qmv::kFlag);
  if (hadColumns) ::setenv(qmv::kColumnsFlag, savedColumns.c_str(), 1); else ::unsetenv(qmv::kColumnsFlag);
  std::cout << "{\"cpu_checks\":\"passed\",\"gpu_work\":false,\"paired_cpu_columns\":" << pairedColumns
            << ",\"paired_gpu_byte_qualification\":\"pending\",\"policy\":\"" << qmv::kC2Policy << "\"}\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 2 || std::string_view(argv[1]) != "--cpu-self-test")
      throw std::invalid_argument("CPU-only helper requires --cpu-self-test; no GPU mode exists");
    checkColumns(); return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
