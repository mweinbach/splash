// CPU-only ABI/geometry/ID/scratch/alias policy checks. No Metal headers/backend.
#include "prefill4k_allrows_qmv.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {
namespace qmv = splash::flash::gathered_i8_qmv;
void require(bool value, const char *reason) {
  if (!value) throw std::runtime_error(reason);
}
template<class Function> void rejects(Function &&function, const char *reason) {
  try { function(); } catch (const std::invalid_argument &) { return; }
  throw std::runtime_error(reason);
}
uint16_t bf16(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
float sanitized(uint16_t value, uint32_t &diagnostics) {
  if ((value & 0x7f80u) == 0x7f80u) { diagnostics |= qmv::kInvalidNumerics; return 0.0f; }
  return number(value);
}
void cpuChecks() {
  require(sizeof(FlashGatheredI8QMVParams) == 16 && qmv::kThreads == 128,
      "host/shader ABI or fixed launch differs");
  require(qmv::kPolicy != "private-allrows-full512-signed-i8-f32-late-row-scale-bf16-dots-m16tight-small-m32m64-large-no-target-q4gpu-v1",
      "QMV reduction identity aliases MPP control");
  for (uint32_t rows : {1u, 2u, 3u, 4u, 8u, 15u, 16u}) {
    const auto g = qmv::geometry(rows);
    require(g.inputBytes == uint64_t{rows} * 5120 && g.idBytes == uint64_t{rows} * 80 &&
        g.intermediateBytes == uint64_t{rows} * 12800 &&
        g.expertDownBytes == uint64_t{rows} * 51200 &&
        g.gateColumnGroups == 160 && g.downColumnGroups == 640,
        "canonical tensor extent or launch differs");
    // Synthetic CPU metadata ranges, never dereferenced.
    const qmv::ByteView input{0x10000000, g.inputBytes}, ids{0x20000000, g.idBytes},
        activated{0x30000000, g.intermediateBytes}, down{0x40000000, g.expertDownBytes},
        diag{0x50000000, 4};
    qmv::validateViews(g, input, ids, activated, diag, false);
    qmv::validateViews(g, activated, ids, down, diag, true);
    rejects([&] { qmv::validateViews(g, {input.address, input.bytes - 1}, ids, activated, diag, false); },
        "undersized hidden view admitted");
    rejects([&] { qmv::validateViews(g, activated, ids, {down.address, down.bytes - 1}, diag, true); },
        "undersized expert-down view admitted");
    rejects([&] { qmv::validateViews(g, input, {ids.address + 1, ids.bytes}, activated, diag, false); },
        "misaligned signed I64 IDs admitted");
    rejects([&] { qmv::validateViews(g, input, ids, {input.address + input.bytes - 2, activated.bytes}, diag, false); },
        "partial write/input overlap admitted");
    rejects([&] { qmv::validateViews(g, input, ids, activated, {activated.address, 4}, false); },
        "diagnostic/output alias admitted");
    rejects([&] { qmv::validateViews(g, input, ids, {0, activated.bytes}, diag, false); },
        "missing output pointer admitted");
    rejects([&] { qmv::validateViews(g, input, ids,
        {std::numeric_limits<uintptr_t>::max() - 1, activated.bytes}, diag, false); },
        "overflowing address extent admitted");
    require(!qmv::overlaps(input, {input.address + input.bytes, 2}),
        "adjacent disjoint view rejected");
    std::vector<int64_t> routeIDs(uint64_t{rows} * 10);
    for (uint32_t row = 0; row < rows; ++row)
      for (uint32_t slot = 0; slot < 10; ++slot)
        routeIDs[row * 10 + slot] = (row * 73 + slot * 53) % 512;
    require(qmv::fixtureIDsValid(routeIDs, rows) &&
        qmv::fixtureIDDiagnostics(routeIDs, rows, 0x80000000u) == 0x80000000u,
        "valid canonical I64 ownership rejected or sticky bits lost");
    for (int64_t invalid : {int64_t{-1}, int64_t{512}, int64_t{1} << 32,
                           std::numeric_limits<int64_t>::max()}) {
      auto bad = routeIDs; bad.back() = invalid;
      require(!qmv::fixtureIDsValid(bad, rows) &&
          qmv::fixtureIDDiagnostics(bad, rows, 0x80000000u) == 0x80000001u,
          "malformed I64 original ID diagnostics differ");
    }
    auto duplicate = routeIDs; duplicate[1] = duplicate[0];
    require(qmv::fixtureIDsValid(duplicate, rows) &&
        qmv::fixtureIDDiagnostics(duplicate, rows, 0x80000000u) == 0x80000001u,
        "duplicate finite computations must flagID without flagNumeric");
    require(qmv::fixtureIDDiagnostics(std::span(routeIDs).first(routeIDs.size() - 1), rows) == 2,
        "malformed fixture ID extent admitted");
  }
  for (uint32_t rows : {0u, 17u, 256u, UINT32_MAX})
    rejects([&] { (void)qmv::geometry(rows); }, "unsupported row count admitted");
  for (uint32_t selections : {0u, 1u, 9u, 11u, UINT32_MAX})
    rejects([&] { (void)qmv::geometry(1, selections); }, "nonK10 gathered policy admitted");
  require(!qmv::validGeometry(1, 10, 511), "partial inventory admitted");
  std::array<uint32_t, 512> ranks{};
  for (uint32_t expert = 0; expert < 512; ++expert) ranks[expert] = (expert * 173 + 31) % 512;
  auto sorted = ranks; std::sort(sorted.begin(), sorted.end());
  for (uint32_t i = 0; i < 512; ++i) require(sorted[i] == i, "nonidentity fixture ranks not bijective");
  for (uint32_t expert = 0; expert < 512; ++expert) {
    uint32_t sticky = 0x80000000u;
    require(qmv::checkedRank(expert, ranks, sticky) == ranks[expert] && sticky == 0x80000000u,
        "originalID mistakenly used as persisted rank");
  }
  for (uint32_t invalid : {512u, UINT32_MAX}) {
    auto bad = ranks; bad[71] = invalid;
    uint32_t sticky = 0x80000000u;
    require(qmv::checkedRank(71, bad, sticky) == UINT32_MAX && sticky == 0x80000001u,
        "Full512 missing/corrupt rank must flagID");
  }
  require(bf16(-0.0f) == 0x8000 && bf16(1.00390625f) == 0x3f80,
      "BF16 fixture rounding or signed zero differs");
  uint32_t sticky = 0x80000000u;
  require(sanitized(0x7fc0, sticky) == 0 && sanitized(0x7f80, sticky) == 0 &&
      sanitized(0xff80, sticky) == 0 && sticky == 0x80000004u,
      "original nonfinite zero/sticky behavior differs");
  require(std::signbit(sanitized(0x8000, sticky)), "finite signed zero changed");
  // Proves the scale boundary is meaningful. Early per-code BF16 rounding is
  // not the candidate's late-scale signed-I8 dot policy.
  const float late = (float(int8_t{127}) + float(int8_t{-126})) * 0.003f;
  const float early = number(bf16(float(int8_t{127}) * 0.003f)) +
      number(bf16(float(int8_t{-126}) * 0.003f));
  require(bf16(late) != bf16(early), "late-scale adversarial fixture lacks discrimination");
  const char *old = std::getenv(qmv::kFlag);
  const std::string saved = old ? old : "";
  const bool had = old != nullptr;
  ::unsetenv(qmv::kFlag); require(!qmv::requested(), "unset flag must select MPP control");
  ::setenv(qmv::kFlag, "0", 1); require(!qmv::requested(), "flag0 must select MPP control");
  ::setenv(qmv::kFlag, "1", 1); require(qmv::requested(), "flag1 must select QMV");
  for (const char *bad : {"", "true", "01", " 1", "2", "-1"}) {
    ::setenv(qmv::kFlag, bad, 1);
    rejects([] { (void)qmv::requested(); }, "malformed runtime policy flag admitted");
  }
  if (had) ::setenv(qmv::kFlag, saved.c_str(), 1); else ::unsetenv(qmv::kFlag);
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 2 || std::string_view(argv[1]) != "--cpu-self-test")
      throw std::invalid_argument("CPU-only helper requires --cpu-self-test; no GPU mode exists");
    cpuChecks();
    std::cout << "{\"cpu_checks\":\"passed\",\"gpu_work\":false,\"numerical_qualification\":\"pending\",\"policy\":\""
              << qmv::kPolicy << "\"}\n";
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
