#include "ops/Linear.hpp"
#include "metal/abi/ExecutionGeometry.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;

// Existing profile encodings keep their values when the new paired variant
// is appended. N256 remains the sequential gate/up choice.
static_assert(uint8_t(LinearTile::N128) == 0);
static_assert(uint8_t(LinearTile::N256) == 1);
static_assert(uint8_t(LinearTile::Paired128) == 2);
static_assert(uint8_t(LinearTile::N64) == 3);
static_assert(uint8_t(LinearTile::Paired256) == 4);

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid Linear plan or buffer was accepted");
}

uint16_t bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}

float fp32(uint16_t value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

// The device policy restated independently of Linear.cpp: rules over the GPU
// family, the GPU core count and the workload's tile counts. A per-shape or
// per-machine exception added to the operator must fail here.
struct ExpectedConfig final {
  LinearTile tile;
  uint32_t groups;
  LinearSimdgroups simdgroups = LinearSimdgroups::Eight;
};

// All non-baseline choices emitted by the 12-pair projection sweep and then
// confirmed in complete decode graphs on the 80-core Apple10 GPU. This table
// is measured evidence, independent of the production policy's inequalities.
constexpr std::array kMeasuredN64Workloads{
    LinearWorkload{{256, 5120}, 24},
    LinearWorkload{{256, 5120}, 32},
    LinearWorkload{{1280, 5120}, 24},
    LinearWorkload{{1280, 5120}, 32},
    LinearWorkload{{5120, 4096}, 24},
    LinearWorkload{{5120, 4096}, 32},
    LinearWorkload{{5120, 6144}, 24, LinearPhase::Decode, LinearEpilogue::Residual},
    LinearWorkload{{5120, 6144}, 32, LinearPhase::Decode, LinearEpilogue::Residual},
    LinearWorkload{{5120, 17408}, 24},
    LinearWorkload{{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual},
    LinearWorkload{{5120, 17408}, 32},
    LinearWorkload{{5120, 17408}, 32, LinearPhase::Decode, LinearEpilogue::Residual},
    LinearWorkload{{5120, 25600}, 24},
    LinearWorkload{{5120, 25600}, 32},
    LinearWorkload{{6144, 5120}, 24},
    LinearWorkload{{6144, 5120}, 32},
    LinearWorkload{{14336, 5120}, 24},
    LinearWorkload{{14336, 5120}, 32},
    LinearWorkload{{16640, 5120}, 24}};

bool measuredN64Winner(LinearWorkload workload) {
  return std::find(kMeasuredN64Workloads.begin(), kMeasuredN64Workloads.end(), workload) !=
      kMeasuredN64Workloads.end();
}

// The separate lossless M8 gate/up measurement selects exactly one operator
// key. Keep this literal evidence separate from the N64 width-three/four set.
constexpr LinearWorkload kMeasuredPairedGateUpWorkload{
    {17408, 5120}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};

// Tiles on the busiest core when `groups` threadgroups are placed round-robin
// on `cores` and group g streams tiles g, g + groups, ...; the operator's
// closed form is restated tile by tile.
uint32_t busiestCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) {
  std::vector<uint32_t> load(cores);
  for (uint32_t tile = 0; tile < tiles; ++tile) ++load[tile % groups % cores];
  return *std::max_element(load.begin(), load.end());
}

// Groups per core up to which the one-tile grid wins, resident groups per
// core (one wave) and tiles per core from which the many-wave grid wins.
struct GroupRule final { uint32_t grid, wave, manyWaves; };

// The Apple10 rule as properties: the grid up to one wave per core and from
// many waves per core; between them the smallest count of at most two-tile
// groups, never above one wave, that leaves every core at ceil(tiles / cores)
// tiles while at least three quarters of the full-grid limit stays resident,
// and one full wave of longer chains when no such count exists.
uint32_t expectedGroups(uint32_t tiles, uint32_t cores, GroupRule rule) {
  if (tiles <= rule.grid * cores || tiles >= rule.manyWaves * cores) return tiles;
  for (uint32_t groups = (tiles + 1) / 2; groups <= rule.wave * cores; ++groups)
    if (groups >= rule.grid * cores * 3 / 4 &&
        busiestCoreTiles(tiles, groups, cores) == (tiles + cores - 1) / cores)
      return groups;
  return rule.wave * cores;
}

ExpectedConfig expectedDecode(uint32_t family, uint32_t cores, LinearMatrix matrix,
                              uint32_t lanes, LinearEpilogue epilogue) {
  constexpr GroupRule n128{4, 4, 12}, m16{5, 4, 12}, n256{3, 3, 8}, gateUp{3, 3, 8},
      fourSimdgroups{8, 8, 24};
  const uint32_t tiles128 = matrix.outputSize / 128;
  const uint32_t tiles256 = matrix.outputSize / 256;
  // Apple9 is unmeasured under the balanced rule and keeps its one-tile grids
  // and the fused gate/up clamp of 2.25 resident groups per core.
  const auto groups = [&](uint32_t tiles, GroupRule rule) {
    return family >= 10 ? expectedGroups(tiles, cores, rule) : tiles;
  };
  if (epilogue == LinearEpilogue::GateUp) {
    if (family == 10 && cores == 80 &&
        LinearWorkload{matrix, lanes * 8, LinearPhase::Decode, epilogue} ==
            kMeasuredPairedGateUpWorkload)
      return {LinearTile::Paired256, 68};
    if (family >= 10) return {LinearTile::N256, expectedGroups(tiles256, cores, gateUp)};
    return {LinearTile::N256, std::min(tiles256, uint32_t(std::lround(2.25 * cores)))};
  }
  if (family == 10 && cores == 80 &&
      measuredN64Winner({matrix, lanes * 8, LinearPhase::Decode, epilogue}))
    return {LinearTile::N64, matrix.outputSize / 64};
  if (lanes == 1) return {LinearTile::Paired128, groups(tiles128, n128)};
  if (lanes == 3 && (family >= 10 ||
      (family == 9 && epilogue == LinearEpilogue::None)))
    return {LinearTile::N128, groups(tiles128, fourSimdgroups), LinearSimdgroups::Four};
  if (lanes >= 3 && epilogue == LinearEpilogue::None && tiles256 >= 2 * cores)
    return {LinearTile::N256, groups(tiles256, n256)};
  return {LinearTile::N128, groups(tiles128, lanes == 2 ? m16 : n128)};
}

std::string expectedPipeline(const ExpectedConfig &expected, uint32_t lanes,
                             LinearEpilogue epilogue) {
  if (epilogue == LinearEpilogue::GateUp && expected.tile == LinearTile::Paired256)
    return "decode_linear_q4_n256_gate_up_paired";
  if (epilogue == LinearEpilogue::GateUp)
    return lanes == 1 ? "decode_linear_q4_n256_gate_up" : lanes == 2 ? "decode_linear_q4_n256_gate_up_m16"
        : lanes == 3 ? "decode_linear_q4_n256_m24" : "decode_linear_q4_n256_m32";
  std::string name = expected.tile == LinearTile::N64 ? "decode_linear_q4_n64" :
      expected.tile == LinearTile::N256 ? "decode_linear_q4_n256" : "decode_linear_q4_n128";
  if (epilogue == LinearEpilogue::Residual) name += "_residual";
  if (expected.tile == LinearTile::Paired128) return name + "_paired";
  if (lanes > 1) name += "_m" + std::to_string(lanes * 8);
  if (expected.simdgroups == LinearSimdgroups::Four) name += "_sg4";
  return name;
}

void baselinePlans() {
  constexpr uint32_t assumedCores = 64;  // policy default when the count is unknown
  for (const uint32_t family : {9U, 10U, 11U}) {
    for (const uint32_t reportedCores : {0U, 10U, 16U, 20U, 40U, 80U}) {
      const uint32_t cores = reportedCores ? reportedCores : assumedCores;
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = reportedCores;
      Q4Linear linear(device);
      for (const uint32_t hidden : {5120U, 2048U}) {
        const bool large = hidden == 5120;
        const uint32_t intermediate = large ? 17408U : 6144U;
        struct Call final { LinearMatrix matrix; LinearEpilogue epilogue; };
        const std::array calls{
            Call{{large ? 16640U : 12544U, hidden}, LinearEpilogue::None},
            Call{{large ? 14336U : 9216U, hidden}, LinearEpilogue::None},
            Call{{hidden, large ? 6144U : 4096U}, LinearEpilogue::Residual},
            Call{{intermediate, hidden}, LinearEpilogue::GateUp},
            Call{{hidden, intermediate}, LinearEpilogue::Residual},
            Call{{hidden, intermediate}, LinearEpilogue::None},
            Call{{hidden, large ? 25600U : 16384U}, LinearEpilogue::None},
            Call{{248320, hidden}, LinearEpilogue::None},
            Call{{hidden / 4, hidden}, LinearEpilogue::None},
            Call{{6144, hidden}, LinearEpilogue::None},
            Call{{256, hidden}, LinearEpilogue::None}};
        for (const auto &call : calls) {
          for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
            const auto plan = linear.plan({call.matrix, lanes * 8, LinearPhase::Decode, call.epilogue});
            const auto expected = expectedDecode(family, cores, call.matrix, lanes, call.epilogue);
            require(plan.configuration() ==
                        LinearConfig{expected.tile, expected.groups, expected.simdgroups},
                    "Linear decode policy differs from its stated rules");
            require(plan.threadsPerThreadgroup() ==
                        (expected.simdgroups == LinearSimdgroups::Four ? 128U : 256U),
                    "Linear decode scope differs from its configuration");
            require(plan.pipeline() == expectedPipeline(expected, lanes, call.epilogue),
                    "Linear decode pipeline differs from its configuration");
            require(plan.secondPipeline().empty() ==
                        !(call.epilogue == LinearEpilogue::GateUp && lanes >= 3),
                    "gate/up dispatch decomposition changed");
          }
        }
        for (const LinearMatrix matrix :
             {LinearMatrix{6144, hidden}, LinearMatrix{large ? 16640U : 12544U, hidden},
              LinearMatrix{hidden, large ? 6144U : 4096U}, LinearMatrix{intermediate, hidden}}) {
          for (const uint32_t rows : {1U, 7U, 31U, 32U, 33U, 127U, 2048U}) {
            for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                       LinearEpilogue::UpWithGate}) {
              const auto plan = linear.plan({matrix, rows, LinearPhase::Prefill, epilogue});
              const double rowTiles = (rows + 31) / 32;
              const bool wide = rowTiles * (matrix.outputSize / 256) >= 8.0 * cores;
              const LinearConfig expected = family >= 10
                  ? LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four}
                  : LinearConfig{epilogue == LinearEpilogue::UpWithGate || wide
                                     ? LinearTile::N256 : LinearTile::N128, 0};
              require(plan.configuration() == expected,
                      "prefill policy differs from its stated rule");
              require(plan.threadsPerThreadgroup() == (family >= 10 ? 128U : 256U),
                      "prefill cooperative execution scope changed");
            }
          }
        }
      }
    }
  }
  // Anchors from the measured machines: 16- and 20-core Apple10 GPUs (M5 Pro)
  // and a 40-core Apple9 GPU (M3 Max). Changing a rule must change these
  // knowingly.
  const auto configured = [](uint32_t family, uint32_t cores, LinearWorkload workload) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = cores;
    return Q4Linear(device).plan(workload).configuration();
  };
  const LinearWorkload gateUp{{17408, 5120}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  require(configured(10, 16, gateUp) == LinearConfig{LinearTile::N256, 36} &&
              configured(10, 20, gateUp) == LinearConfig{LinearTile::N256, 48} &&
              configured(9, 40, gateUp) == LinearConfig{LinearTile::N256, 68} &&
              configured(10, 0, gateUp) == LinearConfig{LinearTile::N256, 68},
          "fused gate/up grid does not follow the balanced two-tile rule");
  // Apple9 keeps the shipped one-tile grids and 2.25 gate/up groups per core.
  require(configured(9, 16, gateUp) == LinearConfig{LinearTile::N256, 36} &&
              configured(9, 20, gateUp) == LinearConfig{LinearTile::N256, 45} &&
              configured(9, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 130} &&
              configured(9, 16, {{16640, 5120}, 16}) == LinearConfig{LinearTile::N128, 130} &&
              configured(9, 20, {{16640, 5120}, 32}) == LinearConfig{LinearTile::N256, 65},
          "Apple9 decode grids changed without a measurement");
  // Balanced two-tile groups above one wave: 130 paired tiles keep three
  // groups per core on 20 cores and one full wave of longer chains on 16; 98
  // tiles land on 60 and 50 groups; the M16 grid holds to five per core.
  require(configured(10, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 70} &&
              configured(10, 16, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 64} &&
              configured(10, 20, {{12544, 2048}, 8}) == LinearConfig{LinearTile::Paired128, 60} &&
              configured(10, 16, {{12544, 2048}, 8}) == LinearConfig{LinearTile::Paired128, 50} &&
              configured(10, 20, {{16640, 5120}, 16}) == LinearConfig{LinearTile::N128, 75} &&
              configured(10, 16, {{16640, 5120}, 16}) == LinearConfig{LinearTile::N128, 64} &&
              configured(10, 20, {{12544, 2048}, 16}) == LinearConfig{LinearTile::N128, 98} &&
              configured(10, 16, {{16640, 5120}, 24}) ==
                  LinearConfig{LinearTile::N128, 96, LinearSimdgroups::Four} &&
              configured(10, 20, {{16640, 5120}, 24}) ==
                  LinearConfig{LinearTile::N128, 130, LinearSimdgroups::Four} &&
              configured(10, 20, {{16640, 5120}, 32}) == LinearConfig{LinearTile::N256, 45} &&
              configured(10, 20, {{248320, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 1940} &&
              configured(9, 40, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 130} &&
              configured(10, 0, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 130},
          "persistent decode groups changed for the measured shapes");
  require(configured(10, 16, {{14336, 5120}, 24}) ==
              LinearConfig{LinearTile::N128, 112, LinearSimdgroups::Four} &&
          configured(10, 16, {{14336, 5120}, 32}) == LinearConfig{LinearTile::N256, 40} &&
          configured(10, 40, {{14336, 5120}, 32}) == LinearConfig{LinearTile::N128, 112} &&
          configured(9, 40, {{14336, 5120}, 24}) ==
              LinearConfig{LinearTile::N128, 112, LinearSimdgroups::Four} &&
          configured(9, 40, {{248320, 5120}, 24}) ==
              LinearConfig{LinearTile::N128, 1940, LinearSimdgroups::Four} &&
          configured(9, 40, {{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual}) ==
              LinearConfig{LinearTile::N128, 40},
          "decode tile rules changed for the measured shapes");
  require(configured(10, 16, {{5120, 17408}, 8}) == LinearConfig{LinearTile::Paired128, 40} &&
          configured(10, 16, {{5120, 17408}, 16}) == LinearConfig{LinearTile::N128, 40} &&
          configured(10, 16, {{6144, 5120}, 2048, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four} &&
          configured(10, 16, {{17408, 5120}, 2048, LinearPhase::Prefill,
                              LinearEpilogue::UpWithGate}) ==
              LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four} &&
          configured(9, 40, {{6144, 5120}, 2048, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N256, 0} &&
          configured(9, 40, {{6144, 5120}, 32, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N128, 0} &&
          configured(9, 40, {{17408, 5120}, 32, LinearPhase::Prefill,
                              LinearEpilogue::UpWithGate}) ==
              LinearConfig{LinearTile::N256, 0},
          "one-lane pipelining or prefill tile rule changed for the measured shapes");
}

void measuredDecodePolicy() {
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 80;
  Q4Linear linear(device);
  struct Call final { LinearMatrix matrix; LinearEpilogue epilogue; };
  // The complete 48-key sweep includes every selected and kept projection,
  // not just an example of the faster geometry.
  const std::array measuredCalls{
      Call{{256, 5120}, LinearEpilogue::None},
      Call{{1280, 5120}, LinearEpilogue::None},
      Call{{5120, 4096}, LinearEpilogue::None},
      Call{{5120, 6144}, LinearEpilogue::Residual},
      Call{{5120, 17408}, LinearEpilogue::None},
      Call{{5120, 17408}, LinearEpilogue::Residual},
      Call{{5120, 25600}, LinearEpilogue::None},
      Call{{6144, 5120}, LinearEpilogue::None},
      Call{{14336, 5120}, LinearEpilogue::None},
      Call{{16640, 5120}, LinearEpilogue::None},
      Call{{17408, 5120}, LinearEpilogue::GateUp},
      Call{{248320, 5120}, LinearEpilogue::None}};
  uint32_t selectedKeys = 0;
  uint32_t pairedGateUpKeys = 0;
  for (const auto &call : measuredCalls) {
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const LinearWorkload workload{call.matrix, lanes * 8, LinearPhase::Decode, call.epilogue};
      const auto plan = linear.plan(workload);
      const auto expected = expectedDecode(10, 80, call.matrix, lanes, call.epilogue);
      require(plan.configuration() == LinearConfig{expected.tile, expected.groups, expected.simdgroups},
              "80-core decode policy differs from the complete measured choice set");
      const bool selected = plan.configuration().tile == LinearTile::N64;
      require(selected == measuredN64Winner(workload),
              "80-core decode changed a measured nonwinner or omitted a winner");
      selectedKeys += selected;
      const bool paired = plan.configuration().tile == LinearTile::Paired256;
      require(paired == (workload == kMeasuredPairedGateUpWorkload),
              "paired gate/up default changed an unmeasured operator key");
      pairedGateUpKeys += paired;
      const auto candidates = linear.candidates(workload);
      require(!candidates.empty() && candidates.size() <= 40 &&
                  candidates.front().configuration() == plan.configuration(),
              "measured decode baseline is not first in the bounded candidate set");
    }
  }
  require(selectedKeys == kMeasuredN64Workloads.size(),
          "80-core decode did not select all 19 emitted winners");
  require(pairedGateUpKeys == 1,
          "80-core decode did not select exactly the measured M8 gate/up key");

  for (const auto workload : kMeasuredN64Workloads) {
    const auto config = linear.plan(workload).configuration();
    require(config == LinearConfig{LinearTile::N64, workload.matrix.outputSize / 64},
            "measured winner did not retain its exact full N64 grid and scope");
    for (const uint32_t family : {9U, 10U, 11U}) {
      for (const uint32_t cores : {0U, 16U, 20U, 40U, 64U, 79U, 81U}) {
        auto otherDevice = device;
        otherDevice.appleGpuFamily = family;
        otherDevice.gpuCoreCount = cores;
        const auto unchanged = Q4Linear(otherDevice).plan(workload).configuration();
        const auto prior = expectedDecode(family, cores ? cores : 64, workload.matrix,
                                          workload.rows / 8, workload.epilogue);
        require(unchanged == LinearConfig{prior.tile, prior.groups, prior.simdgroups} &&
                    unchanged.tile != LinearTile::N64,
                "measured 80-core policy escaped onto an uncalibrated device");
      }
    }
    for (const uint32_t family : {9U, 11U}) {
      auto otherFamily = device;
      otherFamily.appleGpuFamily = family;
      const auto unchanged = Q4Linear(otherFamily).plan(workload).configuration();
      const auto prior = expectedDecode(family, 80, workload.matrix,
                                        workload.rows / 8, workload.epilogue);
      require(unchanged == LinearConfig{prior.tile, prior.groups, prior.simdgroups},
              "measured 80-core policy changed another GPU family");
    }
  }
  for (const LinearMatrix matrix : {LinearMatrix{2048, 4096}, LinearMatrix{2048, 16384},
                                    LinearMatrix{256, 4864}, LinearMatrix{5120, 3840}}) {
    for (const uint32_t rows : {24U, 32U}) {
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual}) {
        const auto config = linear.plan({matrix, rows, LinearPhase::Decode, epilogue}).configuration();
        const auto prior = expectedDecode(10, 80, matrix, rows / 8, epilogue);
        require(config == LinearConfig{prior.tile, prior.groups, prior.simdgroups} &&
                    config.tile != LinearTile::N64,
                "80-core N64 default broadened beyond the measured input geometry");
      }
    }
  }
}

void pairedGateUpContracts() {
  DeviceCapabilities measuredDevice;
  measuredDevice.appleGpuFamily = 10;
  measuredDevice.gpuCoreCount = 80;
  Q4Linear measured(measuredDevice);
  const LinearConfig promoted{LinearTile::Paired256, 68, LinearSimdgroups::Eight};
  const auto selected = measured.plan(kMeasuredPairedGateUpWorkload);
  require(selected.configuration() == promoted && selected.tileColumns() == 256 &&
              selected.storageRows() == 8 && selected.threadsPerThreadgroup() == 256 &&
              selected.pipeline() == "decode_linear_q4_n256_gate_up_paired" &&
              selected.secondPipeline().empty() && !selected.sumsBytes() &&
              !selected.downSumsBytes() && !selected.gateScratchBytes(),
          "paired M8 gate/up changed its fused dispatch or zero-workspace contract");

  for (uint32_t family : {9U, 10U, 11U}) {
    for (uint32_t reportedCores : {0U, 16U, 20U, 40U, 64U, 79U, 80U, 81U, 96U}) {
      auto device = measuredDevice;
      device.appleGpuFamily = family;
      device.gpuCoreCount = reportedCores;
      const auto plan = Q4Linear(device).plan(kMeasuredPairedGateUpWorkload);
      const bool calibrated = family == 10 && reportedCores == 80;
      const auto expected = expectedDecode(family, reportedCores ? reportedCores : 64,
          kMeasuredPairedGateUpWorkload.matrix, 1, LinearEpilogue::GateUp);
      require(plan.configuration() == LinearConfig{expected.tile, expected.groups, expected.simdgroups} &&
                  (plan.configuration().tile == LinearTile::Paired256) == calibrated &&
                  plan.pipeline() == expectedPipeline(expected, 1, LinearEpilogue::GateUp),
              "paired gate/up default escaped the measured device profile");
    }
  }

  for (const LinearMatrix matrix : {LinearMatrix{512, 256}, LinearMatrix{768, 768},
                                   LinearMatrix{17152, 5120}, LinearMatrix{17664, 5120},
                                   LinearMatrix{17408, 4864}, LinearMatrix{17408, 5376},
                                   LinearMatrix{6144, 2048}}) {
    const LinearWorkload workload{matrix, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
    const auto oldDefault = measured.plan(workload);
    const auto expected = expectedDecode(10, 80, matrix, 1, LinearEpilogue::GateUp);
    require(oldDefault.configuration() == LinearConfig{expected.tile, expected.groups, expected.simdgroups} &&
                oldDefault.configuration().tile == LinearTile::N256 &&
                oldDefault.pipeline() == "decode_linear_q4_n256_gate_up",
            "paired gate/up default broadened beyond the measured matrix");
    // Explicit offline plans remain usable for every aligned M8 gate/up
    // matrix. Only the measured default is shape- and device-bound.
    for (uint32_t groups : {1U, matrix.outputSize / 256}) {
      const auto plan = Q4Linear::plan(workload, {LinearTile::Paired256, groups});
      require(plan.configuration().tile == LinearTile::Paired256 &&
                  plan.configuration().groups == groups && plan.tileColumns() == 256 &&
                  plan.threadsPerThreadgroup() == 256 && plan.secondPipeline().empty() &&
                  plan.pipeline() == "decode_linear_q4_n256_gate_up_paired" &&
                  !plan.sumsBytes() && !plan.downSumsBytes() && !plan.gateScratchBytes(),
              "explicit paired M8 gate/up plan was incorrectly measurement-bound");
    }
    const auto candidates = measured.candidates(workload);
    require(!candidates.empty() && candidates.size() <= 40 &&
                std::any_of(candidates.begin(), candidates.end(), [](const LinearPlan &plan) {
                  return plan.configuration().tile == LinearTile::Paired256;
                }) &&
                std::any_of(candidates.begin(), candidates.end(), [](const LinearPlan &plan) {
                  return plan.configuration().tile == LinearTile::N256;
                }),
            "offline M8 gate/up candidates omitted paired or legacy sequential plans");
  }

  // Prior defaults for this matrix's generic affine/residual workloads. M24
  // already falls within the shipped N64 geometry rule even though it is not
  // one of the 19 literal winners in the separate measured projection table.
  constexpr std::array legacyNonGateDefaults{
      ExpectedConfig{LinearTile::Paired128, 136},
      ExpectedConfig{LinearTile::N128, 136},
      ExpectedConfig{LinearTile::N64, 272},
      ExpectedConfig{LinearTile::N128, 136}};
  for (uint32_t rows : {8U, 16U, 24U, 32U}) {
    for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                         LinearEpilogue::GateUp}) {
      const LinearWorkload workload{kMeasuredPairedGateUpWorkload.matrix, rows,
                                    LinearPhase::Decode, epilogue};
      const bool supported = rows == 8 && epilogue == LinearEpilogue::GateUp;
      const auto candidates = measured.candidates(workload);
      require(!candidates.empty() && candidates.size() <= 40 &&
                  std::any_of(candidates.begin(), candidates.end(), [](const LinearPlan &plan) {
                    return plan.configuration().tile == LinearTile::Paired256;
                  }) == supported,
              "paired gate/up candidate escaped its M8 epilogue set");
      if (!supported) {
        rejects([&] { (void)Q4Linear::plan(workload, {LinearTile::Paired256, 1}); });
        const auto expected = epilogue == LinearEpilogue::GateUp
            ? expectedDecode(10, 80, workload.matrix, rows / 8, epilogue)
            : legacyNonGateDefaults[rows / 8 - 1];
        require(measured.plan(workload).configuration() ==
                    LinearConfig{expected.tile, expected.groups, expected.simdgroups} &&
                    measured.plan(workload).configuration().tile != LinearTile::Paired256,
                "paired gate/up changed another decode width or epilogue");
      }
    }
  }
  for (uint32_t rows : {1U, 8U, 32U, 2048U}) {
    for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                         LinearEpilogue::UpWithGate}) {
      const LinearWorkload workload{kMeasuredPairedGateUpWorkload.matrix, rows,
                                    LinearPhase::Prefill, epilogue};
      const auto candidates = measured.candidates(workload);
      require(!candidates.empty() && candidates.size() <= 40 &&
                  std::none_of(candidates.begin(), candidates.end(), [](const LinearPlan &plan) {
                    return plan.configuration().tile == LinearTile::Paired256;
                  }) &&
                  measured.plan(workload).configuration().tile != LinearTile::Paired256,
              "paired M8 gate/up escaped into prefill");
      rejects([&] { (void)Q4Linear::plan(workload, {LinearTile::Paired256, 0}); });
      rejects([&] { (void)Q4Linear::plan(workload, {LinearTile::Paired256, 1}); });
    }
  }

  const LinearWorkload generic{{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  for (uint32_t groups : {0U, 3U})
    rejects([&] { (void)Q4Linear::plan(generic, {LinearTile::Paired256, groups}); });
  for (auto scope : {LinearSimdgroups::Four, static_cast<LinearSimdgroups>(0),
                     static_cast<LinearSimdgroups>(2), static_cast<LinearSimdgroups>(16)})
    rejects([&] { (void)Q4Linear::plan(generic, {LinearTile::Paired256, 1, scope}); });
  for (const LinearMatrix matrix : {LinearMatrix{128, 256}, LinearMatrix{384, 256},
                                   LinearMatrix{512, 64}, LinearMatrix{512, 128},
                                   LinearMatrix{512, 320}})
    rejects([&] { (void)Q4Linear::plan(
        {matrix, 8, LinearPhase::Decode, LinearEpilogue::GateUp}, {LinearTile::Paired256, 1}); });

  // Legacy N256 overrides preserve the original sequential kernel even on
  // the promoted device/key. Rejecting a new invalid profile is transactional.
  const LinearConfig legacy{LinearTile::N256, 68};
  const std::array legacyChoices{LinearChoice{kMeasuredPairedGateUpWorkload, legacy}};
  measured.setChoices(legacyChoices);
  require(measured.plan(kMeasuredPairedGateUpWorkload).configuration() == legacy &&
              measured.plan(kMeasuredPairedGateUpWorkload).pipeline() ==
                  "decode_linear_q4_n256_gate_up",
          "legacy N256 profile was silently reinterpreted as paired gate/up");
  const std::array invalidChoices{LinearChoice{kMeasuredPairedGateUpWorkload,
      {LinearTile::Paired256, 68, LinearSimdgroups::Four}}};
  rejects([&] { measured.setChoices(invalidChoices); });
  require(measured.plan(kMeasuredPairedGateUpWorkload).configuration() == legacy,
          "invalid paired profile update replaced a legacy installed choice");
  measured.setChoices({});
  require(measured.plan(kMeasuredPairedGateUpWorkload).configuration() == promoted,
          "clearing legacy choices failed to restore measured paired gate/up");
}

void candidateCoverage() {
  // The vocabulary projection is much wider than either GPU. Its offline
  // sweep must explore core-relative resident grids, not only the fixed grids
  // measured on smaller devices and the many-wave full output grid.
  for (const uint32_t cores : {16U, 20U, 80U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = 10;
    device.gpuCoreCount = cores;
    Q4Linear linear(device);
    for (const uint32_t rows : {8U, 16U, 24U, 32U}) {
      const LinearWorkload head{{248320, 5120}, rows};
      const auto candidates = linear.candidates(head);
      require(!candidates.empty() && candidates.size() <= 40 &&
                  candidates.front().configuration() == linear.plan(head).configuration(),
              "wide candidate set lost its bound or baseline");
      for (size_t index = 0; index < candidates.size(); ++index) {
        const auto &plan = candidates[index];
        require(plan.configuration().groups && plan.configuration().groups <=
                    head.matrix.outputSize / plan.tileColumns(),
                "wide candidate escaped its output grid");
        for (size_t prior = 0; prior < index; ++prior)
          require(candidates[prior].configuration() != plan.configuration(),
                  "duplicate wide candidates inflated resident coverage");
      }
      for (const auto variant : {
               LinearConfig{LinearTile::N128, 0},
               LinearConfig{LinearTile::N256, 0},
               LinearConfig{LinearTile::Paired128, 0},
               LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four},
               LinearConfig{LinearTile::N64, 0}}) {
        if ((variant.tile == LinearTile::Paired128 && rows != 8) ||
            (variant.simdgroups == LinearSimdgroups::Four && rows != 24))
          continue;
        const uint32_t tiles = head.matrix.outputSize /
            (variant.tile == LinearTile::N64 ? 64 :
             variant.tile == LinearTile::N256 ? 256 : 128);
        bool oneGroupPerCore = false;
        bool fullGrid = false;
        uint32_t interiorGrids = 0;
        bool residentRange = false;
        for (const auto &plan : candidates) {
          const auto config = plan.configuration();
          if (config.tile != variant.tile || config.simdgroups != variant.simdgroups)
            continue;
          oneGroupPerCore |= config.groups == cores;
          fullGrid |= config.groups == tiles;
          interiorGrids += config.groups > cores && config.groups < tiles &&
              config.groups <= uint64_t{cores} * 8;
          residentRange |= config.groups >= uint64_t{cores} * 2 &&
              config.groups <= uint64_t{cores} * 4;
        }
        require(oneGroupPerCore && fullGrid && interiorGrids >= 3 && residentRange,
                "vocabulary candidate set omits device-relative resident grids");
      }
    }
    // Smaller mixer, MLP and selector grids clamp to actual output tiles. They
    // still retain every legal scope and never generate a zero/oversized grid.
    for (const LinearMatrix matrix : {LinearMatrix{16640, 5120},
                                      LinearMatrix{17408, 5120},
                                      LinearMatrix{5120, 17408},
                                      LinearMatrix{256, 5120}}) {
      for (const uint32_t rows : {8U, 16U, 24U, 32U}) {
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                   LinearEpilogue::GateUp}) {
          const LinearWorkload workload{matrix, rows, LinearPhase::Decode, epilogue};
          const auto candidates = linear.candidates(workload);
          require(!candidates.empty() && candidates.size() <= 40 &&
                      candidates.front().configuration() == linear.plan(workload).configuration(),
                  "expanded candidate set lost its bound or baseline");
          for (size_t index = 0; index < candidates.size(); ++index) {
            const auto &plan = candidates[index];
            const auto config = plan.configuration();
            const uint32_t tiles = matrix.outputSize / plan.tileColumns();
            require(config.groups && config.groups <= tiles,
                    "device-relative candidate escaped its output grid");
            for (size_t prior = 0; prior < index; ++prior)
              require(candidates[prior].configuration() != config,
                      "clamped device-relative candidates were not deduplicated");
            const auto sameVariant = [&](const LinearPlan &candidate) {
              return candidate.configuration().tile == config.tile &&
                  candidate.configuration().simdgroups == config.simdgroups;
            };
            require(std::any_of(candidates.begin(), candidates.end(), [&](const LinearPlan &candidate) {
                      return sameVariant(candidate) &&
                          candidate.configuration().groups == std::min(cores, tiles);
                    }) &&
                    std::any_of(candidates.begin(), candidates.end(), [&](const LinearPlan &candidate) {
                      return sameVariant(candidate) && candidate.configuration().groups == tiles;
                    }),
                    "small-grid candidate coverage omitted a core wave or full grid");
          }
        }
      }
    }
  }
  DeviceCapabilities unknown;
  unknown.appleGpuFamily = 10;
  auto assumed = unknown;
  assumed.gpuCoreCount = 64;
  const LinearWorkload head{{248320, 5120}, 24};
  const auto fallback = Q4Linear(unknown).candidates(head);
  const auto explicitCount = Q4Linear(assumed).candidates(head);
  require(fallback.size() == explicitCount.size(),
          "unknown-core candidate set differs from the conservative fallback");
  for (size_t index = 0; index < fallback.size(); ++index)
    require(fallback[index].configuration() == explicitCount[index].configuration(),
            "unknown-core candidate order or grids changed");
}

void narrowDecodeCandidates() {
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 80;
  Q4Linear linear(device);
  for (const uint32_t rows : {8U, 16U, 24U, 32U}) {
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual}) {
      const LinearWorkload workload{{5120, 17408}, rows, LinearPhase::Decode, epilogue};
      const auto candidates = linear.candidates(workload);
      // The language hidden-width projection has only 40 N128 output tiles.
      // N64 makes one group per core possible without changing weight packing.
      require(std::any_of(candidates.begin(), candidates.end(), [](const LinearPlan &candidate) {
                const auto config = candidate.configuration();
                return config.tile == LinearTile::N64 && config.groups == 80 &&
                    config.simdgroups == LinearSimdgroups::Eight &&
                    candidate.tileColumns() == 64 && candidate.threadsPerThreadgroup() == 256;
              }),
              "hidden-width decode omitted the N64 grid covering all 80 cores");
      for (const auto &plan : candidates) {
        const auto config = plan.configuration();
        if (config.tile == LinearTile::N128 || config.tile == LinearTile::Paired128)
          require(config.groups <= 40, "N128 candidate exceeded its hidden-width output grid");
        if (config.tile == LinearTile::N64)
          require(plan.storageRows() == rows && plan.secondPipeline().empty() &&
                      !plan.sumsBytes() && !plan.downSumsBytes() && !plan.gateScratchBytes(),
                  "N64 changed the complete decode operator's storage contract");
      }
      rejects([&] { (void)Q4Linear::plan(workload, {LinearTile::N64, 0}); });
      rejects([&] { (void)Q4Linear::plan(workload, {LinearTile::N64, 81}); });
      rejects([&] { (void)Q4Linear::plan(workload,
          {LinearTile::N64, 80, LinearSimdgroups::Four}); });
    }
    const LinearWorkload gateUp{{17408, 5120}, rows, LinearPhase::Decode, LinearEpilogue::GateUp};
    const auto gateUpCandidates = linear.candidates(gateUp);
    require(std::none_of(gateUpCandidates.begin(), gateUpCandidates.end(), [](const LinearPlan &candidate) {
              return candidate.configuration().tile == LinearTile::N64;
            }), "N64 escaped into the unsupported gate/up candidate set");
    rejects([&] { (void)Q4Linear::plan(gateUp, {LinearTile::N64, 80}); });
  }
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                             LinearEpilogue::UpWithGate}) {
    const LinearWorkload prefill{{5120, 17408}, 2048, LinearPhase::Prefill, epilogue};
    const auto candidates = linear.candidates(prefill);
    require(std::none_of(candidates.begin(), candidates.end(), [](const LinearPlan &candidate) {
              return candidate.configuration().tile == LinearTile::N64;
            }), "N64 escaped into the unsupported prefill candidate set");
    rejects([&] { (void)Q4Linear::plan(prefill, {LinearTile::N64, 0}); });
  }
}

void planContracts() {
  DeviceCapabilities device;
  device.appleGpuFamily = 9;
  Q4Linear linear(device);
  for (const LinearMatrix matrix : {LinearMatrix{512, 256}, LinearMatrix{768, 768},
                                    LinearMatrix{16640, 5120}, LinearMatrix{12544, 2048}}) {
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                 LinearEpilogue::GateUp}) {
        const LinearWorkload workload{matrix, lanes * 8, LinearPhase::Decode, epilogue};
        const auto candidates = linear.candidates(workload);
        require(!candidates.empty() && candidates.size() <= 40,
                "Linear candidates exceed the bounded set");
        require(candidates.front().configuration() == linear.plan(workload).configuration(),
                "Linear baseline is not first candidate");
        uint32_t fourScopeCandidates = 0;
        for (size_t index = 0; index < candidates.size(); ++index) {
          const auto &plan = candidates[index];
          if (plan.configuration().tile == LinearTile::Paired256)
            require(lanes == 1 && epilogue == LinearEpilogue::GateUp &&
                        plan.configuration().simdgroups == LinearSimdgroups::Eight &&
                        plan.tileColumns() == 256 &&
                        plan.pipeline() == "decode_linear_q4_n256_gate_up_paired",
                    "paired256 candidate escaped the precompiled M8 gate/up contract");
          const bool four = plan.configuration().simdgroups == LinearSimdgroups::Four;
          if (four) {
            ++fourScopeCandidates;
            require(lanes == 3 && plan.configuration().tile == LinearTile::N128 &&
                        epilogue != LinearEpilogue::GateUp && plan.secondPipeline().empty(),
                    "four-SIMDgroup candidate escaped its precompiled workload set");
            require(plan.pipeline() == (epilogue == LinearEpilogue::Residual
                        ? "decode_linear_q4_n128_residual_m24_sg4" : "decode_linear_q4_n128_m24_sg4"),
                    "four-SIMDgroup plan chose the wrong pipeline");
          }
          require(plan.threadsPerThreadgroup() == (four ? 128 : 256),
                  "Linear plan scope/thread count disagree");
          require(plan.storageRows() == lanes * 8 && !plan.sumsBytes() && !plan.downSumsBytes(),
                  "decode storage/sums contract changed");
          require(plan.gateScratchBytes() == (epilogue == LinearEpilogue::GateUp && lanes >= 3
                      ? uint64_t{lanes} * 8 * matrix.outputSize * 2 : 0),
                  "decode gate scratch disagrees with decomposition");
          for (size_t prior = 0; prior < index; ++prior)
            require(candidates[prior].configuration() != plan.configuration(),
                    "duplicate Linear candidates");
        }
        require((fourScopeCandidates != 0) ==
                    (lanes == 3 && epilogue != LinearEpilogue::GateUp),
                "Linear candidate set omitted or added four-SIMDgroup plans");
      }
    }
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                               LinearEpilogue::UpWithGate}) {
      // Every prefill epilogue has the eight-simdgroup N256 tile and the
      // four-simdgroup N128 tile; the plain and residual ones also N128/8.
      const auto candidates = linear.candidates({matrix, 2048, LinearPhase::Prefill, epilogue});
      const bool up = epilogue == LinearEpilogue::UpWithGate;
      require(candidates.size() == (up ? 2U : 3U) &&
                  candidates.front().configuration() ==
                      linear.plan({matrix, 2048, LinearPhase::Prefill, epilogue}).configuration(),
              "prefill candidate set changed");
      for (const auto &plan : candidates) {
        const bool four = plan.configuration().simdgroups == LinearSimdgroups::Four;
        require(plan.configuration().groups == 0 && plan.threadsPerThreadgroup() == (four ? 128 : 256) &&
                    (!four || plan.configuration().tile == LinearTile::N128) &&
                    (!up || four || plan.configuration().tile == LinearTile::N256) &&
                    (plan.pipeline().ends_with("_sg4") == four),
                "prefill candidate scope, tile or pipeline name disagree");
      }
      for (uint32_t rows = 1; rows <= 2048; ++rows) {
        const auto plan = linear.plan({matrix, rows, LinearPhase::Prefill, epilogue});
        const uint64_t storageRows = (rows + 31) / 32 * 32;
        require(plan.storageRows() == storageRows &&
                    plan.sumsBytes() == storageRows * (matrix.inputSize / 64) * 4,
                "prefill padding or sums bound incorrect");
        require(plan.gateScratchBytes() == (up ? storageRows * matrix.outputSize * 2 : 0) &&
                    plan.downSumsBytes() == (up ? storageRows * (matrix.outputSize / 64) * 4 : 0),
                "prefill gate/output sums bound incorrect");
      }
    }
  }
  const LinearWorkload valid{{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::None};
  for (const LinearWorkload invalid : {
           LinearWorkload{{0, 256}, 8}, LinearWorkload{{128, 256}, 8},
           LinearWorkload{{384, 256}, 8}, LinearWorkload{{512, 64}, 8},
           LinearWorkload{{512, 320}, 8}, LinearWorkload{{512, 0}, 8},
           LinearWorkload{{512, 256}, 0}, LinearWorkload{{512, 256}, 7},
           LinearWorkload{{512, 256}, 40},
           LinearWorkload{{512, 256}, 8, static_cast<LinearPhase>(255)},
           LinearWorkload{{512, 256}, 8, LinearPhase::Decode, static_cast<LinearEpilogue>(255)},
           LinearWorkload{{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::UpWithGate},
           LinearWorkload{{512, 256}, 8, LinearPhase::Prefill, LinearEpilogue::GateUp},
           LinearWorkload{{512, 256}, 2049, LinearPhase::Prefill}})
    rejects([&] { (void)linear.plan(invalid); });
  for (const LinearConfig invalid : {
           LinearConfig{LinearTile::N128, 0}, LinearConfig{LinearTile::N128, 5},
           LinearConfig{static_cast<LinearTile>(255), 1}})
    rejects([&] { (void)Q4Linear::plan(valid, invalid); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 16}, {LinearTile::Paired128, 1}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 32, LinearPhase::Prefill}, {LinearTile::N128, 1}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 32, LinearPhase::Prefill}, {LinearTile::Paired128, 0}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::Residual}, {LinearTile::N256, 1}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}, {LinearTile::N128, 1}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 8, LinearPhase::Prefill, LinearEpilogue::UpWithGate}, {LinearTile::N128, 0}); });
  const LinearWorkload fourWorkload{{512, 256}, 24, LinearPhase::Decode, LinearEpilogue::None};
  const LinearConfig fourConfig{LinearTile::N128, 1, LinearSimdgroups::Four};
  for (uint32_t scope : {0U, 1U, 2U, 3U, 16U, 255U})
    rejects([&] { (void)Q4Linear::plan(fourWorkload,
        {LinearTile::N128, 1, static_cast<LinearSimdgroups>(scope)}); });
  for (uint32_t rows : {8U, 16U, 32U})
    rejects([&] { (void)Q4Linear::plan({{512, 256}, rows}, fourConfig); });
  for (const auto tile : {LinearTile::N256, LinearTile::Paired128, LinearTile::N64,
                         LinearTile::Paired256})
    rejects([&] { (void)Q4Linear::plan(fourWorkload,
        {tile, 1, LinearSimdgroups::Four}); });
  for (const auto epilogue : {LinearEpilogue::GateUp, LinearEpilogue::UpWithGate})
    rejects([&] { (void)Q4Linear::plan({{512, 256}, 24, LinearPhase::Decode, epilogue},
                                      fourConfig); });
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                             LinearEpilogue::UpWithGate}) {
    const LinearWorkload prefill{{512, 256}, 24, LinearPhase::Prefill, epilogue};
    require(Q4Linear::plan(prefill, {LinearTile::N128, 0, LinearSimdgroups::Four})
                    .threadsPerThreadgroup() == 128,
            "four-SIMDgroup prefill plan was rejected");
    rejects([&] { (void)Q4Linear::plan(prefill, {LinearTile::N256, 0, LinearSimdgroups::Four}); });
    rejects([&] { (void)Q4Linear::plan(prefill, {LinearTile::N128, 1, LinearSimdgroups::Four}); });
  }

  const LinearWorkload other{{768, 768}, 16, LinearPhase::Decode, LinearEpilogue::None};
  const auto original = linear.plan(other).configuration();
  const LinearConfig selected{LinearTile::N256, 1};
  const std::array choices{LinearChoice{other, selected}, LinearChoice{valid, selected}};
  linear.setChoices(choices);
  require(linear.plan(valid).configuration() == selected &&
              linear.plan(other).configuration() == selected,
          "unsorted profile choices did not take effect");
  const std::array duplicate{choices[0], choices[0]};
  rejects([&] { linear.setChoices(duplicate); });
  const std::array invalid{LinearChoice{valid, {LinearTile::N128, 0}}};
  rejects([&] { linear.setChoices(invalid); });
  require(linear.plan(other).configuration() == selected,
          "invalid profile update changed installed choices");
  linear.setChoices({});
  require(linear.plan(other).configuration() == original,
          "clearing profile did not restore the shipped baseline");
  const auto baselineFourWorkload = linear.plan(fourWorkload);
  const std::array fourChoices{LinearChoice{fourWorkload, fourConfig}};
  linear.setChoices(fourChoices);
  require(linear.plan(fourWorkload).configuration() == fourConfig &&
              linear.plan(fourWorkload).threadsPerThreadgroup() == 128,
          "installed four-SIMDgroup choice did not reach the selected plan");
  const std::array invalidScope{LinearChoice{valid, fourConfig}};
  rejects([&] { linear.setChoices(invalidScope); });
  require(linear.plan(fourWorkload).configuration() == fourConfig,
          "invalid scope update changed installed choices");
  linear.setChoices({});
  require(linear.plan(fourWorkload).configuration() == baselineFourWorkload.configuration() &&
              linear.plan(fourWorkload).threadsPerThreadgroup() ==
                  baselineFourWorkload.threadsPerThreadgroup(),
          "clearing choices did not restore the shipped execution scope");
}

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  auto buffer = backend.allocateBuffer(bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352d;
  value ^= value >> 15;
  value *= 0x846ca68b;
  return value ^ (value >> 16);
}

std::array<uint64_t, 3> projectionFingerprint(const Q4Projection &projection) {
  std::array<uint64_t, 3> result{};
  const std::array buffers{projection.weights, projection.scales, projection.biases};
  for (size_t slot = 0; slot < buffers.size(); ++slot) {
    const auto *bytes = static_cast<const uint8_t *>(buffers[slot].contents());
    uint64_t hash = 14695981039346656037ULL;
    for (uint64_t i = 0; i < buffers[slot].sizeBytes(); ++i)
      hash = (hash ^ bytes[i]) * 1099511628211ULL;
    result[slot] = hash;
  }
  return result;
}

Q4Projection projection(metal::MetalBackend &backend, LinearMatrix matrix, uint32_t seed) {
  const uint64_t parameters = uint64_t{matrix.outputSize} * (matrix.inputSize / 64);
  Q4Projection p{allocate(backend, parameters * 32), allocate(backend, parameters * 2),
                  allocate(backend, parameters * 2), matrix.outputSize, matrix.inputSize};
  auto *weights = static_cast<uint8_t *>(p.weights.contents());
  auto *scales = static_cast<uint16_t *>(p.scales.contents());
  auto *biases = static_cast<uint16_t *>(p.biases.contents());
  for (uint64_t i = 0; i < parameters * 32; ++i) weights[i] = mix(uint32_t(i) + seed);
  for (uint64_t i = 0; i < parameters; ++i) {
    const float scale = 0.004f + float(mix(uint32_t(i) + seed) % 17) * 0.0001f;
    scales[i] = bf16(scale);
    biases[i] = bf16(-7.5f * scale);
  }
  return p;
}

// Scalar reference uses the actual StorageN=256 bytes and FP32 per-group
// affine accumulation, including the BF16 boundary before each epilogue.
float affineReference(const Q4Projection &p, const uint16_t *input,
                        uint32_t row, uint32_t column) {
  const auto *weights = static_cast<const uint8_t *>(p.weights.contents());
  const auto *scales = static_cast<const uint16_t *>(p.scales.contents());
  const auto *biases = static_cast<const uint16_t *>(p.biases.contents());
  const uint32_t groups = p.inputSize / 64;
  float result = 0;
  for (uint32_t group = 0; group < groups; ++group) {
    const uint64_t parameter = (uint64_t{column / 256} * groups + group) * 256 + column % 256;
    float partial = 0, sum = 0;
    for (uint32_t k = 0; k < 64; ++k) {
      const uint8_t byte = weights[parameter * 32 + k / 2];
      const uint32_t quantized = (byte >> ((k & 1) * 4)) & 15;
      const float x = fp32(input[uint64_t{row} * p.inputSize + group * 64 + k]);
      sum += x;
      partial += x * quantized;
    }
    result += partial * fp32(scales[parameter]) + sum * fp32(biases[parameter]);
  }
  return fp32(bf16(result));
}

void checkReference(const Q4Projection &p, const Q4Projection &gate,
                      LinearWorkload workload, const LinearBuffers &buffers,
                      const uint16_t *savedResidual = nullptr) {
  const auto *input = static_cast<const uint16_t *>(buffers.input.contents());
  const auto *output = static_cast<const uint16_t *>(buffers.output.contents());
  const auto *residual = savedResidual ? savedResidual
      : static_cast<const uint16_t *>(buffers.residual.contents());
  const auto *gateValues = static_cast<const uint16_t *>(buffers.gateScratch.contents());
  for (const uint32_t row : {0U, workload.rows / 2, workload.rows - 1}) {
    for (const uint32_t column : {0U, 127U, 128U, 255U,
                                  p.outputSize / 2, p.outputSize - 1}) {
      float expected = affineReference(p, input, row, column);
      const uint64_t index = uint64_t{row} * p.outputSize + column;
      if (workload.epilogue == LinearEpilogue::Residual)
        expected += fp32(residual[index]);
      if (workload.epilogue == LinearEpilogue::GateUp ||
          workload.epilogue == LinearEpilogue::UpWithGate) {
        const float gateValue = workload.epilogue == LinearEpilogue::GateUp
            ? affineReference(gate, input, row, column) : fp32(gateValues[index]);
        expected *= gateValue / (1 + std::exp(-gateValue));
      }
      expected = fp32(bf16(expected));
      const float actual = fp32(output[index]);
      if (!std::isfinite(actual) || std::abs(actual - expected) > 0.004f) {
        std::cerr << "reference row=" << row << " col=" << column
                  << " actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error("Linear failed independent packed-Q4 oracle");
      }
    }
  }
  if (workload.epilogue == LinearEpilogue::UpWithGate) {
    const auto *sums = static_cast<const float *>(buffers.downSums.contents());
    const uint32_t quantGroups = p.outputSize / 64;
    for (uint32_t row = 0; row < workload.rows; ++row) {
      for (uint32_t group = 0; group < quantGroups; ++group) {
        double expected = 0;
        for (uint32_t k = 0; k < 64; ++k)
          expected += fp32(output[uint64_t{row} * p.outputSize + group * 64 + k]);
        const uint64_t index = uint64_t{row / 32} * 32 * quantGroups + group * 32 + row % 32;
        require(std::abs(sums[index] - expected) <= 1e-6 * std::max(1.0, std::abs(expected)),
                "fused prefill output sums have wrong layout/value");
      }
    }
  }
}

void bufferContracts(metal::MetalBackend &backend, Q4Linear &linear,
                        LinearBuffers buffers, const Q4Projection &p,
                        const Q4Projection &gate, const LinearPlan &plan) {
  const bool gateUp = plan.workload().epilogue == LinearEpilogue::GateUp;
  const auto add = [&](metal::CommandGraph &graph, LinearBuffers b,
                        const Q4Projection &projection, const Q4Projection *g) {
    linear.add(graph, b, projection, plan, g);
  };
  const std::array members{&LinearBuffers::input, &LinearBuffers::output,
                           &LinearBuffers::sums, &LinearBuffers::residual,
                           &LinearBuffers::gateScratch, &LinearBuffers::downSums};
  for (auto member : members) {
    if (!(buffers.*member)) continue;
    auto shortBuffers = buffers;
    shortBuffers.*member = backend.view(buffers.*member, 0, (buffers.*member).sizeBytes() - 1);
    metal::CommandGraph graph;
    rejects([&] { add(graph, shortBuffers, p, gateUp ? &gate : nullptr); });
    require(graph.empty(), "invalid Linear buffers partially encoded a graph");
  }
  for (auto member : {&Q4Projection::weights, &Q4Projection::scales, &Q4Projection::biases}) {
    auto shortProjection = p;
    shortProjection.*member = backend.view(p.*member, 0, (p.*member).sizeBytes() - 1);
    metal::CommandGraph graph;
    rejects([&] { add(graph, buffers, shortProjection, gateUp ? &gate : nullptr); });
    require(graph.empty(), "invalid projection partially encoded a graph");
  }
  auto mismatch = p;
  mismatch.inputSize += 256;
  metal::CommandGraph graph;
  rejects([&] { add(graph, buffers, mismatch, gateUp ? &gate : nullptr); });
  rejects([&] { add(graph, buffers, p, gateUp ? nullptr : &gate); });
  require(graph.empty(), "invalid Linear gate/projection partially encoded graph");
  if (plan.workload().phase == LinearPhase::Prefill) {
    const auto workload = plan.workload();
    rejects([&] {
      linear.addPrefillSums(graph,
          backend.view(buffers.input, 0, buffers.input.sizeBytes() - 1), buffers.sums,
          workload.matrix, workload.rows);
    });
    rejects([&] {
      linear.addPrefillSums(graph, buffers.input,
          backend.view(buffers.sums, 0, buffers.sums.sizeBytes() - 1),
          workload.matrix, workload.rows);
    });
    for (const LinearMatrix matrix : {LinearMatrix{0, workload.matrix.inputSize},
           LinearMatrix{128, workload.matrix.inputSize},
           LinearMatrix{workload.matrix.outputSize, 0},
           LinearMatrix{workload.matrix.outputSize, 63}})
      rejects([&] { linear.addPrefillSums(graph, buffers.input, buffers.sums,
                                         matrix, workload.rows); });
    for (uint32_t rows : {0U, SPLASH_PREFILL_TOKEN_BUDGET + 1U})
      rejects([&] { linear.addPrefillSums(graph, buffers.input, buffers.sums,
                                         workload.matrix, rows); });
    require(graph.empty(), "invalid prefill sums input partially encoded graph");
  }
}

void numericalCase(metal::MetalBackend &backend, Q4Linear &linear,
                      const Q4Projection &p, const Q4Projection &gate,
                      LinearWorkload workload, bool inPlaceResidual = false) {
  require(!inPlaceResidual || workload.epilogue == LinearEpilogue::Residual,
          "in-place residual fixture requires residual epilogue");
  const auto candidates = linear.candidates(workload);
  const uint32_t storageRows = candidates[0].storageRows();
  auto input = allocate(backend, uint64_t{storageRows} * p.inputSize * 2);
  auto *inputValues = static_cast<uint16_t *>(input.contents());
  for (uint64_t i = 0; i < uint64_t{workload.rows} * p.inputSize; ++i)
    inputValues[i] = bf16(float(int(mix(uint32_t(i) + 1949) % 257) - 128) / 257.0f);
  std::vector<uint16_t> baseline;
  for (const auto &plan : candidates) {
    const uint64_t outputBytes = uint64_t{storageRows} * p.outputSize * 2;
    const uint64_t guardBytes = uint64_t{8} * p.outputSize * 2;
    auto outputBacking = allocate(backend, outputBytes + guardBytes);
    std::memset(static_cast<uint8_t *>(outputBacking.contents()) + outputBytes, 0x5a, guardBytes);
    const uint64_t gateBytes = plan.gateScratchBytes();
    auto gateBacking = allocate(backend, gateBytes ? gateBytes + guardBytes : 0);
    if (gateBytes)
      std::memset(static_cast<uint8_t *>(gateBacking.contents()) + gateBytes, 0x5a, guardBytes);
    LinearBuffers b{input, backend.view(outputBacking, 0, outputBytes),
                     allocate(backend, plan.sumsBytes()), {},
                     gateBytes ? backend.view(gateBacking, 0, gateBytes) : metal::MetalBuffer{},
                     allocate(backend, plan.downSumsBytes())};
    if (workload.epilogue == LinearEpilogue::Residual) {
      b.residual = inPlaceResidual ? b.output : allocate(backend, b.output.sizeBytes());
      auto *residual = static_cast<uint16_t *>(b.residual.contents());
      for (uint64_t i = 0; i < uint64_t{workload.rows} * p.outputSize; ++i)
        residual[i] = bf16(float(int(mix(uint32_t(i) + 7919) % 257) - 128) / 257.0f);
    }
    bufferContracts(backend, linear, b, p, gate, plan);
    metal::CommandGraph graph;
    if (workload.phase == LinearPhase::Prefill)
      linear.addPrefillSums(graph, input, b.sums, workload.matrix, workload.rows);
    if (workload.epilogue == LinearEpilogue::UpWithGate) {
      const auto gatePlan = linear.plan({workload.matrix, workload.rows, LinearPhase::Prefill,
                                         LinearEpilogue::None});
      linear.add(graph, {input, b.gateScratch, b.sums, {}, {}, {}}, gate, gatePlan);
    }
    const size_t prepasses = graph.dispatches().size();
    Q4DispatchStats stats;
    linear.add(graph, b, p, plan, workload.epilogue == LinearEpilogue::GateUp ? &gate : nullptr, &stats);
    const auto &last = graph.dispatches().back();
    require(last.pipelineName == (plan.secondPipeline().empty() ? plan.pipeline() : plan.secondPipeline()),
            "production dispatch differs from Linear plan");
    for (const auto &dispatch : graph.dispatches().subspan(prepasses))
      require(dispatch.threadsPerThreadgroup.x == plan.threadsPerThreadgroup() &&
                  dispatch.threadsPerThreadgroup.y == 1 && dispatch.threadsPerThreadgroup.z == 1,
              "production dispatch threads differ from Linear plan scope");
    if (workload.phase == LinearPhase::Decode) {
      const uint32_t lanes = workload.rows / 8;
      const uint32_t dispatches = plan.secondPipeline().empty() ? 1 : 2;
      require(last.threadgroups.x == plan.configuration().groups &&
                  graph.dispatches().size() == dispatches,
              "Linear decode plan/graph geometry mismatch");
      require(stats.fusedSourceOperations == (lanes == 1 ? 0 : lanes * dispatches) &&
                  stats.m16Dispatches == (lanes == 2 ? dispatches : 0) &&
                  stats.m24Dispatches == (lanes == 3 ? dispatches : 0) &&
                  stats.m32Dispatches == (lanes == 4 ? dispatches : 0),
              "Linear dispatch statistics changed");
    } else {
      require(last.threadgroups.x == storageRows / 32 &&
                  last.threadgroups.y == p.outputSize / plan.tileColumns() &&
                  graph.dispatches().size() == prepasses + 1,
              "Linear prefill plan/graph geometry mismatch");
    }
    const auto snapshot = [](const metal::MetalBuffer &buffer) {
      if (!buffer) return std::vector<uint8_t>{};
      const auto *begin = static_cast<const uint8_t *>(buffer.contents());
      return std::vector<uint8_t>{begin, begin + buffer.sizeBytes()};
    };
    const auto immutableInput = snapshot(b.input);
    std::vector<uint16_t> immutableResidual;
    if (b.residual) {
      const auto *values = static_cast<const uint16_t *>(b.residual.contents());
      immutableResidual.assign(values, values + b.residual.sizeBytes() / sizeof(uint16_t));
    }
    (void)backend.submitCommand(graph.dispatches());
    if (workload.phase == LinearPhase::Decode && workload.rows == 24) {
      const auto firstOutput = snapshot(b.output);
      if (inPlaceResidual)
        std::memcpy(b.residual.contents(), immutableResidual.data(),
                    immutableResidual.size() * sizeof(uint16_t));
      (void)backend.submitCommand(graph.dispatches());
      require(std::memcmp(firstOutput.data(), b.output.contents(), firstOutput.size()) == 0,
              "repeated M24 dispatch changed its output bytes");
    }
    const auto checkGuard = [&](const metal::MetalBuffer &backing, uint64_t payload,
                                 const char *role) {
      const auto *guard = static_cast<const uint8_t *>(backing.contents()) + payload;
      for (uint64_t i = 0; i < guardBytes; ++i) {
        if (guard[i] != 0x5a) {
          std::cerr << role << " guard overwritten matrix=" << p.outputSize << 'x' << p.inputSize
                    << " rows=" << workload.rows << " epilogue=" << uint32_t(workload.epilogue)
                    << " pipeline=" << plan.pipeline() << " first_extra_byte=" << i << '\n';
          throw std::runtime_error("Linear wrote beyond its exact workspace/output view");
        }
      }
    };
    checkGuard(outputBacking, outputBytes, "output");
    if (gateBytes) checkGuard(gateBacking, gateBytes, "gate scratch");
    require(std::memcmp(immutableInput.data(), b.input.contents(), immutableInput.size()) == 0,
            "Linear modified its immutable input");
    if (!inPlaceResidual && !immutableResidual.empty())
      require(std::memcmp(immutableResidual.data(), b.residual.contents(),
                          immutableResidual.size() * sizeof(uint16_t)) == 0,
              "Linear modified its immutable residual");
    try {
      checkReference(p, gate, workload, b,
                     inPlaceResidual ? immutableResidual.data() : nullptr);
    } catch (const std::exception &) {
      std::cerr << "matrix=" << p.outputSize << 'x' << p.inputSize
                << " rows=" << workload.rows << " phase=" << uint32_t(workload.phase)
                << " epilogue=" << uint32_t(workload.epilogue)
                << " tile=" << uint32_t(plan.configuration().tile)
                << " groups=" << plan.configuration().groups
                << " simdgroups=" << uint32_t(plan.configuration().simdgroups)
                << " in_place_residual=" << inPlaceResidual
                << " pipeline=" << plan.pipeline() << '\n';
      throw;
    }
    const auto *output = static_cast<const uint16_t *>(b.output.contents());
    const uint64_t elements = uint64_t{storageRows} * p.outputSize;
    if (baseline.empty()) baseline.assign(output, output + elements);
    require(std::memcmp(baseline.data(), output, elements * 2) == 0,
            "Linear candidate differs from baseline output bytes");
    for (uint64_t i = uint64_t{workload.rows} * p.outputSize; i < elements; ++i)
      require(output[i] == 0, "padded prefill rows were not zero");
  }
}

void pipelineCapabilities(const char *metallib, const DeviceCapabilities &capabilities) {
  Q4Linear linear(capabilities);
  std::map<std::string, uint32_t> names{{"prefill_linear_q4_sums32", 256}};
  const auto collect = [&](LinearWorkload workload) {
    for (const auto &plan : linear.candidates(workload)) {
      for (auto name : {plan.pipeline(), plan.secondPipeline()}) {
        if (name.empty()) continue;
        const auto [found, inserted] = names.emplace(name, plan.threadsPerThreadgroup());
        require(inserted || found->second == plan.threadsPerThreadgroup(),
                "one Linear pipeline was assigned incompatible execution scopes");
      }
    }
  };
  for (uint32_t lanes = 1; lanes <= 4; ++lanes)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                               LinearEpilogue::GateUp})
      collect({{16640, 5120}, lanes * 8, LinearPhase::Decode, epilogue});
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                             LinearEpilogue::UpWithGate})
    collect({{16640, 5120}, 33, LinearPhase::Prefill, epilogue});
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib]] error:&error];
  require(library != nil, "could not load Linear library for resource inspection");
  uint64_t largestStaticMemory = 0;
  uint64_t smallestThreadLimit = std::numeric_limits<uint64_t>::max();
  for (const auto &[name, threads] : names) {
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name.c_str()]];
    require(function != nil, "missing precompiled Linear candidate");
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, "Linear candidate pipeline could not be created");
    largestStaticMemory = std::max(largestStaticMemory, uint64_t(pipeline.staticThreadgroupMemoryLength));
    smallestThreadLimit = std::min(smallestThreadLimit, uint64_t(pipeline.maxTotalThreadsPerThreadgroup));
    require(pipeline.staticThreadgroupMemoryLength <= capabilities.maxThreadgroupMemoryBytes &&
                pipeline.maxTotalThreadsPerThreadgroup >= threads && pipeline.threadExecutionWidth == 32,
            "Linear candidate exceeds current device resources");
  }
  std::cout << "Linear pipelines=" << names.size() << " maximum_static_tg_bytes="
            << largestStaticMemory << " minimum_thread_limit=" << smallestThreadLimit
            << " apple_family=" << capabilities.appleGpuFamily << " PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: linear-plan <production.metallib|--cpu>");
    baselinePlans();
    measuredDecodePolicy();
    pairedGateUpContracts();
    candidateCoverage();
    narrowDecodeCandidates();
    planContracts();
    if (std::string_view(argv[1]) == "--cpu") {
      std::cout << "Linear CPU plans: PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    Q4Linear linear(backend.capabilities());
    // Instrumented shader builds can report additional validation storage;
    // inspect production requirements in the ordinary/API-validation run.
    if (!std::getenv("MTL_SHADER_VALIDATION"))
      pipelineCapabilities(argv[1], backend.capabilities());
    for (const LinearMatrix matrix : {LinearMatrix{512, 256}, LinearMatrix{768, 768},
                                      LinearMatrix{16640, 5120}, LinearMatrix{12544, 2048},
                                      LinearMatrix{5120, 17408},
                                      LinearMatrix{256, 64}, LinearMatrix{512, 320}}) {
      const auto p = projection(backend, matrix, 31);
      const auto gate = projection(backend, matrix, 157);
      const auto immutableProjection = projectionFingerprint(p);
      const auto immutableGateProjection = projectionFingerprint(gate);
      if (matrix.inputSize % 256 == 0)
        for (uint32_t lanes = 1; lanes <= 4; ++lanes)
          for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                     LinearEpilogue::GateUp}) {
            numericalCase(backend, linear, p, gate,
                          {matrix, lanes * 8, LinearPhase::Decode, epilogue});
            if (epilogue == LinearEpilogue::Residual)
              numericalCase(backend, linear, p, gate,
                            {matrix, lanes * 8, LinearPhase::Decode, epilogue}, true);
          }
      for (const uint32_t rows : {1U, 33U, 2048U})
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                   LinearEpilogue::UpWithGate})
          numericalCase(backend, linear, p, gate,
                        {matrix, rows, LinearPhase::Prefill, epilogue});
      require(projectionFingerprint(p) == immutableProjection &&
                  projectionFingerprint(gate) == immutableGateProjection,
              "Linear changed immutable Q4 weights or quantization metadata");
      std::cout << "Linear N=" << matrix.outputSize << " K=" << matrix.inputSize
                << " all epilogues/candidates/row cases PASS\n";
    }
    std::cout << "Linear plans and candidates: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "Linear plans: FAIL: " << error.what() << '\n';
    return 1;
  }
}
