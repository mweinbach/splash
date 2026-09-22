#pragma once
#include "flash/FlashGatheredMPP.hpp"

// Shipping retains the canonical 16-byte original parameter block and old
// bindings. Gate/up probes append eight stage buffers; down appends three.
namespace expert_rhs_tile64_sep22 {
enum : uint32_t {
  kPanel = 64, kThreads = 128, kSelections = 10, kExperts = 512,
  kGateInput = 2560, kGateOutput = 640, kDownInput = 640, kDownOutput = 2560,
  kGateParamsSlot = 9, kDownParamsSlot = 7,
  kGateDotSlot = 9, kGateScaledSlot = 10, kUpDotSlot = 11, kUpScaledSlot = 12,
  kGateBF16Slot = 13, kUpBF16Slot = 14, kSiLuBF16Slot = 15,
  kActivationBF16Slot = 16, kGateProbeParamsSlot = 17,
  kDownDotSlot = 7, kDownScaledSlot = 8, kDownBF16Slot = 9,
  kDownProbeParamsSlot = 10
};
// Same panel base (rank*W + panelColumn)*K; only within-panel K/N order changes.
constexpr uint64_t oldCoefficientOffset(uint32_t rank, uint32_t width,
    uint32_t input, uint32_t column, uint32_t k) {
  return (uint64_t(rank) * width + column) * input + k;
}
constexpr uint64_t tile64CoefficientOffset(uint32_t rank, uint32_t width,
    uint32_t input, uint32_t column, uint32_t k) {
  return (uint64_t(rank) * width + column / 64 * 64) * input + uint64_t(k) * 64 + column % 64;
}
} // namespace expert_rhs_tile64_sep22
static_assert(sizeof(FlashGatheredMPPParams) == 16);
static_assert(alignof(FlashGatheredMPPParams) == 4);
#ifndef __METAL_VERSION__
#include <cstddef>
static_assert(offsetof(FlashGatheredMPPParams, rows) == 0);
static_assert(offsetof(FlashGatheredMPPParams, selections) == 4);
static_assert(offsetof(FlashGatheredMPPParams, experts) == 8);
static_assert(offsetof(FlashGatheredMPPParams, reserved) == 12);
#endif
