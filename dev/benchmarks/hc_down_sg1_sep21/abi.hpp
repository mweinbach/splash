#pragma once
#include "metal/abi/FlashHCFused.h"

// This private experiment retains the exact original 160-byte ABI. Timed
// buffer slots 0..9 are normalized/down/injection/activated/gates/diagnostics;
// params follow at slot 10. Probes retain 0..9 and append F32 tap, BF16 tap,
// then those same params at 12. Every slot must be bound even when a tap is
// disabled or injection is absent. p.write_raw_up controls both probe taps.
namespace splash_hc_down_sg1_sep21 {
using Params = FlashHCFusedParams;
using Matrix = FlashHCFusedMatrix;
enum : uint32_t {
  kTimedParamsSlot = 10,
  kProbeRawF32Slot = 10,
  kProbeRawBF16Slot = 11,
  kProbeParamsSlot = 12,
  kEpilogParamsSlot = 4,
  kDiagnosticInvalid = 2,
  kDiagnosticNonfinite = 4
};

constexpr uint32_t outputCount(const Params &p) {
  return p.lowrank + p.has_injection * p.streams;
}
constexpr uint64_t rawTapIndex(const Params &p, uint32_t row, uint32_t output) {
  return uint64_t(row) * outputCount(p) + output;
}
constexpr uint32_t originalGridX(uint32_t output) { return output / 4; }
constexpr uint32_t originalSIMD(uint32_t output) { return output % 4; }
constexpr bool validSIMDCount(uint32_t sg) { return sg == 1 || sg == 4; }
} // namespace splash_hc_down_sg1_sep21

static_assert(sizeof(FlashHCFusedMatrix) == 32);
static_assert(alignof(FlashHCFusedMatrix) == 8);
static_assert(sizeof(FlashHCFusedParams) == 160);
static_assert(alignof(FlashHCFusedParams) == 8);

#ifndef __METAL_VERSION__
#include <cstddef>
static_assert(offsetof(FlashHCFusedMatrix, input_size) == 0);
static_assert(offsetof(FlashHCFusedMatrix, output_size) == 4);
static_assert(offsetof(FlashHCFusedMatrix, bits) == 8);
static_assert(offsetof(FlashHCFusedMatrix, group_size) == 12);
static_assert(offsetof(FlashHCFusedMatrix, weight_row_stride_bytes) == 16);
static_assert(offsetof(FlashHCFusedMatrix, parameter_row_stride_bytes) == 24);
static_assert(offsetof(FlashHCFusedParams, rows) == 0);
static_assert(offsetof(FlashHCFusedParams, width) == 4);
static_assert(offsetof(FlashHCFusedParams, streams) == 8);
static_assert(offsetof(FlashHCFusedParams, lowrank) == 12);
static_assert(offsetof(FlashHCFusedParams, has_injection) == 16);
static_assert(offsetof(FlashHCFusedParams, write_raw_up) == 20);
static_assert(offsetof(FlashHCFusedParams, arithmetic_mode) == 24);
static_assert(offsetof(FlashHCFusedParams, simdgroups) == 28);
static_assert(offsetof(FlashHCFusedParams, norm_is_float) == 32);
static_assert(offsetof(FlashHCFusedParams, norm_convention) == 36);
static_assert(offsetof(FlashHCFusedParams, reserved0) == 40);
static_assert(offsetof(FlashHCFusedParams, reserved1) == 44);
static_assert(offsetof(FlashHCFusedParams, norm_epsilon) == 48);
static_assert(offsetof(FlashHCFusedParams, reserved2) == 52);
static_assert(offsetof(FlashHCFusedParams, reserved3) == 56);
static_assert(offsetof(FlashHCFusedParams, reserved4) == 60);
static_assert(offsetof(FlashHCFusedParams, down) == 64);
static_assert(offsetof(FlashHCFusedParams, injection) == 96);
static_assert(offsetof(FlashHCFusedParams, up) == 128);
#endif
