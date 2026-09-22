#pragma once
#include "metal/abi/FlashHCFused.h"
#include "metal/abi/FlashFloatDenseCache.h"

// Canonical original parameters remain byte-identical at offset0. This private
// VerifyR4-only producer advertises the HC-up M8 tensor's padded activation rows.
#ifndef __METAL_VERSION__
namespace splash::flash::hc_pad_sep22 {
#endif
struct FlashHCDownPadParams {
  FlashHCFusedParams literal;
  uint32_t padded_rows;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};
#ifndef __METAL_VERSION__
} // namespace splash::flash::hc_pad_sep22
using FlashHCDownPadParams = splash::flash::hc_pad_sep22::FlashHCDownPadParams;
#endif
namespace hc_pad_producer_sep22 {
enum : uint32_t {
  kActiveRows = 4, kPaddedRows = 8, kLowrank = 320,
  kSIMDGroups = 4, kThreads = 128,
  kTimedParamsSlot = 10, kProbeRawF32Slot = 10,
  kProbeRawBF16Slot = 11, kProbeParamsSlot = 12,
  kUpProbeRawF32Slot = 6, kUpProbeParamsSlot = 7
};
} // namespace hc_pad_producer_sep22
static_assert(sizeof(FlashFloatDenseSmallRowsParams) == 32);
static_assert(alignof(FlashFloatDenseSmallRowsParams) == 4);
static_assert(sizeof(FlashHCFusedMatrix) == 32);
static_assert(alignof(FlashHCFusedMatrix) == 8);
static_assert(sizeof(FlashHCFusedParams) == 160);
static_assert(alignof(FlashHCFusedParams) == 8);
static_assert(sizeof(FlashHCDownPadParams) == 176);
static_assert(alignof(FlashHCDownPadParams) == 8);
#ifndef __METAL_VERSION__
#include <cstddef>
static_assert(offsetof(FlashFloatDenseSmallRowsParams, rows) == 0);
static_assert(offsetof(FlashFloatDenseSmallRowsParams, padded_rows) == 4);
static_assert(offsetof(FlashFloatDenseSmallRowsParams, input_size) == 8);
static_assert(offsetof(FlashFloatDenseSmallRowsParams, output_size) == 12);
static_assert(offsetof(FlashFloatDenseSmallRowsParams, output_begin) == 16);
static_assert(offsetof(FlashFloatDenseSmallRowsParams, output_count) == 20);
static_assert(offsetof(FlashFloatDenseSmallRowsParams, tile_rows) == 24);
static_assert(offsetof(FlashFloatDenseSmallRowsParams, tile_outputs) == 28);
static_assert(offsetof(FlashHCDownPadParams, literal) == 0);
static_assert(offsetof(FlashHCDownPadParams, padded_rows) == 160);
static_assert(offsetof(FlashHCDownPadParams, reserved0) == 164);
static_assert(offsetof(FlashHCDownPadParams, reserved1) == 168);
static_assert(offsetof(FlashHCDownPadParams, reserved2) == 172);
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
static_assert(offsetof(FlashHCFusedMatrix, input_size) == 0);
static_assert(offsetof(FlashHCFusedMatrix, output_size) == 4);
static_assert(offsetof(FlashHCFusedMatrix, bits) == 8);
static_assert(offsetof(FlashHCFusedMatrix, group_size) == 12);
static_assert(offsetof(FlashHCFusedMatrix, weight_row_stride_bytes) == 16);
static_assert(offsetof(FlashHCFusedMatrix, parameter_row_stride_bytes) == 24);
#endif
