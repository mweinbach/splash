#pragma once
#include "metal/abi/FlashHCFused.h"

// PRIVATE ordinary R1 component. The canonical HC descriptor is unchanged.
#ifndef __METAL_VERSION__
namespace hc_r1_chronological_terms_sep22 {
#endif
struct HCChronologicalTermsParams {
  FlashHCFusedParams literal;
  uint32_t term_outputs;
  uint32_t term_j;
  uint32_t slice_terms;
  uint32_t reserved0;
};
#ifndef __METAL_VERSION__
} // namespace hc_r1_chronological_terms_sep22
using HCChronologicalTermsParams =
    hc_r1_chronological_terms_sep22::HCChronologicalTermsParams;
#endif

namespace hc_r1_chronological_terms_sep22 {
enum : uint32_t {
  kRows = 1, kWidth = 2560, kStreams = 4, kK = 10240, kLowrank = 320,
  kSIMDGroups = 4, kThreads = 128, kTermJ = 320, kSliceTerms = 8,
  kSlices = 40, kTermPlaneSlot = 10, kParamsSlot = 11,
  kRawF32Slot = 12, kRawBF16Slot = 13, kOriginalDirectTermsSlot = 14,
  kParamsBytes = 176
};
enum : uint64_t {
  kNoInjectionTermBytes = uint64_t(320) * 320 * 32 * sizeof(float),
  kInjectionTermBytes = uint64_t(324) * 320 * 32 * sizeof(float)
};
} // namespace hc_r1_chronological_terms_sep22

static_assert(sizeof(FlashHCFusedMatrix) == 32);
static_assert(alignof(FlashHCFusedMatrix) == 8);
static_assert(sizeof(FlashHCFusedParams) == 160);
static_assert(alignof(FlashHCFusedParams) == 8);
static_assert(sizeof(HCChronologicalTermsParams) == 176);
static_assert(alignof(HCChronologicalTermsParams) == 8);
#ifndef __METAL_VERSION__
#include <cstddef>
static_assert(offsetof(HCChronologicalTermsParams, literal) == 0);
static_assert(offsetof(HCChronologicalTermsParams, term_outputs) == 160);
static_assert(offsetof(HCChronologicalTermsParams, term_j) == 164);
static_assert(offsetof(HCChronologicalTermsParams, slice_terms) == 168);
static_assert(offsetof(HCChronologicalTermsParams, reserved0) == 172);
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
