#pragma once
#include "metal/abi/FlashAffine.h"

// Pair bindings: input0, AW1/AS2/AB3, BW4/BS5/BB6, Aout7/Bout8,
// diagnostics9. Timed params follow at10. Probes append ArawF32 at10,
// BrawF32 at11, pair params at12. Single-plane probes retain the pair ABI.
struct GDNABMergeParams {
  FlashAffineParams a;
  FlashAffineParams b;
};

#ifdef __METAL_VERSION__
#define GDN_AB_CONFIG_REF const constant
#else
#define GDN_AB_CONFIG_REF const
#endif
namespace gdn_ab_merge_sep21 {
enum : uint32_t {
  kInputSize = 2560, kOutputSize = 48, kOutputGroups = 6, kThreads = 64,
  kTimedParamsSlot = 10, kProbeARawSlot = 10,
  kProbeBRawSlot = 11, kProbeParamsSlot = 12
};

// Preserve every original project predicate and restrict this dispatch
// experiment to measured GDN A/B shapes. Expert-stride extents are unused for
// expert0; preserve the original parameter-expert-stride alignment predicate.
constexpr bool validPlane(GDN_AB_CONFIG_REF FlashAffineParams &p) {
  return (p.rows == 1 || p.rows == 4) && p.selections == 1 &&
      p.input_size == 2560 && p.output_size == 48 &&
      p.experts == 1 && p.flags == 0 &&
      ((p.bits == 5 && p.group_size == 128) ||
       (p.bits == 6 && p.group_size == 64)) &&
      p.input_size % p.group_size == 0 &&
      p.weight_row_stride_bytes >= uint64_t(p.input_size) * p.bits / 8 &&
      p.parameter_row_stride_bytes >= uint64_t(p.input_size / p.group_size) * 2 &&
      p.parameter_row_stride_bytes % 2 == 0 &&
      p.parameter_expert_stride_bytes % 2 == 0;
}
constexpr bool validPair(GDN_AB_CONFIG_REF GDNABMergeParams &p) {
  return validPlane(p.a) && validPlane(p.b) && p.a.rows == p.b.rows;
}
constexpr uint64_t rawIndex(uint32_t row, uint32_t output) {
  return uint64_t(row) * 48 + output;
}
} // namespace gdn_ab_merge_sep21
#undef GDN_AB_CONFIG_REF

static_assert(sizeof(FlashAffineParams) == 64);
static_assert(alignof(FlashAffineParams) == 8);
static_assert(sizeof(GDNABMergeParams) == 128);
static_assert(alignof(GDNABMergeParams) == 8);
#ifndef __METAL_VERSION__
#include <cstddef>
static_assert(offsetof(GDNABMergeParams, a) == 0);
static_assert(offsetof(GDNABMergeParams, b) == 64);
static_assert(offsetof(FlashAffineParams, rows) == 0);
static_assert(offsetof(FlashAffineParams, selections) == 4);
static_assert(offsetof(FlashAffineParams, input_size) == 8);
static_assert(offsetof(FlashAffineParams, output_size) == 12);
static_assert(offsetof(FlashAffineParams, experts) == 16);
static_assert(offsetof(FlashAffineParams, bits) == 20);
static_assert(offsetof(FlashAffineParams, group_size) == 24);
static_assert(offsetof(FlashAffineParams, flags) == 28);
static_assert(offsetof(FlashAffineParams, weight_row_stride_bytes) == 32);
static_assert(offsetof(FlashAffineParams, weight_expert_stride_bytes) == 40);
static_assert(offsetof(FlashAffineParams, parameter_row_stride_bytes) == 48);
static_assert(offsetof(FlashAffineParams, parameter_expert_stride_bytes) == 56);
#endif
