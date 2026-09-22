// Private one-layer saved-Full512-I8 gate/up pairing experiment.
// Only the whole-row arrangement in a temporary B plane is new; fitted I8
// coefficients/scales/ranks are unchanged. No down/decode/production edits.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "abi.h"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

// Exact native saved-I8 BF16 activation stages. Keep the explicit rounding
// boundaries and the native compiled fast-exp implementation.
#pragma METAL fp math_mode(fast)
inline bfloat prefill_paired_gate_sep21_native_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::fast::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
#pragma METAL fp math_mode(safe)

template <ushort SG, ushort BlockK, bool IntQuant, bool Audit>
inline bool prefill_paired_gate_sep21_job(
    constant FlashInt8ExpertStoreParams &p,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device uint *diag, uint3 group, uint3 threads, uint tid,
    thread uint &rank, thread uint &begin, thread uint &valid_rows) {
  constexpr uint M = kPrefillPairedGateSep21TileRows;
  if (!p.rows || p.rows > kFlashMoEBucketMaximumRows || !p.selections ||
      p.selections > kFlashMoEBucketMaximumSelections ||
      p.route_capacity != p.rows * p.selections ||
      p.job_capacity != (p.route_capacity + M - 1) / M + 511 ||
      p.tile_rows != M || !p.stored_experts || p.stored_experts > 512 ||
      p.scale_group_size || p.reserved ||
      group.x >= kPrefillPairedGateSep21ColumnBlocks ||
      group.y >= p.job_capacity || group.z ||
      threads.x != uint(SG) * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) flash_mpp_error(diag, 2u);
    return false;
  }
  const uint active = job_count[0];
  if (active > min(p.job_capacity, p.route_capacity) ||
      offsets[512] > p.route_capacity) {
    if (!tid) flash_mpp_error(diag, 2u);
    return false;
  }
  if (group.y >= active) return false;
  const auto job = jobs[group.y];
  if (job.expert >= 512) {
    if (!tid) flash_mpp_error(diag, 1u);
    return false;
  }
  const uint first = offsets[job.expert], end = offsets[job.expert + 1];
  if (first > end || end > p.route_capacity ||
      job.row_begin < first || job.row_begin >= end) {
    if (!tid) flash_mpp_error(diag, 2u);
    return false;
  }
  rank = ranks[job.expert];
  if (rank == UINT_MAX) {
    if (p.stored_experts == 512 && !tid) flash_mpp_error(diag, 1u);
    return false;
  }
  if (rank >= p.stored_experts) {
    if (!tid) flash_mpp_error(diag, 1u);
    return false;
  }
  begin = job.row_begin;
  valid_rows = min(M, end - begin);
  return true;
}

// Including IntQuant and Audit in this uniquely named helper's template
// identity prevents the unrelated previous experiments' weak specializations
// from being coalesced into a different numerical producer during linking.
template <ushort SG, ushort BlockK, bool IntQuant, bool Audit>
inline void prefill_paired_gate_sep21_gate(
    device conditional_t<IntQuant, int8_t, bfloat> *input,
    device int8_t *paired_codes,
    device const float *gate_scale, device const float *up_scale,
    device const uint *ranks, device const uint *offsets,
    device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device bfloat *output, device uint *diag,
    constant FlashInt8ExpertStoreParams &p,
    device const float *activation_scales,
    device float *raw_gate, device float *raw_up,
    device float *scaled_gate, device float *scaled_up,
    device int *dot_gate, device int *dot_up,
    uint3 group, uint3 threads, uint tid,
    threadgroup bfloat *rounded_pairs) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  uint rank, begin, valid_rows;
  if (!prefill_paired_gate_sep21_job<SG, BlockK, IntQuant, Audit>(p, ranks,
      offsets, jobs, job_count, diag, group, threads, tid,
      rank, begin, valid_rows)) return;
  constexpr ushort M = kPrefillPairedGateSep21TileRows;
  constexpr ushort N = kPrefillPairedGateSep21TileColumns;
  constexpr uint K = kPrefillPairedGateSep21InputColumns;
  constexpr uint OutputColumns = kPrefillPairedGateSep21OutputColumns;
  constexpr uint PairedColumns = kPrefillPairedGateSep21PairedColumns;
  const uint output_column = group.x * 64;
  const uint paired_column = group.x * N;
  // Dynamic A row bounds retain native incomplete-job masking: never read
  // source rows from the next expert or round/copy any additional A elements.
  auto a = tensor(input + ulong(begin) * K,
      dextents<int, 2>{int(K), int(valid_rows)}, array<int, 2>{1, int(K)});
  auto b = tensor(paired_codes +
      (ulong(rank) * PairedColumns + paired_column) * K,
      dextents<int, 2>{int(K), N}, array<int, 2>{1, int(K)});
  constexpr auto descriptor = matmul2d_descriptor(M, N,
      BlockK ? int(BlockK) : static_cast<int>(dynamic_extent),
      false, true, false, BlockK ? matmul2d_descriptor::mode::multiply_accumulate
                               : matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  using Dot = conditional_t<IntQuant, int, float>;
  auto dot = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(b), Dot>();
  if constexpr (BlockK) {
#pragma unroll
    for (ushort i = 0; i < dot.get_capacity(); ++i)
      if (dot.is_valid_element(i)) dot[i] = Dot(0);
    for (uint k = 0; k < K; k += BlockK) {
      auto aa = a.slice(k, 0); auto bb = b.slice(k, 0);
      operation.run(aa, bb, dot);
    }
  } else { operation.run(a, b, dot); }
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const uint paired_n = uint(index[0]), row = uint(index[1]);
    if (paired_n >= N || row >= valid_rows) continue;
    const bool is_up = paired_n >= 64;
    const uint n = output_column + (paired_n % 64);
    const ulong at = ulong(begin + row) * OutputColumns + n;
    const float weight_scale = is_up
        ? up_scale[ulong(rank) * OutputColumns + n]
        : gate_scale[ulong(rank) * OutputColumns + n];
    const float raw = float(dot[i]);
    float unscaled = raw;
    if constexpr (IntQuant) {
      const float activation_scale = activation_scales[begin + row];
      unscaled = raw * activation_scale;
      if (!(activation_scale > 0.0f) || !flash_mpp_finite(activation_scale))
        flash_mpp_error(diag, 4u);
    }
    const float scaled = unscaled * weight_scale;
    const bfloat rounded = bfloat(scaled);
    if (!(weight_scale > 0.0f) || !flash_mpp_finite(weight_scale) ||
        !flash_mpp_finite(scaled)) flash_mpp_error(diag, 4u);
    if constexpr (Audit) {
      if (is_up) { raw_up[at] = raw; scaled_up[at] = scaled; }
      else { raw_gate[at] = raw; scaled_gate[at] = scaled; }
      if constexpr (IntQuant) {
        if (is_up) dot_up[at] = dot[i];
        else dot_gate[at] = dot[i];
      }
    }
    // Cooperative destination ownership is opaque. Address each value by its
    // API-provided multidimensional index, then pair through shared memory.
    rounded_pairs[row * N + paired_n] = rounded;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint cell = tid; cell < valid_rows * 64; cell += uint(SG) * 32) {
    const uint row = cell / 64, column = cell % 64;
    const bfloat gv = rounded_pairs[row * N + column];
    const bfloat uv = rounded_pairs[row * N + column + 64];
    const bfloat silu = gv * prefill_paired_gate_sep21_native_sigmoid(gv);
    const bfloat value = silu * uv;
    if (!flash_mpp_finite(value)) flash_mpp_error(diag, 4u);
    output[ulong(begin + row) * OutputColumns + output_column + column] = value;
  }
}

#define PAIRED_GATE_COMMON(A_TYPE) \
    device A_TYPE *a [[buffer(0)]], device int8_t *paired [[buffer(1)]], \
    device const float *gs [[buffer(2)]], device const float *us [[buffer(3)]], \
    device const uint *ranks [[buffer(4)]], device const uint *offsets [[buffer(5)]], \
    device const FlashMoEBucketJob *jobs [[buffer(6)]], \
    device const uint *count [[buffer(7)]], device bfloat *out [[buffer(8)]], \
    device uint *diag [[buffer(9)]], \
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]]
#define PAIRED_GATE_POSITION \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]
#define PAIRED_GATE_BF16(NAME, SG, BLOCK_K) \
kernel void NAME(PAIRED_GATE_COMMON(bfloat), PAIRED_GATE_POSITION) { \
  threadgroup bfloat rounded_pairs[kPrefillPairedGateSep21TileRows * \
      kPrefillPairedGateSep21TileColumns]; \
  prefill_paired_gate_sep21_gate<SG, BLOCK_K, false, false>(a, paired, gs, us, ranks, \
      offsets, jobs, count, out, diag, p, nullptr, nullptr, nullptr, nullptr, \
      nullptr, nullptr, nullptr, group, threads, tid, rounded_pairs); \
}
#define PAIRED_GATE_BF16_AUDIT(NAME, SG, BLOCK_K) \
kernel void NAME(PAIRED_GATE_COMMON(bfloat), \
    device float *raw_g [[buffer(11)]], device float *raw_u [[buffer(12)]], \
    device float *scaled_g [[buffer(13)]], device float *scaled_u [[buffer(14)]], \
    PAIRED_GATE_POSITION) { \
  threadgroup bfloat rounded_pairs[kPrefillPairedGateSep21TileRows * \
      kPrefillPairedGateSep21TileColumns]; \
  prefill_paired_gate_sep21_gate<SG, BLOCK_K, false, true>(a, paired, gs, us, ranks, \
      offsets, jobs, count, out, diag, p, nullptr, raw_g, raw_u, scaled_g, \
      scaled_u, nullptr, nullptr, group, threads, tid, rounded_pairs); \
}
#define PAIRED_GATE_W8A8(NAME, SG) \
kernel void NAME(PAIRED_GATE_COMMON(int8_t), \
    device const float *activation_scales [[buffer(11)]], \
    PAIRED_GATE_POSITION) { \
  threadgroup bfloat rounded_pairs[kPrefillPairedGateSep21TileRows * \
      kPrefillPairedGateSep21TileColumns]; \
  prefill_paired_gate_sep21_gate<SG, 0, true, false>(a, paired, gs, us, ranks, \
      offsets, jobs, count, out, diag, p, activation_scales, nullptr, nullptr, \
      nullptr, nullptr, nullptr, nullptr, group, threads, tid, rounded_pairs); \
}
#define PAIRED_GATE_W8A8_AUDIT(NAME, SG) \
kernel void NAME(PAIRED_GATE_COMMON(int8_t), \
    device const float *activation_scales [[buffer(11)]], \
    device float *raw_g [[buffer(12)]], device float *raw_u [[buffer(13)]], \
    device float *scaled_g [[buffer(14)]], device float *scaled_u [[buffer(15)]], \
    device int *dot_g [[buffer(16)]], device int *dot_u [[buffer(17)]], \
    PAIRED_GATE_POSITION) { \
  threadgroup bfloat rounded_pairs[kPrefillPairedGateSep21TileRows * \
      kPrefillPairedGateSep21TileColumns]; \
  prefill_paired_gate_sep21_gate<SG, 0, true, true>(a, paired, gs, us, ranks, \
      offsets, jobs, count, out, diag, p, activation_scales, raw_g, raw_u, \
      scaled_g, scaled_u, dot_g, dot_u, group, threads, tid, rounded_pairs); \
}

PAIRED_GATE_BF16(prefill_paired_gate_sep21_bf16_gate_up_m32_n128_sg2, 2, 0)
PAIRED_GATE_BF16_AUDIT(prefill_paired_gate_sep21_bf16_gate_up_m32_n128_sg2_audit, 2, 0)
PAIRED_GATE_BF16(prefill_paired_gate_sep21_bf16_gate_up_m32_n128_sg4, 4, 0)
PAIRED_GATE_BF16_AUDIT(prefill_paired_gate_sep21_bf16_gate_up_m32_n128_sg4_audit, 4, 0)
PAIRED_GATE_BF16(prefill_paired_gate_sep21_bf16_gate_up_m32_n128_k128_sg2, 2, 128)
PAIRED_GATE_BF16_AUDIT(prefill_paired_gate_sep21_bf16_gate_up_m32_n128_k128_sg2_audit, 2, 128)
PAIRED_GATE_BF16(prefill_paired_gate_sep21_bf16_gate_up_m32_n128_k128_sg4, 4, 128)
PAIRED_GATE_BF16_AUDIT(prefill_paired_gate_sep21_bf16_gate_up_m32_n128_k128_sg4_audit, 4, 128)
PAIRED_GATE_W8A8(prefill_paired_gate_sep21_w8a8_gate_up_m32_n128_sg2, 2)
PAIRED_GATE_W8A8_AUDIT(prefill_paired_gate_sep21_w8a8_gate_up_m32_n128_sg2_audit, 2)
PAIRED_GATE_W8A8(prefill_paired_gate_sep21_w8a8_gate_up_m32_n128_sg4, 4)
PAIRED_GATE_W8A8_AUDIT(prefill_paired_gate_sep21_w8a8_gate_up_m32_n128_sg4_audit, 4)

#undef PAIRED_GATE_COMMON
#undef PAIRED_GATE_POSITION
#undef PAIRED_GATE_BF16
#undef PAIRED_GATE_BF16_AUDIT
#undef PAIRED_GATE_W8A8
#undef PAIRED_GATE_W8A8_AUDIT
#endif
