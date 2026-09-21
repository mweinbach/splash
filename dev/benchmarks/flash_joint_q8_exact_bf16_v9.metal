// Private original-Q8 coefficient decode. Root alone runs Metal commands.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashInt8Head.h"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

inline bool exact_q8_geometry(constant FlashInt8HeadParams &p) {
  return p.rows >= 2 && p.rows <= 8 && p.padded_rows == 8 &&
      p.input_size == 2560 && p.output_size == 248320 && p.bits == 8 &&
      p.group_size == 64 && p.tile_rows == 8 &&
      (p.tile_outputs == 32 || p.tile_outputs == 64 || p.tile_outputs == 128) &&
      p.weight_row_stride_bytes >= 2560 &&
      p.parameter_row_stride_bytes >= 80 && !(p.parameter_row_stride_bytes % 2);
}

// Audit every actual coefficient against the already qualified cache. The
// helper and cast are exactly the ones in flash_dense_cache_expand_bf16.
kernel void flash_joint_q8_exact_bf16_v9_audit(
    device const uchar *codes [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device const uchar *biases [[buffer(2)]],
    device const ushort *cached [[buffer(3)]],
    device atomic_uint *counts [[buffer(4)]],
    constant FlashInt8HeadParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (!exact_q8_geometry(p)) {
    if (!gid) atomic_fetch_or_explicit(counts + 1, 1u, memory_order_relaxed);
    return;
  }
  if (ulong(gid) >= ulong(p.output_size) * p.input_size) return;
  const uint n = gid / p.input_size, k = gid % p.input_size;
  const ulong offset = ulong(n) * p.parameter_row_stride_bytes + (k / 64) * 2;
  const bfloat s = *reinterpret_cast<device const bfloat *>(scales + offset);
  const bfloat b = *reinterpret_cast<device const bfloat *>(biases + offset);
  const float value = flash_mpp_dequantize_f32(
      uint(codes[ulong(n) * p.weight_row_stride_bytes + k]), s, b);
  const bfloat coefficient = bfloat(value);
  if (as_type<ushort>(coefficient) != cached[gid])
    atomic_fetch_add_explicit(counts, 1u, memory_order_relaxed);
  if (!flash_mpp_finite(value) || !flash_mpp_finite(coefficient))
    atomic_fetch_add_explicit(counts + 1, 1u, memory_order_relaxed);
}

kernel void flash_joint_q8_exact_bf16_v9_lut_audit(
    device const uchar *codes [[buffer(0)]],
    device const ushort *indices [[buffer(1)]],
    device const ushort *dictionary [[buffer(2)]],
    device const ushort *cached [[buffer(3)]],
    device atomic_uint *counts [[buffer(4)]],
    constant FlashInt8HeadParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (!exact_q8_geometry(p)) {
    if (!gid) atomic_fetch_or_explicit(counts + 1, 1u, memory_order_relaxed);
    return;
  }
  if (ulong(gid) >= ulong(p.output_size) * p.input_size) return;
  const uint n = gid / p.input_size, k = gid % p.input_size;
  const uint pair = indices[ulong(n) * 40 + k / 64];
  if (pair >= 41151) {
    atomic_fetch_add_explicit(counts + 1, 1u, memory_order_relaxed);
    return;
  }
  const ushort coefficient = dictionary[pair * 256 +
      codes[ulong(n) * p.weight_row_stride_bytes + k]];
  if (coefficient != cached[gid])
    atomic_fetch_add_explicit(counts, 1u, memory_order_relaxed);
  if ((coefficient & 0x7f80u) == 0x7f80u)
    atomic_fetch_add_explicit(counts + 1, 1u, memory_order_relaxed);
}

template <ushort N, ushort BK, ushort SG, bool LUT = false>
inline void exact_q8_tile(
    device bfloat *padded, device const uchar *codes,
    device const uchar *scaleBytes, device const uchar *biasBytes,
    device bfloat *output, device uint *diagnostics,
    constant FlashInt8HeadParams &p, uint3 group, uint3 threads, uint tid,
    threadgroup bfloat *stagedB) {
  if (!exact_q8_geometry(p) || p.tile_outputs != N || group.z || group.y ||
      group.x >= p.output_size / N || threads.x != SG * 32 ||
      threads.y != 1 || threads.z != 1) {
    if (!tid) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const uint column = group.x * N;
  auto a = tensor(padded, dextents<int, 2>{2560, 8}, array<int, 2>{1, 2560});
  auto b = tensor(stagedB, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});
  auto at = a.template slice<BK, 8>(0, 0);
  auto bt = b.template slice<BK, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(8, N, BK, false, true,
      false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto total = operation.template get_destination_cooperative_tensor<
      decltype(at), decltype(bt), float>();
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i)
    if (total.is_valid_element(i)) total[i] = 0.0f;
  for (uint origin = 0; origin < 2560; origin += BK) {
    for (uint i = tid; i < N * BK; i += SG * 32) {
      const uint n = column + i / BK, k = origin + i % BK;
      bfloat coefficient;
      if constexpr (LUT) {
        const device ushort *indices = reinterpret_cast<device const ushort *>(scaleBytes);
        const device bfloat *dictionary = reinterpret_cast<device const bfloat *>(biasBytes);
        const uint pair = indices[ulong(n) * 40 + k / 64];
        if (pair >= 41151) {
          flash_mpp_error(diagnostics, 2u);
          coefficient = bfloat(0.0f);
        } else {
          coefficient = dictionary[pair * 256 + codes[ulong(n) * p.weight_row_stride_bytes + k]];
        }
      } else {
        const ulong offset = ulong(n) * p.parameter_row_stride_bytes + (k / 64) * 2;
        const bfloat scale = *reinterpret_cast<device const bfloat *>(scaleBytes + offset);
        const bfloat bias = *reinterpret_cast<device const bfloat *>(biasBytes + offset);
        const float reconstructed = flash_mpp_dequantize_f32(
            uint(codes[ulong(n) * p.weight_row_stride_bytes + k]), scale, bias);
        coefficient = bfloat(reconstructed);
        if (!flash_mpp_finite(reconstructed)) flash_mpp_error(diagnostics, 4u);
      }
      if (!flash_mpp_finite(coefficient)) flash_mpp_error(diagnostics, 4u);
      stagedB[i] = coefficient;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    at = a.template slice<BK, 8>(origin, 0);
    operation.run(at, bt, total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i) {
    if (!total.is_valid_element(i)) continue;
    const auto index = total.get_multidimensional_index(i);
    if (uint(index[1]) >= p.rows) continue;
    const bfloat result = bfloat(total[i]);
    if (!flash_mpp_finite(total[i]) || !flash_mpp_finite(result))
      flash_mpp_error(diagnostics, 4u);
    output[ulong(index[1]) * p.output_size + column + index[0]] = result;
  }
}

#define EXACT_Q8_ENTRY(N, BK, SG) \
kernel void flash_joint_q8_exact_bf16_v9_m8_n##N##_k##BK##_s##SG( \
    device bfloat *input [[buffer(0)]], device const uchar *codes [[buffer(1)]], \
    device const uchar *scales [[buffer(2)]], device const uchar *biases [[buffer(3)]], \
    device const float *unused [[buffer(4)]], device bfloat *output [[buffer(5)]], \
    device uint *diagnostics [[buffer(6)]], constant FlashInt8HeadParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  (void)unused; threadgroup bfloat stagedB[N * BK]; \
  exact_q8_tile<N, BK, SG>(input, codes, scales, biases, output, diagnostics, \
                         p, group, threads, tid, stagedB); \
}
EXACT_Q8_ENTRY(64, 64, 4)
EXACT_Q8_ENTRY(64, 128, 4)
EXACT_Q8_ENTRY(128, 64, 4)
EXACT_Q8_ENTRY(128, 128, 4)
EXACT_Q8_ENTRY(64, 128, 2)
EXACT_Q8_ENTRY(32, 64, 4)
EXACT_Q8_ENTRY(64, 64, 8)
#undef EXACT_Q8_ENTRY

#define EXACT_Q8_LUT_ENTRY(N, BK, SG) \
kernel void flash_joint_q8_exact_bf16_v9_lut_m8_n##N##_k##BK##_s##SG( \
    device bfloat *input [[buffer(0)]], device const uchar *codes [[buffer(1)]], \
    device const uchar *indices [[buffer(2)]], device const uchar *dictionary [[buffer(3)]], \
    device const float *unused [[buffer(4)]], device bfloat *output [[buffer(5)]], \
    device uint *diagnostics [[buffer(6)]], constant FlashInt8HeadParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  (void)unused; threadgroup bfloat stagedB[N * BK]; \
  exact_q8_tile<N, BK, SG, true>(input, codes, indices, dictionary, output, diagnostics, \
                         p, group, threads, tid, stagedB); \
}
EXACT_Q8_LUT_ENTRY(64, 64, 4)
EXACT_Q8_LUT_ENTRY(64, 128, 4)
EXACT_Q8_LUT_ENTRY(128, 64, 4)
EXACT_Q8_LUT_ENTRY(128, 128, 4)
EXACT_Q8_LUT_ENTRY(64, 128, 2)
EXACT_Q8_LUT_ENTRY(32, 64, 4)
EXACT_Q8_LUT_ENTRY(64, 64, 8)
#undef EXACT_Q8_LUT_ENTRY
#endif
