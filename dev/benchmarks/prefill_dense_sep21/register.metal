// Copyright © 2025 Apple Inc.
// Private numerical alternative. Fragment coordinates and register MMA packing
// are adapted from the MIT-licensed MLX Steel BaseNAXFrag in the local vendor
// snapshot at prefill4k_dense/steel_vendor (commit 2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99).
// BF16 A/B, ascending BK512 blocks with SK16 register MMAs, persistent F32 C,
// and one final BF16 cast. This changes reduction order from whole-K Splash MPP.
// The relaxed variants also enable descriptor relaxed precision. Neither variant
// is a byte-exact or model-qualified substitution merely because it compiles.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashDenseCache.h"
#include "metal/kernels/common/flash_dense_traversal.h"

#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

using DenseRegisterBF16 = array<bfloat, 8>;
using DenseRegisterF32 = array<float, 8>;

inline void dense_register_error(device atomic_uint *diagnostics, uint flag) {
  atomic_fetch_or_explicit(diagnostics, flag, memory_order_relaxed);
}

// One 16x16 fragment has eight elements per lane: two rows eight apart,
// with four contiguous columns. The coordinate order matches Steel register
// tensors, including the transposed right operand packing used below.
inline uint2 dense_register_coordinate(uint lane) {
  const uint quad = lane >> 2;
  const uint row = ((quad & 4) | ((lane >> 1) & 3));
  const uint column = ((quad & 2) | (lane & 1)) * 4;
  return uint2(column, row);
}

inline DenseRegisterBF16 dense_register_load(const device bfloat *source,
    uint stride, uint rowOrigin, uint kOrigin, uint rowLimit, uint2 coordinate) {
  DenseRegisterBF16 result;
#pragma unroll
  for (uint i = 0; i < 2; ++i) {
    const uint row = rowOrigin + coordinate.y + i * 8;
#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      result[i * 4 + j] = row < rowLimit
          ? source[ulong(row) * stride + kOrigin + coordinate.x + j]
          : bfloat(0.0f);
    }
  }
  return result;
}

template <bool Relaxed>
inline void dense_register_mma(thread DenseRegisterF32 &c0,
    thread DenseRegisterF32 &c1, const thread DenseRegisterBF16 &a,
    const thread DenseRegisterBF16 &b0, const thread DenseRegisterBF16 &b1) {
  constexpr auto descriptor = matmul2d_descriptor(16, 32, 16,
      false, true, Relaxed, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> operation;
  auto left = operation.template get_left_input_cooperative_tensor<
      bfloat, bfloat, float>();
  auto right = operation.template get_right_input_cooperative_tensor<
      bfloat, bfloat, float>();
  auto destination = operation.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(left)>, remove_addrspace_t<decltype(right)>, float>();
#pragma unroll
  for (uint i = 0; i < 8; ++i) {
    left[i] = a[i];
    right[i] = b0[i];
    right[8 + i] = b1[i];
    destination[i] = c0[i];
    destination[8 + i] = c1[i];
  }
  operation.run(left, right, destination);
#pragma unroll
  for (uint i = 0; i < 8; ++i) {
    c0[i] = destination[i];
    c1[i] = destination[8 + i];
  }
}

template <ushort N, bool Relaxed>
inline void dense_register_tile(const device bfloat *input,
    const device bfloat *weights, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashDenseCacheParams &p,
    uint3 group, uint3 threads, uint tid, uint lane) {
  constexpr uint M = 32, RowFragments = M / 16, ColumnFragments = N / 16;
  static_assert(N == 32 || N == 64);
  const uint rowTiles = (p.rows + M - 1) / M;
  const uint columnTiles = p.output_count / N;
  if (!p.rows || p.rows > 8192 || !p.input_size || p.input_size > 32768 ||
      p.input_size % 512 || !p.output_size || p.output_size > 32768 ||
      !p.output_count || p.output_count % N || p.output_begin > p.output_size ||
      p.output_count > p.output_size - p.output_begin || p.tile_rows != M ||
      p.tile_outputs != N || p.reserved > 4 || group.z ||
      !flash_dense_traversal_group_valid(group.xy, rowTiles, columnTiles, p.reserved) ||
      threads.x != 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) dense_register_error(diagnostics, 2u);
    return;
  }
  const uint2 tile = flash_dense_traversal_tile(group.xy, p.reserved);
  // A grouped traversal may contain inactive row tiles in the final group.
  if (tile.y >= rowTiles || tile.x >= columnTiles) return;
  const uint rowOrigin = tile.y * M;
  const uint columnOrigin = p.output_begin + tile.x * N;
  const uint2 coordinate = dense_register_coordinate(lane);
  DenseRegisterF32 accumulators[RowFragments][ColumnFragments];
#pragma unroll
  for (uint m = 0; m < RowFragments; ++m)
#pragma unroll
    for (uint n = 0; n < ColumnFragments; ++n)
#pragma unroll
      for (uint i = 0; i < 8; ++i) accumulators[m][n][i] = 0.0f;

#pragma nounroll
  for (uint block = 0; block < p.input_size; block += 512) {
#pragma nounroll
    for (uint inner = 0; inner < 512; inner += 16) {
      const uint k = block + inner;
      DenseRegisterBF16 left[RowFragments];
      DenseRegisterBF16 right[ColumnFragments];
#pragma unroll
      for (uint m = 0; m < RowFragments; ++m)
        left[m] = dense_register_load(input, p.input_size,
            rowOrigin + m * 16, k, p.rows, coordinate);
#pragma unroll
      for (uint n = 0; n < ColumnFragments; ++n)
        right[n] = dense_register_load(weights, p.input_size,
            columnOrigin + n * 16, k, p.output_size, coordinate);
#pragma unroll
      for (uint m = 0; m < RowFragments; ++m)
#pragma unroll
        for (uint n = 0; n < ColumnFragments; n += 2)
          dense_register_mma<Relaxed>(accumulators[m][n], accumulators[m][n + 1],
              left[m], right[n], right[n + 1]);
    }
  }

#pragma unroll
  for (uint m = 0; m < RowFragments; ++m)
#pragma unroll
    for (uint n = 0; n < ColumnFragments; ++n)
#pragma unroll
      for (uint i = 0; i < 2; ++i) {
        const uint row = rowOrigin + m * 16 + coordinate.y + i * 8;
        if (row >= p.rows) continue;
#pragma unroll
        for (uint j = 0; j < 4; ++j) {
          const uint column = columnOrigin + n * 16 + coordinate.x + j;
          const float value = accumulators[m][n][i * 4 + j];
          const bfloat finalValue = bfloat(value);
          if (!isfinite(value) || !isfinite(float(finalValue)))
            dense_register_error(diagnostics, 4u);
          output[ulong(row) * p.output_size + column] = finalValue;
        }
      }
}

#define DENSE_REGISTER_ENTRY(N, NAME, RELAXED) \
[[max_total_threads_per_threadgroup(32)]] kernel void \
prefill_dense_register_m32_n##N##_sg1_bk512_sk16_##NAME( \
    const device bfloat *input [[buffer(0)]], const device bfloat *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], device atomic_uint *diagnostics [[buffer(3)]], \
    constant FlashDenseCacheParams &p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  dense_register_tile<N, RELAXED>(input, weights, output, diagnostics, p, \
      group, threads, tid, lane); \
}
DENSE_REGISTER_ENTRY(32, strict, false)
DENSE_REGISTER_ENTRY(64, strict, false)
DENSE_REGISTER_ENTRY(32, relaxed, true)
DENSE_REGISTER_ENTRY(64, relaxed, true)
#undef DENSE_REGISTER_ENTRY
