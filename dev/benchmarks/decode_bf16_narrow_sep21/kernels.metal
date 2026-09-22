#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "KernelParams.hpp"

#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#include "metal/kernels/common/flash_affine_mpp_common.h"

using namespace metal;
using namespace mpp::tensor_ops;

// Every check is uniform within a threadgroup, before the cooperative MPP
// call. Diagnostics are sticky atomic ORs, preserving the caller's bits.
template <ushort M, ushort N, ushort SG>
inline bool sep21_bf16_narrow_geometry(
    constant Sep21BF16NarrowParams &p, uint3 group, uint3 threads) {
  const bool rows = M == 8
      ? (p.rows == 1 || p.rows == 2 || p.rows == 4 || p.rows == 8)
      : M == 16 && p.rows == 16;
  return rows && p.tile_rows == M && p.tile_outputs == N &&
      p.padded_rows == M && p.input_size && p.input_size <= 32768 &&
      p.input_size % 32 == 0 && p.output_size && p.output_size % 64 == 0 &&
      p.output_count && p.output_count % N == 0 &&
      p.output_begin <= p.output_size &&
      p.output_count <= p.output_size - p.output_begin &&
      group.x < p.output_count / N && group.y == 0 && group.z == 0 &&
      threads.x == uint(SG) * 32 && threads.y == 1 && threads.z == 1;
}

// Device-BF16 operands, K-major/row-contiguous layout and a single whole-K
// multiply descriptor exactly match flash_dense_small_rows_tile. There is no
// input conversion, numerical split, threadgroup staging or vector-MMA path.
// Tap is a compile-time branch: timed instantiations never store an F32 tap.
template <ushort M, ushort N, ushort SG, bool Tap>
inline void sep21_bf16_narrow_tile(
    device bfloat *padded, device bfloat *weights, device bfloat *output,
    device uint *diagnostics, device float *rawDot,
    constant Sep21BF16NarrowParams &p,
    uint3 group, uint3 threads, uint tid) {
  if (!sep21_bf16_narrow_geometry<M, N, SG>(p, group, threads)) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const uint column = p.output_begin + group.x * N;
  const int k = int(p.input_size);
  auto a = tensor(padded,
      dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto b = tensor(weights + ulong(column) * p.input_size,
      dextents<int, 2>{k, N}, array<int, 2>{1, k});
  constexpr auto descriptor = matmul2d_descriptor(
      M, N, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const uint outputRow = index[1];
    if (outputRow >= p.rows) continue;
    const bfloat value = bfloat(dot[i]);
    if (!flash_mpp_finite(dot[i]) || !flash_mpp_finite(value))
      flash_mpp_error(diagnostics, 4u);
    const ulong destination = ulong(outputRow) * p.output_size + column + index[0];
    output[destination] = value;
    if constexpr (Tap) rawDot[destination] = dot[i];
  }
}

#define SEP21_BF16_NARROW(Name, M, N, SG)                                  \
  kernel void Name(device bfloat *padded [[buffer(0)]],                    \
      device bfloat *weights [[buffer(1)]], device bfloat *output [[buffer(2)]], \
      device uint *diagnostics [[buffer(3)]],                              \
      constant Sep21BF16NarrowParams &params [[buffer(4)]],                 \
      uint3 group [[threadgroup_position_in_grid]],                        \
      uint3 threads [[threads_per_threadgroup]],                          \
      uint tid [[thread_index_in_threadgroup]]) {                         \
    sep21_bf16_narrow_tile<M, N, SG, false>(                               \
        padded, weights, output, diagnostics, nullptr, params, group, threads, tid); \
  }

#define SEP21_BF16_PROBE(Name, M, N, SG)                                   \
  kernel void Name(device bfloat *padded [[buffer(0)]],                    \
      device bfloat *weights [[buffer(1)]], device bfloat *output [[buffer(2)]], \
      device uint *diagnostics [[buffer(3)]],                              \
      device float *rawDot [[buffer(4)]],                                 \
      constant Sep21BF16NarrowParams &params [[buffer(5)]],                 \
      uint3 group [[threadgroup_position_in_grid]],                        \
      uint3 threads [[threads_per_threadgroup]],                          \
      uint tid [[thread_index_in_threadgroup]]) {                         \
    sep21_bf16_narrow_tile<M, N, SG, true>(                                \
        padded, weights, output, diagnostics, rawDot, params, group, threads, tid); \
  }

#define SEP21_BF16_PAIR(MName, M, NName, N, SName, SG)                       \
  SEP21_BF16_NARROW(sep21_bf16_##MName##_##NName##_##SName, M, N, SG)       \
  SEP21_BF16_PROBE(sep21_bf16_probe_##MName##_##NName##_##SName, M, N, SG)

SEP21_BF16_PAIR(m8, 8, n32, 32, s1, 1)
SEP21_BF16_PAIR(m8, 8, n32, 32, s2, 2)
SEP21_BF16_PAIR(m8, 8, n32, 32, s4, 4)
SEP21_BF16_PAIR(m8, 8, n64, 64, s1, 1)
SEP21_BF16_PAIR(m8, 8, n64, 64, s2, 2)
SEP21_BF16_PAIR(m8, 8, n64, 64, s4, 4)
SEP21_BF16_PAIR(m16, 16, n32, 32, s1, 1)
SEP21_BF16_PAIR(m16, 16, n32, 32, s2, 2)
SEP21_BF16_PAIR(m16, 16, n32, 32, s4, 4)
SEP21_BF16_PAIR(m16, 16, n64, 64, s1, 1)
SEP21_BF16_PAIR(m16, 16, n64, 64, s2, 2)
SEP21_BF16_PAIR(m16, 16, n64, 64, s4, 4)

// Raw-dot controls accompany the separate, untouched original N128 shaders.
SEP21_BF16_PROBE(sep21_bf16_probe_m8_n128_s4, 8, 128, 4)
SEP21_BF16_PROBE(sep21_bf16_probe_m16_n128_s4, 16, 128, 4)

#undef SEP21_BF16_PAIR
#undef SEP21_BF16_PROBE
#undef SEP21_BF16_NARROW
#endif
