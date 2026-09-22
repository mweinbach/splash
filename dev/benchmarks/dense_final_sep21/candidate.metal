// Private CPU-compiled dense geometry and HALF-input screen.
// Whole K is accumulated into F32; every destination is finally rounded to BF16.
// Traversal/shape guards retain the production FlashDenseCache ABI.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashDenseCache.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
#include "metal/kernels/common/flash_dense_traversal.h"

inline void dense_final_error(device atomic_uint *diagnostics, uint flag) {
  atomic_fetch_or_explicit(diagnostics, flag, memory_order_relaxed);
}

template <typename Input, ushort M, ushort N, ushort Groups>
inline void dense_final_tile(device Input *input, device Input *weights,
    device bfloat *output, device atomic_uint *diagnostics,
    constant FlashDenseCacheParams &p, uint3 group, uint3 threads, uint tid) {
  if (p.rows != 2048 || !p.input_size || p.input_size > 32768 ||
      p.input_size % 32 || !p.output_size || !p.output_count ||
      p.output_count % N || p.output_begin > p.output_size ||
      p.output_count > p.output_size - p.output_begin ||
      p.tile_rows != M || p.tile_outputs != N || p.reserved > 4 || group.z ||
      !flash_dense_traversal_group_valid(group.xy, p.rows / M,
          p.output_count / N, p.reserved) ||
      threads.x != Groups * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) dense_final_error(diagnostics, 2u);
    return;
  }
  const uint2 tile = flash_dense_traversal_tile(group.xy, p.reserved);
  if (tile.x >= p.output_count / N || tile.y >= p.rows / M) return;
  const uint row = tile.y * M, column = p.output_begin + tile.x * N;
  const int k = int(p.input_size);
  auto a = tensor(input + ulong(row) * p.input_size,
      dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto b = tensor(weights + ulong(column) * p.input_size,
      dextents<int, 2>{k, N}, array<int, 2>{1, k});
  constexpr auto descriptor = matmul2d_descriptor(M, N,
      static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<Groups>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const bfloat value = bfloat(dot[i]);
    if (!isfinite(dot[i]) || !isfinite(float(value))) {
      dense_final_error(diagnostics, 4u);
    }
    output[ulong(row + index[1]) * p.output_size + column + index[0]] = value;
  }
}

#define DENSE_FINAL_ENTRY(TYPE, LABEL, M, N, G) \
[[max_total_threads_per_threadgroup(G * 32)]] \
kernel void dense_final_##LABEL##_m##M##_n##N##_sg##G( \
    device TYPE *input [[buffer(0)]], device TYPE *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], \
    device atomic_uint *diagnostics [[buffer(3)]], \
    constant FlashDenseCacheParams &p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  dense_final_tile<TYPE, M, N, G>( \
      input, weights, output, diagnostics, p, group, threads, tid); \
}

#ifndef DENSE_FINAL_DISABLE_BF16_M256_N32_SG4
DENSE_FINAL_ENTRY(bfloat, bf16, 256, 32, 4)
#endif
#ifndef DENSE_FINAL_DISABLE_BF16_M256_N32_SG8
DENSE_FINAL_ENTRY(bfloat, bf16, 256, 32, 8)
#endif
#ifndef DENSE_FINAL_DISABLE_BF16_M256_N64_SG8
DENSE_FINAL_ENTRY(bfloat, bf16, 256, 64, 8)
#endif
#ifndef DENSE_FINAL_DISABLE_HALF_M128_N64_SG4
DENSE_FINAL_ENTRY(half, half, 128, 64, 4)
#endif
#ifndef DENSE_FINAL_DISABLE_HALF_M128_N64_SG8
DENSE_FINAL_ENTRY(half, half, 128, 64, 8)
#endif
#ifndef DENSE_FINAL_DISABLE_HALF_M256_N32_SG4
DENSE_FINAL_ENTRY(half, half, 256, 32, 4)
#endif
#ifndef DENSE_FINAL_DISABLE_HALF_M256_N32_SG8
DENSE_FINAL_ENTRY(half, half, 256, 32, 8)
#endif
#ifndef DENSE_FINAL_DISABLE_HALF_M256_N64_SG8
DENSE_FINAL_ENTRY(half, half, 256, 64, 8)
#endif
#undef DENSE_FINAL_ENTRY

// ABI: source[0], HALF destination[1], atomic OR diagnostics[2],
// constant uint scalar_count[3]. Dispatch one dimension; tail lanes are valid.
// flag 2: invalid scalar count/dimensional dispatch; flag 4: nonfinite source
// or finite BF16 input overflowed to nonfinite HALF. No clamp is performed.
[[max_total_threads_per_threadgroup(256)]]
kernel void dense_final_bf16_to_half(
    device const bfloat *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant uint &scalar_count [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]]) {
  if (!scalar_count || gid.y || gid.z || threads.x != 256 ||
      threads.y != 1 || threads.z != 1) {
    if (!gid.x) dense_final_error(diagnostics, 2u);
    return;
  }
  if (gid.x >= scalar_count) return;
  const float value = float(input[gid.x]);
  const half converted = half(value);
  if (!isfinite(value) || !isfinite(float(converted))) {
    dense_final_error(diagnostics, 4u);
  }
  output[gid.x] = converted;
}
