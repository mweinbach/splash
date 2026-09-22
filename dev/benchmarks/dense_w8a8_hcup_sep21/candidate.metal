// HC-UP-only numerical alternative; F32 coefficients are fitted directly to I8.
// Signed I8 A[R,K] and B[N,K] accumulate the entire K in exact signed I32.
// Epilogue order is (float(intdot) * input_row_scale) * weight_output_scale,
// then one BF16 conversion. CPU coefficient fitting is outside GPU timings;
// GPU activation row quantization and matmul both belong to complete timings.
//
// Matmul ABI: I8 A[0], I8 B[1], BF16 output[R,N][2], I32 dot[R,N][3],
// F32 input scales[R][4], F32 coefficient scales[N][5], atomic diagnostics[6],
// FlashDenseCacheParams[7]. Normal entries bind buffer3 but do not write it;
// separate _probe entries additionally write every exact integer dot.
//
// Converter ABI: BF16 source[R,K][0], I8 codes[R,K][1], F32 scales[R][2],
// atomic diagnostics[3], DenseW8A8QuantizeParams[4]. One 256-thread group per
// row, with eight SIMD maxima reduced through threadgroup memory. Scale is
// max(abs(finite source))/127, or 1 for an all-zero row; explicit rint gives
// round-to-nearest-even before clamping to [-127,127]. Nonfinite input is
// flagged and receives code0, so no undefined nonfinite-to-integer cast occurs.
// Diagnostics OR bits: 2 invalid geometry/dispatch; 4 nonfinite value/scale.
// K<=32768 bounds even full signed I8 [-128,127] dots by K*16384 <= 536870912;
// certified symmetric codes [-127,127] have K*16129 <= 528515072.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "abi.hpp"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
#include "metal/kernels/common/flash_dense_traversal.h"

inline void dense_w8a8_error(device atomic_uint *diagnostics, uint flag) {
  atomic_fetch_or_explicit(diagnostics, flag, memory_order_relaxed);
}

template <ushort M, ushort N, ushort Groups, bool Probe>
inline void dense_w8a8_tile(device int8_t *input, device int8_t *weights,
    device bfloat *output, device int *integer_output,
    device const float *input_scales, device const float *weight_scales,
    device atomic_uint *diagnostics, constant FlashDenseCacheParams &p,
    uint3 group, uint3 threads, uint tid) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (p.rows != 2048 || p.input_size != 320 || p.output_size != 10240 ||
      !p.output_count || p.output_count % N ||
      p.output_begin > p.output_size ||
      p.output_count > p.output_size - p.output_begin ||
      p.tile_rows != M || p.tile_outputs != N || p.reserved > 4 || group.z ||
      !flash_dense_traversal_group_valid(group.xy, p.rows / M,
          p.output_count / N, p.reserved) ||
      threads.x != uint(Groups) * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) dense_w8a8_error(diagnostics, 2u);
    return;
  }
  const uint2 tile = flash_dense_traversal_tile(group.xy, p.reserved);
  if (tile.x >= p.output_count / N || tile.y >= p.rows / M) return;
  const uint row_origin = tile.y * M;
  const uint column_origin = p.output_begin + tile.x * N;
  const int k = int(p.input_size);
  auto a = tensor(input + ulong(row_origin) * p.input_size,
      dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto b = tensor(weights + ulong(column_origin) * p.input_size,
      dextents<int, 2>{k, N}, array<int, 2>{1, k});
  constexpr auto descriptor = matmul2d_descriptor(M, N,
      static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<Groups>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(b), int>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const uint row = row_origin + index[1];
    const uint column = column_origin + index[0];
    const float input_scale = input_scales[row];
    const float weight_scale = weight_scales[column];
    const float activated = float(dot[i]) * input_scale;
    const float scaled = activated * weight_scale;
    const bfloat value = bfloat(scaled);
    if (!(input_scale > 0.0f) || !(weight_scale > 0.0f) ||
        !isfinite(input_scale) || !isfinite(weight_scale) ||
        !isfinite(activated) || !isfinite(scaled) || !isfinite(float(value))) {
      dense_w8a8_error(diagnostics, 4u);
    }
    const ulong at = ulong(row) * p.output_size + column;
    if constexpr (Probe) integer_output[at] = dot[i];
    output[at] = value;
  }
}

#define DENSE_W8A8_ENTRY(NAME, GROUPS, PROBE) \
[[max_total_threads_per_threadgroup(GROUPS * 32)]] \
kernel void NAME( \
    device int8_t *input [[buffer(0)]], device int8_t *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], device int *integer_output [[buffer(3)]], \
    device const float *input_scales [[buffer(4)]], \
    device const float *weight_scales [[buffer(5)]], \
    device atomic_uint *diagnostics [[buffer(6)]], \
    constant FlashDenseCacheParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  dense_w8a8_tile<128, 64, GROUPS, PROBE>(input, weights, output, integer_output, \
      input_scales, weight_scales, diagnostics, p, group, threads, tid); \
}

DENSE_W8A8_ENTRY(dense_w8a8_m128_n64_sg4, 4, false)
DENSE_W8A8_ENTRY(dense_w8a8_m128_n64_sg8, 8, false)
DENSE_W8A8_ENTRY(dense_w8a8_m128_n64_sg4_probe, 4, true)
DENSE_W8A8_ENTRY(dense_w8a8_m128_n64_sg8_probe, 8, true)
#undef DENSE_W8A8_ENTRY

[[max_total_threads_per_threadgroup(256)]]
kernel void dense_w8a8_bf16_to_i8_t256(
    device const bfloat *input [[buffer(0)]],
    device int8_t *output [[buffer(1)]],
    device float *row_scales [[buffer(2)]],
    device atomic_uint *diagnostics [[buffer(3)]],
    constant DenseW8A8QuantizeParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  if (p.rows != 2048 || p.input_size != 320 || group.x >= p.rows || group.y || group.z ||
      threads.x != 256 || threads.y != 1 || threads.z != 1 || simd_width != 32) {
    if (!tid) dense_w8a8_error(diagnostics, 2u);
    return;
  }
  threadgroup float maxima[8];
  const ulong origin = ulong(group.x) * p.input_size;
  float local_maximum = 0.0f;
  bool nonfinite = false;
  for (uint k = tid; k < p.input_size; k += 256) {
    const float value = float(input[origin + k]);
    if (isfinite(value)) local_maximum = max(local_maximum, abs(value));
    else nonfinite = true;
  }
  if (nonfinite) dense_w8a8_error(diagnostics, 4u);
  const float simd_maximum = simd_max(local_maximum);
  if (!(tid & 31u)) maxima[tid / 32] = simd_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maximum = 0.0f;
#pragma unroll
  for (ushort i = 0; i < 8; ++i) maximum = max(maximum, maxima[i]);
  float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
  if (!(scale > 0.0f) || !isfinite(scale)) {
    if (!tid) dense_w8a8_error(diagnostics, 4u);
    scale = 1.0f;
  }
  if (!tid) row_scales[group.x] = scale;
  for (uint k = tid; k < p.input_size; k += 256) {
    const float value = float(input[origin + k]);
    const float sanitized = isfinite(value) ? value : 0.0f;
    const float rounded = metal::precise::rint(sanitized / scale);
    output[origin + k] = int8_t(clamp(rounded, -127.0f, 127.0f));
  }
}
#endif
