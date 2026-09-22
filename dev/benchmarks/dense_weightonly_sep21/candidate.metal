// Private dense-prefill coefficient requantization experiment. Original BF16
// activations are unchanged; coefficient rows are signed I8 with F32 scales.
// This changes the coefficient representation and makes no accuracy/equality
// claim relative to the original dense BF16 control. No activation converter,
// activation scale, padding, or mutation is required.
//
// ABI: BF16 A[R,K][0], signed I8 B[N,K][1], BF16 output[R,N][2],
// raw F32 unscaled dot[R,N][3], F32 coefficient scales[N][4],
// atomic diagnostics[5], FlashDenseCacheParams[6]. Normal entries bind buffer3
// but never read or write it; separate _probe entries additionally store the
// complete raw F32 dot. BF16 output has the same late scale in both versions.
// Whole-K BF16 x I8 MPP uses F32 destination, relaxed_precision=false,
// multiply mode, one F32 coefficient-row scale, and one final BF16 conversion.
// Diagnostics OR bits: 2 invalid geometry/dispatch; 4 invalid scale/nonfinite.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "abi.hpp"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
#include "metal/kernels/common/flash_dense_traversal.h"

inline void dense_weightonly_error(device atomic_uint *diagnostics, uint flag) {
  atomic_fetch_or_explicit(diagnostics, flag, memory_order_relaxed);
}

template <ushort M, ushort N, ushort Groups, bool Probe>
inline void dense_weightonly_tile(device const bfloat *input,
    device const int8_t *weights, device bfloat *output, device float *raw_dot,
    device const float *weight_scales, device atomic_uint *diagnostics,
    constant FlashDenseCacheParams &p, uint3 group, uint3 threads, uint tid) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (p.rows != 2048 || !p.input_size || p.input_size > 32768 ||
      p.input_size % 32 || !p.output_size || p.output_size > 32768 ||
      !p.output_count || p.output_count % N ||
      p.output_begin > p.output_size ||
      p.output_count > p.output_size - p.output_begin ||
      p.tile_rows != M || p.tile_outputs != N || p.reserved > 4 || group.z ||
      !flash_dense_traversal_group_valid(group.xy, p.rows / M,
          p.output_count / N, p.reserved) ||
      threads.x != uint(Groups) * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) dense_weightonly_error(diagnostics, 2u);
    return;
  }
  const uint2 tile = flash_dense_traversal_tile(group.xy, p.reserved);
  if (tile.x >= p.output_count / N || tile.y >= p.rows / M) return;
  const uint row_origin = tile.y * M;
  const uint column_origin = p.output_begin + tile.x * N;
  const int k = int(p.input_size);
  // The MPP tensor view requires a non-const pointer type. Neither tensor
  // operand nor its original storage is a matmul destination or written here.
  auto a = tensor(const_cast<device bfloat *>(input + ulong(row_origin) * p.input_size),
      dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto b = tensor(const_cast<device int8_t *>(weights + ulong(column_origin) * p.input_size),
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
    const uint row = row_origin + index[1];
    const uint column = column_origin + index[0];
    const float weight_scale = weight_scales[column];
    const float scaled = dot[i] * weight_scale;
    const bfloat value = bfloat(scaled);
    if (!(weight_scale > 0.0f) || !isfinite(weight_scale) ||
        !isfinite(dot[i]) || !isfinite(scaled) || !isfinite(float(value))) {
      dense_weightonly_error(diagnostics, 4u);
    }
    const ulong at = ulong(row) * p.output_size + column;
    if constexpr (Probe) raw_dot[at] = dot[i];
    output[at] = value;
  }
}

#define DENSE_WEIGHTONLY_ENTRY(NAME, M, N, GROUPS, PROBE) \
[[max_total_threads_per_threadgroup(GROUPS * 32)]] \
kernel void NAME( \
    device const bfloat *input [[buffer(0)]], \
    device const int8_t *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], device float *raw_dot [[buffer(3)]], \
    device const float *weight_scales [[buffer(4)]], \
    device atomic_uint *diagnostics [[buffer(5)]], \
    constant FlashDenseCacheParams &p [[buffer(6)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  dense_weightonly_tile<M, N, GROUPS, PROBE>(input, weights, output, raw_dot, \
      weight_scales, diagnostics, p, group, threads, tid); \
}

DENSE_WEIGHTONLY_ENTRY(dense_weightonly_m128_n64_sg4, 128, 64, 4, false)
DENSE_WEIGHTONLY_ENTRY(dense_weightonly_m128_n64_sg8, 128, 64, 8, false)
DENSE_WEIGHTONLY_ENTRY(dense_weightonly_m64_n128_sg8, 64, 128, 8, false)
DENSE_WEIGHTONLY_ENTRY(dense_weightonly_m128_n64_sg4_probe, 128, 64, 4, true)
DENSE_WEIGHTONLY_ENTRY(dense_weightonly_m128_n64_sg8_probe, 128, 64, 8, true)
DENSE_WEIGHTONLY_ENTRY(dense_weightonly_m64_n128_sg8_probe, 64, 128, 8, true)
#undef DENSE_WEIGHTONLY_ENTRY
#endif
