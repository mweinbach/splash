#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "FlashAffineInt8Code.h"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
using namespace metal;
using namespace mpp::tensor_ops;

inline bool int8code_valid(constant FlashAffineInt8CodeParams &p) {
  return p.rows && p.rows <= 16 && p.input_size && p.input_size <= 32768 &&
      p.input_size % 64 == 0 && p.output_size && p.output_size % 64 == 0 &&
      p.group_size == 64 &&
      (p.bits == 4 || p.bits == 5 || p.bits == 6 || p.bits == 8) &&
      (p.tile_rows == 8 || p.tile_rows == 16) &&
      (p.tile_outputs == 64 || p.tile_outputs == 128) &&
      p.padded_rows == (p.rows + p.tile_rows - 1) / p.tile_rows * p.tile_rows &&
      p.padded_rows <= 16 && p.weight_row_stride_bytes >=
          (ulong(p.input_size) * p.bits + 7) / 8 &&
      p.parameter_row_stride_bytes >= ulong(p.input_size / 64) * 2 &&
      p.parameter_row_stride_bytes % 2 == 0;
}

// Expand each original unsigned code once. Every Q4/Q5/Q6/Q8 code remains
// exact UINT8; there is no centering, fitted scale or requantization.
kernel void flash_affine_uint8code_v2_expand(
    device const uchar *packed [[buffer(0)]],
    device uint8_t *codes [[buffer(1)]], device uint *diagnostics [[buffer(2)]],
    constant FlashAffineInt8CodeParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (!int8code_valid(p) || p.rows != 1 || group.y || group.z) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const ulong index = ulong(group.x) * 256 + tid;
  if (index >= ulong(p.output_size) * p.input_size) return;
  const uint n = uint(index / p.input_size), k = uint(index % p.input_size);
  const int q = int(flash_mpp_code(packed + ulong(n) * p.weight_row_stride_bytes,
                                 k, p.bits));
  codes[index] = uint8_t(q);
}

// BF16 activations retain their source words. One SIMD owns each row/G64
// group; its F32 sum is shared by every output tile's affine bias correction.
kernel void flash_affine_uint8code_v2_group_sums(
    device const bfloat *padded [[buffer(0)]],
    device float *sums [[buffer(1)]], device uint *diagnostics [[buffer(2)]],
    constant FlashAffineInt8CodeParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (!int8code_valid(p) || group.x >= p.padded_rows ||
      group.y >= p.input_size / 64) {
    if (lane == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const ulong base = ulong(group.x) * p.input_size + group.y * 64;
  const float sum = simd_sum(float(padded[base + lane * 2]) +
                             float(padded[base + lane * 2 + 1]));
  if (lane == 0) {
    if (!flash_mpp_finite(sum)) flash_mpp_error(diagnostics, 4u);
    sums[ulong(group.x) * (p.input_size / 64) + group.y] = sum;
  }
}

template <ushort M, ushort N>
inline void int8code_tile(
    device bfloat *padded, device uint8_t *codes,
    device const uchar *scaleBytes, device const uchar *biasBytes,
    device const float *sums, device bfloat *output, device uint *diagnostics,
    constant FlashAffineInt8CodeParams &p, uint3 group, uint tid) {
  if (!int8code_valid(p) || p.tile_rows != M || p.tile_outputs != N ||
      p.output_size % N || group.z || group.x >= p.output_size / N ||
      group.y >= p.padded_rows / M) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const uint row = group.y * M, column = group.x * N;
  const int k = int(p.input_size);
  auto a = tensor(padded + ulong(row) * p.input_size,
      dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto b = tensor(codes + ulong(column) * p.input_size,
      dextents<int, 2>{k, N}, array<int, 2>{1, k});
  constexpr auto descriptor = matmul2d_descriptor(
      M, N, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto at = a.template slice<64, M>(0, 0);
  auto bt = b.template slice<64, N>(0, 0);
  auto total = operation.template get_destination_cooperative_tensor<
      decltype(at), decltype(bt), float>();
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i)
    if (total.is_valid_element(i)) total[i] = 0.0f;
  for (uint g = 0; g < p.input_size / 64; ++g) {
    at = a.template slice<64, M>(g * 64, 0);
    bt = b.template slice<64, N>(g * 64, 0);
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(at), decltype(bt), float>();
    operation.run(at, bt, partial);
#pragma unroll
    for (ushort i = 0; i < total.get_capacity(); ++i) {
      if (!total.is_valid_element(i)) continue;
      const auto index = total.get_multidimensional_index(i);
      const uint r = row + index[1], n = column + index[0];
      const ulong offset = ulong(n) * p.parameter_row_stride_bytes + ulong(g) * 2;
      const float scale = float(*reinterpret_cast<device const bfloat *>(scaleBytes + offset));
      const float bias = float(*reinterpret_cast<device const bfloat *>(biasBytes + offset));
      const float scaled = partial[i] * scale;
      const float corrected = scaled + bias * sums[ulong(r) * (p.input_size / 64) + g];
      total[i] += corrected;
    }
  }
#pragma unroll
  for (ushort i = 0; i < total.get_capacity(); ++i) {
    if (!total.is_valid_element(i)) continue;
    const auto index = total.get_multidimensional_index(i);
    const uint r = row + index[1];
    if (r >= p.rows) continue;
    const bfloat result = bfloat(total[i]);
    if (!flash_mpp_finite(total[i]) || !flash_mpp_finite(result))
      flash_mpp_error(diagnostics, 4u);
    output[ulong(r) * p.output_size + column + index[0]] = result;
  }
}
#define INT8CODE_ENTRY(NAME, M, N) \
  kernel void NAME(device bfloat *padded [[buffer(0)]], \
      device uint8_t *codes [[buffer(1)]], \
      device const uchar *scales [[buffer(2)]], \
      device const uchar *biases [[buffer(3)]], \
      device const float *sums [[buffer(4)]], \
      device bfloat *output [[buffer(5)]], device uint *diagnostics [[buffer(6)]], \
      constant FlashAffineInt8CodeParams &p [[buffer(7)]], \
      uint3 group [[threadgroup_position_in_grid]], \
      uint tid [[thread_index_in_threadgroup]]) { \
    int8code_tile<M, N>(padded, codes, scales, biases, sums, output, \
                        diagnostics, p, group, tid); \
  }
INT8CODE_ENTRY(flash_affine_uint8code_v2_m8_n64, 8, 64)
INT8CODE_ENTRY(flash_affine_uint8code_v2_m8_n128, 8, 128)
INT8CODE_ENTRY(flash_affine_uint8code_v2_m16_n64, 16, 64)
INT8CODE_ENTRY(flash_affine_uint8code_v2_m16_n128, 16, 128)
#undef INT8CODE_ENTRY
#endif
