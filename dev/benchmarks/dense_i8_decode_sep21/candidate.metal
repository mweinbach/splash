// Private numerical dense-weight requantization experiment. Original F32
// cached weights are fitted to signed-I8 codes plus F32 row scales by the
// controlled runtime oracle. This changes coefficients; it does not claim
// original F32-cache equality. BF16 input words remain unchanged, with no
// activation quantization, input padding, or mutation.
// ABI: buffer0 BF16 A[R,K], 1 I8 B[N,K] row-major, 2 F32 row scales[N],
// 3 BF16 output[R,N], 4 diagnostics, 5 DenseI8Params. Separate _audit kernels
// append raw scaled F32 output[R,N] at buffer6; timed kernels bind no audit.
// Whole-K device BF16 x I8 MPP accumulates F32, applies each weight scale once
// after the dot, then casts once to BF16. Captured input fixtures must be
// finite. Scale and output finiteness are checked in the producer; there is
// no repeated K-element input scan per output tile. Malformed input coverage
// belongs to separate oracle diagnostics rather than timed preprocessing.
#if __METAL_VERSION__ >= 410
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "dev/benchmarks/dense_i8_decode_sep21/abi.hpp"
#include "metal/kernels/common/flash_affine_mpp_common.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

template <ushort M, ushort N, ushort SG, bool Audit>
inline void dense_i8_decode_sep21_project(
    device const bfloat *input, device const int8_t *weights,
    device const float *scales, device bfloat *output, device uint *diag,
    constant DenseI8Params &p, device float *raw,
    uint3 group, uint3 threads, uint tid) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  if (!p.rows || p.rows > 16 || !p.input_size || p.input_size > 32768 ||
      p.input_size % 32 || !p.output_size || p.output_size > 32768 ||
      p.output_size % N || p.tile_rows != M || p.tile_outputs != N ||
      p.reserved0 || p.reserved1 || p.reserved2 ||
      group.x >= p.output_size / N || group.y >= (p.rows + M - 1) / M ||
      group.z || threads.x != uint(SG) * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) flash_mpp_error(diag, 2u); return;
  }
  const uint row_origin = group.y * M, column_origin = group.x * N;
  const uint valid_rows = min(uint(M), p.rows - row_origin);
  // Logical row extents mask R1/R4 and incomplete R16 tiles directly. No
  // additional input allocation or pad converter is required.
  auto a = tensor(const_cast<device bfloat *>(input + ulong(row_origin) * p.input_size),
      dextents<int, 2>{int(p.input_size), int(valid_rows)},
      array<int, 2>{1, int(p.input_size)});
  auto b = tensor(const_cast<device int8_t *>(weights + ulong(column_origin) * p.input_size),
      dextents<int, 2>{int(p.input_size), N},
      array<int, 2>{1, int(p.input_size)});
  constexpr auto descriptor = matmul2d_descriptor(M, N,
      static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    if (uint(index[1]) >= valid_rows) continue;
    const uint column = column_origin + index[0], row = row_origin + index[1];
    const float scale = scales[column];
    const float result = dot[i] * scale;
    const bfloat value = bfloat(result);
    if (!(scale > 0.0f) || !flash_mpp_finite(scale) ||
        !flash_mpp_finite(result) || !flash_mpp_finite(value))
      flash_mpp_error(diag, 4u);
    const ulong at = ulong(row) * p.output_size + column;
    if constexpr (Audit) raw[at] = result;
    output[at] = value;
  }
}

#define DENSE_I8_COMMON \
    device const bfloat *a [[buffer(0)]], device const int8_t *b [[buffer(1)]], \
    device const float *scales [[buffer(2)]], device bfloat *output [[buffer(3)]], \
    device uint *diag [[buffer(4)]], constant DenseI8Params &p [[buffer(5)]]
#define DENSE_I8_POSITION \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]
#define DENSE_I8_KERNEL(NAME, M, N, SG) \
kernel void NAME(DENSE_I8_COMMON, DENSE_I8_POSITION) { \
  dense_i8_decode_sep21_project<M, N, SG, false>(a, b, scales, output, diag, p, \
      nullptr, group, threads, tid); \
}
#define DENSE_I8_AUDIT_KERNEL(NAME, M, N, SG) \
kernel void NAME(DENSE_I8_COMMON, device float *raw [[buffer(6)]], DENSE_I8_POSITION) { \
  dense_i8_decode_sep21_project<M, N, SG, true>(a, b, scales, output, diag, p, \
      raw, group, threads, tid); \
}

DENSE_I8_KERNEL(dense_i8_decode_sep21_m8_n64_sg4, 8, 64, 4)
DENSE_I8_AUDIT_KERNEL(dense_i8_decode_sep21_m8_n64_sg4_audit, 8, 64, 4)
DENSE_I8_KERNEL(dense_i8_decode_sep21_m8_n128_sg4, 8, 128, 4)
DENSE_I8_AUDIT_KERNEL(dense_i8_decode_sep21_m8_n128_sg4_audit, 8, 128, 4)
DENSE_I8_KERNEL(dense_i8_decode_sep21_m16_n64_sg4, 16, 64, 4)
DENSE_I8_AUDIT_KERNEL(dense_i8_decode_sep21_m16_n64_sg4_audit, 16, 64, 4)
DENSE_I8_KERNEL(dense_i8_decode_sep21_m8_n64_sg2, 8, 64, 2)
DENSE_I8_AUDIT_KERNEL(dense_i8_decode_sep21_m8_n64_sg2_audit, 8, 64, 2)
#undef DENSE_I8_COMMON
#undef DENSE_I8_POSITION
#undef DENSE_I8_KERNEL
#undef DENSE_I8_AUDIT_KERNEL
#endif
