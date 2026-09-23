// Expert-shaped GEMM throughput: E experts x NT column tiles, one M-row tile
// per expert, K = 2560. Variants differ in operand formats and K structure.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int K = 2560, NOUT = 640, G = 64, NG = K / G;

// Full-K single run: A [E*M, K] (TA), B [E*NOUT, K] (TB), per-column scale.
template <typename TA, typename TB, typename TD, int M, int SG>
kernel void gemm_full(device TA *a [[buffer(0)]], device TB *b [[buffer(1)]],
                      device const float *scale [[buffer(2)]], device bfloat *out [[buffer(3)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
  const uint e = tg.y, n0 = tg.x * 64;
  auto ta = tensor(a + ulong(e) * M * K, dextents<int, 2>{K, M}, array<int, 2>{1, K});
  auto tb = tensor(b + (ulong(e) * NOUT + n0) * K, dextents<int, 2>{K, 64}, array<int, 2>{1, K});
  constexpr auto d = matmul2d_descriptor(M, 64, static_cast<int>(dynamic_extent), false, true, false,
                                         matmul2d_descriptor::mode::multiply);
  matmul2d<d, execution_simdgroups<SG>> op;
  auto acc = op.template get_destination_cooperative_tensor<decltype(ta), decltype(tb), TD>();
  op.run(ta, tb, acc);
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto idx = acc.get_multidimensional_index(i);
    out[(ulong(e) * M + idx[1]) * NOUT + n0 + idx[0]] = bfloat(float(acc[i]) * scale[n0 + idx[0]]);
  }
}

// Per-group runs (K=64) with an affine epilogue; B in 4-bit (packed bytes) or 8-bit.
template <typename TA, typename TB, typename TD, int M, int SG, int PER_BYTE, int KS>
kernel void gemm_group(device TA *a [[buffer(0)]], device uchar *b [[buffer(1)]],
                       device const bfloat *sb [[buffer(2)]], device bfloat *out [[buffer(3)]],
                       device const float *xs [[buffer(4)]],
                       uint3 tg [[threadgroup_position_in_grid]]) {
  const uint e = tg.y, n0 = tg.x * 64;
  auto ta = tensor(a + ulong(e) * M * K, dextents<int, 2>{K, M}, array<int, 2>{1, K});
  using BP = typename metal::conditional<PER_BYTE == 1, device TB *, device uchar *>::type;
  tensor<device TB, dextents<int, 2>, tensor_inline> tb(
      reinterpret_cast<BP>(b + (ulong(e) * NOUT + n0) * (K / PER_BYTE)), dextents<int, 2>{K, 64}, array<int, 2>{1, K});
  constexpr auto d = matmul2d_descriptor(M, 64, KS, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<d, execution_simdgroups<SG>> op;
  auto a0 = ta.template slice<KS, M>(0, 0);
  auto b0 = tb.template slice<KS, 64>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  for (uint g = 0; g < K / KS; ++g) {
    auto ag = ta.template slice<KS, M>(g * KS, 0);
    auto bg = tb.template slice<KS, 64>(g * KS, 0);
    auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), TD>();
    op.run(ag, bg, t);
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      const uint n = n0 + idx[0];
      const float s = float(sb[(ulong(e) * NOUT + n) * NG * 2 + g * 2]);
      const float bb = float(sb[(ulong(e) * NOUT + n) * NG * 2 + g * 2 + 1]);
      acc[i] = fma(float(t[i]), s, fma(xs[(ulong(e) * M + idx[1]) * NG + g], bb, acc[i]));
    }
  }
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto idx = acc.get_multidimensional_index(i);
    out[(ulong(e) * M + idx[1]) * NOUT + n0 + idx[0]] = bfloat(acc[i]);
  }
}

#define FULL(NAME, TA, TB, TD, M, SG) \
  template [[host_name(NAME)]] kernel void gemm_full<TA, TB, TD, M, SG>(device TA *, device TB *, device const float *, device bfloat *, uint3);
#define GRP(NAME, TA, TB, TD, M, SG, PB, KS) \
  template [[host_name(NAME)]] kernel void gemm_group<TA, TB, TD, M, SG, PB, KS>(device TA *, device uchar *, device const bfloat *, device bfloat *, device const float *, uint3);

FULL("full_bf16_bf16_m64_sg4", bfloat, bfloat, float, 64, 4)
FULL("full_bf16_i8_m64_sg4", bfloat, int8_t, float, 64, 4)
FULL("full_i8_i8_m64_sg4", int8_t, int8_t, int, 64, 4)
FULL("full_bf16_i8_m64_sg8", bfloat, int8_t, float, 64, 8)
FULL("full_i8_i8_m64_sg8", int8_t, int8_t, int, 64, 8)
FULL("full_i8_i8_m32_sg4", int8_t, int8_t, int, 32, 4)
FULL("full_bf16_i8_m16_sg2", bfloat, int8_t, float, 16, 2)
FULL("full_i8_i8_m16_sg2", int8_t, int8_t, int, 16, 2)
GRP("grp_bf16_u4_m64_sg4", bfloat, uint4b_format, float, 64, 4, 2, 64)
GRP("grp_u8_u4_m64_sg4", uint8_t, uint4b_format, int, 64, 4, 2, 64)
GRP("grp_i8_i8_m64_sg4", int8_t, int8_t, int, 64, 4, 1, 64)
GRP("grp_u8_u4_m64_sg8", uint8_t, uint4b_format, int, 64, 8, 2, 64)
GRP("grp_u8_u4_m32_sg4", uint8_t, uint4b_format, int, 32, 4, 2, 64)
