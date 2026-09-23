// Small-M (<= 8 rows) affine-quantized matmul on the matrix units.
// Exact integer codes are staged as BF16 (0..255 are exact); scale and bias
// apply in F32 per 64-wide K chunk: acc += s * (x . q) + b * sum(x).
// Each simdgroup owns private staging and a K slice; a threadgroup reduction
// combines the SK slices, so every weight byte is read exactly once.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct OptQmvParams {
  uint K;
  uint N;
  uint rows;
  uint x_stride;
  uint y_stride;
  uint reserved;
  ulong w_row_stride;
  ulong p_row_stride;
};

template <int BITS>
inline void opt_codes8(const device uchar *row, uint k0, thread uint *q) {
  const device uchar *p = row + (k0 * BITS) / 8;
  if (BITS == 4) {
    const uint u = *reinterpret_cast<const device uint *>(p);
#pragma unroll
    for (int j = 0; j < 8; ++j) q[j] = (u >> (4 * j)) & 15u;
  } else if (BITS == 8) {
    const uint2 u = *reinterpret_cast<const device uint2 *>(p);
#pragma unroll
    for (int j = 0; j < 4; ++j) { q[j] = (u.x >> (8 * j)) & 255u; q[j + 4] = (u.y >> (8 * j)) & 255u; }
  } else if (BITS == 6) {
    const ushort u0 = reinterpret_cast<const device ushort *>(p)[0];
    const ushort u1 = reinterpret_cast<const device ushort *>(p)[1];
    const ushort u2 = reinterpret_cast<const device ushort *>(p)[2];
    const uint lo = uint(u0) | (uint(u1) << 16);
    const uint hi = uint(u2);
    q[0] = lo & 63u; q[1] = (lo >> 6) & 63u; q[2] = (lo >> 12) & 63u; q[3] = (lo >> 18) & 63u;
    q[4] = (lo >> 24) & 63u; q[5] = ((lo >> 30) | (hi << 2)) & 63u; q[6] = (hi >> 4) & 63u; q[7] = (hi >> 10) & 63u;
  } else { // 5
    const uint lo = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
    const uint hi = uint(p[4]);
    q[0] = lo & 31u; q[1] = (lo >> 5) & 31u; q[2] = (lo >> 10) & 31u; q[3] = (lo >> 15) & 31u;
    q[4] = (lo >> 20) & 31u; q[5] = (lo >> 25) & 31u; q[6] = ((lo >> 30) | (hi << 2)) & 31u; q[7] = (hi >> 3) & 31u;
  }
}

template <int BITS, int G, int NT, int SK>
[[kernel]] void opt_qmm8(
    const device bfloat *x [[buffer(0)]],
    const device uchar *w [[buffer(1)]],
    const device bfloat *scales [[buffer(2)]],
    const device bfloat *biases [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant OptQmvParams &p [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BK = 64;
  threadgroup bfloat stA[SK][8 * BK];
  threadgroup bfloat stB[SK][NT * BK];
  threadgroup float rsum[SK][8];
  threadgroup float red[SK][8 * NT];
  const uint n0 = tg.x * NT;
  auto a = tensor(&stA[simd][0], dextents<int, 2>{BK, 8}, array<int, 2>{1, BK});
  auto b = tensor(&stB[simd][0], dextents<int, 2>{BK, NT}, array<int, 2>{1, BK});
  constexpr auto desc = matmul2d_descriptor(8, NT, BK, false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto acc = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  const uint chunks = p.K / BK;
  const ulong prow = p.p_row_stride / 2;
  for (uint c = simd; c < chunks; c += SK) {
    const uint k0 = c * BK;
    {  // A: 8 rows x 64, lane -> row lane/4, 16 values
      const uint row = lane / 4, kk = (lane % 4) * 16;
      float s = 0.0f;
      threadgroup bfloat *dst = &stA[simd][row * BK + kk];
      if (row < p.rows) {
        const device bfloat *src = x + ulong(row) * p.x_stride + k0 + kk;
#pragma unroll
        for (int i = 0; i < 16; ++i) { const bfloat v = src[i]; dst[i] = v; s += float(v); }
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) dst[i] = bfloat(0.0f);
      }
      s += simd_shuffle_xor(s, 1);
      s += simd_shuffle_xor(s, 2);
      if (lane % 4 == 0) rsum[simd][row] = s;
    }
    // B: NT columns x 64 codes; 8 lanes per column, 8 codes per lane.
#pragma unroll
    for (int it = 0; it < NT / 4; ++it) {
      const uint col = it * 4 + lane / 8, kk = (lane % 8) * 8;
      threadgroup bfloat *dst = &stB[simd][col * BK + kk];
      if (n0 + col < p.N) {
        uint q[8];
        opt_codes8<BITS>(w + ulong(n0 + col) * p.w_row_stride, k0 + kk, q);
#pragma unroll
        for (int j = 0; j < 8; ++j) dst[j] = bfloat(float(q[j]));
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) dst[j] = bfloat(0.0f);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto part = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(a, b, part);
    const uint g = k0 / G;
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = part.get_multidimensional_index(i);
      const uint n = n0 + idx[0];
      const uint nn = n < p.N ? n : 0;
      const float sc = float(scales[ulong(nn) * prow + g]);
      const float bi = float(biases[ulong(nn) * prow + g]);
      acc[i] = fma(sc, part[i], fma(bi, rsum[simd][idx[1]], acc[i]));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto idx = acc.get_multidimensional_index(i);
    red[simd][idx[1] * NT + idx[0]] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = tid; e < p.rows * NT; e += 32 * SK) {
    const uint m = e / NT, n = e % NT;
    if (n0 + n >= p.N) continue;
    float total = 0.0f;
#pragma unroll
    for (int s = 0; s < SK; ++s) total += red[s][m * NT + n];
    y[ulong(m) * p.y_stride + n0 + n] = bfloat(total);
  }
}

#define OPT_QMM(B, G, NT, SK) \
  template [[host_name("opt_qmm8_b" #B "_g" #G "_nt" #NT "_sk" #SK)]] [[kernel]] void \
  opt_qmm8<B, G, NT, SK>(const device bfloat *, const device uchar *, const device bfloat *, \
      const device bfloat *, device bfloat *, constant OptQmvParams &, uint3, uint, uint, uint);
#define OPT_QMM_ALL(B, G) \
  OPT_QMM(B, G, 16, 2) OPT_QMM(B, G, 16, 4) OPT_QMM(B, G, 16, 8) \
  OPT_QMM(B, G, 32, 2) OPT_QMM(B, G, 32, 4) OPT_QMM(B, G, 32, 8) \
  OPT_QMM(B, G, 64, 2) OPT_QMM(B, G, 64, 4)
OPT_QMM_ALL(4, 64)
OPT_QMM_ALL(5, 64)
OPT_QMM_ALL(5, 128)
OPT_QMM_ALL(6, 64)
OPT_QMM_ALL(6, 128)
OPT_QMM_ALL(8, 64)
OPT_QMM_ALL(8, 128)
