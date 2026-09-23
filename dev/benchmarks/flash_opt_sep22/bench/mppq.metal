// Few-row affine-quantized GEMV on the matrix units.
// y[r, n] = sum_g s[n,g] * (x[r, g] . q[n, g]) + sum_g b[n,g] * xsum[r,g]
// The matrix units read the packed codes directly (uint4b_format / uint8_t).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct OptQmvParams {
  uint K, N, rows, x_stride, y_stride, row0;
  ulong w_row_stride, p_row_stride;
};

template <int BITS> struct MppqCode;
template <> struct MppqCode<4> { using T = uint4b_format; static constant constexpr int PER_BYTE = 2; };
template <> struct MppqCode<8> { using T = uint8_t; static constant constexpr int PER_BYTE = 1; };

// V: 0 = scale+bias per group in the epilogue, 1 = bias via a final matmul.
template <int BITS, int G, int M, int NT, int SGM, int SK, int V>
[[kernel]] void mppq(const device bfloat *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
                     const device bfloat *scales [[buffer(2)]], const device bfloat *biases [[buffer(3)]],
                     device bfloat *y [[buffer(4)]], constant OptQmvParams &p [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int MAXG = 160;
  threadgroup float xsum[MAXG * M];            // [g][m]
  threadgroup float red[(SK > 1) ? SK * M * NT : 1];
  const uint ng = p.K / G;
  const uint rows = p.rows;
  const device bfloat *xr = x + ulong(p.row0) * p.x_stride;
  for (uint i = tid; i < ng * M; i += SGM * SK * 32) {
    const uint g = i / M, m = i % M;
    float s = 0.0f;
    if (m < rows) {
      const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(xr + ulong(m) * p.x_stride + g * G);
      for (int j = 0; j < G / 4; ++j) { const float4 v = float4(q[j]); s += v.x + v.y + v.z + v.w; }
    }
    xsum[i] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n0 = tg.x * NT;
  const uint sk = simd / SGM;
  using CT = typename MppqCode<BITS>::T;
  auto a = tensor(const_cast<device bfloat *>(xr), dextents<int, 2>{int(p.K), int(rows)},
                  array<int, 2>{1, int(p.x_stride)});
  tensor<device CT, dextents<int, 2>, tensor_inline> b(
      const_cast<device uchar *>(w + ulong(n0) * p.w_row_stride),
      dextents<int, 2>{int(p.K), int(NT)},
      array<int, 2>{1, int(p.w_row_stride * MppqCode<BITS>::PER_BYTE)});
  constexpr auto desc = matmul2d_descriptor(M, NT, G, false, true, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<SGM>> op;
  auto a0 = a.template slice<G, M>(0, 0);
  auto b0 = b.template slice<G, NT>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  const uint pr = uint(p.p_row_stride / 2);
  if (V == 2) {
    constexpr auto d2 = matmul2d_descriptor(M, NT, G, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d2, execution_simdgroups<SGM>> op2;
    for (uint g = sk; g < ng; g += SK) {
      auto ag = a.template slice<G, M>(g * G, 0);
      auto bg = b.template slice<G, NT>(g * G, 0);
      op2.run(ag, bg, acc);
    }
  } else
  for (uint g = sk; g < ng; g += SK) {
    auto ag = a.template slice<G, M>(g * G, 0);
    auto bg = b.template slice<G, NT>(g * G, 0);
    auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    op.run(ag, bg, t);
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      const uint n = n0 + idx[0], m = idx[1];
      const float s = float(scales[ulong(n) * pr + g]);
      if (V == 0) {
        const float bb = float(biases[ulong(n) * pr + g]);
        acc[i] = fma(t[i], s, fma(xsum[g * M + m], bb, acc[i]));
      } else {
        acc[i] = fma(t[i], s, acc[i]);
      }
    }
  }
  if (V == 1 && sk == 0) {
    // acc += xsum[M, ng] x biases[n0.., ng]^T
    for (uint g = 0; g < ng; ++g) {
#pragma unroll
      for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        const uint n = n0 + idx[0], m = idx[1];
        acc[i] = fma(xsum[g * M + m], float(biases[ulong(n) * pr + g]), acc[i]);
      }
    }
  }
  if (SK > 1) {
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      red[(sk * M + idx[1]) * NT + idx[0]] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < rows * NT; i += SGM * SK * 32) {
      const uint m = i / NT, n = i % NT;
      float t = 0.0f;
      for (uint s = 0; s < SK; ++s) t += red[(s * M + m) * NT + n];
      if (n0 + n < p.N) y[ulong(p.row0 + m) * p.y_stride + n0 + n] = bfloat(t);
    }
  } else {
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      const uint n = n0 + idx[0], m = idx[1];
      if (m < rows && n < p.N) y[ulong(p.row0 + m) * p.y_stride + n] = bfloat(acc[i]);
    }
  }
}

#define MPPQ(B, G, M, NT, SGM, SK, V)                                                         \
  template [[host_name("mppq_b" #B "_g" #G "_m" #M "_nt" #NT "_sgm" #SGM "_sk" #SK "_v" #V)]] \
  [[kernel]] void mppq<B, G, M, NT, SGM, SK, V>(                                             \
      const device bfloat *, const device uchar *, const device bfloat *,                    \
      const device bfloat *, device bfloat *, constant OptQmvParams &, uint3, uint, uint);

#define MPPQ_ALL(B, G)                                                                        \
  MPPQ(B, G, 16, 32, 1, 1, 0) MPPQ(B, G, 16, 64, 1, 1, 0) MPPQ(B, G, 16, 64, 2, 1, 0)       \
  MPPQ(B, G, 16, 64, 4, 1, 0) MPPQ(B, G, 16, 128, 4, 1, 0) MPPQ(B, G, 16, 32, 1, 4, 0)      \
  MPPQ(B, G, 16, 64, 1, 4, 0) MPPQ(B, G, 16, 64, 2, 2, 0) MPPQ(B, G, 16, 32, 1, 8, 0)       \
  MPPQ(B, G, 16, 64, 1, 8, 0) MPPQ(B, G, 16, 64, 1, 4, 1) MPPQ(B, G, 16, 32, 1, 4, 1)

#define MPPQ_MORE(B, G) \
  MPPQ(B, G, 16, 16, 1, 8, 0) MPPQ(B, G, 16, 16, 1, 16, 0) MPPQ(B, G, 16, 32, 1, 16, 0) \
  MPPQ(B, G, 16, 32, 2, 4, 0) MPPQ(B, G, 16, 32, 2, 8, 0) MPPQ(B, G, 16, 16, 1, 4, 0) \
  MPPQ(B, G, 16, 32, 1, 2, 0) MPPQ(B, G, 16, 64, 1, 2, 0)
#define MPPQ_M8(B, G) \
  MPPQ(B, G, 8, 32, 1, 8, 0) MPPQ(B, G, 8, 32, 1, 4, 0) MPPQ(B, G, 8, 16, 1, 8, 0) MPPQ(B, G, 8, 32, 2, 4, 0) \
  MPPQ(B, G, 16, 32, 1, 8, 2) MPPQ(B, G, 8, 32, 1, 8, 2) MPPQ(B, G, 16, 64, 1, 4, 2) MPPQ(B, G, 16, 32, 2, 4, 2)
#define MPPQ_M32(B, G) \
  MPPQ(B, G, 32, 32, 1, 8, 0) MPPQ(B, G, 32, 32, 1, 4, 0) MPPQ(B, G, 32, 32, 2, 4, 0) MPPQ(B, G, 32, 64, 2, 2, 0)
MPPQ_M32(4, 64)
MPPQ_M32(8, 64)
MPPQ_M32(8, 128)
MPPQ_M8(4, 64)
MPPQ_M8(8, 64)
MPPQ_MORE(4, 64)
MPPQ_MORE(8, 64)
MPPQ_MORE(8, 128)
MPPQ_ALL(4, 64)
MPPQ_ALL(8, 64)
MPPQ_ALL(8, 128)
