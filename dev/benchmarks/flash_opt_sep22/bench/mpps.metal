// Staged few-row GEMV on the matrix units: each threadgroup owns NT output
// columns; W chunks (NT x BK) are dequantized into FP16 threadgroup tiles and
// multiplied against the (<= M) activation rows read from device memory.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct OptQmvParams { uint K, N, rows, x_stride, y_stride, row0; ulong w_row_stride, p_row_stride; };

template <int BITS>
inline void mpps_codes(const device uchar *row, uint k, thread float *q) {
  const device uchar *p = row + (k * BITS) / 8;
  if (BITS == 4) {
    const uint v = *reinterpret_cast<const device uint *>(p);
    for (int j = 0; j < 8; ++j) q[j] = float((v >> (4 * j)) & 15u);
  } else if (BITS == 8) {
    const uint2 v = *reinterpret_cast<const device uint2 *>(p);
    for (int j = 0; j < 4; ++j) { q[j] = float((v.x >> (8 * j)) & 255u); q[4 + j] = float((v.y >> (8 * j)) & 255u); }
  } else {
    ulong v = 0;
    for (int b = 0; b < BITS; ++b) v |= ulong(p[b]) << (8 * b);
    for (int j = 0; j < 8; ++j) q[j] = float(uint(v >> (BITS * j)) & ((1u << BITS) - 1u));
  }
}

// SG simdgroups share one NT x BK tile; SK groups of SG split K.
template <int BITS, int G, int M, int NT, int BK, int SG, int SK>
kernel void mpps(const device bfloat *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
                 const device bfloat *scales [[buffer(2)]], const device bfloat *biases [[buffer(3)]],
                 device bfloat *y [[buffer(4)]], constant OptQmvParams &p [[buffer(5)]],
                 uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int THREADS = SG * 32;
  threadgroup half stage[SK][NT * BK];
  threadgroup float red[(SK > 1) ? SK * M * NT : 1];
  const uint sk = simd / SG, local = tid - sk * THREADS;
  const uint n0 = tg.x * NT;
  const int rows = int(p.rows);
  auto a = tensor(const_cast<device bfloat *>(x + ulong(p.row0) * p.x_stride), dextents<int, 2>{int(p.K), rows},
                  array<int, 2>{1, int(p.x_stride)});
  auto b = tensor(&stage[sk][0], dextents<int, 2>{BK, NT}, array<int, 2>{1, BK});
  auto bt = b.template slice<BK, NT>(0, 0);
  constexpr auto desc = matmul2d_descriptor(M, NT, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto a0 = a.template slice<BK, M>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bt), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  const uint pr = uint(p.p_row_stride / 2);
  const uint chunks = p.K / BK;
  constexpr int ITEMS = NT * (BK / 8);
  for (uint c = sk; c < chunks + (SK - 1) - ((chunks + SK - 1) % SK == 0 ? 0 : 0); c += SK) {
    const bool active = c < chunks;
    if (active) {
      const uint k0 = c * BK;
      for (int item = int(local); item < ITEMS; item += THREADS) {
        const uint n = uint(item) / (BK / 8), kk = (uint(item) % (BK / 8)) * 8;
        threadgroup half4 *dst = reinterpret_cast<threadgroup half4 *>(&stage[sk][n * BK + kk]);
        float q[8];
        mpps_codes<BITS>(w + ulong(n0 + n) * p.w_row_stride, k0 + kk, q);
        const uint pi = (n0 + n) * pr + (k0 + kk) / G;
        const float s = float(scales[pi]), bb = float(biases[pi]);
        dst[0] = half4(fma(float4(q[0], q[1], q[2], q[3]), s, bb));
        dst[1] = half4(fma(float4(q[4], q[5], q[6], q[7]), s, bb));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) { auto ac = a.template slice<BK, M>(c * BK, 0); op.run(ac, bt, acc); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (SK > 1) {
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      red[(sk * M + idx[1]) * NT + idx[0]] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(rows) * NT; i += SG * SK * 32) {
      const uint m = i / NT, n = i % NT;
      float t = 0.0f;
      for (uint s = 0; s < SK; ++s) t += red[(s * M + m) * NT + n];
      y[ulong(p.row0 + m) * p.y_stride + n0 + n] = bfloat(t);
    }
  } else {
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      if (int(idx[1]) < rows) y[ulong(p.row0 + idx[1]) * p.y_stride + n0 + idx[0]] = bfloat(acc[i]);
    }
  }
}
#define MS(B, G, M, NT, BK, SG, SK) template [[host_name("mpps_b" #B "_g" #G "_m" #M "_nt" #NT "_bk" #BK "_sg" #SG "_sk" #SK)]] \
  kernel void mpps<B, G, M, NT, BK, SG, SK>(const device bfloat *, const device uchar *, const device bfloat *, const device bfloat *, device bfloat *, constant OptQmvParams &, uint3, uint, uint);
#define MSALL(B, G) \
  MS(B, G, 16, 32, 128, 1, 1) MS(B, G, 16, 32, 128, 1, 4) MS(B, G, 16, 64, 128, 1, 2) MS(B, G, 16, 64, 128, 2, 2) \
  MS(B, G, 16, 32, 64, 1, 4) MS(B, G, 16, 16, 128, 1, 4) MS(B, G, 8, 32, 128, 1, 4) MS(B, G, 16, 32, 128, 1, 8) \
  MS(B, G, 16, 64, 64, 1, 4) MS(B, G, 16, 32, 256, 1, 2)
MSALL(4, 64)
MSALL(5, 128)
MSALL(6, 64)
MSALL(8, 128)
