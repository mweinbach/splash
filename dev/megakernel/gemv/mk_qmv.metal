// Few-row (1..8) affine-quantized GEMV for decode and MTP verification.
// y[r, n] = sum_g s[n,g] * (x[r, g-block] . q[n, g-block]) + b[n,g] * xsum[r,g]
//
// Code layouts (per weight row; row stride = K * BITS / 8 bytes):
//   4-bit, 8-bit: MLX bitstream (unchanged).
//   5-bit, 6-bit: "planar" per quantization group of G codes: G/2 bytes of low
//                 nibbles (MLX 4-bit order) followed by the high bits, G/8
//                 bytes (5-bit: bit j of byte j/8) or G/4 bytes (6-bit: bits
//                 2(j%4)..+1 of byte j/4). Same size as the MLX stream.
#include <metal_stdlib>
using namespace metal;

struct MkQmvParams {
  uint K, N, rows, x_stride, y_stride, row0;
  ulong w_row_stride;  // bytes
  ulong p_row_stride;  // bytes (BF16 scales/biases)
};

// CPL codes of one row starting at code k0 (k0 % CPL == 0) as floats.
template <int BITS, int G, int CPL> struct MkCodes;

template <int G> struct MkCodes<4, G, 8> {
  static inline void load(const device uchar *row, uint k0, thread float *q) {
    const uint w = *reinterpret_cast<const device uint *>(row + k0 / 2);
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = float((w >> (4 * i)) & 15u);
  }
};
template <int G> struct MkCodes<4, G, 16> {
  static inline void load(const device uchar *row, uint k0, thread float *q) {
    const uint2 w = *reinterpret_cast<const device uint2 *>(row + k0 / 2);
#pragma unroll
    for (int i = 0; i < 8; ++i) { q[i] = float((w.x >> (4 * i)) & 15u); q[8 + i] = float((w.y >> (4 * i)) & 15u); }
  }
};
template <int G> struct MkCodes<8, G, 8> {
  static inline void load(const device uchar *row, uint k0, thread float *q) {
    const uint2 w = *reinterpret_cast<const device uint2 *>(row + k0);
#pragma unroll
    for (int i = 0; i < 4; ++i) { q[i] = float((w.x >> (8 * i)) & 255u); q[4 + i] = float((w.y >> (8 * i)) & 255u); }
  }
};
template <int G> struct MkCodes<8, G, 16> {
  static inline void load(const device uchar *row, uint k0, thread float *q) {
    const uint4 w = *reinterpret_cast<const device uint4 *>(row + k0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[i] = float((w.x >> (8 * i)) & 255u); q[4 + i] = float((w.y >> (8 * i)) & 255u);
      q[8 + i] = float((w.z >> (8 * i)) & 255u); q[12 + i] = float((w.w >> (8 * i)) & 255u);
    }
  }
};
template <int G> struct MkCodes<5, G, 8> {
  static inline void load(const device uchar *row, uint k0, thread float *q) {
    const device uchar *grp = row + (k0 / G) * (G * 5 / 8);
    const uint j0 = k0 % G;
    const uint w = *reinterpret_cast<const device uint *>(grp + j0 / 2);
    const uint h = grp[G / 2 + j0 / 8];
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = float(((w >> (4 * i)) & 15u) | (((h >> i) & 1u) << 4));
  }
};
template <int G> struct MkCodes<5, G, 16> {
  static inline void load(const device uchar *row, uint k0, thread float *q) {
    const device uchar *grp = row + (k0 / G) * (G * 5 / 8);
    const uint j0 = k0 % G;
    const uint2 w = *reinterpret_cast<const device uint2 *>(grp + j0 / 2);
    const uint h = *reinterpret_cast<const device ushort *>(grp + G / 2 + j0 / 8);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      q[i] = float(((w.x >> (4 * i)) & 15u) | (((h >> i) & 1u) << 4));
      q[8 + i] = float(((w.y >> (4 * i)) & 15u) | (((h >> (8 + i)) & 1u) << 4));
    }
  }
};
template <int G> struct MkCodes<6, G, 8> {
  static inline void load(const device uchar *row, uint k0, thread float *q) {
    const device uchar *grp = row + (k0 / G) * (G * 6 / 8);
    const uint j0 = k0 % G;
    const uint w = *reinterpret_cast<const device uint *>(grp + j0 / 2);
    const uint h = *reinterpret_cast<const device ushort *>(grp + G / 2 + j0 / 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = float(((w >> (4 * i)) & 15u) | (((h >> (2 * i)) & 3u) << 4));
  }
};
template <int G> struct MkCodes<6, G, 16> {
  static inline void load(const device uchar *row, uint k0, thread float *q) {
    const device uchar *grp = row + (k0 / G) * (G * 6 / 8);
    const uint j0 = k0 % G;
    const uint2 w = *reinterpret_cast<const device uint2 *>(grp + j0 / 2);
    const uint h = *reinterpret_cast<const device uint *>(grp + G / 2 + j0 / 4);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      q[i] = float(((w.x >> (4 * i)) & 15u) | (((h >> (2 * i)) & 3u) << 4));
      q[8 + i] = float(((w.y >> (4 * i)) & 15u) | (((h >> (16 + 2 * i)) & 3u) << 4));
    }
  }
};

template <int CPL>
inline void mk_load_x(const device bfloat *xr, thread float *xv) {
  const device bfloat4 *p = reinterpret_cast<const device bfloat4 *>(xr);
#pragma unroll
  for (int i = 0; i < CPL / 4; ++i) {
    const float4 v = float4(p[i]);
    xv[4 * i] = v.x; xv[4 * i + 1] = v.y; xv[4 * i + 2] = v.z; xv[4 * i + 3] = v.w;
  }
}

// TG = SN * SK simdgroups; simdgroup (sn, sk) computes RN outputs over the
// K steps sk, sk + SK, ... Each step covers 32 * CPL codes of every row.
template <int BITS, int G, int R, int RN, int SN, int SK, int CPL>
[[kernel]] void mk_qmv(
    const device bfloat *x [[buffer(0)]],
    const device uchar *w [[buffer(1)]],
    const device bfloat *scales [[buffer(2)]],
    const device bfloat *biases [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant MkQmvParams &p [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partials[(SK > 1) ? SK * SN * R * RN : 1];
  const uint rows = p.rows;
  const uint sn = simd % SN, sk = simd / SN;
  const uint n0 = (tg * SN + sn) * RN;
  const device bfloat *xb = x + ulong(p.row0) * p.x_stride;
  const uint pr = uint(p.p_row_stride / 2);
  float acc[R][RN];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < RN; ++n) acc[r][n] = 0.0f;
  constexpr uint STEP = 32 * CPL;
  const uint steps = (p.K + STEP - 1) / STEP;
  for (uint step = sk; step < steps; step += SK) {
    const uint k0 = step * STEP + lane * CPL;
    if (k0 >= p.K) break;
    float xv[R][CPL];
    float xs[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint rr = uint(r) < rows ? uint(r) : 0u;
      mk_load_x<CPL>(xb + ulong(rr) * p.x_stride + k0, xv[r]);
      xs[r] = 0.0f;
#pragma unroll
      for (int i = 0; i < CPL; ++i) xs[r] += xv[r][i];
    }
    const uint g = k0 / G;
#pragma unroll
    for (int n = 0; n < RN; ++n) {
      const uint col = n0 + n;
      if (col >= p.N) break;
      float q[CPL];
      MkCodes<BITS, G, CPL>::load(w + ulong(col) * p.w_row_stride, k0, q);
      const float s = float(scales[ulong(col) * pr + g]);
      const float b = float(biases[ulong(col) * pr + g]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < CPL; ++i) d = fma(xv[r][i], q[i], d);
        acc[r][n] = fma(s, d, fma(b, xs[r], acc[r][n]));
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < RN; ++n) acc[r][n] = simd_sum(acc[r][n]);
  if (SK > 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int n = 0; n < RN; ++n) partials[((sk * SN + sn) * R + r) * RN + n] = acc[r][n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk != 0) return;
    if (lane < uint(R * RN)) {
      const uint r = lane / RN, n = lane % RN;
      float total = 0.0f;
      for (uint s = 0; s < uint(SK); ++s) total += partials[((s * SN + sn) * R + r) * RN + n];
      const uint col = n0 + n;
      if (r < rows && col < p.N) y[ulong(p.row0 + r) * p.y_stride + col] = bfloat(total);
    }
  } else if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int n = 0; n < RN; ++n) {
        const uint col = n0 + n;
        if (uint(r) < rows && col < p.N) y[ulong(p.row0 + r) * p.y_stride + col] = bfloat(acc[r][n]);
      }
  }
}

#define MK_QMV(B, G, R, RN, SN, SK, CPL) \
  template [[host_name("mk_qmv_b" #B "_g" #G "_r" #R "_n" #RN "_sn" #SN "_sk" #SK "_c" #CPL)]] \
  [[kernel]] void mk_qmv<B, G, R, RN, SN, SK, CPL>( \
      const device bfloat *, const device uchar *, const device bfloat *, \
      const device bfloat *, device bfloat *, constant MkQmvParams &, uint, uint, uint);

#define MK_QMV_VARIANTS(B, G, R) \
  MK_QMV(B, G, R, 4, 1, 1, 8) MK_QMV(B, G, R, 4, 1, 2, 8) MK_QMV(B, G, R, 4, 1, 4, 8) MK_QMV(B, G, R, 4, 2, 1, 8) \
  MK_QMV(B, G, R, 8, 1, 1, 8) MK_QMV(B, G, R, 8, 1, 2, 8) MK_QMV(B, G, R, 2, 1, 4, 8) MK_QMV(B, G, R, 2, 1, 8, 8) \
  MK_QMV(B, G, R, 4, 1, 1, 16) MK_QMV(B, G, R, 4, 1, 2, 16) MK_QMV(B, G, R, 2, 1, 2, 16) MK_QMV(B, G, R, 2, 1, 4, 16) \
  MK_QMV(B, G, R, 4, 2, 1, 16) MK_QMV(B, G, R, 8, 1, 1, 16) MK_QMV(B, G, R, 4, 1, 8, 8) MK_QMV(B, G, R, 1, 1, 8, 8) \
  MK_QMV(B, G, R, 1, 1, 16, 8) MK_QMV(B, G, R, 2, 1, 16, 8)

#define MK_QMV_ROWS(B, G) MK_QMV_VARIANTS(B, G, 1) MK_QMV_VARIANTS(B, G, 5)

MK_QMV_ROWS(4, 64)
MK_QMV_ROWS(5, 64)
MK_QMV_ROWS(5, 128)
MK_QMV_ROWS(6, 64)
MK_QMV_ROWS(6, 128)
MK_QMV_ROWS(8, 64)
MK_QMV_ROWS(8, 128)
