// Multi-row affine-quantized GEMV for MLX bitstream-packed weights.
// y[r, n] = sum_k x[r, k] * (scale[n, k/G] * q[n, k] + bias[n, k/G])
// Each weight byte is loaded once per threadgroup and reused for every row.
#include <metal_stdlib>
using namespace metal;

struct OptQmvParams {
  uint K;             // input features
  uint N;             // output features
  uint rows;          // activation rows actually present (<= R)
  uint x_stride;      // elements between activation rows
  uint y_stride;      // elements between output rows
  uint reserved;
  ulong w_row_stride; // bytes between packed weight rows
  ulong p_row_stride; // bytes between scale/bias rows
};

// Word type used for aligned per-lane code loads of VPT codes.
template <int BITS, int VPT> struct OptCodeWords;
template <> struct OptCodeWords<4, 16> { typedef uint T; static constant constexpr int W = 2; };
template <> struct OptCodeWords<4, 8>  { typedef uint T; static constant constexpr int W = 1; };
template <> struct OptCodeWords<5, 16> { typedef ushort T; static constant constexpr int W = 5; };
template <> struct OptCodeWords<5, 8>  { typedef uchar T; static constant constexpr int W = 5; };
template <> struct OptCodeWords<6, 16> { typedef uint T; static constant constexpr int W = 3; };
template <> struct OptCodeWords<6, 8>  { typedef ushort T; static constant constexpr int W = 3; };
template <> struct OptCodeWords<8, 16> { typedef uint T; static constant constexpr int W = 4; };
template <> struct OptCodeWords<8, 8>  { typedef uint T; static constant constexpr int W = 2; };

template <int BITS, int VPT>
inline void opt_codes(const device uchar *row, uint k0, thread float *q) {
  typedef typename OptCodeWords<BITS, VPT>::T T;
  constexpr int W = OptCodeWords<BITS, VPT>::W;
  constexpr int WB = sizeof(T) * 8;
  const device T *p = reinterpret_cast<const device T *>(row + (k0 * BITS) / 8);
  T u[W];
#pragma unroll
  for (int i = 0; i < W; ++i) u[i] = p[i];
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int pos = j * BITS;
    const int wi = pos / WB;
    const int sh = pos % WB;
    uint v = uint(u[wi]) >> sh;
    if (sh + BITS > WB) v |= uint(u[wi + 1]) << (WB - sh);
    q[j] = float(v & ((1u << BITS) - 1u));
  }
}

// TG = SN*SK simdgroups. sn selects an RN-wide output slice; sk splits K.
template <int BITS, int G, int R, int RN, int SN, int SK, int VPT>
[[kernel]] void opt_qmv(
    const device bfloat *x [[buffer(0)]],
    const device uchar *w [[buffer(1)]],
    const device bfloat *scales [[buffer(2)]],
    const device bfloat *biases [[buffer(3)]],
    device bfloat *y [[buffer(4)]],
    constant OptQmvParams &p [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partials[(SK > 1) ? SK * SN * R * RN : 1];
  const uint sn = simd % SN, sk = simd / SN;
  const uint n0 = (tg.x * SN + sn) * RN;
  float acc[R][RN];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < RN; ++n) acc[r][n] = 0.0f;
  constexpr uint STEP = 32 * VPT;
  const uint steps = (p.K + STEP - 1) / STEP;
  for (uint step = sk; step < steps; step += SK) {
    const uint k0 = step * STEP + lane * VPT;
    if (k0 >= p.K) continue;
    float xv[R][VPT];
    float xs[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      xs[r] = 0.0f;
      const uint rr = uint(r) < p.rows ? uint(r) : 0u;
      const device bfloat *xr = x + ulong(rr) * p.x_stride + k0;
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        xv[r][i] = float(xr[i]);
        xs[r] += xv[r][i];
      }
    }
    const uint g = k0 / G;
#pragma unroll
    for (int n = 0; n < RN; ++n) {
      const uint col = n0 + n;
      if (col >= p.N) break;
      float q[VPT];
      opt_codes<BITS, VPT>(w + ulong(col) * p.w_row_stride, k0, q);
      const ulong pi = ulong(col) * (p.p_row_stride / 2) + g;
      const float s = float(scales[pi]);
      const float b = float(biases[pi]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < VPT; ++i) d = fma(xv[r][i], q[i], d);
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
        for (int n = 0; n < RN; ++n)
          partials[((sk * SN + sn) * R + r) * RN + n] = acc[r][n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk != 0) return;
    if (lane < uint(R * RN)) {
      const uint r = lane / RN, n = lane % RN;
      float total = 0.0f;
      for (uint s = 0; s < uint(SK); ++s) total += partials[((s * SN + sn) * R + r) * RN + n];
      const uint col = n0 + n;
      if (r < p.rows && col < p.N) y[ulong(r) * p.y_stride + col] = bfloat(total);
    }
  } else if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int n = 0; n < RN; ++n) {
        const uint col = n0 + n;
        if (uint(r) < p.rows && col < p.N) y[ulong(r) * p.y_stride + col] = bfloat(acc[r][n]);
      }
  }
}

#define OPT_QMV(B, G, R, RN, SN, SK, V) \
  template [[host_name("opt_qmv_b" #B "_g" #G "_r" #R "_n" #RN "_sn" #SN "_sk" #SK "_v" #V)]] \
  [[kernel]] void opt_qmv<B, G, R, RN, SN, SK, V>( \
      const device bfloat *, const device uchar *, const device bfloat *, \
      const device bfloat *, device bfloat *, constant OptQmvParams &, uint3, uint, uint);

#define OPT_QMV_SHAPES(B, G, R) \
  OPT_QMV(B, G, R, 4, 2, 1, 16) OPT_QMV(B, G, R, 4, 1, 4, 16) OPT_QMV(B, G, R, 4, 1, 8, 16) \
  OPT_QMV(B, G, R, 4, 2, 1, 8)  OPT_QMV(B, G, R, 4, 1, 4, 8)  OPT_QMV(B, G, R, 4, 1, 8, 8) \
  OPT_QMV(B, G, R, 2, 4, 1, 16) OPT_QMV(B, G, R, 8, 2, 1, 16) OPT_QMV(B, G, R, 4, 4, 1, 16) \
  OPT_QMV(B, G, R, 4, 2, 2, 16) OPT_QMV(B, G, R, 4, 1, 16, 8) OPT_QMV(B, G, R, 4, 1, 16, 16) \
  OPT_QMV(B, G, R, 4, 1, 2, 8) OPT_QMV(B, G, R, 4, 1, 2, 16) \
  OPT_QMV(B, G, R, 2, 1, 8, 8) OPT_QMV(B, G, R, 2, 1, 16, 8)

#define OPT_QMV_ROWS(B, G) \
  OPT_QMV_SHAPES(B, G, 1) OPT_QMV_SHAPES(B, G, 2) OPT_QMV_SHAPES(B, G, 3) OPT_QMV_SHAPES(B, G, 4) \
  OPT_QMV_SHAPES(B, G, 5) OPT_QMV_SHAPES(B, G, 8)

OPT_QMV_ROWS(4, 64)
OPT_QMV_ROWS(5, 64)
OPT_QMV_ROWS(5, 128)
OPT_QMV_ROWS(6, 64)
OPT_QMV_ROWS(6, 128)
OPT_QMV_ROWS(8, 64)
OPT_QMV_ROWS(8, 128)
