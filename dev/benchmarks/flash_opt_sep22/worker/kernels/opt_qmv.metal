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
  uint row0;          // first activation/output row of this chunk
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
  // Row blocks along grid y (tall inputs); a single block when tg.y == 0.
  const uint block_rows = p.rows > tg.y * uint(R) ? metal::min(uint(R), p.rows - tg.y * uint(R)) : 0u;
  if (!block_rows) return;
  const uint row_base = p.row0 + tg.y * uint(R);
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
      const uint rr = uint(r) < block_rows ? uint(r) : 0u;
      const device bfloat *xr = x + ulong(row_base + rr) * p.x_stride + k0;
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
      if (r < block_rows && col < p.N) y[ulong(row_base + r) * p.y_stride + col] = bfloat(total);
    }
  } else if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int n = 0; n < RN; ++n) {
        const uint col = n0 + n;
        if (uint(r) < block_rows && col < p.N) y[ulong(row_base + r) * p.y_stride + col] = bfloat(acc[r][n]);
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
  OPT_QMV(B, G, R, 2, 1, 8, 8) OPT_QMV(B, G, R, 2, 1, 16, 8) OPT_QMV(B, G, R, 4, 4, 2, 8)

#define OPT_QMV_ROWS(B, G) \
  OPT_QMV_SHAPES(B, G, 1) OPT_QMV_SHAPES(B, G, 2) OPT_QMV_SHAPES(B, G, 3) OPT_QMV_SHAPES(B, G, 4) \
  OPT_QMV_SHAPES(B, G, 5)

OPT_QMV_ROWS(4, 64)
OPT_QMV_ROWS(5, 64)
OPT_QMV_ROWS(5, 128)
OPT_QMV_ROWS(6, 64)
OPT_QMV_ROWS(6, 128)
OPT_QMV_ROWS(8, 64)
OPT_QMV_ROWS(8, 128)

// ---------------------------------------------------------------------------
// Hyper-connection kernels built on the same multi-row traversal.
// Arithmetic of the epilogues follows flash_hc_fused.metal exactly; only the
// F32 summation order of the projections differs.

struct OptHCParams {
  uint rows;
  uint has_injection;
  uint inj_bits;
  uint inj_group;
  ulong down_w_row_stride;
  ulong down_p_row_stride;
  ulong inj_w_row_stride;
  ulong inj_p_row_stride;
  ulong up_w_row_stride;
  ulong up_p_row_stride;
  uint row0;
  uint reserved;
};

inline bfloat opt_hc_sigmoid_fast(bfloat x) {
  const bfloat e = bfloat(metal::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}
inline bfloat opt_hc_sigmoid_unary(bfloat x) {
  const bfloat e = bfloat(metal::precise::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}
inline uint opt_code_dynamic(const device uchar *row, uint k, uint bits) {
  const uint bit = k * bits, byte = bit >> 3, shift = bit & 7;
  uint word = uint(row[byte]);
  if (shift + bits > 8) word |= uint(row[byte + 1]) << 8;
  return (word >> shift) & ((1u << bits) - 1u);
}

template <int BITS, int R>
inline void opt_hc_inject_accumulate(const device bfloat *x, const device uchar *iw,
    const device bfloat *is, const device bfloat *ib, constant OptHCParams &p,
    uint simd, uint lane, thread float (&acc)[R][4]) {
  constexpr int VPT = 8, SK = 16;
  constexpr uint K = 10240;
  for (uint step = simd; step < K / (32 * VPT); step += SK) {
    const uint k0 = step * 32 * VPT + lane * VPT;
    float xv[R][VPT];
    float xs[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      xs[r] = 0.0f;
      const uint rr = uint(r) < p.rows ? uint(r) : 0u;
      const device bfloat *xr = x + ulong(p.row0 + rr) * K + k0;
#pragma unroll
      for (int i = 0; i < VPT; ++i) { xv[r][i] = float(xr[i]); xs[r] += xv[r][i]; }
    }
    const uint g = k0 / p.inj_group;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      float q[VPT];
      opt_codes<BITS, VPT>(iw + ulong(n) * p.inj_w_row_stride, k0, q);
      const ulong pi = ulong(n) * (p.inj_p_row_stride / 2) + g;
      const float s = float(is[pi]), b = float(ib[pi]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < VPT; ++i) d = fma(xv[r][i], q[i], d);
        acc[r][n] = fma(s, d, fma(b, xs[r], acc[r][n]));
      }
    }
  }
}

// normalized [rows, 10240] -> activated [rows, 320] (SiLU(raw/4)) and, when
// present, gates [rows, 4] = 2*sigmoid(raw/4) from block_inject_weight.
// Threadgroups [0, 160) cover 320 down outputs (2 each); threadgroup 160
// computes the four injection outputs with a runtime-format traversal.
template <int BITS, int G, int R>
[[kernel]] void opt_hc_down(
    const device bfloat *x [[buffer(0)]],
    const device uchar *dw [[buffer(1)]], const device bfloat *ds [[buffer(2)]],
    const device bfloat *db [[buffer(3)]],
    const device uchar *iw [[buffer(4)]], const device bfloat *is [[buffer(5)]],
    const device bfloat *ib [[buffer(6)]],
    device bfloat *activated [[buffer(7)]], device bfloat *gates [[buffer(8)]],
    constant OptHCParams &p [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int RN = 2, SK = 16, VPT = 8;
  constexpr uint K = 10240;
  threadgroup float partials[SK * R * 4];
  const bool inj = tg.x == 160;
  if (inj && !p.has_injection) return;
  float acc[R][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < 4; ++n) acc[r][n] = 0.0f;
  if (!inj) {
    const uint n0 = tg.x * RN;
    for (uint step = simd; step < K / (32 * VPT); step += SK) {
      const uint k0 = step * 32 * VPT + lane * VPT;
      float xv[R][VPT];
      float xs[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        xs[r] = 0.0f;
        const uint rr = uint(r) < p.rows ? uint(r) : 0u;
        const device bfloat *xr = x + ulong(p.row0 + rr) * K + k0;
#pragma unroll
        for (int i = 0; i < VPT; ++i) { xv[r][i] = float(xr[i]); xs[r] += xv[r][i]; }
      }
      const uint g = k0 / G;
#pragma unroll
      for (int n = 0; n < RN; ++n) {
        const uint col = n0 + n;
        float q[VPT];
        opt_codes<BITS, VPT>(dw + ulong(col) * p.down_w_row_stride, k0, q);
        const ulong pi = ulong(col) * (p.down_p_row_stride / 2) + g;
        const float s = float(ds[pi]), b = float(db[pi]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          float d = 0.0f;
#pragma unroll
          for (int i = 0; i < VPT; ++i) d = fma(xv[r][i], q[i], d);
          acc[r][n] = fma(s, d, fma(b, xs[r], acc[r][n]));
        }
      }
    }
  } else {
    // Four injection outputs with the same vectorized traversal; the
    // injection matrix may use a different bit width than the down matrix.
    switch (p.inj_bits) {
    case 4: opt_hc_inject_accumulate<4, R>(x, iw, is, ib, p, simd, lane, acc); break;
    case 5: opt_hc_inject_accumulate<5, R>(x, iw, is, ib, p, simd, lane, acc); break;
    case 6: opt_hc_inject_accumulate<6, R>(x, iw, is, ib, p, simd, lane, acc); break;
    default: opt_hc_inject_accumulate<8, R>(x, iw, is, ib, p, simd, lane, acc); break;
    }
  }
  const int outs = inj ? 4 : RN;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < 4; ++n) acc[r][n] = simd_sum(acc[r][n]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int n = 0; n < 4; ++n) partials[(simd * R + r) * 4 + n] = acc[r][n];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd != 0 || lane >= uint(R * outs)) return;
  const uint r = lane / outs, n = lane % outs;
  if (r >= p.rows) return;
  float total = 0.0f;
  for (uint s = 0; s < uint(SK); ++s) total += partials[(s * R + r) * 4 + n];
  const bfloat raw = bfloat(total);
  const bfloat divided = bfloat(float(raw) / 4.0f);
  if (!inj) {
    const bfloat sigmoid = opt_hc_sigmoid_fast(divided);
    activated[ulong(p.row0 + r) * 320 + tg.x * RN + n] = bfloat(float(divided) * float(sigmoid));
  } else {
    const bfloat sigmoid = opt_hc_sigmoid_unary(divided);
    gates[ulong(p.row0 + r) * 4 + n] = bfloat(2.0f * float(sigmoid));
  }
}

// activated [rows, 320] -> up [rows, 4, 2560] (never stored) -> mixed
// [rows, 2560] = sum_s sigmoid(up_s) * normalized_s / 4. One simdgroup per h.
template <int BITS, int G, int R>
[[kernel]] void opt_hc_up_mix(
    const device bfloat *normalized [[buffer(0)]],
    const device bfloat *activated [[buffer(1)]],
    const device uchar *uw [[buffer(2)]], const device bfloat *us [[buffer(3)]],
    const device bfloat *ub [[buffer(4)]], device bfloat *mixed [[buffer(5)]],
    constant OptHCParams &p [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int VPT = 8, SN = 4;
  constexpr uint K = 320, W = 2560;
  const uint h = tg.x * SN + simd;
  float acc[R][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < 4; ++n) acc[r][n] = 0.0f;
  for (uint k0 = lane * VPT; k0 < K; k0 += 32 * VPT) {
    float xv[R][VPT];
    float xs[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      xs[r] = 0.0f;
      const uint rr = uint(r) < p.rows ? uint(r) : 0u;
      const device bfloat *xr = activated + ulong(p.row0 + rr) * K + k0;
#pragma unroll
      for (int i = 0; i < VPT; ++i) { xv[r][i] = float(xr[i]); xs[r] += xv[r][i]; }
    }
    const uint g = k0 / G;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint col = s * W + h;
      float q[VPT];
      opt_codes<BITS, VPT>(uw + ulong(col) * p.up_w_row_stride, k0, q);
      const ulong pi = ulong(col) * (p.up_p_row_stride / 2) + g;
      const float sc = float(us[pi]), bi = float(ub[pi]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < VPT; ++i) d = fma(xv[r][i], q[i], d);
        acc[r][s] = fma(sc, d, fma(bi, xs[r], acc[r][s]));
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int s = 0; s < 4; ++s) acc[r][s] = simd_sum(acc[r][s]);
  if (lane >= uint(R) || lane >= p.rows) return;
  const uint r = lane;
  bfloat total = bfloat(0.0f);
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    // Select this lane's row from the (uniform) register array.
    float v = acc[0][s];
#pragma unroll
    for (int rr = 1; rr < R; ++rr) if (uint(rr) == r) v = acc[rr][s];
    const bfloat raw = bfloat(v);
    const bfloat gate = opt_hc_sigmoid_unary(raw);
    const bfloat product = bfloat(float(gate) * float(normalized[(ulong(p.row0 + r) * 4 + s) * W + h]));
    total = bfloat(float(product) + float(total));
  }
  mixed[ulong(p.row0 + r) * W + h] = bfloat(float(total) / 4.0f);
}

#define OPT_HC(B, G, R) \
  template [[host_name("opt_hc_down_b" #B "_g" #G "_r" #R)]] [[kernel]] void opt_hc_down<B, G, R>( \
      const device bfloat *, const device uchar *, const device bfloat *, const device bfloat *, \
      const device uchar *, const device bfloat *, const device bfloat *, device bfloat *, \
      device bfloat *, constant OptHCParams &, uint3, uint, uint); \
  template [[host_name("opt_hc_up_mix_b" #B "_g" #G "_r" #R)]] [[kernel]] void opt_hc_up_mix<B, G, R>( \
      const device bfloat *, const device bfloat *, const device uchar *, const device bfloat *, \
      const device bfloat *, device bfloat *, constant OptHCParams &, uint3, uint, uint);
#define OPT_HC_ROWS(B, G) OPT_HC(B, G, 1) OPT_HC(B, G, 2) OPT_HC(B, G, 3) OPT_HC(B, G, 4) OPT_HC(B, G, 5)
OPT_HC_ROWS(4, 64)
OPT_HC_ROWS(5, 64)
OPT_HC_ROWS(6, 64)
OPT_HC_ROWS(8, 64)

// ---------------------------------------------------------------------------
// Multi-row dense BF16 GEMV (row-major W[N, K]) for small row counts.
template <int R, int RN, int SK>
[[kernel]] void opt_dense_bf16(
    const device bfloat *x [[buffer(0)]],
    const device bfloat *w [[buffer(1)]],
    device bfloat *y [[buffer(2)]],
    constant OptQmvParams &p [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int VPT = 8;
  threadgroup float partials[SK * R * RN];
  const uint n0 = tg.x * RN;
  float acc[R][RN];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < RN; ++n) acc[r][n] = 0.0f;
  for (uint k0 = (simd * 32 + lane) * VPT; k0 < p.K; k0 += SK * 32 * VPT) {
    float xv[R][VPT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint rr = uint(r) < p.rows ? uint(r) : 0u;
      const device bfloat4 *xr = reinterpret_cast<const device bfloat4 *>(x + ulong(p.row0 + rr) * p.x_stride + k0);
      const bfloat4 a = xr[0], b = xr[1];
      xv[r][0] = a.x; xv[r][1] = a.y; xv[r][2] = a.z; xv[r][3] = a.w;
      xv[r][4] = b.x; xv[r][5] = b.y; xv[r][6] = b.z; xv[r][7] = b.w;
    }
#pragma unroll
    for (int n = 0; n < RN; ++n) {
      const uint col = n0 + n;
      if (col >= p.N) break;
      const device bfloat4 *wr = reinterpret_cast<const device bfloat4 *>(w + ulong(col) * p.K + k0);
      const bfloat4 a = wr[0], b = wr[1];
      const float q[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < VPT; ++i) acc[r][n] = fma(xv[r][i], q[i], acc[r][n]);
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < RN; ++n) acc[r][n] = simd_sum(acc[r][n]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int n = 0; n < RN; ++n) partials[(simd * R + r) * RN + n] = acc[r][n];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd != 0 || lane >= uint(R * RN)) return;
  const uint r = lane / RN, n = lane % RN;
  if (r >= p.rows || n0 + n >= p.N) return;
  float total = 0.0f;
  for (uint s = 0; s < uint(SK); ++s) total += partials[(s * R + r) * RN + n];
  y[ulong(p.row0 + r) * p.y_stride + n0 + n] = bfloat(total);
}
#define OPT_DENSE(R) \
  template [[host_name("opt_dense_bf16_r" #R)]] [[kernel]] void opt_dense_bf16<R, 4, 4>( \
      const device bfloat *, const device bfloat *, device bfloat *, constant OptQmvParams &, \
      uint3, uint, uint);
OPT_DENSE(1) OPT_DENSE(2) OPT_DENSE(3) OPT_DENSE(4) OPT_DENSE(5)

// ---------------------------------------------------------------------------
// Gathered MoE expert projections on the original affine expert weights.
// One threadgroup column per (row, selection); tg.y indexes the selection.
struct OptMoEParams {
  uint K;
  uint N;
  uint rows;
  uint selections;
  uint experts;
  uint reserved;
  ulong w_row_stride;
  ulong w_expert_stride;
  ulong p_row_stride;
  ulong p_expert_stride;
};

inline bfloat opt_moe_sigmoid(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

// MODE 1: out[sel, n] = silu(x_row . gate_n) * (x_row . up_n), x_row = x[sel / selections]
// MODE 0: out[sel, n] = x[sel] . w_n
template <int BITS, int G, int RN, int SN, int VPT, int MODE>
[[kernel]] void opt_moe(
    const device bfloat *x [[buffer(0)]],
    const device uchar *w0 [[buffer(1)]], const device bfloat *s0 [[buffer(2)]],
    const device bfloat *b0 [[buffer(3)]],
    const device uchar *w1 [[buffer(4)]], const device bfloat *s1 [[buffer(5)]],
    const device bfloat *b1 [[buffer(6)]],
    const device long *ids [[buffer(7)]],
    device bfloat *out [[buffer(8)]],
    constant OptMoEParams &p [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint sel = tg.y;
  if (sel >= p.rows * p.selections) return;
  const long expert = ids[sel];
  if (expert < 0 || uint(expert) >= p.experts) return;
  const uint n0 = (tg.x * SN + simd) * RN;
  const device bfloat *xr = x + ulong(MODE == 1 ? sel / p.selections : sel) * p.K;
  const ulong we = ulong(expert) * p.w_expert_stride, pe = ulong(expert) * (p.p_expert_stride / 2);
  const ulong prow = p.p_row_stride / 2;
  float acc0[RN], acc1[RN];
#pragma unroll
  for (int n = 0; n < RN; ++n) { acc0[n] = 0.0f; acc1[n] = 0.0f; }
  for (uint k0 = lane * VPT; k0 < p.K; k0 += 32 * VPT) {
    float xv[VPT];
    float xs = 0.0f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) { xv[i] = float(xr[k0 + i]); xs += xv[i]; }
    const uint g = k0 / G;
#pragma unroll
    for (int n = 0; n < RN; ++n) {
      const uint col = n0 + n;
      if (col >= p.N) break;
      {
        float q[VPT];
        opt_codes<BITS, VPT>(w0 + we + ulong(col) * p.w_row_stride, k0, q);
        const ulong pi = pe + ulong(col) * prow + g;
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < VPT; ++i) d = fma(xv[i], q[i], d);
        acc0[n] = fma(float(s0[pi]), d, fma(float(b0[pi]), xs, acc0[n]));
      }
      if (MODE == 1) {
        float q[VPT];
        opt_codes<BITS, VPT>(w1 + we + ulong(col) * p.w_row_stride, k0, q);
        const ulong pi = pe + ulong(col) * prow + g;
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < VPT; ++i) d = fma(xv[i], q[i], d);
        acc1[n] = fma(float(s1[pi]), d, fma(float(b1[pi]), xs, acc1[n]));
      }
    }
  }
#pragma unroll
  for (int n = 0; n < RN; ++n) {
    acc0[n] = simd_sum(acc0[n]);
    if (MODE == 1) acc1[n] = simd_sum(acc1[n]);
  }
  if (lane != 0) return;
#pragma unroll
  for (int n = 0; n < RN; ++n) {
    const uint col = n0 + n;
    if (col >= p.N) break;
    bfloat value;
    if (MODE == 1) {
      const bfloat gate = bfloat(acc0[n]), up = bfloat(acc1[n]);
      const bfloat silu = bfloat(float(gate) * float(opt_moe_sigmoid(gate)));
      value = bfloat(float(silu) * float(up));
    } else {
      value = bfloat(acc0[n]);
    }
    out[ulong(sel) * p.N + col] = value;
  }
}
#define OPT_MOE(B, G, RN, SN, V, M, NAME) \
  template [[host_name(NAME)]] [[kernel]] void opt_moe<B, G, RN, SN, V, M>( \
      const device bfloat *, const device uchar *, const device bfloat *, const device bfloat *, \
      const device uchar *, const device bfloat *, const device bfloat *, const device long *, \
      device bfloat *, constant OptMoEParams &, uint3, uint, uint);
OPT_MOE(4, 64, 4, 2, 16, 1, "opt_moe_gate_up_b4_g64")
OPT_MOE(4, 64, 4, 2, 8, 0, "opt_moe_down_b4_g64")

// ---------------------------------------------------------------------------
// MoE top-k routing, one simdgroup per row (512 experts, 16 per lane).
// Semantics follow flash_moe_route: softmax probabilities rounded to BF16,
// top-k by probability with ascending-ID ties, BF16 running selected sum and
// optional renormalization. The softmax denominator is a parallel F32 sum.
struct OptRouteParams {
  uint rows;
  uint experts;
  uint selections;
  uint normalize_top_k;
};

[[kernel]] void opt_moe_route(
    const device bfloat *logits [[buffer(0)]], device long *expert_ids [[buffer(1)]],
    device bfloat *weights [[buffer(2)]], constant OptRouteParams &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (row >= p.rows) return;
  float v[16];
  float peak = -INFINITY;
#pragma unroll
  for (uint m = 0; m < 16; ++m) {
    const uint e = lane + m * 32;
    v[m] = e < p.experts ? float(logits[ulong(row) * p.experts + e]) : -INFINITY;
    peak = metal::max(peak, v[m]);
  }
  peak = simd_max(peak);
  float s = 0.0f;
#pragma unroll
  for (uint m = 0; m < 16; ++m) {
    const uint e = lane + m * 32;
    v[m] = e < p.experts ? metal::exp(v[m] - peak) : 0.0f;
    s += v[m];
  }
  const float total = simd_sum(s);
#pragma unroll
  for (uint m = 0; m < 16; ++m) {
    const uint e = lane + m * 32;
    v[m] = e < p.experts ? float(bfloat(v[m] / total)) : -1.0f;
  }
  long selected[10];
  float scores[10];
  float selected_sum = 0.0f;
  for (uint slot = 0; slot < p.selections; ++slot) {
    float best = -1.0f;
    uint best_id = UINT_MAX;
#pragma unroll
    for (uint m = 0; m < 16; ++m) {
      const uint e = lane + m * 32;
      if (v[m] > best) { best = v[m]; best_id = e; }
    }
    const float winner = simd_max(best);
    const uint winner_id = simd_min(best == winner ? best_id : UINT_MAX);
#pragma unroll
    for (uint m = 0; m < 16; ++m)
      if (lane + m * 32 == winner_id) v[m] = -1.0f;
    selected[slot] = long(winner_id);
    scores[slot] = winner;
    selected_sum = float(bfloat(selected_sum) + bfloat(winner));
  }
  if (lane != 0) return;
  const float denominator = float(bfloat(selected_sum));
  for (uint slot = 0; slot < p.selections; ++slot) {
    const ulong index = ulong(row) * p.selections + slot;
    expert_ids[index] = selected[slot];
    weights[index] = bfloat(p.normalize_top_k ? scores[slot] / denominator : scores[slot]);
  }
}

// ---------------------------------------------------------------------------
// Parallel replacements for the single-thread MoE bucket scans. One simdgroup
// scans 512 experts (16 consecutive per lane). Outputs and the invalid-input
// behaviour (sticky diagnostic 2, zeroed offsets) match the serial kernels.
struct OptBucketParams {
  uint rows, selections, width, experts, routes, tile_rows, job_capacity, reserved;
};

[[kernel]] void opt_moe_bucket_prefix(
    const device uint *counts [[buffer(0)]], device uint *offsets [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]], constant OptBucketParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint local[16];
  uint sum = 0;
#pragma unroll
  for (uint i = 0; i < 16; ++i) { local[i] = counts[lane * 16 + i]; sum += local[i]; }
  const uint before = simd_prefix_exclusive_sum(sum);
  const uint total = simd_sum(sum);
  if (total > p.routes || p.experts != 512) {
    if (lane == 0) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    for (uint i = lane; i <= 512; i += 32) offsets[i] = 0;
    return;
  }
  uint running = before;
  if (lane == 0) offsets[0] = 0;
#pragma unroll
  for (uint i = 0; i < 16; ++i) { running += local[i]; offsets[lane * 16 + i + 1] = running; }
}

[[kernel]] void opt_moe_bucket_job_prefix(
    const device uint *counts [[buffer(0)]], const device uint *offsets [[buffer(1)]],
    device uint *job_offsets [[buffer(2)]], device uint *job_count [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]], constant OptBucketParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint jobs[16];
  uint sum = 0;
  bool invalid = p.experts != 512 || offsets[0] != 0;
#pragma unroll
  for (uint i = 0; i < 16; ++i) {
    const uint e = lane * 16 + i;
    const uint count = counts[e], begin = offsets[e], end = offsets[e + 1];
    if (begin > p.routes || end > p.routes || end < begin || end - begin != count) invalid = true;
    jobs[i] = count <= p.routes ? (count + p.tile_rows - 1) / p.tile_rows : p.job_capacity + 1;
    sum += jobs[i];
  }
  const uint before = simd_prefix_exclusive_sum(sum);
  const uint total = simd_sum(sum);
  if (simd_any(invalid) || total > p.job_capacity) {
    if (lane == 0) { job_count[0] = 0; atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed); }
    for (uint i = lane; i <= 512; i += 32) job_offsets[i] = 0;
    return;
  }
  uint running = before;
  if (lane == 0) job_offsets[0] = 0;
#pragma unroll
  for (uint i = 0; i < 16; ++i) { running += jobs[i]; job_offsets[lane * 16 + i + 1] = running; }
  if (lane == 31) job_count[0] = total;
}

// Draft chain token feed: tokens[0] = greedy[row].token (0 when the greedy row
// reported an error; the host rejects that row before using any later step).
struct OptGreedyRow { uint token, rank, errors, reserved; };
kernel void opt_greedy_token_feed(const device OptGreedyRow *results [[buffer(0)]],
                                  device long *tokens [[buffer(1)]],
                                  constant uint2 &p [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  const OptGreedyRow r = results[p.x];
  tokens[0] = long(r.errors == 0 && r.token < p.y ? r.token : 0u);
}

// ---------------------------------------------------------------------------
// Shared-expert SwiGLU for few-row windows: gate and up projections of the
// same input in one dispatch, then silu(gate) * up with the BF16 boundaries of
// flash_moe_silu_multiply. Each projection value is summed exactly as the
// opt_qmv variant with the same template arguments would.
inline bfloat opt_swiglu_sigmoid(bfloat source) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
inline bfloat opt_swiglu(bfloat gate, bfloat up, device atomic_uint *diagnostics) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const float gate_value = float(gate), up_value = float(up);
  const bfloat sigmoid = opt_swiglu_sigmoid(gate);
  const bfloat silu = gate * sigmoid;
  const bfloat result = silu * up;
  if (!metal::isfinite(gate_value) || !metal::isfinite(up_value) || !metal::isfinite(float(result))) {
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
    return bfloat(as_type<float>(0x7fc00000u));
  }
  return result;
}

template <int BITS, int G, int R, int RN, int SN, int SK, int VPT>
[[kernel]] void opt_qmv_swiglu(
    const device bfloat *x [[buffer(0)]],
    const device uchar *gw [[buffer(1)]], const device bfloat *gs [[buffer(2)]],
    const device bfloat *gb [[buffer(3)]],
    const device uchar *uw [[buffer(4)]], const device bfloat *us [[buffer(5)]],
    const device bfloat *ub [[buffer(6)]],
    device bfloat *y [[buffer(7)]], device atomic_uint *diagnostics [[buffer(8)]],
    constant OptQmvParams &p [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partials[(SK > 1) ? 2 * SK * SN * R * RN : 1];
  const uint block_rows = p.rows > tg.y * uint(R) ? metal::min(uint(R), p.rows - tg.y * uint(R)) : 0u;
  if (!block_rows) return;
  const uint row_base = p.row0 + tg.y * uint(R);
  const uint sn = simd % SN, sk = simd / SN;
  const uint n0 = (tg.x * SN + sn) * RN;
  float acc[2][R][RN];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int n = 0; n < RN; ++n) acc[m][r][n] = 0.0f;
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
      const uint rr = uint(r) < block_rows ? uint(r) : 0u;
      const device bfloat *xr = x + ulong(row_base + rr) * p.x_stride + k0;
#pragma unroll
      for (int i = 0; i < VPT; ++i) {
        xv[r][i] = float(xr[i]);
        xs[r] += xv[r][i];
      }
    }
    const uint g = k0 / G;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const device uchar *w = m ? uw : gw;
      const device bfloat *scales = m ? us : gs;
      const device bfloat *biases = m ? ub : gb;
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
          acc[m][r][n] = fma(s, d, fma(b, xs[r], acc[m][r][n]));
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int n = 0; n < RN; ++n) acc[m][r][n] = simd_sum(acc[m][r][n]);
  if (SK > 1) {
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
          for (int n = 0; n < RN; ++n)
            partials[(((m * SK + sk) * SN + sn) * R + r) * RN + n] = acc[m][r][n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk != 0) return;
    if (lane < uint(R * RN)) {
      const uint r = lane / RN, n = lane % RN;
      float total[2] = {0.0f, 0.0f};
      for (uint m = 0; m < 2; ++m)
        for (uint s = 0; s < uint(SK); ++s) total[m] += partials[(((m * SK + s) * SN + sn) * R + r) * RN + n];
      const uint col = n0 + n;
      if (r < block_rows && col < p.N)
        y[ulong(row_base + r) * p.y_stride + col] = opt_swiglu(bfloat(total[0]), bfloat(total[1]), diagnostics);
    }
  } else if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int n = 0; n < RN; ++n) {
        const uint col = n0 + n;
        if (uint(r) < block_rows && col < p.N)
          y[ulong(row_base + r) * p.y_stride + col] = opt_swiglu(bfloat(acc[0][r][n]), bfloat(acc[1][r][n]), diagnostics);
      }
  }
}

#define OPT_QMV_SWIGLU(B, G, R, RN, SN, SK, V) \
  template [[host_name("opt_qmv_swiglu_b" #B "_g" #G "_r" #R "_n" #RN "_sn" #SN "_sk" #SK "_v" #V)]] \
  [[kernel]] void opt_qmv_swiglu<B, G, R, RN, SN, SK, V>( \
      const device bfloat *, const device uchar *, const device bfloat *, const device bfloat *, \
      const device uchar *, const device bfloat *, const device bfloat *, device bfloat *, \
      device atomic_uint *, constant OptQmvParams &, uint3, uint, uint);
#define OPT_QMV_SWIGLU_VARIANTS(B, G, R) \
  OPT_QMV_SWIGLU(B, G, R, 4, 1, 4, 8) OPT_QMV_SWIGLU(B, G, R, 2, 1, 16, 8) OPT_QMV_SWIGLU(B, G, R, 2, 1, 8, 8) \
  OPT_QMV_SWIGLU(B, G, R, 4, 1, 4, 16) OPT_QMV_SWIGLU(B, G, R, 4, 1, 8, 16)
#define OPT_QMV_SWIGLU_ROWS(B, G) \
  OPT_QMV_SWIGLU_VARIANTS(B, G, 1) OPT_QMV_SWIGLU_VARIANTS(B, G, 2) OPT_QMV_SWIGLU_VARIANTS(B, G, 3) \
  OPT_QMV_SWIGLU_VARIANTS(B, G, 4) OPT_QMV_SWIGLU_VARIANTS(B, G, 5)
OPT_QMV_SWIGLU_ROWS(8, 64)
OPT_QMV_SWIGLU_ROWS(8, 128)
