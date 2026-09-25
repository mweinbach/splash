// Few-row (<= 8) quantized GEMV on the matrix units over lane-major tiles.
// y[r, n] = sum_g s[n,g] * (x[r, g-block] . q[n, g-block]) + b[n,g] * xsum[r,g]
//
// Weights are repacked (losslessly) into 32-column x 64-code tiles, ordered by
// column block then K. Inside a tile, lane L's 64 codes are the elements of
// the matmul's uint8 right-operand cooperative tensor in element order:
//   4/5/6-bit: 32 bytes of nibbles at L*32; element e = 8u + 4h + b is nibble h
//              of byte b of word u.
//   5-bit:     + 8 bytes at 1024 + L*8; bit e holds element e's bit 4.
//   6-bit:     + 16 bytes at 1024 + L*16; bits 2e..2e+1 hold element e's bits 4-5.
//   8-bit:     64 bytes at L*64; byte e is element e.
// Scales and biases are transposed to [K/G][N].
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct MkQmvParams {
  uint K, N, rows, x_stride, y_stride, row0;
  ulong w_row_stride;  // unused (tiled layout)
  ulong p_row_stride;  // unused (transposed parameters)
};

constant constexpr uint MK_MAXT = 160;  // 64-code tiles along K (K <= 10240)
constant constexpr uint MK_MAXR = 8;

template <int BITS> struct MkTile;
template <> struct MkTile<4> {
  static constant constexpr uint BYTES = 1024;
  struct Raw { uint4 n0, n1; };
  static inline Raw load(const device uchar *t, uint lane) {
    const device uint4 *p = reinterpret_cast<const device uint4 *>(t + lane * 32);
    return {p[0], p[1]};
  }
  static inline void unpack(const thread Raw &r, thread uint *u) {
    const uint n[8] = {r.n0.x, r.n0.y, r.n0.z, r.n0.w, r.n1.x, r.n1.y, r.n1.z, r.n1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) { u[2 * i] = n[i] & 0x0F0F0F0Fu; u[2 * i + 1] = (n[i] >> 4) & 0x0F0F0F0Fu; }
  }
};
template <> struct MkTile<8> {
  static constant constexpr uint BYTES = 2048;
  struct Raw { uint4 a, b, c, d; };
  static inline Raw load(const device uchar *t, uint lane) {
    const device uint4 *p = reinterpret_cast<const device uint4 *>(t + lane * 64);
    return {p[0], p[1], p[2], p[3]};
  }
  static inline void unpack(const thread Raw &r, thread uint *u) {
    u[0] = r.a.x; u[1] = r.a.y; u[2] = r.a.z; u[3] = r.a.w;
    u[4] = r.b.x; u[5] = r.b.y; u[6] = r.b.z; u[7] = r.b.w;
    u[8] = r.c.x; u[9] = r.c.y; u[10] = r.c.z; u[11] = r.c.w;
    u[12] = r.d.x; u[13] = r.d.y; u[14] = r.d.z; u[15] = r.d.w;
  }
};
template <> struct MkTile<5> {
  static constant constexpr uint BYTES = 1280;
  struct Raw { uint4 n0, n1; uint2 h; };
  static inline Raw load(const device uchar *t, uint lane) {
    const device uint4 *p = reinterpret_cast<const device uint4 *>(t + lane * 32);
    return {p[0], p[1], *reinterpret_cast<const device uint2 *>(t + 1024 + lane * 8)};
  }
  static inline void unpack(const thread Raw &r, thread uint *u) {
    MkTile<4>::unpack(MkTile<4>::Raw{r.n0, r.n1}, u);
#pragma unroll
    for (int w = 0; w < 16; ++w) {
      const uint h4 = ((w < 8 ? r.h.x : r.h.y) >> (4 * (w & 7))) & 15u;
      u[w] |= ((h4 * 0x00204081u) & 0x01010101u) << 4;
    }
  }
};
template <> struct MkTile<6> {
  static constant constexpr uint BYTES = 1536;
  struct Raw { uint4 n0, n1, h; };
  static inline Raw load(const device uchar *t, uint lane) {
    const device uint4 *p = reinterpret_cast<const device uint4 *>(t + lane * 32);
    return {p[0], p[1], *reinterpret_cast<const device uint4 *>(t + 1024 + lane * 16)};
  }
  static inline void unpack(const thread Raw &r, thread uint *u) {
    MkTile<4>::unpack(MkTile<4>::Raw{r.n0, r.n1}, u);
    const uint hw[4] = {r.h.x, r.h.y, r.h.z, r.h.w};
#pragma unroll
    for (int w = 0; w < 16; ++w) {
      const uint h8 = (hw[w >> 2] >> (8 * (w & 3))) & 255u;
      const uint s = (h8 & 3u) | ((h8 & 0x0Cu) << 6) | ((h8 & 0x30u) << 12) | ((h8 & 0xC0u) << 18);
      u[w] |= s << 4;
    }
  }
};

// Row sums of x over one 64-code tile for row m = lane / 4, broadcast so each
// lane gets the sum of its destination row ml.
inline float mk_tile_xsum(const device bfloat *xr, uint x_stride, uint rows, uint k0, uint lane, uint ml) {
  const uint m = lane >> 2, c = lane & 3;
  float v = 0.0f;
  if (m < rows) {
    const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(xr + ulong(m) * x_stride + k0 + c * 16);
    const float4 a = float4(q[0]), b = float4(q[1]), d = float4(q[2]), e = float4(q[3]);
    v = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w)) + ((d.x + d.y) + (d.z + d.w)) + ((e.x + e.y) + (e.z + e.w));
  }
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 2);
  return simd_shuffle(v, ushort(ml * 4));
}

// XS: 0 = per-threadgroup x-sum prologue, 1 = per-tile x-sum in the loop.
template <int BITS, int G, int SK, int XS>
[[kernel]] void mk_mpt(const device bfloat *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
                       const device bfloat *sT [[buffer(2)]], const device bfloat *bT [[buffer(3)]],
                       device bfloat *y [[buffer(4)]], constant MkQmvParams &p [[buffer(5)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  constexpr int M = 16, NT = 32, KT = 64;
  constexpr uint T = SK * 32;
  using TL = MkTile<BITS>;
  threadgroup float xsum[XS == 0 ? MK_MAXT * MK_MAXR : 1];  // [tile][m]
  threadgroup float red[(SK > 1 ? SK : 1) * MK_MAXR * NT];
  const uint nt = p.K / KT, rows = p.rows, N = p.N;
  const uint n0 = tg * NT;
  const device bfloat *xr = x + ulong(p.row0) * p.x_stride;
  const device uchar *wt = w + ulong(tg) * nt * TL::BYTES;
  uint k = simd;
  typename TL::Raw cur;
  if (k < nt) cur = TL::load(wt + ulong(k) * TL::BYTES, lane);

  if (XS == 0) {
    for (uint i = tid; i < nt * MK_MAXR; i += T) {
      const uint kk = i / MK_MAXR, m = i % MK_MAXR;
      float s = 0.0f;
      if (m < rows) {
        const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(xr + ulong(m) * p.x_stride + kk * KT);
        for (int j = 0; j < KT / 4; ++j) { const float4 v = float4(q[j]); s += (v.x + v.y) + (v.z + v.w); }
      }
      xsum[i] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  auto a = tensor(const_cast<device bfloat *>(xr), dextents<int, 2>{int(p.K), int(rows)},
                  array<int, 2>{1, int(p.x_stride)});
  // Shape-only view used to derive the operand/destination types.
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> bv(
      const_cast<device uchar *>(w), dextents<int, 2>{KT, NT}, array<int, 2>{1, KT});
  constexpr auto desc = matmul2d_descriptor(M, NT, KT, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto a0 = a.template slice<KT, M>(0, 0);
  auto b0 = bv.template slice<KT, NT>(0, 0);
  using DT = decltype(op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>());
  DT probe = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  const auto id0 = probe.get_multidimensional_index(ushort(0));
  const uint nl = uint(id0[0]), ml = uint(id0[1]);
  float r[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) r[i] = 0.0f;

  auto rb = op.template get_right_input_cooperative_tensor<bfloat, uint8_t, float>();
  thread uint *rbw = reinterpret_cast<thread uint *>(&rb[0]);

  while (k < nt) {
    const uint k1 = k + SK;
    typename TL::Raw nxt;
    if (k1 < nt) nxt = TL::load(wt + ulong(k1) * TL::BYTES, lane);
    const float xs = XS == 0 ? xsum[k * MK_MAXR + ml] : mk_tile_xsum(xr, p.x_stride, rows, k * KT, lane, ml);
    auto ag = a.template slice<KT, M>(k * KT, 0);
    uint u[16];
    TL::unpack(cur, u);
#pragma unroll
    for (int i = 0; i < 16; ++i) rbw[i] = u[i];
    DT t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    op.run(ag, rb, t);
    const uint g = (k * KT) / G;
    const device bfloat *sg = sT + ulong(g) * N + n0 + nl;
    const device bfloat *bg = bT + ulong(g) * N + n0 + nl;
    const float4 s0 = float4(*reinterpret_cast<const device bfloat4 *>(sg));
    const float4 s1 = float4(*reinterpret_cast<const device bfloat4 *>(sg + 16));
    const float4 c0 = float4(*reinterpret_cast<const device bfloat4 *>(bg));
    const float4 c1 = float4(*reinterpret_cast<const device bfloat4 *>(bg + 16));
    r[0] = fma(t[0], s0.x, fma(xs, c0.x, r[0]));
    r[1] = fma(t[1], s0.y, fma(xs, c0.y, r[1]));
    r[2] = fma(t[2], s0.z, fma(xs, c0.z, r[2]));
    r[3] = fma(t[3], s0.w, fma(xs, c0.w, r[3]));
    r[4] = fma(t[8], s1.x, fma(xs, c1.x, r[4]));
    r[5] = fma(t[9], s1.y, fma(xs, c1.y, r[5]));
    r[6] = fma(t[10], s1.z, fma(xs, c1.z, r[6]));
    r[7] = fma(t[11], s1.w, fma(xs, c1.w, r[7]));
    cur = nxt;
    k = k1;
  }

  if (SK > 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) red[(simd * MK_MAXR + ml) * NT + nl + (i & 3) + 16 * (i >> 2)] = r[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < rows * NT; i += T) {
      const uint m = i / NT, n = i % NT;
      float t = 0.0f;
      for (uint s = 0; s < uint(SK); ++s) t += red[(s * MK_MAXR + m) * NT + n];
      y[ulong(p.row0 + m) * p.y_stride + n0 + n] = bfloat(t);
    }
  } else if (ml < rows) {
#pragma unroll
    for (int i = 0; i < 8; ++i) y[ulong(p.row0 + ml) * p.y_stride + n0 + nl + (i & 3) + 16 * (i >> 2)] = bfloat(r[i]);
  }
}

#define MK_MPT(B, G, SK, XS) \
  template [[host_name("mk_mpt_b" #B "_g" #G "_sk" #SK "_x" #XS)]] \
  [[kernel]] void mk_mpt<B, G, SK, XS>(const device bfloat *, const device uchar *, const device bfloat *, \
      const device bfloat *, device bfloat *, constant MkQmvParams &, uint, uint, uint, uint);
#define MK_MPT_ALL(B, G) MK_MPT(B, G, 1, 0) MK_MPT(B, G, 2, 0) MK_MPT(B, G, 4, 0) MK_MPT(B, G, 8, 0) MK_MPT(B, G, 16, 0) \
  MK_MPT(B, G, 1, 1) MK_MPT(B, G, 2, 1) MK_MPT(B, G, 4, 1) MK_MPT(B, G, 8, 1) MK_MPT(B, G, 16, 1)
MK_MPT_ALL(4, 64)
MK_MPT_ALL(5, 64)
MK_MPT_ALL(5, 128)
MK_MPT_ALL(6, 64)
MK_MPT_ALL(6, 128)
MK_MPT_ALL(8, 64)
MK_MPT_ALL(8, 128)
