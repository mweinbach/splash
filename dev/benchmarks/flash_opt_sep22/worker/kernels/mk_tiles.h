#pragma once
// Lane-major tile format and per-block GEMV core shared by the decode
// megakernel phases (mk_dense.metal, mk_hc.metal).
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
// Scales and biases are transposed to [K/G][Npad]; N is padded to a multiple
// of 32 with zero codes and parameters, and padded outputs are not written.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;


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
// lane gets the sum of its destination row ml. XP: device or threadgroup pointer.
template <typename XP>
inline float mk_tile_xsum(XP xr, uint x_stride, uint rows, uint k0, uint lane, uint ml) {
  using V4 = metal::conditional_t<metal::is_same_v<XP, const device bfloat *>, const device bfloat4 *,
                                  const threadgroup bfloat4 *>;
  const uint m = lane >> 2, c = lane & 3;
  float v = 0.0f;
  if (m < rows) {
    const V4 q = reinterpret_cast<V4>(xr + ulong(m) * x_stride + k0 + c * 16);
    const float4 a = float4(q[0]), b = float4(q[1]), d = float4(q[2]), e = float4(q[3]);
    v = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w)) + ((d.x + d.y) + (d.z + d.w)) + ((e.x + e.y) + (e.z + e.w));
  }
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 2);
  return simd_shuffle(v, ushort(ml * 4));
}

// The same for rows ml and ml + 8 (16-row windows).
template <typename XP>
inline float2 mk_tile_xsum16(XP xr, uint x_stride, uint rows, uint k0, uint lane, uint ml) {
  using V4 = metal::conditional_t<metal::is_same_v<XP, const device bfloat *>, const device bfloat4 *,
                                  const threadgroup bfloat4 *>;
  const uint m = lane >> 2, c = lane & 3;
  float2 v = 0.0f;
#pragma unroll
  for (uint h = 0; h < 2; ++h) {
    if (m + 8 * h < rows) {
      const V4 q = reinterpret_cast<V4>(xr + ulong(m + 8 * h) * x_stride + k0 + c * 16);
      const float4 a = float4(q[0]), b = float4(q[1]), d = float4(q[2]), e = float4(q[3]);
      v[h] = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w)) + ((d.x + d.y) + (d.z + d.w)) +
             ((e.x + e.y) + (e.z + e.w));
    }
  }
  v += float2(simd_shuffle_xor(v.x, 1), simd_shuffle_xor(v.y, 1));
  v += float2(simd_shuffle_xor(v.x, 2), simd_shuffle_xor(v.y, 2));
  return float2(simd_shuffle(v.x, ushort(ml * 4)), simd_shuffle(v.y, ushort(ml * 4)));
}


// Accumulates tiles k_first, k_first + k_step, ... < k_end of one 32-column
// block (tiles at wt, parameters sT[g][NP] with biases at + bias_offset),
// keeping D tiles in flight. On return r[i] holds column
// nl + (i & 3) + 16 * ((i >> 2) & 1) of row ml + 8 * (i >> 3); RB = 2 covers
// 16-row windows (rows ml + 8 are the tile's upper half).
template <int BITS, int G, typename XP, int D = 1, int RB = 1>
inline void mk_tile_accumulate(XP xr, uint x_stride, uint K, uint rows, const device uchar *wt,
                               const device bfloat *sT, ulong bias_offset, uint NP, uint n0,
                               uint k_first, uint k_step, uint k_end, uint lane,
                               thread float (&r)[8 * RB], thread uint &nl, thread uint &ml) {
  constexpr int M = 16, NT = 32, KT = 64;
  using TL = MkTile<BITS>;
  using XE = metal::conditional_t<metal::is_same_v<XP, const device bfloat *>, device bfloat *, threadgroup bfloat *>;
  typename TL::Raw ring[D];
#pragma unroll
  for (int d = 0; d < D; ++d) {
    const uint kd = k_first + uint(d) * k_step;
    if (kd < k_end) ring[d] = TL::load(wt + ulong(kd) * TL::BYTES, lane);
  }
  auto a = tensor(const_cast<XE>(xr), dextents<int, 2>{int(K), int(rows)}, array<int, 2>{1, int(x_stride)});
  // Shape-only view used to derive the operand/destination types.
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> bv(
      const_cast<device uchar *>(wt), dextents<int, 2>{KT, NT}, array<int, 2>{1, KT});
  constexpr auto desc = matmul2d_descriptor(M, NT, KT, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto a0 = a.template slice<KT, M>(0, 0);
  auto b0 = bv.template slice<KT, NT>(0, 0);
  using DT = decltype(op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>());
  DT probe = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  const auto id0 = probe.get_multidimensional_index(ushort(0));
  nl = uint(id0[0]);
  ml = uint(id0[1]);
#pragma unroll
  for (int i = 0; i < 8 * RB; ++i) r[i] = 0.0f;
  auto rb = op.template get_right_input_cooperative_tensor<bfloat, uint8_t, float>();
  thread uint *rbw = reinterpret_cast<thread uint *>(&rb[0]);
  for (uint k = k_first; k < k_end;) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      if (k < k_end) {
        float2 xs2 = 0.0f;
        if (RB == 2) xs2 = mk_tile_xsum16(xr, x_stride, rows, k * KT, lane, ml);
        const float xs = RB == 2 ? xs2.x : mk_tile_xsum(xr, x_stride, rows, k * KT, lane, ml);
        auto ag = a.template slice<KT, M>(k * KT, 0);
        uint u[16];
        TL::unpack(ring[d], u);
        const uint kn = k + uint(D) * k_step;
        if (kn < k_end) ring[d] = TL::load(wt + ulong(kn) * TL::BYTES, lane);
#pragma unroll
        for (int i = 0; i < 16; ++i) rbw[i] = u[i];
        DT t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
        op.run(ag, rb, t);
        const uint g = (k * KT) / G;
        const device bfloat *sg = sT + ulong(g) * NP + n0 + nl;
        const device bfloat *bg = sg + bias_offset;
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
        if (RB == 2) {
          const float xu = xs2.y;
          r[8 + 0] = fma(t[4], s0.x, fma(xu, c0.x, r[8 + 0]));
          r[8 + 1] = fma(t[5], s0.y, fma(xu, c0.y, r[8 + 1]));
          r[8 + 2] = fma(t[6], s0.z, fma(xu, c0.z, r[8 + 2]));
          r[8 + 3] = fma(t[7], s0.w, fma(xu, c0.w, r[8 + 3]));
          r[8 + 4] = fma(t[12], s1.x, fma(xu, c1.x, r[8 + 4]));
          r[8 + 5] = fma(t[13], s1.y, fma(xu, c1.y, r[8 + 5]));
          r[8 + 6] = fma(t[14], s1.z, fma(xu, c1.z, r[8 + 6]));
          r[8 + 7] = fma(t[15], s1.w, fma(xu, c1.w, r[8 + 7]));
        }
      }
      k += k_step;
    }
  }
}
