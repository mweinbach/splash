// Hyper-connection phases for 2..8-row decode windows on lane-major tiles
// (mk_tiles.h); SPLASH_MK_QMV=1 (FlashForward::hc).
//   mk_hc_down: normalized [rows, 10240] x {down (320), inject (4, padded to 32)}
//               as split-K partial sums partial[split][8][352] (F32).
//   mk_hc_up:   sums the partials, applies SiLU(raw/4) (and writes the gates
//               2*sigmoid(raw/4)), then up (320 -> 4 x 2560) with the sigmoid-
//               gated stream mix into mixed [rows, 2560]. The up codes are
//               tiled with columns interleaved so each 32-column block holds
//               8 positions x 4 streams (column c: stream c % 4, position
//               8 * block + c / 4).
// BF16 epilogue boundaries follow opt_hc_down / opt_hc_up_mix.
#include "mk_tiles.h"

struct MkHCSegment {
  uint bits, group, padded_n, pad0;
  ulong tile_offset;   // bytes
  ulong param_offset;  // BF16 elements
  ulong bias_offset;   // elements from scales to biases
  ulong pad1;
};
struct MkHCDownParams {
  uint rows, splits, blocks, pad0;  // blocks: 10 (down) or 11 (down + inject)
  MkHCSegment down, inject;
};
struct MkHCUpParams {
  uint rows, splits, has_injection, pad0;
  MkHCSegment up;
};

constant constexpr uint MK_HC_K = 10240, MK_HC_TILES = MK_HC_K / 64, MK_HC_WIDTH = 352, MK_HC_SPLITS = 4;

inline bfloat mk_hc_sigmoid_fast(bfloat x) {
  const bfloat e = bfloat(metal::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}
inline bfloat mk_hc_sigmoid_unary(bfloat x) {
  const bfloat e = bfloat(metal::precise::exp(metal::abs(float(x))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat t = bfloat(1.0f) / d;
  return x < bfloat(0.0f) ? t : bfloat(1.0f) - t;
}

// Grid (blocks, splits); SK simdgroups split each split's K range.
template <int SK>
[[kernel]] void mk_hc_down(const device bfloat *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
                           const device bfloat *params [[buffer(2)]], device float *partial [[buffer(3)]],
                           constant MkHCDownParams &p [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[SK * MK_MAXR * 32];
  const uint block = tg.x, split = tg.y;
  const bool inject = block >= 10;
  constant MkHCSegment &seg = inject ? p.inject : p.down;
  const uint local = inject ? 0 : block;
  const uint per = MK_HC_TILES / MK_HC_SPLITS;
  const uint k_first = split * per + simd, k_end = (split + 1) * per;
  float r[8];
  uint nl = 0, ml = 0;
#define MK_HC_SEG(B, GS) \
  mk_tile_accumulate<B, GS, const device bfloat *, 5>(x, MK_HC_K, MK_HC_K, p.rows, \
      w + seg.tile_offset + ulong(local) * MK_HC_TILES * MkTile<B>::BYTES, params + seg.param_offset, \
      seg.bias_offset, seg.padded_n, local * 32, k_first, SK, k_end, lane, r, nl, ml)
  switch (seg.bits * 1000 + seg.group) {
  case 4064: MK_HC_SEG(4, 64); break;
  case 5064: MK_HC_SEG(5, 64); break;
  case 6064: MK_HC_SEG(6, 64); break;
  case 5128: MK_HC_SEG(5, 128); break;
  case 6128: MK_HC_SEG(6, 128); break;
  case 8128: MK_HC_SEG(8, 128); break;
  default:   MK_HC_SEG(8, 64); break;
  }
#undef MK_HC_SEG
#pragma unroll
  for (int i = 0; i < 8; ++i) red[(simd * MK_MAXR + ml) * 32 + nl + (i & 3) + 16 * (i >> 2)] = r[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < p.rows * 32; i += SK * 32) {
    const uint m = i / 32, n = i % 32;
    float t = 0.0f;
    for (uint s = 0; s < uint(SK); ++s) t += red[(s * MK_MAXR + m) * 32 + n];
    partial[(ulong(split) * MK_MAXR + m) * MK_HC_WIDTH + block * 32 + n] = t;
  }
}

// Grid 80 x 128 threads: simdgroup s of threadgroup t owns column block 4t + s.
template <int BITS, int G>
inline void mk_hc_up_block(const threadgroup bfloat *act, uint rows, const device uchar *w,
                           const device bfloat *params, constant MkHCSegment &seg, uint block,
                           const device bfloat *normalized, device bfloat *mixed, uint lane) {
  float r[8];
  uint nl = 0, ml = 0;
  mk_tile_accumulate<BITS, G, const threadgroup bfloat *, 5>(act, 320, 320, rows,
      w + seg.tile_offset + ulong(block) * 5 * MkTile<BITS>::BYTES, params + seg.param_offset, seg.bias_offset,
      seg.padded_n, block * 32, 0, 1, 5, lane, r, nl, ml);
  if (ml >= rows) return;
#pragma unroll
  for (int pos = 0; pos < 2; ++pos) {
    const uint h = block * 8 + nl / 4 + 4 * pos;
    bfloat total = bfloat(0.0f);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const bfloat raw = bfloat(r[4 * pos + s]);
      const bfloat gate = mk_hc_sigmoid_unary(raw);
      const bfloat product = bfloat(float(gate) * float(normalized[(ulong(ml) * 4 + s) * 2560 + h]));
      total = bfloat(float(product) + float(total));
    }
    mixed[ulong(ml) * 2560 + h] = bfloat(float(total) / 4.0f);
  }
}

[[kernel]] void mk_hc_up(const device bfloat *normalized [[buffer(0)]], const device float *partial [[buffer(1)]],
                         const device uchar *w [[buffer(2)]], const device bfloat *params [[buffer(3)]],
                         device bfloat *mixed [[buffer(4)]], device bfloat *gates [[buffer(5)]],
                         constant MkHCUpParams &p [[buffer(6)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup bfloat act[MK_MAXR * 320];
  const uint rows = p.rows;
  // 2560 activations, 20 per thread; every partial load is issued before use.
  float totals[20];
#pragma unroll
  for (int j = 0; j < 20; ++j) {
    const uint i = tid + 128 * j, m = i / 320, k = i % 320;
    float t = 0.0f;
    if (m < rows) {
#pragma unroll
      for (uint s = 0; s < MK_HC_SPLITS; ++s) t += partial[(ulong(s) * MK_MAXR + m) * MK_HC_WIDTH + k];
    }
    totals[j] = t;
  }
#pragma unroll
  for (int j = 0; j < 20; ++j) {
    const uint i = tid + 128 * j;
    const bfloat raw = bfloat(totals[j]);
    const bfloat divided = bfloat(float(raw) / 4.0f);
    act[i] = i / 320 < rows ? bfloat(float(divided) * float(mk_hc_sigmoid_fast(divided))) : bfloat(0.0f);
  }
  if (tg == 0 && p.has_injection && tid < rows * 4) {
    const uint m = tid / 4, j = tid % 4;
    float total = 0.0f;
    for (uint s = 0; s < MK_HC_SPLITS; ++s) total += partial[(ulong(s) * MK_MAXR + m) * MK_HC_WIDTH + 320 + j];
    const bfloat raw = bfloat(total);
    const bfloat divided = bfloat(float(raw) / 4.0f);
    gates[m * 4 + j] = bfloat(2.0f * float(mk_hc_sigmoid_unary(divided)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint block = tg * 4 + simd;
  const threadgroup bfloat *a = act;
  switch (p.up.bits * 1000 + p.up.group) {
  case 4064: mk_hc_up_block<4, 64>(a, rows, w, params, p.up, block, normalized, mixed, lane); break;
  case 5064: mk_hc_up_block<5, 64>(a, rows, w, params, p.up, block, normalized, mixed, lane); break;
  case 6064: mk_hc_up_block<6, 64>(a, rows, w, params, p.up, block, normalized, mixed, lane); break;
  default:   mk_hc_up_block<8, 64>(a, rows, w, params, p.up, block, normalized, mixed, lane); break;
  }
}

template [[host_name("mk_hc_down_sk8")]] [[kernel]] void mk_hc_down<8>(
    const device bfloat *, const device uchar *, const device bfloat *, device float *,
    constant MkHCDownParams &, uint2, uint, uint, uint);
