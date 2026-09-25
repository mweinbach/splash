// Few-row (2..8) quantized GEMV on the matrix units over lane-major tiles
// (decode verification windows; OptQmv::addQmv with SPLASH_MK_QMV=1). The tile
// format is described in mk_tiles.h.
#include "mk_tiles.h"

struct MkDenseParams {
  uint K, N, rows, x_stride, y_stride, row0;
  uint padded_n;       // parameter row length (multiple of 32)
  uint pad0;
  ulong bias_offset;   // elements from sT to bT
};

// One 32-column block (tiles at wt, parameters at sT[g][NP]) of a projection:
// y[row0 + m, n0 + n] for n0 + n < N. SK simdgroups split K.
template <int BITS, int G, int SK, int RB = 1>
inline void mk_mpt_block(const device bfloat *xr, uint x_stride, uint K, uint rows,
                         const device uchar *wt, const device bfloat *sT, ulong bias_offset,
                         uint N, uint NP, uint n0, device bfloat *y, uint y_stride,
                         uint tid, uint simd, uint lane, threadgroup float *red) {
  constexpr int NT = 32;
  constexpr uint T = SK * 32, RM = 8 * RB;
  float r[8 * RB];
  uint nl, ml;
  mk_tile_accumulate<BITS, G, const device bfloat *, 1, RB>(xr, x_stride, K, rows, wt, sT, bias_offset, NP, n0,
                                                           simd, SK, K / 64, lane, r, nl, ml);
  if (SK > 1) {
#pragma unroll
    for (int i = 0; i < 8 * RB; ++i)
      red[(simd * RM + ml + 8 * (i >> 3)) * NT + nl + (i & 3) + 16 * ((i >> 2) & 1)] = r[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < rows * NT; i += T) {
      const uint m = i / NT, n = i % NT;
      float t = 0.0f;
      for (uint s = 0; s < uint(SK); ++s) t += red[(s * RM + m) * NT + n];
      if (n0 + n < N) y[ulong(m) * y_stride + n0 + n] = bfloat(t);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8 * RB; ++i) {
      const uint m = ml + 8 * (i >> 3), n = n0 + nl + (i & 3) + 16 * ((i >> 2) & 1);
      if (m < rows && n < N) y[ulong(m) * y_stride + n] = bfloat(r[i]);
    }
  }
}

template <int BITS, int G, int SK, int XS>
[[kernel]] void mk_mpt(const device bfloat *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
                       const device bfloat *sT [[buffer(2)]],
                       device bfloat *y [[buffer(3)]], constant MkDenseParams &p [[buffer(4)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[(SK > 1 ? SK : 1) * MK_MAXR * 32];
  const uint nt = p.K / 64;
  mk_mpt_block<BITS, G, SK>(x + ulong(p.row0) * p.x_stride, p.x_stride, p.K, p.rows,
                            w + ulong(tg) * nt * MkTile<BITS>::BYTES, sT, p.bias_offset, p.N,
                            p.padded_n, tg * 32, y + ulong(p.row0) * p.y_stride, p.y_stride,
                            tid, simd, lane, red);
}

// 16-row windows (batched verification of several requests).
template <int BITS, int G, int SK>
[[kernel]] void mk_mpt16(const device bfloat *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
                         const device bfloat *sT [[buffer(2)]],
                         device bfloat *y [[buffer(3)]], constant MkDenseParams &p [[buffer(4)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[SK * 2 * MK_MAXR * 32];
  const uint nt = p.K / 64;
  mk_mpt_block<BITS, G, SK, 2>(x + ulong(p.row0) * p.x_stride, p.x_stride, p.K, p.rows,
                               w + ulong(tg) * nt * MkTile<BITS>::BYTES, sT, p.bias_offset, p.N,
                               p.padded_n, tg * 32, y + ulong(p.row0) * p.y_stride, p.y_stride,
                               tid, simd, lane, red);
}

// Several projections of the same input in one dispatch (e.g. GDN qkv/z/a/b).
// Segment s owns threadgroups [block_begin, next block_begin); its tiles and
// transposed parameters live at byte/element offsets of shared buffers and its
// output is buffer 3 + out_index.
struct MkSegment {
  uint block_begin, n, padded_n, bits;
  uint group, out_index, pad0, pad1;
  ulong tile_offset;   // bytes
  ulong param_offset;  // BF16 elements (scales; biases follow at bias_offset)
  ulong bias_offset;   // elements from scales to biases
  ulong pad2;
};
struct MkMultiParams {
  uint K, rows, segments, x_stride;
  uint y_stride[4];
  MkSegment segment[4];
};

template <int SK>
[[kernel]] void mk_mpt_multi(const device bfloat *x [[buffer(0)]], const device uchar *w [[buffer(1)]],
                             const device bfloat *params [[buffer(2)]],
                             device bfloat *y0 [[buffer(3)]], device bfloat *y1 [[buffer(4)]],
                             device bfloat *y2 [[buffer(5)]], device bfloat *y3 [[buffer(6)]],
                             constant MkMultiParams &p [[buffer(7)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[(SK > 1 ? SK : 1) * MK_MAXR * 32];
  uint s = 0;
  for (uint i = 1; i < p.segments; ++i) if (tg >= p.segment[i].block_begin) s = i;
  constant MkSegment &seg = p.segment[s];
  const uint block = tg - seg.block_begin, nt = p.K / 64;
  device bfloat *y = seg.out_index == 0 ? y0 : seg.out_index == 1 ? y1 : seg.out_index == 2 ? y2 : y3;
  const uint ys = p.y_stride[seg.out_index];
  const device bfloat *sT = params + seg.param_offset;
#define MK_SEG(B, GS) \
  mk_mpt_block<B, GS, SK>(x, p.x_stride, p.K, p.rows, w + seg.tile_offset + ulong(block) * nt * MkTile<B>::BYTES, \
                          sT, seg.bias_offset, seg.n, seg.padded_n, block * 32, y, ys, tid, simd, lane, red)
  const uint key = seg.bits * 1000 + seg.group;
  switch (key) {
  case 4064: MK_SEG(4, 64); break;
  case 5064: MK_SEG(5, 64); break;
  case 5128: MK_SEG(5, 128); break;
  case 6064: MK_SEG(6, 64); break;
  case 6128: MK_SEG(6, 128); break;
  case 8064: MK_SEG(8, 64); break;
  default:   MK_SEG(8, 128); break;
  }
#undef MK_SEG
}
template [[host_name("mk_mpt_multi_sk4")]] [[kernel]] void mk_mpt_multi<4>(
    const device bfloat *, const device uchar *, const device bfloat *, device bfloat *, device bfloat *,
    device bfloat *, device bfloat *, constant MkMultiParams &, uint, uint, uint, uint);

#define MK_MPT(B, G, SK, XS) \
  template [[host_name("mk_mpt_b" #B "_g" #G "_sk" #SK "_x" #XS)]] \
  [[kernel]] void mk_mpt<B, G, SK, XS>(const device bfloat *, const device uchar *, const device bfloat *, \
      device bfloat *, constant MkDenseParams &, uint, uint, uint, uint);
#define MK_MPT16(B, G, SK) \
  template [[host_name("mk_mpt16_b" #B "_g" #G "_sk" #SK)]] \
  [[kernel]] void mk_mpt16<B, G, SK>(const device bfloat *, const device uchar *, const device bfloat *, \
      device bfloat *, constant MkDenseParams &, uint, uint, uint, uint);
#define MK_MPT_ALL(B, G) MK_MPT(B, G, 4, 1) MK_MPT(B, G, 16, 1) MK_MPT16(B, G, 4) MK_MPT16(B, G, 8)
MK_MPT_ALL(4, 64)
MK_MPT_ALL(5, 64)
MK_MPT_ALL(5, 128)
MK_MPT_ALL(6, 64)
MK_MPT_ALL(6, 128)
MK_MPT_ALL(8, 64)
MK_MPT_ALL(8, 128)
