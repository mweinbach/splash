// Decode-window (<= 8 rows) routed-expert kernels on the matrix units.
//   gate_up: inter[u][m][640] = silu(x_m . gate_e) * (x_m . up_e)   for unique expert u
//   down:    out[u][m][2560]  = inter[u][m] . down_e
// LAYOUT 0: codes in lane-major 32x64 tiles per expert (see gemv/mk_mpt.metal).
// LAYOUT 1: original MLX Q4 rows; each lane gathers its 16 ushorts per tile.
// Scales/biases transposed per expert: [K/64][N] scales then [K/64][N] biases.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Mk2Plan { uint rows, unique, pad0, pad1; uint experts[80]; };
struct Mk2Params {
  uint rows, pad0, pad1, pad2;
  ulong code_expert_stride;   // bytes per expert (same for tiled and MLX)
  ulong param_expert_stride;  // BF16 elements per expert (scales + biases)
};

constant constexpr uint MK2_MAXR = 8;

inline uint mk2_spread_nibbles(uint v) {
  // 16-bit value of four 4-bit codes -> four bytes.
  const uint a = (v & 0x00FFu) | ((v & 0xFF00u) << 8);
  return (a & 0x000F000Fu) | ((a & 0x00F000F0u) << 4);
}

template <int LAYOUT> struct Mk2Tile;
template <> struct Mk2Tile<0> {
  struct Raw { uint4 n0, n1; };
  // t: tile base (1 KB), row_stride unused.
  static inline Raw load(const device uchar *blk, uint kt, uint, uint lane, uint, uint) {
    const device uint4 *p = reinterpret_cast<const device uint4 *>(blk + kt * 1024 + lane * 32);
    return {p[0], p[1]};
  }
  static inline void unpack(const thread Raw &r, thread uint *u) {
    const uint n[8] = {r.n0.x, r.n0.y, r.n0.z, r.n0.w, r.n1.x, r.n1.y, r.n1.z, r.n1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) { u[2 * i] = n[i] & 0x0F0F0F0Fu; u[2 * i + 1] = (n[i] >> 4) & 0x0F0F0F0Fu; }
  }
};
template <> struct Mk2Tile<1> {
  struct Raw { ushort v[16]; };
  // blk: first row of the 32-row block; row_stride bytes; kL/nL lane coordinates.
  static inline Raw load(const device uchar *blk, uint kt, uint row_stride, uint, uint kL, uint nL) {
    Raw r;
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
      const device uchar *row = blk + ulong(nL + 8 * nb) * row_stride + kt * 32 + kL / 2;
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) r.v[kb * 4 + nb] = *reinterpret_cast<const device ushort *>(row + 8 * kb);
    }
    return r;
  }
  static inline void unpack(const thread Raw &r, thread uint *u) {
#pragma unroll
    for (int w = 0; w < 16; ++w) u[w] = mk2_spread_nibbles(r.v[w]);
  }
};

// LAYOUT 2: MLX rows; lane L loads row L of the 32-row block (32 bytes = 64
// codes of the tile) with two 16-byte loads, then shuffles redistribute each
// lane's 16 ushorts (rows nL + 8nb, ushort index kL/4 + 4kb).
template <> struct Mk2Tile<2> {
  struct Raw { uint4 a, b; };
  static inline Raw load(const device uchar *blk, uint kt, uint row_stride, uint lane, uint, uint) {
    const device uint4 *p = reinterpret_cast<const device uint4 *>(blk + ulong(lane) * row_stride + kt * 32);
    return {p[0], p[1]};
  }
  static inline void unpack_lane(const thread Raw &r, thread uint *u, uint kL, uint nL) {
    const uint w[8] = {r.a.x, r.a.y, r.a.z, r.a.w, r.b.x, r.b.y, r.b.z, r.b.w};
    const uint hi = (kL >> 3) & 1u, hsel = (kL >> 2) & 1u;
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
      const ushort src = ushort(nL + 8 * nb);
#pragma unroll
      for (int kb = 0; kb < 4; ++kb) {
        const uint lo = simd_shuffle(w[2 * kb], src), up = simd_shuffle(w[2 * kb + 1], src);
        const uint word = hi ? up : lo;
        u[kb * 4 + nb] = mk2_spread_nibbles(hsel ? (word >> 16) : (word & 0xFFFFu));
      }
    }
  }
  static inline void unpack(const thread Raw &, thread uint *) {}
};

inline float mk2_tile_xsum(const device bfloat *x, uint x_stride, uint rows, uint k0, uint lane, uint ml) {
  const uint m = lane >> 2, c = lane & 3;
  float v = 0.0f;
  if (m < rows) {
    const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(x + ulong(m) * x_stride + k0 + c * 16);
    const float4 a = float4(q[0]), b = float4(q[1]), d = float4(q[2]), e = float4(q[3]);
    v = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w)) + ((d.x + d.y) + (d.z + d.w)) + ((e.x + e.y) + (e.z + e.w));
  }
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 2);
  return simd_shuffle(v, ushort(ml * 4));
}

inline bfloat mk2_sigmoid_fast(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
inline bfloat mk2_silu_mul(float gate_acc, float up_acc) {
  const bfloat gate = bfloat(gate_acc), up = bfloat(up_acc);
  const bfloat silu = bfloat(float(gate) * float(mk2_sigmoid_fast(gate)));
  return bfloat(float(silu) * float(up));
}

// Grid (N/32 column blocks, unique experts); SK simdgroups split K. NMAT = 2 for
// gate+up (SwiGLU epilogue), 1 for down.
template <int LAYOUT, int NMAT, int SK, int D = 1>
inline void mk2_body(const device bfloat *x, uint x_stride, uint K, uint N, uint rows,
                     const device uchar *w0, const device uchar *w1,
                     const device bfloat *p0, const device bfloat *p1,
                     device bfloat *y, uint y_stride, uint nb, uint tid, uint simd, uint lane,
                     threadgroup float *red) {
  constexpr int M = 16, NT = 32, KT = 64;
  using TL = Mk2Tile<LAYOUT>;
  const uint nt = K / KT, ng = K / 64, n0 = nb * NT;
  const uint row_stride = K / 2;
  // LAYOUT 0: column block nb's tiles are contiguous; LAYOUT 1: rows n0.. of the MLX matrix.
  const device uchar *b0 = LAYOUT == 0 ? w0 + ulong(nb) * nt * 1024 : w0 + ulong(n0) * row_stride;
  const device uchar *b1 = LAYOUT == 0 ? w1 + ulong(nb) * nt * 1024 : w1 + ulong(n0) * row_stride;
  (void)b1;
  const uint kL = 4 * (lane & 1) + 8 * ((lane >> 3) & 1), nL = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1);

  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{int(K), int(rows)}, array<int, 2>{1, int(x_stride)});
  tensor<device uint8_t, dextents<int, 2>, tensor_inline> bv(
      const_cast<device uchar *>(w0), dextents<int, 2>{KT, NT}, array<int, 2>{1, KT});
  constexpr auto desc = matmul2d_descriptor(M, NT, KT, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto a0 = a.template slice<KT, M>(0, 0);
  auto bs = bv.template slice<KT, NT>(0, 0);
  using DT = decltype(op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs), float>());
  DT probe = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs), float>();
  const auto id0 = probe.get_multidimensional_index(ushort(0));
  const uint nl = uint(id0[0]), ml = uint(id0[1]);
  float r0[8], r1[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { r0[i] = 0.0f; r1[i] = 0.0f; }
  auto rb = op.template get_right_input_cooperative_tensor<bfloat, uint8_t, float>();
  thread uint *rbw = reinterpret_cast<thread uint *>(&rb[0]);

  typename TL::Raw q0[D], q1[D];
#pragma unroll
  for (int d = 0; d < D; ++d) {
    const uint kd = simd + uint(d) * SK;
    if (kd < nt) { q0[d] = TL::load(b0, kd, row_stride, lane, kL, nL); if (NMAT == 2) q1[d] = TL::load(b1, kd, row_stride, lane, kL, nL); }
  }
  for (uint k = simd; k < nt;) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      if (k < nt) {
        const float xs = mk2_tile_xsum(x, x_stride, rows, k * KT, lane, ml);
        auto ag = a.template slice<KT, M>(k * KT, 0);
        const uint g = k;  // G = 64
        uint u0[16], u1[16];
        TL::unpack(q0[d], u0);
        if (NMAT == 2) TL::unpack(q1[d], u1);
        const uint kn = k + uint(D) * SK;
        if (kn < nt) { q0[d] = TL::load(b0, kn, row_stride, lane, kL, nL); if (NMAT == 2) q1[d] = TL::load(b1, kn, row_stride, lane, kL, nL); }
#pragma unroll
        for (int mat = 0; mat < NMAT; ++mat) {
#pragma unroll
          for (int i = 0; i < 16; ++i) rbw[i] = mat ? u1[i] : u0[i];
          DT t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs), float>();
          op.run(ag, rb, t);
          const device bfloat *sg = (mat ? p1 : p0) + ulong(g) * N + n0 + nl;
          const device bfloat *bg = sg + ulong(ng) * N;
          const float4 s0 = float4(*reinterpret_cast<const device bfloat4 *>(sg));
          const float4 s1 = float4(*reinterpret_cast<const device bfloat4 *>(sg + 16));
          const float4 c0v = float4(*reinterpret_cast<const device bfloat4 *>(bg));
          const float4 c1v = float4(*reinterpret_cast<const device bfloat4 *>(bg + 16));
          thread float *r = mat ? r1 : r0;
          r[0] = fma(t[0], s0.x, fma(xs, c0v.x, r[0]));
          r[1] = fma(t[1], s0.y, fma(xs, c0v.y, r[1]));
          r[2] = fma(t[2], s0.z, fma(xs, c0v.z, r[2]));
          r[3] = fma(t[3], s0.w, fma(xs, c0v.w, r[3]));
          r[4] = fma(t[8], s1.x, fma(xs, c1v.x, r[4]));
          r[5] = fma(t[9], s1.y, fma(xs, c1v.y, r[5]));
          r[6] = fma(t[10], s1.z, fma(xs, c1v.z, r[6]));
          r[7] = fma(t[11], s1.w, fma(xs, c1v.w, r[7]));
        }
      }
      k += SK;
    }
  }
  // Cross-simdgroup reduction: red[mat][sk][8][32].
  constexpr uint PLANE = SK * MK2_MAXR * NT;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint idx = (simd * MK2_MAXR + ml) * NT + nl + (i & 3) + 16 * (i >> 2);
    red[idx] = r0[i];
    if (NMAT == 2) red[PLANE + idx] = r1[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < rows * NT; i += SK * 32) {
    const uint m = i / NT, n = i % NT;
    float t0 = 0.0f, t1 = 0.0f;
    for (uint s = 0; s < uint(SK); ++s) {
      t0 += red[(s * MK2_MAXR + m) * NT + n];
      if (NMAT == 2) t1 += red[PLANE + (s * MK2_MAXR + m) * NT + n];
    }
    y[ulong(m) * y_stride + n0 + n] = NMAT == 2 ? mk2_silu_mul(t0, t1) : bfloat(t0);
  }
}

template <int LAYOUT, int SK, int D>
[[kernel]] void mk2_gate_up(const device bfloat *x [[buffer(0)]],
                            const device uchar *gw [[buffer(1)]], const device uchar *uw [[buffer(2)]],
                            const device bfloat *gp [[buffer(3)]], const device bfloat *up [[buffer(4)]],
                            device bfloat *inter [[buffer(5)]],   // [u][8][640]
                            const device Mk2Plan *plan [[buffer(6)]],
                            constant Mk2Params &p [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[2 * SK * MK2_MAXR * 32];
  const uint u = tg.y;
  if (u >= plan->unique) return;
  const uint e = plan->experts[u];
  mk2_body<LAYOUT, 2, SK, D>(x, 2560, 2560, 640, p.rows, gw + ulong(e) * p.code_expert_stride,
                          uw + ulong(e) * p.code_expert_stride, gp + ulong(e) * p.param_expert_stride,
                          up + ulong(e) * p.param_expert_stride, inter + ulong(u) * MK2_MAXR * 640, 640,
                          tg.x, tid, simd, lane, red);
}

template <int LAYOUT, int SK, int D>
[[kernel]] void mk2_down(const device bfloat *inter [[buffer(0)]],
                         const device uchar *dw [[buffer(1)]], const device bfloat *dp [[buffer(2)]],
                         device bfloat *out [[buffer(3)]],   // [u][8][2560]
                         const device Mk2Plan *plan [[buffer(4)]],
                         constant Mk2Params &p [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[SK * MK2_MAXR * 32];
  const uint u = tg.y;
  if (u >= plan->unique) return;
  const uint e = plan->experts[u];
  mk2_body<LAYOUT, 1, SK, D>(inter + ulong(u) * MK2_MAXR * 640, 640, 640, 2560, p.rows,
                          dw + ulong(e) * p.code_expert_stride, dw, dp + ulong(e) * p.param_expert_stride, dp,
                          out + ulong(u) * MK2_MAXR * 2560, 2560, tg.x, tid, simd, lane, red);
}

#define MK2_GU(L, SK, D) template [[host_name("mk2_gate_up_l" #L "_sk" #SK "_d" #D)]] [[kernel]] void mk2_gate_up<L, SK, D>( \
    const device bfloat *, const device uchar *, const device uchar *, const device bfloat *, const device bfloat *, \
    device bfloat *, const device Mk2Plan *, constant Mk2Params &, uint2, uint, uint, uint);
#define MK2_DN(L, SK, D) template [[host_name("mk2_down_l" #L "_sk" #SK "_d" #D)]] [[kernel]] void mk2_down<L, SK, D>( \
    const device bfloat *, const device uchar *, const device bfloat *, device bfloat *, const device Mk2Plan *, \
    constant Mk2Params &, uint2, uint, uint, uint);
MK2_GU(0, 8, 1) MK2_GU(0, 8, 2) MK2_GU(0, 8, 5) MK2_GU(0, 4, 5) MK2_GU(0, 4, 10) MK2_GU(0, 2, 5)
MK2_DN(0, 2, 1) MK2_DN(0, 2, 5) MK2_DN(0, 5, 2) MK2_DN(0, 1, 5) MK2_DN(0, 1, 10)
