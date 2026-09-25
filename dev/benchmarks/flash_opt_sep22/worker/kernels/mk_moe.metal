// Megakernel decode MoE phases for 1..8 row windows (decode and MTP verify).
//
//   mk_moe_router      router logits (BF16 W), shared-expert SwiGLU (gate+up)
//                      and the shared-expert gate logit, one dispatch.
//   mk_moe_plan        routing (softmax/top-10) for every row and the union of
//                      selected experts, one threadgroup.
//   mk_moe_gate_up     every unique expert's gate/up weights are read once by the
//                      matrix units for the whole window (rows that did not
//                      select the expert are computed and discarded).
//   mk_moe_down        union expert down projections plus the shared down.
//   mk_moe_gate_up_t,  the same two phases over lane-major expert tiles
//   mk_moe_down_t      (mk_tiles.h; SPLASH_MK_MOE_TILED=1).
//   mk_moe_bucket_*    tiled experts for 16-row bucket jobs of the blocked MoE
//                      (prefill windows below 1,024 rows).
//   mk_moe_combine     MLX-order weighted combine, sigmoid-gated shared expert,
//                      HC injection into the residual streams and the next
//                      grouped RMS norm.
//
// Element-wise BF16 boundaries follow opt_moe_route, opt_moe, opt_qmv_swiglu,
// flash_moe_combine and flash_hc_fused_inject_norm.
#include "mk_tiles.h"
#include "metal/abi/FlashMoEBuckets.h"

constant constexpr uint MK_WIDTH = 2560;
constant constexpr uint MK_EXPERTS = 512;
constant constexpr uint MK_SEL = 10;
constant constexpr uint MK_INTER = 640;
constant constexpr uint MK_MAX_ROWS = 16;
constant constexpr uint MK_MAX_ROUTES = MK_MAX_ROWS * MK_SEL;
constant constexpr uint MK_M = 16;  // matrix-unit row tile (rows >= window are discarded)

struct MkMoEPlan {
  uint rows, unique, pad0, pad1;
  uint experts[MK_MAX_ROUTES];              // unique expert ids, ascending
  uint route_of[MK_MAX_ROUTES][MK_MAX_ROWS]; // route index (row*10+slot) or ~0u
  uint route_expert[MK_MAX_ROUTES];
  float route_weight[MK_MAX_ROUTES];        // BF16-rounded normalized top-k weights
};

struct MkAffine {
  ulong w_row_stride;   // bytes between packed rows
  ulong w_expert_stride;
  ulong p_row_stride;   // elements between scale/bias rows
  ulong p_expert_stride;
  uint bits, group, pad0, pad1;
};

struct MkRouterParams {
  uint rows;
  uint pad0, pad1, pad2;
  MkAffine shared_gate, shared_up, shared_logit;
};

struct MkMoEParams {
  uint rows;
  uint pad0, pad1, pad2;
  MkAffine gate, up, down, shared_down;
};

struct MkCombineParams {
  uint rows;
  uint has_norm;        // write the next grouped RMS norm
  uint norm_convention; // 0: (1 + w), 1: w
  float epsilon;
};

// ---------------------------------------------------------------------------
// Code extraction (MLX bitstream packing), identical to opt_codes.
template <int BITS, int VPT> struct MkCodeWords;
template <> struct MkCodeWords<4, 8> { typedef uint T; static constant constexpr int W = 1; };
template <> struct MkCodeWords<8, 8> { typedef uint T; static constant constexpr int W = 2; };

template <int BITS, int VPT>
inline void mk_codes(const device uchar *row, uint k0, thread float *q) {
  typedef typename MkCodeWords<BITS, VPT>::T T;
  constexpr int W = MkCodeWords<BITS, VPT>::W;
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

inline bfloat mk_sigmoid_fast(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
inline bfloat mk_sigmoid_precise(bfloat source) {
  const bfloat exponent = bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

inline bfloat mk_swiglu(bfloat gate, bfloat up) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const bfloat sigmoid = mk_sigmoid_fast(gate);
  const bfloat silu = gate * sigmoid;
  return silu * up;
}

inline bfloat mk_expert_silu_mul(float gate_acc, float up_acc) {
  const bfloat gate = bfloat(gate_acc), up = bfloat(up_acc);
  const bfloat silu = bfloat(float(gate) * float(mk_sigmoid_fast(gate)));
  return bfloat(float(silu) * float(up));
}

template <int R>
inline void mk_load_x(const device bfloat *x, uint x_stride, uint k0,
                      thread float (&xv)[R][8], thread float (&xs)[R]) {
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const device bfloat4 *xr = reinterpret_cast<const device bfloat4 *>(x + ulong(r) * x_stride + k0);
    const bfloat4 a = xr[0], b = xr[1];
    xv[r][0] = a.x; xv[r][1] = a.y; xv[r][2] = a.z; xv[r][3] = a.w;
    xv[r][4] = b.x; xv[r][5] = b.y; xv[r][6] = b.z; xv[r][7] = b.w;
    xs[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) xs[r] += xv[r][i];
  }
}

// acc[r][n] += s*dot(x_r, q_n) + b*sum(x_r) over one 8-wide K slice.
template <int BITS, int R, int RN>
inline void mk_dot_affine(const thread float (&xv)[R][8], const thread float (&xs)[R],
                          const device uchar *w, const device bfloat *s, const device bfloat *b,
                          MkAffine a, uint n0, uint N, uint k0, thread float (&acc)[R][RN]) {
  const uint g = k0 / a.group;
#pragma unroll
  for (int n = 0; n < RN; ++n) {
    const uint col = n0 + n;
    if (col >= N) break;
    float q[8];
    mk_codes<BITS, 8>(w + ulong(col) * a.w_row_stride, k0, q);
    const ulong pi = ulong(col) * a.p_row_stride + g;
    const float sc = float(s[pi]), bi = float(b[pi]);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float d = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) d = fma(xv[r][i], q[i], d);
      acc[r][n] = fma(sc, d, fma(bi, xs[r], acc[r][n]));
    }
  }
}

// ---------------------------------------------------------------------------
// Router logits, shared SwiGLU and shared gate logit (SIMD, R exact rows).
// Threadgroup = 8 simdgroups = 2 items x 4-way split K (640 each).
// Items [0, 128): router columns 4i..4i+3. Items [128, 288): shared columns
// 4j..4j+3 (gate and up). Item 288: shared gate logit.
template <int R>
[[kernel]] void mk_moe_router(
    const device bfloat *x [[buffer(0)]],
    const device bfloat *router_w [[buffer(1)]],
    const device uchar *gw [[buffer(2)]], const device bfloat *gs [[buffer(3)]], const device bfloat *gb [[buffer(4)]],
    const device uchar *uw [[buffer(5)]], const device bfloat *us [[buffer(6)]], const device bfloat *ub [[buffer(7)]],
    const device uchar *lw [[buffer(8)]], const device bfloat *ls [[buffer(9)]], const device bfloat *lb [[buffer(10)]],
    device bfloat *logits [[buffer(11)]], device bfloat *shared_inter [[buffer(12)]],
    device bfloat *shared_logit [[buffer(13)]],
    constant MkRouterParams &p [[buffer(14)]],
    uint tg [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int RN = 4, SK = 4;
  constexpr uint KS = MK_WIDTH / SK;
  threadgroup float partials[2][SK][2][R][RN];
  const uint slot = simd / SK, sk = simd % SK;
  const uint item = tg * 2 + slot;
  const bool router = item < 128, shared = item >= 128 && item < 288, logit = item == 288;
  float acc0[R][RN], acc1[R][RN];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < RN; ++n) { acc0[r][n] = 0.0f; acc1[r][n] = 0.0f; }
  if (router || shared || logit) {
    for (uint k0 = sk * KS + lane * 8; k0 < (sk + 1) * KS; k0 += 256) {
      float xv[R][8], xs[R];
      mk_load_x<R>(x, MK_WIDTH, k0, xv, xs);
      if (router) {
#pragma unroll
        for (int n = 0; n < RN; ++n) {
          const device bfloat4 *wr = reinterpret_cast<const device bfloat4 *>(router_w + ulong(item * RN + n) * MK_WIDTH + k0);
          const bfloat4 a = wr[0], b = wr[1];
          const float q[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
          for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc0[r][n] = fma(xv[r][i], q[i], acc0[r][n]);
        }
      } else if (shared) {
        const uint n0 = (item - 128) * RN;
        if (p.shared_gate.bits == 8) {
          mk_dot_affine<8, R, RN>(xv, xs, gw, gs, gb, p.shared_gate, n0, MK_INTER, k0, acc0);
          mk_dot_affine<8, R, RN>(xv, xs, uw, us, ub, p.shared_up, n0, MK_INTER, k0, acc1);
        } else {
          mk_dot_affine<4, R, RN>(xv, xs, gw, gs, gb, p.shared_gate, n0, MK_INTER, k0, acc0);
          mk_dot_affine<4, R, RN>(xv, xs, uw, us, ub, p.shared_up, n0, MK_INTER, k0, acc1);
        }
      } else {
        if (p.shared_logit.bits == 8)
          mk_dot_affine<8, R, RN>(xv, xs, lw, ls, lb, p.shared_logit, 0, 1, k0, acc0);
        else
          mk_dot_affine<4, R, RN>(xv, xs, lw, ls, lb, p.shared_logit, 0, 1, k0, acc0);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int n = 0; n < RN; ++n) { acc0[r][n] = simd_sum(acc0[r][n]); acc1[r][n] = simd_sum(acc1[r][n]); }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int n = 0; n < RN; ++n) {
        partials[slot][sk][0][r][n] = acc0[r][n];
        partials[slot][sk][1][r][n] = acc1[r][n];
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sk != 0 || lane >= uint(R * RN) || !(router || shared || logit)) return;
  const uint r = lane / RN, n = lane % RN;
  float t0 = 0.0f, t1 = 0.0f;
  for (uint s = 0; s < uint(SK); ++s) { t0 += partials[slot][s][0][r][n]; t1 += partials[slot][s][1][r][n]; }
  if (router) {
    logits[r * MK_EXPERTS + item * RN + n] = bfloat(t0);
  } else if (shared) {
    shared_inter[r * MK_INTER + (item - 128) * RN + n] = mk_swiglu(bfloat(t0), bfloat(t1));
  } else if (n == 0) {
    shared_logit[r] = bfloat(t0);
  }
}

// ---------------------------------------------------------------------------
// Matrix-unit router + shared SwiGLU + shared gate logit. Grid 37 x 256 threads:
// tg < 16: router columns [32 tg, +32), 8 simdgroups split K.
// tg < 36: shared columns [32 (tg-16), +32); simdgroups 0-3 gate, 4-7 up.
// tg == 36: shared gate logit (SIMD reduction).
[[kernel]] void mk_moe_router_mpp(
    const device bfloat *x [[buffer(0)]],
    const device bfloat *router_w [[buffer(1)]],
    const device uchar *gw [[buffer(2)]], const device bfloat *gs [[buffer(3)]], const device bfloat *gb [[buffer(4)]],
    const device uchar *uw [[buffer(5)]], const device bfloat *us [[buffer(6)]], const device bfloat *ub [[buffer(7)]],
    const device uchar *lw [[buffer(8)]], const device bfloat *ls [[buffer(9)]], const device bfloat *lb [[buffer(10)]],
    device bfloat *logits [[buffer(11)]], device bfloat *shared_inter [[buffer(12)]],
    device bfloat *shared_logit [[buffer(13)]],
    constant MkRouterParams &p [[buffer(14)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int M = MK_M, NT = 32;
  constexpr uint K = MK_WIDTH;
  const uint rows = p.rows;
  threadgroup float red[8 * M * NT];
  threadgroup float xsum[40 * M];
  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{int(K), int(rows)}, array<int, 2>{1, int(K)});
  if (tg < 16) {
    constexpr int G = 64;
    const uint n0 = tg * NT;
    auto bw = tensor(const_cast<device bfloat *>(router_w + ulong(n0) * K), dextents<int, 2>{int(K), NT}, array<int, 2>{1, int(K)});
    constexpr auto desc = matmul2d_descriptor(M, NT, G, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<1>> op;
    auto a0 = a.template slice<G, M>(0, 0);
    auto b0 = bw.template slice<G, NT>(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (uint c = simd; c < K / G; c += 8) {
      auto ac = a.template slice<G, M>(c * G, 0);
      auto bc = bw.template slice<G, NT>(c * G, 0);
      op.run(ac, bc, acc);
    }
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      red[(simd * M + idx[1]) * NT + idx[0]] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < rows * NT; i += 256) {
      const uint m = i / NT, n = i % NT;
      float t = 0.0f;
      for (uint s = 0; s < 8; ++s) t += red[(s * M + m) * NT + n];
      logits[m * MK_EXPERTS + n0 + n] = bfloat(t);
    }
    return;
  }
  if (tg < 36) {
    constexpr int G = 128;
    constexpr uint NG = K / G;
    const uint n0 = (tg - 16) * NT;
    for (uint i = tid; i < NG * M; i += 256) {
      const uint g = i / M, m = i % M;
      float sum = 0.0f;
      if (m < rows) {
        const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(x + ulong(m) * K + g * G);
        for (int j = 0; j < G / 4; ++j) { const float4 v = float4(q[j]); sum += v.x + v.y + v.z + v.w; }
      }
      xsum[i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint mat = simd / 4, sk = simd % 4;
    const MkAffine af = mat ? p.shared_up : p.shared_gate;
    const device uchar *w = mat ? uw : gw;
    const device bfloat *sc = mat ? us : gs, *bi = mat ? ub : gb;
    tensor<device uint8_t, dextents<int, 2>, tensor_inline> bw(
        const_cast<device uchar *>(w + ulong(n0) * af.w_row_stride), dextents<int, 2>{int(K), NT},
        array<int, 2>{1, int(af.w_row_stride)});
    constexpr auto desc = matmul2d_descriptor(M, NT, G, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<1>> op;
    auto a0 = a.template slice<G, M>(0, 0);
    auto b0 = bw.template slice<G, NT>(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (uint g = sk; g < NG; g += 4) {
      auto ag = a.template slice<G, M>(g * G, 0);
      auto bg = bw.template slice<G, NT>(g * G, 0);
      auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
      op.run(ag, bg, t);
#pragma unroll
      for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        const ulong pi = ulong(n0 + idx[0]) * af.p_row_stride + g;
        acc[i] = fma(t[i], float(sc[pi]), fma(xsum[g * M + idx[1]], float(bi[pi]), acc[i]));
      }
    }
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      red[((mat * 4 + sk) * M + idx[1]) * NT + idx[0]] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < rows * NT; i += 256) {
      const uint m = i / NT, n = i % NT;
      float t0 = 0.0f, t1 = 0.0f;
      for (uint s = 0; s < 4; ++s) { t0 += red[(s * M + m) * NT + n]; t1 += red[((4 + s) * M + m) * NT + n]; }
      shared_inter[m * MK_INTER + n0 + n] = mk_swiglu(bfloat(t0), bfloat(t1));
    }
    return;
  }
  // Shared gate logit: simdgroup s handles K slice [320 s, +320); lanes over K.
  float acc[MK_MAX_ROWS];
  for (uint r = 0; r < MK_MAX_ROWS; ++r) acc[r] = 0.0f;
  const MkAffine af = p.shared_logit;
  for (uint k0 = simd * 320 + lane * 8; k0 < (simd + 1) * 320; k0 += 256) {
    float q[8];
    if (af.bits == 8) mk_codes<8, 8>(lw, k0, q); else mk_codes<4, 8>(lw, k0, q);
    const uint g = k0 / af.group;
    const float sc = float(ls[g]), bi = float(lb[g]);
    for (uint r = 0; r < rows; ++r) {
      const device bfloat4 *xr = reinterpret_cast<const device bfloat4 *>(x + ulong(r) * K + k0);
      const float4 x0 = float4(xr[0]), x1 = float4(xr[1]);
      const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
      float d = 0.0f, xs = 0.0f;
      for (uint i = 0; i < 8; ++i) { d = fma(xv[i], q[i], d); xs += xv[i]; }
      acc[r] = fma(sc, d, fma(bi, xs, acc[r]));
    }
  }
  for (uint r = 0; r < rows; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) red[simd * MK_MAX_ROWS + r] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < rows) {
    float t = 0.0f;
    for (uint s = 0; s < 8; ++s) t += red[s * MK_MAX_ROWS + tid];
    shared_logit[tid] = bfloat(t);
  }
}

// ---------------------------------------------------------------------------
// Routing (opt_moe_route semantics) for one row by one simdgroup. Lane 0
// records each winner in threadgroup memory and the selected-expert bitmask.
inline void mk_route_row_tg(const device bfloat *logits, uint row, uint lane,
                            threadgroup uint *route_expert, threadgroup float *route_weight,
                            threadgroup atomic_uint *mask) {
  float v[16];
  float peak = -INFINITY;
#pragma unroll
  for (uint m = 0; m < 16; ++m) {
    v[m] = float(logits[row * MK_EXPERTS + lane + m * 32]);
    peak = metal::max(peak, v[m]);
  }
  peak = simd_max(peak);
  float s = 0.0f;
#pragma unroll
  for (uint m = 0; m < 16; ++m) { v[m] = metal::exp(v[m] - peak); s += v[m]; }
  const float total = simd_sum(s);
#pragma unroll
  for (uint m = 0; m < 16; ++m) v[m] = float(bfloat(v[m] / total));
  float selected_sum = 0.0f;
  for (uint slot = 0; slot < MK_SEL; ++slot) {
    float best = -1.0f;
    uint best_id = UINT_MAX;
#pragma unroll
    for (uint m = 0; m < 16; ++m)
      if (v[m] > best) { best = v[m]; best_id = lane + m * 32; }
    const float winner = simd_max(best);
    const uint winner_id = simd_min(best == winner ? best_id : UINT_MAX);
#pragma unroll
    for (uint m = 0; m < 16; ++m)
      if (lane + m * 32 == winner_id) v[m] = -1.0f;
    if (lane == 0) {
      route_expert[row * MK_SEL + slot] = winner_id;
      route_weight[row * MK_SEL + slot] = winner;
      atomic_fetch_or_explicit(&mask[winner_id / 32], 1u << (winner_id % 32), memory_order_relaxed);
    }
    selected_sum = float(bfloat(selected_sum) + bfloat(winner));
  }
  if (lane != 0) return;
  const float denominator = float(bfloat(selected_sum));
  for (uint slot = 0; slot < MK_SEL; ++slot)
    route_weight[row * MK_SEL + slot] = float(bfloat(route_weight[row * MK_SEL + slot] / denominator));
}

inline void mk_route_row(const device bfloat *logits, uint row, uint lane,
                         threadgroup uint *route_expert, threadgroup float *route_weight) {
  float v[16];
  float peak = -INFINITY;
#pragma unroll
  for (uint m = 0; m < 16; ++m) {
    v[m] = float(logits[row * MK_EXPERTS + lane + m * 32]);
    peak = metal::max(peak, v[m]);
  }
  peak = simd_max(peak);
  float s = 0.0f;
#pragma unroll
  for (uint m = 0; m < 16; ++m) { v[m] = metal::exp(v[m] - peak); s += v[m]; }
  const float total = simd_sum(s);
#pragma unroll
  for (uint m = 0; m < 16; ++m) v[m] = float(bfloat(v[m] / total));
  uint selected[MK_SEL];
  float scores[MK_SEL];
  float selected_sum = 0.0f;
  for (uint slot = 0; slot < MK_SEL; ++slot) {
    float best = -1.0f;
    uint best_id = UINT_MAX;
#pragma unroll
    for (uint m = 0; m < 16; ++m)
      if (v[m] > best) { best = v[m]; best_id = lane + m * 32; }
    const float winner = simd_max(best);
    const uint winner_id = simd_min(best == winner ? best_id : UINT_MAX);
#pragma unroll
    for (uint m = 0; m < 16; ++m)
      if (lane + m * 32 == winner_id) v[m] = -1.0f;
    selected[slot] = winner_id;
    scores[slot] = winner;
    selected_sum = float(bfloat(selected_sum) + bfloat(winner));
  }
  if (lane != 0) return;
  const float denominator = float(bfloat(selected_sum));
  for (uint slot = 0; slot < MK_SEL; ++slot) {
    route_expert[row * MK_SEL + slot] = selected[slot];
    route_weight[row * MK_SEL + slot] = float(bfloat(scores[slot] / denominator));
  }
}

// One threadgroup of 256 threads: routing of every row, then the ascending
// union of selected experts and each (unique expert, row) route index.
[[kernel]] void mk_moe_plan(
    const device bfloat *logits [[buffer(0)]],
    device MkMoEPlan *plan [[buffer(1)]],
    constant MkMoEParams &p [[buffer(2)]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint route_expert[MK_MAX_ROUTES];
  threadgroup float route_weight[MK_MAX_ROUTES];
  threadgroup atomic_uint mask[16];
  threadgroup uint prefix[17];
  const uint rows = p.rows, routes = rows * MK_SEL;
  if (tid < 16) atomic_store_explicit(&mask[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint row = simd; row < rows; row += 8) mk_route_row_tg(logits, row, lane, route_expert, route_weight, mask);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    const uint word = lane < 16 ? atomic_load_explicit(&mask[lane], memory_order_relaxed) : 0u;
    const uint count = popcount(word);
    const uint before = simd_prefix_exclusive_sum(count);
    if (lane < 16) prefix[lane] = before;
    if (lane == 16) prefix[16] = before;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint total = prefix[16];
  for (uint i = tid; i < total * MK_MAX_ROWS; i += 256) plan->route_of[i / MK_MAX_ROWS][i % MK_MAX_ROWS] = ~0u;
  threadgroup_barrier(mem_flags::mem_device);
  if (tid < routes) {
    const uint e = route_expert[tid];
    const uint word = atomic_load_explicit(&mask[e / 32], memory_order_relaxed);
    const uint u = prefix[e / 32] + popcount(word & ((1u << (e % 32)) - 1u));
    plan->experts[u] = e;
    plan->route_of[u][tid / MK_SEL] = tid;
    plan->route_expert[tid] = e;
    plan->route_weight[tid] = route_weight[tid];
  }
  if (tid == 0) { plan->rows = rows; plan->unique = total; }
}

// ---------------------------------------------------------------------------
// Matrix-unit gathered GEMV: C[16, NT] = X[16 (rows < R valid), K] * W_e[n0.., K]^T
// with per-group scale/bias epilogue; SK simdgroups split K, then reduce.
template <typename CT> struct MkMppCode;
template <> struct MkMppCode<uint4b_format> { static constant constexpr int PER_BYTE = 2; };
template <> struct MkMppCode<uint8_t> { static constant constexpr int PER_BYTE = 1; };

// Grid (640 / 32 tiles, max unique experts); 4 simdgroups split K (10 groups each).
[[kernel]] void mk_moe_gate_up(
    const device bfloat *x [[buffer(0)]],
    const device uchar *gw [[buffer(1)]], const device bfloat *gs [[buffer(2)]], const device bfloat *gb [[buffer(3)]],
    const device uchar *uw [[buffer(4)]], const device bfloat *us [[buffer(5)]], const device bfloat *ub [[buffer(6)]],
    device bfloat *inter [[buffer(7)]],        // [unique][16][640]
    const device MkMoEPlan *plan [[buffer(8)]],
    constant MkMoEParams &p [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int G = 64, NT = 32, SK = 4, M = MK_M;
  constexpr uint K = MK_WIDTH, NG = K / G;
  const uint u = tg.y;
  if (u >= plan->unique) return;
  const uint expert = plan->experts[u];
  const uint rows = p.rows;
  threadgroup float xsum[NG * M];
  threadgroup float red[SK * 2 * M * NT];
  for (uint i = tid; i < NG * M; i += SK * 32) {
    const uint g = i / M, m = i % M;
    float sum = 0.0f;
    if (m < rows) {
      const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(x + ulong(m) * K + g * G);
      for (int j = 0; j < G / 4; ++j) { const float4 v = float4(q[j]); sum += v.x + v.y + v.z + v.w; }
    }
    xsum[i] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n0 = tg.x * NT;
  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{int(K), int(rows)}, array<int, 2>{1, int(K)});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bw0(
      const_cast<device uchar *>(gw + ulong(expert) * p.gate.w_expert_stride + ulong(n0) * p.gate.w_row_stride),
      dextents<int, 2>{int(K), NT}, array<int, 2>{1, int(p.gate.w_row_stride * 2)});
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bw1(
      const_cast<device uchar *>(uw + ulong(expert) * p.up.w_expert_stride + ulong(n0) * p.up.w_row_stride),
      dextents<int, 2>{int(K), NT}, array<int, 2>{1, int(p.up.w_row_stride * 2)});
  constexpr auto desc = matmul2d_descriptor(M, NT, G, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto a0 = a.template slice<G, M>(0, 0);
  auto bs0 = bw0.template slice<G, NT>(0, 0);
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
#pragma unroll
  for (ushort i = 0; i < acc0.get_capacity(); ++i) if (acc0.is_valid_element(i)) { acc0[i] = 0.0f; acc1[i] = 0.0f; }
  const device bfloat *se0 = gs + ulong(expert) * p.gate.p_expert_stride, *be0 = gb + ulong(expert) * p.gate.p_expert_stride;
  const device bfloat *se1 = us + ulong(expert) * p.up.p_expert_stride, *be1 = ub + ulong(expert) * p.up.p_expert_stride;
  for (uint g = simd; g < NG; g += SK) {
    auto ag = a.template slice<G, M>(g * G, 0);
    {
      auto bg = bw0.template slice<G, NT>(g * G, 0);
      auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
      op.run(ag, bg, t);
#pragma unroll
      for (ushort i = 0; i < acc0.get_capacity(); ++i) {
        if (!acc0.is_valid_element(i)) continue;
        const auto idx = acc0.get_multidimensional_index(i);
        const ulong pi = ulong(n0 + idx[0]) * p.gate.p_row_stride + g;
        acc0[i] = fma(t[i], float(se0[pi]), fma(xsum[g * M + idx[1]], float(be0[pi]), acc0[i]));
      }
    }
    {
      auto bg = bw1.template slice<G, NT>(g * G, 0);
      auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
      op.run(ag, bg, t);
#pragma unroll
      for (ushort i = 0; i < acc1.get_capacity(); ++i) {
        if (!acc1.is_valid_element(i)) continue;
        const auto idx = acc1.get_multidimensional_index(i);
        const ulong pi = ulong(n0 + idx[0]) * p.up.p_row_stride + g;
        acc1[i] = fma(t[i], float(se1[pi]), fma(xsum[g * M + idx[1]], float(be1[pi]), acc1[i]));
      }
    }
  }
#pragma unroll
  for (ushort i = 0; i < acc0.get_capacity(); ++i) {
    if (!acc0.is_valid_element(i)) continue;
    const auto idx = acc0.get_multidimensional_index(i);
    red[((simd * 2 + 0) * M + idx[1]) * NT + idx[0]] = acc0[i];
    red[((simd * 2 + 1) * M + idx[1]) * NT + idx[0]] = acc1[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < rows * NT; i += SK * 32) {
    const uint m = i / NT, n = i % NT;
    if (plan->route_of[u][m] == ~0u) continue;
    float t0 = 0.0f, t1 = 0.0f;
    for (uint s = 0; s < uint(SK); ++s) { t0 += red[((s * 2) * M + m) * NT + n]; t1 += red[((s * 2 + 1) * M + m) * NT + n]; }
    inter[(ulong(u) * M + m) * MK_INTER + n0 + n] = mk_expert_silu_mul(t0, t1);
  }
}

// Grid (2560 / 32 tiles, max unique experts + 1); 5 simdgroups split K.
// y = unique handles the shared expert down (BF16 input shared_inter[rows]).
template <typename CT, int G, int SK>
inline void mk_down_tile(const device bfloat *x, uint rows, const device uchar *w,
                         const device bfloat *s, const device bfloat *b, MkAffine af, uint expert,
                         uint n0, uint tid, uint simd, threadgroup float *xsum,
                         threadgroup float *red) {
  constexpr int NT = 32, M = MK_M;
  constexpr uint K = MK_INTER, NG = K / G;
  constexpr int PER_BYTE = MkMppCode<CT>::PER_BYTE;
  for (uint i = tid; i < NG * M; i += SK * 32) {
    const uint g = i / M, m = i % M;
    float sum = 0.0f;
    if (m < rows) {
      const device bfloat4 *q = reinterpret_cast<const device bfloat4 *>(x + ulong(m) * K + g * G);
      for (int j = 0; j < G / 4; ++j) { const float4 v = float4(q[j]); sum += v.x + v.y + v.z + v.w; }
    }
    xsum[i] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto a = tensor(const_cast<device bfloat *>(x), dextents<int, 2>{int(K), int(rows)}, array<int, 2>{1, int(K)});
  tensor<device CT, dextents<int, 2>, tensor_inline> bw(
      const_cast<device uchar *>(w + ulong(expert) * af.w_expert_stride + ulong(n0) * af.w_row_stride),
      dextents<int, 2>{int(K), NT}, array<int, 2>{1, int(af.w_row_stride * PER_BYTE)});
  constexpr auto desc = matmul2d_descriptor(M, NT, G, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto a0 = a.template slice<G, M>(0, 0);
  auto bs0 = bw.template slice<G, NT>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  const device bfloat *se = s + ulong(expert) * af.p_expert_stride, *be = b + ulong(expert) * af.p_expert_stride;
  for (uint g = simd; g < NG; g += SK) {
    auto ag = a.template slice<G, M>(g * G, 0);
    auto bg = bw.template slice<G, NT>(g * G, 0);
    auto t = op.template get_destination_cooperative_tensor<decltype(a0), decltype(bs0), float>();
    op.run(ag, bg, t);
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto idx = acc.get_multidimensional_index(i);
      const ulong pi = ulong(n0 + idx[0]) * af.p_row_stride + g;
      acc[i] = fma(t[i], float(se[pi]), fma(xsum[g * M + idx[1]], float(be[pi]), acc[i]));
    }
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto idx = acc.get_multidimensional_index(i);
    red[(simd * M + idx[1]) * NT + idx[0]] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

[[kernel]] void mk_moe_down(
    const device bfloat *inter [[buffer(0)]],         // [unique][16][640]
    const device bfloat *shared_inter [[buffer(1)]],  // [rows][640]
    const device uchar *dw [[buffer(2)]], const device bfloat *ds [[buffer(3)]], const device bfloat *db [[buffer(4)]],
    const device uchar *sw [[buffer(5)]], const device bfloat *ss [[buffer(6)]], const device bfloat *sb [[buffer(7)]],
    device bfloat *expert_down [[buffer(8)]],          // [route][2560]
    device bfloat *shared_down [[buffer(9)]],          // [rows][2560]
    const device MkMoEPlan *plan [[buffer(10)]],
    constant MkMoEParams &p [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int NT = 32, M = MK_M, SK = 5;
  threadgroup float xsum[10 * M];
  threadgroup float red[SK * M * NT];
  const uint u = tg.y, U = plan->unique, rows = p.rows;
  if (u > U) return;
  const uint n0 = tg.x * NT;
  const bool shared = u == U;
  if (shared) {
    if (p.shared_down.bits == 8 && p.shared_down.group == 128)
      mk_down_tile<uint8_t, 128, SK>(shared_inter, rows, sw, ss, sb, p.shared_down, 0, n0, tid, simd, xsum, red);
    else if (p.shared_down.bits == 8)
      mk_down_tile<uint8_t, 64, SK>(shared_inter, rows, sw, ss, sb, p.shared_down, 0, n0, tid, simd, xsum, red);
    else
      mk_down_tile<uint4b_format, 64, SK>(shared_inter, rows, sw, ss, sb, p.shared_down, 0, n0, tid, simd, xsum, red);
  } else {
    mk_down_tile<uint4b_format, 64, SK>(inter + ulong(u) * M * MK_INTER, rows, dw, ds, db, p.down,
                                        plan->experts[u], n0, tid, simd, xsum, red);
  }
  for (uint i = tid; i < rows * NT; i += SK * 32) {
    const uint m = i / NT, n = i % NT;
    const uint route = shared ? m : plan->route_of[u][m];
    if (route == ~0u) continue;
    float t = 0.0f;
    for (uint s = 0; s < uint(SK); ++s) t += red[(s * M + m) * NT + n];
    (shared ? shared_down : expert_down)[ulong(route) * MK_WIDTH + n0 + n] = bfloat(t);
  }
}

// ---------------------------------------------------------------------------
// Expert phases over lane-major tiles. Per expert: gate/up tiles [20 blocks][40]
// and down tiles [80 blocks][10] (1 KB each), parameters [K/64][N] scales then
// biases. The shared expert's down projection uses its dense tile copy.
struct MkMoETiledParams {
  uint rows, pad0, pad1, pad2;
  ulong gate_stride, down_stride;              // code bytes per expert
  ulong gate_param_stride, down_param_stride;  // BF16 parameter elements per expert
  uint shared_bits, shared_group, shared_padded_n, pad3;
  ulong shared_bias_offset;
};

// Grid (640 / 32, max unique), SK simdgroups split K; RB = 2 for 9..16 rows.
template <int RB, int SK>
[[kernel]] void mk_moe_gate_up_t(
    const device bfloat *x [[buffer(0)]],
    const device uchar *gt [[buffer(1)]], const device uchar *ut [[buffer(2)]],
    const device bfloat *gp [[buffer(3)]], const device bfloat *upar [[buffer(4)]],
    device bfloat *inter [[buffer(5)]],        // [unique][16][640]
    const device MkMoEPlan *plan [[buffer(6)]],
    constant MkMoETiledParams &p [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint RM = 8 * RB, PLANE = SK * RM * 32;
  threadgroup float red[2 * PLANE];
  const uint unique = tg.y, nb = tg.x, rows = p.rows;
  if (unique >= plan->unique) return;
  const ulong expert = plan->experts[unique];
  float rg[8 * RB], ru[8 * RB];
  uint nl = 0, ml = 0;
  mk_tile_accumulate<4, 64, const device bfloat *, 1, RB>(x, MK_WIDTH, MK_WIDTH, rows,
      gt + expert * p.gate_stride + ulong(nb) * 40 * 1024, gp + expert * p.gate_param_stride, 40 * MK_INTER,
      MK_INTER, nb * 32, simd, SK, 40, lane, rg, nl, ml);
  mk_tile_accumulate<4, 64, const device bfloat *, 1, RB>(x, MK_WIDTH, MK_WIDTH, rows,
      ut + expert * p.gate_stride + ulong(nb) * 40 * 1024, upar + expert * p.gate_param_stride, 40 * MK_INTER,
      MK_INTER, nb * 32, simd, SK, 40, lane, ru, nl, ml);
#pragma unroll
  for (int i = 0; i < 8 * RB; ++i) {
    const uint idx = (simd * RM + ml + 8 * (i >> 3)) * 32 + nl + (i & 3) + 16 * ((i >> 2) & 1);
    red[idx] = rg[i];
    red[PLANE + idx] = ru[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < rows * 32; i += SK * 32) {
    const uint m = i / 32, n = i % 32;
    if (plan->route_of[unique][m] == ~0u) continue;
    float g = 0.0f, u = 0.0f;
    for (uint s = 0; s < SK; ++s) { g += red[(s * RM + m) * 32 + n]; u += red[PLANE + (s * RM + m) * 32 + n]; }
    inter[(ulong(unique) * MK_M + m) * MK_INTER + nb * 32 + n] = mk_expert_silu_mul(g, u);
  }
}

// Grid (2560 / 32, max unique + 1), 64 threads: two simdgroups split K.
// y = unique is the shared expert (BF16 input shared_inter[rows]).
template <int RB>
[[kernel]] void mk_moe_down_t(
    const device bfloat *inter [[buffer(0)]], const device bfloat *shared_inter [[buffer(1)]],
    const device uchar *dt [[buffer(2)]], const device bfloat *dp [[buffer(3)]],
    const device uchar *st [[buffer(4)]], const device bfloat *sp [[buffer(5)]],
    device bfloat *expert_down [[buffer(6)]], device bfloat *shared_down [[buffer(7)]],
    const device MkMoEPlan *plan [[buffer(8)]],
    constant MkMoETiledParams &p [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint SK = 2, RM = 8 * RB;
  threadgroup float red[SK * RM * 32];
  const uint unique = tg.y, U = plan->unique, nb = tg.x, rows = p.rows;
  if (unique > U) return;
  const bool shared = unique == U;
  float r[8 * RB];
  uint nl = 0, ml = 0;
  if (!shared) {
    const ulong expert = plan->experts[unique];
    mk_tile_accumulate<4, 64, const device bfloat *, 1, RB>(inter + ulong(unique) * MK_M * MK_INTER, MK_INTER,
        MK_INTER, rows, dt + expert * p.down_stride + ulong(nb) * 10 * 1024, dp + expert * p.down_param_stride,
        10 * MK_WIDTH, MK_WIDTH, nb * 32, simd, SK, 10, lane, r, nl, ml);
  } else {
#define MK_SHARED(B, GS) \
    mk_tile_accumulate<B, GS, const device bfloat *, 1, RB>(shared_inter, MK_INTER, MK_INTER, rows, \
        st + ulong(nb) * 10 * MkTile<B>::BYTES, sp, p.shared_bias_offset, p.shared_padded_n, nb * 32, simd, SK, 10, \
        lane, r, nl, ml)
    switch (p.shared_bits * 1000 + p.shared_group) {
    case 8128: MK_SHARED(8, 128); break;
    case 8064: MK_SHARED(8, 64); break;
    default:   MK_SHARED(4, 64); break;
    }
#undef MK_SHARED
  }
#pragma unroll
  for (int i = 0; i < 8 * RB; ++i)
    red[(simd * RM + ml + 8 * (i >> 3)) * 32 + nl + (i & 3) + 16 * ((i >> 2) & 1)] = r[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < rows * 32; i += SK * 32) {
    const uint m = i / 32, n = i % 32;
    const uint route = shared ? m : plan->route_of[unique][m];
    if (route == ~0u) continue;
    float t = 0.0f;
    for (uint s = 0; s < SK; ++s) t += red[(s * RM + m) * 32 + n];
    (shared ? shared_down : expert_down)[ulong(route) * MK_WIDTH + nb * 32 + n] = bfloat(t);
  }
}

template [[host_name("mk_moe_gate_up_t")]] [[kernel]] void mk_moe_gate_up_t<1, 8>(
    const device bfloat *, const device uchar *, const device uchar *, const device bfloat *, const device bfloat *,
    device bfloat *, const device MkMoEPlan *, constant MkMoETiledParams &, uint2, uint, uint, uint);
template [[host_name("mk_moe_gate_up_t16")]] [[kernel]] void mk_moe_gate_up_t<2, 4>(
    const device bfloat *, const device uchar *, const device uchar *, const device bfloat *, const device bfloat *,
    device bfloat *, const device MkMoEPlan *, constant MkMoETiledParams &, uint2, uint, uint, uint);
template [[host_name("mk_moe_down_t")]] [[kernel]] void mk_moe_down_t<1>(
    const device bfloat *, const device bfloat *, const device uchar *, const device bfloat *, const device uchar *,
    const device bfloat *, device bfloat *, device bfloat *, const device MkMoEPlan *, constant MkMoETiledParams &,
    uint2, uint, uint, uint);
template [[host_name("mk_moe_down_t16")]] [[kernel]] void mk_moe_down_t<2>(
    const device bfloat *, const device bfloat *, const device uchar *, const device bfloat *, const device uchar *,
    const device bfloat *, device bfloat *, device bfloat *, const device MkMoEPlan *, constant MkMoETiledParams &,
    uint2, uint, uint, uint);

// Blocked-MoE bucket jobs (FlashMoEBlocked M16 tiles) on the expert tiles: each
// job is up to 16 packed rows of one expert. Same buffer contracts as the
// blocked gate/up (packed inputs -> packed activated) and down scatter kernels.
struct MkBucketParams {
  uint route_capacity, job_capacity, pad0, pad1;
  ulong gate_stride, down_stride;              // code bytes per expert
  ulong gate_param_stride, down_param_stride;  // BF16 parameter elements per expert
};

inline bool mk_bucket_job(const device uint *offsets, const device FlashMoEBucketJob *jobs,
                          const device uint *job_count, uint job, constant MkBucketParams &p,
                          thread uint &expert, thread uint &begin, thread uint &rows) {
  const uint active = job_count[0];
  if (active > p.job_capacity || job >= active || offsets[512] > p.route_capacity) return false;
  const FlashMoEBucketJob selected = jobs[job];
  if (selected.expert >= 512) return false;
  const uint b = offsets[selected.expert], e = offsets[selected.expert + 1];
  if (b > e || e > p.route_capacity || selected.row_begin < b || selected.row_begin >= e) return false;
  expert = selected.expert;
  begin = selected.row_begin;
  rows = min(16u, e - begin);
  return true;
}

// Grid (640 / 32, job capacity), 128 threads: four simdgroups split K.
[[kernel]] void mk_moe_bucket_gate_up(
    const device bfloat *packed [[buffer(0)]],
    const device uchar *gt [[buffer(1)]], const device uchar *ut [[buffer(2)]],
    const device bfloat *gp [[buffer(3)]], const device bfloat *upar [[buffer(4)]],
    const device uint *offsets [[buffer(5)]], const device FlashMoEBucketJob *jobs [[buffer(6)]],
    const device uint *job_count [[buffer(7)]], device bfloat *activated [[buffer(8)]],
    constant MkBucketParams &p [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint SK = 4, RM = 16, PLANE = SK * RM * 32;
  threadgroup float red[2 * PLANE];
  uint expert = 0, begin = 0, rows = 0;
  if (!mk_bucket_job(offsets, jobs, job_count, tg.y, p, expert, begin, rows)) return;
  const uint nb = tg.x;
  const device bfloat *x = packed + ulong(begin) * MK_WIDTH;
  float rg[16], ru[16];
  uint nl = 0, ml = 0;
  mk_tile_accumulate<4, 64, const device bfloat *, 1, 2>(x, MK_WIDTH, MK_WIDTH, rows,
      gt + ulong(expert) * p.gate_stride + ulong(nb) * 40 * 1024, gp + ulong(expert) * p.gate_param_stride,
      40 * MK_INTER, MK_INTER, nb * 32, simd, SK, 40, lane, rg, nl, ml);
  mk_tile_accumulate<4, 64, const device bfloat *, 1, 2>(x, MK_WIDTH, MK_WIDTH, rows,
      ut + ulong(expert) * p.gate_stride + ulong(nb) * 40 * 1024, upar + ulong(expert) * p.gate_param_stride,
      40 * MK_INTER, MK_INTER, nb * 32, simd, SK, 40, lane, ru, nl, ml);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const uint idx = (simd * RM + ml + 8 * (i >> 3)) * 32 + nl + (i & 3) + 16 * ((i >> 2) & 1);
    red[idx] = rg[i];
    red[PLANE + idx] = ru[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < rows * 32; i += SK * 32) {
    const uint m = i / 32, n = i % 32;
    float g = 0.0f, u = 0.0f;
    for (uint s = 0; s < SK; ++s) { g += red[(s * RM + m) * 32 + n]; u += red[PLANE + (s * RM + m) * 32 + n]; }
    activated[ulong(begin + m) * MK_INTER + nb * 32 + n] = mk_expert_silu_mul(g, u);
  }
}

// Grid (2560 / 32, job capacity), 64 threads: two simdgroups split K.
[[kernel]] void mk_moe_bucket_down(
    const device bfloat *activated [[buffer(0)]],
    const device uchar *dt [[buffer(1)]], const device bfloat *dp [[buffer(2)]],
    const device uint *offsets [[buffer(3)]], const device FlashMoEBucketJob *jobs [[buffer(4)]],
    const device uint *job_count [[buffer(5)]], const device uint *map [[buffer(6)]],
    device bfloat *scattered [[buffer(7)]],
    constant MkBucketParams &p [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint SK = 2, RM = 16;
  threadgroup float red[SK * RM * 32];
  uint expert = 0, begin = 0, rows = 0;
  if (!mk_bucket_job(offsets, jobs, job_count, tg.y, p, expert, begin, rows)) return;
  const uint nb = tg.x;
  float r[16];
  uint nl = 0, ml = 0;
  mk_tile_accumulate<4, 64, const device bfloat *, 1, 2>(activated + ulong(begin) * MK_INTER, MK_INTER, MK_INTER,
      rows, dt + ulong(expert) * p.down_stride + ulong(nb) * 10 * 1024, dp + ulong(expert) * p.down_param_stride,
      10 * MK_WIDTH, MK_WIDTH, nb * 32, simd, SK, 10, lane, r, nl, ml);
#pragma unroll
  for (int i = 0; i < 16; ++i)
    red[(simd * RM + ml + 8 * (i >> 3)) * 32 + nl + (i & 3) + 16 * ((i >> 2) & 1)] = r[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < rows * 32; i += SK * 32) {
    const uint m = i / 32, n = i % 32;
    const uint route = map[begin + m];
    if (route >= p.route_capacity) continue;
    float t = 0.0f;
    for (uint s = 0; s < SK; ++s) t += red[(s * RM + m) * 32 + n];
    scattered[ulong(route) * MK_WIDTH + nb * 32 + n] = bfloat(t);
  }
}

// ---------------------------------------------------------------------------
// Combine + shared + HC inject (+ next grouped RMS norm).
// Grid (rows, 4 streams), 640 threads, 4 columns per thread.
[[kernel]] void mk_moe_combine(
    const device bfloat *expert_down [[buffer(0)]],
    const device bfloat *shared_down [[buffer(1)]],
    const device bfloat *shared_logit [[buffer(2)]],
    const device MkMoEPlan *plan [[buffer(3)]],
    const device bfloat *gates [[buffer(4)]],
    device bfloat *hyper [[buffer(5)]],
    const device bfloat *norm_weight [[buffer(6)]],
    device bfloat *normalized [[buffer(7)]],
    constant MkCombineParams &p [[buffer(8)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  const uint row = grid.x, stream = grid.y;
  if (row >= p.rows) return;
  const bfloat raw_gate = shared_logit[row];
  const bfloat shared_scale = mk_sigmoid_precise(raw_gate);
  const bfloat gate = gates[row * 4 + stream];
  const ulong base = (ulong(row) * 4 + stream) * MK_WIDTH;
  bfloat weights[MK_SEL];
  for (uint slot = 0; slot < MK_SEL; ++slot) weights[slot] = bfloat(plan->route_weight[row * MK_SEL + slot]);
  bfloat values[4];
  float square_sum = 0.0f;
  for (uint e = 0; e < 4; ++e) {
    const uint column = tid * 4 + e;
    bfloat partials[8];
    for (uint y = 0; y < 8; ++y) {
      bfloat partial = bfloat(0.0f);
      for (uint slot = y; slot < MK_SEL; slot += 8) {
        const uint route = row * MK_SEL + slot;
        const bfloat down = expert_down[ulong(route) * MK_WIDTH + column];
        partial = partial + down * weights[slot];
      }
      partials[y] = partial;
    }
    bfloat routed = partials[0];
    for (uint y = 1; y < 8; ++y) routed = routed + partials[y];
    const bfloat shared = shared_down[ulong(row) * MK_WIDTH + column];
    const bfloat branch = routed + shared * shared_scale;
    const bfloat product = bfloat(float(branch) * float(gate));
    values[e] = bfloat(float(hyper[base + column]) + float(product));
    const float value = float(values[e]);
    square_sum += value * value;
  }
  for (uint e = 0; e < 4; ++e) hyper[base + tid * 4 + e] = values[e];
  if (!p.has_norm) return;
  square_sum = simd_sum(square_sum);
  threadgroup float partials_tg[32];
  if (simd == 0) partials_tg[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) partials_tg[simd] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    const float total = simd_sum(partials_tg[lane]);
    if (lane == 0) partials_tg[0] = metal::precise::rsqrt(total / float(MK_WIDTH) + p.epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inverse = partials_tg[0];
  for (uint e = 0; e < 4; ++e) {
    const uint column = tid * 4 + e;
    const float raw = float(norm_weight[stream * MK_WIDTH + column]);
    const float scale = p.norm_convention == 0 ? 1.0f + raw : raw;
    normalized[base + column] = bfloat(float(values[e]) * inverse * scale);
  }
}

#define MK_ROUTER(R) \
  template [[host_name("mk_moe_router_r" #R)]] [[kernel]] void mk_moe_router<R>( \
      const device bfloat *, const device bfloat *, const device uchar *, const device bfloat *, \
      const device bfloat *, const device uchar *, const device bfloat *, const device bfloat *, \
      const device uchar *, const device bfloat *, const device bfloat *, device bfloat *, \
      device bfloat *, device bfloat *, constant MkRouterParams &, uint, uint, uint);
MK_ROUTER(1) MK_ROUTER(2) MK_ROUTER(3) MK_ROUTER(4) MK_ROUTER(5) MK_ROUTER(6) MK_ROUTER(7) MK_ROUTER(8)
