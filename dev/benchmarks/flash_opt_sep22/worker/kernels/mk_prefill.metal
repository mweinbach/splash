// Megakernel prefill phases.
#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Same contract as private_moe_poison_route32 (canonical routes excluded from
// every bucket get a NaN down row and sticky flags 1|4), one thread per route.
kernel void mk_moe_poison_routes(device const uint *inverse [[buffer(0)]], device bfloat *output [[buffer(1)]],
                                 device atomic_uint *diag [[buffer(2)]],
                                 constant FlashMoEBlockedDownParams &params [[buffer(3)]],
                                 uint route [[thread_position_in_grid]]) {
  if (route >= params.route_capacity || inverse[route] < params.route_capacity) return;
  atomic_fetch_or_explicit(diag, 5u, memory_order_relaxed);
  const bfloat nan = bfloat(as_type<float>(0x7fc00000u));
  for (uint column = 0; column < 2560; ++column) output[ulong(route) * 2560 + column] = nan;
}

inline bfloat mk_pf_sigmoid(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

inline bool mk_pf_job(device const uint *offsets, device const FlashMoEBucketJob *jobs,
                      device const uint *job_count, uint job, uint capacity, uint routes,
                      thread uint &expert, thread uint &begin, thread uint &end) {
  const uint active = job_count[0];
  if (active > capacity || job >= active || offsets[512] > routes) return false;
  const FlashMoEBucketJob selected = jobs[job];
  if (selected.expert >= 512) return false;
  const uint b = offsets[selected.expert], e = offsets[selected.expert + 1];
  if (b > e || e > routes || selected.row_begin < b || selected.row_begin >= e) return false;
  expert = selected.expert; begin = selected.row_begin; end = e;
  return true;
}

// Gate/up for a 64-row bucket tile and 64 intermediate features: staged
// B = [gate rows n0..n0+63 ; up rows n0..n0+63] (128 x BK per K chunk),
// C[64 x 128] on 8 simdgroups, SwiGLU epilogue through threadgroup memory.
// Output: intermediate[packed row][640].
// VEC: vectorized staging stores. NOSTAGE: timing upper bound (skips staging).
// ASTAGE: A rows are staged in threadgroup memory too (BK must be 32).
template <int BK, bool VEC, bool NOSTAGE, bool ASTAGE>
inline void mk_moe_prefill_gate_up_impl(
    device bfloat *input, device const uchar *gw, device const bfloat *gs, device const bfloat *gb,
    device const uchar *uw, device const bfloat *us, device const bfloat *ub,
    device const uint *offsets, device const FlashMoEBucketJob *jobs, device const uint *job_count,
    device bfloat *output, constant FlashMoEBlockedGateParams &params, uint3 group, uint tid,
    threadgroup bfloat *stage) {
  constexpr ushort M = 64, N = 128;
  constexpr uint CHUNKS = 2560 / BK;
  constexpr uint CODES = BK * N / 256;          // codes staged per thread per chunk (32 or 16)
  constexpr uint PER_ROW = BK / CODES;          // threads per staged row (2)
  const constant FlashMoEFusedParams &p = params.affine;
  uint expert = 0, begin = 0, end = 0;
  if (!mk_pf_job(offsets, jobs, job_count, group.y, params.job_capacity, params.route_capacity,
                 expert, begin, end)) return;
  const uint n0 = group.x * 64;
  const uint srow = tid / PER_ROW, part = (tid % PER_ROW) * CODES;
  const bool up = srow >= 64;
  const uint n = n0 + (up ? srow - 64 : srow);
  const device uchar *rw = (up ? uw : gw) + ulong(expert) * (up ? p.up_weight_expert_stride_bytes : p.gate_weight_expert_stride_bytes) +
      ulong(n) * (up ? p.up_weight_row_stride_bytes : p.gate_weight_row_stride_bytes);
  const ulong pe = ulong(expert) * ((up ? p.up_parameter_expert_stride_bytes : p.gate_parameter_expert_stride_bytes) / 2);
  const ulong pr = ulong(n) * ((up ? p.up_parameter_row_stride_bytes : p.gate_parameter_row_stride_bytes) / 2);
  const device bfloat *rs = (up ? us : gs) + pe + pr, *rb = (up ? ub : gb) + pe + pr;
  const uint rows = min(uint(M), end - begin);
  threadgroup bfloat *bstage = stage;
  threadgroup bfloat *astage = stage + 2 * N * BK;
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<8>> op;
  auto b0 = tensor(bstage, dextents<int, 2>{BK, N}, array<int, 2>{1, BK}).template slice<BK, N>(0, 0);
  auto b1 = tensor(bstage + N * BK, dextents<int, 2>{BK, N}, array<int, 2>{1, BK}).template slice<BK, N>(0, 0);
  auto adev = tensor(input + ulong(begin) * 2560, dextents<int, 2>{2560, int(rows)}, array<int, 2>{1, 2560});
  auto as0 = tensor(astage, dextents<int, 2>{BK, M}, array<int, 2>{1, BK}).template slice<BK, M>(0, 0);
  auto as1 = tensor(astage + M * BK, dextents<int, 2>{BK, M}, array<int, 2>{1, BK}).template slice<BK, M>(0, 0);
  auto a0 = adev.template slice<BK, M>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  auto stage_chunk = [&](uint k0, uint slot) {
    if (ASTAGE) {
      // 64 rows x BK(32) bf16 = 4 KB: 256 threads x 16 B.
      const uint ar = tid / 4, ac = (tid % 4) * 8;
      uint4 v = uint4(0);
      if (ar < rows) v = *reinterpret_cast<const device uint4 *>(input + ulong(begin + ar) * 2560 + k0 + ac);
      *reinterpret_cast<threadgroup uint4 *>(astage + slot * M * BK + ar * BK + ac) = v;
    }
    if (NOSTAGE) return;
    const uint k = k0 + part;
    const float sc = float(rs[k / 64]), bi = float(rb[k / 64]);
    threadgroup bfloat *d = bstage + slot * N * BK + srow * BK + part;
    if (CODES == 32) {
      const uint4 w = *reinterpret_cast<const device uint4 *>(rw + k / 2);
      const uint words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (uint i = 0; i < 4; ++i) {
        if (VEC) {
          bfloat4 lo, hi;
#pragma unroll
          for (uint j = 0; j < 4; ++j) lo[j] = bfloat(fma(float((words[i] >> (4 * j)) & 15u), sc, bi));
#pragma unroll
          for (uint j = 0; j < 4; ++j) hi[j] = bfloat(fma(float((words[i] >> (4 * (j + 4))) & 15u), sc, bi));
          *reinterpret_cast<threadgroup bfloat4 *>(d + i * 8) = lo;
          *reinterpret_cast<threadgroup bfloat4 *>(d + i * 8 + 4) = hi;
        } else {
#pragma unroll
          for (uint j = 0; j < 8; ++j) d[i * 8 + j] = bfloat(fma(float((words[i] >> (4 * j)) & 15u), sc, bi));
        }
      }
    } else {
      const uint2 w = *reinterpret_cast<const device uint2 *>(rw + k / 2);
      const uint words[2] = {w.x, w.y};
#pragma unroll
      for (uint i = 0; i < 2; ++i) {
        bfloat4 lo, hi;
#pragma unroll
        for (uint j = 0; j < 4; ++j) lo[j] = bfloat(fma(float((words[i] >> (4 * j)) & 15u), sc, bi));
#pragma unroll
        for (uint j = 0; j < 4; ++j) hi[j] = bfloat(fma(float((words[i] >> (4 * (j + 4))) & 15u), sc, bi));
        *reinterpret_cast<threadgroup bfloat4 *>(d + i * 8) = lo;
        *reinterpret_cast<threadgroup bfloat4 *>(d + i * 8 + 4) = hi;
      }
    }
  };
  stage_chunk(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
    if (chunk + 1 < CHUNKS) stage_chunk((chunk + 1) * BK, (chunk + 1) & 1);
    if (ASTAGE) {
      if (chunk & 1) op.run(as1, b1, acc); else op.run(as0, b0, acc);
    } else {
      auto ac = adev.template slice<BK, M>(chunk * BK, 0);
      if (chunk & 1) op.run(ac, b1, acc); else op.run(ac, b0, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float *c = reinterpret_cast<threadgroup float *>(stage);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    c[index[1] * N + index[0]] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < rows * 64; i += 256) {
    const uint m = i / 64, nn = i % 64;
    const bfloat gate = bfloat(c[m * N + nn]), upv = bfloat(c[m * N + 64 + nn]);
    const bfloat silu = bfloat(float(gate) * float(mk_pf_sigmoid(gate)));
    output[ulong(begin + m) * 640 + n0 + nn] = bfloat(float(silu) * float(upv));
  }
}

// Register-pipelined variant (BK = 32): the codes/A for chunk c+2 load into
// registers while chunk c multiplies; chunk c+1 is dequantized from registers
// loaded one iteration earlier. A and B are both double-buffered (24 KB).
// NT = B rows per tile (128 = 64 gate + 64 up, or 128 down features);
// GATE selects the SwiGLU/packed output epilogue vs the scattered down output.
template <uint K, bool GATE, uint ABL = 0>
inline void mk_moe_prefill_pipelined(
    device bfloat *input, device const uchar *w0, device const bfloat *s0, device const bfloat *b0,
    device const uchar *w1, device const bfloat *s1, device const bfloat *b1,
    ulong w0e, ulong w0r, ulong p0e, ulong p0r, ulong w1e, ulong w1r, ulong p1e, ulong p1r,
    device const uint *offsets, device const FlashMoEBucketJob *jobs, device const uint *job_count,
    uint job_capacity, uint route_capacity, device const uint *map, device bfloat *output,
    device uint *diag, uint3 group, uint tid, threadgroup bfloat *stage) {
  constexpr ushort M = 64, N = 128, BK = 32;
  constexpr uint CHUNKS = K / BK;
  uint expert = 0, begin = 0, end = 0;
  if (!mk_pf_job(offsets, jobs, job_count, group.y, job_capacity, route_capacity, expert, begin, end)) return;
  const uint n0 = group.x * (GATE ? 64 : 128);
  // B staging: thread -> row tid/2 (0..127), 16 codes at (tid%2)*16.
  const uint srow = tid / 2, part = (tid % 2) * 16;
  const bool second = GATE && srow >= 64;
  const uint n = GATE ? n0 + (second ? srow - 64 : srow) : n0 + srow;
  const device uchar *rw = (second ? w1 : w0) + ulong(expert) * (second ? w1e : w0e) + ulong(n) * (second ? w1r : w0r);
  const ulong pbase = ulong(expert) * ((second ? p1e : p0e) / 2) + ulong(n) * ((second ? p1r : p0r) / 2);
  const device bfloat *rs = (second ? s1 : s0) + pbase, *rb = (second ? b1 : b0) + pbase;
  // A staging: thread -> row tid/4 (0..63), 8 values at (tid%4)*8.
  const uint rows = min(uint(M), end - begin);
  const uint ar = tid / 4, acol = (tid % 4) * 8;
  const device bfloat *arow = input + ulong(begin + min(ar, rows - 1)) * K + acol;
  const bool avalid = ar < rows;
  threadgroup bfloat *bst = stage;              // 2 x [128][32]
  threadgroup bfloat *ast = stage + 2 * N * BK; // 2 x [64][32]
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<8>> op;
  auto bt0 = tensor(bst, dextents<int, 2>{BK, N}, array<int, 2>{1, BK}).template slice<BK, N>(0, 0);
  auto bt1 = tensor(bst + N * BK, dextents<int, 2>{BK, N}, array<int, 2>{1, BK}).template slice<BK, N>(0, 0);
  auto at0 = tensor(ast, dextents<int, 2>{BK, M}, array<int, 2>{1, BK}).template slice<BK, M>(0, 0);
  auto at1 = tensor(ast + M * BK, dextents<int, 2>{BK, M}, array<int, 2>{1, BK}).template slice<BK, M>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(at0), decltype(bt0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  uint2 code; uint4 av; float sc, bi;
  auto fetch = [&](uint chunk) {
    if ((ABL & 1) && chunk > 1) return;
    const uint k = chunk * BK + part;
    code = *reinterpret_cast<const device uint2 *>(rw + k / 2);
    sc = float(rs[k / 64]); bi = float(rb[k / 64]);
    av = avalid ? *reinterpret_cast<const device uint4 *>(arow + chunk * BK) : uint4(0);
  };
  auto commit = [&](uint slot) {
    threadgroup bfloat *d = bst + slot * N * BK + srow * BK + part;
    const uint words[2] = {code.x, code.y};
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      bfloat4 lo, hi;
      if (ABL & 2) {
        *reinterpret_cast<threadgroup uint2 *>(d + i * 8) = uint2(words[i], words[i]);
        *reinterpret_cast<threadgroup uint2 *>(d + i * 8 + 4) = uint2(words[i], words[i]);
        continue;
      }
#pragma unroll
      for (uint j = 0; j < 4; ++j) lo[j] = bfloat(fma(float((words[i] >> (4 * j)) & 15u), sc, bi));
#pragma unroll
      for (uint j = 0; j < 4; ++j) hi[j] = bfloat(fma(float((words[i] >> (4 * (j + 4))) & 15u), sc, bi));
      *reinterpret_cast<threadgroup bfloat4 *>(d + i * 8) = lo;
      *reinterpret_cast<threadgroup bfloat4 *>(d + i * 8 + 4) = hi;
    }
    *reinterpret_cast<threadgroup uint4 *>(ast + slot * M * BK + ar * BK + acol) = av;
  };
  fetch(0);
  commit(0);
  fetch(1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
    if (chunk + 1 < CHUNKS) {
      commit((chunk + 1) & 1);
      if (chunk + 2 < CHUNKS) fetch(chunk + 2);
    }
    if (!(ABL & 4)) { if (chunk & 1) op.run(at1, bt1, acc); else op.run(at0, bt0, acc); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (GATE) {
    // Accumulator element i (column n < 64, gate) and i + 8 (column n + 64,
    // up) of the same row belong to this thread for the 64x128 SG8 layout.
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if ((i & 15) >= 8 || !acc.is_valid_element(i)) continue;
      const auto index = acc.get_multidimensional_index(i);
      const auto pair = acc.get_multidimensional_index(i + 8);
      const uint m = index[1], nn = index[0];
      if (pair[0] != index[0] + 64 || pair[1] != index[1] || nn >= 64) {
        atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(diag), 2u, memory_order_relaxed);
        continue;
      }
      if (m >= rows) continue;
      const bfloat gate = bfloat(acc[i]), upv = bfloat(acc[i + 8]);
      const bfloat silu = bfloat(float(gate) * float(mk_pf_sigmoid(gate)));
      output[ulong(begin + m) * 640 + n0 + nn] = bfloat(float(silu) * float(upv));
    }
  } else {
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto index = acc.get_multidimensional_index(i);
      const uint m = index[1];
      if (m >= rows) continue;
      const uint route = map[begin + m];
      if (route >= route_capacity) continue;
      output[ulong(route) * 2560 + n0 + index[0]] = bfloat(acc[i]);
    }
  }
}

[[kernel]] void mk_moe_prefill_gate_up_pipe(
    device bfloat *input [[buffer(0)]],
    device const uchar *gw [[buffer(1)]], device const bfloat *gs [[buffer(2)]],
    device const bfloat *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]],
    device const bfloat *us [[buffer(5)]], device const bfloat *ub [[buffer(6)]],
    device const uint *offsets [[buffer(7)]], device const FlashMoEBucketJob *jobs [[buffer(8)]],
    device const uint *job_count [[buffer(9)]], device bfloat *output [[buffer(10)]],
    device uint *diag [[buffer(11)]], constant FlashMoEBlockedGateParams &params [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  alignas(16) threadgroup bfloat stage[2 * 128 * 32 + 2 * 64 * 32];
  const constant FlashMoEFusedParams &p = params.affine;
  mk_moe_prefill_pipelined<2560, true>(input, gw, gs, gb, uw, us, ub,
      p.gate_weight_expert_stride_bytes, p.gate_weight_row_stride_bytes,
      p.gate_parameter_expert_stride_bytes, p.gate_parameter_row_stride_bytes,
      p.up_weight_expert_stride_bytes, p.up_weight_row_stride_bytes,
      p.up_parameter_expert_stride_bytes, p.up_parameter_row_stride_bytes,
      offsets, jobs, job_count, params.job_capacity, params.route_capacity, offsets, output, diag, group, tid, stage);
}

[[kernel]] void mk_moe_prefill_down_pipe(
    device bfloat *input [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const bfloat *s [[buffer(2)]], device const bfloat *b [[buffer(3)]],
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]],
    device const uint *job_count [[buffer(6)]], device const uint *map [[buffer(7)]],
    device bfloat *output [[buffer(8)]], device uint *diag [[buffer(9)]],
    constant FlashMoEBlockedDownParams &params [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  alignas(16) threadgroup bfloat stage[2 * 128 * 32 + 2 * 64 * 32];
  const constant FlashMoEDownFusedParams &p = params.affine;
  mk_moe_prefill_pipelined<640, false>(input, w, s, b, w, s, b,
      p.weight_expert_stride_bytes, p.weight_row_stride_bytes,
      p.parameter_expert_stride_bytes, p.parameter_row_stride_bytes,
      p.weight_expert_stride_bytes, p.weight_row_stride_bytes,
      p.parameter_expert_stride_bytes, p.parameter_row_stride_bytes,
      offsets, jobs, job_count, params.job_capacity, params.route_capacity, map, output, diag, group, tid, stage);
}


#define MK_PF_ABL(NAME, ABL) \
[[kernel]] void NAME( \
    device bfloat *input [[buffer(0)]], \
    device const uchar *gw [[buffer(1)]], device const bfloat *gs [[buffer(2)]], \
    device const bfloat *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]], \
    device const bfloat *us [[buffer(5)]], device const bfloat *ub [[buffer(6)]], \
    device const uint *offsets [[buffer(7)]], device const FlashMoEBucketJob *jobs [[buffer(8)]], \
    device const uint *job_count [[buffer(9)]], device bfloat *output [[buffer(10)]], \
    device uint *diag [[buffer(11)]], constant FlashMoEBlockedGateParams &params [[buffer(12)]], \
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
  alignas(16) threadgroup bfloat stage[2 * 128 * 32 + 2 * 64 * 32]; \
  const constant FlashMoEFusedParams &p = params.affine; \
  mk_moe_prefill_pipelined<2560, true, ABL>(input, gw, gs, gb, uw, us, ub, \
      p.gate_weight_expert_stride_bytes, p.gate_weight_row_stride_bytes, \
      p.gate_parameter_expert_stride_bytes, p.gate_parameter_row_stride_bytes, \
      p.up_weight_expert_stride_bytes, p.up_weight_row_stride_bytes, \
      p.up_parameter_expert_stride_bytes, p.up_parameter_row_stride_bytes, \
      offsets, jobs, job_count, params.job_capacity, params.route_capacity, offsets, output, diag, group, tid, stage); \
}
MK_PF_ABL(mk_abl_nofetch, 1) MK_PF_ABL(mk_abl_nodequant, 2) MK_PF_ABL(mk_abl_nomma, 4) MK_PF_ABL(mk_abl_mmaonly, 3) MK_PF_ABL(mk_abl_nothing, 7)

#define MK_PF_GATE(NAME, BK, VEC, NOSTAGE, ASTAGE) \
[[kernel]] void NAME( \
    device bfloat *input [[buffer(0)]], \
    device const uchar *gw [[buffer(1)]], device const bfloat *gs [[buffer(2)]], \
    device const bfloat *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]], \
    device const bfloat *us [[buffer(5)]], device const bfloat *ub [[buffer(6)]], \
    device const uint *offsets [[buffer(7)]], device const FlashMoEBucketJob *jobs [[buffer(8)]], \
    device const uint *job_count [[buffer(9)]], device bfloat *output [[buffer(10)]], \
    device uint *diag [[buffer(11)]], constant FlashMoEBlockedGateParams &params [[buffer(12)]], \
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
  alignas(16) threadgroup bfloat stage[(2 * 128 * BK + (ASTAGE ? 2 * 64 * BK : 0)) > 8192 * 2 ? \
      (2 * 128 * BK + (ASTAGE ? 2 * 64 * BK : 0)) : 8192 * 2]; \
  mk_moe_prefill_gate_up_impl<BK, VEC, NOSTAGE, ASTAGE>(input, gw, gs, gb, uw, us, ub, offsets, jobs, \
      job_count, output, params, group, tid, stage); \
  (void)diag; \
}
MK_PF_GATE(mk_moe_prefill_gate_up, 64, false, false, false)
MK_PF_GATE(mk_moe_prefill_gate_up_vec, 64, true, false, false)
MK_PF_GATE(mk_moe_prefill_gate_up_nostage, 64, true, true, false)
MK_PF_GATE(mk_moe_prefill_gate_up_bk32, 32, true, false, false)
MK_PF_GATE(mk_moe_prefill_gate_up_bk32a, 32, true, false, true)
MK_PF_GATE(mk_moe_prefill_gate_up_bk32a_nostage, 32, true, true, true)

// Down for a 64-row bucket tile and 128 output features, K = 640 (10 chunks).
[[kernel]] void mk_moe_prefill_down(
    device bfloat *input [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const bfloat *s [[buffer(2)]], device const bfloat *b [[buffer(3)]],
    device const uint *offsets [[buffer(4)]],
    device const FlashMoEBucketJob *jobs [[buffer(5)]],
    device const uint *job_count [[buffer(6)]], device const uint *map [[buffer(7)]],
    device bfloat *output [[buffer(8)]], device uint *diag [[buffer(9)]],
    constant FlashMoEBlockedDownParams &params [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr ushort M = 64, N = 128, BK = 64;
  alignas(16) threadgroup bfloat stage[2 * N * BK];
  const constant FlashMoEDownFusedParams &p = params.affine;
  uint expert = 0, begin = 0, end = 0;
  if (!mk_pf_job(offsets, jobs, job_count, group.y, params.job_capacity, params.route_capacity,
                 expert, begin, end)) return;
  const uint n0 = group.x * N;
  const uint srow = tid / 2, half_k = (tid % 2) * 32;
  const device uchar *rw = w + ulong(expert) * p.weight_expert_stride_bytes + ulong(n0 + srow) * p.weight_row_stride_bytes;
  const ulong pbase = ulong(expert) * (p.parameter_expert_stride_bytes / 2) + ulong(n0 + srow) * (p.parameter_row_stride_bytes / 2);
  const device bfloat *rs = s + pbase, *rb = b + pbase;
  const uint rows = min(uint(M), end - begin);
  auto a = tensor(input + ulong(begin) * 640, dextents<int, 2>{640, int(rows)}, array<int, 2>{1, 640});
  constexpr auto descriptor = matmul2d_descriptor(M, N, BK, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<8>> op;
  auto b0 = tensor(stage, dextents<int, 2>{BK, N}, array<int, 2>{1, BK}).template slice<BK, N>(0, 0);
  auto b1 = tensor(stage + N * BK, dextents<int, 2>{BK, N}, array<int, 2>{1, BK}).template slice<BK, N>(0, 0);
  auto a0 = a.template slice<BK, M>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  auto stage_chunk = [&](uint k0, threadgroup bfloat *dst) {
    const uint k = k0 + half_k;
    const uint4 wv = *reinterpret_cast<const device uint4 *>(rw + k / 2);
    const float sc = float(rs[k / 64]), bi = float(rb[k / 64]);
    const uint words[4] = {wv.x, wv.y, wv.z, wv.w};
    threadgroup bfloat *d = dst + srow * BK + half_k;
#pragma unroll
    for (uint i = 0; i < 4; ++i)
#pragma unroll
      for (uint j = 0; j < 8; ++j)
        d[i * 8 + j] = bfloat(fma(float((words[i] >> (4 * j)) & 15u), sc, bi));
  };
  stage_chunk(0, stage);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint chunk = 0; chunk < 10; ++chunk) {
    if (chunk + 1 < 10) stage_chunk((chunk + 1) * BK, stage + ((chunk + 1) & 1) * N * BK);
    auto ac = a.template slice<BK, M>(chunk * BK, 0);
    if (chunk & 1) op.run(ac, b1, acc); else op.run(ac, b0, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    const uint m = index[1], n = n0 + index[0];
    if (m >= rows) continue;
    const uint route = map[begin + m];
    if (route >= params.route_capacity) continue;
    output[ulong(route) * 2560 + n] = bfloat(acc[i]);
  }
  (void)diag;
}
