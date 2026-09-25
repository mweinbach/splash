// Prefill routed-expert GEMMs over 64-row bucket jobs (packed rows per expert).
// Each simdgroup owns one 16-row block and half of the tile's features, so row
// blocks past the bucket's end issue no matrix work. Weights are dequantized
// to BF16 in threadgroup memory (double-buffered, register-pipelined).
//   gate_up: tile = 64 features (gate + up); SG (rb, h) -> rows rb*16.., features h*32..
//   down:    tile = 128 outputs;             SG (rb, h) -> rows rb*16.., outputs h*64..
#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

inline bfloat pf2_sigmoid(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

inline bool pf2_job(device const uint *offsets, device const FlashMoEBucketJob *jobs,
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

// Stage one 32-code chunk of 128 B-rows: thread -> B-row tid/2, codes (tid%2)*16.
struct Pf2Stage {
  const device uchar *row;
  const device bfloat *scales, *biases;
  uint part;
  uint2 code;
  float sc, bi;
  inline void fetch(uint chunk) {
    const uint k = chunk * 32 + part;
    code = *reinterpret_cast<const device uint2 *>(row + k / 2);
    sc = float(scales[k / 64]);
    bi = float(biases[k / 64]);
  }
  inline void commit(threadgroup bfloat *d) const {
    const uint words[2] = {code.x, code.y};
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

// K: input width; BR: B-rows per simdgroup slice (64 for both kernels).
template <uint K, bool GATE>
inline void pf2_body(device bfloat *input, Pf2Stage st, uint begin, uint rows, uint n0,
                     device const uint *map, uint route_capacity, device bfloat *output,
                     uint tid, uint simd, threadgroup bfloat *bst) {
  constexpr ushort BK = 32, BR = 64, M = 16;
  constexpr uint CHUNKS = K / BK;
  const uint rb = simd / 2, h = simd % 2;
  const bool active = rb * M < rows;
  const uint srow = tid / 2;
  // A staging: thread -> row tid/4 (0..63), 8 values at (tid%4)*8.
  threadgroup bfloat *ast = bst + 2 * 128 * BK;
  const uint ar = tid / 4, acol = (tid % 4) * 8;
  const bool avalid = ar < rows;
  const device bfloat *arow = input + ulong(begin + min(ar, rows - 1)) * K + acol;
  uint4 av;
  auto afetch = [&](uint chunk) { av = avalid ? *reinterpret_cast<const device uint4 *>(arow + chunk * BK) : uint4(0); };
  auto acommit = [&](uint slot) { *reinterpret_cast<threadgroup uint4 *>(ast + slot * 64 * BK + ar * BK + acol) = av; };
  auto a = tensor(input + ulong(begin) * K, dextents<int, 2>{int(K), int(rows)}, array<int, 2>{1, int(K)});
  constexpr auto desc = matmul2d_descriptor(M, BR, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto b0 = tensor(bst + h * BR * BK, dextents<int, 2>{BK, BR}, array<int, 2>{1, BK}).template slice<BK, BR>(0, 0);
  auto b1 = tensor(bst + 128 * BK + h * BR * BK, dextents<int, 2>{BK, BR}, array<int, 2>{1, BK}).template slice<BK, BR>(0, 0);
  auto at0 = tensor(ast + rb * M * BK, dextents<int, 2>{BK, M}, array<int, 2>{1, BK}).template slice<BK, M>(0, 0);
  auto at1 = tensor(ast + 64 * BK + rb * M * BK, dextents<int, 2>{BK, M}, array<int, 2>{1, BK}).template slice<BK, M>(0, 0);
  auto a0 = a.template slice<BK, M>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  st.fetch(0); afetch(0);
  st.commit(bst + srow * BK + st.part); acommit(0);
  st.fetch(1); afetch(1);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
    if (chunk + 1 < CHUNKS) {
      st.commit(bst + ((chunk + 1) & 1) * 128 * BK + srow * BK + st.part);
      acommit((chunk + 1) & 1);
      if (chunk + 2 < CHUNKS) { st.fetch(chunk + 2); afetch(chunk + 2); }
    }
    if (active) {
      if (chunk & 1) op.run(at1, b1, acc); else op.run(at0, b0, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!active) return;
  const auto id0 = acc.get_multidimensional_index(ushort(0));
  const uint nl = uint(id0[0]), ml = uint(id0[1]);
  if (GATE) {
#pragma unroll
    for (ushort i = 0; i < 16; ++i) {
      const uint n = nl + (i & 3) + 16 * (i >> 3), m = rb * M + ml + 8 * ((i >> 2) & 1);
      if (m >= rows) continue;
      const bfloat gate = bfloat(acc[i]), upv = bfloat(acc[i + 16]);
      const bfloat silu = bfloat(float(gate) * float(pf2_sigmoid(gate)));
      output[ulong(begin + m) * 640 + n0 + h * 32 + n] = bfloat(float(silu) * float(upv));
    }
  } else {
#pragma unroll
    for (ushort i = 0; i < 32; ++i) {
      const uint n = nl + (i & 3) + 16 * (i >> 3), m = rb * M + ml + 8 * ((i >> 2) & 1);
      if (m >= rows) continue;
      const uint route = map[begin + m];
      if (route >= route_capacity) continue;
      output[ulong(route) * 2560 + n0 + h * 64 + n] = bfloat(acc[i]);
    }
  }
}

// Single-buffered stage (12 KB, two threadgroups per core): the next chunk's
// codes and activations load into registers while the current chunk multiplies.
template <uint K, bool GATE>
inline void pf2s_body(device bfloat *input, Pf2Stage st, uint begin, uint rows, uint n0,
                      device const uint *map, uint route_capacity, device bfloat *output,
                      uint tid, uint simd, threadgroup bfloat *bst) {
  constexpr ushort BK = 32, BR = 64, M = 16;
  constexpr uint CHUNKS = K / BK;
  const uint rb = simd / 2, h = simd % 2;
  const bool active = rb * M < rows;
  const uint srow = tid / 2;
  threadgroup bfloat *ast = bst + 128 * BK;
  const uint ar = tid / 4, acol = (tid % 4) * 8;
  const bool avalid = ar < rows;
  const device bfloat *arow = input + ulong(begin + min(ar, rows - 1)) * K + acol;
  uint4 av;
  auto afetch = [&](uint chunk) { av = avalid ? *reinterpret_cast<const device uint4 *>(arow + chunk * BK) : uint4(0); };
  auto a = tensor(input + ulong(begin) * K, dextents<int, 2>{int(K), int(rows)}, array<int, 2>{1, int(K)});
  constexpr auto desc = matmul2d_descriptor(M, BR, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto b0 = tensor(bst + h * BR * BK, dextents<int, 2>{BK, BR}, array<int, 2>{1, BK}).template slice<BK, BR>(0, 0);
  auto at0 = tensor(ast + rb * M * BK, dextents<int, 2>{BK, M}, array<int, 2>{1, BK}).template slice<BK, M>(0, 0);
  auto a0 = a.template slice<BK, M>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) if (acc.is_valid_element(i)) acc[i] = 0.0f;
  st.fetch(0); afetch(0);
  for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
    st.commit(bst + srow * BK + st.part);
    *reinterpret_cast<threadgroup uint4 *>(ast + ar * BK + acol) = av;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chunk + 1 < CHUNKS) { st.fetch(chunk + 1); afetch(chunk + 1); }
    if (active) op.run(at0, b0, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!active) return;
  const auto id0 = acc.get_multidimensional_index(ushort(0));
  const uint nl = uint(id0[0]), ml = uint(id0[1]);
  if (GATE) {
#pragma unroll
    for (ushort i = 0; i < 16; ++i) {
      const uint n = nl + (i & 3) + 16 * (i >> 3), m = rb * M + ml + 8 * ((i >> 2) & 1);
      if (m >= rows) continue;
      const bfloat gate = bfloat(acc[i]), upv = bfloat(acc[i + 16]);
      const bfloat silu = bfloat(float(gate) * float(pf2_sigmoid(gate)));
      output[ulong(begin + m) * 640 + n0 + h * 32 + n] = bfloat(float(silu) * float(upv));
    }
  } else {
#pragma unroll
    for (ushort i = 0; i < 32; ++i) {
      const uint n = nl + (i & 3) + 16 * (i >> 3), m = rb * M + ml + 8 * ((i >> 2) & 1);
      if (m >= rows) continue;
      const uint route = map[begin + m];
      if (route >= route_capacity) continue;
      output[ulong(route) * 2560 + n0 + h * 64 + n] = bfloat(acc[i]);
    }
  }
}

[[kernel]] void mk_pf2_gate_up(
    device bfloat *input [[buffer(0)]],
    device const uchar *gw [[buffer(1)]], device const bfloat *gs [[buffer(2)]],
    device const bfloat *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]],
    device const bfloat *us [[buffer(5)]], device const bfloat *ub [[buffer(6)]],
    device const uint *offsets [[buffer(7)]], device const FlashMoEBucketJob *jobs [[buffer(8)]],
    device const uint *job_count [[buffer(9)]], device bfloat *output [[buffer(10)]],
    device uint *diag [[buffer(11)]], constant FlashMoEBlockedGateParams &params [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  alignas(16) threadgroup bfloat bst[2 * 128 * 32 + 2 * 64 * 32];
  const constant FlashMoEFusedParams &p = params.affine;
  uint expert = 0, begin = 0, end = 0;
  if (!pf2_job(offsets, jobs, job_count, group.y, params.job_capacity, params.route_capacity, expert, begin, end)) return;
  const uint n0 = group.x * 64, rows = min(64u, end - begin);
  // B-row r of the stage: h = r / 64, gate for (r % 64) < 32 else up, feature n0 + h*32 + r % 32.
  const uint srow = tid / 2, hh = srow / 64, rr = srow % 64;
  const bool up = rr >= 32;
  const uint n = n0 + hh * 32 + (rr & 31);
  Pf2Stage st;
  st.row = (up ? uw : gw) + ulong(expert) * (up ? p.up_weight_expert_stride_bytes : p.gate_weight_expert_stride_bytes) +
           ulong(n) * (up ? p.up_weight_row_stride_bytes : p.gate_weight_row_stride_bytes);
  const ulong pe = ulong(expert) * ((up ? p.up_parameter_expert_stride_bytes : p.gate_parameter_expert_stride_bytes) / 2) +
                   ulong(n) * ((up ? p.up_parameter_row_stride_bytes : p.gate_parameter_row_stride_bytes) / 2);
  st.scales = (up ? us : gs) + pe;
  st.biases = (up ? ub : gb) + pe;
  st.part = (tid % 2) * 16;
  pf2_body<2560, true>(input, st, begin, rows, n0, offsets, params.route_capacity, output, tid, simd, bst);
  (void)diag;
}

[[kernel]] void mk_pf2_down(
    device bfloat *input [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const bfloat *s [[buffer(2)]], device const bfloat *b [[buffer(3)]],
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]],
    device const uint *job_count [[buffer(6)]], device const uint *map [[buffer(7)]],
    device bfloat *output [[buffer(8)]], device uint *diag [[buffer(9)]],
    constant FlashMoEBlockedDownParams &params [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  alignas(16) threadgroup bfloat bst[2 * 128 * 32 + 2 * 64 * 32];
  const constant FlashMoEDownFusedParams &p = params.affine;
  uint expert = 0, begin = 0, end = 0;
  if (!pf2_job(offsets, jobs, job_count, group.y, params.job_capacity, params.route_capacity, expert, begin, end)) return;
  const uint n0 = group.x * 128, rows = min(64u, end - begin);
  const uint n = n0 + tid / 2;
  Pf2Stage st;
  st.row = w + ulong(expert) * p.weight_expert_stride_bytes + ulong(n) * p.weight_row_stride_bytes;
  const ulong pe = ulong(expert) * (p.parameter_expert_stride_bytes / 2) + ulong(n) * (p.parameter_row_stride_bytes / 2);
  st.scales = s + pe;
  st.biases = b + pe;
  st.part = (tid % 2) * 16;
  pf2_body<640, false>(input, st, begin, rows, n0, map, params.route_capacity, output, tid, simd, bst);
  (void)diag;
}

[[kernel]] void mk_pf2s_gate_up(
    device bfloat *input [[buffer(0)]],
    device const uchar *gw [[buffer(1)]], device const bfloat *gs [[buffer(2)]],
    device const bfloat *gb [[buffer(3)]], device const uchar *uw [[buffer(4)]],
    device const bfloat *us [[buffer(5)]], device const bfloat *ub [[buffer(6)]],
    device const uint *offsets [[buffer(7)]], device const FlashMoEBucketJob *jobs [[buffer(8)]],
    device const uint *job_count [[buffer(9)]], device bfloat *output [[buffer(10)]],
    device uint *diag [[buffer(11)]], constant FlashMoEBlockedGateParams &params [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  alignas(16) threadgroup bfloat bst[128 * 32 + 64 * 32];
  const constant FlashMoEFusedParams &p = params.affine;
  uint expert = 0, begin = 0, end = 0;
  if (!pf2_job(offsets, jobs, job_count, group.y, params.job_capacity, params.route_capacity, expert, begin, end)) return;
  const uint n0 = group.x * 64, rows = min(64u, end - begin);
  // B-row r of the stage: h = r / 64, gate for (r % 64) < 32 else up, feature n0 + h*32 + r % 32.
  const uint srow = tid / 2, hh = srow / 64, rr = srow % 64;
  const bool up = rr >= 32;
  const uint n = n0 + hh * 32 + (rr & 31);
  Pf2Stage st;
  st.row = (up ? uw : gw) + ulong(expert) * (up ? p.up_weight_expert_stride_bytes : p.gate_weight_expert_stride_bytes) +
           ulong(n) * (up ? p.up_weight_row_stride_bytes : p.gate_weight_row_stride_bytes);
  const ulong pe = ulong(expert) * ((up ? p.up_parameter_expert_stride_bytes : p.gate_parameter_expert_stride_bytes) / 2) +
                   ulong(n) * ((up ? p.up_parameter_row_stride_bytes : p.gate_parameter_row_stride_bytes) / 2);
  st.scales = (up ? us : gs) + pe;
  st.biases = (up ? ub : gb) + pe;
  st.part = (tid % 2) * 16;
  pf2s_body<2560, true>(input, st, begin, rows, n0, offsets, params.route_capacity, output, tid, simd, bst);
  (void)diag;
}

[[kernel]] void mk_pf2s_down(
    device bfloat *input [[buffer(0)]], device const uchar *w [[buffer(1)]],
    device const bfloat *s [[buffer(2)]], device const bfloat *b [[buffer(3)]],
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]],
    device const uint *job_count [[buffer(6)]], device const uint *map [[buffer(7)]],
    device bfloat *output [[buffer(8)]], device uint *diag [[buffer(9)]],
    constant FlashMoEBlockedDownParams &params [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  alignas(16) threadgroup bfloat bst[128 * 32 + 64 * 32];
  const constant FlashMoEDownFusedParams &p = params.affine;
  uint expert = 0, begin = 0, end = 0;
  if (!pf2_job(offsets, jobs, job_count, group.y, params.job_capacity, params.route_capacity, expert, begin, end)) return;
  const uint n0 = group.x * 128, rows = min(64u, end - begin);
  const uint n = n0 + tid / 2;
  Pf2Stage st;
  st.row = w + ulong(expert) * p.weight_expert_stride_bytes + ulong(n) * p.weight_row_stride_bytes;
  const ulong pe = ulong(expert) * (p.parameter_expert_stride_bytes / 2) + ulong(n) * (p.parameter_row_stride_bytes / 2);
  st.scales = s + pe;
  st.biases = b + pe;
  st.part = (tid % 2) * 16;
  pf2s_body<640, false>(input, st, begin, rows, n0, map, params.route_capacity, output, tid, simd, bst);
  (void)diag;
}
