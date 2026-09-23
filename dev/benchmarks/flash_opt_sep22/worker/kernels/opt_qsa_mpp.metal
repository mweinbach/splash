// Selected-block GQA partitions for few-row verification (drop-in for
// flash_qsa_mpp_online_partition, same bindings, grid and threadgroup).
// The query tile is staged once per threadgroup, each tile resolves its
// selected token rows once, and K/V rows are staged with 16-byte loads.
// Threadgroup tile contents and every matrix/softmax operation match the
// original kernel, so partition results are unchanged.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashQSAMPP.h"
#include "metal/abi/FlashQSAFast.h"

#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

inline void opt_qsa_failure(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}
inline uint opt_qsa_visible(FlashQSAParams p, uint row) {
  const uint count = p.begin + row + 1;
  return metal::min(count / 4, 512u) * 4 + count % 4;
}
inline uint opt_qsa_token(device const uint *selected, FlashQSAParams p, uint row, uint slot) {
  const uint completed = (p.begin + row + 1) / 4;
  const uint picked = metal::min(completed, 512u);
  return slot < picked * 4
      ? selected[ulong(row) * 512 + slot / 4] * 4 + slot % 4
      : completed * 4 + slot - picked * 4;
}

kernel void opt_qsa_online_partition(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device const uint *selected [[buffer(3)]],
    device float *statistics [[buffer(4)]],
    device float *numerators [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashQSAFastParams &params [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const FlashQSAParams p = params.common;
  const uint row = group.x, kv = group.y, partition = group.z;
  if (row >= p.rows || kv >= 2 || partition >= params.partitions) return;
  const uint visible = opt_qsa_visible(p, row);
  const uint length = (visible + params.partitions - 1) / params.partitions;
  const uint start = metal::min(partition * length, visible);
  const uint stop = metal::min(start + length, visible);
  // aq holds all 256 query dimensions of the twelve heads (rows 12..15 zero).
  threadgroup bfloat aq[16 * 256], bk[64 * 64], bv[64 * 64];
  threadgroup float weights[16 * 64];
  threadgroup float maximum[16], sum[16], alpha[16];
  // Token row of each tile slot (UINT_MAX: padding or rejected slot).
  threadgroup uint rowOf[64];
  if (tid < 16) {
    maximum[tid] = -INFINITY;
    sum[tid] = 0.0f;
  }
  for (uint i = tid; i < 16 * 32; i += 128) {
    const uint h = i / 32, part = i % 32;
    uint4 value = uint4(0u);
    if (h < 12)
      value = *reinterpret_cast<const device uint4 *>(
          queries + (ulong(row) * 24 + kv * 12 + h) * 256 + part * 8);
    *reinterpret_cast<threadgroup uint4 *>(aq + h * 256 + part * 8) = value;
  }
  auto qt = tensor(aq, dextents<int, 2>{256, 16}, array<int, 2>{1, 256});
  auto kt = tensor(bk, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto pt = tensor(weights, dextents<int, 2>{64, 16}, array<int, 2>{1, 64});
  auto vt = tensor(bv, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto k = kt.slice<64, 64>(0, 0);
  auto prob = pt.slice<64, 16>(0, 0);
  auto v = vt.slice<64, 64>(0, 0);
  auto q0 = qt.slice<64, 16>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      16, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto out0 = operation.get_destination_cooperative_tensor<
      decltype(prob), decltype(v), float>();
  auto out1 = operation.get_destination_cooperative_tensor<
      decltype(prob), decltype(v), float>();
  auto out2 = operation.get_destination_cooperative_tensor<
      decltype(prob), decltype(v), float>();
  auto out3 = operation.get_destination_cooperative_tensor<
      decltype(prob), decltype(v), float>();
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i)
    if (out0.is_valid_element(i))
      out0[i] = out1[i] = out2[i] = out3[i] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint tokenBegin = start; tokenBegin < stop; tokenBegin += 64) {
    if (tid < 64) {
      const uint slot = tokenBegin + tid;
      uint token = UINT_MAX;
      if (slot < stop) {
        token = opt_qsa_token(selected, p, row, slot);
        if (token >= p.capacity || token > p.begin + row) {
          opt_qsa_failure(diagnostics, 1u << 9);
          token = UINT_MAX;
        }
      }
      rowOf[tid] = token;
    }
    auto scores = operation.get_destination_cooperative_tensor<
        decltype(q0), decltype(k), float>();
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i)
      if (scores.is_valid_element(i)) scores[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      const uint dimension = chunk * 64;
      // 64 slots x 8 sixteen-byte parts; bk[slot * 64 + d].
      for (uint i = tid; i < 64 * 8; i += 128) {
        const uint slot = i / 8, part = i % 8;
        const uint token = rowOf[slot];
        uint4 value = uint4(0u);
        if (token != UINT_MAX)
          value = *reinterpret_cast<const device uint4 *>(
              keys + (ulong(token) * 2 + kv) * 256 + dimension + part * 8);
        *reinterpret_cast<threadgroup uint4 *>(bk + slot * 64 + part * 8) = value;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto q = qt.slice<64, 16>(dimension, 0);
      auto partial = operation.get_destination_cooperative_tensor<
          decltype(q0), decltype(k), float>();
      operation.run(q, k, partial);
#pragma unroll
      for (ushort i = 0; i < scores.get_capacity(); ++i)
        if (scores.is_valid_element(i)) scores[i] += partial[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i) {
      if (!scores.is_valid_element(i)) continue;
      const auto index = scores.get_multidimensional_index(i);
      const uint h = index[1], slot = index[0];
      weights[h * 64 + slot] = h < 12 && tokenBegin + slot < stop
          ? scores[i] * 0.0625f : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint h = tid / 8, hLane = tid % 8;
    float tileMaximum = -INFINITY;
    for (uint slot = hLane; slot < 64; slot += 8)
      tileMaximum = metal::max(tileMaximum, weights[h * 64 + slot]);
    tileMaximum = metal::max(tileMaximum, simd_shuffle_xor(tileMaximum, 4));
    tileMaximum = metal::max(tileMaximum, simd_shuffle_xor(tileMaximum, 2));
    tileMaximum = metal::max(tileMaximum, simd_shuffle_xor(tileMaximum, 1));
    const float newMaximum = metal::max(maximum[h], tileMaximum);
    const float rescale = h < 12 ? metal::exp(maximum[h] - newMaximum) : 0.0f;
    float tileSum = 0.0f;
    for (uint slot = hLane; slot < 64; slot += 8) {
      const float weight = h < 12
          ? metal::exp(weights[h * 64 + slot] - newMaximum) : 0.0f;
      weights[h * 64 + slot] = weight;
      tileSum += weight;
    }
    tileSum += simd_shuffle_xor(tileSum, 4);
    tileSum += simd_shuffle_xor(tileSum, 2);
    tileSum += simd_shuffle_xor(tileSum, 1);
    if (hLane == 0) {
      alpha[h] = rescale;
      maximum[h] = newMaximum;
      sum[h] = sum[h] * rescale + tileSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint dimensionChunk = 0; dimensionChunk < 4; ++dimensionChunk) {
      const uint dimension = dimensionChunk * 64;
      // bv[d * 64 + slot]: sixteen-byte row loads, transposed stores.
      for (uint i = tid; i < 64 * 8; i += 128) {
        const uint slot = i / 8, part = i % 8;
        const uint token = rowOf[slot];
        uint4 raw = uint4(0u);
        if (token != UINT_MAX)
          raw = *reinterpret_cast<const device uint4 *>(
              values + (ulong(token) * 2 + kv) * 256 + dimension + part * 8);
        const bfloat *lanes = reinterpret_cast<const thread bfloat *>(&raw);
#pragma unroll
        for (uint j = 0; j < 8; ++j) bv[(part * 8 + j) * 64 + slot] = lanes[j];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.get_destination_cooperative_tensor<
          decltype(prob), decltype(v), float>();
      operation.run(prob, v, partial);
#pragma unroll
      for (ushort i = 0; i < out0.get_capacity(); ++i) {
        if (!out0.is_valid_element(i)) continue;
        const auto index = out0.get_multidimensional_index(i);
        const float scale = alpha[index[1]];
        if (dimensionChunk == 0) out0[i] = out0[i] * scale + partial[i];
        if (dimensionChunk == 1) out1[i] = out1[i] * scale + partial[i];
        if (dimensionChunk == 2) out2[i] = out2[i] * scale + partial[i];
        if (dimensionChunk == 3) out3[i] = out3[i] * scale + partial[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (tid < 12) {
    const ulong index = (ulong(row) * 24 + kv * 12 + tid) *
                       params.maximum_partitions + partition;
    statistics[index * 2] = maximum[tid];
    statistics[index * 2 + 1] = sum[tid];
    if (stop > start && (!(sum[tid] > 0.0f) || !isfinite(sum[tid]) ||
                         !isfinite(maximum[tid])))
      opt_qsa_failure(diagnostics, 1u << 8);
  }
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i) {
    if (!out0.is_valid_element(i)) continue;
    const auto coord = out0.get_multidimensional_index(i);
    const uint h = coord[1], d = coord[0];
    if (h >= 12) continue;
    const ulong index = (ulong(row) * 24 + kv * 12 + h) *
                       params.maximum_partitions + partition;
    numerators[index * 256 + d] = out0[i];
    numerators[index * 256 + 64 + d] = out1[i];
    numerators[index * 256 + 128 + d] = out2[i];
    numerators[index * 256 + 192 + d] = out3[i];
    if (!isfinite(out0[i]) || !isfinite(out1[i]) || !isfinite(out2[i]) ||
        !isfinite(out3[i])) opt_qsa_failure(diagnostics, 1u << 8);
  }
}
