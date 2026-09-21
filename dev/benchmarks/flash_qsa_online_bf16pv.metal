#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashQSAMPP.h"
#include "metal/abi/FlashQSAFast.h"

#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

inline void qsa_mpp_failure(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}
inline uint qsa_mpp_visible(FlashQSAParams p, uint row) {
  const uint count = p.begin + row + 1;
  return metal::min(count / 4, 512u) * 4 + count % 4;
}
inline uint qsa_mpp_token(device const uint *selected,
                           FlashQSAParams p, uint row, uint slot) {
  const uint completed = (p.begin + row + 1) / 4;
  const uint picked = metal::min(completed, 512u);
  return slot < picked * 4
      ? selected[ulong(row) * 512 + slot / 4] * 4 + slot % 4
      : completed * 4 + slot - picked * 4;
}
inline bfloat qsa_mpp_gate(bfloat attention, bfloat source) {
  const bfloat e = bfloat(precise::exp(metal::abs(float(source))));
  const bfloat d = bfloat(1.0f) + e;
  const bfloat tail = bfloat(1.0f) / d;
  const bfloat sigmoid = source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
  return attention * sigmoid;
}

// Each query/KV-head group computes all twelve shared-key query heads in one
// padded sixteen-row MPP tile. Its selected cache rows are read once, instead
// of once per query head. No gathered K/V tensor is materialized.
// Private candidate only. BF16 QK operands and F32 QK/softmax/merge are kept.
// Tile-local unnormalized exponentials are rounded to BF16 for BF16 x BF16
// MPP PV with F32 accumulation. This is NOT globally normalized checkpoint
// BF16 probability semantics; the report names this numerical alternative.
// Peak shared-sheet bytes stay 4096 because its preceding score epoch is F32.
union QSABF16PVSheet {
  float scores[16 * 64];
  bfloat probabilities[16 * 64];
};
kernel void flash_qsa_mpp_online_partition_bf16pv(
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
  const uint visible = qsa_mpp_visible(p, row);
  const uint length = (visible + params.partitions - 1) / params.partitions;
  const uint start = metal::min(partition * length, visible);
  const uint stop = metal::min(start + length, visible);
  threadgroup bfloat aq[16 * 64], bk[64 * 64], bv[64 * 64];
  threadgroup QSABF16PVSheet sheet;
  threadgroup float maximum[16], sum[16], alpha[16];
  if (tid < 16) {
    maximum[tid] = -INFINITY;
    sum[tid] = 0.0f;
  }
  auto qt = tensor(aq, dextents<int, 2>{64, 16}, array<int, 2>{1, 64});
  auto kt = tensor(bk, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto pt = tensor(sheet.probabilities, dextents<int, 2>{64, 16}, array<int, 2>{1, 64});
  auto vt = tensor(bv, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto q = qt.slice<64, 16>(0, 0);
  auto k = kt.slice<64, 64>(0, 0);
  auto prob = pt.slice<64, 16>(0, 0);
  auto v = vt.slice<64, 64>(0, 0);
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
    auto scores = operation.get_destination_cooperative_tensor<
        decltype(q), decltype(k), float>();
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i)
      if (scores.is_valid_element(i)) scores[i] = 0.0f;
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      const uint dimension = chunk * 64;
      for (uint i = tid; i < 16 * 64; i += 128) {
        const uint h = i / 64, d = i % 64;
        aq[i] = h < 12
            ? queries[(ulong(row) * 24 + kv * 12 + h) * 256 + dimension + d]
            : bfloat(0.0f);
      }
      for (uint i = tid; i < 64 * 64; i += 128) {
        const uint slot = tokenBegin + i / 64, d = i % 64;
        bfloat value = bfloat(0.0f);
        if (slot < stop) {
          const uint token = qsa_mpp_token(selected, p, row, slot);
          if (token >= p.capacity || token > p.begin + row)
            qsa_mpp_failure(diagnostics, 1u << 9);
          else
            value = keys[(ulong(token) * 2 + kv) * 256 + dimension + d];
        }
        bk[i] = value;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.get_destination_cooperative_tensor<
          decltype(q), decltype(k), float>();
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
      sheet.scores[h * 64 + slot] = h < 12 && tokenBegin + slot < stop
          ? scores[i] * 0.0625f : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Eight SIMD lanes share one head. XOR reductions stay inside each
    // eight-lane sub-group; no full-SIMD head cross-contamination occurs.
    const uint h = tid / 8, hLane = tid % 8;
    float tileMaximum = -INFINITY;
    for (uint slot = hLane; slot < 64; slot += 8)
      tileMaximum = metal::max(tileMaximum, sheet.scores[h * 64 + slot]);
    tileMaximum = metal::max(tileMaximum, simd_shuffle_xor(tileMaximum, 4));
    tileMaximum = metal::max(tileMaximum, simd_shuffle_xor(tileMaximum, 2));
    tileMaximum = metal::max(tileMaximum, simd_shuffle_xor(tileMaximum, 1));
    const float newMaximum = metal::max(maximum[h], tileMaximum);
    const float rescale = h < 12 ? metal::exp(maximum[h] - newMaximum) : 0.0f;
    float tileSum = 0.0f;
    float localWeights[8];
#pragma unroll
    for (uint item = 0; item < 8; ++item) {
      const uint slot = hLane + item * 8;
      const float weight = h < 12
          ? metal::exp(sheet.scores[h * 64 + slot] - newMaximum) : 0.0f;
      localWeights[item] = weight;
      tileSum += weight;
    }
    // No BF16 store may overwrite the F32 score sheet until all lanes have
    // copied their eight F32 weights into registers. This union aliases only
    // disjoint score/PV epochs; QK, maximum, denominator and merging stay F32.
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint item = 0; item < 8; ++item)
      sheet.probabilities[h * 64 + hLane + item * 8] = bfloat(localWeights[item]);
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
      for (uint i = tid; i < 64 * 64; i += 128) {
        const uint d = i / 64, slot = tokenBegin + i % 64;
        bfloat value = bfloat(0.0f);
        if (slot < stop) {
          const uint token = qsa_mpp_token(selected, p, row, slot);
          if (token >= p.capacity || token > p.begin + row)
            qsa_mpp_failure(diagnostics, 1u << 9);
          else
            value = values[(ulong(token) * 2 + kv) * 256 + dimension + d];
        }
        bv[i] = value;
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
      qsa_mpp_failure(diagnostics, 1u << 8);
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
        !isfinite(out3[i])) qsa_mpp_failure(diagnostics, 1u << 8);
  }
}

#endif
