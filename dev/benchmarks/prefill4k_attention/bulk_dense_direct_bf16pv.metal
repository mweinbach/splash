// Private NUMERICAL ALTERNATIVE: dense 2048-row M32/p1 attention with BF16 PV.
// BF16 QK operands; F32 scores/maxima/denominators/alpha/bank updates; BF16
// tile-local UNNORMALIZED exp weights times BF16 V, with strict F32 MPP PV.
// This is NOT globally normalized BF16 probability semantics or a claim of
// checkpoint/model matching. No GPU numerical/generation/service evidence.
// Compared with normal p4 attention, p1 also changes bank origins/merge order.
// ABI buffers: 0 prepared Q,1 full K,2 full V,3 raw Q/gate projection,
// 4 BF16 output,5 U32 diagnostics,6 FlashQSAFastParams; grid (768,2,1),
// exactly 128 threads; only begin 0/rows 2048/capacity >=2048/partitions 1.
// Shared banks: query/alpha union 4096 bytes, K/V epoch bank 8192 bytes,
// F32-score/BF16-P epoch union 8192 bytes; total 20480 bytes, unchanged peak.
// Both M32 head batches copy localWeights[2][8] before the first alias barrier;
// BF16 stores then publish the PV operand through a second CTA barrier.
// F32 denominator uses ORIGINAL unrounded weights, never the BF16 values.
// Final register normalization and staged BF16 sigmoid gate retain direct v1.
// Direct-v1 source SHA256 dev/benchmarks/prefill4k_attention/bulk_dense_direct.metal: 0cc0edd6acc0f5b03df2083a7ef885814b5d242c432e427c79641d5eb8cd4539
#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashQSAFast.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

inline bfloat bulk_dense_direct_bf16pv_gate(bfloat attention, bfloat source);
inline void bulk_dense_direct_bf16pv_error(device atomic_uint *diagnostics, uint reason) {
  atomic_fetch_or_explicit(diagnostics, reason, memory_order_relaxed);
}
template <ushort M> union BulkDenseDirectBF16PVQueryStorage {
  bfloat operand[M * 64];
  float alpha[M];
};
// Epoch-alias sheet: all F32 scores are read into per-lane registers before
// any BF16-P store; the second barrier publishes the BF16 operand for MPP PV.
template <ushort M> union BulkDenseDirectBF16PVWeightStorage {
  float scores[M * 64];
  bfloat probabilities[M * 64];
};
static_assert(sizeof(BulkDenseDirectBF16PVWeightStorage<32>) == 8192,
              "M32 F32-score/BF16-P epoch sheet extent");
// Dense-only temporal sharing. Each flattened row owns one real query/head;
// common K/V staging covers the union of their original partition windows.
// QK retains four K64 partials. PV uses BF16 tile-local UNNORMALIZED exp
// weights and BF16 V, with F32 accumulation; F32 softmax denominators remain.
// This does not reproduce globally normalized checkpoint BF16 probabilities.
template <ushort M>
inline void bulk_dense_direct_bf16pv_online(device const bfloat *queries,
    device const bfloat *keys, device const bfloat *values,
    device const bfloat *qProjection, device bfloat *output,
    device atomic_uint *diagnostics,
    thread const FlashQSAFastParams &params, uint3 group, uint tid,
    threadgroup BulkDenseDirectBF16PVQueryStorage<M> &queryMemory,
    threadgroup bfloat *kvMemory,
    threadgroup BulkDenseDirectBF16PVWeightStorage<M> &weightMemory) {
  static_assert(M == 32, "private two-batch BF16 PV candidate is M32 only");
  threadgroup float *weights = weightMemory.scores;
  const FlashQSAParams p = params.common;
  if (p.begin != 0 || p.rows != 2048 || p.capacity < 2048 ||
      params.partitions != 1 || group.y >= 2 || group.z != 0) {
    if (!tid) bulk_dense_direct_bf16pv_error(diagnostics, 1u << 9);
    return;
  }
  const uint flatBase = group.x * M, kv = group.y, partition = group.z;
  if (flatBase >= p.rows * 12) return;
  const uint firstQuery = flatBase / 12;
  const uint lastQuery = min((flatBase + M - 1) / 12, p.rows - 1);
  const uint firstCount = p.begin + firstQuery + 1, lastCount = p.begin + lastQuery + 1;
  const uint firstLength = (firstCount + params.partitions - 1) / params.partitions;
  const uint lastLength = (lastCount + params.partitions - 1) / params.partitions;
  const uint commonStart = min(partition * firstLength, firstCount);
  const uint commonStop = min(partition * lastLength + lastLength, lastCount);
  float runningMaximum[M / 16], runningSum[M / 16];
#pragma unroll
  for (uint batch = 0; batch < M / 16; ++batch) {
    runningMaximum[batch] = -INFINITY;
    runningSum[batch] = 0.0f;
  }
  auto qt = tensor(queryMemory.operand, dextents<int, 2>{64, M}, array<int, 2>{1, 64});
  auto kt = tensor(kvMemory, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto pt = tensor(weightMemory.probabilities, dextents<int, 2>{64, M}, array<int, 2>{1, 64});
  auto vt = tensor(kvMemory, dextents<int, 2>{64, 64}, array<int, 2>{1, 64});
  auto q = qt.template slice<64, M>(0, 0);
  auto k = kt.slice<64, 64>(0, 0);
  auto prob = pt.template slice<64, M>(0, 0);
  auto v = vt.slice<64, 64>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      M, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto out0 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
  auto out1 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
  auto out2 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
  auto out3 = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i)
    if (out0.is_valid_element(i)) out0[i] = out1[i] = out2[i] = out3[i] = 0.0f;
  for (uint tokenBegin = commonStart; tokenBegin < commonStop; tokenBegin += 64) {
    auto scores = operation.template get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
#pragma unroll
    for (ushort i = 0; i < scores.get_capacity(); ++i)
      if (scores.is_valid_element(i)) scores[i] = 0.0f;
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      for (uint i = tid; i < M * 64; i += 128) {
        const uint flat = flatBase + i / 64, row = flat / 12, h = kv * 12 + flat % 12;
        queryMemory.operand[i] = flat < p.rows * 12
            ? queries[(ulong(row) * 24 + h) * 256 + chunk * 64 + i % 64] : bfloat(0.0f);
      }
      for (uint i = tid; i < 64 * 64; i += 128) {
        const uint token = tokenBegin + i / 64;
        kvMemory[i] = token < commonStop
            ? keys[(ulong(token) * 2 + kv) * 256 + chunk * 64 + i % 64] : bfloat(0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.template get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
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
      const uint h = index[1], token = tokenBegin + index[0], flat = flatBase + h;
      const uint count = p.begin + flat / 12 + 1;
      const uint length = (count + params.partitions - 1) / params.partitions;
      const uint start = min(partition * length, count), stop = min(start + length, count);
      weights[h * 64 + index[0]] = flat < p.rows * 12 && token >= start && token < stop
          ? scores[i] * 0.0625f : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Both head batches must preserve their eight F32 exponentials per lane
    // before the aliased sheet enters its BF16-P epoch. No sheet stores here.
    float localWeights[2][8];
#pragma unroll
    for (uint batch = 0; batch < M / 16; ++batch) {
      const uint h = batch * 16 + tid / 8, hLane = tid % 8, flat = flatBase + h;
      const bool real = flat < p.rows * 12;
      float tileMaximum = -INFINITY;
      for (uint slot = hLane; slot < 64; slot += 8)
        tileMaximum = max(tileMaximum, weights[h * 64 + slot]);
      tileMaximum = max(tileMaximum, simd_shuffle_xor(tileMaximum, 4));
      tileMaximum = max(tileMaximum, simd_shuffle_xor(tileMaximum, 2));
      tileMaximum = max(tileMaximum, simd_shuffle_xor(tileMaximum, 1));
      const float newMaximum = max(runningMaximum[batch], tileMaximum);
      const float rescale = real && isfinite(newMaximum)
          ? exp(runningMaximum[batch] - newMaximum) : 0.0f;
      float tileSum = 0.0f;
      for (uint slot = hLane; slot < 64; slot += 8) {
        const float weight = real && isfinite(newMaximum) ? exp(weights[h * 64 + slot] - newMaximum) : 0.0f;
        localWeights[batch][slot / 8] = weight;
        tileSum += weight;
      }
      tileSum += simd_shuffle_xor(tileSum, 4);
      tileSum += simd_shuffle_xor(tileSum, 2);
      tileSum += simd_shuffle_xor(tileSum, 1);
      runningMaximum[batch] = newMaximum;
      runningSum[batch] = runningSum[batch] * rescale + tileSum;
      // QK has finished reading the query staging bank. Reuse it for alpha;
      // max/sum remain per-head SIMD registers until the next token tile.
      if (!hLane) queryMemory.alpha[h] = rescale;
    }
    // Every lane has read ALL M32 heads/batches from the F32 score sheet.
    // BF16 stores below alias those bytes; this barrier must precede ANY store.
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint batch = 0; batch < 2; ++batch) {
      const uint h = batch * 16 + tid / 8, hLane = tid % 8;
#pragma unroll
      for (uint item = 0; item < 8; ++item)
        weightMemory.probabilities[h * 64 + hLane + item * 8] =
            bfloat(localWeights[batch][item]);
    }
    // All BF16-P stores are complete before any cooperative PV operand read.
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      for (uint i = tid; i < 64 * 64; i += 128) {
        const uint token = tokenBegin + i % 64;
        kvMemory[i] = token < commonStop
            ? values[(ulong(token) * 2 + kv) * 256 + chunk * 64 + i / 64] : bfloat(0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto partial = operation.template get_destination_cooperative_tensor<decltype(prob), decltype(v), float>();
      operation.run(prob, v, partial);
#pragma unroll
      for (ushort i = 0; i < out0.get_capacity(); ++i) {
        if (!out0.is_valid_element(i)) continue;
        const auto index = out0.get_multidimensional_index(i);
        const float rescale = queryMemory.alpha[index[1]];
        if (chunk == 0) out0[i] = out0[i] * rescale + partial[i];
        if (chunk == 1) out1[i] = out1[i] * rescale + partial[i];
        if (chunk == 2) out2[i] = out2[i] * rescale + partial[i];
        if (chunk == 3) out3[i] = out3[i] * rescale + partial[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
#pragma unroll
  for (uint batch = 0; batch < M / 16; ++batch) {
    const uint h = batch * 16 + tid / 8, flat = flatBase + h;
    if (tid % 8 || flat >= p.rows * 12) continue;
    // Every PV bank has finished consuming alpha. Reuse the same query bank
    // for each head's final F32 denominator; no global numerator/stat sheet.
    queryMemory.alpha[h] = runningSum[batch];
    if (!(runningSum[batch] > 0) || !isfinite(runningSum[batch]) || !isfinite(runningMaximum[batch]))
      bulk_dense_direct_bf16pv_error(diagnostics, 1u << 8);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (ushort i = 0; i < out0.get_capacity(); ++i) {
    if (!out0.is_valid_element(i)) continue;
    const auto coord = out0.get_multidimensional_index(i);
    const uint flat = flatBase + coord[1];
    if (flat >= p.rows * 12) continue;
    const uint row = flat / 12, head = kv * 12 + flat % 12, d = coord[0];
    const float denominator = queryMemory.alpha[coord[1]];
#pragma unroll
    for (uint chunk = 0; chunk < 4; ++chunk) {
      float numerator;
      if (chunk == 0) numerator = out0[i];
      else if (chunk == 1) numerator = out1[i];
      else if (chunk == 2) numerator = out2[i];
      else numerator = out3[i];
      const uint column = chunk * 64 + d;
      const bfloat attention = bfloat(numerator / denominator);
      const bfloat gate = qProjection[ulong(row) * 12288 + head * 512 + 256 + column];
      const bfloat gated = bulk_dense_direct_bf16pv_gate(attention, gate);
      output[(ulong(row) * 24 + head) * 256 + column] = gated;
      if (!(denominator > 0.0f) || !isfinite(denominator) ||
          !isfinite(numerator) || !isfinite(float(gated)) || !isfinite(float(gate)))
        bulk_dense_direct_bf16pv_error(diagnostics, 1u << 8);
    }
  }
}

// Gate follows the qualified scalar reducer contraction/reassociation policy.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline bfloat bulk_dense_direct_bf16pv_gate(bfloat attention, bfloat source) {
  const bfloat exponential = bfloat(precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponential;
  const bfloat tail = bfloat(1.0f) / denominator;
  const bfloat sigmoid = source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
  return attention * sigmoid;
}


[[max_total_threads_per_threadgroup(128)]]
kernel void flash_qsa_prefill_dense_direct_bf16pv_m32(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device const bfloat *qProjection [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device atomic_uint *diagnostics [[buffer(5)]],
    constant FlashQSAFastParams &params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (any(groupSize != uint3(128, 1, 1))) {
    if (!tid) bulk_dense_direct_bf16pv_error(diagnostics, 1u << 9);
    return;
  }
  FlashQSAFastParams localParams = params;
  threadgroup BulkDenseDirectBF16PVQueryStorage<32> queryMemory;
  threadgroup bfloat kvMemory[64 * 64];
  threadgroup BulkDenseDirectBF16PVWeightStorage<32> weightMemory;
  bulk_dense_direct_bf16pv_online<32>(queries, keys, values, qProjection, output,
      diagnostics, localParams, group, tid, queryMemory, kvMemory, weightMemory);
}
#endif
