// Isolated chunked GDN prefill experiment. Changes arithmetic association;
// qualification must compare F32 carried state and continued sequences.
// Source q/k/v/beta remain BF16; gates, solve, state and outputs accumulate F32.
// Derivation/reference: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py
// This is an independent implementation using relative alpha products, so a
// zero/underflowed prefix is never divided into another prefix.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashGDN.h"
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
using namespace metal;
using namespace mpp::tensor_ops;

inline void gdc_error(device atomic_uint &diagnostics, uint flag) {
  atomic_fetch_or_explicit(&diagnostics, flag, memory_order_relaxed);
}

template <ushort Values, ushort Time, bool Audit, bool Cached = false>
inline void gdc_chunked(device const bfloat *mixed,
    device const float *decay, device const bfloat *beta,
    device float *recurrent, device bfloat *output,
    device atomic_uint &diagnostics, constant FlashGDNParams &p,
    uint3 group, uint3 threads, uint tid,
    threadgroup bfloat *keys, threadgroup float *state,
    threadgroup float *gram, threadgroup float *score,
    threadgroup float *delta, threadgroup float *stateQuery,
    threadgroup float *alphas, threadgroup float *betas,
    threadgroup float *prefix,
    device float *history, device float *deltaAudit, device float *outputAudit) {
  if (!p.rows || p.rows > 2048 || !p.lanes || p.lanes > 32 ||
      p.key_heads != 16 || p.value_heads != 48 ||
      p.key_dimension != 128 || p.value_dimension != 128 ||
      p.convolution_taps != 4 || !isfinite(p.norm_epsilon) || p.norm_epsilon <= 0.0f ||
      p.convolution_lane_stride_bytes < ulong(3) * 10240 * 2 ||
      p.convolution_lane_stride_bytes % 2 ||
      p.recurrent_lane_stride_bytes < ulong(48) * 128 * 128 * 4 ||
      p.recurrent_lane_stride_bytes % 4 ||
      threads.x != 128 || threads.y != 1 || threads.z != 1 ||
      group.x >= 48 || group.y >= 128 / Values || group.z >= p.lanes ||
      (Audit && group.x != 0)) {
    if (!tid) gdc_error(diagnostics, FlashGDNInvalidParameters);
    return;
  }
  const uint head = group.x, batch = group.z, keyHead = head / 3;
  const uint valueBegin = group.y * Values;
  const ulong stateBase = ulong(batch) * p.recurrent_lane_stride_bytes / 4 +
                         (head * 128 + valueBegin) * 128;
  auto kt = tensor(keys, dextents<int, 2>{128, Time}, array<int, 2>{1, 128});
  auto st = tensor(state, dextents<int, 2>{128, Values}, array<int, 2>{1, 128});
  auto dt = tensor(delta, dextents<int, 2>{Time, Values}, array<int, 2>{1, Time});
  auto pt = tensor(score, dextents<int, 2>{Time, Time}, array<int, 2>{1, Time});
  auto k = kt.template slice<128, Time>(0, 0);
  auto s = st.template slice<128, Values>(0, 0);
  auto d = dt.template slice<Time, Values>(0, 0);
  auto attention = pt.template slice<Time, Time>(0, 0);
  constexpr auto gramDescriptor = matmul2d_descriptor(Time, Time, 128,
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<gramDescriptor, execution_simdgroups<4>> gramOp;
  constexpr auto projectDescriptor = matmul2d_descriptor(Values, Time, 128,
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<projectDescriptor, execution_simdgroups<4>> projectOp;
  constexpr auto outputDescriptor = matmul2d_descriptor(Values, Time, Time,
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<outputDescriptor, execution_simdgroups<4>> outputOp;
  constexpr auto updateDescriptor = matmul2d_descriptor(Values, 128, Time,
      false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<updateDescriptor, execution_simdgroups<4>> updateOp;
  // Apple permits cooperative tensors as inputs only with a single-SIMD
  // execution scope. Cached variants partition V across the four SIMD groups
  // for state projection/update; Gram and output products remain SG4.
  constexpr auto cachedProjectDescriptor = matmul2d_descriptor(Values / 4, Time, 128,
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<cachedProjectDescriptor, execution_simdgroup> cachedProjectOp;
  constexpr auto cachedUpdateDescriptor = matmul2d_descriptor(Values / 4, 128, Time,
      false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<cachedUpdateDescriptor, execution_simdgroup> cachedUpdateOp;
  const uint cachedValueBegin = (tid / 32) * (Values / 4);
  auto cachedD = dt.template slice<Time, Values / 4>(0, int(cachedValueBegin));
  auto cachedS = st.template slice<128, Values / 4>(0, int(cachedValueBegin));
  // The register variant retains the entire F32 state as the output layout of
  // the update operation. Input layout conversion is explicit and stays F32.
  auto stateCache = cachedUpdateOp.template get_destination_cooperative_tensor<
      decltype(cachedD), decltype(k), float>();
  if constexpr (Cached) {
#pragma unroll
    for (ushort i = 0; i < stateCache.get_capacity(); ++i) {
      if (!stateCache.is_valid_element(i)) continue;
      const auto index = stateCache.get_multidimensional_index(i);
      stateCache[i] = recurrent[stateBase + (cachedValueBegin + index[1]) * 128 + index[0]];
    }
  } else {
    for (uint i = tid; i < uint(Values) * 128; i += 128)
      state[i] = recurrent[stateBase + i];
  }

  for (uint begin = 0; begin < p.rows; begin += Time) {
    const uint count = min(uint(Time), p.rows - begin);
    for (uint i = tid; i < uint(Time) * 128; i += 128) {
      const uint token = i / 128;
      keys[i] = token < count ? mixed[
          (ulong(batch) * p.rows + begin + token) * 10240 +
          2048 + keyHead * 128 + i % 128] : bfloat(0.0f);
    }
    if (tid < Time) {
      const ulong gate = (ulong(batch) * p.rows + begin + tid) * 48 + head;
      alphas[tid] = tid < count ? decay[gate] : 1.0f;
      betas[tid] = tid < count ? float(beta[gate]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < Time) {
      float product = 1.0f;
      for (uint token = 0; token <= tid; ++token) product *= alphas[token];
      prefix[tid] = product;
    }

    auto kk = gramOp.template get_destination_cooperative_tensor<
        decltype(k), decltype(k), float>();
    gramOp.run(k, k, kk);
#pragma unroll
    for (ushort i = 0; i < kk.get_capacity(); ++i) {
      if (!kk.is_valid_element(i)) continue;
      const auto index = kk.get_multidimensional_index(i);
      gram[index[1] * Time + index[0]] = kk[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(Time) * Time; i += 128) {
      const uint token = i / Time, previous = i % Time;
      float product = 1.0f;
      for (uint a = previous + 1; a <= token; ++a) product *= alphas[a];
      gram[i] = token < count && previous < token
          ? gram[i] * betas[token] * product : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Queries are read directly from device memory: retaining both Q and K
    // would exceed the 32 KiB bound for the Time32/Values16 tile. Clamp rows
    // in a tensor extent for the tail; MPP slice loads pad outside extents.
    // MPP's tensor element traits reject const bfloat operands. This cast
    // changes only type metadata; q is exclusively an input to multiplication.
    auto qt = tensor(const_cast<device bfloat *>(mixed) +
        (ulong(batch) * p.rows + begin) * 10240 + keyHead * 128,
        dextents<int, 2>{128, int(count)}, array<int, 2>{1, 10240});
    // Keep dynamic extents for q: a static Time extent would declare padded
    // rows valid even when this is the final partial chunk.
    auto q = qt.slice(0, 0);
    auto qk = gramOp.template get_destination_cooperative_tensor<
        decltype(q), decltype(k), float>();
    gramOp.run(q, k, qk);
#pragma unroll
    for (ushort i = 0; i < qk.get_capacity(); ++i) {
      if (!qk.is_valid_element(i)) continue;
      const auto index = qk.get_multidimensional_index(i);
      const uint token = index[1], previous = index[0];
      float product = 1.0f;
      for (uint a = previous + 1; a <= token; ++a) product *= alphas[a];
      score[token * Time + previous] = token < count && previous <= token
          ? qk[i] * product : 0.0f;
    }
    if constexpr (Cached) {
      auto cachedInput = cachedProjectOp.template get_left_input_cooperative_tensor<
          float, bfloat, float>(stateCache);
      auto sk = cachedProjectOp.template get_destination_cooperative_tensor<
          decltype(cachedS), decltype(k), float>();
      cachedProjectOp.run(cachedInput, k, sk);
#pragma unroll
      for (ushort i = 0; i < sk.get_capacity(); ++i) {
        if (!sk.is_valid_element(i)) continue;
        const auto index = sk.get_multidimensional_index(i);
        const uint token = index[0], v = cachedValueBegin + index[1];
        const float value = token < count ? float(mixed[
            (ulong(batch) * p.rows + begin + token) * 10240 +
            4096 + head * 128 + valueBegin + v]) : 0.0f;
        delta[v * Time + token] = betas[token] * (value - prefix[token] * sk[i]);
      }
    } else {
      auto sk = projectOp.template get_destination_cooperative_tensor<
          decltype(s), decltype(k), float>();
      projectOp.run(s, k, sk);
#pragma unroll
      for (ushort i = 0; i < sk.get_capacity(); ++i) {
        if (!sk.is_valid_element(i)) continue;
        const auto index = sk.get_multidimensional_index(i);
        const uint token = index[0], v = index[1];
        const float value = token < count ? float(mixed[
            (ulong(batch) * p.rows + begin + token) * 10240 +
            4096 + head * 128 + valueBegin + v]) : 0.0f;
        delta[v * Time + token] = betas[token] * (value - prefix[token] * sk[i]);
      }
    }
    if constexpr (Cached) {
      auto cachedInput = cachedProjectOp.template get_left_input_cooperative_tensor<
          float, bfloat, float>(stateCache);
      auto sq = cachedProjectOp.template get_destination_cooperative_tensor<
          decltype(cachedS), decltype(q), float>();
      cachedProjectOp.run(cachedInput, q, sq);
#pragma unroll
      for (ushort i = 0; i < sq.get_capacity(); ++i) {
        if (!sq.is_valid_element(i)) continue;
        const auto index = sq.get_multidimensional_index(i);
        stateQuery[(cachedValueBegin + index[1]) * Time + index[0]] = prefix[index[0]] * sq[i];
      }
    } else {
      auto sq = projectOp.template get_destination_cooperative_tensor<
          decltype(s), decltype(q), float>();
      projectOp.run(s, q, sq);
#pragma unroll
      for (ushort i = 0; i < sq.get_capacity(); ++i) {
        if (!sq.is_valid_element(i)) continue;
        const auto index = sq.get_multidimensional_index(i);
        stateQuery[index[1] * Time + index[0]] = prefix[index[0]] * sq[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Independent value rows solve the unit-lower-triangular delta system.
    // The dependent chain is now Time*(Time-1)/2 scalar multiply/adds;
    // no per-token SIMD reduction or state-tile update remains in the chain.
    if (tid < Values) {
      for (uint token = 0; token < count; ++token) {
        float value = delta[tid * Time + token];
        for (uint previous = 0; previous < token; ++previous)
          value -= gram[token * Time + previous] * delta[tid * Time + previous];
        delta[tid * Time + token] = value;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto result = outputOp.template get_destination_cooperative_tensor<
        decltype(d), decltype(attention), float>();
    outputOp.run(d, attention, result);
#pragma unroll
    for (ushort i = 0; i < result.get_capacity(); ++i) {
      if (!result.is_valid_element(i)) continue;
      const auto index = result.get_multidimensional_index(i);
      const uint token = index[0], v = index[1];
      if (token >= count) continue;
      const float value = stateQuery[v * Time + token] + result[i];
      output[(ulong(batch) * p.rows + begin + token) * 6144 +
             head * 128 + valueBegin + v] = bfloat(value);
      if (!isfinite(value)) gdc_error(diagnostics, FlashGDNNonFinite);
      if constexpr (Audit) {
        deltaAudit[(ulong(batch) * p.rows + begin + token) * 128 + valueBegin + v] = delta[v * Time + token];
        outputAudit[(ulong(batch) * p.rows + begin + token) * 128 + valueBegin + v] = value;
      }
    }
    if constexpr (Audit) {
      // Audit entry reconstructs every full F32 state row with the chunk
      // formula before scaling/updating the carried tile. Never benchmark it.
      auto writeHistory = [&](uint v, uint dimension, float sourceState) {
        for (uint token = 0; token < count; ++token) {
          float value = prefix[token] * sourceState;
          for (uint previous = 0; previous <= token; ++previous) {
            float product = 1.0f;
            for (uint a = previous + 1; a <= token; ++a) product *= alphas[a];
            value += product * delta[v * Time + previous] * float(keys[previous * 128 + dimension]);
          }
          history[((ulong(batch) * p.rows + begin + token) * 128 + valueBegin + v) * 128 + dimension] = value;
        }
      };
      if constexpr (Cached) {
#pragma unroll
        for (ushort i = 0; i < stateCache.get_capacity(); ++i) {
          if (!stateCache.is_valid_element(i)) continue;
          const auto index = stateCache.get_multidimensional_index(i);
          writeHistory(cachedValueBegin + index[1], index[0], stateCache[i]);
        }
      } else {
        for (uint i = tid; i < uint(Values) * 128; i += 128)
          writeHistory(i / 128, i % 128, state[i]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < uint(Values) * Time; i += 128) {
      const uint token = i % Time;
      float product = token < count ? 1.0f : 0.0f;
      for (uint a = token + 1; a < count; ++a) product *= alphas[a];
      delta[i] *= product;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if constexpr (Cached) {
      auto update = cachedUpdateOp.template get_destination_cooperative_tensor<
          decltype(cachedD), decltype(k), float>();
      cachedUpdateOp.run(cachedD, k, update);
#pragma unroll
      for (ushort i = 0; i < update.get_capacity(); ++i) {
        if (!update.is_valid_element(i)) continue;
        stateCache[i] = prefix[count - 1] * stateCache[i] + update[i];
        if (!isfinite(stateCache[i])) gdc_error(diagnostics, FlashGDNNonFinite);
      }
    } else {
      auto update = updateOp.template get_destination_cooperative_tensor<
          decltype(d), decltype(k), float>();
      updateOp.run(d, k, update);
#pragma unroll
      for (ushort i = 0; i < update.get_capacity(); ++i) {
        if (!update.is_valid_element(i)) continue;
        const auto index = update.get_multidimensional_index(i);
        const uint item = index[1] * 128 + index[0];
        state[item] = prefix[count - 1] * state[item] + update[i];
        if (!isfinite(state[item])) gdc_error(diagnostics, FlashGDNNonFinite);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if constexpr (Cached) {
#pragma unroll
    for (ushort i = 0; i < stateCache.get_capacity(); ++i) {
      if (!stateCache.is_valid_element(i)) continue;
      const auto index = stateCache.get_multidimensional_index(i);
      recurrent[stateBase + (cachedValueBegin + index[1]) * 128 + index[0]] = stateCache[i];
    }
  } else {
    for (uint i = tid; i < uint(Values) * 128; i += 128)
      recurrent[stateBase + i] = state[i];
  }
}

#define GDC_MEMORY(V,T,S) \
  threadgroup bfloat keys[T * 128]; \
  threadgroup float state[S], gram[T * T], score[T * T]; \
  threadgroup float delta[V * T], stateQuery[V * T]; \
  threadgroup float alphas[T], betas[T], prefix[T];
#define GDC_ARGS \
  device const bfloat *mixed [[buffer(0)]], device const float *decay [[buffer(1)]], \
  device const bfloat *beta [[buffer(2)]], device float *recurrent [[buffer(3)]], \
  device bfloat *output [[buffer(4)]], device atomic_uint &diagnostics [[buffer(5)]], \
  constant FlashGDNParams &p [[buffer(6)]]
#define GDC_THREADS \
  uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
  uint tid [[thread_index_in_threadgroup]]
#define GDC_ENTRY(V,T) \
  [[max_total_threads_per_threadgroup(128)]] kernel void private_gdn_chunk_v##V##_t##T( \
      GDC_ARGS, GDC_THREADS) { \
    GDC_MEMORY(V,T,V*128) \
    gdc_chunked<V,T,false>(mixed,decay,beta,recurrent,output,diagnostics,p,group,threads,tid, \
        keys,state,gram,score,delta,stateQuery,alphas,betas,prefix,nullptr,nullptr,nullptr); \
  } \
  [[max_total_threads_per_threadgroup(128)]] kernel void private_gdn_chunk_audit_v##V##_t##T( \
      GDC_ARGS, device float *history [[buffer(7)]], device float *deltaAudit [[buffer(8)]], \
      device float *outputAudit [[buffer(9)]], GDC_THREADS) { \
    GDC_MEMORY(V,T,V*128) \
    gdc_chunked<V,T,true>(mixed,decay,beta,recurrent,output,diagnostics,p,group,threads,tid, \
        keys,state,gram,score,delta,stateQuery,alphas,betas,prefix,history,deltaAudit,outputAudit); \
  }
GDC_ENTRY(16,16)
GDC_ENTRY(16,32)
GDC_ENTRY(32,16)
#define GDC_REGISTER_ENTRY(V,T) \
  [[max_total_threads_per_threadgroup(128)]] kernel void private_gdn_chunk_register_v##V##_t##T( \
      GDC_ARGS, GDC_THREADS) { \
    GDC_MEMORY(V,T,1) \
    gdc_chunked<V,T,false,true>(mixed,decay,beta,recurrent,output,diagnostics,p,group,threads,tid, \
        keys,state,gram,score,delta,stateQuery,alphas,betas,prefix,nullptr,nullptr,nullptr); \
  } \
  [[max_total_threads_per_threadgroup(128)]] kernel void private_gdn_chunk_register_audit_v##V##_t##T( \
      GDC_ARGS, device float *history [[buffer(7)]], device float *deltaAudit [[buffer(8)]], \
      device float *outputAudit [[buffer(9)]], GDC_THREADS) { \
    GDC_MEMORY(V,T,1) \
    gdc_chunked<V,T,true,true>(mixed,decay,beta,recurrent,output,diagnostics,p,group,threads,tid, \
        keys,state,gram,score,delta,stateQuery,alphas,betas,prefix,history,deltaAudit,outputAudit); \
  }
GDC_REGISTER_ENTRY(32,16)
GDC_REGISTER_ENTRY(32,32)
#undef GDC_REGISTER_ENTRY
#undef GDC_ENTRY
#undef GDC_THREADS
#undef GDC_ARGS
#undef GDC_MEMORY
