// Isolated scalar GDN explicit-FMA source probe, derived from canonical_audit.metal.
// Gate multiplication, beta/residual sequencing and both SIMD reductions retain
// native order. Only explicit dot/update FMA calls change scalar rounding.
// This experiment is not numerically qualified or promoted to the runtime.
#include "metal/abi/FlashGDN.h"
#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline void gsf_flag(device atomic_uint &diagnostics, uint flag) {
  atomic_fetch_or_explicit(&diagnostics, flag, memory_order_relaxed);
}

inline bool gsf_valid(FlashGDNParams p, device atomic_uint &diagnostics) {
  const bool valid = p.rows && p.rows <= 2048 && p.lanes && p.lanes <= 32 &&
      p.key_heads == 16 && p.value_heads == 48 && p.key_dimension == 128 &&
      p.value_dimension == 128 && p.convolution_taps == 4 &&
      isfinite(p.norm_epsilon) && p.norm_epsilon > 0.0f &&
      p.convolution_lane_stride_bytes >= ulong(3) * 10240 * 2 &&
      p.convolution_lane_stride_bytes % 2 == 0 &&
      p.recurrent_lane_stride_bytes >= ulong(48) * 128 * 128 * 4 &&
      p.recurrent_lane_stride_bytes % 4 == 0;
  if (!valid) gsf_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

template <uint Values, uint Time, bool Audit, bool DotFma, bool UpdateFma>
inline void gsf_recurrence(
    device const bfloat *mixed, device const float *decay,
    device const bfloat *beta, device float *recurrent,
    device bfloat *output, device atomic_uint &diagnostics,
    constant FlashGDNParams &p, uint3 group, uint3 threads, uint thread_index,
    uint lane, uint simd_group, threadgroup bfloat *queries,
    threadgroup bfloat *keys, threadgroup bfloat *values,
    threadgroup float *decays, threadgroup bfloat *betas,
    device float *history, device float *deltaAudit, device float *outputAudit) {
  if (!gsf_valid(p, diagnostics) || group.x >= 48 || (Audit && group.x != 0) ||
      group.y >= 128 / Values ||
      group.z >= p.lanes || threads.x != Values * 32 ||
      threads.y != 1 || threads.z != 1) {
    gsf_flag(diagnostics, FlashGDNInvalidParameters);
    return;
  }
  constexpr uint Threads = Values * 32;
  constexpr uint KeyStride = 136;
  constexpr uint ValueStride = Values + 8;
  const uint head = group.x, batch = group.z;
  const uint key_head = head / 3;
  const uint value_begin = group.y * Values;
  const uint dimension = value_begin + simd_group;
  const ulong state_base = ulong(batch) * p.recurrent_lane_stride_bytes / 4 +
                           (head * 128 + dimension) * 128 + 4 * lane;
  float state[4];
  for (uint i = 0; i < 4; ++i) state[i] = recurrent[state_base + i];
  for (uint begin = 0; begin < p.rows; begin += Time) {
    const uint count = min(Time, p.rows - begin);
    // Every key/query element is read once per value block, rather than once
    // per value dimension. Cooperative loads stay coalesced along K.
    for (uint item = thread_index; item < count * 128; item += Threads) {
      const uint token = item / 128, key_dimension = item % 128;
      const ulong source = (ulong(batch) * p.rows + begin + token) * 10240 +
                            key_head * 128 + key_dimension;
      queries[token * KeyStride + key_dimension] = mixed[source];
      keys[token * KeyStride + key_dimension] = mixed[source + 2048];
    }
    for (uint item = thread_index; item < count * Values; item += Threads) {
      const uint token = item / Values, value_dimension = item % Values;
      const ulong source = (ulong(batch) * p.rows + begin + token) * 10240 +
                            4096 + head * 128 + value_begin + value_dimension;
      values[token * ValueStride + value_dimension] = mixed[source];
    }
    for (uint token = thread_index; token < count; token += Threads) {
      const ulong source = (ulong(batch) * p.rows + begin + token) * 48 + head;
      decays[token] = decay[source]; betas[token] = beta[source];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Token order, gating arithmetic and SIMD reductions match the
    // qualified recurrence. Only selected explicit FMA operations change rounding.
    for (uint token = 0; token < count; ++token) {
      float q[4], k[4];
      float memory = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        q[i] = float(queries[token * KeyStride + 4 * lane + i]);
        k[i] = float(keys[token * KeyStride + 4 * lane + i]);
        state[i] = state[i] * decays[token];
        if constexpr (DotFma) memory = fma(state[i], k[i], memory);
        else memory += state[i] * k[i];
      }
      memory = simd_sum(memory);
      const float value = float(values[token * ValueStride + simd_group]);
      const float delta = (value - memory) * float(betas[token]);
      float result = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        if constexpr (UpdateFma) state[i] = fma(k[i], delta, state[i]);
        else state[i] = state[i] + k[i] * delta;
        if constexpr (DotFma) result = fma(state[i], q[i], result);
        else result += state[i] * q[i];
      }
      result = simd_sum(result);
      if constexpr (Audit) {
        for (uint i = 0; i < 4; ++i)
          history[((ulong(batch) * p.rows + begin + token) * 128 + dimension) * 128 +
                  4 * lane + i] = state[i];
      }
      if (lane == 0) {
        if constexpr (Audit) {
          deltaAudit[(ulong(batch) * p.rows + begin + token) * 128 + dimension] = delta;
          outputAudit[(ulong(batch) * p.rows + begin + token) * 128 + dimension] = result;
        }
        output[(ulong(batch) * p.rows + begin + token) * 6144 +
                head * 128 + dimension] = bfloat(result);
        if (!isfinite(result)) gsf_flag(diagnostics, FlashGDNNonFinite);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 4; ++i) {
    recurrent[state_base + i] = state[i];
    if (!isfinite(state[i])) gsf_flag(diagnostics, FlashGDNNonFinite);
  }
}

// Normal grid {48, 128/Values, lanes}; audit grid {1, 128/Values, lanes}.
// Every group uses Values*32 threads. Audit buffers keep compact head-0 layout.
#define GSF_ARGS \
  device const bfloat *mixed [[buffer(0)]], device const float *decay [[buffer(1)]], \
  device const bfloat *beta [[buffer(2)]], device float *recurrent [[buffer(3)]], \
  device bfloat *output [[buffer(4)]], device atomic_uint &diagnostics [[buffer(5)]], \
  constant FlashGDNParams &p [[buffer(6)]]
#define GSF_THREADS \
  uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
  uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
  uint simd_group [[simdgroup_index_in_threadgroup]]
#define GSF_MEMORY(V,T) \
  threadgroup bfloat queries[T * 136], keys[T * 136]; \
  threadgroup bfloat values[T * (V + 8)], betas[T]; \
  threadgroup float decays[T];
#define GSF_ENTRY(NAME,AUDIT_NAME,V,T,DOT,UPDATE) \
  [[max_total_threads_per_threadgroup(V * 32)]] kernel void NAME(GSF_ARGS, GSF_THREADS) { \
    GSF_MEMORY(V,T) \
    gsf_recurrence<V,T,false,DOT,UPDATE>(mixed,decay,beta,recurrent,output,diagnostics,p, \
        group,threads,thread_index,lane,simd_group,queries,keys,values,decays,betas, \
        nullptr,nullptr,nullptr); \
  } \
  [[max_total_threads_per_threadgroup(V * 32)]] kernel void AUDIT_NAME( \
      GSF_ARGS, device float *history [[buffer(7)]], device float *deltaAudit [[buffer(8)]], \
      device float *outputAudit [[buffer(9)]], GSF_THREADS) { \
    GSF_MEMORY(V,T) \
    gsf_recurrence<V,T,true,DOT,UPDATE>(mixed,decay,beta,recurrent,output,diagnostics,p, \
        group,threads,thread_index,lane,simd_group,queries,keys,values,decays,betas, \
        history,deltaAudit,outputAudit); \
  }
GSF_ENTRY(private_gdn_scalar_fma_v8_t8, private_gdn_scalar_fma_audit_v8_t8, 8, 8, true, true)
GSF_ENTRY(private_gdn_scalar_fma_v8_t16, private_gdn_scalar_fma_audit_v8_t16, 8, 16, true, true)
GSF_ENTRY(private_gdn_scalar_fma_v16_t8, private_gdn_scalar_fma_audit_v16_t8, 16, 8, true, true)
GSF_ENTRY(private_gdn_scalar_fma_v16_t16, private_gdn_scalar_fma_audit_v16_t16, 16, 16, true, true)
GSF_ENTRY(private_gdn_scalar_fma_v16_t32, private_gdn_scalar_fma_audit_v16_t32, 16, 32, true, true)
GSF_ENTRY(private_gdn_scalar_fma_dot_v16_t16, private_gdn_scalar_fma_dot_audit_v16_t16, 16, 16, true, false)
GSF_ENTRY(private_gdn_scalar_fma_update_v16_t16, private_gdn_scalar_fma_update_audit_v16_t16, 16, 16, false, true)
#undef GSF_ENTRY
#undef GSF_MEMORY
#undef GSF_THREADS
#undef GSF_ARGS
