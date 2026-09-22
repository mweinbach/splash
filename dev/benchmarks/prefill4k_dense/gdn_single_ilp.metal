// Private two-column ILP GDN, ordered F32 time/dot contract unchanged.
#include "metal/abi/FlashGDN.h"
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline void private_single_ilp_flag(device atomic_uint &diagnostics, uint flag) {
  atomic_fetch_or_explicit(&diagnostics, flag, memory_order_relaxed);
}
inline bool private_single_ilp_valid(FlashGDNParams p, device atomic_uint &diagnostics) {
  const bool valid = p.rows && p.rows <= 4096 && p.lanes && p.lanes <= 4 &&
      p.key_heads == 16 && p.value_heads == 48 && p.key_dimension == 128 &&
      p.value_dimension == 128 && p.convolution_taps == 4 &&
      isfinite(p.norm_epsilon) && p.norm_epsilon > 0.0f &&
      p.convolution_lane_stride_bytes >= ulong(3) * 10240 * 2 &&
      p.convolution_lane_stride_bytes % 2 == 0 &&
      p.recurrent_lane_stride_bytes >= ulong(48) * 128 * 128 * 4 &&
      p.recurrent_lane_stride_bytes % 4 == 0;
  if (!valid) private_single_ilp_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}
template <uint Values, uint Time, uint Simds>
inline void private_single_ilp_recurrence(device const bfloat *mixed, device const float *decay,
    device const bfloat *beta, device float *recurrent, device bfloat *output,
    device atomic_uint &diagnostics, constant FlashGDNParams &p,
    uint3 group, uint tid, uint lane, uint sg, threadgroup bfloat *queries,
    threadgroup bfloat *keys, threadgroup bfloat *values,
    threadgroup float *decays, threadgroup bfloat *betas) {
  if (!private_single_ilp_valid(p, diagnostics) || group.x >= 48 || group.y >= 128 / Values || group.z >= p.lanes)
    return;
  const uint actualRows = p.rows;
  if (!actualRows) return;
  if (actualRows > 2048 || actualRows > p.rows) {
    private_single_ilp_flag(diagnostics, FlashGDNInvalidParameters); return;
  }
  constexpr uint Threads = Simds * 32, PerSimd = Values / Simds;
  constexpr uint KeyStride = 136, ValueStride = Values + 8;
  const uint head = group.x, batch = group.z, keyHead = head / 3, valueBegin = group.y * Values;
  float state[PerSimd][4];
#pragma unroll
  for (uint r = 0; r < PerSimd; ++r) {
    const uint dimension = valueBegin + r * Simds + sg;
    const ulong base = ulong(batch)*p.recurrent_lane_stride_bytes/4 + (head * 128 + dimension) * 128 + 4 * lane;
#pragma unroll
    for (uint i = 0; i < 4; ++i) state[r][i] = recurrent[base + i];
  }
  for (uint begin = 0; begin < actualRows; begin += Time) {
    const uint count = min(Time, actualRows - begin);
    for (uint item = tid; item < count * 128; item += Threads) {
      const uint token = item / 128, d = item % 128;
      const ulong src = (ulong(batch) * p.rows + begin + token) * 10240 + keyHead * 128 + d;
      queries[token * KeyStride + d] = mixed[src];
      keys[token * KeyStride + d] = mixed[src + 2048];
    }
    for (uint item = tid; item < count * Values; item += Threads) {
      const uint token = item / Values, d = item % Values;
      values[token * ValueStride + d] = mixed[(ulong(batch) * p.rows + begin + token) * 10240 +
                                             4096 + head * 128 + valueBegin + d];
    }
    for (uint token = tid; token < count; token += Threads) {
      const ulong src = (ulong(batch) * p.rows + begin + token) * 48 + head;
      decays[token] = decay[src]; betas[token] = beta[src];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint token = 0; token < count; ++token) {
      float q[4], k[4], memory[PerSimd], delta[PerSimd], result[PerSimd];
#pragma unroll
      for (uint r = 0; r < PerSimd; ++r) memory[r] = result[r] = 0.0f;
#pragma unroll
      for (uint i = 0; i < 4; ++i) {
        q[i] = float(queries[token * KeyStride + 4 * lane + i]);
        k[i] = float(keys[token * KeyStride + 4 * lane + i]);
#pragma unroll
        for (uint r = 0; r < PerSimd; ++r) {
          state[r][i] = state[r][i] * decays[token];
          memory[r] += state[r][i] * k[i];
        }
      }
#pragma unroll
      for (uint r = 0; r < PerSimd; ++r) {
        memory[r] = simd_sum(memory[r]);
        const float value = float(values[token * ValueStride + r * Simds + sg]);
        delta[r] = (value - memory[r]) * float(betas[token]);
      }
#pragma unroll
      for (uint i = 0; i < 4; ++i)
#pragma unroll
        for (uint r = 0; r < PerSimd; ++r) {
          state[r][i] = state[r][i] + k[i] * delta[r];
          result[r] += state[r][i] * q[i];
        }
#pragma unroll
      for (uint r = 0; r < PerSimd; ++r) {
        result[r] = simd_sum(result[r]);
        if (!lane) {
          const uint dimension = valueBegin + r * Simds + sg;
          output[(ulong(batch) * p.rows + begin + token) * 6144 + head * 128 + dimension] = bfloat(result[r]);
          if (!isfinite(result[r])) private_single_ilp_flag(diagnostics, FlashGDNNonFinite);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (uint r = 0; r < PerSimd; ++r) {
    const uint dimension = valueBegin + r * Simds + sg;
    const ulong base = ulong(batch)*p.recurrent_lane_stride_bytes/4 + (head * 128 + dimension) * 128 + 4 * lane;
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
      recurrent[base + i] = state[r][i];
      if (!isfinite(state[r][i])) private_single_ilp_flag(diagnostics, FlashGDNNonFinite);
    }
  }
}


#define SINGLE_ILP_ENTRY(NAME,V,T,S) \
[[max_total_threads_per_threadgroup(S*32)]] kernel void NAME( \
    device const bfloat *mixed [[buffer(0)]],device const float *decay [[buffer(1)]], \
    device const bfloat *beta [[buffer(2)]],device float *state [[buffer(3)]], \
    device bfloat *output [[buffer(4)]],device atomic_uint &diagnostics [[buffer(5)]], \
    constant FlashGDNParams &p [[buffer(6)]], \
    uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]], \
    uint3 threads [[threads_per_threadgroup]],uint lane [[thread_index_in_simdgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]]) { \
  if (threads.x!=S*32 || threads.y!=1 || threads.z!=1) { \
    if (!tid) private_single_ilp_flag(diagnostics,FlashGDNInvalidParameters);return; \
  } \
  threadgroup bfloat queries[T*136],keys[T*136],values[T*(V+8)],betas[T]; \
  threadgroup float decays[T]; \
  private_single_ilp_recurrence<V,T,S>(mixed,decay,beta,state,output,diagnostics,p,group,tid,lane,sg,queries,keys,values,decays,betas); \
}
SINGLE_ILP_ENTRY(private_gdn_ilp_v16_t16_s8,16,16,8)
SINGLE_ILP_ENTRY(private_gdn_ilp_v16_t32_s8,16,32,8)
#undef SINGLE_ILP_ENTRY
