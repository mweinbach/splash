#include "metal/abi/FlashGDNLazyRollback.h"
#include <metal_stdlib>

using namespace metal;

// Lazy-rollback GDN verification with the token-independent work hoisted out
// of the recurrence (drop-in for flash_gdn_lazy_verify_sg16: same bindings,
// grid (48, lanes) and 512 threads). Every token's q/k convolution and norms,
// value convolution and gates are computed in parallel first; the recurrence
// then needs no barriers; output norms run in parallel last. Each value is
// produced by the same operations in the same order as the original kernel.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

constant uint OptLGFKeyWidth = 2048;
constant uint OptLGFValueWidth = 6144;
constant uint OptLGFConvWidth = 10240;

inline void opt_lgf_flag(device atomic_uint &diagnostics, uint flag) {
  atomic_fetch_or_explicit(&diagnostics, flag, memory_order_relaxed);
}

inline bool opt_lgf_valid(FlashGDNParams p, device atomic_uint &diagnostics) {
  const bool valid = p.rows && p.rows <= 16 && p.lanes && p.lanes <= 4 &&
      p.key_heads == 16 && p.value_heads == 48 && p.key_dimension == 128 &&
      p.value_dimension == 128 && p.convolution_taps == 4 &&
      isfinite(p.norm_epsilon) && p.norm_epsilon > 0.0f &&
      p.convolution_lane_stride_bytes >= ulong(3) * OptLGFConvWidth * 2 &&
      p.convolution_lane_stride_bytes % 2 == 0 &&
      (p.lanes <= 1 || p.convolution_lane_stride_bytes <=
          (ulong(-1) - ulong(3) * OptLGFConvWidth * 2) / (p.lanes - 1)) &&
      p.recurrent_lane_stride_bytes >= ulong(48) * 128 * 128 * 4 &&
      p.recurrent_lane_stride_bytes % 4 == 0 &&
      (p.lanes <= 1 || p.recurrent_lane_stride_bytes <=
          (ulong(-1) - ulong(48) * 128 * 128 * 4) / (p.lanes - 1));
  if (!valid) opt_lgf_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

inline void opt_lgf_check(float value, device atomic_uint &diagnostics) {
  if (!isfinite(value)) opt_lgf_flag(diagnostics, FlashGDNNonFinite);
}

inline bfloat opt_lgf_sigmoid_compiled(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f + float(exponent));
  const bfloat tail = bfloat(1.0f / float(denominator));
  return float(source) < 0.0f ? tail : bfloat(1.0f - float(tail));
}

inline bfloat opt_lgf_sigmoid_unary(bfloat source) {
  const bfloat exponent =
      bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f + float(exponent));
  const bfloat tail = bfloat(1.0f / float(denominator));
  return float(source) < 0.0f ? tail : bfloat(1.0f - float(tail));
}

inline bfloat opt_lgf_softplus(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(-metal::abs(float(source))));
  const float xp1 = 1.0f + float(exponent);
  const bfloat logarithm = xp1 == 1.0f ? exponent :
      bfloat(float(exponent) * (metal::log(xp1) / (xp1 - 1.0f)));
  return bfloat(max(float(source), 0.0f) + float(logarithm));
}

inline float opt_lgf_sigmoid_z(float source) {
  const float tail =
      1.0f / (1.0f + metal::precise::exp(metal::abs(source)));
  return source < 0.0f ? tail : 1.0f - tail;
}

template <bool Separate = false>
inline bfloat opt_lgf_conv(device const bfloat *qkv,
                       device const bfloat *weights,
                       device const bfloat *history, FlashGDNParams p,
                       uint batch, uint token, uint channel) {
  device const bfloat *lane_history =
      history + (Separate ? 0 : ulong(batch) * p.convolution_lane_stride_bytes / 2);
  device const bfloat *lane_input = qkv + ulong(batch) * p.rows * OptLGFConvWidth;
  float total = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    const uint position = token + tap;
    const bfloat source = position < 3
        ? lane_history[position * OptLGFConvWidth + channel]
        : lane_input[ulong(position - 3) * OptLGFConvWidth + channel];
    total += float(source) * float(weights[channel * 4 + tap]);
  }
  const bfloat convolution = bfloat(total);
  return bfloat(float(convolution) * float(opt_lgf_sigmoid_compiled(convolution)));
}

template <bool Separate = false>
inline void opt_lgf_qk(device const bfloat *qkv,
                   device const bfloat *weights,
                   device const bfloat *history, FlashGDNParams p,
                   uint batch, uint token, uint key_head, uint lane,
                   thread bfloat *query, thread bfloat *key) {
  float q[4], k[4];
  bfloat qp = bfloat(0.0f), kp = bfloat(0.0f);
  for (uint i = 0; i < 4; ++i) {
    q[i] = float(opt_lgf_conv<Separate>(qkv, weights, history, p, batch, token,
                         key_head * 128 + 4 * lane + i));
    k[i] = float(opt_lgf_conv<Separate>(qkv, weights, history, p, batch, token,
                         OptLGFKeyWidth + key_head * 128 + 4 * lane + i));
    qp = bfloat(float(qp) + float(bfloat(q[i] * q[i])));
    kp = bfloat(float(kp) + float(bfloat(k[i] * k[i])));
  }
  const bfloat qs = bfloat(simd_sum(float(qp)));
  const bfloat ks = bfloat(simd_sum(float(kp)));
  const bfloat epsilon = bfloat(1e-6f);
  const bfloat qd = bfloat(float(qs) + float(epsilon));
  const bfloat kd = bfloat(float(ks) + float(epsilon));
  const float qi = float(bfloat(metal::precise::rsqrt(float(qd))));
  const float ki = float(bfloat(metal::precise::rsqrt(float(kd))));
  const bfloat scale = bfloat(0.08838834764831845f);
  for (uint i = 0; i < 4; ++i) {
    query[i] = bfloat(float(bfloat(q[i] * qi)) * float(scale));
    key[i] = bfloat(k[i] * ki);
  }
}

inline void opt_lgf_gates(device const bfloat *a, device const bfloat *b,
                      device const bfloat *a_log,
                      device const bfloat *time_bias, ulong index, uint head,
                      thread float &decay, thread bfloat &beta) {
  const bfloat sum = bfloat(float(a[index]) + float(time_bias[head]));
  decay = metal::exp(-metal::exp(float(a_log[head])) * float(opt_lgf_softplus(sum)));
  beta = opt_lgf_sigmoid_unary(b[index]);
}

constant ulong OptLazyRecurrentLaneBytes = ulong(48) * 128 * 128 * 4;

inline bool opt_lazy_snapshot_stride_valid(ulong stride, uint lanes,
                                       device atomic_uint &diagnostics) {
  const bool valid = stride >= OptLazyRecurrentLaneBytes && stride % 4 == 0 &&
      (lanes <= 1 || stride <= (ulong(-1) - OptLazyRecurrentLaneBytes) / (lanes - 1));
  if (!valid) opt_lgf_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

inline bool opt_lazy_grid_valid(uint3 group_count, uint3 threads, uint simd_width,
                            uint lanes, uint groups, uint required_threads,
                            device atomic_uint &diagnostics) {
  const bool valid = group_count.x == groups && group_count.y == lanes &&
      group_count.z == 1 && threads.x == required_threads && threads.y == 1 &&
      threads.z == 1 && simd_width == 32;
  if (!valid) opt_lgf_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

inline bool opt_lazy_replay_valid(constant FlashGDNLazyReplayParams &params,
                              device atomic_uint &diagnostics) {
  if (!opt_lgf_valid(params.gdn, diagnostics) ||
      !opt_lazy_snapshot_stride_valid(params.snapshot_lane_stride_bytes,
                                   params.gdn.lanes, diagnostics)) return false;
  for (uint lane = 0; lane < params.gdn.lanes; ++lane) {
    if (params.retained[lane] > params.gdn.rows) {
      opt_lgf_flag(diagnostics, FlashGDNInvalidParameters);
      return false;
    }
  }
  return true;
}


kernel void opt_gdn_lazy_verify(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *a [[buffer(2)]],
    device const bfloat *b [[buffer(3)]],
    device const bfloat *weights [[buffer(4)]],
    device const bfloat *a_log [[buffer(5)]],
    device const bfloat *time_bias [[buffer(6)]],
    device const bfloat *norm [[buffer(7)]],
    device const bfloat *history [[buffer(8)]],
    device float *recurrent [[buffer(9)]],
    device bfloat *mixed [[buffer(10)]],
    device float *decay [[buffer(11)]],
    device bfloat *beta [[buffer(12)]],
    device bfloat *recurrent_rows [[buffer(13)]],
    device bfloat *output [[buffer(14)]],
    device atomic_uint &diagnostics [[buffer(15)]],
    device float *initial_snapshot [[buffer(16)]],
    constant FlashGDNLazyVerifyParams &params [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_count [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (!opt_lazy_grid_valid(group_count, threads, simd_width, params.gdn.lanes,
                           48, 512, diagnostics) ||
      !opt_lgf_valid(params.gdn, diagnostics) ||
      !opt_lazy_snapshot_stride_valid(params.snapshot_lane_stride_bytes,
                                      params.gdn.lanes, diagnostics)) return;
  const FlashGDNParams p = params.gdn;
  if (group.x >= 48 || group.y >= p.lanes) return;
  constexpr uint Simdgroups = 16, ValueRows = 128 / Simdgroups, MaxRows = 16;
  threadgroup bfloat query[MaxRows][128], key[MaxRows][128], value[MaxRows][128], rows[MaxRows][128];
  threadgroup float gates[MaxRows][3];
  const uint head = group.x, batch = group.y, key_head = head / 3;
  const bool shared_writer = head % 3 == 0;
  // Phase 1: token-independent operands for every row of the window.
  if (simd_group < p.rows) {
    const uint token = simd_group;
    const ulong row = ulong(batch) * p.rows + token;
    bfloat q[4], k[4];
    opt_lgf_qk(qkv, weights, history, p, batch, token, key_head, lane, q, k);
    for (uint i = 0; i < 4; ++i) {
      query[token][4 * lane + i] = q[i]; key[token][4 * lane + i] = k[i];
      if (shared_writer) {
        mixed[row * OptLGFConvWidth + key_head * 128 + 4 * lane + i] = q[i];
        mixed[row * OptLGFConvWidth + OptLGFKeyWidth + key_head * 128 + 4 * lane + i] = k[i];
      }
      opt_lgf_check(float(q[i]), diagnostics); opt_lgf_check(float(k[i]), diagnostics);
    }
  }
  for (uint item = thread_index; item < p.rows * 128; item += 512) {
    const uint token = item / 128, c = item % 128;
    const ulong row = ulong(batch) * p.rows + token;
    const uint channel = 2 * OptLGFKeyWidth + head * 128 + c;
    value[token][c] = opt_lgf_conv(qkv, weights, history, p, batch, token, channel);
    mixed[row * OptLGFConvWidth + channel] = value[token][c];
    opt_lgf_check(float(value[token][c]), diagnostics);
  }
  if (thread_index < p.rows) {
    const uint token = thread_index;
    const ulong row = ulong(batch) * p.rows + token;
    float d; bfloat be;
    opt_lgf_gates(a, b, a_log, time_bias, row * 48 + head, head, d, be);
    gates[token][0] = d; gates[token][1] = float(be);
    decay[row * 48 + head] = d; beta[row * 48 + head] = be;
    opt_lgf_check(d, diagnostics); opt_lgf_check(float(be), diagnostics);
  }
  // Phase 2: the recurrence. Each SIMD group owns its value rows, so tokens
  // follow one another without threadgroup barriers.
  float state[ValueRows][4];
  const ulong state_lane = ulong(batch) * p.recurrent_lane_stride_bytes / 4;
  for (uint r = 0; r < ValueRows; ++r) {
    const uint dimension = r * Simdgroups + simd_group;
    const ulong base = state_lane + (head * 128 + dimension) * 128 + 4 * lane;
    const ulong snapshot_base = ulong(batch) * params.snapshot_lane_stride_bytes / 4 +
        (head * 128 + dimension) * 128 + 4 * lane;
    for (uint i = 0; i < 4; ++i) {
      state[r][i] = recurrent[base + i];
      initial_snapshot[snapshot_base + i] = state[r][i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint token = 0; token < p.rows; ++token) {
    const ulong row = ulong(batch) * p.rows + token;
    float q[4], k[4];
    for (uint i = 0; i < 4; ++i) {
      q[i] = float(query[token][4 * lane + i]); k[i] = float(key[token][4 * lane + i]);
    }
    const float decayGate = gates[token][0], betaGate = gates[token][1];
    for (uint r = 0; r < ValueRows; ++r) {
      const uint dimension = r * Simdgroups + simd_group;
      float memory = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        state[r][i] = state[r][i] * decayGate; memory += state[r][i] * k[i];
      }
      memory = simd_sum(memory);
      const float delta = (float(value[token][dimension]) - memory) * betaGate;
      float result = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        state[r][i] = state[r][i] + k[i] * delta; result += state[r][i] * q[i];
      }
      result = simd_sum(result);
      if (lane == 0) {
        rows[token][dimension] = bfloat(result);
        recurrent_rows[row * OptLGFValueWidth + head * 128 + dimension] = rows[token][dimension];
        opt_lgf_check(result, diagnostics);
      }
    }
  }
  for (uint r = 0; r < ValueRows; ++r) {
    const uint dimension = r * Simdgroups + simd_group;
    const ulong base = state_lane + (head * 128 + dimension) * 128 + 4 * lane;
    for (uint i = 0; i < 4; ++i) {
      recurrent[base + i] = state[r][i]; opt_lgf_check(state[r][i], diagnostics);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Phase 3: per-token RMS norm, weight and output gate.
  if (simd_group < p.rows) {
    const uint token = simd_group;
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
      const float item = float(rows[token][4 * lane + i]); sum += item * item;
    }
    sum = simd_sum(sum);
    if (lane == 0) gates[token][2] = metal::precise::rsqrt(sum / 128 + p.norm_epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint item = thread_index; item < p.rows * 128; item += 512) {
    const uint token = item / 128, c = item % 128;
    const ulong row = ulong(batch) * p.rows + token;
    const bfloat normalized = bfloat(float(rows[token][c]) * gates[token][2]);
    const bfloat weighted = bfloat(float(normalized) * float(norm[c]));
    const ulong index = row * OptLGFValueWidth + head * 128 + c;
    const bfloat result = bfloat(float(weighted) * opt_lgf_sigmoid_z(float(z[index])));
    output[index] = result; opt_lgf_check(float(result), diagnostics);
  }
}
