#include <metal_stdlib>
#include "metal/abi/FlashMoE.h"
#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline bfloat pointwise_nan() { return bfloat(as_type<float>(0x7fc00000u)); }
inline bfloat pointwise_sigmoid(bfloat source) {
  const bfloat exponent = bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponent;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
inline bool pointwise_geometry(constant FlashMoEPointwiseParams &p,
                               uint2 threads) {
  return p.rows && p.rows <= 8192 && p.width && p.width <= 2560 &&
      p.selections && p.selections <= 10 && p.experts && p.experts <= 512 &&
      p.selections <= p.experts && threads.x == 256 && threads.y == 1;
}
inline uint pointwise_route_error(const device long *ids,
    const device bfloat *weights, constant FlashMoEPointwiseParams &p,
    ulong row, uint slot) {
  const ulong route = row * p.selections + slot;
  const long expert = ids[route];
  uint error = expert < 0 || ulong(expert) >= p.experts ? 1u : 0u;
  for (uint previous = 0; previous < slot; ++previous)
    if (ids[row * p.selections + previous] == expert) error |= 1u;
  const float weight = float(weights[route]);
  if (!metal::isfinite(weight) || weight < 0.0f || weight > 1.0f) error |= 4u;
  return error;
}
inline void pointwise_finish(const device bfloat *expert_down,
    const device bfloat *shared_down, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashMoEPointwiseParams &p,
    ulong row, uint column, thread const bfloat *weights,
    bfloat shared_sigmoid, uint error) {
  // Exact source BF16 storage boundaries and slot reduction order.
  bfloat partials[8];
  for (uint y = 0; y < 8; ++y) {
    bfloat partial = bfloat(0.0f);
    for (uint slot = y; slot < p.selections; slot += 8) {
      const bfloat down = expert_down[(row * p.selections + slot) * p.width + column];
      const bfloat weighted = down * weights[slot];
      if (!metal::isfinite(float(down)) || !metal::isfinite(float(weighted))) error |= 4u;
      partial = partial + weighted;
    }
    partials[y] = partial;
  }
  bfloat routed = partials[0];
  for (uint y = 1; y < 8; ++y) routed = routed + partials[y];
  const ulong index = row * p.width + column;
  const bfloat shared = shared_down[index];
  const bfloat scaled_shared = shared * shared_sigmoid;
  const bfloat combined = routed + scaled_shared;
  if (!metal::isfinite(float(shared)) || !metal::isfinite(float(routed)) ||
      !metal::isfinite(float(scaled_shared)) || !metal::isfinite(float(combined))) error |= 4u;
  if (error) {
    atomic_fetch_or_explicit(diagnostics, error, memory_order_relaxed);
    output[index] = pointwise_nan();
  } else output[index] = combined;
}

template <bool ParallelSlots>
inline void pointwise_simd(const device bfloat *expert_down,
    const device long *expert_ids, const device bfloat *route_weights,
    const device bfloat *shared_down, const device bfloat *shared_gate,
    device bfloat *output, device atomic_uint *diagnostics,
    constant FlashMoEPointwiseParams &p, uint2 group, uint tid,
    uint2 threads, uint lane) {
  if (!pointwise_geometry(p, threads)) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (group.y >= p.rows || group.x * 256 >= p.width) return;
  const ulong row = group.y;
  const uint column = group.x * 256 + tid;
  uint error = 0u, sigmoid_bits = 0u;
  bfloat weights[10];
  if constexpr (ParallelSlots) {
    uint weight_bits = 0u;
    if (lane < p.selections) {
      error = pointwise_route_error(expert_ids, route_weights, p, row, lane);
      weight_bits = uint(as_type<ushort>(route_weights[row * p.selections + lane]));
    }
    if (!lane) {
      const bfloat gate = shared_gate[row];
      if (!metal::isfinite(float(gate))) error |= 4u;
      sigmoid_bits = uint(as_type<ushort>(pointwise_sigmoid(gate)));
    }
    error = simd_or(error);
    sigmoid_bits = simd_broadcast(sigmoid_bits, 0);
    for (uint slot = 0; slot < p.selections; ++slot)
      weights[slot] = as_type<bfloat>(ushort(simd_broadcast(weight_bits, ushort(slot))));
  } else {
    if (!lane) {
      for (uint slot = 0; slot < p.selections; ++slot) {
        error |= pointwise_route_error(expert_ids, route_weights, p, row, slot);
        weights[slot] = route_weights[row * p.selections + slot];
      }
      const bfloat gate = shared_gate[row];
      if (!metal::isfinite(float(gate))) error |= 4u;
      sigmoid_bits = uint(as_type<ushort>(pointwise_sigmoid(gate)));
    }
    error = simd_broadcast(error, 0);
    sigmoid_bits = simd_broadcast(sigmoid_bits, 0);
    for (uint slot = 0; slot < p.selections; ++slot) {
      const uint bits = !lane ? uint(as_type<ushort>(weights[slot])) : 0u;
      weights[slot] = as_type<bfloat>(ushort(simd_broadcast(bits, 0)));
    }
  }
  // Partial SIMD groups must take all collectives before columns return.
  if (column >= p.width) return;
  pointwise_finish(expert_down, shared_down, output, diagnostics, p, row, column,
      weights, as_type<bfloat>(ushort(sigmoid_bits)), error);
}
#define POINTWISE_SIMD(NAME, PARALLEL) \
kernel void NAME(const device bfloat *down [[buffer(0)]], \
    const device long *ids [[buffer(1)]], const device bfloat *weights [[buffer(2)]], \
    const device bfloat *shared [[buffer(3)]], const device bfloat *gate [[buffer(4)]], \
    device bfloat *output [[buffer(5)]], device atomic_uint *diag [[buffer(6)]], \
    constant FlashMoEPointwiseParams &p [[buffer(7)]], \
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint2 threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  pointwise_simd<PARALLEL>(down,ids,weights,shared,gate,output,diag,p,group,tid,threads,lane); \
}
POINTWISE_SIMD(private_moe_combine_simd_lane0, false)
POINTWISE_SIMD(private_moe_combine_simd_slots, true)
#undef POINTWISE_SIMD

template <bool WholeRow>
inline void pointwise_cta(const device bfloat *down, const device long *ids,
    const device bfloat *weights, const device bfloat *shared,
    const device bfloat *gate, device bfloat *output, device atomic_uint *diag,
    constant FlashMoEPointwiseParams &p, uint2 group, uint tid, uint2 threads,
    threadgroup bfloat *row_weights, threadgroup bfloat *shared_sigmoid,
    threadgroup uint *row_error) {
  if (!pointwise_geometry(p, threads)) {
    if (!tid) atomic_fetch_or_explicit(diag, 2u, memory_order_relaxed);
    return;
  }
  if (group.y >= p.rows || group.x * 256 >= p.width || (WholeRow && group.x)) return;
  if (!tid) {
    uint error = 0u;
    for (uint slot = 0; slot < p.selections; ++slot) {
      error |= pointwise_route_error(ids, weights, p, group.y, slot);
      row_weights[slot] = weights[ulong(group.y) * p.selections + slot];
    }
    const bfloat raw_gate = gate[group.y];
    if (!metal::isfinite(float(raw_gate))) error |= 4u;
    *shared_sigmoid = pointwise_sigmoid(raw_gate);
    *row_error = error;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bfloat local_weights[10];
  for (uint slot = 0; slot < p.selections; ++slot) local_weights[slot] = row_weights[slot];
  const bfloat local_sigmoid = *shared_sigmoid;
  const uint local_error = *row_error;
  for (uint column = group.x * 256 + tid; column < p.width; column += 256) {
    pointwise_finish(down, shared, output, diag, p, group.y, column,
        local_weights, local_sigmoid, local_error);
    if constexpr (!WholeRow) break;
  }
}
#define POINTWISE_CTA(NAME, WHOLE_ROW) \
kernel void NAME(const device bfloat *down [[buffer(0)]], const device long *ids [[buffer(1)]], \
    const device bfloat *weights [[buffer(2)]], const device bfloat *shared [[buffer(3)]], \
    const device bfloat *gate [[buffer(4)]], device bfloat *output [[buffer(5)]], \
    device atomic_uint *diag [[buffer(6)]], constant FlashMoEPointwiseParams &p [[buffer(7)]], \
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint2 threads [[threads_per_threadgroup]]) { \
  threadgroup bfloat row_weights[10], shared_sigmoid; \
  threadgroup uint row_error; \
  pointwise_cta<WHOLE_ROW>(down,ids,weights,shared,gate,output,diag,p,group,tid,threads, \
      row_weights,&shared_sigmoid,&row_error); \
}
POINTWISE_CTA(private_moe_combine_cta, false)
POINTWISE_CTA(private_moe_combine_cta_row, true)
#undef POINTWISE_CTA

inline bool pointwise_poison_geometry(constant FlashMoEBlockedDownParams &params,
    uint3 group, uint3 threads, uint thread_count) {
  const constant FlashMoEDownFusedParams &p = params.affine;
  return p.rows && p.rows <= kFlashMoEBucketMaximumRows && p.selections &&
      p.selections <= kFlashMoEBucketMaximumSelections && p.input_size == 640 &&
      p.output_size == 2560 && p.experts == 512 &&
      params.route_capacity == p.rows * p.selections && !group.z &&
      threads.x == thread_count && threads.y == 1 && threads.z == 1;
}
template <uint Threads>
inline void pointwise_poison_route(device const uint *inverse, device bfloat *output,
    device atomic_uint *diag, constant FlashMoEBlockedDownParams &params,
    uint3 group, uint3 threads, uint tid, uint lane, threadgroup uint *excluded) {
  if (!pointwise_poison_geometry(params, group, threads, Threads)) {
    if (!tid) atomic_fetch_or_explicit(diag, 2u, memory_order_relaxed);
    return;
  }
  const uint route = group.y;
  if (group.x || route >= params.route_capacity) return;
  uint invalid;
  if constexpr (Threads == 32) {
    invalid = !lane ? uint(inverse[route] >= params.route_capacity) : 0u;
    invalid = simd_broadcast(invalid, 0);
  } else {
    if (!tid) *excluded = uint(inverse[route] >= params.route_capacity);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    invalid = *excluded;
  }
  if (!invalid) return;
  if (!tid) atomic_fetch_or_explicit(diag, 5u, memory_order_relaxed);
  for (uint column = tid; column < 2560; column += Threads)
    output[ulong(route) * 2560 + column] = pointwise_nan();
}
#define POINTWISE_POISON(NAME, THREADS) \
kernel void NAME(device const uint *inverse [[buffer(0)]], device bfloat *output [[buffer(1)]], \
    device atomic_uint *diag [[buffer(2)]], constant FlashMoEBlockedDownParams &p [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup uint excluded; \
  pointwise_poison_route<THREADS>(inverse,output,diag,p,group,threads,tid,lane,&excluded); \
}
POINTWISE_POISON(private_moe_poison_route256, 256)
POINTWISE_POISON(private_moe_poison_route32, 32)
#undef POINTWISE_POISON

// Frozen production poison body, using its identical validation and grid.
kernel void flash_moe_blocked_poison_excluded_routes(
    device const uint *inverse [[buffer(0)]], device bfloat *output [[buffer(1)]],
    device atomic_uint *diag [[buffer(2)]], constant FlashMoEBlockedDownParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (!pointwise_poison_geometry(params, group, threads, 256)) {
    if (!tid) atomic_fetch_or_explicit(diag, 2u, memory_order_relaxed);
    return;
  }
  const uint column = group.x * 256 + tid, route = group.y;
  if (column >= 2560 || route >= params.route_capacity) return;
  if (inverse[route] >= params.route_capacity) {
    if (!tid) atomic_fetch_or_explicit(diag, 5u, memory_order_relaxed);
    output[ulong(route) * 2560 + column] = pointwise_nan();
  }
}
