#include "metal/abi/FlashHCFused.h"
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline void private_prefill_hc_inject_norm_sep21_check(float x, device atomic_uint *d) {
  if ((as_type<uint>(x) & 0x7f800000u) == 0x7f800000u)
    atomic_fetch_or_explicit(d, 4u, memory_order_relaxed);
}
// Literal copy of the qualified small-row fusion arithmetic, with its own
// bounded row/launch guard. One group owns one complete (row, stream) plane.
kernel void private_prefill_hc_inject_norm_sep21(
    const device bfloat *hyper [[buffer(0)]], const device bfloat *branch [[buffer(1)]],
    const device bfloat *gates [[buffer(2)]], const device uchar *weight [[buffer(3)]],
    device bfloat *updated [[buffer(4)]], device bfloat *normalized [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashHCFusedParams &p [[buffer(7)]],
    uint3 grid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 group_size [[threads_per_threadgroup]]) {
  if (p.rows < 512 || p.rows > 2048 || p.width != 2560 || p.streams != 4 ||
      p.lowrank != 320 || p.arithmetic_mode > 1 || p.has_injection > 1 ||
      p.write_raw_up > 1 || (p.simdgroups != 4 && p.simdgroups != 8) ||
      p.norm_is_float > 1 || p.norm_convention > 1 ||
      !isfinite(p.norm_epsilon) || p.norm_epsilon <= 0 ||
      group_size.x != 640 || group_size.y != 1 || group_size.z != 1) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (grid.x >= p.rows || grid.y >= p.streams || grid.z) return;
  const ulong base = (ulong(grid.x) * p.streams + grid.y) * p.width;
  const bfloat gate = gates[ulong(grid.x) * p.streams + grid.y];
  bfloat values[4];
  float square_sum = 0.0f;
  for (uint element = 0; element < 4; ++element) {
    const uint column = tid * 4 + element;
    const bfloat product = bfloat(float(branch[ulong(grid.x) * p.width + column]) * float(gate));
    values[element] = bfloat(float(hyper[base + column]) + float(product));
    const float value = float(values[element]);
    square_sum += value * value;
  }
  square_sum = simd_sum(square_sum);
  threadgroup float partials[32];
  if (simd == 0) partials[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!lane) partials[simd] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0) {
    const float total = simd_sum(partials[lane]);
    if (!lane) partials[0] = metal::precise::rsqrt(total / float(p.width) + p.norm_epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint element = 0; element < 4; ++element) {
    const uint column = tid * 4 + element;
    const ulong offset = ulong(grid.y) * p.width + column;
    const float raw = p.norm_is_float
        ? reinterpret_cast<const device float *>(weight)[offset]
        : float(reinterpret_cast<const device bfloat *>(weight)[offset]);
    const float scale = p.norm_convention == 0 ? 1.0f + raw : raw;
    const float norm = float(values[element]) * partials[0];
    const bfloat result = bfloat(norm * scale);
    updated[base + column] = values[element]; normalized[base + column] = result;
    private_prefill_hc_inject_norm_sep21_check(float(values[element]), diagnostics);
    private_prefill_hc_inject_norm_sep21_check(float(result), diagnostics);
  }
}
