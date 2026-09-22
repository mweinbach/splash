// Untimed stage certificate. Normal timed control/candidate both use the actual
// shared flash_hc_mix kernel. This exact source copy additionally records the
// BF16 gate, product and each sequential stream sum for quality measurement.
#include <metal_stdlib>
#include "abi.hpp"
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
inline bfloat hcup_sigmoid(bfloat source) {
  const bfloat exponential = bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponential;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}
// Buffers: normalized[0], raw[1], mixed[2], gates[3], products[4], sums[5],
// diagnostics[6], FlashHCParams[7]. All stage planes are BF16 [R,4,H].
kernel void hcup_post_probe(
    device const bfloat *normalized [[buffer(0)]], device const bfloat *raw [[buffer(1)]],
    device bfloat *mixed [[buffer(2)]], device bfloat *gates [[buffer(3)]],
    device bfloat *products [[buffer(4)]], device bfloat *sums [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]], constant FlashHCParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (p.rows != 2048 || p.width != 2560 || p.streams != 4 ||
      group.x >= 10 || group.y >= 2048 || group.z ||
      threads.x != 256 || threads.y != 1 || threads.z != 1) {
    if (!tid) atomic_fetch_or_explicit(diagnostics,2u,memory_order_relaxed);
    return;
  }
  const uint column = group.x * 256 + tid;
  bfloat total = bfloat(0.0f);
  for (uint stream = 0; stream < 4; ++stream) {
    const ulong at = (ulong(group.y) * 4 + stream) * 2560 + column;
    const bfloat gate = hcup_sigmoid(raw[at]);
    const bfloat product = bfloat(float(gate) * float(normalized[at]));
    total = bfloat(float(product) + float(total));
    gates[at] = gate; products[at] = product; sums[at] = total;
    if (!isfinite(float(gate)) || !isfinite(float(product)) || !isfinite(float(total)))
      atomic_fetch_or_explicit(diagnostics,4u,memory_order_relaxed);
  }
  const bfloat result = bfloat(float(total) / float(p.streams));
  mixed[ulong(group.y) * 2560 + column] = result;
  if (!isfinite(float(result))) atomic_fetch_or_explicit(diagnostics,4u,memory_order_relaxed);
}
