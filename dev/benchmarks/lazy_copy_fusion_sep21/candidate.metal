#include "metal/abi/FlashGDN.h"
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Native carry validation/body copied literally from the frozen pointwise
// FlashGDN source. Only the three old-history stores are added before carry.
inline bool sep21_carry_parameters(FlashGDNParams p, device atomic_uint &d) {
  const bool valid =
      p.rows >= 1 && p.rows <= 8192 && p.lanes >= 1 && p.lanes <= 32 &&
      p.key_heads == 16 && p.value_heads == 48 && p.key_dimension == 128 &&
      p.value_dimension == 128 && p.convolution_taps == 4 &&
      isfinite(p.norm_epsilon) && p.norm_epsilon > 0.0f &&
      p.convolution_lane_stride_bytes >= ulong(3) * 10240 * sizeof(bfloat) &&
      p.convolution_lane_stride_bytes % sizeof(bfloat) == 0 &&
      p.recurrent_lane_stride_bytes >= ulong(48) * 128 * 128 * sizeof(float) &&
      p.recurrent_lane_stride_bytes % sizeof(float) == 0;
  if (!valid) atomic_fetch_or_explicit(&d, uint(FlashGDNInvalidParameters), memory_order_relaxed);
  return valid;
}

kernel void private_gdn_lazy_snapshot_convolution_carry_sep21(
    device const bfloat *qkv [[buffer(0)]], device bfloat *history [[buffer(1)]],
    device bfloat *initial_history [[buffer(2)]], device atomic_uint &diagnostics [[buffer(3)]],
    constant FlashGDNParams &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  if (!sep21_carry_parameters(p, diagnostics)) return;
  const uint channel = group.x * 256 + tid;
  const uint lane = group.y;
  if (channel >= 10240 || lane >= p.lanes) return;
  device bfloat *lane_history = history + ulong(lane) * p.convolution_lane_stride_bytes / sizeof(bfloat);
  device const bfloat *lane_input = qkv + ulong(lane) * p.rows * 10240;
  device bfloat *saved = initial_history + ulong(lane) * 3 * 10240;
  // All old source rows for this unique channel are saved before any overwrite.
  // The persistent verification dispatch has already finished every history read.
  for (uint row = 0; row < 3; ++row)
    saved[row * 10240 + channel] = lane_history[row * 10240 + channel];
  // Ascending destination order preserves old source rows for rows<3.
  for (uint row = 0; row < 3; ++row) {
    const uint source = p.rows + row;
    lane_history[row * 10240 + channel] = source < 3
        ? lane_history[source * 10240 + channel]
        : lane_input[ulong(source - 3) * 10240 + channel];
  }
}
