#include <metal_stdlib>
#include "metal/abi/FlashGDN.h"
using namespace metal;
inline bool wys_v5_control_valid(constant FlashGDNParams &p) {
  return p.rows && p.rows<=2048 && p.lanes && p.lanes<=32 &&
      p.key_heads==16 && p.value_heads==48 && p.key_dimension==128 && p.value_dimension==128 &&
      p.recurrent_lane_stride_bytes>=ulong(48)*128*128*4 && !(p.recurrent_lane_stride_bytes%4);
}
template <bool Restore>
inline void wys_v5_control_copy(device uint *state, device uint *snapshot, device atomic_uint *flags,
    device atomic_uint &diagnostics, constant FlashGDNParams &p,
    uint3 group, uint3 threads, uint tid) {
  if (!wys_v5_control_valid(p) || group.x>=48 || group.y>=64 || group.z>=p.lanes ||
      threads.x!=256 || threads.y!=1 || threads.z!=1) {
    if (!tid) atomic_fetch_or_explicit(&diagnostics,uint(FlashGDNInvalidParameters),memory_order_relaxed);
    return;
  }
  const uint head=group.x, batch=group.z, element=group.y*256+tid;
  const ulong source=ulong(batch)*p.recurrent_lane_stride_bytes/4+head*16384+element;
  const ulong saved=(ulong(batch)*48+head)*16384+element;
  if constexpr (Restore) {
    if (atomic_load_explicit(&flags[batch*48+head],memory_order_relaxed)) state[source]=snapshot[saved];
  } else {
    snapshot[saved]=state[source];
    if (!group.y && !tid) atomic_store_explicit(&flags[batch*48+head],0u,memory_order_relaxed);
  }
}
[[max_total_threads_per_threadgroup(256)]] kernel void private_gdn_wy_v5_snapshot(
    device uint *state [[buffer(0)]], device uint *snapshot [[buffer(1)]],
    device atomic_uint *flags [[buffer(2)]], device atomic_uint &diagnostics [[buffer(3)]],
    constant FlashGDNParams &p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  wys_v5_control_copy<false>(state,snapshot,flags,diagnostics,p,group,threads,tid);
}
[[max_total_threads_per_threadgroup(256)]] kernel void private_gdn_wy_v5_restore(
    device uint *snapshot [[buffer(0)]], device uint *state [[buffer(1)]],
    device atomic_uint *flags [[buffer(2)]], device atomic_uint &diagnostics [[buffer(3)]],
    constant FlashGDNParams &p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  wys_v5_control_copy<true>(state,snapshot,flags,diagnostics,p,group,threads,tid);
}
