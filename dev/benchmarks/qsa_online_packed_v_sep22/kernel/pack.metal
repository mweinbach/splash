#include <metal_stdlib>
#include "PackedV.hpp"
#pragma METAL fp math_mode(safe)
using namespace metal;

// BF16 bit transpose only: no floating cast, arithmetic or diagnostic scan.
[[max_total_threads_per_threadgroup(256)]]
kernel void qsa_online_packed_v_pack(
    device const ushort *source [[buffer(0)]],
    device ushort *packed [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant QSAOnlinePackedVParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 totalGroups [[threadgroups_per_grid]],
    uint3 groupSize [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (params.rows != 2048 || params.kvHeads != 2 || params.dimensions != 256 ||
      params.reserved || any(totalGroups != uint3(64, 8, 2)) ||
      any(groupSize != uint3(256, 1, 1)) || group.x >= 64 || group.y >= 8 || group.z >= 2) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 1u << 9, memory_order_relaxed);
    return;
  }
  threadgroup ushort tile[32 * 33];
  const uint lane = tid % 32, warp = tid / 32;
#pragma unroll
  for (uint j = 0; j < 32; j += 8) {
    const uint token = group.x * 32 + warp + j;
    const uint dimension = group.y * 32 + lane;
    tile[(warp + j) * 33 + lane] = source[(ulong(token) * 2 + group.z) * 256 + dimension];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (uint j = 0; j < 32; j += 8) {
    const uint token = group.x * 32 + lane;
    const uint dimension = group.y * 32 + warp + j;
    packed[(ulong(group.z) * 256 + dimension) * 2048 + token] = tile[lane * 33 + warp + j];
  }
}
