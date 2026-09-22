#include <metal_stdlib>
#include "metal/abi/FlashGDN.h"
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
using namespace metal;
enum : uint { GWY_NORM = 8u };
inline void gwy_weight_norm(device const float *prepared, ulong base, uint token,
    uint simdLane, thread float &result, thread uint &reason) {
  float sum = 0.0f;
  bool bad = false;
  for (uint i = 0; i < 4; ++i) {
    const uint raw = reinterpret_cast<device const uint *>(prepared)[base + token * 128 + 4 * simdLane + i];
    const float x = as_type<float>(raw);
    const float square = x * x;
    if (((raw & 0x7fffffffu) && square < 0x1p-126f) ||
        !isfinite(square) || square >= 0x1.fffffep127f) bad = true;
    sum += square;
  }
  const float norm = simd_sum(sum);
  if (!isfinite(norm) || norm >= 0x1.fffffep127f) bad = true;
  reason = simd_any(bad) ? uint(GWY_NORM) : 0u;
  result = sqrt(norm) * 1.000125f;
}

template <ushort Time, ushort Groups>
inline void norm_reference(device const float *prepared, device float *sidecar,
    device atomic_uint &diagnostics, constant FlashGDNParams &p,
    uint3 group, uint3 threads, uint tid) {
  const uint chunks=(p.rows+Time-1)/Time;
  if (!p.rows || p.rows>2048 || !p.lanes || p.lanes>32 || group.x>=48 ||
      group.y>=chunks || group.z>=p.lanes || threads.x!=Groups*32 ||
      threads.y!=1 || threads.z!=1) {
    if (!tid) atomic_fetch_or_explicit(&diagnostics,uint(FlashGDNInvalidParameters),memory_order_relaxed);
    return;
  }
  const ulong item=(ulong(group.z)*chunks+group.y)*48+group.x;
  const ulong base=item*(3*Time*128+Time*Time+Time);
  const uint simdGroup=tid/32,simdLane=tid%32;
  for (uint token=simdGroup;token<Time;token+=Groups) {
    float value; uint reason;
    gwy_weight_norm(prepared,base,token,simdLane,value,reason);
    if (!simdLane) {
      sidecar[item*2*Time+token]=value;
      reinterpret_cast<device uint *>(sidecar)[item*2*Time+Time+token]=reason;
    }
  }
}
#define NORM_ENTRY(T,G) \
[[max_total_threads_per_threadgroup(G*32)]] kernel void private_gdn_wy_norm_reference_t##T##_sg##G( \
    device const float *prepared [[buffer(0)]], device float *sidecar [[buffer(1)]], \
    device atomic_uint &diagnostics [[buffer(2)]], constant FlashGDNParams &p [[buffer(3)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  norm_reference<T,G>(prepared,sidecar,diagnostics,p,group,threads,tid); \
}
NORM_ENTRY(32,4)
NORM_ENTRY(32,8)
#undef NORM_ENTRY
