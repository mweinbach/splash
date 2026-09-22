// Observes stable guard decisions; does not write flags, recurrence, output,
// prepared coefficients or request metadata. Exactly one thread is the writer.
#include <metal_stdlib>
using namespace metal;
struct WYTelemetryParams { uint heads; uint reserved; };
static_assert(sizeof(WYTelemetryParams)==8);
inline void wyt_add(device ulong *counters,uint slot,ulong amount) {
  const ulong old=counters[slot],limit=0xfffffffffffffffful;
  if (amount>limit-old) { counters[slot]=limit; counters[31]=1; }
  else counters[slot]=old+amount;
}
template <bool Replay>
inline void wyt_observe(device const uint *flags,device ulong *counters,
    constant WYTelemetryParams &p,uint3 group,uint3 threads,uint tid) {
  if (group.x || group.y || group.z || threads.x!=32 || threads.y!=1 ||
      threads.z!=1 || p.heads!=48 || p.reserved || tid) return;
  ulong selected=0,zero=0; uint reasonMask=0;
  for (uint head=0;head<48;++head) {
    const uint reason=flags[head];
    selected+=reason!=0; zero+=reason==0; reasonMask|=reason;
    if constexpr (Replay) {
      wyt_add(counters,7,ulong((reason&1u)!=0));
      wyt_add(counters,8,ulong((reason&2u)!=0));
      wyt_add(counters,9,ulong((reason&4u)!=0));
      wyt_add(counters,10,ulong((reason&8u)!=0));
      wyt_add(counters,11,ulong((reason&~15u)!=0));
      if (!(reason&~15u)) wyt_add(counters,12+(reason&15u),1);
    }
  }
  if constexpr (Replay) {
    wyt_add(counters,1,1); wyt_add(counters,5,selected); wyt_add(counters,6,zero);
    counters[29]=selected; counters[30]=ulong(reasonMask);
  } else {
    wyt_add(counters,0,1); wyt_add(counters,2,48);
    wyt_add(counters,3,zero); wyt_add(counters,4,zero);
    counters[28]=zero;
  }
}
#define WYT_ENTRY(NAME,REPLAY) \
[[max_total_threads_per_threadgroup(32)]] kernel void NAME( \
    device const uint *flags [[buffer(0)]],device ulong *counters [[buffer(1)]], \
    constant WYTelemetryParams &p [[buffer(2)]],uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) { \
  wyt_observe<REPLAY>(flags,counters,p,group,threads,tid); \
}
WYT_ENTRY(private_gdn_wy_telemetry_eligible,false)
WYT_ENTRY(private_gdn_wy_telemetry_replay,true)
#undef WYT_ENTRY
