// FIRST compact-integer-only R4 setup; every pack/MPP/preparation kernel stays original.
#include <metal_stdlib>
#include "abi.hpp"
using namespace metal;
kernel void expert_r4_compact_native_sep22_plan(
    device const long *ids [[buffer(0)]],device uint *counts [[buffer(1)]],
    device uint *offsets [[buffer(2)]],device uint *routeMap [[buffer(3)]],
    device uint *inverse [[buffer(4)]],device uint *jobOffsets [[buffer(5)]],
    device FlashMoEBucketJob *jobs [[buffer(6)]],device uint *jobCount [[buffer(7)]],
    device atomic_uint *diag [[buffer(8)]],constant FlashMoEBucketParams &p [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],uint3 grid [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
  const bool valid=p.rows==4&&p.selections==10&&p.width==2560&&p.experts==512&&p.routes==40&&
      p.tile_rows==16&&p.job_capacity==514&&!p.reserved&&all(grid==uint3(1))&&all(threads==uint3(256,1,1));
  if(!valid) {
    if(!tid)atomic_fetch_or_explicit(diag,2u,memory_order_relaxed);
    // One writer on a bad grid. A caller validates fixed extents beforehand;
    // zero live ranges/job count before original consumers can see stale data.
    if(all(group==uint3(0))&&!tid) {
      for(uint e=0;e<512;++e)counts[e]=0;
      for(uint e=0;e<513;++e){offsets[e]=0;jobOffsets[e]=0;}
      for(uint r=0;r<40;++r){routeMap[r]=UINT_MAX;inverse[r]=UINT_MAX;}
      for(uint j=0;j<514;++j)jobs[j]={UINT_MAX,0};
      *jobCount=0;
    }
    return;
  }
  for(uint e=tid;e<512;e+=256)counts[e]=0;
  for(uint e=tid;e<513;e+=256){offsets[e]=0;jobOffsets[e]=0;}
  for(uint j=tid;j<514;j+=256)jobs[j]={UINT_MAX,0};
  if(tid<40){routeMap[tid]=UINT_MAX;inverse[tid]=UINT_MAX;}
  if(!tid)*jobCount=0;
  threadgroup_barrier(mem_flags::mem_device);
  if(!tid) {
    for(uint route=0;route<40;++route) {
      const long id=ids[route];
      if(id<0||id>=512){atomic_fetch_or_explicit(diag,1u,memory_order_relaxed);continue;}
      for(uint previous=route/10*10;previous<route;++previous)
        if(ids[previous]==id)atomic_fetch_or_explicit(diag,1u,memory_order_relaxed);
      ++counts[uint(id)];
    }
    uint total=0,active=0;
    for(uint expert=0;expert<512;++expert) {
      offsets[expert]=total;jobOffsets[expert]=active;
      for(uint route=0;route<40;++route)if(ids[route]==long(expert)) {
        routeMap[total]=route;inverse[route]=total;++total;
      }
      for(uint row=offsets[expert];row<total;row+=16)jobs[active++]={expert,row};
    }
    offsets[512]=total;jobOffsets[512]=active;*jobCount=active;
  }
}
