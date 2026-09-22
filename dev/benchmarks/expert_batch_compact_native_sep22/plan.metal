// Distinct PARALLEL integer-only fixed R8/R16 setup; every float consumer stays original.
#include <metal_stdlib>
#include "abi.hpp"
using namespace metal;
kernel void expert_batch_compact_native_sep22_plan(
    device const long *ids [[buffer(0)]],device uint *counts [[buffer(1)]],
    device uint *offsets [[buffer(2)]],device uint *routeMap [[buffer(3)]],
    device uint *inverse [[buffer(4)]],device uint *jobOffsets [[buffer(5)]],
    device FlashMoEBucketJob *jobs [[buffer(6)]],device uint *jobCount [[buffer(7)]],
    device atomic_uint *diag [[buffer(8)]],constant FlashMoEBucketParams &p [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],uint3 grid [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
  const bool valid=p.rows==kCompactNativeBatchRows&&p.selections==10&&p.width==2560&&p.experts==512&&p.routes==kCompactNativeBatchRoutes&&
      p.tile_rows==16&&p.job_capacity==kCompactNativeBatchJobCapacity&&!p.reserved&&all(grid==uint3(1))&&all(threads==uint3(256,1,1));
  if(!valid) {
    if(!tid)atomic_fetch_or_explicit(diag,2u,memory_order_relaxed);
    // One writer on a bad grid. A caller validates fixed extents beforehand;
    // zero live ranges/job count before original consumers can see stale data.
    if(all(group==uint3(0))&&!tid) {
      for(uint e=0;e<512;++e)counts[e]=0;
      for(uint e=0;e<513;++e){offsets[e]=0;jobOffsets[e]=0;}
      for(uint r=0;r<kCompactNativeBatchRoutes;++r){routeMap[r]=UINT_MAX;inverse[r]=UINT_MAX;}
      for(uint j=0;j<kCompactNativeBatchJobCapacity;++j)jobs[j]={UINT_MAX,0};
      *jobCount=0;
    }
    return;
  }
  threadgroup long cachedIDs[kCompactNativeBatchRoutes];
  threadgroup uint cachedCounts[512];
  threadgroup uint groupTotals[8],groupPrefixes[8];
  const uint lane=tid&31u,simd=tid/32u;
  if(tid<kCompactNativeBatchRoutes)cachedIDs[tid]=ids[tid];
  for(uint e=tid;e<513;e+=256){offsets[e]=0;jobOffsets[e]=0;}
  for(uint j=tid;j<kCompactNativeBatchJobCapacity;j+=256)jobs[j]={UINT_MAX,0};
  if(tid<kCompactNativeBatchRoutes){routeMap[tid]=UINT_MAX;inverse[tid]=UINT_MAX;}
  if(!tid)*jobCount=0;
  threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
  if(tid<kCompactNativeBatchRoutes) {
    const long id=cachedIDs[tid];
    if(id<0||id>=512)atomic_fetch_or_explicit(diag,1u,memory_order_relaxed);
    else {
      uint position=0;
      for(uint r=0;r<kCompactNativeBatchRoutes;++r) {
        const long other=cachedIDs[r];
        position+=uint(other>=0&&other<512&&(other<id||(other==id&&r<tid)));
        if(r>=tid/10*10&&r<tid&&other==id)atomic_fetch_or_explicit(diag,1u,memory_order_relaxed);
      }
      routeMap[position]=tid;inverse[tid]=position;
    }
  }
  const uint e0=tid*2,e1=e0+1;
  uint c0=0,c1=0;
  for(uint r=0;r<kCompactNativeBatchRoutes;++r){const long id=cachedIDs[r];c0+=uint(id==long(e0));c1+=uint(id==long(e1));}
  cachedCounts[e0]=c0;cachedCounts[e1]=c1;counts[e0]=c0;counts[e1]=c1;
  const uint routeLanePrefix=simd_prefix_exclusive_sum(c0+c1),routeGroupTotal=simd_sum(c0+c1);
  if(!lane)groupTotals[simd]=routeGroupTotal;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(!simd){const uint value=lane<8?groupTotals[lane]:0u;const uint prefix=simd_prefix_exclusive_sum(value);if(lane<8)groupPrefixes[lane]=prefix;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint first=groupPrefixes[simd]+routeLanePrefix;
  offsets[e0]=first;offsets[e1]=first+c0;if(tid==255)offsets[512]=first+c0+c1;
  // Reuse the same eight-group scan scratch for exact M16 job prefixes.
  const uint j0=(cachedCounts[e0]+15)/16,j1=(cachedCounts[e1]+15)/16;
  const uint jobLanePrefix=simd_prefix_exclusive_sum(j0+j1),jobGroupTotal=simd_sum(j0+j1);
  if(!lane)groupTotals[simd]=jobGroupTotal;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(!simd){const uint value=lane<8?groupTotals[lane]:0u;const uint prefix=simd_prefix_exclusive_sum(value);if(lane<8)groupPrefixes[lane]=prefix;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint jobFirst=groupPrefixes[simd]+jobLanePrefix;
  jobOffsets[e0]=jobFirst;jobOffsets[e1]=jobFirst+j0;
  if(tid==255){jobOffsets[512]=jobFirst+j0+j1;*jobCount=jobFirst+j0+j1;}
  // Each expert owner emits only its disjoint active job range. All unused
  // capacitykCompactNativeBatchJobCapacity records retain the sentinel set before the first barrier.
  for(uint j=0;j<j0;++j)jobs[jobFirst+j]={e0,first+j*16};
  for(uint j=0;j<j1;++j)jobs[jobFirst+j0+j]={e1,first+c0+j*16};
}
