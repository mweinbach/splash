#include "bandwidth.hpp"

inline uint4 bw_pattern(ulong vector, uint seed) {
  const uint first = uint(vector * 4) * kBWMultiplier + seed;
  return uint4(first, first + kBWMultiplier, first + 2u*kBWMultiplier,
               first + 3u*kBWMultiplier);
}

kernel void bw_initialize(device uint4 *data [[buffer(0)]],
    constant BWParams &p [[buffer(1)]], uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const ulong begin = ulong(group)*p.chunkVectors;
  const ulong end = min(begin+p.chunkVectors,p.vectors);
  for(ulong i=begin+tid;i<end;i+=kBWThreads) data[i]=bw_pattern(i,p.seedA);
}

// Exact component sums retain every loaded uint4 and are checked for every CTA.
// uint64 sums avoid the repeated-CTA modulo32 checksum degeneracy.
// Metal SIMD arithmetic reductions accept32-bit integers. Shuffle explicit
// low/high uint32 pairs, reconstructulong, and add with exact64-bit carries.
inline ulong bw_simd_sum64(ulong value,uint lane) {
  for(uint offset=16;offset;offset/=2){
    const uint2 parts=simd_shuffle_down(uint2(uint(value),uint(value>>32)),offset);
    if(lane+offset<32)value+=ulong(parts.x)|(ulong(parts.y)<<32);
  }
  return value;
}
inline bool bw_geometry(device BWRecord *records,constant BWParams &p,
    uint group,uint tid,uint threads,uint width) {
  if(threads==kBWThreads&&width==32)return true;
  if(!tid){BWRecord r={};r.badWords=0xffffffffu;r.stamp=p.stamp;records[group]=r;}
  return false;
}
inline void bw_finish(device BWRecord *records, constant BWParams &p,
    ulong4 sum, uint count, uint bad, uint first, uint last,
    threadgroup ulong4 *groupSums, threadgroup uint2 *groupCounts,
    uint tid, uint group, uint lane, uint simdgroup) {
  const ulong4 reduced = ulong4(bw_simd_sum64(sum.x,lane),bw_simd_sum64(sum.y,lane),
                                bw_simd_sum64(sum.z,lane),bw_simd_sum64(sum.w,lane));
  const uint2 counts=uint2(simd_sum(count),simd_sum(bad));
  if(!lane){groupSums[simdgroup]=reduced;groupCounts[simdgroup]=counts;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid<32){
    const ulong4 s=tid<8?groupSums[tid]:ulong4(0);
    const uint2 c=tid<8?groupCounts[tid]:uint2(0);
    const ulong4 total=ulong4(bw_simd_sum64(s.x,lane),bw_simd_sum64(s.y,lane),
                            bw_simd_sum64(s.z,lane),bw_simd_sum64(s.w,lane));
    const uint2 nc=uint2(simd_sum(c.x),simd_sum(c.y));
    if(!tid){
      BWRecord r;
      r.sums[0]=total.x;r.sums[1]=total.y;r.sums[2]=total.z;r.sums[3]=total.w;
      r.vectors=nc.x;r.first=first;r.last=last;r.badWords=nc.y;r.stamp=p.stamp;
      records[group]=r;
    }
  }
}

kernel void bw_read(device const uint4 *a [[buffer(0)]],
    device const uint4 *b [[buffer(1)]],device BWRecord *records [[buffer(2)]],
    constant BWParams &p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint threads [[threads_per_threadgroup]],
    uint width [[threads_per_simdgroup]]) {
  threadgroup ulong4 sums[8];threadgroup uint2 counts[8];
  if(!bw_geometry(records,p,group,tid,threads,width))return;
  const ulong begin=ulong(group)*p.chunkVectors,end=min(begin+p.chunkVectors,p.vectors);
  ulong4 sum(0);uint count=0;
  for(ulong i=begin+tid;i<end;i+=kBWThreads){
    const uint4 v=i<p.vectorsPerBuffer?a[i]:b[i-p.vectorsPerBuffer];
    sum+=ulong4(v);++count;
  }
  uint first=0,last=0;
  if(!tid){
    first=begin<p.vectorsPerBuffer?a[begin].x:b[begin-p.vectorsPerBuffer].x;
    const ulong j=end-1;
    last=j<p.vectorsPerBuffer?a[j].w:b[j-p.vectorsPerBuffer].w;
  }
  bw_finish(records,p,sum,count,0,first,last,sums,counts,tid,group,lane,sg);
}

kernel void bw_copy(device const uint4 *source [[buffer(0)]],
    device uint4 *destination [[buffer(1)]],device BWRecord *records [[buffer(2)]],
    constant BWParams &p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint threads [[threads_per_threadgroup]],
    uint width [[threads_per_simdgroup]]) {
  threadgroup uint groupCounts[8];
  if(!bw_geometry(records,p,group,tid,threads,width))return;
  const ulong begin=ulong(group)*p.chunkVectors,end=min(begin+p.chunkVectors,p.vectors);
  uint count=0;
  for(ulong i=begin+tid;i<end;i+=kBWThreads){destination[i]=source[i];++count;}
  const uint n=simd_sum(count);
  if(!lane)groupCounts[sg]=n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid<32){
    const uint total=simd_sum(tid<8?groupCounts[tid]:0u);
    if(!tid){BWRecord r;r.sums[0]=r.sums[1]=r.sums[2]=r.sums[3]=0;
      r.vectors=total;r.first=r.last=r.badWords=0;r.stamp=p.stamp;records[group]=r;}
  }
}

// Outside all timed scopes, validates every destination/source word against its
// immutable index/seed pattern; a checksum alone is not copy correctness proof.
kernel void bw_validate(device const uint4 *data [[buffer(0)]],
    device BWRecord *records [[buffer(1)]],constant BWParams &p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],uint width [[threads_per_simdgroup]]) {
  threadgroup ulong4 sums[8];threadgroup uint2 counts[8];
  if(!bw_geometry(records,p,group,tid,threads,width))return;
  const ulong begin=ulong(group)*p.chunkVectors,end=min(begin+p.chunkVectors,p.vectors);
  ulong4 sum(0);uint count=0,bad=0;
  for(ulong i=begin+tid;i<end;i+=kBWThreads){
    const uint4 v=data[i],expected=bw_pattern(i,p.seedA);
    sum+=ulong4(v);++count;
    bad+=uint(v.x!=expected.x)+uint(v.y!=expected.y)+uint(v.z!=expected.z)+uint(v.w!=expected.w);
  }
  const uint first=!tid?data[begin].x:0,last=!tid?data[end-1].w:0;
  bw_finish(records,p,sum,count,bad,first,last,sums,counts,tid,group,lane,sg);
}
