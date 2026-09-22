#include <metal_stdlib>
#include "metal/abi/FlashMoEBuckets.h"
using namespace metal;
inline bool prefill4k_hit_geometry(constant FlashMoEBucketParams &p,uint threads) {
  return p.rows && p.rows <=8192 && p.selections && p.selections <=10 && p.width ==2560 &&
      p.experts ==512 && p.routes ==p.rows *p.selections && !p.reserved && threads ==256 &&
      (p.tile_rows ==32 || p.tile_rows ==64 || p.tile_rows ==128) && p.job_capacity ==(p.routes +p.tile_rows -1) /p.tile_rows +511;
}
// Compact only stored experts into an independently tiled list. Original
// offsets/route order stay unchanged and Q4 misses keep their original jobs.
kernel void prefill4k_int8tiles_hit_prefix(device const uint *counts [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],device const uint *ranks [[buffer(2)]],
    device uint *job_offsets [[buffer(3)]],device uint *job_count [[buffer(4)]],
    device atomic_uint *diag [[buffer(5)]],constant FlashMoEBucketParams &p [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],uint threads [[threads_per_threadgroup]]) {
  if (!prefill4k_hit_geometry(p,threads)) {
    if (!tid) { job_count[0] =0; atomic_fetch_or_explicit(diag,2u,memory_order_relaxed); } return;
  }
  if (tid) return;
  uint total =0, routes =0;
  job_count[0] =0; job_offsets[0] =0;
  bool invalid =offsets[0] !=0;
  for (uint expert =0; expert <512; ++expert) {
    const uint count =counts[expert],begin =offsets[expert],end =offsets[expert +1];
    if (begin !=routes || begin >p.routes || end >p.routes || end <begin || end -begin !=count) invalid =true;
    const uint add =count >p.routes ? p.job_capacity +1 :
        ranks[expert] ==UINT_MAX ? 0 : (count +p.tile_rows -1) /p.tile_rows;
    if (add >p.job_capacity -total) invalid =true;
    if (invalid) {
      atomic_fetch_or_explicit(diag,2u,memory_order_relaxed);
      for (uint i =0; i <=512; ++i) job_offsets[i] =0;
      return;
    }
    total +=add; routes =end; job_offsets[expert +1] =total;
  }
  job_count[0] =total;
}
kernel void prefill4k_int8tiles_hit_jobs(device const uint *offsets [[buffer(0)]],
    device const uint *job_offsets [[buffer(1)]],device const uint *job_count [[buffer(2)]],
    device FlashMoEBucketJob *jobs [[buffer(3)]],device atomic_uint *diag [[buffer(4)]],
    constant FlashMoEBucketParams &p [[buffer(5)]],uint index [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint threads [[threads_per_threadgroup]]) {
  if (!prefill4k_hit_geometry(p,threads)) { if (!tid) atomic_fetch_or_explicit(diag,2u,memory_order_relaxed); return; }
  if (index >=p.job_capacity) return;
  const uint count =job_count[0];
  if (count >p.job_capacity || index >=count) {
    jobs[index] ={UINT_MAX,0};
    if (count >p.job_capacity && !tid) atomic_fetch_or_explicit(diag,2u,memory_order_relaxed);
    return;
  }
  uint lower =0,upper =512;
  while (lower <upper) { const uint middle =(lower +upper) /2;
    if (job_offsets[middle] <=index) lower =middle +1; else upper =middle; }
  const uint expert =lower ? lower -1 : UINT_MAX;
  if (expert >=512) { jobs[index] ={UINT_MAX,0}; atomic_fetch_or_explicit(diag,1u,memory_order_relaxed); return; }
  const uint first =job_offsets[expert],last =job_offsets[expert +1],begin =offsets[expert],end =offsets[expert +1];
  if (first >index || last <=index || last >count || begin >p.routes || end >p.routes || end <=begin ||
      index -first >=(end -begin +p.tile_rows -1) /p.tile_rows) {
    jobs[index] ={UINT_MAX,0}; atomic_fetch_or_explicit(diag,2u,memory_order_relaxed); return;
  }
  jobs[index] ={expert,begin +(index -first) *p.tile_rows};
}
