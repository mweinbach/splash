// Isolated hybrid expert job lists. Each expert selects M16/M32/M64 from
// its bucket count. Existing consumers bind the corresponding compact list.
#include <metal_stdlib>
#include "metal/abi/FlashMoEBuckets.h"
using namespace metal;
inline bool flash_moe_hybrid_bucket_base_geometry(constant FlashMoEBucketParams &p,
                                            uint threads) {
  return p.rows && p.rows <= kFlashMoEBucketMaximumRows && p.selections &&
      p.selections <= kFlashMoEBucketMaximumSelections &&
      p.width == 2560 && p.experts == 512 &&
      p.routes == p.rows * p.selections && !p.reserved && threads == 256;
}

inline bool flash_moe_hybrid_bucket_job_geometry(constant FlashMoEBucketParams &p,
                                           uint threads) {
  if (!flash_moe_hybrid_bucket_base_geometry(p, threads) ||
      (p.tile_rows != 16 && p.tile_rows != 32 && p.tile_rows != 64)) return false;
  return p.job_capacity == (p.routes + p.tile_rows - 1) / p.tile_rows + 511;
}

inline uint flash_moe_hybrid_tile(uint count) {
  return count <= 16 ? 16u : count <= 32 ? 32u : 64u;
}

// 0 counts, 1 offsets, 2 jobOffsets U32[513], 3 jobCount U32[1],
// 4 diagnostics, 5 parameters. No CPU readback of any dynamic count.
kernel void flash_moe_hybrid_bucket_job_prefix(
    const device uint *counts [[buffer(0)]],
    const device uint *offsets [[buffer(1)]],
    device uint *job_offsets [[buffer(2)]],
    device uint *job_count [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],
    constant FlashMoEBucketParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  if (!flash_moe_hybrid_bucket_job_geometry(p, threads)) {
    if (!tid) {
      job_count[0] = 0;
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    }
    return;
  }
  if (tid) return;
  uint total = 0;
  uint routes = 0;
  job_count[0] = 0;
  job_offsets[0] = 0;
  bool invalid = offsets[0] != 0;
  for (uint expert = 0; expert < p.experts; ++expert) {
    const uint count = counts[expert];
    const uint begin = offsets[expert];
    const uint end = offsets[expert + 1];
    if (begin != routes || begin > p.routes || end > p.routes ||
        end < begin || end - begin != count) invalid = true;
    // Never perform potentially overflowing count+M-1 for malformed buffers.
    const uint expert_jobs = count > p.routes ? p.job_capacity + 1 :
        flash_moe_hybrid_tile(count) == p.tile_rows ?
          (count + p.tile_rows - 1) / p.tile_rows : 0u;
    if (expert_jobs > p.job_capacity - total) invalid = true;
    if (invalid) {
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
      for (uint index = 0; index <= p.experts; ++index) job_offsets[index] = 0;
      return;
    }
    total += expert_jobs;
    routes = end;
    job_offsets[expert + 1] = total;
  }
  job_count[0] = total;
}

// Fixed-capacity GPU emission. An upper-bound search skips repeated offsets
// from empty experts, then maps each active job to its expert/packed row pair.
// 0 offsets, 1 jobOffsets, 2 jobCount, 3 jobs pairs, 4 diag, 5 params.
kernel void flash_moe_hybrid_bucket_jobs(
    const device uint *offsets [[buffer(0)]],
    const device uint *job_offsets [[buffer(1)]],
    const device uint *job_count [[buffer(2)]],
    device FlashMoEBucketJob *jobs [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],
    constant FlashMoEBucketParams &p [[buffer(5)]],
    uint index [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  if (!flash_moe_hybrid_bucket_job_geometry(p, threads)) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (index >= p.job_capacity) return;
  const uint count = job_count[0];
  if (count > p.job_capacity || index >= count) {
    jobs[index] = {UINT_MAX, 0};
    if (count > p.job_capacity && !tid)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  uint lower = 0;
  uint upper = p.experts;
  while (lower < upper) {
    const uint middle = (lower + upper) / 2;
    if (job_offsets[middle] <= index) lower = middle + 1;
    else upper = middle;
  }
  const uint expert = lower ? lower - 1 : UINT_MAX;
  if (expert >= p.experts) {
    jobs[index] = {UINT_MAX, 0};
    atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint first_job = job_offsets[expert];
  const uint end_job = job_offsets[expert + 1];
  const uint begin = offsets[expert];
  const uint end = offsets[expert + 1];
  if (first_job > index || end_job <= index || end_job > count ||
      begin > p.routes || end > p.routes || end <= begin ||
      flash_moe_hybrid_tile(end - begin) != p.tile_rows ||
      index - first_job >= (end - begin + p.tile_rows - 1) / p.tile_rows) {
    jobs[index] = {UINT_MAX, 0};
    atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  jobs[index] = {expert, begin + (index - first_job) * p.tile_rows};
}
