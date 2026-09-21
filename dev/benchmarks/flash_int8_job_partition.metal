// Private stable GPU job-list partition. Does not require indirect dispatch
// support or host count readback. Projection launches may use the two smaller
// static class capacities while retaining their original parameter capacity.
#include <metal_stdlib>
#include "FlashInt8JobPartitionABI.h"

using namespace metal;

inline void flash_i8_job_partition_error(device atomic_uint *diagnostics,
                                         uint bits) {
  atomic_fetch_or_explicit(diagnostics, bits, memory_order_relaxed);
}

inline void flash_i8_job_partition_clear(device FlashMoEBucketJob *hits,
    device FlashMoEBucketJob *misses, uint hit_capacity, uint miss_capacity,
    uint tid) {
  for (uint i = tid; i < hit_capacity; i += 256)
    hits[i] = {UINT_MAX, 0};
  for (uint i = tid; i < miss_capacity; i += 256)
    misses[i] = {UINT_MAX, 0};
}

kernel void flash_int8_job_partition(
    device const FlashMoEBucketJob *source_jobs [[buffer(0)]],
    device const uint *source_count [[buffer(1)]],
    device const uint *ranks [[buffer(2)]],
    device const uint *offsets [[buffer(3)]],
    device FlashMoEBucketJob *hit_jobs [[buffer(4)]],
    device FlashMoEBucketJob *miss_jobs [[buffer(5)]],
    device uint *counts [[buffer(6)]],
    device atomic_uint *diagnostics [[buffer(7)]],
    constant FlashInt8JobPartitionParams &p [[buffer(8)]],
    uint3 grid [[threadgroups_per_grid]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  // Even malformed metadata cannot expose stale nonzero consumer counts.
  // The caller must always provide counts U32[2] and diagnostics U32[1].
  if (!tid && !group.x && !group.y && !group.z) { counts[0] = 0; counts[1] = 0; }
  const bool geometry = flashInt8JobPartitionGeometry(p.rows, p.selections,
      p.tile_rows, p.stored_experts) &&
      p.route_capacity == p.rows * p.selections &&
      p.source_job_capacity == flashInt8JobPartitionSourceCapacity(
          p.rows, p.selections, p.tile_rows, p.stored_experts) &&
      p.hit_job_capacity == flashInt8JobPartitionHitCapacity(
          p.rows, p.selections, p.tile_rows, p.stored_experts) &&
      p.miss_job_capacity == flashInt8JobPartitionMissCapacity(
          p.rows, p.selections, p.tile_rows, p.stored_experts) &&
      !p.reserved0 && !p.reserved1 && !p.reserved2 && !p.reserved3;
  if (!geometry || grid.x != 1 || grid.y != 1 || grid.z != 1 ||
      group.x || group.y || group.z || threads.x != 256 || threads.y != 1 ||
      threads.z != 1 || simd_width != 32) {
    if (!tid && !group.x && !group.y && !group.z)
      flash_i8_job_partition_error(diagnostics, 2u);
    return;
  }
  flash_i8_job_partition_clear(hit_jobs, miss_jobs, p.hit_job_capacity,
                              p.miss_job_capacity, tid);
  threadgroup_barrier(mem_flags::mem_device);

  const uint active = source_count[0];
  if (active > p.source_job_capacity) {
    if (!tid) flash_i8_job_partition_error(diagnostics, 2u);
    return;
  }

  threadgroup uint validation_flags[8];
  threadgroup uint expected_jobs[8];
  threadgroup uint failure[1];
  uint local_flags = offsets[0] != 0 || offsets[512] > p.route_capacity ? 2u : 0u;
  uint local_expected = 0;
  for (uint expert = tid; expert < 512; expert += 256) {
    const uint first = offsets[expert], end = offsets[expert + 1];
    if (first > end || end > p.route_capacity) local_flags |= 2u;
    else local_expected += (end - first + p.tile_rows - 1) / p.tile_rows;
  }
  // Validate the whole canonical list before storing a single active record.
  // Checking adjacent records detects omissions, duplicates, out-of-order
  // experts and row steps, so a short but superficially legal count fails.
  for (uint i = tid; i < active; i += 256) {
    const auto job = source_jobs[i];
    if (job.expert >= 512) { local_flags |= 1u; continue; }
    const uint first = offsets[job.expert], end = offsets[job.expert + 1];
    if (first > end || end > p.route_capacity || job.row_begin < first ||
        job.row_begin >= end) { local_flags |= 2u; continue; }
    const uint rank = ranks[job.expert];
    if (rank != UINT_MAX && rank >= p.stored_experts) local_flags |= 3u;
    if ((job.row_begin - first) % p.tile_rows) local_flags |= 2u;
    if (!i) {
      if (job.row_begin != first || first != 0) local_flags |= 2u;
    } else {
      const auto previous = source_jobs[i - 1];
      if (previous.expert >= 512) local_flags |= 1u;
      else if (previous.expert > job.expert) local_flags |= 2u;
      else if (previous.expert == job.expert) {
        if (previous.row_begin > p.route_capacity ||
            job.row_begin != previous.row_begin + p.tile_rows) local_flags |= 2u;
      } else {
        const uint previous_end = offsets[previous.expert + 1];
        if (previous.row_begin > p.route_capacity ||
            previous.row_begin + p.tile_rows < previous_end ||
            previous_end != first || job.row_begin != first) local_flags |= 2u;
      }
    }
    if (i + 1 == active &&
        (job.row_begin + p.tile_rows < end || end != offsets[512])) local_flags |= 2u;
  }
  const uint group_flags = simd_or(local_flags);
  const uint group_expected = simd_sum(local_expected);
  if (!lane) {
    validation_flags[simd] = group_flags;
    expected_jobs[simd] = group_expected;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!simd) {
    const uint combined_flags = simd_or(lane < 8 ? validation_flags[lane] : 0u);
    const uint combined_expected = simd_sum(lane < 8 ? expected_jobs[lane] : 0u);
    if (!lane) failure[0] = combined_flags | (combined_expected != active ? 2u : 0u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (failure[0]) {
    if (!tid) flash_i8_job_partition_error(diagnostics, failure[0]);
    return;
  }

  threadgroup uint hit_totals[8], miss_totals[8];
  threadgroup uint hit_prefixes[8], miss_prefixes[8];
  threadgroup uint tile_totals[2];
  uint previous_hits = 0, previous_misses = 0;
  for (uint base = 0; base < active; base += 256) {
    const uint i = base + tid;
    FlashMoEBucketJob job = {UINT_MAX, 0};
    uint hit = 0, miss = 0;
    if (i < active) {
      job = source_jobs[i];
      hit = uint(ranks[job.expert] != UINT_MAX);
      miss = 1u - hit;
    }
    const uint hit_lane_prefix = simd_prefix_exclusive_sum(hit);
    const uint miss_lane_prefix = simd_prefix_exclusive_sum(miss);
    const uint hit_group_total = simd_sum(hit), miss_group_total = simd_sum(miss);
    if (!lane) { hit_totals[simd] = hit_group_total; miss_totals[simd] = miss_group_total; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!simd) {
      const uint h = lane < 8 ? hit_totals[lane] : 0u;
      const uint m = lane < 8 ? miss_totals[lane] : 0u;
      const uint hp = simd_prefix_exclusive_sum(h), mp = simd_prefix_exclusive_sum(m);
      const uint ht = simd_sum(h), mt = simd_sum(m);
      if (lane < 8) { hit_prefixes[lane] = hp; miss_prefixes[lane] = mp; }
      if (!lane) { tile_totals[0] = ht; tile_totals[1] = mt; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (previous_hits + tile_totals[0] > p.hit_job_capacity ||
        previous_misses + tile_totals[1] > p.miss_job_capacity) {
      if (!tid) flash_i8_job_partition_error(diagnostics, 2u);
      flash_i8_job_partition_clear(hit_jobs, miss_jobs, p.hit_job_capacity,
                                  p.miss_job_capacity, tid);
      return;
    }
    if (hit) {
      const uint destination = previous_hits + hit_prefixes[simd] + hit_lane_prefix;
      if (destination < p.hit_job_capacity) hit_jobs[destination] = job;
    }
    if (miss) {
      const uint destination = previous_misses + miss_prefixes[simd] + miss_lane_prefix;
      if (destination < p.miss_job_capacity) miss_jobs[destination] = job;
    }
    previous_hits += tile_totals[0];
    previous_misses += tile_totals[1];
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (!tid) { counts[0] = previous_hits; counts[1] = previous_misses; }
}
