#pragma once

#include "metal/abi/FlashMoEBuckets.h"

// Private benchmark ABI. Every buffer must be disjoint and have its complete
// extent checked by the caller before submission. Bind one 256-thread group.
// Source jobs must be the complete canonical stable bucket job list emitted
// by flash_moe_bucket_jobs: ascending expert, then row_begin in M-row steps.
// Source count U32[1], ranks U32[512], offsets U32[513]. Inputs stay immutable.
// Hit/miss arrays each have the corresponding capacity below. All inactive
// records are {UINT32_MAX,0}, including on any source validation failure.
// Counts U32[2] are {hit_count,miss_count}, published only after all copies.
// Bind count views at byte offsets0/4 to existing projection consumers; the
// record format and original frozen source_job_capacity remain unchanged.
enum : uint32_t {
  kFlashInt8JobPartitionSourceJobs = 0,
  kFlashInt8JobPartitionSourceCount = 1,
  kFlashInt8JobPartitionRanks = 2,
  kFlashInt8JobPartitionOffsets = 3,
  kFlashInt8JobPartitionHitJobs = 4,
  kFlashInt8JobPartitionMissJobs = 5,
  kFlashInt8JobPartitionCounts = 6,
  kFlashInt8JobPartitionDiagnostics = 7,
  kFlashInt8JobPartitionParameters = 8,
};

struct FlashInt8JobPartitionParams {
  uint32_t rows;
  uint32_t selections;
  uint32_t route_capacity;
  uint32_t tile_rows;
  uint32_t stored_experts;
  uint32_t source_job_capacity;
  uint32_t hit_job_capacity;
  uint32_t miss_job_capacity;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
};

// Inputs are deliberately bounded before any addition/multiplication. Zero
// reports invalid metadata; no helper divides by an unchecked tile size.
inline constexpr bool flashInt8JobPartitionGeometry(uint32_t rows,
    uint32_t selections, uint32_t tile_rows, uint32_t stored_experts) {
  return rows && rows <= kFlashMoEBucketMaximumRows && selections &&
      selections <= kFlashMoEBucketMaximumSelections &&
      (tile_rows == 8 || tile_rows == 16 || tile_rows == 32 || tile_rows == 64) &&
      stored_experts && stored_experts <= 128;
}

inline constexpr uint32_t flashInt8JobPartitionSourceCapacity(uint32_t rows,
    uint32_t selections, uint32_t tile_rows, uint32_t stored_experts) {
  if (!flashInt8JobPartitionGeometry(rows, selections, tile_rows, stored_experts))
    return 0;
  return (rows * selections + tile_rows - 1) / tile_rows + 511;
}

inline constexpr uint32_t flashInt8JobPartitionHitCapacity(uint32_t rows,
    uint32_t selections, uint32_t tile_rows, uint32_t stored_experts) {
  if (!flashInt8JobPartitionGeometry(rows, selections, tile_rows, stored_experts))
    return 0;
  return (rows * selections + tile_rows - 1) / tile_rows + stored_experts - 1;
}

inline constexpr uint32_t flashInt8JobPartitionMissCapacity(uint32_t rows,
    uint32_t selections, uint32_t tile_rows, uint32_t stored_experts) {
  if (!flashInt8JobPartitionGeometry(rows, selections, tile_rows, stored_experts))
    return 0;
  return (rows * selections + tile_rows - 1) / tile_rows + 512 - stored_experts - 1;
}

static_assert(sizeof(FlashInt8JobPartitionParams) == 48,
              "Private INT8 job partition parameters are twelve uint32 values");
