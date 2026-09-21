#pragma once
#include "metal/abi/FlashMoEBuckets.h"

// Private offline signed INT8 coefficient candidate. This is a new weight
// representation, not the checkpoint's unsigned affine code representation.
struct FlashExpertInt8Params {
  uint32_t rows;
  uint32_t selections;
  uint32_t route_capacity;
  uint32_t job_capacity;
  uint32_t tile_rows;
  uint32_t stored_experts;
  uint32_t scale_group_size; // 0 means one F32 scale per output row.
  uint32_t reserved;
};
static_assert(sizeof(FlashExpertInt8Params) == 32);
