#pragma once
#include "metal/abi/FlashHCFused.h"
struct HCDownPackedParams {
  FlashHCFusedParams literal;
  uint32_t padded_rows;
  uint32_t tile_outputs;
  uint32_t partitions;
  uint32_t write_debug;
};
static_assert(sizeof(HCDownPackedParams) == 176);
