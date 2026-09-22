#pragma once
#ifndef __METAL_VERSION__
#include <stdint.h>
#endif

// Private benchmark ABI. K partitions preserve the full row stride.
struct UltraSplitKParams {
  uint32_t rows, input_size, output_size;
  uint32_t padded_rows, padded_outputs;
  uint32_t partition, tile_rows, tile_outputs;
};
static_assert(sizeof(UltraSplitKParams) == 32);
