#pragma once
#if !defined(__METAL_VERSION__)
#include <cstdint>
using Prefill4KQ27UInt = uint32_t;
#else
using Prefill4KQ27UInt = uint;
#endif
struct Prefill4KQ27Params {
  Prefill4KQ27UInt rows, input_size, output_size, tile_rows;
  Prefill4KQ27UInt tile_outputs, traversal, reserved0, reserved1;
};
#if !defined(__METAL_VERSION__)
static_assert(sizeof(Prefill4KQ27Params) == 32);
#endif
