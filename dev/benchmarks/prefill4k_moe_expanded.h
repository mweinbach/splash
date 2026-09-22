#pragma once
#include "metal/abi/FlashMoEBlocked.h"
struct Prefill4KMoEExpandedParams {
  uint32_t input_size, output_size, experts, reserved;
  uint64_t weight_row_stride_bytes, weight_expert_stride_bytes;
  uint64_t parameter_row_stride_bytes, parameter_expert_stride_bytes;
};
static_assert(sizeof(Prefill4KMoEExpandedParams) == 48);
