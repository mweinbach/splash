#pragma once

#include "metal/abi/FlashDenseCache.h"

// Matmul uses the unchanged 32-byte FlashDenseCacheParams ABI. Converter
// dispatch is one 256-thread group per row; source/codes are [rows,input_size].
struct DenseW8A8QuantizeParams {
  uint32_t rows;
  uint32_t input_size;
};

static_assert(sizeof(DenseW8A8QuantizeParams) == 8,
              "Dense W8A8 row quantization parameters are two uint32 values");
