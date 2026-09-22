#pragma once
#include "metal/abi/FlashDenseCache.h"
#include "metal/abi/FlashHC.h"
struct DenseW8A8QuantizeParams { uint32_t rows, input_size; };
static_assert(sizeof(DenseW8A8QuantizeParams) == 8);
