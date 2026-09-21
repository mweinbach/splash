#pragma once

#include "metal/abi/FlashAffine.h"

struct FlashMoEDecodeF32CoefficientParams {
  FlashAffineParams affine;
  uint32_t expert;
  uint32_t output_begin;
  uint32_t input_begin;
  uint32_t reserved;
};
static_assert(sizeof(FlashMoEDecodeF32CoefficientParams)==80,
              "Private coefficient sampler ABI is80 bytes");
