#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using DenseI8UInt = uint;
#else
#include <stdint.h>
using DenseI8UInt = uint32_t;
#endif

// Private numerical weight-requantization experiment. BF16 activations remain
// unchanged; the runtime fits row-major signed-I8 codes plus positive F32 row
// scales from original F32 cached coefficients. All-zero rows use scale1.
struct DenseI8Params {
  DenseI8UInt rows;
  DenseI8UInt input_size;
  DenseI8UInt output_size;
  DenseI8UInt tile_rows;
  DenseI8UInt tile_outputs;
  DenseI8UInt reserved0;
  DenseI8UInt reserved1;
  DenseI8UInt reserved2;
};

static_assert(sizeof(DenseI8Params) == 32,
              "Dense I8 decode parameters are eight uint32 values");
