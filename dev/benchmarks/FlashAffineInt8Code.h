#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif
// Private G64 pilot. Codes are exact signed centered integers; coefficients
// remain the original BF16 scale/bias, and all affine epilogs remain F32.
struct FlashAffineInt8CodeParams {
  uint32_t rows, padded_rows, input_size, output_size;
  uint32_t bits, group_size, tile_rows, tile_outputs;
  uint64_t weight_row_stride_bytes, parameter_row_stride_bytes;
};
static_assert(sizeof(FlashAffineInt8CodeParams) == 48,
              "Private INT8 code ABI size changed");
