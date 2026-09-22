// Compile-only SDK probe. Never dispatch this kernel.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef PROBE_M
#define PROBE_M 16
#endif
#ifndef PROBE_N
#define PROBE_N 128
#endif
#ifndef PROBE_K
#define PROBE_K 128
#endif
#ifndef PROBE_SG
#define PROBE_SG 4
#endif
#ifndef PROBE_LEFT
#define PROBE_LEFT float
#endif
#ifndef PROBE_RIGHT
#define PROBE_RIGHT float
#endif
#ifndef PROBE_ADDRESS
#define PROBE_ADDRESS device
#endif
#ifndef PROBE_CONST
#define PROBE_CONST
#endif
#ifndef PROBE_COOPERATIVE_INPUT
#define PROBE_COOPERATIVE_INPUT 0
#endif
#ifndef PROBE_RELAXED
#define PROBE_RELAXED false
#endif
#ifndef PROBE_TRANSPOSE_RIGHT
#define PROBE_TRANSPOSE_RIGHT false
#endif
#ifndef PROBE_ACCUMULATE
#define PROBE_ACCUMULATE 0
#endif

#if PROBE_TRANSPOSE_RIGHT
#define PROBE_RIGHT_EXTENTS {PROBE_K, PROBE_N}
#define PROBE_RIGHT_STRIDES {1, PROBE_K}
#else
#define PROBE_RIGHT_EXTENTS {PROBE_N, PROBE_K}
#define PROBE_RIGHT_STRIDES {1, PROBE_N}
#endif

kernel void matmul_sdk_probe(
    PROBE_ADDRESS PROBE_CONST PROBE_LEFT *left,
    PROBE_ADDRESS PROBE_CONST PROBE_RIGHT *right,
    device float *destination [[buffer(2)]]) {
  auto a = tensor(left, dextents<int, 2>{PROBE_K, PROBE_M},
                  array<int, 2>{1, PROBE_K});
  auto b = tensor(right, dextents<int, 2>PROBE_RIGHT_EXTENTS,
                  array<int, 2>PROBE_RIGHT_STRIDES);
  constexpr auto descriptor = matmul2d_descriptor(
      PROBE_M, PROBE_N, PROBE_K, false, PROBE_TRANSPOSE_RIGHT,
      PROBE_RELAXED, PROBE_ACCUMULATE
          ? matmul2d_descriptor::mode::multiply_accumulate
          : matmul2d_descriptor::mode::multiply);
  static_assert(descriptor.relaxed_precision == PROBE_RELAXED);
  matmul2d<descriptor, execution_simdgroups<PROBE_SG>> op;
  auto result = op.template get_destination_cooperative_tensor<
      decltype(a), decltype(b), float>();
#if PROBE_ACCUMULATE
  for (ushort i = 0; i < result.get_capacity(); ++i)
    if (result.is_valid_element(i)) result[i] = 0.0f;
#endif
#if PROBE_COOPERATIVE_INPUT
  auto input = op.template get_left_input_cooperative_tensor<
      PROBE_LEFT, PROBE_RIGHT, float>();
  input.load(a);
  op.run(input, b, result);
#else
  op.run(a, b, result);
#endif
  for (ushort i = 0; i < result.get_capacity(); ++i) {
    if (!result.is_valid_element(i)) continue;
    auto index = result.get_multidimensional_index(i);
    destination[index[1] * PROBE_N + index[0]] = result[i];
  }
}
