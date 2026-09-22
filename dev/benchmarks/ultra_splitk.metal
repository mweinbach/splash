#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "ultra_splitk_params.h"
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
using namespace metal;
using namespace mpp::tensor_ops;

inline void ultra_splitk_error(device atomic_uint *diagnostics, uint flag) {
  atomic_fetch_or_explicit(diagnostics, flag, memory_order_relaxed);
}
inline bool ultra_splitk_geometry(constant UltraSplitKParams &p) {
  return p.rows && p.rows <= 8192 && p.input_size && p.input_size <= 32768 &&
      p.input_size % 32 == 0 && p.output_size && p.output_size <= 32768 &&
      p.tile_rows && p.tile_outputs &&
      p.padded_rows == (p.rows + p.tile_rows - 1) / p.tile_rows * p.tile_rows &&
      p.padded_outputs == (p.output_size + p.tile_outputs - 1) /
          p.tile_outputs * p.tile_outputs &&
      (!p.partition || (p.partition >= 32 && p.partition % 32 == 0));
}
template<typename W, ushort M, ushort N, ushort SIMD>
inline void ultra_splitk_tile(device bfloat *input, device W *weights,
    device bfloat *output, device float *partial, device atomic_uint *diagnostics,
    constant UltraSplitKParams &p, uint3 group, uint3 threads, uint tid) {
  const uint partitions = p.partition ? (p.input_size + p.partition - 1) / p.partition : 1;
  if (!ultra_splitk_geometry(p) || p.tile_rows != M || p.tile_outputs != N ||
      group.x >= p.padded_outputs / N || group.y >= p.padded_rows / M ||
      group.z >= partitions || threads.x != SIMD * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) ultra_splitk_error(diagnostics, 2u); return;
  }
  const uint row = group.y * M, column = group.x * N;
  const uint begin = p.partition ? group.z * p.partition : 0;
  const int k = int(p.partition ? min(p.partition, p.input_size - begin) : p.input_size);
  const int stride = int(p.input_size);
  auto a = tensor(input + ulong(row) * stride + begin,
      dextents<int, 2>{k, M}, array<int, 2>{1, stride});
  auto b = tensor(weights + ulong(column) * stride + begin,
      dextents<int, 2>{k, N}, array<int, 2>{1, stride});
  constexpr auto descriptor = matmul2d_descriptor(M, N, static_cast<int>(dynamic_extent),
      false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SIMD>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const uint r = row + index[1], n = column + index[0];
    if (!isfinite(dot[i])) ultra_splitk_error(diagnostics, 4u);
    if (p.partition) {
      // FP32 partials never round to BF16 before the final reduction.
      partial[(ulong(group.z) * p.padded_rows + r) * p.padded_outputs + n] = dot[i];
    } else if (r < p.rows && n < p.output_size) {
      const bfloat value = bfloat(dot[i]);
      if (!isfinite(float(value))) ultra_splitk_error(diagnostics, 4u);
      output[ulong(r) * p.output_size + n] = value;
    }
  }
}
#define ULTRA_SPLITK_ENTRY(Name,W,M,N,SIMD) \
kernel void Name(device bfloat *input [[buffer(0)]], device W *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], device float *partial [[buffer(3)]], \
    device atomic_uint *diagnostics [[buffer(4)]], constant UltraSplitKParams &p [[buffer(5)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  ultra_splitk_tile<W,M,N,SIMD>(input,weights,output,partial,diagnostics,p,group,threads,tid); \
}
ULTRA_SPLITK_ENTRY(ultra_splitk_m8_n128_s4,bfloat,8,128,4)
ULTRA_SPLITK_ENTRY(ultra_splitk_m16_n128_s4,bfloat,16,128,4)
ULTRA_SPLITK_ENTRY(ultra_splitk_m64_n128_s8,bfloat,64,128,8)
ULTRA_SPLITK_ENTRY(ultra_splitk_f32_m8_n64_s4,float,8,64,4)
ULTRA_SPLITK_ENTRY(ultra_splitk_f32_m16_n64_s4,float,16,64,4)
ULTRA_SPLITK_ENTRY(ultra_splitk_f32_m8_n128_s4,float,8,128,4)
ULTRA_SPLITK_ENTRY(ultra_splitk_f32_m16_n128_s4,float,16,128,4)

kernel void ultra_splitk_pad(device bfloat *input [[buffer(0)]],
    device bfloat *padded [[buffer(1)]], device atomic_uint *diagnostics [[buffer(2)]],
    constant UltraSplitKParams &p [[buffer(3)]], uint3 position [[thread_position_in_grid]]) {
  const uint index = position.x;
  if (!ultra_splitk_geometry(p) || position.y || position.z) {
    if (!index) ultra_splitk_error(diagnostics, 2u); return;
  }
  if (index >= p.padded_rows * p.input_size) return;
  padded[index] = index < p.rows * p.input_size ? input[index] : bfloat(0.0f);
}

kernel void ultra_splitk_reduce(device float *partial [[buffer(0)]],
    device bfloat *output [[buffer(1)]], device atomic_uint *diagnostics [[buffer(2)]],
    constant UltraSplitKParams &p [[buffer(3)]], uint3 position [[thread_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]]) {
  const uint index = position.x;
  if (!ultra_splitk_geometry(p) || !p.partition || position.y || position.z || threads.x != 256 ||
      threads.y != 1 || threads.z != 1) {
    if (!index) ultra_splitk_error(diagnostics, 2u); return;
  }
  if (index >= p.rows * p.output_size) return;
  const uint row = index / p.output_size, column = index % p.output_size;
  const uint count = (p.input_size + p.partition - 1) / p.partition;
  float sum = 0.0f;
  for (uint part = 0; part < count; ++part)
    sum += partial[(ulong(part) * p.padded_rows + row) * p.padded_outputs + column];
  const bfloat value = bfloat(sum);
  if (!isfinite(sum) || !isfinite(float(value))) ultra_splitk_error(diagnostics, 4u);
  output[index] = value;
}
