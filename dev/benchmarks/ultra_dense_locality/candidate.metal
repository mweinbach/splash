// Isolated traversal/geometry experiment. There is no die or memory affinity.
// FlashDenseCacheParams.reserved encodes traversal only in this benchmark:
// 0 column-fast, 1 row-fast, 2/3/4 rows grouped by 2/4/8 (MLX style).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashDenseCache.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;

inline void locality_error(device atomic_uint *diagnostics, uint flag) {
  atomic_fetch_or_explicit(diagnostics, flag, memory_order_relaxed);
}

template <ushort M, ushort N, ushort Groups>
inline void locality_tile(device bfloat *input, device bfloat *weights,
    device bfloat *output, device atomic_uint *diagnostics,
    constant FlashDenseCacheParams &p, uint3 group, uint3 threads, uint tid) {
  if (!p.rows || p.rows > 8192 || !p.input_size || p.input_size > 32768 ||
      p.input_size % 32 || !p.output_size || !p.output_count ||
      p.output_count % N || p.output_begin > p.output_size ||
      p.output_count > p.output_size - p.output_begin || p.tile_rows != M ||
      p.tile_outputs != N || p.reserved > 4 || group.z ||
      threads.x != Groups * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) locality_error(diagnostics, 2u);
    return;
  }
  const uint rowTiles = (p.rows + M - 1) / M, columnTiles = p.output_count / N;
  uint tileRow = group.y, tileColumn = group.x;
  if (p.reserved == 1) {
    tileRow = group.x; tileColumn = group.y;
  } else if (p.reserved >= 2) {
    const uint log = p.reserved - 1;
    tileRow = (group.y << log) + (group.x & ((1u << log) - 1));
    tileColumn = group.x >> log;
  }
  // Swizzled dispatches may have unused groups in the final row block.
  if (tileRow >= rowTiles || tileColumn >= columnTiles) return;
  const uint row = tileRow * M, column = p.output_begin + tileColumn * N;
  const int k = int(p.input_size);
  // Host allocates/copies positive-zero padding to whole M rows. No OOB reads.
  auto a = tensor(input + ulong(row) * p.input_size,
      dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto b = tensor(weights + ulong(column) * p.input_size,
      dextents<int, 2>{k, N}, array<int, 2>{1, k});
  constexpr auto descriptor = matmul2d_descriptor(M, N,
      static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<Groups>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    if (row + index[1] >= p.rows) continue;
    const bfloat value = bfloat(dot[i]);
    if (!isfinite(dot[i]) || !isfinite(float(value))) locality_error(diagnostics, 4u);
    output[ulong(row + index[1]) * p.output_size + column + index[0]] = value;
  }
}
#define LOCALITY_ENTRY(M,N,G) \
[[max_total_threads_per_threadgroup(G*32)]] kernel void ultra_locality_m##M##_n##N( \
    device bfloat *input [[buffer(0)]], device bfloat *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], device atomic_uint *diagnostics [[buffer(3)]], \
    constant FlashDenseCacheParams &p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) { \
  locality_tile<M,N,G>(input,weights,output,diagnostics,p,group,threads,tid); \
}
LOCALITY_ENTRY(8,64,4)
LOCALITY_ENTRY(8,128,4)
LOCALITY_ENTRY(16,64,4)
LOCALITY_ENTRY(16,128,4)
LOCALITY_ENTRY(32,64,4)
LOCALITY_ENTRY(32,128,4)
LOCALITY_ENTRY(64,64,8)
LOCALITY_ENTRY(64,128,8)
LOCALITY_ENTRY(128,64,8)
#undef LOCALITY_ENTRY
