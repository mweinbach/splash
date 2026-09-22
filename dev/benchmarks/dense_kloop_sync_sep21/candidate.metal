// Isolated device-BF16 accumulation-loop synchronization screen.
// Controls use whole-K multiply; candidates accumulate complete static-K
// blocks in one F32 cooperative destination and round to BF16 only once.
// Every partial final block uses dynamic K extents for bounds checking.
// Reduction ordering may differ from the controls; host output certificates
// determine numerical agreement. This source makes no equality claim.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashDenseCache.h"
#pragma METAL fp math_mode(safe)
using namespace metal;
using namespace mpp::tensor_ops;
#include "metal/kernels/common/flash_dense_traversal.h"

inline void dense_kloop_error(device atomic_uint *diagnostics, uint flag) {
  atomic_fetch_or_explicit(diagnostics, flag, memory_order_relaxed);
}

template <typename Dot>
inline void dense_kloop_store(Dot &dot, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashDenseCacheParams &p,
    uint row, uint column) {
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const bfloat value = bfloat(dot[i]);
    if (!isfinite(dot[i]) || !isfinite(float(value))) {
      dense_kloop_error(diagnostics, 4u);
    }
    output[ulong(row + index[1]) * p.output_size + column + index[0]] = value;
  }
}

template <ushort Groups, ushort BlockK = 0, ushort SyncFrequency = 0>
inline void dense_kloop_tile(device bfloat *input, device bfloat *weights,
    device bfloat *output, device atomic_uint *diagnostics,
    constant FlashDenseCacheParams &p, uint3 group, uint3 threads, uint tid) {
  constexpr ushort M = 128, N = 64;
  static_assert(Groups == 4 || Groups == 8, "Only SG4/SG8 are screened");
  static_assert(BlockK == 0 || BlockK == 256 || BlockK == 512 || BlockK == 1024,
      "Only whole K, K256, K512, and K1024 are screened");
  static_assert(SyncFrequency == 0 || SyncFrequency == 1 ||
      SyncFrequency == 2 || SyncFrequency == 4,
      "Only synchronization frequencies 0/1/2/4 are screened");
  if (p.rows != 2048 || !p.input_size || p.input_size > 32768 ||
      p.input_size % 32 || !p.output_size || !p.output_count ||
      p.output_count % N || p.output_begin > p.output_size ||
      p.output_count > p.output_size - p.output_begin ||
      p.tile_rows != M || p.tile_outputs != N || p.reserved > 4 || group.z ||
      !flash_dense_traversal_group_valid(group.xy, p.rows / M,
          p.output_count / N, p.reserved) ||
      threads.x != Groups * 32 || threads.y != 1 || threads.z != 1) {
    if (!tid) dense_kloop_error(diagnostics, 2u);
    return;
  }
  const uint2 tile = flash_dense_traversal_tile(group.xy, p.reserved);
  if (tile.x >= p.output_count / N || tile.y >= p.rows / M) return;
  const uint row = tile.y * M, column = p.output_begin + tile.x * N;
  const int k = int(p.input_size);
  auto a = tensor(input + ulong(row) * p.input_size,
      dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto b = tensor(weights + ulong(column) * p.input_size,
      dextents<int, 2>{k, N}, array<int, 2>{1, k});
  if constexpr (BlockK == 0) {
    // Exact descriptor of the existing strict device-BF16 whole-K control.
    constexpr auto descriptor = matmul2d_descriptor(M, N,
        static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<Groups>> operation;
    auto dot = operation.template get_destination_cooperative_tensor<
        decltype(a), decltype(b), float>();
    operation.run(a, b, dot);
    dense_kloop_store(dot, output, diagnostics, p, row, column);
  } else {
    constexpr auto descriptor = matmul2d_descriptor(M, N, BlockK,
        false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<Groups>> operation;
    auto dot = operation.template get_destination_cooperative_tensor<
        decltype(a), decltype(b), float>();
#pragma unroll
    for (ushort i = 0; i < dot.get_capacity(); ++i) {
      if (dot.is_valid_element(i)) dot[i] = 0.f;
    }
    const uint fullBlocks = p.input_size / BlockK;
    for (uint step = 0; step < fullBlocks; ++step) {
      if constexpr (SyncFrequency != 0) {
        if (step % SyncFrequency == 0) threadgroup_barrier(mem_flags::mem_none);
      }
      const int k0 = int(step * BlockK);
      auto ak = a.template slice<BlockK, M>(k0, 0);
      auto bk = b.template slice<BlockK, N>(k0, 0);
      operation.run(ak, bk, dot);
    }
    const uint tailBegin = fullBlocks * BlockK;
    if (tailBegin < p.input_size) {
      if constexpr (SyncFrequency != 0) {
        if (fullBlocks % SyncFrequency == 0) {
          threadgroup_barrier(mem_flags::mem_none);
        }
      }
      // At actual K2560/BK1024 this is K512. At synthetic K288/BK512
      // or BK1024 this is the sole operation: no full static slice is made.
      // dynamic_extent preserves the remaining K extent; operation.run
      // bounds-checks the missing descriptor lanes as required by MPP.
      auto ak = a.template slice<dynamic_extent, M>(int(tailBegin), 0);
      auto bk = b.template slice<dynamic_extent, N>(int(tailBegin), 0);
      operation.run(ak, bk, dot);
    }
    dense_kloop_store(dot, output, diagnostics, p, row, column);
  }
}

#define DENSE_KLOOP_WHOLE_ENTRY(G) \
[[max_total_threads_per_threadgroup(G * 32)]] \
kernel void dense_kloop_bf16_m128_n64_sg##G##_whole( \
    device bfloat *input [[buffer(0)]], device bfloat *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], \
    device atomic_uint *diagnostics [[buffer(3)]], \
    constant FlashDenseCacheParams &p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  dense_kloop_tile<G>( \
      input, weights, output, diagnostics, p, group, threads, tid); \
}

#define DENSE_KLOOP_BLOCK_ENTRY(G, K, S) \
[[max_total_threads_per_threadgroup(G * 32)]] \
kernel void dense_kloop_bf16_m128_n64_sg##G##_k##K##_sync##S( \
    device bfloat *input [[buffer(0)]], device bfloat *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], \
    device atomic_uint *diagnostics [[buffer(3)]], \
    constant FlashDenseCacheParams &p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]) { \
  dense_kloop_tile<G, K, S>( \
      input, weights, output, diagnostics, p, group, threads, tid); \
}

#define DENSE_KLOOP_SYNC_ENTRIES(G, K) \
DENSE_KLOOP_BLOCK_ENTRY(G, K, 0) \
DENSE_KLOOP_BLOCK_ENTRY(G, K, 1) \
DENSE_KLOOP_BLOCK_ENTRY(G, K, 2) \
DENSE_KLOOP_BLOCK_ENTRY(G, K, 4)

DENSE_KLOOP_WHOLE_ENTRY(4)
DENSE_KLOOP_WHOLE_ENTRY(8)
DENSE_KLOOP_SYNC_ENTRIES(4, 256)
DENSE_KLOOP_SYNC_ENTRIES(4, 512)
DENSE_KLOOP_SYNC_ENTRIES(4, 1024)
DENSE_KLOOP_SYNC_ENTRIES(8, 256)
DENSE_KLOOP_SYNC_ENTRIES(8, 512)
DENSE_KLOOP_SYNC_ENTRIES(8, 1024)
#undef DENSE_KLOOP_SYNC_ENTRIES
#undef DENSE_KLOOP_BLOCK_ENTRY
#undef DENSE_KLOOP_WHOLE_ENTRY
