// Copyright © 2025 Apple Inc.
// Private BF16 screen using unmodified MLX Steel reduction primitives pinned to
// 2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99. The wrapper is adapted from Apple's
// steel_gemm_fused_nax.h; vendored copyright notices and MIT license are retained.
// This is the official relaxed-precision NAX arithmetic, not Splash whole-K MPP.
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "metal/abi/FlashDenseCache.h"
#include "metal/kernels/common/flash_dense_traversal.h"

using namespace metal;
using namespace mlx::steel;

inline void steel_screen_error(device atomic_uint *diagnostics, uint flag) {
  atomic_fetch_or_explicit(diagnostics, flag, memory_order_relaxed);
}

template <ushort BM, ushort BN, ushort WM, ushort WN, bool AlignedN, bool AlignedK>
inline void steel_screen_compute(const device bfloat *input,
    const device bfloat *weights, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashDenseCacheParams &p,
    uint2 tile, uint simdGroup) {
  constexpr short SM = BM / WM, SN = BN / WN, SK = 32, BK = 512;
  const uint rowOrigin = tile.y * BM, columnWithin = tile.x * BN;
  const uint columnOrigin = p.output_begin + columnWithin;
  const short tm = SM * (simdGroup / WN), tn = SN * (simdGroup % WN);
  const short sgpSM = SM;
  const short sgpSN = AlignedN ? SN : short(max(0, min(int(SN),
      int(p.output_count) - int(columnWithin) - int(tn))));
  const device bfloat *a = input + ulong(rowOrigin + tm) * p.input_size;
  // Inactive SIMD subtiles still participate in the BK loop's threadgroup
  // barriers. Give them an in-bounds placeholder; gemm_loop never reads it.
  const uint physicalColumn = min(columnOrigin + uint(tn),
      p.output_begin + p.output_count - 1);
  const device bfloat *b = weights + ulong(physicalColumn) * p.input_size;
  auto dot = gemm_loop<bfloat, SM, SN, SK, BK, false, true, true,
      AlignedN, AlignedK, float>(a, b, int(p.input_size), int(p.input_size),
      int(p.input_size), int(p.input_size / BK), sgpSM, sgpSN);
  if (!sgpSN) return;
#pragma unroll
  for (short i = 0; i < decltype(dot)::kElemsPerTile; ++i) {
    const float value = dot.elems()[i];
    if (!isfinite(value) || !isfinite(float(bfloat(value))))
      steel_screen_error(diagnostics, 4u);
  }
  device bfloat *destination = output + ulong(rowOrigin + tm) * p.output_size +
      columnOrigin + uint(tn);
  if constexpr (AlignedN)
    dot.store(destination, int(p.output_size));
  else
    dot.store_safe(destination, int(p.output_size), short2(sgpSN, sgpSM));
}

template <ushort BM, ushort BN, ushort WM, ushort WN>
inline void steel_screen_tile(const device bfloat *input,
    const device bfloat *weights, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashDenseCacheParams &p,
    uint3 group, uint3 threads, uint tid, uint simdGroup) {
  static_assert(WM * WN == 8);
  const uint columnTiles = (p.output_count + BN - 1) / BN;
  if (p.rows != 2048 || !p.input_size || p.input_size > 32768 ||
      p.input_size % 32 || !p.output_size || p.output_size > 32768 ||
      !p.output_count || p.output_count % 64 ||
      p.output_begin > p.output_size || p.output_count > p.output_size - p.output_begin ||
      p.tile_rows != BM || p.tile_outputs != BN || p.reserved > 4 || group.z ||
      !flash_dense_traversal_group_valid(group.xy, p.rows / BM,
          columnTiles, p.reserved) ||
      threads.x != 256 || threads.y != 1 || threads.z != 1) {
    if (!tid) steel_screen_error(diagnostics, 2u);
    return;
  }
  const uint2 tile = flash_dense_traversal_tile(group.xy, p.reserved);
  if (tile.x >= columnTiles || tile.y >= p.rows / BM) return;
  // These predicates are uniform across all eight SIMD groups. The unmodified
  // official aligned loop would omit a non-BK512 tail, so select its safe tail
  // implementation for K320/K640 and other supported nonmultiples of 512.
  if (p.output_count % BN == 0) {
    if (p.input_size % 512 == 0)
      steel_screen_compute<BM,BN,WM,WN,true,true>(input,weights,output,diagnostics,p,tile,simdGroup);
    else
      steel_screen_compute<BM,BN,WM,WN,true,false>(input,weights,output,diagnostics,p,tile,simdGroup);
  } else {
    if (p.input_size % 512 == 0)
      steel_screen_compute<BM,BN,WM,WN,false,true>(input,weights,output,diagnostics,p,tile,simdGroup);
    else
      steel_screen_compute<BM,BN,WM,WN,false,false>(input,weights,output,diagnostics,p,tile,simdGroup);
  }
}

#define STEEL_SCREEN_ENTRY(BM,BN,WM,WN) \
[[max_total_threads_per_threadgroup(256)]] kernel void prefill4k_steel_bf16_m##BM##_n##BN##_bk512_wm##WM##_wn##WN( \
    const device bfloat *input [[buffer(0)]], const device bfloat *weights [[buffer(1)]], \
    device bfloat *output [[buffer(2)]], device atomic_uint *diagnostics [[buffer(3)]], \
    constant FlashDenseCacheParams &p [[buffer(4)]], \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]], uint simdGroup [[simdgroup_index_in_threadgroup]]) { \
  steel_screen_tile<BM,BN,WM,WN>(input,weights,output,diagnostics,p,group,threads,tid,simdGroup); \
}
STEEL_SCREEN_ENTRY(128,64,4,2)
STEEL_SCREEN_ENTRY(64,128,2,4)
#undef STEEL_SCREEN_ENTRY
