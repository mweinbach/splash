// Candidate-only dispatch traversal. All dots call the production helpers;
// changing group coordinates preserves every lane's arithmetic and reduction.
// There is no explicit die or memory affinity.
#include "metal/kernels/shared/flash_affine.metal"
#include "metal/kernels/shared/flash_expert_qmv.metal"

template <ushort Mode>
inline uint3 ultra_quantized_group(uint3 group, constant FlashAffineParams &p) {
  if constexpr (Mode == 1) return uint3(group.y, group.x, group.z); // row-fast
  if constexpr (Mode == 2) return uint3(group.y, group.z, group.x); // selection-fast
  if constexpr (Mode == 3) return uint3(group.y, group.x / p.selections,
                             group.x % p.selections); // route-fast
  if constexpr (Mode == 4 || Mode == 5 || Mode == 6) {
    constexpr uint width = 1u << (Mode - 3);
    return uint3(group.x / width, group.y * width + group.x % width, group.z);
  }
  if constexpr (Mode == 7 || Mode == 8) {
    constexpr uint width = 1u << (Mode - 6);
    return uint3(group.x / width, group.y, group.z * width + group.x % width);
  }
  return group;
}

#define ULTRA_QMV_ENTRY(NAME, MODE, ...) \
kernel void NAME( \
    const device bfloat *input [[buffer(0)]], \
    const device uchar *weights [[buffer(1)]], \
    const device uchar *scales [[buffer(2)]], \
    const device uchar *biases [[buffer(3)]], \
    const device long *ids [[buffer(4)]], \
    device bfloat *output [[buffer(5)]], \
    device atomic_uint *diagnostics [[buffer(6)]], \
    constant FlashAffineParams &p [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  (void)threads; \
  if (!p.selections) { \
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed); \
    return; \
  } \
  const uint3 mapped = ultra_quantized_group<MODE>(group, p); \
  __VA_ARGS__ \
}

#define ULTRA_AFFINE(MODE) ULTRA_QMV_ENTRY(ultra_affine_m##MODE, MODE, \
  flash_affine_specialized_project<4, 64, 1, true>(input, weights, scales, \
      biases, ids, output, diagnostics, p, mapped, simd, lane);)
#define ULTRA_CONTIG16(MODE) ULTRA_QMV_ENTRY(ultra_contig16_m##MODE, MODE, \
  expert_qmv_contig<16, 4, 2>(input, weights, scales, biases, ids, output, \
      diagnostics, p, mapped, threads, simd, lane);)
#define ULTRA_CONTIG8(MODE) ULTRA_QMV_ENTRY(ultra_contig8_m##MODE, MODE, \
  expert_qmv_contig<8, 2, 4>(input, weights, scales, biases, ids, output, \
      diagnostics, p, mapped, threads, simd, lane);)
#define ULTRA_MODE(MODE) ULTRA_AFFINE(MODE) ULTRA_CONTIG16(MODE) ULTRA_CONTIG8(MODE)
ULTRA_MODE(0)
ULTRA_MODE(1)
ULTRA_MODE(2)
ULTRA_MODE(3)
ULTRA_MODE(4)
ULTRA_MODE(5)
ULTRA_MODE(6)
ULTRA_MODE(7)
ULTRA_MODE(8)
#undef ULTRA_MODE
#undef ULTRA_CONTIG8
#undef ULTRA_CONTIG16
#undef ULTRA_AFFINE
#undef ULTRA_QMV_ENTRY

// Stable GPU route jobs: each route computes its rank by expert ID, using the
// canonical route as the tie-break. At the verifier's maximum160 routes this
// is a single small dispatch; its time is included for every candidate repeat.
kernel void ultra_quantized_route_jobs(
    const device long *ids [[buffer(0)]], device uint *jobs [[buffer(1)]],
    device atomic_uint *diagnostics [[buffer(2)]],
    constant FlashAffineParams &p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (!p.rows || p.rows > 16 || (p.selections != 1 && p.selections != 10)) {
    if (!tid) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint count = p.rows * p.selections;
  if (tid >= count) return;
  const long expert = ids[tid];
  uint rank = 0;
  for (uint other = 0; other < count; ++other) {
    const long candidate = ids[other];
    rank += candidate < expert || (candidate == expert && other < tid);
  }
  jobs[rank] = tid;
}

#define ULTRA_SORTED_ENTRY(NAME, ROUTE_FAST, ...) \
kernel void NAME( \
    const device bfloat *input [[buffer(0)]], \
    const device uchar *weights [[buffer(1)]], \
    const device uchar *scales [[buffer(2)]], \
    const device uchar *biases [[buffer(3)]], \
    const device long *ids [[buffer(4)]], \
    device bfloat *output [[buffer(5)]], \
    device atomic_uint *diagnostics [[buffer(6)]], \
    const device uint *jobs [[buffer(7)]], \
    constant FlashAffineParams &p [[buffer(8)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 threads [[threads_per_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  (void)threads; \
  if (!p.rows || p.rows > 16 || (p.selections != 1 && p.selections != 10)) { \
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed); \
    return; \
  } \
  const uint job = ROUTE_FAST ? group.x : group.y; \
  if (job >= p.rows * p.selections || group.z) return; \
  const uint route = jobs[job]; \
  if (route >= p.rows * p.selections) { \
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed); \
    return; \
  } \
  const uint3 mapped = uint3(ROUTE_FAST ? group.y : group.x, \
      route / p.selections, route % p.selections); \
  __VA_ARGS__ \
}
#define ULTRA_SORTED_AFFINE(FAST) ULTRA_SORTED_ENTRY(ultra_affine_sorted##FAST, FAST, \
  flash_affine_specialized_project<4, 64, 1, true>(input, weights, scales, \
      biases, ids, output, diagnostics, p, mapped, simd, lane);)
#define ULTRA_SORTED_CONTIG16(FAST) ULTRA_SORTED_ENTRY(ultra_contig16_sorted##FAST, FAST, \
  expert_qmv_contig<16, 4, 2>(input, weights, scales, biases, ids, output, \
      diagnostics, p, mapped, threads, simd, lane);)
#define ULTRA_SORTED_CONTIG8(FAST) ULTRA_SORTED_ENTRY(ultra_contig8_sorted##FAST, FAST, \
  expert_qmv_contig<8, 2, 4>(input, weights, scales, biases, ids, output, \
      diagnostics, p, mapped, threads, simd, lane);)
#define ULTRA_SORTED(FAST) ULTRA_SORTED_AFFINE(FAST) ULTRA_SORTED_CONTIG16(FAST) ULTRA_SORTED_CONTIG8(FAST)
ULTRA_SORTED(0)
ULTRA_SORTED(1)
#undef ULTRA_SORTED
#undef ULTRA_SORTED_CONTIG8
#undef ULTRA_SORTED_CONTIG16
#undef ULTRA_SORTED_AFFINE
#undef ULTRA_SORTED_ENTRY
