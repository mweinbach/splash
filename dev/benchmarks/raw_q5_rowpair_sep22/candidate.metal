#include "pair.metalh"

kernel void raw_q5_rowpair_sep22_timed(
    const device bfloat *input [[buffer(0)]],
    const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]],
    const device uchar *biases [[buffer(3)]],
    const device long *expert_ids [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]],
    constant FlashAffineParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  (void)expert_ids; // Dense flags0 never reads this original binding.
  if (any(threads != uint3(64, 1, 1))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  raw_q5_rowpair_sep22::project_pair(input, weights, scales, biases, output,
      diagnostics, p, group, simd_group, lane);
}
