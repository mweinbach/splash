#include "control_tap.metalh"

kernel void r5_raw_odd_rowpair_sep22_control_probe_q4_g64(
    const device bfloat *input [[buffer(0)]], const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]], const device uchar *biases [[buffer(3)]],
    const device long *expert_ids [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]], constant FlashAffineParams &p [[buffer(7)]],
    device float *raw_f32 [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  (void)expert_ids;
  if (any(threads != uint3(64, 1, 1)) || simd_width != 32 || simd_group >= 2 || lane >= 32) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  r5_raw_odd_control_tap::project<4, 64>(input, weights, scales, biases, expert_ids, output, diagnostics, raw_f32, p, group, simd_group, lane);
}

kernel void r5_raw_odd_rowpair_sep22_control_probe_q5_g64(
    const device bfloat *input [[buffer(0)]], const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]], const device uchar *biases [[buffer(3)]],
    const device long *expert_ids [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]], constant FlashAffineParams &p [[buffer(7)]],
    device float *raw_f32 [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  (void)expert_ids;
  if (any(threads != uint3(64, 1, 1)) || simd_width != 32 || simd_group >= 2 || lane >= 32) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  r5_raw_odd_control_tap::project<5, 64>(input, weights, scales, biases, expert_ids, output, diagnostics, raw_f32, p, group, simd_group, lane);
}

kernel void r5_raw_odd_rowpair_sep22_control_probe_q5_g128(
    const device bfloat *input [[buffer(0)]], const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]], const device uchar *biases [[buffer(3)]],
    const device long *expert_ids [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]], constant FlashAffineParams &p [[buffer(7)]],
    device float *raw_f32 [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  (void)expert_ids;
  if (any(threads != uint3(64, 1, 1)) || simd_width != 32 || simd_group >= 2 || lane >= 32) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  r5_raw_odd_control_tap::project<5, 128>(input, weights, scales, biases, expert_ids, output, diagnostics, raw_f32, p, group, simd_group, lane);
}

kernel void r5_raw_odd_rowpair_sep22_control_probe_q6_g64(
    const device bfloat *input [[buffer(0)]], const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]], const device uchar *biases [[buffer(3)]],
    const device long *expert_ids [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]], constant FlashAffineParams &p [[buffer(7)]],
    device float *raw_f32 [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  (void)expert_ids;
  if (any(threads != uint3(64, 1, 1)) || simd_width != 32 || simd_group >= 2 || lane >= 32) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  r5_raw_odd_control_tap::project<6, 64>(input, weights, scales, biases, expert_ids, output, diagnostics, raw_f32, p, group, simd_group, lane);
}
