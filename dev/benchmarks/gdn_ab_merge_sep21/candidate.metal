// Private scheduling-only GDN A/B pair; preserve the active QMV_F32 profile.
#include "abi.hpp"
#define splash_mlx_qmv_f32xsum_v1 gdn_ab_original_qmv
#define flash_affine_mlx_qmv_f32xsum_v1_q4_g64 gdn_ab_original_q4_g64
#define flash_affine_mlx_qmv_f32xsum_v1_q4_g128 gdn_ab_original_q4_g128
#define flash_affine_mlx_qmv_f32xsum_v1_q5_g64 gdn_ab_original_q5_g64
#define flash_affine_mlx_qmv_f32xsum_v1_q5_g128 gdn_ab_original_q5_g128
#define flash_affine_mlx_qmv_f32xsum_v1_q6_g64 gdn_ab_original_q6_g64
#define flash_affine_mlx_qmv_f32xsum_v1_q6_g128 gdn_ab_original_q6_g128
#define flash_affine_mlx_qmv_f32xsum_v1_q8_g64 gdn_ab_original_q8_g64
#define flash_affine_mlx_qmv_f32xsum_v1_q8_g128 gdn_ab_original_q8_g128
#include "metal/kernels/shared/flash_affine_qmv_f32.metal"
#undef splash_mlx_qmv_f32xsum_v1
#undef flash_affine_mlx_qmv_f32xsum_v1_q4_g64
#undef flash_affine_mlx_qmv_f32xsum_v1_q4_g128
#undef flash_affine_mlx_qmv_f32xsum_v1_q5_g64
#undef flash_affine_mlx_qmv_f32xsum_v1_q5_g128
#undef flash_affine_mlx_qmv_f32xsum_v1_q6_g64
#undef flash_affine_mlx_qmv_f32xsum_v1_q6_g128
#undef flash_affine_mlx_qmv_f32xsum_v1_q8_g64
#undef flash_affine_mlx_qmv_f32xsum_v1_q8_g128

// Exact source extraction with one raw sum store plus driver plumbing.
#include "gdn_ab_qmv_probe_generated.metal"

inline bool gdn_ab_launch_valid(
    constant GDNABMergeParams &p, uint3 actual_threads,
    device atomic_uint *diagnostics, uint lane) {
  if (!gdn_ab_merge_sep21::validPair(p) ||
      any(actual_threads != uint3(64, 1, 1))) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return false;
  }
  return true;
}

// Flags0 makes the original expert_ids argument unread.
inline void gdn_ab_timed_plane(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    device bfloat *output, device atomic_uint *diagnostics,
    constant FlashAffineParams &p, uint3 group, uint simd, uint lane) {
  const device long *dummy_ids = reinterpret_cast<const device long *>(input);
  if (p.bits == 5)
    gdn_ab_original_qmv::project<5, 128>(
        input, weights, scales, biases, dummy_ids, output, diagnostics, p,
        uint3(group.x, group.y, 0), simd, lane);
  else
    gdn_ab_original_qmv::project<6, 64>(
        input, weights, scales, biases, dummy_ids, output, diagnostics, p,
        uint3(group.x, group.y, 0), simd, lane);
}

inline void gdn_ab_probe_plane(
    const device bfloat *input, const device uchar *weights,
    const device uchar *scales, const device uchar *biases,
    device bfloat *output, device atomic_uint *diagnostics,
    device float *raw_f32, constant FlashAffineParams &p,
    uint3 group, uint simd, uint lane) {
  const device long *dummy_ids = reinterpret_cast<const device long *>(input);
  if (p.bits == 5)
    gdn_ab_original_qmv::project_tap<5, 128>(
        input, weights, scales, biases, dummy_ids, output, diagnostics, raw_f32,
        p, uint3(group.x, group.y, 0), simd, lane);
  else
    gdn_ab_original_qmv::project_tap<6, 64>(
        input, weights, scales, biases, dummy_ids, output, diagnostics, raw_f32,
        p, uint3(group.x, group.y, 0), simd, lane);
}

kernel void flash_gdn_ab_merge_qmv_f32_v1(
    const device bfloat *input [[buffer(0)]],
    const device uchar *aw [[buffer(1)]], const device uchar *as [[buffer(2)]],
    const device uchar *ab [[buffer(3)]], const device uchar *bw [[buffer(4)]],
    const device uchar *bs [[buffer(5)]], const device uchar *bb [[buffer(6)]],
    device bfloat *a_output [[buffer(7)]], device bfloat *b_output [[buffer(8)]],
    device atomic_uint *diagnostics [[buffer(9)]],
    constant GDNABMergeParams &p [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 actual_threads [[threads_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  // Validate both planes before any input/weight/output access.
  if (!gdn_ab_launch_valid(p, actual_threads, diagnostics, lane)) return;
  if (group.x >= 6 || group.y >= p.a.rows || group.z >= 2) return;
  if (group.z == 0)
    gdn_ab_timed_plane(input, aw, as, ab, a_output, diagnostics, p.a,
                       group, simd, lane);
  else
    gdn_ab_timed_plane(input, bw, bs, bb, b_output, diagnostics, p.b,
                       group, simd, lane);
}

template <short Plane>
inline void gdn_ab_probe_entry(
    const device bfloat *input,
    const device uchar *aw, const device uchar *as, const device uchar *ab,
    const device uchar *bw, const device uchar *bs, const device uchar *bb,
    device bfloat *a_output, device bfloat *b_output,
    device atomic_uint *diagnostics, device float *a_raw, device float *b_raw,
    constant GDNABMergeParams &p, uint3 group, uint3 actual_threads,
    uint simd, uint lane) {
  if (!gdn_ab_launch_valid(p, actual_threads, diagnostics, lane)) return;
  if (group.x >= 6 || group.y >= p.a.rows ||
      group.z >= uint(Plane < 0 ? 2 : 1)) return;
  const uint selected = Plane < 0 ? group.z : uint(Plane);
  if (selected == 0)
    gdn_ab_probe_plane(input, aw, as, ab, a_output, diagnostics, a_raw, p.a,
                       group, simd, lane);
  else
    gdn_ab_probe_plane(input, bw, bs, bb, b_output, diagnostics, b_raw, p.b,
                       group, simd, lane);
}

#define GDN_AB_PROBE_ENTRY(NAME, PLANE) \
kernel void NAME( \
    const device bfloat *input [[buffer(0)]], \
    const device uchar *aw [[buffer(1)]], const device uchar *as [[buffer(2)]], \
    const device uchar *ab [[buffer(3)]], const device uchar *bw [[buffer(4)]], \
    const device uchar *bs [[buffer(5)]], const device uchar *bb [[buffer(6)]], \
    device bfloat *a_output [[buffer(7)]], device bfloat *b_output [[buffer(8)]], \
    device atomic_uint *diagnostics [[buffer(9)]], \
    device float *a_raw [[buffer(10)]], device float *b_raw [[buffer(11)]], \
    constant GDNABMergeParams &p [[buffer(12)]], \
    uint3 group [[threadgroup_position_in_grid]], \
    uint3 actual_threads [[threads_per_threadgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
  gdn_ab_probe_entry<PLANE>(input, aw, as, ab, bw, bs, bb, a_output, b_output, \
      diagnostics, a_raw, b_raw, p, group, actual_threads, simd, lane); \
}
GDN_AB_PROBE_ENTRY(flash_gdn_ab_merge_probe_qmv_f32_v1, -1)
GDN_AB_PROBE_ENTRY(flash_gdn_ab_single_probe_a_qmv_f32_v1, 0)
GDN_AB_PROBE_ENTRY(flash_gdn_ab_single_probe_b_qmv_f32_v1, 1)
#undef GDN_AB_PROBE_ENTRY
