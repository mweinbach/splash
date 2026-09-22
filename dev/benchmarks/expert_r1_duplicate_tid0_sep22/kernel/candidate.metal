// Original R1 math; only candidate duplicate loop is thread-zero-owned.
#define gathered_mpp_finite r1_duplicate_tid0_native_gathered_mpp_finite
#define gathered_mpp_error r1_duplicate_tid0_native_gathered_mpp_error
#define gathered_mpp_nan r1_duplicate_tid0_native_gathered_mpp_nan
#define gathered_mpp_sigmoid r1_duplicate_tid0_native_gathered_mpp_sigmoid
#define gathered_mpp_rank r1_duplicate_tid0_native_gathered_mpp_rank
#define gathered_mpp_scan r1_duplicate_tid0_native_gathered_mpp_scan
#define gathered_mpp_execute r1_duplicate_tid0_native_gathered_mpp_execute
#include "native_helper.metal"
#undef gathered_mpp_finite
#undef gathered_mpp_error
#undef gathered_mpp_nan
#undef gathered_mpp_sigmoid
#undef gathered_mpp_rank
#undef gathered_mpp_scan
#undef gathered_mpp_execute
#define gathered_mpp_finite r1_duplicate_tid0_candidate_gathered_mpp_finite
#define gathered_mpp_error r1_duplicate_tid0_candidate_gathered_mpp_error
#define gathered_mpp_nan r1_duplicate_tid0_candidate_gathered_mpp_nan
#define gathered_mpp_sigmoid r1_duplicate_tid0_candidate_gathered_mpp_sigmoid
#define gathered_mpp_rank r1_duplicate_tid0_candidate_gathered_mpp_rank
#define gathered_mpp_scan r1_duplicate_tid0_candidate_gathered_mpp_scan
#define gathered_mpp_execute r1_duplicate_tid0_candidate_gathered_mpp_execute
#include "candidate_helper.metal"
#undef gathered_mpp_finite
#undef gathered_mpp_error
#undef gathered_mpp_nan
#undef gathered_mpp_sigmoid
#undef gathered_mpp_rank
#undef gathered_mpp_scan
#undef gathered_mpp_execute
kernel void r1_duplicate_tid0_gu_native(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]],
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]],
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]],
    device uint *diag [[buffer(8)]], constant FlashGatheredMPPParams &p [[buffer(9)]],
    uint3 physical [[threadgroup_position_in_grid]],
    uint3 total [[threadgroups_per_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  // Reject physical extents before transposition, avoiding wrapped/aliased
  // logical groups. Every participating thread takes the same guard branch.
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      physical.x >= 10u || physical.y >= 1u || physical.z >= 10u ||
      total.x != 10u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) r1_duplicate_tid0_native_gathered_mpp_error(diag,2u); return;
  }
  const uint3 logical = physical;
  r1_duplicate_tid0_native_gathered_mpp_execute<true>(x,g,gs,u,us,ranks,ids,out,diag,p,logical,threads,tid,safe_a,&nonfinite);
}
kernel void r1_duplicate_tid0_gu_candidate(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]],
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]],
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]],
    device uint *diag [[buffer(8)]], constant FlashGatheredMPPParams &p [[buffer(9)]],
    uint3 physical [[threadgroup_position_in_grid]],
    uint3 total [[threadgroups_per_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  // Reject physical extents before transposition, avoiding wrapped/aliased
  // logical groups. Every participating thread takes the same guard branch.
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      physical.x >= 10u || physical.y >= 1u || physical.z >= 10u ||
      total.x != 10u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) r1_duplicate_tid0_candidate_gathered_mpp_error(diag,2u); return;
  }
  const uint3 logical = physical;
  r1_duplicate_tid0_candidate_gathered_mpp_execute<true>(x,g,gs,u,us,ranks,ids,out,diag,p,logical,threads,tid,safe_a,&nonfinite);
}
kernel void r1_duplicate_tid0_down_native(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]],
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const long *ids [[buffer(4)]], device bfloat *out [[buffer(5)]],
    device uint *diag [[buffer(6)]], constant FlashGatheredMPPParams &p [[buffer(7)]],
    uint3 physical [[threadgroup_position_in_grid]],
    uint3 total [[threadgroups_per_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  // Reject physical extents before transposition, avoiding wrapped/aliased
  // logical groups. Every participating thread takes the same guard branch.
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      physical.x >= 40u || physical.y >= 1u || physical.z >= 10u ||
      total.x != 40u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) r1_duplicate_tid0_native_gathered_mpp_error(diag,2u); return;
  }
  const uint3 logical = physical;
  r1_duplicate_tid0_native_gathered_mpp_execute<false>(x,w,s,w,s,ranks,ids,out,diag,p,logical,threads,tid,safe_a,&nonfinite);
}
kernel void r1_duplicate_tid0_down_candidate(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]],
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const long *ids [[buffer(4)]], device bfloat *out [[buffer(5)]],
    device uint *diag [[buffer(6)]], constant FlashGatheredMPPParams &p [[buffer(7)]],
    uint3 physical [[threadgroup_position_in_grid]],
    uint3 total [[threadgroups_per_grid]], uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  // Reject physical extents before transposition, avoiding wrapped/aliased
  // logical groups. Every participating thread takes the same guard branch.
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      physical.x >= 40u || physical.y >= 1u || physical.z >= 10u ||
      total.x != 40u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) r1_duplicate_tid0_candidate_gathered_mpp_error(diag,2u); return;
  }
  const uint3 logical = physical;
  r1_duplicate_tid0_candidate_gathered_mpp_execute<false>(x,w,s,w,s,ranks,ids,out,diag,p,logical,threads,tid,safe_a,&nonfinite);
}
