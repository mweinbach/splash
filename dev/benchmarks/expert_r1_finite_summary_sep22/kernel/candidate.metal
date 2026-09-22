// Flag-only R1 finite summary: original deviceA and original bad-input scan.
#define gathered_mpp_finite r1_finite_summary_private_gathered_mpp_finite
#define gathered_mpp_error r1_finite_summary_private_gathered_mpp_error
#define gathered_mpp_nan r1_finite_summary_private_gathered_mpp_nan
#define gathered_mpp_sigmoid r1_finite_summary_private_gathered_mpp_sigmoid
#define gathered_mpp_rank r1_finite_summary_private_gathered_mpp_rank
#define gathered_mpp_scan r1_finite_summary_private_gathered_mpp_scan
#define gathered_mpp_execute r1_finite_summary_private_gathered_mpp_execute
#include "native_helper.metal"
#include "summary.metalh"
#include "consumer_helper.metalh"
kernel void r1_finite_summary_gu_native(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]],
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]],
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]],
    device uint *diag [[buffer(8)]], constant FlashGatheredMPPParams &p [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 total [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= 10u || group.y || group.z >= 10u ||
      total.x != 10u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) gathered_mpp_error(diag,2u); return;
  }
  gathered_mpp_execute<true>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite);
}
kernel void r1_finite_summary_gu_consumer(
    device const bfloat *x [[buffer(0)]], device int8_t *g [[buffer(1)]],
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]],
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]],
    device const long *ids [[buffer(6)]], device bfloat *out [[buffer(7)]],
    device uint *diag [[buffer(8)]], constant FlashGatheredMPPParams &p [[buffer(9)]],
    device const uint *packet [[buffer(10)]],
    constant R1FiniteSummaryInvocation &v [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 total [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[2560]; threadgroup atomic_uint nonfinite;
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= 10u || group.y || group.z >= 10u ||
      total.x != 10u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) gathered_mpp_error(diag,2u); return;
  }
  const bool needsOriginalScan = r1_finite_summary_needs_scan(packet,v,1u,2560u,0);
  r1_finite_summary_private_gathered_mpp_execute_consumer<true>(x,g,gs,u,us,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,needsOriginalScan);
}
kernel void r1_finite_summary_down_native(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]],
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const long *ids [[buffer(4)]], device bfloat *out [[buffer(5)]],
    device uint *diag [[buffer(6)]], constant FlashGatheredMPPParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 total [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= 40u || group.y || group.z >= 10u ||
      total.x != 40u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) gathered_mpp_error(diag,2u); return;
  }
  gathered_mpp_execute<false>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite);
}
kernel void r1_finite_summary_down_consumer(
    device const bfloat *x [[buffer(0)]], device int8_t *w [[buffer(1)]],
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const long *ids [[buffer(4)]], device bfloat *out [[buffer(5)]],
    device uint *diag [[buffer(6)]], constant FlashGatheredMPPParams &p [[buffer(7)]],
    device const uint *packet [[buffer(8)]],
    constant R1FiniteSummaryInvocation &v [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]], uint3 total [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup bfloat safe_a[640]; threadgroup atomic_uint nonfinite;
  if (p.rows != 1 || p.selections != 10 || p.experts != 512 || p.reserved ||
      group.x >= 40u || group.y || group.z >= 10u ||
      total.x != 40u || total.y != 1u || total.z != 10u ||
      threads.x != 128u || threads.y != 1u || threads.z != 1u) {
    if (!tid) gathered_mpp_error(diag,2u); return;
  }
  const bool needsOriginalScan = r1_finite_summary_needs_scan(packet,v,2u,640u,group.z);
  r1_finite_summary_private_gathered_mpp_execute_consumer<false>(x,w,s,w,s,ranks,ids,out,diag,p,group,threads,tid,safe_a,&nonfinite,needsOriginalScan);
}
