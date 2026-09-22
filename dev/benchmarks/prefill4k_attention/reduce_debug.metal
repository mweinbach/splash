#include <metal_stdlib>
#include "metal/abi/FlashQSAFast.h"
using namespace metal;
#pragma METAL fp math_mode(safe)
kernel void prefill4k_qsa_attention_debug(
    device const float *statistics [[buffer(0)]],device const float *numerators [[buffer(1)]],
    device float *attention [[buffer(2)]],device bfloat *rounded [[buffer(3)]],
    device atomic_uint *diagnostics [[buffer(4)]],constant FlashQSAFastParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
  const uint row=group.x,head=group.y;
  if (row>=params.common.rows || head>=24 || tid>=256) return;
  const ulong base=(ulong(row)*24+head)*params.maximum_partitions;
  float maximum=-INFINITY;
  for (uint p=0;p<params.partitions;++p) maximum=max(maximum,statistics[(base+p)*2]);
  float sum=0.0f,value=0.0f;
  for (uint p=0;p<params.partitions;++p) {
    const float localSum=statistics[(base+p)*2+1];
    if (localSum==0.0f) continue;
    const float factor=exp(statistics[(base+p)*2]-maximum);
    sum+=localSum*factor;value+=numerators[(base+p)*256+tid]*factor;
  }
  const float result=value/sum;
  const ulong out=(ulong(row)*24+head)*256+tid;
  attention[out]=result;rounded[out]=bfloat(result);
  if (!(sum>0.0f) || !isfinite(result))
    atomic_fetch_or_explicit(diagnostics,1u<<8,memory_order_relaxed);
}
