#include <metal_stdlib>
#include "FlashMTPGPUChainFourGuard.h"
using namespace metal;
inline uint four_record_errors(FlashGreedyGPURowResult r,uint vocabulary) {
  if(r.errors&kFlashGreedyGPUErrorNonfinite) return kFlashMTPGPUChainNonfinite;
  return r.errors||r.reserved||r.token>=vocabulary||r.rank<0x80u||r.rank>0xff7fu||r.rank==0x7fffu
      ?kFlashMTPGPUChainInvalidGreedy:0u;
}
inline bool four_eos(uint token) {return token==248044u||token==248046u;}
kernel void flash_mtp_gpu_chain_four_begin(
    device const FlashGreedyGPURowResult *seed [[buffer(0)]],
    device const uint *prior_diagnostics [[buffer(1)]],
    device const uint *static_groups [[buffer(2)]],
    device uint *indirect_groups [[buffer(3)]],device long *next_token_i64 [[buffer(4)]],
    device FlashMTPGPUChainFourControl *control [[buffer(5)]],
    constant FlashMTPGPUChainFourGuardParams &p [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if(group.x||group.y||group.z||tid) return;
  auto result=p.body_index?control[0]:
      FlashMTPGPUChainFourControl{0,0,0,0,0,0,0,0,{UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX}};
  result.body_enabled=0;
  const bool valid=threads.x==1&&threads.y==1&&threads.z==1&&p.vocabulary&&p.vocabulary<=248320&&
      p.requested_depth<=4&&p.remaining&&p.capacity&&p.capacity<=262144&&p.begin<=p.capacity&&
      p.dispatch_count<=256&&p.body_index<=2&&!p.reserved;
  if(!valid) result.errors|=kFlashMTPGPUChainParameters;
  else {
    const uint depth=min(p.requested_depth,p.remaining-1);
    if(!p.body_index&&depth) {
      result.errors=prior_diagnostics[0]?kFlashMTPGPUChainDiagnostics:four_record_errors(seed[0],p.vocabulary);
      if(!result.errors) {
        result.proposals[0]=seed[0].token;result.proposal_count=1;
        result.finished_eos=uint(four_eos(seed[0].token));
      }
    }
    if(!result.errors&&!result.finished_eos&&depth>p.body_index+1) {
      if(result.proposal_count!=p.body_index+1||result.consumed_pairs!=p.body_index)
        result.errors=kFlashMTPGPUChainParameters;
      else if(p.body_index>=p.capacity-p.begin) result.errors=kFlashMTPGPUChainCapacity;
      else if(!p.dispatch_count) result.errors=kFlashMTPGPUChainParameters;
      else {
        result.body_enabled=1;
        next_token_i64[0]=long(result.proposals[p.body_index]);
      }
    }
  }
  control[0]=result;
  for(uint i=0;i<min(p.dispatch_count,256u)*3;++i)
    indirect_groups[i]=result.body_enabled?static_groups[i]:0u;
}
kernel void flash_mtp_gpu_chain_four_finish(
    device const FlashGreedyGPURowResult *next [[buffer(0)]],
    device const uint *body_diagnostics [[buffer(1)]],
    device FlashMTPGPUChainFourControl *control [[buffer(2)]],
    constant FlashMTPGPUChainFourGuardParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if(group.x||group.y||group.z||tid) return;
  auto result=control[0];
  if(!result.body_enabled) return;
  result.consumed_pairs=p.body_index+1;
  if(threads.x!=1||threads.y!=1||threads.z!=1||p.body_index>2)
    result.errors|=kFlashMTPGPUChainParameters;
  else {
    result.errors|=body_diagnostics[0]?kFlashMTPGPUChainDiagnostics:four_record_errors(next[0],p.vocabulary);
    if(!result.errors) {
      result.proposals[p.body_index+1]=next[0].token;result.proposal_count=p.body_index+2;
      result.finished_eos=uint(four_eos(next[0].token));
    }
  }
  control[0]=result;
}
