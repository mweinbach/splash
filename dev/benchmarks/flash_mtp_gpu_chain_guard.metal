#include <metal_stdlib>
#include "FlashMTPGPUChainGuard.h"
using namespace metal;

inline uint mtp_chain_record_errors(FlashGreedyGPURowResult record,uint vocabulary) {
  if(record.errors & kFlashGreedyGPUErrorNonfinite) return kFlashMTPGPUChainNonfinite;
  if(record.errors||record.reserved||record.token>=vocabulary||
      record.rank<0x80u||record.rank>0xff7fu||record.rank==0x7fffu)
    return kFlashMTPGPUChainInvalidGreedy;
  return 0;
}
inline bool mtp_chain_eos(uint token) {return token==248044u||token==248046u;}

// One ordered handoff dispatch writes every body group triplet, including
// positive zeros for every skipped dispatch. No body token/suffix record is
// touched after EOS, exhausted budget or an invalid exact greedy record.
kernel void flash_mtp_gpu_chain_guard_begin(
    device const FlashGreedyGPURowResult *seed [[buffer(0)]],
    device const uint *prior_diagnostics [[buffer(1)]],
    device const uint *static_groups [[buffer(2)]],
    device uint *indirect_groups [[buffer(3)]],
    device long *next_token_i64 [[buffer(4)]],
    device FlashMTPGPUChainControl *control [[buffer(5)]],
    constant FlashMTPGPUChainGuardParams &p [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
  if(group.x||group.y||group.z||tid) return;
  FlashMTPGPUChainControl result{0,0,0,0,0,0,0,0,{UINT_MAX,UINT_MAX}};
  if(threads.x!=1||threads.y!=1||threads.z!=1||!p.vocabulary||p.vocabulary>248320||
      p.requested_depth>2||!p.remaining||!p.capacity||p.capacity>262144||
      p.begin>p.capacity||p.dispatch_count>256||p.reserved0||p.reserved1) {
    result.errors=kFlashMTPGPUChainParameters;
  } else {
    const uint depth=min(p.requested_depth,p.remaining-1);
    if(depth) {
      result.errors=prior_diagnostics[0]?kFlashMTPGPUChainDiagnostics:
          mtp_chain_record_errors(seed[0],p.vocabulary);
      if(!result.errors) {
        result.proposals[0]=seed[0].token;
        result.proposal_count=1;
        result.finished_eos=uint(mtp_chain_eos(seed[0].token));
        if(depth>1&&!result.finished_eos) {
          if(p.begin==p.capacity) result.errors=kFlashMTPGPUChainCapacity;
          else if(!p.dispatch_count) result.errors=kFlashMTPGPUChainParameters;
          else {
            result.body_enabled=1;
            next_token_i64[0]=long(seed[0].token);
          }
        }
      }
    }
  }
  control[0]=result;
  // A malformed oversized count cannot name valid allocations. Host rejects
  // it; the shader touches at most the supported count on direct guard tests.
  for(uint i=0;i<min(p.dispatch_count,256u)*3;++i)
    indirect_groups[i]=result.body_enabled?static_groups[i]:0u;
}

// Body and its exact greedy reductions are indirect dispatches. This direct
// finish never reads their unwritten suffix after a skipped body.
kernel void flash_mtp_gpu_chain_guard_finish(
    device const FlashGreedyGPURowResult *second [[buffer(0)]],
    device const uint *body_diagnostics [[buffer(1)]],
    device FlashMTPGPUChainControl *control [[buffer(2)]],
    constant FlashMTPGPUChainGuardParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],uint tid [[thread_index_in_threadgroup]]) {
  if(group.x||group.y||group.z||tid) return;
  auto result=control[0];
  if(!result.body_enabled) return;
  result.consumed_pairs=1;
  if(threads.x!=1||threads.y!=1||threads.z!=1) {
    result.errors|=kFlashMTPGPUChainParameters;
  } else {
    result.errors|=body_diagnostics[0]?kFlashMTPGPUChainDiagnostics:
        mtp_chain_record_errors(second[0],p.vocabulary);
    if(!result.errors) {
      result.proposals[1]=second[0].token;
      result.proposal_count=2;
      result.consumed_pairs=1;
      result.finished_eos=uint(mtp_chain_eos(second[0].token));
    }
  }
  control[0]=result;
}
