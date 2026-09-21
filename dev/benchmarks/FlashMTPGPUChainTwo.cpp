#include "FlashMTPGPUChainTwo.hpp"
#include "metal/abi/FlashForward.h"
#include <cstring>
#include <limits>
#include <stdexcept>

namespace splash::flash::mtp_gpu_chain_candidate {
namespace {
constexpr uint32_t kDispatches=256,kHyper=10240;
void requireBuffer(const metal::MetalBuffer &buffer,uint64_t bytes,uint64_t alignment) {
  if(!buffer||buffer.storage()!=metal::BufferStorage::Shared||!buffer.contents()||
      buffer.sizeBytes()<bytes||reinterpret_cast<uintptr_t>(buffer.contents())%alignment)
    throw std::invalid_argument("private chain buffer visibility, extent or alignment invalid");
}
bool overlap(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  const auto aa=reinterpret_cast<uintptr_t>(a.contents()),bb=reinterpret_cast<uintptr_t>(b.contents());
  if(!aa||!bb||aa>UINTPTR_MAX-a.sizeBytes()||bb>UINTPTR_MAX-b.sizeBytes())
    throw std::invalid_argument("private chain address extent invalid");
  return aa<bb+b.sizeBytes()&&bb<aa+a.sizeBytes();
}
}
TwoProposalWorkspace::TwoProposalWorkspace(FlashMTPForward &head)
    :head_(head),backend_(head.executionBackend()) {
  const auto allocate=[&](uint64_t bytes,const char *name) {
    return backend_.allocateBuffer((bytes+16383)&~uint64_t{16383},metal::BufferStorage::Shared,name);
  };
  staticGroups_=allocate(uint64_t{kDispatches}*12,"private-chain-static-groups");
  indirectGroups_=allocate(uint64_t{kDispatches}*12,"private-chain-gpu-groups");
  token_=allocate(8,"private-chain-next-token-i64");
  secondGreedy_=allocate(16,"private-chain-second-greedy-record");
  control_=allocate(sizeof(FlashMTPGPUChainControl),"private-chain-completion-control");
  diagnostics_=allocate(4,"private-chain-sticky-body-diagnostics");
  hiddenSnapshot_=allocate(uint64_t{kHyper}*2,"private-chain-owned-premixer-snapshot");
  indirectSource_=backend_.registerIndirectDispatchSource(indirectGroups_,head_.immutableOperands());
}
std::vector<metal::MetalBuffer> TwoProposalWorkspace::mutableBuffers() const {
  return {staticGroups_,indirectGroups_,token_,secondGreedy_,control_,diagnostics_,hiddenSnapshot_};
}
TwoProposalResult TwoProposalWorkspace::run(FlashMTPState &state,const FlashMTPResult &seed,
    uint32_t requestedDepth,uint32_t remaining,metal::MetalBuffer priorDiagnostics) {
  std::lock_guard lock(head_.executionMutex());
  if(state.poisoned()||requestedDepth>2||!remaining||seed.logicalLength!=state.logicalLength()||
      !seed.hiddenRows||seed.greedyRows!=1||seed.logitRows!=1)
    throw std::invalid_argument("private chain requires latest healthy scalar seed and depth0..2");
  requireBuffer(seed.hiddenBF16,uint64_t{seed.hiddenRows}*kHyper*2,2);
  requireBuffer(seed.greedyResultsU32,sizeof(FlashGreedyGPURowResult),4);
  requireBuffer(priorDiagnostics,4,4);
  const auto immutable=head_.immutableOperands();
  auto writable=mutableBuffers();
  for(const auto &a:writable) {
    for(const auto &b:immutable) if(overlap(a,b))
      throw std::invalid_argument("private chain workspace aliases model operands");
    for(const auto &b:{seed.hiddenBF16,seed.greedyResultsU32,priorDiagnostics})
      if(overlap(a,b)) throw std::invalid_argument("private chain workspace aliases seed");
  }
  for(size_t i=0;i<writable.size();++i) for(size_t j=i+1;j<writable.size();++j)
    if(overlap(writable[i],writable[j])) throw std::invalid_argument("private chain workspace aliases itself");
  // Sentinels permit proof that every skipped token/argmax/feature suffix is
  // untouched. Only the tiny guard's control and dimensions may change.
  for(const auto &buffer:writable) std::memset(buffer.contents(),0xa7,buffer.sizeBytes());
  std::memset(diagnostics_.contents(),0,diagnostics_.sizeBytes());
  const uint64_t originalLength=state.logicalLength();
  if(originalLength>UINT32_MAX) throw std::invalid_argument("private chain position is too large");
  metal::CommandGraph body;
  if(originalLength<state.capacity()) {
    const auto previousHidden=backend_.view(seed.hiddenBF16,uint64_t{seed.hiddenRows-1}*kHyper*2,uint64_t{kHyper}*2);
    const auto appended=head_.appendPairBody(state,body,uint32_t(originalLength),token_,previousHidden,
        diagnostics_,secondGreedy_);
    const FlashForwardCopyParams copy{uint64_t{kHyper}*2/4};
    body.add("flash_forward_copy_words",{appended.hiddenBF16,hiddenSnapshot_},copy,
        {(copy.words+255)/256,1,1},{256,1,1});
  }
  if(body.dispatches().size()>kDispatches) throw std::invalid_argument("private chain dispatch list exceeds guard capacity");
  auto *dimensions=static_cast<uint32_t *>(staticGroups_.contents());
  for(size_t i=0;i<body.dispatches().size();++i) {
    const auto &groups=body.dispatches()[i].threadgroups;
    for(uint64_t value:{groups.x,groups.y,groups.z})
      if(value>UINT32_MAX) throw std::invalid_argument("private chain indirect triplet exceedsuint32");
    dimensions[i*3]=uint32_t(groups.x);dimensions[i*3+1]=uint32_t(groups.y);dimensions[i*3+2]=uint32_t(groups.z);
  }
  const FlashMTPGPUChainGuardParams parameters{248320,requestedDepth,remaining,
      uint32_t(originalLength),state.capacity(),uint32_t(body.dispatches().size()),0,0};
  metal::CommandGraph begin,finish;
  begin.add("flash_mtp_gpu_chain_guard_begin",{seed.greedyResultsU32,priorDiagnostics,staticGroups_,
      indirectGroups_,token_,control_},parameters,{1,1,1},{1,1,1});
  finish.add("flash_mtp_gpu_chain_guard_finish",{secondGreedy_,diagnostics_,control_},
      parameters,{1,1,1},{1,1,1});
  std::vector<metal::ComputeDispatch> command;
  command.insert(command.end(),begin.dispatches().begin(),begin.dispatches().end());
  for(size_t i=0;i<body.dispatches().size();++i) {
    auto dispatch=body.dispatches()[i];
    dispatch.indirectGroups=metal::ComputeDispatch::IndirectGroups{indirectSource_,uint64_t{i}*12};
    command.push_back(std::move(dispatch));
  }
  command.insert(command.end(),finish.dispatches().begin(),finish.dispatches().end());
  metal::CommandTiming timing;
  FlashMTPGPUChainControl control{};
  try {
    timing=backend_.submitCommand(command);
    std::memcpy(&control,control_.contents(),sizeof(control));
    if(control.proposal_count>2||control.consumed_pairs>1||control.reserved0||control.reserved1||control.reserved2)
      throw std::runtime_error("private chain returned malformed completion control");
    head_.completePairTransaction(state,originalLength,control.consumed_pairs,
        control.body_enabled&&control.errors);
  } catch(...) {
    if(!state.poisoned()) head_.completePairTransaction(state,originalLength,0,true);
    throw;
  }
  return {timing,control,secondGreedy_,hiddenSnapshot_,originalLength,state.logicalLength(),
      uint32_t(body.dispatches().size())};
}
}
