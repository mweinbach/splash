#include "FlashMTPGPUChainFour.hpp"
#include "metal/abi/FlashForward.h"
#include <cstring>
#include <stdexcept>
namespace splash::flash::mtp_gpu_chain_four_candidate {
namespace {
constexpr uint32_t kDispatches=256,kHyper=10240;
void requireBuffer(const metal::MetalBuffer &b,uint64_t bytes,uint64_t alignment) {
  if(!b||b.storage()!=metal::BufferStorage::Shared||!b.contents()||b.sizeBytes()<bytes||
      reinterpret_cast<uintptr_t>(b.contents())%alignment)
    throw std::invalid_argument("private four-chain buffer visibility, extent or alignment invalid");
}
bool overlap(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  const auto aa=reinterpret_cast<uintptr_t>(a.contents()),bb=reinterpret_cast<uintptr_t>(b.contents());
  if(!aa||!bb||aa>UINTPTR_MAX-a.sizeBytes()||bb>UINTPTR_MAX-b.sizeBytes())
    throw std::invalid_argument("private four-chain address extent invalid");
  return aa<bb+b.sizeBytes()&&bb<aa+a.sizeBytes();
}
}
FourProposalWorkspace::FourProposalWorkspace(FlashMTPForward &head)
    :head_(head),backend_(head.executionBackend()) {
  const auto allocate=[&](uint64_t bytes,const char *name) {
    return backend_.allocateBuffer((bytes+16383)&~uint64_t{16383},metal::BufferStorage::Shared,name);
  };
  const auto immutable=head_.immutableOperands();
  for(uint32_t i=0;i<3;++i) {
    staticGroups_[i]=allocate(uint64_t{kDispatches}*12,"private-four-chain-static-groups");
    indirectGroups_[i]=allocate(uint64_t{kDispatches}*12,"private-four-chain-gpu-groups");
    tokens_[i]=allocate(8,"private-four-chain-token-i64");
    greedy_[i]=allocate(16,"private-four-chain-greedy-record");
    hidden_[i]=allocate(uint64_t{kHyper}*2,"private-four-chain-premixer-snapshot");
    indirectSources_[i]=backend_.registerIndirectDispatchSource(indirectGroups_[i],immutable);
  }
  control_=allocate(sizeof(FlashMTPGPUChainFourControl),"private-four-chain-control");
  diagnostics_=allocate(4,"private-four-chain-sticky-diagnostics");
}
std::vector<metal::MetalBuffer> FourProposalWorkspace::mutableBuffers() const {
  std::vector<metal::MetalBuffer> out{control_,diagnostics_};
  for(uint32_t i=0;i<3;++i) for(const auto &b:{staticGroups_[i],indirectGroups_[i],tokens_[i],greedy_[i],hidden_[i]}) out.push_back(b);
  return out;
}
FourProposalResult FourProposalWorkspace::run(FlashMTPState &state,const FlashMTPResult &seed,
    uint32_t depth,uint32_t remaining,metal::MetalBuffer priorDiagnostics) {
  std::lock_guard lock(head_.executionMutex());
  if(state.poisoned()||depth>4||!remaining||seed.logicalLength!=state.logicalLength()||
      !seed.hiddenRows||seed.greedyRows!=1||seed.logitRows!=1)
    throw std::invalid_argument("private four-chain requires healthy scalar seed and depth0..4");
  requireBuffer(seed.hiddenBF16,uint64_t{seed.hiddenRows}*kHyper*2,2);
  requireBuffer(seed.greedyResultsU32,16,4);requireBuffer(priorDiagnostics,4,4);
  const auto immutable=head_.immutableOperands();
  const auto mutableViews=mutableBuffers();
  for(const auto &a:mutableViews) {
    for(const auto &b:immutable) if(overlap(a,b)) throw std::invalid_argument("four-chain workspace aliases model");
    for(const auto &b:{seed.hiddenBF16,seed.greedyResultsU32,priorDiagnostics})
      if(overlap(a,b)) throw std::invalid_argument("four-chain workspace aliases seed");
  }
  for(size_t i=0;i<mutableViews.size();++i) for(size_t j=i+1;j<mutableViews.size();++j)
    if(overlap(mutableViews[i],mutableViews[j])) throw std::invalid_argument("four-chain workspace self-aliases");
  for(const auto &b:mutableViews) std::memset(b.contents(),0xa7,b.sizeBytes());
  std::memset(diagnostics_.contents(),0,diagnostics_.sizeBytes());
  const uint64_t original=state.logicalLength();
  if(original>UINT32_MAX) throw std::invalid_argument("four-chain position exceedsuint32");
  std::array<metal::CommandGraph,3> bodies,begins,finishes;
  std::array<uint32_t,3> counts{};
  auto previous=backend_.view(seed.hiddenBF16,uint64_t{seed.hiddenRows-1}*kHyper*2,uint64_t{kHyper}*2);
  for(uint32_t i=0;i<3;++i) {
    if(original+i<state.capacity()) {
      const auto body=head_.appendPairBody(state,bodies[i],uint32_t(original+i),tokens_[i],
          previous,diagnostics_,greedy_[i]);
      const FlashForwardCopyParams copy{uint64_t{kHyper}*2/4};
      bodies[i].add("flash_forward_copy_words",{body.hiddenBF16,hidden_[i]},copy,
          {(copy.words+255)/256,1,1},{256,1,1});
      previous=hidden_[i];
    }
    if(bodies[i].dispatches().size()>kDispatches) throw std::invalid_argument("four-chain body dispatch capacity exceeded");
    counts[i]=uint32_t(bodies[i].dispatches().size());
    auto *dimensions=static_cast<uint32_t *>(staticGroups_[i].contents());
    for(uint32_t j=0;j<counts[i];++j) {
      const auto &groups=bodies[i].dispatches()[j].threadgroups;
      for(auto value:{groups.x,groups.y,groups.z}) if(value>UINT32_MAX)
        throw std::invalid_argument("four-chain indirect dimensions exceeduint32");
      dimensions[j*3]=uint32_t(groups.x);dimensions[j*3+1]=uint32_t(groups.y);dimensions[j*3+2]=uint32_t(groups.z);
    }
    const FlashMTPGPUChainFourGuardParams p{248320,depth,remaining,uint32_t(original),state.capacity(),counts[i],i,0};
    begins[i].add("flash_mtp_gpu_chain_four_begin",{seed.greedyResultsU32,priorDiagnostics,staticGroups_[i],
        indirectGroups_[i],tokens_[i],control_},p,{1,1,1},{1,1,1});
    finishes[i].add("flash_mtp_gpu_chain_four_finish",{greedy_[i],diagnostics_,control_},p,{1,1,1},{1,1,1});
  }
  std::vector<metal::ComputeDispatch> command;
  for(uint32_t i=0;i<3;++i) {
    command.insert(command.end(),begins[i].dispatches().begin(),begins[i].dispatches().end());
    for(uint32_t j=0;j<counts[i];++j) {
      auto d=bodies[i].dispatches()[j];
      d.indirectGroups=metal::ComputeDispatch::IndirectGroups{indirectSources_[i],uint64_t{j}*12};
      command.push_back(std::move(d));
    }
    command.insert(command.end(),finishes[i].dispatches().begin(),finishes[i].dispatches().end());
  }
  metal::CommandTiming timing;FlashMTPGPUChainFourControl control{};
  try {
    timing=backend_.submitCommand(command);std::memcpy(&control,control_.contents(),sizeof(control));
    if(control.proposal_count>4||control.consumed_pairs>3||control.reserved0||control.reserved1||control.reserved2)
      throw std::runtime_error("four-chain malformed completion control");
    const bool failedBody=control.consumed_pairs&&control.errors&&control.errors!=kFlashMTPGPUChainCapacity;
    head_.completePairTransaction(state,original,control.errors?0:control.consumed_pairs,failedBody);
  } catch(...) {
    if(!state.poisoned()) head_.completePairTransaction(state,original,0,true);
    throw;
  }
  return {timing,control,greedy_,hidden_,original,state.logicalLength(),counts};
}
}
