#pragma once
#include "FlashMTPGPUChainCandidate.hpp"
#include "FlashMTPGPUChainGuard.h"

namespace splash::flash::mtp_gpu_chain_candidate {

struct TwoProposalResult {
  metal::CommandTiming timing;
  FlashMTPGPUChainControl control;
  metal::MetalBuffer secondGreedy, hiddenSnapshot;
  uint64_t originalLength=0, logicalLength=0;
  uint32_t indirectDispatches=0;
};
class TwoProposalWorkspace final {
public:
  explicit TwoProposalWorkspace(FlashMTPForward &head);
  // Seed was produced by this head's latest completed committed fold. Caller
  // keeps state/seed allocations alive. This function holds the head lock
  // through encoding, one submission, completion and length publication.
  [[nodiscard]] TwoProposalResult run(FlashMTPState &state,
      const FlashMTPResult &seed,uint32_t requestedDepth,uint32_t remaining,
      metal::MetalBuffer priorDiagnostics);
  [[nodiscard]] std::vector<metal::MetalBuffer> mutableBuffers() const;
private:
  FlashMTPForward &head_;
  metal::MetalBackend &backend_;
  metal::MetalBuffer staticGroups_,indirectGroups_,token_,secondGreedy_,
      control_,diagnostics_,hiddenSnapshot_;
  metal::IndirectDispatchSource indirectSource_;
};
}
