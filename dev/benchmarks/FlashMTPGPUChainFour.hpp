#pragma once
#include "FlashMTPGPUChainFourCandidate.hpp"
#include "FlashMTPGPUChainFourGuard.h"
#include <array>
namespace splash::flash::mtp_gpu_chain_four_candidate {
struct FourProposalResult {
  metal::CommandTiming timing;
  FlashMTPGPUChainFourControl control;
  std::array<metal::MetalBuffer,3> greedyRecords,hiddenSnapshots;
  uint64_t originalLength=0,logicalLength=0;
  std::array<uint32_t,3> indirectDispatches{};
};
class FourProposalWorkspace final {
public:
  explicit FourProposalWorkspace(FlashMTPForward &head);
  [[nodiscard]] FourProposalResult run(FlashMTPState &state,const FlashMTPResult &seed,
      uint32_t requestedDepth,uint32_t remaining,metal::MetalBuffer priorDiagnostics);
  [[nodiscard]] std::vector<metal::MetalBuffer> mutableBuffers() const;
private:
  FlashMTPForward &head_;
  metal::MetalBackend &backend_;
  std::array<metal::MetalBuffer,3> staticGroups_,indirectGroups_,tokens_,greedy_,hidden_;
  std::array<metal::IndirectDispatchSource,3> indirectSources_;
  metal::MetalBuffer control_,diagnostics_;
};
}
