#pragma once

#include "FlashMTPGPUChainPolicy.hpp"
#include <algorithm>
#include <array>
#include <cstdint>

namespace splash::flash::mtp_gpu_chain_four_policy {
namespace greedy = splash::flash::mtp_gpu_chain_policy;
inline constexpr uint32_t kMaximumDepth=4,kMaximumBodies=3,kMaximumDispatches=256;
inline constexpr uint32_t kErrorNonfinite=1,kErrorInvalidGreedy=2,kErrorDiagnostics=4;
inline constexpr uint32_t kErrorCapacity=8,kErrorMalformedParams=16,kNoToken=UINT32_MAX;
inline constexpr const char *kSemantics=
    "private-seeded-exact-greedy-four-proposal-ordered-r1-indirect-eos-quota-v1";
struct Params {
  uint32_t vocabulary=248320,requestedDepth=4,remaining=5,begin=0;
  uint32_t capacity=4096,dispatchCount=1,bodyIndex=0,reserved=0;
};
struct Control {
  uint32_t proposalCount=0,consumedPairs=0,bodyEnabled=0,finishedEOS=0;
  uint32_t errors=0,reserved[3]{};
  uint32_t proposals[4]{kNoToken,kNoToken,kNoToken,kNoToken};
};
static_assert(sizeof(Params)==32);
static_assert(sizeof(Control)==48);
using Groups=std::array<uint32_t,3>;
[[nodiscard]] constexpr uint32_t paramsError(const Params &p) noexcept {
  return !p.vocabulary||p.vocabulary>248320||p.requestedDepth>4||!p.remaining||
      !p.capacity||p.capacity>262144||p.begin>p.capacity||p.dispatchCount>256||
      p.bodyIndex>2||p.reserved?kErrorMalformedParams:0;
}
[[nodiscard]] inline uint32_t boundedDepth(uint32_t requested,uint32_t remaining) {
  if(requested>4||!remaining)
    throw std::invalid_argument("private four-proposal chain requires depth0..4 and a positive budget");
  return std::min(requested,remaining-1);
}
[[nodiscard]] constexpr uint32_t controlError(const Control &c) noexcept {
  return c.proposalCount>4||c.consumedPairs>3||c.bodyEnabled>1||c.finishedEOS>1||
      c.reserved[0]||c.reserved[1]||c.reserved[2]?kErrorMalformedParams:0;
}
// Params.begin is the initial folded length, never an incremented CPU length.
// Only body0 reads the external seed; later bodies reuse the validated prefix.
// The bool return identifies a seed read, useful for strict no-work tests.
[[nodiscard]] inline bool beginStep(Control &c,const Params &p,
    const FlashGreedyGPURowResult &seed,uint32_t priorDiagnostics=0) noexcept {
  if(!p.bodyIndex)c=Control{};
  else c.bodyEnabled=0;
  c.errors|=paramsError(p)|controlError(c);
  if(c.errors)return false;
  const uint32_t depth=std::min(p.requestedDepth,p.remaining-1);
  if(!depth)return false;
  bool readSeed=false;
  if(!p.bodyIndex) {
    if(priorDiagnostics){c.errors=kErrorDiagnostics;return false;}
    readSeed=true;
    c.errors=greedy::recordError(seed,p.vocabulary);
    if(c.errors)return readSeed;
    c.proposals[0]=seed.token;c.proposalCount=1;
    c.finishedEOS=greedy::stopToken(seed.token);
  }
  if(c.finishedEOS||depth<=p.bodyIndex+1)return readSeed;
  // A healthy active prefix has exactly one validated token beyond the pairs
  // already consumed. No skipped or failed predecessor can revive a suffix.
  if(c.consumedPairs!=p.bodyIndex||c.proposalCount!=p.bodyIndex+1) {
    c.errors=kErrorMalformedParams;return readSeed;
  }
  if(priorDiagnostics){c.errors=kErrorDiagnostics;return readSeed;}
  if(uint64_t{p.begin}+p.bodyIndex>=p.capacity) {
    c.errors=kErrorCapacity;return readSeed;
  }
  if(!p.dispatchCount){c.errors=kErrorMalformedParams;return readSeed;}
  c.bodyEnabled=1;
  return readSeed;
}
[[nodiscard]] constexpr Groups indirectGroups(const Control &c,Groups original) noexcept {
  return c.bodyEnabled?original:Groups{0,0,0};
}
inline void writeNextToken(const Control &c,const Params &p,int64_t &token) noexcept {
  if(c.bodyEnabled&&p.bodyIndex<3)token=int64_t(c.proposals[p.bodyIndex]);
}
// The bool return identifies a suffix-record read. A skipped body neither
// consumes a pair nor inspects its intentionally untouched record/diagnostics.
[[nodiscard]] inline bool finishStep(Control &c,const Params &p,
    const FlashGreedyGPURowResult &next,uint32_t diagnostics=0) noexcept {
  if(!c.bodyEnabled)return false;
  ++c.consumedPairs; // Actual execution, even when its result is fatal.
  if(diagnostics){c.errors|=kErrorDiagnostics;return false;}
  c.errors|=paramsError(p)|controlError(c);
  if(c.errors)return false;
  if(c.consumedPairs!=p.bodyIndex+1||c.proposalCount!=p.bodyIndex+1) {
    c.errors=kErrorMalformedParams;return false;
  }
  c.errors=greedy::recordError(next,p.vocabulary);
  if(c.errors)return true;
  c.proposals[p.bodyIndex+1]=next.token;++c.proposalCount;
  c.finishedEOS=greedy::stopToken(next.token);
  return true;
}
struct Publication {
  Control control;
  uint32_t logicalLength=0,rollbackLength=0;
  bool ready=false,success=false,poisonHead=false;
};
[[nodiscard]] inline Publication publish(const Control &c,uint32_t initialBegin,
    bool commandCompleted,bool commandSucceeded=true) noexcept {
  Publication out;out.control=c;
  out.logicalLength=out.rollbackLength=initialBegin;
  if(!commandCompleted)return out;
  out.ready=true;out.control.errors|=controlError(c);
  if(!commandSucceeded) {
    out.control.errors|=kErrorDiagnostics;
    out.poisonHead=bool(c.consumedPairs||c.bodyEnabled);
    return out;
  }
  if(out.control.errors) {
    // Capacity-only truncation followed exclusively healthy cache updates.
    // Their logical offset remains the original fold and may safely retry.
    out.poisonHead=c.consumedPairs&&out.control.errors!=kErrorCapacity;
    return out;
  }
  out.success=true;out.logicalLength=initialBegin+c.consumedPairs;
  return out;
}
[[nodiscard]] constexpr uint32_t poolBegin(uint32_t begin) noexcept{return begin/4;}
[[nodiscard]] constexpr uint32_t newlyCompletedPools(uint32_t begin,uint32_t consumed) noexcept {
  return consumed<=3&&uint64_t{begin}+consumed<=262144?
      (begin+consumed)/4-begin/4:0;
}
} // namespace splash::flash::mtp_gpu_chain_four_policy
