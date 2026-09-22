#pragma once
#include "coalesced.hpp"
namespace splash::flash::prefill4k {
struct TwoPassWorkspace final {
  DenseCoalescedWorkspace prepared;
  metal::MetalBuffer packedQueries,scoresAndProbabilities,rawAttention;
};
constexpr uint64_t twoPassExtraBytes() noexcept {return 25165824ULL+402653184ULL+50331648ULL;}
constexpr uint64_t twoPassPlannedBytes() noexcept {return 31457280ULL+twoPassExtraBytes();}
bool twoPassGeometry(uint32_t begin,uint32_t rows,uint32_t capacity,bool verification) noexcept;
TwoPassWorkspace allocateTwoPassWorkspace(metal::MetalBackend &backend,uint64_t reservedBytes);
void addTwoPassQSA(metal::MetalBackend &backend,metal::CommandGraph &graph,
    const FlashQSAFastInputs &input,FlashQSAState &state,FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &fast,TwoPassWorkspace &workspace,uint32_t begin,uint32_t rows,
    bool verification=false,bool packedV=false);
}
