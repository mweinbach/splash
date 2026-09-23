#pragma once
#include "flash/FlashQSAMPP.hpp"

namespace splash::flash::prefill4k {
// Dedicated bounded workspace: callers must reserve plannedBytes before
// construction when integrating this experiment with an engine budget.
struct DenseCoalescedWorkspace final {
  uint32_t maximumRows = 0;
  metal::MetalBuffer queries;
  metal::MetalBuffer indexQueries;
  metal::MetalBuffer selectedBlocks;
};
[[nodiscard]] uint64_t denseCoalescedPlannedBytes(uint32_t maximumRows);
[[nodiscard]] DenseCoalescedWorkspace allocateDenseCoalescedWorkspace(
    metal::MetalBackend &backend, uint32_t maximumRows);
[[nodiscard]] bool denseCoalescedEligible(uint32_t begin, uint32_t rows,
                                         uint32_t capacity) noexcept;
// Bulk preparation/pooling/dense chronological selection only. Attention
// retains the existing per128-row route, partition policy and arithmetic.
void addDenseCoalescedQSA(metal::MetalBackend &backend,
    metal::CommandGraph &graph, const FlashQSAFastInputs &input,
    FlashQSAState &state, FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &fast, DenseCoalescedWorkspace &bulk,
    uint32_t begin, uint32_t rows);
// Actual current route, with ordinary128-row scratch reused after attention.
void addOrdinaryChunkedQSA(metal::MetalBackend &backend,
    metal::CommandGraph &graph, const FlashQSAFastInputs &input,
    FlashQSAState &state, FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &fast, uint32_t begin, uint32_t rows);
void addOrdinaryQSAChunk(metal::CommandGraph &graph,
    const FlashQSAFastInputs &input, FlashQSAState &state,
    FlashQSAWorkspace &ordinary, FlashQSAFastWorkspace &fast,
    uint32_t begin, uint32_t rows);
[[nodiscard]] FlashQSAFastInputs sliceCoalescedInputs(
    metal::MetalBackend &backend, const FlashQSAFastInputs &input,
    uint32_t offset, uint32_t rows);
} // namespace splash::flash::prefill4k
