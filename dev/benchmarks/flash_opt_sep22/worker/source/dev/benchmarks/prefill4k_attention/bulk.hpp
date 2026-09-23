#pragma once
#include "coalesced.hpp"
namespace splash::flash::prefill4k {
struct BulkExactWorkspace final {
  DenseCoalescedWorkspace prepared;
  FlashQSAFastWorkspace partials;
};
[[nodiscard]] uint64_t bulkExactPlannedBytes();
[[nodiscard]] BulkExactWorkspace allocateBulkExactWorkspace(metal::MetalBackend &backend);
void addBulkExactQSA(metal::MetalBackend &backend,metal::CommandGraph &graph,
    const FlashQSAFastInputs &input,FlashQSAState &state,FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &ordinaryFast,BulkExactWorkspace &bulk,uint32_t begin,uint32_t rows,bool sg8=false);
void addBulkDirectQSA(metal::MetalBackend &backend,metal::CommandGraph &graph,
    const FlashQSAFastInputs &input,FlashQSAState &state,FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &ordinaryFast,DenseCoalescedWorkspace &prepared,uint32_t begin,uint32_t rows,bool bf16PV=false);
} // namespace splash::flash::prefill4k
