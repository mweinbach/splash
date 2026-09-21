#pragma once
#include "metal/MetalBackend.hpp"

namespace splash::flash::candidate {
// Rewrite only qualified v5 INT8 hit/Q4 direct-miss producer names and grid Y.
// Original graph owns all borrowed parameter bytes and must remain alive.
[[nodiscard]] std::vector<metal::ComputeDispatch> persistentInt8JobDispatches(
    std::span<const metal::ComputeDispatch> source, uint32_t gateGridY = 64,
    uint32_t downGridY = 16, bool gate = true, bool down = true);
} // namespace splash::flash::candidate
