#pragma once

#include "metal/MetalBackend.hpp"

#include <span>
#include <vector>

namespace splash::flash::candidate {

// Reuses the existing packed expert jobs and blocked-MoE ABI. The returned
// dispatch bytes borrow the source CommandGraph's storage; keep it alive.
// This isolated transform never selects a production runtime policy.
[[nodiscard]] std::vector<metal::ComputeDispatch> moEWideDispatches(
    std::span<const metal::ComputeDispatch> source, bool wideGateUp = true,
    bool wideDown = true);

} // namespace splash::flash::candidate
