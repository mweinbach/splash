#pragma once

#include "metal/CommandGraph.hpp"

namespace splash::flash::candidate {

// Private graph rewrite; source owns original borrowed parameter payloads.
// packedInputs must expose routeCapacity+63 rows, and preparedDown must expose
// the same number of 640-channel rows. No production policy is changed.
class MoEDirectACommands final {
public:
  MoEDirectACommands(std::span<const metal::ComputeDispatch> source,
                    metal::MetalBuffer preparedDown,
                    bool directGateUp = true, bool directDown = true,
                    bool inplaceDown = false);
  MoEDirectACommands(const MoEDirectACommands &) = delete;
  MoEDirectACommands &operator=(const MoEDirectACommands &) = delete;
  MoEDirectACommands(MoEDirectACommands &&) noexcept = default;
  MoEDirectACommands &operator=(MoEDirectACommands &&) noexcept = default;

  [[nodiscard]] std::span<const metal::ComputeDispatch> dispatches() const {
    return dispatches_;
  }

private:
  metal::CommandGraph preparation_;
  std::vector<metal::ComputeDispatch> dispatches_;
};

} // namespace splash::flash::candidate
