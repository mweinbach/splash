#pragma once

#include "metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

namespace splash::flash::candidate {

[[nodiscard]] uint32_t moETallJobCapacity(uint32_t rows, uint32_t selections);

// Owns all modified M64 parameter bytes. Unchanged dispatches still borrow
// the source graph's bytes; keep the source CommandGraph alive until submit.
// Production M8/M16/M32 jobs and policies are never modified.
class MoETallDispatches final {
public:
  explicit MoETallDispatches(std::span<const metal::ComputeDispatch> source,
                            uint32_t simdgroups = 4);
  MoETallDispatches(const MoETallDispatches &) = delete;
  MoETallDispatches &operator=(const MoETallDispatches &) = delete;
  [[nodiscard]] std::span<const metal::ComputeDispatch> dispatches() const {
    return dispatches_;
  }

private:
  std::deque<std::vector<std::byte>> payloads_;
  std::vector<metal::ComputeDispatch> dispatches_;
};

} // namespace splash::flash::candidate
