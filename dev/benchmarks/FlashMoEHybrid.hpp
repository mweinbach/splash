#pragma once

#include "metal/MetalBackend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

namespace splash::flash::candidate {

struct MoEHybridJobList {
  uint32_t tile = 0;
  uint32_t capacity = 0;
  uint32_t launchCapacity = 0;
  metal::MetalBuffer offsets;
  metal::MetalBuffer count;
  metal::MetalBuffer jobs;
  std::array<metal::MetalBuffer,3> allocations;
  [[nodiscard]] bool canariesClean() const;
};

// Each expert's bucket count selects M16 <=16, M32 <=32, otherwise M64.
// GPU compact lists feed the qualified production M16/M32 SG4 and M64 SG8
// consumers verbatim. The original source graph must survive submission.
class MoEHybridDispatches final {
public:
  MoEHybridDispatches(metal::MetalBackend &backend,
                     std::span<const metal::ComputeDispatch> source);
  MoEHybridDispatches(const MoEHybridDispatches &) = delete;
  MoEHybridDispatches &operator=(const MoEHybridDispatches &) = delete;
  [[nodiscard]] std::span<const metal::ComputeDispatch> dispatches() const {
    return dispatches_;
  }
  [[nodiscard]] const std::array<MoEHybridJobList,3> &lists() const { return lists_; }

private:
  std::array<MoEHybridJobList,3> lists_;
  std::deque<std::vector<std::byte>> payloads_;
  std::vector<metal::ComputeDispatch> dispatches_;
};

} // namespace splash::flash::candidate
