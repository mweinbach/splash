#pragma once
#include "metal/CommandGraph.hpp"
#include "FlashInt8JobPartitionABI.h"

namespace splash::flash::candidate {
struct Int8PartitionedJobScratch final {
  metal::MetalBuffer hits, misses, counts, hitCount, missCount;
  uint32_t hitCapacity = 0, missCapacity = 0;
};
[[nodiscard]] Int8PartitionedJobScratch allocateInt8PartitionedJobs(
    metal::MetalBackend &backend, uint32_t rows, uint32_t selections,
    uint32_t tileRows, uint32_t storedExperts);

// Source keeps original producer payloads alive. The preparation graph owns
// the partition ABI; original v5 arithmetic kernels/bindings remain unchanged.
class PartitionedInt8Commands final {
public:
  PartitionedInt8Commands(std::span<const metal::ComputeDispatch> source,
                         const Int8PartitionedJobScratch &scratch);
  PartitionedInt8Commands(const PartitionedInt8Commands &) = delete;
  PartitionedInt8Commands &operator=(const PartitionedInt8Commands &) = delete;
  PartitionedInt8Commands(PartitionedInt8Commands &&) noexcept = default;
  PartitionedInt8Commands &operator=(PartitionedInt8Commands &&) noexcept = default;
  [[nodiscard]] std::span<const metal::ComputeDispatch> dispatches() const { return dispatches_; }
private:
  metal::CommandGraph preparation_;
  std::vector<metal::ComputeDispatch> dispatches_;
};
} // namespace splash::flash::candidate
