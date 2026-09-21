#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>

namespace splash::flash {

// Private experiment state, deliberately outside FlashWeights' production
// layout and API. The worker owns the callback only through startup loading.
struct PrivateOwnedOriginalStorage final {
  bool enabled = false;
  bool verifiedPayloadHashes = false;
  uint64_t nativeBaseCount = 0;
  uint64_t openedMappingCount = 0;
  uint64_t releasedMappingCount = 0;
  uint64_t activeMappings = 0;
  uint64_t maximumActiveMappings = 0;
  uint64_t temporaryMappingPeakBytes = 0;
  uint64_t copiedBytes = 0;
  uint64_t verifiedCopyBytes = 0;
  uint64_t maximumAdmissionBytes = 0;
  uint64_t admittedShardCopies = 0;
  double copySeconds = 0;
  double comparisonSeconds = 0;
  double payloadHashSeconds = 0;
  double loadSeconds = 0;
};

[[nodiscard]] bool privateOwnedOriginalRequested();
void privateOwnedOriginalSetAdmission(metal::AllocationAdmission admission);
[[nodiscard]] PrivateOwnedOriginalStorage privateOwnedOriginalStorage() noexcept;

} // namespace splash::flash
