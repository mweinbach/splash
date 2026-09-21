#pragma once
#include <cstdint>
#include <limits>

namespace splash::metal::private_indirect {

inline bool validArgumentsRange(uint64_t viewLength, uint64_t offset) noexcept {
  return offset % 4 == 0 && offset <= viewLength && 12 <= viewLength - offset;
}
inline bool validThreads(uint64_t x, uint64_t y, uint64_t z) noexcept {
  return x && y && z && x <= std::numeric_limits<uint64_t>::max() / y &&
      x * y <= std::numeric_limits<uint64_t>::max() / z;
}
// Registration treats invalid ranges conservatively as overlap. Distinct
// valid native allocations cannot alias, regardless of their view offsets.
inline bool overlap(uintptr_t baseIdentity, uint64_t offset, uint64_t length,
    uintptr_t otherBaseIdentity, uint64_t otherOffset, uint64_t otherLength) noexcept {
  if (!baseIdentity || !otherBaseIdentity || !length || !otherLength ||
      offset > std::numeric_limits<uint64_t>::max() - length ||
      otherOffset > std::numeric_limits<uint64_t>::max() - otherLength) return true;
  return baseIdentity == otherBaseIdentity && offset < otherOffset + otherLength &&
      otherOffset < offset + length;
}
} // namespace splash::metal::private_indirect
