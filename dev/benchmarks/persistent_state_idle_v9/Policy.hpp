#pragma once
#include <cstdint>
namespace splash::flash::private_persistent_state {
inline constexpr uint64_t kPlanes=134,kBytes=349388800;
inline bool resetEligible(bool owner, bool healthy, bool pending, bool outstanding) noexcept {
  return owner && healthy && !pending && !outstanding;
}
}
