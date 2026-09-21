#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace splash::flash::private_prefill_frontier {
inline bool parseSwitch(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_PRIVATE_PREFILL_FRONTIER_SPLIT must be 0 or 1");
}
inline bool eligible(bool enabled, uint32_t rows, uint32_t begin,
    bool verification) noexcept {
  return enabled && rows == 128 && begin == 0 && !verification;
}
struct Execution {
  bool split = false;
  uint64_t dispatches = 0;
  metal::CommandTiming first{}, remainder{};
};
inline thread_local Execution lastExecution{};
} // namespace splash::flash::private_prefill_frontier
