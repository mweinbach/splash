#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>
namespace splash::metal::private_completed_retention {
inline bool parseSwitch(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_PRIVATE_COMPLETED_COMMAND_RETENTION must be 0 or 1");
}
// External store mutex protects every operation; displaced ownership is returned
// so ObjC/C++ objects retire after releasing that mutex.
template<class T> struct Slots {
  std::array<T, 2> held{};
  size_t next = 0;
  bool closed = false;
  T publish(T candidate) noexcept {
    if (closed) return candidate;
    T retired = std::move(held[next]);
    held[next] = std::move(candidate);
    next = (next + 1) % held.size();
    return retired;
  }
  std::array<T, 2> close() noexcept {
    closed = true;
    auto retired = std::move(held);
    held = {};
    return retired;
  }
};
} // namespace splash::metal::private_completed_retention
