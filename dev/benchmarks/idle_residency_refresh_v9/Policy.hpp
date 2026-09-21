#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
namespace splash::metal::private_idle_residency {
inline constexpr double kIdleThresholdSeconds = 5.0;
inline bool parseSwitch(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY_REFRESH must be 0 or 1");
}
inline bool shouldRefresh(bool enabled, bool setExists, bool stopping,
    bool outstanding, double lastCompleted, double now) noexcept {
  return enabled && setExists && !stopping && !outstanding &&
      std::isfinite(lastCompleted) && std::isfinite(now) &&
      lastCompleted > 0.0 && now - lastCompleted > kIdleThresholdSeconds;
}
struct Refresh {
  bool requested = false, succeeded = false;
  uint64_t count = 0, sequence = 0;
  double lastCompletedSeconds = 0.0, beganSeconds = 0.0,
      idleSeconds = 0.0, apiSeconds = 0.0;
  uint64_t baseCount = 0, bytes = 0;
};
inline thread_local Refresh lastRefresh{};
} // namespace splash::metal::private_idle_residency
