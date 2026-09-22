#pragma once
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::adaptive_expert_tail_sep21 {
inline bool parseSwitch(const char *raw) {
  if (!raw || std::string_view(raw) == "0") return false;
  if (std::string_view(raw) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SEP21 must be 0 or 1");
}
inline bool requested() {
  // Startup freezes/validates before paths, metadata or a backend are touched.
  static const bool selected = parseSwitch(std::getenv("SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SEP21"));
  return selected;
}
inline bool selected(std::string_view phase, uint32_t tileRows, bool enabled) {
  return enabled && tileRows == 32 && (phase == "gate_up" || phase == "down_scatter");
}
inline std::string pipeline(const char *phase, uint32_t tileRows, bool enabled) {
  if (tileRows != 16 && tileRows != 32 && tileRows != 64)
    throw std::invalid_argument("saved INT8 expert tile must be M16/M32/M64");
  if (selected(phase, tileRows, enabled))
    return std::string("adaptive_expert_tail_sep21_") + phase + "_m16_tail";
  return std::string("flash_int8_expert_store_") + phase + "_m" + std::to_string(tileRows) +
      (tileRows == 64 ? "_n64_sg8" : "_n64");
}
inline constexpr std::string_view marker(bool enabled) {
  return enabled ? ";private-native-m32-jobs-m16-tail-validle16-sg4-exact-f32-bf16-sep21-v1" : "";
}
} // namespace splash::flash::adaptive_expert_tail_sep21
