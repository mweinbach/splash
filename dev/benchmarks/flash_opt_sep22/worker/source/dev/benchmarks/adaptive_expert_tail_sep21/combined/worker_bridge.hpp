#pragma once
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace splash::flash::adaptive_expert_tail_sg2k128_sep21 {
inline bool parseSwitch(const char *raw) {
  if (!raw || std::string_view(raw) == "0") return false;
  if (std::string_view(raw) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21 must be 0 or 1");
}
inline bool requested() {
  static const bool selected = parseSwitch(std::getenv("SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21"));
  return selected;
}
inline constexpr const char *producerNameFor(bool gate, uint32_t selectedSG2, bool tailEnabled) {
  if (selectedSG2 != 7) return "";
  if (tailEnabled) return gate ? "adaptive_expert_tail_sg2k128_sep21_gate_up_m16_tail" :
      "adaptive_expert_tail_sg2k128_sep21_down_scatter_m16_tail";
  return gate ? "prefill_moe_sep21_memory_fixed_gate_up_m32_n64_k128_sg2" :
      "prefill_moe_sep21_memory_fixed_down_scatter_m32_n64_k128_sg2";
}
inline constexpr const char *markerFor(uint32_t selectedSG2, bool tailEnabled) {
  return selectedSG2 == 7 && tailEnabled ?
      ";private-prefill-moe-sg2-k128-m16-tail-validle16-original-m32-jobs-r2048-exact-f32-bf16-sep21-v1" : "";
}
} // namespace splash::flash::adaptive_expert_tail_sg2k128_sep21
