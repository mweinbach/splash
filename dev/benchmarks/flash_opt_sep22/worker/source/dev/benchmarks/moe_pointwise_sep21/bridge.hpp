#pragma once
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashMoEBlocked.h"
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace splash::flash::pointwise_sep21 {
inline bool parseSwitch(const char *raw) {
  if (!raw || std::string_view(raw) == "0") return false;
  if (std::string_view(raw) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_MOE_POINTWISE_SEP21 must be0 or1");
}
inline bool requested() {
  // The worker explicitly freezes this value before its backend is created.
  static const bool selected = parseSwitch(std::getenv("SPLASH_FLASH_MOE_POINTWISE_SEP21"));
  return selected;
}
enum class CombineRoute { Original, SimdSlots, WholeRow };
inline constexpr CombineRoute combineRoute(uint32_t rows, uint32_t width,
    uint32_t experts, uint32_t selections, bool enabled) {
  if (!enabled || !rows || rows > 8192 || width != 2560 || experts != 512 || selections != 10)
    return CombineRoute::Original;
  if (rows <= 16) return CombineRoute::SimdSlots;
  if (rows >= 256) return CombineRoute::WholeRow;
  return CombineRoute::Original;
}
inline constexpr bool poisonEnabled(uint32_t rows, uint32_t selections, bool enabled) {
  return enabled && rows >= 256 && rows <= 8192 && selections && selections <= 10;
}
inline constexpr std::string_view marker(bool enabled) {
  return enabled ? ";private-moe-pointwise-exact-bf16-simdslots-r1to16-ctarow-r256plus-poison-sg1-r256plus-sep21-v1" : "";
}
inline void addPoison(metal::CommandGraph &graph, std::vector<metal::MetalBuffer> buffers,
    const FlashMoEBlockedDownParams &p) {
  const bool enabled = poisonEnabled(p.affine.rows, p.affine.selections, requested());
  graph.add(enabled ? "private_moe_poison_route32" : "flash_moe_blocked_poison_excluded_routes",
      std::move(buffers), p, {enabled ? 1u : 10u, p.route_capacity, 1},
      {enabled ? 32u : 256u, 1, 1});
}
} // namespace splash::flash::pointwise_sep21
