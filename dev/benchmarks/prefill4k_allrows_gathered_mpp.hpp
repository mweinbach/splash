#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif
struct FlashGatheredMPPParams { uint32_t rows, selections, experts, reserved; };
static_assert(sizeof(FlashGatheredMPPParams) == 16);
#ifndef __METAL_VERSION__
#include "prefill4k_gathered_mpp_views.hpp"
namespace splash::flash::gathered_mpp {
inline constexpr const char *kFlag = "SPLASH_FLASH_ALLROWS_GATHERED_MPP";
inline constexpr uint32_t kThreads = 128;
inline constexpr std::string_view kPolicy =
    "private-direct-gathered-signed-i8-bf16-mpp-m16n64-dynamicK-multiply-sg4-validRows1-late-row-scale-bf16-swiglu-finite-deviceA-nonfinite-localA-rows1to16-v1";
inline bool requested() {
  const char *raw = std::getenv(kFlag);
  if (!raw || std::string_view(raw) == "0") return false;
  if (std::string_view(raw) == "1") return true;
  throw std::invalid_argument("SPLASH_FLASH_ALLROWS_GATHERED_MPP must be exactly0 or1");
}
using Geometry = gathered_mpp_view::Geometry;
using ByteView = gathered_mpp_view::ByteView;
inline Geometry geometry(uint32_t rows, uint32_t selections = 10) {
  auto g = gathered_mpp_view::geometry(rows, selections);
  g.gateColumnGroups = 10; g.downColumnGroups = 40;
  return g;
}
inline void validateViews(const Geometry &g, ByteView input, ByteView ids,
                          ByteView output, ByteView diag, bool down) {
  gathered_mpp_view::validateViews(g, input, ids, output, diag, down);
}
} // namespace splash::flash::gathered_mpp
#endif
