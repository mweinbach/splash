#pragma once
#include "flash/FlashHCFused.hpp"
namespace splash::flash::candidate {
enum class HCDownPackedMode : uint32_t { Rows2 = 0, Rows4 = 1 };
inline const char *hcDownPackedModeName(HCDownPackedMode mode) {
  switch (mode) {
    case HCDownPackedMode::Rows2: return "packed-literal-lane32-reuse2-original-f32coeff";
    case HCDownPackedMode::Rows4: return "packed-literal-lane32-reuse4-original-f32coeff";
  }
  throw std::invalid_argument("invalid packed HC-down mode");
}
struct HCDownPackedDebug { metal::MetalBuffer rawBF16, rawF32; };
void addHCDownPackedCandidate(metal::MetalBackend &backend, metal::CommandGraph &graph,
    metal::MetalBuffer normalized, const FlashAffineProjection &down,
    const FlashAffineProjection *injection, metal::MetalBuffer activated,
    metal::MetalBuffer gates, metal::MetalBuffer diagnostic, uint32_t rows,
    HCDownPackedMode mode, const HCDownPackedDebug &debug = {});
void addHCDownPackedLiteralWitness(metal::MetalBackend &backend, metal::CommandGraph &graph,
    metal::MetalBuffer normalized, const FlashAffineProjection &down,
    const FlashAffineProjection *injection, metal::MetalBuffer activated,
    metal::MetalBuffer gates, metal::MetalBuffer diagnostic, uint32_t rows,
    const HCDownPackedDebug &debug);
}
