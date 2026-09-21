#pragma once
#include "flash/FlashHCFused.hpp"
#include "flash/FlashFloatDenseCache.hpp"

namespace splash::flash::candidate {
enum class HCDownF32Mode : uint32_t {
  LiteralRowReuse = 0, M8N32 = 1, M8N64 = 2, M8N32Split4 = 3, M8N32Split8 = 4,
};
inline const char *hcDownF32ModeName(HCDownF32Mode mode) {
  switch (mode) {
  case HCDownF32Mode::LiteralRowReuse: return "cached-f32-literal-lane32-row-reuse";
  case HCDownF32Mode::M8N32: return "bf16xf32-whole-k-m8n32-reduction-alternative";
  case HCDownF32Mode::M8N64: return "bf16xf32-whole-k-m8n64-reduction-alternative";
  case HCDownF32Mode::M8N32Split4: return "bf16xf32-split4-m8n32-f32-fold-reduction-alternative";
  case HCDownF32Mode::M8N32Split8: return "bf16xf32-split8-m8n32-f32-fold-reduction-alternative";
  }
  throw std::invalid_argument("invalid HC down candidate mode");
}
struct HCDownF32Scratch { metal::MetalBuffer padded, partials; };
struct HCDownF32Debug { metal::MetalBuffer rawBF16, rawF32; };

// Private only; no model lookup, allocation or GPU submission. Exact cached
// F32 coefficients are checked externally against the original source. Raw
// injection coefficients retain the literal original format and traversal.
void addHCDownF32Candidate(metal::MetalBackend &backend, metal::CommandGraph &graph,
    metal::MetalBuffer normalized, const FlashTensor &cachedDown,
    const FlashAffineProjection &originalDown,
    const FlashAffineProjection *injection, metal::MetalBuffer activated,
    metal::MetalBuffer gates, metal::MetalBuffer diagnostic, uint32_t rows,
    HCDownF32Mode mode, const HCDownF32Scratch &scratch = {},
    const HCDownF32Debug &debug = {});

// Untimed witness cloned from the production literal down producer. Includes
// original SIMD sum, BF16 raw dot, division, activation and injection stages.
void addHCDownLiteralWitness(metal::MetalBackend &backend, metal::CommandGraph &graph,
    metal::MetalBuffer normalized, const FlashAffineProjection &down,
    const FlashAffineProjection *injection, metal::MetalBuffer activated,
    metal::MetalBuffer gates, metal::MetalBuffer diagnostic, uint32_t rows,
    const HCDownF32Debug &debug);
} // namespace splash::flash::candidate
