#pragma once

#include "flash/FlashDenseCache.hpp"

namespace splash::flash::candidate {

inline constexpr const char *kSharedExpertFusedSemantics =
    "cached-bf16-whole-k-mpp-f32accum-rounded-gate-up-compiled-bf16-swiglu-v1";

// Private prefill candidate; immutable BF16[640,2560] operands only.
// No cache conversion, model lookup, CPU dot product or GPU submit occurs.
// Tail scratch needs at most31*640 BF16 elements per plane. Complete tiles
// do not use scratch; an incomplete final tile uses literal qualified vector
// projections and the canonical pointwise producer, preserving its traversal.
struct SharedExpertFusedTail {
  metal::MetalBuffer gate, up;
};

[[nodiscard]] uint32_t sharedExpertFusedTileRows(FlashAffineMPPTile tile);
[[nodiscard]] uint32_t sharedExpertFusedTileOutputs(FlashAffineMPPTile tile);

void addSharedExpertFused(metal::MetalBackend &backend, metal::CommandGraph &graph,
    const FlashTensor &gate, const FlashTensor &up, metal::MetalBuffer input,
    metal::MetalBuffer activated, metal::MetalBuffer diagnostics, uint32_t rows,
    FlashAffineMPPTile tile, const SharedExpertFusedTail &tail = {});

// Qualification only; mirrors the same compiled core while additionally
// materializing the two rounded BF16 dot planes. Never use for timing claims.
void addSharedExpertFusedTaps(metal::MetalBackend &backend, metal::CommandGraph &graph,
    const FlashTensor &gate, const FlashTensor &up, metal::MetalBuffer input,
    metal::MetalBuffer activated, metal::MetalBuffer diagnostics, uint32_t rows,
    FlashAffineMPPTile tile, metal::MetalBuffer gateTap, metal::MetalBuffer upTap);

} // namespace splash::flash::candidate
