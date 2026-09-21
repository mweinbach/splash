#pragma once

#include "flash/FlashMoEBlocked.hpp"

namespace splash::flash::candidate {
inline constexpr const char *kMoEDecodeF32Semantics =
    "private-selected-expert-original-f32coef-bf16-input-mixed-mpp-m8n64-mac-k64-bf16-swiglu-scatter-v1";

void addMoEDecodeF32GateUp(metal::CommandGraph &graph,
    const FlashAffineProjection &gate,const FlashAffineProjection &up,
    const FlashMoEBlockedScratch &scratch,metal::MetalBuffer gateTap,
    metal::MetalBuffer upTap,metal::MetalBuffer diagnostics,uint32_t rows);
void addMoEDecodeF32Down(metal::CommandGraph &graph,
    const FlashAffineProjection &down,const FlashMoEBlockedScratch &scratch,
    metal::MetalBuffer diagnostics,uint32_t rows);
void addMoEDecodeBF16GateUpTaps(metal::CommandGraph &graph,
    const FlashAffineProjection &gate,const FlashAffineProjection &up,
    const FlashMoEBlockedScratch &scratch,metal::MetalBuffer gateTap,
    metal::MetalBuffer upTap,metal::MetalBuffer diagnostics,uint32_t rows);
void addMoEDecodeF32CoefficientSample(metal::CommandGraph &graph,
    const FlashAffineProjection &projection,uint32_t expert,uint32_t outputBegin,
    uint32_t inputBegin,metal::MetalBuffer output,metal::MetalBuffer diagnostics,
    bool bf16Operand=false);
}
