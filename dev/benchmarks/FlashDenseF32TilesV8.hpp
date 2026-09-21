#pragma once
#include "flash/FlashFloatDenseCache.hpp"
namespace splash::flash::candidate {
void addDenseF32TileV8(metal::MetalBackend &, metal::CommandGraph &,
    metal::MetalBuffer, const FlashTensor &, metal::MetalBuffer,
    metal::MetalBuffer, uint32_t, FlashFloatDenseSmallRowsWorkspace &, uint32_t);
}
