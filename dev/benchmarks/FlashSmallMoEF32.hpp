#pragma once
#include "flash/FlashAffine.hpp"
#include "FlashSmallMoEF32ABI.h"

namespace splash::flash::candidate {
struct SmallMoEF32Scratch final {
  metal::MetalBuffer jobs,count;
  uint32_t routeCapacity=0;
};
[[nodiscard]] SmallMoEF32Scratch allocateSmallMoEF32Scratch(metal::MetalBackend &backend,uint32_t rows);
void addSmallMoEF32Jobs(metal::CommandGraph &graph,metal::MetalBuffer ids,
    const SmallMoEF32Scratch &scratch,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t groupRows);
void addSmallMoEF32Gate(metal::CommandGraph &graph,metal::MetalBuffer input,
    const FlashAffineProjection &gate,const FlashAffineProjection &up,metal::MetalBuffer ids,
    const SmallMoEF32Scratch &scratch,metal::MetalBuffer gateOutput,metal::MetalBuffer upOutput,
    metal::MetalBuffer activation,metal::MetalBuffer gateF32,metal::MetalBuffer upF32,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t groupRows,bool writeTaps);
void addSmallMoEF32Down(metal::CommandGraph &graph,metal::MetalBuffer input,
    const FlashAffineProjection &down,metal::MetalBuffer ids,const SmallMoEF32Scratch &scratch,
    metal::MetalBuffer output,metal::MetalBuffer sumF32,metal::MetalBuffer diagnostics,
    uint32_t rows,uint32_t groupRows,bool writeTaps);
void addSmallMoEF32Reference(metal::CommandGraph &graph,metal::MetalBuffer input,
    const FlashAffineProjection &projection,metal::MetalBuffer ids,metal::MetalBuffer output,
    metal::MetalBuffer sumF32,metal::MetalBuffer diagnostics,uint32_t rows,bool inputPerSelection);
} // namespace splash::flash::candidate
