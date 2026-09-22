#pragma once
#include "frozen/metal/abi/FlashGDN.h"
#ifndef __METAL_VERSION__
#include <cstddef>
#endif
struct TileChunkParams {
  FlashGDNParams gdn;
  uint32_t mode;
  uint32_t reserved;
};
static_assert(sizeof(TileChunkParams)==56);
#ifndef __METAL_VERSION__
static_assert(offsetof(TileChunkParams,mode)==48);
static_assert(offsetof(TileChunkParams,reserved)==52);
#endif
