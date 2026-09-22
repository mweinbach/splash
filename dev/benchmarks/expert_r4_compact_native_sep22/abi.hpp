#pragma once
#include "metal/abi/FlashMoEBuckets.h"
// Use the unchanged native M16 job and parameter layouts.
enum : uint32_t {
  kCompactNativeR4Rows=4,
  kCompactNativeR4Routes=40,
  kCompactNativeR4TileRows=16,
  kCompactNativeR4JobCapacity=514,
};
static_assert(sizeof(FlashMoEBucketParams)==32);
static_assert(sizeof(FlashMoEBucketJob)==8);
