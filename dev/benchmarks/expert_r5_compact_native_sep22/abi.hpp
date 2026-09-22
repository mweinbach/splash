#pragma once
#include "metal/abi/FlashMoEBuckets.h"
#ifndef EXPERT_R5_ROWS
#error "EXPERT_R5_ROWS must be fixed to 5 by the sealed component build"
#endif
static_assert(EXPERT_R5_ROWS==5);
enum : uint32_t {
 kCompactNativeR5Rows=EXPERT_R5_ROWS,
 kCompactNativeR5Routes=EXPERT_R5_ROWS*10,
 kCompactNativeR5TileRows=16,
 kCompactNativeR5JobCapacity=(EXPERT_R5_ROWS*10+15)/16+511,
 kCompactNativeR5BackingCapacity=(EXPERT_R5_ROWS*10+7)/8+511,
 kCompactNativeR5OperandRows=EXPERT_R5_ROWS*10+63,
};
static_assert(sizeof(FlashMoEBucketParams)==32);
static_assert(sizeof(FlashMoEBucketJob)==8);
