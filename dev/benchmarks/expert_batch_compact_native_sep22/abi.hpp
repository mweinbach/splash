#pragma once
#include "metal/abi/FlashMoEBuckets.h"
#ifndef EXPERT_BATCH_ROWS
#error "EXPERT_BATCH_ROWS must be fixed to 8 or 16 by the sealed component build"
#endif
static_assert(EXPERT_BATCH_ROWS==8||EXPERT_BATCH_ROWS==16);
enum : uint32_t {
 kCompactNativeBatchRows=EXPERT_BATCH_ROWS,
 kCompactNativeBatchRoutes=EXPERT_BATCH_ROWS*10,
 kCompactNativeBatchTileRows=16,
 kCompactNativeBatchJobCapacity=(EXPERT_BATCH_ROWS*10+15)/16+511,
 kCompactNativeBatchBackingCapacity=(EXPERT_BATCH_ROWS*10+7)/8+511,
 kCompactNativeBatchOperandRows=EXPERT_BATCH_ROWS*10+63,
};
static_assert(sizeof(FlashMoEBucketParams)==32);
static_assert(sizeof(FlashMoEBucketJob)==8);
