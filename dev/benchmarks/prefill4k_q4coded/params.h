#pragma once
#include "metal/abi/FlashInt8ExpertStore.h"
// Selector0 covers only UINT_MAX rank misses; selector1 covers all experts.
struct Prefill4KQ4CodedGateParams { FlashInt8ExpertStoreGateParams source; uint32_t selector,reserved0,reserved1,reserved2; };
struct Prefill4KQ4CodedDownParams { FlashInt8ExpertStoreDownParams source; uint32_t selector,reserved0,reserved1,reserved2; };
struct Prefill4KQ4CodedSumParams { uint32_t routes,width,groups,reserved; };
static_assert(sizeof(Prefill4KQ4CodedGateParams) ==144);
static_assert(sizeof(Prefill4KQ4CodedDownParams) ==112);
static_assert(sizeof(Prefill4KQ4CodedSumParams) ==16);
