#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif
#include "metal/abi/FlashMoEBuckets.h"
struct FlashExpertR4CohortParams {
  uint32_t rows,selections,experts,output_tile,slots,epoch,reserved0,reserved1;
};
struct FlashExpertR4CohortProbeParams {
  FlashExpertR4CohortParams cohort;
  uint32_t input_size,output_size,per_route_input,reserved;
};
static_assert(sizeof(FlashExpertR4CohortParams)==32);
static_assert(sizeof(FlashExpertR4CohortProbeParams)==48);
static_assert(alignof(FlashExpertR4CohortParams)==4);
static_assert(alignof(FlashExpertR4CohortProbeParams)==4);
enum : uint32_t {
  kExpertR4MetaReady=0x4d345031u,
  kExpertR4GateReady=0x47344e16u,
  kExpertR4DownReady=0x44344e16u,
  kExpertR4GateCTAs=400u,
  kExpertR4DownCTAs=1600u,
  kExpertR4MetadataBytes=6796u,
};
