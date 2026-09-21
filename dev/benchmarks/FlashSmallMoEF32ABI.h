#pragma once
#include "metal/abi/FlashAffine.h"

struct FlashSmallMoEF32Job {
  int64_t expert;
  uint32_t route_count;
  uint32_t routes[4];
  uint32_t reserved;
};
struct FlashSmallMoEF32JobParams {
  uint32_t rows, selections, route_capacity, group_rows;
  uint32_t reserved0, reserved1, reserved2, reserved3;
};
struct FlashSmallMoEF32GateParams {
  FlashAffineParams gate;
  FlashAffineParams up;
  uint32_t group_rows, job_capacity, write_f32_taps, reserved;
};
struct FlashSmallMoEF32DownParams {
  FlashAffineParams affine;
  uint32_t group_rows, job_capacity, write_f32_taps, reserved;
};
static_assert(sizeof(FlashSmallMoEF32Job)==32);
static_assert(sizeof(FlashSmallMoEF32JobParams)==32);
static_assert(sizeof(FlashSmallMoEF32GateParams)==144);
static_assert(sizeof(FlashSmallMoEF32DownParams)==80);
