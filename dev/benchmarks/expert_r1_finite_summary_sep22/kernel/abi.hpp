#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using R1SummaryUInt = uint;
#else
#include <cstdint>
using R1SummaryUInt = uint32_t;
#endif

struct R1FiniteSummaryInvocation {
  R1SummaryUInt epoch, role, reserved0, reserved1;
};
static_assert(sizeof(R1FiniteSummaryInvocation) == 16);
enum : R1SummaryUInt {
  kR1FiniteSummaryMagic = 0x46535231u,
  kR1FiniteSummaryGURole = 1u,
  kR1FiniteSummaryDownRole = 2u,
  kR1FiniteSummaryWords = 16u,
};
