#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif
struct FlashGEMVDecodeR1Params { uint32_t rows,selections,experts,reserved; };
static_assert(sizeof(FlashGEMVDecodeR1Params)==16);
static_assert(alignof(FlashGEMVDecodeR1Params)==4);
