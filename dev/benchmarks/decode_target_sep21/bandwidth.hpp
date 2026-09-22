#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using BWU32 = uint;
using BWU64 = ulong;
#else
#include <cstdint>
using BWU32 = uint32_t;
using BWU64 = uint64_t;
#endif
struct BWParams {
  BWU64 vectorsPerBuffer, vectors, chunkVectors;
  BWU32 seedA, seedB, stamp, reserved;
};
struct BWRecord {
  BWU64 sums[4], vectors;
  BWU32 first, last, badWords, stamp;
};
#ifdef __METAL_VERSION__
constant constexpr BWU32 kBWMultiplier = 2654435761u;
constant constexpr BWU32 kBWThreads = 256;
constant constexpr BWU64 kBWChunkVectors = 65536;
#else
constexpr BWU32 kBWMultiplier = 2654435761u;
constexpr BWU32 kBWThreads = 256;
constexpr BWU64 kBWChunkVectors = 65536;
#endif
#ifndef __METAL_VERSION__
static_assert(sizeof(BWParams) == 40);
static_assert(sizeof(BWRecord) == 56);
#endif
