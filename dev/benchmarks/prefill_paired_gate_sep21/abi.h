#pragma once

#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashMoEBuckets.h"

// Private one-layer saved-Full512-I8 gate/up pairing experiment. The original
// signed I8 code rows, fitted positive F32 row scales, and rank map are not
// changed. A Root-owned temporary plane only rearranges whole code rows:
// paired[rank, block*128 + side*64 + c, k] =
// original_side[rank, block*64 + c, k], side 0 gate / 1 up.
// Full512 extent [512,1280,2560] is exactly 1,677,721,600 bytes.
enum : uint32_t {
  kPrefillPairedGateSep21TileRows = 32,
  kPrefillPairedGateSep21TileColumns = 128,
  kPrefillPairedGateSep21OutputColumns = 640,
  kPrefillPairedGateSep21InputColumns = 2560,
  kPrefillPairedGateSep21ColumnBlocks = 10,
  kPrefillPairedGateSep21PairedColumns = 1280,
  kPrefillPairedGateSep21StoredExperts = 512,
  kPrefillPairedGateSep21SharedBytes = 8192,
};

// All producers use unchanged 32-byte FlashInt8ExpertStoreParams.
// Grid {10, job_capacity, 1}; threadgroup {SG*32,1,1}, SG 2 or 4.
// The input/output route ordering is the original packed bucket ordering.
// Valid row tails are bounded by the bucket's offsets, never padded or copied.
// There is one whole-K M32N128 matmul (or one accumulate per ascending K128
// slice for BF16 _k128 variants), then the rounded paired projections are
// placed in 8KiB threadgroup BF16 memory. One CTA barrier precedes the native
// BF16 sigmoid/SwiGLU sequence. No lane/warp ownership assumption is made.
// Scales retain the saved-I8 positive/finite contract; diagnostics retain bits
// 1 invalid expert/rank, 2 malformed parameters/job, 4 nonfinite/invalid scale.
// Down, staging, route scatter, quantizers, and decode are unchanged.
//
// BF16 ABI:
//  0 packed BF16 A [route,2560] (read only)
//  1 temporary paired signed I8 B [stored_experts,1280,2560] (read only)
//  2 original positive F32 gate scales [stored_experts,640] (read only)
//  3 original positive F32 up scales [stored_experts,640] (read only)
//  4 original ranks[512], 5 original offsets[513], 6 original M32 jobs,
//  7 original jobCount, 8 activated BF16 output [route,640],
//  9 sticky diagnostics, 10 FlashInt8ExpertStoreParams.
// BF16 _audit appends 11 raw F32 gate dot, 12 raw F32 up dot,
// 13 scaled F32 gate, 14 scaled F32 up, each packed [route,640]. Audit commands
// are excluded from timing; timed kernels have no audit buffer arguments.
//
// W8A8 ABI uses the same indices except 0 is quantized I8 A [route,2560],
// and 11 is positive F32 activation scales [route]. W8A8 _audit appends
// 12 raw F32 gate dot, 13 raw F32 up dot, 14 scaled F32 gate,
// 15 scaled F32 up, 16 exact raw I32 gate dot, 17 exact raw I32 up dot.
// Raw F32 is float(rawI32); scaled F32 is explicitly
// (float(rawI32) * activationScale) * originalWeightScale, preserving the
// existing W8A8 association with contraction and reassociation disabled.
// Include both original gate/down activation quantizers in full-chain timing.
// Worst integer magnitude 127*128*2560 = 41,615,360 < INT32_MAX.
// Rounded BF16 G/U can be reconstructed from scaled F32 audit buffers; the
// actual activated BF16 output is also checked against the native producer.

static_assert(sizeof(FlashInt8ExpertStoreParams) == 32);
static_assert(kPrefillPairedGateSep21ColumnBlocks * 64 ==
              kPrefillPairedGateSep21OutputColumns);
static_assert(kPrefillPairedGateSep21ColumnBlocks *
                  kPrefillPairedGateSep21TileColumns ==
              kPrefillPairedGateSep21PairedColumns);
static_assert(kPrefillPairedGateSep21TileRows *
                  kPrefillPairedGateSep21TileColumns * 2 ==
              kPrefillPairedGateSep21SharedBytes);
