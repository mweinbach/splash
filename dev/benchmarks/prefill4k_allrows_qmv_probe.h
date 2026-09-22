#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Untimed numeric taps, independent of the activated/timed producer kernels.
// Legal phases: K2560/N640/per_route_input=0, or K640/N2560/per_route_input=1.
// Rows are 1..16, selections=10, experts=512, reserved=0.
struct FlashQMVProbeParams {
  uint32_t rows;
  uint32_t selections;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t per_route_input;
  uint32_t columns;
  uint32_t experts;
  uint32_t reserved;
};
static_assert(sizeof(FlashQMVProbeParams) == 32,
              "QMV numeric probes use eight uint32 values");
static_assert(alignof(FlashQMVProbeParams) == 4,
              "QMV numeric probe alignment must match host and Metal");

// flash_qmv_probe_c1/c2, columns=1/2:
// 0 BF16 input, 1 signed I8 codes, 2 F32 row scales, 3 U32 ranks[512],
// 4 original I64 IDs[R,10], 5 F32 raw dot[R,10,N], 6 F32 scaled[R,10,N],
// 7 BF16 projection[R,10,N], 8 sticky U32 diagnostic, 9 params.
// Input shape is [R,K] for per_route_input=0, or [R,10,K] for =1.
// Codes/scales must allocate all 512 persisted ranks: [512,N,K]/[512,N].
// Dispatch {N/(4*columns),R,10}, threads {128,1,1}.
// Both variants use identical per-output lane K strides and late scaling.
// Standalone tap policy: invalid ID/rank poisons all projection states and ORs
// 5 in either phase; this is not the timed gate producer's interim bit policy.

// flash_qmv_probe_mpp_m16, columns=0:
// 0 sanitized packed BF16[R*10,K], 1 signed I8 codes,
// 2 F32 row scales, 3 U32 ranks[512], 4 U32 offsets[513],
// 5 FlashMoEBucketJob jobs, 6 U32 jobCount[1], 7 U32 routeMap[R*10],
// 8 F32 raw dot[R,10,N], 9 F32 scaled[R,10,N], 10 BF16 projection[R,10,N],
// 11 sticky U32 diagnostic, 12 params.
// Dispatch {N/64,R*10,1}, threads {128,1,1}; jobCount must be <= R*10.
// Input must be the existing producer's sanitized packed operand. This tap
// preserves the whole-K M16/N64/SG4 MPP descriptor rather than staging K chunks.
// Outputs scatter through routeMap. Excluded canonical routes become NaN.
// As in the paired producer, caller-validated jobs must cover the live prefix
// exactly once with disjoint M16 tiles; routeMap's live prefix must be unique.
// The allocation uses full route capacity; offsets[512] gives the live prefix.
// Carry the pack/sanitizer's sticky diagnostic into this tap: its sanitized
// operand no longer contains the original nonfinite-input evidence.
// Invalid ranks use the same standalone NaN/5 policy; malformed metadata uses 2.

// flash_qmv_probe_unpack_activation, K2560/N640/per_route_input=0/columns=0:
// 0 packed BF16 activation[R*10,640], 1 U32 canonicalToPacked[R*10],
// 2 canonical BF16 activation[R,10,640], 3 sticky U32 diagnostic, 4 params.
// Dispatch {3,R*10,1}, threads {256,1,1}; excluded routes become NaN.
// The packed allocation uses full route capacity, including its dead tail.
