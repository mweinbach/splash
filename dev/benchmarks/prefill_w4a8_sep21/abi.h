#pragma once

#include "metal/abi/FlashMoEBuckets.h"
#include "metal/abi/FlashMoEBlocked.h"

// Isolated numerical activation/association experiment. Original checkpoint
// coefficients and their BF16 scale/bias storage remain unchanged. Centered
// signed I4 sidecars contain byte-exact original packed Q4 XOR 0x88; Down alone
// has a zero-padded physical B row stride of 768 I4 elements (384 bytes).
// Neither this ABI nor these kernels are part of the production/decode path.
enum : uint32_t {
  kPrefillW4A8TileRows = 32,
  kPrefillW4A8GuardRows = 63,
  kPrefillW4A8QuantThreads = 256,
};

struct PrefillW4A8QuantParams {
  FlashMoEBucketParams bucket;
  uint32_t physical_rows;       // routes + 63; dispatch one CTA per row.
  uint32_t input_row_stride;    // BF16 elements: 2560 or 640.
  uint32_t code_row_stride;     // I8 bytes/elements: 2560 or 640.
  uint32_t sum_row_stride;      // I32 elements: 40 or 10.
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
};

struct PrefillW4A8GateParams {
  FlashMoEBlockedGateParams source;
  uint32_t physical_rows;
  uint32_t activation_row_stride; // I8 elements, 2560.
  uint32_t sum_row_stride;        // I32 elements, 40.
  uint32_t reserved;
  uint64_t gate_centered_row_stride_bytes;
  uint64_t gate_centered_expert_stride_bytes;
  uint64_t up_centered_row_stride_bytes;
  uint64_t up_centered_expert_stride_bytes;
};

struct PrefillW4A8DownParams {
  FlashMoEBlockedDownParams source;
  uint32_t physical_rows;
  uint32_t activation_row_stride; // I8 elements, 640.
  uint32_t sum_row_stride;        // I32 elements, 10.
  uint32_t reserved;
  uint64_t centered_row_stride_bytes;
  uint64_t centered_expert_stride_bytes;
};

struct PrefillW4A8ProbeSample {
  uint32_t job_index;
  uint32_t row_within_job;
  uint32_t column_within_tile;
  uint32_t column_tile;
};

struct PrefillW4A8ProbeParams {
  FlashMoEBucketParams bucket;  // Job geometry; width is logical K.
  uint32_t physical_rows;
  uint32_t input_row_stride;
  uint32_t sum_row_stride;
  uint32_t sample_count;
  uint32_t plane;              // 0 gate, 1 up, 2 Down.
  uint32_t tile_outputs;       // N of the selected native M32 variant.
  uint32_t reserved0;
  uint32_t reserved1;
  uint64_t centered_row_stride_bytes;
  uint64_t centered_expert_stride_bytes;
  uint64_t parameter_row_stride_bytes;
  uint64_t parameter_expert_stride_bytes;
};

// Quant ABI: 0 BF16 input, 1 offsets[513], 2 I8 codes, 3 F32 row scales,
// 4 I32 group sums, 5 sticky diagnostics (1 route, 2 parameter, 4 nonfinite),
// 6 PrefillW4A8QuantParams. Guard/inactive rows do not read source input and
// are explicitly written as all-zero codes/sums and scale 1.
// Gate ABI: 0 I8 codes, 1 F32 row scales, 2 I32 G64 sums,
// 3 gate centered I4 bytes, 4 original gate BF16 scales, 5 original gate biases,
// 6 up centered I4 bytes, 7 original up scales, 8 original up biases,
// 9 offsets, 10 jobs, 11 jobCount, 12 activated BF16 output, 13 diagnostics,
// 14 PrefillW4A8GateParams. _audit adds 15 raw F32 gate, 16 raw F32 up.
// Down ABI: 0 I8 codes, 1 F32 row scales, 2 I32 G64 sums,
// 3 centered I4 bytes, 4 original scales, 5 original biases, 6 offsets,
// 7 jobs, 8 jobCount, 9 packed-to-canonical routeMap, 10 BF16 route output,
// 11 diagnostics, 12 PrefillW4A8DownParams. _audit adds 13 raw F32 Down,
// scattered using the same canonical routeMap as the BF16 output.
// Matrix dispatch grid {output_size/N, job_capacity, 1}, threadgroup {SG*32,1,1}.
// Native I32 compact probe ABI: 0 I8 codes, 1 I32 G64 sums, 2 centered I4,
// 3 original BF16 scale, 4 original BF16 bias, 5 offsets, 6 jobs, 7 jobCount,
// 8 PrefillW4A8ProbeSample[], 9 I32 dot, 10 I32 activation sum,
// 11 F32 corrected bias, 12 diagnostics, 13 PrefillW4A8ProbeParams.
// Probe grid {sample_count,K/64,1}, threadgroup {SG*32,1,1}. Output index is
// sample*(K/64)+group. Host initializes dot/sum to INT32_MIN and bias to NaN;
// exactly one native cooperative result cell per sample/group must write.
static_assert(sizeof(PrefillW4A8QuantParams) == 64, "W4A8 quant ABI is 64 bytes");
static_assert(sizeof(PrefillW4A8GateParams) == 160, "W4A8 gate ABI is 160 bytes");
static_assert(sizeof(PrefillW4A8DownParams) == 112, "W4A8 Down ABI is 112 bytes");
static_assert(sizeof(PrefillW4A8ProbeSample) == 16, "W4A8 probe sample ABI is 16 bytes");
static_assert(sizeof(PrefillW4A8ProbeParams) == 96, "W4A8 probe ABI is 96 bytes");
