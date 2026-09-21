#pragma once
#include "metal/abi/FlashGreedyGPU.h"

enum : uint32_t {
  kFlashMTPGPUChainNonfinite = 1,
  kFlashMTPGPUChainInvalidGreedy = 2,
  kFlashMTPGPUChainDiagnostics = 4,
  kFlashMTPGPUChainCapacity = 8,
  kFlashMTPGPUChainParameters = 16,
};
struct FlashMTPGPUChainFourGuardParams {
  uint32_t vocabulary, requested_depth, remaining, begin;
  uint32_t capacity, dispatch_count, body_index, reserved;
};
struct FlashMTPGPUChainFourControl {
  uint32_t proposal_count, consumed_pairs, body_enabled, finished_eos;
  uint32_t errors, reserved0, reserved1, reserved2;
  uint32_t proposals[4];
};
static_assert(sizeof(FlashMTPGPUChainFourGuardParams)==32);
static_assert(sizeof(FlashMTPGPUChainFourControl)==48);
