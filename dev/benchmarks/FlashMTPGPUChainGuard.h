#pragma once
#include "metal/abi/FlashGreedyGPU.h"

enum : uint32_t {
  kFlashMTPGPUChainNonfinite = 1,
  kFlashMTPGPUChainInvalidGreedy = 2,
  kFlashMTPGPUChainDiagnostics = 4,
  kFlashMTPGPUChainCapacity = 8,
  kFlashMTPGPUChainParameters = 16,
};
struct FlashMTPGPUChainGuardParams {
  uint32_t vocabulary, requested_depth, remaining, begin;
  uint32_t capacity, dispatch_count, reserved0, reserved1;
};
struct FlashMTPGPUChainControl {
  uint32_t proposal_count, consumed_pairs, body_enabled, finished_eos;
  uint32_t errors, reserved0, reserved1, reserved2;
  uint32_t proposals[2];
};
static_assert(sizeof(FlashMTPGPUChainGuardParams)==32);
static_assert(sizeof(FlashMTPGPUChainControl)==40);
