#pragma once

#include "metal/abi/FlashGreedyGPU.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace splash::flash::mtp_gpu_chain_policy {

inline constexpr uint32_t kMaximumDepth = 2, kMaximumVocabulary = 248320;
inline constexpr uint32_t kMaximumCapacity = 262144, kNoToken = UINT32_MAX;
inline constexpr uint32_t kErrorNonfinite = 1, kErrorInvalidGreedy = 2;
inline constexpr uint32_t kErrorDiagnostics = 4, kErrorCapacity = 8;
inline constexpr uint32_t kErrorMalformedParams = 16;
inline constexpr const char *kSemantics =
    "private-seeded-exact-greedy-two-proposal-r1-head-indirect-eos-quota-v1";

// CPU reference layout only. The parent's independently owned shader ABI must
// be checked against these field meanings; this header never creates a device.
struct Params {
  uint32_t vocabulary = kMaximumVocabulary;
  uint32_t requestedDepth = 2, remaining = 3, begin = 0, capacity = 4096;
  uint32_t dispatchCount = 1;
  uint32_t reserved[2]{};
};
struct Control {
  uint32_t proposalCount = 0, consumedPairs = 0, bodyEnabled = 0, finishedEOS = 0;
  uint32_t errors = 0, reserved[3]{};
  uint32_t proposals[2]{kNoToken, kNoToken};
};
static_assert(sizeof(Params) == 32);
static_assert(sizeof(Control) == 40);
using Groups = std::array<uint32_t, 3>;

[[nodiscard]] constexpr bool stopToken(uint32_t token) noexcept {
  return token == 248044 || token == 248046;
}
[[nodiscard]] constexpr uint32_t paramsError(const Params &p) noexcept {
  return !p.vocabulary || p.vocabulary > kMaximumVocabulary ||
      p.requestedDepth > kMaximumDepth || !p.remaining || !p.capacity ||
      p.capacity > kMaximumCapacity || p.begin > p.capacity ||
      p.reserved[0] || p.reserved[1] ? kErrorMalformedParams : 0;
}
[[nodiscard]] inline uint32_t boundedDepth(uint32_t requested, uint32_t remaining) {
  if (requested > kMaximumDepth || !remaining)
    throw std::invalid_argument("private two-proposal chain requires depth0..2 and a positive budget");
  return std::min(requested, remaining - 1);
}
// Exact acceptance rules and nonfinite priority from greedyGPUResultToken.
// Do not promote a malformed rank or a reserved/error record to a token.
[[nodiscard]] constexpr uint32_t recordError(
    const FlashGreedyGPURowResult &r, uint32_t vocabulary) noexcept {
  if (r.errors & kFlashGreedyGPUErrorNonfinite) return kErrorNonfinite;
  return !vocabulary || vocabulary > kMaximumVocabulary || r.errors ||
      r.reserved || r.token >= vocabulary || r.rank < 0x80 ||
      r.rank > 0xff7f || r.rank == 0x7fff ? kErrorInvalidGreedy : 0;
}
[[nodiscard]] inline uint32_t recordToken(
    const FlashGreedyGPURowResult &r, uint32_t vocabulary) {
  const auto error = recordError(r, vocabulary);
  if (error == kErrorNonfinite)
    throw std::runtime_error("non-finite Flash vocabulary logit");
  if (error)
    throw std::runtime_error("Flash MTP GPU greedy result has invalid status or extent");
  return r.token;
}

struct Prepared {
  Params params;
  Control control;
  uint32_t depth = 0;
  bool seedRead = false;
};
[[nodiscard]] inline Prepared prepare(
    const Params &params, const FlashGreedyGPURowResult &seed,
    uint32_t priorDiagnostics = 0,
    std::array<uint32_t, 2> existingProposals = {kNoToken, kNoToken}) {
  Prepared out;
  out.params = params;
  out.control.proposals[0] = existingProposals[0];
  out.control.proposals[1] = existingProposals[1];
  out.control.errors = paramsError(params);
  if (out.control.errors) return out;
  out.depth = boundedDepth(params.requestedDepth, params.remaining);
  if (!out.depth) return out; // Ignore unneeded seed and prior diagnostics.
  if (priorDiagnostics) {
    out.control.errors = kErrorDiagnostics;
    return out;
  }
  out.seedRead = true;
  out.control.errors = recordError(seed, params.vocabulary);
  if (out.control.errors) return out;
  out.control.proposals[0] = seed.token;
  out.control.proposalCount = 1;
  out.control.finishedEOS = stopToken(seed.token);
  if (out.depth < 2 || out.control.finishedEOS) return out;
  if (params.begin == params.capacity) {
    out.control.errors = kErrorCapacity;
    return out;
  }
  if (!params.dispatchCount) {
    out.control.errors = kErrorMalformedParams;
    return out;
  }
  out.control.bodyEnabled = 1;
  return out;
}
[[nodiscard]] constexpr Groups indirectGroups(
    const Prepared &prepared, Groups staticGroups) noexcept {
  return prepared.control.bodyEnabled ? staticGroups : Groups{0, 0, 0};
}
inline void writeNextToken(const Prepared &prepared, int64_t &nextI64) noexcept {
  if (prepared.control.bodyEnabled)
    nextI64 = int64_t(prepared.control.proposals[0]);
}

struct Completion {
  Control control;
  uint32_t logicalLength = 0, rollbackLength = 0;
  bool ready = false, poisonHead = false;
};
// State advancement is a publication after BOTH command completion and
// diagnostic/second-record validation. An actual failed body poisons its head;
// a rejected seed/zero-work guard never does. The GPU suffix is not read for a
// skipped body, even if its old words are invalid or nonfinite.
[[nodiscard]] inline Completion complete(
    const Prepared &prepared, const FlashGreedyGPURowResult &second,
    uint32_t bodyDiagnostics, bool commandCompleted,
    bool commandSucceeded = true) noexcept {
  Completion out;
  out.control = prepared.control;
  out.logicalLength = out.rollbackLength = prepared.params.begin;
  if (!commandCompleted) return out;
  out.ready = true;
  if (!commandSucceeded) {
    out.control.errors |= kErrorDiagnostics;
    out.control.proposalCount = 0;
    out.poisonHead = bool(prepared.control.bodyEnabled);
    return out;
  }
  if (!prepared.control.bodyEnabled) return out;
  out.control.consumedPairs = 1;
  if (bodyDiagnostics) {
    out.control.errors |= kErrorDiagnostics;
    out.poisonHead = true;
    return out;
  }
  out.control.errors |= recordError(second, prepared.params.vocabulary);
  if (out.control.errors) {
    out.poisonHead = true;
    return out;
  }
  out.control.proposals[1] = second.token;
  out.control.proposalCount = 2;
  out.control.finishedEOS = stopToken(second.token);
  out.logicalLength = prepared.params.begin + 1;
  return out;
}
[[nodiscard]] constexpr uint32_t poolBegin(uint32_t begin) noexcept {
  return begin / 4;
}
[[nodiscard]] constexpr uint32_t newlyCompletedPools(
    uint32_t begin, uint32_t consumedPairs) noexcept {
  return consumedPairs <= 1 && begin < kMaximumCapacity
      ? (begin + consumedPairs) / 4 - begin / 4 : 0;
}

} // namespace splash::flash::mtp_gpu_chain_policy
