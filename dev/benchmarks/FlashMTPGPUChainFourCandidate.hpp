#pragma once

#include "flash/FlashMTP.hpp"
#include "flash/FlashWeights.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace splash::metal { class CommandGraph; }

namespace splash::flash::mtp_gpu_chain_four_candidate {

inline constexpr const char *kFlashMTPSemantics =
    "qwen4-lightning-head-global-hidden-rms-bf16-fuse-qsa-hc-moe-v1";

enum class FlashMTPLogits : uint8_t { None, Last, All };

class FlashMTPForward;
class FlashBatchMTPForward;

// The trained head has one QSA cache and no GDN/PLE state. Its offset counts
// folded (previous hidden, next token) pairs rather than backbone tokens.
class FlashMTPState final {
public:
  FlashMTPState();
  ~FlashMTPState();
  FlashMTPState(const FlashMTPState &) = delete;
  FlashMTPState &operator=(const FlashMTPState &) = delete;
  FlashMTPState(FlashMTPState &&) noexcept;
  FlashMTPState &operator=(FlashMTPState &&) noexcept;
  [[nodiscard]] uint64_t logicalLength() const noexcept;
  [[nodiscard]] uint32_t capacity() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class FlashMTPForward;
  friend class FlashBatchMTPForward;
};

struct FlashMTPResult final {
  metal::CommandTiming timing;
  // Borrowed BF16[logitRows,vocabularySize], absent for None. Both result
  // buffers belong to the shared head arena until the next forward() call.
  metal::MetalBuffer logitsBF16;
  uint32_t logitRows = 0;
  // All four streams BEFORE the head's final HC mixer: BF16[rows,10240].
  // Feeding the final row into this same head's next call is supported.
  metal::MetalBuffer hiddenBF16;
  uint32_t hiddenRows = 0;
  uint64_t logicalLength = 0;
  // Optional borrowed FlashGreedyGPURowResult[greedyRows], exact BF16 greedy
  // IDs from the same command as the head. Absent when disabled or >16 rows.
  metal::MetalBuffer greedyResultsU32{};
  uint32_t greedyRows = 0;
};

// Original checkpoint's embedded one-layer Lightning head. This class only
// folds head pairs; the caller owns target verification and token acceptance.
class FlashMTPForward final {
public:
  FlashMTPForward(metal::MetalBackend &backend, const FlashWeights &weights,
                  uint32_t capacity, uint32_t maximumRows = 128);
  ~FlashMTPForward();
  FlashMTPForward(const FlashMTPForward &) = delete;
  FlashMTPForward &operator=(const FlashMTPForward &) = delete;
  FlashMTPForward(FlashMTPForward &&) noexcept;
  FlashMTPForward &operator=(FlashMTPForward &&) noexcept;

  [[nodiscard]] FlashMTPState createState();
  [[nodiscard]] FlashMTPResult
  forward(FlashMTPState &state, metal::MetalBuffer previousHiddenBF16,
          std::span<const uint32_t> nextTokens,
          FlashMTPLogits logits = FlashMTPLogits::Last);
  // Private append bridge: no token CPU writes, diagnostic reset, submit,
  // readback or logical-length update. Caller holds executionMutex() across
  // build/submission/completion and keeps state/owner alive. Result buffers
  // are borrowed; GPU token IDs must be I64[1], previousHidden BF16[1,10240].
  [[nodiscard]] std::mutex &executionMutex();
  [[nodiscard]] metal::MetalBackend &executionBackend();
  [[nodiscard]] std::vector<metal::MetalBuffer> immutableOperands() const;
  // Private qualification metadata only: CPU can prove cache/scratch bytes
  // remain unchanged when every head-body dispatch was GPU-skipped.
  [[nodiscard]] std::vector<metal::MetalBuffer> stateBuffers(const FlashMTPState &state) const;
  [[nodiscard]] std::vector<metal::MetalBuffer> scratchBuffers() const;
  [[nodiscard]] FlashMTPResult appendPairBody(
      FlashMTPState &state, metal::CommandGraph &graph, uint32_t knownPosition,
      metal::MetalBuffer tokenIDsI64, metal::MetalBuffer previousHiddenBF16,
      metal::MetalBuffer diagnostics, metal::MetalBuffer uniqueGreedyResult);
  void completePairTransaction(FlashMTPState &state, uint64_t originalLength,
                               uint32_t consumedPairs, bool gpuBodyFailure);
  // QSA rollback restores the logical offset. A future append replaces its
  // stale token rows and every newly completed compression block.
  void truncate(FlashMTPState &state, uint64_t retainedLength);
  [[nodiscard]] static uint64_t requestStateBytes(uint32_t capacity);
  [[nodiscard]] uint64_t workspaceBytes() const noexcept;
  // Attention policy is shared by sequential and joint head execution.
  [[nodiscard]] const char *attentionRouteSemantics() const noexcept;
  // Proposal projection choices only; target verification remains external.
  // attentionRouteSemantics also appends an enabled proposal tag for status.
  [[nodiscard]] const char *projectionRouteSemantics() const noexcept;
  [[nodiscard]] static uint64_t attentionWorkspacePlannedBytes(uint32_t maximumRows);
  [[nodiscard]] static uint64_t denseCachePlannedBytes(const FlashWeights &weights);
  [[nodiscard]] std::vector<metal::MetalBuffer> cachedOperandsOnly() const;
  [[nodiscard]] static uint64_t workspacePlannedBytes(uint32_t capacity,
                                                      uint32_t maximumRows = 128);

private:
  friend class FlashBatchMTPForward;
  [[nodiscard]] metal::MetalBackend &batchBackend() const;
  [[nodiscard]] const FlashWeights &batchWeights() const;
  [[nodiscard]] uint32_t batchCapacity() const;
  [[nodiscard]] std::mutex &batchMutex() const;
  [[nodiscard]] const FlashDenseCache *batchDenseCache() const;
  [[nodiscard]] bool batchFuseHC() const;
  [[nodiscard]] bool batchQSAF32() const;
  [[nodiscard]] bool batchQSAMPP() const;
  void batchAddQSA(metal::CommandGraph &graph, const FlashQSAFastInputs &inputs,
      FlashQSAState &state, FlashQSAWorkspace &workspace,
      FlashQSAFastWorkspace &fastWorkspace, uint32_t begin, uint32_t rows);
  [[nodiscard]] bool ownsState(const FlashMTPState &state) const noexcept;
  void batchProject(metal::CommandGraph &graph, const std::string &prefix,
      const metal::MetalBuffer &input, const metal::MetalBuffer &output,
      const metal::MetalBuffer &diagnostics, uint32_t rows,
      uint32_t logicalRows = 0, bool allowQmvProposals = true);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::flash::mtp_gpu_chain_four_candidate
