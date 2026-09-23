#pragma once
#include "flash/FlashMTP.hpp"
#include <array>
#include <memory>

namespace splash::flash {
// Private standalone teacher operation. Shares coefficients, owner and QSA
// workspace with the authoritative128-row trained head; owns only scratch.
class FlashMTPTeacherBulkForward final {
public:
  static constexpr uint32_t maximumRows = 2048, rowQuantum = 128;
  static constexpr uint64_t plannedBytes = 310181888ULL;
  explicit FlashMTPTeacherBulkForward(FlashMTPForward &head);
  ~FlashMTPTeacherBulkForward();
  FlashMTPTeacherBulkForward(const FlashMTPTeacherBulkForward &) = delete;
  FlashMTPTeacherBulkForward &operator=(const FlashMTPTeacherBulkForward &) = delete;
  [[nodiscard]] metal::CommandTiming primeTeacherCache(
      FlashMTPState &state, metal::MetalBuffer actualTargetPremixerHidden,
      std::span<const uint32_t> nextTokens);
  [[nodiscard]] uint64_t workspaceBytes() const noexcept;
  // Standalone oracle only: complete physical scratch views for redzones and
  // alias/no-write guards. This header is private benchmark machinery.
  [[nodiscard]] std::array<metal::MetalBuffer,18> oracleWorkspaceBuffers() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace splash::flash
