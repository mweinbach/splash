#pragma once
#include "flash/FlashGDNFused.hpp"

#include <memory>
#include <span>

// Private synchronous GPU-graph prototype. No model projections are replayed.
// The caller completes each begin graph before commit and each commit graph
// before another begin. An aborted trial's live state is terminal/discarded.
class GdnLazyRollback final {
public:
  GdnLazyRollback(splash::metal::MetalBackend &backend, uint32_t maximumRows,
                  uint32_t maximumLanes);
  ~GdnLazyRollback();
  GdnLazyRollback(const GdnLazyRollback &) = delete;
  GdnLazyRollback &operator=(const GdnLazyRollback &) = delete;
  GdnLazyRollback(GdnLazyRollback &&) noexcept;
  GdnLazyRollback &operator=(GdnLazyRollback &&) noexcept;

  [[nodiscard]] uint64_t begin(splash::metal::CommandGraph &graph,
      const splash::flash::FlashGDNWeights &weights,
      const splash::flash::FlashGDNBuffers &buffers,
      const splash::flash::FlashGDNState &state, uint32_t rows, uint32_t lanes,
      float epsilon = 1e-6f);
  void commit(splash::metal::CommandGraph &graph, uint64_t ticket,
              std::span<const uint32_t> retained);
  void abort(uint64_t ticket);
  [[nodiscard]] bool pending() const noexcept;
  [[nodiscard]] uint64_t allocationBytes() const noexcept;
  [[nodiscard]] static uint64_t plannedBytes(uint32_t maximumRows, uint32_t maximumLanes);
  [[nodiscard]] uint32_t maximumRows() const noexcept;
  [[nodiscard]] uint32_t maximumLanes() const noexcept;
  [[nodiscard]] bool canariesIntact() const noexcept;
  [[nodiscard]] splash::flash::FlashGDNBuffers savedBuffers() const;
  [[nodiscard]] splash::metal::MetalBuffer initialRecurrent() const;
  [[nodiscard]] splash::metal::MetalBuffer initialConvolution() const;
  [[nodiscard]] constexpr uint64_t initialRecurrentLaneStrideBytes() const noexcept {
    return splash::flash::flashGDNRecurrentLaneBytes();
  }
  [[nodiscard]] constexpr uint64_t initialConvolutionLaneStrideBytes() const noexcept {
    return splash::flash::flashGDNConvolutionLaneBytes();
  }
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
