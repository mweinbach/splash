#include "flash_gdn_lazy_rollback.hpp"
#include "flash_gdn_lazy_rollback.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace splash::flash;
using namespace splash::metal;
constexpr uint64_t kAlignment = 16384, kGuard = 64;
std::atomic<uint64_t> nextTicket{1};
[[noreturn]] void fail(const char *text) { throw std::invalid_argument(text); }
uint64_t uniqueTicket() {
  auto next = nextTicket.load(std::memory_order_relaxed);
  for (;;) {
    if (next == UINT64_MAX) fail("lazy GDN ticket space exhausted");
    if (nextTicket.compare_exchange_weak(next, next + 1, std::memory_order_relaxed)) return next;
  }
}
uint64_t aligned(uint64_t value) { return (value + kAlignment - 1) & ~(kAlignment - 1); }
void geometry(uint32_t rows, uint32_t lanes) {
  if (!rows || rows > 16 || !lanes || lanes > 4) fail("lazy GDN supports rows1..16 and lanes1..4");
}
void shared(const MetalBuffer &b) {
  if (!b || b.storage() != BufferStorage::Shared || !b.contents()) fail("lazy GDN requires Shared buffers");
}
void disjoint(const MetalBuffer &a, const MetalBuffer &b) {
  shared(a); shared(b);
  const auto x = reinterpret_cast<uintptr_t>(a.contents()), y = reinterpret_cast<uintptr_t>(b.contents());
  if (x <= y ? uint64_t(y - x) < a.sizeBytes() : uint64_t(x - y) < b.sizeBytes())
    fail("lazy GDN writable buffer aliases immutable input, state or tape");
}
void copy(CommandGraph &graph, const MetalBuffer &source, const MetalBuffer &destination,
          uint64_t bytes) {
  graph.add("private_gdn_lazy_copy", {source, destination}, GdnLazyCopyParams{bytes},
      {(bytes + 255) / 256, 1, 1}, {256, 1, 1});
}
uint64_t stride(uint64_t requested, uint64_t tight, uint32_t alignment) {
  if (!requested) return tight;
  if (requested < tight || requested % alignment) fail("lazy GDN invalid padded state stride");
  return requested;
}
} // namespace

struct GdnLazyRollback::Impl final {
  enum Plane : size_t { InitialState, InitialHistory, RawQKV, Mixed, Decay, Beta, Count };
  MetalBackend &backend;
  const uint32_t maxRows, maxLanes;
  std::array<MetalBuffer, Count> allocations, buffers;
  std::array<uint64_t, Count> logical{};
  uint64_t allocated = 0, generation = 0;
  bool inTrial = false;
  uint32_t rows = 0, lanes = 0;
  FlashGDNState state;
  FlashGDNBuffers originals;
  FlashGDNParams params{};

  Impl(MetalBackend &value, uint32_t r, uint32_t l) : backend(value), maxRows(r), maxLanes(l) {
    geometry(r, l);
    if (r == 1) return; // Existing R1 policy has no rollback tape.
    logical = {uint64_t{l} * flashGDNRecurrentLaneBytes(), uint64_t{l} * flashGDNConvolutionLaneBytes(),
        uint64_t{l} * r * 10240 * 2, uint64_t{l} * r * 10240 * 2,
        uint64_t{l} * r * 48 * 4, uint64_t{l} * r * 48 * 2};
    const uint64_t before = backend.memoryStats().allocatedBytes;
    for (size_t i = 0; i < Count; ++i) {
      allocations[i] = backend.allocateBuffer(aligned(logical[i] + kGuard), BufferStorage::Shared,
          "private lazy GDN initial state and saved prework");
      std::memset(allocations[i].contents(), 0xa5, allocations[i].sizeBytes());
      buffers[i] = backend.view(allocations[i], 0, logical[i]);
    }
    const auto after = backend.memoryStats().allocatedBytes;
    if (after < before || after - before > GdnLazyRollback::plannedBytes(r, l))
      fail("lazy GDN allocation exceeds planned arena");
    allocated = after - before;
  }
  MetalBuffer view(Plane p, uint64_t bytes) const { return backend.view(buffers[p], 0, bytes); }
  bool guards() const noexcept {
    if (maxRows == 1) return true;
    for (size_t i = 0; i < Count; ++i) {
      const auto *tail = static_cast<const uint8_t *>(allocations[i].contents()) + logical[i];
      if (!std::all_of(tail, tail + kGuard, [](uint8_t value) { return value == 0xa5; })) return false;
    }
    return true;
  }
};

GdnLazyRollback::GdnLazyRollback(MetalBackend &b, uint32_t r, uint32_t l)
    : impl_(std::make_unique<Impl>(b, r, l)) {}
GdnLazyRollback::~GdnLazyRollback() = default;
GdnLazyRollback::GdnLazyRollback(GdnLazyRollback &&) noexcept = default;
GdnLazyRollback &GdnLazyRollback::operator=(GdnLazyRollback &&) noexcept = default;
uint64_t GdnLazyRollback::plannedBytes(uint32_t rows, uint32_t lanes) {
  geometry(rows, lanes);
  if (rows == 1) return 0;
  uint64_t result = 0;
  for (uint64_t bytes : {uint64_t{lanes} * flashGDNRecurrentLaneBytes(),
      uint64_t{lanes} * flashGDNConvolutionLaneBytes(), uint64_t{lanes} * rows * 10240 * 2,
      uint64_t{lanes} * rows * 10240 * 2, uint64_t{lanes} * rows * 48 * 4,
      uint64_t{lanes} * rows * 48 * 2}) result += aligned(bytes + kGuard);
  return result;
}
bool GdnLazyRollback::pending() const noexcept { return impl_ && impl_->inTrial; }
uint64_t GdnLazyRollback::allocationBytes() const noexcept { return impl_ ? impl_->allocated : 0; }
uint32_t GdnLazyRollback::maximumRows() const noexcept { return impl_ ? impl_->maxRows : 0; }
uint32_t GdnLazyRollback::maximumLanes() const noexcept { return impl_ ? impl_->maxLanes : 0; }
bool GdnLazyRollback::canariesIntact() const noexcept { return impl_ && impl_->guards(); }
FlashGDNBuffers GdnLazyRollback::savedBuffers() const {
  if (!impl_->inTrial || impl_->rows == 1) fail("lazy GDN saved prework requires an active multirow trial");
  auto b = impl_->originals;
  b.qkv = impl_->view(Impl::RawQKV, uint64_t{impl_->lanes} * impl_->rows * 10240 * 2);
  b.mixed = impl_->view(Impl::Mixed, b.qkv.sizeBytes());
  b.decay = impl_->view(Impl::Decay, uint64_t{impl_->lanes} * impl_->rows * 48 * 4);
  b.beta = impl_->view(Impl::Beta, uint64_t{impl_->lanes} * impl_->rows * 48 * 2);
  return b;
}
MetalBuffer GdnLazyRollback::initialRecurrent() const {
  return impl_->rows > 1 ? impl_->view(Impl::InitialState, uint64_t{impl_->lanes} * flashGDNRecurrentLaneBytes()) : MetalBuffer{};
}
MetalBuffer GdnLazyRollback::initialConvolution() const {
  return impl_->rows > 1 ? impl_->view(Impl::InitialHistory, uint64_t{impl_->lanes} * flashGDNConvolutionLaneBytes()) : MetalBuffer{};
}
uint64_t GdnLazyRollback::begin(CommandGraph &graph, const FlashGDNWeights &weights,
    const FlashGDNBuffers &b, const FlashGDNState &state, uint32_t rows, uint32_t lanes, float epsilon) {
  geometry(rows, lanes);
  if (impl_->inTrial || rows > impl_->maxRows || lanes > impl_->maxLanes)
    fail("lazy GDN begin has a pending trial or exceeds its arena");
  CommandGraph validated;
  addGDNFused(validated, weights, b, state, rows, lanes, FlashGDNFusion::PersistentHead512, epsilon);
  const std::array regions{b.qkv, b.z, b.a, b.b, b.mixed, b.decay, b.beta, b.recurrentRows,
      b.output, b.diagnostics, state.convolution, state.recurrent, weights.convolution->buffer,
      weights.aLog->buffer, weights.timeBias->buffer, weights.norm->buffer};
  for (size_t i = 0; i < regions.size(); ++i) {
    shared(regions[i]); (void)impl_->backend.view(regions[i], 0, regions[i].sizeBytes());
    for (size_t j = i + 1; j < regions.size(); ++j) disjoint(regions[i], regions[j]);
    if (rows > 1) for (const auto &tape : impl_->buffers) disjoint(regions[i], tape);
  }
  const uint64_t convStride = stride(state.convolutionLaneStrideBytes, flashGDNConvolutionLaneBytes(), 2);
  const uint64_t recStride = stride(state.recurrentLaneStrideBytes, flashGDNRecurrentLaneBytes(), 4);
  const FlashGDNParams p{rows, lanes, 16, 48, 128, 128, 4, epsilon, convStride, recStride};
  const auto ticket = uniqueTicket();
  if (rows == 1) {
    addGDNFused(graph, weights, b, state, rows, lanes, FlashGDNFusion::PersistentHead512, epsilon);
  } else {
    const auto qkv = impl_->view(Impl::RawQKV, uint64_t{lanes} * rows * 10240 * 2);
    const auto mixed = impl_->view(Impl::Mixed, qkv.sizeBytes());
    const auto decay = impl_->view(Impl::Decay, uint64_t{lanes} * rows * 48 * 4);
    const auto beta = impl_->view(Impl::Beta, uint64_t{lanes} * rows * 48 * 2);
    const auto initialState = impl_->view(Impl::InitialState, uint64_t{lanes} * flashGDNRecurrentLaneBytes());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const auto before = impl_->backend.view(state.convolution, uint64_t{lane} * convStride, flashGDNConvolutionLaneBytes());
      const auto saved = impl_->backend.view(impl_->buffers[Impl::InitialHistory],
          uint64_t{lane} * flashGDNConvolutionLaneBytes(), flashGDNConvolutionLaneBytes());
      copy(graph, before, saved, flashGDNConvolutionLaneBytes());
    }
    copy(graph, b.qkv, qkv, qkv.sizeBytes());
    graph.add("private_gdn_lazy_verify_sg16", {qkv, b.z, b.a, b.b, weights.convolution->buffer,
        weights.aLog->buffer, weights.timeBias->buffer, weights.norm->buffer, state.convolution,
        state.recurrent, mixed, decay, beta, b.recurrentRows, b.output, b.diagnostics, initialState},
        GdnLazyVerifyParams{p, flashGDNRecurrentLaneBytes()}, {48, lanes, 1}, {512, 1, 1});
    graph.add("flash_gdn_convolution_carry", {qkv, state.convolution, b.diagnostics}, p,
        {40, lanes, 1}, {256, 1, 1});
  }
  impl_->rows = rows; impl_->lanes = lanes; impl_->state = state; impl_->originals = b;
  impl_->params = p; impl_->inTrial = true; impl_->generation = ticket;
  return ticket;
}
void GdnLazyRollback::commit(CommandGraph &graph, uint64_t ticket, std::span<const uint32_t> retained) {
  if (!impl_->inTrial || ticket != impl_->generation || retained.size() != impl_->lanes)
    fail("lazy GDN commit has a missing, stale or foreign ticket");
  GdnLazyReplayParams p{impl_->params, {0, 0, 0, 0}, flashGDNRecurrentLaneBytes()};
  bool partial = false;
  for (uint32_t lane = 0; lane < impl_->lanes; ++lane) {
    if (retained[lane] > impl_->rows) fail("lazy GDN retained prefix exceeds incoming rows");
    // Zero means a terminal/dropped lane, matching the target verifier. Its
    // consumed live state is discarded; it is not rolled back or promoted.
    p.retained[lane] = retained[lane] ? retained[lane] : impl_->rows;
    partial |= retained[lane] != 0 && retained[lane] != impl_->rows;
  }
  if (partial && impl_->rows > 1) {
    const auto b = savedBuffers();
    graph.add("private_gdn_lazy_replay_sg16", {b.mixed, b.decay, b.beta, initialRecurrent(),
        impl_->state.recurrent, b.diagnostics}, p, {48, impl_->lanes, 1}, {512, 1, 1});
    graph.add("private_gdn_lazy_restore_convolution", {initialConvolution(), b.qkv,
        impl_->state.convolution, b.diagnostics}, p, {40, impl_->lanes, 1}, {256, 1, 1});
  }
  impl_->inTrial = false;
}
void GdnLazyRollback::abort(uint64_t ticket) {
  if (!impl_->inTrial || ticket != impl_->generation) fail("lazy GDN abort has a stale or missing ticket");
  // Live state has consumed the trial and must be discarded by its owner.
  // No state is promoted, restored, or advertised healthy on abort.
  impl_->inTrial = false;
}
