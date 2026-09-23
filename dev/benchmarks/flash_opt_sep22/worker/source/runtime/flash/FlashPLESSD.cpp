#include "flash/FlashPLESSD.hpp"

#include "metal/abi/FlashPLESSD.h"

#include <dispatch/dispatch.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384;

uint64_t rounded(uint64_t bytes) {
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}

void validateGeometry(uint32_t lanes, uint32_t rows) {
  if (!lanes || lanes > 4 || !rows || rows > (lanes == 1 ? 8192u : 2048u))
    throw std::invalid_argument("PRIVATE Flash PLE SSD staging requires singleton rows1..8192 or lanes2..4 rows1..2048");
}

std::span<const int64_t> hashArray(const FlashTensor &tensor, uint64_t count,
                                 const char *name) {
  if (tensor.dtype != FlashDType::I64 || tensor.shape != std::vector<uint64_t>{count} ||
      tensor.logicalBytes < count * 8 || !tensor.buffer ||
      tensor.buffer.sizeBytes() < count * 8 || !tensor.buffer.contents())
    throw std::invalid_argument(std::string("Flash PLE SSD requires shared stored ") + name);
  return {static_cast<const int64_t *>(tensor.buffer.contents()), static_cast<size_t>(count)};
}
} // namespace

struct FlashPLESSD::Prefetch final {
  dispatch_queue_t queue = dispatch_queue_create("splash.ple-ssd-prefetch", DISPATCH_QUEUE_SERIAL);
  dispatch_group_t group = dispatch_group_create();
  void wait() const { dispatch_group_wait(group, DISPATCH_TIME_FOREVER); }
  ~Prefetch() {
    wait();
    dispatch_release(group);
    dispatch_release(queue);
  }
};

namespace {
bool plePrefetchEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_PLE_PREFETCH");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
} // namespace

FlashPLESSD::FlashPLESSD(metal::MetalBackend &backend,
                       std::shared_ptr<FlashPLESSDStore> store,
                       const FlashPLEWeights &weights, uint32_t lanes,
                       uint32_t rows)
    : backend_(backend), store_(std::move(store)),
      maximumLanes_(lanes), maximumRows_(rows) {
  validateGeometry(lanes, rows);
  if (!store_ || !weights.diskTableRows || !weights.shards.empty() ||
      weights.diskTableRows != store_->tableRows())
    throw std::invalid_argument("Flash PLE SSD staging requires checked disk-only tables");
  (void)hashArray(weights.multipliers, 3, "multipliers");
  (void)hashArray(weights.headVocabularySizes, 16, "head vocabulary sizes");
  (void)hashArray(weights.headOffsets, 16, "head offsets");
  const uint64_t selections = uint64_t{lanes} * rows * kFlashPLEHeads;
  expectedIDs_ = backend.allocateBuffer(rounded(selections * 8),
      metal::BufferStorage::Shared, "flash-ple-ssd expected canonical IDs");
  rows_ = backend.allocateBuffer(rounded(selections * kFlashPLESSDRowBytes),
      metal::BufferStorage::Shared, "flash-ple-ssd bounded original Q4 rows");
  if (!expectedIDs_.contents() || !rows_.contents())
    throw std::logic_error("Flash PLE SSD staging is not Shared");
  prefetch_ = std::make_unique<Prefetch>();
}

FlashPLESSD::~FlashPLESSD() = default;

void FlashPLESSD::prepare(std::span<const int64_t> tokens,
                         std::span<const int64_t> histories,
                         const FlashPLEWeights &weights, FlashPLEGeometry geometry) {
  preparedLanes_ = preparedRows_ = 0;
  prefetch_->wait();
  validateGeometry(geometry.lanes, geometry.rows);
  if (geometry.lanes > maximumLanes_ || geometry.rows > maximumRows_ ||
      tokens.size() != uint64_t{geometry.lanes} * geometry.rows ||
      histories.size() != uint64_t{geometry.lanes} * 2 ||
      weights.diskTableRows != store_->tableRows() || !weights.shards.empty())
    throw std::invalid_argument("Flash PLE SSD preparation exceeds arena or has changed tables");
  std::vector<int64_t> privateHistory(histories.begin(), histories.end());
  const auto ids = computePLENgramIDs(tokens, privateHistory,
      hashArray(weights.multipliers, 3, "multipliers"),
      hashArray(weights.headVocabularySizes, 16, "head vocabulary sizes"),
      hashArray(weights.headOffsets, 16, "head offsets"), geometry, store_->tableRows());
  store_->lookupRows(ids, {static_cast<uint8_t *>(rows_.contents()),
                           ids.size() * kFlashPLESSDRowBytes});
  std::memcpy(expectedIDs_.contents(), ids.data(), ids.size() * sizeof(int64_t));
  preparedLanes_ = geometry.lanes;
  preparedRows_ = geometry.rows;
}

void FlashPLESSD::prefetch(std::span<const int64_t> tokens,
                          std::span<const int64_t> histories,
                          const FlashPLEWeights &weights, FlashPLEGeometry geometry) {
  if (!plePrefetchEnabled() || tokens.empty() || geometry.lanes != 1 ||
      geometry.rows != tokens.size() || geometry.rows > maximumRows_ ||
      histories.size() != 2 || weights.diskTableRows != store_->tableRows())
    return;
  std::vector<int64_t> privateHistory(histories.begin(), histories.end());
  auto ids = std::make_shared<std::vector<int64_t>>(computePLENgramIDs(tokens, privateHistory,
      hashArray(weights.multipliers, 3, "multipliers"),
      hashArray(weights.headVocabularySizes, 16, "head vocabulary sizes"),
      hashArray(weights.headOffsets, 16, "head offsets"), geometry, store_->tableRows()));
  auto store = store_;
  dispatch_group_async(prefetch_->group, prefetch_->queue, ^{
    try { store->prefetchRows(*ids); } catch (...) {}
  });
}

void FlashPLESSD::addHashGather(metal::CommandGraph &graph,
                              const FlashPLEWeights &weights,
                              metal::MetalBuffer tokens, metal::MetalBuffer history,
                              metal::MetalBuffer ids, metal::MetalBuffer output,
                              metal::MetalBuffer diagnostics,
                              FlashPLEGeometry geometry, bool deferredPreparation) {
  if (weights.tableRows() != store_->tableRows())
    throw std::logic_error("Flash PLE SSD gather has no matching completed preparation");
  if (deferredPreparation) {
    validateGeometry(geometry.lanes, geometry.rows);
    if (geometry.lanes > maximumLanes_ || geometry.rows > maximumRows_)
      throw std::invalid_argument("Flash PLE SSD deferred gather exceeds arena");
  } else if (!preparedRows_ || geometry.lanes != preparedLanes_ || geometry.rows != preparedRows_)
    throw std::logic_error("Flash PLE SSD gather has no matching completed preparation");
  const uint32_t flat = geometry.lanes * geometry.rows;
  const uint64_t selections = uint64_t{flat} * kFlashPLEHeads;
  if (!output || output.sizeBytes() < selections * kFlashPLEHeadWidth * 2 ||
      !diagnostics || diagnostics.sizeBytes() < 4 ||
      weights.sharedScale.dtype != FlashDType::BF16 ||
      weights.sharedScale.shape != std::vector<uint64_t>{1} ||
      weights.sharedScale.logicalBytes < 2 || !weights.sharedScale.buffer ||
      weights.sharedScale.buffer.sizeBytes() < 2)
    throw std::invalid_argument("Flash PLE SSD gather has invalid output or diagnostics");
  const std::array<metal::MetalBuffer, 3> inputs{tokens, history, ids};
  for (const auto &input : inputs)
    if (input.sameView(output) || input.sameView(diagnostics))
      throw std::invalid_argument("Flash PLE SSD input and writable views must be distinct");
  if (output.sameView(diagnostics))
    throw std::invalid_argument("Flash PLE SSD writable views must be distinct");
  for (const auto &arena : {expectedIDs_, rows_})
    if (output.sameView(arena) || diagnostics.sameView(arena))
      throw std::invalid_argument("Flash PLE SSD output must not alias its staging");
  // A preparation belongs to one graph; every later graph must fetch its own
  // incoming token/history IDs, including after partial speculative restores.
  if (!deferredPreparation) preparedLanes_ = preparedRows_ = 0;
  addPLENgramIDs(graph, weights, tokens, history, ids, diagnostics, geometry);
  const FlashPLESSDParams params{flat, kFlashPLEHeads, kFlashPLEHeadWidth,
                                 static_cast<uint32_t>(kFlashPLESSDRowBytes), store_->tableRows()};
  graph.add("flash_ple_ssd_gather", {ids,
      backend_.view(expectedIDs_, 0, selections * 8),
      backend_.view(rows_, 0, selections * kFlashPLESSDRowBytes),
      weights.sharedScale.buffer, output, diagnostics}, params,
      {1, flat, kFlashPLEHeads}, {kFlashPLEHeadWidth, 1, 1});
}

uint64_t FlashPLESSD::allocatedBytes() const noexcept {
  return expectedIDs_.sizeBytes() + rows_.sizeBytes();
}

std::vector<metal::MetalBuffer> FlashPLESSD::scratchBuffers() const {
  return {expectedIDs_, rows_};
}

uint64_t FlashPLESSD::plannedBytes(uint32_t lanes, uint32_t rows) {
  validateGeometry(lanes, rows);
  const uint64_t selections = uint64_t{lanes} * rows * kFlashPLEHeads;
  return rounded(selections * 8) + rounded(selections * kFlashPLESSDRowBytes);
}

} // namespace splash::flash
