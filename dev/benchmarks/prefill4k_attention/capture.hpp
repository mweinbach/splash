#pragma once
#include "flash/FlashQSA.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashForward.h"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace prefill4k_attention {
struct Capture final {
  uint32_t layer=0,rows=128,queryOffset=1920,prefixRows=2048;
  splash::metal::MetalBuffer rawProjection,queries,keys,values;
};
inline std::vector<Capture> captures;
inline std::set<uint32_t> captured;
inline constexpr uint64_t kCaptureBytesPerLayer=uint64_t(128)*12288*2+
    uint64_t(128)*6144*2+uint64_t(2048)*512*2*2;
inline constexpr uint64_t kCapturePlannedBytes=3*kCaptureBytesPerLayer;
inline constexpr uint64_t kAllRowsCaptureBytesPerLayer=uint64_t(2048)*12288*2+
    uint64_t(2048)*6144*2+uint64_t(2048)*512*2*2;
inline constexpr uint64_t kAllRowsCapturePlannedBytes=3*kAllRowsCaptureBytesPerLayer;
static_assert(kCapturePlannedBytes==26738688 && kAllRowsCapturePlannedBytes==239075328,
              "private attention capture admission must cover all three layers");
inline bool captureEnabled() {
  const char *path=std::getenv("PREFILL4K_ATTENTION_CAPTURE");
  return path && *path;
}
inline bool captureAllRows() {
  const char *raw=std::getenv("PREFILL4K_ATTENTION_CAPTURE_ALL_ROWS");
  if (!raw || !*raw || std::string(raw)=="0") return false;
  if (std::string(raw)=="1") return true;
  throw std::runtime_error("PREFILL4K_ATTENTION_CAPTURE_ALL_ROWS must be absent/0/1");
}
// The private attribution overlay calls this BEFORE governor.tryReserve and
// target construction. Actual buffers are allocated later when the layer runs.
inline uint64_t capturePlannedBytes() {
  if (!captureEnabled()) return 0;
  return captureAllRows() ? kAllRowsCapturePlannedBytes : kCapturePlannedBytes;
}
inline void copy(splash::metal::CommandGraph &graph,const splash::metal::MetalBuffer &source,
    const splash::metal::MetalBuffer &destination,uint64_t bytes) {
  if (!bytes || bytes%4 || source.sizeBytes()<bytes || destination.sizeBytes()<bytes)
    throw std::runtime_error("attention capture extent invalid");
  const uint64_t words=bytes/4;
  graph.add("flash_forward_copy_words",{source,destination},FlashForwardCopyParams{words},
            {(words+255)/256,1,1});
}
class CaptureScope final {
public:
  CaptureScope(splash::metal::MetalBackend &backend,splash::metal::CommandGraph &graph,
      uint32_t layer,const splash::flash::FlashQSAState &state,
      const splash::metal::MetalBuffer &projection,uint32_t begin,uint32_t rows)
      : backend_(backend),graph_(graph),keysSource_(state.keys),valuesSource_(state.values) {
    if (!captureEnabled() || begin || rows!=2048 || captured.contains(layer) ||
        (layer!=3 && layer!=27 && layer!=47)) return;
    Capture value; value.layer=layer;
    if (captureAllRows()) { value.rows=2048;value.queryOffset=0; }
    auto allocate=[&](uint64_t bytes,const char *label) {
      return backend.allocateBuffer(bytes,splash::metal::BufferStorage::Shared,label);
    };
    value.rawProjection=allocate(uint64_t(value.rows)*12288*2,"private-qsa-raw-gate-projection");
    value.queries=allocate(uint64_t(value.rows)*6144*2,"private-qsa-prepared-query");
    value.keys=allocate(uint64_t(2048)*512*2,"private-qsa-visible-rotated-keys");
    value.values=allocate(uint64_t(2048)*512*2,"private-qsa-visible-values");
    const auto raw=backend.view(projection,uint64_t(value.queryOffset)*12288*2,
                                value.rawProjection.sizeBytes());
    copy(graph_,raw,value.rawProjection,value.rawProjection.sizeBytes());
    index_=captures.size();captures.push_back(std::move(value));captured.insert(layer);
  }
  // Queue a copy after each original window's preparation/attention, before its
  // ordinary prepared-query scratch is reused. The normalization math is never
  // recomputed. Tail mode copies only overlap with rows [1920,2048).
  void preparedWindow(const splash::metal::MetalBuffer &queries,
                      uint32_t offset,uint32_t count) {
    if (!index_) return;
    auto &value=captures.at(*index_);
    if (!count || offset>=value.prefixRows || count>value.prefixRows-offset ||
        queries.sizeBytes()<uint64_t(count)*6144*2)
      throw std::runtime_error("attention capture prepared window extent invalid");
    const uint32_t first=std::max(offset,value.queryOffset);
    const uint32_t end=std::min(offset+count,value.queryOffset+value.rows);
    if (first>=end) return;
    if (first!=value.queryOffset+preparedRows_)
      throw std::runtime_error("attention capture prepared windows omitted/repeated/out of order");
    const uint64_t bytes=uint64_t(end-first)*6144*2;
    const auto source=backend_.view(queries,uint64_t(first-offset)*6144*2,bytes);
    const auto destination=backend_.view(value.queries,uint64_t(first-value.queryOffset)*6144*2,bytes);
    copy(graph_,source,destination,bytes);
    preparedRows_+=end-first;
  }
  // Compatibility for previously generated tail-only overlays. All-rows mode
  // requires the per-window overlay to avoid copying an incomplete query bank.
  void prepared(const splash::metal::MetalBuffer &queries) {
    if (!index_) return;
    const auto &value=captures.at(*index_);
    if (value.rows!=128)
      throw std::runtime_error("all-row attention capture requires regenerated preparedWindow overlay");
    preparedWindow(queries,value.queryOffset,value.rows);
  }
  ~CaptureScope() noexcept(false) {
    if (!index_ || std::uncaught_exceptions()) return;
    auto &value=captures.at(*index_);
    if (preparedRows_!=value.rows)
      throw std::runtime_error("attention capture omitted prepared query rows");
    copy(graph_,keysSource_,value.keys,value.keys.sizeBytes());
    copy(graph_,valuesSource_,value.values,value.values.sizeBytes());
  }
private:
  splash::metal::MetalBackend &backend_;
  splash::metal::CommandGraph &graph_;
  splash::metal::MetalBuffer keysSource_,valuesSource_;
  std::optional<size_t> index_;
  uint32_t preparedRows_=0;
};
void writeCaptures();
} // namespace prefill4k_attention
