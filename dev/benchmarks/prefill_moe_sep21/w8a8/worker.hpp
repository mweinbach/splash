#pragma once

// Private whole-worker NUMERICAL prefill experiment. Persisted I8 weights and
// F32 weight scales remain immutable. Only four activation workspace buffers
// are allocated here; audits and model loading belong to the separate oracle.
#include "flash/FlashMoEBlocked.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace splash::flash::prefill_w8a8_sep21 {

inline constexpr uint64_t kAlignment=16384,kGuardBytes=64;
inline constexpr uint32_t kSelections=10,kPaddingRows=63;
inline constexpr uint32_t kMaximumRows=8192,kEligibleRows=2048;

[[nodiscard]] inline uint32_t parseMode(const char *value) {
  if (!value || std::string_view(value)=="0") return 0;
  if (std::string_view(value)=="1") return 1;
  if (std::string_view(value)=="2") return 2;
  throw std::invalid_argument("SPLASH_FLASH_PREFILL_MOE_W8A8 must be 0, 1 or 2");
}

// The worker must call this during initial policy validation, before loading
// weights or allocating scratch. Later environment changes cannot alter it.
[[nodiscard]] inline uint32_t mode() {
  static const uint32_t frozen=parseMode(std::getenv("SPLASH_FLASH_PREFILL_MOE_W8A8"));
  return frozen;
}

[[nodiscard]] constexpr bool eligibleFor(uint32_t rows,FlashMoEBlockedTile tile,
                                        bool verification,uint32_t requestedMode) noexcept {
  return (requestedMode==1 || requestedMode==2) && !verification &&
      rows==kEligibleRows && tile==FlashMoEBlockedTile::M32N64;
}
[[nodiscard]] inline bool eligible(uint32_t rows,FlashMoEBlockedTile tile,bool verification) {
  return eligibleFor(rows,tile,verification,mode());
}

// These strings describe the changed arithmetic and exact routing scope.
// Empty inactive strings preserve baseline model/cache numerical identity.
[[nodiscard]] constexpr const char *policyFor(uint32_t requestedMode) noexcept {
  if (requestedMode==1)
    return "prefill-moe-w8a8-sep21-main-nonverification-r2048-native-m32-canonical-buckets-"
        "bf16-a-perrow-maxabs-f32scale-maxdiv127-zero-scale1-rne-i8-clamp127-"
        "nonfinite-zero-diagnostic-i8b-original-f32-late-weightscale-whole-k-i32-mpp-"
        "m32n64-sg4-f32-intdot-times-ascaletimes-wscale-no-contract-no-reassociate-"
        "original-bf16-swiglu-bf16-canonical-combine-globalpad63-two-gpu-quantizers-v1";
  if (requestedMode==2)
    return "prefill-moe-w8a8-sep21-main-nonverification-r2048-native-m32-canonical-buckets-"
        "bf16-a-perrow-maxabs-f32scale-maxdiv127-zero-scale1-rne-i8-clamp127-"
        "nonfinite-zero-diagnostic-i8b-original-f32-late-weightscale-whole-k-i32-mpp-"
        "m32n64-sg2-f32-intdot-times-ascaletimes-wscale-no-contract-no-reassociate-"
        "original-bf16-swiglu-bf16-canonical-combine-globalpad63-two-gpu-quantizers-v1";
  return "";
}
[[nodiscard]] constexpr const char *markerFor(uint32_t requestedMode) noexcept {
  if (requestedMode==1)
    return ";prefill-moe-w8a8-sep21-main-nonverification-r2048-native-m32-canonical-buckets-"
        "bf16-a-perrow-maxabs-f32scale-maxdiv127-zero-scale1-rne-i8-clamp127-"
        "nonfinite-zero-diagnostic-i8b-original-f32-late-weightscale-whole-k-i32-mpp-"
        "m32n64-sg4-f32-intdot-times-ascaletimes-wscale-no-contract-no-reassociate-"
        "original-bf16-swiglu-bf16-canonical-combine-globalpad63-two-gpu-quantizers-v1";
  if (requestedMode==2)
    return ";prefill-moe-w8a8-sep21-main-nonverification-r2048-native-m32-canonical-buckets-"
        "bf16-a-perrow-maxabs-f32scale-maxdiv127-zero-scale1-rne-i8-clamp127-"
        "nonfinite-zero-diagnostic-i8b-original-f32-late-weightscale-whole-k-i32-mpp-"
        "m32n64-sg2-f32-intdot-times-ascaletimes-wscale-no-contract-no-reassociate-"
        "original-bf16-swiglu-bf16-canonical-combine-globalpad63-two-gpu-quantizers-v1";
  return "";
}
[[nodiscard]] inline const char *policy() {return policyFor(mode());}
[[nodiscard]] inline const char *marker() {return markerFor(mode());}

[[nodiscard]] constexpr const char *producerNameFor(bool gate,uint32_t requestedMode) noexcept {
  if (requestedMode==1)
    return gate ? "private_w8a8_worker_gate_up_m32_n64_sg4" :
        "private_w8a8_worker_down_scatter_m32_n64_sg4";
  if (requestedMode==2)
    return gate ? "private_w8a8_worker_gate_up_m32_n64_sg2" :
        "private_w8a8_worker_down_scatter_m32_n64_sg2";
  return "";
}
[[nodiscard]] constexpr uint32_t producerThreadsFor(uint32_t requestedMode) noexcept {
  return requestedMode==1 ? 128 : requestedMode==2 ? 64 : 0;
}
[[nodiscard]] inline const char *producerName(bool gate) {return producerNameFor(gate,mode());}
[[nodiscard]] inline uint32_t producerThreads() {return producerThreadsFor(mode());}

struct QuantParams final {uint32_t rows,width,reserved0,reserved1;};
static_assert(sizeof(QuantParams)==16,"W8A8 row quantizer has uint4 ABI");

struct Counters final {
  bool enabled=false;
  uint32_t mode=0,maxRows=0;
  uint64_t plannedBytes=0,logicalBytes=0,actualAllocatedBytes=0;
  uint64_t gateCalls=0,gateRows=0,downCalls=0,downRows=0,quantDispatches=0;
};

class Workspace final {
 public:
  // Plain logical views bind into graph commands and immutable-alias checks.
  metal::MetalBuffer qGate,qDown,gateScales,downScales;

  [[nodiscard]] static std::array<uint64_t,4> sizes(uint32_t maxRows) {
    if (!maxRows || maxRows>kMaximumRows)
      throw std::invalid_argument("W8A8 workspace maximum rows must be 1..8192");
    const uint64_t padded=uint64_t{maxRows}*kSelections+kPaddingRows;
    return {padded*2560,padded*640,padded*sizeof(float),padded*sizeof(float)};
  }
  [[nodiscard]] static uint64_t logicalBytes(uint32_t maxRows) {
    uint64_t total=0;for (uint64_t bytes:sizes(maxRows)) total+=bytes;return total;
  }
  [[nodiscard]] static uint64_t plannedBytes(uint32_t maxRows) {
    uint64_t total=0;
    for (uint64_t bytes:sizes(maxRows)) total+=rounded(bytes+kGuardBytes);
    return total;
  }

  // Caller independently reserves plannedBytes(maxRows) BEFORE this constructor.
  // Nonmovable atomics are intentional: optional<Workspace>::emplace owns it.
  Workspace(metal::MetalBackend &backend,uint32_t maxRows)
      :maxRows_(maxRows),mode_(prefill_w8a8_sep21::mode()),logical_(sizes(maxRows)),
       planned_(plannedBytes(maxRows)),logicalTotal_(logicalBytes(maxRows)) {
    if (!mode_) throw std::invalid_argument("inactive W8A8 policy must not allocate workspace");
    const uint64_t before=backend.memoryStats().allocatedBytes;
    std::array<metal::MetalBuffer,4> views;
    constexpr std::array<const char *,4> labels{
        "prefill W8A8 governed guarded I8 gate input",
        "prefill W8A8 governed guarded I8 down input",
        "prefill W8A8 governed guarded F32 gate row scales",
        "prefill W8A8 governed guarded F32 down row scales"};
    for (uint32_t index=0;index<allocations_.size();++index) {
      const uint64_t allocated=rounded(logical_[index]+kGuardBytes);
      allocations_[index]=backend.allocateBuffer(allocated,metal::BufferStorage::Shared,labels[index]);
      if (!allocations_[index].contents())
        throw std::runtime_error("W8A8 activation workspace is not addressable");
      views[index]=backend.view(allocations_[index],0,logical_[index]);
      std::memset(allocations_[index].contents(),0xa5,logical_[index]);
      std::memset(static_cast<uint8_t *>(allocations_[index].contents())+logical_[index],
          0x5a,allocated-logical_[index]);
    }
    qGate=views[0];qDown=views[1];gateScales=views[2];downScales=views[3];
    const uint64_t after=backend.memoryStats().allocatedBytes;
    if (after<before || after-before>planned_)
      throw std::runtime_error("W8A8 activation workspace exceeded pre-constructor admission");
    actual_=after-before;
    validate(maxRows_);
  }
  Workspace(const Workspace &)=delete;
  Workspace &operator=(const Workspace &)=delete;
  Workspace(Workspace &&)=delete;
  Workspace &operator=(Workspace &&)=delete;

  void validate(uint32_t rows) const {
    if (!rows || rows>maxRows_ || maxRows_>kMaximumRows)
      throw std::invalid_argument("W8A8 requested rows exceed governed workspace capacity");
    const auto expected=sizes(maxRows_);
    const auto views=buffers();
    for (uint32_t index=0;index<views.size();++index) {
      requireBytes(views[index],expected[index]);
      if (!allocations_[index].contents() ||
          allocations_[index].sizeBytes()<expected[index]+kGuardBytes)
        throw std::invalid_argument("W8A8 activation guard geometry differs");
      for (uint32_t other=index+1;other<views.size();++other)
        if (overlaps(views[index],views[other]))
          throw std::invalid_argument("W8A8 activation workspace views overlap");
    }
  }
  [[nodiscard]] std::array<metal::MetalBuffer,4> buffers() const {
    return {qGate,qDown,gateScales,downScales};
  }
  [[nodiscard]] bool canaries() const {
    for (uint32_t index=0;index<allocations_.size();++index) {
      if (!allocations_[index].contents() ||
          allocations_[index].sizeBytes()<logical_[index]+kGuardBytes) return false;
      const auto *begin=static_cast<const uint8_t *>(allocations_[index].contents())+logical_[index];
      if (!std::all_of(begin,begin+allocations_[index].sizeBytes()-logical_[index],
          [](uint8_t value){return value==0x5a;})) return false;
    }
    return true;
  }

  void addQuantGate(metal::CommandGraph &graph,metal::MetalBuffer source,
                    metal::MetalBuffer diagnostics,uint32_t rows) const {
    addQuant(graph,std::move(source),std::move(diagnostics),rows,true);
  }
  void addQuantDown(metal::CommandGraph &graph,metal::MetalBuffer source,
                    metal::MetalBuffer diagnostics,uint32_t rows) const {
    addQuant(graph,std::move(source),std::move(diagnostics),rows,false);
  }
  // These are graph-construction counts, not GPU completion measurements.
  [[nodiscard]] Counters counters() const noexcept {
    return {mode_!=0,mode_,maxRows_,planned_,logicalTotal_,actual_,
        gateCalls_.load(std::memory_order_relaxed),gateRows_.load(std::memory_order_relaxed),
        downCalls_.load(std::memory_order_relaxed),downRows_.load(std::memory_order_relaxed),
        quantDispatches_.load(std::memory_order_relaxed)};
  }

 private:
  uint32_t maxRows_,mode_;
  std::array<uint64_t,4> logical_;
  uint64_t planned_,logicalTotal_,actual_=0;
  std::array<metal::MetalBuffer,4> allocations_;
  mutable std::atomic<uint64_t> gateCalls_{0},gateRows_{0},downCalls_{0},downRows_{0},quantDispatches_{0};

  [[nodiscard]] static constexpr uint64_t rounded(uint64_t bytes) noexcept {
    return (bytes+kAlignment-1)&~(kAlignment-1);
  }
  static void requireBytes(const metal::MetalBuffer &buffer,uint64_t bytes) {
    if (!buffer || !buffer.contents() || buffer.sizeBytes()<bytes)
      throw std::invalid_argument("W8A8 insufficient addressable activation/scale/source/diagnostic bytes");
  }
  [[nodiscard]] static bool overlaps(const metal::MetalBuffer &left,
                                    const metal::MetalBuffer &right) noexcept {
    if (left.sameView(right)) return true;
    const auto a=reinterpret_cast<uintptr_t>(left.contents());
    const auto b=reinterpret_cast<uintptr_t>(right.contents());
    if (!a || !b) return false;
    return a<=b ? uint64_t(b-a)<left.sizeBytes() : uint64_t(a-b)<right.sizeBytes();
  }
  void addQuant(metal::CommandGraph &graph,metal::MetalBuffer source,
                metal::MetalBuffer diagnostics,uint32_t rows,bool gate) const {
    validate(rows);
    if (!eligibleFor(rows,FlashMoEBlockedTile::M32N64,false,mode_))
      throw std::invalid_argument("W8A8 quantizer is restricted to active main R2048 prefill");
    const uint32_t padded=rows*kSelections+kPaddingRows,width=gate ? 2560 : 640;
    requireBytes(source,uint64_t{padded}*width*2);requireBytes(diagnostics,4);
    if (overlaps(source,diagnostics))
      throw std::invalid_argument("W8A8 source overlaps diagnostics");
    for (const auto &buffer:buffers())
      if (overlaps(buffer,source) || overlaps(buffer,diagnostics))
        throw std::invalid_argument("W8A8 quantizer writable workspace overlaps source/diagnostics");
    graph.add(gate ? "prefill_moe_sep21_w8a8_quantize_gate_t256" :
            "prefill_moe_sep21_w8a8_quantize_down_t128",
        {source,gate ? qGate : qDown,gate ? gateScales : downScales,diagnostics},
        QuantParams{padded,width,0,0},{padded,1,1},{gate ? 256u : 128u,1,1});
    if (gate) {
      gateCalls_.fetch_add(1,std::memory_order_relaxed);gateRows_.fetch_add(rows,std::memory_order_relaxed);
    } else {
      downCalls_.fetch_add(1,std::memory_order_relaxed);downRows_.fetch_add(rows,std::memory_order_relaxed);
    }
    quantDispatches_.fetch_add(1,std::memory_order_relaxed);
  }
};

} // namespace splash::flash::prefill_w8a8_sep21
