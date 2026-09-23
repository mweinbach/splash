#include "dev/benchmarks/expert_r5_verify_worker_sep22/policy.hpp"
#include "dev/benchmarks/raw_q4_verify_worker_sep22/policy.hpp"
#include "dev/benchmarks/guard_hc_fast_composite_sep22/policy.hpp"
#include "dev/benchmarks/hc_pad_verify_worker_sep22/worker_bridge.hpp"
#include "dev/benchmarks/expert_r4_preflight_bundle_sep22/policy.hpp"
#include "dev/benchmarks/expert_r4_compact_verify_worker_sep22/bridge.hpp"
#include "dev/benchmarks/gdn_ab_merge_sep21/bridge.hpp"
#include "dev/benchmarks/prefill_qsa_twopass_sep21/worker_bridge.hpp"
#include "dev/benchmarks/prefill_hc_inject_norm_sep21/worker_bridge.hpp"
#include "dev/benchmarks/dense_w8a8_sep21/worker_bridge.hpp"
#include "dev/benchmarks/gdn_chunk_sep21/worker_bridge.hpp"
#include "dev/benchmarks/adaptive_expert_tail_sep21/combined/worker_bridge.hpp"
#include "dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy.hpp"
#include "dev/benchmarks/moe_pointwise_sep21/bridge.hpp"
// Private all-row Full512 target executor overlay v1.
#include "bulk.hpp"
#include "flash/OptQmv.hpp"
#include "flash/FlashForward.hpp"

#include "flash/FlashAffine.hpp"
#include "flash/FlashBatchPrefill.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashSharedExpertFused.hpp"
#include "flash/FlashDenseSmallRows.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "flash/FlashInt8Head.hpp"
#include "flash/FlashExpertDenseCache.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include "flash/FlashInt8ExpertStoreMetadata.hpp"
#include "flash/FlashExpertCachePlan.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashGDN.hpp"
#include "flash/FlashGDNFused.hpp"
#include "flash/FlashGDNLazyRollback.hpp"
#include "flash/FlashGDNStaged.hpp"
#include "flash/FlashHC.hpp"
#include "flash/FlashHCFused.hpp"
#include "flash/FlashMoE.hpp"
#include "flash/FlashMTPWindow.hpp"
#include "flash/FlashPrefillDenseTiles.hpp"
#include "flash/FlashPLE.hpp"
#include "flash/FlashPLEFused.hpp"
#include "flash/FlashPLESSD.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashQSA.hpp"
#include "flash/FlashQSAFast.hpp"
#include "flash/FlashQSAMPP.hpp"
#include "flash/FlashRequestStateInternal.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashForward.h"

#include <array>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::flash {
namespace {
constexpr uint64_t kAlignment = 16384;
constexpr uint32_t kWidth = 2560;
constexpr uint32_t kHyper = 10240;
constexpr uint32_t kSelections = 10;

bool fusionEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value) return false;
  if (std::string_view(value) == "1") return true;
  if (std::string_view(value) == "0") return false;
  throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

// Private experiment: validate dependencies before admission/construction.
bool privateBulkQSAPrefillEnabled() {
  if (!fusionEnabled("SPLASH_FLASH_QSA_BULK_PREFILL")) return false;
  if (!fusionEnabled("SPLASH_FLASH_QSA_F32") ||
      !fusionEnabled("SPLASH_FLASH_QSA_MPP") ||
      !fusionEnabled("SPLASH_FLASH_QSA_ROW_TILES") ||
      !qsaOnlineMPPRowTilesEnabled())
    throw std::invalid_argument("SPLASH_FLASH_QSA_BULK_PREFILL requires QSA_F32=1, QSA_MPP=1 and QSA_ROW_TILES=1");
  return true;
}

bool privateBulkQSAPrefillSG8Enabled(bool bulkEnabled) {
  if (!fusionEnabled("SPLASH_FLASH_QSA_BULK_PREFILL_SG8")) return false;
  if (!bulkEnabled)
    throw std::invalid_argument("SPLASH_FLASH_QSA_BULK_PREFILL_SG8 requires SPLASH_FLASH_QSA_BULK_PREFILL=1 and its dependencies");
  return true;
}

std::optional<FlashExpertCachePlan> hotExpertPlan(const FlashWeights &weights) {
  const char *path = std::getenv("SPLASH_FLASH_HOT_EXPERT_PLAN");
  if (!path) return std::nullopt;
  if (!*path) throw std::invalid_argument("SPLASH_FLASH_HOT_EXPERT_PLAN must name a local plan");
  return loadFlashExpertCachePlan(path, weights.sourceIdentity());
}
std::optional<std::filesystem::path> int8ExpertDirectory() {
  const char *path = std::getenv("SPLASH_FLASH_INT8_EXPERT_STORE");
  if (!path || std::string_view(path) == "0") return std::nullopt;
  if (!*path) throw std::invalid_argument("SPLASH_FLASH_INT8_EXPERT_STORE must name a saved local store");
  if (std::getenv("SPLASH_FLASH_HOT_EXPERT_PLAN"))
    throw std::invalid_argument("choose one selected-expert operand store: INT8 or BF16 hot cache");
  return std::filesystem::path(path);
}

uint64_t checkedMultiply(uint64_t left, uint64_t right) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left)
    throw std::overflow_error("Flash forward byte count overflows");
  return left * right;
}
uint64_t roundAllocation(uint64_t bytes) {
  if (!bytes || bytes > std::numeric_limits<uint64_t>::max() - (kAlignment - 1))
    throw std::overflow_error("Flash forward allocation extent is invalid");
  return (bytes + kAlignment - 1) & ~(kAlignment - 1);
}
void validateCapacity(uint32_t capacity) {
  if (!capacity || capacity > 262144)
    throw std::invalid_argument("Flash request capacity must be 1..262144");
}
void clear(const metal::MetalBuffer &buffer) {
  if (!buffer || buffer.storage() != metal::BufferStorage::Shared || !buffer.contents())
    throw std::logic_error("Flash cold state requires CPU-accessible shared storage");
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
}

enum class Scratch : size_t {
  TokenIDs, Embedding, Hyper, HCNormalized, HCUp, HCDown, HCRawInjection,
  HCInjectionWeights, Mixed, Branch, QProjection, KProjection, VProjection,
  IndexProjection, GDNMixed, GDNZ, GDNA, GDNB, GDNDecay, GDNBeta,
  GDNRecurrentRows, AttentionOutput, Router, ExpertIDs, RouteWeights,
  ExpertGate, ExpertUp, ExpertIntermediate, ExpertDown, SharedGate, SharedUp,
  SharedIntermediate, SharedDown, SharedGateLogit, NgramIDs, PLEEmbedding,
  PLEKey, PLEValue, PLENormalizedKeys, PLENormalizedQueries, PLEGated,
  PLENormalizedConvolution, PLEOutput, HeadLogits, Diagnostics, Count,
};
constexpr size_t kScratchCount = static_cast<size_t>(Scratch::Count);

} // namespace

struct FlashForward::Impl final {
  metal::MetalBackend &backend;
  const FlashWeights &weights;
  const FlashDescriptor descriptor;
  const uint32_t capacity;
  const uint32_t maximumRows;
  const uint32_t maximumLogitRows;
  const uint32_t maximumVerifyRows;
  const bool fuseHC = fusionEnabled("SPLASH_FLASH_FUSE_HC");
  const bool hcPadVerifyR4 = hc_pad_verify_sep22::requested();
  const bool hcUpF32 = flashHCUpF32MPPEnabled();
  FlashHCUpEncodedCounters hcUpCounters;
  const bool fuseGDN = fusionEnabled("SPLASH_FLASH_FUSE_GDN");
  const bool stagedGDN = fusionEnabled("SPLASH_FLASH_GDN_STAGED");
  const bool cacheDense = fusionEnabled("SPLASH_FLASH_DENSE_CACHE");
  const bool fuseSharedExpert = flashSharedExpertFusedFlag(std::getenv("SPLASH_FLASH_SHARED_EXPERT_FUSED"));
  const bool blockMoE = fusionEnabled("SPLASH_FLASH_BLOCKED_MOE");
  const bool allRowsInt8Target = fusionEnabled("SPLASH_FLASH_ALLROWS_FULL512_TARGET");
  const bool onlineQSA = fusionEnabled("SPLASH_FLASH_QSA_F32");
  const bool mppQSA = fusionEnabled("SPLASH_FLASH_QSA_MPP");
  const bool bulkQSAPrefill = privateBulkQSAPrefillEnabled();
  const bool bulkQSAPrefillSG8 = privateBulkQSAPrefillSG8Enabled(bulkQSAPrefill);
  const bool smallDense = fusionEnabled("SPLASH_FLASH_DENSE_SMALL_ROWS");
  const bool cacheFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE");
  const bool mergeGDNAB = gdn_ab_merge::requested();
  const bool codeHead = fusionEnabled("SPLASH_FLASH_INT8_HEAD");
  const bool selectiveFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE");
  const bool rawQ4Rowpair = raw_q4_verify_sep22::requested();
  const bool captureRoutes = fusionEnabled("SPLASH_FLASH_CAPTURE_EXPERT_IDS");
  const bool fusePLELookup = fusionEnabled("SPLASH_FLASH_PLE_LOOKUP_FUSED");
  const bool gpuGreedy = fusionEnabled("SPLASH_FLASH_GPU_GREEDY");
  const bool lazyGDN = flashGDNLazyRollbackEnabled();
  std::shared_ptr<const uint8_t> owner = std::make_shared<const uint8_t>(0);
  std::array<metal::MetalBuffer, kScratchCount> scratch;
  FlashQSAWorkspace qsaWorkspace;
  FlashQSAFastWorkspace qsaFastWorkspace;
  std::optional<prefill4k::BulkExactWorkspace> bulkQSAWorkspace;
  std::optional<prefill_qsa_twopass_sep21::Workspace> twoPassQSAWorkspace;
  FlashPLEWeights pleWeights;
  std::unique_ptr<FlashPLEFused> pleLookup;
  std::unique_ptr<FlashPLESSD> pleSSD;
  FlashGreedyGPUWorkspace greedyWorkspace;
  metal::MetalBuffer greedyResults;
  std::unique_ptr<FlashDenseCache> denseCache;
  const bool denseW8Requested = dense_w8a8_sep21::requested();
  std::unique_ptr<dense_w8a8_sep21::Cache> denseW8Cache;
  std::unique_ptr<dense_w8a8_sep21::Workspace> denseW8Workspace;
  std::unique_ptr<FlashDenseSmallRowsWorkspace> smallDenseWorkspace;
  std::unique_ptr<FlashFloatDenseCache> floatDenseCache;
  std::unique_ptr<FlashFloatDenseSmallRowsWorkspace> floatDenseWorkspace;
  std::unique_ptr<FlashInt8Head> int8Head;
  std::array<std::unique_ptr<FlashExpertDenseCache>, 48> expertCaches;
  std::unique_ptr<FlashInt8ExpertStore> int8ExpertStore;
  std::string expertPlanIdentity;
  metal::MetalBuffer capturedRoutes;
  uint32_t capturedRows = 0;
  FlashMoEBlockedScratch blockedScratch;
  metal::MetalBuffer verifyRecurrent;
  metal::MetalBuffer verifyConvolution;
  std::array<std::unique_ptr<FlashGDNLazyRollback>, 48> lazyGDNRecords;
  std::array<uint64_t, 48> lazyGDNTickets{};
  metal::MetalBuffer beforePLEHistory;
  metal::MetalBuffer beforePLEConvolution;
  metal::MetalBuffer retainedCount;
  metal::MetalBuffer beforeGDNConvolution;
  std::array<uint32_t, 48> gdnSlots{};
  std::weak_ptr<const uint8_t> pendingIdentity;
  uint64_t pendingBegin = 0;
  uint32_t pendingRows = 0;
  uint64_t workspaceBytes = 0;
  std::mutex mutex;
  // Advanced by every trunk graph build/submission/commit; a prepared window
  // is only submitted if nothing ran since it was built.
  uint64_t trunkSerial = 0;
  std::vector<metal::MetalBuffer> repackedCodes;
  std::shared_ptr<FlashRequestStatePool> statePool = std::make_shared<FlashRequestStatePool>();

  Impl(metal::MetalBackend &value, const FlashWeights &model, uint32_t context,
       uint32_t rows, uint32_t verifyRows)
      : backend(value), weights(model), descriptor(model.descriptor()),
        capacity(context), maximumRows(rows), maximumLogitRows(std::min(rows, uint32_t{128})),
        maximumVerifyRows(verifyRows),
        pleWeights(FlashPLEWeights::fromWeights(model)) {
    descriptor.validate();
    compact_native_r5_verify_sep22::validateConstruction(maximumVerifyRows,allRowsInt8Target,blockMoE);
    raw_q4_verify_sep22::validateDependencies();
    raw_q4_verify_sep22::validateInventory(weights);
    guard_hc_fast_composite_sep22::validate(compact_native_r4_verify_sep22::requested(),
        compact_r4_preflight_sep22::requested(),hc_pad_verify_sep22::requested());
    hc_pad_verify_sep22::validateDependencies(hcUpF32,fuseHC,cacheFloat);
    if(hcPadVerifyR4 && maximumRows<8)
      throw std::invalid_argument("HC padding requires an admitted eight-row HCDown scratch view");
    if (!descriptor.pleParametersLoaded)
      throw std::invalid_argument("Flash forward requires checked stored PLE parameters");
    if (mppQSA && !onlineQSA)
      throw std::invalid_argument("SPLASH_FLASH_QSA_MPP requires SPLASH_FLASH_QSA_F32=1");
    if (selectiveFloat && (!cacheFloat || !flashAffineFastEnabled()))
      throw std::invalid_argument("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE requires FLOAT_DENSE_CACHE=1 and QMV_F32=1");
    if (fuseSharedExpert && !cacheDense)
      throw std::invalid_argument("SPLASH_FLASH_SHARED_EXPERT_FUSED requires SPLASH_FLASH_DENSE_CACHE=1");
    validateCapacity(capacity);
    if (!maximumRows || maximumRows > kFlashGDNMaximumRows || maximumRows > kFlashMoEMaxRows)
      throw std::invalid_argument("PRIVATE Flash forward maximum rows must be 1..8192");
    if (maximumVerifyRows > kFlashSingletonMaximumVerifyRows || maximumVerifyRows > maximumRows)
      throw std::invalid_argument("Flash target verification rows must be 0..16 and fit scratch");
    // Bounded metadata preflight precedes every backend workspace/cache allocation.
    if (allRowsInt8Target) {
      if (!blockMoE)
        throw std::invalid_argument("private all-row Full512 target requires blocked MoE");
      const auto directory = int8ExpertDirectory();
      if (!directory)
        throw std::invalid_argument("private all-row Full512 target requires a saved INT8 Store");
      const auto metadata = loadFlashInt8ExpertStoreMetadata(*directory, weights.sourceIdentity(),
          weights.manifestFingerprint(), weights.normConvention());
      for (uint32_t layer = 0; layer < 48; ++layer) {
        const auto &ids = metadata.layers[layer].selectedIDs;
        if (ids.size() != 512)
          throw std::invalid_argument("private all-row target requires all 512 experts in every layer");
        for (uint32_t expert = 0; expert < 512; ++expert)
          if (ids[expert] != expert)
            throw std::invalid_argument("private all-row target requires canonical complete expert IDs");
      }
    }
    const uint64_t before = backend.memoryStats().allocatedBytes;
    if (weights.pleSSDStreamingEnabled())
      pleSSD = std::make_unique<FlashPLESSD>(backend, weights.pleSSDStore(),
          pleWeights, 1, maximumRows);
    else if (fusePLELookup && backend.supportsArgumentBuffersTier2())
      pleLookup = std::make_unique<FlashPLEFused>(backend, pleWeights);
    if (gpuGreedy) {
      const auto greedyRows = std::min(maximumLogitRows, uint32_t{kFlashGreedyGPUMaximumRows});
      greedyWorkspace = allocateGreedyGPUWorkspace(backend, greedyRows, descriptor.vocabularySize);
      greedyResults = backend.allocateBuffer(uint64_t{greedyRows} * sizeof(FlashGreedyGPURowResult),
          metal::BufferStorage::Shared, "flash-forward compact GPU greedy records");
    }
    if (cacheFloat) {
      if (smallDense) throw std::invalid_argument("choose one small-row coefficient policy: BF16 or F32");
      const auto prefixes = FlashFloatDenseCache::defaultPrefixes(weights, !codeHead);
      floatDenseCache = std::make_unique<FlashFloatDenseCache>(backend, weights, prefixes);
      floatDenseWorkspace = std::make_unique<FlashFloatDenseSmallRowsWorkspace>(backend);
      raw_q4_verify_sep22::validateCachePresence(weights,*floatDenseCache);
    }
    if (codeHead)
      int8Head = std::make_unique<FlashInt8Head>(backend, weights,
          floatDenseWorkspace ? floatDenseWorkspace->paddedInput() : metal::MetalBuffer{});
    if (const auto plan = hotExpertPlan(weights)) {
      if (!blockMoE) throw std::invalid_argument("hot-expert cache requires blocked prefill enabled");
      expertPlanIdentity = plan->planSha256;
      for (uint32_t layer = 0; layer < descriptor.layers; ++layer)
        expertCaches[layer] = std::make_unique<FlashExpertDenseCache>(backend, weights,
            "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp",
            plan->selectedExperts[layer]);
    }
    if (const auto directory = int8ExpertDirectory()) {
      if (!blockMoE) throw std::invalid_argument("saved INT8 experts require blocked prefill enabled");
      int8ExpertStore = std::make_unique<FlashInt8ExpertStore>(backend, weights, *directory);
    }
    if (cacheDense) {
      const auto prefixes = FlashDenseCache::defaultPrefixes(weights, true);
      denseCache = std::make_unique<FlashDenseCache>(backend, weights, prefixes);
      if (smallDense)
        smallDenseWorkspace = std::make_unique<FlashDenseSmallRowsWorkspace>(backend);
    }
    if(hcPadVerifyR4) {
      const auto checkHC=[&](const std::string &prefix,bool injection) {
        const auto &d=weights.projection(prefix+".input_mix_weight_down");
        const auto &u=weights.projection(prefix+".input_mix_weight_up");
        const auto *i=injection ? &weights.projection(prefix+".block_inject_weight") : nullptr;
        if(!floatDenseCache || !floatDenseWorkspace || !floatDenseCache->contains(prefix+".input_mix_weight_up") ||
            !flashHCUpF32MPPGeometry(prefix+".input_mix_weight_up",4,10240,320) ||
            !supportsHCFused(d,u,i,{4,kWidth,4,static_cast<float>(descriptor.normEpsilon)}))
          throw std::invalid_argument("HC padding requires all97 original fused-down/cached-F32-up bindings");
        const auto &t=floatDenseCache->tensor(prefix+".input_mix_weight_up");
        if(t.dtype!=FlashDType::F32 || t.shape!=std::vector<uint64_t>{10240,320} || t.logicalBytes!=13107200ULL)
          throw std::invalid_argument("HC padding original cached up view differs");
      };
      for(uint32_t layer=0;layer<48;++layer) {
        const auto prefix="language_model.model.layers."+std::to_string(layer);
        checkHC(prefix+".attn_hyper_connection",true);checkHC(prefix+".mlp_hyper_connection",true);
      }
      checkHC("language_model.model.hyper_connection_mixer",false);
    }
    // Resource policy is fixed for both selector values. Source BF16 operands
    // are inspected/fitted only here, during authorized model startup.
    if (dense_w8a8_sep21::requiresCache(maximumRows)) {
      if (!denseCache) throw std::invalid_argument("private dense W8A8 worker requires DENSE_CACHE=1 at maximumRows>=2048");
      denseW8Cache = std::make_unique<dense_w8a8_sep21::Cache>(backend,weights,*denseCache);
      denseW8Workspace = std::make_unique<dense_w8a8_sep21::Workspace>(backend);
    }
    if (captureRoutes)
      capturedRoutes = backend.allocateBuffer(
          roundAllocation(uint64_t{descriptor.layers} * maximumRows * kSelections * 8),
          metal::BufferStorage::Shared, "flash-forward-expert-route-capture");
    const auto allocate = [&](Scratch slot, uint64_t bytes, const char *label) {
      scratch[static_cast<size_t>(slot)] = backend.allocateBuffer(
          roundAllocation(bytes), metal::BufferStorage::Shared, label);
    };
    const auto bf = [&](Scratch slot, uint64_t width, const char *label) {
      allocate(slot, checkedMultiply(checkedMultiply(maximumRows, width), 2), label);
    };
    allocate(Scratch::TokenIDs, uint64_t{maximumRows} * 8, "flash-forward-token-ids");
    bf(Scratch::Embedding, kWidth, "flash-forward-embedding");
    bf(Scratch::Hyper, kHyper, "flash-forward-hyper-state");
    bf(Scratch::HCNormalized, kHyper, "flash-forward-hc-normalized");
    bf(Scratch::HCUp, kHyper, "flash-forward-hc-up");
    bf(Scratch::HCDown, 320, "flash-forward-hc-down");
    bf(Scratch::HCRawInjection, 4, "flash-forward-hc-raw-injection");
    bf(Scratch::HCInjectionWeights, 4, "flash-forward-hc-injection-weights");
    bf(Scratch::Mixed, kWidth, "flash-forward-mixed-input");
    bf(Scratch::Branch, kWidth, "flash-forward-branch");
    bf(Scratch::QProjection, 12288, "flash-forward-q-projection");
    bf(Scratch::KProjection, 512, "flash-forward-k-projection");
    bf(Scratch::VProjection, 512, "flash-forward-v-projection");
    bf(Scratch::IndexProjection, 640, "flash-forward-index-projection");
    bf(Scratch::GDNMixed, 10240, "flash-forward-gdn-convolved");
    bf(Scratch::GDNZ, 6144, "flash-forward-gdn-z");
    bf(Scratch::GDNA, 48, "flash-forward-gdn-a");
    bf(Scratch::GDNB, 48, "flash-forward-gdn-b");
    allocate(Scratch::GDNDecay, uint64_t{maximumRows} * 48 * 4, "flash-forward-gdn-decay");
    bf(Scratch::GDNBeta, 48, "flash-forward-gdn-beta");
    bf(Scratch::GDNRecurrentRows, 6144, "flash-forward-gdn-recurrent-rows");
    bf(Scratch::AttentionOutput, 6144, "flash-forward-attention-output");
    bf(Scratch::Router, 512, "flash-forward-router-logits");
    allocate(Scratch::ExpertIDs, uint64_t{maximumRows} * kSelections * 8, "flash-forward-expert-ids");
    bf(Scratch::RouteWeights, kSelections, "flash-forward-route-weights");
    bf(Scratch::ExpertGate, kSelections * 640, "flash-forward-expert-gate");
    bf(Scratch::ExpertUp, kSelections * 640, "flash-forward-expert-up");
    bf(Scratch::ExpertIntermediate, kSelections * 640, "flash-forward-expert-intermediate");
    bf(Scratch::ExpertDown, kSelections * kWidth, "flash-forward-expert-down");
    bf(Scratch::SharedGate, 640, "flash-forward-shared-gate");
    bf(Scratch::SharedUp, 640, "flash-forward-shared-up");
    bf(Scratch::SharedIntermediate, 640, "flash-forward-shared-intermediate");
    bf(Scratch::SharedDown, kWidth, "flash-forward-shared-down");
    bf(Scratch::SharedGateLogit, 1, "flash-forward-shared-gate-logit");
    allocate(Scratch::NgramIDs, uint64_t{maximumRows} * 16 * 8, "flash-forward-ngram-ids");
    bf(Scratch::PLEEmbedding, kWidth, "flash-forward-ple-embedding");
    bf(Scratch::PLEKey, kHyper, "flash-forward-ple-key");
    bf(Scratch::PLEValue, kWidth, "flash-forward-ple-value");
    bf(Scratch::PLENormalizedKeys, kHyper, "flash-forward-ple-normalized-keys");
    bf(Scratch::PLENormalizedQueries, kHyper, "flash-forward-ple-normalized-queries");
    bf(Scratch::PLEGated, kHyper, "flash-forward-ple-gated-values");
    bf(Scratch::PLENormalizedConvolution, kHyper, "flash-forward-ple-normalized-convolution");
    bf(Scratch::PLEOutput, kHyper, "flash-forward-ple-output");
    allocate(Scratch::HeadLogits, uint64_t{maximumLogitRows} * descriptor.vocabularySize * 2,
              "flash-forward-head-logits");
    allocate(Scratch::Diagnostics, 4, "flash-forward-diagnostics");
    qsaWorkspace = allocateQSAWorkspace(backend, std::min(maximumRows, uint32_t{128}), capacity);
    if (onlineQSA)
      qsaFastWorkspace = mppQSA
          ? allocateQSAOnlineMPPWorkspace(backend, std::min(maximumRows, uint32_t{128}), 32)
          : allocateQSAFastWorkspace(backend, std::min(maximumRows, uint32_t{128}), 4);
    if (blockMoE && (allRowsInt8Target || maximumRows >= 256))
      blockedScratch = allocateMoEBlockedScratch(backend, maximumRows, kSelections);
    if (bulkQSAPrefill && maximumRows >= 2048) {
      bulkQSAWorkspace.emplace(prefill4k::allocateBulkExactWorkspace(backend));
      const auto &bulk = *bulkQSAWorkspace;
      const uint64_t bulkPlaneBytes = bulk.prepared.queries.sizeBytes() +
          bulk.prepared.indexQueries.sizeBytes() + bulk.prepared.selectedBlocks.sizeBytes() +
          bulk.partials.partitionStatistics.sizeBytes() + bulk.partials.partitionValues.sizeBytes();
      if (bulkPlaneBytes != prefill4k::bulkExactPlannedBytes())
        throw std::logic_error("private bulk QSA five-plane allocation/admission mismatch");
    }
    if (prefill_qsa_twopass_sep21::plannedExtraBytes(maximumRows,prefill_qsa_twopass_sep21::requested())) {
      if (!bulkQSAWorkspace) throw std::logic_error("private QSA needs admitted original bulk prefix");
      twoPassQSAWorkspace.emplace(backend,bulkQSAWorkspace->prepared);
    }
    uint32_t gdnSlot = 0;
    for (uint32_t layer = 0; layer < descriptor.layers; ++layer)
      if (descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet)
        gdnSlots[layer] = gdnSlot++;
    if (maximumVerifyRows > 1) {
      const uint64_t prefixes = maximumVerifyRows - 1;
      if (lazyGDN) {
        for (uint32_t layer = 0; layer < descriptor.layers; ++layer)
          if (descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet)
            lazyGDNRecords[layer] = std::make_unique<FlashGDNLazyRollback>(backend, maximumVerifyRows, 1);
      } else {
        verifyRecurrent = backend.allocateBuffer(36 * prefixes * roundAllocation(flashGDNRecurrentLaneBytes()),
            metal::BufferStorage::Shared, "flash-verify-recurrent-prefixes");
        verifyConvolution = backend.allocateBuffer(36 * prefixes * roundAllocation(flashGDNConvolutionLaneBytes()),
            metal::BufferStorage::Shared, "flash-verify-convolution-prefixes");
      }
      beforePLEHistory = backend.allocateBuffer(roundAllocation(2 * sizeof(int64_t)),
          metal::BufferStorage::Shared, "flash-verify-ple-before-history");
      beforePLEConvolution = backend.allocateBuffer(roundAllocation(uint64_t{9} * kHyper * 2),
          metal::BufferStorage::Shared, "flash-verify-ple-before-convolution");
      retainedCount = backend.allocateBuffer(roundAllocation(sizeof(uint32_t)),
          metal::BufferStorage::Shared, "flash-verify-retained-count");
      if (fuseGDN && !lazyGDN)
        beforeGDNConvolution = backend.allocateBuffer(roundAllocation(flashGDNConvolutionLaneBytes()),
            metal::BufferStorage::Shared, "flash-verify-gdn-before-convolution");
    }
    const uint64_t after = backend.memoryStats().allocatedBytes;
    if (after < before) throw std::logic_error("Flash workspace allocation ledger regressed");
    workspaceBytes = after - before;
    if (bulkQSAWorkspace && workspaceBytes < prefill4k::bulkExactPlannedBytes())
      throw std::logic_error("private bulk QSA planes omitted from allocation ledger");
  }

  metal::MetalBuffer view(Scratch slot, uint64_t bytes) {
    return backend.view(scratch[static_cast<size_t>(slot)], 0, bytes);
  }
  metal::MetalBuffer bf(Scratch slot, uint32_t rows, uint32_t width) {
    return view(slot, uint64_t{rows} * width * 2);
  }

  void copy(metal::CommandGraph &graph, const metal::MetalBuffer &input,
            const metal::MetalBuffer &output, uint64_t bytes) {
    if (!bytes || bytes % 4 || input.sizeBytes() < bytes || output.sizeBytes() < bytes || input.sameView(output))
      throw std::invalid_argument("Flash verification raw-copy extent/alias is invalid");
    const uint64_t words = bytes / 4;
    graph.add("flash_forward_copy_words", {input, output}, FlashForwardCopyParams{words},
              {(words + 255) / 256, 1, 1});
  }

  metal::MetalBuffer prefixTape(bool recurrence, uint32_t layer, uint32_t row) {
    const uint64_t bytes = recurrence ? flashGDNRecurrentLaneBytes() : flashGDNConvolutionLaneBytes();
    const uint64_t cell = uint64_t{gdnSlots[layer]} * (maximumVerifyRows - 1) + row;
    return backend.view(recurrence ? verifyRecurrent : verifyConvolution,
                         cell * roundAllocation(bytes), bytes);
  }

  void abortLazyRecords() noexcept {
    for (uint32_t layer = 0; layer < descriptor.layers; ++layer) {
      auto &record = lazyGDNRecords[layer];
      if (record && record->pending()) {
        try { record->abort(lazyGDNTickets[layer]); } catch (...) {}
      }
      lazyGDNTickets[layer] = 0;
    }
  }
  void finishPending() noexcept {
    abortLazyRecords(); pendingIdentity.reset(); pendingRows = 0; pendingBegin = 0;
  }

  void project(metal::CommandGraph &graph, const std::string &prefix,
               const metal::MetalBuffer &input, const metal::MetalBuffer &output,
               const metal::MetalBuffer &diagnostics, uint32_t rows,
               bool verification = false, bool singletonMain = false) {
    if (rows <= opt::kQmvMaximumSplitRows && prefix != "language_model.lm_head" &&
        opt::addQmv(graph, input, weights.projection(prefix), output, rows)) return;
    if (rows > opt::kQmvMaximumSplitRows && prefix != "language_model.lm_head" &&
        opt::addQmvTall(graph, input, weights.projection(prefix), output, rows)) return;
    // Short prefill windows: chunked matrix-unit/SIMD kernels on the original
    // codes instead of the BF16 dense cache's two-tile plus per-row tail.
    if (rows > opt::kQmvMaximumSplitRows && rows <= opt::prefillQmvRows() &&
        opt::prefillQmvEnabled() && prefix != "language_model.lm_head" &&
        opt::addQmv(graph, input, weights.projection(prefix), output, rows)) return;
    if (denseW8Requested && singletonMain && !verification && rows == 2048 &&
        denseW8Cache && denseW8Workspace && denseW8Cache->contains(prefix) &&
        dense_w8a8_sep21::addProjection(graph,*denseW8Cache,prefix,input,output,
            diagnostics,rows,verification,*denseW8Workspace,4)) return;
    if (int8Head && prefix == "language_model.lm_head") {
      if (rows >= 2 && rows <= 16) {
        int8Head->addProjection(graph, input, output, diagnostics, rows);
        return;
      }
      if (rows == 1) {
        addAffine(graph, input, weights.projection(prefix), output, diagnostics, rows);
        return;
      }
    }
    if (floatDenseCache && rows >= 2 && rows <= 16 && floatDenseCache->contains(prefix)) {
      if (selectiveFloat) {
        const auto &projection = weights.projection(prefix);
        const auto tile = flashFloatDenseSmallRowsPolicy(prefix, rows,
            projection.outputSize, projection.inputSize, projection.bits, projection.groupSize);
        if (tile)
          floatDenseCache->addSmallRows(graph, prefix, input, output, diagnostics, rows,
              *floatDenseWorkspace, *tile);
        else {
          if (rawQ4Rowpair && raw_q4_verify_sep22::selected(prefix,rows,verification,singletonMain,projection))
            raw_q4_verify_sep22::add(graph,prefix,input,projection,output,diagnostics,rows,verification,singletonMain);
          else
            addAffine(graph, input, projection, output, diagnostics, rows);
        }
        return;
      }
      floatDenseCache->addSmallRows(graph, prefix, input, output, diagnostics, rows,
          *floatDenseWorkspace, rows > 8 ? FlashFloatDenseSmallRowsTile::M16N64
                                       : FlashFloatDenseSmallRowsTile::M8N64);
    } else if (denseCache && smallDenseWorkspace && rows < 16 && denseCache->contains(prefix)) {
      const auto &projection = weights.projection(prefix);
      addDenseBF16SmallRows(backend, graph, input, denseCache->tensor(prefix), output,
          diagnostics, rows, *smallDenseWorkspace,
          projection.outputSize >= 1024 ? FlashDenseSmallRowsTile::M8N128
                                       : FlashDenseSmallRowsTile::M8N64);
    } else if (denseCache && rows >= 16 && denseCache->contains(prefix)) {
      const auto &projection = weights.projection(prefix);
      const auto tile = rows >= 128 && projection.outputSize >= 1024
          ? FlashAffineMPPTile::M32N128 : FlashAffineMPPTile::M16N64;
      denseCache->addProjection(graph, prefix, input, output, diagnostics, rows, tile);
    } else {
      addAffine(graph, input, weights.projection(prefix), output, diagnostics, rows);
    }
  }

  void hc(metal::CommandGraph &graph, const std::string &prefix,
          uint32_t rows, bool injection, bool normalizedReady = false, bool verification = false) {
    hc_pad_verify_sep22::requireFrozen();
    const FlashHCGeometry geometry{rows, kWidth, 4, static_cast<float>(descriptor.normEpsilon)};
    const auto normalized = bf(Scratch::HCNormalized, rows, kHyper);
    const bool padVerify = hcPadVerifyR4 && hc_pad_verify_sep22::eligible(verification,rows,maximumRows) &&
        hcUpF32 && fuseHC && floatDenseCache && floatDenseWorkspace &&
        flashHCUpF32MPPGeometry(prefix+".input_mix_weight_up",rows,10240,320) &&
        floatDenseCache->contains(prefix+".input_mix_weight_up");
    const auto down = bf(Scratch::HCDown, padVerify ? 8 : rows, 320);
    const auto up = bf(Scratch::HCUp, rows, kHyper);
    const std::string norm = prefix + ".hc_norm.weight";
    if (!normalizedReady)
      addHCGroupedNorm(graph, bf(Scratch::Hyper, rows, kHyper), weights.tensor(norm),
                       normalized, geometry, weights.normConvention(norm));
    const auto &downProjection = weights.projection(prefix + ".input_mix_weight_down");
    const auto &upProjection = weights.projection(prefix + ".input_mix_weight_up");
    const auto *injectionProjection = injection
        ? &weights.projection(prefix + ".block_inject_weight") : nullptr;
    if ((rows <= opt::kQmvMaximumSplitRows ||
         (rows <= opt::prefillQmvRows() && opt::prefillQmvEnabled())) &&
        opt::addHCDown(graph, normalized, downProjection, injectionProjection,
            bf(Scratch::HCDown, rows, 320),
            injection ? bf(Scratch::HCInjectionWeights, rows, 4) : metal::MetalBuffer{}, rows)) {
      if (!opt::addHCUpMix(graph, normalized, bf(Scratch::HCDown, rows, 320), upProjection,
              bf(Scratch::Mixed, rows, kWidth), rows))
        throw std::logic_error("optimized HC up-mix rejected a supported down projection");
      return;
    }
    if (fuseHC && supportsHCFused(downProjection, upProjection, injectionProjection, geometry)) {
      const auto diagnostics = view(Scratch::Diagnostics, 4);
      if(padVerify) {
        hc_pad_verify_sep22::addChain(backend,graph,normalized,downProjection,injectionProjection,down,
            injection ? bf(Scratch::HCInjectionWeights,rows,4) : metal::MetalBuffer{},
            floatDenseCache->tensor(prefix+".input_mix_weight_up"),bf(Scratch::Mixed,rows,kWidth),diagnostics,
            *floatDenseWorkspace,geometry,verification,maximumRows);
        hcUpCounters.recordAttempt(rows);hcUpCounters.recordEligible(rows);hcUpCounters.recordCachedEncoded(rows);
        return;
      }
      addHCFusedDown(graph, normalized, downProjection, injectionProjection, down,
          injection ? bf(Scratch::HCInjectionWeights, rows, 4) : metal::MetalBuffer{},
          diagnostics, geometry);
      if (!cachedHCUp(graph, prefix + ".input_mix_weight_up", normalized, down,
          bf(Scratch::Mixed, rows, kWidth), diagnostics, rows))
        addHCFusedUpMix(graph, normalized, down, upProjection,
            bf(Scratch::Mixed, rows, kWidth), diagnostics, geometry);
      return;
    }
    project(graph, prefix + ".input_mix_weight_down", normalized, down,
        view(Scratch::Diagnostics, 4), rows);
    const FlashForwardActivationParams activation{rows, 320, 4};
    graph.add("flash_forward_hc_silu", {down, down, view(Scratch::Diagnostics, 4)},
              activation, {uint64_t{rows} * 320 / 256 + (uint64_t{rows} * 320 % 256 != 0), 1, 1});
    project(graph, prefix + ".input_mix_weight_up", down, up,
        view(Scratch::Diagnostics, 4), rows);
    if (injection) {
      const auto raw = bf(Scratch::HCRawInjection, rows, 4);
      project(graph, prefix + ".block_inject_weight", normalized, raw,
          view(Scratch::Diagnostics, 4), rows);
      addHCMixWithInjection(graph, normalized, up, raw, bf(Scratch::Mixed, rows, kWidth),
                             bf(Scratch::HCInjectionWeights, rows, 4), geometry);
    } else {
      addHCMix(graph, normalized, up, bf(Scratch::Mixed, rows, kWidth), geometry);
    }
  }

  bool cachedHCUp(metal::CommandGraph &graph, const std::string &prefix,
      const metal::MetalBuffer &normalized, const metal::MetalBuffer &activated,
      const metal::MetalBuffer &mixed, const metal::MetalBuffer &diagnostics,
      uint32_t rows) {
    hcUpCounters.recordAttempt(rows);
    if (!flashHCUpF32MPPGeometry(prefix, rows, 10240, 320)) {
      ++hcUpCounters.skippedUnsupportedGeometry;
      return false;
    }
    hcUpCounters.recordEligible(rows);
    if (!hcUpF32 || !fuseHC || !floatDenseCache || !floatDenseWorkspace) {
      ++hcUpCounters.skippedDependenciesOff;
      return false;
    }
    if (!floatDenseCache->contains(prefix)) {
      ++hcUpCounters.skippedMissingOperand;
      return false;
    }
    addHCFusedUpMixF32Cache(backend, graph, normalized, activated,
        floatDenseCache->tensor(prefix), mixed, diagnostics,
        {rows, kWidth, 4, static_cast<float>(descriptor.normEpsilon)}, *floatDenseWorkspace);
    hcUpCounters.recordCachedEncoded(rows);
    return true;
  }
};

FlashRequestState::FlashRequestState() = default;
FlashRequestState::~FlashRequestState() {
  if (!impl_ || impl_->poisoned || impl_->pendingVerification) return;
  const auto pool = impl_->pool.lock();
  if (!pool || !opt::statePoolEnabled()) return;
  // Expire this request's identity now: no pending verification or batch
  // member can refer to the pooled storage afterwards.
  impl_->identity = std::make_shared<const uint8_t>(0);
  std::lock_guard lock(pool->mutex);
  if (pool->states.size() < FlashRequestStatePool::kMaximumStates)
    pool->states.push_back(std::move(impl_));
}
FlashRequestState::FlashRequestState(FlashRequestState &&) noexcept = default;
FlashRequestState &FlashRequestState::operator=(FlashRequestState &&) noexcept = default;
uint64_t FlashRequestState::logicalLength() const noexcept { return impl_ ? impl_->length : 0; }
uint32_t FlashRequestState::capacity() const noexcept { return impl_ ? impl_->capacity : 0; }
bool FlashRequestState::poisoned() const noexcept { return !impl_ || impl_->poisoned; }

namespace {
// Dense projections whose 5/6-bit codes get an 8-bit copy for the matrix-unit
// GEMV (batched verification windows and short prefill).
std::vector<std::string> repackPrefixes(const FlashWeights &weights) {
  std::vector<std::string> result;
  if (!opt::repackEnabled()) return result;
  const auto &descriptor = weights.descriptor();
  for (uint32_t layer = 0; layer < descriptor.layers; ++layer) {
    const std::string base = "language_model.model.layers." + std::to_string(layer);
    const bool gdn = descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet;
    const std::array<const char *, 3> names = gdn
        ? std::array<const char *, 3>{".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.out_proj"}
        : std::array<const char *, 3>{".self_attn.q_proj", ".self_attn.o_proj", nullptr};
    for (const char *name : names) {
      if (!name) continue;
      if (opt::repackEligible(weights.projection(base + name))) result.push_back(base + name);
    }
  }
  return result;
}
} // namespace

uint64_t FlashForward::repackPlannedBytes(const FlashWeights &weights) {
  uint64_t total = 0;
  for (const auto &prefix : repackPrefixes(weights)) {
    const auto &projection = weights.projection(prefix);
    total += roundAllocation(uint64_t{projection.outputSize} * projection.inputSize);
  }
  return total;
}

FlashForward::FlashForward(metal::MetalBackend &backend, const FlashWeights &weights,
                           uint32_t capacity, uint32_t maximumRows, uint32_t maximumVerifyRows)
    : impl_(std::make_unique<Impl>(backend, weights, capacity, maximumRows, maximumVerifyRows)) {
  for (const auto &prefix : repackPrefixes(weights)) {
    const auto &projection = weights.projection(prefix);
    auto codes = opt::repackCodes(backend, projection);
    opt::registerRepackedCodes(projection.weights->buffer.contents(), codes);
    impl_->repackedCodes.push_back(std::move(codes));
  }
}
FlashForward::~FlashForward() = default;
FlashForward::FlashForward(FlashForward &&) noexcept = default;
FlashForward &FlashForward::operator=(FlashForward &&) noexcept = default;

uint64_t FlashForward::requestStateBytes(uint32_t capacity) {
  validateCapacity(capacity);
  uint64_t bytes = 36 * (roundAllocation(flashGDNConvolutionLaneBytes()) +
                          roundAllocation(flashGDNRecurrentLaneBytes()));
  const uint64_t blocks = (uint64_t{capacity} + 3) / 4;
  const uint64_t qsa = 2 * roundAllocation(uint64_t{capacity} * 512 * 2) +
      roundAllocation(uint64_t{capacity} * 128 * 2) + roundAllocation(blocks * 128 * 2) +
      roundAllocation(uint64_t{capacity} * 8);
  bytes += 12 * qsa;
  bytes += roundAllocation(2 * sizeof(int64_t));
  bytes += roundAllocation(uint64_t{9} * kHyper * 2);
  return bytes;
}

uint64_t FlashForward::workspaceBytes() const noexcept { return impl_ ? impl_->workspaceBytes : 0; }
uint64_t FlashForward::pleSSDStagingBytes() const noexcept {
  return impl_ && impl_->pleSSD ? impl_->pleSSD->allocatedBytes() : 0;
}
uint64_t FlashForward::verificationGDNStorageBytes() const noexcept {
  if (!impl_) return 0;
  uint64_t bytes = impl_->verifyRecurrent.sizeBytes() + impl_->verifyConvolution.sizeBytes();
  for (const auto &record : impl_->lazyGDNRecords) if (record) bytes += record->allocationBytes();
  return bytes;
}
bool FlashForward::lazyGDNRollbackEnabled() const noexcept { return impl_ && impl_->lazyGDN; }
FlashGDNLazyRollbackCounters FlashForward::lazyGDNRollbackCounters() const noexcept {
  FlashGDNLazyRollbackCounters result;
  if (impl_) for (const auto &record : impl_->lazyGDNRecords)
    if (record) result.add(record->counters());
  return result;
}

uint64_t FlashForward::workspacePlannedBytes(uint32_t capacity, uint32_t maximumRows,
                                            uint32_t maximumVerifyRows) {
  validateCapacity(capacity);
  if (!maximumRows || maximumRows > kFlashGDNMaximumRows || maximumVerifyRows > kFlashSingletonMaximumVerifyRows || maximumVerifyRows > maximumRows)
    throw std::invalid_argument("Flash workspace admission geometry is unsupported");
  // The39 BF16 row buffers retain their current allocation geometry. Logits
  // and QSA have separately capped row extents below.
  constexpr std::array<uint32_t, 39> widths{
      kWidth, kHyper, kHyper, kHyper, 320, 4, 4, kWidth, kWidth, 12288,
      512, 512, 640, 10240, 6144, 48, 48, 48, 6144, 6144, 512, kSelections,
      kSelections * 640, kSelections * 640, kSelections * 640, kSelections * kWidth,
      640, 640, 640, kWidth, 1, kWidth, kHyper, kWidth, kHyper, kHyper,
      kHyper, kHyper, kHyper};
  uint64_t total = 0;
  for (uint32_t width : widths) total += roundAllocation(uint64_t{maximumRows} * width * 2);
  for (uint64_t extent : std::array<uint64_t, 5>{uint64_t{maximumRows} * 8,
      uint64_t{maximumRows} * 48 * 4, uint64_t{maximumRows} * kSelections * 8,
      uint64_t{maximumRows} * 16 * 8, 4}) total += roundAllocation(extent);
  const uint64_t cappedRows = std::min(maximumRows, uint32_t{128});
  total += roundAllocation(cappedRows * 248320 * 2);
  const uint64_t blocks = (uint64_t{capacity} + 3) / 4;
  for (uint64_t width : std::array<uint64_t, 6>{6144 * 2, 512 * 2, blocks * 4,
      512 * 4, uint64_t{24} * kFlashQSATokenWidth * 4, uint64_t{24} * kFlashQSATokenWidth * 2})
    total += roundAllocation(cappedRows * width);
  // Admit the largest optional online scratch before GPU construction. Scalar
  // F32 attention uses the same strided layout with only four active partitions.
  total += roundAllocation(cappedRows * 24 * 32 * 2 * 4);
  total += roundAllocation(cappedRows * 24 * 32 * 256 * 4);
  total += verificationWorkspaceBytes(maximumVerifyRows);
  // Conservative allowance for optional padding and diagnostic route capture.
  total += roundAllocation(uint64_t{16} * 32768 * 2);
  total += roundAllocation(uint64_t{48} * maximumRows * kSelections * 8);
  if (maximumVerifyRows > 1 && !flashGDNLazyRollbackEnabled())
    total += roundAllocation(flashGDNConvolutionLaneBytes());
  if (flashPLESSDStreamingValue(std::getenv("SPLASH_FLASH_PLE_SSD_STREAMING")))
    total += FlashPLESSD::plannedBytes(1, maximumRows);
  else if (fusionEnabled("SPLASH_FLASH_PLE_LOOKUP_FUSED"))
    total += roundAllocation(kFlashPLEFusedMaximumArgumentBytes);
  if (fusionEnabled("SPLASH_FLASH_GPU_GREEDY"))
    total += greedyGPUWorkspacePlannedBytes(
        std::min(maximumRows, uint32_t{kFlashGreedyGPUMaximumRows}), 248320);
  if (fusionEnabled("SPLASH_FLASH_ALLROWS_FULL512_TARGET") && maximumRows < 256)
    total += flashMoEBlockedWorkspacePlannedBytes(maximumRows, kSelections, kAlignment);
  const bool privateBulkPrefillEnabled = privateBulkQSAPrefillEnabled();
  (void)privateBulkQSAPrefillSG8Enabled(privateBulkPrefillEnabled);
  if (privateBulkPrefillEnabled && maximumRows >= 2048)
    total += prefill4k::bulkExactPlannedBytes();
  total += prefill_qsa_twopass_sep21::plannedExtraBytes(maximumRows,prefill_qsa_twopass_sep21::requested());
  if (dense_w8a8_sep21::requiresCache(maximumRows))
    total += dense_w8a8_sep21::Workspace::plannedBytes();
  return total;
}

std::string FlashForward::kernelRoutes() const {
  if (!impl_) throw std::logic_error("Flash forward is not initialized");
  return std::string(flashAffineSemantics()) +
      (impl_->mergeGDNAB ? std::string(";") + gdn_ab_merge::kSemantics : "") +
      std::string(prefill_qsa_twopass_sep21::selectionMarker(prefill_qsa_twopass_sep21::requested())) +
      std::string(prefill_hc_inject_norm_sep21::selectionMarker(prefill_hc_inject_norm_sep21::requested())) +
      std::string(dense_w8a8_sep21::selectionMarker(impl_->denseW8Requested)) +
      std::string(dense_w8a8_sep21::kCacheMarker) +
      (impl_->denseW8Cache ? impl_->denseW8Cache->identitySha256() : "none-maxrows-below2048") +
      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +
      (impl_->hcPadVerifyR4 ? hc_pad_verify_sep22::marker : "") +
      compact_native_r4_verify_sep22::implementationMarker() +
      compact_r4_preflight_sep22::marker() +
      guard_hc_fast_composite_sep22::marker() +
      raw_q4_verify_sep22::marker() +
      compact_native_r5_verify_sep22::marker() +
      std::string(fixed_sg2_prefill_sep21::marker()) +
      std::string(adaptive_expert_tail_sg2k128_sep21::markerFor(
          fixed_sg2_prefill_sep21::selection(),adaptive_expert_tail_sg2k128_sep21::requested())) +
      (impl_->bulkQSAWorkspace ? ";private-qsa-bulk-prefill-begin0-r2048-w128-m16p1-p4-m32p4-v2" : "") +
      (impl_->bulkQSAWorkspace && impl_->bulkQSAPrefillSG8 ? ";private-qsa-bulk-prefill-temporal-sg8-w4to15-m32-p4-t256-v1" : "") +
      (impl_->allRowsInt8Target ? ";private-allrows-full512-target-m16-below256-v1" : "") +
      (impl_->int8ExpertStore && impl_->int8ExpertStore->gatheredMPPEnabled()
          ? std::string(";") + std::string(gathered_mpp::kPolicy) : "") +
      ";private-singleton-prefill-arena-max8192-causal-qsa128-v1" +
      (impl_->fuseHC ? ";hc-fused-literal-sg4" : ";hc-separate") +
      (impl_->fuseGDN ? ";gdn-decode-persistent512-verify-capture512" : ";gdn-separate") +
      (impl_->lazyGDN ? kFlashGDNLazyRollbackRoute : "") +
      (impl_->stagedGDN
          ? (gdn_prefill_fma_sep21::requested()
              ? std::string(gdn_prefill_fma_sep21::marker(true))
              : std::string(";gdn-prefill-staged-v16-t16"))
          : std::string{}) +
      (impl_->denseCache ? ";dense-cache-bf16-whole-k:" + impl_->denseCache->identitySha256() : ";dense-raw") +
      (impl_->denseCache && flashDenseM64OutEnabled() ? std::string(kFlashDenseM64OutSemantics) : "") +
      (impl_->denseCache && flashPrefillDenseTilesEnabled() ? std::string(kFlashPrefillDenseTilesSemantics) : "") +
      (impl_->smallDenseWorkspace ? ";dense-all-rows-bf16-static-operands-padded-m8" : "") +
      (impl_->floatDenseCache ? ";dense-f32-original-coefficients-multirow-mpp:" + impl_->floatDenseCache->identitySha256() : "") +
      (impl_->int8Head ? std::string(";") + kFlashInt8HeadSemantics + ":" + impl_->int8Head->identitySha256() +
          ";" + impl_->int8Head->codeStorageSemantics() : "") +
      (impl_->selectiveFloat ? std::string(";") + kFlashFloatDenseSmallRowsPolicySemantics : "") +
      (impl_->hcUpF32 && impl_->fuseHC && impl_->floatDenseCache
          ? std::string(";") + kFlashHCUpF32MPPSemantics : "") +
      (impl_->fuseSharedExpert ? std::string(";") + kFlashSharedExpertFusedSemantics : "") +
      (impl_->expertPlanIdentity.empty() ? "" : ";hot-expert-bf16-whole-k:" + impl_->expertPlanIdentity) +
      (impl_->int8ExpertStore ? std::string(";") + kFlashInt8ExpertStoreSemantics + ":" +
          impl_->int8ExpertStore->identitySha256() : "") +
      (impl_->pleSSD ? kFlashPLESSDRoute
                    : impl_->pleLookup ? ";ple-exact-i64-hash-direct128-q4g32-tier2"
                       : impl_->fusePLELookup ? ";ple-gather8-tier2-unavailable"
                                              : ";ple-source-gather8") +
      ";" + flashPLEPostRouteSemantics() +
      (impl_->gpuGreedy ? ";greedy-gpu-bf16-argmax-inline" : ";greedy-host-vocabulary") +
      (impl_->blockMoE ? std::string(";") + flashMoEBlockedRouteSemantics() : ";moe-vector") +
      (impl_->onlineQSA ? (impl_->mppQSA ? qsaOnlineMPPRouteSemantics()
                                       : ";qsa-fused-prep-f32-online-partitions4")
                        : ";qsa-bf16-probabilities");
}

FlashHCUpEncodedCounters FlashForward::hcUpEncodedCounters() const {
  if (!impl_) return {};
  std::lock_guard lock(impl_->mutex);
  return impl_->hcUpCounters;
}

uint64_t FlashForward::qsaOutF32N32EncodedCalls() const {
  if (!impl_) return 0;
  std::lock_guard lock(impl_->mutex);
  return impl_->floatDenseCache ? impl_->floatDenseCache->qsaOutF32N32Dispatches() : 0;
}

uint64_t FlashForward::qsaOutF32N32EncodedRealRows() const {
  if (!impl_) return 0;
  std::lock_guard lock(impl_->mutex);
  return impl_->floatDenseCache ? impl_->floatDenseCache->qsaOutF32N32RealRows() : 0;
}

uint64_t FlashForward::expertCachePlannedBytes(const FlashWeights &weights) {
  if (const auto directory = int8ExpertDirectory())
    return FlashInt8ExpertStore::plannedBytes(weights, *directory);
  const auto plan = hotExpertPlan(weights);
  if (!plan) return 0;
  uint64_t total = 0;
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const auto bytes = FlashExpertDenseCache::plannedBytes(weights,
        "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp",
        plan->selectedExperts[layer]);
    if (bytes > UINT64_MAX - total) throw std::overflow_error("hot-expert plan bytes overflow");
    total += bytes;
  }
  return total;
}

uint64_t FlashForward::floatDenseCachePlannedBytes(const FlashWeights &weights) {
  if (!fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE")) return 0;
  return FlashFloatDenseCache::plannedBytes(weights,
      FlashFloatDenseCache::defaultPrefixes(weights, !fusionEnabled("SPLASH_FLASH_INT8_HEAD")));
}

uint64_t FlashForward::int8HeadPlannedBytes(const FlashWeights &weights) {
  if (!fusionEnabled("SPLASH_FLASH_INT8_HEAD")) return 0;
  return FlashInt8Head::plannedBytes(weights, fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_CACHE"));
}

const FlashTensor *FlashForward::cachedVocabulary() const {
  return impl_ && impl_->denseCache ? &impl_->denseCache->tensor("language_model.lm_head") : nullptr;
}

const FlashTensor *FlashForward::cachedFloatVocabulary() const {
  return impl_ && impl_->floatDenseCache && impl_->floatDenseCache->contains("language_model.lm_head")
      ? &impl_->floatDenseCache->tensor("language_model.lm_head") : nullptr;
}

FlashPersistedOperandStatus FlashForward::persistedOperandStatus() const {
  FlashPersistedOperandStatus result;
  if (!impl_) return result;
  if (impl_->denseCache) {
    result.bf16Tensors = impl_->denseCache->persistedTensorCount();
    result.bf16PayloadBytes = impl_->denseCache->persistedPayloadBytes();
    result.storeManifestSha256 = impl_->denseCache->operandStoreIdentitySha256();
  }
  if (impl_->floatDenseCache) {
    result.f32Tensors = impl_->floatDenseCache->persistedTensorCount();
    result.f32PayloadBytes = impl_->floatDenseCache->persistedPayloadBytes();
    const auto &identity = impl_->floatDenseCache->operandStoreIdentitySha256();
    if (!result.storeManifestSha256.empty() && !identity.empty() && result.storeManifestSha256 != identity)
      throw std::logic_error("Flash dense caches were loaded from different operand stores");
    if (!identity.empty()) result.storeManifestSha256 = identity;
  }
  return result;
}

std::vector<metal::MetalBuffer> FlashForward::cachedOperandsOnly() const {
  if (!impl_) return {};
  // Also checks that selected BF16/F32 caches came from the same saved store.
  (void)persistedOperandStatus();
  std::vector<metal::MetalBuffer> result;
  const auto append = [&](std::vector<metal::MetalBuffer> selected) {
    for (auto &buffer : selected) {
      if (!buffer || buffer.storage() != metal::BufferStorage::Shared ||
          !buffer.contents() || !buffer.sizeBytes())
        throw std::logic_error("Flash saved residency operand is not a mapped Shared buffer");
      result.push_back(std::move(buffer));
    }
  };
  if (impl_->denseCache) append(impl_->denseCache->persistedWeightBuffers());
  append(impl_->repackedCodes);
  if (impl_->denseW8Cache) append(impl_->denseW8Cache->immutableWeightBuffers());
  if (impl_->floatDenseCache) append(impl_->floatDenseCache->persistedWeightBuffers());
  // This store enumerates its derived base/rank allocations, never the raw
  // source projections retained for the Q4 miss route.
  // The Q4 expert route never reads the INT8 store; keeping it out of the
  // residency set leaves its 113 GB mapping untouched.
  if (impl_->int8ExpertStore && !opt::moeReplacesInt8())
    append(impl_->int8ExpertStore->immutableWeightBuffers());
  return result;
}

metal::MetalBuffer FlashForward::capturedExpertIDs(uint32_t &rows, uint32_t &stride) const {
  rows = impl_ ? impl_->capturedRows : 0;
  stride = impl_ ? impl_->maximumRows : 0;
  return impl_ ? impl_->capturedRoutes : metal::MetalBuffer{};
}

void FlashForward::batchProject(metal::CommandGraph &graph, const std::string &prefix,
                               const metal::MetalBuffer &input, const metal::MetalBuffer &output,
                               const metal::MetalBuffer &diagnostics, uint32_t rows) {
  impl_->project(graph, prefix, input, output, diagnostics, rows);
}

bool FlashForward::batchDenseSmallRowsEnabled() const noexcept {
  return impl_ && (bool(impl_->smallDenseWorkspace) || bool(impl_->floatDenseCache) || bool(impl_->int8Head));
}

bool FlashForward::batchSharedExpertFused(metal::CommandGraph &graph,
    const std::string &prefix, const metal::MetalBuffer &input,
    const metal::MetalBuffer &activated, const metal::MetalBuffer &diagnostics,
    uint32_t rows, const metal::MetalBuffer &tailGate,
    const metal::MetalBuffer &tailUp) {
  if (!impl_ || !flashSharedExpertFusedEligible(impl_->fuseSharedExpert,
      bool(impl_->denseCache), rows)) return false;
  const auto gate = prefix + ".gate_proj", up = prefix + ".up_proj";
  if (!impl_->denseCache->contains(gate) || !impl_->denseCache->contains(up)) return false;
  addSharedExpertFusedPrefill(impl_->backend, graph, impl_->denseCache->tensor(gate),
      impl_->denseCache->tensor(up), input, activated, diagnostics, rows,
      {tailGate, tailUp});
  return true;
}

bool FlashForward::batchFloatDenseEnabled() const noexcept {
  return impl_ && bool(impl_->floatDenseCache);
}

bool FlashForward::batchHCFusedUpMixF32(metal::CommandGraph &graph,
    const std::string &prefix, const metal::MetalBuffer &normalized,
    const metal::MetalBuffer &activated, const metal::MetalBuffer &mixed,
    const metal::MetalBuffer &diagnostics, uint32_t rows) {
  return impl_ && impl_->cachedHCUp(graph, prefix, normalized, activated, mixed, diagnostics, rows);
}

bool FlashForward::batchInt8HeadEnabled() const noexcept {
  return impl_ && bool(impl_->int8Head);
}

const FlashExpertDenseCache *FlashForward::batchExpertCache(uint32_t layer) const {
  if (!impl_ || layer >= 48) throw std::invalid_argument("invalid hot-expert layer index");
  return impl_->expertCaches[layer].get();
}
bool FlashForward::allRowsInt8TargetEnabled() const noexcept {
  return impl_ && impl_->allRowsInt8Target;
}

const FlashInt8ExpertStore *FlashForward::batchInt8ExpertStore() const noexcept {
  return impl_ ? impl_->int8ExpertStore.get() : nullptr;
}

const FlashPLEFused *FlashForward::batchPLELookup() const noexcept {
  return impl_ ? impl_->pleLookup.get() : nullptr;
}

void FlashForward::batchValidateExternalDestination(const metal::MetalBuffer &destination) const {
  if (!impl_ || !impl_->owner || !destination ||
      destination.storage() != metal::BufferStorage::Shared)
    throw std::invalid_argument("Flash source feature destination requires initialized Shared ownership");
  validateFlashBatchPrefillCopyRange(destination.contents(), destination.sizeBytes(), destination.sizeBytes());
  const auto reject = [&](const metal::MetalBuffer &buffer) {
    // A Private allocation cannot acquire a Shared view through backend.view.
    if (flashBatchPrefillCopyRangesOverlap(destination.contents(), destination.sizeBytes(),
                                          buffer.contents(), buffer.sizeBytes()))
      throw std::invalid_argument("Flash feature destination overlaps source workspace or immutable weights");
  };
  for (const auto &buffer : impl_->scratch) reject(buffer);
  for (const auto &buffer : {impl_->qsaWorkspace.queries, impl_->qsaWorkspace.indexQueries,
      impl_->qsaWorkspace.blockScores, impl_->qsaWorkspace.selectedBlocks,
      impl_->qsaWorkspace.attentionScores, impl_->qsaWorkspace.probabilities,
      impl_->qsaFastWorkspace.partitionStatistics, impl_->qsaFastWorkspace.partitionValues,
      impl_->greedyWorkspace.partials, impl_->greedyResults, impl_->capturedRoutes,
      impl_->verifyRecurrent, impl_->verifyConvolution, impl_->beforePLEHistory,
      impl_->beforePLEConvolution, impl_->retainedCount, impl_->beforeGDNConvolution}) reject(buffer);
  if (impl_->bulkQSAWorkspace) {
    const auto &bulk = *impl_->bulkQSAWorkspace;
    for (const auto &buffer : {bulk.prepared.queries, bulk.prepared.indexQueries,
        bulk.prepared.selectedBlocks, bulk.partials.partitionStatistics,
        bulk.partials.partitionValues}) reject(buffer);
  }
  if (impl_->twoPassQSAWorkspace) {
    const auto &two = *impl_->twoPassQSAWorkspace;
    for (const auto &buffer : {two.arena,two.qualified.prepared.queries,
        two.qualified.prepared.indexQueries,two.qualified.prepared.selectedBlocks,
        two.qualified.packedQueries,two.qualified.scoresAndProbabilities,two.qualified.rawAttention}) reject(buffer);
  }
  const auto &blocked = impl_->blockedScratch;
  for (const auto &buffer : {blocked.buckets.counts, blocked.buckets.offsets, blocked.buckets.routeMap,
      blocked.buckets.canonicalToPacked, blocked.buckets.packedInputs, blocked.buckets.jobOffsets,
      blocked.buckets.jobCount, blocked.buckets.tileJobs, blocked.packedActivated, blocked.scatteredDown})
    reject(buffer);
  if (impl_->smallDenseWorkspace) reject(impl_->smallDenseWorkspace->paddedInput());
  if (impl_->floatDenseWorkspace) reject(impl_->floatDenseWorkspace->paddedInput());
  if (impl_->int8Head)
    for (const auto &buffer : impl_->int8Head->scratchBuffers()) reject(buffer);
  if (impl_->pleLookup) reject(impl_->pleLookup->sources_);
  if (impl_->pleSSD)
    for (const auto &buffer : impl_->pleSSD->scratchBuffers()) reject(buffer);
  for (const auto &buffer : impl_->weights.immutableWeightBuffers()) reject(buffer);
  for (const auto &record : impl_->lazyGDNRecords)
    if (record) for (const auto &buffer : record->arenaBuffers()) reject(buffer);
  if (impl_->denseCache)
    for (const auto &buffer : impl_->denseCache->immutableWeightBuffers()) reject(buffer);
  if (impl_->denseW8Cache)
    for (const auto &buffer : impl_->denseW8Cache->immutableWeightBuffers()) reject(buffer);
  if (impl_->denseW8Workspace)
    for (const auto &buffer : {impl_->denseW8Workspace->codes,impl_->denseW8Workspace->inputScales,
        impl_->denseW8Workspace->dummyDot}) reject(buffer);
  if (impl_->floatDenseCache)
    for (const auto &buffer : impl_->floatDenseCache->immutableWeightBuffers()) reject(buffer);
  if (impl_->int8Head)
    for (const auto &buffer : impl_->int8Head->immutableWeightBuffers()) reject(buffer);
  for (const auto &cache : impl_->expertCaches)
    if (cache) for (const auto &buffer : cache->immutableWeightBuffers()) reject(buffer);
  if (impl_->int8ExpertStore)
    for (const auto &buffer : impl_->int8ExpertStore->immutableWeightBuffers()) reject(buffer);
}

bool FlashForward::ownsState(const FlashRequestState &state) const noexcept {
  return impl_ && state.impl_ && state.impl_->owner == impl_->owner &&
      !state.impl_->poisoned && !state.impl_->pendingVerification &&
      state.impl_->capacity == impl_->capacity;
}

metal::MetalBackend &FlashForward::batchBackend() const {
  if (!impl_) throw std::logic_error("Flash trunk is not initialized"); return impl_->backend;
}
const FlashWeights &FlashForward::batchWeights() const {
  if (!impl_) throw std::logic_error("Flash trunk is not initialized"); return impl_->weights;
}
uint32_t FlashForward::batchCapacity() const {
  if (!impl_) throw std::logic_error("Flash trunk is not initialized"); return impl_->capacity;
}
std::mutex &FlashForward::batchMutex() const {
  if (!impl_) throw std::logic_error("Flash trunk is not initialized"); return impl_->mutex;
}

uint64_t FlashForward::verificationWorkspaceBytes(uint32_t maximumVerifyRows) {
  if (maximumVerifyRows > kFlashSingletonMaximumVerifyRows)
    throw std::invalid_argument("Flash verification maximum rows exceeds16");
  if (maximumVerifyRows <= 1) return 0;
  const uint64_t gdn = flashGDNLazyRollbackEnabled()
      ? uint64_t{36} * FlashGDNLazyRollback::plannedBytes(maximumVerifyRows, 1)
      : uint64_t{36} * (maximumVerifyRows - 1) *
          (roundAllocation(flashGDNRecurrentLaneBytes()) + roundAllocation(flashGDNConvolutionLaneBytes()));
  return gdn +
      roundAllocation(2 * sizeof(int64_t)) + roundAllocation(uint64_t{9} * kHyper * 2) +
      roundAllocation(sizeof(uint32_t));
}

FlashRequestState FlashForward::createState() {
  if (!impl_) throw std::logic_error("Flash forward is not initialized");
  opt::ScopedTimer timer("forward.create_state");
  std::lock_guard lock(impl_->mutex);
  FlashRequestState result;
  std::unique_ptr<FlashRequestState::Impl> reused;
  {
    std::lock_guard poolLock(impl_->statePool->mutex);
    if (!impl_->statePool->states.empty()) {
      reused = std::move(impl_->statePool->states.back());
      impl_->statePool->states.pop_back();
    }
  }
  if (reused && reused->owner == impl_->owner && reused->capacity == impl_->capacity) {
    // Every position a previous request could have written lies below its
    // final length plus one verification window; clear that prefix of each
    // attention cache and all recurrent/convolution/PLE state.
    auto &state = *reused;
    const uint64_t dirty = std::min<uint64_t>(state.capacity, state.length + 64);
    const auto zero = [](const metal::MetalBuffer &buffer, uint64_t bytes) {
      std::memset(buffer.contents(), 0, std::min<uint64_t>(bytes, buffer.sizeBytes()));
    };
    for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
      if (impl_->descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
        clear(state.gdn[layer].convolution); clear(state.gdn[layer].recurrent);
      } else {
        auto &qsa = state.qsa[layer];
        zero(qsa.keys, dirty * 512 * 2); zero(qsa.values, dirty * 512 * 2);
        zero(qsa.rawIndexKeys, dirty * 128 * 2);
        zero(qsa.pooledKeys, ((dirty + 3) / 4 + 1) * 128 * 2);
        zero(qsa.indexPositions, dirty * 8);
      }
    }
    clear(state.pleHistory); clear(state.pleConvolution);
    auto *history = static_cast<int64_t *>(state.pleHistory.contents());
    history[0] = history[1] = impl_->descriptor.pleHistoryEos;
    state.length = 0; state.poisoned = false; state.pendingVerification = false;
    state.identity = std::make_shared<const uint8_t>(0);
    result.impl_ = std::move(reused);
    return result;
  }
  reused.reset();
  result.impl_ = std::make_unique<FlashRequestState::Impl>();
  auto &state = *result.impl_;
  state.pool = impl_->statePool;
  state.owner = impl_->owner;
  state.capacity = impl_->capacity;
  for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
    if (impl_->descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
      state.gdn[layer].convolution = impl_->backend.allocateBuffer(
          roundAllocation(flashGDNConvolutionLaneBytes()), metal::BufferStorage::Shared,
          "flash-request-gdn-convolution");
      state.gdn[layer].recurrent = impl_->backend.allocateBuffer(
          roundAllocation(flashGDNRecurrentLaneBytes()), metal::BufferStorage::Shared,
          "flash-request-gdn-recurrent");
      clear(state.gdn[layer].convolution); clear(state.gdn[layer].recurrent);
    } else {
      state.qsa[layer] = allocateQSAState(impl_->backend, state.capacity);
      clear(state.qsa[layer].keys); clear(state.qsa[layer].values);
      clear(state.qsa[layer].rawIndexKeys); clear(state.qsa[layer].pooledKeys);
      clear(state.qsa[layer].indexPositions);
    }
  }
  state.pleHistory = impl_->backend.allocateBuffer(roundAllocation(2 * sizeof(int64_t)),
      metal::BufferStorage::Shared, "flash-request-ple-history");
  state.pleConvolution = impl_->backend.allocateBuffer(roundAllocation(uint64_t{9} * kHyper * 2),
      metal::BufferStorage::Shared, "flash-request-ple-convolution");
  clear(state.pleHistory); clear(state.pleConvolution);
  auto *history = static_cast<int64_t *>(state.pleHistory.contents());
  history[0] = history[1] = impl_->descriptor.pleHistoryEos;
  return result;
}

FlashForwardResult FlashForward::forward(FlashRequestState &request,
                                         std::span<const uint32_t> tokens,
                                         bool returnAllLogits, bool captureHidden) {
  return forwardImpl(request, tokens, returnAllLogits, captureHidden, false);
}

FlashForwardResult FlashForward::verify(FlashRequestState &request,
                                        std::span<const uint32_t> tokens) {
  return forwardImpl(request, tokens, true, true, true);
}

void FlashForward::prefetchPLE(FlashRequestState &request, std::span<const uint32_t> tokens) {
  if (!impl_ || !impl_->pleSSD || tokens.empty() || tokens.size() > 8) return;
  std::lock_guard lock(impl_->mutex);
  if (!request.impl_ || request.impl_->owner != impl_->owner || request.impl_->poisoned) return;
  auto &state = *request.impl_;
  const auto *history = static_cast<const int64_t *>(state.pleHistory.contents());
  if (!history) return;
  std::array<int64_t, 8> ids{};
  for (size_t index = 0; index < tokens.size(); ++index) {
    if (tokens[index] >= impl_->descriptor.vocabularySize) return;
    ids[index] = tokens[index];
  }
  const uint32_t rows = static_cast<uint32_t>(tokens.size());
  const FlashPLEGeometry geometry{1, rows, kWidth, 4, impl_->descriptor.pleHistoryEos,
      impl_->descriptor.vocabularySize, static_cast<float>(impl_->descriptor.normEpsilon)};
  impl_->pleSSD->prefetch({ids.data(), rows}, {history, 2}, impl_->pleWeights, geometry);
}

struct FlashForward::Window {
  metal::CommandGraph graph;
  metal::MetalBuffer tokenIDs, diag, logits, hyper, compactGreedy;
  FlashPLEGeometry pleGeometry{};
  uint32_t rows = 0, begin = 0, logitRows = 0;
  bool verification = false, captureHidden = false;
  // A verification build has begun lazy GDN trials that must be kept by a
  // successful submission or aborted.
  bool lazyPending = false;
  uint64_t serial = 0;
};

FlashForwardResult FlashForward::forwardImpl(FlashRequestState &request,
                                             std::span<const uint32_t> tokens,
                                             bool returnAllLogits, bool captureHidden,
                                             bool verification) {
  if (!impl_) throw std::logic_error("Flash forward is not initialized");
  std::lock_guard lock(impl_->mutex);
  if (tokens.empty() || tokens.size() > impl_->maximumRows)
    throw std::invalid_argument("Flash forward token window exceeds rows/context capacity");
  auto window = buildWindow(request, static_cast<uint32_t>(tokens.size()), returnAllLogits,
                            captureHidden, verification);
  return executeWindow(request, *window, tokens);
}

std::unique_ptr<FlashForward::Window> FlashForward::buildWindow(
    FlashRequestState &request, uint32_t rows, bool returnAllLogits, bool captureHidden,
    bool verification) {
  if (!request.impl_ || request.impl_->owner != impl_->owner || request.impl_->poisoned)
    throw std::invalid_argument("Flash forward requires its own healthy request state");
  auto &state = *request.impl_;
  if (state.pendingVerification && !impl_->pendingRows)
    throw std::logic_error("resolve the pending batch verification before a trunk call");
  if (impl_->pendingRows && impl_->pendingIdentity.expired())
    impl_->finishPending(); // The cancelled/destroyed request cannot be resumed.
  if (impl_->pendingRows)
    throw std::logic_error("resolve the previous Flash verification before another trunk call");
  if (verification && (!impl_->maximumVerifyRows || rows > impl_->maximumVerifyRows))
    throw std::invalid_argument("Flash target verification window exceeds provisioned prefix tape");
  if (!rows || rows > impl_->maximumRows || state.length > state.capacity ||
      rows > state.capacity - state.length)
    throw std::invalid_argument("Flash forward token window exceeds rows/context capacity");
  if (returnAllLogits && rows > impl_->maximumLogitRows)
    throw std::invalid_argument("Flash all-logits window exceeds bounded head storage");
  const uint32_t begin = static_cast<uint32_t>(state.length);
  const bool privateTwoPassQSA = prefill_qsa_twopass_sep21::mainEligible(
      begin,rows,verification,true,prefill_qsa_twopass_sep21::requested());
  if (privateTwoPassQSA) prefill_qsa_twopass_sep21::recordForward();
  const bool privatePrefillHC = prefill_hc_inject_norm_sep21::mainEligible(
      rows, verification, true, prefill_hc_inject_norm_sep21::requested());
  if (privatePrefillHC) prefill_hc_inject_norm_sep21::recordForward();
  impl_->capturedRows = 0;
  auto tokenIDs = impl_->view(Scratch::TokenIDs, uint64_t{rows} * 8);
  if (!tokenIDs.contents()) throw std::logic_error("Flash token input is not shared");
  auto diag = impl_->view(Scratch::Diagnostics, 4);
  clear(diag);
  const auto bf = [&](Scratch slot, uint32_t width) { return impl_->bf(slot, rows, width); };
  const auto affine = [&](metal::CommandGraph &graph, const std::string &prefix,
                          const metal::MetalBuffer &input, const metal::MetalBuffer &output) {
    impl_->project(graph, prefix, input, output, diag, rows, verification, true);
  };
  auto window = std::make_unique<Window>();
  metal::CommandGraph &graph = window->graph;
  struct LazyTrialGuard {
    Impl *source;
    bool keep = false;
    ~LazyTrialGuard() { if (!keep) source->abortLazyRecords(); }
  } lazyGuard{impl_.get(), !verification};
  const auto hyper = bf(Scratch::Hyper, kHyper);
  const auto mixed = bf(Scratch::Mixed, kWidth);
  const auto branch = bf(Scratch::Branch, kWidth);
  const auto embedding = bf(Scratch::Embedding, kWidth);
  const FlashHCGeometry hcGeometry{rows, kWidth, 4, static_cast<float>(impl_->descriptor.normEpsilon)};
  const FlashPLEGeometry pleGeometry{1, rows, kWidth, 4, impl_->descriptor.pleHistoryEos,
      impl_->descriptor.vocabularySize, static_cast<float>(impl_->descriptor.normEpsilon)};
  std::optional<opt::ScopedTimer> graphTimer;
  graphTimer.emplace(rows <= 8 ? "forward.small.graph_build" : "forward.large.graph_build");
  addAffineEmbedding(graph, impl_->weights.projection("language_model.model.embed_tokens"),
                       tokenIDs, embedding, diag, rows);
  addHCExpand(graph, embedding, hyper, hcGeometry);
  bool normalizedReady = false;
  for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
    const std::string prefix = "language_model.model.layers." + std::to_string(layer);
    if (layer == impl_->descriptor.pleLayerIndices.front()) {
      if (verification && rows > 1) {
        impl_->copy(graph, state.pleHistory, impl_->beforePLEHistory, 2 * sizeof(int64_t));
        impl_->copy(graph, state.pleConvolution, impl_->beforePLEConvolution, uint64_t{9} * kHyper * 2);
      }
      const auto ngram = impl_->view(Scratch::NgramIDs, uint64_t{rows} * 16 * 8);
      if (impl_->pleSSD) {
        // Rows are staged by executeWindow() once the tokens are known.
        impl_->pleSSD->addHashGather(graph, impl_->pleWeights, tokenIDs,
            state.pleHistory, ngram, bf(Scratch::PLEEmbedding, kWidth), diag, pleGeometry, true);
      } else if (!impl_->pleLookup || !impl_->pleLookup->addHashGather(graph, tokenIDs,
          state.pleHistory, ngram, bf(Scratch::PLEEmbedding, kWidth), diag, pleGeometry)) {
        addPLENgramIDs(graph, impl_->pleWeights, tokenIDs, state.pleHistory, ngram, diag, pleGeometry);
        addPLEGather(graph, impl_->pleWeights, ngram, bf(Scratch::PLEEmbedding, kWidth), diag, pleGeometry);
      }
      affine(graph, prefix + ".ple.key_proj", bf(Scratch::PLEEmbedding, kWidth), bf(Scratch::PLEKey, kHyper));
      affine(graph, prefix + ".ple.value_proj", bf(Scratch::PLEEmbedding, kWidth), bf(Scratch::PLEValue, kWidth));
      const FlashPLEPostScratch post{bf(Scratch::PLENormalizedKeys, kHyper),
          bf(Scratch::PLENormalizedQueries, kHyper), bf(Scratch::PLEGated, kHyper),
          bf(Scratch::PLENormalizedConvolution, kHyper)};
      addPLEPostProjectAndInject(graph, impl_->pleWeights, hyper, bf(Scratch::PLEKey, kHyper),
          bf(Scratch::PLEValue, kWidth), post, state.pleConvolution,
          bf(Scratch::PLEOutput, kHyper), diag, pleGeometry);
      normalizedReady = false;
    }
    impl_->hc(graph, prefix + ".attn_hyper_connection", rows, true, normalizedReady, verification);
    normalizedReady = false;
    const auto attentionOutput = bf(Scratch::AttentionOutput, 6144);
    if (impl_->descriptor.layerKinds[layer] == FlashLayerKind::GatedDeltaNet) {
      const std::string attention = prefix + ".linear_attn";
      const auto qkv = bf(Scratch::QProjection, 10240);
      affine(graph, attention + ".in_proj_qkv", mixed, qkv);
      affine(graph, attention + ".in_proj_z", mixed, bf(Scratch::GDNZ, 6144));
      if (impl_->mergeGDNAB && (rows == 1 || rows == 4)) {
        (void)gdn_ab_merge::requested();
        const auto &a = impl_->weights.projection(attention + ".in_proj_a");
        const auto &b = impl_->weights.projection(attention + ".in_proj_b");
        if (gdn_ab_merge::geometry(a, rows) && gdn_ab_merge::geometry(b, rows) && flashAffineFastEnabled()) {
          gdn_ab_merge::add(graph, mixed, a, b, bf(Scratch::GDNA, 48), bf(Scratch::GDNB, 48), diag, rows);
        } else {
          affine(graph, attention + ".in_proj_a", mixed, bf(Scratch::GDNA, 48));
          affine(graph, attention + ".in_proj_b", mixed, bf(Scratch::GDNB, 48));
        }
      } else {
        affine(graph, attention + ".in_proj_a", mixed, bf(Scratch::GDNA, 48));
        affine(graph, attention + ".in_proj_b", mixed, bf(Scratch::GDNB, 48));
      }
      const FlashGDNWeights weights{&impl_->weights.tensor(attention + ".conv1d.weight"),
          &impl_->weights.tensor(attention + ".A_log"), &impl_->weights.tensor(attention + ".dt_bias"),
          &impl_->weights.tensor(attention + ".norm.weight")};
      const FlashGDNBuffers buffers{qkv, bf(Scratch::GDNZ, 6144), bf(Scratch::GDNA, 48),
          bf(Scratch::GDNB, 48), bf(Scratch::GDNMixed, 10240),
          impl_->view(Scratch::GDNDecay, uint64_t{rows} * 48 * 4), bf(Scratch::GDNBeta, 48),
          bf(Scratch::GDNRecurrentRows, 6144), attentionOutput, diag};
      if (verification && rows > 1) {
        if (impl_->lazyGDN) {
          impl_->lazyGDNTickets[layer] = impl_->lazyGDNRecords[layer]->begin(graph, weights, buffers,
              state.gdn[layer], rows, 1, static_cast<float>(impl_->descriptor.normEpsilon));
        } else if (impl_->fuseGDN) {
          impl_->copy(graph, state.gdn[layer].convolution, impl_->beforeGDNConvolution,
              flashGDNConvolutionLaneBytes());
          const uint64_t recurrentStride = roundAllocation(flashGDNRecurrentLaneBytes());
          const uint64_t firstCell = uint64_t{impl_->gdnSlots[layer]} * (impl_->maximumVerifyRows - 1);
          const auto tape = impl_->backend.view(impl_->verifyRecurrent,
              firstCell * recurrentStride, uint64_t{rows - 1} * recurrentStride);
          const FlashGDNCapture capture{tape, recurrentStride,
              uint64_t{rows - 1} * recurrentStride, rows - 1, 512};
          addGDNFusedCaptured(graph, weights, buffers, state.gdn[layer], capture, rows, 1,
              static_cast<float>(impl_->descriptor.normEpsilon));
          const uint64_t historyRowBytes = uint64_t{10240} * 2;
          for (uint32_t kept = 1; kept < rows; ++kept) {
            const auto destination = impl_->prefixTape(false, layer, kept - 1);
            const auto history = *flashMTPConvolutionPrefix(kept);
            if (history.oldRows) {
              const uint64_t bytes = uint64_t{history.oldRows} * historyRowBytes;
              impl_->copy(graph, impl_->backend.view(impl_->beforeGDNConvolution,
                  uint64_t{history.oldBegin} * historyRowBytes, bytes),
                  impl_->backend.view(destination, 0, bytes), bytes);
            }
            const uint64_t bytes = uint64_t{history.inputRows} * historyRowBytes;
            impl_->copy(graph, impl_->backend.view(qkv,
                uint64_t{history.inputBegin} * historyRowBytes, bytes),
                impl_->backend.view(destination, uint64_t{history.destinationInputBegin} * historyRowBytes, bytes), bytes);
          }
        } else {
        const auto slice = [&](const metal::MetalBuffer &buffer, uint32_t row, uint32_t width, uint32_t elementBytes = 2) {
          const uint64_t extent = uint64_t{width} * elementBytes;
          return impl_->backend.view(buffer, uint64_t{row} * extent, extent);
        };
        for (uint32_t row = 0; row < rows; ++row) {
          const FlashGDNBuffers single{slice(buffers.qkv, row, 10240), slice(buffers.z, row, 6144),
              slice(buffers.a, row, 48), slice(buffers.b, row, 48), slice(buffers.mixed, row, 10240),
              slice(buffers.decay, row, 48, 4), slice(buffers.beta, row, 48),
              slice(buffers.recurrentRows, row, 6144), slice(buffers.output, row, 6144), diag};
          addGDN(graph, weights, single, state.gdn[layer], 1, 1,
                   static_cast<float>(impl_->descriptor.normEpsilon));
          if (row + 1 < rows) {
            impl_->copy(graph, state.gdn[layer].convolution, impl_->prefixTape(false, layer, row),
                          flashGDNConvolutionLaneBytes());
            impl_->copy(graph, state.gdn[layer].recurrent, impl_->prefixTape(true, layer, row),
                          flashGDNRecurrentLaneBytes());
          }
        }
        }
      } else {
        if (impl_->stagedGDN && rows >= 64)
          addGDNStagedPrefill(graph, weights, buffers, state.gdn[layer], rows, 1,
              FlashGDNStageTile::Values16Time16,
              static_cast<float>(impl_->descriptor.normEpsilon));
        else if (impl_->fuseGDN)
          addGDNFused(graph, weights, buffers, state.gdn[layer], rows, 1,
              rows == 1 ? FlashGDNFusion::PersistentHead512 : FlashGDNFusion::Prepare,
              static_cast<float>(impl_->descriptor.normEpsilon));
        else
          addGDN(graph, weights, buffers, state.gdn[layer], rows, 1,
                   static_cast<float>(impl_->descriptor.normEpsilon));
      }
      affine(graph, attention + ".out_proj", attentionOutput, branch);
    } else {
      const std::string attention = prefix + ".self_attn";
      const auto q = bf(Scratch::QProjection, 12288);
      const auto k = bf(Scratch::KProjection, 512);
      const auto v = bf(Scratch::VProjection, 512);
      const auto index = bf(Scratch::IndexProjection, 640);
      affine(graph, attention + ".q_proj", mixed, q);
      affine(graph, attention + ".k_proj", mixed, k);
      affine(graph, attention + ".v_proj", mixed, v);
      affine(graph, attention + ".indexer.index_qk_proj", mixed, index);
      const std::string qNorm = attention + ".q_norm.weight", kNorm = attention + ".k_norm.weight";
      const std::string iqNorm = attention + ".indexer.q_layernorm.weight";
      const std::string ikNorm = attention + ".indexer.k_layernorm.weight";
      // QSA keeps its qualified128-row scratch geometry. Each complete chunk
      // appends/normalizes/pools/selects/attends before that scratch is reused.
      const auto slice = [&](const metal::MetalBuffer &buffer, uint32_t offset,
                              uint32_t count, uint32_t width) {
        return impl_->backend.view(buffer, uint64_t{offset} * width * 2,
                                     uint64_t{count} * width * 2);
      };
      if (impl_->bulkQSAWorkspace && !verification && begin == 0 && rows == 2048) {
        const FlashQSAFastInputs bulkInputs{q, k, v, index,
            &impl_->weights.tensor(qNorm), &impl_->weights.tensor(kNorm),
            &impl_->weights.tensor(iqNorm), &impl_->weights.tensor(ikNorm),
            attentionOutput, diag, {},
            impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
            impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
            impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta};
        if (privateTwoPassQSA) {
          if (!impl_->twoPassQSAWorkspace) throw std::logic_error("private QSA arena missing after admission");
          prefill4k::addTwoPassQSA(impl_->backend,graph,bulkInputs,state.qsa[layer],
              impl_->qsaWorkspace,impl_->qsaFastWorkspace,impl_->twoPassQSAWorkspace->qualified,
              begin,rows,verification,true); // Only Root-qualified packed-V route.
          prefill_qsa_twopass_sep21::recordLayer();
        } else {
          prefill4k::addBulkExactQSA(impl_->backend, graph, bulkInputs, state.qsa[layer],
              impl_->qsaWorkspace, impl_->qsaFastWorkspace, *impl_->bulkQSAWorkspace, begin, rows,
              impl_->bulkQSAPrefillSG8);
        }
      } else {
      for (uint32_t offset = 0; offset < rows;) {
        const uint32_t count = std::min(rows - offset, impl_->qsaWorkspace.maximumRows);
        if (impl_->onlineQSA) {
          const FlashQSAFastInputs inputs{slice(q, offset, count, 12288),
              slice(k, offset, count, 512), slice(v, offset, count, 512),
              slice(index, offset, count, 640), &impl_->weights.tensor(qNorm),
              &impl_->weights.tensor(kNorm), &impl_->weights.tensor(iqNorm),
              &impl_->weights.tensor(ikNorm), slice(attentionOutput, offset, count, 6144), diag, {},
              impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
              impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
              impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta};
          const uint32_t partitions = impl_->mppQSA
              ? qsaOnlineMPPRoutePartitions(begin + offset, count) : 0;
          if (partitions)
            addQSAOnlineMPP(graph, inputs, state.qsa[layer], impl_->qsaWorkspace,
                impl_->qsaFastWorkspace, begin + offset, count, partitions, true);
          else
            addQSAFast(graph, inputs, state.qsa[layer], impl_->qsaWorkspace,
                impl_->qsaFastWorkspace, begin + offset, count,
                FlashQSAFastMode::PartitionedF32Probabilities, 4, true);
        } else {
        addQSA(graph, slice(q, offset, count, 12288), slice(k, offset, count, 512),
            slice(v, offset, count, 512), slice(index, offset, count, 640),
            impl_->weights.tensor(qNorm), impl_->weights.tensor(kNorm),
            impl_->weights.tensor(iqNorm), impl_->weights.tensor(ikNorm), state.qsa[layer],
            impl_->qsaWorkspace, slice(attentionOutput, offset, count, 6144), diag, begin + offset, count,
            impl_->weights.normConvention(qNorm), impl_->weights.normConvention(kNorm),
            impl_->weights.normConvention(iqNorm), impl_->weights.normConvention(ikNorm),
            impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta);
        }
        offset += count;
      }
      }
      affine(graph, attention + ".o_proj", attentionOutput, branch);
    }
    const std::string mlpNorm = prefix + ".mlp_hyper_connection.hc_norm.weight";
    if (privatePrefillHC) {
      prefill_hc_inject_norm_sep21::addPrivatePrefillHCInjectNormSep21(graph, hyper, branch,
          bf(Scratch::HCInjectionWeights, 4), impl_->weights.tensor(mlpNorm), hyper,
          bf(Scratch::HCNormalized, kHyper), diag, hcGeometry, impl_->weights.normConvention(mlpNorm));
      prefill_hc_inject_norm_sep21::recordAttentionToMlp();
      normalizedReady = true;
    } else if (impl_->fuseHC && rows <= 32) {
      addHCFusedInjectNorm(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4),
          impl_->weights.tensor(mlpNorm), hyper, bf(Scratch::HCNormalized, kHyper), diag,
          hcGeometry, impl_->weights.normConvention(mlpNorm));
      normalizedReady = true;
    } else {
      addHCInject(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4), hyper, hcGeometry);
    }
    impl_->hc(graph, prefix + ".mlp_hyper_connection", rows, true, normalizedReady, verification);
    const std::string mlp = prefix + ".mlp";
    const auto router = bf(Scratch::Router, 512);
    const auto ids = impl_->view(Scratch::ExpertIDs, uint64_t{rows} * kSelections * 8);
    const auto route = bf(Scratch::RouteWeights, kSelections);
    if (impl_->denseCache && rows >= 16)
      addDenseBF16WholeK(impl_->backend, graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"),
          router, diag, rows, FlashAffineMPPTile::M16N64);
    else if (!opt::addDenseBF16Rows(graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"), router, rows))
      addDenseBF16(graph, mixed, impl_->weights.tensor(mlp + ".gate.weight"), router, diag, rows);
    addRoute(graph, router, ids, route, diag, rows, 512, kSelections);
    if (impl_->capturedRoutes)
      impl_->copy(graph, ids, impl_->backend.view(impl_->capturedRoutes,
          uint64_t{layer} * impl_->maximumRows * kSelections * 8,
          uint64_t{rows} * kSelections * 8), uint64_t{rows} * kSelections * 8);
    const bool optExperts = rows <= opt::kQmvMaximumRows && opt::moeEnabled() && opt::addMoEExperts(graph, mixed,
        impl_->weights.projection(mlp + ".switch_mlp.gate_proj"),
        impl_->weights.projection(mlp + ".switch_mlp.up_proj"),
        impl_->weights.projection(mlp + ".switch_mlp.down_proj"), ids,
        bf(Scratch::ExpertIntermediate, kSelections * 640),
        bf(Scratch::ExpertDown, kSelections * kWidth), rows, kSelections);
    const bool gatheredMPP = !optExperts && !opt::moeReplacesInt8() && impl_->allRowsInt8Target && impl_->int8ExpertStore &&
        impl_->int8ExpertStore->gatheredMPPEnabled() && rows <= impl_->int8ExpertStore->gatheredMPPMaximumRows();
    const bool blocked = !optExperts && impl_->blockMoE && (impl_->allRowsInt8Target || rows >= 256);
    const bool compactR4Verify = !optExperts && !opt::moeReplacesInt8() && verification && impl_->allRowsInt8Target && impl_->int8ExpertStore &&
        compact_native_r4_verify_sep22::eligible(rows,verification) && impl_->int8ExpertStore->compactNativeR4VerifyEnabled();
    if (optExperts) {
    } else if (compactR4Verify && compact_r4_preflight_sep22::requested()) {
      impl_->int8ExpertStore->addCompactNativeR4VerifyChain(graph,layer,mixed,ids,impl_->blockedScratch,diag,rows,kSelections);
    } else if (compactR4Verify) {
      impl_->int8ExpertStore->addCompactNativeR4VerifyPack(graph,layer,mixed,ids,impl_->blockedScratch,diag,rows,kSelections);
      impl_->int8ExpertStore->addCompactNativeR4VerifyGateUp(graph,layer,impl_->blockedScratch,diag,rows,kSelections);
      impl_->int8ExpertStore->addCompactNativeR4VerifyDown(graph,layer,impl_->blockedScratch,diag,rows,kSelections);
    } else if (gatheredMPP) {
      impl_->int8ExpertStore->addGatheredMPPGateUp(graph, layer, mixed, ids,
          bf(Scratch::ExpertIntermediate, kSelections * 640), diag, rows, kSelections);
      impl_->int8ExpertStore->addGatheredMPPDown(graph, layer,
          bf(Scratch::ExpertIntermediate, kSelections * 640), ids,
          bf(Scratch::ExpertDown, kSelections * kWidth), diag, rows, kSelections);
    } else if (blocked) {
      const auto tile = impl_->allRowsInt8Target && rows < 256
          ? FlashMoEBlockedTile::M16N64 : flashMoEBlockedTile(rows, bool(impl_->expertCaches[layer]));
      // With the Q4 expert route the INT8 store is never referenced, so it
      // is never made GPU-resident; prefill uses the original Q4 experts.
      const bool useInt8Store = impl_->int8ExpertStore && !opt::moeReplacesInt8();
      const bool compactR5=impl_->allRowsInt8Target&&useInt8Store&&tile==FlashMoEBlockedTile::M16N64&&compact_native_r5_verify_sep22::eligible(rows,verification,true);
      if(compactR5)
        compact_native_r5_verify_sep22::addSetup(graph,mixed,ids,impl_->blockedScratch,diag,rows,tile,verification,true);
      else
        addMoEBlockedPack(graph, mixed, ids, impl_->blockedScratch, diag, rows, tile);
      if (useInt8Store) {
        if (fixed_sg2_prefill_sep21::eligible(rows,tile,verification)) {
          impl_->int8ExpertStore->addFixedSG2PrefillGateUp(graph,layer,impl_->blockedScratch,diag,rows,tile);
          impl_->int8ExpertStore->addFixedSG2PrefillDownScatter(graph,layer,impl_->blockedScratch,diag,rows,tile);
        } else {
          impl_->int8ExpertStore->addGateUp(graph, layer, impl_->blockedScratch, diag, rows, tile);
          impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);
          if(compactR5)compact_native_r5_verify_sep22::recordCompletedGraph(rows);
        }
      } else if (impl_->expertCaches[layer]) {
        impl_->expertCaches[layer]->addGateUp(graph, impl_->blockedScratch, diag, rows, tile);
        impl_->expertCaches[layer]->addDownScatter(graph, impl_->blockedScratch, diag, rows, tile);
      } else {
        addMoEBlockedGateUp(graph, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"),
            impl_->weights.projection(mlp + ".switch_mlp.up_proj"), impl_->blockedScratch,
            diag, rows, tile);
        addMoEBlockedDownScatter(graph, impl_->weights.projection(mlp + ".switch_mlp.down_proj"),
            impl_->blockedScratch, diag, rows, tile);
      }
    } else {
    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"), ids,
        bf(Scratch::ExpertGate, kSelections * 640), diag, rows, kSelections);
    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.up_proj"), ids,
        bf(Scratch::ExpertUp, kSelections * 640), diag, rows, kSelections);
    addSiLUMultiply(graph, bf(Scratch::ExpertGate, kSelections * 640),
        bf(Scratch::ExpertUp, kSelections * 640), bf(Scratch::ExpertIntermediate, kSelections * 640),
        diag, rows, 640, kSelections);
    addGatheredAffine(graph, bf(Scratch::ExpertIntermediate, kSelections * 640),
        impl_->weights.projection(mlp + ".switch_mlp.down_proj"), ids,
        bf(Scratch::ExpertDown, kSelections * kWidth), diag, rows, kSelections, true);
    }
    if (!opt::addSharedSwiGLU(graph, mixed, impl_->weights.projection(mlp + ".shared_expert.gate_proj"),
            impl_->weights.projection(mlp + ".shared_expert.up_proj"),
            bf(Scratch::SharedIntermediate, 640), diag, rows) &&
        !batchSharedExpertFused(graph, mlp + ".shared_expert", mixed,
        bf(Scratch::SharedIntermediate, 640), diag, rows,
        bf(Scratch::SharedGate, 640), bf(Scratch::SharedUp, 640))) {
      affine(graph, mlp + ".shared_expert.gate_proj", mixed, bf(Scratch::SharedGate, 640));
      affine(graph, mlp + ".shared_expert.up_proj", mixed, bf(Scratch::SharedUp, 640));
      addSiLUMultiply(graph, bf(Scratch::SharedGate, 640), bf(Scratch::SharedUp, 640),
          bf(Scratch::SharedIntermediate, 640), diag, rows, 640);
    }
    affine(graph, mlp + ".shared_expert.down_proj", bf(Scratch::SharedIntermediate, 640),
        bf(Scratch::SharedDown, kWidth));
    affine(graph, mlp + ".shared_expert_gate", mixed, bf(Scratch::SharedGateLogit, 1));
    addCombine(graph, (compactR4Verify || (blocked && !gatheredMPP)) ? impl_->blockedScratch.scatteredDown : bf(Scratch::ExpertDown, kSelections * kWidth), ids, route,
        bf(Scratch::SharedDown, kWidth), bf(Scratch::SharedGateLogit, 1), branch,
        diag, rows, kWidth, 512, kSelections);
    const bool nextHasPLE = layer + 1 == impl_->descriptor.pleLayerIndices.front();
    const bool nextIsTerminalMixer = layer + 1 == impl_->descriptor.layers;
    const bool privatePrefillNextNorm =
        prefill_hc_inject_norm_sep21::nextNormEligible(privatePrefillHC, nextHasPLE, nextIsTerminalMixer);
    if (privatePrefillHC && nextHasPLE) prefill_hc_inject_norm_sep21::recordPLEExcluded();
    if (privatePrefillHC && nextIsTerminalMixer) prefill_hc_inject_norm_sep21::recordTerminalExcluded();
    normalizedReady = (impl_->fuseHC && rows <= 32 && !nextHasPLE) || privatePrefillNextNorm;
    if (normalizedReady) {
      const std::string nextHC = layer + 1 == impl_->descriptor.layers
          ? "language_model.model.hyper_connection_mixer"
          : "language_model.model.layers." + std::to_string(layer + 1) + ".attn_hyper_connection";
      const auto nextNorm = nextHC + ".hc_norm.weight";
      if (privatePrefillNextNorm) {
        prefill_hc_inject_norm_sep21::addPrivatePrefillHCInjectNormSep21(graph, hyper, branch,
            bf(Scratch::HCInjectionWeights, 4), impl_->weights.tensor(nextNorm), hyper,
            bf(Scratch::HCNormalized, kHyper), diag, hcGeometry, impl_->weights.normConvention(nextNorm));
        prefill_hc_inject_norm_sep21::recordMlpToNext();
      } else {
        addHCFusedInjectNorm(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4),
            impl_->weights.tensor(nextNorm), hyper, bf(Scratch::HCNormalized, kHyper), diag,
            hcGeometry, impl_->weights.normConvention(nextNorm));
      }
    } else {
      addHCInject(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4), hyper, hcGeometry);
    }
  }
  impl_->hc(graph, "language_model.model.hyper_connection_mixer", rows, false, normalizedReady, verification);
  const uint32_t logitRows = returnAllLogits ? rows : 1;
  const auto headInput = returnAllLogits ? mixed : impl_->backend.view(mixed,
      uint64_t{rows - 1} * kWidth * 2, uint64_t{kWidth} * 2);
  const auto logits = impl_->bf(Scratch::HeadLogits, logitRows, impl_->descriptor.vocabularySize);
  impl_->project(graph, "language_model.lm_head", headInput, logits, diag, logitRows);
  metal::MetalBuffer compactGreedy;
  if (impl_->gpuGreedy && logitRows <= kFlashGreedyGPUMaximumRows) {
    compactGreedy = impl_->backend.view(impl_->greedyResults, 0,
        uint64_t{logitRows} * sizeof(FlashGreedyGPURowResult));
    addGreedyGPU(graph, logits, impl_->greedyWorkspace, compactGreedy,
        logitRows, impl_->descriptor.vocabularySize);
  }
  graphTimer.reset();
  window->tokenIDs = tokenIDs; window->diag = diag; window->logits = logits;
  window->hyper = hyper; window->compactGreedy = compactGreedy;
  window->pleGeometry = pleGeometry; window->rows = rows; window->begin = begin;
  window->logitRows = logitRows; window->verification = verification;
  window->captureHidden = captureHidden; window->lazyPending = verification;
  window->serial = ++impl_->trunkSerial;
  lazyGuard.keep = true; // The window now owns any begun lazy trial.
  return window;
}

FlashForwardResult FlashForward::executeWindow(FlashRequestState &request, Window &window,
                                               std::span<const uint32_t> tokens) {
  struct LazyAbort {
    Impl *source; Window &window;
    ~LazyAbort() { if (window.lazyPending) { source->abortLazyRecords(); window.lazyPending = false; } }
  } lazyAbort{impl_.get(), window};
  if (!request.impl_ || request.impl_->owner != impl_->owner || request.impl_->poisoned)
    throw std::invalid_argument("Flash forward requires its own healthy request state");
  auto &state = *request.impl_;
  if (window.serial != impl_->trunkSerial || tokens.size() != window.rows ||
      state.length != window.begin || impl_->pendingRows)
    throw std::logic_error("Flash prepared window no longer matches the trunk state");
  const uint32_t rows = window.rows;
  auto *hostIDs = static_cast<int64_t *>(window.tokenIDs.contents());
  for (uint32_t index = 0; index < rows; ++index) {
    if (tokens[index] >= impl_->descriptor.vocabularySize)
      throw std::invalid_argument("Flash input contains an out-of-vocabulary token");
    hostIDs[index] = tokens[index];
  }
  clear(window.diag);
  if (impl_->pleSSD) {
    const auto *history = static_cast<const int64_t *>(state.pleHistory.contents());
    if (!history) throw std::logic_error("Flash PLE SSD request history is not Shared");
    opt::ScopedTimer pleTimer(rows <= 8 ? "forward.small.ple_prepare" : "forward.large.ple_prepare");
    impl_->pleSSD->prepare({hostIDs, rows}, {history, 2}, impl_->pleWeights, window.pleGeometry);
  }
  metal::CommandTiming timing;
  try {
    opt::ScopedTimer submitTimer(rows <= 8 ? "forward.small.submit_wait" : "forward.large.submit_wait");
    timing = impl_->backend.submitCommand(window.graph.dispatches());
    uint32_t status = 0;
    std::memcpy(&status, window.diag.contents(), sizeof(status));
    if (status) throw std::runtime_error("Flash forward sticky diagnostics failed: " + std::to_string(status));
  } catch (...) {
    if (impl_->pleSSD) impl_->pleSSD->consumePreparation();
    state.poisoned = true;
    throw;
  }
  if (impl_->pleSSD) impl_->pleSSD->consumePreparation();
  state.length += rows;
  impl_->capturedRows = impl_->capturedRoutes ? rows : 0;
  if (window.verification) {
    state.pendingVerification = true;
    impl_->pendingIdentity = state.identity;
    impl_->pendingBegin = window.begin;
    impl_->pendingRows = rows;
  }
  window.lazyPending = false;
  ++impl_->trunkSerial;
  return {timing, window.logits, window.logitRows, state.length, state.capacity,
      window.captureHidden ? window.hyper : metal::MetalBuffer{}, window.compactGreedy,
      window.compactGreedy ? window.logitRows : 0};
}

struct FlashPreparedVerify::Impl final {
  std::unique_ptr<FlashForward::Window> window;
  FlashForward::Impl *owner = nullptr;
};

FlashPreparedVerify::FlashPreparedVerify() = default;
FlashPreparedVerify::~FlashPreparedVerify() { discard(); }
FlashPreparedVerify::FlashPreparedVerify(FlashPreparedVerify &&) noexcept = default;
FlashPreparedVerify &FlashPreparedVerify::operator=(FlashPreparedVerify &&other) noexcept {
  if (this != &other) { discard(); impl_ = std::move(other.impl_); }
  return *this;
}
FlashPreparedVerify::operator bool() const noexcept { return impl_ && impl_->window; }
uint32_t FlashPreparedVerify::rows() const noexcept { return *this ? impl_->window->rows : 0; }
void FlashPreparedVerify::discard() noexcept {
  if (impl_ && impl_->window && impl_->window->lazyPending && impl_->owner) {
    std::lock_guard lock(impl_->owner->mutex);
    if (impl_->owner->trunkSerial == impl_->window->serial) {
      impl_->owner->abortLazyRecords();
      ++impl_->owner->trunkSerial;
    }
    impl_->window->lazyPending = false;
  }
  impl_.reset();
}

FlashPreparedVerify FlashForward::prepareVerify(FlashRequestState &request, uint32_t rows) {
  FlashPreparedVerify prepared;
  if (!impl_ || !rows) return prepared;
  prepared.impl_ = std::make_unique<FlashPreparedVerify::Impl>();
  prepared.impl_->owner = impl_.get();
  std::lock_guard lock(impl_->mutex);
  try {
    prepared.impl_->window = buildWindow(request, rows, true, true, true);
  } catch (...) {
    prepared.impl_.reset(); // buildWindow undid its partial trial; verify() rebuilds.
  }
  return prepared;
}

FlashForwardResult FlashForward::verifyPrepared(FlashRequestState &request,
    FlashPreparedVerify prepared, std::span<const uint32_t> tokens) {
  if (!impl_) throw std::logic_error("Flash forward is not initialized");
  {
    std::lock_guard lock(impl_->mutex);
    const bool usable = prepared && prepared.impl_->owner == impl_.get() &&
        prepared.impl_->window->rows == tokens.size() &&
        prepared.impl_->window->serial == impl_->trunkSerial && !impl_->pendingRows &&
        request.impl_ && request.impl_->owner == impl_->owner && !request.impl_->poisoned &&
        request.impl_->length == prepared.impl_->window->begin;
    if (usable) {
      auto window = std::move(prepared.impl_->window);
      prepared.impl_.reset();
      return executeWindow(request, *window, tokens);
    }
  }
  prepared.discard();
  return verify(request, tokens);
}

metal::CommandTiming FlashForward::commitVerify(FlashRequestState &request, uint32_t retained) {
  if (!impl_) throw std::logic_error("Flash forward is not initialized");
  std::lock_guard lock(impl_->mutex);
  ++impl_->trunkSerial;
  if (!request.impl_ || request.impl_->owner != impl_->owner || request.impl_->poisoned ||
      impl_->pendingIdentity.lock() != request.impl_->identity || !impl_->pendingRows ||
      !retained || retained > impl_->pendingRows)
    throw std::invalid_argument("Flash verification commit has no matching healthy prefix");
  auto &state = *request.impl_;
  if (state.length != impl_->pendingBegin + impl_->pendingRows)
    throw std::logic_error("Flash verification logical length changed before commit");
  struct LazyCommitGuard {
    Impl *source;
    FlashRequestState::Impl *state;
    bool keep = false;
    ~LazyCommitGuard() {
      if (!keep && source->lazyGDN) {
        state->poisoned = true; state->pendingVerification = false; source->finishPending();
      }
    }
  } lazyGuard{impl_.get(), &state};
  if (retained == impl_->pendingRows) {
    if (impl_->lazyGDN && impl_->pendingRows > 1) {
      metal::CommandGraph noWork;
      for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer)
        if (impl_->lazyGDNRecords[layer])
          impl_->lazyGDNRecords[layer]->commit(noWork, impl_->lazyGDNTickets[layer],
              std::span<const uint32_t>(&retained, 1));
      if (!noWork.empty()) throw std::logic_error("fully accepted lazy GDN must not replay");
    }
    state.pendingVerification = false; impl_->finishPending(); lazyGuard.keep = true; return {};
  }
  auto diag = impl_->view(Scratch::Diagnostics, 4);
  clear(diag);
  std::memcpy(impl_->retainedCount.contents(), &retained, sizeof(retained));
  metal::CommandGraph graph;
  for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
    if (impl_->descriptor.layerKinds[layer] != FlashLayerKind::GatedDeltaNet) continue;
    if (impl_->lazyGDN) {
      impl_->lazyGDNRecords[layer]->commit(graph, impl_->lazyGDNTickets[layer],
          std::span<const uint32_t>(&retained, 1));
    } else {
      impl_->copy(graph, impl_->prefixTape(false, layer, retained - 1), state.gdn[layer].convolution,
                    flashGDNConvolutionLaneBytes());
      impl_->copy(graph, impl_->prefixTape(true, layer, retained - 1), state.gdn[layer].recurrent,
                    flashGDNRecurrentLaneBytes());
    }
  }
  const uint32_t rows = impl_->pendingRows;
  const FlashPLEGeometry geometry{1, rows, kWidth, 4, impl_->descriptor.pleHistoryEos,
      impl_->descriptor.vocabularySize, static_cast<float>(impl_->descriptor.normEpsilon)};
  addPLERestorePrefix(graph, impl_->beforePLEHistory, impl_->view(Scratch::TokenIDs, uint64_t{rows} * 8),
      impl_->beforePLEConvolution, impl_->bf(Scratch::PLENormalizedConvolution, rows, kHyper),
      impl_->retainedCount, state.pleHistory, state.pleConvolution, diag, geometry);
  metal::CommandTiming timing;
  try {
    timing = impl_->backend.submitCommand(graph.dispatches());
    uint32_t status = 0; std::memcpy(&status, diag.contents(), sizeof(status));
    if (status) throw std::runtime_error("Flash prefix restore diagnostics failed: " + std::to_string(status));
  } catch (...) {
    state.poisoned = true; state.pendingVerification = false; impl_->finishPending(); throw;
  }
  state.length = impl_->pendingBegin + retained;
  state.pendingVerification = false;
  impl_->finishPending();
  lazyGuard.keep = true;
  return timing;
}

void FlashForward::abortVerify(FlashRequestState &request) noexcept {
  if (!impl_) return;
  std::lock_guard lock(impl_->mutex);
  ++impl_->trunkSerial;
  if (request.impl_ && impl_->pendingIdentity.lock() == request.impl_->identity) {
    request.impl_->poisoned = true; request.impl_->pendingVerification = false; impl_->finishPending();
  }
}

} // namespace splash::flash
