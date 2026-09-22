#pragma once
#include "flash/FlashExpertDenseCache.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashMoEDirectA.h"
#include <cstring>
#include <stdexcept>

namespace splash::flash::prefill4k_bf16 {
inline std::string pipeline(const char *phase, FlashMoEBlockedTile tile) {
  const uint32_t m = uint32_t(tile);
  if (m !=16 && m !=32 && m !=64) throw std::invalid_argument("BF16 bridge tile must be16/32/64");
  return std::string("prefill4k_bf16cache_") +phase +"_m" +std::to_string(m) +(m ==64 ? "_n64_sg8" : "_n64");
}
template<class P> inline P parameter(const metal::ComputeDispatch &d, uint32_t index) {
  if (d.bytes.size() !=1 || d.bytes[0].index !=index || d.bytes[0].sizeBytes !=sizeof(P) || !d.bytes[0].data)
    throw std::invalid_argument("BF16 bridge producer parameter contract differs");
  P value; std::memcpy(&value,d.bytes[0].data,sizeof(value)); return value;
}
inline void addGate(metal::CommandGraph &graph, const FlashExpertDenseCache &cache,
    const FlashAffineProjection &gate, const FlashAffineProjection &up,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diag,
    uint32_t rows, FlashMoEBlockedTile tile) {
  if (!flashMoEDirectAEnabled()) throw std::invalid_argument("BF16 bridge requires current Direct-A policy");
  metal::CommandGraph validated;
  addMoEBlockedGateUp(validated,gate,up,s,diag,rows,tile);
  if (validated.dispatches().size() !=1 || !validated.dispatches()[0].pipelineName.starts_with("flash_moe_direct_a_gate_up_"))
    throw std::invalid_argument("BF16 bridge requires unique current gate producer");
  const auto blocked = parameter<FlashMoEBlockedGateParams>(validated.dispatches()[0],12);
  const uint32_t hot = uint32_t(cache.selectedExpertIDs().size());
  const FlashInt8ExpertStoreParams hit{rows,10,rows *10,blocked.job_capacity,uint32_t(tile),hot,0,0};
  const FlashInt8ExpertStoreGateParams miss{blocked,hot,1,0,0};
  const uint32_t threads = tile ==FlashMoEBlockedTile::M64N64 ? 256 :128;
  const auto &g = cache.cachedProjection(FlashExpertCachePlane::Gate).buffer;
  const auto &u = cache.cachedProjection(FlashExpertCachePlane::Up).buffer;
  const auto &ranks = cache.expertRanks();
  // Slots2/4 are unused legacy scale bindings in the private shader. Rank
  // storage supplies a retained readonly view; no F32 scale is loaded/applied.
  graph.add(pipeline("gate_up",tile),{s.buckets.packedInputs,g,ranks,u,ranks,ranks,
      s.buckets.offsets,s.buckets.tileJobs,s.buckets.jobCount,s.packedActivated,diag},hit,
      {10,blocked.job_capacity,1},{threads,1,1});
  graph.add(pipeline("gate_up_miss_direct",tile),{s.buckets.packedInputs,gate.weights->buffer,
      gate.scales->buffer,gate.biases->buffer,up.weights->buffer,up.scales->buffer,up.biases->buffer,
      s.buckets.offsets,s.buckets.tileJobs,s.buckets.jobCount,s.packedActivated,diag,ranks},miss,
      {10,blocked.job_capacity,1},{threads,1,1});
}
inline void addDown(metal::CommandGraph &graph, const FlashExpertDenseCache &cache,
    const FlashAffineProjection &down, const FlashMoEBlockedScratch &s,
    metal::MetalBuffer diag, uint32_t rows, FlashMoEBlockedTile tile) {
  if (!flashMoEDirectAEnabled()) throw std::invalid_argument("BF16 bridge requires current Direct-A policy");
  metal::CommandGraph validated;
  addMoEBlockedDownScatter(validated,down,s,diag,rows,tile);
  const metal::ComputeDispatch *producer = nullptr;
  for (const auto &d : validated.dispatches()) {
    if (d.pipelineName.starts_with("flash_moe_direct_a_down_scatter_")) {
      if (producer) throw std::invalid_argument("BF16 bridge duplicate down producer");
      producer =&d;
    } else if (d.pipelineName =="flash_moe_blocked_poison_excluded_routes") {
      graph.add(d.pipelineName,{s.buckets.canonicalToPacked,s.scatteredDown,diag},
          parameter<FlashMoEBlockedDownParams>(d,3),d.threadgroups,d.threadsPerThreadgroup);
    } else if (d.pipelineName =="flash_moe_direct_a_prepare_down") {
      graph.add(d.pipelineName,{s.packedActivated,s.buckets.offsets,s.packedActivated,diag},
          parameter<FlashMoEDirectAPrepareParams>(d,4),d.threadgroups,d.threadsPerThreadgroup);
    } else throw std::invalid_argument("BF16 bridge unexpected preparatory down dispatch");
  }
  if (!producer) throw std::invalid_argument("BF16 bridge missing current down producer");
  const auto blocked = parameter<FlashMoEBlockedDownParams>(*producer,10);
  const uint32_t hot = uint32_t(cache.selectedExpertIDs().size());
  const FlashInt8ExpertStoreParams hit{rows,10,rows *10,blocked.job_capacity,uint32_t(tile),hot,0,0};
  const FlashInt8ExpertStoreDownParams miss{blocked,hot,1,0,0};
  const uint32_t threads = tile ==FlashMoEBlockedTile::M64N64 ? 256 :128;
  const auto &weights = cache.cachedProjection(FlashExpertCachePlane::Down).buffer;
  const auto &ranks = cache.expertRanks();
  graph.add(pipeline("down_scatter",tile),{s.packedActivated,weights,ranks,ranks,s.buckets.offsets,
      s.buckets.tileJobs,s.buckets.jobCount,s.buckets.routeMap,s.scatteredDown,diag},hit,
      {40,blocked.job_capacity,1},{threads,1,1});
  graph.add(pipeline("down_miss_direct",tile),{s.packedActivated,down.weights->buffer,
      down.scales->buffer,down.biases->buffer,s.buckets.offsets,s.buckets.tileJobs,
      s.buckets.jobCount,s.buckets.routeMap,s.scatteredDown,diag,ranks},miss,
      {40,blocked.job_capacity,1},{threads,1,1});
}
} // namespace splash::flash::prefill4k_bf16
