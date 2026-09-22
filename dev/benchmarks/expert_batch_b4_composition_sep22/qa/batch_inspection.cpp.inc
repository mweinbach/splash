// Private oracle clone only. The original batch methods above are unchanged.
#include "metal/abi/FlashMoEBuckets.h"
#include <unordered_map>
#include <unordered_set>
namespace splash::flash {
namespace {std::unordered_map<const void *,uint64_t> gOracleBatchGuardDeltas;}
std::vector<FlashDeepPrefixOracle::Plane> FlashDeepPrefixOracle::batchMoEPlanes(const FlashBatchVerify &batch){
  if(!batch.impl_)throw std::logic_error("batch inspection target unavailable");
  const auto &s=*batch.impl_;const auto &b=s.blockedScratch.buckets;
  if(b.rowCapacity!=16||b.selectionCapacity!=10||b.routeCapacity!=160||b.jobCapacity!=531)
    throw std::logic_error("actual admitted batch arena is not max16/531");
  std::vector<Plane> out;
  const auto add=[&](const char *name,Type type,const metal::MetalBuffer &buffer,uint64_t live){
    if(!buffer||!buffer.contents()||live>buffer.sizeBytes())throw std::logic_error("batch MoE plane extent invalid");
    out.push_back({name,type,buffer,live});
  };
  add("moe.counts",Type::U32,b.counts,512*4);
  add("moe.offsets",Type::U32,b.offsets,513*4);
  add("moe.route_map",Type::U32,b.routeMap,160*4);
  add("moe.inverse",Type::U32,b.canonicalToPacked,160*4);
  add("moe.packed_inputs",Type::BF16,b.packedInputs,223ULL*2560*2);
  add("moe.job_offsets",Type::U32,b.jobOffsets,513*4);
  add("moe.job_count",Type::U32,b.jobCount,4);
  add("moe.jobs",Type::U32,b.tileJobs,531ULL*sizeof(FlashMoEBucketJob));
  add("moe.prepared_gu",Type::BF16,s.blockedScratch.packedActivated,223ULL*640*2);
  add("moe.canonical_down",Type::BF16,s.blockedScratch.scatteredDown,160ULL*2560*2);
  return out;
}
std::vector<FlashDeepPrefixOracle::Plane> FlashDeepPrefixOracle::batchPlanes(const FlashBatchVerify &batch){
  if(!batch.impl_)throw std::logic_error("batch inspection target unavailable");
  const auto &s=*batch.impl_;auto out=batchMoEPlanes(batch);
  for(size_t i=0;i<s.scratch.size();++i){
    const auto &b=s.scratch[i];if(!b||!b.contents())throw std::logic_error("batch scratch inaccessible");
    // Scratch has allocator padding and mixed integer roles; finite checks are
    // performed on borrowed outputs and all persistent active numeric state.
    out.push_back({"batch.scratch."+std::to_string(i),Type::U32,b,b.sizeBytes()});
  }
  const auto temporary=[&](const char *name,const metal::MetalBuffer &buffer){
    if(!buffer||(buffer.storage()==metal::BufferStorage::Shared&&!buffer.contents())){
      std::ostringstream error;error<<"batch auxiliary scratch inaccessible: plane="<<name
        <<" present="<<bool(buffer)<<" bytes="<<buffer.sizeBytes()<<" CPU_visible="<<bool(buffer.contents())
        <<" storage="<<(buffer.storage()==metal::BufferStorage::Private?"Private":"Shared")
        <<" onlineQSA="<<s.onlineQSA<<" mppQSA="<<s.mppQSA
        <<" qsa.maximumRows="<<s.qsa.maximumRows<<" qsa.capacity="<<s.qsa.capacity
        <<" fast.maximumRows="<<s.qsaFast.maximumRows<<" fast.maximumPartitions="<<s.qsaFast.maximumPartitions
        <<" auxiliary_inventory_before_seed=[";
      bool comma=false;const auto describe=[&](const char *label,const metal::MetalBuffer &b){
        if(comma)error<<',';comma=true;error<<label<<"{present:"<<bool(b)<<",bytes:"<<b.sizeBytes()<<",CPU_visible:"<<bool(b.contents())
          <<",storage:"<<(b.storage()==metal::BufferStorage::Private?"Private":"Shared")<<'}';};
      describe("qsa.queries",s.qsa.queries);describe("qsa.index_queries",s.qsa.indexQueries);
      describe("qsa.block_scores",s.qsa.blockScores);describe("qsa.selected_blocks",s.qsa.selectedBlocks);
      describe("qsa.attention_scores",s.qsa.attentionScores);describe("qsa.probabilities",s.qsa.probabilities);
      describe("qsa.partition_statistics",s.qsaFast.partitionStatistics);describe("qsa.partition_values",s.qsaFast.partitionValues);
      describe("greedy.partials",s.greedyWorkspace.partials);describe("greedy.results",s.greedyResults);
      if(s.pleSSD){uint32_t i=0;for(const auto &b:s.pleSSD->scratchBuffers())describe(("ple.scratch."+std::to_string(i++)).c_str(),b);}
      error<<']';throw std::logic_error(error.str());
    }
    out.push_back({name,Type::U32,buffer,buffer.sizeBytes()});
  };
  temporary("batch.qsa.queries",s.qsa.queries);temporary("batch.qsa.index_queries",s.qsa.indexQueries);
  temporary("batch.qsa.block_scores",s.qsa.blockScores);temporary("batch.qsa.selected_blocks",s.qsa.selectedBlocks);
  temporary("batch.qsa.attention_scores",s.qsa.attentionScores);temporary("batch.qsa.probabilities",s.qsa.probabilities);
  if(s.onlineQSA){temporary("batch.qsa.partition_statistics",s.qsaFast.partitionStatistics);temporary("batch.qsa.partition_values",s.qsaFast.partitionValues);}
  if(s.greedyWorkspace.partials)temporary("batch.greedy_partials",s.greedyWorkspace.partials);
  if(s.pleSSD){uint32_t i=0;for(const auto &buffer:s.pleSSD->scratchBuffers()){
    const auto name="batch.ple_ssd."+std::to_string(i++);out.push_back({name,Type::U32,buffer,buffer.sizeBytes()});}}
  if(!s.lazyGDN)throw std::logic_error("batch proof requires original lazy rollback");
  const std::array<Type,6> types{Type::F32,Type::BF16,Type::BF16,Type::BF16,Type::F32,Type::BF16};
  const std::array<const char *,6> names{"initial_state","initial_history","raw_qkv","mixed","decay","beta"};
  uint32_t layers=0;
  for(uint32_t layer=0;layer<48;++layer)if(s.lazyGDNLayers[layer]){
    ++layers;const auto &r=*s.lazyGDNLayers[layer];const auto arenas=r.arenaBuffers();
    if(r.maximumRows()!=4||r.maximumLanes()!=4||arenas.size()!=6)throw std::logic_error("batch lazy shape drift");
    for(size_t i=0;i<6;++i)out.push_back({"batch.gdn."+std::to_string(layer)+"."+names[i],types[i],arenas[i],arenas[i].sizeBytes()});
  }
  if(layers!=36)throw std::logic_error("batch GDN inventory differs");
  if(s.greedyResults)out.push_back({"batch.greedy_backing",Type::U32,s.greedyResults,s.greedyResults.sizeBytes()});
  uint64_t bytes=0,raw=0,roundedBytes=0;std::unordered_set<uintptr_t> owners;
  for(const auto &p:out){const auto owner=p.buffer.oracleOwnerIdentitySep22();
    if(!owner||!owners.insert(owner).second)throw std::logic_error("batch physical inventory contains a missing/duplicate native owner");
    bytes+=p.buffer.oracleChargedBytesSep22();raw+=p.buffer.sizeBytes();roundedBytes+=(p.buffer.sizeBytes()+16383)&~uint64_t{16383};}
  const auto guard=gOracleBatchGuardDeltas.find(batch.impl_.get());const auto guardDelta=guard==gOracleBatchGuardDeltas.end()?0:guard->second;
  if(bytes!=s.allocatedBytes+guardDelta){std::ostringstream error;error<<"complete batch owner charge inventory !=workspace: actual_owner_charges="<<bytes
    <<" original_workspace="<<s.allocatedBytes<<" actual_guard_delta="<<guardDelta<<" legal_lengths="<<raw<<" rounded_lengths="<<roundedBytes;
    throw std::logic_error(error.str());}
  return out;
}
std::string FlashDeepPrefixOracle::batchMetadata(const FlashBatchVerify &batch){
  if(!batch.impl_)throw std::logic_error("batch inspection target unavailable");
  const auto &s=*batch.impl_;std::ostringstream out;
  out<<"{\"capacity\":"<<s.capacity<<",\"maximum_lanes\":"<<s.maximumLanes<<",\"maximum_rows\":"<<s.maximumRows
    <<",\"pending_rows\":"<<s.pendingRows<<",\"pending_lanes\":[";
  for(size_t i=0;i<s.pendingLanes.size();++i){if(i)out<<',';const auto &p=s.pendingLanes[i];const bool live=!p.identity.expired();
    out<<"{\"begin\":"<<p.begin<<",\"identity_live\":"<<(live?"true":"false")<<'}';}
  out<<"],\"lazy_tickets\":[";for(uint32_t i=0;i<48;++i){if(i)out<<',';out<<s.lazyGDNTickets[i];}
  out<<"],\"arena\":{\"routes\":160,\"jobs\":531,\"operand_rows\":223}}";return out.str();
}
bool FlashDeepPrefixOracle::batchCanaries(const FlashBatchVerify &batch){
  if(!batch.impl_)return false;for(const auto &r:batch.impl_->lazyGDNLayers)if(r&&!r->canariesIntact())return false;return true;
}
uint64_t FlashDeepPrefixOracle::batchGuardPlannedDelta(const FlashBatchVerify &batch){
  uint64_t delta=0;for(const auto &p:batchMoEPlanes(batch)){
    const auto newLength=((p.buffer.sizeBytes()+16383)&~uint64_t{16383})+16384;
    const auto oldCharge=p.buffer.oracleChargedBytesSep22();if(newLength<oldCharge)throw std::logic_error("guarded base plan below original owner charge");
    delta+=newLength-oldCharge;}
  if(delta>guardedBatchAllowance())throw std::logic_error("actual guard plan exceeds admitted source-derived maximum");return delta;
}
std::vector<FlashDeepPrefixOracle::Guard> FlashDeepPrefixOracle::guardBatchMoE(metal::MetalBackend &backend,FlashBatchVerify &batch){
  if(!batch.impl_)throw std::logic_error("batch guard target unavailable");
  auto &s=*batch.impl_;auto &b=s.blockedScratch.buckets;std::vector<Guard> guards;uint64_t actualDelta=0;
  const auto planned=batchGuardPlannedDelta(batch);
  const auto wrap=[&](metal::MetalBuffer &buffer){const auto bytes=buffer.sizeBytes();
    if(!buffer||!buffer.contents()||!bytes)throw std::logic_error("batch guard source unavailable");
    // One explicit additional 16 KiB page per independently owned plane.
    // This keeps the diagnostic guard delta exactly planned even when an
    // original logical length was not page aligned.
    const auto oldCharge=buffer.oracleChargedBytesSep22(),newLength=((bytes+16383)&~uint64_t{16383})+16384;
    auto base=backend.allocateBuffer(newLength,metal::BufferStorage::Shared,"oracle batch MoE redzones");
    if(base.oracleChargedBytesSep22()!=newLength)throw std::logic_error("native guarded owner charge differs from aligned base plan");
    actualDelta+=base.oracleChargedBytesSep22()-oldCharge;
    std::memset(base.contents(),0x5a,base.sizeBytes());auto view=backend.view(base,64,bytes);
    std::memcpy(view.contents(),buffer.contents(),bytes);buffer=view;guards.push_back({base,view,bytes});};
  wrap(b.counts);wrap(b.offsets);wrap(b.routeMap);wrap(b.canonicalToPacked);wrap(b.packedInputs);
  wrap(b.jobOffsets);wrap(b.jobCount);wrap(b.tileJobs);wrap(s.blockedScratch.packedActivated);wrap(s.blockedScratch.scatteredDown);
  if(guards.size()!=10||actualDelta!=planned)throw std::logic_error("batch redzone census/actual planned charge differs");
  gOracleBatchGuardDeltas[batch.impl_.get()]=actualDelta;return guards;
}
}
