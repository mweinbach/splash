#!/usr/bin/env python3
"""Seal separate default-off physical R8/R16 target-only compact integer batch worker."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_batch_b4_composition_sep22')
QUALIFIED=Path('dev/benchmarks/expert_batch_compact_native_sep22')
CHANGED={'runtime/flash/FlashInt8ExpertStore.hpp','runtime/flash/FlashInt8ExpertStore.mm','runtime/flash/FlashBatchVerify.cpp','runtime/flash/FlashWorker.mm'}
OVERRIDES={'runtime/flash/FlashGDNBatchILP.cpp','runtime/flash/FlashWorker.mm'}
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def replace(text,before,after,count=1):
 if text.count(before)!=count:raise ValueError('Batch compact source anchor drift:'+before[:100])
 return text.replace(before,after)
def transform(relative,text):
 if relative in CHANGED:text=f'#include "{PRIVATE}/bridge.hpp"\n'+text
 if relative=='runtime/flash/FlashInt8ExpertStore.hpp':
  text=replace(text,'private:\n  struct Impl;', '''  [[nodiscard]] bool compactNativeBatchVerifyEnabled() const;
  [[nodiscard]] compact_native_batch_verify_sep22::Counters compactNativeBatchVerifyCounters() const;
  void addCompactNativeBatchVerifyPack(metal::CommandGraph &graph,uint32_t layer,
      metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &scratch,
      metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
  void addCompactNativeBatchVerifyGateUp(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
  void addCompactNativeBatchVerifyDown(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
private:
  struct Impl;''')
 if relative=='runtime/flash/FlashInt8ExpertStore.mm':
  text='#include "metal/abi/FlashMoEBuckets.h"\n'+text
  text=replace(text,'  const bool gatheredMPP = gathered_mpp::requested();', '''  const bool compactBatchVerify = compact_native_batch_verify_sep22::requested();
  mutable std::array<std::atomic<uint64_t>,2> compactPlanCalls{},compactPlanRows{},compactGateCalls{},compactGateRows{},compactDownCalls{},compactDownRows{};
  const bool gatheredMPP = gathered_mpp::requested();''')
  text=replace(text,'    numericalIdentity = hash(derivative.data(), derivative.size());', '''    if (compactBatchVerify)
      derivative += std::string("batch_r8r16_target_verify_execution_policy=")+compact_native_batch_verify_sep22::implementationMarker()+"\\n";
    numericalIdentity = hash(derivative.data(), derivative.size());''')
  methods='''bool FlashInt8ExpertStore::compactNativeBatchVerifyEnabled() const {
  if (!impl_) fail("compact batch verifier Store disposed");
  if (compact_native_batch_verify_sep22::requested()!=impl_->compactBatchVerify)
    fail("compact batch verifier flag changed after construction");
  if (impl_->compactBatchVerify && (!gatheredMPPEnabled()||gatheredMPPMaximumRows()!=4||!pointwise_sep21::requested()))
    fail("compact batch verifier requires frozen gathered cap exactly4 and native pointwiseM16");
  return impl_->compactBatchVerify;
}
compact_native_batch_verify_sep22::Counters FlashInt8ExpertStore::compactNativeBatchVerifyCounters() const {
  compact_native_batch_verify_sep22::Counters result;result.enabled=compactNativeBatchVerifyEnabled();
  auto read=[&](uint32_t i) {return compact_native_batch_verify_sep22::WidthCounters{
      impl_->compactPlanCalls[i].load(std::memory_order_relaxed),impl_->compactPlanRows[i].load(std::memory_order_relaxed),
      impl_->compactGateCalls[i].load(std::memory_order_relaxed),impl_->compactGateRows[i].load(std::memory_order_relaxed),
      impl_->compactDownCalls[i].load(std::memory_order_relaxed),impl_->compactDownRows[i].load(std::memory_order_relaxed)};};
  result.r8=read(0);result.r16=read(1);return result;
}
void FlashInt8ExpertStore::addCompactNativeBatchVerifyPack(metal::CommandGraph &graph,uint32_t index,
    metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &s,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeBatchVerifyEnabled()||(rows!=8&&rows!=16)||selections!=10) fail("compact batch verifier only physical R8/R16 S10");
  const auto &layer=impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size()!=512) fail("compact batch verifier requires Full512 inventory");
  allRowsScratch(s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  requireBytes(input,uint64_t{rows}*2560*2);requireBytes(ids,uint64_t{rows}*10*8);requireBytes(layer.ranks,512*4);
  disjoint(input,ids);
  for (const auto &b:{s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,s.buckets.packedInputs,
      s.buckets.jobOffsets,s.buckets.jobCount,s.buckets.tileJobs,s.packedActivated,s.scatteredDown,diagnostics}) {
    disjoint(input,b);disjoint(ids,b);impl_->immutableDisjoint(b);
  }
  impl_->immutableDisjoint(input);impl_->immutableDisjoint(ids);
  const uint32_t routes=rows*10,jobs=moEBucketJobCapacity(rows,10,16),counter=rows==8?0:1;
  graph.add(rows==8?"expert_batch_r8_compact_native_sep22_plan":"expert_batch_r16_compact_native_sep22_plan",
      {ids,s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,
      s.buckets.jobOffsets,s.buckets.tileJobs,s.buckets.jobCount,diagnostics},
      FlashMoEBucketParams{rows,10,2560,512,routes,16,jobs,0},{1,1,1},{256,1,1});
  graph.add("flash_moe_direct_a_pack",{input,s.buckets.offsets,s.buckets.routeMap,s.buckets.packedInputs,diagnostics},
      FlashMoEBucketParams{rows,10,2560,512,routes,0,0,0},{routes+63,1,1},{256,1,1});
  impl_->compactPlanCalls[counter].fetch_add(1,std::memory_order_relaxed);impl_->compactPlanRows[counter].fetch_add(rows,std::memory_order_relaxed);
}
void FlashInt8ExpertStore::addCompactNativeBatchVerifyGateUp(metal::CommandGraph &graph,uint32_t index,
    const FlashMoEBlockedScratch &s,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeBatchVerifyEnabled()||(rows!=8&&rows!=16)||selections!=10) fail("compact batch verifier gate only physical R8/R16 S10");
  addGateUp(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  const uint32_t counter=rows==8?0:1;impl_->compactGateCalls[counter].fetch_add(1,std::memory_order_relaxed);impl_->compactGateRows[counter].fetch_add(rows,std::memory_order_relaxed);
}
void FlashInt8ExpertStore::addCompactNativeBatchVerifyDown(metal::CommandGraph &graph,uint32_t index,
    const FlashMoEBlockedScratch &s,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeBatchVerifyEnabled()||(rows!=8&&rows!=16)||selections!=10) fail("compact batch verifier down only physical R8/R16 S10");
  addDownScatter(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  const uint32_t counter=rows==8?0:1;impl_->compactDownCalls[counter].fetch_add(1,std::memory_order_relaxed);impl_->compactDownRows[counter].fetch_add(rows,std::memory_order_relaxed);
}
'''
  text=replace(text,'} // namespace splash::flash',methods+'} // namespace splash::flash')
 if relative=='runtime/flash/FlashBatchVerify.cpp':
  text=replace(text,'    if (gatheredMPP) {', '''    const bool compactBatchVerify = impl_->allRowsInt8Target && impl_->int8ExpertStore &&
        compact_native_batch_verify_sep22::eligible(lanes,rows,true) && impl_->int8ExpertStore->compactNativeBatchVerifyEnabled();
    if (compactBatchVerify) {
      impl_->int8ExpertStore->addCompactNativeBatchVerifyPack(graph,layer,mixed,expertIDs,impl_->blockedScratch,diag,flattened,kSelections);
      impl_->int8ExpertStore->addCompactNativeBatchVerifyGateUp(graph,layer,impl_->blockedScratch,diag,flattened,kSelections);
      impl_->int8ExpertStore->addCompactNativeBatchVerifyDown(graph,layer,impl_->blockedScratch,diag,flattened,kSelections);
    } else if (gatheredMPP) {''')
 if relative=='runtime/flash/FlashWorker.mm':
  text=replace(text,'      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.', '''      compact_native_batch_verify_sep22::validateDependencies(environmentSwitch("SPLASH_FLASH_ALLROWS_FULL512_TARGET"),
          gathered_mpp::requested(),gathered_mpp::requestedMaximumRows(),environmentSwitch("SPLASH_FLASH_BLOCKED_MOE"),
          environmentSwitch("SPLASH_FLASH_MOE_DIRECT_A"),environmentSwitch("SPLASH_FLASH_MOE_Q4X8"),pointwise_sep21::requested());
      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.''')
  text=replace(text,'  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};', '''  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};
  const auto compactBatchGraphs=persistedExperts ? persistedExperts->compactNativeBatchVerifyCounters() : compact_native_batch_verify_sep22::Counters{};''')
  text=replace(text,'json::quote(forward_.kernelRoutes())','json::quote(forward_.kernelRoutes()+compact_native_batch_verify_sep22::implementationMarker())')
  text=replace(text,'      << R"(,"gdn_verification_storage":{"lazy_enabled":)"', '''      << R"(,"compact_native_batch_verify":{"schema":"parallel-integer-original-M16-six-stage-R8R16-target-verify-v1","scope":"batch target rows4 active lanes2/4 physicalR8/R16 only; graph construction not GPU completion","enabled":)"
      <<(compactBatchGraphs.enabled ? "true" : "false")<<R"(,"requested":)"<<(compact_native_batch_verify_sep22::requested() ? "true" : "false")
      <<R"(,"source_identity_sha256":)"<<json::quote(compact_native_batch_verify_sep22::kSourceIdentitySha256)
      <<R"(,"dispatches_per_layer":6,"base_native_dispatches_per_layer":10,"additional_gpu_allocation_bytes":0,"full_model_quality_qualified":false,"r8":{"physical_rows":8,"planner_threadgroup_bytes":2752,"plan_graph_calls":)"
      <<compactBatchGraphs.r8.planCalls<<R"(,"plan_graph_rows":)"<<compactBatchGraphs.r8.planRows<<R"(,"gate_graph_calls":)"<<compactBatchGraphs.r8.gateCalls
      <<R"(,"gate_graph_rows":)"<<compactBatchGraphs.r8.gateRows<<R"(,"down_graph_calls":)"<<compactBatchGraphs.r8.downCalls<<R"(,"down_graph_rows":)"<<compactBatchGraphs.r8.downRows
      <<R"(},"r16":{"physical_rows":16,"planner_threadgroup_bytes":3392,"plan_graph_calls":)"
      <<compactBatchGraphs.r16.planCalls<<R"(,"plan_graph_rows":)"<<compactBatchGraphs.r16.planRows<<R"(,"gate_graph_calls":)"<<compactBatchGraphs.r16.gateCalls
      <<R"(,"gate_graph_rows":)"<<compactBatchGraphs.r16.gateRows<<R"(,"down_graph_calls":)"<<compactBatchGraphs.r16.downCalls<<R"(,"down_graph_rows":)"<<compactBatchGraphs.r16.downRows<<"}}"
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
 return text
