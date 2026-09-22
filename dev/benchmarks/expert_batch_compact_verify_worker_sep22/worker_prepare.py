#!/usr/bin/env python3
"""Seal separate default-off physical R8/R16 target-only compact integer batch worker."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_batch_compact_verify_worker_sep22')
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

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--math-parent',type=Path,default=ROOT/'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5');p.add_argument('--clock-parent',type=Path,default=ROOT/'build/current-batch-native-clock-sep22-v5');p.add_argument('--output',type=Path,default=ROOT/'build/compact-native-batch-verify-clock-sep22-worker-v1');a=p.parse_args();base=a.math_parent.resolve();clock=a.clock_parent.resolve();out=a.output.resolve()
 if out.exists()or ROOT/'build'not in out.parents:raise ValueError('NEW private worker output required')
 pm=base/'overlay-manifest.json';parent=json.loads(pm.read_text());clockseal=json.loads((clock/'compiled-cpu-seal.json').read_text())
 if not clockseal.get('pass')or clockseal.get('qualified_math_parent')!=str(base)or not clockseal.get('retained_clock_Worker_source_and_object_identical'):raise ValueError('Expected guarded nativeclock v5 authenticated math ancestry')
 clockexpected={'runtime/flash/FlashWorker.mm':clockseal['worker_source_sha256'],'runtime/flash/FlashGDNBatchILP.cpp':clockseal['guard_source_sha256']}
 for rel,digest in clockexpected.items():
  if sha((clock/'source'/rel).read_bytes())!=digest:raise ValueError('Nativeclock source drift:'+rel)
 plans={};qs={}
 for rows in (8,16):
  q=ROOT/f'build/expert-batch-r{rows}-compact-native-sep22-component-v2';seal=json.loads((q/'source-seal.json').read_text());review=json.loads((q/'independent-source-review-v2.json').read_text())
  if not review.get('pass')or review.get('source_identity_sha256')!=seal['source_identity_sha256']:raise ValueError('Reviewed qualified batch planner closure required')
  record=next(r for r in seal['sources']if r['path']==str(QUALIFIED/'plan.metal'));plans[rows]=(q/'source'/QUALIFIED/'plan.metal').read_bytes()
  if sha(plans[rows])!=record['sha256']:raise ValueError('Qualified planner drift')
  qs[str(rows)]=seal['source_identity_sha256']
 if plans[8]!=plans[16]:raise ValueError('Both qualified fixed variants must have same parameterized integer source')
 parts={'qualified_plan':sha(plans[8]),'qualified_components':qs,'bridge':sha((ROOT/PRIVATE/'bridge.hpp').read_bytes()),'worker_transform':sha(Path(__file__).read_bytes()),'parent_manifest':sha(pm.read_bytes()),'clock_parent_seal':sha((clock/'compiled-cpu-seal.json').read_bytes()),'scope':'batch-target-verification-rows4-actual-active-lanes2or4-physicalR8R16-Full512-cap4-pointwise-nativeM16-sixstage-v1'};identity=sha(json.dumps(parts,sort_keys=True,separators=(',',':')).encode());files=[]
 for r in parent['files']:
  rel=r['path'];original=(base/'source'/rel).read_bytes();digest=r.get('sha256')or r.get('overlay_sha256')
  if sha(original)!=digest:raise ValueError('Math parent source drift:'+rel)
  data=(clock/'source'/rel).read_bytes()if rel in OVERRIDES else original
  changed=transform(rel,data.decode()).encode();write(out/'source'/rel,changed);files.append({'path':rel,'parent_sha256':sha(original),'effective_parent_sha256':sha(data),'sha256':sha(changed),'changed':data!=changed,'nativeclock_override':rel in OVERRIDES})
 header=Path('dev/benchmarks/current_batch_sep22/NativeLifecycleTrace.hpp');data=(clock/'source'/header).read_bytes()
 if sha(data)!=clockseal['native_trace_header_sha256']:raise ValueError('Nativeclock header drift')
 write(out/'source'/header,data);files.append({'path':str(header),'sha256':sha(data),'nativeclock_override':True})
 for name in('bridge.hpp','policy_cpu.cpp','worker_prepare.py','worker.mk','worker_witness.py','counter_audit.py','INTEGRATION.md'):
  rel=PRIVATE/name;data=(ROOT/rel).read_bytes();write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'new':True})
 for rows in (8,16):
  rel=PRIVATE/f'plan-r{rows}.metal';data=plans[rows].replace(b'expert_batch_compact_native_sep22_plan',f'expert_batch_r{rows}_compact_native_sep22_plan'.encode());write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'qualified_copy_export_renamed_only':True})
 rel=PRIVATE/'abi.hpp';data=(ROOT/'build/expert-batch-r8-compact-native-sep22-component-v2/source'/QUALIFIED/'abi.hpp').read_bytes();write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'qualified_copy':True})
 rel=PRIVATE/'source_identity.hpp';data=f'#pragma once\nnamespace splash::flash::compact_native_batch_verify_sep22 {{inline constexpr char kSourceIdentitySha256[]="{identity}";}}\n'.encode();write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'generated':True})
 rows=parent['rebuild'];names=[r['object']for r in rows]+['teacher_bulk'];link='REBUILD_NAMES := '+' '.join(names)+'\n'
 for r in rows:link+=f'SRC_{r["object"]} := $(BUILD)/source/{r["source"]}\n'
 link+='SRC_teacher_bulk := $(BUILD)/source/dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp\n';frozen=[];core=[]
 for r in parent['frozen_objects']:
  src=base/r['path'];data=src.read_bytes()
  if sha(data)!=r['sha256']:raise ValueError('Parent Core drift')
  rel=Path('reused/core')/src.name;write(out/rel,data);core.append(str(rel));frozen.append({'path':str(rel),'sha256':sha(data),'source':str(src)})
 link+='CORE := '+' '.join('$(BUILD)/'+r for r in core)+'\n';ancestor=Path(parent['base'])
 if (ancestor/'splash.metallib').read_bytes()!=(base/'splash.metallib').read_bytes()or(base/'splash.metallib').read_bytes()!=(clock/'splash.metallib').read_bytes():raise ValueError('Shader ancestor/teacher/nativeclock libraries differ')
 airs=[]
 for line in(ancestor/'link-inputs.mk').read_text().splitlines():
  if not line.startswith('AIRS := '):continue
  for token in line.split(' := ',1)[1].split():
   src=ancestor/Path(token[9:]);rel=Path('reused/air')/src.name;data=src.read_bytes();write(out/rel,data);airs.append(str(rel));frozen.append({'path':str(rel),'sha256':sha(data),'source':str(src)})
 src=ancestor/'candidate.air';rel=Path('reused/air/ab-merge-candidate.air');data=src.read_bytes();write(out/rel,data);airs.append(str(rel));frozen.append({'path':str(rel),'sha256':sha(data),'source':str(src)})
 link+='AIRS := '+' '.join('$(BUILD)/'+r for r in airs)+'\n';write(out/'link-inputs.mk',link.encode())
 manifest={'schema':'compact-native-R8R16-target-verify-nativeclock-worker-source-v1','base':str(base),'clock_parent':str(clock),'parent_manifest_sha256':sha(pm.read_bytes()),'files':files,'rebuild':rows,'all_noncore_host_consumers_rebuilt':True,'frozen_inputs':frozen,'link_make_sha256':sha(link.encode()),'qualified_parallel_source_identities':qs,'source_identity_sha256':identity,'identity_parts':parts,'changed_paths':sorted(CHANGED),'nativeclock_overrides':sorted(OVERRIDES),'additional_gpu_allocation_bytes':0,'component_all13_patterns_exact_root_confirmed':True,'full_model_quality_qualified':False,'gpu_work':False,'model_payload_reads':False,'capture_payload_reads':False}
 write(out/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps({'prepared':str(out),'source_identity_sha256':identity,'sources':len(files),'host_rebuild_tus':len(names),'gpu_work':False}))
if __name__=='__main__':main()
