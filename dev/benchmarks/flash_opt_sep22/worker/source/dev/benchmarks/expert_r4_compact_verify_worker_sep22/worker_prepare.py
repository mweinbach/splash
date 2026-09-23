#!/usr/bin/env python3
"""Seal qualified parallel INTEGER planner for singleton-main R4 verify only."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_r4_compact_verify_worker_sep22')
QUALIFIED=Path('dev/benchmarks/expert_r4_compact_native_parallel_sep22')
CHANGED={'runtime/flash/FlashInt8ExpertStore.hpp','runtime/flash/FlashInt8ExpertStore.mm','runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm'}
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def replace(text,before,after,count=1):
 if text.count(before)!=count:raise ValueError('Compact verify source anchor drift:'+before[:100])
 return text.replace(before,after)
def transform(relative,text):
 if relative in CHANGED:text=f'#include "{PRIVATE}/bridge.hpp"\n'+text
 if relative=='runtime/flash/FlashInt8ExpertStore.hpp':
  text=replace(text,'private:\n  struct Impl;', '''  [[nodiscard]] bool compactNativeR4VerifyEnabled() const;
  [[nodiscard]] compact_native_r4_verify_sep22::Counters compactNativeR4VerifyCounters() const;
  void addCompactNativeR4VerifyPack(metal::CommandGraph &graph,uint32_t layer,
      metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &scratch,
      metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
  void addCompactNativeR4VerifyGateUp(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
  void addCompactNativeR4VerifyDown(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
private:
  struct Impl;''')
 if relative=='runtime/flash/FlashInt8ExpertStore.mm':
  text=f'#include "{PRIVATE}/abi.hpp"\n'+text
  text=replace(text,'  const bool gatheredMPP = gathered_mpp::requested();', '''  const bool compactR4Verify = compact_native_r4_verify_sep22::requested();
  mutable std::atomic<uint64_t> compactR4PlanCalls{0},compactR4PlanRows{0},compactR4GateCalls{0},compactR4GateRows{0},compactR4DownCalls{0},compactR4DownRows{0};
  const bool gatheredMPP = gathered_mpp::requested();''')
  text=replace(text,'    numericalIdentity = hash(derivative.data(), derivative.size());', '''    if (compactR4Verify)
      derivative += std::string("singleton_r4_verify_execution_policy=")+compact_native_r4_verify_sep22::implementationMarker()+"\\n";
    numericalIdentity = hash(derivative.data(), derivative.size());''')
  methods='''bool FlashInt8ExpertStore::compactNativeR4VerifyEnabled() const {
  if (!impl_) fail("compact R4 verifier Store disposed");
  if (compact_native_r4_verify_sep22::requested()!=impl_->compactR4Verify)
    fail("compact R4 verifier flag changed after construction");
  if (impl_->compactR4Verify && (!gatheredMPPEnabled()||gatheredMPPMaximumRows()!=4))
    fail("compact R4 verifier requires original gathered cap exactly4");
  return impl_->compactR4Verify;
}
compact_native_r4_verify_sep22::Counters FlashInt8ExpertStore::compactNativeR4VerifyCounters() const {
  return {compactNativeR4VerifyEnabled(),impl_->compactR4PlanCalls.load(std::memory_order_relaxed),impl_->compactR4PlanRows.load(std::memory_order_relaxed),
      impl_->compactR4GateCalls.load(std::memory_order_relaxed),impl_->compactR4GateRows.load(std::memory_order_relaxed),
      impl_->compactR4DownCalls.load(std::memory_order_relaxed),impl_->compactR4DownRows.load(std::memory_order_relaxed)};
}
void FlashInt8ExpertStore::addCompactNativeR4VerifyPack(metal::CommandGraph &graph,uint32_t index,
    metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &s,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeR4VerifyEnabled()||rows!=4||selections!=10) fail("compact verifier only canonical R4/S10");
  const auto &layer=impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size()!=512) fail("compact verifier requires Full512 inventory");
  allRowsScratch(s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  requireBytes(input,uint64_t{4}*2560*2);requireBytes(ids,40*8);requireBytes(layer.ranks,512*4);
  disjoint(input,ids);
  for (const auto &b:{s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,s.buckets.packedInputs,
      s.buckets.jobOffsets,s.buckets.jobCount,s.buckets.tileJobs,s.packedActivated,s.scatteredDown,diagnostics}) {
    disjoint(input,b);disjoint(ids,b);impl_->immutableDisjoint(b);
  }
  impl_->immutableDisjoint(input);impl_->immutableDisjoint(ids);
  graph.add("expert_r4_compact_native_sep22_plan",{ids,s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,
      s.buckets.jobOffsets,s.buckets.tileJobs,s.buckets.jobCount,diagnostics},FlashMoEBucketParams{4,10,2560,512,40,16,514,0},{1,1,1},{256,1,1});
  graph.add("flash_moe_direct_a_pack",{input,s.buckets.offsets,s.buckets.routeMap,s.buckets.packedInputs,diagnostics},
      FlashMoEBucketParams{4,10,2560,512,40,0,0,0},{103,1,1},{256,1,1});
  impl_->compactR4PlanCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4PlanRows.fetch_add(rows,std::memory_order_relaxed);
}
void FlashInt8ExpertStore::addCompactNativeR4VerifyGateUp(metal::CommandGraph &graph,uint32_t index,
    const FlashMoEBlockedScratch &s,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeR4VerifyEnabled()||rows!=4||selections!=10) fail("compact verifier gate only R4/S10");
  addGateUp(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  impl_->compactR4GateCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4GateRows.fetch_add(rows,std::memory_order_relaxed);
}
void FlashInt8ExpertStore::addCompactNativeR4VerifyDown(metal::CommandGraph &graph,uint32_t index,
    const FlashMoEBlockedScratch &s,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeR4VerifyEnabled()||rows!=4||selections!=10) fail("compact verifier down only R4/S10");
  addDownScatter(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  impl_->compactR4DownCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4DownRows.fetch_add(rows,std::memory_order_relaxed);
}
'''
  text=replace(text,'} // namespace splash::flash',methods+'} // namespace splash::flash')
 if relative=='runtime/flash/FlashForward.cpp':
  text=replace(text,'      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +', '''      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +
      compact_native_r4_verify_sep22::implementationMarker() +''')
  text=replace(text,'    if (gatheredMPP) {', '''    const bool compactR4Verify = verification && impl_->allRowsInt8Target && impl_->int8ExpertStore &&
        compact_native_r4_verify_sep22::eligible(rows,verification) && impl_->int8ExpertStore->compactNativeR4VerifyEnabled();
    if (compactR4Verify) {
      impl_->int8ExpertStore->addCompactNativeR4VerifyPack(graph,layer,mixed,ids,impl_->blockedScratch,diag,rows,kSelections);
      impl_->int8ExpertStore->addCompactNativeR4VerifyGateUp(graph,layer,impl_->blockedScratch,diag,rows,kSelections);
      impl_->int8ExpertStore->addCompactNativeR4VerifyDown(graph,layer,impl_->blockedScratch,diag,rows,kSelections);
    } else if (gatheredMPP) {''')
  text=replace(text,'blocked && !gatheredMPP ? impl_->blockedScratch.scatteredDown', '(compactR4Verify || (blocked && !gatheredMPP)) ? impl_->blockedScratch.scatteredDown')
 if relative=='runtime/flash/FlashWorker.mm':
  text=replace(text,'      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.', '''      compact_native_r4_verify_sep22::validateDependencies(environmentSwitch("SPLASH_FLASH_ALLROWS_FULL512_TARGET"),
          gathered_mpp::requested(),gathered_mpp::requestedMaximumRows(),environmentSwitch("SPLASH_FLASH_BLOCKED_MOE"),
          environmentSwitch("SPLASH_FLASH_MOE_DIRECT_A"),environmentSwitch("SPLASH_FLASH_MOE_Q4X8"));
      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.''')
  text=replace(text,'  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};', '''  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};
  const auto compactR4Graphs=persistedExperts ? persistedExperts->compactNativeR4VerifyCounters() : compact_native_r4_verify_sep22::Counters{};''')
  text=replace(text,'      << R"(,"gdn_verification_storage":{"lazy_enabled":)"', '''      << R"(,"compact_native_r4_verify":{"schema":"parallel-integer-original-M16-six-stage-R4-verify-v1","scope":"singleton main physical R4 verification only; graph construction not GPU completion","enabled":)"
      <<(compactR4Graphs.enabled ? "true" : "false")<<R"(,"requested":)"<<(compact_native_r4_verify_sep22::requested() ? "true" : "false")
      <<R"(,"source_identity_sha256":)"<<json::quote(compact_native_r4_verify_sep22::kSourceIdentitySha256)
      <<R"(,"dispatches_per_layer":6,"base_gather_dispatches_per_layer":2,"planner_threadgroup_bytes":2432,"additional_gpu_allocation_bytes":0,"full_model_quality_qualified":false,"plan_graph_calls":)"
      <<compactR4Graphs.planCalls<<R"(,"plan_graph_rows":)"<<compactR4Graphs.planRows<<R"(,"gate_graph_calls":)"<<compactR4Graphs.gateCalls
      <<R"(,"gate_graph_rows":)"<<compactR4Graphs.gateRows<<R"(,"down_graph_calls":)"<<compactR4Graphs.downCalls<<R"(,"down_graph_rows":)"<<compactR4Graphs.downRows<<'}'
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
 return text
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT/'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5');p.add_argument('--output',type=Path,default=ROOT/'build/compact-native-r4-verify-teacher-sep22-worker-v1b');a=p.parse_args();base=a.base.resolve();out=a.output.resolve()
 if out.exists()or ROOT/'build'not in out.parents:raise ValueError('NEW private worker output required')
 pm=base/'overlay-manifest.json';parent=json.loads(pm.read_text());qseal=json.loads((ROOT/'build/expert-r4-compact-native-sep22-component-v2/source-seal.json').read_text())
 plan=(ROOT/QUALIFIED/'plan.metal').read_bytes();qrecord=next(r for r in qseal['sources']if r['path']==str(QUALIFIED/'plan.metal'))
 if sha(plan)!=qrecord['sha256']:raise ValueError('Qualified parallel planner source drift')
 parts={'qualified_plan':sha(plan),'qualified_component_source_identity':qseal['source_identity_sha256'],'bridge':sha((ROOT/PRIVATE/'bridge.hpp').read_bytes()),'worker_transform':sha(Path(__file__).read_bytes()),'parent_manifest':sha(pm.read_bytes()),'scope':'singleton-main-verification-physicalR4-Full512-cap4-S10-nativeM16-step16-sixstage-v1'};identity=sha(json.dumps(parts,sort_keys=True,separators=(',',':')).encode());files=[]
 for r in parent['files']:
  rel=r['path'];data=(base/'source'/rel).read_bytes();digest=r.get('sha256')or r.get('overlay_sha256')
  if sha(data)!=digest:raise ValueError('Parent source drift:'+rel)
  changed=transform(rel,data.decode()).encode();write(out/'source'/rel,changed);files.append({'path':rel,'parent_sha256':sha(data),'sha256':sha(changed),'changed':data!=changed})
 for name in('bridge.hpp','policy_cpu.cpp','worker_prepare.py','worker.mk','worker_witness.py'):
  rel=PRIVATE/name;data=(ROOT/rel).read_bytes();write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'new':True})
 for name,data in(('plan.metal',plan),('abi.hpp',(ROOT/QUALIFIED/'abi.hpp').read_bytes())):
  rel=PRIVATE/name;write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'qualified_copy':True})
 rel=PRIVATE/'source_identity.hpp';data=f'#pragma once\nnamespace splash::flash::compact_native_r4_verify_sep22 {{inline constexpr char kSourceIdentitySha256[]="{identity}";}}\n'.encode();write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'generated':True})
 # Preserve the parent's complete host source closure: rebuild every non-core
 # TU, including all consumers of the changed Store/Forward class definitions.
 rows=parent['rebuild'];names=[r['object']for r in rows]+['teacher_bulk'];link='REBUILD_NAMES := '+' '.join(names)+'\n'
 for r in rows:link+=f'SRC_{r["object"]} := $(BUILD)/source/{r["source"]}\n'
 link+='SRC_teacher_bulk := $(BUILD)/source/dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp\n'
 frozen=[];core=[]
 for r in parent['frozen_objects']:
  src=base/r['path'];data=src.read_bytes()
  if sha(data)!=r['sha256']:raise ValueError('Parent Core drift')
  rel=Path('reused/core')/src.name;write(out/rel,data);core.append(str(rel));frozen.append({'path':str(rel),'sha256':sha(data),'source':str(src)})
 link+='CORE := '+' '.join('$(BUILD)/'+r for r in core)+'\n'
 # Teacher modifies host code only; its frozen AB-QSA ancestor supplies the
 # exact original shader closure plus one newly qualified integer AIR.
 ancestor=Path(parent['base']);ancestorLib=(ancestor/'splash.metallib').read_bytes();baseLib=(base/'splash.metallib').read_bytes()
 if ancestorLib!=baseLib:raise ValueError('Teacher library differs from its shader ancestor')
 airs=[]
 for line in(ancestor/'link-inputs.mk').read_text().splitlines():
  if not line.startswith('AIRS := '):continue
  for token in line.split(' := ',1)[1].split():
   src=ancestor/Path(token[9:]);rel=Path('reused/air')/src.name;data=src.read_bytes();write(out/rel,data);airs.append(str(rel));frozen.append({'path':str(rel),'sha256':sha(data),'source':str(src)})
 src=ancestor/'candidate.air';rel=Path('reused/air/ab-merge-candidate.air');data=src.read_bytes();write(out/rel,data);airs.append(str(rel));frozen.append({'path':str(rel),'sha256':sha(data),'source':str(src)})
 link+='AIRS := '+' '.join('$(BUILD)/'+r for r in airs)+'\n';write(out/'link-inputs.mk',link.encode())
 manifest={'schema':'compact-native-R4-singleton-verify-teacher-worker-source-v1','base':str(base),'parent_manifest_sha256':sha(pm.read_bytes()),'files':files,'rebuild':rows,'all_noncore_host_consumers_rebuilt':True,'frozen_inputs':frozen,'link_make_sha256':sha(link.encode()),'qualified_parallel_source_identity':qseal['source_identity_sha256'],'source_identity_sha256':identity,'identity_parts':parts,'changed_paths':sorted(CHANGED),'additional_gpu_allocation_bytes':0,'component_all_six_patterns_exact_root_confirmed':True,'full_model_quality_qualified':False,'gpu_work':False,'model_payload_reads':False,'capture_payload_reads':False}
 write(out/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
 for name in('splash-flash.config','splash.metallib.config'):
  src=base/name
  if src.exists():write(out/name,src.read_bytes().rstrip(b'\n')+b'-compact-native-R4-singleton-verify-v1\n')
 print(json.dumps({'prepared':str(out),'source_identity_sha256':identity,'sources':len(files),'host_rebuild_tus':len(names),'gpu_work':False}))
if __name__=='__main__':main()
