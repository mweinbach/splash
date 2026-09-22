#!/usr/bin/env python3
"""Source-only transformations from sealed v13; no payload or GPU work."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
BASE=ROOT/'build/batchverify-exact-compact-sep22-v13/source/dev/benchmarks/batch_verify_exact_sep22/oracle.mm'
def once(s,a,b):
 if s.count(a)!=1:raise ValueError('source anchor drift:'+a[:100])
 return s.replace(a,b,1)
def transform(s):
 s=once(s,'#include "flash/FlashBatchVerify.hpp"','#include "flash/FlashBatchVerify.hpp"\n#include "flash/FlashBatchPrefill.hpp"\n#include "dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp"')
 s=once(s,'bool expectedKnown=false,','bool expectedKnown=false,') if False else s
 s=s.replace('<<",\\\"expected_unique_frames\\\":26,\\\"completed_repeated_frames\\\":"','<<",\\\"expected_unique_frames\\\":null,\\\"completed_repeated_frames\\\":"')
 s=s.replace('<<",\\\"expected_repeated_frames\\\":54"','<<",\\\"expected_repeated_frames\\\":null"')
 # Sink metadata/payload opening is deferred until all exact budgets/charges are proved.
 a=s.index('  Store(fs::path spill,');b=s.index('  void frame(',a)
 s=s[:a]+'''  Store(fs::path spill,bool exporting,FailureProgress &progress):spill_(std::move(spill)),exporting_(exporting),progress_(progress){}
  void activate(){require(!active_,"snapshot sink already activated");
    progress_.section=exporting_?"spill_directory":"export_manifest";
    if(exporting_){require(!fs::exists(spill_),"fresh spill directory required");fs::create_directories(spill_);}
    else{const auto manifest=dictionary(spill_/"complete.json");require([manifest[@"complete"] isEqual:@YES]&&string(manifest[@"schema"])=="batchverify-export-v1","completed export required");
      for(id item:static_cast<NSArray *>(manifest[@"frames"])){const auto entry=static_cast<NSDictionary *>(item);frames_.push_back({string(entry[@"label"]),string(entry[@"sha256"]),number(entry[@"bytes"]),number(entry[@"planes"]),number(entry[@"live_bytes"])});}
      require(!frames_.empty(),"nonempty complete campaign export required");}active_=true;}
'''+s[b:]
 s=once(s,'progress_.begin(label,repeat);const auto bytes=extent','require(active_,"snapshot sink not admitted/activated");progress_.begin(label,repeat);const auto bytes=extent')
 s=once(s,'private:fs::path spill_;bool exporting_;','private:fs::path spill_;bool exporting_,active_=false;')
 old='  for(const char *n:{"SPLASH_FLASH_BATCH","SPLASH_FLASH_BATCH_MTP","SPLASH_FLASH_BATCH_MTP_PREFILL","SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_MTP","SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT","SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"})require(!selected(n),std::string("excluded worker route: ")+n);'
 new='''  for(const char *n:{"SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT","SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"})require(!selected(n),std::string("excluded worker route: ")+n);
  for(const char *n:{"SPLASH_FLASH_BATCH","SPLASH_FLASH_BATCH_MTP","SPLASH_FLASH_BATCH_MTP_PREFILL","SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_MTP","SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY","SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY","SPLASH_FLASH_BATCH_QSA_BULK_PREFILL","SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22"})require(selected(n),std::string("actual narrowed BQSA4 dependency absent: ")+n);
  require(std::getenv("SPLASH_FLASH_MTP_DRAFT_DEPTH")&&std::string_view(std::getenv("SPLASH_FLASH_MTP_DRAFT_DEPTH"))=="3","actual narrowed BQSA4 MTPdepth3 required");
  require(batch_prefill_twopass_sep22::requested(),"strict B4-only BQSA policy required");
  require(std::string_view(batch_prefill_twopass_sep22::schema)=="batch-real4-MTP3-allfresh2048-existing-packedV-twopass-v2","old real2/4 BQSA policy excluded");'''
 s=once(s,old,new)
 anchor='"SPLASH_FLASH_GDN_BATCH_ILP"};'
 s=once(s,anchor,'"SPLASH_FLASH_GDN_BATCH_ILP","SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22","SPLASH_FLASH_BATCH_QSA_BULK_PREFILL","SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY","SPLASH_FLASH_MTP","SPLASH_FLASH_MTP_DRAFT_DEPTH","SPLASH_FLASH_BATCH_MTP","SPLASH_FLASH_BATCH_MTP_PREFILL","SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY"};')
 s=once(s,'constexpr std::array<const char *,18> kCheckpoints{','constexpr std::array<const char *,21> kCheckpoints{')
 s=once(s,'"recovery.completed"};','"recovery.completed", "same-r8.future.completed", "b2.initial", "fresh-r8.future.completed"};')
 s=once(s,'  uint64_t planned=0,measured=0,guardAllowance=', '  uint64_t prefillPlanned=0,prefillMeasured=0,hostSnapshotPlanned=0,hostSnapshotActual=0,hostRequiredBeforeSink=0,fullCheckpointBound=0;\n  uint64_t groupedB4Prefills=0,groupedB2Prefills=0,B4LayerDelta=0,B2LayerDelta=0;\n  uint64_t planned=0,measured=0,guardAllowance=')
 s=once(s,'    <<",\\\"debug_copy_commands\\\":"<<debugCopyCommands', '''    <<",\\\"actual_prefill_workspace_planned\\\":"<<prefillPlanned<<",\\\"actual_prefill_workspace_measured\\\":"<<prefillMeasured
    <<",\\\"host_exact_rejection_snapshot_planned\\\":"<<hostSnapshotPlanned<<",\\\"host_exact_rejection_snapshot_actual\\\":"<<hostSnapshotActual
    <<",\\\"host_required_before_sink\\\":"<<hostRequiredBeforeSink<<",\\\"full_checkpoint_bound\\\":"<<fullCheckpointBound
    <<",\\\"actual_grouped_B4_prefills\\\":"<<groupedB4Prefills<<",\\\"actual_grouped_B2_prefills\\\":"<<groupedB2Prefills
    <<",\\\"actual_B4_twoPass_layer_delta\\\":"<<B4LayerDelta<<",\\\"actual_B2_twoPass_layer_delta\\\":"<<B2LayerDelta
    <<",\\\"debug_copy_commands\\\":"<<debugCopyCommands''')
 s=once(s,'const uint64_t outputBytes=12*','const uint64_t outputBytes=14*')
 s=once(s,'  std::vector<uint8_t> b(1024,0x5a);', '''  const uint64_t verifyPlan=FlashBatchVerify::workspacePlannedBytes(kCapacity,4,4),prefillPlan=FlashBatchPrefill::workspacePlannedBytes(kCapacity,4,kRows);
  const uint64_t checkpointBound=state4+verifyPlan+(4ULL<<20),hostCopy=5*FlashForward::requestStateBytes(kCapacity)+verifyPlan+(16ULL<<20),partitionBound=(1ULL<<30)+2*checkpointBound;
  require(partitionBound<kLimit,"CPU declared two-selected partition spill exceeds4GiB");
  std::vector<uint8_t> b(1024,0x5a);''')
 s=once(s,'<<conservativeOne<<",\\\"stream_mutation_rejected', '<<conservativeOne<<",\\\"native_verify_plan\\\":"<<verifyPlan<<",\\\"native_actual_batch_prefill_plan\\\":"<<prefillPlan<<",\\\"host_full_rejection_copy_plan_upper\\\":"<<hostCopy<<",\\\"selected2_spill_plan_upper\\\":"<<partitionBound<<",\\\"stream_mutation_rejected')

 s=once(s,'      AllocationBreakdown &allocation,BatchAllocation &batchAllocation,std::span<const uint32_t> prompt,std::vector<std::string> selected)',
       '      FlashBatchPrefill &prefill,std::vector<uint8_t> &hostSnapshot,AllocationBreakdown &allocation,BatchAllocation &batchAllocation,std::span<const uint32_t> prompt,std::vector<std::string> selected)')
 s=once(s,'batch_(batch),store_(store),allocation_', 'batch_(batch),store_(store),prefill_(prefill),hostSnapshot_(hostSnapshot),allocation_')
 s=once(s,'    const uint64_t fullCheckpointBound=4*FlashForward::requestStateBytes(kCapacity)+batchSerialized+(4ULL<<20);','    const uint64_t fullCheckpointBound=4*FlashForward::requestStateBytes(kCapacity)+batchSerialized+(4ULL<<20);batchAllocation_.fullCheckpointBound=fullCheckpointBound;')
 s=once(s,'    for(uint32_t lane=0;lane<4;++lane)fresh(lane,"initial.prefill."+std::to_string(lane));','    for(uint32_t lane=0;lane<4;++lane)allocateOwner(lane);')
 a='''    point("initial");
    verify({0,1},4,"fresh-r8");commit({0,1},{4,1},"fresh-r8");future({0,1},"fresh-r8.future",1);
    verify({0,1,2,3},4,"fresh-r16");commit({0,1,2,3},{4,3,2,1},"fresh-r16");
    verify({0,1},4,"same-r8");commit({0,1},{4,4},"same-r8");'''
 b='''    ledger();const auto host=governor_.snapshot();
    batchAllocation_.hostRequiredBeforeSink=batchAllocation_.partitionSpillPreflight+(16ULL<<20);
    require(host.hostMeasurementValid&&host.growthAllowed&&host.pressure==engine::MemoryPressure::Normal&&host.systemPressure==engine::MemoryPressure::Normal&&host.hostHeadroomBytes>=batchAllocation_.hostRequiredBeforeSink,"host/GUV preflight denied before snapshot sink");
    store_.activate();groupedFresh({0,1,2,3},"initial.grouped-b4");point("initial");
    verify({0,1,2,3},4,"fresh-r16");commit({0,1,2,3},{0,1,2,4},"fresh-r16",0,false);
    verify({1,2},4,"same-r8");commit({1,2},{4,4},"same-r8");future({1,2},"same-r8.future",1);point("same-r8.future.completed");
    allocateOwner(0);allocateOwner(1);groupedFresh({0,1},"independent.grouped-b2");point("b2.initial");
    verify({0,1},4,"fresh-r8");commit({0,1},{4,1},"fresh-r8");future({0,1},"fresh-r8.future",1);point("fresh-r8.future.completed");'''
 s=once(s,a,b)
 s=once(s,'Store &store_;\n  AllocationBreakdown','Store &store_;FlashBatchPrefill &prefill_;std::vector<uint8_t> &hostSnapshot_;\n  AllocationBreakdown')
 # Exact local snapshots: fixed one admitted arena, whole physical requests + every batch plane.
 a=s.index('  std::string stateWitness()const');b=s.index('  void point(',a)
 s=s[:a]+'''  struct ExactSnapshot {
    struct Entry{std::string label;const uint8_t *data;uint64_t bytes,offset;};
    std::vector<Entry> entries;uint64_t bytes=0;
    void check(const std::vector<uint8_t> &arena)const{for(const auto &p:entries)
      require(!std::memcmp(p.data,arena.data()+p.offset,p.bytes),"rejected API mutated full physical bytes: "+p.label);}
  };
  ExactSnapshot exactSnapshot(const FlashRequestState *extra=nullptr){for(const auto &m:mirrors_)debugCopy(m.original,m.view,m.original.sizeBytes());
    ExactSnapshot out;const auto capture=[&](std::string label,const metal::MetalBuffer &buffer,const uint8_t *data){
      require(data&&buffer.sizeBytes()<=hostSnapshot_.size()-out.bytes,"exact rejection snapshot exceeds admitted host arena");
      out.entries.push_back({std::move(label),data,buffer.sizeBytes(),out.bytes});std::memcpy(hostSnapshot_.data()+out.bytes,data,buffer.sizeBytes());out.bytes+=buffer.sizeBytes();};
    for(uint32_t lane=0;lane<4;++lane)if(slots_[lane].state)for(const auto &p:Access::planes(*slots_[lane].state))capture("lane."+std::to_string(lane)+"."+p.label,p.buffer,static_cast<const uint8_t *>(p.buffer.contents()));
    for(const auto &p:Access::batchPlanes(batch_)){
      const auto *data=static_cast<const uint8_t *>(p.buffer.contents());
      if(p.buffer.storage()==metal::BufferStorage::Private){auto it=std::find_if(mirrors_.begin(),mirrors_.end(),[&](const auto &m){return m.label==p.label&&m.original.sameView(p.buffer);});require(it!=mirrors_.end(),"private rejection plane lacks mirror");data=static_cast<const uint8_t *>(it->view.contents());}
      capture(p.label,p.buffer,data);}
    if(extra)for(const auto &p:Access::planes(*extra))capture("sibling."+p.label,p.buffer,static_cast<const uint8_t *>(p.buffer.contents()));return out;}
  void checkExactSnapshot(const ExactSnapshot &before){for(const auto &m:mirrors_)debugCopy(m.original,m.view,m.original.sizeBytes());before.check(hostSnapshot_);checks();}
'''+s[b:]
 s=once(s,'  void fresh(uint32_t lane,const std::string &label){','  void allocateOwner(uint32_t lane){')
 s=once(s,'    const auto r=target_.forward(*slot.state,prompt_,false,true);','''  }
  void groupedFresh(const std::vector<uint32_t> &lanes,const std::string &label){
    auto p=ptrs(lanes);for(auto *state:p)require(state->logicalLength()==0&&!state->poisoned()&&target_.ownsState(*state),"actual grouped prefill owner not fresh");
    std::vector<uint32_t> input;for(size_t lane=0;lane<lanes.size();++lane)input.insert(input.end(),prompt_.begin(),prompt_.end());
    auto &c=batch_prefill_twopass_sep22::counters();const auto before=c.layers.load();const auto r=prefill_.forwardBatch(p,input,kRows,true);const auto after=c.layers.load();
    require(r.lanes==lanes.size()&&r.rows==kRows&&r.greedyRows==lanes.size()&&r.logicalLengths.size()==lanes.size(),"actual grouped prefill output shape differs");
    const uint64_t expected=lanes.size()==4?48:0;require(after==before+expected,"actual B4/B2 narrowed twoPass layer counter differs");
    if(lanes.size()==4){++batchAllocation_.groupedB4Prefills;batchAllocation_.B4LayerDelta+=after-before;}else{++batchAllocation_.groupedB2Prefills;batchAllocation_.B2LayerDelta+=after-before;}
    require(prefill_.canariesIntact(),"actual grouped prefill arena canary changed");std::vector<View> views;
    const auto add=[&](const char *name,Type type,const metal::MetalBuffer &buffer,uint64_t bytes){require(buffer&&buffer.contents()&&buffer.sizeBytes()==bytes,"actual grouped prefill borrowed extent differs");views.push_back({name,type,static_cast<const uint8_t *>(buffer.contents()),bytes,bytes});};
    add("hidden",Type::BF16,r.hiddenBF16,uint64_t{lanes.size()}*kRows*kHyper*2);add("logits",Type::BF16,r.logitsBF16,uint64_t{lanes.size()}*kVocabulary*2);add("greedy",Type::U32,r.greedyResultsU32,uint64_t{lanes.size()}*sizeof(FlashGreedyGPURowResult));
    store_.frame(label+".output","{\\\"lanes\\\":"+std::to_string(lanes.size())+",\\\"rows_per_lane\\\":2048}",views);
    const auto *greedyRows=static_cast<const FlashGreedyGPURowResult *>(r.greedyResultsU32.contents());
    for(size_t i=0;i<lanes.size();++i){auto &slot=slots_[lanes[i]];require(r.logicalLengths[i]==kRows&&target_.ownsState(*slot.state),"actual grouped prefill state differs");slot.anchor=greedyGPUResultToken(greedyRows[i],kVocabulary);}point(label);
  }
  void fresh(uint32_t lane,const std::string &label){allocateOwner(lane);auto &slot=slots_[lane];
    const auto r=target_.forward(*slot.state,prompt_,false,true);''')
 s=s.replace('const auto before=metadata(),beforeState=stateWitness();','const auto before=metadata();const auto beforeState=exactSnapshot();')
 s=s.replace('metadata()==before&&stateWitness()==beforeState','metadata()==before')
 s=once(s,'"expired lane rejection changed pending ownership/state");','"expired lane rejection changed pending ownership/state");checkExactSnapshot(beforeState);')
 s=once(s,'"rejected pending action changed exact metadata/state");','"rejected pending action changed exact metadata/state");checkExactSnapshot(beforeState);')
 s=once(s,'"source wrapper replacement rejection mutated original pending state");','"source wrapper replacement rejection mutated original pending state");checkExactSnapshot(beforeState);')
 s=once(s,'rejected([&]{(void)target_.forward(*slots_[0].state,one);},"aborted scalar forward");++invalidChecks_;', 'const auto abortedBefore=metadata();const auto abortedSnapshot=exactSnapshot();rejected([&]{(void)target_.forward(*slots_[0].state,one);},"aborted scalar forward");++invalidChecks_;require(metadata()==abortedBefore,"aborted scalar rejected API changed metadata");checkExactSnapshot(abortedSnapshot);')
 s=once(s,'rejected([&]{(void)batch_.verifyBatch(bad,one,1);},"aborted batch verify");++invalidChecks_;', 'const auto abortedBatchSnapshot=exactSnapshot();rejected([&]{(void)batch_.verifyBatch(bad,one,1);},"aborted batch verify");++invalidChecks_;require(metadata()==abortedBefore,"aborted batch rejected API changed metadata");checkExactSnapshot(abortedBatchSnapshot);')
 s=once(s,'},"moved source verify");++invalidChecks_;', '},"moved source verify");++invalidChecks_;checkExactSnapshot(beforeState);')
 s=once(s,'},"moved source commit");++invalidChecks_;', '},"moved source commit");++invalidChecks_;checkExactSnapshot(beforeState);')
 s=once(s,'      reject([&]{(void)batch_.commitBatch(substitute,full);},"wrong-cohort-sibling");checkGuards(guards);', '      const auto siblingMetadata=Access::metadata(target_,sibling);const auto siblingBefore=exactSnapshot(&sibling);reject([&]{(void)batch_.commitBatch(substitute,full);},"wrong-cohort-sibling");checkExactSnapshot(siblingBefore);require(Access::metadata(target_,sibling)==siblingMetadata,"wrong-cohort rejected API changed sibling metadata");checkGuards(guards);')


 s=once(s,'      allocation.state=batchAllocation.stateAllowance;allocation.reservation=', '      batchAllocation.prefillPlanned=FlashBatchPrefill::workspacePlannedBytes(kCapacity,4,kRows);\n      allocation.state=batchAllocation.stateAllowance;allocation.reservation=batchAllocation.prefillPlanned+')
 s=once(s,'admission->commit();','''admission->commit();
      stage="actual-batch-prefill";const auto beforePrefill=backend.memoryStats().allocatedBytes;FlashBatchPrefill prefill(backend,weights,target,kCapacity,4,kRows);const auto afterPrefill=backend.memoryStats().allocatedBytes;
      require(afterPrefill>=beforePrefill&&afterPrefill-beforePrefill==prefill.workspaceBytes()&&prefill.workspaceBytes()<=batchAllocation.prefillPlanned,"actual batch prefill native charge exceeds admitted category");batchAllocation.prefillMeasured=afterPrefill-beforePrefill;
      uint64_t batchPhysical=0;for(const auto &plane:Access::batchPlanes(batch))batchPhysical+=plane.buffer.sizeBytes();
      batchAllocation.hostSnapshotPlanned=5*FlashForward::requestStateBytes(kCapacity)+batchPhysical+(16ULL<<20);
      auto hostAdmission=governor.tryReserve(batchAllocation.hostSnapshotPlanned);require(bool(hostAdmission),"host exact rejection copy admission denied before allocation/sink");
      std::vector<uint8_t> hostSnapshot(batchAllocation.hostSnapshotPlanned);batchAllocation.hostSnapshotActual=hostSnapshot.size();
      require(hostSnapshot.capacity()==hostSnapshot.size(),"host exact snapshot allocator capacity exceeds admitted size");''')
 s=once(s,'actual max16 BatchVerify same-owner transition/state/future; no trained head or service','current BQSA4 grouped fresh4/independent oldSG8 fresh2 then actual max16 BatchVerify same-owner transition/state/future; no trained head or service')
 s=once(s,'<<json::quote(target.batchInt8ExpertStore()->identitySha256())', '<<json::quote(target.batchInt8ExpertStore()->identitySha256())<<",\\\"target_raw_numeric_parent_SHA\\\":"<<json::quote(target.batchInt8ExpertStore()->compactNativeBatchVerifyNumericParent())<<",\\\"BQSA_source_policy_SHA\\\":"<<json::quote(batch_prefill_twopass_sep22::sourceSHA)')
 s=once(s,'<<json::quote(batch_prefill_twopass_sep22::sourceSHA)\n        <<",\\\"numeric_flags', '<<json::quote(batch_prefill_twopass_sep22::sourceSHA)<<",\\\"BQSA_numeric_parent_routes_raw\\\":"<<json::quote(target.kernelRoutes())\n        <<",\\\"numeric_flags')
 s=once(s,'      stage="campaign";Campaign campaign(backend,governor,target,batch,store,allocation,batchAllocation,prompt,selectedPoints);', '      stage="campaign";Campaign campaign(backend,governor,target,batch,store,prefill,hostSnapshot,allocation,batchAllocation,prompt,selectedPoints);')
 s=once(s,'\",\\\"kernel_routes\\\":"<<json::quote(target.kernelRoutes())', '\",\\\"BQSA_numeric_parent_routes_raw\\\":"<<json::quote(target.kernelRoutes())<<",\\\"target_numeric_parent_SHA\\\":"<<json::quote(target.batchInt8ExpertStore()->compactNativeBatchVerifyNumericParent())<<",\\\"BQSA_policy_SHA\\\":"<<json::quote(batch_prefill_twopass_sep22::sourceSHA)<<",\\\"kernel_routes\\\":"<<json::quote(target.kernelRoutes())')
 return s
if __name__=='__main__':(HERE/'oracle.mm').write_text(transform(BASE.read_text()))
