#!/usr/bin/env python3
"""CPU-only bounded actual-model QA clone; no model/capture/fixture data access."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess
ROOT=Path("/Users/mweinbach/Projects/splash")
HERE=Path(__file__).resolve().parent
NATIVE=ROOT/"build/trunkverify-rawQ4-GDN26-VerifyR4-sep22-v1"
PRIVATE="dev/benchmarks/trunk_verify_exact_sep22"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(s,old,new):
 if s.count(old)!=1:raise ValueError("bounded QA source anchor drift:"+old[:90])
 return s.replace(old,new)
def transform(s,worker_exe_sha):
 s='#include "dev/benchmarks/immutable_interval_worker_sep22/policy.hpp"\n'+s
 s=s.replace("expected_unique_frames\\\":26","expected_unique_frames\\\":10").replace("expected_repeated_frames\\\":54","expected_repeated_frames\\\":0")
 s=s.replace("frames_.size()==26","frames_.size()==10").replace("completed_==26&&repeated_==54","completed_==10&&repeated_==0")
 s=s.replace("all26 expected frames required","all10 bounded expected frames required")
 s=s.replace("TRUNK Prefill+Verify4+Commit1..4+correction/future matrix; no trained head","bounded TRUNK Prefill2K+Verify4+keep2+genuine correctionR1; no trained head")
 s=s.replace("registered COMPOSITE compact1 HC1 bundle1 exact matrix","bounded samebinary immutable96 flag0/1 host-decision preservation")
 s=s.replace("663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438",worker_exe_sha)
 s=s.replace("15*FlashForward::requestStateBytes(kCapacity)+(200ULL<<20)+(128ULL<<20)",
             "4*FlashForward::requestStateBytes(kCapacity)+(80ULL<<20)+3*(160ULL<<20)")
 s=s.replace('\"unique_frames\\\":26','\"unique_frames\\\":10').replace('\"state_frames\\\":15','\"state_frames\\\":4')
 s=once(s,'require(exporting!=bool(SPLASH_VERIFY_CANDIDATE),"binary role mismatch");require(!exporting,"registered composite comparator cannot export");',
        'require(immutable_interval_worker_sep22::requested()==!exporting,"samebinary flag0export/flag1compare role required");')
 s=once(s,"early();progress.role=", "immutable_interval_worker_sep22::startup();early();progress.role=")
 s=once(s,'      stage="target";FlashForward target(', '''      constexpr uint64_t hostPlan=2ULL<<30;
      auto hostAdmission=governor.tryReserve(hostPlan);
      require(bool(hostAdmission),"separate HostRAM2GiB admission BEFORE any large local snapshot or payload write");
      {
      stage="target";FlashForward target(''')
 s=once(s,"      LocalTails tails(target);uint64_t localChecks=0,invalidChecks=0;","""      // QA-only known initialization of ORIGINAL unused owned snapshot/control
      // buffers BEFORE first trial. No request state/history/weights are touched.
      for (const auto &p:Access::tapeUndefined(target)) {
        require(p.buffer.storage()==metal::BufferStorage::Shared && p.buffer.contents() &&
            p.begin<=p.buffer.sizeBytes(),"exact Shared undefined owned-tail initialization bound");
        std::memset(static_cast<uint8_t*>(p.buffer.contents())+p.begin,0xa5,p.buffer.sizeBytes()-p.begin);
      }
      LocalTails tails(target);uint64_t localChecks=0,invalidChecks=0;
      const uint64_t entireCampaignBound=4*FlashForward::requestStateBytes(kCapacity)+(80ULL<<20)+3*(160ULL<<20);
      const auto inventory=tapeViews(target);
      const uint64_t actualTapeExtent=extent(kTapeMeta,inventory);
      require(actualTapeExtent<(160ULL<<20),"actual complete physical tape inventory below source campaign allowance");
      uint64_t snapshotBytes=0;for(const auto &p:tails.entries)snapshotBytes+=p.bytes.size();
      require(snapshotBytes<hostPlan,"actual local full unused-tail host snapshots fit held HostRAM2GiB admission");
      require(entireCampaignBound<kLimit,"WHOLE bounded campaign source upper bound including headers before first payload write");""")
 s=once(s,"for(uint32_t retained:{1u,2u,3u,4u,0u})","for(uint32_t retained:{2u})")
 s=once(s,"const bool repeat=retained!=1;","const bool repeat=false;")
 s=once(s,"        stage=\"verify\";auto verified=target.verify(state,verification);","""        const auto indexedBefore=immutable_interval_worker_sep22::indexedAccepts.load();
        const auto originalBefore=immutable_interval_worker_sep22::originalCallbacks.load();
        stage="verify";auto verified=target.verify(state,verification);
        const auto indexedDelta=immutable_interval_worker_sep22::indexedAccepts.load()-indexedBefore;
        const auto originalDelta=immutable_interval_worker_sep22::originalCallbacks.load()-originalBefore;
        require(indexedDelta==(exporting?0:624) && originalDelta==(exporting?624:0),
            "actual VerifyR4 48x13 indexed/original query dispositions differ");""")
 s=once(s,"        if(retained==1)for(uint32_t i=0;i<4;++i)","        for(uint32_t i=0;i<4;++i)")
 s=once(s,'store.state("commit-r"+std::to_string(retained)+".state",target,state);store.frame("verify.tapes",kTapeMeta,tapeViews(target),true);',
        'store.state("commit-r"+std::to_string(retained)+".state",target,state);store.frame("commit.tapes",kTapeMeta,tapeViews(target),false);')
 s=once(s,'store.output(label+".output",result,rows);store.state(label+".state",target,state);store.frame("verify.tapes",kTapeMeta,tapeViews(target),true);',
        'store.output(label+".output",result,rows);store.state(label+".state",target,state);store.frame("future.tapes",kTapeMeta,tapeViews(target),false);')
 s=once(s,'append(1,"correction-r"+std::to_string(retained));if(retained==2||retained==4)append(4,"future4-r"+std::to_string(retained));if(retained==3||retained==4)append(8,"future8-r"+std::to_string(retained));',
        'append(1,"correction-r"+std::to_string(retained));')
 s=s.replace("actualBundle.calls==240&&actualBundle.rows==960","actualBundle.calls==48&&actualBundle.rows==192")
 s=s.replace("hcCalls==485&&hcRows==1940&&hcPads==485","hcCalls==97&&hcRows==388&&hcPads==97")
 s=s.replace("actualRawQ4Calls==130&&actualRawQ4Rows==520","actualRawQ4Calls==26&&actualRawQ4Rows==104")
 s=once(s,'std::vector<View> tapeViews(const FlashForward &target){std::vector<View> out;for(const auto &p:Access::tapePlanes(target))out.push_back({p.label,p.type,static_cast<const uint8_t *>(p.buffer.contents()),p.live,p.buffer.sizeBytes()});return out;}',
        '''std::vector<View> tapeViews(const FlashForward &target){std::vector<View> out;for(const auto &p:Access::tapePlanes(target))out.push_back({p.label,p.type,static_cast<const uint8_t *>(p.buffer.contents()),p.live,p.buffer.sizeBytes()});
  for(const auto &p:Access::tapeUndefined(target)){
    require(p.buffer.storage()==metal::BufferStorage::Shared && p.buffer.contents() && p.begin<=p.buffer.sizeBytes(),
        "complete initialized owned physical tail bounds");
    if(p.buffer.sizeBytes()>p.begin)out.push_back({"known_initialized_physical_tail."+p.label,Type::U32,
        static_cast<const uint8_t*>(p.buffer.contents())+p.begin,p.buffer.sizeBytes()-p.begin,p.buffer.sizeBytes()-p.begin});
  }return out;}''')
 s=s.replace("defined ranges only; undefined tails locally unchanged","full original known-initialized owned physical snapshot/control tails; QA-only")
 s=once(s,'const uint32_t actualGuardCases=Access::actualBundleGuardProbes(target);',
        '''const auto beforeAliases=immutable_interval_worker_sep22::originalCallbacks.load();
      const uint32_t actualGuardCases=Access::actualBundleGuardProbes(target);
      const auto aliasFallbackDelta=immutable_interval_worker_sep22::originalCallbacks.load()-beforeAliases;
      require(immutable_interval_worker_sep22::finalizedTables.load()==(exporting?0:1) &&
          immutable_interval_worker_sep22::finalizedSpans.load()==(exporting?0:96) &&
          immutable_interval_worker_sep22::indexableTables.load()==(exporting?0:1),
          "actual completed Store immutable96 table qualification differs");
      (void)aliasFallbackDelta;''')
 s=once(s,'std::ostringstream o;o<<"{\\\"actual_rawQ4_rowpair_calls\\\":"',
        'std::ostringstream o;o<<"{\\\"immutable_index_requested\\\":"<<(exporting?"false":"true")<<",\\\"immutable_index_source_policy_sha256\\\":"<<json::quote(immutable_interval_worker_sep22::sourcePolicySHA)<<",\\\"VerifyR4_indexed_accepts\\\":"<<(exporting?0:624)<<",\\\"VerifyR4_original_callbacks\\\":"<<(exporting?624:0)<<",\\\"owned_alias_original_callback_delta\\\":"<<aliasFallbackDelta<<",\\\"known_initialized_original_unused_snapshot_control_tail_scope\\\":true,\\\"samebinary_flag0_control_has_getenv_counter_overhead\\\":true,\\\"actual_rawQ4_rowpair_calls\\\":"')
 s=once(s,'own=o.str();allocation.governor=governor.snapshot();allocation.finalAllocated=backend.memoryStats().allocatedBytes;',
        '''own=o.str();allocation.finalAllocated=backend.memoryStats().allocatedBytes;
      require(allocation.finalAllocated>=allocation.mappedInitial && allocation.finalAllocated-allocation.mappedInitial<=planned,
          "actual live native ledger inside admitted target/state categories");
      }
      require(backend.memoryStats().allocatedBytes==allocation.mappedInitial,"actual target/state/tapes owner teardown returns to original mapped-model ledger");
      hostAdmission->commit();allocation.governor=governor.snapshot();allocation.governorStage="after_all_owned_target_state_tail_snapshots_teardown";
      require(allocation.governor.reservedBytes==0 && allocation.governor.deniedReservations==0 && allocation.governor.hostMeasurementValid && allocation.governor.growthAllowed,
          "actual healthy governor zero reservations after full owned teardown");''')
 s=once(s,'uint64_t fixed=0,experts=0,f32=0,head=0,bf16=0,blocked=0,w8Cache=0,state=0,margin=16ULL<<20;',
        'uint64_t fixed=0,experts=0,f32=0,head=0,bf16=0,blocked=0,w8Cache=0,state=0,margin=16ULL<<20;\n  uint64_t separateHostRAMAdmission=2ULL<<30;')
 s=once(s,'<<",\\\"one_request_state\\\":"<<state<<",\\\"diagnostic_margin\\\":"<<margin<<",\\\"reservation_total\\\":"<<reservation',
        '<<",\\\"one_request_state\\\":"<<state<<",\\\"diagnostic_margin\\\":"<<margin<<",\\\"native_target_state_reservation_total\\\":"<<reservation<<",\\\"held_separate_HostRAM_admission\\\":"<<separateHostRAMAdmission<<",\\\"reservation_total\\\":"<<(reservation+separateHostRAMAdmission)')
 s=s.replace('"qualification_complete\\\":','"bounded_host_state_comparison_complete\\\":')
 return s
def main():
 p=argparse.ArgumentParser();p.add_argument("--worker",type=Path,required=True);p.add_argument("--build",type=Path,required=True)
 a=p.parse_args();worker=a.worker.resolve();out=a.build.resolve()
 if out.exists():raise ValueError("fresh bounded QA clone required")
 seal=json.loads((worker/"compiled-cpu-seal.json").read_text())
 if not seal["pass"]:raise ValueError("complete worker CPU closure required")
 for r in seal["objects"]+seal["artifacts"]:
  if sha(worker/r["path"])!=r["sha256"]:raise ValueError("worker code/object drift")
 out.mkdir(parents=True);shutil.copytree(worker/"source",out/"source")
 own=out/"source"/PRIVATE
 own.mkdir(parents=True,exist_ok=True)
 for name in ["inspect.hpp","inspection.cpp.inc"]:shutil.copy2(NATIVE/"source"/PRIVATE/name,own/name)
 inspection=own/"inspection.cpp.inc";it=inspection.read_text()
 it=once(it,'bool oldRejected=false,newRejected=false;metal::CommandGraph oldGraph,newGraph;',
         'bool oldRejected=false,newRejected=false;std::string oldCategory,newCategory,oldText,newText;metal::CommandGraph oldGraph,newGraph;')
 it=once(it,'catch(const std::invalid_argument&){oldRejected=true;}catch(const std::logic_error&){oldRejected=true;}',
         'catch(const std::invalid_argument&e){oldRejected=true;oldCategory="invalid_argument";oldText=e.what();}catch(const std::logic_error&e){oldRejected=true;oldCategory="logic_error";oldText=e.what();}')
 it=once(it,'catch(const std::invalid_argument&){newRejected=true;}catch(const std::logic_error&){newRejected=true;}',
         'catch(const std::invalid_argument&e){newRejected=true;newCategory="invalid_argument";newText=e.what();}catch(const std::logic_error&e){newRejected=true;newCategory="logic_error";newText=e.what();}')
 it=once(it,'if(!oldRejected||!newRejected||oldRejected!=newRejected||!newGraph.empty())',
         'if(!oldRejected||!newRejected||oldRejected!=newRejected||oldCategory!=newCategory||oldText!=newText||!newGraph.empty())')
 inspection.write_text(it)
 oldHeader=out/"source/runtime/flash/FlashForward.hpp";text=oldHeader.read_text();text=once(text,"private:\n  friend class FlashBatchForward;","private:\n  friend class FlashBatchForward;\n  friend class FlashDeepPrefixOracle;");oldHeader.write_text(text)
 forward=out/"source/runtime/flash/FlashForward.cpp";forward.write_text('#include "inspect.hpp"\n'+forward.read_text()+"\n"+(own/"inspection.cpp.inc").read_text())
 original=(NATIVE/"source"/PRIVATE/"oracle.mm").read_text();oracle=transform(original,sha(worker/"splash-flash"));(own/"oracle.mm").write_text(oracle)
 flags=["-std=c++20","-O3","-Wall","-Wextra","-Werror","-Wno-deprecated-declarations","-fobjc-arc","-mmacosx-version-min=27.0","-DSPLASH_INT8_EXPERIMENT=1",
        "-I"+str(out/"source"),"-I"+str(out/"source/runtime"),"-I"+str(out/"source/dev/benchmarks/prefill4k_attention"),"-I"+str(own),"-I"+str(out)]
 (out/"objects").mkdir();objects=[];census=[];commands=[]
 link=(worker/"link-inputs.mk").read_text();srcs={m.group(1):m.group(2)for m in re.finditer(r"^SRC_(\S+) := \$\(BUILD\)/source/(.*)$",link,re.M)}
 for r in seal["objects"]:
  sourcePath=srcs.get(re.sub(r"^\d+-","",Path(r["path"]).stem)) or srcs.get(Path(r["path"]).stem)
  originalObject=worker/r["path"];obj=out/"objects"/Path(r["path"]).name
  if sourcePath:
   source=out/"source"/sourcePath
   deps=subprocess.run(["xcrun","-sdk","macosx","clang++",*flags,"-MM",str(source)],cwd=ROOT,check=True,capture_output=True,text=True).stdout
   consumer="/runtime/flash/FlashForward.hpp" in deps;census.append({"source":str(source),"Forward_private_friend_consumer":consumer,"excluded_Worker":sourcePath.endswith("FlashWorker.mm"),"dependencies":deps})
   if sourcePath.endswith("FlashWorker.mm"):continue
   if consumer:
    command=["xcrun","-sdk","macosx","clang++",*flags,"-MMD","-MP","-c",str(source),"-o",str(obj)];commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
   else:shutil.copy2(originalObject,obj)
  else:shutil.copy2(originalObject,obj);consumer=False
  objects.append({"path":str(obj),"sha256":sha(obj),"source_parent":str(originalObject),"source_parent_sha256":r["sha256"],"private_Fwd_header_recompiled":consumer})
 if len(objects)!=53 or len(census)!=50:raise ValueError("exact53nonWorker/50census required")
 shutil.copy2(worker/"splash.metallib",out/"splash.metallib")
 provenance={"schema":"bounded-immutable-index-actual-state-code-provenance-v1","worker":str(worker),"worker_sha256":sha(worker/"splash-flash"),"worker_seal_sha256":sha(worker/"compiled-cpu-seal.json"),"metallib_sha256":sha(out/"splash.metallib"),"source_policy":seal["source_policy_sha256"],"no_numerical_graph_or_library_change":True,"QA_only_known_unused_control_snapshot_init":True}
 (out/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
 (out/"TrunkVerifyBuildProvenance.hpp").write_text('#pragma once\ninline constexpr const char*kPrefillExactProvenancePath='+json.dumps(str(out/"provenance.json"))+';\ninline constexpr const char*kPrefillExactBuildProvenance=R"PROV('+json.dumps(provenance,sort_keys=True)+')PROV";\n')
 source=own/"oracle.mm";command=["xcrun","-sdk","macosx","clang++",*flags,str(source),*[x["path"]for x in objects],"-framework","Foundation","-framework","Metal","-framework","IOKit","-o",str(out/"oracle")]
 commands.append(command);subprocess.run(command,cwd=ROOT,check=True)
 cpu=json.loads(subprocess.run([str(out/"oracle"),"--cpu-only"],cwd=ROOT,check=True,capture_output=True,text=True).stdout)
 ready={"schema":"immutable-index-bounded-samebinary-Verify4-keep2-R1-CPU-closure-v1","pass":True,"GPU_executed":False,"model_capture_response_operand_export_payload_read_or_hashed":False,
  "worker":str(worker),"worker_seal_sha256":sha(worker/"compiled-cpu-seal.json"),"oracle_sha256":sha(out/"oracle"),"metallib_sha256":sha(out/"splash.metallib"),
  "objects":objects,"actual50TU_friend_header_census":census,"headers":[{"path":str(oldHeader),"sha256":sha(oldHeader)},{"path":str(own/"inspect.hpp"),"sha256":sha(own/"inspect.hpp")}],
  "source_files":[{"path":str(p),"sha256":sha(p)}for p in sorted((out/"source").rglob("*"))if p.is_file()],"source_original_native_sha256":sha(NATIVE/"source"/PRIVATE/"oracle.mm"),
  "compiler_commands":commands,"CPU":cpu,"bounded_expected_unique_frames":10,"bounded_expected_repeated_frames":0,"all134fullstate_216fulltapes_originalphysical_unusedtails_knowninit":True,
  "all8_actual_owned_guard_probes_exact_error_category_and_text":True,"separate_held_host_admission_bytes":2<<30,"source_campaign_spill_upper_bound_bytes":cpu["spill_upper_bound_bytes"],
  "Root_GPU_qualification_pending":True,"whole_performance_or_original22_qualified":False}
 (out/"CPU_READY.json").write_text(json.dumps(ready,indent=2)+"\n")
 print(json.dumps({"CPU_READY":str(out/"CPU_READY.json"),"oracle_sha256":ready["oracle_sha256"],"CPU":cpu}))
if __name__=="__main__":main()
