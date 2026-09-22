#!/usr/bin/env python3
"""CPU-only fresh R5 oracle. Never opens model, operand, token or export data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
WORKER = ROOT / 'build/R5-integer-currentQ4-fixed4-sep22-worker-v2'
PARENT = ROOT / 'build/trunkverify-rawQ4-GDN26-VerifyR4-sep22-v1/source/dev/benchmarks/trunk_verify_exact_sep22'
JOURNAL = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new, label):
    if text.count(old) != 1:
        raise ValueError('unique source anchor changed: ' + label)
    JOURNAL.append({'label': label, 'old': old, 'new': new})
    return text.replace(old, new, 1)


def oracle_source():
    text = (PARENT / 'oracle.mm').read_text()
    text = once(text, '#include "dev/benchmarks/raw_q4_verify_worker_sep22/policy.hpp"', '#include "dev/benchmarks/expert_r5_verify_worker_sep22/policy.hpp"\n#include "dev/benchmarks/raw_q4_verify_worker_sep22/policy.hpp"', 'R5 policy include')
    for old, new, label in [
        ('kVerifyRows=4', 'kVerifyRows=5', 'real maximum rows'),
        ('"expected_unique_frames\\\":26', '"expected_unique_frames\\\":29', 'failure unique count'),
        ('"expected_repeated_frames\\\":54', '"expected_repeated_frames\\\":61', 'failure repeat count'),
        ('frames_.size()==26', 'frames_.size()==29', 'manifest frame count'),
        ('all26 expected frames required', 'all29 expected frames required', 'manifest message'),
        ('completed_==26&&repeated_==54', 'completed_==29&&repeated_==61', 'completion count'),
        ('15*FlashForward::requestStateBytes(kCapacity)+(200ULL<<20)+(128ULL<<20)', '17*FlashForward::requestStateBytes(kCapacity)+(96ULL<<20)+36*FlashGDNLazyRollback::plannedBytes(5,1)+(2ULL<<20)', 'source formula spill upper bound'),
        ('"unique_frames\\\":26,\\\"state_frames\\\":15', '"unique_frames\\\":29,\\\"state_frames\\\":17', 'CPU expected counts'),
        ('require(!exporting,"registered composite comparator cannot export");', '', 'enable paired current flag0 export'),
        ('std::array<uint32_t,4> correction{};', 'std::array<uint32_t,5> correction{};', 'five corrections'),
        ('inputs.count==4&&fixes.count==4', 'inputs.count==5&&fixes.count==5', 'five metadata tokens'),
        ('for(uint32_t i=0;i<4;++i){verification.push_back', 'for(uint32_t i=0;i<5;++i){verification.push_back', 'five control tokens'),
        ('retained:{1u,2u,3u,4u,0u}', 'retained:{1u,2u,3u,4u,5u,0u}', 'six lifecycle trials'),
        ('verification={greedy(body,0),prompt[kRows-3],prompt[kRows-2],prompt[kRows-1]}', 'verification={greedy(body,0),prompt[kRows-4],prompt[kRows-3],prompt[kRows-2],prompt[kRows-1]}', 'five current verify inputs'),
        ('verified.logitRows==4', 'verified.logitRows==5', 'five logits'),
        ('store.output("verify.output",verified,4,repeat)', 'store.output("verify.output",verified,5,repeat)', 'five verify output rows'),
        ('if(retained==1)for(uint32_t i=0;i<4;++i)', 'if(retained==1)for(uint32_t i=0;i<5;++i)', 'five correction records'),
        ('store.output("verify.output",verified,4,true)', 'store.output("verify.output",verified,5,true)', 'five invalid-operation replay rows'),
        ('target.commitVerify(state,5);},"excess retained"', 'target.commitVerify(state,6);},"excess retained"', 'invalid six retained'),
        ('if(retained==4)require(commit.gpuSeconds', 'if(retained==5)require(commit.gpuSeconds', 'full five acceptance'),
        ('TRUNK Prefill+Verify4+Commit1..4+correction/future matrix; no trained head', 'TRUNK Prefill+Verify5+Commit1..5+zero-retained rejection/abort+correction/future matrix; no trained head', 'new explicit scope'),
        ('for(uint32_t i=0;i<4;++i){if(i)c<<\',\';c<<verification', 'for(uint32_t i=0;i<5;++i){if(i)c<<\',\';c<<verification', 'five verification metadata'),
        ('for(uint32_t i=0;i<4;++i){if(i)c<<\',\';c<<correction', 'for(uint32_t i=0;i<5;++i){if(i)c<<\',\';c<<correction', 'five correction metadata'),
        ('registered COMPOSITE compact1 HC1 bundle1 exact matrix', 'current Q4 source R5 flag0 blocked10 versus flag1 integer6 main-state exact matrix', 'report scope'),
    ]:
        text = once(text, old, new, label)
    # These three repeated scalar checks all belong to pending Verify5 status.
    old = 'std::array<uint64_t,3>{4,kRows,36}'
    if text.count(old) != 2:
        raise ValueError('pending scalar anchors changed')
    JOURNAL.append({'label':'both pending scalar checks','old':old,'new':'std::array<uint64_t,3>{5,kRows,36}','count':2})
    text = text.replace(old, 'std::array<uint64_t,3>{5,kRows,36}')
    start = text.index('void early(){')
    end = text.index('\nstd::vector<View> tapeViews', start)
    old = text[start:end]
    new = '''void early(){
  raw_q4_verify_sep22::validateDependencies();
  require(raw_q4_verify_sep22::requested(),"current qualified rawQ4 flag1 required");
  require(std::string_view(compact_native_r5_verify_sep22::kSourceIdentitySha256)=="4bb7b637c2b6b60520159ad3724a8d9e3867c7c538dd8d260caa4fc7d184769a","current R5 source differs");
  const bool candidate=bool(SPLASH_VERIFY_CANDIDATE);
  require(selected(compact_native_r5_verify_sep22::kFlag)==candidate,"R5 flag must match private binary role");
  compact_native_r5_verify_sep22::validateDepth(selected("SPLASH_FLASH_MTP"),4,true);
  const char*depth=std::getenv("SPLASH_FLASH_MTP_DRAFT_DEPTH");require(depth&&std::string_view(depth)=="4","explicit fixed4 environment required");
  for(const char*n:{"SPLASH_FLASH_ALLROWS_FULL512_TARGET","SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22","SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22","SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22","SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22","SPLASH_FLASH_GDN_LAZY_ROLLBACK","SPLASH_FLASH_GPU_GREEDY"})require(selected(n),std::string("current parent flag1 required: ")+n);
  require(!selected("SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21"),"phase-Q4 excluded");
  for(const char*n:{"SPLASH_FLASH_BATCH","SPLASH_FLASH_BATCH_MTP","SPLASH_FLASH_BATCH_MTP_PREFILL","SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT","SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"})require(!selected(n),std::string("excluded route: ")+n);
  (void)commonPolicy();
}'''
    text = once(text, old, new, 'exact current R5 role admission')
    text = once(text, 'saved VerifyR4 operands\\\",\\\"maximum_rows\\\":4', 'saved VerifyR5 operands\\\",\\\"maximum_rows\\\":5', 'five-row tape metadata')
    old = 'require(raw_q4_verify_sep22::graphCalls.load()-rawBeforeVerify==26&&raw_q4_verify_sep22::graphRows.load()-rawRowsBeforeVerify==104,"Verify4 actual rawQ4 family26 counter differs");'
    new = 'require(raw_q4_verify_sep22::graphCalls.load()==rawBeforeVerify&&raw_q4_verify_sep22::graphRows.load()==rawRowsBeforeVerify,"R5 must not execute R4-only rawQ4 pairing");require(compact_native_r5_verify_sep22::graphCalls.load()-r5BeforeVerify==(SPLASH_VERIFY_CANDIDATE?48:0)&&compact_native_r5_verify_sep22::graphRows.load()-r5RowsBeforeVerify==(SPLASH_VERIFY_CANDIDATE?240:0),"actual R5 layer census differs");'
    text = once(text, old, new, 'actual R5 window counters')
    text = once(text, 'stage="verify";auto verified', 'const auto r5BeforeVerify=compact_native_r5_verify_sep22::graphCalls.load(),r5RowsBeforeVerify=compact_native_r5_verify_sep22::graphRows.load();\n        stage="verify";auto verified', 'R5 before snapshot')
    start = text.index('      const auto actualBundle=')
    end = text.index('\n    }\n    backendDestroyed=true;', start)
    old = text[start:end]
    new = '''      const auto actualBundle=target.batchInt8ExpertStore()->compactR4PreflightCounters();const auto actualPlan=target.batchInt8ExpertStore()->compactNativeR4VerifyCounters();
      const uint64_t hcCalls=hc_pad_verify_sep22::graphCalls.load(),hcRows=hc_pad_verify_sep22::graphRows.load(),hcPads=hc_pad_verify_sep22::savedPaddingDispatches.load();
      const uint64_t rawCalls=raw_q4_verify_sep22::graphCalls.load(),rawRows=raw_q4_verify_sep22::graphRows.load();
      require(actualBundle.calls==0&&actualBundle.rows==0&&actualPlan.planCalls==0&&actualPlan.planRows==0&&hcCalls==0&&hcRows==0&&hcPads==0&&rawCalls==0&&rawRows==0,"R5/future matrix executed excluded R4-only chain");
      const uint64_t r5Calls=compact_native_r5_verify_sep22::graphCalls.load(),r5Rows=compact_native_r5_verify_sep22::graphRows.load();
      require(r5Calls==(SPLASH_VERIFY_CANDIDATE?288:0)&&r5Rows==(SPLASH_VERIFY_CANDIDATE?1440:0),"six actual Verify5 trials x48 layers required");
      const uint32_t guardCases=Access::actualBundleGuardProbes(target);require(guardCases==8,"preserved R4 metadata guard census differs");const auto r5GuardCases=Access::actualR5GuardProbes(target);require(r5GuardCases==9,"fresh actual R5 canonical negative metadata guards differ");
      std::ostringstream o;o<<"{\\\"R5_enabled\\\":"<<(SPLASH_VERIFY_CANDIDATE?"true":"false")<<",\\\"actual_R5_calls\\\":"<<r5Calls<<",\\\"actual_R5_rows\\\":"<<r5Rows<<",\\\"actual_rawQ4_calls\\\":"<<rawCalls<<",\\\"actual_HC_R4_calls\\\":"<<hcCalls<<",\\\"actual_bundle_R4_calls\\\":"<<actualBundle.calls<<",\\\"preserved_R4_owned_metadata_guard_cases\\\":"<<guardCases<<",\\\"actual_R5_owned_metadata_guard_cases\\\":"<<r5GuardCases<<",\\\"local_pointer_redzone_tape_checks\\\":"<<localChecks<<",\\\"invalid_operation_checks\\\":"<<invalidChecks<<",\\\"source_identity_sha256\\\":"<<json::quote(compact_native_r5_verify_sep22::kSourceIdentitySha256)<<",\\\"worker\\\":"<<kPrefillExactBuildProvenance<<",\\\"kernel_routes\\\":"<<json::quote(target.kernelRoutes())<<'}';own=o.str();allocation.governor=governor.snapshot();allocation.finalAllocated=backend.memoryStats().allocatedBytes;
      require(allocation.governor.reservedBytes==0&&allocation.governor.deniedReservations==0,"final governor reserved/denied must be zero");'''
    text = once(text, old, new, 'fresh R5 final census no inherited score')
    old = 'template<class F>void rejected(F &&f,const char *label){bool bad=false;'
    new = 'template<class F>void rejected(metal::MetalBackend&backend,F &&f,const char *label){const auto submits=backend.submissionCount();bool bad=false;'
    text = once(text, old, new, 'each invalid operation authenticates no submit')
    text = once(text, 'require(bad,std::string("invalid operation accepted: ")+label);}', 'require(bad,std::string("invalid operation accepted: ")+label);require(backend.submissionCount()==submits,std::string("invalid operation submitted: ")+label);}', 'invalid no-submit gate')
    old = 'rejected([&]'
    if text.count(old) != 8:
        raise ValueError('eight original invalid calls changed')
    JOURNAL.append({'label':'eight rejected calls use actual backend','old':old,'new':'rejected(backend,[&]','count':8})
    text = text.replace(old, 'rejected(backend,[&]')
    text = once(text, 'LocalTails tails(target);uint64_t localChecks=0,invalidChecks=0;', '''LocalTails tails(target);uint64_t localChecks=0,invalidChecks=0;
      {FlashRequestState nullState;const bool nullPoisoned=nullState.poisoned();const std::array<uint32_t,1> one{prompt[0]};const auto submits=backend.submissionCount(),owned=backend.memoryStats().allocatedBytes;
        rejected(backend,[&]{(void)target.forward(nullState,one);},"null forward");
        rejected(backend,[&]{(void)target.verify(nullState,one);},"null verify");
        rejected(backend,[&]{(void)target.commitVerify(nullState,1);},"null commit");invalidChecks+=3;
        target.abortVerify(nullState);require(backend.submissionCount()==submits&&backend.memoryStats().allocatedBytes==owned&&nullState.poisoned()==nullPoisoned&&!Access::pending(nullState)&&Access::tapeStatus(target)==std::array<uint64_t,3>{0,0,0},"null abort changed ownership/tape/submit");tails.check();require(Access::tapeCanaries(target),"null operations damaged tapes");++localChecks;
      }''', 'null operation lifecycle before actual state')
    text = once(text, 'if(verification.empty())verification=', '''if(retained==1){const std::array<uint32_t,6> excess{prompt[0],prompt[1],prompt[2],prompt[3],prompt[4],prompt[5]};
          const auto noPending=[&]{store.state("prefill.state",target,state,true);binding.check(target,state);checkGuards(guards);tails.check();require(Access::tapeStatus(target)==std::array<uint64_t,3>{0,0,0},"rejected fresh operation armed tape");++localChecks;};
          rejected(backend,[&]{(void)target.forward(state,{});},"empty forward");noPending();
          rejected(backend,[&]{(void)target.verify(state,{});},"empty verify");noPending();
          rejected(backend,[&]{(void)target.verify(state,excess);},"six-row verify exceeds owned five-row tape");noPending();
          rejected(backend,[&]{(void)target.commitVerify(state,1);},"commit without pending");noPending();invalidChecks+=4;
        }
        if(verification.empty())verification=''', 'empty/excess/no-pending no-submit preserves full state')
    # Four additional repeated full physical state checks above.
    text = once(text, 'completed_==29&&repeated_==61', 'completed_==29&&repeated_==65', 'new no-submit repeated frames')
    text = once(text, '"expected_repeated_frames\\\":61', '"expected_repeated_frames\\\":65', 'new failure expected repeated')
    text = once(text, 'uint64_t beforeState=0,afterState=0,stateDelta=0,finalAllocated=0;', 'uint64_t beforeState=0,afterState=0,stateDelta=0,finalAllocated=0,afterTargetDestruction=0,afterModelDestruction=0;bool stopped=false;', 'actual destruction ledger fields')
    text = once(text, '<<",\\\"final_allocated\\\":"<<finalAllocated', '<<",\\\"after_target_destruction\\\":"<<afterTargetDestruction<<",\\\"after_model_destruction\\\":"<<afterModelDestruction<<",\\\"backend_stopped\\\":"<<(stopped?"true":"false")<<",\\\"final_allocated\\\":"<<finalAllocated', 'destruction ledger report')
    text = once(text, 'backendCreated=true;const auto weights=', 'backendCreated=true;{const auto weights=', 'nested actual mapped owner lifetime')
    text = once(text, 'stage="target";FlashForward target', 'stage="target";{FlashForward target', 'nested target owner lifetime')
    text = once(text, 'require(allocation.governor.reservedBytes==0&&allocation.governor.deniedReservations==0,"final governor reserved/denied must be zero");', '''require(allocation.governor.reservedBytes==0&&allocation.governor.deniedReservations==0,"final governor reserved/denied must be zero");
      }
      allocation.afterTargetDestruction=backend.memoryStats().allocatedBytes;require(allocation.afterTargetDestruction==allocation.mappedInitial,"target/state/tape owned bytes remain after destruction");allocation.governor=governor.snapshot();require(allocation.governor.reservedBytes==0&&allocation.governor.deniedReservations==0,"governor residue after target destruction");
      }
      allocation.afterModelDestruction=backend.memoryStats().allocatedBytes;require(allocation.afterModelDestruction==0,"mapped/native owned bytes remain after all owners destroyed");backend.drainSparseUnmaps();require(backend.healthy(),"backend unhealthy after synchronous proof drain");backend.stop();allocation.stopped=true;''', 'actual target model owner zero drain stop before backend destruction')
    # These checks precede token parsing, checkpoint RAM scratch, and disk writes.
    text = once(text, 'const auto prompt=tokens(argv[5]);', '''const uint64_t spillBound=17*FlashForward::requestStateBytes(kCapacity)+(96ULL<<20)+36*FlashGDNLazyRollback::plannedBytes(5,1)+(2ULL<<20);require(spillBound<kLimit,"R5 entire campaign physical spill bound exceeds4GiB before any checkpoint");
    const auto hostAvailable=engine::queryHostAvailableMemory();const auto hostReserve=std::max<uint64_t>(16ULL<<30,NSProcessInfo.processInfo.physicalMemory/10);require(hostAvailable&&*hostAvailable>hostReserve+(64ULL<<20),"bounded64MiB host streaming/metadata admission denied before checkpoint scratch");
    const auto prompt=tokens(argv[5]);''', 'before data scratch host and exact source geometry admission')
    text = once(text, 'const auto weights=FlashWeights::load(backend,argv[4]);', 'const auto weights=FlashWeights::load(backend,argv[4]);require(weights.sourceIdentity()=="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"&&weights.manifestFingerprint()=="edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0","original source/layout must match before target");', 'exact current model metadata before target')
    text = once(text, 'workspace=target.workspaceBytes();', 'require(target.batchInt8ExpertStore()&&target.batchInt8ExpertStore()->identitySha256()=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","original Full512 store identity differs before trunk");workspace=target.workspaceBytes();', 'exact original Full512 store before trunk')
    text = once(text, 'auto body=target.forward(state,prompt,false,true);', 'const auto r5BeforePref=compact_native_r5_verify_sep22::graphCalls.load(),r5RowsBeforePref=compact_native_r5_verify_sep22::graphRows.load();\n        auto body=target.forward(state,prompt,false,true);require(compact_native_r5_verify_sep22::graphCalls.load()==r5BeforePref&&compact_native_r5_verify_sep22::graphRows.load()==r5RowsBeforePref,"Prefill executed excluded R5 setup");', 'actual prefill excludes new R5')
    text = once(text, 'const auto before=state.logicalLength();auto result=target.forward(state,input,true,true);', 'const auto r5BeforeFuture=compact_native_r5_verify_sep22::graphCalls.load(),r5RowsBeforeFuture=compact_native_r5_verify_sep22::graphRows.load();const auto before=state.logicalLength();auto result=target.forward(state,input,true,true);require(compact_native_r5_verify_sep22::graphCalls.load()==r5BeforeFuture&&compact_native_r5_verify_sep22::graphRows.load()==r5RowsBeforeFuture,"ordinary future AR/four/eight rows executed excluded R5 setup");', 'actual ordinary future excludes new R5')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    args = parser.parse_args()
    build = (ROOT / args.build).resolve()
    if build.exists():
        raise SystemExit('fresh output required')
    ready = json.loads((WORKER / 'CPU_READY.json').read_text())
    if not ready['pass'] or ready['GPU_work'] or len(ready['compiled_objects']) != 54:
        raise SystemExit('current CPU closure required')
    source = oracle_source()
    HERE.mkdir(exist_ok=True)
    (HERE / 'oracle.mm').write_text(source)
    (HERE / 'source_delta.json').write_text(json.dumps({'parent':str(PARENT/'oracle.mm'),'parent_sha256':sha(PARENT/'oracle.mm'),'journal':JOURNAL},indent=2)+'\n')
    print(json.dumps({'source_prepared':str(HERE/'oracle.mm'),'source_sha256':sha(HERE/'oracle.mm'),'journal_entries':len(JOURNAL),'GPU_work':False,'build_pending':str(build)}))


if __name__ == '__main__':
    main()
