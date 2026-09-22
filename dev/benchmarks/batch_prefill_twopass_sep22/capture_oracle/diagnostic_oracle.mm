// Root-only replay of real batch QSA projections. --cpu-self-test is pure CPU.
// component_helpers.mm and base_oracle.mm are frozen copies of the original
// qualified component oracle; all numerical envelopes/functions remain exact.
#include "component_helpers.mm"
#include "batch_policy.hpp"

namespace {
constexpr uint64_t kActualReplayReservation=4ULL<<30;
constexpr uint32_t kActualCapacity=16384;
constexpr const char *kExpectedModelSource="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
constexpr const char *kExpectedRestoredLibrary="1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf";
constexpr const char *kExpectedParentWorker="2a9f674a04b32e0e21a40f95add0cc8c17c313c39c4760342b1b419ae0e43309";
constexpr const char *kExpectedCaptureReceipt="7310f721467ee733686e0a5a09e50c49d461c75c0c03e1420f792e031613411e";
constexpr const char *kExpectedLayout="edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0";
constexpr const char *kExpectedParentSeal="4a6fb14387ccfd5897b3cdc86fca0a89bbcf30ca8e42d616a60a40c933e374fc";
std::string textField(NSDictionary *d,NSString *key) {
  id v=d[key];require([v isKindOfClass:NSString.class],"capture manifest string field missing");
  return std::string([(NSString *)v UTF8String]);
}
uint64_t integerField(NSDictionary *d,NSString *key) {
  id v=d[key];require([v isKindOfClass:NSNumber.class],"capture manifest numeric field missing");
  const double n=[(NSNumber *)v doubleValue];require(std::isfinite(n)&&n>=0&&n<=double(UINT64_MAX)&&std::floor(n)==n,"capture manifest integer invalid");
  return [(NSNumber *)v unsignedLongLongValue];
}
bool trueField(NSDictionary *d,NSString *key) {
  id v=d[key];return [v isKindOfClass:NSNumber.class]&&[(NSNumber *)v boolValue];
}
NSDictionary *dictionary(id value) {require([value isKindOfClass:NSDictionary.class],"capture manifest object missing");return (NSDictionary *)value;}
std::string fileHash(const std::filesystem::path &p) {
  std::ifstream f(p,std::ios::binary);require(bool(f),"capture source file unreadable");SHA256 h;std::array<char,65536> chunk{};
  while(f) {f.read(chunk.data(),chunk.size());if(f.gcount())h.add(chunk.data(),uint64_t(f.gcount()));}
  require(f.eof(),"capture file read failed");return h.finish();
}
NormConvention convention(NSString *s) {
  require([s isKindOfClass:NSString.class],"capture norm convention missing");
  if([s isEqualToString:@"OnePlusWeight"])return NormConvention::OnePlusWeight;
  require([s isEqualToString:@"DirectGamma"],"capture norm convention unknown");return NormConvention::DirectGamma;
}
struct CaptureMetadata {uint32_t lanes=0,layer=0,lane=0;std::string source;};
CaptureMetadata validateMetadata(NSDictionary *manifest,std::string_view expectedClone) {CaptureMetadata m;
    require(textField(manifest,@"schema")=="splash-current-batch-qsa-projection-capture-v1","capture manifest schema differs");
    require(trueField(manifest,@"execution_complete")&&trueField(manifest,@"completed")&&trueField(manifest,@"cohortValidated"),"capture native completion/fresh cohort not proven");
    require(integerField(manifest,@"rows")==2048&&integerField(manifest,@"capacity")==kActualCapacity&&integerField(manifest,@"begin")==0,"capture shape/context differs");
    m.lanes=uint32_t(integerField(manifest,@"actualLanes"));m.layer=uint32_t(integerField(manifest,@"layer"));m.lane=uint32_t(integerField(manifest,@"lane"));
    require((m.lanes==2||m.lanes==4)&&m.layer==3&&m.lane==0,"capture must be real B2/B4 lane0 layer3");
    require([manifest[@"epsilon"] doubleValue]==1e-6&&[manifest[@"theta"] doubleValue]==1e7,"capture descriptor epsilon/theta differs");
    m.source=textField(manifest,@"sourceIdentity");require(m.source==kExpectedModelSource,"capture model source differs from qualified current source");
    require(expectedClone.size()==64&&std::all_of(expectedClone.begin(),expectedClone.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');})&&textField(manifest,@"capture_clone_executable_sha256")==expectedClone,"capture producer differs from externally pinned Root clone");
    require(textField(manifest,@"metallib_sha256")==kExpectedRestoredLibrary,"capture library differs from exact restored library");
    NSDictionary *provenance=dictionary(manifest[@"provenance"]);require(textField(provenance,@"parent_seal_sha256")==kExpectedParentSeal&&textField(provenance,@"parent_Worker_sha256")==kExpectedParentWorker&&textField(provenance,@"capture_source_receipt_sha256")==kExpectedCaptureReceipt,"capture parent/source/Worker closure differs");
    require(trueField(provenance,@"original_body_after_debug_only_normalization_exact")&&!trueField(provenance,@"public_header_layout_or_Metal_math_changes"),"capture original math/source scope differs");
    require(textField(manifest,@"layout")==kExpectedLayout,"capture descriptor layout differs");
    NSDictionary *preservation=dictionary(manifest[@"preservation"]),*allocation=dictionary(manifest[@"allocation"]);
    require(trueField(preservation,@"all_bytes_equal")&&trueField(preservation,@"same_Forward_Batch_model_process")&&trueField(preservation,@"only_one_cohort_live")&&integerField(preservation,@"all_physical_state_planes_per_lane")==134,"capture old-path preservation proof incomplete");
    require(trueField(allocation,@"backend_destroyed")&&trueField(allocation,@"target_state_capture_teardown_to_model_ledger")&&trueField(allocation,@"host_measurement_valid")&&trueField(allocation,@"growth_allowed")&&!integerField(allocation,@"denied_reservations"),"capture teardown/host admission proof incomplete");
    NSDictionary *graph=dictionary(manifest[@"graph"]);NSArray *producer=graph[@"legacy_producers"],*submissions=graph[@"native_submissions_per_cohort"];
    require([producer isKindOfClass:NSArray.class]&&producer.count==6&&integerField(graph,@"debug_copy_dispatches")==9&&integerField(graph,@"legacy_QSA_dispatches")==6&&integerField(graph,@"output_copy_ordinal")==integerField(graph,@"input_copy_ordinal")+14,"capture actual copy/producer graph differs");
    require([submissions isKindOfClass:NSArray.class]&&submissions.count==2&&[submissions[0] unsignedLongLongValue]==1&&[submissions[1] unsignedLongLongValue]==1,"capture native submission contract differs");
    const std::array<std::string,6> names{"flash_qsa_fast_prepare","","flash_qsa_select_blocks","flash_qsa_mpp_prefill_bulk_early_2048","flash_qsa_mpp_prefill_bulk_temporal_sg8_2048","flash_qsa_fast_prefill_bulk_reduce_2048"};
    for(size_t i=0;i<6;++i) {NSDictionary *record=dictionary(producer[i]);const std::string s=textField(record,@"pipeline");require(integerField(record,@"ordinal")==integerField(graph,@"input_copy_ordinal")+8+i,"capture actual producer ordinal differs");require(i==1?s.starts_with("flash_qsa_pool_rope_"):s==names[i],"capture producer pipeline source differs");}
 return m;
}
struct Captured final {
  NSDictionary *manifest; std::filesystem::path directory; std::string manifestHash,source;
  std::array<MetalBuffer,4> projections;std::array<FlashTensor,4> norms;
  std::array<NormConvention,4> conventions;MetalBuffer originalOutput;
  uint32_t lanes=0,layer=0,lane=0;
  Captured(MetalBackend &b,const std::filesystem::path &path,std::string_view expectedClone):directory(path.parent_path()),manifestHash(fileHash(path)) {
    NSError *error=nil;NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
    manifest=dictionary([NSJSONSerialization JSONObjectWithData:data options:0 error:&error]);require(!error,"capture manifest JSON parse failed");
    const auto metadata=validateMetadata(manifest,expectedClone);lanes=metadata.lanes;layer=metadata.layer;lane=metadata.lane;source=metadata.source;
    NSDictionary *files=dictionary(manifest[@"files"]);
    const std::array<NSString *,4> labels{@"q",@"k",@"v",@"index"};const std::array<uint64_t,4> bytes{50331648,2097152,2097152,2621440};
    const std::array<uint32_t,4> widths{12288,512,512,640};
    const auto load=[&](NSString *name,uint64_t expected) {
      NSDictionary *record=dictionary(files[name]);require(integerField(record,@"bytes")==expected,"capture file declared extent differs");
      const auto relative=std::filesystem::path(textField(record,@"path"));require(!relative.is_absolute(),"capture filenames must be relative to manifest");
      const auto file=std::filesystem::canonical(directory/relative),parent=std::filesystem::canonical(directory);
      const auto rel=std::filesystem::relative(file,parent);require(!rel.empty()&&*rel.begin()!="..","capture path escapes immutable capture folder");
      require(std::filesystem::file_size(file)==expected&&fileHash(file)==textField(record,@"sha256"),"capture file actual extent/hash differs");
      auto result=b.allocateBuffer(expected,BufferStorage::Shared,"Root actual immutable captured tensor");std::ifstream f(file,std::ios::binary);f.read(static_cast<char *>(result.contents()),std::streamsize(expected));require(uint64_t(f.gcount())==expected,"capture tensor read truncated");return result;
    };
    for(size_t i=0;i<4;++i) {NSDictionary *record=dictionary(files[labels[i]]);require(textField(record,@"dtype")=="BF16","captured projection dtype differs");
      NSArray *shape=record[@"shape"];require([shape isKindOfClass:NSArray.class]&&shape.count==2&&[shape[0] unsignedLongLongValue]==2048&&[shape[1] unsignedLongLongValue]==widths[i],"captured projection shape differs");projections[i]=load(labels[i],bytes[i]);}
    {NSDictionary *record=dictionary(files[@"output"]);NSArray *shape=record[@"shape"];require(textField(record,@"dtype")=="BF16"&&[shape isKindOfClass:NSArray.class]&&shape.count==2&&[shape[0] unsignedLongLongValue]==2048&&[shape[1] unsignedLongLongValue]==6144,"actual old output dtype/shape differs");}
    originalOutput=load(@"output",25165824);
    const std::array<NSString *,4> ns{@"q_norm",@"k_norm",@"index_q_norm",@"index_k_norm"};
    const std::array<uint32_t,4> normWidth{256,256,128,128};
    NSDictionary *cs=dictionary(manifest[@"conventions"]);require(cs.count==4,"all four actual norm conventions required");
    for(size_t i=0;i<4;++i) {
      NSDictionary *record=dictionary(files[ns[i]]);const auto dtype=textField(record,@"dtype");require(dtype=="BF16"||dtype=="F32","actual norm dtype differs");
      auto &t=norms[i];t.dtype=dtype=="F32"?FlashDType::F32:FlashDType::BF16;NSArray *shape=record[@"shape"];
      require([shape isKindOfClass:NSArray.class]&&shape.count==1&&[shape[0] unsignedLongLongValue]==normWidth[i],"actual QSA norm shape differs");
      t.shape={normWidth[i]};t.logicalBytes=uint64_t(normWidth[i])*(t.dtype==FlashDType::F32?4:2);t.buffer=load(ns[i],t.logicalBytes);conventions[i]=convention(cs[ns[i]]);
    }
    for(const auto &p:projections)for(uint64_t i=0;i<p.sizeBytes()/2;++i)require(std::isfinite(number(static_cast<const uint16_t *>(p.contents())[i])),"actual captured projection nonfinite");
    for(const auto &t:norms)for(uint64_t i=0;i<t.shape[0];++i)require(std::isfinite(t.dtype==FlashDType::F32?static_cast<const float *>(t.buffer.contents())[i]:number(static_cast<const uint16_t *>(t.buffer.contents())[i])),"actual captured norm nonfinite");
  }
  FlashQSAFastInputs input(MetalBuffer output,MetalBuffer diag) const {return {projections[0],projections[1],projections[2],projections[3],&norms[0],&norms[1],&norms[2],&norms[3],output,diag,{},conventions[0],conventions[1],conventions[2],conventions[3],1e-6,1e7};}
  std::string hash() const {return buffersHash({projections[0],projections[1],projections[2],projections[3],norms[0].buffer,norms[1].buffer,norms[2].buffer,norms[3].buffer,originalOutput});}
};
uint64_t actualMetadataCPUChecks() {
 const std::string clone(64,'a');NSMutableArray *producer=[NSMutableArray array];
 const std::array<const char *,6> names{"flash_qsa_fast_prepare","flash_qsa_pool_rope_bf16","flash_qsa_select_blocks","flash_qsa_mpp_prefill_bulk_early_2048","flash_qsa_mpp_prefill_bulk_temporal_sg8_2048","flash_qsa_fast_prefill_bulk_reduce_2048"};
 for(size_t i=0;i<6;++i)[producer addObject:@{@"ordinal":@(108+i),@"pipeline":[NSString stringWithUTF8String:names[i]]}];
 NSDictionary *valid=@{@"schema":@"splash-current-batch-qsa-projection-capture-v1",@"execution_complete":@YES,@"completed":@YES,@"cohortValidated":@YES,@"rows":@2048,@"capacity":@16384,@"begin":@0,@"actualLanes":@4,@"layer":@3,@"lane":@0,@"epsilon":@1e-6,@"theta":@1e7,@"sourceIdentity":[NSString stringWithUTF8String:kExpectedModelSource],@"capture_clone_executable_sha256":[NSString stringWithUTF8String:clone.c_str()],@"metallib_sha256":[NSString stringWithUTF8String:kExpectedRestoredLibrary],@"layout":[NSString stringWithUTF8String:kExpectedLayout],@"provenance":@{@"parent_seal_sha256":[NSString stringWithUTF8String:kExpectedParentSeal],@"parent_Worker_sha256":[NSString stringWithUTF8String:kExpectedParentWorker],@"capture_source_receipt_sha256":[NSString stringWithUTF8String:kExpectedCaptureReceipt],@"original_body_after_debug_only_normalization_exact":@YES,@"public_header_layout_or_Metal_math_changes":@NO},@"preservation":@{@"all_bytes_equal":@YES,@"same_Forward_Batch_model_process":@YES,@"only_one_cohort_live":@YES,@"all_physical_state_planes_per_lane":@134},@"allocation":@{@"backend_destroyed":@YES,@"target_state_capture_teardown_to_model_ledger":@YES,@"host_measurement_valid":@YES,@"growth_allowed":@YES,@"denied_reservations":@0},@"graph":@{@"legacy_producers":producer,@"native_submissions_per_cohort":@[@1,@1],@"debug_copy_dispatches":@9,@"legacy_QSA_dispatches":@6,@"input_copy_ordinal":@100,@"output_copy_ordinal":@114}};
 uint64_t checks=0;for(uint32_t width:{2u,4u}) {NSMutableDictionary *d=[valid mutableCopy];d[@"actualLanes"]=@(width);require(validateMetadata(d,clone).lanes==width,"CPU real width metadata selector failed");++checks;}
 for(uint32_t fault=0;fault<22;++fault) {
  NSData *encoded=[NSJSONSerialization dataWithJSONObject:valid options:0 error:nil];NSMutableDictionary *d=[NSJSONSerialization JSONObjectWithData:encoded options:NSJSONReadingMutableContainers error:nil];
  switch(fault) {
   case 0:d[@"schema"]=@"wrong";break;case 1:d[@"execution_complete"]=@NO;break;case 2:d[@"completed"]=@NO;break;case 3:d[@"cohortValidated"]=@NO;break;
   case 4:d[@"rows"]=@2047;break;case 5:d[@"capacity"]=@4096;break;case 6:d[@"begin"]=@1;break;case 7:d[@"actualLanes"]=@3;break;
   case 8:d[@"layer"]=@7;break;case 9:d[@"lane"]=@1;break;case 10:d[@"epsilon"]=@1e-5;break;case 11:d[@"theta"]=@1e6;break;
   case 12:d[@"sourceIdentity"]=@"foreign";break;case 13:d[@"capture_clone_executable_sha256"]=@"foreign";break;case 14:d[@"metallib_sha256"]=@"foreign";break;
   case 15:d[@"layout"]=@"foreign";break;case 16:((NSMutableDictionary *)d[@"provenance"])[@"parent_seal_sha256"]=@"foreign";break;
   case 17:((NSMutableDictionary *)d[@"preservation"])[@"all_physical_state_planes_per_lane"]=@133;break;
   case 18:((NSMutableDictionary *)d[@"allocation"])[@"backend_destroyed"]=@NO;break;
   case 19:((NSMutableDictionary *)d[@"graph"])[@"output_copy_ordinal"]=@113;break;
   case 20:((NSMutableDictionary *)d[@"graph"])[@"native_submissions_per_cohort"]=@[@1,@2];break;
   default:((NSMutableDictionary *)((NSArray *)((NSDictionary *)d[@"graph"])[@"legacy_producers"])[4])[@"pipeline"]=@"arbitrary";break;
  }
  bool rejected=false;try{(void)validateMetadata(d,clone);}catch(const std::exception &){rejected=true;}require(rejected,"CPU metadata refusal missing fault"+std::to_string(fault));++checks;
 }
 return checks;
}
struct RowDiagnostics {uint64_t failures=0,firstFailure=2048;double maximumL2=0,minimumCosine=1;std::string json;};
RowDiagnostics allRowDiagnostics(const Captured &capture,std::span<const uint16_t> actual,std::span<const uint16_t> expected,const float *raw,const float *rawBase) {
 require(actual.size()==2048ULL*6144&&expected.size()==actual.size(),"diagnostic exact real2048 output spans");RowDiagnostics result;std::ostringstream out;out<<std::setprecision(17)<<'[';
 for(uint32_t row=0;row<2048;++row){const auto a=actual.subspan(uint64_t(row)*6144,6144),b=expected.subspan(uint64_t(row)*6144,6144);const auto error=compare(a,b);
  const bool pass=!error.nonfinite&&error.relativeL2()<=kSourceBF16L2&&error.cosine()>=kSourceCosine;
  if(!pass){++result.failures;result.firstFailure=std::min<uint64_t>(result.firstFailure,row);}result.maximumL2=std::max(result.maximumL2,error.relativeL2());result.minimumCosine=std::min(result.minimumCosine,error.cosine());
  uint64_t actualZeros=0,referenceZeros=0,firstDiff=6144;FloatError rawError;for(uint32_t head=0;head<24;++head)for(uint32_t d=0;d<256;++d){const uint32_t column=head*256+d;const auto x=a[column],y=b[column];actualZeros+=number(x)==0;referenceZeros+=number(y)==0;if(x!=y)firstDiff=std::min<uint64_t>(firstDiff,column);const uint64_t at=(uint64_t(head/12)*24576+row*12+head%12)*256+d;rawError.add(raw[at],rawBase[at]);}
  if(row)out<<',';out<<"{\"row\":"<<row<<",\"valid_visible_tokens\":"<<row+1<<",\"legacy_shape_bucket\":"<<splash::json::quote(row<128?"early128":"temporalSG8")<<",\"required_per_row_pass\":"<<(pass?"true":"false")<<",\"BF16_error\":";error.write(out);
  out<<",\"actual_l2_norm\":"<<std::sqrt(error.squareActual)<<",\"reference_l2_norm\":"<<std::sqrt(error.squareReference)<<",\"absolute_l2_error\":"<<std::sqrt(error.squareError)<<",\"actual_zero_cells\":"<<actualZeros<<",\"reference_zero_cells\":"<<referenceZeros<<",\"first_differing_output_column\":"<<(firstDiff==6144?-1:int64_t(firstDiff))<<",\"first_differing_head\":"<<(firstDiff==6144?-1:int64_t(firstDiff/256))<<",\"raw_F32_error\":";rawError.write(out);
  out<<",\"projection_scales\":{";const std::array<const char *,4>names{"q_query_and_gate","k","v","index"};const std::array<uint32_t,4>widths{12288,512,512,640};
  for(uint32_t plane=0;plane<4;++plane){double sum=0,maxAbs=0;uint64_t zero=0,nonfinite=0;const auto*p=static_cast<const uint16_t *>(capture.projections[plane].contents())+uint64_t(row)*widths[plane];for(uint32_t d=0;d<widths[plane];++d){const double value=number(p[d]);nonfinite+=!std::isfinite(value);if(std::isfinite(value)){sum+=value*value;maxAbs=std::max(maxAbs,std::abs(value));zero+=value==0;}}if(plane)out<<',';out<<splash::json::quote(names[plane])<<":{\"elements\":"<<widths[plane]<<",\"rms\":"<<std::sqrt(sum/widths[plane])<<",\"max_abs\":"<<maxAbs<<",\"zeros\":"<<zero<<",\"nonfinite\":"<<nonfinite<<'}';}
  out<<"},\"head_error_buckets\":[";for(uint32_t head=0;head<24;++head){if(head)out<<',';const auto e=compare(a.subspan(head*256,256),b.subspan(head*256,256));out<<"{\"head\":"<<head<<",\"error\":";e.write(out);out<<'}';}out<<"]}";
 }
 out<<']';result.json=out.str();return result;
}
}

int main(int argc,char **argv) {
 @autoreleasepool {
  std::string phase="arguments",metrics,manifestHash,finalReport;uint64_t rowChecks=0;
  try {
   if(argc==2&&std::string(argv[1])=="--cpu-self-test") {std::cout<<"{\"pass\":true,\"cpu_checks\":"<<(cpuChecks()+actualMetadataCPUChecks())<<",\"actual_metadata_checks\":24,\"gpu_commands\":0,\"actual_input_required\":true}\n";return 0;}
   require(argc==5,"usage: actual-qsa-oracle METALLIB CAPTURE_MANIFEST EXPECTED_CAPTURE_CLONE_SHA256 FRESH_REPORT | --cpu-self-test");require(!std::filesystem::exists(argv[4]),"actual QSA report must be fresh");
   require(single("BATCH_QSA_REPLAY_ROOT_GPU",0,1)==1,"GPU replay requires explicit Root authorization");
   const uint32_t repeats=single("BATCH_QSA_REPLAY_PAIRS",20,32);require(repeats>=4&&!(repeats%4),"timing pairs must be multiple of4");
   // Includes both509MB workspaces, a guarded509MB helper replica,234MB legacy,
   // three16384 five-plane caches +guard copies, actual84MB projections/output,
   // ordinary128-row scratch/partial buffers, output/raw guards and host snapshots.
   require(3ULL*509607936+234356736+6ULL*(uint64_t(kActualCapacity)*512*4+uint64_t(kActualCapacity)*128*2+uint64_t(kActualCapacity/4)*128*2+uint64_t(kActualCapacity)*8)+512ULL*1024*1024<kActualReplayReservation,"complete replay reservation bound exceeded");
   require(fileHash(argv[1])==kExpectedRestoredLibrary,"replay library differs from externally qualified restored artifact");
   phase="backend-and-admission";{MetalBackend backend(argv[1]);const uint64_t backendInitial=backend.memoryStats().allocatedBytes;const uint64_t physical=NSProcessInfo.processInfo.physicalMemory;
   const uint64_t hostReserve=splash::engine::EngineMemoryPolicy::hostAvailableReserveBytes(physical);require(physical>hostReserve,"original host reserve unavailable");
   splash::engine::MemoryGovernor governor(backend,std::min<uint64_t>(8ULL<<30,physical-hostReserve),hostReserve);{auto reservation=governor.tryReserve(kActualReplayReservation);require(bool(reservation),"real governor denied complete replay before allocations");
   phase="actual-capture-load";Captured capture(backend,argv[2],argv[3]);manifestHash=capture.manifestHash;const auto immutable=capture.hash();
   Guarded legacyOut(backend,2048ULL*6144),nativeOut(backend,2048ULL*6144),helperOut(backend,2048ULL*6144);
   auto diagA=backend.allocateBuffer(4,BufferStorage::Shared,"actual legacy diagnostics"),diagB=backend.allocateBuffer(4,BufferStorage::Shared,"actual native diagnostics"),diagC=backend.allocateBuffer(4,BufferStorage::Shared,"actual unchanged helper diagnostics");
   for(const auto &d:{diagA,diagB,diagC})*static_cast<uint32_t *>(d.contents())=kSticky;
   auto a=allocateQSAState(backend,kActualCapacity),b=allocateQSAState(backend,kActualCapacity),c=allocateQSAState(backend,kActualCapacity);
   StateGuards guardA(backend,a),guardB(backend,b),guardC(backend,c);poisonPadding(a);poisonPadding(b);poisonPadding(c);
   auto wa=allocateQSAWorkspace(backend,128,kActualCapacity),wb=allocateQSAWorkspace(backend,128,kActualCapacity),wc=allocateQSAWorkspace(backend,128,kActualCapacity);
   auto fa=allocateQSAOnlineMPPWorkspace(backend,128,32),fb=allocateQSAOnlineMPPWorkspace(backend,128,32),fc=allocateQSAOnlineMPPWorkspace(backend,128,32);
   auto bulk=allocateBulkExactWorkspace(backend);splash::flash::batch_prefill_twopass_sep22::Workspace nativeWorkspace(backend);
   auto &testWorkspace=nativeWorkspace.qualified;auto helperWorkspace=allocateTwoPassWorkspace(backend,twoPassPlannedBytes());
   GuardedBytes qGuard(backend,25165824),scoreGuard(backend,402653184),rawGuard(backend,50331648),baseRaw(backend,50331648);
   GuardedBytes preparedQGuard(backend,25165824),preparedIndexGuard(backend,2097152),preparedSelectionGuard(backend,4194304);
   helperWorkspace.packedQueries=qGuard.view;helperWorkspace.scoresAndProbabilities=scoreGuard.view;helperWorkspace.rawAttention=rawGuard.view;
   helperWorkspace.prepared.queries=preparedQGuard.view;helperWorkspace.prepared.indexQueries=preparedIndexGuard.view;helperWorkspace.prepared.selectedBlocks=preparedSelectionGuard.view;
   require(backend.memoryStats().peakAllocatedBytes<=kActualReplayReservation,"actual replay allocatedSize peak exceeded complete admission");
   const auto ia=capture.input(legacyOut.view,diagA),ib=capture.input(nativeOut.view,diagB),ic=capture.input(helperOut.view,diagC);
   CommandGraph baseline,test,helper;addBulkExactQSA(backend,baseline,ia,a,wa,fa,bulk,0,2048,true);
   addTwoPassQSA(backend,test,ib,b,wb,fb,testWorkspace,0,2048,false,true);addTwoPassQSA(backend,helper,ic,c,wc,fc,helperWorkspace,0,2048,false,true);
   require(baseline.dispatches().size()==6&&test.dispatches().size()==9&&helper.dispatches().size()==9,"actual component graph shapes differ");
   phase="old-authoritative-ordinary-prefix";std::vector<uint8_t> queries(25165824),indexQueries(2097152);std::vector<uint32_t> selection(2048*512);
   for(uint32_t offset=0;offset<2048;offset+=128) {CommandGraph chunk;addOrdinaryQSAChunk(chunk,sliceCoalescedInputs(backend,ia,offset,128),a,wa,fa,offset,128);(void)backend.submitCommand(chunk.dispatches());
    std::memcpy(queries.data()+uint64_t(offset)*6144*2,wa.queries.contents(),128*6144*2);std::memcpy(indexQueries.data()+uint64_t(offset)*512*2,wa.indexQueries.contents(),128*512*2);std::memcpy(selection.data()+uint64_t(offset)*512,wa.selectedBlocks.contents(),128*512*4);}
   const std::vector<uint16_t> originalOutput(legacyOut.values().begin(),legacyOut.values().end());
   require(!std::memcmp(originalOutput.data(),capture.originalOutput.contents(),25165824),"captured old layer output differs from authoritative SAME projections ordinary QSA");
   phase="legacy-native-helper-outputs";(void)backend.submitCommand(baseline.dispatches());require(compare(legacyOut.values(),originalOutput).mismatches==0,"SG8 replay changed old authoritative output");
   CommandGraph extract;const FlashQSAFastParams extractParams{{2048,0,kActualCapacity,kActualCapacity/4,0,512,0,0,1e-6f,1e7f,0,0},0,0,4,4};
   extract.add("sep21_qsa_twopass_control_raw",{bulk.partials.partitionStatistics,bulk.partials.partitionValues,baseRaw.view},extractParams,{2048,24,1},{256,1,1});(void)backend.submitCommand(extract.dispatches());
   const auto qkOnly=range(test,0,5);(void)backend.submitCommand(qkOnly.dispatches());Certificate scoreCertificate;const auto reference=f64Reference(testWorkspace,b,scoreCertificate);
   {std::ostringstream o;o<<"{\"QK_f64_certificate\":";scoreCertificate.write(o);o<<'}';metrics=o.str();}
   require(!scoreCertificate.failures&&!scoreCertificate.signFailures,"actual QK failed original preregistered F64 operand envelope");
   const auto *scores=static_cast<const float *>(testWorkspace.scoresAndProbabilities.contents());uint64_t maskedCells=0;
   for(uint32_t kv=0;kv<2;++kv)for(uint32_t flat=0;flat<24576;++flat)for(uint32_t token=0;token<2048;++token){const float score=scores[(uint64_t(kv)*24576+flat)*2048+token];if(token>flat/12){require(score==-INFINITY,"actual QK future score lacks exact -Inf mask");++maskedCells;}else require(std::isfinite(score),"actual QK live score nonfinite");}
   (void)backend.submitCommand(test.dispatches());(void)backend.submitCommand(helper.dispatches());equalState(a,b);equalState(a,c);
   require(compare(nativeOut.values(),helperOut.values()).mismatches==0,"native dedicated batch workspace differs bitwise from unchanged singleton SAME-input helper");
   require(!std::memcmp(testWorkspace.rawAttention.contents(),helperWorkspace.rawAttention.contents(),50331648),"native batch raw attention differs bitwise from unchanged SAME-input helper");
   for(const auto *w:{&testWorkspace,&helperWorkspace}) {
    require(!std::memcmp(queries.data(),w->prepared.queries.contents(),queries.size()),"actual original Q norm/RoPE bytes changed");require(!std::memcmp(indexQueries.data(),w->prepared.indexQueries.contents(),indexQueries.size()),"actual original index-Q bytes changed");require(!std::memcmp(selection.data(),w->prepared.selectedBlocks.contents(),selection.size()*4),"actual original chronology/block selection changed");}
   const auto *raw=static_cast<const float *>(testWorkspace.rawAttention.contents()),*rawBase=static_cast<const float *>(baseRaw.view.contents());FloatError rawError;
   for(uint64_t i=0;i<50331648/4;++i)rawError.add(raw[i],rawBase[i]);const auto outputError=compare(nativeOut.values(),legacyOut.values());Certificate rawCertificate,baselineCertificate;Error gatedF64;
   for(const auto &r:reference)for(uint32_t i=0;i<columns.size();++i) {const uint64_t at=(uint64_t(r.head/12)*24576+r.row*12+r.head%12)*256+columns[i];rawCertificate.add(raw[at],r.raw[i],kRawAbs,kRawRelative);baselineCertificate.add(rawBase[at],r.raw[i],kRawAbs,kRawRelative);
    const uint64_t out=(uint64_t(r.row)*24+r.head)*256+columns[i];const auto gate=static_cast<const uint16_t *>(capture.projections[0].contents())[uint64_t(r.row)*12288+r.head*512+256+columns[i]];gatedF64.add(nativeOut.values()[out],gated(bf16(float(r.raw[i])),gate));}
   std::ostringstream quality;quality<<std::setprecision(16)<<"{\"raw_source_error\":";rawError.write(quality);quality<<",\"BF16_source_error\":";outputError.write(quality);quality<<",\"QK_f64_certificate\":";scoreCertificate.write(quality);quality<<",\"raw_f64_certificate\":";rawCertificate.write(quality);quality<<",\"baseline_raw_f64_certificate\":";baselineCertificate.write(quality);quality<<",\"staged_gate_f64_sample_error\":";gatedF64.write(quality);quality<<'}';metrics=quality.str();
   phase="collect-all-real-row-diagnostics";const auto rows=allRowDiagnostics(capture,nativeOut.values(),legacyOut.values(),raw,rawBase);rowChecks=2048;
   const bool oldGlobalPass=!rawError.nonfinite&&!rawError.signFlips&&rawError.l2()<=kSourceRawL2&&rawError.cosine()>=kSourceCosine&&!outputError.nonfinite&&outputError.relativeL2()<=kSourceBF16L2&&outputError.cosine()>=kSourceCosine;
   const bool f64Pass=!scoreCertificate.failures&&!scoreCertificate.signFailures&&!rawCertificate.failures&&!rawCertificate.signFailures&&!baselineCertificate.failures&&!baselineCertificate.signFailures&&!gatedF64.nonfinite&&gatedF64.relativeL2()<=kSourceBF16L2&&gatedF64.cosine()>=kSourceCosine;
   equalState(a,b);equalState(a,c);qGuard.check();scoreGuard.check();rawGuard.check();baseRaw.check();preparedQGuard.check();preparedIndexGuard.check();preparedSelectionGuard.check();guardA.check();guardB.check();guardC.check();legacyOut.check(true);nativeOut.check(true);helperOut.check(true);
   require(capture.hash()==immutable,"diagnostic immutable actual capture changed");require(backend.memoryStats().peakAllocatedBytes<=kActualReplayReservation,"diagnostic allocatedSize peak exceeded complete admission");for(const auto &d:{diagA,diagB,diagC})require(*static_cast<uint32_t *>(d.contents())==kSticky,"diagnostic valid graph numeric flags");reservation->commit();const auto admission=governor.snapshot();require(!admission.reservedBytes&&!admission.deniedReservations&&admission.hostMeasurementValid&&admission.growthAllowed,"diagnostic real governor final admission");
   std::ostringstream report;report<<std::setprecision(17)<<"{\"schema\":\"actual-batch-QSA-all2048-row-diagnostic-v1\",\"diagnostic_execution_complete\":true,\"numerical_qualification_claimed\":false,\"normal_policy_qualified\":false,\"timing_executed\":false,\"warm_executed\":false,\"captured_old_output_ordinary_and_SG8_exact\":true,\"batch_dedicated_vs_unchanged_singleton_SAME_input_output_and_raw_bitwise\":true,\"prepared_Q_index_selection_and_all5_cache_planes_exact\":true,\"actual_source_identity\":"<<splash::json::quote(capture.source)<<",\"capture_manifest_sha256\":"<<splash::json::quote(manifestHash)<<",\"capture_producer_sha256\":"<<splash::json::quote(argv[3])<<",\"actual_lanes\":"<<capture.lanes<<",\"causal_negative_inf_QK_cells\":"<<maskedCells<<",\"row_count\":2048,\"capacity\":16384,\"layer\":3,\"lane\":0,\"old_original_component_global_envelopes_pass\":"<<(oldGlobalPass?"true":"false")<<",\"original_F64_QK_raw_gate_envelopes_pass\":"<<(f64Pass?"true":"false")<<",\"new_required_batch_every_row_gate_pass\":"<<(rows.failures?"false":"true")<<",\"every_row_gate_relative_l2_max\":"<<kSourceBF16L2<<",\"every_row_gate_cosine_min\":"<<kSourceCosine<<",\"every_row_gate_nonfinite_max\":0,\"required_per_row_failure_count\":"<<rows.failures<<",\"first_required_per_row_failure\":"<<(rows.failures?int64_t(rows.firstFailure):-1)<<",\"maximum_row_relative_l2\":"<<rows.maximumL2<<",\"minimum_row_cosine\":"<<rows.minimumCosine<<",\"global_quality\":"<<quality.str()<<",\"all2048_row_diagnostics\":"<<rows.json<<",\"standalone_admitted_before_allocations\":"<<kActualReplayReservation<<",\"actual_native_allocatedSize_peak\":"<<backend.memoryStats().peakAllocatedBytes<<",\"governor_reserved_bytes\":0,\"governor_denied_reservations\":0,\"host_available_bytes\":"<<admission.hostAvailableBytes<<",\"host_reserve_bytes\":"<<admission.hostReserveBytes<<",\"host_headroom_bytes\":"<<admission.hostHeadroomBytes<<",\"host_measurement_valid\":true,\"growth_allowed\":true,\"no_floating_threshold_or_GPU_math_changes\":true}";finalReport=report.str();}
   require(backend.memoryStats().allocatedBytes==backendInitial,"diagnostic owned buffers did not return to initial native ledger");const auto released=governor.snapshot();require(!released.reservedBytes&&!released.deniedReservations&&released.hostMeasurementValid,"diagnostic final real reservation ledger invalid");finalReport.pop_back();finalReport+=",\"native_owned_allocation_ledger_returned_to_initial\":true,\"native_initial_allocated_bytes\":"+std::to_string(backendInitial)+",\"native_final_allocated_bytes\":"+std::to_string(backend.memoryStats().allocatedBytes)+",\"final_governor_reserved_bytes\":0}";}
   finalReport.pop_back();finalReport+=",\"backend_destroyed_before_publish\":true}";writeReport(argv[4],finalReport);std::cout<<"{\"diagnostic_execution_complete\":true,\"normal_policy_qualified\":false,\"timing_executed\":false,\"report\":"<<splash::json::quote(argv[4])<<"}\n";return 0;
  }catch(const std::exception &e){if(argc==5&&!std::filesystem::exists(argv[4]))try{writeReport(argv[4],"{\"execution_complete\":false,\"valid\":false,\"phase\":"+splash::json::quote(phase)+",\"error\":"+splash::json::quote(e.what())+",\"capture_manifest_sha256\":"+splash::json::quote(manifestHash)+",\"completed_per_row_checks\":"+std::to_string(rowChecks)+",\"failed_stage_metrics\":"+(metrics.empty()?"null":metrics)+"}");}catch(...){}std::cerr<<"actual batch QSA replay: "<<e.what()<<'\n';return 1;}
 }
}
