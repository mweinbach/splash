// Shared Root-only REF/CANDIDATE harness. Each executable links its own Batch TU.
#define main unused_legacy_batch_prefill_oracle_main
#include "dev/benchmarks/flash_batch_prefill_oracle.mm"
#undef main
#include "MainProofChecks.hpp"
#include "MainAlternativeProvenance.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "dev/benchmarks/dense_w8a8_sep21/worker_cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include <CommonCrypto/CommonDigest.h>
#ifdef SPLASH_BATCH_INTENDED_REFERENCE
#include "reference_qsa.hpp"
#else
#include "dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp"
#endif

namespace {
namespace fs=std::filesystem;
using main_proof_v3::Progress;
constexpr uint32_t capacity=4096,rows=2048;
constexpr uint64_t limit=4ULL<<30,hostAllowance=16ULL<<20;
[[maybe_unused]] constexpr uint64_t arenaBytes=509607936;
struct FailureTrace {std::string phase="arguments",checkpoint,plane;uint64_t offset=0,completedFrames=0;} failure;
struct View {std::string name;FlashDType type;uint64_t active,bytes;const void *data;};
struct Owned {std::vector<uint16_t> hidden,logits;std::vector<FlashGreedyGPURowResult> greedy;};
uint32_t exactWidth(std::string_view value){
  if(value=="2")return 2;if(value=="4")return 4;
  throw std::invalid_argument("intended math proof width must be literal2 or4");
}
uint64_t predicted(uint32_t width){
  // Full physical states at Prefill and after each of three genuine AR steps.
  const auto state=FlashForward::requestStateBytes(capacity);
  const uint64_t outputs=uint64_t(width)*(rows*kHyper*2ULL+kVocabulary*2ULL+sizeof(FlashGreedyGPURowResult))+
      uint64_t(3)*width*(kHyper*2ULL+kVocabulary*2ULL+sizeof(FlashGreedyGPURowResult));
  const uint64_t records=uint64_t(width)*4*134+3+uint64_t(width)*3*3;
  return uint64_t(width)*4*state+outputs+records*256+(1ULL<<20);
}
uint64_t exactCampaign(FlashBatchPrefill&batch,const std::vector<FlashRequestState>&states,
    uint32_t width,const std::string&source,const std::string&layout){
  const std::string magic="batch-intended-arithmetic-physical-campaign-v1";
  uint64_t bytes=8+magic.size()+16+8+source.size()+8+layout.size();
  const auto record=[](const std::string&name,uint64_t size){return 32+name.size()+size;};
  for(uint32_t step=0;step<4;++step){
    const std::string label=step?"genuine-greedy-AR"+std::to_string(step):"prefill";
    bytes+=16+label.size();
    for(uint32_t lane=0;lane<width;++lane)
      for(const auto&p:batch.inspectState(states[lane]))
        bytes+=record("lane"+std::to_string(lane)+"."+p.name,p.buffer.sizeBytes());
    if(!step){
      bytes+=record("prefill.hidden",uint64_t(width)*rows*kHyper*2);
      bytes+=record("prefill.logits",uint64_t(width)*kVocabulary*2);
      bytes+=record("prefill.complete_greedy",uint64_t(width)*sizeof(FlashGreedyGPURowResult));
    }else for(uint32_t lane=0;lane<width;++lane){
      const auto prefix="lane"+std::to_string(lane)+".future.";
      bytes+=record(prefix+"hidden",kHyper*2ULL)+record(prefix+"logits",kVocabulary*2ULL)+
        record(prefix+"complete_greedy",sizeof(FlashGreedyGPURowResult));
    }
  }
  require(bytes<predicted(width)&&bytes<limit,"EXACT actual-view whole campaign serialization preflight");
  return bytes;
}
class Stream {
 public:
  Stream(const fs::path&path,bool write,uint64_t maximum):path_(path),write_(write),maximum_(maximum){
    require(maximum<limit,"whole campaign spill bound before payload write");
    require(CC_SHA256_Init(&hash_),"stream SHA init");
    if(write){require(!fs::exists(path)&&!fs::exists(path.string()+".writing"),"fresh frame output");
      out_.open(path.string()+".writing",std::ios::binary);require(bool(out_),"frame create");}
    else{require(fs::is_regular_file(path)&&fs::file_size(path)==maximum,"exact regular reference frame extent");
      in_.open(path,std::ios::binary);require(bool(in_),"reference frame open");}
  }
  void bytes(const void*data,uint64_t count){
    failure.offset=count_;
    require(data||!count,"null frame span");require(count<=maximum_-count_,"exact campaign bound before each payload part");
    const auto*p=static_cast<const uint8_t*>(data);std::array<uint8_t,65536> scratch{};
    while(count){const size_t n=size_t(std::min<uint64_t>(count,scratch.size()));
      if(write_){out_.write(reinterpret_cast<const char*>(p),std::streamsize(n));require(bool(out_),"frame write");}
      else{in_.read(reinterpret_cast<char*>(scratch.data()),std::streamsize(n));require(in_.gcount()==std::streamsize(n),"truncated reference frame");
        if(std::memcmp(p,scratch.data(),n)){size_t at=0;while(at<n&&p[at]==scratch[at])++at;
          failure.offset=count_+at;throw std::runtime_error("intended-reference byte mismatch offset"+std::to_string(count_+at));}}
      require(CC_SHA256_Update(&hash_,p,CC_LONG(n)),"stream SHA update");count_+=n;p+=n;count-=n;
    }
  }
  void number(uint64_t value){bytes(&value,sizeof(value));}
  void text(const std::string&value){require(value.size()<65536,"bounded frame metadata");number(value.size());bytes(value.data(),value.size());}
  void frame(const std::string&label,const std::vector<View>&views){
    failure.phase="checkpoint";failure.checkpoint=label;failure.plane.clear();
    uint64_t expected=8+label.size()+8;for(const auto&v:views)expected+=8+v.name.size()+24+v.bytes;
    require(expected<=maximum_-count_,"whole checkpoint exact extent before write");
    text(label);number(views.size());
    for(const auto&v:views){failure.plane=v.name;require(v.data&&v.bytes&&v.active<=v.bytes,"complete readable checkpoint view");
      text(v.name);number(uint64_t(v.type));number(v.active);number(v.bytes);bytes(v.data,v.bytes);++planes_;}
    ++frames_;failure.completedFrames=frames_;
  }
  std::string finish(){
    require(count_==maximum_,"exact full campaign extent consumed");
    if(write_){out_.close();require(bool(out_),"frame close");fs::rename(path_.string()+".writing",path_);}
    else require(in_.peek()==std::char_traits<char>::eof(),"trailing reference frame bytes");
    std::array<uint8_t,CC_SHA256_DIGEST_LENGTH>d{};require(CC_SHA256_Final(d.data(),&hash_),"stream SHA final");
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto byte:d)out<<std::setw(2)<<unsigned(byte);return out.str();
  }
  uint64_t count()const{return count_;}uint64_t frames()const{return frames_;}uint64_t planes()const{return planes_;}
 private:
  fs::path path_;bool write_;uint64_t maximum_,count_=0,frames_=0,planes_=0;
  std::ifstream in_;std::ofstream out_;CC_SHA256_CTX hash_{};
};
void finiteSpan(const void*data,uint64_t bytes,FlashDType type,const std::string&name){
  const auto word=main_proof_v3::wordBytes(type);require(data&&bytes&&bytes%word==0,"typed full extent "+name);
  if(type!=FlashDType::BF16&&type!=FlashDType::F32)return;
  const auto*p=static_cast<const uint8_t*>(data);
  for(uint64_t i=0;i<bytes;i+=word){uint32_t bits=0;std::memcpy(&bits,p+i,word);if(word==2)bits<<=16;
    require(std::isfinite(std::bit_cast<float>(bits)),"nonfinite physical/output plane "+name);}
}
std::vector<View> stateViews(FlashBatchPrefill&batch,FlashForward&trunk,
    const std::vector<FlashRequestState>&states,const FlashDescriptor&descriptor,uint64_t context,Progress&progress){
  std::vector<View> views;const auto expected=main_proof_v3::expectedLayout(descriptor,context,progress);
  for(size_t lane=0;lane<states.size();++lane){const auto&s=states[lane];
    require(s.capacity()==capacity&&s.logicalLength()==context&&!s.poisoned()&&trunk.ownsState(s),"healthy owned nonpending state context");
    const auto planes=batch.inspectState(s);std::vector<main_proof_v3::PlaneMetadata> metadata;uint64_t physical=0;
    for(const auto&p:planes){require(p.buffer.storage()==metal::BufferStorage::Shared,"state Shared only");
      metadata.push_back({p.name,p.dtype,p.activeBytes,p.buffer.sizeBytes(),reinterpret_cast<uintptr_t>(p.buffer.contents())});}
    main_proof_v3::validatePlaneMetadata(metadata,expected,progress,"actual intended-policy state");
    for(const auto&p:planes){finiteSpan(p.buffer.contents(),p.buffer.sizeBytes(),p.dtype,p.name);physical+=p.buffer.sizeBytes();
      views.push_back({"lane"+std::to_string(lane)+"."+p.name,p.dtype,p.activeBytes,p.buffer.sizeBytes(),p.buffer.contents()});}
    require(physical==FlashForward::requestStateBytes(capacity),"all134 physical readable state bytes");
  }
  return views;
}
void outputViews(std::vector<View>&views,std::string prefix,const metal::MetalBuffer&hidden,
    const metal::MetalBuffer&logits,const metal::MetalBuffer&greedy,uint32_t count,uint32_t incoming,Progress&progress){
  const std::array<metal::MetalBuffer,3> buffers{hidden,logits,greedy};
  const std::array<uint64_t,3> bytes{uint64_t(count)*incoming*kHyper*2,uint64_t(count)*kVocabulary*2,uint64_t(count)*sizeof(FlashGreedyGPURowResult)};
  const std::array<FlashDType,3> types{FlashDType::BF16,FlashDType::BF16,FlashDType::U32};const std::array<const char*,3> names{"hidden","logits","complete_greedy"};
  for(size_t i=0;i<3;++i){main_proof_v3::validateOutputSpan(buffers[i].contents(),buffers[i].sizeBytes(),bytes[i],types[i],progress,prefix+names[i]);
    views.push_back({prefix+names[i],types[i],bytes[i],bytes[i],buffers[i].contents()});}
  const auto*g=static_cast<const FlashGreedyGPURowResult*>(greedy.contents());for(uint32_t row=0;row<count;++row)(void)greedyGPUResultToken(g[row],kVocabulary);
}
void publish(const fs::path&path,const std::string&text){
  require(!fs::exists(path)&&!fs::exists(path.string()+".writing"),"fresh report");std::ofstream out(path.string()+".writing",std::ios::binary);out<<text<<'\n';out.close();
  require(bool(out),"report write");fs::rename(path.string()+".writing",path);
}
uint64_t cpu(){
  require(sizeof(metal::CommandTiming)==200&&std::endian::native==std::endian::little,"source ABI/little endian");
  require(FlashForward::requestStateBytes(capacity)==232603648,"exact4K physical state source");
  require(predicted(4)<limit&&predicted(2)<predicted(4),"whole perwidth prefill+threeAR campaign <4GiB");
  return main_proof_v3::cpuSelfTest()+3;
}
int run(int argc,char**argv){
  require(argc==8,"oracle --export|--compare LIB PACKAGE TOKENS2048 2|4 FRESH_OR_REFERENCE_FRAME FRESH_REPORT");
  const bool exporting=std::string_view(argv[1])=="--export";require(exporting||std::string_view(argv[1])=="--compare","explicit Root role");
#ifdef SPLASH_BATCH_INTENDED_REFERENCE
  require(exporting,"intended reference executable exports only");constexpr const char*role="intended-reference";
#else
  require(!exporting,"service candidate executable compares only");constexpr const char*role="service-candidate";
  require(batch_prefill_twopass_sep22::requested(),"candidate new policy flag1 required");
#endif
  const auto width=exactWidth(argv[5]);const auto bound=predicted(width);require(bound<limit,"preregistered entire campaign preflight");
  require(!fs::exists(argv[7]),"fresh report");require(flag("SPLASH_FLASH_GPU_GREEDY",false)&&flashPrivateBatchBulkPrefillEnabled(),"actual original greedy/bulk profile");
  const auto fixture=loadTokens(argv[4]);require(fixture.size()==rows,"exact original2048 tokens");
  std::array<uint8_t,CC_SHA256_DIGEST_LENGTH> fixtureDigest{};
  require(CC_SHA256(fixture.data(),CC_LONG(fixture.size()*4),fixtureDigest.data()),"fixture word SHA");
  std::ostringstream fixtureSHA;fixtureSHA<<std::hex<<std::setfill('0');for(auto b:fixtureDigest)fixtureSHA<<std::setw(2)<<unsigned(b);
  require(fixtureSHA.str()=="0a383d21f5c784b0616d589847ca6cb04c69bf654729f2344e2b916e542f36b4","original canonical fixture words");
  Progress progress;uint64_t peak=0,targetDelta=0,batchDelta=0,stateDelta=0,referenceDelta=0,calls=0,exactBytes=0;std::string source,layout,digest;uint64_t frameBytes=0,frames=0,planes=0;
  {
    failure.phase="backend-model";metal::MetalBackend backend(argv[2]);const auto weights=FlashWeights::load(backend,argv[3]);source=weights.sourceIdentity();layout=weights.manifestFingerprint();
    require(source=="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"&&layout=="edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0","exact model/source layout");
    const auto mapped=backend.memoryStats().allocatedBytes;const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);
    engine::MemoryGovernor governor(backend,physical-reserve,reserve);
    uint64_t targetPlan=FlashForward::workspacePlannedBytes(capacity,rows,4)+FlashForward::expertCachePlannedBytes(weights)+FlashForward::floatDenseCachePlannedBytes(weights)+FlashForward::int8HeadPlannedBytes(weights);
    if(flag("SPLASH_FLASH_DENSE_CACHE",false))targetPlan+=FlashDenseCache::plannedBytes(weights,FlashDenseCache::defaultPrefixes(weights,true));
    if(flag("SPLASH_FLASH_BLOCKED_MOE",false))targetPlan+=flashMoEBlockedWorkspacePlannedBytes(rows,10);
    if(dense_w8a8_sep21::requiresCache(rows))targetPlan+=dense_w8a8_sep21::Cache::plannedBytes();
    const auto batchPlan=FlashBatchPrefill::workspacePlannedBytes(capacity,4,rows),statePlan=uint64_t(width)*FlashForward::requestStateBytes(capacity);
#ifdef SPLASH_BATCH_INTENDED_REFERENCE
    constexpr uint64_t extra=arenaBytes;
#else
    constexpr uint64_t extra=0;
#endif
    auto lease=governor.tryReserve(targetPlan+batchPlan+statePlan+extra+hostAllowance);require(bool(lease),"all target/batch/state/reference/host admission before constructors");
    {
      auto before=backend.memoryStats().allocatedBytes;FlashForward trunk(backend,weights,capacity,rows,4);targetDelta=backend.memoryStats().allocatedBytes-before;
      require(targetDelta==trunk.workspaceBytes()&&targetDelta<=targetPlan,"target actual category charge");
      before=backend.memoryStats().allocatedBytes;FlashBatchPrefill batch(backend,weights,trunk,capacity,4,rows);batchDelta=backend.memoryStats().allocatedBytes-before;
      require(batchDelta==batch.workspaceBytes()&&batchDelta<=batchPlan,"batch actual category charge");
#ifdef SPLASH_BATCH_INTENDED_REFERENCE
      batch_qsa_intended_reference::Session session(backend,arenaBytes);referenceDelta=session.actualCharge;
#endif
      before=backend.memoryStats().allocatedBytes;std::vector<FlashRequestState> states;states.reserve(width);for(uint32_t lane=0;lane<width;++lane)states.push_back(trunk.createState());
      stateDelta=backend.memoryStats().allocatedBytes-before;require(stateDelta<=statePlan,"native actual states admission");lease->commit();
      std::vector<FlashRequestState*> pointers;std::vector<uint32_t> input;std::vector<std::vector<metal::MetalBuffer>> initialViews;
      for(auto&s:states){require(s.logicalLength()==0&&trunk.ownsState(s)&&!s.poisoned(),"fresh whole cohort");pointers.push_back(&s);input.insert(input.end(),fixture.begin(),fixture.end());
        std::vector<metal::MetalBuffer> views;for(const auto&p:batch.inspectState(s))views.push_back(p.buffer);require(views.size()==134,"fresh physical134views");initialViews.push_back(std::move(views));}
      exactBytes=exactCampaign(batch,states,width,source,layout);
      Stream stream(argv[6],exporting,exactBytes);stream.text("batch-intended-arithmetic-physical-campaign-v1");stream.number(width);stream.number(capacity);stream.text(source);stream.text(layout);
#ifndef SPLASH_BATCH_INTENDED_REFERENCE
      const auto oldCalls=batch_prefill_twopass_sep22::counters().layers.load(),oldForwards=batch_prefill_twopass_sep22::counters().forwards.load();
#endif
      failure.phase="actual-prefill";auto submitted=backend.submissionCount();FlashBatchPrefillResult result;
#ifdef SPLASH_BATCH_INTENDED_REFERENCE
      {batch_qsa_intended_reference::Install install(session);result=batch.forwardBatch(pointers,input,rows,true);}
      calls=session.encodedLaneLayerCalls;require(session.completedForwards==1&&calls==uint64_t(12)*width,"reference all12QSA real lanes complete");
#else
      result=batch.forwardBatch(pointers,input,rows,true);calls=batch_prefill_twopass_sep22::counters().layers.load()-oldCalls;
      require(calls==uint64_t(12)*width&&batch_prefill_twopass_sep22::counters().forwards.load()-oldForwards==1,"candidate all12QSA real lanes complete");
#endif
      require(backend.submissionCount()-submitted==1&&backend.healthy()&&batch.canariesIntact(),"actual healthy synchronous native batch");
      require(result.lanes==width&&result.rows==rows&&result.capacity==capacity&&result.greedyRows==width&&!result.hiddenDeliveredToDestination&&result.logicalLengths==std::vector<uint64_t>(width,rows),"actual prefill result metadata");
      auto views=stateViews(batch,trunk,states,weights.descriptor(),rows,progress);outputViews(views,"prefill.",result.hiddenBF16,result.logitsBF16,result.greedyResultsU32,width,rows,progress);stream.frame("prefill",views);
      std::vector<uint32_t> next;const auto*g=static_cast<const FlashGreedyGPURowResult*>(result.greedyResultsU32.contents());for(uint32_t lane=0;lane<width;++lane)next.push_back(greedyGPUResultToken(g[lane],kVocabulary));result={};
      for(uint32_t step=0;step<3;++step){
        failure.phase="actual-future-AR"+std::to_string(step+1);
        std::vector<Owned> outputs;outputs.reserve(width);
        for(uint32_t lane=0;lane<width;++lane){const std::array<uint32_t,1>incoming{next[lane]};submitted=backend.submissionCount();auto r=trunk.forward(states[lane],incoming,false,true);
          require(backend.submissionCount()-submitted==1&&backend.healthy()&&r.logicalLength==rows+step+1&&r.capacity==capacity&&r.logitRows==1&&r.greedyRows==1,"genuine future AR completed metadata");
          std::vector<View> validated;outputViews(validated,"future.",r.hiddenBF16,r.logitsBF16,r.greedyResultsU32,1,1,progress);
          outputs.push_back({copied<uint16_t>(r.hiddenBF16,kHyper*2ULL),copied<uint16_t>(r.logitsBF16,kVocabulary*2ULL),copied<FlashGreedyGPURowResult>(r.greedyResultsU32,sizeof(FlashGreedyGPURowResult))});
          next[lane]=greedyGPUResultToken(outputs.back().greedy[0],kVocabulary);
        }
        views=stateViews(batch,trunk,states,weights.descriptor(),rows+step+1,progress);
        for(uint32_t lane=0;lane<width;++lane){const auto&o=outputs[lane];const auto prefix="lane"+std::to_string(lane)+".future.";
          views.push_back({prefix+"hidden",FlashDType::BF16,o.hidden.size()*2,o.hidden.size()*2,o.hidden.data()});
          views.push_back({prefix+"logits",FlashDType::BF16,o.logits.size()*2,o.logits.size()*2,o.logits.data()});
          views.push_back({prefix+"complete_greedy",FlashDType::U32,o.greedy.size()*sizeof(FlashGreedyGPURowResult),o.greedy.size()*sizeof(FlashGreedyGPURowResult),o.greedy.data()});}
        stream.frame("genuine-greedy-AR"+std::to_string(step+1),views);
      }
      for(uint32_t lane=0;lane<width;++lane){const auto now=batch.inspectState(states[lane]);for(size_t i=0;i<now.size();++i)require(now[i].buffer.sameView(initialViews[lane][i]),"local request storage ownership/views unchanged");}
      peak=backend.memoryStats().peakAllocatedBytes-mapped;require(peak<=targetPlan+batchPlan+statePlan+extra+hostAllowance&&batch.canariesIntact()&&backend.healthy(),"actual peak/guard/health within admission");
      digest=stream.finish();frameBytes=stream.count();frames=stream.frames();planes=stream.planes();
      require(frames==4&&planes==uint64_t(width)*4*134+3+uint64_t(width)*3*3,"complete physical pref+threeAR plane/checkpoint census");
    }
    require(backend.memoryStats().allocatedBytes==mapped,"all owned arena/state/target teardown to mapped model");
    const auto audit=governor.snapshot();require(!audit.deniedReservations&&!audit.reservedBytes&&audit.hostMeasurementValid&&audit.growthAllowed,"actual zero governor/healthy host teardown");
  }
  std::ostringstream report;report<<"{\"schema\":\"batch-twoPass-intended-full-state-arithmetic-v1\",\"pass\":true,\"role\":"<<json::quote(role)<<",\"GPU_executed\":true,\"intended_arithmetic_state_comparison_complete\":"<<(exporting?"false":"true")<<",\"numerical_alternative\":true,\"old_SG8_EVERYROW_equivalence\":false,\"old_extra_row_gate_failed_rows\":116,\"original22_or_Head_Worker_qualified\":false,\"capacity\":4096,\"real_lanes\":"<<width<<",\"prefill_rows\":2048,\"all12QSA_lane_layer_calls\":"<<calls<<",\"physical_state_planes\":134,\"checkpoints\":"<<frames<<",\"planes\":"<<planes<<",\"full_frame_bytes\":"<<frameBytes<<",\"whole_spill_preflight_bound\":"<<bound<<",\"frame_sha256\":"<<json::quote(digest)<<",\"all_full_hidden_logits_complete_greedy_and_inactive_state_bytes_exact\":"<<(exporting?"false":"true")<<",\"genuine_greedy_AR_contexts\":[2049,2050,2051],\"one_Model_Forward_Batch_process\":true,\"reference_arena_charge\":"<<referenceDelta<<",\"target_actual_charge\":"<<targetDelta<<",\"batch_actual_charge\":"<<batchDelta<<",\"state_actual_charge\":"<<stateDelta<<",\"measured_native_peak_growth\":"<<peak<<",\"final_governor_reserved_bytes\":0,\"owned_teardown_to_model\":true,\"backend_destroyed\":true,\"source\":"<<json::quote(source)<<",\"layout\":"<<json::quote(layout)<<",\"provenance\":"<<kMainAlternativeProvenance<<'}';
  publish(argv[7],report.str());std::cout<<report.str()<<'\n';return 0;
}
}
int main(int argc,char**argv){@autoreleasepool{
  try{
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){std::cout<<"{\"pass\":true,\"CPU_checks\":"<<cpu()<<",\"GPU_executed\":false,\"data_payload_reads\":0,\"B4_spill_bound\":"<<predicted(4)<<"}\n";return 0;}
    if(argc==2&&std::string_view(argv[1])=="--help"){std::cout<<"oracle --export|--compare LIB PACKAGE TOKENS2048 2|4 FRESH_OR_REFERENCE_FRAME FRESH_REPORT\nRoot only intended NEW arithmetic; not old SG8 EVERYROW equivalence; --cpu-self-test opens no model/device.\n";return 0;}
    return run(argc,argv);
  }catch(const std::exception&e){std::cerr<<"intended full batch math proof: "<<e.what()<<'\n';
    if(argc==8&&!fs::exists(argv[7]))try{publish(std::string(argv[7])+".failure.json","{\"pass\":false,\"intended_arithmetic_state_comparison_complete\":false,\"earlier_GPU_milestones_not_asserted\":true,\"reason\":"+json::quote(e.what())+",\"phase\":"+json::quote(failure.phase)+",\"checkpoint\":"+json::quote(failure.checkpoint)+",\"plane\":"+json::quote(failure.plane)+",\"mismatch_or_stream_offset\":"+std::to_string(failure.offset)+",\"completed_checkpoints\":"+std::to_string(failure.completedFrames)+'}');}catch(...){}
    return 1;
  }
}}
