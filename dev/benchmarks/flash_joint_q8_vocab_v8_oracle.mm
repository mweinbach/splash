// Root-only private GPU oracle. --help constructs no Metal device.
#define main splash_original_head_profile_unused_v8
#include "flash_trained_head_profile_v8_oracle.mm"
#undef main
#include "flash/FlashInt8Head.hpp"

namespace {
constexpr double kStrictL2=1e-4;
using States=std::vector<FlashMTPState>;
struct Snapshot {
  metal::CommandTiming timing;
  double seconds=0;
  std::vector<uint16_t> hidden,logits;
  std::vector<uint32_t> greedy,cpuGreedy;
  std::vector<uint64_t> lengths;
};
std::vector<uint16_t> copied(const metal::MetalBuffer &buffer,uint64_t words) {
  require(buffer&&buffer.contents()&&buffer.sizeBytes()>=words*2,"joint snapshot missing BF16 words");
  const auto *p=static_cast<const uint16_t *>(buffer.contents());return {p,p+words};
}
uint32_t fullGreedy(std::span<const uint16_t> words) {
  float maximum=-INFINITY;uint32_t best=0;
  for(uint32_t index=0;index<words.size();++index) {
    const float value=std::bit_cast<float>(uint32_t{words[index]}<<16);
    require(std::isfinite(value),"joint vocabulary is nonfinite");
    if(value>maximum){maximum=value;best=index;}
  }return best;
}
States clone(FlashMTPForward &head,const std::vector<Seed> &seeds) {
  States out;out.reserve(seeds.size());
  for(const auto &seed:seeds){out.push_back(head.createState());head.privateCopyStateV8(out.back(),seed.head);}
  return out;
}
void restore(FlashMTPForward &head,States &states,const std::vector<Seed> &seeds) {
  for(size_t lane=0;lane<seeds.size();++lane)head.privateCopyStateV8(states[lane],seeds[lane].head);
}
bool qsaExact(FlashMTPForward &head,const States &a,const States &b) {
  for(size_t lane=0;lane<a.size();++lane) {
    if(a[lane].logicalLength()!=b[lane].logicalLength()||a[lane].capacity()!=b[lane].capacity()||a[lane].poisoned()||b[lane].poisoned())return false;
    const auto aa=head.privateQSAStateBuffersV8(a[lane]),bb=head.privateQSAStateBuffersV8(b[lane]);
    require(aa.size()==5&&bb.size()==5,"joint QSA plane count differs");
    for(size_t plane=0;plane<aa.size();++plane) {
      require(aa[plane].contents()&&bb[plane].contents()&&aa[plane].sizeBytes()==bb[plane].sizeBytes(),"joint QSA plane extent differs");
      if(std::memcmp(aa[plane].contents(),bb[plane].contents(),aa[plane].sizeBytes()))return false;
    }
  }return true;
}
std::string qsaHashes(FlashMTPForward &head,const States &states) {
  std::ostringstream out;out<<'[';
  for(size_t lane=0;lane<states.size();++lane) {
    if(lane)out<<',';out<<"{\"lane\":"<<lane<<",\"length\":"<<states[lane].logicalLength()<<",\"poisoned\":"<<(states[lane].poisoned()?"true":"false")<<",\"full_plane_sha256\":[";
    const auto planes=head.privateQSAStateBuffersV8(states[lane]);
    for(size_t plane=0;plane<planes.size();++plane){if(plane)out<<',';out<<json::quote(digest(planes[plane].contents(),planes[plane].sizeBytes()));}
    out<<"]}";
  }out<<']';return out.str();
}
Snapshot body(FlashBatchMTPForward &joint,States &states,metal::MetalBuffer features,std::span<const uint32_t> next) {
  std::vector<FlashMTPState *> pointers;for(auto &state:states)pointers.push_back(&state);
  const std::array<uint32_t,4> counts{1,1,1,1};
  const auto start=Clock::now();const auto result=joint.forward(pointers,features,next,counts,FlashMTPLogits::Last);
  Snapshot out;out.seconds=std::chrono::duration<double>(Clock::now()-start).count();out.timing=result.timing;
  require(result.lanes==4&&result.logitRows==4&&result.greedyRows==4&&result.hiddenBF16.sizeBytes()==4*kHyper*2,"joint B4R1 result geometry differs");
  require(std::isfinite(out.timing.gpuSeconds)&&out.timing.gpuSeconds>0&&std::isfinite(out.seconds)&&out.seconds>0,"joint timing ABI invalid");
  out.hidden=copied(result.hiddenBF16,4*kHyper);out.logits=copied(result.logitsBF16,4*kVocabulary);out.lengths=result.logicalLengths;
  for(uint32_t lane=0;lane<4;++lane) {
    require(out.lengths[lane]==states[lane].logicalLength()&&!states[lane].poisoned(),"joint state publication differs");
    out.greedy.push_back(token(result.greedyResultsU32,result.logitsBF16,lane));
    out.cpuGreedy.push_back(fullGreedy(std::span<const uint16_t>(out.logits).subspan(uint64_t{lane}*kVocabulary,kVocabulary)));
  }return out;
}
struct Metrics { bool finite=true;uint64_t mismatches=0;uint32_t maxULP=0;double l2=0,maxAbsolute=0; };
uint32_t ordered(uint16_t value){return value&0x8000?uint32_t{0x8000}-(value&0x7fff):uint32_t{0x8000}+value;}
Metrics compareWords(std::span<const uint16_t> a,std::span<const uint16_t> b) {
  require(a.size()==b.size(),"joint numerical comparison extent differs");
  Metrics out;long double error=0,norm=0;
  for(size_t index=0;index<a.size();++index) {
    const float aa=std::bit_cast<float>(uint32_t{a[index]}<<16),bb=std::bit_cast<float>(uint32_t{b[index]}<<16);
    if(!std::isfinite(aa)||!std::isfinite(bb)){out.finite=false;continue;}
    const long double delta=static_cast<long double>(aa)-bb;error+=delta*delta;norm+=static_cast<long double>(aa)*aa;
    if(a[index]!=b[index])++out.mismatches;
    const auto x=ordered(a[index]),y=ordered(b[index]);out.maxULP=std::max(out.maxULP,x>y?x-y:y-x);out.maxAbsolute=std::max(out.maxAbsolute,double(std::abs(delta)));
  }out.l2=norm>0?double(std::sqrt(error/norm)):(error==0?0:INFINITY);return out;
}
void writeMetrics(std::ostream &out,const Metrics &m) {
  out<<"{\"finite\":"<<(m.finite?"true":"false")<<",\"bf16_word_mismatches\":"<<m.mismatches<<",\"max_bf16_ulp\":"<<m.maxULP<<",\"max_absolute_error\":";profiling::writeNumber(out,m.maxAbsolute);
  out<<",\"relative_l2\":";profiling::writeNumber(out,m.l2);out<<",\"strict_relative_l2_bound\":"<<kStrictL2<<",\"strict_bound_pass\":"<<(m.finite&&m.l2<=kStrictL2?"true":"false")<<'}';
}
void writeTiming(std::ostream &out,const Snapshot &s) {
  out<<"{\"normal_call_seconds\":"<<s.seconds<<",\"command_gpu_seconds\":"<<s.timing.gpuSeconds<<",\"command_wall_seconds\":"<<s.timing.wallSeconds<<",\"host_command_subphases\":";writeHost(out,s.timing.host);out<<'}';
}
void privateUsage() {
  std::cout<<"usage: flash-joint-q8-vocab-v8-oracle METALLIB PACKAGE FIXTURE_DIR REPORT_JSON\n"
    "Root-only GPU. --help initializes no device. Fresh 200-byte CommandTiming host ABI.\n"
    "B4xR1 contexts128/2048; FLASH_JOINT_Q8_CONTEXT=128|2048 optionally selects one context.\n"
    "One startup warmup per route per context; six paired normal AB,BA,AB,BA,AB,BA timings.\n"
    "Exact teacher-primed QSA restore before every iteration. Copies/hashes/state/CPU greedy are outside timed head calls.\n"
    "Strict each-lane full-vocabulary L2<=1e-4, exact greedy IDs/full hidden/full QSA; numerical rejection report retained, exit2.\n"
    "Continuation/truncate-overwrite are deferred in this bounded initial screen. No service claim.\n";
}
} // namespace

int main(int argc,char **argv) {
 @autoreleasepool {try {
  static_assert(sizeof(metal::CommandTiming)==200,"fresh private ABI200 host objects required");
  if(argc==2&&std::string_view(argv[1])=="--help"){privateUsage();return 0;}
  require(argc==5,"usage: flash-joint-q8-vocab-v8-oracle METALLIB PACKAGE FIXTURE_DIR REPORT_JSON");
  std::vector<uint32_t> contexts{128,2048};
  if(std::getenv("FLASH_JOINT_Q8_CONTEXT")){const auto context=integer("FLASH_JOINT_Q8_CONTEXT",0,128,2048);require(context==128||context==2048,"context must be128 or2048");contexts={context};}
  require(!std::filesystem::exists(argv[4]),"choose fresh joint report");
  metal::MetalBackend backend(argv[1]);const auto weights=FlashWeights::load(backend,argv[2]);
  const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);
  engine::MemoryGovernor governor(backend,physical-reserve,reserve);
  uint64_t planned=FlashForward::workspacePlannedBytes(kCapacity,2048,16)+FlashForward::expertCachePlannedBytes(weights)+FlashForward::floatDenseCachePlannedBytes(weights)+FlashForward::int8HeadPlannedBytes(weights)+FlashMTPForward::workspacePlannedBytes(kCapacity,128)+2*FlashBatchMTPForward::workspacePlannedBytes(kCapacity,4,4,true)+FlashInt8Head::plannedBytes(weights)+4*(FlashForward::requestStateBytes(kCapacity)+3*FlashMTPForward::requestStateBytes(kCapacity))+(512ULL<<20);
  if(enabled("SPLASH_FLASH_DENSE_CACHE"))planned+=FlashDenseCache::plannedBytes(weights,FlashDenseCache::defaultPrefixes(weights,true))+FlashMTPForward::denseCachePlannedBytes(weights);
  if(enabled("SPLASH_FLASH_BLOCKED_MOE"))planned+=flashMoEBlockedWorkspacePlannedBytes(2048,10);
  auto reservation=governor.tryReserve(planned);require(bool(reservation),"governor denied private joint arenas");
  FlashForward target(backend,weights,kCapacity,2048,16);FlashMTPForward head(backend,weights,kCapacity,128);
  require(setenv("SPLASH_PRIVATE_JOINT_Q8_VOCAB","0",1)==0,"cannot choose original route");
  FlashBatchMTPForward original(head,4,4,target.cachedVocabulary());
  require(setenv("SPLASH_PRIVATE_JOINT_Q8_VOCAB","1",1)==0,"cannot choose candidate route");
  FlashBatchMTPForward candidate(head,4,4,target.cachedVocabulary());
  require(unsetenv("SPLASH_PRIVATE_JOINT_Q8_VOCAB")==0,"cannot clear route switch");
  require(!original.privateQ8VocabularyEnabledV8()&&candidate.privateQ8VocabularyEnabledV8(),"route switch was not captured per instance");
  require(std::string_view(candidate.privateVocabularyRouteV8())=="head-original-q8-byte-view-zero-copy-v1","candidate is not original zero-copy Q8");
  metal::ResidencyLease residency;if(enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
    auto operands=target.cachedOperandsOnly();const auto extra=head.cachedOperandsOnly();operands.insert(operands.end(),extra.begin(),extra.end());
    if(!operands.empty())residency=backend.requestWeightResidency(operands,"private joint original saved source/head operands");
  }reservation->commit();
  std::vector<std::string> records;bool accepted=true,allHidden=true,allState=true,allGreedy=true,allVocabulary=true;
  for(uint32_t context:contexts) {
    std::vector<Seed> seeds;seeds.reserve(4);
    auto features=backend.allocateBuffer(4*kHyper*2,metal::BufferStorage::Shared,"private-joint-q8-four-real-target-features");
    std::vector<uint32_t> next;
    for(uint32_t lane=0;lane<4;++lane) {
      const auto name="ctx"+std::to_string(context)+"-width4-lane"+std::to_string(lane)+".tokens.json";
      const auto prompt=loadTokens((std::filesystem::path(argv[3])/name).string());require(prompt.size()==context,"fixture context differs");
      seeds.push_back(prepare(backend,target,head,prompt));next.push_back(seeds.back().anchor);
      std::memcpy(static_cast<uint8_t *>(features.contents())+uint64_t{lane}*kHyper*2,static_cast<const uint8_t *>(seeds.back().targetFeatures.contents())+seeds.back().targetFeatures.sizeBytes()-kHyper*2,kHyper*2);
    }
    States a=clone(head,seeds),b=clone(head,seeds);
    require(qsaExact(head,a,b),"initial cloned state differs");
    (void)body(original,a,features,next);(void)body(candidate,b,features,next);
    for(uint32_t pair=0;pair<6;++pair) {
      restore(head,a,seeds);restore(head,b,seeds);require(qsaExact(head,a,b),"paired restored seed differs");
      Snapshot aa,bb;
      if(pair%2==0){aa=body(original,a,features,next);bb=body(candidate,b,features,next);}else{bb=body(candidate,b,features,next);aa=body(original,a,features,next);}
      const auto hidden=compareWords(aa.hidden,bb.hidden);
      const bool hiddenExact=aa.hidden==bb.hidden&&hidden.finite,stateExact=qsaExact(head,a,b),greedyExact=aa.greedy==bb.greedy&&aa.greedy==aa.cpuGreedy&&bb.greedy==bb.cpuGreedy;
      bool vocabularyPass=true;std::vector<Metrics> laneMetrics;
      for(uint32_t lane=0;lane<4;++lane){laneMetrics.push_back(compareWords(std::span<const uint16_t>(aa.logits).subspan(uint64_t{lane}*kVocabulary,kVocabulary),std::span<const uint16_t>(bb.logits).subspan(uint64_t{lane}*kVocabulary,kVocabulary)));vocabularyPass=vocabularyPass&&laneMetrics.back().finite&&laneMetrics.back().l2<=kStrictL2;}
      const bool pass=hiddenExact&&stateExact&&greedyExact&&vocabularyPass&&aa.lengths==bb.lengths;
      accepted=accepted&&pass;allHidden=allHidden&&hiddenExact;allState=allState&&stateExact;allGreedy=allGreedy&&greedyExact;allVocabulary=allVocabulary&&vocabularyPass;
      std::ostringstream record;record<<std::setprecision(17);
      record<<"{\"context\":"<<context<<",\"pair\":"<<pair<<",\"order\":"<<json::quote(pair%2==0?"AB":"BA")<<",\"accepted\":"<<(pass?"true":"false")<<",\"full_premixer_bf16_exact\":"<<(hiddenExact?"true":"false")<<",\"premixer_finite\":"<<(hidden.finite?"true":"false")
        <<",\"original_premixer_sha256\":"<<json::quote(digest(aa.hidden.data(),aa.hidden.size()*2))<<",\"candidate_premixer_sha256\":"<<json::quote(digest(bb.hidden.data(),bb.hidden.size()*2))<<",\"qsa_full_buffers_exact\":"<<(stateExact?"true":"false")<<",\"exact_greedy_all_lanes\":"<<(greedyExact?"true":"false")<<",\"each_lane_strict_vocabulary_bound_pass\":"<<(vocabularyPass?"true":"false")<<",\"batch_vocabulary\":";
      writeMetrics(record,compareWords(aa.logits,bb.logits));record<<",\"lanes\":[";
      for(uint32_t lane=0;lane<4;++lane){if(lane)record<<',';record<<"{\"lane\":"<<lane<<",\"original_greedy\":"<<aa.greedy[lane]<<",\"candidate_greedy\":"<<bb.greedy[lane]<<",\"original_cpu_greedy\":"<<aa.cpuGreedy[lane]<<",\"candidate_cpu_greedy\":"<<bb.cpuGreedy[lane]<<",\"original_length\":"<<aa.lengths[lane]<<",\"candidate_length\":"<<bb.lengths[lane]<<",\"vocabulary\":";writeMetrics(record,laneMetrics[lane]);record<<'}';}
      record<<"],\"original_timing\":";writeTiming(record,aa);record<<",\"candidate_timing\":";writeTiming(record,bb);
      if(pair==0)record<<",\"original_qsa_full_hashes\":"<<qsaHashes(head,a)<<",\"candidate_qsa_full_hashes\":"<<qsaHashes(head,b);
      record<<'}';records.push_back(record.str());
    }
  }
  std::ofstream report(argv[4]);require(bool(report),"cannot write joint report");report<<std::setprecision(17);
  report<<"{\"schema\":\"splash-private-joint-q8-vocabulary-initial-screen-v8\",\"execution_complete\":true,\"gpu_executed\":true,\"command_timing_abi_bytes\":"<<sizeof(metal::CommandTiming)
    <<",\"accepted\":"<<(accepted?"true":"false")<<",\"numerical_rejection\":"<<(!allVocabulary?"true":"false")<<",\"status\":"<<json::quote(accepted?"accepted_initial_screen_only":"rejected")<<",\"strict_each_lane_relative_l2_bound\":"<<kStrictL2<<",\"strict_bound_relaxed\":false,\"full_premixer_bf16_exact\":"<<(allHidden?"true":"false")<<",\"qsa_full_buffers_exact\":"<<(allState?"true":"false")<<",\"exact_greedy_all_lanes\":"<<(allGreedy?"true":"false")<<",\"each_lane_strict_vocabulary_bound_pass\":"<<(allVocabulary?"true":"false")
    <<",\"warmup_per_route_per_context\":1,\"normal_pairs_per_context\":6,\"pair_order\":\"AB,BA,AB,BA,AB,BA; repeated ABBA\",\"exact_seed_restore_before_every_iteration\":true,\"output_copies_and_hashes_outside_timing\":true,\"timing_scope\":\"one joint trained-head API and command; no target/restore/inspection copies in measured API; no HTTP/service claim\""
    <<",\"original_vocabulary_route\":"<<json::quote(original.privateVocabularyRouteV8())<<",\"candidate_code_storage\":"<<json::quote(candidate.privateVocabularyRouteV8())<<",\"candidate_identity_sha256\":"<<json::quote(candidate.privateVocabularyIdentityV8())<<",\"candidate_operand_format\":"<<json::quote(kFlashInt8HeadOperandFormat)<<",\"candidate_projection_semantics\":"<<json::quote(kFlashInt8HeadSemantics)
    <<",\"numerical_alternative\":\"original cached vocabulary rounds dequantized coefficients to BF16; candidate uses unchanged original unsigned Q8 codes/BF16 scale/bias/group-factored F32 MPP/BF16 output\",\"candidate_scope\":\"private Last real logitRows2..4 branch only\",\"production_modified\":false,\"future_continuation_qualification\":\"deferred\",\"truncate_overwrite_qualification\":\"deferred\""
    <<",\"source_identity_sha256\":"<<json::quote(weights.sourceIdentity())<<",\"manifest_fingerprint_sha256\":"<<json::quote(weights.manifestFingerprint())<<",\"metallib_sha256\":"<<json::quote(hexadecimal(backend.metallibSha256()))<<",\"fixture_provenance_sha256\":"<<json::quote(fileDigest((std::filesystem::path(argv[3])/"fixture-provenance.json").string()))<<",\"measurements\":[";
  for(size_t index=0;index<records.size();++index){if(index)report<<',';report<<records[index];}report<<"]}\n";require(bool(report),"joint report write failed");
  std::cout<<"Private joint Q8 screen complete; accepted="<<(accepted?"true":"false")<<" report="<<argv[4]<<'\n';return accepted?0:2;
 }catch(const std::exception &error){std::cerr<<"Private joint Q8 oracle failed: "<<error.what()<<'\n';return 1;}}
}
