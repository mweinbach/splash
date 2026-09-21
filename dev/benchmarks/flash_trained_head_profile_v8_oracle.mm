// Private diagnostic; --help never initializes Metal. Root alone runs GPU.
#include "flash/FlashForward.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include "flash/FlashMTP.hpp"
#include "flash/FlashBatchMTPForward.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "metal/ProfilingJson.hpp"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using namespace splash;
using namespace splash::flash;
using Clock=std::chrono::steady_clock;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
uint32_t integer(const char *name, uint32_t fallback, uint32_t minimum, uint32_t maximum) {
  const char *raw = std::getenv(name); if (!raw) return fallback;
  uint64_t value = 0; require(*raw != 0, "empty attribution integer setting");
  for (const char *p = raw; *p; ++p) {
    require(*p >= '0' && *p <= '9', "attribution integer setting is not decimal");
    value = value * 10 + unsigned(*p - '0'); require(value <= maximum, "attribution integer setting exceeds limit");
  }
  require(value >= minimum, "attribution integer setting is below limit"); return uint32_t(value);
}
bool enabled(const char *name) {
  const char *raw = std::getenv(name); if (!raw) return false;
  require(std::string_view(raw) == "0" || std::string_view(raw) == "1", "attribution switch must be 0 or 1");
  return std::string_view(raw) == "1";
}
std::vector<uint32_t> loadTokens(const std::string &path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data != nil, "cannot read attribution tokens");
  NSError *error = nil; id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error == nil && [object isKindOfClass:[NSArray class]], "attribution tokens must be a JSON array");
  std::vector<uint32_t> result;
  for (id entry in static_cast<NSArray *>(object)) {
    require([entry isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)entry) != CFBooleanGetTypeID(),
        "attribution token must be an integer, not a boolean");
    const double value = static_cast<NSNumber *>(entry).doubleValue;
    require(std::isfinite(value) && value >= 0 && value < 248320 && std::floor(value) == value, "invalid attribution token");
    result.push_back(uint32_t(value));
  }
  require(!result.empty(), "attribution prompt is empty"); return result;
}
std::string digest(const void *bytes, uint64_t count) {
  CC_SHA256_CTX context{}; CC_SHA256_Init(&context);
  auto *next = static_cast<const uint8_t *>(bytes);
  while (count) { const CC_LONG part = CC_LONG(std::min<uint64_t>(count, UINT32_MAX));
    CC_SHA256_Update(&context, next, part); next += part; count -= part; }
  std::array<uint8_t, 32> result{}; CC_SHA256_Final(result.data(), &context);
  static constexpr char hex[] = "0123456789abcdef"; std::string output;
  for (uint8_t byte : result) { output += hex[byte >> 4]; output += hex[byte & 15]; } return output;
}
std::string hexadecimal(std::span<const uint8_t> bytes) {
  static constexpr char hex[] = "0123456789abcdef"; std::string output;
  for (uint8_t byte : bytes) { output += hex[byte >> 4]; output += hex[byte & 15]; }
  return output;
}
std::string fileDigest(const std::string &path) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data != nil, "cannot read attribution provenance"); return digest(data.bytes, data.length);
}
void writeHost(std::ostream &out, const metal::CommandHostTiming &h) {
  out << "{\"timed_commands\":" << h.timedCommands << ",\"commit_samples\":" << h.commitSamples
      << ",\"scheduled_callback_samples\":" << h.scheduledCallbackSamples
      << ",\"completed_callback_samples\":" << h.completedCallbackSamples
      << ",\"pre_commit_memory_samples\":" << h.preCommitMemorySamples
      << ",\"post_commit_memory_samples\":" << h.postCommitMemorySamples
      << ",\"scheduled_memory_samples\":" << h.scheduledMemorySamples
      << ",\"completed_memory_samples\":" << h.completedMemorySamples
      << ",\"ticket_wait_calls\":" << h.ticketWaitCalls;
  const auto field = [&](const char *name, double value) { out << ',' << json::quote(name) << ':'; profiling::writeNumber(out, value); };
  field("preparation_seconds", h.preparationSeconds); field("encoding_seconds", h.encodingSeconds);
  field("encoding_end_to_commit_begin_seconds", h.beforeCommitSeconds);
  field("sparse_dependency_wait_seconds", h.dependencyWaitSeconds); field("commit_seconds", h.commitSeconds);
  field("commit_to_scheduled_callback_seconds", h.commitToScheduledCallbackSeconds);
  field("commit_to_completed_callback_seconds", h.commitToCompletedCallbackSeconds);
  field("completed_callback_to_wall_end_seconds", h.completionCallbackBeforeWallEndSeconds);
  field("submission_return_latency_seconds", h.submissionReturnSeconds);
  field("ticket_blocking_wait_seconds", h.ticketBlockingWaitSeconds);
  field("pre_commit_memory_query_seconds", h.preCommitMemorySampleSeconds);
  field("post_commit_memory_query_seconds", h.postCommitMemorySampleSeconds);
  field("scheduled_memory_query_seconds", h.scheduledMemorySampleSeconds);
  field("completed_memory_query_seconds", h.completedMemorySampleSeconds);
  out << ",\"intervals_can_overlap\":true,\"scheduled_latency_scope\":\"commit begin to callback arrival, not actual GPU scheduling\"}";
}

struct Family { uint64_t dispatches=0, timed=0; double seconds=0; };
struct Classifier {
  uint32_t logitRows=1,physicalRows=1;
  uint32_t fc=0;
  bool insideMoE=false, shared=false;
  uint64_t routes=0;
  static bool projection(std::string_view name) {
    return name.starts_with("flash_affine") || name.starts_with("flash_dense") ||
           name.starts_with("flash_float_dense") || name.starts_with("flash_int8_head");
  }
  std::string family(std::string_view name,const metal::CommandDispatchTimestamp &metadata) {
    if(name=="flash_affine_embedding")return "embedding";
    if(name.ends_with("_pad"))return "padding";
    if(name.starts_with("flash_greedy"))return "greedy";
    if(name.starts_with("flash_int8_head"))return "vocabulary";
    if(projection(name)) {
      const uint32_t outputIndex=name.starts_with("flash_affine")?5:2;
      for(const auto &binding:metadata.bindings)
        if(!binding.inlineBytes&&binding.index==outputIndex&&
           binding.sizeBytes==uint64_t{logitRows}*248320*2)return "vocabulary";
      if(fc<2)return fc++==0?"fc_embedding":"fc_hidden";
    }
    if(name=="flash_moe_route") {insideMoE=true;shared=false;++routes;return "moe";}
    if(name=="flash_moe_combine") {insideMoE=false;shared=false;return "moe";}
    if(insideMoE&&projection(name))shared=true;
    if(shared&&(projection(name)||name=="flash_moe_silu_multiply"))return "shared_expert";
    if(projection(name)) {
      std::map<uint32_t,uint64_t> sizes;
      for(const auto &binding:metadata.bindings)if(!binding.inlineBytes)sizes[binding.index]=binding.sizeBytes;
      const uint32_t outputIndex=name.starts_with("flash_affine")?5:2;
      const auto source=sizes[1],params=sizes[2],output=sizes[outputIndex];
      if(name.starts_with("flash_affine")) {
        if(source==23592960&&output==uint64_t{physicalRows}*12288*2)return "attention_q_projection";
        if(source==983040&&output==uint64_t{physicalRows}*512*2&&params==40960)return "attention_k_projection";
        if(source==983040&&output==uint64_t{physicalRows}*512*2&&params==20480)return "attention_v_projection";
        if(source==1228800&&output==uint64_t{physicalRows}*640*2)return "attention_index_projection";
        if(source==9830400&&output==uint64_t{physicalRows}*2560*2)return "attention_o_projection";
      }
      if(name=="flash_dense_bf16_project"&&source==2621440&&output==uint64_t{physicalRows}*512*2)return "moe_router_projection";
    }
    if(name.starts_with("flash_qsa"))return "qsa";
    if(name.starts_with("flash_hc")||name.starts_with("flash_forward_hc"))return "hc";
    if(name.starts_with("flash_moe")||name.starts_with("flash_expert")||name.starts_with("flash_int8_expert"))return "moe";
    if(projection(name))return "dense";
    if(name=="flash_forward_copy_words")return "copies";
    return "other";
  }
};
void writeFamilies(std::ostream &out,const metal::CommandDispatchProfile &profile,uint32_t logitRows,uint32_t physicalRows) {
  Classifier classifier;classifier.logitRows=logitRows;classifier.physicalRows=physicalRows;
  std::map<std::string,Family> families;double timedSeconds=0;uint64_t timed=0;
  for(const auto &dispatch:profile.dispatches) {
    auto &family=families[classifier.family(dispatch.pipelineName,dispatch)];++family.dispatches;
    if(dispatch.timestampsValid){++family.timed;++timed;family.seconds+=dispatch.gpuSeconds;timedSeconds+=dispatch.gpuSeconds;}
  }
  out<<"{\"classification_scope\":\"head pipeline families; first two non-embedding projections are fc_embedding/fc_hidden; vocabulary identified by exact output extent; shared expert interval ends at canonical combine; attention/router projection roles identified by checked original source weight and output extents\""
     <<",\"classification_complete\":"<<(!profile.dispatchMetadataTruncated&&profile.dispatches.size()==profile.dispatchCount&&classifier.fc==2&&classifier.routes==1?"true":"false")
     <<",\"model_layer_routes\":"<<classifier.routes<<",\"timed_dispatches\":"<<timed
     <<",\"full_command_gpu_seconds\":"<<profile.timing.gpuSeconds<<",\"sum_timed_dispatch_gpu_seconds\":"<<timedSeconds
     <<",\"command_minus_timed_dispatch_seconds\":"<<profile.timing.gpuSeconds-timedSeconds<<",\"families\":{";
  bool comma=false;for(const auto &[name,family]:families){if(comma)out<<',';comma=true;
    out<<json::quote(name)<<":{\"dispatches\":"<<family.dispatches<<",\"timed_dispatches\":"<<family.timed
       <<",\"gpu_seconds\":"<<family.seconds<<",\"fraction_of_full_command_gpu\":";
    if(profile.timing.gpuSeconds>0&&family.timed)out<<family.seconds/profile.timing.gpuSeconds;else out<<"null";
    out<<'}';
  }out<<"}}";
}

constexpr uint32_t kHyper=10240,kVocabulary=248320,kCapacity=8192;
void usage() {
  std::cout<<"usage: flash-trained-head-profile-v8-oracle METALLIB PACKAGE FIXTURE_DIR REPORT_JSON\n"
    "Root-only GPU. --help creates no device. Requires fresh ABI-200 host objects.\n"
    "FLASH_HEAD_PROFILE_MODE=normal|command|stage|dispatch; CONTEXT=128|2048; PHASE=proposal|committed_fold|joint_proposal; ROWS=1|4|8.\n"
    "FLASH_HEAD_PROFILE_WARMUP=1, REPEATS=1. Unset selectors run all supported cases.\n"
    "R1 uses a true last target premixer feature and anchor after teacher-priming.\n"
    "R4/R8 folds use true greedy target continuation pairs, Last vocabulary output.\n"
    "Joint B4xR1 uses four real benchmark lanes and the production cached vocabulary.\n"
    "No diagnostic route copies or state hashes in the measured command. BF16 output hashes read only after completion.\n"
    "Stage changes encoder boundaries; dispatch adds timestamp barriers; timings are attribution diagnostics.\n";
}
bool stop(uint32_t token){return token==248044||token==248046;}
uint32_t token(const metal::MetalBuffer &greedy,const metal::MetalBuffer &logits,uint32_t row=0) {
  if(greedy&&greedy.contents()) {
    require(greedy.sizeBytes()>=uint64_t{row+1}*sizeof(FlashGreedyGPURowResult),"missing greedy result row");
    return greedyGPUResultToken(static_cast<const FlashGreedyGPURowResult *>(greedy.contents())[row],kVocabulary);
  }
  require(logits&&logits.contents()&&logits.sizeBytes()>=uint64_t{row+1}*kVocabulary*2,"missing BF16 vocabulary row");
  const auto *values=static_cast<const uint16_t *>(logits.contents())+uint64_t{row}*kVocabulary;
  float maximum=-INFINITY;uint32_t best=0;
  for(uint32_t index=0;index<kVocabulary;++index) {
    const float value=std::bit_cast<float>(uint32_t{values[index]}<<16);require(std::isfinite(value),"nonfinite vocabulary output");
    if(value>maximum){maximum=value;best=index;}
  }return best;
}
std::string bfDigest(const metal::MetalBuffer &buffer,uint64_t words) {
  require(buffer&&buffer.contents()&&buffer.sizeBytes()>=words*2,"missing BF16 digest storage");
  const auto *values=static_cast<const uint16_t *>(buffer.contents());
  for(uint64_t index=0;index<words;++index)require(std::isfinite(std::bit_cast<float>(uint32_t{values[index]}<<16)),"nonfinite hidden/logit digest input");
  return digest(buffer.contents(),words*2);
}
struct Seed {
  FlashRequestState target;
  FlashMTPState head;
  metal::MetalBuffer targetFeatures;
  uint32_t anchor=0;
  std::string promptSha,lastFeatureSha;
};
Seed prepare(metal::MetalBackend &backend,FlashForward &target,FlashMTPForward &head,std::span<const uint32_t> prompt) {
  Seed out;out.target=target.createState();out.head=head.createState();
  const auto result=target.forward(out.target,prompt,false,true);
  require(result.hiddenBF16&&result.hiddenBF16.sizeBytes()>=uint64_t{prompt.size()}*kHyper*2,"target premixer extent differs");
  out.targetFeatures=backend.allocateBuffer(uint64_t{prompt.size()}*kHyper*2,metal::BufferStorage::Shared,"private-trained-head-owned-real-target-features");
  require(result.hiddenBF16.contents()&&out.targetFeatures.contents(),"target features are not shared readable storage");
  const uint64_t featureBytes=uint64_t{prompt.size()}*kHyper*2;
  std::memcpy(out.targetFeatures.contents(),result.hiddenBF16.contents(),featureBytes);
  out.targetFeatures=backend.view(out.targetFeatures,0,featureBytes);
  out.anchor=token(result.greedyResultsU32,result.logitsBF16);
  require(!stop(out.anchor),"target anchor is EOS; refusing invented draft body");
  out.promptSha=digest(prompt.data(),prompt.size()*sizeof(uint32_t));
  for(uint32_t begin=0;begin+1<prompt.size();begin+=128) {
    const uint32_t count=std::min<uint32_t>(128,uint32_t(prompt.size())-begin-1);
    (void)head.forward(out.head,backend.view(out.targetFeatures,uint64_t{begin}*kHyper*2,uint64_t{count}*kHyper*2),prompt.subspan(begin+1,count),FlashMTPLogits::None);
  }
  require(out.head.logicalLength()==prompt.size()-1&&out.target.logicalLength()==prompt.size(),"true teacher-prime offset differs");
  const auto last=backend.view(out.targetFeatures,uint64_t{prompt.size()-1}*kHyper*2,kHyper*2);
  out.lastFeatureSha=bfDigest(last,kHyper);return out;
}
struct Fold {
  metal::MetalBuffer features;
  std::vector<uint32_t> nextTokens;
  uint32_t pending=0;
};
Fold trueTargetFold(metal::MetalBackend &backend,FlashForward &target,Seed &seed,uint32_t rows) {
  Fold out;out.features=backend.allocateBuffer(uint64_t{rows}*kHyper*2,metal::BufferStorage::Shared,"private-trained-head-committed-real-target-pairs");
  const auto last=backend.view(seed.targetFeatures,seed.targetFeatures.sizeBytes()-kHyper*2,kHyper*2);
  std::memcpy(out.features.contents(),last.contents(),kHyper*2);uint32_t next=seed.anchor;
  for(uint32_t row=0;row<rows;++row) {
    require(!stop(next),"true target continuation reached EOS before requested committed fold");
    out.nextTokens.push_back(next);
    const auto result=target.forward(seed.target,std::span<const uint32_t>(&next,1),false,true);
    next=token(result.greedyResultsU32,result.logitsBF16);
    if(row+1<rows)std::memcpy(static_cast<uint8_t *>(out.features.contents())+uint64_t{row+1}*kHyper*2,result.hiddenBF16.contents(),kHyper*2);
  }
  out.pending=next;return out;
}
struct Capture {
  metal::CommandTiming timing;
  double callSeconds=0;
  std::vector<metal::CommandDispatchProfile> profiles;
  std::string hiddenSha,logitsSha;
  std::vector<uint32_t> predictions;
};
void profileOn(metal::MetalBackend &backend,metal::CommandDispatchProfilingMode mode) {
  require(backend.takeCommandDispatchProfiles().empty(),"stale diagnostic profiles remained");backend.setCommandDispatchProfiling(mode);
}
void finishProfile(metal::MetalBackend &backend,Capture &capture,metal::CommandDispatchProfilingMode mode) {
  capture.profiles=backend.takeCommandDispatchProfiles();backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
  require(capture.profiles.size()==(mode==metal::CommandDispatchProfilingMode::Off?0u:1u),"missing or extra original head profile");
  require(std::isfinite(capture.timing.gpuSeconds)&&capture.timing.gpuSeconds>0&&std::isfinite(capture.callSeconds)&&capture.callSeconds>0,"invalid ABI or command timing");
}
Capture runSingle(metal::MetalBackend &backend,FlashMTPForward &head,Seed &seed,metal::MetalBuffer features,
    std::span<const uint32_t> next,metal::CommandDispatchProfilingMode mode) {
  profileOn(backend,mode);const auto start=Clock::now();const auto result=head.forward(seed.head,features,next,FlashMTPLogits::Last);
  Capture out;out.callSeconds=std::chrono::duration<double>(Clock::now()-start).count();out.timing=result.timing;
  finishProfile(backend,out,mode);require(result.hiddenRows==next.size()&&result.logitRows==1&&result.greedyRows==1,"scalar original head geometry differs");
  out.hiddenSha=bfDigest(result.hiddenBF16,uint64_t{result.hiddenRows}*kHyper);
  out.logitsSha=bfDigest(result.logitsBF16,kVocabulary);out.predictions.push_back(token(result.greedyResultsU32,result.logitsBF16));return out;
}
Capture runJoint(metal::MetalBackend &backend,FlashBatchMTPForward &joint,std::vector<Seed> &seed,metal::CommandDispatchProfilingMode mode) {
  std::vector<FlashMTPState *> states;std::vector<uint32_t> next;const std::array<uint32_t,4> counts{1,1,1,1};
  auto featuresAllocation=backend.allocateBuffer(4*kHyper*2,metal::BufferStorage::Shared,"private-trained-head-joint-four-owned-features");
  const auto features=backend.view(featuresAllocation,0,4*kHyper*2);
  for(uint32_t lane=0;lane<4;++lane) {
    states.push_back(&seed[lane].head);next.push_back(seed[lane].anchor);
    std::memcpy(static_cast<uint8_t *>(features.contents())+uint64_t{lane}*kHyper*2,
      static_cast<const uint8_t *>(seed[lane].targetFeatures.contents())+seed[lane].targetFeatures.sizeBytes()-kHyper*2,kHyper*2);
  }
  profileOn(backend,mode);const auto start=Clock::now();const auto result=joint.forward(states,features,next,counts,FlashMTPLogits::Last);
  Capture out;out.callSeconds=std::chrono::duration<double>(Clock::now()-start).count();out.timing=result.timing;
  finishProfile(backend,out,mode);require(result.lanes==4&&result.logitRows==4&&result.greedyRows==4,"joint original head geometry differs");
  out.hiddenSha=bfDigest(result.hiddenBF16,4*kHyper);out.logitsSha=bfDigest(result.logitsBF16,4*kVocabulary);
  for(uint32_t lane=0;lane<4;++lane){require(result.logicalLengths[lane]==seed[lane].head.logicalLength(),"joint head publication differs");out.predictions.push_back(token(result.greedyResultsU32,result.logitsBF16,lane));}return out;
}
} // namespace
int main(int argc,char **argv) {
 @autoreleasepool {try {
  static_assert(sizeof(metal::CommandTiming)==200,"fresh host ABI200 objects required");
  if(argc==2&&std::string_view(argv[1])=="--help"){usage();return 0;}
  require(argc==5,"usage: flash-trained-head-profile-v8-oracle METALLIB PACKAGE FIXTURE_DIR REPORT_JSON");
  const std::string mode=std::getenv("FLASH_HEAD_PROFILE_MODE")?std::getenv("FLASH_HEAD_PROFILE_MODE"):"normal";
  auto sampling=metal::CommandDispatchProfilingMode::Off;
  if(mode=="command")sampling=metal::CommandDispatchProfilingMode::Command;
  else if(mode=="stage")sampling=metal::CommandDispatchProfilingMode::StagePerDispatch;
  else if(mode=="dispatch")sampling=metal::CommandDispatchProfilingMode::DispatchBoundary;
  else require(mode=="normal","invalid profile mode");
  const std::string selector=std::getenv("FLASH_HEAD_PROFILE_PHASE")?std::getenv("FLASH_HEAD_PROFILE_PHASE"):"all";
  require(selector=="all"||selector=="proposal"||selector=="committed_fold"||selector=="joint_proposal","invalid phase selector");
  std::vector<uint32_t> contexts{128,2048};if(std::getenv("FLASH_HEAD_PROFILE_CONTEXT")){const auto value=integer("FLASH_HEAD_PROFILE_CONTEXT",0,128,2048);require(value==128||value==2048,"context must be128 or2048");contexts={value};}
  const uint32_t selectedRows=integer("FLASH_HEAD_PROFILE_ROWS",0,0,8);require(!selectedRows||selectedRows==1||selectedRows==4||selectedRows==8,"rows must be1/4/8");
  require(!(selectedRows==1&&selector=="committed_fold")&&!(selectedRows>1&&(selector=="proposal"||selector=="joint_proposal")),"phase/rows selection has no supported geometry");
  const uint32_t warmup=integer("FLASH_HEAD_PROFILE_WARMUP",1,0,2),repeats=integer("FLASH_HEAD_PROFILE_REPEATS",1,1,8);
  for(const auto &path:{std::string(argv[4]),std::string(argv[4])+".commands.jsonl",std::string(argv[4])+".trace.jsonl"})require(!std::filesystem::exists(path),"choose fresh report outputs");
  std::ofstream commands(std::string(argv[4])+".commands.jsonl"),trace(std::string(argv[4])+".trace.jsonl");require(commands&&trace,"cannot create trace outputs");
  commands<<std::setprecision(std::numeric_limits<double>::max_digits10);trace<<std::setprecision(std::numeric_limits<double>::max_digits10);
  metal::MetalBackend backend(argv[1]);const auto weights=FlashWeights::load(backend,argv[2]);
  const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);
  engine::MemoryGovernor governor(backend,physical-reserve,reserve);
  uint64_t planned=FlashForward::workspacePlannedBytes(kCapacity,2048,16)+FlashForward::expertCachePlannedBytes(weights)+
    FlashForward::floatDenseCachePlannedBytes(weights)+FlashForward::int8HeadPlannedBytes(weights)+FlashMTPForward::workspacePlannedBytes(kCapacity,128)+
    FlashBatchMTPForward::workspacePlannedBytes(kCapacity,4,4,true)+4*(FlashForward::requestStateBytes(kCapacity)+FlashMTPForward::requestStateBytes(kCapacity))+(256ULL<<20);
  if(enabled("SPLASH_FLASH_DENSE_CACHE"))planned+=FlashDenseCache::plannedBytes(weights,FlashDenseCache::defaultPrefixes(weights,true))+FlashMTPForward::denseCachePlannedBytes(weights);
  if(enabled("SPLASH_FLASH_BLOCKED_MOE"))planned+=flashMoEBlockedWorkspacePlannedBytes(2048,10);
  auto reservation=governor.tryReserve(planned);require(bool(reservation),"governor denied diagnostic arenas");
  FlashForward target(backend,weights,kCapacity,2048,16);FlashMTPForward head(backend,weights,kCapacity,128);
  FlashBatchMTPForward joint(head,4,4,target.cachedVocabulary());
  metal::ResidencyLease residency;if(enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
    auto operands=target.cachedOperandsOnly();const auto extra=head.cachedOperandsOnly();operands.insert(operands.end(),extra.begin(),extra.end());
    if(!operands.empty())residency=backend.requestWeightResidency(operands,"private trained head original saved source/head operands");
  }reservation->commit();
  std::vector<std::string> records;uint64_t profiled=0;
  for(uint32_t context:contexts)for(const std::string phase:{"proposal","committed_fold","joint_proposal"}) {
    if(selector!="all"&&selector!=phase)continue;
    const bool batched=phase=="joint_proposal";const uint32_t lanes=batched?4:1;
    const std::vector<uint32_t> rowChoices=phase=="committed_fold"?std::vector<uint32_t>{4,8}:std::vector<uint32_t>{1};
    for(uint32_t rows:rowChoices) {
      if(selectedRows&&selectedRows!=rows)continue;
      std::vector<std::vector<uint32_t>> prompts;for(uint32_t lane=0;lane<lanes;++lane) {
        const std::string name="ctx"+std::to_string(context)+"-width"+std::to_string(lanes)+"-lane"+std::to_string(lane)+".tokens.json";
        prompts.push_back(loadTokens((std::filesystem::path(argv[3])/name).string()));require(prompts.back().size()==context,"exact fixture context differs");
      }
      for(uint32_t trial=0;trial<warmup+repeats;++trial) {
        std::vector<Seed> seed;seed.reserve(lanes);for(uint32_t lane=0;lane<lanes;++lane)seed.push_back(prepare(backend,target,head,prompts[lane]));
        const bool measured=trial>=warmup;const auto selectedMode=measured?sampling:metal::CommandDispatchProfilingMode::Off;
        Fold fold;Capture capture,reference;bool outputParity=false;
        if(batched) {
          if(measured&&sampling!=metal::CommandDispatchProfilingMode::Off) {reference=runJoint(backend,joint,seed,metal::CommandDispatchProfilingMode::Off);for(auto &lane:seed)head.truncate(lane.head,context-1);}
          capture=runJoint(backend,joint,seed,selectedMode);
        }
        else if(phase=="committed_fold") {
          fold=trueTargetFold(backend,target,seed[0],rows);
          if(measured&&sampling!=metal::CommandDispatchProfilingMode::Off) {reference=runSingle(backend,head,seed[0],fold.features,fold.nextTokens,metal::CommandDispatchProfilingMode::Off);head.truncate(seed[0].head,context-1);}
          capture=runSingle(backend,head,seed[0],fold.features,fold.nextTokens,selectedMode);
        }
        else {
          const auto last=backend.view(seed[0].targetFeatures,seed[0].targetFeatures.sizeBytes()-kHyper*2,kHyper*2);
          const std::span<const uint32_t> next(&seed[0].anchor,1);
          if(measured&&sampling!=metal::CommandDispatchProfilingMode::Off) {reference=runSingle(backend,head,seed[0],last,next,metal::CommandDispatchProfilingMode::Off);head.truncate(seed[0].head,context-1);}
          capture=runSingle(backend,head,seed[0],last,next,selectedMode);
        }
        if(measured&&sampling!=metal::CommandDispatchProfilingMode::Off) {
          require(capture.hiddenSha==reference.hiddenSha&&capture.logitsSha==reference.logitsSha&&capture.predictions==reference.predictions,"sampled original head outputs differ from normal original replay");outputParity=true;
        }
        for(const auto &lane:seed)require(lane.head.logicalLength()==context-1+rows&&!lane.head.poisoned(),"head offset/health differs after original body");
        if(!measured)continue;
        const std::string label=phase+"-ctx"+std::to_string(context)+"-B"+std::to_string(lanes)+"-R"+std::to_string(rows)+"-sample"+std::to_string(trial-warmup);
        std::ostringstream record;record<<std::setprecision(17);
        record<<"{\"case\":"<<json::quote(label)<<",\"phase\":"<<json::quote(phase)<<",\"context_per_lane\":"<<context<<",\"lanes\":"<<lanes
          <<",\"true_rows_per_lane\":"<<rows<<",\"physical_rows\":"<<lanes*rows<<",\"logit_rows\":"<<lanes
          <<",\"head_length_before\":"<<context-1<<",\"head_length_after\":"<<context-1+rows<<",\"gpu_seconds\":"<<capture.timing.gpuSeconds
          <<",\"normal_replay_output_parity_checked\":"<<(outputParity?"true":"false")
          <<",\"normal_replay_gpu_seconds\":"<<(outputParity?std::to_string(reference.timing.gpuSeconds):"null")
          <<",\"command_wall_seconds\":"<<capture.timing.wallSeconds<<",\"head_api_call_seconds\":"<<capture.callSeconds
          <<",\"diagnostic_dispatches_added\":0,\"state_hash_reads_before_timed_call\":false,\"head_outputs_bf16_sha256\":"<<json::quote(capture.hiddenSha)
          <<",\"vocabulary_bf16_sha256\":"<<json::quote(capture.logitsSha)<<",\"host_command_subphases\":";writeHost(record,capture.timing.host);
        record<<",\"head_greedy_predictions\":[";for(size_t index=0;index<capture.predictions.size();++index){if(index)record<<',';record<<capture.predictions[index];}
        record<<"],\"committed_target_fold_tokens\":[";for(size_t index=0;index<fold.nextTokens.size();++index){if(index)record<<',';record<<fold.nextTokens[index];}
        record<<"],\"continuation_pending_token\":"<<(phase=="committed_fold"?std::to_string(fold.pending):"null")<<",\"lanes_seed\":[";
        for(uint32_t lane=0;lane<lanes;++lane){if(lane)record<<',';record<<"{\"lane\":"<<lane<<",\"anchor\":"<<seed[lane].anchor
          <<",\"prompt_u32le_sha256\":"<<json::quote(seed[lane].promptSha)<<",\"last_true_target_premixer_bf16_sha256\":"<<json::quote(seed[lane].lastFeatureSha)<<'}';}
        record<<"]}";records.push_back(record.str());commands<<record.str()<<'\n';
        for(const auto &profile:capture.profiles) {++profiled;trace<<"{\"case\":"<<json::quote(label)<<",\"phase\":"<<json::quote(phase)
          <<",\"context_per_lane\":"<<context<<",\"lanes\":"<<lanes<<",\"true_rows_per_lane\":"<<rows<<",\"physical_rows\":"<<lanes*rows
          <<",\"attribution_scope\":\"one original trained head command; stage/dispatch perturb scheduling; no diagnostic copies\",\"family_attribution\":";
          writeFamilies(trace,profile,lanes,lanes*rows);trace<<",\"command\":";profiling::writeJson(trace,profile);trace<<"}\n";}
        commands.flush();trace.flush();require(commands&&trace,"trace output write failed");
      }
    }
  }
  require(!records.empty(),"selection produced no head cases");
  std::ofstream report(argv[4]);require(bool(report),"cannot create report");report<<std::setprecision(17);
  report<<"{\"schema\":\"splash-original-trained-head-attribution-v8-private-v2\",\"execution_complete\":true,\"gpu_executed\":true,\"command_timing_abi_bytes\":"<<sizeof(metal::CommandTiming)
    <<",\"scope\":\"original trained MTP head; true target premixer features; exact benchmark contexts; teacher primed prompt state; true greedy target continuation folds; no HTTP performance claim\""
    <<",\"profile_mode\":"<<json::quote(mode)<<",\"warmup_per_case\":"<<warmup<<",\"measured_repeats_per_case\":"<<repeats
    <<",\"diagnostic_dispatches_added_per_head\":0,\"state_hash_reads_before_timed_call\":false,\"output_hash_reads_after_timed_call\":true"
    <<",\"source_identity_sha256\":"<<json::quote(weights.sourceIdentity())<<",\"manifest_fingerprint_sha256\":"<<json::quote(weights.manifestFingerprint())
    <<",\"kernel_routes\":"<<json::quote(target.kernelRoutes())<<",\"head_attention_route\":"<<json::quote(head.attentionRouteSemantics())
    <<",\"head_projection_route\":"<<json::quote(head.projectionRouteSemantics())<<",\"joint_head_attention_route\":"<<json::quote(joint.attentionRouteSemantics())
    <<",\"metallib_sha256\":"<<json::quote(hexadecimal(backend.metallibSha256()))<<",\"fixture_provenance_sha256\":"<<json::quote(fileDigest((std::filesystem::path(argv[3])/"fixture-provenance.json").string()))
    <<",\"saved_residency_buffers\":"<<residency.bufferCount()<<",\"saved_residency_bytes\":"<<residency.byteCount()<<",\"cases\":"<<records.size()<<",\"counter_profiles\":"<<profiled<<",\"measurements\":[";
  for(size_t index=0;index<records.size();++index){if(index)report<<',';report<<records[index];}report<<"]}\n";require(bool(report),"report write failed");
  std::cout<<"Original trained head attribution complete; cases="<<records.size()<<" report="<<argv[4]<<'\n';return 0;
 }catch(const std::exception &error){std::cerr<<"Original trained head attribution failed: "<<error.what()<<'\n';return 1;}}
}
