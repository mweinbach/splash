// Root-only TRUNKPREF preservation proof. --cpu-only constructs no backend.
// Teacher export and phase comparison must run in separate processes.
#include "flash/FlashForward.hpp"
#include "flash/FlashRequestStateInternal.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include "dev/benchmarks/dense_w8a8_sep21/worker_cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "BatchVerifyBuildProvenance.hpp"
#include "flash/FlashBatchVerify.hpp"
#include "metal/abi/FlashMoEBuckets.h"
#include "metal/abi/FlashForward.h"
#include "metal/CommandGraph.hpp"
#include "inspect.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


namespace {
using namespace splash;
using namespace splash::flash;
namespace fs=std::filesystem;
using Access=FlashDeepPrefixOracle; using Type=Access::Type;
constexpr uint32_t kCapacity=4096,kRows=2048,kVerifyRows=4,kVocabulary=248320,kHyper=10240;
constexpr uint64_t kLimit=4ULL<<30,kScratch=1ULL<<20;
constexpr const char *kFixtureSHA="4985e55294b83c72cb9e51e00c40f918460b6c4f560cb5b32d4be3662e540b57";
constexpr const char *kTokensSHA="0a383d21f5c784b0616d589847ca6cb04c69bf654729f2344e2b916e542f36b4";
static_assert(sizeof(metal::CommandTiming)==200);
void require(bool ok,const std::string &message) { if (!ok) throw std::runtime_error(message); }
struct Hash {
  CC_SHA256_CTX state{}; Hash(){CC_SHA256_Init(&state);}
  void add(const void *raw,uint64_t count) {
    const auto *p=static_cast<const uint8_t *>(raw);
    while (count) {const auto n=CC_LONG(std::min<uint64_t>(count,UINT32_MAX)); CC_SHA256_Update(&state,p,n);p+=n;count-=n;}
  }
  std::string finish() {
    std::array<uint8_t,32> result{};CC_SHA256_Final(result.data(),&state);std::ostringstream out;
    for (uint8_t value:result) out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(value);
    return out.str();
  }
};
std::string bytesHash(const void *raw,uint64_t bytes){Hash h;h.add(raw,bytes);return h.finish();}
std::string fileHash(const fs::path &path) {
  std::ifstream in(path,std::ios::binary);require(bool(in),"cannot open provenance/input file: "+path.string());
  Hash h;std::array<char,65536> scratch{};
  while (in) {in.read(scratch.data(),scratch.size());if(in.gcount())h.add(scratch.data(),uint64_t(in.gcount()));}
  require(in.eof(),"provenance/input read failed");return h.finish();
}
NSDictionary *dictionary(const fs::path &path) {
  require(fs::is_regular_file(path) && fs::file_size(path)<=2ULL<<20,"bounded JSON declaration invalid");
  NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  NSError *error=nil;id result=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error==nil && [result isKindOfClass:[NSDictionary class]],"JSON declaration must be dictionary");
  return static_cast<NSDictionary *>(result);
}
std::string string(id value) {require([value isKindOfClass:[NSString class]],"JSON string required");return static_cast<NSString *>(value).UTF8String;}
uint64_t number(id value) {require([value isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)value)!=CFBooleanGetTypeID(),"JSON integer required");
  const double n=static_cast<NSNumber *>(value).doubleValue;require(std::isfinite(n)&&n>=0&&std::floor(n)==n&&n<=double(kLimit),"JSON integer outside bound");return uint64_t(n);}
std::vector<uint32_t> tokens(const fs::path &path) {
  require(fs::is_regular_file(path) && fs::file_size(path)<=2ULL<<20,"prompt extent invalid");
  require(fileHash(path)==kFixtureSHA,"canonical prompt file SHA differs");
  NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  NSError *error=nil;id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error==nil && [value isKindOfClass:[NSArray class]],"prompt must be token array");
  std::vector<uint32_t> out;
  for (id item:static_cast<NSArray *>(value)) {const uint64_t n=number(item);require(n<kVocabulary,"token exceeds vocabulary");out.push_back(uint32_t(n));}
  require(out.size()==kRows,"exactly2048 canonical prompt tokens required");
  require(bytesHash(out.data(),out.size()*4)==kTokensSHA,"canonical token-word SHA differs");return out;
}
struct View {std::string label;Type type;const uint8_t *data;uint64_t live,bytes;};
uint32_t wordWidth(Type type) noexcept {
  switch(type){case Type::BF16:return 2;case Type::F32:case Type::U32:return 4;case Type::I64:return 8;}
  return 0;
}
const char *typeName(Type type) noexcept {
  switch(type){case Type::BF16:return "BF16";case Type::F32:return "F32";case Type::I64:return "I64";case Type::U32:return "U32";}
  return "unknown";
}
std::string bounded(std::string_view text,size_t limit){return std::string(text.substr(0,limit));}
// This survives Store/Stream destruction and never owns or reads tensor data.
struct FailureProgress {
  std::string role="unassigned",frame,file,section="arguments",plane,dtype,kind="exception";
  uint64_t expectedBytes=0,observedBytes=0,absoluteByte=0,planeByte=0,payloadBegin=0;
  uint64_t currentMatchedBytes=0,completedUnique=0,completedRepeated=0,completedPlanes=0;
  uint64_t spilledBytes=0,comparedBytes=0,comparedPlanes=0;
  uint32_t width=0,actualByte=0,controlByte=0;
  bool expectedKnown=false,observedKnown=false,absoluteKnown=false,planeByteKnown=false;
  bool payloadSection=false,byteValuesKnown=false,repeated=false;
  std::vector<std::string> completedLabels,repeatedLabels;
  void begin(const std::string &label,bool repeat=false){
    frame=label;file=label.empty()?"":label+".bin";section="frame_preflight";plane.clear();dtype.clear();kind="exception";
    expectedBytes=observedBytes=absoluteByte=planeByte=payloadBegin=currentMatchedBytes=0;
    width=actualByte=controlByte=0;expectedKnown=observedKnown=absoluteKnown=planeByteKnown=false;
    payloadSection=byteValuesKnown=false;repeated=repeat;
  }
  void at(const std::string &part,uint64_t position,const View *view=nullptr,bool payload=false){
    section=part;absoluteByte=position;absoluteKnown=true;payloadSection=payload;planeByteKnown=false;
    if(view){plane=view->label;dtype=typeName(view->type);width=wordWidth(view->type);}
    else{plane.clear();dtype.clear();width=0;}
    if(payload){payloadBegin=position;planeByte=0;planeByteKnown=true;}
  }
  void offset(uint64_t position){
    absoluteByte=position;absoluteKnown=true;
    if(payloadSection&&position>=payloadBegin){planeByte=position-payloadBegin;planeByteKnown=true;}
  }
  void byteMismatch(uint64_t position,uint8_t actual,uint8_t control){
    kind="byte_mismatch";offset(position);actualByte=actual;controlByte=control;byteValuesKnown=true;
  }
  std::string json()const{
    std::ostringstream out;out<<"{\"role\":"<<splash::json::quote(bounded(role,16))<<",\"frame_label\":"<<splash::json::quote(bounded(frame,256))
      <<",\"frame_file\":"<<splash::json::quote(bounded(file,256))<<",\"repeated_frame\":"<<(repeated?"true":"false")
      <<",\"section\":"<<splash::json::quote(bounded(section,64))<<",\"plane\":"<<splash::json::quote(bounded(plane,256))
      <<",\"dtype\":"<<splash::json::quote(dtype)<<",\"word_width\":"<<width<<",\"failure_kind\":"<<splash::json::quote(kind)
      <<",\"expected_frame_bytes\":"<<(expectedKnown?std::to_string(expectedBytes):"null")
      <<",\"observed_frame_bytes\":"<<(observedKnown?std::to_string(observedBytes):"null")
      <<",\"absolute_byte\":"<<(absoluteKnown?std::to_string(absoluteByte):"null")
      <<",\"plane_relative_byte\":"<<(planeByteKnown?std::to_string(planeByte):"null")
      <<",\"word_index\":"<<(planeByteKnown&&width?std::to_string(planeByte/width):"null")
      <<",\"byte_in_word\":"<<(planeByteKnown&&width?std::to_string(planeByte%width):"null")
      <<",\"actual_byte\":"<<(byteValuesKnown?std::to_string(actualByte):"null")
      <<",\"control_byte\":"<<(byteValuesKnown?std::to_string(controlByte):"null")
      <<",\"current_frame_matched_bytes\":"<<currentMatchedBytes<<",\"completed_unique_frames\":"<<completedUnique
      <<",\"expected_unique_frames\":26,\"completed_repeated_frames\":"<<completedRepeated<<",\"expected_repeated_frames\":54"
      <<",\"completed_planes\":"<<completedPlanes<<",\"spill_bytes\":"<<spilledBytes
      <<",\"bytes_compared\":"<<comparedBytes<<",\"planes_compared\":"<<comparedPlanes<<",\"completed_frame_labels\":[";
    for(size_t i=0;i<std::min<size_t>(completedLabels.size(),64);++i){if(i)out<<',';out<<splash::json::quote(bounded(completedLabels[i],32));}
    out<<"],\"completed_repeated_labels\":[";
    for(size_t i=0;i<std::min<size_t>(repeatedLabels.size(),256);++i){if(i)out<<',';out<<splash::json::quote(bounded(repeatedLabels[i],32));}
    out<<"]}";return out.str();
  }
};
class FrameFailure final:public std::runtime_error {
public:explicit FrameFailure(const std::string &message):std::runtime_error(message){}
};
struct AllocationBreakdown {
  uint64_t fixed=0,experts=0,f32=0,head=0,bf16=0,blocked=0,w8Cache=0,state=0,margin=16ULL<<20;
  uint64_t plannedTarget=0,reservation=0,mappedInitial=0,afterTarget=0,targetDelta=0,workspace=0;
  uint64_t beforeState=0,afterState=0,stateDelta=0,finalAllocated=0;
  uint64_t bf16SavedCount=0,bf16SavedPayload=0,f32SavedCount=0,f32SavedPayload=0;
  engine::MemoryGovernorSnapshot governor{};std::string governorStage="not_created";
  bool targetMeasured=false,stateMeasured=false;
  std::string json()const{
    std::ostringstream out;out<<"{\"planned\":{\"fixed_workspace_including_w8_activation\":"<<fixed
      <<",\"expert_store_and_cache\":"<<experts<<",\"float_dense_cache\":"<<f32
      <<",\"int8_head\":"<<head<<",\"bf16_dense_cache\":"<<bf16<<",\"blocked_moe\":"<<blocked
      <<",\"dense_w8_coefficients\":"<<w8Cache<<",\"target_total\":"<<plannedTarget
      <<",\"one_request_state\":"<<state<<",\"diagnostic_margin\":"<<margin<<",\"reservation_total\":"<<reservation
      <<"},\"actual\":{\"model_mapped_initial\":"<<mappedInitial<<",\"after_target\":"<<afterTarget
      <<",\"target_delta\":"<<targetDelta<<",\"forward_workspace\":"<<workspace
      <<",\"before_state\":"<<beforeState<<",\"after_state\":"<<afterState<<",\"state_delta\":"<<stateDelta
      <<",\"final_allocated\":"<<finalAllocated<<",\"bf16_saved_tensors\":"<<bf16SavedCount<<",\"bf16_saved_payload_bytes\":"<<bf16SavedPayload
      <<",\"f32_saved_tensors\":"<<f32SavedCount<<",\"f32_saved_payload_bytes\":"<<f32SavedPayload
      <<",\"workspace_plus_state_plan\":"<<(targetMeasured?workspace+state:0)
      <<",\"reservation_shortfall\":"<<(targetMeasured&&workspace+state>reservation?workspace+state-reservation:0)
      <<",\"reservation_slack\":"<<(targetMeasured&&workspace+state<=reservation?reservation-workspace-state:0)
      <<",\"target_measured\":"<<(targetMeasured?"true":"false")
      <<",\"state_measured\":"<<(stateMeasured?"true":"false")
      <<"},\"guards\":{\"target_ledger_matches_workspace\":"<<(targetMeasured&&targetDelta==workspace?"true":"false")
      <<",\"target_fits_category_plan\":"<<(targetMeasured&&targetDelta<=plannedTarget?"true":"false")
      <<",\"workspace_plus_state_plan_fits_reservation\":"<<(targetMeasured&&workspace<=reservation&&state<=reservation-workspace?"true":"false")
      <<",\"state_fits_state_plan\":"<<(stateMeasured&&stateDelta<=state?"true":"false")
      <<",\"workspace_plus_actual_state_fits_reservation\":"<<(stateMeasured&&workspace<=reservation&&stateDelta<=reservation-workspace?"true":"false")
      <<",\"live_backend_delta_fits_reservation\":"<<(targetMeasured&&finalAllocated>=mappedInitial&&finalAllocated-mappedInitial<=reservation?"true":"false")
      <<"},\"governor_snapshot\":{\"stage\":"<<splash::json::quote(governorStage)<<",\"limit_bytes\":"<<governor.limitBytes
      <<",\"observed_resident_bytes\":"<<governor.observedResidentBytes<<",\"reserved_bytes\":"<<governor.reservedBytes
      <<",\"headroom_bytes\":"<<governor.headroomBytes<<",\"denied_reservations\":"<<governor.deniedReservations
      <<",\"host_measurement_valid\":"<<(governor.hostMeasurementValid?"true":"false")<<",\"host_headroom_bytes\":"<<governor.hostHeadroomBytes
      <<",\"growth_allowed\":"<<(governor.growthAllowed?"true":"false")<<"}}";
    return out.str();
  }
};
struct BackendLifetime {
  bool &created,&destroyed;
  ~BackendLifetime(){if(created)destroyed=true;}
};
void finite(const View &plane,FailureProgress *progress=nullptr,uint64_t payloadBegin=0) {
  if(progress)progress->at("plane_finite",payloadBegin,&plane,true);
  if (plane.type==Type::I64 || plane.type==Type::U32) return;
  const uint32_t width=plane.type==Type::BF16?2:4;require(plane.bytes%width==0,"partial numeric word");
  for(uint64_t i=0;i<plane.bytes;i+=width){uint32_t bits=0;std::memcpy(&bits,plane.data+i,width);if(width==2)bits<<=16;
    if(!std::isfinite(std::bit_cast<float>(bits))){if(progress){progress->kind="nonfinite_word";progress->offset(payloadBegin+i);}
      throw FrameFailure("nonfinite checkpoint word: "+plane.label+" byte="+std::to_string(i));}}
}
std::array<uint8_t,8> little(uint64_t value){std::array<uint8_t,8> out{};for(uint32_t i=0;i<8;++i)out[i]=uint8_t(value>>(8*i));return out;}
uint64_t extent(const std::string &meta,const std::vector<View> &planes,FailureProgress *progress=nullptr){
  if(progress)progress->at("frame_extent_metadata",0);
  require(meta.size()<=65536 && !planes.empty() && planes.size()<=1024,"frame declaration invalid");
  uint64_t n=16+8+meta.size()+8;
  for(const auto &p:planes){if(progress)progress->at("frame_extent_plane",n,&p);
    require(!p.label.empty()&&p.label.size()<=256&&p.live<=p.bytes&&(p.data||!p.bytes),"plane declaration invalid");
    const uint64_t add=8+p.label.size()+24;require(add<=kLimit-n,"frame bound exceeded");n+=add;require(p.bytes<=kLimit-n,"frame bound exceeded");n+=p.bytes;}
  return n;
}
class Stream {
public:
  Stream(fs::path path,bool write,uint64_t expected,FailureProgress *progress=nullptr):path_(std::move(path)),write_(write),expected_(expected),scratch_(kScratch),progress_(progress){
    if(progress_){progress_->file=path_.filename().string();progress_->expectedBytes=expected;progress_->expectedKnown=true;
      progress_->at(write_?"frame_file_create":"frame_file_extent",0);}
    if(write_){require(!fs::exists(path_)&&!fs::exists(path_.string()+".writing"),"fresh frame path required");out_.open(path_.string()+".writing",std::ios::binary);require(bool(out_),"cannot open frame output");}
    else{const bool regular=fs::is_regular_file(path_);if(regular&&progress_){progress_->observedBytes=fs::file_size(path_);progress_->observedKnown=true;}
      require(regular&&fs::file_size(path_)==expected_,"frame exact physical extent differs");in_.open(path_,std::ios::binary);require(bool(in_),"cannot open frame input");}
  }
  uint64_t position()const noexcept{return count_;}
  void section(const std::string &label,const View *view=nullptr,bool payload=false){if(progress_)progress_->at(label,count_,view,payload);}
  void part(const void *raw,uint64_t bytes){require(raw||!bytes,"null frame payload");require(bytes<=kLimit-count_,"frame limit exceeded");
    const auto *p=static_cast<const uint8_t *>(raw);while(bytes){const uint64_t n=std::min<uint64_t>(bytes,scratch_.size());
      if(write_){out_.write(reinterpret_cast<const char *>(p),std::streamsize(n));require(bool(out_),"frame write failed");}
      else{in_.read(reinterpret_cast<char *>(scratch_.data()),std::streamsize(n));
        if(in_.gcount()!=std::streamsize(n)){if(progress_){progress_->kind="truncated_frame";progress_->offset(count_+uint64_t(std::max<std::streamsize>(0,in_.gcount())));}
          throw FrameFailure("truncated frame");}
        if(std::memcmp(p,scratch_.data(),size_t(n))){uint64_t i=0;while(i<n&&p[i]==scratch_[i])++i;
          if(progress_){progress_->byteMismatch(count_+i,p[i],scratch_[i]);progress_->currentMatchedBytes=count_+i;}
          throw FrameFailure("exact frame mismatch: "+path_.filename().string()+" byte="+std::to_string(count_+i));}}
      hash_.add(p,n);p+=n;bytes-=n;count_+=n;if(progress_){progress_->offset(count_);progress_->currentMatchedBytes=count_;}}}
  void num(uint64_t n){const auto b=little(n);part(b.data(),b.size());}
  void text(const std::string &s,const std::string &label,const View *view=nullptr){section(label+"_length",view);num(s.size());section(label,view);part(s.data(),s.size());}
  std::string finish(){section("frame_finish");require(count_==expected_,"frame consumed extent differs");if(write_){out_.close();require(bool(out_),"frame close failed");fs::rename(path_.string()+".writing",path_);require(fs::file_size(path_)==expected_,"published frame extent differs");}
    else require(in_.peek()==std::char_traits<char>::eof(),"trailing frame bytes");return hash_.finish();}
private:
  fs::path path_;bool write_;uint64_t expected_,count_=0;std::ifstream in_;std::ofstream out_;std::vector<uint8_t> scratch_;Hash hash_;FailureProgress *progress_;
};
void serialize(Stream &stream,const std::string &meta,const std::vector<View> &planes,FailureProgress *progress=nullptr){
  constexpr std::array<uint8_t,16> magic{'S','P','L','A','S','H','P','R','E','F','E','X',1,0,0,0};
  stream.section("magic");stream.part(magic.data(),magic.size());stream.text(meta,"metadata");stream.section("plane_count");stream.num(planes.size());
  for(const auto &p:planes){stream.text(p.label,"plane_label",&p);stream.section("plane_type",&p);stream.num(uint32_t(p.type));
    stream.section("plane_live_bytes",&p);stream.num(p.live);stream.section("plane_physical_bytes",&p);stream.num(p.bytes);
    finite(p,progress,stream.position());stream.section("plane_payload",&p,true);stream.part(p.data,p.bytes);}
}
struct Frame {std::string label,sha;uint64_t bytes,planes,live;};
class Store {
public:
  Store(fs::path spill,bool exporting,FailureProgress &progress):spill_(std::move(spill)),exporting_(exporting),progress_(progress){
    progress_.section=exporting_?"spill_directory":"export_manifest";
    if(exporting_){require(!fs::exists(spill_),"fresh spill directory required");fs::create_directories(spill_);}
    else{const auto manifest=dictionary(spill_/"complete.json");require([manifest[@"complete"] isEqual:@YES]&&string(manifest[@"schema"])=="batchverify-export-v1","completed export required");
      for(id item:static_cast<NSArray *>(manifest[@"frames"])){const auto entry=static_cast<NSDictionary *>(item);frames_.push_back({string(entry[@"label"]),string(entry[@"sha256"]),number(entry[@"bytes"]),number(entry[@"planes"]),number(entry[@"live_bytes"])});}
      require(!frames_.empty(),"nonempty complete campaign export required");}}
  void frame(const std::string &label,const std::string &meta,const std::vector<View> &planes,bool repeat=false){
    progress_.begin(label,repeat);const auto bytes=extent(meta,planes,&progress_);uint64_t live=0;for(const auto &p:planes)live+=p.live;
    progress_.expectedBytes=bytes;progress_.expectedKnown=true;progress_.at("frame_manifest_preflight",0);progress_.absoluteKnown=false;
    const auto it=std::find_if(frames_.begin(),frames_.end(),[&](const auto &f){return f.label==label;});
    const bool write=exporting_&&!repeat;require(write?it==frames_.end():it!=frames_.end(),"missing/duplicate frame declaration");
    if(write)require(bytes<=kLimit-spilled_,"aggregate spill exceeds4GiB before write");
    else require(it->bytes==bytes&&it->planes==planes.size()&&it->live==live,"frame metadata/geometry differs");
    Stream stream(spill_/(label+".bin"),write,bytes,&progress_);serialize(stream,meta,planes,&progress_);const auto sha=stream.finish();progress_.section="frame_digest";
    if(write){frames_.push_back({label,sha,bytes,planes.size(),live});spilled_+=bytes;}
    else {require(sha==it->sha,"frame provenance digest differs");compared_+=bytes;checked_+=planes.size();}
    if(!repeat){++completed_;progress_.completedLabels.push_back(label);}else{++repeated_;progress_.repeatedLabels.push_back(label);}
    progress_.completedUnique=completed_;progress_.completedRepeated=repeated_;progress_.completedPlanes+=planes.size();
    progress_.spilledBytes=spilled_;progress_.comparedBytes=compared_;progress_.comparedPlanes=checked_;progress_.section="frame_complete";}
  void state(const std::string &label,const FlashForward &target,const FlashRequestState &state,bool repeat=false,bool pending=false,bool poisoned=false){
    progress_.begin(label,repeat);progress_.section="state_preflight";
    require(Access::rawOwns(target,state)&&state.capacity()==kCapacity&&state.poisoned()==poisoned&&Access::pending(state)==pending,"state must be healthy/resolved/locally owned");
    std::vector<View> views;uint64_t bytes=0;std::vector<std::pair<uintptr_t,uintptr_t>> ranges;
    for(const auto &p:Access::planes(state)){views.push_back({p.label,p.type,static_cast<const uint8_t *>(p.buffer.contents()),p.live,p.buffer.sizeBytes()});progress_.at("state_plane_preflight",0,&views.back());progress_.absoluteKnown=false;bytes+=p.buffer.sizeBytes();
      const uintptr_t begin=reinterpret_cast<uintptr_t>(p.buffer.contents());require(p.buffer.sizeBytes()<=UINTPTR_MAX-begin,"state range overflow");ranges.emplace_back(begin,begin+p.buffer.sizeBytes());}
    std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();++i)require(ranges[i].first>=ranges[i-1].second,"persistent planes overlap");
    require(bytes==FlashForward::requestStateBytes(kCapacity),"physical planes differ from admission formula");frame(label,Access::metadata(target,state),views,repeat);}
  void output(const std::string &label,const FlashForwardResult &result,uint32_t consumed,bool repeat=false){
    progress_.begin(label,repeat);progress_.section="output_preflight";
    require(result.capacity==kCapacity&&result.logitRows>0&&result.logitRows<=16&&result.logitRows<=consumed,"output shape invalid");std::vector<View> views;
    const auto add=[&](std::string name,Type type,const metal::MetalBuffer &buffer,uint64_t bytes){
      const View declared{name,type,nullptr,bytes,bytes};progress_.at("output_plane_preflight",0,&declared);progress_.absoluteKnown=false;
      require(buffer&&buffer.contents()&&buffer.sizeBytes()==bytes,"output exact borrowed extent invalid: "+name);views.push_back({std::move(name),type,static_cast<const uint8_t *>(buffer.contents()),bytes,bytes});};
    add("hidden",Type::BF16,result.hiddenBF16,uint64_t{consumed}*kHyper*2);add("logits",Type::BF16,result.logitsBF16,uint64_t{result.logitRows}*kVocabulary*2);
    require(result.greedyRows==result.logitRows&&result.greedyResultsU32,"GPU compact greedy required");
    add("compact_greedy",Type::U32,result.greedyResultsU32,uint64_t{result.greedyRows}*sizeof(FlashGreedyGPURowResult));
    for(uint32_t r=0;r<result.greedyRows;++r)(void)greedyGPUResultToken(static_cast<const FlashGreedyGPURowResult *>(result.greedyResultsU32.contents())[r],kVocabulary);
    std::ostringstream meta;meta<<"{\"length\":"<<result.logicalLength<<",\"capacity\":"<<result.capacity<<",\"consumed_rows\":"<<consumed<<",\"logit_rows\":"<<result.logitRows<<",\"greedy_rows\":"<<result.greedyRows<<'}';frame(label,meta.str(),views,repeat);}
  void complete(const std::string &common,const std::string &different){
    progress_.section="completion_preflight";
    require(completed_>0,"at least one selected checkpoint required");
    if(!exporting_)require(completed_==frames_.size(),"comparator omitted an exported campaign frame");
    if(exporting_){std::ostringstream out;out<<"{\"schema\":\"batchverify-export-v1\",\"complete\":true,\"common\":"<<common<<",\"producer\":"<<different<<",\"spill_bytes\":"<<spilled_<<",\"frames\":[";
      for(size_t i=0;i<frames_.size();++i){if(i)out<<',';const auto &f=frames_[i];out<<"{\"label\":"<<json::quote(f.label)<<",\"sha256\":"<<json::quote(f.sha)<<",\"bytes\":"<<f.bytes<<",\"planes\":"<<f.planes<<",\"live_bytes\":"<<f.live<<'}';}out<<"]}\n";publish(spill_/"complete.json",out.str());}}
  static void publish(const fs::path &path,const std::string &text){require(!fs::exists(path)&&!fs::exists(path.string()+".writing"),"fresh report/declaration path required");std::ofstream out(path.string()+".writing");out<<text;out.close();require(bool(out),"JSON publish failed");fs::rename(path.string()+".writing",path);}
  uint64_t completed()const{return completed_;}uint64_t repeated()const{return repeated_;}uint64_t spilled()const{return spilled_;}uint64_t compared()const{return compared_;}uint64_t checked()const{return checked_;}
private:fs::path spill_;bool exporting_;std::vector<Frame> frames_;uint64_t spilled_=0,compared_=0,checked_=0,completed_=0,repeated_=0;FailureProgress &progress_;
};
bool selected(const char *name){const char *v=std::getenv(name);require(v&&(*v=='0'||*v=='1')&&!v[1],std::string("strict explicit flag required: ")+name);return *v=='1';}
std::string commonPolicy(){
  constexpr std::array names{"SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21","SPLASH_FLASH_ALLROWS_GATHERED_MPP","SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS","SPLASH_FLASH_BLOCKED_MOE","SPLASH_FLASH_DENSE_CACHE","SPLASH_FLASH_DENSE_M64_OUT","SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21","SPLASH_FLASH_EXPERT_QMV","SPLASH_FLASH_FLOAT_DENSE_CACHE","SPLASH_FLASH_FLOAT_DENSE_SELECTIVE","SPLASH_FLASH_FUSE_GDN","SPLASH_FLASH_FUSE_HC","SPLASH_FLASH_GDN_LAZY_ROLLBACK","SPLASH_FLASH_GDN_PREFILL_FMA_SEP21","SPLASH_FLASH_GDN_STAGED","SPLASH_FLASH_GPU_GREEDY","SPLASH_FLASH_HC_UP_F32_MPP","SPLASH_FLASH_INT8_HEAD","SPLASH_FLASH_MOE_DIRECT_A","SPLASH_FLASH_MOE_M64","SPLASH_FLASH_MOE_POINTWISE_SEP21","SPLASH_FLASH_MOE_Q4X8","SPLASH_FLASH_PLE_LOOKUP_FUSED","SPLASH_FLASH_PLE_POST_FUSED","SPLASH_FLASH_PLE_SSD_STREAMING","SPLASH_FLASH_PREFILL_DENSE_TILES","SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21","SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT","SPLASH_FLASH_QMV_F32","SPLASH_FLASH_QSA_BULK_PREFILL","SPLASH_FLASH_QSA_BULK_PREFILL_SG8","SPLASH_FLASH_QSA_F32","SPLASH_FLASH_QSA_MPP","SPLASH_FLASH_QSA_OUT_F32_N32","SPLASH_FLASH_QSA_ROW_TILES","SPLASH_FLASH_SHARED_EXPERT_FUSED","SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21","SPLASH_FLASH_GDN_AB_MERGE_SEP21","SPLASH_FLASH_GDN_BATCH_ILP"};
  std::ostringstream out;out<<'{';for(size_t i=0;i<names.size();++i){const char *v=std::getenv(names[i]);require(v,std::string("common flag absent: ")+names[i]);if(i)out<<',';out<<json::quote(names[i])<<':'<<json::quote(v);}out<<'}';return out.str();
}
void early(){
  require(selected("SPLASH_FLASH_ALLROWS_FULL512_TARGET"),"pure I8 parent/candidate required");
  require(!selected("SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21"),"phase-Q4 profile excluded");
  require(!selected("SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22"),"singleton compact excluded");
  require(!selected("SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22"),"singleton HC-pad excluded");
  for(const char *n:{"SPLASH_FLASH_BATCH","SPLASH_FLASH_BATCH_MTP","SPLASH_FLASH_BATCH_MTP_PREFILL","SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_MTP","SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT","SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"})require(!selected(n),std::string("excluded worker route: ")+n);
  require(selected("SPLASH_FLASH_GDN_LAZY_ROLLBACK")&&selected("SPLASH_FLASH_GPU_GREEDY"),"guarded lazy tapes/greedy required");
  (void)commonPolicy();
}
std::vector<View> tapeViews(const FlashForward &target){std::vector<View> out;for(const auto &p:Access::tapePlanes(target))out.push_back({p.label,p.type,static_cast<const uint8_t *>(p.buffer.contents()),p.live,p.buffer.sizeBytes()});return out;}
struct LocalTails {
  struct Entry{Access::Undefined region;std::vector<uint8_t> bytes;};std::vector<Entry> entries;
  explicit LocalTails(const FlashForward &target){for(auto p:Access::tapeUndefined(target)){
    require(p.buffer&&p.buffer.contents()&&p.begin<=p.buffer.sizeBytes(),"undefined tape region invalid");const auto *data=static_cast<const uint8_t *>(p.buffer.contents())+p.begin;entries.push_back({p,{data,data+(p.buffer.sizeBytes()-p.begin)}});}}
  void allowRetained(uint32_t value){for(auto &p:entries)if(p.region.label=="retained_count.inactive"){require(p.bytes.size()>=4,"retained snapshot short");std::memcpy(p.bytes.data(),&value,4);}}
  void check()const{for(const auto &p:entries)require(!std::memcmp(static_cast<const uint8_t *>(p.region.buffer.contents())+p.region.begin,p.bytes.data(),p.bytes.size()),"undefined tape tail changed: "+p.region.label);}
};
struct Bindings {
  std::array<uintptr_t,3> state;std::vector<std::pair<uintptr_t,uint64_t>> planes,tapes;
  Bindings(const FlashForward &target,const FlashRequestState &request):state(Access::binding(request)){
    for(const auto &p:Access::planes(request))planes.push_back({reinterpret_cast<uintptr_t>(p.buffer.contents()),p.buffer.sizeBytes()});
    for(const auto &p:Access::tapePlanes(target))tapes.push_back({reinterpret_cast<uintptr_t>(p.buffer.contents()),p.buffer.sizeBytes()});}
  void check(const FlashForward &target,const FlashRequestState &request)const{
    require(state==Access::binding(request),"request owner/identity changed");std::vector<std::pair<uintptr_t,uint64_t>> a,b;
    for(const auto &p:Access::planes(request))a.push_back({reinterpret_cast<uintptr_t>(p.buffer.contents()),p.buffer.sizeBytes()});
    for(const auto &p:Access::tapePlanes(target))b.push_back({reinterpret_cast<uintptr_t>(p.buffer.contents()),p.buffer.sizeBytes()});
    require(a==planes&&b==tapes,"state/tape pointer or extent changed");require(Access::tapeCanaries(target),"lazy tape redzone changed");}
};
template<class F>void rejected(F &&f,const char *label){bool bad=false;try{f();}catch(const std::invalid_argument &){bad=true;}catch(const std::logic_error &){bad=true;}require(bad,std::string("invalid operation accepted: ")+label);}
void checkGuards(const std::vector<Access::Guard> &guards){for(const auto &g:guards)g.check();}
uint32_t greedy(const FlashForwardResult &r,uint32_t row){require(r.greedyRows>row&&r.greedyResultsU32.contents(),"greedy record absent");return greedyGPUResultToken(static_cast<const FlashGreedyGPURowResult *>(r.greedyResultsU32.contents())[row],kVocabulary);}
}

namespace {
constexpr std::array<const char *,18> kCheckpoints{
  "initial", "fresh-r8.pending", "fresh-r8.committed", "fresh-r16.pending", "fresh-r16.committed",
  "same-r8.pending", "same-r16.pending", "drop4.committed", "drop3.committed", "drop2.committed",
  "fallback1.committed", "reconstituted4.pending", "mixed-terminal.committed", "future.completed",
  "original-widths.completed", "abort.pending", "abort.completed", "recovery.completed"};
struct StateSlot {
  std::unique_ptr<FlashRequestState> state;
  std::vector<Access::Guard> guards;
  std::array<uintptr_t,3> binding{};
  std::vector<std::pair<uintptr_t,uint64_t>> planes;
  uint32_t anchor=0;
};
struct BatchAllocation {
  uint64_t planned=0,measured=0,guardAllowance=Access::guardedBatchAllowance(),guardMeasured=0;
  uint64_t stateAllowance=0,maximumStateMeasured=0,maximumLiveDelta=0,siblingAllowance=0,partitionSpillPreflight=0;
  uint64_t guardPlannedExact=0,mirrorPlanned=32768,mirrorMeasured=0,debugCopyCommands=0,debugCopyBytes=0;
  std::string json()const{std::ostringstream o;o<<"{\"batch_workspace_planned\":"<<planned<<",\"batch_workspace_actual\":"<<measured
    <<",\"batch_guard_allowance\":"<<guardAllowance<<",\"batch_guard_actual\":"<<guardMeasured
    <<",\"four_state_guarded_allowance\":"<<stateAllowance<<",\"maximum_state_delta\":"<<maximumStateMeasured
    <<",\"maximum_live_backend_delta\":"<<maximumLiveDelta<<",\"independently_reserved_sibling_allowance\":"<<siblingAllowance
    <<",\"partition_spill_preflight_upper_bound\":"<<partitionSpillPreflight<<",\"batch_guard_exact_after_owner_metadata\":"<<guardPlannedExact
    <<",\"private_mirror_planned\":"<<mirrorPlanned<<",\"private_mirror_actual\":"<<mirrorMeasured
    <<",\"debug_copy_commands\":"<<debugCopyCommands<<",\"debug_copy_bytes\":"<<debugCopyBytes<<'}';return o.str();}
};
std::vector<std::string> selectors(const char *raw){std::stringstream s(raw);std::vector<std::string> out;std::string value;
  while(std::getline(s,value,',')){require(std::find(kCheckpoints.begin(),kCheckpoints.end(),value)!=kCheckpoints.end(),"unknown checkpoint selector: "+value);
    require(std::find(out.begin(),out.end(),value)==out.end(),"duplicate checkpoint selector");out.push_back(value);}
  require(!out.empty()&&out.size()<=2,"one or two full checkpoints per bounded partition required");return out;}
void cpuBatch(){
  require(FlashForward::requestStateBytes(kCapacity)==232603648ULL,"physical state formula changed");
  require(dense_w8a8_sep21::Cache::plannedBytes()==1890975744ULL,"W8 coefficient admission changed");
  require(Access::guardedStateAllowance()==2195456ULL&&Access::guardedBatchAllowance()==281436ULL,"guard allowances changed");
  require(16*((kVocabulary+kFlashGreedyGPUValuesPerPartition-1)/kFlashGreedyGPUValuesPerPartition)*sizeof(FlashGreedyGPURowResult)==31232,"original Private greedy plane shape changed");
  require(moEBucketJobCapacity(16,10,8)==531,"admitted maximum job capacity changed");
  const uint64_t state4=4*FlashForward::requestStateBytes(kCapacity),tapes=36*FlashGDNLazyRollback::plannedBytes(4,4);
  const uint64_t conservativeOne=state4+tapes+(1024ULL<<20);require(conservativeOne<kLimit,"one full checkpoint bound exceeds4GiB");
  // All always-recorded outputs and <=100 MoE points fit this declaration;
  // each frame is further constrained by the generic serializer bounds.
  const uint64_t outputBytes=12*(uint64_t{kRows}*kHyper*2+kVocabulary*2+sizeof(FlashGreedyGPURowResult))
    +(172ULL+61)*(kHyper*2+kVocabulary*2+sizeof(FlashGreedyGPURowResult));
  const uint64_t maximumMoEBytes=2375680ULL;
  require(outputBytes+100*maximumMoEBytes+(32ULL<<20)<(1ULL<<30),"constant campaign spill upper bound too small");
  std::vector<uint8_t> b(1024,0x5a);View v{"synthetic",Type::U32,b.data(),b.size(),b.size()};
  const auto p=fs::temp_directory_path()/("batchverify-cpu-"+std::to_string(NSProcessInfo.processInfo.processIdentifier));require(!fs::exists(p),"fresh CPU path required");
  const auto n=extent("{}",{v});Stream w(p,true,n);serialize(w,"{}",{v});const auto digest=w.finish();Stream r(p,false,n);serialize(r,"{}",{v});require(r.finish()==digest,"stream parity failed");fs::remove(p);
  // Mutation must fail at its true byte coordinate, without changing the gate.
  Stream w2(p,true,n);serialize(w2,"{}",{v});(void)w2.finish();b[777]^=1;bool rejectedMutation=false;
  try{Stream r2(p,false,n);serialize(r2,"{}",{v});(void)r2.finish();}catch(const FrameFailure &){rejectedMutation=true;}
  require(rejectedMutation,"stream accepted mutated payload");fs::remove(p);
  std::cout<<"{\"pass\":true,\"gpu_executed\":false,\"model_payload_bytes_read\":0,\"input_or_export_payload_bytes_read\":0,\"request_physical_planes\":134,\"batch_lazy_planes\":216,\"max_jobs\":531,\"max_routes\":160,\"max_operand_rows\":223,\"one_checkpoint_conservative_spill_bound\":"<<conservativeOne<<",\"stream_mutation_rejected\":true,\"checkpoints\":[";
  for(size_t i=0;i<kCheckpoints.size();++i){if(i)std::cout<<',';std::cout<<json::quote(kCheckpoints[i]);}std::cout<<"]}\n";
}
class Campaign {
public:
  Campaign(metal::MetalBackend &backend,engine::MemoryGovernor &governor,FlashForward &target,FlashBatchVerify &batch,Store &store,
      AllocationBreakdown &allocation,BatchAllocation &batchAllocation,std::span<const uint32_t> prompt,std::vector<std::string> selected)
      :backend_(backend),governor_(governor),target_(target),batch_(batch),store_(store),allocation_(allocation),batchAllocation_(batchAllocation),prompt_(prompt),selected_(std::move(selected)){
    uint64_t batchSerialized=0;for(const auto &p:Access::batchPlanes(batch_))batchSerialized+=p.buffer.sizeBytes();
    const uint64_t fullCheckpointBound=4*FlashForward::requestStateBytes(kCapacity)+batchSerialized+(4ULL<<20);
    batchAllocation_.partitionSpillPreflight=(1ULL<<30)+selected_.size()*fullCheckpointBound;
    require(batchAllocation_.partitionSpillPreflight<kLimit,"selected partition spill preflight exceeds4GiB before any payload write");
    batchAllocation_.guardPlannedExact=Access::batchGuardPlannedDelta(batch_);
    const auto before=backend_.memoryStats().allocatedBytes;batchGuards_=Access::guardBatchMoE(backend_,batch_);
    const auto after=backend_.memoryStats().allocatedBytes;require(after>=before,"batch guard ledger regressed");batchAllocation_.guardMeasured=after-before;
    require(batchAllocation_.guardMeasured==batchAllocation_.guardPlannedExact&&batchAllocation_.guardMeasured<=Access::guardedBatchAllowance(),"batch guard actual !=exact owner-derived plan");
    // Define every otherwise indeterminate general/QSA/greedy/PLE/MoE scratch
    // byte before cross-process comparisons. Preserve the original explicit
    // A5 initialization and canaries of all216 batch.gdn.* lazy tape arenas.
    // Original and compact algorithms receive exactly the same finite seed.
    for(const auto &p:Access::batchPlanes(batch_))if(!p.label.starts_with("batch.gdn.")){
      if(p.buffer.storage()==metal::BufferStorage::Private){
        require(p.label=="batch.greedy_partials"&&p.buffer.sizeBytes()==31232,"unexpected Private scratch scope/length");
        const auto oldAllocated=backend_.memoryStats().allocatedBytes;
        auto backing=backend_.allocateBuffer(batchAllocation_.mirrorPlanned,metal::BufferStorage::Shared,"oracle Private greedy readback mirror");
        const auto newAllocated=backend_.memoryStats().allocatedBytes;
        require(newAllocated>=oldAllocated&&newAllocated-oldAllocated==batchAllocation_.mirrorPlanned&&backing.oracleChargedBytesSep22()==batchAllocation_.mirrorPlanned,"Private mirror actual !=exact admitted plan");
        batchAllocation_.mirrorMeasured+=newAllocated-oldAllocated;std::memset(backing.contents(),0x3a,backing.sizeBytes());
        mirrors_.push_back({p.label,p.buffer,backing,backend_.view(backing,0,p.buffer.sizeBytes())});
        debugCopy(mirrors_.back().view,p.buffer,p.buffer.sizeBytes());
      }else{require(p.buffer.contents(),"Shared scratch unexpectedly inaccessible before initialization");std::memset(p.buffer.contents(),0x3a,p.buffer.sizeBytes());}}
    require(mirrors_.size()==1&&batchAllocation_.mirrorMeasured==batchAllocation_.mirrorPlanned,"exactly one admitted Private diagnostic mirror required");
    for(const auto &p:Access::batchPlanes(batch_))batchBindings_.push_back({p.buffer.oracleOwnerIdentitySep22(),p.buffer.sizeBytes()});
    allocation_.beforeState=backend_.memoryStats().allocatedBytes;
    for(uint32_t lane=0;lane<4;++lane)fresh(lane,"initial.prefill."+std::to_string(lane));
    allocation_.afterState=backend_.memoryStats().allocatedBytes;
    require(allocation_.afterState>=allocation_.beforeState,"four-state ledger regressed");
    allocation_.stateDelta=allocation_.afterState-allocation_.beforeState;
    require(allocation_.stateDelta==batchAllocation_.stateAllowance,"four actual guarded states !=exact state category");allocation_.stateMeasured=true;
  }
  void run(){
    point("initial");
    verify({0,1},4,"fresh-r8");commit({0,1},{4,1},"fresh-r8");future({0,1},"fresh-r8.future",1);
    verify({0,1,2,3},4,"fresh-r16");commit({0,1,2,3},{4,3,2,1},"fresh-r16");
    verify({0,1},4,"same-r8");commit({0,1},{4,4},"same-r8");
    verify({0,1,2,3},4,"same-r16");commit({0,1,2,3},{4,4,4,4},"same-r16");
    verify({0,1,2,3},4,"drop4");
    invalidPending({0,1,2,3},"drop4.invalid");
    commit({0,1,2,3},{1,2,3,0},"drop4",3,false);
    verify({0,1,2},4,"drop3");commit({0,1,2},{4,2,0},"drop3",2,false);
    verify({0,1},4,"drop2");commit({0,1},{1,0},"drop2",1,true);
    verify({0},4,"fallback1");commit({0},{4},"fallback1");
    for(uint32_t i=1;i<4;++i)fresh(i,"reconstituted.prefill."+std::to_string(i));
    verify({0,1,2,3},4,"reconstituted4");
    // Destroyed lane0 becomes an expired weak ticket. Moving survivor1 keeps
    // its same heap implementation/owner identity, which must remain legal.
    auto moved=std::make_unique<FlashRequestState>(std::move(*slots_[1].state));slots_[1].state=std::move(moved);
    commit({0,1,2,3},{0,1,2,4},"mixed-terminal",0,true);
    future({1,2,3},"future.greedy",5);futureRows({1,2,3},4,"future.rows4");futureRows({1,2,3},8,"future.rows8");point("future.completed");
    fresh(0,"original-widths.prefill.0");
    for(uint32_t rows=1;rows<=3;++rows){
      const auto a="original-widths.lanes4.rows"+std::to_string(rows);verify({0,1,2,3},rows,a);commit({0,1,2,3},{rows,rows,rows,rows},a);
      const auto b="original-widths.lanes2.rows"+std::to_string(rows);verify({0,1},rows,b);commit({0,1},{rows,rows},b);
    }point("original-widths.completed");
    verify({0,1,2,3},4,"abort");
    // Abort is legal after a pending wrapper is destroyed and another moves.
    slots_[3].state.reset();slots_[3].guards.clear();slots_[3].planes.clear();
    auto movedAbort=std::make_unique<FlashRequestState>(std::move(*slots_[0].state));slots_[0].state=std::move(movedAbort);
    batch_.abortBatch();batch_.abortBatch();
    for(auto &slot:slots_)if(slot.state)require(slot.state->poisoned()&&!Access::pending(*slot.state)&&!target_.ownsState(*slot.state),"abort survivor not terminal");
    point("abort.completed");
    const std::array<uint32_t,1> one{prompt_[0]};
    rejected([&]{(void)target_.forward(*slots_[0].state,one);},"aborted scalar forward");++invalidChecks_;point("abort.reject-scalar");
    const std::array<FlashRequestState *,1> bad{slots_[0].state.get()};
    rejected([&]{(void)batch_.verifyBatch(bad,one,1);},"aborted batch verify");++invalidChecks_;point("abort.reject-batch");
    for(uint32_t i=0;i<4;++i)fresh(i,"recovery.prefill."+std::to_string(i));
    verify({0,1,2,3},4,"recovery");commit({0,1,2,3},{4,4,4,4},"recovery");future({0,1,2,3},"recovery.future",2);point("recovery.completed");
    for(const auto &label:selected_)require(std::find(selectedCompleted_.begin(),selectedCompleted_.end(),label)!=selectedCompleted_.end(),"selected checkpoint never completed: "+label);
    require(verifyCommands_==17&&commitCommands_==16&&invalidChecks_==13&&!batch_.pending(),"full required actual command/control campaign incomplete");
  }
  std::string summary()const{std::ostringstream o;o<<"{\"local_pointer_owner_redzone_checks\":"<<localChecks_<<",\"invalid_operation_checks\":"<<invalidChecks_
    <<",\"actual_verify_commands\":"<<verifyCommands_<<",\"actual_commit_commands\":"<<commitCommands_<<",\"selected_full_checkpoints\":[";
    for(size_t i=0;i<selectedCompleted_.size();++i){if(i)o<<',';o<<json::quote(selectedCompleted_[i]);}
    o<<"],\"source_replacement_guard_runtime_proved\":true,\"foreign_trunk_rejection_runtime_proved\":false,\"worker_deadline_callback_proved\":false}";return o.str();}
private:
  metal::MetalBackend &backend_;engine::MemoryGovernor &governor_;FlashForward &target_;FlashBatchVerify &batch_;Store &store_;
  AllocationBreakdown &allocation_;BatchAllocation &batchAllocation_;std::span<const uint32_t> prompt_;std::vector<std::string> selected_,selectedCompleted_;
  std::array<StateSlot,4> slots_;std::vector<Access::Guard> batchGuards_;std::vector<std::pair<uintptr_t,uint64_t>> batchBindings_;
  struct PrivateMirror {std::string label;metal::MetalBuffer original,backing,view;};std::vector<PrivateMirror> mirrors_;
  std::vector<uint32_t> pendingLanes_,pendingCorrections_;uint32_t pendingRows_=0;
  FlashBatchVerifyResult pendingResult_;
  std::string pendingOutputLabel_;
  uint64_t localChecks_=0,invalidChecks_=0,verifyCommands_=0,commitCommands_=0,extraAllowance_=0;
  std::vector<FlashRequestState *> ptrs(const std::vector<uint32_t> &lanes){std::vector<FlashRequestState *> p;for(auto lane:lanes){require(lane<4&&slots_[lane].state,"missing active state");p.push_back(slots_[lane].state.get());}return p;}
  void ledger(){const auto now=backend_.memoryStats().allocatedBytes;require(now>=allocation_.mappedInitial,"backend ledger regressed");
    batchAllocation_.maximumLiveDelta=std::max(batchAllocation_.maximumLiveDelta,now-allocation_.mappedInitial);
    require(now-allocation_.mappedInitial<=allocation_.reservation+extraAllowance_,"live backend delta exceeds original categories/guard reservations");allocation_.finalAllocated=now;}
  void checks(){ledger();require(Access::batchCanaries(batch_),"batch lazy canary changed");checkGuards(batchGuards_);
    std::vector<std::pair<uintptr_t,uint64_t>> bindings;for(const auto &p:Access::batchPlanes(batch_))bindings.push_back({p.buffer.oracleOwnerIdentitySep22(),p.buffer.sizeBytes()});require(bindings==batchBindings_,"batch scratch/tape owner extent changed");
    for(const auto &slot:slots_)if(slot.state){require(Access::binding(*slot.state)==slot.binding,"request owner implementation/identity changed");checkGuards(slot.guards);
      std::vector<std::pair<uintptr_t,uint64_t>> planes;for(const auto &p:Access::planes(*slot.state))planes.push_back({reinterpret_cast<uintptr_t>(p.buffer.contents()),p.buffer.sizeBytes()});require(planes==slot.planes,"request plane owner extent changed");}
    ++localChecks_;}
  std::string metadata()const{std::ostringstream o;o<<"{\"batch\":"<<Access::batchMetadata(batch_)<<",\"states\":[";
    for(uint32_t i=0;i<4;++i){if(i)o<<',';if(slots_[i].state)o<<Access::metadata(target_,*slots_[i].state);else o<<"null";}o<<"],\"Private_planes\":[";
    for(size_t i=0;i<mirrors_.size();++i){if(i)o<<',';const auto &m=mirrors_[i];o<<"{\"label\":"<<json::quote(m.label)<<",\"original_storage\":\"Private\",\"original_length\":"<<m.original.sizeBytes()
      <<",\"original_owner_charge\":"<<m.original.oracleChargedBytesSep22()<<",\"mirror_readable_length\":"<<m.view.sizeBytes()<<",\"mirror_owner_charge\":"<<m.backing.oracleChargedBytesSep22()<<'}';}o<<"]}";return o.str();}
  void debugCopy(const metal::MetalBuffer &from,const metal::MetalBuffer &to,uint64_t bytes){
    require(bytes&&bytes%4==0&&from.sizeBytes()>=bytes&&to.sizeBytes()>=bytes&&from.oracleOwnerIdentitySep22()!=to.oracleOwnerIdentitySep22(),"diagnostic copy extent/owner alias invalid");
    metal::CommandGraph graph;const uint64_t words=bytes/4;
    graph.add("flash_forward_copy_words",{from,to},FlashForwardCopyParams{words},{(words+255)/256,1,1});
    auto ticket=backend_.submitCommandAsync(graph.dispatches());require(bool(ticket),"diagnostic copy ticket absent");(void)ticket.wait();
    require(ticket.ready()&&backend_.healthy(),"diagnostic copy did not complete on healthy backend");++batchAllocation_.debugCopyCommands;batchAllocation_.debugCopyBytes+=bytes;
  }
  std::string stateWitness()const{Hash hash;for(const auto &slot:slots_)if(slot.state)for(const auto &p:Access::planes(*slot.state))hash.add(p.buffer.contents(),p.buffer.sizeBytes());return hash.finish();}
  void point(const std::string &label){checks();std::vector<View> moe;
    for(const auto &p:Access::batchMoEPlanes(batch_))moe.push_back({p.label,p.type,static_cast<const uint8_t *>(p.buffer.contents()),p.live,p.buffer.sizeBytes()});
    store_.frame(label+".max16-arena",metadata(),moe);
    if(std::find(selected_.begin(),selected_.end(),label)==selected_.end())return;
    for(const auto &m:mirrors_)debugCopy(m.original,m.view,m.original.sizeBytes());
    std::vector<View> all;for(uint32_t lane=0;lane<4;++lane)if(slots_[lane].state)for(const auto &p:Access::planes(*slots_[lane].state))
      all.push_back({"lane."+std::to_string(lane)+"."+p.label,p.type,static_cast<const uint8_t *>(p.buffer.contents()),p.live,p.buffer.sizeBytes()});
    for(const auto &p:Access::batchPlanes(batch_)){
      const uint8_t *data=static_cast<const uint8_t *>(p.buffer.contents());
      if(p.buffer.storage()==metal::BufferStorage::Private){const auto it=std::find_if(mirrors_.begin(),mirrors_.end(),[&](const auto &m){return m.label==p.label&&m.original.sameView(p.buffer);});
        require(it!=mirrors_.end()&&it->view.sizeBytes()==p.buffer.sizeBytes(),"Private plane lacks exact original-view readback mirror");data=static_cast<const uint8_t *>(it->view.contents());}
      require(data,"checkpoint plane inaccessible after completed readback");all.push_back({p.label,p.type,data,p.live,p.buffer.sizeBytes()});}
    store_.frame(label+".whole-state-and-batch-tapes",metadata(),all);selectedCompleted_.push_back(label);
  }
  void fresh(uint32_t lane,const std::string &label){require(!batch_.pending(),"fresh owner while batch pending");
    auto &slot=slots_[lane];slot.guards.clear();slot.state.reset();slot.planes.clear();
    const auto before=backend_.memoryStats().allocatedBytes;slot.state=std::make_unique<FlashRequestState>(target_.createState());slot.guards=Access::guardState(backend_,*slot.state);
    const auto after=backend_.memoryStats().allocatedBytes;require(after>=before,"state ledger regressed");const auto delta=after-before;
    const auto allowance=FlashForward::requestStateBytes(kCapacity)+Access::guardedStateAllowance();require(delta==allowance,"guarded request actual delta !=exact original category allowance");
    batchAllocation_.maximumStateMeasured=std::max(batchAllocation_.maximumStateMeasured,delta);slot.binding=Access::binding(*slot.state);
    for(const auto &p:Access::planes(*slot.state))slot.planes.push_back({reinterpret_cast<uintptr_t>(p.buffer.contents()),p.buffer.sizeBytes()});
    const auto r=target_.forward(*slot.state,prompt_,false,true);require(r.logicalLength==kRows&&target_.ownsState(*slot.state),"canonical 2K prefill state differs");
    store_.output(label+".output",r,kRows);slot.anchor=greedy(r,0);point(label);
  }
  void output(const std::string &label,const FlashBatchVerifyResult &r,bool repeat=false){
    require(r.capacity==kCapacity&&r.lanes>0&&r.lanes<=4&&r.rows>0&&r.rows<=4&&r.logitsBF16&&r.hiddenBF16,"batch output geometry invalid");
    const uint32_t count=r.lanes*r.rows;require(r.greedyRows==count&&r.greedyResultsU32,"actual all batch greedy records required");std::vector<View> v;
    const auto add=[&](const char *name,Type type,const metal::MetalBuffer &b,uint64_t bytes){require(b&&b.contents()&&b.sizeBytes()==bytes,"batch borrowed output extent invalid");v.push_back({name,type,static_cast<const uint8_t *>(b.contents()),bytes,bytes});};
    add("hidden",Type::BF16,r.hiddenBF16,uint64_t{count}*kHyper*2);add("logits",Type::BF16,r.logitsBF16,uint64_t{count}*kVocabulary*2);
    add("greedy",Type::U32,r.greedyResultsU32,uint64_t{count}*sizeof(FlashGreedyGPURowResult));
    for(uint32_t i=0;i<count;++i)(void)greedyGPUResultToken(static_cast<const FlashGreedyGPURowResult *>(r.greedyResultsU32.contents())[i],kVocabulary);
    std::ostringstream meta;meta<<"{\"lanes\":"<<r.lanes<<",\"rows\":"<<r.rows<<",\"logical_lengths\":[";for(size_t i=0;i<r.logicalLengths.size();++i){if(i)meta<<',';meta<<r.logicalLengths[i];}meta<<"]}";
    store_.frame(label,meta.str(),v,repeat);
  }
  void verify(std::vector<uint32_t> lanes,uint32_t rows,const std::string &label){require(!batch_.pending(),"unresolved trial before verify");auto p=ptrs(lanes);std::vector<uint32_t> input;
    for(auto lane:lanes){input.push_back(slots_[lane].anchor);for(uint32_t row=1;row<rows;++row)input.push_back(prompt_[(lane*31+row*17+verifyCommands_*13)%prompt_.size()]);}
    const auto *store=target_.batchInt8ExpertStore();require(store,"Full512 store missing");const auto before=store->compactNativeBatchVerifyCounters();
    const auto r=batch_.verifyBatch(p,input,rows);pendingResult_=r;pendingOutputLabel_=label+".output";++verifyCommands_;require(batch_.pending()&&r.lanes==lanes.size()&&r.rows==rows,"actual pending verify geometry differs");output(pendingOutputLabel_,r);
    const auto after=store->compactNativeBatchVerifyCounters();const bool eligible=rows==4&&(lanes.size()==2||lanes.size()==4);
    const auto gate=[&](const auto &a,const auto &b,uint32_t physical,bool wanted){const uint64_t calls=wanted?48:0,physicalRows=calls*physical;
      require(b.planCalls==a.planCalls+calls&&b.gateCalls==a.gateCalls+calls&&b.downCalls==a.downCalls+calls,"actual compact per-width graph construction call delta differs");
      require(b.planRows==a.planRows+physicalRows&&b.gateRows==a.gateRows+physicalRows&&b.downRows==a.downRows+physicalRows,"actual compact per-width physical row delta differs");};
    gate(before.r8,after.r8,8,bool(SPLASH_VERIFY_CANDIDATE)&&eligible&&lanes.size()==2);gate(before.r16,after.r16,16,bool(SPLASH_VERIFY_CANDIDATE)&&eligible&&lanes.size()==4);
    pendingLanes_=std::move(lanes);pendingRows_=rows;pendingCorrections_.clear();const auto *records=static_cast<const FlashGreedyGPURowResult *>(r.greedyResultsU32.contents());
    for(uint32_t i=0;i<r.lanes*rows;++i)pendingCorrections_.push_back(greedyGPUResultToken(records[i],kVocabulary));
    point(label+".pending");
  }
  void commit(std::vector<uint32_t> lanes,std::vector<uint32_t> retained,const std::string &label,int terminal=-1,bool destroy=false){
    require(lanes==pendingLanes_&&lanes.size()==retained.size(),"commit local cohort differs");auto p=ptrs(lanes);std::vector<uint64_t> begins;
    for(auto lane:lanes)begins.push_back(slots_[lane].state->logicalLength()-pendingRows_);
    if(terminal>=0){const auto at=std::find(lanes.begin(),lanes.end(),uint32_t(terminal));require(at!=lanes.end()&&retained[size_t(at-lanes.begin())]==0,"terminal lane is not retained0");
      const auto index=size_t(at-lanes.begin());p[index]=nullptr;if(destroy){slots_[terminal].state.reset();slots_[terminal].guards.clear();slots_[terminal].planes.clear();
        const auto before=metadata(),beforeState=stateWitness();auto illegal=retained;illegal[index]=1;
        rejected([&]{(void)batch_.commitBatch(p,illegal);},"expired lane with nonzero retained");++invalidChecks_;
        require(batch_.pending()&&metadata()==before&&stateWitness()==beforeState,"expired lane rejection changed pending ownership/state");
        output(pendingOutputLabel_,pendingResult_,true);point(label+".reject-expired-nonzero");}}
    const auto timing=batch_.commitBatch(p,retained);++commitCommands_;require(!batch_.pending(),"commit did not release pending batch");
    bool needsReplay=false;for(size_t i=0;i<lanes.size();++i){auto &slot=slots_[lanes[i]];if(!retained[i]){if(slot.state)require(slot.state->poisoned()&&!Access::pending(*slot.state),"terminal live-null lane not poisoned");continue;}
      needsReplay|=retained[i]<pendingRows_;require(slot.state&&slot.state->logicalLength()==begins[i]+retained[i]&&!slot.state->poisoned()&&target_.ownsState(*slot.state),"survivor mixed prefix metadata differs");
      slot.anchor=pendingCorrections_[i*pendingRows_+retained[i]-1];}
    if(!needsReplay)require(timing.gpuSeconds==0&&timing.wallSeconds==0,"full/terminal-only commit submitted restore work");
    pendingLanes_.clear();pendingCorrections_.clear();pendingRows_=0;point(label+".committed");
  }
  void future(const std::vector<uint32_t> &lanes,const std::string &label,uint32_t steps){for(uint32_t step=0;step<steps;++step)for(auto lane:lanes){auto &slot=slots_[lane];const uint32_t token=slot.anchor;
    const auto before=slot.state->logicalLength();const auto r=target_.forward(*slot.state,std::span<const uint32_t>(&token,1),true,true);
    require(r.logicalLength==before+1&&target_.ownsState(*slot.state),"future greedy ordinary state differs");const auto tag=label+".step"+std::to_string(step)+".lane"+std::to_string(lane);
    store_.output(tag+".output",r,1);slot.anchor=greedy(r,0);point(tag);}}
  void futureRows(const std::vector<uint32_t> &lanes,uint32_t rows,const std::string &label){for(auto lane:lanes){auto &slot=slots_[lane];std::vector<uint32_t> input(prompt_.begin()+lane,prompt_.begin()+lane+rows);input[0]=slot.anchor;
    const auto before=slot.state->logicalLength();const auto r=target_.forward(*slot.state,input,true,true);require(r.logicalLength==before+rows&&r.logitRows==rows,"future replacement row geometry differs");
    const auto tag=label+".lane"+std::to_string(lane);store_.output(tag+".output",r,rows);slot.anchor=greedy(r,rows-1);point(tag);}}
  void invalidPending(std::vector<uint32_t> lanes,const std::string &label){auto p=ptrs(lanes);const auto input=std::span<const uint32_t>(prompt_).first(4);const auto before=metadata(),beforeState=stateWitness();
    const auto reject=[&](auto action,const std::string &name){rejected(action,name.c_str());++invalidChecks_;require(metadata()==before&&stateWitness()==beforeState,"rejected pending action changed exact metadata/state");
      output("drop4.output",pendingResult_,true);point(label+"."+name);};
    reject([&]{(void)batch_.verifyBatch(std::span<FlashRequestState *const>(p).first(1),input.first(1),1);},"verify-while-pending");
    reject([&]{(void)target_.forward(*p[0],input.first(1));},"scalar-while-batch-pending");
    auto reversed=p;std::reverse(reversed.begin(),reversed.end());const std::array<uint32_t,4> full{4,4,4,4};
    reject([&]{(void)batch_.commitBatch(reversed,full);},"reordered-owner");
    auto duplicate=p;duplicate[1]=duplicate[0];reject([&]{(void)batch_.commitBatch(duplicate,full);},"duplicate-owner");
    auto null=p;null[3]=nullptr;reject([&]{(void)batch_.commitBatch(null,full);},"nonzero-null");
    const std::array<uint32_t,4> excessive{1,2,3,5};reject([&]{(void)batch_.commitBatch(p,excessive);},"excessive-prefix");
    // Moving the source wrapper allocates no second graph/coefficient cache.
    // The arena keeps its original source-object reference and rejects that
    // temporarily empty object before any GPU work or pending-ticket mutation.
    {FlashForward held(std::move(target_));
      rejected([&]{(void)batch_.verifyBatch(std::span<FlashRequestState *const>(p).first(1),input.first(1),1);},"moved source verify");++invalidChecks_;
      rejected([&]{(void)batch_.commitBatch(p,full);},"moved source commit");++invalidChecks_;
      target_=std::move(held);}
    require(metadata()==before&&stateWitness()==beforeState,"source wrapper replacement rejection mutated original pending state");
    output("drop4.output",pendingResult_,true);point(label+".moved-source-rejected");
    const auto allowance=FlashForward::requestStateBytes(kCapacity)+Access::guardedStateAllowance();auto reservation=governor_.tryReserve(allowance);require(bool(reservation),"governor denied wrong-cohort sibling probe");
    extraAllowance_=allowance;batchAllocation_.siblingAllowance=allowance;
    {auto sibling=target_.createState();auto guards=Access::guardState(backend_,sibling);auto substitute=p;substitute[3]=&sibling;
      reject([&]{(void)batch_.commitBatch(substitute,full);},"wrong-cohort-sibling");checkGuards(guards);require(sibling.logicalLength()==0&&!sibling.poisoned()&&target_.ownsState(sibling),"wrong cohort rejection changed sibling");}
    // The temporary sibling is gone before releasing its exact reservation.
    extraAllowance_=0;ledger();
  }
};
}

int main(int argc,char **argv){@autoreleasepool{
  bool backendCreated=false,backendDestroyed=false;std::string stage="arguments";fs::path report;AllocationBreakdown allocation;BatchAllocation batchAllocation;FailureProgress progress;
  try{
    if(argc>=7)report=argv[6];if(argc==2&&std::string_view(argv[1])=="--cpu-only"){cpuBatch();return 0;}
    require(argc==9&&std::string_view(argv[1])=="--gpu","oracle --gpu export|compare METALLIB PACKAGE TOKENS FRESH_REPORT SPILL CHECKPOINTS");
    const bool exporting=std::string_view(argv[2])=="export";require(exporting||std::string_view(argv[2])=="compare","role invalid");require(exporting!=bool(SPLASH_VERIFY_CANDIDATE),"binary role mismatch");
    require(!fs::exists(report)&&!fs::exists(report.string()+".partial"),"fresh report required");early();
    require(selected("SPLASH_FLASH_COMPACT_NATIVE_BATCH_VERIFY_SEP22")==bool(SPLASH_VERIFY_CANDIDATE),"batch compact role mismatch");
    progress.role=exporting?"export":"compare";const auto selectedPoints=selectors(argv[8]);const auto prompt=tokens(argv[5]);
    const auto provenance=dictionary(kPrefillExactProvenancePath);require(fileHash(argv[3])==string(provenance[@"metallib_sha256"]),"worker library drift");
    Store store(argv[7],exporting,progress);std::string common,own;
    {
      BackendLifetime lifetime{backendCreated,backendDestroyed};stage="backend";metal::MetalBackend backend(argv[3]);backendCreated=true;
      stage="weights";const auto weights=FlashWeights::load(backend,argv[4]);const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);
      engine::MemoryGovernor governor(backend,physical-reserve,reserve);allocation.mappedInitial=backend.memoryStats().allocatedBytes;
      allocation.fixed=FlashForward::workspacePlannedBytes(kCapacity,kRows,kVerifyRows);allocation.experts=FlashForward::expertCachePlannedBytes(weights);
      allocation.f32=FlashForward::floatDenseCachePlannedBytes(weights);allocation.head=FlashForward::int8HeadPlannedBytes(weights);
      if(selected("SPLASH_FLASH_DENSE_CACHE"))allocation.bf16=FlashDenseCache::plannedBytes(weights,FlashDenseCache::defaultPrefixes(weights,true));
      if(selected("SPLASH_FLASH_BLOCKED_MOE"))allocation.blocked=flashMoEBlockedWorkspacePlannedBytes(kRows,10);
      if(dense_w8a8_sep21::requiresCache(kRows))allocation.w8Cache=dense_w8a8_sep21::Cache::plannedBytes();
      allocation.plannedTarget=allocation.fixed+allocation.experts+allocation.f32+allocation.head+allocation.bf16+allocation.blocked+allocation.w8Cache;
      batchAllocation.planned=FlashBatchVerify::workspacePlannedBytes(kCapacity,4,4);batchAllocation.stateAllowance=4*(FlashForward::requestStateBytes(kCapacity)+Access::guardedStateAllowance());
      allocation.state=batchAllocation.stateAllowance;allocation.reservation=allocation.plannedTarget+allocation.state+batchAllocation.planned+batchAllocation.guardAllowance+batchAllocation.mirrorPlanned+allocation.margin;
      auto admission=governor.tryReserve(allocation.reservation);require(bool(admission),"governor denied original categories/four states/max16 batch/guards admission");
      stage="target";FlashForward target(backend,weights,kCapacity,kRows,kVerifyRows);allocation.workspace=target.workspaceBytes();allocation.afterTarget=backend.memoryStats().allocatedBytes;
      require(allocation.afterTarget>=allocation.mappedInitial,"target ledger regressed");allocation.targetDelta=allocation.afterTarget-allocation.mappedInitial;allocation.targetMeasured=true;
      require(allocation.targetDelta==allocation.workspace,"target actual delta !=workspace ledger");require(allocation.targetDelta<=allocation.plannedTarget,"target actual delta exceeds category plan");
      stage="batch";const auto beforeBatch=backend.memoryStats().allocatedBytes;FlashBatchVerify batch(backend,weights,target,kCapacity,4,4);const auto afterBatch=backend.memoryStats().allocatedBytes;
      require(afterBatch>=beforeBatch,"batch ledger regressed");batchAllocation.measured=afterBatch-beforeBatch;require(batchAllocation.measured==batch.workspaceBytes(),"batch actual delta !=workspace ledger");
      require(batchAllocation.measured<=batchAllocation.planned,"batch actual delta exceeds exact original plan");admission->commit();
      const auto commonFlags=commonPolicy();std::ostringstream c;c<<"{\"scope\":\"actual max16 BatchVerify same-owner transition/state/future; no trained head or service\",\"model_source\":"<<json::quote(weights.sourceIdentity())
        <<",\"model_layout\":"<<json::quote(weights.manifestFingerprint())<<",\"full512_store\":"<<json::quote(target.batchInt8ExpertStore()->identitySha256())
        <<",\"numeric_flags\":"<<commonFlags<<",\"capacity\":"<<kCapacity<<",\"prefill_rows\":"<<kRows<<",\"selected_checkpoints\":[";
      for(size_t i=0;i<selectedPoints.size();++i){if(i)c<<',';c<<json::quote(selectedPoints[i]);}c<<"]}";common=c.str();
      if(!exporting){const auto prior=dictionary(fs::path(argv[7])/"complete.json");NSError *error=nil;id parsed=[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:common.data() length:common.size()] options:0 error:&error];
        require(error==nil&&[prior[@"common"] isEqual:parsed],"control model/numeric flags/selected replay checkpoints differ before execution");}
      stage="campaign";Campaign campaign(backend,governor,target,batch,store,allocation,batchAllocation,prompt,selectedPoints);campaign.run();
      allocation.governor=governor.snapshot();allocation.governorStage="completed_actual_campaign";
      require(allocation.governor.deniedReservations==0&&allocation.governor.hostMeasurementValid&&allocation.governor.growthAllowed,"actual governor final measurement/denial guard failed");
      std::ostringstream o;o<<"{\"worker\":"<<kPrefillExactBuildProvenance<<",\"kernel_routes\":"<<json::quote(target.kernelRoutes())<<",\"checks\":"<<campaign.summary()<<'}';own=o.str();
    }
    backendDestroyed=true;store.complete(common,own);std::ostringstream out;out<<"{\"schema\":\"batchverify-two-process-maxarena-exact-v1\",\"pass\":true,\"partition_complete\":true,\"whole_campaign_all_partitions_qualified\":false"
      <<",\"role\":"<<json::quote(exporting?"export":"compare")<<",\"backend_destroyed\":true,\"trained_head_proved\":false,\"worker_cancel_deadline_proved\":false,\"foreign_trunk_runtime_proved\":false"
      <<",\"frames\":"<<store.completed()<<",\"spill_bytes\":"<<store.spilled()<<",\"bytes_compared\":"<<store.compared()<<",\"planes_compared\":"<<store.checked()
      <<",\"allocation\":"<<allocation.json()<<",\"batch_allocation\":"<<batchAllocation.json()<<",\"common\":"<<common<<",\"producer\":"<<own<<"}\n";
    Store::publish(report,out.str());std::cout<<out.str();return 0;
  }catch(const std::exception &error){std::ostringstream out;out<<"{\"pass\":false,\"partition_complete\":false,\"stage\":"<<json::quote(stage)<<",\"error\":"<<json::quote(error.what())
    <<",\"backend_created\":"<<(backendCreated?"true":"false")<<",\"backend_destroyed\":"<<(backendDestroyed?"true":"false")<<",\"progress\":"<<progress.json()
    <<",\"allocation\":"<<allocation.json()<<",\"batch_allocation\":"<<batchAllocation.json()<<"}\n";std::cerr<<out.str();if(!report.empty())try{Store::publish(report.string()+".failure.json",out.str());}catch(...){}return 1;}
}}
