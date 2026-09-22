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
#include "TrunkVerifyBuildProvenance.hpp"
#include "inspect.hpp"
#include "dev/benchmarks/hc_pad_verify_worker_sep22/worker_bridge.hpp"

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
  require(meta.size()<=65536 && !planes.empty() && planes.size()<=256,"frame declaration invalid");
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
    else{const auto manifest=dictionary(spill_/"complete.json");require([manifest[@"complete"] isEqual:@YES]&&string(manifest[@"schema"])=="trunkverify-export-v1","completed export required");
      for(id item:static_cast<NSArray *>(manifest[@"frames"])){const auto entry=static_cast<NSDictionary *>(item);frames_.push_back({string(entry[@"label"]),string(entry[@"sha256"]),number(entry[@"bytes"]),number(entry[@"planes"]),number(entry[@"live_bytes"])});}
      require(frames_.size()==26,"all26 expected frames required");}}
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
    require(completed_==26&&repeated_==54,"all verifier campaign frames/replays required");
    if(exporting_){std::ostringstream out;out<<"{\"schema\":\"trunkverify-export-v1\",\"complete\":true,\"common\":"<<common<<",\"producer\":"<<different<<",\"spill_bytes\":"<<spilled_<<",\"frames\":[";
      for(size_t i=0;i<frames_.size();++i){if(i)out<<',';const auto &f=frames_[i];out<<"{\"label\":"<<json::quote(f.label)<<",\"sha256\":"<<json::quote(f.sha)<<",\"bytes\":"<<f.bytes<<",\"planes\":"<<f.planes<<",\"live_bytes\":"<<f.live<<'}';}out<<"]}\n";publish(spill_/"complete.json",out.str());}}
  static void publish(const fs::path &path,const std::string &text){require(!fs::exists(path)&&!fs::exists(path.string()+".writing"),"fresh report/declaration path required");std::ofstream out(path.string()+".writing");out<<text;out.close();require(bool(out),"JSON publish failed");fs::rename(path.string()+".writing",path);}
  uint64_t completed()const{return completed_;}uint64_t repeated()const{return repeated_;}uint64_t spilled()const{return spilled_;}uint64_t compared()const{return compared_;}uint64_t checked()const{return checked_;}
private:fs::path spill_;bool exporting_;std::vector<Frame> frames_;uint64_t spilled_=0,compared_=0,checked_=0,completed_=0,repeated_=0;FailureProgress &progress_;
};
bool selected(const char *name){const char *v=std::getenv(name);require(v&&(*v=='0'||*v=='1')&&!v[1],std::string("strict explicit flag required: ")+name);return *v=='1';}
std::string commonPolicy(){
  constexpr std::array names{"SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21","SPLASH_FLASH_ALLROWS_GATHERED_MPP","SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS","SPLASH_FLASH_BLOCKED_MOE","SPLASH_FLASH_DENSE_CACHE","SPLASH_FLASH_DENSE_M64_OUT","SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21","SPLASH_FLASH_EXPERT_QMV","SPLASH_FLASH_FLOAT_DENSE_CACHE","SPLASH_FLASH_FLOAT_DENSE_SELECTIVE","SPLASH_FLASH_FUSE_GDN","SPLASH_FLASH_FUSE_HC","SPLASH_FLASH_GDN_LAZY_ROLLBACK","SPLASH_FLASH_GDN_PREFILL_FMA_SEP21","SPLASH_FLASH_GDN_STAGED","SPLASH_FLASH_GPU_GREEDY","SPLASH_FLASH_HC_UP_F32_MPP","SPLASH_FLASH_INT8_HEAD","SPLASH_FLASH_MOE_DIRECT_A","SPLASH_FLASH_MOE_M64","SPLASH_FLASH_MOE_POINTWISE_SEP21","SPLASH_FLASH_MOE_Q4X8","SPLASH_FLASH_PLE_LOOKUP_FUSED","SPLASH_FLASH_PLE_POST_FUSED","SPLASH_FLASH_PLE_SSD_STREAMING","SPLASH_FLASH_PREFILL_DENSE_TILES","SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21","SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT","SPLASH_FLASH_QMV_F32","SPLASH_FLASH_QSA_BULK_PREFILL","SPLASH_FLASH_QSA_BULK_PREFILL_SG8","SPLASH_FLASH_QSA_F32","SPLASH_FLASH_QSA_MPP","SPLASH_FLASH_QSA_OUT_F32_N32","SPLASH_FLASH_QSA_ROW_TILES","SPLASH_FLASH_SHARED_EXPERT_FUSED","SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21","SPLASH_FLASH_GDN_AB_MERGE_SEP21"};
  std::ostringstream out;out<<'{';for(size_t i=0;i<names.size();++i){const char *v=std::getenv(names[i]);require(v,std::string("common flag absent: ")+names[i]);if(i)out<<',';out<<json::quote(names[i])<<':'<<json::quote(v);}out<<'}';return out.str();
}
void early(){
  require(selected("SPLASH_FLASH_ALLROWS_FULL512_TARGET"),"pure I8 parent/candidate required");
  require(!selected("SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21"),"phase-Q4 profile excluded");
  require(selected("SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22"),"registered composite requires COMPACT1");
  require(selected("SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22"),"registered composite requires HC1");
  require(selected("SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22"),"registered composite requires BUNDLE1");
  require(selected("SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22"),"registered composite requires explicit registration flag1");
  for(const char *n:{"SPLASH_FLASH_BATCH","SPLASH_FLASH_BATCH_MTP","SPLASH_FLASH_BATCH_MTP_PREFILL","SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_MTP","SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT","SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"})require(!selected(n),std::string("excluded worker route: ")+n);
  require(selected("SPLASH_FLASH_GDN_LAZY_ROLLBACK")&&selected("SPLASH_FLASH_GPU_GREEDY"),"guarded lazy tapes/greedy required");
  (void)commonPolicy();
}
std::vector<View> tapeViews(const FlashForward &target){std::vector<View> out;for(const auto &p:Access::tapePlanes(target))out.push_back({p.label,p.type,static_cast<const uint8_t *>(p.buffer.contents()),p.live,p.buffer.sizeBytes()});return out;}
constexpr const char *kTapeMeta="{\"source\":\"saved VerifyR4 operands\",\"maximum_rows\":4,\"physical_lazy_arenas\":216,\"PLE_scope\":\"defined ranges only; undefined tails locally unchanged\"}";
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
void cpu(){
  const uint64_t bound=15*FlashForward::requestStateBytes(kCapacity)+(200ULL<<20)+(128ULL<<20);
  require(bound<kLimit,"campaign spill bound exceeds4GiB");require(FlashForward::requestStateBytes(kCapacity)==232603648ULL,"physical state formula changed");
  require(dense_w8a8_sep21::Cache::plannedBytes()==1890975744ULL,"W8 coefficient admission changed");
  require(Access::guardedStateAllowance()==2195456ULL,"redzone backing allowance differs");
  std::vector<uint8_t> bytes(1024,0x5a);View p{"synthetic",Type::U32,bytes.data(),bytes.size(),bytes.size()};
  const auto path=fs::temp_directory_path()/("trunkverify-cpu-"+std::to_string(NSProcessInfo.processInfo.processIdentifier));require(!fs::exists(path),"fresh CPU path required");
  const auto n=extent("{}",{p});Stream w(path,true,n);serialize(w,"{}",{p});const auto sha=w.finish();Stream r(path,false,n);serialize(r,"{}",{p});require(r.finish()==sha,"stream parity failed");fs::remove(path);
  std::cout<<"{\"pass\":true,\"gpu_executed\":false,\"model_payload_bytes_read\":0,\"spill_upper_bound_bytes\":"<<bound<<",\"unique_frames\":26,\"state_frames\":15,\"request_physical_planes\":134,\"tape_initialized_physical_arenas\":216}\n";
}
}
int main(int argc,char **argv){@autoreleasepool{
  bool backendCreated=false,backendDestroyed=false;std::string stage="arguments";fs::path report;AllocationBreakdown allocation;FailureProgress progress;
  try{
    if(argc>=7)report=argv[6];if(argc==2&&std::string_view(argv[1])=="--cpu-only"){cpu();return 0;}
    require(argc==8&&std::string_view(argv[1])=="--gpu","oracle --gpu export|compare METALLIB PACKAGE TOKENS FRESH_REPORT SPILL");
    const bool exporting=std::string_view(argv[2])=="export";require(exporting||std::string_view(argv[2])=="compare","role invalid");require(exporting!=bool(SPLASH_VERIFY_CANDIDATE),"binary role mismatch");require(!exporting,"registered composite comparator cannot export");
    require(!fs::exists(report)&&!fs::exists(report.string()+".partial"),"fresh report required");early();progress.role=exporting?"export":"compare";
    const auto prompt=tokens(argv[5]);const auto provenance=dictionary(kPrefillExactProvenancePath);
    require(fileHash(argv[3])==string(provenance[@"metallib_sha256"]),"worker library drift");
    Store store(argv[7],exporting,progress);stage="backend";uint64_t workspace=0,planned=0;std::vector<uint32_t> verification;
    std::array<uint32_t,4> correction{};std::string common,own;
    {
      BackendLifetime lifetime{backendCreated,backendDestroyed};metal::MetalBackend backend(argv[3]);backendCreated=true;const auto weights=FlashWeights::load(backend,argv[4]);
      const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      allocation.mappedInitial=backend.memoryStats().allocatedBytes;allocation.fixed=FlashForward::workspacePlannedBytes(kCapacity,kRows,kVerifyRows);
      allocation.experts=FlashForward::expertCachePlannedBytes(weights);allocation.f32=FlashForward::floatDenseCachePlannedBytes(weights);allocation.head=FlashForward::int8HeadPlannedBytes(weights);
      if(selected("SPLASH_FLASH_DENSE_CACHE"))allocation.bf16=FlashDenseCache::plannedBytes(weights,FlashDenseCache::defaultPrefixes(weights,true));
      if(selected("SPLASH_FLASH_BLOCKED_MOE"))allocation.blocked=flashMoEBlockedWorkspacePlannedBytes(kRows,10);
      if(dense_w8a8_sep21::requiresCache(kRows))allocation.w8Cache=dense_w8a8_sep21::Cache::plannedBytes();
      allocation.state=FlashForward::requestStateBytes(kCapacity)+Access::guardedStateAllowance();allocation.plannedTarget=allocation.fixed+allocation.experts+allocation.f32+allocation.head+allocation.bf16+allocation.blocked+allocation.w8Cache;
      planned=allocation.reservation=allocation.plannedTarget+allocation.state+allocation.margin;auto admission=governor.tryReserve(planned);require(bool(admission),"governor denied target/state admission");
      stage="target";FlashForward target(backend,weights,kCapacity,kRows,kVerifyRows);workspace=target.workspaceBytes();allocation.workspace=workspace;allocation.afterTarget=backend.memoryStats().allocatedBytes;
      require(allocation.afterTarget>=allocation.mappedInitial,"target ledger regressed");allocation.targetDelta=allocation.afterTarget-allocation.mappedInitial;allocation.targetMeasured=true;
      require(allocation.targetDelta==workspace,"target ledger !=workspace");require(allocation.targetDelta<=allocation.plannedTarget,"target exceeds categories");require(workspace+allocation.state<=planned,"target/state exceeds reservation");admission->commit();
      if(!exporting){const auto prior=dictionary(fs::path(argv[7])/"complete.json");const auto priorCommon=static_cast<NSDictionary *>(prior[@"common"]);
        require(target.batchInt8ExpertStore()&&string(priorCommon[@"full512_store"])==target.batchInt8ExpertStore()->identitySha256(),"Full512 store differs before trunk");
        require(string(priorCommon[@"model_source"])==weights.sourceIdentity()&&string(priorCommon[@"model_layout"])==weights.manifestFingerprint(),"source/layout differs before trunk");
        const auto flags=commonPolicy();NSError *error=nil;id parsed=[NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:flags.data() length:flags.size()] options:0 error:&error];require(error==nil&&[priorCommon[@"prefill_flags"] isEqual:parsed],"Prefill flags differ before trunk");
        const auto inputs=static_cast<NSArray *>(priorCommon[@"verification_tokens"]),fixes=static_cast<NSArray *>(priorCommon[@"correction_tokens"]);require(inputs.count==4&&fixes.count==4,"control token metadata differs");
        for(uint32_t i=0;i<4;++i){verification.push_back(uint32_t(number(inputs[i])));correction[i]=uint32_t(number(fixes[i]));require(verification.back()<kVocabulary&&correction[i]<kVocabulary,"control token metadata invalid");}}
      LocalTails tails(target);uint64_t localChecks=0,invalidChecks=0;
      for(uint32_t retained:{1u,2u,3u,4u,0u}){
        stage="state";allocation.stateMeasured=false;allocation.stateDelta=allocation.afterState=0;allocation.beforeState=backend.memoryStats().allocatedBytes;
        auto state=target.createState();auto guards=Access::guardState(backend,state);allocation.afterState=backend.memoryStats().allocatedBytes;allocation.finalAllocated=allocation.afterState;
        require(allocation.afterState>=allocation.beforeState,"state ledger regressed");allocation.stateDelta=allocation.afterState-allocation.beforeState;allocation.stateMeasured=true;
        require(allocation.stateDelta<=allocation.state,"state exceeds plan");require(workspace+allocation.stateDelta<=planned,"target/state exceeds reservation");require(allocation.finalAllocated-allocation.mappedInitial<=planned,"live ledger exceeds reservation");
        Bindings binding(target,state);const bool repeat=retained!=1;stage="prefill";
        auto body=target.forward(state,prompt,false,true);store.output("prefill.output",body,kRows,repeat);store.state("prefill.state",target,state,repeat);
        if(verification.empty())verification={greedy(body,0),prompt[kRows-3],prompt[kRows-2],prompt[kRows-1]};
        stage="verify";auto verified=target.verify(state,verification);require(verified.logitRows==4&&Access::pending(state)&&!target.ownsState(state),"pending verification predicates differ");require(Access::tapeStatus(target)==std::array<uint64_t,3>{4,kRows,36},"pending tape scalar state differs");
        store.output("verify.output",verified,4,repeat);store.state("verify.pending.state",target,state,repeat,true,false);store.frame("verify.tapes",kTapeMeta,tapeViews(target),repeat);
        if(retained==1)for(uint32_t i=0;i<4;++i){if(exporting)correction[i]=greedy(verified,i);else require(correction[i]==greedy(verified,i),"control correction record differs");}
        const auto unchanged=[&]{require(Access::tapeStatus(target)==std::array<uint64_t,3>{4,kRows,36},"invalid operation changed tape status");binding.check(target,state);checkGuards(guards);tails.check();store.output("verify.output",verified,4,true);store.state("verify.pending.state",target,state,true,true,false);store.frame("verify.tapes",kTapeMeta,tapeViews(target),true);++localChecks;};
        if(!retained){
          rejected([&]{(void)target.commitVerify(state,0);},"zero retained");++invalidChecks;unchanged();
          rejected([&]{(void)target.commitVerify(state,5);},"excess retained");++invalidChecks;unchanged();
          const std::array<uint32_t,1> one{verification[0]};rejected([&]{(void)target.forward(state,one);},"forward while pending");++invalidChecks;unchanged();
          rejected([&]{(void)target.verify(state,verification);},"verify while pending");++invalidChecks;unchanged();
          auto extra=governor.tryReserve(allocation.state);require(bool(extra),"governor denied wrong-identity sibling probe");
          {auto sibling=target.createState();auto siblingGuards=Access::guardState(backend,sibling);rejected([&]{(void)target.commitVerify(sibling,1);},"wrong request identity");++invalidChecks;target.abortVerify(sibling);require(!sibling.poisoned()&&!Access::pending(sibling),"foreign abort changed sibling");checkGuards(siblingGuards);unchanged();}
          target.abortVerify(state);require(state.poisoned()&&!Access::pending(state)&&!target.ownsState(state),"abort did not poison/disarm");require(Access::tapeStatus(target)==std::array<uint64_t,3>{0,0,0},"abort left live tape trial");
          store.state("abort.state",target,state,false,false,true);store.frame("verify.tapes",kTapeMeta,tapeViews(target),true);checkGuards(guards);tails.check();require(Access::tapeCanaries(target),"abort damaged tape redzones");
          const auto poisonUnchanged=[&]{require(Access::tapeStatus(target)==std::array<uint64_t,3>{0,0,0},"poison operation rearmed tapes");store.state("abort.state",target,state,true,false,true);store.frame("verify.tapes",kTapeMeta,tapeViews(target),true);binding.check(target,state);checkGuards(guards);tails.check();++localChecks;};
          rejected([&]{(void)target.forward(state,one);},"poison forward");poisonUnchanged();rejected([&]{(void)target.verify(state,verification);},"poison verify");poisonUnchanged();rejected([&]{(void)target.commitVerify(state,1);},"poison commit");poisonUnchanged();invalidChecks+=3;continue;
        }
        stage="commit";const auto commit=target.commitVerify(state,retained);require(!Access::pending(state)&&!state.poisoned()&&state.logicalLength()==kRows+retained&&target.ownsState(state),"commit state predicates differ");require(Access::tapeStatus(target)==std::array<uint64_t,3>{0,0,0},"commit left live tape trial");
        if(retained==4)require(commit.gpuSeconds==0&&commit.wallSeconds==0,"full accept replayed");else {require(Access::retainedValue(target)==retained,"retained control word differs");tails.allowRetained(retained);}
        store.state("commit-r"+std::to_string(retained)+".state",target,state);store.frame("verify.tapes",kTapeMeta,tapeViews(target),true);binding.check(target,state);checkGuards(guards);tails.check();
        const auto append=[&](uint32_t rows,const std::string &label){stage=label;std::vector<uint32_t> input;
          if(rows==1)input={correction[retained-1]};else input.assign(prompt.begin(),prompt.begin()+rows);
          const auto before=state.logicalLength();auto result=target.forward(state,input,true,true);require(state.logicalLength()==before+rows&&result.logitRows==rows,"future output/state rows differ");
          store.output(label+".output",result,rows);store.state(label+".state",target,state);store.frame("verify.tapes",kTapeMeta,tapeViews(target),true);binding.check(target,state);checkGuards(guards);tails.check();++localChecks;};
        append(1,"correction-r"+std::to_string(retained));if(retained==2||retained==4)append(4,"future4-r"+std::to_string(retained));if(retained==3||retained==4)append(8,"future8-r"+std::to_string(retained));
      }
      std::ostringstream c;c<<"{\"scope\":\"TRUNK Prefill+Verify4+Commit1..4+correction/future matrix; no trained head\",\"model_source\":"<<json::quote(weights.sourceIdentity())<<",\"model_layout\":"<<json::quote(weights.manifestFingerprint())<<",\"full512_store\":"<<json::quote(target.batchInt8ExpertStore()->identitySha256())<<",\"prefill_flags\":"<<commonPolicy()<<",\"verification_tokens\":[";
      for(uint32_t i=0;i<4;++i){if(i)c<<',';c<<verification[i];}c<<"],\"correction_tokens\":[";for(uint32_t i=0;i<4;++i){if(i)c<<',';c<<correction[i];}c<<"]}";common=c.str();
      if(!exporting){const auto prior=dictionary(fs::path(argv[7])/"complete.json");NSData *data=[NSData dataWithBytes:common.data() length:common.size()];NSError *error=nil;id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];require(error==nil&&[prior[@"common"] isEqual:value],"common source/controls/input policy differs");}
      const auto actualBundle=target.batchInt8ExpertStore()->compactR4PreflightCounters();const auto actualPlan=target.batchInt8ExpertStore()->compactNativeR4VerifyCounters();
      require(actualBundle.enabled&&actualBundle.calls==240&&actualBundle.rows==960,"actual bundle verification census differs");
      require(actualPlan.planCalls==actualBundle.calls&&actualPlan.planRows==actualBundle.rows,"actual bundle/plan counters differ");
      const uint64_t hcCalls=hc_pad_verify_sep22::graphCalls.load(std::memory_order_relaxed),hcRows=hc_pad_verify_sep22::graphRows.load(std::memory_order_relaxed),hcPads=hc_pad_verify_sep22::savedPaddingDispatches.load(std::memory_order_relaxed);
      require(hcCalls==485&&hcRows==1940&&hcPads==485,"actual HC97 x five VerifyR4 census differs");
      const uint32_t actualGuardCases=Access::actualBundleGuardProbes(target);require(actualGuardCases==8,"actual model-buffer guard census differs");
      std::ostringstream o;o<<"{\"actual_HC_VerifyR4_calls\":"<<hcCalls<<",\"actual_HC_VerifyR4_rows\":"<<hcRows<<",\"actual_HC_padding_dispatches_saved\":"<<hcPads<<",\"actual_owned_buffer_guard_cases\":"<<actualGuardCases<<",\"actual_bundle_verify_calls\":"<<actualBundle.calls<<",\"actual_bundle_verify_rows\":"<<actualBundle.rows<<",\"private_lexical_token_fault_injection_claimed\":false,\"source_harness_cases_separate\":11630,\"worker\":"<<kPrefillExactBuildProvenance<<",\"kernel_routes\":"<<json::quote(target.kernelRoutes())<<",\"local_pointer_redzone_tape_checks\":"<<localChecks<<",\"invalid_operation_checks\":"<<invalidChecks<<'}';own=o.str();allocation.governor=governor.snapshot();allocation.finalAllocated=backend.memoryStats().allocatedBytes;
    }
    backendDestroyed=true;store.complete(common,own);std::ostringstream out;out<<"{\"pass\":true,\"qualification_complete\":"<<(exporting?"false":"true")<<",\"role\":"<<json::quote(exporting?"export":"compare")<<",\"scope\":\"registered COMPOSITE compact1 HC1 bundle1 exact matrix\",\"teacher_head_proved\":false,\"worker_residency_proved\":false,\"backend_destroyed\":true,\"frames\":"<<store.completed()<<",\"repeated_frames\":"<<store.repeated()<<",\"spill_bytes\":"<<store.spilled()<<",\"bytes_compared\":"<<store.compared()<<",\"planes_compared\":"<<store.checked()<<",\"allocation\":"<<allocation.json()<<",\"common\":"<<common<<",\"producer\":"<<own<<"}\n";Store::publish(report,out.str());std::cout<<out.str();return 0;
  }catch(const std::exception &error){std::ostringstream out;out<<"{\"pass\":false,\"qualification_complete\":false,\"stage\":"<<json::quote(stage)<<",\"error\":"<<json::quote(error.what())<<",\"backend_created\":"<<(backendCreated?"true":"false")<<",\"backend_destroyed\":"<<(backendDestroyed?"true":"false")<<",\"progress\":"<<progress.json()<<",\"allocation\":"<<allocation.json()<<"}\n";std::cerr<<out.str();if(!report.empty())try{Store::publish(report.string()+".failure.json",out.str());}catch(...){}return 1;}
}}
