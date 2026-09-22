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
#include "PrefillExactBuildProvenance.hpp"
#if SPLASH_PHASE_ORACLE
#include "dev/benchmarks/prefill_decode_composition_sep21/phase.hpp"
#endif
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

namespace splash::flash {
// This existing RequestState friend adds no public or private header API.
class FlashDeepPrefixOracle final {
public:
  enum class Type : uint32_t { BF16=1, F32=2, I64=3, U32=4 };
  struct Plane { std::string label; Type type; metal::MetalBuffer buffer; uint64_t live; };
  static std::vector<Plane> planes(const FlashRequestState &request) {
    if (!request.impl_) throw std::logic_error("uninitialized snapshot state");
    const auto &s=*request.impl_; std::vector<Plane> out;
    const auto add=[&](std::string label,Type type,const metal::MetalBuffer &buffer,uint64_t live) {
      if (!buffer || !buffer.contents() || live>buffer.sizeBytes())
        throw std::logic_error("invalid state plane extent: "+label);
      out.push_back({std::move(label),type,buffer,live});
    };
    for (uint32_t layer=0;layer<48;++layer) {
      const auto p="layer."+std::to_string(layer)+".";
      if (s.gdn[layer].recurrent) {
        add(p+"gdn.convolution",Type::BF16,s.gdn[layer].convolution,flashGDNConvolutionLaneBytes());
        add(p+"gdn.recurrent",Type::F32,s.gdn[layer].recurrent,flashGDNRecurrentLaneBytes());
      } else {
        const auto &q=s.qsa[layer];
        add(p+"qsa.keys",Type::BF16,q.keys,s.length*1024);
        add(p+"qsa.values",Type::BF16,q.values,s.length*1024);
        add(p+"qsa.raw_index_keys",Type::BF16,q.rawIndexKeys,s.length*256);
        add(p+"qsa.pooled_keys",Type::BF16,q.pooledKeys,(s.length/4)*256);
        add(p+"qsa.index_positions",Type::I64,q.indexPositions,s.length*8);
      }
    }
    add("ple.history",Type::I64,s.pleHistory,16);
    add("ple.convolution",Type::BF16,s.pleConvolution,9ULL*10240*2);
    if (out.size()!=134) throw std::logic_error("state inventory must contain134 planes");
    return out;
  }
  static std::array<uintptr_t,3> binding(const FlashRequestState &request) {
    if (!request.impl_) throw std::logic_error("uninitialized owner binding");
    return {reinterpret_cast<uintptr_t>(request.impl_.get()),
      reinterpret_cast<uintptr_t>(request.impl_->owner.get()),
      reinterpret_cast<uintptr_t>(request.impl_->identity.get())};
  }
  static std::string metadata(const FlashForward &target,const FlashRequestState &request) {
    if (!request.impl_) throw std::logic_error("uninitialized state metadata");
    const auto &s=*request.impl_; std::ostringstream out;
    out<<"{\"length\":"<<s.length<<",\"capacity\":"<<s.capacity
      <<",\"poisoned\":"<<(s.poisoned?"true":"false")
      <<",\"pending\":"<<(s.pendingVerification?"true":"false")
      <<",\"owned\":"<<(target.ownsState(request)?"true":"false")<<",\"geometry\":[";
    for (uint32_t i=0;i<48;++i) {
      if (i) out<<',';
      if (s.gdn[i].recurrent) out<<'['<<i<<",\"gdn\","<<s.gdn[i].convolutionLaneStrideBytes
        <<','<<s.gdn[i].recurrentLaneStrideBytes<<']';
      else out<<'['<<i<<",\"qsa\","<<s.qsa[i].capacity<<']';
    }
    out<<"]}"; return out.str();
  }
  static bool pending(const FlashRequestState &request) {
    return request.impl_ && request.impl_->pendingVerification;
  }
};
}

namespace {
using namespace splash;
using namespace splash::flash;
namespace fs=std::filesystem;
using Access=FlashDeepPrefixOracle; using Type=Access::Type;
constexpr uint32_t kCapacity=4096,kRows=2048,kVerifyRows=4,kVocabulary=248320,kHyper=10240;
constexpr uint64_t kLimit=4ULL<<30,kScratch=1ULL<<20;
constexpr std::array<uint32_t,7> kTails{0,1,4,8,9,15,16};
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
      <<",\"expected_unique_frames\":28,\"completed_repeated_frames\":"<<completedRepeated<<",\"expected_repeated_frames\":12"
      <<",\"completed_planes\":"<<completedPlanes<<",\"spill_bytes\":"<<spilledBytes
      <<",\"bytes_compared\":"<<comparedBytes<<",\"planes_compared\":"<<comparedPlanes<<",\"completed_frame_labels\":[";
    for(size_t i=0;i<std::min<size_t>(completedLabels.size(),28);++i){if(i)out<<',';out<<splash::json::quote(bounded(completedLabels[i],32));}
    out<<"],\"completed_repeated_labels\":[";
    for(size_t i=0;i<std::min<size_t>(repeatedLabels.size(),12);++i){if(i)out<<',';out<<splash::json::quote(bounded(repeatedLabels[i],32));}
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
  require(meta.size()<=65536 && !planes.empty() && planes.size()<=134,"frame declaration invalid");
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
    else{const auto manifest=dictionary(spill_/"complete.json");require([manifest[@"complete"] isEqual:@YES]&&string(manifest[@"schema"])=="trunkpref-export-v1","completed export required");
      for(id item:static_cast<NSArray *>(manifest[@"frames"])){const auto entry=static_cast<NSDictionary *>(item);frames_.push_back({string(entry[@"label"]),string(entry[@"sha256"]),number(entry[@"bytes"]),number(entry[@"planes"]),number(entry[@"live_bytes"])});}
      require(frames_.size()==28,"all28 expected frames required");}}
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
  void state(const std::string &label,const FlashForward &target,const FlashRequestState &state,bool repeat=false){
    progress_.begin(label,repeat);progress_.section="state_preflight";
    require(target.ownsState(state)&&state.capacity()==kCapacity&&!state.poisoned()&&!Access::pending(state),"state must be healthy/resolved/locally owned");
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
    require(completed_==28&&repeated_==12,"all canonical/tail/future frames and repeated bodies required");
    if(exporting_){std::ostringstream out;out<<"{\"schema\":\"trunkpref-export-v1\",\"complete\":true,\"common\":"<<common<<",\"producer\":"<<different<<",\"spill_bytes\":"<<spilled_<<",\"frames\":[";
      for(size_t i=0;i<frames_.size();++i){if(i)out<<',';const auto &f=frames_[i];out<<"{\"label\":"<<json::quote(f.label)<<",\"sha256\":"<<json::quote(f.sha)<<",\"bytes\":"<<f.bytes<<",\"planes\":"<<f.planes<<",\"live_bytes\":"<<f.live<<'}';}out<<"]}\n";publish(spill_/"complete.json",out.str());}}
  static void publish(const fs::path &path,const std::string &text){require(!fs::exists(path)&&!fs::exists(path.string()+".writing"),"fresh report/declaration path required");std::ofstream out(path.string()+".writing");out<<text;out.close();require(bool(out),"JSON publish failed");fs::rename(path.string()+".writing",path);}
  uint64_t completed()const{return completed_;}uint64_t repeated()const{return repeated_;}uint64_t spilled()const{return spilled_;}uint64_t compared()const{return compared_;}uint64_t checked()const{return checked_;}
private:fs::path spill_;bool exporting_;std::vector<Frame> frames_;uint64_t spilled_=0,compared_=0,checked_=0,completed_=0,repeated_=0;FailureProgress &progress_;
};
bool selected(const char *name){const char *value=std::getenv(name);require(value&&(*value=='0'||*value=='1')&&!value[1],std::string("explicit strict flag required: ")+name);return *value=='1';}
std::string prefPolicy(){
  constexpr std::array names{"SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21","SPLASH_FLASH_ALLROWS_GATHERED_MPP","SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS","SPLASH_FLASH_BLOCKED_MOE","SPLASH_FLASH_DENSE_CACHE","SPLASH_FLASH_DENSE_M64_OUT","SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21","SPLASH_FLASH_EXPERT_QMV","SPLASH_FLASH_FLOAT_DENSE_CACHE","SPLASH_FLASH_FLOAT_DENSE_SELECTIVE","SPLASH_FLASH_FUSE_GDN","SPLASH_FLASH_FUSE_HC","SPLASH_FLASH_GDN_LAZY_ROLLBACK","SPLASH_FLASH_GDN_PREFILL_FMA_SEP21","SPLASH_FLASH_GDN_STAGED","SPLASH_FLASH_GPU_GREEDY","SPLASH_FLASH_HC_UP_F32_MPP","SPLASH_FLASH_INT8_HEAD","SPLASH_FLASH_MOE_DIRECT_A","SPLASH_FLASH_MOE_M64","SPLASH_FLASH_MOE_POINTWISE_SEP21","SPLASH_FLASH_MOE_Q4X8","SPLASH_FLASH_PLE_LOOKUP_FUSED","SPLASH_FLASH_PLE_POST_FUSED","SPLASH_FLASH_PLE_SSD_STREAMING","SPLASH_FLASH_PREFILL_DENSE_TILES","SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21","SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT","SPLASH_FLASH_QMV_F32","SPLASH_FLASH_QSA_BULK_PREFILL","SPLASH_FLASH_QSA_BULK_PREFILL_SG8","SPLASH_FLASH_QSA_F32","SPLASH_FLASH_QSA_MPP","SPLASH_FLASH_QSA_OUT_F32_N32","SPLASH_FLASH_QSA_ROW_TILES","SPLASH_FLASH_SHARED_EXPERT_FUSED","SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21","SPLASH_FLASH_GDN_AB_MERGE_SEP21"};
  std::ostringstream out;out<<"{\"scope\":\"TRUNKPREF only: forward, no Decode, Verify, trained teacher prime or head state\",\"semantics\":"<<json::quote(kFlashForwardSemantics)<<",\"flags\":{";
  for(size_t i=0;i<names.size();++i){const char *v=std::getenv(names[i]);require(v,std::string("body flag absent: ")+names[i]);if(i)out<<',';out<<json::quote(names[i])<<':'<<json::quote(v);}out<<"}}";return out.str();}
void early(){
  require(selected("SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21")==bool(SPLASH_PHASE_ORACLE),"oracle role/phase flag mismatch");
  require(selected("SPLASH_FLASH_ALLROWS_FULL512_TARGET")!=bool(SPLASH_PHASE_ORACLE),"oracle role/allrows policy mismatch");
  for(const char *name:{"SPLASH_FLASH_BATCH","SPLASH_FLASH_BATCH_MTP","SPLASH_FLASH_BATCH_MTP_PREFILL","SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT","SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"})require(!selected(name),std::string("out-of-scope flag enabled: ")+name);
  require(selected("SPLASH_FLASH_ALLROWS_GATHERED_MPP")&&selected("SPLASH_FLASH_GPU_GREEDY"),"gathered/greedy required");
  require(std::getenv("SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS")&&std::string_view(std::getenv("SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS"))=="4","qualified gathered cap4 required");
  for(const char *name:{"SPLASH_FLASH_HOT_EXPERT_PLAN","SPLASH_FLASH_CAPTURE_EXPERT_IDS"})require(!std::getenv(name)||std::string_view(std::getenv(name))=="0",std::string("unsupported diagnostic/plan flag: ")+name);
  (void)prefPolicy();
}
std::string failureJSON(const std::exception &error,const std::string &stage,const FailureProgress &progress,
    const AllocationBreakdown &allocation,bool backendCreated,bool backendDestroyed,const fs::path &report,
    const std::string &publicationError={}){
  std::ostringstream out;out<<"{\"schema\":\"trunkpref-oracle-failure-v1\",\"pass\":false,\"qualification_complete\":false"
    <<",\"phase\":"<<json::quote(bounded(stage,64))<<",\"error\":"<<json::quote(bounded(error.what(),768))
    <<",\"backend_created\":"<<(backendCreated?"true":"false")<<",\"backend_destroyed\":"<<(backendDestroyed?"true":"false")
    <<",\"failure_report\":"<<json::quote(bounded(report.empty()?"":report.string()+".failure.json",768))
    <<",\"progress\":"<<progress.json()<<",\"allocation\":"<<allocation.json();
  if(!publicationError.empty())out<<",\"failure_report_publish_error\":"<<json::quote(bounded(publicationError,384));
  out<<"}\n";return out.str();
}
void publishFailure(const std::exception &error,const std::string &stage,const FailureProgress &progress,
    const AllocationBreakdown &allocation,bool backendCreated,bool backendDestroyed,const fs::path &report)noexcept{
  try{
    const auto failure=failureJSON(error,stage,progress,allocation,backendCreated,backendDestroyed,report);
    try{require(!report.empty(),"report path unavailable");Store::publish(report.string()+".failure.json",failure);std::cerr<<failure;}
    catch(const std::exception &publication){std::cerr<<failureJSON(error,stage,progress,allocation,backendCreated,backendDestroyed,report,publication.what());}
    catch(...){std::cerr<<failureJSON(error,stage,progress,allocation,backendCreated,backendDestroyed,report,"unknown publication error");}
  }catch(...){std::cerr<<"{\"schema\":\"trunkpref-oracle-failure-v1\",\"pass\":false,\"qualification_complete\":false,\"failure_report_publish_error\":\"failure diagnostic construction failed\"}\n";}
}
void cpu(){
  std::vector<uint8_t> words((1ULL<<20)+17,0x5a);View view{"synthetic",Type::U32,words.data(),words.size(),words.size()};
  const auto temp=fs::temp_directory_path()/("splash-trunkpref-cpu-"+std::to_string(NSProcessInfo.processInfo.processIdentifier));require(!fs::exists(temp),"fresh CPU fixture path required");
  const auto n=extent("{}",{view});Stream writer(temp,true,n);serialize(writer,"{}",{view});const auto sha=writer.finish();Stream reader(temp,false,n);serialize(reader,"{}",{view});require(reader.finish()==sha,"synthetic streaming parity failed");
  words.back()^=1;FailureProgress last;last.begin("synthetic.last-byte");bool mismatch=false;
  try{Stream r(temp,false,n,&last);serialize(r,"{}",{view},&last);}catch(const FrameFailure &){mismatch=true;}
  require(mismatch&&last.kind=="byte_mismatch"&&last.frame=="synthetic.last-byte"&&last.section=="plane_payload"&&last.plane=="synthetic"&&
    last.absoluteKnown&&last.absoluteByte==n-1&&last.planeByteKnown&&last.planeByte==words.size()-1&&last.width==4&&
    last.byteValuesKnown&&last.actualByte==0x5b&&last.controlByte==0x5a&&last.currentMatchedBytes==n-1,
    "synthetic multi-chunk last-byte context differs");words.back()^=1;
  FailureProgress header;header.begin("synthetic.header");bool headerMismatch=false;
  try{Stream r(temp,false,n,&header);serialize(r,"{]",{view},&header);}catch(const FrameFailure &){headerMismatch=true;}
  require(headerMismatch&&header.section=="metadata"&&header.absoluteByte==25&&!header.planeByteKnown&&header.width==0&&header.plane.empty(),
    "synthetic metadata mismatch was misclassified as payload");
  FailureProgress shortFrame;shortFrame.begin("synthetic.extent");bool truncated=false;
  try{Stream r(temp,false,n+1,&shortFrame);}catch(const std::runtime_error &){truncated=true;}
  require(truncated&&shortFrame.section=="frame_file_extent"&&shortFrame.expectedKnown&&shortFrame.expectedBytes==n+1&&
    shortFrame.observedKnown&&shortFrame.observedBytes==n,"extent mismatch context differs");
  FailureProgress invalid;invalid.begin("synthetic.preflight");bool invalidExtent=false;
  try{(void)extent("{}",{{"invalid",Type::BF16,nullptr,2,2}},&invalid);}catch(const std::runtime_error &){invalidExtent=true;}
  require(invalidExtent&&invalid.section=="frame_extent_plane"&&invalid.plane=="invalid"&&invalid.width==2,
    "payload preflight lost plane context");
  const std::array<uint16_t,1> nan{0x7fc0};bool rejected=false;try{finite({"nan",Type::BF16,reinterpret_cast<const uint8_t *>(nan.data()),2,2});}catch(const std::runtime_error &){rejected=true;}require(rejected,"nonfinite checkpoint not rejected");
  require(little(0x0102030405060708ULL)==std::array<uint8_t,8>{8,7,6,5,4,3,2,1},"frame endian differs");
  require(FlashForward::requestStateBytes(kCapacity)==232603648ULL,"134-plane physical formula differs");
  uint64_t cacheCensus=0;
  for(uint32_t layer=0;layer<48;++layer){const std::array<uint32_t,2> outputs=layer%4==3?std::array<uint32_t,2>{12288,0}:std::array<uint32_t,2>{10240,6144};
    for(uint32_t n:outputs)if(n)cacheCensus+=((uint64_t{n}*2560+16383)/16384)*16384+((uint64_t{n}*4+16383)/16384)*16384;}
  require(cacheCensus==1890975744ULL&&dense_w8a8_sep21::Cache::plannedBytes()==cacheCensus,"independent W8 coefficient census differs");
  require(dense_w8a8_sep21::requiresCache(kRows)&&!dense_w8a8_sep21::requiresCache(kRows-1),"W8 cache must be planned by constructor capacity");
  const uint64_t upper=14*FlashForward::requestStateBytes(kCapacity)+uint64_t{kRows}*kHyper*2+150000000ULL;
  require(upper<kLimit,"accepted spill footprint exceeds4GiB");fs::remove(temp);
  std::cout<<"{\"pass\":true,\"gpu_executed\":false,\"model_payload_bytes_read\":0,\"checks\":[\"streaming_exact_last_byte\",\"frame_extent\",\"finite\",\"little_endian\",\"134_plane_formula\",\"spill_bound\",\"independent_w8_cache_census_and_capacity_gate\",\"multi_chunk_last_byte_context\",\"metadata_context\",\"extent_and_payload_preflight_context\"],\"dense_w8_cache_bytes\":"<<cacheCensus<<",\"spill_upper_bound_bytes\":"<<upper<<"}\n";
}
}

int main(int argc,char **argv){@autoreleasepool{
  bool backendCreated=false,backendDestroyed=false;std::string progress="arguments";fs::path report;AllocationBreakdown allocation;FailureProgress diagnostic;
  try{
    if(argc>=7&&argv[6])report=argv[6];
    if(argc==2&&std::string_view(argv[1])=="--cpu-only"){cpu();return 0;}
    require(argc==8&&std::string_view(argv[1])=="--gpu","usage: oracle --gpu export|compare METALLIB PACKAGE TOKENS_JSON FRESH_REPORT SPILL_DIRECTORY | --cpu-only");
    const bool exporting=std::string_view(argv[2])=="export";require(exporting||std::string_view(argv[2])=="compare","export or compare required");
    diagnostic.role=exporting?"export":"compare";
    require(exporting!=bool(SPLASH_PHASE_ORACLE),"separate-process binary role mismatch");require(!fs::exists(report)&&!fs::exists(report.string()+".writing"),"fresh Root report required");
    diagnostic.section="policy_preflight";early();diagnostic.section="prompt_preflight";const auto prompt=tokens(argv[5]);const std::string policy=prefPolicy();
    diagnostic.section="provenance_preflight";
    const auto provenance=dictionary(kPrefillExactProvenancePath);require(fileHash(argv[3])==string(provenance[@"metallib_sha256"]),"loaded library differs from sealed worker");
    for(NSString *key in @[@"sources",@"objects"])for(id item:static_cast<NSArray *>(provenance[key])){const auto entry=static_cast<NSDictionary *>(item);require(fileHash(string(entry[@"path"]))==string(entry[@"sha256"]),"frozen compile closure differs");}
    Store store(argv[7],exporting,diagnostic);std::string common,different;uint64_t peak=0,planned=0,workspace=0;engine::MemoryGovernorSnapshot finalGovernor{};
    progress="backend_construct";diagnostic.section="backend_construct";
    {
      BackendLifetime lifetime{backendCreated,backendDestroyed};metal::MetalBackend backend(argv[3]);backendCreated=true;const auto weights=FlashWeights::load(backend,argv[4]);
      require(weights.descriptor().layers==48&&weights.descriptor().vocabularySize==kVocabulary,"actual descriptor differs");
      progress="weights_loaded";diagnostic.section="admission_plan";allocation.mappedInitial=backend.memoryStats().allocatedBytes;allocation.finalAllocated=allocation.mappedInitial;
      const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);
      engine::MemoryGovernor governor(backend,physical-reserve,reserve);
      allocation.fixed=FlashForward::workspacePlannedBytes(kCapacity,kRows,kVerifyRows);
      allocation.experts=FlashForward::expertCachePlannedBytes(weights);allocation.f32=FlashForward::floatDenseCachePlannedBytes(weights);
      allocation.head=FlashForward::int8HeadPlannedBytes(weights);allocation.state=FlashForward::requestStateBytes(kCapacity);
      if(selected("SPLASH_FLASH_DENSE_CACHE"))allocation.bf16=FlashDenseCache::plannedBytes(weights,FlashDenseCache::defaultPrefixes(weights,true));
      if(selected("SPLASH_FLASH_BLOCKED_MOE"))allocation.blocked=flashMoEBlockedWorkspacePlannedBytes(kRows,10);
      // The constructor creates this fixed derivative whenever maximumRows>=2048,
      // independently of kernel selection. Forward's fixed planner admits its
      // activation Workspace only; mirror the Worker's separate coefficient charge.
      if(dense_w8a8_sep21::requiresCache(kRows))allocation.w8Cache=dense_w8a8_sep21::Cache::plannedBytes();
      allocation.plannedTarget=allocation.fixed+allocation.experts+allocation.f32+allocation.head+allocation.bf16+allocation.blocked+allocation.w8Cache;
      planned=allocation.reservation=allocation.plannedTarget+allocation.state+allocation.margin;
      allocation.governor=governor.snapshot();allocation.governorStage="before_reservation";
      progress="reservation_preflight";diagnostic.section="reservation_preflight";Store::publish(report.string()+".reservation-checkpoint.json","{\"qualification_complete\":false,\"allocation\":"+allocation.json()+"}\n");
      auto reservation=governor.tryReserve(planned);require(bool(reservation),"governor denied fixed trunk/cache/one-state admission");
      {
        progress="target_construct";diagnostic.section="target_construct";FlashForward target(backend,weights,kCapacity,kRows,kVerifyRows);workspace=target.workspaceBytes();
        progress="target_constructed";diagnostic.section="target_ledger";allocation.workspace=workspace;allocation.afterTarget=backend.memoryStats().allocatedBytes;allocation.finalAllocated=allocation.afterTarget;
        require(allocation.afterTarget>=allocation.mappedInitial,"target allocation ledger regressed");allocation.targetDelta=allocation.afterTarget-allocation.mappedInitial;allocation.targetMeasured=true;
        const auto measuredOperands=target.persistedOperandStatus();allocation.bf16SavedCount=measuredOperands.bf16Tensors;allocation.bf16SavedPayload=measuredOperands.bf16PayloadBytes;
        allocation.f32SavedCount=measuredOperands.f32Tensors;allocation.f32SavedPayload=measuredOperands.f32PayloadBytes;
        allocation.governor=governor.snapshot();allocation.governorStage="target_constructed_reservation_in_flight";
        Store::publish(report.string()+".allocation-checkpoint.json","{\"qualification_complete\":false,\"allocation\":"+allocation.json()+"}\n");
        require(allocation.targetDelta==workspace,"target ledger delta differs from Forward workspace ledger");
        require(allocation.targetDelta<=allocation.plannedTarget,"actual target allocation exceeds category plan");
        require(workspace+FlashForward::requestStateBytes(kCapacity)<=planned,"actual trunk/one-state arena exceeds reservation");
        reservation->commit();
        allocation.governor=governor.snapshot();allocation.governorStage="after_target_commit";
        diagnostic.section="common_policy";const auto *experts=target.batchInt8ExpertStore();require(experts&&experts->mappedBytes()==121173442560ULL,"actual Full512 source store differs");
        std::ostringstream shared;shared<<"{\"input_file_sha256\":"<<json::quote(kFixtureSHA)<<",\"input_u32_sha256\":"<<json::quote(kTokensSHA)<<",\"model_source_identity\":"<<json::quote(weights.sourceIdentity())<<",\"model_manifest_fingerprint\":"<<json::quote(weights.manifestFingerprint())<<",\"pref_i8_numerical_identity\":"<<json::quote(experts->numericalIdentitySha256())<<",\"pref_i8_store_identity\":"<<json::quote(experts->identitySha256())<<",\"numeric_pref_policy\":"<<policy<<'}';common=shared.str();
        if(!exporting){const auto declaration=dictionary(fs::path(argv[7])/"complete.json");NSString *text=[NSString stringWithUTF8String:common.c_str()];NSError *error=nil;id value=[NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:&error];require(error==nil&&[declaration[@"common"] isEqual:value],"source/input/numeric Prefill policy differs before any trunk call");}
        const auto operands=target.persistedOperandStatus();std::ostringstream own;own<<"{\"role\":"<<json::quote(SPLASH_PHASE_ORACLE?"phase":"teacher")<<",\"kernel_routes\":"<<json::quote(target.kernelRoutes())<<",\"f32_saved_tensors\":"<<operands.f32Tensors<<",\"f32_saved_bytes\":"<<operands.f32PayloadBytes<<",\"artifact_provenance\":"<<kPrefillExactBuildProvenance;
#if SPLASH_PHASE_ORACLE
        diagnostic.section="producer_identity";
        own<<",\"phase_global_numerical_identity\":"<<json::quote(phase_q4_sep21::identity(experts->numericalIdentitySha256()));
#endif
        own<<'}';different=own.str();
        for(uint32_t tail:kTails){diagnostic.begin("body.output",tail!=0);diagnostic.section="state_allocation";allocation.stateMeasured=false;allocation.stateDelta=0;allocation.afterState=0;allocation.beforeState=backend.memoryStats().allocatedBytes;auto state=target.createState();allocation.afterState=backend.memoryStats().allocatedBytes;allocation.finalAllocated=allocation.afterState;
          require(allocation.afterState>=allocation.beforeState,"request allocation ledger regressed");allocation.stateDelta=allocation.afterState-allocation.beforeState;allocation.stateMeasured=true;
          allocation.governor=governor.snapshot();allocation.governorStage="one_request_live";
          progress="state_created-r"+std::to_string(tail);if(!tail)Store::publish(report.string()+".state-allocation-checkpoint.json","{\"qualification_complete\":false,\"allocation\":"+allocation.json()+"}\n");
          require(allocation.stateDelta<=allocation.state,"actual request allocation exceeds state plan");
          require(workspace+allocation.stateDelta<=planned,"actual trunk/request arena exceeds reservation");
          require(allocation.finalAllocated>=allocation.mappedInitial&&allocation.finalAllocated-allocation.mappedInitial<=planned,"live backend allocation exceeds reservation growth");
          const auto binding=Access::binding(state);progress="body-r"+std::to_string(tail);std::cerr<<progress<<'\n';
          diagnostic.section="trunk_forward";const auto body=target.forward(state,prompt,false,true);require(state.logicalLength()==kRows,"body logical length differs");
          store.output("body.output",body,kRows,tail!=0);store.state("body.state",target,state,tail!=0);
          if(tail){const auto input=std::span<const uint32_t>(prompt).subspan(kRows-16,tail);progress="tail-r"+std::to_string(tail);diagnostic.begin("tail-r"+std::to_string(tail)+".output");diagnostic.section="trunk_forward";const auto result=target.forward(state,input,true,true);
            require(state.logicalLength()==kRows+tail,"tail logical length differs");store.output("tail-r"+std::to_string(tail)+".output",result,tail);store.state("tail-r"+std::to_string(tail)+".state",target,state);}
          progress="future-r"+std::to_string(tail);diagnostic.begin("future-r"+std::to_string(tail)+".output");diagnostic.section="trunk_forward";const auto result=target.forward(state,std::span<const uint32_t>(prompt).first(8),true,true);
          require(state.logicalLength()==kRows+tail+8,"future logical length differs");store.output("future-r"+std::to_string(tail)+".output",result,8);store.state("future-r"+std::to_string(tail)+".state",target,state);
          require(Access::binding(state)==binding,"trunk changed request owner/identity binding");}
#if SPLASH_PHASE_ORACLE
        diagnostic.begin("");diagnostic.section="prefill_graph_census";
        const auto &p=phase_q4_sep21::counters[0];require(p.i8Calls.load()==960&&p.i8Rows.load()==693360&&p.q4Calls.load()==0&&p.q4Rows.load()==0,"Prefill-only phase graph census differs");
        for(uint32_t i=1;i<3;++i){const auto &c=phase_q4_sep21::counters[i];require(!c.i8Calls.load()&&!c.i8Rows.load()&&!c.q4Calls.load()&&!c.q4Rows.load(),"Decode/Verify graphs unexpectedly constructed");}
#endif
        diagnostic.section="final_governor";finalGovernor=governor.snapshot();allocation.governor=finalGovernor;allocation.governorStage="prefill_complete";allocation.finalAllocated=backend.memoryStats().allocatedBytes;
      }
      peak=backend.memoryStats().peakAllocatedBytes;
    }
    backendDestroyed=true;progress="completed";store.complete(common,different);std::ostringstream out;
    out<<"{\"schema\":\"trunkpref-two-process-exact-v1\",\"pass\":true,\"qualification_complete\":"<<(exporting?"false":"true")<<",\"scope\":\"TRUNKPREF only\",\"teacher_head_prime_and_state_proved\":false,\"worker_residency_lease_proved\":false,\"role\":"<<json::quote(exporting?"export":"compare")<<",\"backend_created\":true,\"backend_destroyed\":true,\"checkpoints\":14,\"frames\":"<<store.completed()<<",\"repeated_body_frames\":"<<store.repeated()<<",\"spill_bytes\":"<<store.spilled()<<",\"bytes_compared\":"<<store.compared()<<",\"planes_compared\":"<<store.checked()<<",\"one_request_at_a_time\":true,\"capacity\":"<<kCapacity<<",\"workspace_bytes\":"<<workspace<<",\"planned_reservation_bytes\":"<<planned<<",\"peak_backend_allocated_bytes\":"<<peak<<",\"final_diagnostic_governor\":{\"observed_resident_bytes\":"<<finalGovernor.observedResidentBytes<<",\"reserved_bytes\":"<<finalGovernor.reservedBytes<<",\"headroom_bytes\":"<<finalGovernor.headroomBytes<<",\"denied_reservations\":"<<finalGovernor.deniedReservations<<",\"host_measurement_valid\":"<<(finalGovernor.hostMeasurementValid?"true":"false")<<",\"host_headroom_bytes\":"<<finalGovernor.hostHeadroomBytes<<",\"growth_allowed\":"<<(finalGovernor.growthAllowed?"true":"false")<<"},\"common\":"<<common<<",\"producer\":"<<different<<",\"allocation\":"<<allocation.json()<<"}\n";
    diagnostic.section="report_publish";Store::publish(report,out.str());std::cout<<out.str();return 0;
  }catch(const std::exception &error){
    publishFailure(error,progress,diagnostic,allocation,backendCreated,backendDestroyed,report);
    return 1;
  }catch(...){
    diagnostic.kind="unknown_native_exception";
    const std::runtime_error error("non-standard native exception");
    publishFailure(error,progress,diagnostic,allocation,backendCreated,backendDestroyed,report);
    return 1;
  }
}}
