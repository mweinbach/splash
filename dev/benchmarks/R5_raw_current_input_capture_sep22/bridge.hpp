#pragma once
#include "contracts.hpp"
#include "engine/MemoryGovernor.hpp"
#include "metal/abi/FlashForward.h"
#include "metal/abi/FlashAffine.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace splash::flash::r5_raw_current_capture_sep22 {
inline constexpr char kFlag[]="SPLASH_FLASH_CAPTURE_R5_RAW_INPUT_SEP22";
inline constexpr char kProofFlag[]="SPLASH_FLASH_CAPTURE_R5_RAW_PROOF_SEP22";
inline constexpr char kDirectory[]="SPLASH_FLASH_CAPTURE_R5_RAW_DIRECTORY_SEP22";
inline void require(bool ok,const char *why){if(!ok)throw std::logic_error(why);}
inline std::string env(const char *name,const char *fallback=""){const char *v=std::getenv(name);return v?v:fallback;}
inline bool strict(const std::string &v){require(v=="0"||v=="1","R5 capture flag must be strict0/1");return v=="1";}
struct Config final {
 bool capture=false,proof=false;std::string directory;std::array<std::string,5> frozen;
 static Config read(){Config c;c.frozen={env(kFlag,"0"),env(kProofFlag,"0"),env(kDirectory),env("SPLASH_FLASH_MTP_DRAFT_DEPTH"),env("SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22","0")};c.capture=strict(c.frozen[0]);c.proof=strict(c.frozen[1]);if(!c.proof&&!c.capture)return c;
  require(c.proof,"capture requires paired state proof instrumentation");require(c.frozen[3]=="4"&&c.frozen[4]=="1","capture requires current fixed4/R5 integer policy");c.directory=c.frozen[2];require(!c.directory.empty()&&c.directory.size()<=4096&&std::filesystem::path(c.directory).is_absolute(),"fresh absolute capture directory required");return c;}
 static Config frozenConfig(){const Config now=read();static const Config first=now;require(now.capture==first.capture&&now.proof==first.proof&&(!(now.proof||now.capture)||now.frozen==first.frozen),"capture configuration changed");return first;}
 void recheck()const{const auto now=frozenConfig();require(capture==now.capture&&proof==now.proof&&(!(proof||capture)||frozen==now.frozen),"capture owner configuration changed");}
};
inline void validateStartup(){(void)Config::frozenConfig();}
inline bool overlap(const metal::MetalBuffer&a,const metal::MetalBuffer&b){require(a&&b&&a.contents()&&b.contents(),"capture requires shared owned/view metadata");const auto x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());return x<=y?uint64_t(y-x)<a.sizeBytes():uint64_t(x-y)<b.sizeBytes();}
inline void writeAll(int fd,const void *data,uint64_t bytes){const auto *p=static_cast<const uint8_t*>(data);while(bytes){const size_t n=size_t(std::min<uint64_t>(bytes,65536));const auto written=::write(fd,p,n);if(written<0&&errno==EINTR)continue;require(written>0,"bounded capture write failed");p+=written;bytes-=uint64_t(written);}}
inline void freshWrite(const std::filesystem::path&path,const void *data,uint64_t bytes){const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);require(fd>=0,"fresh capture file required");try{require(::fcntl(fd,F_NOCACHE,1)==0,"bounded capture file-cache policy unavailable");writeAll(fd,data,bytes);}catch(...){::close(fd);throw;}require(::close(fd)==0,"capture close failed");}
class Owner;
class Scope;
inline thread_local Scope *current=nullptr;
class Owner final {
public:
 struct Slot final { Role role;metal::MetalBuffer base,view,input,output,diagnostics;bool seen=false; };
 Owner(metal::MetalBackend &backend,engine::MemoryGovernor &governor,std::vector<Role> roles)
  :config(Config::frozenConfig()){
  if(!config.proof)return;
  require(roles.size()==kRoles,"capture inventory must contain113 canonical roles");const auto before=governor.snapshot();require(before.hostMeasurementValid&&before.growthAllowed&&before.hostHeadroomBytes>=kHostHeadroomRequired+kOwnerReservation,"capture host128MiB+owner16MiB admission refused");
  reservation_=governor.tryReserve(kOwnerReservation);require(bool(reservation_),"capture16MiB reservation refused");
  hostArena_=std::make_unique<uint8_t[]>(kHostArenaBytes);std::memset(hostArena_.get(),0,kHostArenaBytes);
  const auto held=governor.snapshot();require(held.hostMeasurementValid&&held.growthAllowed&&held.hostHeadroomBytes>=kHostArenaBytes,"capture post-host-preallocation headroom invalid");
  const auto start=backend.memoryStats().allocatedBytes;uint64_t logical=0,aligned=0;slots.reserve(kRoles);
  for(auto &r:roles){Slot slot;logical+=logicalBytes(r);aligned+=guardedBytes(r);slot.role=std::move(r);
   if(config.capture){slot.base=backend.allocateBuffer(guardedBytes(slot.role),metal::BufferStorage::Shared,"private real R5 raw-input guarded slot");require(slot.base&&slot.base.contents(),"capture slot unavailable");std::memset(slot.base.contents(),0x5a,slot.base.sizeBytes());slot.view=backend.view(slot.base,64,logicalBytes(slot.role));}
   slots.push_back(std::move(slot));}
  require(logical==kLogicalBytes&&aligned==kAlignedBytes,"capture113-role byte census differs");
  const auto finish=backend.memoryStats().allocatedBytes;require(finish>=start&&finish-start<=kOwnerReservation,"capture owner exceeds16MiB ledger");ownerDelta=finish-start;
  require(std::filesystem::create_directory(config.directory),"fresh capture directory required");
 }
 Owner(const Owner&)=delete;Owner&operator=(const Owner&)=delete;
 void canaries()const{if(!config.capture)return;for(const auto &s:slots){const auto *p=static_cast<const uint8_t*>(s.base.contents());require(s.view.sizeBytes()==logicalBytes(s.role)&&s.view.contents()==p+64,"capture owner/view changed");for(uint64_t i=0;i<64;++i)require(p[i]==0x5a,"capture prefix guard changed");for(uint64_t i=64+s.view.sizeBytes();i<s.base.sizeBytes();++i)require(p[i]==0x5a,"capture suffix/inactive guard changed");}}
 void rememberInputs(){if(!config.capture)return;uint64_t at=0;for(const auto&s:slots){std::memcpy(hostArena_.get()+at,s.view.contents(),s.view.sizeBytes());at+=s.view.sizeBytes();}require(at==kLogicalBytes,"capture saved-input host census differs");}
 void inputsUnchanged()const{if(!config.capture)return;uint64_t at=0;for(const auto&s:slots){require(std::memcmp(hostArena_.get()+at,s.view.contents(),s.view.sizeBytes())==0,"captured input changed during commit/future");at+=s.view.sizeBytes();}}
 Config config;std::vector<Slot> slots;uint64_t completed=0,copies=0,ownerDelta=0;bool captured=false,failed=false;
private:
 std::optional<engine::MemoryGovernor::Reservation> reservation_;std::unique_ptr<uint8_t[]>hostArena_;
};
class Scope final {
public:
 Scope(Owner &owner,uint64_t id,uint64_t generation,uint32_t depth,uint32_t rows):owner_(owner){
  if(!owner.config.proof||id!=kRequestId||generation!=kGeneration||depth!=4||rows!=5)return;
  require(!current&&!owner.failed,"nested/sticky-invalid genuine R5 capture scope");owner.config.recheck();ordinal_=owner.completed+1;eligible_=true;selected_=ordinal_==kOrdinal;current=this;
  if(selected_){require(!owner.captured,"capture ordinal reused");for(auto&s:owner.slots){require(!s.seen,"capture slot reused");}}
 }
 ~Scope(){if(current==this)current=nullptr;if(eligible_&&!finished_)owner_.failed=true;}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
 bool selected()const noexcept{return selected_;}
 void input(metal::CommandGraph &graph,const std::string &role,const metal::MetalBuffer &input,const FlashAffineProjection&p,const metal::MetalBuffer&output,const metal::MetalBuffer&diagnostics,uint32_t rows,bool verification,bool singletonMain,bool cacheContains,bool nullPolicy){
  if(!selected_)return;
  require(!finished_&&rows==5&&verification&&singletonMain&&cacheContains&&nullPolicy,"excluded capture producer context");
  auto found=std::find_if(owner_.slots.begin(),owner_.slots.end(),[&](const auto&s){return s.role.name==role;});if(found==owner_.slots.end())return;auto &s=*found;const auto &q=s.role.projection;
  require(!s.seen&&p.experts==1&&p.inputSize==q.inputSize&&p.outputSize==q.outputSize&&p.bits==q.bits&&p.groupSize==q.groupSize&&p.weightRowStrideBytes==q.weightRowStrideBytes&&p.parameterRowStrideBytes==q.parameterRowStrideBytes&&p.weightExpertStrideBytes==q.weightExpertStrideBytes&&p.parameterExpertStrideBytes==q.parameterExpertStrideBytes,"capture producer geometry/stride changed");
  require(p.weights==q.weights&&p.scales==q.scales&&p.biases==q.biases&&p.weights&&p.scales&&p.biases&&p.weights->dtype==FlashDType::U32&&p.scales->dtype==FlashDType::BF16&&p.biases->dtype==FlashDType::BF16,"capture original quantized view changed");
  require(input&&output&&diagnostics&&input.contents()&&output.contents()&&diagnostics.contents()&&reinterpret_cast<uintptr_t>(input.contents())%4==0&&input.sizeBytes()>=logicalBytes(s.role)&&output.sizeBytes()>=uint64_t{5}*p.outputSize*2&&diagnostics.sizeBytes()>=4&&!input.sameView(output),"capture raw producer buffer guard failed");
  if(owner_.config.capture){for(const auto&b:{input,output,diagnostics,p.weights->buffer,p.scales->buffer,p.biases->buffer})require(!overlap(s.base,b),"capture slot aliases actual source/output/status");require(logicalBytes(s.role)%4==0,"capture copy words alignment differs");}
  // All context/source/alias/extents checks complete before graph mutation.
  s.input=input;s.output=output;s.diagnostics=diagnostics;s.seen=true;++recorded_;
  if(owner_.config.capture){const uint64_t words=logicalBytes(s.role)/4;graph.add("flash_forward_copy_words",{input,s.view},FlashForwardCopyParams{words},{(words+255)/256,1,1},{256,1,1});++owner_.copies;}
 }
 void graph(const metal::CommandGraph &graph,bool verification,uint32_t rows,uint64_t begin){
  if(!eligible_)return;require(!finished_&&++graphs_==1&&verification&&rows==5,"genuine R5 scope must receive one Verify5 graph");begin_=begin;
  if(!selected_)return;require(recorded_==kRoles&&graph.dispatches().size()<=4096,"selected real R5 producer/graph bound differs");std::array<uint32_t,7> count{};uint32_t actual=0;const auto dispatches=graph.dispatches();
  for(size_t di=0;di<dispatches.size();++di)for(uint32_t i=0;i<kShapes.size();++i){const auto &d=dispatches[di];const auto&s=kShapes[i];if(d.pipelineName!=s.pipeline||d.bytes.size()!=1||d.bytes[0].sizeBytes!=sizeof(FlashAffineParams))continue;require(d.bytes[0].data!=nullptr,"shipping inline parameters unavailable");FlashAffineParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));if(p.input_size!=s.K||p.output_size!=s.N)continue;
   require(p.rows==5&&p.selections==1&&p.experts==1&&p.bits==s.bits&&p.group_size==s.group&&p.flags==0&&d.buffers.size()==7&&d.threadgroups.x==s.N/8&&d.threadgroups.y==5&&d.threadgroups.z==1&&d.threadsPerThreadgroup.x==64&&d.threadsPerThreadgroup.y==1&&d.threadsPerThreadgroup.z==1,"actual captured shipping producer ABI/grid differs");
   auto match=std::find_if(owner_.slots.begin(),owner_.slots.end(),[&](const auto&slot){return slot.role.shape==i&&slot.seen&&slot.input.sameView(d.buffers[0].buffer)&&slot.role.projection.weights->buffer.sameView(d.buffers[1].buffer)&&slot.role.projection.scales->buffer.sameView(d.buffers[2].buffer)&&slot.role.projection.biases->buffer.sameView(d.buffers[3].buffer)&&slot.output.sameView(d.buffers[5].buffer)&&slot.diagnostics.sameView(d.buffers[6].buffer);});require(match!=owner_.slots.end()&&d.buffers[4].buffer.sameView(d.buffers[0].buffer)&&p.weight_row_stride_bytes==match->role.projection.weightRowStrideBytes&&p.parameter_row_stride_bytes==match->role.projection.parameterRowStrideBytes&&p.weight_expert_stride_bytes==match->role.projection.weightExpertStrideBytes&&p.parameter_expert_stride_bytes==match->role.projection.parameterExpertStrideBytes&&d.bytes[0].index==7,"actual production bindings/strides differ from input capture hook");for(uint32_t bi=0;bi<7;++bi)require(d.buffers[bi].index==bi,"actual production slot indices differ");
   if(owner_.config.capture){require(di>0,"capture copy lacks original successor");const auto &copy=dispatches[di-1];require(copy.pipelineName=="flash_forward_copy_words"&&copy.buffers.size()==2&&copy.bytes.size()==1&&copy.bytes[0].data&&copy.bytes[0].sizeBytes==sizeof(FlashForwardCopyParams)&&copy.buffers[0].buffer.sameView(d.buffers[0].buffer)&&copy.buffers[1].buffer.sameView(match->view),"capture copy is not immediate production predecessor");FlashForwardCopyParams cp{};std::memcpy(&cp,copy.bytes[0].data,sizeof(cp));require(cp.words==logicalBytes(match->role)/4&&copy.threadgroups.x==(cp.words+255)/256&&copy.threadgroups.y==1&&copy.threadgroups.z==1&&copy.threadsPerThreadgroup.x==256&&copy.threadsPerThreadgroup.y==1&&copy.threadsPerThreadgroup.z==1,"capture copy original ABI/extent/grid differs");}
   ++count[i];++actual;}
  require(actual==kRoles,"actual selected113 large shipping producers differ");for(uint32_t i=0;i<count.size();++i)require(count[i]==kShapes[i].calls,"actual seven-shape shipping census differs");
 }
 void complete(){if(!eligible_)return;require(!finished_&&graphs_==1,"capture completion lacks one synchronous actual graph");finished_=true;if(current==this)current=nullptr;++owner_.completed;if(!selected_)return;require(owner_.completed==kOrdinal&&owner_.copies==(owner_.config.capture?kRoles:0),"selected ordinal/copy census differs");owner_.canaries();owner_.captured=true;}
 uint64_t begin()const{return begin_;}
private:Owner &owner_;uint64_t ordinal_=0,begin_=0;uint32_t recorded_=0,graphs_=0;bool eligible_=false,selected_=false,finished_=false;
};
inline void captureInput(metal::CommandGraph&g,const std::string&r,const metal::MetalBuffer&i,const FlashAffineProjection&p,const metal::MetalBuffer&o,const metal::MetalBuffer&d,uint32_t rows,bool verification,bool singletonMain,bool cacheContains,bool nullPolicy){if(current)current->input(g,r,i,p,o,d,rows,verification,singletonMain,cacheContains,nullPolicy);}
inline void recordGraph(const metal::CommandGraph&g,bool verification,uint32_t rows,uint64_t begin){if(current)current->graph(g,verification,rows,begin);}
}
