// Private bounded R1 duplicate-tid0 oracle. CPU mode opens no GPU or payload.
// Root alone may call the explicit GPU path; no worker or whole-model claim.
#import <Foundation/Foundation.h>
#include "flash/FlashGatheredMPP.hpp"
#include "flash/FlashInt8ExpertStoreMetadata.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/CommandGraph.hpp"
#include "dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp"
#include "dev/benchmarks/gemv_decode_sep21_v1b/quality.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace splash;
using namespace splash::flash;
using namespace splash::metal;
namespace {
constexpr uint32_t Routes=10,Pairs=18,Sticky=0x80000000u;
constexpr uint64_t GuardBytes=16384,OneExpertBytes=4930560,FixtureBytes=49305600;
constexpr FlashGatheredMPPParams Params{1,10,512,0};
static_assert(sizeof(FlashGatheredMPPParams)==16 && alignof(FlashGatheredMPPParams)==4);
static_assert(offsetof(FlashGatheredMPPParams,rows)==0 && offsetof(FlashGatheredMPPParams,selections)==4 &&
    offsetof(FlashGatheredMPPParams,experts)==8 && offsetof(FlashGatheredMPPParams,reserved)==12);
void require(bool value,const std::string &reason) {if(!value) throw std::runtime_error(reason);}
uint16_t bf(float x) {const auto word=std::bit_cast<uint32_t>(x);return uint16_t((word+0x7fffu+((word>>16)&1u))>>16);}
float number(uint16_t x) {return std::bit_cast<float>(uint32_t(x)<<16);}
std::string hexDigest(std::span<const uint8_t> bytes) {
  constexpr char hex[]="0123456789abcdef";std::string result;
  for(auto value:bytes) {result+=hex[value>>4];result+=hex[value&15];}return result;
}
uint32_t numeric(const char *raw,uint32_t maximum) {
  const std::string value(raw);require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"numeric CLI value");
  size_t used=0;const auto parsed=std::stoul(value,&used);require(used==value.size() && parsed<=maximum,"bounded numeric CLI value");return uint32_t(parsed);
}
struct Progress final {
  std::filesystem::path path;std::string phase="arguments",check,error,library,fixture,inputHash,idsHash,ranksHash;
  uint64_t checks=0,bytes=0,guardCases=0,payloadRead=0,first=UINT64_MAX;
  uint32_t expectedByte=0,actualByte=0,expectedBF16Word=0,actualBF16Word=0,expectedF32Word=0,actualF32Word=0;
  uint64_t planned=128ULL<<20,initialAllocated=0,actualAllocated=0,peakAllocated=0,afterAllocated=UINT64_MAX;
  uint64_t initialDevice=0,deviceCurrentDelta=0,devicePeakDelta=0,denied=UINT64_MAX,hostReserve=0;
  bool gpu=false,backendInitialized=false,backendStopped=false,backendDestroyed=false,hostValid=false,growth=false;
  void write(bool complete,bool pass) const {
    if(path.empty()) return;const auto temp=path.string()+".writing";std::ofstream out(temp);require(bool(out),"progress open");
    out<<"{\"schema\":\"expert-r1-duplicate-tid0-progress-v1\",\"completed\":"<<(complete?"true":"false")
       <<",\"pass\":"<<(pass?"true":"false")<<",\"GPU_executed\":"<<(gpu?"true":"false")
       <<",\"phase\":"<<json::quote(phase)<<",\"check\":"<<json::quote(check)<<",\"error\":"<<json::quote(error)
       <<",\"coefficient_payload_bytes_read\":"<<payloadRead<<",\"checks\":"<<checks<<",\"bytes_compared\":"<<bytes
       <<",\"guard_cases\":"<<guardCases<<",\"first_different_byte\":";
    if(first==UINT64_MAX) out<<"null";else out<<first;
    out<<",\"first_mismatch_numeric\":";
    if(first==UINT64_MAX)out<<"null";else out<<"{\"expected_byte\":"<<expectedByte<<",\"actual_byte\":"<<actualByte
      <<",\"aligned_expected_BF16_word\":"<<expectedBF16Word<<",\"aligned_actual_BF16_word\":"<<actualBF16Word
      <<",\"aligned_expected_F32_bits\":"<<expectedF32Word<<",\"aligned_actual_F32_bits\":"<<actualF32Word<<'}';
    out<<",\"metallib_sha256\":"<<json::quote(library)<<",\"selected_coefficient_sha256\":"<<json::quote(fixture)
       <<",\"allocation\":{\"planned_bytes\":"<<planned<<",\"initial_owned_bytes\":"<<initialAllocated
       <<",\"actual_owned_delta_bytes\":"<<actualAllocated<<",\"peak_owned_delta_bytes\":"<<peakAllocated
       <<",\"current_device_delta_bytes\":"<<deviceCurrentDelta<<",\"peak_device_delta_bytes\":"<<devicePeakDelta
       <<",\"after_owned_delta_bytes\":"<<afterAllocated<<",\"denied_reservations\":"<<denied
       <<",\"host_measurement_valid\":"<<(hostValid?"true":"false")<<",\"growth_allowed\":"<<(growth?"true":"false")
       <<"},\"backend_stopped\":"<<(backendStopped?"true":"false")<<",\"backend_destroyed\":"<<(backendDestroyed?"true":"false")<<"}\n";
    out.close();require(bool(out),"progress close");std::filesystem::rename(temp,path);
  }
};
struct BackendLifetime final {Progress &p;~BackendLifetime(){if(p.backendInitialized)p.backendDestroyed=true;}};
struct Allocation final {MetalBuffer base,view;};
MetalBuffer allocate(MetalBackend &backend,uint64_t bytes,std::vector<Allocation> &all) {
  const uint64_t rounded=(bytes+GuardBytes-1)&~(GuardBytes-1);
  auto base=backend.allocateBuffer(rounded+2*GuardBytes,BufferStorage::Shared,"bounded-r1-duplicate-tid0");
  std::memset(base.contents(),0xa5,base.sizeBytes());auto view=backend.view(base,GuardBytes,bytes);std::memset(view.contents(),0,bytes);
  all.push_back({base,view});return view;
}
void canaries(const std::vector<Allocation> &all) {
  for(const auto &a:all) {const auto *p=static_cast<const uint8_t *>(a.base.contents());
    for(uint64_t i=0;i<GuardBytes;++i) require(p[i]==0xa5,"leading canary");
    for(uint64_t i=GuardBytes+a.view.sizeBytes();i<a.base.sizeBytes();++i) require(p[i]==0xa5,"trailing/padding canary");
  }
}
std::vector<uint8_t> copied(MetalBuffer b) {const auto *p=static_cast<const uint8_t *>(b.contents());return {p,p+b.sizeBytes()};}
bool unchanged(MetalBuffer b,const std::vector<uint8_t> &saved) {return b.sizeBytes()==saved.size() && !std::memcmp(b.contents(),saved.data(),saved.size());}
void equal(MetalBuffer a,MetalBuffer b,Progress &p,const std::string &label) {
  p.check=label;require(a.sizeBytes()==b.sizeBytes(),label+" extent");
  const auto *av=static_cast<const uint8_t *>(a.contents()),*bv=static_cast<const uint8_t *>(b.contents());
  for(uint64_t i=0;i<a.sizeBytes();++i) if(av[i]!=bv[i]) {
    p.first=i;p.expectedByte=av[i];p.actualByte=bv[i];uint16_t aw16=0,bw16=0;uint32_t aw32=0,bw32=0;
    const auto offset16=i/2*2,offset32=i/4*4;
    if(offset16+2<=a.sizeBytes()){std::memcpy(&aw16,av+offset16,2);std::memcpy(&bw16,bv+offset16,2);}
    if(offset32+4<=a.sizeBytes()){std::memcpy(&aw32,av+offset32,4);std::memcpy(&bw32,bv+offset32,4);}
    p.expectedBF16Word=aw16;p.actualBF16Word=bw16;p.expectedF32Word=aw32;p.actualF32Word=bw32;
    throw std::runtime_error(label+" bit mismatch");
  }
  ++p.checks;p.bytes+=a.sizeBytes();
}
void finite(MetalBuffer b,bool bf16) {
  if(bf16) {const auto *v=static_cast<const uint16_t *>(b.contents());for(uint64_t i=0;i<b.sizeBytes()/2;++i)require(std::isfinite(number(v[i])),"finite BF16 output/tap");}
  else {const auto *v=static_cast<const float *>(b.contents());for(uint64_t i=0;i<b.sizeBytes()/4;++i)require(std::isfinite(v[i]),"finite raw/scaled F32 tap");}
}
uint32_t diag(MetalBuffer b) {uint32_t value;std::memcpy(&value,b.contents(),4);return value;}
void resetDiag(MetalBuffer b) {std::memcpy(b.contents(),&Sticky,4);}
void disjoint(MetalBuffer a,MetalBuffer b) {
  const uintptr_t av=reinterpret_cast<uintptr_t>(a.contents()),bv=reinterpret_cast<uintptr_t>(b.contents());
  require(a && b && av && bv,"addressable shared views");
  require(av<=bv ? uint64_t(bv-av)>=a.sizeBytes() : uint64_t(av-bv)>=b.sizeBytes(),"mutable/immutable view overlap");
}
std::array<int64_t,10> routeIDs(uint32_t base,bool permuted) {
  require(base<=502,"ten selected expert bounds");constexpr std::array<uint32_t,10> order{9,0,8,1,7,2,6,3,5,4};
  std::array<int64_t,10> ids{};for(uint32_t slot=0;slot<10;++slot) ids[slot]=base+(permuted?order[slot]:slot);return ids;
}
std::vector<uint16_t> hiddenFixture() {
  std::vector<uint16_t> hidden(2560);double square=0;
  for(uint32_t k=0;k<2560;++k) {const double v=int((k*73+17)%257)-128;square+=v*v;}
  const double rms=std::sqrt(square/2560);require(rms>0 && std::isfinite(rms),"synthetic RMS divisor");
  for(uint32_t k=0;k<2560;++k) hidden[k]=bf(float(int((k*73+17)%257)-128)/float(rms));return hidden;
}
double rowRMS(std::span<const uint16_t> values) {double square=0;for(auto bits:values){const double v=number(bits);require(std::isfinite(v),"finite input fixture");square+=v*v;}return std::sqrt(square/values.size());}
bool same(DispatchSize a,DispatchSize b) {return a.x==b.x && a.y==b.y && a.z==b.z;}
DispatchSize gateGrid(bool candidate) {(void)candidate;return {10,1,10};}
DispatchSize downGrid(bool candidate) {(void)candidate;return {40,1,10};}
void fullGrid(bool candidate,DispatchSize gate,DispatchSize down,DispatchSize threads={128,1,1}) {
  require(same(gate,gateGrid(candidate)) && same(down,downGrid(candidate)) && same(threads,{128,1,1}),"host rejects partial/excess grid or threads before submission");
}
struct Bank final {std::array<MetalBuffer,3> codes,scales;std::filesystem::path source;struct stat before{};uint32_t expertBase=0;};
void boundedRanks(std::span<const uint32_t> ranks,uint32_t base) {
  require(ranks.size()==512 && base<=502,"exact bounded rank inventory");
  for(uint32_t expert=0;expert<512;++expert) {
    const auto rank=ranks[expert];
    if(expert<base || expert>=base+10)require(rank==UINT32_MAX,"nonselected ranks must remain invalid");
    else require(rank<10 || rank>=512,"host rejects finite out-of-fixture rank before submission");
  }
}
bool sameFile(const struct stat &a,const struct stat &b) {
  return S_ISREG(b.st_mode) && !(b.st_mode&0222) && a.st_dev==b.st_dev && a.st_ino==b.st_ino && a.st_size==b.st_size &&
      a.st_mtimespec.tv_sec==b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec==b.st_mtimespec.tv_nsec &&
      a.st_ctimespec.tv_sec==b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec==b.st_ctimespec.tv_nsec;
}
std::vector<uint8_t> slice(int fd,uint64_t offset,uint64_t bytes) {
  std::vector<uint8_t> values(bytes);uint64_t done=0;
  while(done<bytes) {const auto count=::pread(fd,values.data()+done,bytes-done,off_t(offset+done));require(count>0,"bounded readonly coefficient pread");done+=uint64_t(count);}return values;
}
Bank loadBank(MetalBackend &backend,const FlashInt8ExpertStoreLayer &entry,uint32_t base,std::vector<Allocation> &all,Progress &p) {
  qmv_one_layer::detail::validateEntry(entry);require(base<=502,"ten selected expert bounds");
  Bank bank;bank.source=entry.path;bank.expertBase=base;const int fd=::open(entry.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,"readonly coefficient open");
  try {
    require(!::fstat(fd,&bank.before) && S_ISREG(bank.before.st_mode) && !(bank.before.st_mode&0222) &&
        uint64_t(bank.before.st_size)==entry.bytes,"canonical readonly Full512 file stat only");
    CC_SHA256_CTX hash{};require(CC_SHA256_Init(&hash),"fixture SHA256 init");
    for(uint32_t plane=0;plane<3;++plane) {
      const uint32_t width=plane==2?2560:640,k=plane==2?640:2560;const uint64_t codeBytes=uint64_t(width)*k,scaleBytes=uint64_t(width)*4;
      bank.codes[plane]=allocate(backend,10*codeBytes,all);bank.scales[plane]=allocate(backend,10*scaleBytes,all);
      for(uint32_t rank=0;rank<10;++rank) {
        const auto code=slice(fd,entry.codes[plane].offset+uint64_t(base+rank)*codeBytes,codeBytes);
        const auto scale=slice(fd,entry.scales[plane].offset+uint64_t(base+rank)*scaleBytes,scaleBytes);
        p.payloadRead+=code.size()+scale.size();require(std::find(code.begin(),code.end(),uint8_t{128})==code.end(),"excluded signed symmetric code -128");
        for(uint32_t n=0;n<width;++n){float value;std::memcpy(&value,scale.data()+4*n,4);require(value>0 && std::isfinite(value),"positive finite persisted F32 row scale");}
        std::memcpy(static_cast<uint8_t *>(bank.codes[plane].contents())+rank*codeBytes,code.data(),code.size());
        std::memcpy(static_cast<uint8_t *>(bank.scales[plane].contents())+rank*scaleBytes,scale.data(),scale.size());
        require(CC_SHA256_Update(&hash,code.data(),CC_LONG(code.size())) && CC_SHA256_Update(&hash,scale.data(),CC_LONG(scale.size())),"fixture SHA256 update");
      }
    }
    std::array<uint8_t,32> digest{};require(CC_SHA256_Final(digest.data(),&hash),"fixture SHA256 final");p.fixture=hexDigest(digest);
    struct stat after{};require(!::fstat(fd,&after) && sameFile(bank.before,after),"coefficient source stat unchanged during bounded reads");
  } catch(...) {::close(fd);throw;}::close(fd);require(p.payloadRead==FixtureBytes,"exact ten-expert 49,305,600-byte coefficient reads");return bank;
}
void sourceUnchanged(const Bank &bank) {
  const int fd=::open(bank.source.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,"readonly final source-stat open");struct stat after{};
  const bool good=!::fstat(fd,&after) && sameFile(bank.before,after);::close(fd);require(good,"selected coefficient source metadata unchanged");
}
struct Output final {MetalBuffer act,down,diagnostics;std::array<MetalBuffer,8> g;std::array<MetalBuffer,3> d;MetalBuffer guCensus,downCensus;};
Output output(MetalBackend &backend,std::vector<Allocation> &all) {
  Output out;out.act=allocate(backend,Routes*640*2,all);out.down=allocate(backend,Routes*2560*2,all);out.diagnostics=allocate(backend,4,all);
  for(uint32_t i=0;i<8;++i)out.g[i]=allocate(backend,Routes*640*(i<4?4:2),all);
  for(uint32_t i=0;i<3;++i)out.d[i]=allocate(backend,Routes*2560*(i<2?4:2),all);
  out.guCensus=allocate(backend,100*4,all);out.downCensus=allocate(backend,400*4,all);return out;
}
void poison(const Output &out,bool taps) {
  std::memset(out.act.contents(),0xff,out.act.sizeBytes());std::memset(out.down.contents(),0xff,out.down.sizeBytes());resetDiag(out.diagnostics);
  if(taps){for(auto b:out.g)std::memset(b.contents(),0xff,b.sizeBytes());for(auto b:out.d)std::memset(b.contents(),0xff,b.sizeBytes());}
  std::memset(out.guCensus.contents(),0,out.guCensus.sizeBytes());std::memset(out.downCensus.contents(),0,out.downCensus.sizeBytes());
}
void census(MetalBuffer b,uint32_t entries,uint32_t expected) {
  require(b.sizeBytes()==uint64_t(entries)*4,"exact CTA census extent");const auto *v=static_cast<const uint32_t *>(b.contents());
  for(uint32_t i=0;i<entries;++i)require(v[i]==expected,"actual logical CTA completion census");
}
enum class Kind:uint32_t { Original,Candidate,NativeTap,CandidateTap,NativeWrapper };
bool isCandidate(Kind kind){return kind==Kind::Candidate || kind==Kind::CandidateTap;}
bool isTap(Kind kind){return kind==Kind::NativeTap || kind==Kind::CandidateTap;}
void preflight(const Bank &bank,MetalBuffer x,MetalBuffer ranks,MetalBuffer ids,const Output &out,Kind kind,
    MetalBuffer downInput,DispatchSize gu,DispatchSize down,DispatchSize threads={128,1,1}) {
  fullGrid(isCandidate(kind),gu,down,threads);
  require(x.sizeBytes()==2560*2 && ranks.sizeBytes()==512*4 && ids.sizeBytes()==10*8 &&
      out.act.sizeBytes()==10*640*2 && out.down.sizeBytes()==10*2560*2 && out.diagnostics.sizeBytes()==4 &&
      downInput.sizeBytes()==10*640*2,"exact bounded R1 primitive host extents");
  boundedRanks(std::span<const uint32_t>(static_cast<const uint32_t *>(ranks.contents()),512),bank.expertBase);
  std::vector<MetalBuffer> immutable{x,ranks,ids};if(!downInput.sameView(out.act))immutable.push_back(downInput);
  for(uint32_t i=0;i<3;++i){require(bank.codes[i].sizeBytes()==10*1638400 && bank.scales[i].sizeBytes()==uint64_t(i==2?2560:640)*10*4,"exact ten-rank original coefficient banks");immutable.push_back(bank.codes[i]);immutable.push_back(bank.scales[i]);}
  std::vector<MetalBuffer> writes{out.act,out.down,out.diagnostics};
  if(isTap(kind)){writes.insert(writes.end(),out.g.begin(),out.g.end());writes.insert(writes.end(),out.d.begin(),out.d.end());writes.push_back(out.guCensus);writes.push_back(out.downCensus);}
  for(size_t i=0;i<writes.size();++i){for(auto b:immutable)disjoint(writes[i],b);for(size_t j=0;j<i;++j)disjoint(writes[i],writes[j]);}
}
struct Plan final {CommandGraph owner;std::vector<ComputeDispatch> commands;};
Plan graph(const Bank &bank,MetalBuffer x,MetalBuffer ranks,MetalBuffer ids,const Output &out,Kind kind,
    FlashGatheredMPPParams params=Params,MetalBuffer sharedDownInput={}) {
  const bool candidate=isCandidate(kind),tap=isTap(kind);const auto downInput=sharedDownInput?sharedDownInput:out.act;
  preflight(bank,x,ranks,ids,out,kind,downInput,gateGrid(candidate),downGrid(candidate));
  const char *guName="flash_gathered_mpp_gate_up_m16_n64_sg4",*downName="flash_gathered_mpp_down_m16_n64_sg4";
  if(kind==Kind::Candidate){guName="r1_duplicate_tid0_gu_candidate";downName="r1_duplicate_tid0_down_candidate";}
  if(kind==Kind::NativeWrapper){guName="r1_duplicate_tid0_gu_native";downName="r1_duplicate_tid0_down_native";}
  if(kind==Kind::NativeTap){guName="r1_duplicate_tid0_gu_native_tap";downName="r1_duplicate_tid0_down_native_tap";}
  if(kind==Kind::CandidateTap){guName="r1_duplicate_tid0_gu_candidate_tap";downName="r1_duplicate_tid0_down_candidate_tap";}
  Plan plan;plan.owner.add(guName,{x,bank.codes[0],bank.scales[0],bank.codes[1],bank.scales[1],ranks,ids,out.act,out.diagnostics},params,gateGrid(candidate),{128,1,1});
  plan.owner.add(downName,{downInput,bank.codes[2],bank.scales[2],ranks,ids,out.down,out.diagnostics},params,downGrid(candidate),{128,1,1});
  for(const auto &d:plan.owner.dispatches())plan.commands.push_back(d);
  require(plan.commands.size()==2 && plan.commands[0].bytes.size()==1 && plan.commands[0].bytes[0].index==9 &&
      plan.commands[1].bytes.size()==1 && plan.commands[1].bytes[0].index==7,"original two-dispatch 16-byte inline ABI");
  if(tap){for(uint32_t i=0;i<8;++i)plan.commands[0].buffers.push_back({10+i,out.g[i]});plan.commands[0].buffers.push_back({18,out.guCensus});
    for(uint32_t i=0;i<3;++i)plan.commands[1].buffers.push_back({8+i,out.d[i]});plan.commands[1].buffers.push_back({11,out.downCensus});}
  return plan;
}
CommandTiming submit(MetalBackend &backend,std::span<const ComputeDispatch> commands) {
  const auto result=backend.submitCommand(commands);
  require(std::isfinite(result.gpuSeconds) && result.gpuSeconds>0 && std::isfinite(result.wallSeconds) && result.wallSeconds>0,"finite positive GPU/wall command timestamps");return result;
}
void allImmutable(const std::vector<MetalBuffer> &buffers,const std::vector<std::vector<uint8_t>> &saved) {
  require(buffers.size()==saved.size(),"immutable snapshot count");for(size_t i=0;i<buffers.size();++i)require(unchanged(buffers[i],saved[i]),"coefficients/input/IDs/ranks/frozen activation immutable");
}
void qualify(const std::array<Output,5> &outputs,Progress &p) {
  for(const auto &out:outputs){require(diag(out.diagnostics)==Sticky,"healthy sticky diagnostics");finite(out.act,true);finite(out.down,true);}
  constexpr std::array<const char *,8> guStages{"GU raw gate F32","GU scaled gate F32","GU raw up F32","GU scaled up F32",
      "GU gate BF16","GU up BF16","GU SiLU BF16","GU SwiGLU activation BF16"};
  constexpr std::array<const char *,3> downStages{"down raw dot F32","down scaled dot F32","down BF16"};
  // Raw dots are checked first so an arithmetic failure identifies its stage
  // before a later activation rounding mismatch obscures the first cause.
  for(uint32_t i=0;i<8;++i){equal(outputs[2].g[i],outputs[3].g[i],p,guStages[i]);finite(outputs[2].g[i],i>=4);finite(outputs[3].g[i],i>=4);}
  for(uint32_t i=0;i<3;++i){equal(outputs[2].d[i],outputs[3].d[i],p,downStages[i]);finite(outputs[2].d[i],i>=2);finite(outputs[3].d[i],i>=2);}
  for(uint32_t i=1;i<5;++i){equal(outputs[0].act,outputs[i].act,p,"own-chain variant"+std::to_string(i)+" activation BF16");equal(outputs[0].down,outputs[i].down,p,"own-chain variant"+std::to_string(i)+" down BF16");}
  for(uint32_t i=2;i<4;++i){equal(outputs[i].act,outputs[i].g[7],p,"shipping activation versus activation tap");equal(outputs[i].down,outputs[i].d[2],p,"shipping down versus down tap");census(outputs[i].guCensus,100,1);census(outputs[i].downCensus,400,1);}
}
gemv_quality::MetricReport routeNumericGuard(MetalBuffer reference,MetalBuffer candidate,uint32_t width) {
  require(reference.sizeBytes()==uint64_t(Routes)*width*2 && candidate.sizeBytes()==reference.sizeBytes(),"per-route numeric guard extents");
  const auto *a=static_cast<const uint16_t *>(reference.contents()),*b=static_cast<const uint16_t *>(candidate.contents());
  static_assert(gemv_quality::kMaximumRelativeL2==1e-4 && gemv_quality::kMinimumCosine==.999999);
  const auto result=gemv_quality::metrics(std::span<const uint16_t>(a,Routes*width),std::span<const uint16_t>(b,Routes*width),width);
  require(result.pass(),"frozen original global/per-route finite relative-L21e-4/cosine.999999 guard");return result;
}
void ledger(MetalBackend &backend,engine::MemoryGovernor &gov,Progress &p) {
  const auto stats=backend.memoryStats();p.actualAllocated=allocationDelta(p.initialAllocated,stats.allocatedBytes);
  p.peakAllocated=allocationDelta(p.initialAllocated,stats.peakAllocatedBytes);
  p.deviceCurrentDelta=allocationDelta(p.initialDevice,stats.deviceCurrentAllocatedBytes);
  p.devicePeakDelta=allocationDelta(p.initialDevice,stats.devicePeakAllocatedBytes);
  const auto state=gov.snapshot();p.denied=state.deniedReservations;p.hostValid=state.hostMeasurementValid;p.growth=state.growthAllowed;
  require(p.actualAllocated<=p.planned && p.peakAllocated<=p.planned && p.deviceCurrentDelta<=p.planned && p.devicePeakDelta<=p.planned,"owned and real sampled current/peak deltas within reserved 128MiB");
  require(!stats.sparseVirtualBytes && !stats.sparseResidentBytes && !stats.peakSparseResidentBytes,"no sparse backing");
  require(!p.denied && p.hostValid && p.growth && state.pressure==engine::MemoryPressure::Normal && state.systemPressure==engine::MemoryPressure::Normal && state.hostAvailableBytes>state.hostReserveBytes,"normal governor/no denials/valid host growth");
  require(backend.healthy(),"healthy backend ledger boundary");
}
struct TimingSet final {std::array<std::vector<CommandTiming>,2> samples;std::array<double,2> warmGPU{};std::array<uint32_t,2> warmCommands{};};
double median(const std::vector<CommandTiming> &values,bool gpu) {
  require(values.size()==Pairs,"18 balanced samples each");std::vector<double> sorted;for(const auto &t:values)sorted.push_back(gpu?t.gpuSeconds:t.wallSeconds);
  std::sort(sorted.begin(),sorted.end());return (sorted[8]+sorted[9])/2;
}
void timings(std::ostream &out,const std::vector<CommandTiming> &values,bool gpu) {
  out<<'[';for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<(gpu?values[i].gpuSeconds:values[i].wallSeconds)*1000;}out<<']';
}
uint64_t cpu() {
  uint64_t checks=0;require(gemv_quality::cpuSelfTest(),"qualified frozen global/per-route quality selftest");++checks;
  require(bf(1.00390625f)==0x3f80 && bf(1.01171875f)==0x3f82 && bf(-0.0f)==0x8000,"BF16 RNE/signed-zero goldens");++checks;
  require(OneExpertBytes==3*1638400+uint64_t(640+640+2560)*4 && FixtureBytes==10*OneExpertBytes,"exact ten-expert bounded coefficient byte golden");++checks;
  for(bool permuted:{false,true}){auto ids=routeIDs(0,permuted);auto sorted=ids;std::sort(sorted.begin(),sorted.end());for(uint32_t i=0;i<10;++i){require(sorted[i]==i,"ten distinct original experts/permuted routes");++checks;}}
  const auto hidden=hiddenFixture();const double rms=rowRMS(hidden);require(rms>.999 && rms<1.001,"normalized true-row-RMS BF16 fixture");++checks;
  for(bool candidate:{false,true}) {
    fullGrid(candidate,gateGrid(candidate),downGrid(candidate));++checks;
    for(uint32_t stage=0;stage<2;++stage){std::vector<uint32_t> seen(stage?400:100,0);const auto physical=stage?downGrid(candidate):gateGrid(candidate);
      for(uint32_t z=0;z<physical.z;++z)for(uint32_t x=0;x<physical.x;++x){const uint32_t logicalX=x,logicalZ=z;
        require(logicalX<(stage?40u:10u) && logicalZ<10,"original logical producer grid bounds");++seen[logicalZ*(stage?40:10)+logicalX];}
      require(std::all_of(seen.begin(),seen.end(),[](uint32_t value){return value==1;}),"CPU whole-grid CTA bijection");checks+=seen.size();}
    for(uint32_t test=0;test<6;++test){auto gu=gateGrid(candidate),down=downGrid(candidate);DispatchSize threads{128,1,1};
      if(test==0)--gu.x;if(test==1)++gu.z;if(test==2)--down.x;if(test==3)++down.z;if(test==4)threads.x=64;if(test==5)threads.y=2;
      bool rejected=false;try{fullGrid(candidate,gu,down,threads);}catch(const std::runtime_error &){rejected=true;}require(rejected,"CPU host partial/excess/threads rejection");++checks;}
  }
  bool rejected=false;try{(void)routeIDs(503,false);}catch(const std::runtime_error &){rejected=true;}require(rejected,"CPU expert selection bound");++checks;
  std::array<uint32_t,512> ranks{};ranks.fill(UINT32_MAX);for(uint32_t i=0;i<10;++i)ranks[i]=i;boundedRanks(ranks,0);++checks;
  for(auto invalid:{10u,511u}){ranks[0]=invalid;rejected=false;try{boundedRanks(ranks,0);}catch(const std::runtime_error &){rejected=true;}require(rejected,"CPU finite out-of-fixture rank refused");++checks;}
  for(auto safe:{512u,UINT32_MAX}){ranks[0]=safe;boundedRanks(ranks,0);++checks;}return checks;
}
} // namespace

int main(int argc,char **argv) { @autoreleasepool { Progress p;try {
  if(argc==2 && std::string_view(argv[1])=="--cpu-self-test") {std::cout<<"{\"valid\":true,\"checks\":"<<cpu()<<",\"GPU_work\":false,\"payload_reads\":false,\"native_GU_CTAs\":100,\"native_down_CTAs\":400}\n";return 0;}
  if(argc==2 && std::string_view(argv[1])=="--help") {std::cout<<"Root-only: oracle METALLIB FULL512_STORE FRESH_REPORT [LAYER0..47] [EXPERT_BASE0..502] [normal|permuted]\n";return 0;}
  require(argc>=4 && argc<=7,"oracle arguments");require(!std::filesystem::exists(argv[3]) && !std::filesystem::exists(std::string(argv[3])+".writing") && !std::filesystem::exists(std::string(argv[3])+".final-writing"),"fresh report required");
  p.path=argv[3];p.write(false,false);const uint32_t layer=argc>4?numeric(argv[4],47):0,base=argc>5?numeric(argv[5],502):0;
  const std::string_view order=argc>6?argv[6]:"normal";require(order=="normal" || order=="permuted","normal/permuted route order");const bool permuted=order=="permuted";
  p.phase="metadata-only";const auto metadata=loadFlashInt8ExpertStoreMetadata(argv[2],"ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e","edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);
  require(metadata.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","registered canonical Full512 metadata identity");
  std::array<TimingSet,1> measured;std::array<gemv_quality::MetricReport,2> initialQuality,finalQuality;
  const auto selectedIDs=routeIDs(base,permuted);double inputRMS=0;uint64_t timedSubmissions=0;
  {BackendLifetime lifetime{p};p.phase="backend-normal-governor-admission";MetalBackend backend(argv[1]);p.backendInitialized=true;
    const auto initial=backend.memoryStats();p.initialAllocated=initial.allocatedBytes;p.initialDevice=initial.deviceCurrentAllocatedBytes;p.library=hexDigest(backend.metallibSha256());require(!p.initialAllocated,"no owned model allocation at backend baseline");
    const uint64_t physical=NSProcessInfo.processInfo.physicalMemory;p.hostReserve=std::max<uint64_t>(16ULL<<30,physical/10);require(physical>p.hostReserve,"host reserve protected");
    engine::MemoryGovernor gov(backend,physical-p.hostReserve,p.hostReserve);const auto beforeAdmission=gov.snapshot();
    require(beforeAdmission.hostMeasurementValid && beforeAdmission.growthAllowed && beforeAdmission.pressure==engine::MemoryPressure::Normal && beforeAdmission.systemPressure==engine::MemoryPressure::Normal && !beforeAdmission.deniedReservations,"normal valid growth before 128MiB admission");
    auto lease=gov.tryReserve(p.planned);require(bool(lease) && !gov.snapshot().deniedReservations,"normal 128MiB reserve before any fixture allocation");
    {std::vector<Allocation> allocations;p.phase="Root-only-ten-selected-readonly-slices";const auto bank=loadBank(backend,metadata.layers[layer],base,allocations,p);
      const auto x=allocate(backend,2560*2,allocations),ranks=allocate(backend,512*4,allocations),ids=allocate(backend,10*8,allocations);
      const auto hidden=hiddenFixture();inputRMS=rowRMS(hidden);std::memcpy(x.contents(),hidden.data(),x.sizeBytes());std::memcpy(ids.contents(),selectedIDs.data(),ids.sizeBytes());
      auto *rankData=static_cast<uint32_t *>(ranks.contents());std::fill_n(rankData,512,UINT32_MAX);for(uint32_t rank=0;rank<10;++rank)rankData[base+rank]=rank;
      for(uint32_t slot=0;slot<10;++slot)require(rankData[selectedIDs[slot]]<10,"healthy fixture stays within ten packed coefficient ranks");
      p.inputHash=qmv_one_layer::detail::hash(x.contents(),x.sizeBytes());p.idsHash=qmv_one_layer::detail::hash(ids.contents(),ids.sizeBytes());p.ranksHash=qmv_one_layer::detail::hash(ranks.contents(),ranks.sizeBytes());
      std::array<Output,5> outputs{output(backend,allocations),output(backend,allocations),output(backend,allocations),output(backend,allocations),output(backend,allocations)};
      std::array<Plan,5> plans{graph(bank,x,ranks,ids,outputs[0],Kind::Original),graph(bank,x,ranks,ids,outputs[1],Kind::Candidate),graph(bank,x,ranks,ids,outputs[2],Kind::NativeTap),graph(bank,x,ranks,ids,outputs[3],Kind::CandidateTap),graph(bank,x,ranks,ids,outputs[4],Kind::NativeWrapper)};
      std::vector<MetalBuffer> immutable{x,ranks,ids};for(uint32_t i=0;i<3;++i){immutable.push_back(bank.codes[i]);immutable.push_back(bank.scales[i]);}
      std::vector<std::vector<uint8_t>> saved;for(auto buffer:immutable)saved.push_back(copied(buffer));ledger(backend,gov,p);lease->commit();
      p.phase="original-AIR-ownchain-taps-and-actual-completion-census";
      for(uint32_t i=0;i<5;++i){poison(outputs[i],i==2 || i==3);p.gpu=true;(void)submit(backend,plans[i].commands);}qualify(outputs,p);
      initialQuality={routeNumericGuard(outputs[0].act,outputs[1].act,640),routeNumericGuard(outputs[0].down,outputs[1].down,2560)};allImmutable(immutable,saved);canaries(allocations);
      p.phase="literal-malformed-sticky-localA-source-preservation";
      for(uint32_t test=0;test<6;++test) {
        std::memcpy(x.contents(),saved[0].data(),saved[0].size());std::memcpy(ranks.contents(),saved[1].data(),saved[1].size());std::memcpy(ids.contents(),saved[2].data(),saved[2].size());
        auto *xb=static_cast<uint16_t *>(x.contents());auto *id=static_cast<int64_t *>(ids.contents());
        if(test==0)xb[0]=0x7fc1;if(test==1)xb[127]=0x7f80;if(test==2){id[0]=-1;xb[0]=0x7fc1;}if(test==3)rankData[selectedIDs[0]]=512;if(test==4)id[0]=id[1];if(test==5)id[0]=512;
        const auto changedX=copied(x),changedRanks=copied(ranks),changedIDs=copied(ids);
        for(uint32_t i=0;i<5;++i){p.check="malformed case="+std::to_string(test)+" kind="+std::to_string(i);poison(outputs[i],i==2 || i==3);(void)submit(backend,plans[i].commands);}
        for(uint32_t i=1;i<5;++i){equal(outputs[0].act,outputs[i].act,p,"malformed activation bits/NaN policy");equal(outputs[0].down,outputs[i].down,p,"malformed down bits/NaN policy");require(diag(outputs[i].diagnostics)==diag(outputs[0].diagnostics),"same malformed sticky bits");}
        for(uint32_t stage=0;stage<8;++stage)equal(outputs[2].g[stage],outputs[3].g[stage],p,"malformed case="+std::to_string(test)+" GU raw/scaled/BF16 stage="+std::to_string(stage));
        for(uint32_t stage=0;stage<3;++stage)equal(outputs[2].d[stage],outputs[3].d[stage],p,"malformed case="+std::to_string(test)+" down raw/scaled/BF16 stage="+std::to_string(stage));
        const uint32_t expected=test<2?Sticky|4:test<4?Sticky|5:test==4?Sticky|1:Sticky|5;require(diag(outputs[0].diagnostics)==expected,"literal original malformed diagnostic policy");
        require(unchanged(x,changedX) && unchanged(ranks,changedRanks) && unchanged(ids,changedIDs),"nonfinite localA does not mutate source/route/rank operands");canaries(allocations);++p.guardCases;
      }
      std::memcpy(x.contents(),saved[0].data(),saved[0].size());std::memcpy(ranks.contents(),saved[1].data(),saved[1].size());std::memcpy(ids.contents(),saved[2].data(),saved[2].size());
      p.phase="literal-zero-negative-NaN-Inf-row-scale-policy";
      constexpr std::array<uint32_t,4> badScaleBits{0x00000000u,0xbf800000u,0x7fc00000u,0x7f800000u};
      for(uint32_t plane=0;plane<3;++plane)for(auto bits:badScaleBits) {
        const uint64_t scaleIndex=uint64_t(rankData[selectedIDs[0]])*(plane==2?2560:640);
        std::memcpy(static_cast<uint8_t *>(bank.scales[plane].contents())+scaleIndex*4,&bits,4);
        const auto changedScale=copied(bank.scales[plane]);
        for(uint32_t i=0;i<5;++i){p.check="bad-scale plane="+std::to_string(plane)+" bits="+std::to_string(bits)+" kind="+std::to_string(i);poison(outputs[i],i==2 || i==3);(void)submit(backend,plans[i].commands);}
        for(uint32_t i=1;i<5;++i){equal(outputs[0].act,outputs[i].act,p,"bad-scale activation exact original bits");equal(outputs[0].down,outputs[i].down,p,"bad-scale down exact original bits");require(diag(outputs[i].diagnostics)==diag(outputs[0].diagnostics),"same bad-scale sticky bits");}
        for(uint32_t stage=0;stage<8;++stage)equal(outputs[2].g[stage],outputs[3].g[stage],p,"bad-scale GU raw/scaled/BF16 stage="+std::to_string(stage));
        for(uint32_t stage=0;stage<3;++stage)equal(outputs[2].d[stage],outputs[3].d[stage],p,"bad-scale down raw/scaled/BF16 stage="+std::to_string(stage));
        require(diag(outputs[0].diagnostics)==(Sticky|4) && unchanged(bank.scales[plane],changedScale),"literal bad-scale sticky4/source preserved");
        std::memcpy(bank.scales[plane].contents(),saved[4+2*plane].data(),bank.scales[plane].sizeBytes());canaries(allocations);++p.guardCases;
      }
      p.phase="reserved-param-and-thread-reject-before-output-write";
      for(uint32_t test=0;test<2;++test)for(uint32_t i=0;i<5;++i) {
        auto bad=Params;if(test==0)bad.reserved=1;const Kind kind=static_cast<Kind>(i);auto guardedPlan=graph(bank,x,ranks,ids,outputs[i],kind,bad);
        if(test==1)for(auto &d:guardedPlan.commands)d.threadsPerThreadgroup={64,1,1};
        const auto act=copied(outputs[i].act),down=copied(outputs[i].down);resetDiag(outputs[i].diagnostics);(void)submit(backend,guardedPlan.commands);
        require(diag(outputs[i].diagnostics)==(Sticky|2) && unchanged(outputs[i].act,act) && unchanged(outputs[i].down,down),"all variants reject reserved/thread params without output mutation");++p.guardCases;
      }
      p.phase="host-fullgrid-alias-submission-count-guards";
      for(uint32_t variant=0;variant<2;++variant)for(uint32_t test=0;test<7;++test) {
        const auto before=backend.submissionCount();bool rejected=false;auto out=outputs[variant];const auto kind=variant?Kind::Candidate:Kind::Original;
        auto gu=gateGrid(variant),down=downGrid(variant);DispatchSize threads{128,1,1};
        if(test==0)--gu.x;if(test==1)++gu.z;if(test==2)--down.x;if(test==3)++down.z;if(test==4)threads.x=64;
        if(test==5)out.act=backend.view(bank.codes[0],0,out.act.sizeBytes());if(test==6)out.act=backend.view(out.down,0,out.act.sizeBytes());
        try{preflight(bank,x,ranks,ids,out,kind,out.act,gu,down,threads);}catch(const std::runtime_error &){rejected=true;}
        require(rejected && before==backend.submissionCount(),"host fullgrid/alias/extents rejected before submission");++p.guardCases;
      }
      {const auto before=backend.submissionCount();rankData[selectedIDs[0]]=10;bool rejected=false;
        try{preflight(bank,x,ranks,ids,outputs[0],Kind::Original,outputs[0].act,gateGrid(false),downGrid(false));}catch(const std::runtime_error &e){require(std::string_view(e.what())=="host rejects finite out-of-fixture rank before submission","specific bounded-rank host guard");rejected=true;}
        require(rejected && before==backend.submissionCount(),"out-of-fixture rank host rejected without submit");std::memcpy(ranks.contents(),saved[1].data(),saved[1].size());++p.guardCases;}
      p.phase="private-wholegrid-safety-not-original-partial-grid-policy";
      for(uint32_t variant=2;variant<4;++variant)for(uint32_t stage=0;stage<2;++stage)for(uint32_t excess=0;excess<2;++excess) {
        poison(outputs[variant],true);const auto act=copied(outputs[variant].act),down=copied(outputs[variant].down);
        auto d=plans[variant].commands[stage];if(excess)++d.threadgroups.x;else --d.threadgroups.x;
        (void)submit(backend,std::span<const ComputeDispatch>(&d,1));require(diag(outputs[variant].diagnostics)==(Sticky|2) && unchanged(outputs[variant].act,act) && unchanged(outputs[variant].down,down),"private wholegrid guard rejects without output writes");
        census(outputs[variant].guCensus,100,0);census(outputs[variant].downCensus,400,0);canaries(allocations);++p.guardCases;
      }
      p.phase="final-qualification-and-freeze-before-warm";
      for(uint32_t i=0;i<5;++i){poison(outputs[i],i==2 || i==3);(void)submit(backend,plans[i].commands);}qualify(outputs,p);
      allImmutable(immutable,saved);sourceUnchanged(bank);canaries(allocations);ledger(backend,gov,p);resetDiag(outputs[0].diagnostics);resetDiag(outputs[1].diagnostics);
      std::array<std::array<std::span<const ComputeDispatch>,2>,1> commands{{
        {std::span<const ComputeDispatch>(plans[0].commands),std::span<const ComputeDispatch>(plans[1].commands)}
      }};
      p.phase="inclusive-ownchain2-vs2-read-free-150mswarm-and18-balanced-pairs";
      const auto beforeTiming=backend.submissionCount();
      // From here through the final timed submit there are no .contents(),
      // guards, diagnostics, output scans, resets, snapshots, or allocator reads.
      for(uint32_t mode=0;mode<1;++mode) {
        for(uint32_t variant=0;variant<2;++variant)while(measured[mode].warmGPU[variant]<.150) {
          require(measured[mode].warmCommands[variant]<100000,"bounded warm timestamps available");
          const auto t=submit(backend,commands[mode][variant]);measured[mode].warmGPU[variant]+=t.gpuSeconds;++measured[mode].warmCommands[variant];
        }
        for(uint32_t pair=0;pair<Pairs;++pair)for(uint32_t position=0;position<2;++position) {
          const uint32_t variant=(pair+position)%2;measured[mode].samples[variant].push_back(submit(backend,commands[mode][variant]));
        }
      }
      timedSubmissions=backend.submissionCount()-beforeTiming;uint64_t expectedSubmissions=Pairs*2;
      for(const auto &mode:measured)for(auto count:mode.warmCommands)expectedSubmissions+=count;
      require(timedSubmissions==expectedSubmissions,"warm/timed submission count exact");
      p.phase="LAST-shipping-bits-and-final-taps";allImmutable(immutable,saved);canaries(allocations);
      require(diag(outputs[0].diagnostics)==Sticky && diag(outputs[1].diagnostics)==Sticky,"LAST shipping healthy sticky diagnostics");
      equal(outputs[0].act,outputs[1].act,p,"LAST ownchain activation BF16 bits");equal(outputs[0].down,outputs[1].down,p,"LAST ownchain down BF16 bits");finite(outputs[0].act,true);finite(outputs[0].down,true);
      finalQuality={routeNumericGuard(outputs[0].act,outputs[1].act,640),routeNumericGuard(outputs[0].down,outputs[1].down,2560)};
      const auto lastAct=copied(outputs[0].act),lastDown=copied(outputs[0].down);
      for(uint32_t i=2;i<5;++i){poison(outputs[i],i==2 || i==3);(void)submit(backend,plans[i].commands);require(unchanged(outputs[i].act,lastAct) && unchanged(outputs[i].down,lastDown),"final taps/native selfcontrol bound to actual LAST shipping snapshots");}
      qualify(outputs,p);
      allImmutable(immutable,saved);sourceUnchanged(bank);canaries(allocations);ledger(backend,gov,p);
      for(const auto &mode:measured)for(uint32_t variant=0;variant<2;++variant){require(mode.warmGPU[variant]>=.150 && mode.samples[variant].size()==Pairs,"all component variants warmed and18sample complete");}
    }
    p.phase="fixtures-and-graphs-release-to-baseline";p.afterAllocated=allocationDelta(p.initialAllocated,backend.memoryStats().allocatedBytes);
    require(!p.afterAllocated,"all fixture/graph buffers released to baseline before backend stop");
    require(!gov.snapshot().deniedReservations && backend.healthy(),"healthy release/no governor denials");backend.stop();p.backendStopped=true;
    require(backend.memoryStats().allocatedBytes==0,"backend stop retains zero fixture ledger");
  }
  require(p.backendStopped && p.backendDestroyed,"backend stop/destructor before success publication");p.phase="complete";
  const auto final=p.path.string()+".final-writing";std::ofstream out(final);require(bool(out),"final report open");out<<std::setprecision(17);
  out<<"{\"schema\":\"expert-r1-duplicate-tid0-component-v1\",\"pass\":true,\"qualification_complete\":true,\"GPU_executed\":true"
     <<",\"layer\":"<<layer<<",\"expert_base\":"<<base<<",\"route_order\":"<<json::quote(order)<<",\"route_ids\":[";
  for(uint32_t i=0;i<10;++i){if(i)out<<',';out<<selectedIDs[i];}
  out<<"],\"rows\":1,\"selections\":10,\"ten_distinct_original_experts_packed_ranks\":true,\"synthetic_normalized_input\":true,\"actual_current_capture\":false"
     <<",\"input_row_rms\":"<<inputRMS<<",\"coefficient_fixture_bytes\":"<<p.payloadRead<<",\"selected_coefficient_sha256\":"<<json::quote(p.fixture)
     <<",\"input_sha256\":"<<json::quote(p.inputHash)<<",\"route_ids_sha256\":"<<json::quote(p.idsHash)<<",\"ranks_sha256\":"<<json::quote(p.ranksHash)
     <<",\"metallib_sha256\":"<<json::quote(p.library)<<",\"wholemodel_speed_qualified\":false,\"model_quality_qualified\":false"
     <<",\"Full512_payload_mapped_or_constructed\":false,\"full_raw_scaled_F32_and_BF16_gate_up_SwiGLU_down_exact\":true"
     <<",\"two_ownchain_complete_BF16_exact\":true,\"posttiming_taps_bound_to_LAST_shipping_snapshots\":true"
     <<",\"valid_duplicate_raw_stage_parity\":true,\"malformed_invalid_rank_raw_fields_unwritten_sentinel\":true,\"malformed_tap_byte_parity_is_numerical_qualification\":false"
     <<",\"original_global_and_per_route_quality_guards_retained\":true,\"maximum_relative_L2\":1e-4,\"minimum_cosine\":0.999999,\"quality_source\":\"dev/benchmarks/gemv_decode_sep21_v1b/quality.hpp\""
     <<",\"healthy_untimed_actual_GU_completion_cells\":100,\"healthy_untimed_actual_down_completion_cells\":400"
     <<",\"original_untapped_AIR_is_only_performance_control\":true,\"private_native_wrapper_untimed_selfcontrol_only\":true"
     <<",\"gate_native_grid\":[10,1,10],\"gate_candidate_grid\":[10,1,10],\"down_native_grid\":[40,1,10],\"down_candidate_grid\":[40,1,10]"
     <<",\"logical_axes\":[\"physical.x\",\"physical.y\",\"physical.z\"],\"producer_threads\":128"
     <<",\"guard_cases\":"<<p.guardCases<<",\"checks\":"<<p.checks<<",\"bytes_compared\":"<<p.bytes
     <<",\"timing_CPU_buffer_reads\":false,\"sample_pairs_each_component\":18,\"balanced_pair_positions\":true,\"minimum_warm_GPU_seconds_each_variant_each_component\":0.15"
     <<",\"warm_and_timed_submission_count\":"<<timedSubmissions<<",\"initial_activation_quality\":";initialQuality[0].json(out);
  out<<",\"initial_down_quality\":";initialQuality[1].json(out);out<<",\"LAST_shipping_activation_quality\":";finalQuality[0].json(out);
  out<<",\"LAST_shipping_down_quality\":";finalQuality[1].json(out);out<<",\"components\":[";
  constexpr std::array<const char *,1> modes{"two-dispatch-ownchain"};
  for(uint32_t mode=0;mode<1;++mode){if(mode)out<<',';out<<"{\"kind\":"<<json::quote(modes[mode])<<",\"control_warm_GPU_ms\":"<<measured[mode].warmGPU[0]*1000
      <<",\"candidate_warm_GPU_ms\":"<<measured[mode].warmGPU[1]*1000<<",\"control_warm_commands\":"<<measured[mode].warmCommands[0]<<",\"candidate_warm_commands\":"<<measured[mode].warmCommands[1]
      <<",\"control_GPU_ms\":";timings(out,measured[mode].samples[0],true);out<<",\"candidate_GPU_ms\":";timings(out,measured[mode].samples[1],true);
    out<<",\"control_wall_ms\":";timings(out,measured[mode].samples[0],false);out<<",\"candidate_wall_ms\":";timings(out,measured[mode].samples[1],false);
    out<<",\"control_GPU_median_ms\":"<<median(measured[mode].samples[0],true)*1000<<",\"candidate_GPU_median_ms\":"<<median(measured[mode].samples[1],true)*1000
       <<",\"control_wall_median_ms\":"<<median(measured[mode].samples[0],false)*1000<<",\"candidate_wall_median_ms\":"<<median(measured[mode].samples[1],false)*1000
       <<",\"ratio\":"<<median(measured[mode].samples[0],true)/median(measured[mode].samples[1],true)<<'}';}
  out<<"],\"all_immutable_buffers_source_metadata_and_canaries_intact\":true,\"allocation\":{\"planned_bytes\":"<<p.planned<<",\"actual_owned_delta_bytes\":"<<p.actualAllocated
     <<",\"peak_owned_delta_bytes\":"<<p.peakAllocated<<",\"current_device_delta_bytes\":"<<p.deviceCurrentDelta<<",\"peak_device_delta_bytes\":"<<p.devicePeakDelta
     <<",\"after_fixture_owned_delta_bytes\":"<<p.afterAllocated<<",\"denied_reservations\":"<<p.denied<<",\"host_reserve_bytes\":"<<p.hostReserve
     <<",\"host_measurement_valid\":"<<(p.hostValid?"true":"false")<<",\"growth_allowed\":"<<(p.growth?"true":"false")
     <<"},\"backend_stopped_with_zero_fixture_ledger\":true,\"backend_destroyed_before_publication\":true"
     <<",\"timing_scope\":\"bounded hot ten-expert synthetic R1 primitive only; component qualification, no whole-model or decode-throughput promotion\"}\n";
  out.close();require(bool(out),"final report close");std::filesystem::rename(final,p.path);return 0;
} catch(const std::exception &e){p.error=e.what();try{p.write(true,false);}catch(...){}std::cerr<<"R1 duplicate-tid0 oracle failed: "<<e.what()<<'\n';return 1;} } }
