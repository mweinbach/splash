// Root-only bounded RHS layout experiment. CPU mode opens no payload or GPU.
#include <Foundation/Foundation.h>
#include "flash/FlashGatheredMPP.hpp"
#include "flash/FlashInt8ExpertStoreMetadata.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/CommandGraph.hpp"
#include "dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp"
#include "packing.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
using namespace splash;
using namespace splash::flash;
using namespace splash::metal;
namespace {
constexpr uint32_t Rows=4,Routes=40;
constexpr uint64_t GuardBytes=16384,FixtureBytes=4930560;
constexpr FlashGatheredMPPParams Params{4,10,512,0};
void require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
uint16_t bf(float x){return uint16_t(std::bit_cast<uint32_t>(x)>>16);}
float f32(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
struct Progress {
 std::filesystem::path path;std::string phase="arguments",check,error,library,fixture;
 uint64_t checks=0,bytes=0,guards=0,payloadRead=0,first=UINT64_MAX;bool gpu=false;
 uint64_t planned=64ULL<<20,initialAllocated=0,actualAllocated=0,peakAllocated=0,afterAllocated=UINT64_MAX,denied=UINT64_MAX;
 bool backendInitialized=false,backendDestroyed=false;double oldMedian=0,newMedian=0;
 void write(bool complete,bool pass)const{
  if(path.empty())return;auto temp=path.string()+".writing";std::ofstream o(temp);
  require(bool(o),"progress open");o<<"{\"schema\":\"single-expert-RHS-tile64-progress-v1\",\"completed\":"<<(complete?"true":"false")
   <<",\"pass\":"<<(pass?"true":"false")<<",\"GPU_executed\":"<<(gpu?"true":"false")<<",\"phase\":"<<json::quote(phase)
   <<",\"check\":"<<json::quote(check)<<",\"error\":"<<json::quote(error)<<",\"metallib_sha256\":"<<json::quote(library)
   <<",\"fixture_sha256\":"<<json::quote(fixture)<<",\"Root_coefficient_payload_bytes_read\":"<<payloadRead<<",\"checks\":"<<checks
   <<",\"bytes_compared\":"<<bytes<<",\"guard_cases\":"<<guards<<",\"first_different_byte\":";
  if(first==UINT64_MAX)o<<"null";else o<<first;o<<",\"allocation\":{\"planned_bytes\":"<<planned<<",\"initial_bytes\":"<<initialAllocated<<",\"actual_fixture_bytes\":"<<actualAllocated<<",\"peak_fixture_bytes\":"<<peakAllocated<<",\"after_fixture_bytes\":"<<afterAllocated<<",\"Gov_denied_reservations\":"<<denied
   <<"},\"backend_initialized\":"<<(backendInitialized?"true":"false")<<",\"backend_destroyed\":"<<(backendDestroyed?"true":"false")<<"}\n";o.close();require(bool(o),"progress close");std::filesystem::rename(temp,path);
 }
};
struct BackendLifetime {Progress&p;~BackendLifetime(){if(p.backendInitialized)p.backendDestroyed=true;}};
struct Allocation {MetalBuffer base,view;};
Allocation allocate(MetalBackend&b,uint64_t bytes,std::vector<Allocation>&all){
 const uint64_t rounded=(bytes+GuardBytes-1)&~(GuardBytes-1);auto base=b.allocateBuffer(rounded+2*GuardBytes,BufferStorage::Shared,"bounded-rhs-tile64");
 std::memset(base.contents(),0xa5,base.sizeBytes());auto view=b.view(base,GuardBytes,bytes);std::memset(view.contents(),0,bytes);
 Allocation a{base,view};all.push_back(a);return a;
}
void canaries(const std::vector<Allocation>&all){for(const auto&a:all){const auto*p=static_cast<const uint8_t*>(a.base.contents());
 const uint64_t activeEnd=GuardBytes+a.view.sizeBytes();for(uint64_t i=0;i<GuardBytes;++i)require(p[i]==0xa5,"leading canary");
 for(uint64_t i=activeEnd;i<a.base.sizeBytes();++i)require(p[i]==0xa5,"trailing/padding canary");}}
void equal(MetalBuffer a,MetalBuffer b,Progress&p,const std::string&label){p.check=label;require(a.sizeBytes()==b.sizeBytes(),label+" extent");
 const auto*x=static_cast<const uint8_t*>(a.contents()),*y=static_cast<const uint8_t*>(b.contents());
 for(uint64_t i=0;i<a.sizeBytes();++i)if(x[i]!=y[i]){p.first=i;throw std::runtime_error(label+" byte mismatch");}
 ++p.checks;p.bytes+=a.sizeBytes();}
void finite(MetalBuffer b,bool half){if(half){for(uint64_t i=0;i<b.sizeBytes()/2;++i)require(std::isfinite(f32(static_cast<const uint16_t*>(b.contents())[i])),"nonfinite BF16 output");}
 else for(uint64_t i=0;i<b.sizeBytes()/4;++i)require(std::isfinite(static_cast<const float*>(b.contents())[i]),"nonfinite F32 tap");}
uint32_t diag(MetalBuffer b){uint32_t x;std::memcpy(&x,b.contents(),4);return x;}
void reset(MetalBuffer b){uint32_t v=0x80000000u;std::memcpy(b.contents(),&v,4);}
std::vector<uint8_t> copied(MetalBuffer b){const auto*p=static_cast<const uint8_t*>(b.contents());return{p,p+b.sizeBytes()};}
bool unchanged(MetalBuffer b,const std::vector<uint8_t>&x){return b.sizeBytes()==x.size()&&!std::memcmp(b.contents(),x.data(),x.size());}
void disjoint(MetalBuffer a,MetalBuffer b){const uintptr_t x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());
 require(a&&b&&x&&y,"addressable shared views");require(x<=y?uint64_t(y-x)>=a.sizeBytes():uint64_t(x-y)>=b.sizeBytes(),"mutable/immutable view overlap");}
struct Bank {std::array<MetalBuffer,3>old,newCodes,scales;};
std::vector<uint8_t> slice(int fd,uint64_t offset,uint64_t length){std::vector<uint8_t>v(length);uint64_t done=0;
 while(done<length){const auto n=::pread(fd,v.data()+done,length-done,off_t(offset+done));require(n>0,"bounded coefficient pread");done+=uint64_t(n);}return v;}
Bank load(MetalBackend&b,const FlashInt8ExpertStoreLayer&e,uint32_t expert,std::vector<Allocation>&alloc,Progress&p){
 qmv_one_layer::detail::validateEntry(e);require(expert<512,"expert index");int fd=::open(e.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,"readonly expert file open");
 struct stat before{},after{};require(!::fstat(fd,&before)&&S_ISREG(before.st_mode)&&!(before.st_mode&0222)&&uint64_t(before.st_size)==e.bytes,"readonly canonical Full512 file stat");
 Bank bank;std::vector<uint8_t> fixture;
 try{for(uint32_t plane=0;plane<3;++plane){uint32_t width=plane==2?2560:640,k=plane==2?640:2560;const auto bytes=uint64_t(width)*k;
  const auto code=slice(fd,e.codes[plane].offset+uint64_t(expert)*bytes,bytes);const auto scale=slice(fd,e.scales[plane].offset+uint64_t(expert)*width*4,uint64_t(width)*4);
  p.payloadRead+=code.size()+scale.size();fixture.insert(fixture.end(),code.begin(),code.end());fixture.insert(fixture.end(),scale.begin(),scale.end());
  require(std::find(code.begin(),code.end(),uint8_t{128})==code.end(),"excluded symmetric code-128");
  for(uint32_t i=0;i<width;++i){float f;std::memcpy(&f,scale.data()+4*i,4);require(std::isfinite(f)&&f>0,"positive finite F32 scale");}
  std::vector<int8_t>old(bytes);std::memcpy(old.data(),code.data(),bytes);const auto next=rhs_tile64::pack(old,width,k);
  bank.old[plane]=allocate(b,bytes,alloc).view;bank.newCodes[plane]=allocate(b,bytes,alloc).view;bank.scales[plane]=allocate(b,scale.size(),alloc).view;
  std::memcpy(bank.old[plane].contents(),code.data(),bytes);std::memcpy(bank.newCodes[plane].contents(),next.data(),bytes);std::memcpy(bank.scales[plane].contents(),scale.data(),scale.size());
  for(uint32_t n=0;n<width;++n)for(uint32_t kk=0;kk<k;++kk)require(old[rhs_tile64::oldIndex(width,k,n,kk)]==next[rhs_tile64::newIndex(width,k,n,kk)],"all original signed coefficient values equal");}
  require(!::fstat(fd,&after)&&before.st_dev==after.st_dev&&before.st_ino==after.st_ino&&before.st_size==after.st_size&&before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec&&before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec,"coefficient file changed during bounded read");
 }catch(...){::close(fd);throw;}::close(fd);require(p.payloadRead==FixtureBytes,"exact <5MB coefficient slice count");p.fixture=qmv_one_layer::detail::hash(fixture.data(),fixture.size());return bank;
}
struct Output {MetalBuffer act,down,diagnostics;std::array<MetalBuffer,8>g;std::array<MetalBuffer,3>d;};
Output output(MetalBackend&b,std::vector<Allocation>&all){Output o;o.act=allocate(b,Routes*640*2,all).view;o.down=allocate(b,Routes*2560*2,all).view;o.diagnostics=allocate(b,4,all).view;
 for(uint32_t i=0;i<8;++i)o.g[i]=allocate(b,Routes*640*(i<4?4:2),all).view;
 for(uint32_t i=0;i<3;++i)o.d[i]=allocate(b,Routes*2560*(i<2?4:2),all).view;return o;}
void preflight(const Bank&bank,MetalBuffer x,MetalBuffer ranks,MetalBuffer ids,const Output&o,bool tile,bool probe,DispatchSize gateGrid={10,4,10},DispatchSize downGrid={40,4,10}){
 require(gateGrid.x==10&&gateGrid.y==4&&gateGrid.z==10&&downGrid.x==40&&downGrid.y==4&&downGrid.z==10,"host rejects partial or excess grid before submission");
 require(x.sizeBytes()==Rows*2560*2&&ranks.sizeBytes()==512*4&&ids.sizeBytes()==Routes*8&&o.act.sizeBytes()==Routes*640*2&&o.down.sizeBytes()==Routes*2560*2&&o.diagnostics.sizeBytes()==4,"exact primitive host extents");
 std::vector<MetalBuffer>immutable{x,ranks,ids};for(uint32_t i=0;i<3;++i){require(bank.old[i].sizeBytes()==1638400&&bank.newCodes[i].sizeBytes()==1638400&&bank.scales[i].sizeBytes()==uint64_t(i==2?2560:640)*4,"exact oneexpert banks");immutable.push_back(bank.old[i]);immutable.push_back(bank.newCodes[i]);immutable.push_back(bank.scales[i]);}
 std::vector<MetalBuffer>writes{o.act,o.down,o.diagnostics};if(probe){writes.insert(writes.end(),o.g.begin(),o.g.end());writes.insert(writes.end(),o.d.begin(),o.d.end());}
 for(size_t a=0;a<writes.size();++a){for(auto r:immutable)disjoint(writes[a],r);for(size_t c=0;c<a;++c)disjoint(writes[a],writes[c]);}(void)tile;
}
CommandGraph graph(const Bank&bank,MetalBuffer x,MetalBuffer ranks,MetalBuffer ids,const Output&o,bool tile,bool probe,FlashGatheredMPPParams params=Params){
 preflight(bank,x,ranks,ids,o,tile,probe);const auto&w=tile?bank.newCodes:bank.old;CommandGraph g;
 std::vector<MetalBuffer>gate{x,w[0],bank.scales[0],w[1],bank.scales[1],ranks,ids,o.act,o.diagnostics};
 std::vector<MetalBuffer>down{o.act,w[2],bank.scales[2],ranks,ids,o.down,o.diagnostics};
 if(probe){gate.insert(gate.end(),o.g.begin(),o.g.end());down.insert(down.end(),o.d.begin(),o.d.end());}
 const std::string gateName=probe?(tile?"flash_expert_rhs_tile64_candidate_gate_up_probe_m16_n64_sg4":"flash_expert_rhs_tile64_reference_gate_up_probe_m16_n64_sg4"):(tile?"flash_expert_rhs_tile64_gate_up_m16_n64_sg4":"flash_gathered_mpp_gate_up_m16_n64_sg4");
 const std::string downName=probe?(tile?"flash_expert_rhs_tile64_candidate_down_probe_m16_n64_sg4":"flash_expert_rhs_tile64_reference_down_probe_m16_n64_sg4"):(tile?"flash_expert_rhs_tile64_down_m16_n64_sg4":"flash_gathered_mpp_down_m16_n64_sg4");
 g.add(gateName,std::move(gate),params,{10,4,10},{128,1,1});g.add(downName,std::move(down),params,{40,4,10},{128,1,1});return g;
}
double run(MetalBackend&b,const CommandGraph&g){const auto t=b.submitCommand(g.dispatches());require(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>0&&std::isfinite(t.wallSeconds)&&t.wallSeconds>0,"timing ABI");return t.gpuSeconds;}
uint32_t cpu(){require(rhs_tile64::cpuBijection()==3276800,"CPU tile64 bijection/logical-view count");bool bad=false;try{(void)rhs_tile64::bankBytes(65,2560);}catch(const std::invalid_argument&){bad=true;}require(bad,"CPU bad width rejected");bad=false;try{rhs_tile64::requireInlineStrides(64,1);}catch(const std::invalid_argument&){bad=true;}require(bad,"CPU failedv3 illegalfirststride view rejected");return 3276802;}
}
int main(int argc,char**argv){@autoreleasepool{Progress p;try{
 if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){std::cout<<"{\"valid\":true,\"checks\":"<<cpu()<<",\"GPU_work\":false,\"payload_reads\":false}\n";return 0;}
 if(argc==2&&std::string_view(argv[1])=="--help"){std::cout<<"Root-only: oracle METALLIB FULL512_STORE FRESH_REPORT [LAYER0..47] [EXPERT0..511] [REAL_R4_NORMALIZED_BF16_FILE]\n";return 0;}
 require(argc>=4&&argc<=7,"oracle arguments");require(!std::filesystem::exists(argv[3])&&!std::filesystem::exists(std::string(argv[3])+".writing"),"fresh report required");p.path=argv[3];p.write(false,false);
 uint32_t layer=argc>4?uint32_t(std::stoul(argv[4])):0,expert=argc>5?uint32_t(std::stoul(argv[5])):0;require(layer<48&&expert<512,"layer/expert bounds");
 p.phase="metadata";const auto metadata=loadFlashInt8ExpertStoreMetadata(argv[2],"ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e","edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);
 require(metadata.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","registered Full512 metadata identity");
 {BackendLifetime lifetime{p};p.phase="backend-and-admission";MetalBackend backend(argv[1]);p.backendInitialized=true;p.initialAllocated=backend.memoryStats().allocatedBytes;const auto digest=backend.metallibSha256();constexpr char hex[]="0123456789abcdef";for(auto v:digest){p.library+=hex[v>>4];p.library+=hex[v&15];}
 const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);engine::MemoryGovernor gov(backend,physical-reserve,reserve);
 auto lease=gov.tryReserve(p.planned);p.denied=gov.snapshot().deniedReservations;require(bool(lease)&&p.denied==0,"bounded <64MiB fixture/scratch admission");{std::vector<Allocation>alloc;
 p.phase="Root-only-bounded-oneexpert-read";const auto bank=load(backend,metadata.layers[layer],expert,alloc,p);
 const auto x=allocate(backend,Rows*2560*2,alloc).view,ranks=allocate(backend,512*4,alloc).view,ids=allocate(backend,Routes*8,alloc).view;
 auto*xb=static_cast<uint16_t*>(x.contents());for(uint32_t r=0;r<Rows;++r)for(uint32_t k=0;k<2560;++k)xb[r*2560+k]=bf(float(int((k*17+r*31)%251)-125)/64.0f);
 if(argc==7){std::ifstream in(argv[6],std::ios::binary);require(bool(in),"Root real normalized input file");in.read(reinterpret_cast<char*>(xb),x.sizeBytes());require(in.gcount()==std::streamsize(x.sizeBytes())&&in.peek()==std::char_traits<char>::eof(),"exact real R4 normalized BF16 input extent");}
 for(uint32_t i=0;i<Rows*2560;++i)require(std::isfinite(f32(xb[i])),"healthy finite input fixture");
 std::memset(ranks.contents(),0,ranks.sizeBytes());auto*id=static_cast<int64_t*>(ids.contents());for(uint32_t r=0;r<Rows;++r)for(uint32_t s=0;s<10;++s)id[r*10+s]=s;
 std::array<Output,4>out{output(backend,alloc),output(backend,alloc),output(backend,alloc),output(backend,alloc)};
 p.actualAllocated=allocationDelta(p.initialAllocated,backend.memoryStats().allocatedBytes);p.peakAllocated=allocationDelta(p.initialAllocated,backend.memoryStats().peakAllocatedBytes);
 require(p.actualAllocated<=p.planned&&p.peakAllocated<=p.planned&&backend.memoryStats().sparseVirtualBytes==0&&backend.memoryStats().sparseResidentBytes==0,"actual bounded fixture allocation within reservation and no sparse backing");lease->commit();
 std::array<CommandGraph,4>gs{graph(bank,x,ranks,ids,out[0],false,false),graph(bank,x,ranks,ids,out[1],true,false),graph(bank,x,ranks,ids,out[2],false,true),graph(bank,x,ranks,ids,out[3],true,true)};
 std::vector<MetalBuffer>immutable{x,ranks,ids};for(uint32_t i=0;i<3;++i){immutable.push_back(bank.old[i]);immutable.push_back(bank.newCodes[i]);immutable.push_back(bank.scales[i]);}std::vector<std::vector<uint8_t>>saved;for(auto b:immutable)saved.push_back(copied(b));
 p.phase="untapped-probe-rawF32-parity";for(uint32_t i=0;i<4;++i){reset(out[i].diagnostics);p.gpu=true;(void)run(backend,gs[i]);require(diag(out[i].diagnostics)==0x80000000u,"healthy diagnostics");finite(out[i].act,true);finite(out[i].down,true);}
 for(uint32_t i=1;i<4;++i){equal(out[0].act,out[i].act,p,"full gate/up activation parity");equal(out[0].down,out[i].down,p,"full down BF16 parity");}
 for(uint32_t j=0;j<8;++j){finite(out[2].g[j],j>=4);finite(out[3].g[j],j>=4);equal(out[2].g[j],out[3].g[j],p,"all gate/up raw-scaled-F32 and BF16 taps" );}
 for(uint32_t j=0;j<3;++j){finite(out[2].d[j],j>=2);finite(out[3].d[j],j>=2);equal(out[2].d[j],out[3].d[j],p,"all down raw-scaled-F32 and BF16 taps");}
 equal(out[2].act,out[2].g[7],p,"original shipping activation versus activation tap");equal(out[3].act,out[3].g[7],p,"candidate shipping activation versus activation tap");equal(out[2].down,out[2].d[2],p,"original down tap");equal(out[3].down,out[3].d[2],p,"candidate down tap");
 p.phase="sticky-fallback-and-rank-guards";for(uint32_t test=0;test<5;++test){std::memcpy(x.contents(),saved[0].data(),saved[0].size());std::memcpy(ranks.contents(),saved[1].data(),saved[1].size());std::memcpy(ids.contents(),saved[2].data(),saved[2].size());
  if(test==0)xb[0]=0x7fc1;if(test==1)xb[2560+127]=0x7f80;if(test==2){id[0]=-1;xb[0]=0x7fc1;}if(test==3)static_cast<uint32_t*>(ranks.contents())[0]=512;if(test==4)id[0]=1;
  for(uint32_t i=0;i<4;++i){reset(out[i].diagnostics);(void)run(backend,gs[i]);}for(uint32_t i=1;i<4;++i){equal(out[0].act,out[i].act,p,"malformed activation byte/NaN parity");equal(out[0].down,out[i].down,p,"malformed down byte/NaN parity");require(diag(out[0].diagnostics)==diag(out[i].diagnostics),"same malformed sticky bits");}
  const auto expected=test<2?0x80000004u:test<4?0x80000005u:0x80000001u;require(diag(out[0].diagnostics)==expected,"literal original malformed policy");++p.guards;
 }
 std::memcpy(x.contents(),saved[0].data(),saved[0].size());std::memcpy(ranks.contents(),saved[1].data(),saved[1].size());std::memcpy(ids.contents(),saved[2].data(),saved[2].size());
 p.phase="reserved-param-reject-before-outputwrite";{auto bad=Params;bad.reserved=1;for(uint32_t i=0;i<4;++i){const auto a=copied(out[i].act),d=copied(out[i].down);reset(out[i].diagnostics);auto g=graph(bank,x,ranks,ids,out[i],i==1||i==3,i>=2,bad);(void)run(backend,g);require(diag(out[i].diagnostics)==0x80000002u&&unchanged(out[i].act,a)&&unchanged(out[i].down,d),"all variants reject reserved params without output mutation");}++p.guards;}
 p.phase="host-partial-grid-and-alias";for(uint32_t i=0;i<3;++i){const auto before=backend.submissionCount();bool rejected=false;try{if(i==0)preflight(bank,x,ranks,ids,out[1],true,false,{9,4,10});else if(i==1)preflight(bank,x,ranks,ids,out[1],true,false,{10,3,10});else{auto a=out[1];a.act=backend.view(bank.newCodes[0],0,Routes*640*2);preflight(bank,x,ranks,ids,a,true,false);}}catch(const std::runtime_error&error){const auto reason=std::string_view(error.what());require(reason==(i<2?"host rejects partial or excess grid before submission":"mutable/immutable view overlap"),"specific intended host guard");rejected=true;}require(rejected&&before==backend.submissionCount(),"host rejected before submit");++p.guards;}
 for(size_t i=0;i<immutable.size();++i)require(unchanged(immutable[i],saved[i]),"all old/new immutable coefficients/input/rank/ids unchanged");canaries(alloc);require(backend.healthy(),"healthy backend before timing");
 p.phase="shipping-read-free-warm-and18-balanced";reset(out[0].diagnostics);reset(out[1].diagnostics);
 for(uint32_t i=0;i<2;++i){double warmed=0;while(warmed<0.150)warmed+=run(backend,gs[i]);}
 std::array<std::vector<double>,2>timings;for(uint32_t pair=0;pair<18;++pair)for(uint32_t pos=0;pos<2;++pos){const uint32_t v=(pair+pos)%2;timings[v].push_back(run(backend,gs[v]));}
 for(size_t i=0;i<immutable.size();++i)require(unchanged(immutable[i],saved[i]),"immutable after timing");canaries(alloc);equal(out[0].act,out[1].act,p,"posttiming activation");equal(out[0].down,out[1].down,p,"posttiming down");require(diag(out[0].diagnostics)==0x80000000u&&diag(out[1].diagnostics)==0x80000000u,"posttiming sticky");
 p.phase="posttiming-probes-bound-to-LAST-shipping";{const auto lastAct=copied(out[0].act),lastDown=copied(out[0].down);for(uint32_t i=2;i<4;++i){reset(out[i].diagnostics);(void)run(backend,gs[i]);require(diag(out[i].diagnostics)==0x80000000u,"final probes sticky");require(unchanged(out[i].act,lastAct)&&unchanged(out[i].down,lastDown),"final tap replay equals actual LAST shipping snapshots");}
 for(uint32_t j=0;j<8;++j)equal(out[2].g[j],out[3].g[j],p,"LAST shipping coupled gate/up raw/scaled/BF16 taps");for(uint32_t j=0;j<3;++j)equal(out[2].d[j],out[3].d[j],p,"LAST shipping coupled down raw/scaled/BF16 taps");}
 for(size_t i=0;i<immutable.size();++i)require(unchanged(immutable[i],saved[i]),"immutable after final probes");canaries(alloc);
 p.peakAllocated=allocationDelta(p.initialAllocated,backend.memoryStats().peakAllocatedBytes);p.denied=gov.snapshot().deniedReservations;
 require(p.peakAllocated<=p.planned&&p.denied==0&&backend.healthy(),"no allocation growth beyond reservation or Gov denials");
 const auto median=[](std::vector<double>v){std::sort(v.begin(),v.end());return(v[8]+v[9])/2;};p.oldMedian=median(timings[0]);p.newMedian=median(timings[1]);
 }p.phase="fixture-release-before-backend";p.afterAllocated=allocationDelta(p.initialAllocated,backend.memoryStats().allocatedBytes);require(p.afterAllocated==0,"all owned fixture/graph buffers released to zero before backend teardown");
 }require(p.backendDestroyed,"backend destroyed before success publication");p.phase="complete";
 auto temp=p.path.string()+".final-writing";std::ofstream o(temp);require(bool(o),"final report open");o.precision(17);o<<"{\"schema\":\"single-expert-RHS-tile64-component-v1\",\"pass\":true,\"qualification_complete\":true,\"GPU_executed\":true,\"layer\":"<<layer<<",\"original_expert\":"<<expert
  <<",\"coefficient_fixture_bytes\":"<<p.payloadRead<<",\"fixture_sha256\":"<<json::quote(p.fixture)<<",\"metallib_sha256\":"<<json::quote(p.library)<<",\"repeated_oneexpert_synthetic_distinctIDs_rank0\":true,\"true_routed_Full512_top10\":false,\"wholemodel_speed_qualified\":false"
  <<",\"full_raw_scaled_F32_and_BF16_gate_up_activation_down_exact\":true,\"posttiming_raw_taps_coupled_to_LAST_shipping_snapshots\":true,\"original_untapped_AIR_control_preserved\":true,\"all_immutable_buffers_and_canaries_intact\":true,\"guard_cases\":"<<p.guards<<",\"sample_pairs\":18,\"warm_GPU_seconds_each_atleast\":0.15,\"old_shipping_GPU_median_ms\":"<<p.oldMedian*1000<<",\"tile_shipping_GPU_median_ms\":"<<p.newMedian*1000<<",\"ratio\":"<<p.oldMedian/p.newMedian
  <<",\"allocation\":{\"planned_bytes\":"<<p.planned<<",\"actual_fixture_bytes\":"<<p.actualAllocated<<",\"peak_fixture_bytes\":"<<p.peakAllocated<<",\"after_fixture_bytes\":"<<p.afterAllocated<<",\"Gov_denied_reservations\":"<<p.denied<<"},\"backend_destroyed\":"<<(p.backendDestroyed?"true":"false")<<",\"timing_scope\":\"hot repeated oneexpert primitive only; no wholemodel bandwidth or20percent claim\"}\n";o.close();require(bool(o),"final report close");std::filesystem::rename(temp,p.path);return 0;
 }catch(const std::exception&e){p.error=e.what();try{p.write(true,false);}catch(...){}std::cerr<<"RHS tile64 oracle failed: "<<e.what()<<'\n';return 1;}}}
