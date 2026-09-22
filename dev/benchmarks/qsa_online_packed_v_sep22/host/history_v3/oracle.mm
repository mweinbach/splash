// Standalone Root-only original ONLINE QSA packed-V component. No worker/model.
// Source references: frozen prefill4k_attention/bulk.cpp, coalesced.cpp and
// bulk_attention_sg8.metal. No two-pass arena, debug re-reduction or tolerance.
#import <Foundation/Foundation.h>
#include "ABI.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <string_view>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
using namespace splash;
using namespace splash::flash;
using namespace splash::metal;
namespace p4=splash::flash::prefill4k;
namespace h=qsa_online_packed_v_host;
namespace {
constexpr uint32_t Sticky=0x40000000u;
static_assert(sizeof(CommandTiming)==200,"frozen parent timing ABI");
constexpr const char *ManifestPin="135a73ac1edef19dabae195e09fed6d099c58def2bff06affc51f546f4183225";
void require(bool value,const std::string &label){if(!value)throw std::runtime_error(label);}
uint16_t bf(float x){const auto u=std::bit_cast<uint32_t>(x);if((u&0x7f800000u)==0x7f800000u)return uint16_t((u>>16)|((u&0x7fffffu)?0x40u:0u));return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);}
float number(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
std::string hash(const void *data,uint64_t bytes){CC_SHA256_CTX context{};require(CC_SHA256_Init(&context),"SHA init");const auto *p=static_cast<const uint8_t *>(data);
  while(bytes){const auto n=CC_LONG(std::min<uint64_t>(bytes,1ULL<<30));require(CC_SHA256_Update(&context,p,n),"SHA update");p+=n;bytes-=n;}std::array<uint8_t,32>d{};require(CC_SHA256_Final(d.data(),&context),"SHA final");
  constexpr char hex[]="0123456789abcdef";std::string result;for(auto b:d){result+=hex[b>>4];result+=hex[b&15];}return result;}
std::string digest(MetalBuffer b){return hash(b.contents(),b.sizeBytes());}
struct Report final {
  std::filesystem::path path;std::string phase="arguments",check,error,manifestHash,library,mode;
  uint64_t checks=0,bytes=0,guards=0,first=UINT64_MAX,payload=0,enumerated=0,hostEnumerated=0,engineLimit=0,finalReserved=UINT64_MAX,current=0,peak=0,deviceCurrent=0,devicePeak=0,after=UINT64_MAX,denials=UINT64_MAX;
  uint32_t expected=0,actual=0;bool gpu=false,stopped=false,destroyed=false,hostValid=false,growth=false;
  void write(bool complete,bool pass)const{if(path.empty())return;const auto temporary=path.string()+".writing";std::ofstream o(temporary);require(bool(o),"progress open");
    o<<"{\"schema\":\"original-online-packedV-component-progress-v1\",\"completed\":"<<(complete?"true":"false")<<",\"pass\":"<<(pass?"true":"false")<<",\"GPU_executed\":"<<(gpu?"true":"false")
     <<",\"phase\":"<<json::quote(phase)<<",\"check\":"<<json::quote(check)<<",\"error\":"<<json::quote(error)<<",\"fixture\":"<<json::quote(mode)<<",\"checks\":"<<checks<<",\"bytes_compared\":"<<bytes<<",\"guards\":"<<guards
     <<",\"capture_payload_bytes_read\":"<<payload<<",\"first_mismatch_byte\":";if(first==UINT64_MAX)o<<"null";else o<<first;
    o<<",\"expected_word\":"<<expected<<",\"actual_word\":"<<actual<<",\"enumerated_charged_bytes\":"<<enumerated<<",\"enumerated_host_phase_upper_bound\":"<<hostEnumerated<<",\"native_reserved_bytes\":1073741824,\"host_reserved_bytes\":536870912,\"combined_component_admission_bytes\":1610612736,\"engine_physical_minus_host_reserve_limit\":"<<engineLimit<<",\"current_owned_delta\":"<<current<<",\"peak_owned_delta\":"<<peak<<",\"current_device_delta\":"<<deviceCurrent<<",\"peak_device_delta\":"<<devicePeak
     <<",\"after_owned_bytes\":"<<after<<",\"Gov_denials\":"<<denials<<",\"host_measurement_valid\":"<<(hostValid?"true":"false")<<",\"growth_allowed\":"<<(growth?"true":"false")<<",\"backend_stopped\":"<<(stopped?"true":"false")<<",\"backend_destroyed\":"<<(destroyed?"true":"false")<<"}\n";o.close();require(bool(o),"progress close");std::filesystem::rename(temporary,path);}
};
struct Lifetime final{Report &r;~Lifetime(){r.destroyed=true;}};
struct Owner final{MetalBuffer base,view;};
MetalBuffer allocate(MetalBackend &b,uint64_t n,std::vector<Owner>&all){auto base=b.allocateBuffer(h::charge(n),BufferStorage::Shared,"bounded-online-packedV-owner");std::memset(base.contents(),0xa5,base.sizeBytes());auto view=b.view(base,h::guard,n);all.push_back({base,view});return view;}
void canaries(const std::vector<Owner>&all){for(const auto&o:all){const auto*p=static_cast<const uint8_t*>(o.base.contents());for(uint64_t i=0;i<h::guard;++i)require(p[i]==0xa5,"leading canary");for(uint64_t i=h::guard+o.view.sizeBytes();i<o.base.sizeBytes();++i)require(p[i]==0xa5,"trailing/padding canary");}}
std::vector<uint8_t> snapshot(MetalBuffer b){const auto*p=static_cast<const uint8_t*>(b.contents());return{p,p+b.sizeBytes()};}
void exact(MetalBuffer b,std::span<const uint8_t> a,Report&r,const std::string&label){r.check=label;require(b.sizeBytes()==a.size(),label+" extent");const auto*p=static_cast<const uint8_t*>(b.contents());
  for(uint64_t i=0;i<a.size();++i)if(p[i]!=a[i]){r.first=i;const uint64_t at=i/4*4;if(at+4<=a.size()){std::memcpy(&r.expected,a.data()+at,4);std::memcpy(&r.actual,p+at,4);}throw std::runtime_error(label+" bit mismatch");}++r.checks;r.bytes+=a.size();}
void exact(MetalBuffer a,MetalBuffer b,Report&r,const std::string&label){require(a.sizeBytes()==b.sizeBytes(),label+" extent");exact(b,std::span<const uint8_t>(static_cast<const uint8_t*>(a.contents()),a.sizeBytes()),r,label);}
void resetDiag(MetalBuffer b,uint32_t value=Sticky){std::memcpy(b.contents(),&value,4);}
uint32_t diag(MetalBuffer b){uint32_t value;std::memcpy(&value,b.contents(),4);return value;}
void disjoint(MetalBuffer a,MetalBuffer b){const uintptr_t x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());require(x && y && (x<=y?uint64_t(y-x)>=a.sizeBytes():uint64_t(x-y)>=b.sizeBytes()),"host mutable/immutable alias refusal");}
struct Input final {
  uint32_t rows=0;MetalBuffer q,k,v,index,out,diag;std::array<FlashTensor,4> norms;
  Input(MetalBackend &b,uint32_t n,std::vector<Owner>&all):rows(n){q=allocate(b,uint64_t(n)*12288*2,all);k=allocate(b,uint64_t(n)*512*2,all);v=allocate(b,uint64_t(n)*512*2,all);index=allocate(b,uint64_t(n)*640*2,all);out=allocate(b,uint64_t(n)*6144*2,all);diag=allocate(b,4,all);
    for(uint32_t i=0;i<4;++i){const uint32_t width=i<2?256:128;norms[i].dtype=FlashDType::BF16;norms[i].shape={width};norms[i].logicalBytes=width*2;norms[i].buffer=allocate(b,width*2,all);std::memset(norms[i].buffer.contents(),0,width*2);}resetDiag(diag);}
  FlashQSAFastInputs args()const{return{q,k,v,index,&norms[0],&norms[1],&norms[2],&norms[3],out,diag,{},NormConvention::OnePlusWeight,NormConvention::OnePlusWeight,NormConvention::OnePlusWeight,NormConvention::OnePlusWeight,1e-6,1e7};}
  std::vector<MetalBuffer> immutable()const{std::vector<MetalBuffer>v{q,k,this->v,index};for(const auto&n:norms)v.push_back(n.buffer);return v;}
};
void synthetic(Input &x,uint32_t seed){const auto fill=[&](MetalBuffer b,uint32_t width,uint32_t salt){auto*p=static_cast<uint16_t*>(b.contents());for(uint32_t row=0;row<x.rows;++row){double square=0;for(uint32_t c=0;c<width;++c){const double v=int((uint64_t(c)*73+uint64_t(row)*17+seed+salt)%257)-128;square+=v*v;}const float rms=float(std::sqrt(square/width));require(rms>0,"true RMS fixture");for(uint32_t c=0;c<width;++c)p[uint64_t(row)*width+c]=bf(float(int((uint64_t(c)*73+uint64_t(row)*17+seed+salt)%257)-128)/rms);}};
  fill(x.q,12288,1);fill(x.k,512,2);fill(x.v,512,3);fill(x.index,640,4);}
void copyInput(const Input&a,const Input&b){const auto x=a.immutable(),y=b.immutable();for(uint32_t i=0;i<x.size();++i){require(x[i].sizeBytes()==y[i].sizeBytes(),"input copy extent");std::memcpy(y[i].contents(),x[i].contents(),x[i].sizeBytes());}}
std::vector<uint8_t> readPayload(const std::filesystem::path&folder,NSDictionary *files,NSString *name,uint64_t bytes,NSArray *shape,Report&r){NSDictionary *e=files[name];require([e isKindOfClass:[NSDictionary class]],"capture field");
  require([e[@"dtype"] isEqual:@"BF16"] && [e[@"shape"] isEqual:shape] && [e[@"bytes"] unsignedLongLongValue]==bytes,"capture dtype/shape/bytes");const std::string filename([e[@"path"] UTF8String]);require(filename==std::string([name UTF8String])+".bf16","canonical capture filename");
  const int fd=::open((folder/filename).c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,"Root capture payload open");struct stat before{},after{};require(!::fstat(fd,&before) && S_ISREG(before.st_mode) && uint64_t(before.st_size)==bytes,"capture exact file bytes");
  std::vector<uint8_t>data(bytes);uint64_t done=0;while(done<bytes){const auto n=::pread(fd,data.data()+done,bytes-done,off_t(done));if(n<=0){::close(fd);throw std::runtime_error("capture payload read");}done+=uint64_t(n);}const bool unchanged=!::fstat(fd,&after) && before.st_dev==after.st_dev && before.st_ino==after.st_ino && before.st_size==after.st_size && before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec && before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec;::close(fd);require(unchanged,"capture file unchanged during Root read");
  require(hash(data.data(),data.size())==std::string([e[@"sha256"] UTF8String]),"Root capture payload hash");r.payload+=bytes;return data;}
std::vector<uint8_t> actualInput(const std::filesystem::path&manifest,Input&x,Report&r){std::ifstream in(manifest,std::ios::binary);require(bool(in),"Root capture manifest");const std::string text((std::istreambuf_iterator<char>(in)),{});r.manifestHash=hash(text.data(),text.size());require(r.manifestHash==ManifestPin,"pinned old B4 capture manifest");NSData *data=[NSData dataWithBytes:text.data() length:text.size()];NSError *error=nil;NSDictionary *m=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(m && !error && [m[@"schema"] isEqual:@"splash-current-batch-qsa-projection-capture-v1"] && [m[@"pass"] boolValue] && [m[@"cohortValidated"] boolValue] && [m[@"completed"] boolValue],"capture schema/pass/cohort/completion");
  require([m[@"actualLanes"] unsignedIntValue]==4 && [m[@"lane"] unsignedIntValue]==0 && [m[@"layer"] unsignedIntValue]==3 && [m[@"rows"] unsignedIntValue]==2048 && [m[@"begin"] unsignedIntValue]==0 && [m[@"capacity"] unsignedIntValue]==16384,"old B4 lane0 layer3 capture geometry");
  require([m[@"epsilon"] doubleValue]==1e-6 && [m[@"theta"] doubleValue]==1e7 && [m[@"token_words_sha256"] isEqual:@"0a383d21f5c784b0616d589847ca6cb04c69bf654729f2344e2b916e542f36b4"],"original capture epsilon/theta/U32 cohort");
  NSDictionary *files=m[@"files"],*conventions=m[@"conventions"];const auto folder=manifest.parent_path();const std::array<NSString*,4>names{@"q",@"k",@"v",@"index"};const std::array<MetalBuffer,4>buffers{x.q,x.k,x.v,x.index};
  for(uint32_t i=0;i<4;++i){const uint32_t width=i==0?12288:i==3?640:512;const auto payload=readPayload(folder,files,names[i],buffers[i].sizeBytes(),@[@2048,@(width)],r);std::memcpy(buffers[i].contents(),payload.data(),payload.size());}
  const std::array<NSString*,4>normNames{@"q_norm",@"k_norm",@"index_q_norm",@"index_k_norm"};for(uint32_t i=0;i<4;++i){require([conventions[normNames[i]] isEqual:@"OnePlusWeight"],"frozen capture norm convention");const auto payload=readPayload(folder,files,normNames[i],x.norms[i].logicalBytes,@[@(i<2?256:128)],r);std::memcpy(x.norms[i].buffer.contents(),payload.data(),payload.size());}
  return readPayload(folder,files,@"output",uint64_t(2048)*6144*2,@[@2048,@6144],r);}
FlashQSAState state(MetalBackend&b,uint32_t capacity,std::vector<Owner>&all){return{capacity,allocate(b,uint64_t(capacity)*512*2,all),allocate(b,uint64_t(capacity)*512*2,all),allocate(b,uint64_t(capacity)*128*2,all),allocate(b,uint64_t((capacity+3)/4)*128*2,all),allocate(b,uint64_t(capacity)*8,all)};}
std::array<MetalBuffer,5>planes(const FlashQSAState&s){return{s.keys,s.values,s.rawIndexKeys,s.pooledKeys,s.indexPositions};}
FlashQSAWorkspace ordinary(MetalBackend&b,uint32_t c,std::vector<Owner>&all){return{128,c,allocate(b,128ULL*6144*2,all),allocate(b,128ULL*512*2,all),allocate(b,128ULL*((c+3)/4)*4,all),allocate(b,128ULL*512*4,all),allocate(b,128ULL*24*2051*4,all),allocate(b,128ULL*24*2051*2,all)};}
FlashQSAFastWorkspace fast(MetalBackend&b,uint32_t n,uint32_t p,std::vector<Owner>&all){return{n,p,allocate(b,uint64_t(n)*24*p*2*4,all),allocate(b,uint64_t(n)*24*p*256*4,all)};}
p4::BulkExactWorkspace bulk(MetalBackend&b,std::vector<Owner>&all){return{{2048,allocate(b,2048ULL*6144*2,all),allocate(b,2048ULL*512*2,all),allocate(b,2048ULL*512*4,all)},fast(b,2048,4,all)};}
void copyDispatch(CommandGraph&g,const ComputeDispatch&d){std::vector<MetalBuffer>buffers;for(const auto&binding:d.buffers){require(binding.index==buffers.size(),"original helper binding order");buffers.push_back(binding.buffer);}require(d.bytes.size()==1,"original helper inline ABI");
  if(d.bytes[0].sizeBytes==48){FlashQSAParams p;std::memcpy(&p,d.bytes[0].data,48);g.add(d.pipelineName,buffers,p,d.threadgroups,d.threadsPerThreadgroup);}else{require(d.bytes[0].sizeBytes==64,"original helper params64");FlashQSAFastParams p;std::memcpy(&p,d.bytes[0].data,64);g.add(d.pipelineName,buffers,p,d.threadgroups,d.threadsPerThreadgroup);}}
CommandGraph originalPlan(MetalBackend&b,const Input&i,FlashQSAState&s,FlashQSAWorkspace&o,FlashQSAFastWorkspace&f,p4::BulkExactWorkspace&w){CommandGraph g;p4::addBulkExactQSA(b,g,i.args(),s,o,f,w,0,2048,true);require(g.dispatches().size()==6,"original prefix3+online3 helper topology");return g;}
CommandGraph range(const CommandGraph&g,uint32_t first,uint32_t stop){CommandGraph result;for(uint32_t i=first;i<stop;++i)copyDispatch(result,g.dispatches()[i]);return result;}
bool size(DispatchSize a,DispatchSize b){return a.x==b.x && a.y==b.y && a.z==b.z;}
void topology(std::span<const ComputeDispatch>d,bool packed){require(d.size()==(packed?4:3),"host complete original3/candidate4 topology");const uint32_t start=packed?1:0;if(packed)require(d[0].pipelineName=="qsa_online_packed_v_pack" && size(d[0].threadgroups,{64,8,2}) && size(d[0].threadsPerThreadgroup,{256,1,1}),"host exact pack grid/threads");
  const std::array<DispatchSize,3>grids{{{3328,1,1},{4608,1,1},{2048,24,1}}};for(uint32_t i=0;i<3;++i){const auto&v=d[start+i];require(size(v.threadgroups,grids[i]) && size(v.threadsPerThreadgroup,{i?256u:128u,1,1}),"host exact original online consumer grid/threads");
    require(v.bytes.size()==1 && v.bytes[0].data && v.bytes[0].sizeBytes==64 && v.bytes[0].index==(i==2?5:7) && v.buffers.size()==(i==2?5:7),"host original online buffer/Params64 ABI");FlashQSAFastParams p;std::memcpy(&p,v.bytes[0].data,64);
    require(p.common.rows==2048 && !p.common.begin && (p.common.capacity==4096 || p.common.capacity==16384) && p.partitions==4 && p.maximum_partitions==4 && !p.common.reserved,"host fixed original online eligibility/parameters");}
  require(d[start].buffers[0].buffer.sameView(d[start+1].buffers[0].buffer) && d[start].buffers[1].buffer.sameView(d[start+1].buffers[1].buffer) && d[start].buffers[2].buffer.sameView(d[start+1].buffers[2].buffer) && d[start].buffers[3].buffer.sameView(d[start+1].buffers[3].buffer),"host shared immutable producer operands");
  require(d[start].buffers[4].buffer.sameView(d[start+1].buffers[4].buffer) && d[start].buffers[5].buffer.sameView(d[start+1].buffers[5].buffer) && d[start].buffers[4].buffer.sameView(d[start+2].buffers[0].buffer) && d[start].buffers[5].buffer.sameView(d[start+2].buffers[1].buffer),"host live producer/reducer coupled sheets");
}
CommandGraph candidate(const CommandGraph&source,MetalBuffer values,MetalBuffer packed,MetalBuffer diag){require(values.sizeBytes()>=h::packedBytes && packed.sizeBytes()==h::packedBytes,"host exact packedV source/destination extent");disjoint(values,packed);CommandGraph g;g.add("qsa_online_packed_v_pack",{values,packed,diag},h::packParams,{64,8,2},{256,1,1});
  topology(source.dispatches(),false);for(const auto&d:source.dispatches())for(const auto&binding:d.buffers)disjoint(packed,binding.buffer);
  for(uint32_t i=0;i<3;++i){auto d=source.dispatches()[i];d.pipelineName=i==0?"qsa_online_packed_v_candidate_early":i==1?"qsa_online_packed_v_candidate_temporal":"qsa_online_packed_v_candidate_reduce";if(i<2)d.buffers[2].buffer=packed;copyDispatch(g,d);}topology(g.dispatches(),true);return g;}
CommandGraph native(const CommandGraph&source){CommandGraph g;for(uint32_t i=0;i<3;++i){auto d=source.dispatches()[i];d.pipelineName=i==0?"qsa_online_packed_v_native_early":i==1?"qsa_online_packed_v_native_temporal":"qsa_online_packed_v_native_reduce";copyDispatch(g,d);}return g;}
ComputeDispatch tap(const CommandGraph&g,bool packed,MetalBuffer raw,MetalBuffer rounded){require(raw.sizeBytes()==h::quotientBytes && rounded.sizeBytes()==h::roundedBytes,"host exact live tap extents");auto d=g.dispatches()[packed?3:2];
  disjoint(raw,rounded);for(const auto&binding:d.buffers){disjoint(raw,binding.buffer);disjoint(rounded,binding.buffer);}d.pipelineName=packed?"qsa_online_packed_v_candidate_reduce_tap":"qsa_online_packed_v_native_reduce_tap";d.buffers.push_back({6,raw});d.buffers.push_back({7,rounded});return d;}
CommandTiming submit(MetalBackend&b,std::span<const ComputeDispatch>d){const auto t=b.submitCommand(d);require(t.gpuSeconds>0 && std::isfinite(t.gpuSeconds) && t.wallSeconds>0 && std::isfinite(t.wallSeconds),"positive finite GPU/wall timing");return t;}
void packExact(MetalBuffer source,MetalBuffer packed,Report&r){const auto*a=static_cast<const uint16_t*>(source.contents()),*b=static_cast<const uint16_t*>(packed.contents());for(uint32_t kv=0;kv<2;++kv)for(uint32_t d=0;d<256;++d)for(uint32_t t=0;t<2048;++t)require(a[(uint64_t(t)*2+kv)*256+d]==b[(uint64_t(kv)*256+d)*2048+t],"all packedV ushort source bits");++r.checks;r.bytes+=h::packedBytes;}
void normalFinite(const p4::BulkExactWorkspace&w,MetalBuffer out){const auto*s=static_cast<const float*>(w.partials.partitionStatistics.contents()),*n=static_cast<const float*>(w.partials.partitionValues.contents());for(uint32_t row=0;row<2048;++row)for(uint32_t head=0;head<24;++head)for(uint32_t p=0;p<(row<128?1u:4u);++p){const auto at=(uint64_t(row)*24+head)*4+p;require(std::isfinite(s[at*2]) && s[at*2+1]>0 && std::isfinite(s[at*2+1]),"active original statistics finite");for(uint32_t d=0;d<256;++d)require(std::isfinite(n[at*256+d]),"active original numerator finite");}const auto*b=static_cast<const uint16_t*>(out.contents());for(uint64_t i=0;i<out.sizeBytes()/2;++i)require(std::isfinite(number(b[i])),"gated BF16 finite");}
void sheets(const p4::BulkExactWorkspace&a,const p4::BulkExactWorkspace&b,Report&r){exact(a.partials.partitionStatistics,b.partials.partitionStatistics,r,"full F32 statistics including inactive sentinel");exact(a.partials.partitionValues,b.partials.partitionValues,r,"full F32 numerators including inactive sentinel");}
void cacheEqual(const FlashQSAState&a,const FlashQSAState&b,Report&r){const auto x=planes(a),y=planes(b);for(uint32_t i=0;i<5;++i)exact(x[i],y[i],r,"physical cache plane="+std::to_string(i));}
void coupled(MetalBackend&b,const CommandGraph&a,const CommandGraph&c,MetalBuffer raw,MetalBuffer round,const Input&x,const Input&y,Report&r){const auto lastA=snapshot(x.out),lastB=snapshot(y.out);
  const auto poison=[&]{std::memset(raw.contents(),0xff,raw.sizeBytes());std::memset(round.contents(),0xff,round.sizeBytes());};
  const auto finite=[&]{const auto*f=static_cast<const float*>(raw.contents());const auto*q=static_cast<const uint16_t*>(round.contents());for(uint64_t i=0;i<raw.sizeBytes()/4;++i)require(std::isfinite(f[i]) && std::isfinite(number(q[i])),"all live quotient/rounded tap cells written and finite");};
  auto ta=tap(a,false,raw,round);poison();(void)submit(b,std::span<const ComputeDispatch>(&ta,1));finite();exact(x.out,lastA,r,"native live tap bound LAST shipping gated bits");const auto f=snapshot(raw),q=snapshot(round);
  auto tc=tap(c,true,raw,round);poison();(void)submit(b,std::span<const ComputeDispatch>(&tc,1));finite();exact(y.out,lastB,r,"candidate live tap bound LAST shipping gated bits");exact(raw,f,r,"live same-SSA reducer F32 quotient");exact(round,q,r,"live same-SSA reducer rounded BF16");}
double median(const std::vector<CommandTiming>&v,bool gpu){require(v.size()==18,"18 samples each");std::vector<double>x;for(const auto&t:v)x.push_back(gpu?t.gpuSeconds:t.wallSeconds);std::sort(x.begin(),x.end());return(x[8]+x[9])/2;}
uint64_t cpu(){require(h::plannedBytes(16384)==945078272 && h::plannedBytes(4096)==885112832 && h::plannedBytes(16384)<=h::reservation,"honest guarded owner enumeration");require(p4::bulkExactPlannedBytes()==234356736,"original online workspace geometry");require(h::packedBytes==2048ULL*2*256*2 && h::charge(h::packedBytes)==2129920,"packed logical/charged bytes");uint64_t checks=3;
  std::vector<uint8_t>seen(2048*2*256);for(uint32_t t=0;t<2048;++t)for(uint32_t kv=0;kv<2;++kv)for(uint32_t d=0;d<256;++d){const auto at=(kv*256+d)*2048+t;require(at<seen.size() && !seen[at]++,"CPU bit-transpose bijection");++checks;}return checks;}
} // namespace

int main(int argc,char**argv){@autoreleasepool{Report r;try{
  if(argc==2 && std::string_view(argv[1])=="--cpu-self-test"){std::cout<<"{\"valid\":true,\"checks\":"<<cpu()<<",\"GPU_work\":false,\"payload_reads\":false,\"planned_Governor_bytes\":1073741824}\n";return 0;}
  require(argc==4 || argc==5,"oracle METALLIB FRESH_REPORT --synthetic | --actual MANIFEST");const bool real=std::string_view(argv[3])=="--actual";
  require((real && argc==5) || (std::string_view(argv[3])=="--synthetic" && argc==4),"explicit synthetic/actual argument count");
  const char*authorization=std::getenv("QSA_ONLINE_PACKED_V_ROOT_GPU");require(authorization && std::string_view(authorization)=="1","Root GPU authorization required");
  require(!std::filesystem::exists(argv[2]) && !std::filesystem::exists(std::string(argv[2])+".writing") && !std::filesystem::exists(std::string(argv[2])+".final-writing"),"fresh report required");
  r.path=argv[2];r.mode=real?"actual old B4 lane0 layer3 U32 0a383; synthetic future projections":"true-RMS synthetic projected prefix and future projections";
  const uint32_t capacity=real?16384:4096;r.enumerated=h::plannedBytes(capacity);r.hostEnumerated=h::hostVectorPlan(capacity).peak;
  require(r.enumerated<=h::reservation && r.hostEnumerated<=h::hostReservation && r.enumerated+r.hostEnumerated<=h::combinedReservation,"enumerated native1GiB/host512MiB component bounds before allocation");r.write(false,false);
  std::array<std::vector<CommandTiming>,2>timings;std::array<double,2>warm{};std::array<uint32_t,2>warmCount{};uint64_t timedCount=0;
  {Lifetime lifetime{r};MetalBackend backend(argv[1]);const auto initial=backend.memoryStats();require(initial.allocatedBytes==0,"standalone no-model zero owned baseline");
    const auto lib=backend.metallibSha256();std::ostringstream libText;libText<<std::hex<<std::setfill('0');for(auto b:lib)libText<<std::setw(2)<<unsigned(b);r.library=libText.str();const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);require(physical>reserve,"host reserve");
    r.engineLimit=physical-reserve;engine::MemoryGovernor governor(backend,r.engineLimit,reserve);const auto normal=governor.snapshot();require(normal.hostMeasurementValid && normal.growthAllowed && normal.pressure==engine::MemoryPressure::Normal && normal.systemPressure==engine::MemoryPressure::Normal,"normal valid host governor");
    auto admission=governor.tryReserve(h::reservation);require(bool(admission),"Native1GiB reservation before first fixture owner/vector/payload");
    auto hostAdmission=governor.tryReserve(h::hostReservation);require(bool(hostAdmission) && governor.snapshot().reservedBytes==h::combinedReservation,"separately tagged Host512MiB held before fixture owner/vector/payload");
    const auto ledger=[&]{const auto m=backend.memoryStats();r.current=m.allocatedBytes;r.peak=m.peakAllocatedBytes;r.deviceCurrent=allocationDelta(initial.deviceCurrentAllocatedBytes,m.deviceCurrentAllocatedBytes);r.devicePeak=allocationDelta(initial.deviceCurrentAllocatedBytes,m.devicePeakAllocatedBytes);
      const auto g=governor.snapshot();r.denials=g.deniedReservations;r.hostValid=g.hostMeasurementValid;r.growth=g.growthAllowed;
      require(r.current<=r.enumerated && r.peak<=r.enumerated && r.deviceCurrent<=h::reservation && r.devicePeak<=h::reservation,"actual owned/device current and peak within admitted1GiB");
      require(r.devicePeak+r.hostEnumerated<=h::combinedReservation && g.reservedBytes==h::combinedReservation,"held native/host admissions and conservative combined bound within1.5GiB");
      require(!m.sparseVirtualBytes && !m.sparseResidentBytes && !m.peakSparseResidentBytes && !r.denials && r.hostValid && r.growth && g.pressure==engine::MemoryPressure::Normal && g.systemPressure==engine::MemoryPressure::Normal && backend.healthy(),"no sparse/denials; normal healthy host growth");};
    {std::vector<Owner>owners;Input x(backend,2048,owners),y(backend,2048,owners);std::vector<uint8_t>capturedOutput;
      if(real)capturedOutput=actualInput(std::filesystem::absolute(argv[4]),x,r);else synthetic(x,20260922);copyInput(x,y);
      auto a=state(backend,capacity,owners),b=state(backend,capacity,owners);auto o=ordinary(backend,capacity,owners);auto f=fast(backend,128,32,owners);auto wa=bulk(backend,owners),wb=bulk(backend,owners);
      for(const auto&s:{a,b}){for(auto v:planes(s))std::memset(v.contents(),0,v.sizeBytes());for(auto v:{s.keys,s.values}){auto*p=static_cast<uint16_t*>(v.contents());std::fill(p+2048*512,p+uint64_t(capacity)*512,uint16_t{0x7fc1});}}
      Input futureA(backend,128,owners),futureB(backend,128,owners);
      const auto packed=allocate(backend,h::packedBytes,owners),raw=allocate(backend,h::quotientBytes,owners),rounded=allocate(backend,h::roundedBytes,owners);
      auto fullA=originalPlan(backend,x,a,o,f,wa),fullB=originalPlan(backend,y,b,o,f,wb);auto prefixA=range(fullA,0,3),prefixB=range(fullB,0,3),old=range(fullA,3,6),originalB=range(fullB,3,6);auto test=candidate(originalB,b.values,packed,y.diag),self=native(old);
      topology(old.dispatches(),false);topology(test.dispatches(),true);
      std::vector<MetalBuffer>immutable=x.immutable();const auto iy=y.immutable();immutable.insert(immutable.end(),iy.begin(),iy.end());std::vector<std::string>sourceHashes;for(auto v:immutable)sourceHashes.push_back(digest(v));
      r.phase="qualified-original-prefix-prepared-and-all5cacheplanes";r.gpu=true;(void)submit(backend,prefixA.dispatches());(void)submit(backend,prefixB.dispatches());cacheEqual(a,b,r);
      exact(wa.prepared.queries,wb.prepared.queries,r,"prefix preparedQ");exact(wa.prepared.indexQueries,wb.prepared.indexQueries,r,"prefix preparedIndexQ");exact(wa.prepared.selectedBlocks,wb.prepared.selectedBlocks,r,"prefix chronological selection");
      if(real){r.phase="captured-old-output-anchor-before-candidate";(void)submit(backend,old.dispatches());require(diag(x.diag)==Sticky,"captured original output diagnostics");normalFinite(wa,x.out);exact(x.out,capturedOutput,r,"actual old B4 captured original output first");}
      const auto pa=planes(a),pb=planes(b);immutable.insert(immutable.end(),pa.begin(),pa.end());immutable.insert(immutable.end(),pb.begin(),pb.end());
      for(auto v:{wa.prepared.queries,wa.prepared.indexQueries,wa.prepared.selectedBlocks,wb.prepared.queries,wb.prepared.indexQueries,wb.prepared.selectedBlocks})immutable.push_back(v);
      std::vector<std::string>preparedHashes;for(auto v:immutable)preparedHashes.push_back(digest(v));
      const auto unchanged=[&]{for(uint32_t i=0;i<immutable.size();++i)require(digest(immutable[i])==preparedHashes[i],"projected/norm/prepared/selected/all5cache immutable");};
      const auto recover=[&]{for(auto v:{wa.partials.partitionStatistics,wa.partials.partitionValues,wb.partials.partitionStatistics,wb.partials.partitionValues,x.out,y.out})std::memset(v.contents(),0xa5,v.sizeBytes());resetDiag(x.diag);resetDiag(y.diag);
        (void)submit(backend,old.dispatches());(void)submit(backend,test.dispatches());require(diag(x.diag)==Sticky && diag(y.diag)==Sticky,"normal online sticky diagnostics");sheets(wa,wb,r);exact(x.out,y.out,r,"complete original3/candidate4 gated BF16");normalFinite(wa,x.out);normalFinite(wb,y.out);packExact(b.values,packed,r);coupled(backend,old,test,raw,rounded,x,y,r);unchanged();canaries(owners);};
      r.phase="full-coupled-online-F32-BF16-and-captured-original-output";recover();if(real)exact(x.out,capturedOutput,r,"actual old B4 captured original output fullbytes");const auto oldStats=digest(wa.partials.partitionStatistics),oldNums=digest(wa.partials.partitionValues),oldOut=digest(x.out);
      (void)submit(backend,self.dispatches());require(digest(wa.partials.partitionStatistics)==oldStats && digest(wa.partials.partitionValues)==oldNums && digest(x.out)==oldOut && diag(x.diag)==Sticky,"untimed private native auth matches shipping original");
      r.phase="host-extent-alias-topology-refusal-before-submit";
      for(uint32_t fault=0;fault<6;++fault){r.check="host fault="+std::to_string(fault);const auto count=backend.submissionCount();bool refused=false;
        try{if(fault==0){const auto shortV=backend.view(packed,0,h::packedBytes-2);(void)candidate(originalB,b.values,shortV,y.diag);}else if(fault==1){const auto alias=backend.view(b.values,0,h::packedBytes);(void)candidate(originalB,b.values,alias,y.diag);}else if(fault==5){const auto alias=backend.view(x.q,0,h::quotientBytes);(void)tap(old,false,alias,rounded);}else{std::vector<ComputeDispatch>d(test.dispatches().begin(),test.dispatches().end());if(fault==2)--d[1].threadgroups.x;if(fault==3)++d[2].threadgroups.x;if(fault==4)d[0].threadsPerThreadgroup.x=128;topology(d,true);}}catch(const std::runtime_error&e){if(fault==5)require(std::string_view(e.what())=="host mutable/immutable alias refusal","specific live tap alias refusal");refused=true;}
        require(refused && count==backend.submissionCount(),"host refused before GPU submission");++r.guards;}
      r.phase="pack-GPU-negative-bounded-descriptor-sentinel";
      for(uint32_t fault=0;fault<4;++fault){r.check="GPU pack fault="+std::to_string(fault);auto d=test.dispatches()[0];auto bad=h::packParams;if(fault==0){bad.reserved=1;d.bytes[0].data=&bad;}if(fault==1)--d.threadgroups.x;if(fault==2)++d.threadgroups.z;if(fault==3)d.threadsPerThreadgroup.x=128;
        std::memset(packed.contents(),0xa5,packed.sizeBytes());const auto sentinel=snapshot(packed);resetDiag(y.diag);(void)submit(backend,std::span<const ComputeDispatch>(&d,1));require(diag(y.diag)==(Sticky|kFlashQSAInvalidPosition),"literal pack shape bit9");exact(packed,sentinel,r,"GPU bad pack full sentinel");unchanged();canaries(owners);++r.guards;}
      r.phase="consumer-original-uniform-parameter-thread-rejection";
      for(uint32_t variant=0;variant<2;++variant)for(uint32_t stage=0;stage<3;++stage)for(uint32_t fault=0;fault<2;++fault){r.check="GPU consumer variant="+std::to_string(variant)+" stage="+std::to_string(stage)+" fault="+std::to_string(fault);
        const auto &graph=variant?test:old;auto d=graph.dispatches()[stage+(variant?1:0)];FlashQSAFastParams bad;std::memcpy(&bad,d.bytes[0].data,64);if(fault==0){bad.common.begin=1;d.bytes[0].data=&bad;}else d.threadsPerThreadgroup.x=64;
        const auto stats=snapshot(variant?wb.partials.partitionStatistics:wa.partials.partitionStatistics),nums=snapshot(variant?wb.partials.partitionValues:wa.partials.partitionValues),out=snapshot(variant?y.out:x.out);resetDiag(variant?y.diag:x.diag);
        (void)submit(backend,std::span<const ComputeDispatch>(&d,1));require(diag(variant?y.diag:x.diag)==(Sticky|kFlashQSAInvalidPosition),"literal original uniform bit9");exact(variant?wb.partials.partitionStatistics:wa.partials.partitionStatistics,stats,r,"uniform rejection full stats unchanged");exact(variant?wb.partials.partitionValues:wa.partials.partitionValues,nums,r,"uniform rejection full nums unchanged");exact(variant?y.out:x.out,out,r,"uniform rejection full output unchanged");++r.guards;}
      r.phase="NaN-Inf-invalid-selection-original-bit-policy";
      const auto originalQuery=snapshot(wa.prepared.queries),candidateQuery=snapshot(wb.prepared.queries),originalValue=snapshot(a.values),candidateValue=snapshot(b.values),originalSelection=snapshot(wa.prepared.selectedBlocks),candidateSelection=snapshot(wb.prepared.selectedBlocks);
      for(uint32_t fault=0;fault<3;++fault){r.check="malformed online fault="+std::to_string(fault);if(fault<2){const uint16_t bits=fault?0x7f80:0x7fc1;std::memcpy((fault?a.values:wa.prepared.queries).contents(),&bits,2);std::memcpy((fault?b.values:wb.prepared.queries).contents(),&bits,2);}else{const uint32_t bad=UINT32_MAX;std::memcpy(static_cast<uint8_t*>(wa.prepared.selectedBlocks.contents())+3*512*4,&bad,4);std::memcpy(static_cast<uint8_t*>(wb.prepared.selectedBlocks.contents())+3*512*4,&bad,4);}
        resetDiag(x.diag);resetDiag(y.diag);for(auto v:{wa.partials.partitionStatistics,wa.partials.partitionValues,wb.partials.partitionStatistics,wb.partials.partitionValues,x.out,y.out})std::memset(v.contents(),0xa5,v.sizeBytes());
        (void)submit(backend,old.dispatches());(void)submit(backend,test.dispatches());require(diag(x.diag)==diag(y.diag),"original malformed sticky parity no invented bits");sheets(wa,wb,r);exact(x.out,y.out,r,"malformed original output bits not numeric qualification");packExact(b.values,packed,r);canaries(owners);++r.guards;
        std::memcpy(wa.prepared.queries.contents(),originalQuery.data(),originalQuery.size());std::memcpy(wb.prepared.queries.contents(),candidateQuery.data(),candidateQuery.size());std::memcpy(a.values.contents(),originalValue.data(),originalValue.size());std::memcpy(b.values.contents(),candidateValue.data(),candidateValue.size());std::memcpy(wa.prepared.selectedBlocks.contents(),originalSelection.data(),originalSelection.size());std::memcpy(wb.prepared.selectedBlocks.contents(),candidateSelection.data(),candidateSelection.size());}
      r.phase="normal-recovery-before-future-causal-proof";recover();
      const auto prefixOut=snapshot(x.out);std::array<std::vector<uint8_t>,5>earlier;const auto stateA=planes(a);for(uint32_t i=0;i<5;++i){const uint64_t n=i==3?512*128*2:i==4?2048*8:2048*(i==2?128:512)*2;const auto*p=static_cast<const uint8_t*>(stateA[i].contents());earlier[i]={p,p+n};}
      r.phase="ordinary-future1-3-7-128-all5physical-cache-planes";uint32_t begin=2048;
      for(uint32_t n:{1u,3u,7u,128u}){auto &u=futureA,&v=futureB;u.rows=v.rows=n;synthetic(u,begin);if(n==128)for(auto input:{u.q,u.k,u.v,u.index})static_cast<uint16_t*>(input.contents())[0]^=0x8000u;
        for(uint32_t i=0;i<4;++i)std::memcpy(u.norms[i].buffer.contents(),x.norms[i].buffer.contents(),u.norms[i].logicalBytes);copyInput(u,v);std::memset(u.out.contents(),0xa5,u.out.sizeBytes());std::memset(v.out.contents(),0xa5,v.out.sizeBytes());resetDiag(u.diag);resetDiag(v.diag);
        CommandGraph gu,gv;p4::addOrdinaryQSAChunk(gu,u.args(),a,o,f,begin,n);p4::addOrdinaryQSAChunk(gv,v.args(),b,o,f,begin,n);
        (void)submit(backend,gu.dispatches());(void)submit(backend,gv.dispatches());require(diag(u.diag)==Sticky && diag(v.diag)==Sticky,"original future diagnostics");exact(u.out,v.out,r,"synthetic future output bits");cacheEqual(a,b,r);const auto now=planes(a);for(uint32_t i=0;i<5;++i){const auto*p=static_cast<const uint8_t*>(now[i].contents());require(!std::memcmp(p,earlier[i].data(),earlier[i].size()),"earlier physical prefix unchanged by future append");}
        exact(x.out,prefixOut,r,"stored prefix output retained during future append");canaries(owners);ledger();begin+=n;}
      for(uint32_t i=0;i<sourceHashes.size();++i)require(digest(immutable[i])==sourceHashes[i],"all projected/norm operands remain immutable after future proof");ledger();
      r.phase="causal-replay-after-future128-perturbations-before-timing";
      resetDiag(x.diag);resetDiag(y.diag);(void)submit(backend,old.dispatches());(void)submit(backend,test.dispatches());
      require(diag(x.diag)==Sticky && diag(y.diag)==Sticky,"future causal replay normal diagnostics");
      exact(x.out,prefixOut,r,"rerun original earlier output after perturbed future KV");exact(y.out,prefixOut,r,"rerun packed earlier output after perturbed future KV");
      require(digest(wa.partials.partitionStatistics)==oldStats && digest(wb.partials.partitionStatistics)==oldStats && digest(wa.partials.partitionValues)==oldNums && digest(wb.partials.partitionValues)==oldNums,"rerun full earlier F32 stats/numerators after perturbed future KV");
      sheets(wa,wb,r);packExact(b.values,packed,r);coupled(backend,old,test,raw,rounded,x,y,r);cacheEqual(a,b,r);canaries(owners);ledger();
      preparedHashes.clear();for(auto v:immutable)preparedHashes.push_back(digest(v)); // Freeze the CURRENT five-plane state after future proof.
      r.phase="normal-recovery-and-freeze-before-inclusive-timing";recover();ledger();const auto countBefore=backend.submissionCount();
      // Warm/timed region: submissions/timestamp bookkeeping ONLY. No CPU
      // buffer read/write, poison/reset, guards, hashes, snapshots or allocator reads.
      for(uint32_t variant=0;variant<2;++variant)while(warm[variant]<.150){require(warmCount[variant]<100000,"warm timestamp bound");const auto t=submit(backend,(variant?test:old).dispatches());warm[variant]+=t.gpuSeconds;++warmCount[variant];}
      for(uint32_t pair=0;pair<18;++pair)for(uint32_t position=0;position<2;++position){const uint32_t variant=(pair+position)%2;timings[variant].push_back(submit(backend,(variant?test:old).dispatches()));}
      timedCount=backend.submissionCount()-countBefore;require(timedCount==36+warmCount[0]+warmCount[1],"exact normal warm/timed submission count");
      r.phase="LAST-shipping-coupled-taps-and-immutability";require(diag(x.diag)==Sticky && diag(y.diag)==Sticky,"LAST normal diagnostics");sheets(wa,wb,r);exact(x.out,y.out,r,"LAST shipping gated BF16");coupled(backend,old,test,raw,rounded,x,y,r);unchanged();canaries(owners);ledger();
    }r.after=backend.memoryStats().allocatedBytes;require(r.after==0,"all fixture/graph owners and host vectors released before admissions");hostAdmission.reset();admission.reset();r.finalReserved=governor.snapshot().reservedBytes;require(r.finalReserved==0,"both held native/host reservations released to zero after all owners/vectors");backend.stop();r.stopped=true;require(backend.memoryStats().allocatedBytes==0,"zero owned ledger after stop");
  }require(r.stopped && r.destroyed,"stop/destructor before success publication");r.phase="complete";
  const auto temporary=r.path.string()+".final-writing";std::ofstream out(temporary);require(bool(out),"final open");out<<std::setprecision(17)<<"{\"schema\":\"original-online-packedV-component-v1\",\"pass\":true,\"GPU_executed\":true,\"fixture\":"<<json::quote(r.mode)<<",\"capture_manifest_sha256\":"<<json::quote(r.manifestHash)
   <<",\"original_online_control_dispatches\":3,\"candidate_inclusive_dispatches\":4,\"coupled_F32_statistics_numerators_quotient_and_BF16_exact\":true,\"inactive_partition_sentinel_bytes_exact\":true,\"all5cache_planes_and_original_future_appends_exact\":true,\"synthetic_future_projections\":true,\"explicit_future_perturbations_preserve_earlier_bits\":true,\"complete_numerical_qualification\":true,\"normal_inclusive_timing_complete\":true"
   <<",\"packedV_logical_bytes\":2097152,\"packedV_charged_owner_bytes\":2129920,\"enumerated_charged_bytes\":"<<r.enumerated<<",\"planned_Governor_bytes\":1073741824,\"actual_owned_delta\":"<<r.current<<",\"actual_peak_owned_delta\":"<<r.peak<<",\"actual_current_device_delta\":"<<r.deviceCurrent<<",\"actual_peak_device_delta\":"<<r.devicePeak
   <<",\"native_admission_bytes\":1073741824,\"host_admission_bytes\":536870912,\"combined_component_admission_bytes\":1610612736,\"engine_physical_minus_host_reserve_limit\":"<<r.engineLimit<<",\"enumerated_host_phase_upper_bound\":"<<r.hostEnumerated<<",\"measured_native_peak_plus_host_planned_bound\":"<<std::max(r.peak,r.devicePeak)+r.hostEnumerated<<",\"host_bookkeeping_allowance\":67108864,\"host_usage_measured\":false,\"final_reserved_bytes\":"<<r.finalReserved
   <<",\"after_fixture_owned_bytes\":"<<r.after<<",\"Gov_denials\":"<<r.denials<<",\"host_measurement_valid\":"<<(r.hostValid?"true":"false")<<",\"growth_allowed\":"<<(r.growth?"true":"false")<<",\"backend_destroyed_before_publication\":true"
   <<",\"checks\":"<<r.checks<<",\"bytes_compared\":"<<r.bytes<<",\"guard_cases\":"<<r.guards<<",\"capture_payload_bytes_read_by_Root\":"<<r.payload<<",\"pairs\":18,\"control_warm_GPU_ms\":"<<warm[0]*1000<<",\"candidate_warm_GPU_ms\":"<<warm[1]*1000<<",\"warm_and_timed_submissions\":"<<timedCount
   <<",\"control_GPU_median_ms\":"<<median(timings[0],true)*1000<<",\"candidate_GPU_median_ms\":"<<median(timings[1],true)*1000<<",\"control_wall_median_ms\":"<<median(timings[0],false)*1000<<",\"candidate_wall_median_ms\":"<<median(timings[1],false)*1000<<",\"GPU_ratio\":"<<median(timings[0],true)/median(timings[1],true)
   <<",\"CPU_buffer_access_in_warm_timed_window\":false,\"whole_model_or_canonicalB1_qualified\":false,\"model_future_greedy_trajectory_qualified\":false}\n";out.close();require(bool(out),"final close");std::filesystem::rename(temporary,r.path);return 0;
}catch(const std::exception&e){r.error=e.what();try{r.write(true,false);}catch(...){}std::cerr<<"online packedV oracle: "<<e.what()<<'\n';return 1;}}}
