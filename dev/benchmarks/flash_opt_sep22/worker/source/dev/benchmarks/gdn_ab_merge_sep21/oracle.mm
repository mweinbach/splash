// Root-only exact A/B scheduling qualifier. CPU mode loads no model/device.
#include "bridge.hpp"
#include "engine/Json.hpp"
#include "Provenance.hpp"
#include "dev/benchmarks/FlashFloatBoundaryAudit.hpp"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace splash;using namespace splash::flash;using namespace splash::metal;
namespace m=gdn_ab_merge;constexpr uint32_t sticky=0x40000000;
void require(bool value,const char *reason){if(!value)throw std::runtime_error(reason);}
std::string hash(const void *data,uint64_t bytes){CC_SHA256_CTX c{};CC_SHA256_Init(&c);const auto *p=static_cast<const uint8_t *>(data);
  while(bytes){const auto part=CC_LONG(std::min<uint64_t>(bytes,1ULL<<30));CC_SHA256_Update(&c,p,part);p+=part;bytes-=part;}std::array<uint8_t,32> digest{};CC_SHA256_Final(digest.data(),&c);
  constexpr char hex[]="0123456789abcdef";std::string out;for(auto x:digest){out+=hex[x>>4];out+=hex[x&15];}return out;}
struct Guard{MetalBuffer base,view;uint64_t bytes;
  Guard(MetalBackend &b,uint64_t length):bytes(length){base=b.allocateBuffer(length+128,BufferStorage::Shared,"GDNAB guarded qualifier");view=b.view(base,64,length);clear();}
  void clear(){std::memset(base.contents(),0x5a,base.sizeBytes());std::memset(view.contents(),0xa5,bytes);}
  void check()const{const auto *p=static_cast<const uint8_t *>(base.contents());for(uint32_t i=0;i<64;++i)require(p[i]==0x5a&&p[64+bytes+i]==0x5a,"GDNABcanaryoverwritten");}
};
std::vector<uint32_t> choices(const char *name,std::initializer_list<uint32_t> defaults,uint32_t limit){const char *raw=std::getenv(name);if(!raw)return defaults;
  std::stringstream stream(raw);std::string item;std::vector<uint32_t> out;while(std::getline(stream,item,',')){require(!item.empty()&&item.front()!='-',"invalidGDNABintegerselector");size_t n=0;const auto x=std::stoul(item,&n);require(n==item.size()&&x<=limit,"GDNABintegerselectorbound");out.push_back(uint32_t(x));}require(!out.empty(),"emptyGDNABselector");return out;}
double median(std::vector<double> x){require(!x.empty(),"emptyGDNABtimings");std::sort(x.begin(),x.end());return x.size()%2?x[x.size()/2]:(x[x.size()/2-1]+x[x.size()/2])*.5;}
struct Times{std::vector<double> gpu,wall;void add(CommandTiming t){require(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>0&&std::isfinite(t.wallSeconds)&&t.wallSeconds>0,"invalidGDNABtiming");gpu.push_back(t.gpuSeconds);wall.push_back(t.wallSeconds);}
  void write(std::ostream &o)const{o<<"{\"median_gpu_ms\":"<<median(gpu)*1000<<",\"median_wall_ms\":"<<median(wall)*1000<<",\"gpu_seconds\":[";for(size_t i=0;i<gpu.size();++i){if(i)o<<',';o<<gpu[i];}o<<"]}";}};
uint64_t mismatches(MetalBuffer a,MetalBuffer b,uint64_t bytes){const auto *x=static_cast<const uint8_t *>(a.contents()),*y=static_cast<const uint8_t *>(b.contents());uint64_t n=0;for(uint64_t i=0;i<bytes;++i)n+=x[i]!=y[i];return n;}
GDNABMergeParams params(const CommandGraph &graph){const auto &d=graph.dispatches()[0];GDNABMergeParams p{};require(d.bytes.size()==1&&d.bytes[0].sizeBytes==sizeof(p),"GDNABparamABI");std::memcpy(&p,d.bytes[0].data,sizeof(p));return p;}
void rawDispatch(CommandGraph &out,const CommandGraph &source,const GDNABMergeParams &p,DispatchSize groups={6,1,2},DispatchSize threads={64,1,1}){
  std::vector<MetalBuffer> b;for(const auto &x:source.dispatches()[0].buffers)b.push_back(x.buffer);out.add(source.dispatches()[0].pipelineName,std::move(b),p,groups,threads);}

bool run(MetalBackend &backend,const FlashWeights &weights,uint32_t layer,uint32_t rows,uint32_t pairs,std::ostream &out){
  const auto prefix="language_model.model.layers."+std::to_string(layer)+".linear_attn";
  const auto &a=weights.projection(prefix+".in_proj_a"),&b=weights.projection(prefix+".in_proj_b");
  Guard input(backend,uint64_t{rows}*2560*2);auto *x=static_cast<uint16_t *>(input.view.contents());
  const char *capture=std::getenv("FLASH_GDN_AB_INPUT");if(capture){const auto bytes=std::filesystem::file_size(capture);require(bytes>=input.bytes&&bytes%(2560*2)==0,"GDNABcaptureextent");
    std::ifstream file(capture,std::ios::binary);file.read(reinterpret_cast<char *>(x),input.bytes);require(bool(file),"GDNABcaptureread");}
  else for(uint64_t i=0;i<input.bytes/2;++i)x[i]=benchmark::bf16(float(int((i*71+i/2560*17)%2047)-1023)/1024);
  const auto inputSHA=hash(input.view.contents(),input.bytes);
  std::vector<std::tuple<MetalBuffer,uint64_t,std::string>> sources;for(const auto *p:{&a,&b})for(const auto *t:{p->weights,p->scales,p->biases})sources.emplace_back(t->buffer,t->logicalBytes,hash(t->buffer.contents(),t->logicalBytes));
  std::array<Guard,4> ao{Guard(backend,rows*48*2),Guard(backend,rows*48*2),Guard(backend,rows*48*2),Guard(backend,rows*48*2)};
  std::array<Guard,4> bo{Guard(backend,rows*48*2),Guard(backend,rows*48*2),Guard(backend,rows*48*2),Guard(backend,rows*48*2)};
  std::array<Guard,4> diag{Guard(backend,4),Guard(backend,4),Guard(backend,4),Guard(backend,4)};
  std::array<Guard,2> ar{Guard(backend,rows*48*4),Guard(backend,rows*48*4)},br{Guard(backend,rows*48*4),Guard(backend,rows*48*4)};
  for(auto &d:diag)std::memcpy(d.view.contents(),&sticky,4);
  std::array<CommandGraph,4> graphs;
  addAffine(graphs[0],input.view,a,ao[0].view,diag[0].view,rows);addAffine(graphs[0],input.view,b,bo[0].view,diag[0].view,rows);
  m::add(graphs[1],input.view,a,b,ao[1].view,bo[1].view,diag[1].view,rows);
  m::add(graphs[2],input.view,a,b,ao[2].view,bo[2].view,diag[2].view,rows,{ar[0].view,br[0].view},0);
  m::add(graphs[2],input.view,a,b,ao[2].view,bo[2].view,diag[2].view,rows,{ar[0].view,br[0].view},1);
  m::add(graphs[3],input.view,a,b,ao[3].view,bo[3].view,diag[3].view,rows,{ar[1].view,br[1].view});
  const auto healthy=[&]{input.check();for(size_t i=0;i<4;++i){ao[i].check();bo[i].check();diag[i].check();uint32_t status;std::memcpy(&status,diag[i].view.contents(),4);require(status==sticky,"GDNABsticky diagnosticchanged");}for(size_t i=0;i<2;++i){ar[i].check();br[i].check();}};
  for(auto &graph:graphs)(void)backend.submitCommand(graph.dispatches());healthy();
  require(!mismatches(ao[0].view,ao[2].view,ao[0].bytes)&&!mismatches(bo[0].view,bo[2].view,bo[0].bytes),"GDNABstock_single_probe mismatch");
  const uint64_t af=mismatches(ar[0].view,ar[1].view,ar[0].bytes),bf=mismatches(br[0].view,br[1].view,br[0].bytes),ab=mismatches(ao[0].view,ao[1].view,ao[0].bytes),bb=mismatches(bo[0].view,bo[1].view,bo[0].bytes);
  const bool exact=!af&&!bf&&!ab&&!bb&&!mismatches(ao[1].view,ao[3].view,ao[1].bytes)&&!mismatches(bo[1].view,bo[3].view,bo[1].bytes);
  require(exact,"GDNABfullF32_BF16producer mismatch");
  // Host failures must not mutate the target graph; original per-projection
  // view guards and pair cross-plane readonly/output guards both remain active.
  uint32_t aliases=0;
  const auto rejected=[&](MetalBuffer left,MetalBuffer right,MetalBuffer d,uint32_t r){CommandGraph empty;bool rejected=false;try{m::add(empty,input.view,a,b,left,right,d,r);}catch(const std::exception &){rejected=true;}require(rejected&&empty.empty(),"GDNABhostguardmutatedgraph");++aliases;};
  rejected(ao[1].view,ao[1].view,diag[1].view,rows);rejected(input.view,bo[1].view,diag[1].view,rows);rejected(a.weights->buffer,bo[1].view,diag[1].view,rows);
  rejected(ao[1].view,bo[1].view,ao[1].view,rows);rejected(backend.view(ao[1].view,0,2),bo[1].view,diag[1].view,rows);rejected(ao[1].view,bo[1].view,diag[1].view,2);
  const auto original=params(graphs[1]);
  for(uint32_t bad:{0u,1u,2u,3u,4u,5u,6u,7u}){auto p=original;switch(bad){case 0:p.a.rows=2;break;case 1:p.b.input_size=2559;break;case 2:p.a.output_size=47;break;case 3:p.b.flags=1;break;case 4:p.a.bits=4;break;case 5:p.b.group_size=32;break;case 6:p.a.weight_row_stride_bytes=1;break;case 7:p.b.parameter_row_stride_bytes=1;break;}
    const auto beforeA=hash(ao[1].view.contents(),ao[1].bytes),beforeB=hash(bo[1].view.contents(),bo[1].bytes);CommandGraph trap;rawDispatch(trap,graphs[1],p,{6,rows,2});(void)backend.submitCommand(trap.dispatches());
    uint32_t status;std::memcpy(&status,diag[1].view.contents(),4);require(status==(sticky|2u),"GDNABbadbothplane diagnostic");require(hash(ao[1].view.contents(),ao[1].bytes)==beforeA&&hash(bo[1].view.contents(),bo[1].bytes)==beforeB,"GDNABbadparamswroteoutput");std::memcpy(diag[1].view.contents(),&sticky,4);}
  // Poison the two plane outputs. Same prepared candidate must reconstruct
  // them; no hidden dependency on separate original A/B commands.
  const auto goldenA=hash(ao[1].view.contents(),ao[1].bytes),goldenB=hash(bo[1].view.contents(),bo[1].bytes);
  std::memset(ao[1].view.contents(),0xa5,ao[1].bytes);std::memset(bo[1].view.contents(),0xa5,bo[1].bytes);(void)backend.submitCommand(graphs[1].dispatches());
  require(hash(ao[1].view.contents(),ao[1].bytes)==goldenA&&hash(bo[1].view.contents(),bo[1].bytes)==goldenB,"GDNABpoisonreplay");healthy();
  std::array<double,2> warm{};uint64_t calls=0;
  while(warm[0]<.150||warm[1]<.150){for(uint32_t route:{0u,1u}){const auto t=backend.submitCommand(graphs[route].dispatches());require(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>0,"GDNABwarmtiming");warm[route]+=t.gpuSeconds;require(++calls<=50000,"GDNABwarmbudget");}}
  std::array<Times,2> times;std::array<std::array<Times,2>,2> positions;
  for(uint32_t cycle=0;cycle<pairs;++cycle)for(uint32_t pos=0;pos<2;++pos){const uint32_t route=(cycle+pos)%2;const auto t=backend.submitCommand(graphs[route].dispatches());times[route].add(t);positions[route][pos].add(t);}
  healthy();require(hash(input.view.contents(),input.bytes)==inputSHA,"GDNABinputmutated");for(const auto &[buffer,bytes,before]:sources)require(hash(buffer.contents(),bytes)==before,"GDNABimmutable sourcechanged");
  require(hash(ao[1].view.contents(),ao[1].bytes)==goldenA&&hash(bo[1].view.contents(),bo[1].bytes)==goldenB,"GDNABtimedoutputchanged");
  out<<"{\"layer\":"<<layer<<",\"rows\":"<<rows<<",\"a_bits\":"<<a.bits<<",\"a_group\":"<<a.groupSize<<",\"b_bits\":"<<b.bits<<",\"b_group\":"<<b.groupSize
      <<",\"exact_rawF32_and_BF16\":true,\"a_rawF32_byte_mismatches\":"<<af<<",\"b_rawF32_byte_mismatches\":"<<bf<<",\"a_BF16_byte_mismatches\":"<<ab<<",\"b_BF16_byte_mismatches\":"<<bb
      <<",\"host_alias_extent_guard_rejections\":"<<aliases<<",\"malformed_shader_param_no_write_cases\":8,\"guards_sticky_sources_immutable\":true,\"poisonreplay\":true"
      <<",\"input_policy\":"<<json::quote(capture?"caller supplied captured BF16 prefix slice; provenance callerattested":"declared deterministic synthetic BF16 input; not livedecode")
      <<",\"control_dispatches\":2,\"candidate_dispatches\":1,\"gpu_warm_seconds\":["<<warm[0]<<','<<warm[1]<<"],\"warm_commands\":"<<calls<<",\"cpu_payload_touches_warm_to_endtiming\":0,\"control\":";times[0].write(out);out<<",\"merged\":";times[1].write(out);
  out<<",\"position_strata\":[";for(uint32_t p=0;p<2;++p){if(p)out<<',';out<<"{\"control\":";positions[0][p].write(out);out<<",\"merged\":";positions[1][p].write(out);out<<'}';}out<<"]}";return true;
}
void cpu(){static_assert(sizeof(CommandTiming)==200);static_assert(sizeof(GDNABMergeParams)==128);uint32_t cases=0;
  for(uint32_t rows:{1u,4u})for(uint32_t a:{5u,6u})for(uint32_t b:{5u,6u}){GDNABMergeParams p{};p.a.rows=p.b.rows=rows;p.a.bits=a;p.b.bits=b;++cases;}
  std::cout<<"{\"pass\":true,\"gpu_work\":false,\"model_payload_reads\":0,\"ABI_params\":128,\"ABI_timing\":200,\"formatrow_cases\":"<<cases<<"}\n";}
} // namespace
int main(int argc,char **argv){@autoreleasepool{try{
  if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){cpu();return 0;}
  require(argc==5&&std::string_view(argv[1])=="--gpu","oracle --gpu LIB PACKAGE NEW_REPORT");require(!std::filesystem::exists(argv[4]),"freshGDNABreportrequired");
  const auto layers=choices("FLASH_GDN_AB_LAYERS",{0,1,21},47),rows=choices("FLASH_GDN_AB_ROWS",{1,4},4);const uint32_t pairs=choices("FLASH_GDN_AB_PAIRS",{10},40)[0];require(pairs&&pairs%2==0,"evenGDNABpairsrequired");
  require(flashAffineFastEnabled(),"GDNABactiveQMV_F32required");MetalBackend backend(argv[2]);const auto weights=FlashWeights::load(backend,argv[3]);
  std::ofstream out(argv[4]);require(bool(out),"GDNABreportcreate");out<<std::setprecision(17)<<"{\"schema\":\"splash-gdn-ab-exact-scheduling-oracle-v1\",\"provenance\":"<<kGDNABProvenance<<",\"model_quality_qualified\":false,\"cases\":[";
  bool first=true;for(uint32_t layer:layers)for(uint32_t count:rows){require(count==1||count==4,"GDNABscopeR1/R4");if(!first)out<<',';first=false;(void)run(backend,weights,layer,count,pairs,out);}out<<"],\"pass\":true}\n";require(bool(out),"GDNABreportwrite");return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}}
