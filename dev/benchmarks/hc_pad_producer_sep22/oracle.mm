// Root-owned R4 verifier HC component. --cpu-only creates no backend/model.
#include "bridge.hpp"
#include "HCProducerPadProvenance.hpp"
#include "engine/Json.hpp"
#include "engine/MemoryGovernor.hpp"
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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace splash;using namespace splash::flash;using namespace splash::metal;
namespace c=splash::flash::hc_pad_sep22;namespace fs=std::filesystem;
constexpr uint32_t kSticky=0x40000000u;
static_assert(sizeof(CommandTiming)==200);
void require(bool ok,const std::string &why){if(!ok)throw std::runtime_error(why);}
uint16_t bf16(float value){uint32_t bits=std::bit_cast<uint32_t>(value);return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);}
std::string hash(const void *raw,uint64_t bytes){CC_SHA256_CTX state{};CC_SHA256_Init(&state);const auto *p=static_cast<const uint8_t *>(raw);
  while(bytes){const auto n=CC_LONG(std::min<uint64_t>(bytes,UINT32_MAX));CC_SHA256_Update(&state,p,n);p+=n;bytes-=n;}std::array<uint8_t,32> result{};CC_SHA256_Final(result.data(),&state);
  std::ostringstream out;for(auto b:result)out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(b);return out.str();}
std::string fileHash(const fs::path &path){std::ifstream in(path,std::ios::binary);require(bool(in),"cannot open source/artifact provenance");std::vector<char> bytes(fs::file_size(path));in.read(bytes.data(),std::streamsize(bytes.size()));require(bool(in),"source/artifact provenance read failed");return hash(bytes.data(),bytes.size());}
struct Guard {
  MetalBuffer base,view;uint64_t bytes;
  Guard(MetalBackend &backend,uint64_t count):bytes(count){base=backend.allocateBuffer(count+128,BufferStorage::Shared,"HCpad guard");std::memset(base.contents(),0x5a,count+128);view=backend.view(base,64,count);std::memset(view.contents(),0xa5,count);}
  void check()const{const auto *p=static_cast<const uint8_t *>(base.contents());for(uint32_t i=0;i<64;++i)require(p[i]==0x5a&&p[64+bytes+i]==0x5a,"HCpad canary changed");}
};
void exact(const MetalBuffer &a,const MetalBuffer &b,uint64_t bytes,const char *label){require(a.sizeBytes()>=bytes&&b.sizeBytes()>=bytes,"comparison extent too short");require(!std::memcmp(a.contents(),b.contents(),bytes),std::string("not byte exact: ")+label);}
std::vector<std::string> prefixes(){std::vector<std::string> out;for(uint32_t layer=0;layer<48;++layer)for(const char *role:{"attn_hyper_connection","mlp_hyper_connection"})out.push_back("language_model.model.layers."+std::to_string(layer)+"."+role);out.push_back("language_model.model.hyper_connection_mixer");require(out.size()==97,"HC prefix inventory not97");return out;}
struct Route {
  Guard activation,gates,padding,diag,mixed,rawDownF,rawDownB,rawUpF,rawUpB;
  Route(MetalBackend &b,uint32_t rawStride):activation(b,5120),gates(b,32),padding(b,5120),diag(b,4),mixed(b,4*2560*2),rawDownF(b,4*rawStride*4),rawDownB(b,4*rawStride*2),rawUpF(b,4*10240*4),rawUpB(b,4*10240*2){*static_cast<uint32_t *>(diag.view.contents())=kSticky;}
  void check()const{for(const Guard *g:{&activation,&gates,&padding,&diag,&mixed,&rawDownF,&rawDownB,&rawUpF,&rawUpB})g->check();require(*static_cast<const uint32_t *>(diag.view.contents())==kSticky,"HCpad sticky diagnostic changed");}
};
struct Fixture {
  std::string prefix,inputSHA;Guard normalized;const FlashAffineProjection &down;const FlashAffineProjection *injection;const FlashTensor &up;
  std::array<std::unique_ptr<Route>,4> route;
  Fixture(MetalBackend &b,const FlashWeights &weights,const FlashFloatDenseCache &cache,std::string name,uint32_t salt,const fs::path &inputDirectory):prefix(std::move(name)),normalized(b,4*10240*2),down(weights.projection(prefix+".input_mix_weight_down")),
    injection(weights.contains(prefix+".block_inject_weight.weight")?&weights.projection(prefix+".block_inject_weight"):nullptr),up(cache.tensor(prefix+".input_mix_weight_up")){
    auto *values=static_cast<uint16_t *>(normalized.view.contents());
    if(!inputDirectory.empty()){const auto path=inputDirectory/(prefix+".bf16");require(fs::file_size(path)==normalized.bytes,"captured normalized input size differs");std::ifstream in(path,std::ios::binary);in.read(reinterpret_cast<char *>(values),std::streamsize(normalized.bytes));require(bool(in),"captured normalized input read failed");}
    else for(uint64_t i=0;i<normalized.bytes/2;++i)values[i]=bf16(float(int((i*73+i/10240*17+salt*11)%2047)-1023)/1024.0f);
    for(uint64_t i=0;i<normalized.bytes/2;++i)require(std::isfinite(std::bit_cast<float>(uint32_t(values[i])<<16)),"normalized input not finite");
    inputSHA=hash(normalized.view.contents(),normalized.bytes);for(auto &r:route)r=std::make_unique<Route>(b,320+(injection?4:0));}
  void append(MetalBackend &b,CommandGraph &graph,FlashFloatDenseSmallRowsWorkspace &workspace,uint32_t index,bool probe){
    auto &r=*route[index];const bool candidate=index%2;
    c::addDown(graph,normalized.view,down,injection,r.activation.view,r.gates.view,r.diag.view,c::Scope::Verify,candidate,probe?c::DownDebug{r.rawDownF.view,r.rawDownB.view}:c::DownDebug{});
    c::addUp(b,graph,normalized.view,r.activation.view,up,r.mixed.view,r.diag.view,workspace,r.padding.view,c::Scope::Verify,candidate,probe?c::UpDebug{r.rawUpB.view,r.rawUpF.view}:c::UpDebug{});}
  void check()const{normalized.check();for(const auto &r:route)r->check();require(hash(normalized.view.contents(),normalized.bytes)==inputSHA,"immutable normalized input changed");}
  void qualify()const{
    const auto &a=*route[0],&b=*route[1],&pa=*route[2],&pb=*route[3];const uint64_t raw=4*(320+(injection?4:0));
    exact(a.activation.view,pa.activation.view,2560,"control down probe activation");exact(b.activation.view,pb.activation.view,5120,"candidate down probe activation/padding");
    exact(a.activation.view,b.activation.view,2560,"producer active BF16");exact(a.gates.view,b.gates.view,32,"producer injection");exact(a.gates.view,pa.gates.view,32,"control probe injection");exact(b.gates.view,pb.gates.view,32,"candidate probe injection");
    exact(pa.rawDownF.view,pb.rawDownF.view,raw*4,"producer full raw F32");exact(pa.rawDownB.view,pb.rawDownB.view,raw*2,"producer full raw BF16");
    exact(pa.rawUpF.view,pb.rawUpF.view,4*10240*4,"up full raw F32");exact(pa.rawUpB.view,pb.rawUpB.view,4*10240*2,"up full raw BF16");
    exact(a.mixed.view,b.mixed.view,4*2560*2,"up BF16 mixed");exact(a.mixed.view,pa.mixed.view,4*2560*2,"control up probe mixed");exact(b.mixed.view,pb.mixed.view,4*2560*2,"candidate up probe mixed");
    exact(a.padding.view,b.activation.view,5120,"all padded tensor8rows");exact(pa.padding.view,pb.activation.view,5120,"probe padded tensor8rows");
    const auto *words=static_cast<const uint16_t *>(b.activation.view.contents());for(uint32_t i=1280;i<2560;++i)require(words[i]==0,"candidate inactive rows not positive zero");
    check();}
};
struct Times {std::vector<double> gpu,wall;void add(CommandTiming t){require(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>1e-9&&std::isfinite(t.wallSeconds)&&t.wallSeconds>1e-9,"invalid command timing");gpu.push_back(t.gpuSeconds);wall.push_back(t.wallSeconds);}static double median(std::vector<double> x){require(!x.empty(),"empty timings");std::sort(x.begin(),x.end());return x.size()%2?x[x.size()/2]:(x[x.size()/2-1]+x[x.size()/2])/2;}
  void json(std::ostream &out)const{out<<"{\"median_gpu_ms\":"<<median(gpu)*1000<<",\"median_wall_ms\":"<<median(wall)*1000<<",\"gpu_ms\":[";for(size_t i=0;i<gpu.size();++i){if(i)out<<',';out<<gpu[i]*1000;}out<<"]}";}};
template<class F>void rejected(F &&f,const char *label){bool bad=false;try{f();}catch(const std::exception &){bad=true;}require(bad,std::string("host guard failed: ")+label);}
uint32_t hostGuards(MetalBackend &backend,Fixture &f,FlashFloatDenseSmallRowsWorkspace &workspace){uint32_t checks=0;auto &r=*f.route[1];
  for(auto scope:{c::Scope::Prefill,c::Scope::Ordinary,c::Scope::Head,c::Scope::Batch}){CommandGraph g;rejected([&]{c::addDown(g,f.normalized.view,f.down,f.injection,r.activation.view,r.gates.view,r.diag.view,scope,true);},"caller scope");require(g.empty(),"invalid scope mutated graph");++checks;}
  {CommandGraph g;rejected([&]{c::addDown(g,f.normalized.view,f.down,f.injection,backend.view(r.activation.view,0,2560),r.gates.view,r.diag.view,c::Scope::Verify,true);},"short expanded scratch");require(g.empty(),"short output mutated graph");++checks;}
  {CommandGraph g;rejected([&]{c::addDown(g,f.normalized.view,f.down,f.injection,backend.view(f.normalized.view,0,5120),r.gates.view,r.diag.view,c::Scope::Verify,true);},"producer alias");require(g.empty(),"producer alias mutated graph");++checks;}
  {CommandGraph g;rejected([&]{c::addUp(backend,g,f.normalized.view,r.activation.view,f.up,r.mixed.view,r.diag.view,workspace,r.activation.view,c::Scope::Verify,false);},"consumer alias");require(g.empty(),"consumer alias mutated graph");++checks;}
  return checks;}
uint32_t shaderGuards(MetalBackend &backend,Fixture &f,FlashFloatDenseSmallRowsWorkspace &workspace){
  auto &r=*f.route[3];CommandGraph valid;f.append(backend,valid,workspace,3,true);const auto &down=valid.dispatches()[0],&up=valid.dispatches()[1];
  const auto literal=c::params<c::FlashHCDownPadParams>(down,12);const auto upLiteral=c::params<FlashFloatDenseSmallRowsParams>(up,7);uint32_t checks=0;
  const std::array outputs{r.activation.view,r.gates.view,r.rawDownF.view,r.rawDownB.view,r.mixed.view,r.rawUpF.view,r.rawUpB.view};
  const auto negative=[&](const std::string &name,std::vector<MetalBuffer> bound,const auto &p,DispatchSize groups,DispatchSize threads){
    std::vector<std::string> before;for(const auto &b:outputs)before.push_back(hash(b.contents(),b.sizeBytes()));*static_cast<uint32_t *>(r.diag.view.contents())=kSticky;
    CommandGraph g;g.add(name,std::move(bound),p,groups,threads);(void)backend.submitCommand(g.dispatches());
    require(*static_cast<const uint32_t *>(r.diag.view.contents())==(kSticky|2u),"malformed/partial shader did not preserve sticky+shape flag");
    for(size_t i=0;i<outputs.size();++i)require(hash(outputs[i].contents(),outputs[i].sizeBytes())==before[i],"malformed shader wrote payload");
    for(const Guard *b:{&r.activation,&r.gates,&r.rawDownF,&r.rawDownB,&r.mixed,&r.rawUpF,&r.rawUpB,&r.diag})b->check();++checks;};
  for(uint32_t which=0;which<8;++which){auto p=literal;
    if(which==0)p.literal.rows=3;else if(which==1)p.padded_rows=7;else if(which==2)p.reserved0=1;else if(which==3)p.literal.arithmetic_mode=1;
    else if(which==4)p.literal.down.input_size=10239;else if(which==5)p.literal.injection.output_size=3;else if(which==6)p.literal.simdgroups=8;else p.literal.write_raw_up=2;
    negative(down.pipelineName,c::buffers(down),p,down.threadgroups,down.threadsPerThreadgroup);}
  negative(down.pipelineName,c::buffers(down),literal,{down.threadgroups.x-1,4,1},{128,1,1});
  negative(down.pipelineName,c::buffers(down),literal,{down.threadgroups.x,3,1},{128,1,1});
  negative(down.pipelineName,c::buffers(down),literal,{down.threadgroups.x+1,4,1},{128,1,1});
  negative(down.pipelineName,c::buffers(down),literal,down.threadgroups,{64,1,1});
  for(uint32_t which=0;which<4;++which){auto p=upLiteral;if(which==0)p.rows=3;else if(which==1)p.padded_rows=7;else if(which==2)p.output_begin=1;else p.tile_rows=16;
    negative(up.pipelineName,c::buffers(up),p,up.threadgroups,up.threadsPerThreadgroup);}
  *static_cast<uint32_t *>(r.diag.view.contents())=kSticky;return checks;
}
uint32_t nonfiniteGuard(MetalBackend &backend,Fixture &f){auto *x=static_cast<uint16_t *>(f.normalized.view.contents());const uint16_t saved=x[0];x[0]=0x7fc0;
  for(uint32_t i=0;i<2;++i){auto &r=*f.route[i];*static_cast<uint32_t *>(r.diag.view.contents())=kSticky;CommandGraph graph;
    c::addDown(graph,f.normalized.view,f.down,f.injection,r.activation.view,r.gates.view,r.diag.view,c::Scope::Verify,bool(i));(void)backend.submitCommand(graph.dispatches());require(*static_cast<const uint32_t *>(r.diag.view.contents())==(kSticky|4u),"producer failed activated finite sticky check");*static_cast<uint32_t *>(r.diag.view.contents())=kSticky;}
  x[0]=saved;return 2;}
void cpu(){require(sizeof(c::FlashHCDownPadParams)==176&&sizeof(FlashHCFusedParams)==160&&sizeof(FlashFloatDenseSmallRowsParams)==32,"ABI changed");
  for(auto s:{c::Scope::Prefill,c::Scope::Ordinary,c::Scope::Head,c::Scope::Batch})require(!c::eligible(s,4,5120),"nonverify eligibility");for(uint32_t r:{0,1,2,3,5,8,16})require(!c::eligible(c::Scope::Verify,r,5120),"wrong rows eligibility");require(c::eligible(c::Scope::Verify,4,5120)&&!c::eligible(c::Scope::Verify,4,5119),"scratch eligibility");
  std::array<uint32_t,2560> owners{};for(uint32_t n=0;n<320;++n)for(uint32_t r=4;r<8;++r)++owners[r*320+n];for(uint32_t i=0;i<2560;++i)require(owners[i]==uint32_t(i>=1280),"padding owner collision/active clobber");require(prefixes().size()==97,"inventory97");
  std::cout<<"{\"pass\":true,\"gpu_executed\":false,\"model_payload_bytes_read\":0,\"checks\":[\"ABI176_160_32_timing200\",\"verifyR4_only\",\"scratch5120\",\"padding_unique_owners_noactive_writes\",\"all97prefixes\"]}\n";}
}
int main(int argc,char **argv){@autoreleasepool{std::string phase="arguments";fs::path report;try{
  if(argc==2&&std::string_view(argv[1])=="--cpu-only"){cpu();return 0;}require(argc==5&&std::string_view(argv[1])=="--gpu","usage: oracle --gpu METALLIB PACKAGE FRESH_REPORT | --cpu-only");report=argv[4];require(!fs::exists(report)&&!fs::exists(report.string()+".partial"),"fresh report required");
  require(fileHash(argv[2])==kHCProducerPadMetallibSHA,"sealed library changed");uint32_t pairs=10;if(const char *p=std::getenv("HC_PAD_PAIRS")){size_t used=0;pairs=uint32_t(std::stoul(p,&used));require(used==std::strlen(p),"pairs malformed");}require(pairs&&pairs<=32&&pairs%2==0,"positive even pair count required");
  require(std::getenv("SPLASH_FLASH_ALLROWS_FULL512_TARGET")&&std::string_view(std::getenv("SPLASH_FLASH_ALLROWS_FULL512_TARGET"))=="1","component needs parent remaining-original loader");
  require(std::getenv("SPLASH_FLASH_PLE_SSD_STREAMING")&&std::string_view(std::getenv("SPLASH_FLASH_PLE_SSD_STREAMING"))=="1","component needs parent SSD geometry");
  require(std::getenv("SPLASH_FLASH_OPERAND_STORE"),"component needs original saved F32 operands");
  const fs::path inputDirectory=std::getenv("HC_PAD_INPUT_DIRECTORY")?std::getenv("HC_PAD_INPUT_DIRECTORY"):"";
  phase="backend";MetalBackend backend(argv[2]);const auto weights=FlashWeights::load(backend,argv[3]);auto names=prefixes();std::vector<std::string> upNames;for(const auto &name:names)upNames.push_back(name+".input_mix_weight_up");
  const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);engine::MemoryGovernor governor(backend,physical-reserve,reserve);const uint64_t planned=FlashFloatDenseCache::plannedBytes(weights,upNames)+(256ULL<<20);
  auto admission=governor.tryReserve(planned);require(bool(admission),"governor denied selected97 F32 cache/fixtures");phase="cache";FlashFloatDenseCache cache(backend,weights,upNames);FlashFloatDenseSmallRowsWorkspace workspace(backend);
  require(cache.persistedTensorCount()==97&&cache.persistedPayloadBytes()==1271398400ULL,"all97 original F32 up operands must be saved full views");
  admission->commit();
  std::vector<std::unique_ptr<Fixture>> fixtures;std::array<CommandGraph,4> graphs;std::vector<std::pair<MetalBuffer,std::string>> immutable;
  const auto retain=[&](const FlashTensor &t){for(const auto &entry:immutable)if(entry.first.sameView(t.buffer))return;immutable.emplace_back(t.buffer,hash(t.buffer.contents(),t.buffer.sizeBytes()));};
  phase="prepare";for(uint32_t i=0;i<97;++i){auto f=std::make_unique<Fixture>(backend,weights,cache,names[i],i,inputDirectory);for(const auto *p:{&f->down,f->injection})if(p)for(const auto *t:{p->weights,p->scales,p->biases})retain(*t);retain(f->up);for(uint32_t r=0;r<4;++r)f->append(backend,graphs[r],workspace,r,r>=2);fixtures.push_back(std::move(f));}
  require(graphs[0].dispatches().size()==291&&graphs[1].dispatches().size()==194&&graphs[2].dispatches().size()==291&&graphs[3].dispatches().size()==194,"97-call pad savings changed");
  const uint32_t hostChecks=hostGuards(backend,*fixtures.front(),workspace);phase="shader_guards";const uint32_t shaderChecks=shaderGuards(backend,*fixtures.front(),workspace);const uint32_t nonfiniteChecks=nonfiniteGuard(backend,*fixtures.front());phase="prove";for(uint32_t r:{0u,2u,3u,1u})(void)backend.submitCommand(graphs[r].dispatches());for(const auto &f:fixtures)f->qualify();
  // All payload/canary reads finish before sustained GPU warm starts.
  phase="warm";std::array<double,2> warmed{};uint32_t warmCommands=0;while(warmed[0]<.150||warmed[1]<.150)for(uint32_t r:{0u,1u}){const auto t=backend.submitCommand(graphs[r].dispatches());require(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>1e-9,"warm timer invalid");warmed[r]+=t.gpuSeconds;require(++warmCommands<10000,"warm did not converge");}
  phase="timing";std::array<Times,2> times;std::array<std::array<Times,2>,2> positions;for(uint32_t p=0;p<pairs;++p)for(uint32_t order=0;order<2;++order){const uint32_t r=(p+order)%2;const auto t=backend.submitCommand(graphs[r].dispatches());times[r].add(t);positions[r][order].add(t);}
  // Payload reads resume only after every timed position has completed.
  phase="final_proof";for(const auto &f:fixtures)f->qualify();for(const auto &[buffer,initial]:immutable)require(hash(buffer.contents(),buffer.sizeBytes())==initial,"immutable coefficient changed");
  std::ofstream out(report.string()+".partial");require(bool(out),"report open failed");out<<std::setprecision(17)<<"{\"schema\":\"hc-down-pad-up-r4-verify-component-v1\",\"pass\":true,\"scope\":\"isolated97 HC down/pad/F32up calls; no Pref/AR/head/batch/fullworker claim\",\"cases\":97,\"rows\":4,\"padded_rows\":8,\"additional_coefficient_or_workspace_bytes\":0,\"removed_padding_dispatches\":97,\"control_dispatches\":291,\"candidate_dispatches\":194,\"all_active_producer_F32_BF16_activation_injection_exact\":true,\"all_up_F32_BF16_mix_exact\":true,\"all_inactive_positive_zero\":true,\"canaries_sticky_immutable_clean\":true,\"host_guard_checks\":"<<hostChecks<<",\"shader_no_write_guard_checks\":"<<shaderChecks<<",\"nonfinite_sticky_checks\":"<<nonfiniteChecks<<",\"input_policy\":"<<json::quote(inputDirectory.empty()?"deterministic synthetic normalized perrole":"Root captured normalized perrole")<<",\"coefficient_source\":\"original packed down/injection and original saved F32 up\",\"host_timing_ABI\":200,\"source_identity\":"<<json::quote(weights.sourceIdentity())<<",\"manifest_identity\":"<<json::quote(weights.manifestFingerprint())<<",\"library_sha256\":"<<json::quote(kHCProducerPadMetallibSHA)<<",\"build_provenance\":"<<kHCProducerPadProvenance<<",\"GPUwarm_seconds\":["<<warmed[0]<<','<<warmed[1]<<"],\"CPU_payload_touches_between_warm_and_timing\":0,\"control_timing\":";times[0].json(out);out<<",\"candidate_timing\":";times[1].json(out);out<<",\"position_strata\":[";
  for(uint32_t p=0;p<2;++p){if(p)out<<',';out<<"{\"control\":";positions[0][p].json(out);out<<",\"candidate\":";positions[1][p].json(out);out<<'}';}out<<"]}\n";out.close();require(bool(out),"report finish failed");fs::rename(report.string()+".partial",report);std::cout<<"{\"pass\":true,\"report\":"<<json::quote(report.string())<<"}\n";return 0;
}catch(const std::exception &error){std::cerr<<"HCpad: "<<error.what()<<'\n';if(!report.empty()){std::ofstream out(report.string()+".failure.json");out<<"{\"pass\":false,\"qualification_complete\":false,\"phase\":"<<json::quote(phase)<<",\"error\":"<<json::quote(error.what())<<"}\n";}return 1;}}}
