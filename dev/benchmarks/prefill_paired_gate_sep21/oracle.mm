// Root-only one-layer primitive. CPU self-test creates no backend or mapping.
#include "packing.hpp"
#include "abi.h"
#include "flash/FlashMoE.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "metal/abi/FlashMoEDirectA.h"
#include "dev/benchmarks/prefill_moe_sep21/w8a8/cache.hpp"
#include "dev/benchmarks/prefill_moe_sep21/w8a8/precision.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>

namespace {
using namespace splash::metal;using namespace splash::flash;
namespace pair=splash::bench::paired_gate;namespace one=splash::flash::qmv_one_layer;
namespace w8=splash::flash::prefill_moe_w8a8;
constexpr uint32_t kSticky=0x80000000;
void require(bool ok,const char *why){if(!ok)throw std::invalid_argument(why);}
uint32_t integer(const char *raw,uint32_t low,uint32_t high){std::string s(raw);require(!s.empty()&&std::all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}),"CLI integer invalid");size_t n=0;auto v=std::stoul(s,&n);require(n==s.size()&&v>=low&&v<=high,"CLI integer outside bounds");return uint32_t(v);}
uint16_t bf16(float x){const uint32_t b=std::bit_cast<uint32_t>(x);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
float number(uint16_t b){return std::bit_cast<float>(uint32_t(b)<<16);}
struct Guard{MetalBuffer base;uint64_t bytes;bool clean()const{const auto *p=static_cast<const uint8_t *>(base.contents());return std::all_of(p,p+128,[](uint8_t v){return v==0x5a;})&&std::all_of(p+128+bytes,p+256+bytes,[](uint8_t v){return v==0x5a;});}};
MetalBuffer guarded(MetalBackend &b,uint64_t bytes,std::vector<Guard>&guards){auto base=b.allocateBuffer(bytes+256,BufferStorage::Shared,"paired gate guarded plane");std::memset(base.contents(),0x5a,base.sizeBytes());auto view=b.view(base,128,bytes);std::memset(view.contents(),0xa5,bytes);guards.push_back({base,bytes});return view;}
template<class T>MetalBuffer upload(MetalBackend &b,const std::vector<T>&v,std::vector<Guard>&g){auto x=guarded(b,v.size()*sizeof(T),g);std::memcpy(x.contents(),v.data(),x.sizeBytes());return x;}
void guardScratch(MetalBackend &b,FlashMoEBlockedScratch &s,std::vector<Guard>&g){for(auto *x:{&s.buckets.counts,&s.buckets.offsets,&s.buckets.routeMap,&s.buckets.canonicalToPacked,&s.buckets.packedInputs,&s.buckets.jobOffsets,&s.buckets.jobCount,&s.buckets.tileJobs,&s.packedActivated,&s.scatteredDown})*x=guarded(b,x->sizeBytes(),g);}
std::string hash(MetalBuffer x){return one::detail::hash(x.contents(),x.sizeBytes());}
struct Error{uint64_t cells=0,changed=0,nonfinite=0;double ee=0,aa=0,rr=0,dot=0,maxAbs=0;
  void add(uint16_t a,uint16_t b){cells++;changed+=a!=b;const double x=number(a),y=number(b);if(!std::isfinite(x)||!std::isfinite(y)){nonfinite++;return;}const double e=x-y;ee+=e*e;aa+=x*x;rr+=y*y;dot+=x*y;maxAbs=std::max(maxAbs,std::abs(e));}
  double l2()const{return std::sqrt(ee/std::max(rr,1e-300));}double cosine()const{return aa==0&&rr==0?1:dot/std::sqrt(std::max(aa*rr,1e-300));}
  bool guard()const{return !nonfinite&&l2()<=w8::precision::kMaximumProducerRelativeL2&&cosine()>=w8::precision::kMinimumProducerCosine;}
  void write(std::ostream&o)const{o<<"{\"cells\":"<<cells<<",\"changed_bf16\":"<<changed<<",\"nonfinite\":"<<nonfinite<<",\"relative_l2\":"<<l2()<<",\"cosine\":"<<cosine()<<",\"max_abs\":"<<maxAbs<<'}';}
};
Error compare(MetalBuffer a,MetalBuffer b,uint64_t cells){Error e;const auto*x=static_cast<const uint16_t *>(a.contents()),*y=static_cast<const uint16_t *>(b.contents());for(uint64_t i=0;i<cells;++i)e.add(x[i],y[i]);return e;}
uint64_t changed(MetalBuffer a,MetalBuffer b,uint64_t cells,uint32_t bytes){const auto*x=static_cast<const uint8_t *>(a.contents()),*y=static_cast<const uint8_t *>(b.contents());uint64_t n=0;for(uint64_t i=0;i<cells;++i)n+=std::memcmp(x+i*bytes,y+i*bytes,bytes)!=0;return n;}
double median(std::vector<double>x){require(!x.empty(),"empty timing vector");std::sort(x.begin(),x.end());return x.size()%2?x[x.size()/2]:(x[x.size()/2]+x[x.size()/2-1])/2;}
struct Variant{
  std::string name;bool paired=false,quantized=false,fixed=false;uint32_t sg=2;
  FlashMoEBlockedScratch scratch;MetalBuffer output,diag;
  std::array<MetalBuffer,2>codes,activationScales;
  std::array<MetalBuffer,4>raw;std::array<MetalBuffer,2>rawInteger;
  CommandGraph graph;std::vector<ComputeDispatch>commands,audits;
  std::array<std::string,3>hashes;std::array<Error,3>vsFloat,vsMatched;
  std::array<uint64_t,4>rawMismatches{};std::array<uint64_t,2>integerMismatches{};
  std::vector<double>gpu,wall;double warmedGPU=0;uint32_t warmCommands=0,shaderRejections=0;
  bool strictPass=true,numericalGuard=true,replay=true;
};
FlashInt8ExpertStoreParams params(uint32_t rows){return{rows,10,rows*10,moEBucketJobCapacity(rows,10,32),32,512,0,0};}
void quant(CommandGraph &g,const Variant &v,uint32_t rows,uint32_t phase){const uint32_t k=phase?640:2560;
  g.add(phase?"prefill_moe_sep21_w8a8_quantize_down_t128":"prefill_moe_sep21_w8a8_quantize_gate_t256",{phase?v.scratch.packedActivated:v.scratch.buckets.packedInputs,v.codes[phase],v.activationScales[phase],v.diag},w8::QuantParams{rows*10+63,k,0,0},{rows*10+63,1,1},{phase?128u:256u,1,1});
}
void build(CommandGraph &g,const Variant &v,const one::OneLayerPayload &layer,const pair::PairPayload &paired,MetalBuffer hidden,MetalBuffer ids,MetalBuffer rw,MetalBuffer shared,MetalBuffer sharedGate,uint32_t rows){
  const auto p=params(rows);addMoEBlockedPack(g,hidden,ids,v.scratch,v.diag,rows,FlashMoEBlockedTile::M32N64);if(v.quantized)quant(g,v,rows,0);
  if(v.paired){const std::string name="prefill_paired_gate_sep21_"+std::string(v.quantized?"w8a8":"bf16")+"_gate_up_m32_n128_"+(v.fixed?"k128_":"")+"sg"+std::to_string(v.sg);
    g.add(name,{v.quantized?v.codes[0]:v.scratch.buckets.packedInputs,paired.paired,layer.scales[0],layer.scales[1],layer.ranks,v.scratch.buckets.offsets,v.scratch.buckets.tileJobs,v.scratch.buckets.jobCount,v.scratch.packedActivated,v.diag},p,{10,p.job_capacity,1},{v.sg*32,1,1});
  }else{const auto name=v.quantized?"prefill_moe_sep21_w8a8_gate_up_m32_n64_sg2":v.fixed?"prefill_paired_control_gate_m32_n64_k128_sg2":"prefill_paired_control_gate_m32_n64_whole_sg2";
    g.add(name,{v.quantized?v.codes[0]:v.scratch.buckets.packedInputs,layer.codes[0],layer.scales[0],layer.codes[1],layer.scales[1],layer.ranks,v.scratch.buckets.offsets,v.scratch.buckets.tileJobs,v.scratch.buckets.jobCount,v.scratch.packedActivated,v.diag},p,{10,p.job_capacity,1},{64,1,1});
  }
  const FlashMoEBlockedDownParams dp{{rows,10,640,2560,512,0,0,0,320,819200,20,51200},rows*10,p.job_capacity,32,0};
  g.add("flash_moe_blocked_poison_excluded_routes",{v.scratch.buckets.canonicalToPacked,v.scratch.scatteredDown,v.diag},dp,{10,rows*10,1},{256,1,1});
  g.add("flash_moe_direct_a_prepare_down",{v.scratch.packedActivated,v.scratch.buckets.offsets,v.scratch.packedActivated,v.diag},FlashMoEDirectAPrepareParams{rows*10,640,63,0},{rows*10+63,1,1},{256,1,1});if(v.quantized)quant(g,v,rows,1);
  g.add(v.quantized?"prefill_moe_sep21_w8a8_down_scatter_m32_n64_sg2":"prefill_moe_sep21_memory_fixed_down_scatter_m32_n64_k128_sg2",{v.quantized?v.codes[1]:v.scratch.packedActivated,layer.codes[2],layer.scales[2],layer.ranks,v.scratch.buckets.offsets,v.scratch.buckets.tileJobs,v.scratch.buckets.jobCount,v.scratch.buckets.routeMap,v.scratch.scatteredDown,v.diag},p,{40,p.job_capacity,1},{64,1,1});
  addCombine(g,v.scratch.scatteredDown,ids,rw,shared,sharedGate,v.output,v.diag,rows,2560,512,10);
}
void overlay(Variant &v){for(const auto &base:v.graph.dispatches()){
  auto fast=base,audit=base;const bool gate=base.pipelineName.find("gate_up")!=std::string::npos||base.pipelineName.starts_with("prefill_paired_control_gate_");const bool down=base.pipelineName.starts_with("prefill_moe_sep21_w8a8_down_");
  if(v.quantized&&(gate||down)){const uint32_t scaleIndex=gate?(v.paired?11:12):11;fast.buffers.push_back({scaleIndex,v.activationScales[gate?0:1]});audit.buffers=fast.buffers;}
  if(gate){audit.pipelineName+="_audit";const uint32_t first=v.paired?(v.quantized?12:11):12;
    if(v.quantized&&!v.paired){audit.buffers.push_back({13,v.raw[2]});audit.buffers.push_back({14,v.raw[3]});audit.buffers.push_back({15,v.rawInteger[0]});audit.buffers.push_back({16,v.rawInteger[1]});}
    else{for(uint32_t i=0;i<4;++i)audit.buffers.push_back({first+i,v.raw[i]});if(v.quantized){audit.buffers.push_back({16,v.rawInteger[0]});audit.buffers.push_back({17,v.rawInteger[1]});}}
  }
  v.commands.push_back(std::move(fast));v.audits.push_back(std::move(audit));
}}
bool sameBuckets(const Variant &a,const Variant &b){for(auto p:{std::pair{a.scratch.buckets.offsets,b.scratch.buckets.offsets},std::pair{a.scratch.buckets.routeMap,b.scratch.buckets.routeMap},std::pair{a.scratch.buckets.canonicalToPacked,b.scratch.buckets.canonicalToPacked},std::pair{a.scratch.buckets.jobCount,b.scratch.buckets.jobCount},std::pair{a.scratch.buckets.tileJobs,b.scratch.buckets.tileJobs}})if(p.first.sizeBytes()!=p.second.sizeBytes()||std::memcmp(p.first.contents(),p.second.contents(),p.first.sizeBytes()))return false;return true;}
void rejectMalformed(MetalBackend &b,Variant &v,uint32_t rows){const auto d=std::find_if(v.commands.begin(),v.commands.end(),[](const auto &x){return x.pipelineName.starts_with("prefill_paired_gate_sep21_");});require(d!=v.commands.end(),"paired gate missing");const auto outputHash=hash(v.scratch.packedActivated);
  for(uint32_t which=0;which<6;++which){auto bad=*d;auto p=params(rows);switch(which){case 0:p.rows=0;break;case 1:p.tile_rows=16;break;case 2:p.stored_experts=513;break;case 3:p.reserved=1;break;case 4:p.scale_group_size=64;break;default:bad.threadsPerThreadgroup.x=32;break;}
    bad.bytes={{d->bytes[0].index,&p,sizeof(p)}};*static_cast<uint32_t *>(v.diag.contents())=kSticky;(void)b.submitCommand({&bad,1});require(*static_cast<uint32_t *>(v.diag.contents())==(kSticky|2),"malformed paired params did not fail closed");v.shaderRejections++;}
  require(hash(v.scratch.packedActivated)==outputHash,"malformed paired params wrote activation");*static_cast<uint32_t *>(v.diag.contents())=kSticky;
}
}
int main(int argc,char **argv){@autoreleasepool{std::filesystem::path report;
  try{
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){pair::cpuSelfTest();w8::precision::cpuSelfTest();require(sizeof(FlashInt8ExpertStoreParams)==32,"paired ABI drift");std::cout<<"{\"valid\":true,\"gpu_work\":false,\"payload_bytes_read\":0}\n";return 0;}
    require(argc>=7&&std::string_view(argv[1])=="--gpu","usage: paired oracle --gpu LIB FULL512 LAYER ROWS NEWreport [--pattern spread/hot --cycles2..6]");report=argv[6];require(!std::filesystem::exists(report),"paired report must be fresh");const uint32_t layerIndex=integer(argv[4],0,47),rows=integer(argv[5],1024,2048);uint32_t cycles=2;std::string pattern="spread";
    for(int i=7;i<argc;i+=2){require(i+1<argc,"missing CLI value");if(std::string_view(argv[i])=="--cycles")cycles=integer(argv[i+1],2,6);else if(std::string_view(argv[i])=="--pattern")pattern=argv[i+1];else throw std::invalid_argument("unknown CLI argument");}require(pattern=="spread"||pattern=="hot","paired pattern invalid");
    const auto meta=loadFlashInt8ExpertStoreMetadata(argv[3],"ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e","edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);require(meta.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","certified Full512 metadata differs");
    ::setenv("SPLASH_FLASH_MOE_Q4X8","1",1);::setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1);MetalBackend b(argv[2]);const uint64_t reserve=16ULL<<30,physical=NSProcessInfo.processInfo.physicalMemory;require(physical>reserve,"paired host reserve unavailable");splash::engine::MemoryGovernor gov(b,physical-reserve,reserve);const uint64_t planned=one::oneLayerPlannedBytes(meta.layers[layerIndex])+pair::pairPlannedBytes()+(16ULL<<30);auto admission=gov.tryReserve(planned);require(bool(admission),"paired one-layer admission denied");const auto before=b.memoryStats().allocatedBytes;auto layer=one::OneLayerPayload::load(b,meta,layerIndex);auto packed=pair::PairPayload::pack(b,layer);
    std::vector<Guard>guards;std::vector<uint16_t>x(uint64_t(rows)*2560);std::vector<int64_t>ids(uint64_t(rows)*10);
    for(uint32_t row=0;row<rows;++row){double square=0;for(uint32_t k=0;k<2560;++k){const double v=int32_t((k*173+row*31)%211)-105;square+=v*v;}const double rms=std::sqrt(square/2560);for(uint32_t k=0;k<2560;++k)x[uint64_t(row)*2560+k]=bf16(float((int32_t((k*173+row*31)%211)-105)/rms));for(uint32_t slot=0;slot<10;++slot)ids[uint64_t(row)*10+slot]=(slot*53+(pattern=="hot"?0:row*73))%512;}
    auto hidden=upload(b,x,guards),idBuffer=upload(b,ids,guards);auto rw=guarded(b,uint64_t(rows)*10*2,guards),shared=guarded(b,uint64_t(rows)*2560*2,guards),sharedGate=guarded(b,uint64_t(rows)*2,guards);std::fill_n(static_cast<uint16_t *>(rw.contents()),rows*10,bf16(.1f));std::memset(shared.contents(),0,shared.sizeBytes());std::memset(sharedGate.contents(),0,sharedGate.sizeBytes());
    std::vector<std::unique_ptr<Variant>>vs;const auto alloc=[&](std::string name,bool pair,bool quantized,bool fixed,uint32_t sg){auto v=std::make_unique<Variant>();v->name=std::move(name);v->paired=pair;v->quantized=quantized;v->fixed=fixed;v->sg=sg;v->scratch=allocateMoEBlockedScratch(b,rows);guardScratch(b,v->scratch,guards);v->diag=guarded(b,4,guards);*static_cast<uint32_t *>(v->diag.contents())=kSticky;v->output=guarded(b,uint64_t(rows)*2560*2,guards);for(auto &r:v->raw){r=guarded(b,uint64_t(rows)*10*640*4,guards);std::fill_n(static_cast<uint32_t *>(r.contents()),r.sizeBytes()/4,0x7fc00000);}if(quantized){for(uint32_t p=0;p<2;++p){v->codes[p]=guarded(b,uint64_t(rows*10+63)*(p?640:2560),guards);v->activationScales[p]=guarded(b,uint64_t(rows*10+63)*4,guards);v->rawInteger[p]=guarded(b,uint64_t(rows)*10*640*4,guards);std::fill_n(static_cast<int32_t *>(v->rawInteger[p].contents()),v->rawInteger[p].sizeBytes()/4,INT32_MIN);}}build(v->graph,*v,layer,packed,hidden,idBuffer,rw,shared,sharedGate,rows);overlay(*v);return v;};
    vs.push_back(alloc("two_N64_whole_SG2",false,false,false,2));vs.push_back(alloc("best_two_N64_K128_SG2",false,false,true,2));
    for(bool fixed:{false,true})for(uint32_t sg:{2u,4u})vs.push_back(alloc("paired_N128_"+std::string(fixed?"K128_":"whole_")+"SG"+std::to_string(sg),true,false,fixed,sg));
    vs.push_back(alloc("best_two_N64_W8_SG2",false,true,false,2));for(uint32_t sg:{2u,4u})vs.push_back(alloc("paired_N128_W8_SG"+std::to_string(sg),true,true,false,sg));
    bool pass=true,buckets=true;for(size_t i=0;i<vs.size();++i){auto &v=*vs[i];(void)b.submitCommand(v.commands);require(*static_cast<uint32_t *>(v.diag.contents())==kSticky,"paired initial shader diagnostic");buckets&=sameBuckets(v,*vs[0]);v.hashes={hash(v.scratch.packedActivated),hash(v.scratch.scatteredDown),hash(v.output)};(void)b.submitCommand(v.audits);require(*static_cast<uint32_t *>(v.diag.contents())==kSticky,"paired audit shader diagnostic");require(v.hashes==std::array<std::string,3>{hash(v.scratch.packedActivated),hash(v.scratch.scatteredDown),hash(v.output)},"paired fast/audit stages differ");
      if(v.quantized&&!v.paired){const auto *g=static_cast<const int32_t *>(v.rawInteger[0].contents()),*u=static_cast<const int32_t *>(v.rawInteger[1].contents());auto *gf=static_cast<float *>(v.raw[0].contents()),*uf=static_cast<float *>(v.raw[1].contents());for(uint64_t j=0;j<uint64_t(rows)*10*640;++j){require(g[j]!=INT32_MIN&&u[j]!=INT32_MIN,"old W8 I32 audit unwritten");gf[j]=float(g[j]);uf[j]=float(u[j]);}}
      // NaN sentinels must never pass a false raw-bit parity comparison.
      for(const auto &raw:v.raw){const auto *f=static_cast<const float *>(raw.contents());for(uint64_t j=0;j<raw.sizeBytes()/4;++j)require(std::isfinite(f[j]),"gate raw/scaled F32 audit unwritten or nonfinite");}
      if(v.quantized)for(const auto &raw:v.rawInteger){const auto *d=static_cast<const int32_t *>(raw.contents());for(uint64_t j=0;j<raw.sizeBytes()/4;++j)require(d[j]!=INT32_MIN,"paired integer gate audit unwritten");}
      const size_t match=v.quantized?6:v.fixed?1:0;const auto &control=*vs[match];v.vsMatched={compare(v.scratch.packedActivated,control.scratch.packedActivated,uint64_t(rows)*10*640),compare(v.scratch.scatteredDown,control.scratch.scatteredDown,uint64_t(rows)*10*2560),compare(v.output,control.output,uint64_t(rows)*2560)};v.vsFloat={compare(v.scratch.packedActivated,vs[1]->scratch.packedActivated,uint64_t(rows)*10*640),compare(v.scratch.scatteredDown,vs[1]->scratch.scatteredDown,uint64_t(rows)*10*2560),compare(v.output,vs[1]->output,uint64_t(rows)*2560)};
      for(uint32_t r=0;r<4;++r){v.rawMismatches[r]=changed(v.raw[r],control.raw[r],uint64_t(rows)*10*640,4);v.strictPass&=v.rawMismatches[r]==0;}if(v.quantized)for(uint32_t r=0;r<2;++r){v.integerMismatches[r]=changed(v.rawInteger[r],control.rawInteger[r],uint64_t(rows)*10*640,4);v.strictPass&=v.integerMismatches[r]==0;}
      for(const auto &e:v.vsMatched)v.strictPass&=!e.changed&&!e.nonfinite;for(const auto &e:v.vsFloat)v.numericalGuard&=e.guard();if(v.paired)rejectMalformed(b,v,rows);pass&=v.strictPass&&(v.quantized?v.numericalGuard:true);
    }
    // All controls are now produced; recompute comparisons against the actual
    // fixedK128 reference so the earlier whole-control report cannot use an
    // uninitialized best-control scratch. W8 thresholds apply to quantized only.
    for(auto &owned:vs){auto &v=*owned;v.vsFloat={compare(v.scratch.packedActivated,vs[1]->scratch.packedActivated,uint64_t(rows)*10*640),compare(v.scratch.scatteredDown,vs[1]->scratch.scatteredDown,uint64_t(rows)*10*2560),compare(v.output,vs[1]->output,uint64_t(rows)*2560)};v.numericalGuard=true;for(const auto &e:v.vsFloat)v.numericalGuard&=e.guard();}
    // At least150ms GPU work per candidate/control, not a cold two-call guess.
    for(auto &v:vs)while(v->warmedGPU<.15){const auto t=b.submitCommand(v->commands);require(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>0,"paired warm GPU timing invalid");v->warmedGPU+=t.gpuSeconds;v->warmCommands++;require(v->warmCommands<1000,"paired warmup runaway");}
    const uint32_t samples=cycles*vs.size();for(uint32_t sample=0;sample<samples;++sample)for(uint32_t slot=0;slot<vs.size();++slot){auto &v=*vs[(sample+slot)%vs.size()];const auto t=b.submitCommand(v.commands);require(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>0,"paired timing invalid");v.gpu.push_back(t.gpuSeconds*1000);v.wall.push_back(t.wallSeconds*1000);}
    for(auto &v:vs){v->replay=v->hashes==std::array<std::string,3>{hash(v->scratch.packedActivated),hash(v->scratch.scatteredDown),hash(v->output)};pass&=v->replay&&*static_cast<uint32_t *>(v->diag.contents())==kSticky;}const bool canaries=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();})&&packed.guardsClean();packed.unchanged(layer);const bool inputExact=!std::memcmp(x.data(),hidden.contents(),hidden.sizeBytes())&&!std::memcmp(ids.data(),idBuffer.contents(),idBuffer.sizeBytes());pass&=canaries&&buckets&&inputExact;const auto used=allocationDelta(before,b.memoryStats().allocatedBytes);require(used<=planned,"paired admission ledger exceeded");admission->commit();
    std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":\"savedI8_paired_single_N128_gate_v1\",\"execution_complete\":true,\"pass\":"<<(pass?"true":"false")<<",\"model_quality_qualified\":false,\"layer\":"<<layerIndex<<",\"rows\":"<<rows<<",\"pattern\":"<<splash::json::quote(pattern)<<",\"synthetic_trueRMS_BF16_fixture\":true,\"source_one_layer_payload_bytes\":2524446720,\"temporary_pair_plane_bytes\":1677721600,\"pair_codes_bijective_and_source_immutable\":true,\"same_bucket_job_ownership\":"<<(buckets?"true":"false")<<",\"canaries_pass\":"<<(canaries?"true":"false")<<",\"input_immutable\":"<<(inputExact?"true":"false")<<",\"planned_bytes\":"<<planned<<",\"actual_allocated_bytes\":"<<used<<",\"float_down_all_fixedK128_SG2\":true,\"W8_down_all_existing_whole_SG2\":true,\"timing_policy\":\"150msGPUwarmEach; balanced cyclic positions; zeroCPUtensoraccess after warmups until all timed calls finish\",\"variants\":[";
    for(size_t i=0;i<vs.size();++i){if(i)out<<',';const auto &v=*vs[i];out<<"{\"name\":"<<splash::json::quote(v.name)<<",\"paired_single_operation\":"<<(v.paired?"true":"false")<<",\"W8_activation_numerical_alternative\":"<<(v.quantized?"true":"false")<<",\"strict_rawF32_scaledF32_BF16_chain_parity\":"<<(v.strictPass?"true":"false")<<",\"W8_existing_preregistered_L2_cos_guard\":"<<(v.numericalGuard?"true":"false")<<",\"median_gpu_ms\":"<<median(v.gpu)<<",\"median_wall_ms\":"<<median(v.wall)<<",\"speedup_vs_ACTUAL_best_float_K128\":"<<median(vs[1]->gpu)/median(v.gpu)<<",\"speedup_vs_actual_best_W8\":"<<median(vs[6]->gpu)/median(v.gpu)<<",\"warmed_GPU_seconds\":"<<v.warmedGPU<<",\"warm_commands\":"<<v.warmCommands<<",\"shader_failclosed_checks\":"<<v.shaderRejections<<",\"quantizers_in_timing\":"<<(v.quantized?2:0)<<",\"rawF32_GU_scaledF32_GU_mismatches\":[";for(uint32_t p=0;p<4;++p){if(p)out<<',';out<<v.rawMismatches[p];}out<<"],\"I32_GU_mismatches\":["<<v.integerMismatches[0]<<','<<v.integerMismatches[1]<<"],\"matched_activation_down_combine_error\":[";for(uint32_t p=0;p<3;++p){if(p)out<<',';v.vsMatched[p].write(out);}out<<"],\"vs_best_float_activation_down_combine_error\":[";for(uint32_t p=0;p<3;++p){if(p)out<<',';v.vsFloat[p].write(out);}out<<"],\"gpu_ms\":[";for(size_t t=0;t<v.gpu.size();++t){if(t)out<<',';out<<v.gpu[t];}out<<"]}";}out<<"],\"pair_witness\":{\"paired_sha256\":"<<splash::json::quote(packed.pairedSHA)<<",\"original_gate_code_sha256\":"<<splash::json::quote(packed.originalCodeSHA[0])<<",\"original_up_code_sha256\":"<<splash::json::quote(packed.originalCodeSHA[1])<<",\"source_base_sha256\":"<<splash::json::quote(packed.sourceImmutableWitness[0])<<",\"source_rank_sha256\":"<<splash::json::quote(packed.sourceImmutableWitness[1])<<",\"code_bytes_certified\":"<<packed.codeBytesCertified<<",\"planned_pair_bytes\":"<<packed.plannedBytes<<",\"actual_pair_bytes\":"<<packed.allocatedBytes<<",\"canaries_pass\":"<<(packed.guardsClean()?"true":"false")<<"}}\n";std::ofstream f(report);require(bool(f),"paired report write failed");f<<out.str();b.stop();std::cout<<"{\"pass\":"<<(pass?"true":"false")<<",\"report\":"<<splash::json::quote(report.string())<<"}\n";return pass?0:2;
  }catch(const std::exception&e){if(!report.empty()&&!std::filesystem::exists(report)){std::ofstream f(report);f<<"{\"execution_complete\":false,\"error\":"<<splash::json::quote(e.what())<<"}\n";}std::cerr<<e.what()<<'\n';return 1;}
}}
