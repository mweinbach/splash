// Root owns --gpu. CPU self-test/metadata planning never create a backend.
#include "loader.hpp"
#include "precision.hpp"
#include "abi.h"
#include "flash/FlashMoE.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashMoEDirectA.h"
#include "dev/benchmarks/prefill_moe_sep21/w8a8/cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include <iostream>
#include <optional>
#include <sstream>

namespace {
using namespace splash::metal;
using namespace splash::flash;
namespace w4=splash::bench::w4a8;
namespace one=splash::flash::qmv_one_layer;
namespace w8=splash::flash::prefill_moe_w8a8;
constexpr uint32_t kSticky=0x80000000u;
void need(bool ok,const std::string &why){if(!ok)throw std::invalid_argument(why);}
struct Guard {MetalBuffer base;uint64_t bytes;bool clean()const {
  const auto *p=static_cast<const uint8_t *>(base.contents());
  return std::all_of(p,p+128,[](uint8_t x){return x==0x5a;})&&std::all_of(p+128+bytes,p+256+bytes,[](uint8_t x){return x==0x5a;});
}};
MetalBuffer guarded(MetalBackend &b,uint64_t bytes,std::vector<Guard> &guards){
  auto base=b.allocateBuffer(bytes+256,BufferStorage::Shared,"bounded W4A8 guarded buffer");std::memset(base.contents(),0x5a,base.sizeBytes());
  auto view=b.view(base,128,bytes);std::memset(view.contents(),0xa5,bytes);guards.push_back({base,bytes});return view;
}
template<class T>MetalBuffer upload(MetalBackend &b,const std::vector<T> &v,std::vector<Guard> &g){auto x=guarded(b,v.size()*sizeof(T),g);std::memcpy(x.contents(),v.data(),x.sizeBytes());return x;}
void guardScratch(MetalBackend &b,FlashMoEBlockedScratch &s,std::vector<Guard> &g){
  for(auto *x:{&s.buckets.counts,&s.buckets.offsets,&s.buckets.routeMap,&s.buckets.canonicalToPacked,&s.buckets.packedInputs,&s.buckets.jobOffsets,&s.buckets.jobCount,&s.buckets.tileJobs,&s.packedActivated,&s.scatteredDown})*x=guarded(b,x->sizeBytes(),g);
}
uint32_t number(const char *raw,uint32_t low,uint32_t high){std::string s(raw);need(!s.empty()&&std::all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}),"CLI integer must be unsigned decimal");size_t n=0;const auto v=std::stoul(s,&n);need(n==s.size()&&v>=low&&v<=high,"CLI integer outside bounds");return uint32_t(v);}
std::vector<uint16_t> hidden(uint32_t rows){std::vector<uint16_t> x(uint64_t(rows)*2560);for(uint32_t row=0;row<rows;++row){
  double q=0;for(uint32_t k=0;k<2560;++k){const double v=int32_t((k*173+row*31)%211)-105;q+=v*v;}const double rms=std::sqrt(q/2560);
  for(uint32_t k=0;k<2560;++k)x[uint64_t(row)*2560+k]=w4::bf16(float((int32_t((k*173+row*31)%211)-105)/rms));}return x;
}
std::vector<int64_t> routes(uint32_t rows,std::string_view pattern){need(pattern=="spread"||pattern=="hot","pattern must be spread/hot");std::vector<int64_t> result(uint64_t(rows)*10);
  for(uint32_t row=0;row<rows;++row)for(uint32_t slot=0;slot<10;++slot)result[uint64_t(row)*10+slot]=(slot*53+(pattern=="hot"?0:row*73))%512;return result;
}
struct Error {uint64_t cells=0,changed=0,nonfinite=0;double ee=0,rr=0,aa=0,dot=0,maxAbs=0;
  void add(uint16_t a,uint16_t b){cells++;changed+=a!=b;const double x=w4::number(a),y=w4::number(b);if(!std::isfinite(x)||!std::isfinite(y)){nonfinite++;return;}const double e=x-y;ee+=e*e;rr+=y*y;aa+=x*x;dot+=x*y;maxAbs=std::max(maxAbs,std::abs(e));}
  double l2()const{return std::sqrt(ee/std::max(rr,1e-300));}double cosine()const{return aa==0&&rr==0?1:dot/std::sqrt(std::max(aa*rr,1e-300));}
  bool pass()const{return !nonfinite&&l2()<=w4::kMaximumProducerRelativeL2&&cosine()>=w4::kMinimumProducerCosine;}
  void write(std::ostream &o)const{o<<"{\"cells\":"<<cells<<",\"changed_bf16\":"<<changed<<",\"nonfinite\":"<<nonfinite<<",\"relative_l2\":"<<l2()<<",\"cosine\":"<<cosine()<<",\"max_abs\":"<<maxAbs<<'}';}
};
Error compare(MetalBuffer a,MetalBuffer b,uint64_t cells){need(a.sizeBytes()>=cells*2&&b.sizeBytes()>=cells*2,"error extent too small");Error e;const auto *x=static_cast<const uint16_t *>(a.contents()),*y=static_cast<const uint16_t *>(b.contents());for(uint64_t i=0;i<cells;++i)e.add(x[i],y[i]);return e;}
double median(std::vector<double> x){need(!x.empty(),"empty timing vector");std::sort(x.begin(),x.end());return x.size()%2?x[x.size()/2]:(x[x.size()/2]+x[x.size()/2-1])/2;}
struct Variant {
  std::string name;uint32_t n=0,sg=0;bool i8=false,w8=false;
  FlashMoEBlockedScratch scratch;MetalBuffer diag,output;
  std::array<MetalBuffer,2> codes,scales,sums;
  std::array<MetalBuffer,3> raw;
  CommandGraph graph,audit;std::vector<ComputeDispatch> auditCommands;
  std::unique_ptr<w8::Workspace> w8workspace;std::unique_ptr<w8::QuantizedCommands>w8commands;
  std::vector<double> gpu,wall;std::array<Error,3>errors;std::array<std::string,3>hashes;
  struct CertificateLocation{uint32_t plane,job,expert,packedRow,canonicalRoute,column;};std::vector<CertificateLocation> locations;std::vector<w4::RowCertificate> certificates;std::array<w4::QuantizationReport,2> quantReports;std::array<w4::OutputMetrics,3> sampledLinearMetrics;
  uint64_t certifiedGroups=0;bool certificatePass=true,replay=true;
  std::span<const ComputeDispatch> commands()const{return w8?std::span<const ComputeDispatch>(w8commands->commands):graph.dispatches();}
};
FlashMoEBlockedGateParams gateParams(uint32_t rows){return{{rows,10,2560,640,512,0,0,0,1280,819200,80,51200,1280,819200,80,51200},rows*10,moEBucketJobCapacity(rows,10,32),32,0};}
FlashMoEBlockedDownParams downParams(uint32_t rows){return{{rows,10,640,2560,512,0,0,0,320,819200,20,51200},rows*10,moEBucketJobCapacity(rows,10,32),32,0};}
void quant(CommandGraph &g,const Variant &v,uint32_t rows,uint32_t phase){const uint32_t k=phase?640:2560,physical=rows*10+63;
  const PrefillW4A8QuantParams p{{rows,10,k,512,rows*10,0,0,0},physical,k,k,k/64,0,0,0,0};
  g.add(phase?"prefill_w4a8_sep21_quant_down":"prefill_w4a8_sep21_quant_packed",{phase?v.scratch.packedActivated:v.scratch.buckets.packedInputs,v.scratch.buckets.offsets,v.codes[phase],v.scales[phase],v.sums[phase],v.diag},p,{physical,1,1},{256,1,1});
}
void w4chain(CommandGraph &g,const w4::LayerPayload &layer,const Variant &v,MetalBuffer x,MetalBuffer ids,MetalBuffer route,MetalBuffer shared,MetalBuffer sharedGate,uint32_t rows){
  addMoEBlockedPack(g,x,ids,v.scratch,v.diag,rows,FlashMoEBlockedTile::M32N64);quant(g,v,rows,0);
  const std::string suffix="m32_n"+std::to_string(v.n)+"_sg"+std::to_string(v.sg);
  const PrefillW4A8GateParams gp{gateParams(rows),rows*10+63,2560,40,0,1280,819200,1280,819200};
  std::vector<MetalBuffer> gb{v.codes[0],v.scales[0],v.sums[0],layer.centered[0],layer.tensors[1].buffer,layer.tensors[2].buffer,layer.centered[1],layer.tensors[4].buffer,layer.tensors[5].buffer,v.scratch.buckets.offsets,v.scratch.buckets.tileJobs,v.scratch.buckets.jobCount,v.scratch.packedActivated,v.diag};
  g.add("prefill_w4a8_sep21_gate_"+suffix,gb,gp,{640/v.n,gp.source.job_capacity,1},{v.sg*32,1,1});
  const auto dp=downParams(rows);g.add("flash_moe_blocked_poison_excluded_routes",{v.scratch.buckets.canonicalToPacked,v.scratch.scatteredDown,v.diag},dp,{10,rows*10,1},{256,1,1});
  g.add("flash_moe_direct_a_prepare_down",{v.scratch.packedActivated,v.scratch.buckets.offsets,v.scratch.packedActivated,v.diag},FlashMoEDirectAPrepareParams{rows*10,640,63,0},{rows*10+63,1,1},{256,1,1});quant(g,v,rows,1);
  const PrefillW4A8DownParams wp{dp,rows*10+63,640,10,0,384,983040};
  std::vector<MetalBuffer> db{v.codes[1],v.scales[1],v.sums[1],layer.centered[2],layer.tensors[7].buffer,layer.tensors[8].buffer,v.scratch.buckets.offsets,v.scratch.buckets.tileJobs,v.scratch.buckets.jobCount,v.scratch.buckets.routeMap,v.scratch.scatteredDown,v.diag};
  g.add("prefill_w4a8_sep21_down_"+suffix,db,wp,{2560/v.n,dp.job_capacity,1},{v.sg*32,1,1});
  addCombine(g,v.scratch.scatteredDown,ids,route,shared,sharedGate,v.output,v.diag,rows,2560,512,10);
}
void i8chain(CommandGraph &g,const one::OneLayerPayload &layer,const Variant &v,MetalBuffer x,MetalBuffer ids,MetalBuffer route,MetalBuffer shared,MetalBuffer sharedGate,uint32_t rows){
  addMoEBlockedPack(g,x,ids,v.scratch,v.diag,rows,FlashMoEBlockedTile::M32N64);const uint32_t count=moEBucketJobCapacity(rows,10,32);const FlashInt8ExpertStoreParams p{rows,10,rows*10,count,32,512,0,0};
  g.add("flash_int8_expert_store_gate_up_m32_n64",{v.scratch.buckets.packedInputs,layer.codes[0],layer.scales[0],layer.codes[1],layer.scales[1],layer.ranks,v.scratch.buckets.offsets,v.scratch.buckets.tileJobs,v.scratch.buckets.jobCount,v.scratch.packedActivated,v.diag},p,{10,count,1},{128,1,1});
  const auto dp=downParams(rows);g.add("flash_moe_blocked_poison_excluded_routes",{v.scratch.buckets.canonicalToPacked,v.scratch.scatteredDown,v.diag},dp,{10,rows*10,1},{256,1,1});
  g.add("flash_moe_direct_a_prepare_down",{v.scratch.packedActivated,v.scratch.buckets.offsets,v.scratch.packedActivated,v.diag},FlashMoEDirectAPrepareParams{rows*10,640,63,0},{rows*10+63,1,1},{256,1,1});
  g.add("flash_int8_expert_store_down_scatter_m32_n64",{v.scratch.packedActivated,layer.codes[2],layer.scales[2],layer.ranks,v.scratch.buckets.offsets,v.scratch.buckets.tileJobs,v.scratch.buckets.jobCount,v.scratch.buckets.routeMap,v.scratch.scatteredDown,v.diag},p,{40,count,1},{128,1,1});
  addCombine(g,v.scratch.scatteredDown,ids,route,shared,sharedGate,v.output,v.diag,rows,2560,512,10);
}
bool bucketsEqual(const Variant &a,const Variant &b){for(auto pair:{std::pair{a.scratch.buckets.counts,b.scratch.buckets.counts},std::pair{a.scratch.buckets.offsets,b.scratch.buckets.offsets},std::pair{a.scratch.buckets.routeMap,b.scratch.buckets.routeMap},std::pair{a.scratch.buckets.canonicalToPacked,b.scratch.buckets.canonicalToPacked},std::pair{a.scratch.buckets.jobOffsets,b.scratch.buckets.jobOffsets},std::pair{a.scratch.buckets.jobCount,b.scratch.buckets.jobCount},std::pair{a.scratch.buckets.tileJobs,b.scratch.buckets.tileJobs}})if(pair.first.sizeBytes()!=pair.second.sizeBytes()||std::memcmp(pair.first.contents(),pair.second.contents(),pair.first.sizeBytes()))return false;return true;}
void certify(MetalBackend &backend,const w4::LayerPayload &layer,Variant &v,uint32_t rows,std::vector<Guard> &guards){
  (void)backend.submitCommand(v.auditCommands);const uint32_t routes=rows*10,physical=routes+63;
  for(uint32_t phase=0;phase<2;++phase)v.quantReports[phase]=w4::certifyQuantizedRows(static_cast<const uint16_t *>((phase?v.scratch.packedActivated:v.scratch.buckets.packedInputs).contents()),static_cast<const int8_t *>(v.codes[phase].contents()),static_cast<const float *>(v.scales[phase].contents()),routes,phase?640:2560);
  v.certificatePass=v.quantReports[0].pass()&&v.quantReports[1].pass();
  const auto *jobs=static_cast<const FlashMoEBucketJob *>(v.scratch.buckets.tileJobs.contents()),*unused=jobs;(void)unused;
  const uint32_t activeJobs=*static_cast<const uint32_t *>(v.scratch.buckets.jobCount.contents());const auto *offset=static_cast<const uint32_t *>(v.scratch.buckets.offsets.contents()),*map=static_cast<const uint32_t *>(v.scratch.buckets.routeMap.contents());
  need(activeJobs&&activeJobs<=moEBucketJobCapacity(rows,10,32)&&offset[512]==routes,"invalid finite sample job inventory");
  for(uint32_t plane=0;plane<3;++plane){const uint32_t k=plane==2?640:2560,n=plane==2?2560:640,g=k/64,phase=plane==2;std::vector<float> actualLinears;std::vector<double> originalRefs,absoluteEnvelopes;
    std::vector<PrefillW4A8ProbeSample> samples;for(uint32_t j=0;j<16;++j){const uint32_t index=uint64_t(j)*(activeJobs-1)/15;const auto job=jobs[index];const uint32_t valid=std::min(32u,offset[job.expert+1]-job.row_begin);for(uint32_t r:{0u,valid-1})for(uint32_t c:{0u,1u,n/4,n/2,n-2,n-1})samples.push_back({index,r,c%v.n,c/v.n});}
    auto sampleBuffer=upload(backend,samples,guards);auto dot=guarded(backend,uint64_t(samples.size())*g*4,guards),sum=guarded(backend,dot.sizeBytes(),guards),bias=guarded(backend,dot.sizeBytes(),guards);
    std::fill_n(static_cast<int32_t *>(dot.contents()),samples.size()*g,INT32_MIN);std::fill_n(static_cast<int32_t *>(sum.contents()),samples.size()*g,INT32_MIN);std::fill_n(static_cast<uint32_t *>(bias.contents()),samples.size()*g,0x7fc00000u);
    const PrefillW4A8ProbeParams p{{rows,10,k,512,routes,32,moEBucketJobCapacity(rows,10,32),0},physical,k,g,uint32_t(samples.size()),plane,v.n,0,0,plane==2?384u:1280u,plane==2?983040u:819200u,k/32,uint64_t(n)*k/32};CommandGraph probe;
    probe.add("prefill_w4a8_sep21_probe_m32_n"+std::to_string(v.n)+"_sg"+std::to_string(v.sg),{v.codes[phase],v.sums[phase],layer.centered[plane],layer.tensors[plane*3+1].buffer,layer.tensors[plane*3+2].buffer,v.scratch.buckets.offsets,v.scratch.buckets.tileJobs,v.scratch.buckets.jobCount,sampleBuffer,dot,sum,bias,v.diag},p,{samples.size(),g,1},{v.sg*32,1,1});(void)backend.submitCommand(probe.dispatches());
    const auto *sourceA=static_cast<const uint16_t *>((phase?v.scratch.packedActivated:v.scratch.buckets.packedInputs).contents()),*scale=static_cast<const uint16_t *>(layer.tensors[plane*3+1].buffer.contents()),*sourceBias=static_cast<const uint16_t *>(layer.tensors[plane*3+2].buffer.contents());
    const auto *original=static_cast<const uint8_t *>(layer.tensors[plane*3].buffer.contents()),*center=static_cast<const uint8_t *>(layer.centered[plane].contents());const auto *qa=static_cast<const int8_t *>(v.codes[phase].contents());const auto *as=static_cast<const float *>(v.scales[phase].contents()),*raw=static_cast<const float *>(v.raw[plane].contents());
    const auto *dots=static_cast<const int32_t *>(dot.contents()),*sums=static_cast<const int32_t *>(sum.contents());const auto *biases=static_cast<const float *>(bias.contents());
    for(uint32_t s=0;s<samples.size();++s){const auto sample=samples[s];const auto job=jobs[sample.job_index];const uint32_t row=job.row_begin+sample.row_within_job,c=sample.column_tile*v.n+sample.column_within_tile;const uint64_t weightRow=uint64_t(job.expert)*n+c;
      for(uint32_t group=0;group<g;++group)need(dots[s*g+group]!=INT32_MIN&&sums[s*g+group]!=INT32_MIN&&std::isfinite(biases[s*g+group]),"compact native group probe left unwritten sample");
      const w4::SourceRowView source{{sourceA+uint64_t(row)*k,k},{original+weightRow*(k/2),k/2},{scale+weightRow*g,g},{sourceBias+weightRow*g,g},{center+weightRow*(plane==2?384:1280),k/2}};
      const w4::ObservedRowView observed{as[row],{qa+uint64_t(row)*k,k},{dots+uint64_t(s)*g,g},{sums+uint64_t(s)*g,g},{biases+uint64_t(s)*g,g},raw[uint64_t(plane==2?map[row]:row)*n+c]};
      v.locations.push_back({plane,sample.job_index,job.expert,row,map[row],c});v.certificates.push_back(w4::certifyRow(source,observed));v.certificatePass&=v.certificates.back().pass();v.certifiedGroups+=g;actualLinears.push_back(observed.linearOutputF32);originalRefs.push_back(v.certificates.back().originalAffineF64);absoluteEnvelopes.push_back(v.certificates.back().totalEnvelope);
    }
    v.sampledLinearMetrics[plane]=w4::compareOutputs(actualLinears,originalRefs,absoluteEnvelopes);v.certificatePass&=v.sampledLinearMetrics[plane].pass();
  }
}
}
int main(int argc,char **argv){@autoreleasepool{std::filesystem::path report;
  try{
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){w4::cpuSelfTest();need(sizeof(PrefillW4A8QuantParams)==64&&sizeof(PrefillW4A8GateParams)==160&&sizeof(PrefillW4A8DownParams)==112&&sizeof(PrefillW4A8ProbeParams)==96,"W4A8 ABI mismatch");std::cout<<"{\"valid\":true,\"gpu_work\":false,\"payload_bytes_read\":0,\"CPU_precision_and_ABI_checks\":true}\n";return 0;}
    if(argc==4&&std::string_view(argv[1])=="--plan"){auto m=w4::inspect(argv[2],number(argv[3],0,47));std::cout<<"{\"gpu_work\":false,\"model_payload_bytes_read\":0,\"original_layer_bytes\":"<<m.originalBytes<<",\"temporary_centered_layer_bytes\":"<<m.centeredBytes<<",\"manifest_sha256\":"<<splash::json::quote(m.manifestSHA)<<"}\n";return 0;}
    need(argc>=6&&std::string_view(argv[1])=="--gpu","usage: oracle --gpu METALLIB ORIGINAL_NATIVE_PACKAGE NEW_REPORT ROWS [--layer0..47 --pattern spread/hot --cycles2..8 --i8-storeFull512] | --cpu-self-test | --plan PACKAGE LAYER");report=argv[4];need(!std::filesystem::exists(report),"report must be fresh");const uint32_t rows=number(argv[5],1024,2048);uint32_t layerIndex=0,cycles=2;std::string pattern="spread",i8Path;
    for(int i=6;i<argc;i+=2){need(i+1<argc,"missing CLI value");const std::string arg=argv[i];if(arg=="--layer")layerIndex=number(argv[i+1],0,47);else if(arg=="--cycles")cycles=number(argv[i+1],2,8);else if(arg=="--pattern")pattern=argv[i+1];else if(arg=="--i8-store")i8Path=argv[i+1];else throw std::invalid_argument("unknown CLI argument");}
    const auto meta=w4::inspect(argv[3],layerIndex);std::optional<FlashInt8ExpertStoreMetadata> i8meta;if(!i8Path.empty()){i8meta=loadFlashInt8ExpertStoreMetadata(i8Path,w4::kSourceIdentity,"edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);need(i8meta->identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","certified I8 layer metadata differs");}
    ::setenv("SPLASH_FLASH_MOE_Q4X8","1",1);::setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1);::setenv("SPLASH_FLASH_MOE_M64","1",1);
    MetalBackend backend(argv[2]);const uint64_t reserve=16ULL<<30,physical=NSProcessInfo.processInfo.physicalMemory;need(physical>reserve,"cannot preserve host reserve");splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
    const uint64_t planned=meta.originalBytes+meta.centeredBytes+(i8meta?one::oneLayerPlannedBytes(i8meta->layers[layerIndex]):0)+(12ULL<<30);auto reservation=governor.tryReserve(planned);need(bool(reservation),"bounded one-layer W4A8 reservation denied");const uint64_t before=backend.memoryStats().allocatedBytes;auto layer=w4::LayerPayload::load(backend,meta);std::optional<one::OneLayerPayload> i8layer;one::OneLayerPayload::ImmutableHashes i8immutable{};if(i8meta){i8layer=one::OneLayerPayload::load(backend,*i8meta,layerIndex);i8immutable=i8layer->immutableHashes();}
    std::vector<Guard> guards;const auto source=hidden(rows);const auto idsHost=routes(rows,pattern);auto x=upload(backend,source,guards),ids=upload(backend,idsHost,guards);auto rw=guarded(backend,uint64_t(rows)*10*2,guards),shared=guarded(backend,uint64_t(rows)*2560*2,guards),sharedGate=guarded(backend,uint64_t(rows)*2,guards);std::fill_n(static_cast<uint16_t *>(rw.contents()),rows*10,w4::bf16(.1f));std::memset(shared.contents(),0,shared.sizeBytes());std::memset(sharedGate.contents(),0,sharedGate.sizeBytes());
    std::vector<std::unique_ptr<Variant>> variants;const auto allocate=[&](std::string name,uint32_t n,uint32_t sg){auto v=std::make_unique<Variant>();v->name=std::move(name);v->n=n;v->sg=sg;v->scratch=allocateMoEBlockedScratch(backend,rows);guardScratch(backend,v->scratch,guards);v->diag=guarded(backend,4,guards);*static_cast<uint32_t *>(v->diag.contents())=kSticky;v->output=guarded(backend,uint64_t(rows)*2560*2,guards);return v;};
    auto native=allocate("original_Q4_DirectA",64,4);addMoEBlockedPack(native->graph,x,ids,native->scratch,native->diag,rows,FlashMoEBlockedTile::M32N64);addMoEBlockedGateUp(native->graph,layer.projection(0),layer.projection(1),native->scratch,native->diag,rows,FlashMoEBlockedTile::M32N64);addMoEBlockedDownScatter(native->graph,layer.projection(2),native->scratch,native->diag,rows,FlashMoEBlockedTile::M32N64);addCombine(native->graph,native->scratch.scatteredDown,ids,rw,shared,sharedGate,native->output,native->diag,rows,2560,512,10);variants.push_back(std::move(native));
    for(uint32_t n:{32u,64u})for(uint32_t sg:{1u,2u}){auto v=allocate("W4A8_m32_n"+std::to_string(n)+"_sg"+std::to_string(sg),n,sg);for(uint32_t p=0;p<2;++p){const uint32_t k=p?640:2560;v->codes[p]=guarded(backend,uint64_t(rows*10+63)*k,guards);v->scales[p]=guarded(backend,uint64_t(rows*10+63)*4,guards);v->sums[p]=guarded(backend,uint64_t(rows*10+63)*(k/64)*4,guards);}for(uint32_t p=0;p<3;++p){v->raw[p]=guarded(backend,uint64_t(rows)*10*(p==2?2560:640)*4,guards);std::fill_n(static_cast<uint32_t *>(v->raw[p].contents()),v->raw[p].sizeBytes()/4,0x7fc00000u);}w4chain(v->graph,layer,*v,x,ids,rw,shared,sharedGate,rows);w4chain(v->audit,layer,*v,x,ids,rw,shared,sharedGate,rows);for(const auto &base:v->audit.dispatches()){auto d=base;if(d.pipelineName.starts_with("prefill_w4a8_sep21_gate_")){need(d.bytes.size()==1&&d.bytes[0].index==14,"audit gate parameter index drift");d.pipelineName+="_audit";d.buffers.push_back({15,v->raw[0]});d.buffers.push_back({16,v->raw[1]});}else if(d.pipelineName.starts_with("prefill_w4a8_sep21_down_")){need(d.bytes.size()==1&&d.bytes[0].index==12,"audit Down parameter index drift");d.pipelineName+="_audit";d.buffers.push_back({13,v->raw[2]});}v->auditCommands.push_back(std::move(d));}variants.push_back(std::move(v));}
    if(i8layer){auto v=allocate("Full512_I8_BF16A",64,4);v->i8=true;i8chain(v->graph,*i8layer,*v,x,ids,rw,shared,sharedGate,rows);variants.push_back(std::move(v));auto z=allocate("Full512_W8A8_SG2",64,2);z->i8=true;z->w8=true;i8chain(z->graph,*i8layer,*z,x,ids,rw,shared,sharedGate,rows);z->w8workspace=std::make_unique<w8::Workspace>(w8::Workspace::allocate(backend,rows));z->w8commands=std::make_unique<w8::QuantizedCommands>(z->graph.dispatches(),rows,w8::Variant{2},*i8layer,*z->w8workspace);variants.push_back(std::move(z));}
    bool pass=true,bucketExact=true,variantExact=true;for(auto &v:variants){(void)backend.submitCommand(v->commands());need(*static_cast<uint32_t *>(v->diag.contents())==kSticky,"initial W4A8 shader diagnostic");bucketExact&=bucketsEqual(*variants[0],*v);v->errors={compare(v->scratch.packedActivated,variants[0]->scratch.packedActivated,uint64_t(rows)*10*640),compare(v->scratch.scatteredDown,variants[0]->scratch.scatteredDown,uint64_t(rows)*10*2560),compare(v->output,variants[0]->output,uint64_t(rows)*2560)};v->hashes={w4::hash(v->scratch.packedActivated),w4::hash(v->scratch.scatteredDown),w4::hash(v->output)};if(v->n&&v->sg<=2&&!v->i8){for(const auto &e:v->errors)pass&=e.pass();certify(backend,layer,*v,rows,guards);pass&=v->certificatePass;need(v->hashes==std::array<std::string,3>{w4::hash(v->scratch.packedActivated),w4::hash(v->scratch.scatteredDown),w4::hash(v->output)},"fast/audit W4A8 BF16 activation/down/combine differs");}}
    for(size_t i=2;i<5;++i){variantExact&=compare(variants[i]->scratch.packedActivated,variants[1]->scratch.packedActivated,uint64_t(rows)*10*640).changed==0;variantExact&=compare(variants[i]->scratch.scatteredDown,variants[1]->scratch.scatteredDown,uint64_t(rows)*10*2560).changed==0;variantExact&=compare(variants[i]->output,variants[1]->output,uint64_t(rows)*2560).changed==0;}
    // No mutable tensor CPU access throughout the warm and timed command train.
    for(uint32_t round=0;round<2;++round)for(uint32_t i=0;i<variants.size();++i)(void)backend.submitCommand(variants[(i+round)%variants.size()]->commands());
    const uint32_t samples=cycles*variants.size();for(uint32_t sample=0;sample<samples;++sample)for(uint32_t slot=0;slot<variants.size();++slot){auto &v=*variants[(sample+slot)%variants.size()];const auto t=backend.submitCommand(v.commands());need(t.gpuSeconds>0&&std::isfinite(t.gpuSeconds),"invalid GPU timing");v.gpu.push_back(t.gpuSeconds*1000);v.wall.push_back(t.wallSeconds*1000);}
    for(auto &v:variants){v->replay=v->hashes==std::array<std::string,3>{w4::hash(v->scratch.packedActivated),w4::hash(v->scratch.scatteredDown),w4::hash(v->output)};pass&=v->replay&&*static_cast<uint32_t *>(v->diag.contents())==kSticky;}
    const bool canaries=std::all_of(guards.begin(),guards.end(),[](const Guard &g){return g.clean();});layer.unchanged();if(i8layer)i8layer->checkImmutableHashes(i8immutable);const bool inputsExact=!std::memcmp(source.data(),x.contents(),x.sizeBytes())&&!std::memcmp(idsHost.data(),ids.contents(),ids.sizeBytes());pass&=bucketExact&&variantExact&&canaries&&inputsExact;const uint64_t used=allocationDelta(before,backend.memoryStats().allocatedBytes);need(used<=planned,"bounded W4A8 oracle exceeded reservation");reservation->commit();
    std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":\"splash-one-layer-W4A8-original-Q4-affine-v1\",\"execution_complete\":true,\"pass\":"<<(pass?"true":"false")<<",\"model_quality_qualified\":false,\"numerical_activation_and_association_alternative\":true,\"original_decode_Q4_not_modified\":true,\"layer\":"<<layerIndex<<",\"rows\":"<<rows<<",\"pattern\":"<<splash::json::quote(pattern)<<",\"input_policy\":\"synthetic trueRMS BF16, not captured model activation\",\"preregistered_maximum_producer_L2\":0.03,\"preregistered_minimum_cosine\":0.9995,\"original_source_manifest_sha256\":"<<splash::json::quote(meta.manifestSHA)<<",\"original_selected_plane_bytes\":"<<meta.originalBytes<<",\"temporary_centered_code_bytes\":"<<meta.centeredBytes<<",\"full_source_shard_checksum_scans\":false,\"selected_tensor_range_hashes_observed_and_immutable\":true,\"centered_code_bytes_reversible_certified\":"<<layer.exactCodeBytesCertified<<",\"bounded_planned_bytes\":"<<planned<<",\"actual_allocated_bytes\":"<<used<<",\"bucket_ownership_exact\":"<<(bucketExact?"true":"false")<<",\"W4A8_variant_BF16_stage_consistency\":"<<(variantExact?"true":"false")<<",\"canaries_pass\":"<<(canaries?"true":"false")<<",\"original_inputs_immutable\":"<<(inputsExact?"true":"false")<<",\"timing_policy\":\"complete GPU bucket/quantize gate/affine/SwiGLU/quantize down/scatter/combine; cyclic balanced positions; two warm rounds; noCPUtensoraccess during command train\",\"variants\":[";
    for(size_t i=0;i<variants.size();++i){if(i)out<<',';const auto &v=*variants[i];out<<"{\"name\":"<<splash::json::quote(v.name)<<",\"original_Q4_weight_values_preserved_mathematically\":"<<(!v.i8?"true":"false")<<",\"Q4_coefficient_BF16_boundary_preserved\":"<<(i==0?"true":"false")<<",\"quantizer_commands_included\":"<<(i==0||(v.i8&&!v.w8)?0:2)<<",\"dispatches\":"<<v.commands().size()<<",\"median_gpu_ms\":"<<median(v.gpu)<<",\"median_wall_ms\":"<<median(v.wall)<<",\"speedup_vs_originalQ4\":"<<median(variants[0]->gpu)/median(v.gpu)<<",\"replay_exact\":"<<(v.replay?"true":"false")<<",\"errors_vs_originalQ4\":[";for(uint32_t p=0;p<3;++p){if(p)out<<',';v.errors[p].write(out);}out<<"],\"certificate_applies\":"<<((i>0&&i<5)?"true":"false")<<",\"certificate_pass\":"<<((i>0&&i<5)?(v.certificatePass?"true":"false"):"null")<<",\"native_replay_probe_output_cells_checked\":"<<v.certifiedGroups<<",\"quantization_reports\":[";if(i&&i<5){v.quantReports[0].write(out);out<<',';v.quantReports[1].write(out);}out<<"],\"sampled_raw_linear_metrics_per_plane\":[";if(i>0&&i<5){for(uint32_t p=0;p<3;++p){if(p)out<<',';v.sampledLinearMetrics[p].write(out);}}out<<"],\"native_replay_probe_scope\":\"same descriptor/SG/equivalent logical inputs; masked probe A view differs from static full-M core; I32/bias are replay observations, full-core final F32 independently bounded\",\"probe_cell_counts_include_duplicate_endpoint_samples\":true,\"row_certificates\":[";for(size_t c=0;c<v.certificates.size();++c){if(c)out<<',';const auto &l=v.locations[c];out<<"{\"plane\":"<<l.plane<<",\"job\":"<<l.job<<",\"expert\":"<<l.expert<<",\"packed_row\":"<<l.packedRow<<",\"canonical_route\":"<<l.canonicalRoute<<",\"column\":"<<l.column<<",\"source_activation_scope\":"<<splash::json::quote(l.plane==2?"candidate_BF16_SwiGLU_upstream_conditional_projection_bound":"original_shared_BF16_normalized_packed_input")<<",\"certificate\":";v.certificates[c].write(out);out<<'}';}out<<"],\"gpu_ms\":[";for(size_t t=0;t<v.gpu.size();++t){if(t)out<<',';out<<v.gpu[t];}out<<"]}";}out<<"],\"original_selected_tensors\":[";for(uint32_t f=0;f<9;++f){if(f)out<<',';const auto &r=meta.tensors[f];out<<"{\"name\":"<<splash::json::quote(r.name)<<",\"shard\":"<<splash::json::quote(r.path)<<",\"offset\":"<<r.offset<<",\"length\":"<<r.bytes<<",\"source_offset\":"<<r.sourceOffset<<",\"observed_sha256\":"<<splash::json::quote(layer.sourceHashes[f])<<'}';}out<<"]}\n";std::ofstream file(report);need(bool(file),"cannot create report");file<<out.str();backend.stop();std::cout<<"{\"pass\":"<<(pass?"true":"false")<<",\"report\":"<<splash::json::quote(report.string())<<"}\n";return pass?0:2;
  }catch(const std::exception &e){if(!report.empty()&&!std::filesystem::exists(report)){std::ofstream f(report);f<<"{\"execution_complete\":false,\"error\":"<<splash::json::quote(e.what())<<"}\n";}std::cerr<<e.what()<<'\n';return 1;}
}}
