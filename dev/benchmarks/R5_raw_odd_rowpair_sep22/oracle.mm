// Root-only bounded native-affine R5 component. CPU/help return before any
// metadata read, device/backend creation or payload access.
#include "policy.hpp"
#include "storage.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "dev/benchmarks/FlashFloatBoundaryAudit.hpp"
#include "Provenance.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace {
namespace q=R5_raw_odd_rowpair_sep22;
namespace fs=std::filesystem;
using namespace splash::metal;
constexpr uint32_t sticky=0x80000000u;
void require(bool value,const std::string&message) {
  if(!value)throw std::runtime_error(message);
}
float number(uint16_t value){return std::bit_cast<float>(uint32_t(value)<<16);}
uint16_t bf16(float value){return splash::flash::benchmark::bf16(value);}
std::string js(double value) {
  if(!std::isfinite(value))return "null";
  std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();
}
std::string text(id value) {
  require([value isKindOfClass:NSString.class],"R5 metadata string");
  return [(NSString*)value UTF8String];
}
uint64_t integer(id value) {
  require([value isKindOfClass:NSNumber.class]&&[(NSNumber*)value longLongValue]>=0,
    "R5 metadata nonnegative integer");
  return [(NSNumber*)value unsignedLongLongValue];
}
NSDictionary* json(const fs::path&path) {
  const uint64_t bytes=fs::file_size(path);
  require(bytes&&bytes<(4ULL<<20),"R5 bounded metadata file size");
  NSData*data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  NSError*error=nil;
  id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(!error&&[value isKindOfClass:NSDictionary.class],"R5 metadata JSON dictionary");
  return value;
}
void atomicJSON(const fs::path&path,const std::string&body) {
  const fs::path temporary=path.string()+".partial";
  std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
  out<<body<<'\n';out.close();require(bool(out),"R5 report write");
  fs::rename(temporary,path);
}
struct Options {
  std::string library,package,report,prefix,inputPacket;
  std::vector<uint32_t>indices{0};
  bool syntheticCases=false;
};
struct Audit {
  fs::path report;
  std::string stage="start",reason;
  bool backendCreated=false,backendDestroyed=false,payloadRead=false,GPUExecuted=false;
  uint64_t finalOwner=0,finalReserved=0,finalObserved=0,denials=0;
  uint64_t vendorBefore=0,vendorFinal=0,vendorPeak=0,ownerPeak=0;
  uint64_t finalSparseResident=0,finalSparseVirtual=0,finalPendingUnmaps=0,observedPeak=0;
  bool backendHealthy=false,backendDrained=false;
  bool teardownSampled=false;
  std::vector<std::string>completedCases;
  std::string casesJSON()const {
    std::ostringstream out;out<<'[';
    for(size_t i=0;i<completedCases.size();++i){if(i)out<<',';out<<completedCases[i];}
    out<<']';return out.str();
  }
  bool ownedHandlesZero()const noexcept{return finalOwner==0;}
  bool sparseResourcesZero()const noexcept {
    return !finalSparseResident&&!finalSparseVirtual&&!finalPendingUnmaps;
  }
  bool sampledProcessCacheWithinBudget()const noexcept {
    return vendorBefore<=q::governorLimit&&vendorFinal<=q::governorLimit&&
      vendorPeak<=q::governorLimit&&finalObserved<=q::governorLimit&&
      observedPeak<=q::governorLimit&&ownerPeak<=q::governorLimit;
  }
  bool resourceGatePass()const noexcept {
    return backendDestroyed&&teardownSampled&&ownedHandlesZero()&&sparseResourcesZero()&&
      !finalReserved&&!denials&&backendHealthy&&backendDrained&&sampledProcessCacheWithinBudget();
  }
  void checkpoint(const std::string&next) {
    stage=next;
    std::ostringstream out;
    out<<"{\"schema\":\"R5-raw-odd-rowpair-checkpoint-v2\",\"completed\":false,\"stage\":"
      <<splash::json::quote(stage)<<",\"backend_created\":"<<(backendCreated?"true":"false")
      <<",\"payload_read\":"<<(payloadRead?"true":"false")
      <<",\"GPU_executed\":"<<(GPUExecuted?"true":"false")
      <<",\"completed_shape_cases\":"<<completedCases.size()
      <<",\"cases\":"<<casesJSON()<<'}';
    atomicJSON(report.string()+".checkpoint.json",out.str());
  }
  [[noreturn]] void fail(const std::string&message) {
    reason=message;
    std::ostringstream out;
    out<<"{\"schema\":\"R5-raw-odd-rowpair-failure-checkpoint-v2\",\"pass\":false,\"completed\":false,\"stage\":"
      <<splash::json::quote(stage)<<",\"reason\":"<<splash::json::quote(message)
      <<",\"backend_created\":"<<(backendCreated?"true":"false")
      <<",\"payload_read\":"<<(payloadRead?"true":"false")
      <<",\"GPU_executed\":"<<(GPUExecuted?"true":"false")
      <<",\"completed_shape_cases\":"<<completedCases.size()
      <<",\"cases\":"<<casesJSON()<<'}';
    atomicJSON(report.string()+".failure.checkpoint.json",out.str());
    throw std::runtime_error(message);
  }
  void must(bool value,const std::string&message){if(!value)fail(message);}
  std::string teardownJSON()const {
    std::ostringstream out;
    out<<"{\"backend_destroyed\":"<<(backendDestroyed?"true":"false")
      <<",\"owner_allocated_bytes_last_pre_destructor\":"<<finalOwner
      <<",\"Gov_reserved_bytes_last_pre_destructor\":"<<finalReserved
      <<",\"Gov_observed_resident_bytes_last_pre_destructor\":"<<finalObserved
      <<",\"Gov_observed_peak_sampled_bytes\":"<<observedPeak
      <<",\"sparse_resident_bytes_last_pre_destructor\":"<<finalSparseResident
      <<",\"sparse_virtual_bytes_last_pre_destructor\":"<<finalSparseVirtual
      <<",\"pending_sparse_unmaps_last_pre_destructor\":"<<finalPendingUnmaps
      <<",\"owner_Gov_sample_scope\":\"after all case handles released, immediately before backend/Gov destruction\""
      <<",\"device_current_allocated_bytes_after_backend_and_autorelease_destruction\":"<<vendorFinal
      <<",\"device_initial_allocated_bytes\":"<<vendorBefore
      <<",\"highest_sampled_device_bytes\":"<<vendorPeak
      <<",\"owner_peak_allocated_bytes\":"<<ownerPeak
      <<",\"Gov_denied_reservations\":"<<denials
      <<",\"teardown_sampled\":"<<(teardownSampled?"true":"false")
      <<",\"owned_handles_zero\":"<<(ownedHandlesZero()?"true":"false")
      <<",\"sparse_resources_zero\":"<<(sparseResourcesZero()?"true":"false")
      <<",\"reserved_zero\":"<<(!finalReserved?"true":"false")
      <<",\"denials_zero\":"<<(!denials?"true":"false")
      <<",\"backend_healthy_before_destroy\":"<<(backendHealthy?"true":"false")
      <<",\"backend_drained_before_destroy\":"<<(backendDrained?"true":"false")
      <<",\"sampled_process_cache_within_budget\":"<<(sampledProcessCacheWithinBudget()?"true":"false")
      <<",\"resource_gate_pass\":"<<(resourceGatePass()?"true":"false")
      <<",\"sampled_process_cache_budget_bytes\":"<<q::governorLimit
      <<",\"device_and_Gov_observed_are_process_cache_samples_not_owned_leak_proof\":true"
      <<",\"global_physical_or_device_free_zero_claimed\":false"
      <<",\"device_sampling_is_process_counter_not_DRAM_traffic\":true}";
    return out.str();
  }
};
struct Source {
  q::Shape shape;
  std::array<std::string,3>files;
  std::array<uint64_t,3>offset{},length{};
  std::string identity,manifestSHA;
};
Source selectSource(const Options&options,q::Shape wanted) {
  const fs::path package=options.package,manifest=package/"manifest.json";
  NSDictionary*root=json(manifest);
  require(text(root[@"schema"])=="splash-local-qwen4-affine-v1","R5 original native manifest schema");
  Source result;result.identity=text(root[@"source_identity_sha256"]);
  require(result.identity=="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
    "R5 original source identity");
  NSDictionary*tensors=root[@"tensors"];NSDictionary*quant=root[@"quantization"];
  require([tensors isKindOfClass:NSDictionary.class]&&[quant isKindOfClass:NSDictionary.class],
    "R5 native tensor/quantization metadata");
  std::vector<std::string>prefixes;
  if(!options.prefix.empty())prefixes.push_back(options.prefix);
  else {
    for(NSString*key in tensors) {
      const std::string name=text(key);
      constexpr std::string_view suffix=".weight";
      if(name.starts_with("language_model.")&&name.ends_with(suffix))
        prefixes.push_back(name.substr(0,name.size()-suffix.size()));
    }
    std::sort(prefixes.begin(),prefixes.end());
  }
  bool found=false;
  for(const auto&prefix:prefixes) {
    NSString*p=[NSString stringWithUTF8String:prefix.c_str()];
    NSDictionary*w=tensors[[p stringByAppendingString:@".weight"]];
    NSDictionary*s=tensors[[p stringByAppendingString:@".scales"]];
    NSDictionary*b=tensors[[p stringByAppendingString:@".biases"]];
    if(!w||!s||!b)continue;
    NSDictionary*role=quant[p]?:quant;
    if(!role[@"bits"]||!role[@"group_size"]||!role[@"mode"])continue;
    if(integer(role[@"bits"])!=wanted.bits||integer(role[@"group_size"])!=wanted.group||
        text(role[@"mode"])!="affine")continue;
    NSArray*ws=w[@"shape"];NSArray*ss=s[@"shape"];NSArray*bs=b[@"shape"];
    if(![ws isKindOfClass:NSArray.class]||ws.count!=2||
       ![ss isKindOfClass:NSArray.class]||ss.count!=2||
       ![bs isKindOfClass:NSArray.class]||bs.count!=2)continue;
    if(integer(ws[0])!=wanted.N||integer(ws[1])!=uint64_t(wanted.K)*wanted.bits/32||
       integer(ss[0])!=wanted.N||integer(ss[1])!=wanted.K/wanted.group||
       integer(bs[0])!=wanted.N||integer(bs[1])!=wanted.K/wanted.group)continue;
    wanted.prefix=prefix;result.shape=wanted;
    uint32_t i=0;
    for(NSDictionary*t in @[w,s,b]) {
      require(text(t[@"dtype"])==(i?"BF16":"U32"),"R5 native coefficient dtype");
      const uint64_t minimum=i?q::parameterRowBytes(wanted):q::codeRowBytes(wanted);
      const uint64_t stride=t[@"row_stride_bytes"]?integer(t[@"row_stride_bytes"]):minimum;
      if(i==0)result.shape.weightRowStride=stride;
      else if(i==1)result.shape.parameterRowStride=stride;
      else require(stride==result.shape.parameterRowStride,"R5 scale/bias stride disagreement");
      result.offset[i]=integer(t[@"offset"]);result.length[i]=integer(t[@"length"]);
      require(result.length[i]>=q::extent(wanted.N,stride,minimum),
        "R5 selected native coefficient extent");
      require(result.length[i]<=q::selectedSourceLimit,"R5 selected tensor exceeds64MiB");
      const fs::path shard=text(t[@"shard"]);
      require(!shard.is_absolute()&&shard.string().find("..")==std::string::npos,
        "R5 native shard relative path");
      result.files[i]=(package/shard).string();++i;
    }
    // Read only the selected logical projection extents, never a whole shard.
    result.length=q::coefficientBytes(result.shape);
    uint64_t total=0;for(uint64_t n:result.length)total=q::add(total,n);
    require(total<=q::selectedSourceLimit,"R5 combined selected coefficient spans exceed64MiB");
    found=true;break;
  }
  require(found,"R5 admitted observed shape missing original native metadata role");
  NSData*data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:manifest.c_str()]];
  result.manifestSHA=q::hash(data.bytes,data.length);
  return result;
}
struct Graph {FlashAffineParams p{};ComputeDispatch d;};
Graph graph(const q::Shape&s,const std::array<MetalBuffer,7>&buffers,bool paired,bool tap,
    MetalBuffer raw={},FlashAffineParams p={},DispatchSize groups={0,0,0}) {
  Graph result;result.p=p.rows?p:q::parameters(s);
  result.d.pipelineName=q::kernel(s,paired,tap);
  for(uint32_t i=0;i<7;++i)result.d.buffers.push_back({i,buffers[i]});
  if(tap)result.d.buffers.push_back({8,raw});
  result.d.threadgroups=groups.x?groups:DispatchSize{s.N/8,paired?3u:5u,1};
  result.d.threadsPerThreadgroup={64,1,1};return result;
}
CommandTiming submit(MetalBackend&backend,Graph&g) {
  g.d.bytes={{7,&g.p,sizeof(g.p)}};return backend.submitCommand({&g.d,1});
}
void validated(const q::Shape&s,const std::array<MetalBuffer,7>&buffers,
    const FlashAffineParams&p,bool paired,MetalBuffer raw={},DispatchSize groups={0,0,0},
    DispatchSize threads={64,1,1}) {
  std::array<q::Span,7>spans{};
  for(size_t i=0;i<7;++i) {
    require(buffers[i]&&buffers[i].storage()==BufferStorage::Shared&&buffers[i].contents(),
      "R5 actual Shared view validation");
    spans[i]={reinterpret_cast<uintptr_t>(buffers[i].contents()),buffers[i].sizeBytes()};
  }
  if(!groups.x)groups={s.N/8,paired?3u:5u,1};
  q::validate(s,p,spans,paired,groups.x,groups.y,groups.z,threads.x,threads.y,threads.z,
    {reinterpret_cast<uintptr_t>(raw.contents()),raw.sizeBytes()});
}
struct Result {q::Guarded out,timed,raw,diag;};
void reset(Result&r) {
  r.out.poison();r.timed.poison();r.raw.poison();
  *static_cast<uint32_t*>(r.diag.view.contents())=sticky;
}
bool clean(const Result&r){return r.out.clean()&&r.timed.clean()&&r.raw.clean()&&r.diag.clean();}
std::array<MetalBuffer,7>bindings(MetalBuffer input,const std::array<MetalBuffer,3>&source,
    const Result&r,bool timed=false) {
  return {input,source[0],source[1],source[2],input,timed?r.timed.view:r.out.view,r.diag.view};
}
struct Comparison {
  uint64_t bf16Differences=0,rawDifferences=0,firstPairDifferences=0,secondPairDifferences=0,tailDifferences=0,nonfinite=0;
  std::array<double,5>l2{},cosine{};
  bool pass=true;
};
Comparison compare(const q::Shape&s,const Result&a,const Result&b,bool finite) {
  Comparison result;
  const auto*x=static_cast<const uint16_t*>(a.out.view.contents());
  const auto*y=static_cast<const uint16_t*>(b.out.view.contents());
  const auto*rx=static_cast<const uint32_t*>(a.raw.view.contents());
  const auto*ry=static_cast<const uint32_t*>(b.raw.view.contents());
  for(uint32_t row=0;row<5;++row) {
    double error=0,nx=0,ny=0,dot=0;
    for(uint32_t n=0;n<s.N;++n) {
      const uint64_t index=uint64_t(row)*s.N+n;
      const uint64_t differences=uint64_t(x[index]!=y[index])+uint64_t(rx[index]!=ry[index]);
      result.bf16Differences+=x[index]!=y[index];result.rawDifferences+=rx[index]!=ry[index];
      if(row<2)result.firstPairDifferences+=differences;
      else if(row<4)result.secondPairDifferences+=differences;
      else result.tailDifferences+=differences;
      const double u=number(x[index]),v=number(y[index]);
      if(!std::isfinite(u)||!std::isfinite(v)){++result.nonfinite;continue;}
      error+=(u-v)*(u-v);nx+=u*u;ny+=v*v;dot+=u*v;
    }
    result.l2[row]=nx?std::sqrt(error/nx):(error?INFINITY:0);
    result.cosine[row]=nx&&ny?dot/std::sqrt(nx*ny):(nx==ny?1:0);
    if(finite)result.pass&=std::isfinite(result.l2[row])&&result.l2[row]<=1e-4&&result.cosine[row]>=.999999;
  }
  result.pass&=!result.bf16Differences&&!result.rawDifferences&&clean(a)&&clean(b)&&
    *static_cast<const uint32_t*>(a.diag.view.contents())==*static_cast<const uint32_t*>(b.diag.view.contents());
  if(finite)result.pass&=!result.nonfinite;
  return result;
}
std::string comparisonJSON(const Comparison&c) {
  std::ostringstream out;
  out<<"{\"pass\":"<<(c.pass?"true":"false")<<",\"BF16_differences\":"<<c.bf16Differences
    <<",\"raw_F32_bit_differences\":"<<c.rawDifferences
    <<",\"rows0_1_differences\":"<<c.firstPairDifferences
    <<",\"rows2_3_differences\":"<<c.secondPairDifferences
    <<",\"literal_row4_differences\":"<<c.tailDifferences
    <<",\"nonfinite_words\":"<<c.nonfinite<<",\"relative_l2\":[";
  for(uint32_t r=0;r<5;++r){if(r)out<<',';out<<js(c.l2[r]);}
  out<<"],\"cosine\":[";for(uint32_t r=0;r<5;++r){if(r)out<<',';out<<js(c.cosine[r]);}
  out<<"]}";return out.str();
}
void pattern(const q::Shape&s,MetalBuffer input,uint32_t mode) {
  auto*x=static_cast<uint16_t*>(input.contents());
  for(uint32_t r=0;r<5;++r)for(uint32_t k=0;k<s.K;++k) {
    float value=float(int((k*37+r*13)%257)-128)/128;
    if(mode==1)value=float(int((k*37)%257)-128)/128;
    if(mode==2)value=0;
    if(mode==3)value=(k&1)?-1:1;
    if(mode==4)value=std::ldexp(value,40);
    if(mode==10)value=r==4?value:0;
    if(mode==11)value=r==4?-value:float(int((k*17)%127)-63)/64;
    if(mode==12)value=float(int((k*19+(4-r)*31)%509)-254)/256;
    x[uint64_t(r)*s.K+k]=bf16(value);
  }
  if(mode==5)x[0]=0x7fc1;
  if(mode==6)x[s.K+17]=0x7f80;
  if(mode==7)x[4*s.K+33]=0xff80;
  if(mode==8)for(uint64_t i=0;i<uint64_t(5)*s.K;++i)x[i]=uint16_t((i%2?0x8000:0)|(1+i%127));
  if(mode==9)for(uint64_t i=0;i<uint64_t(5)*s.K;++i)x[i]=uint16_t(i%2?0x8000:0);
}
uint32_t packedCode(const uint8_t*row,uint32_t k,uint32_t bits,uint64_t rowBytes) {
  const uint64_t bit=uint64_t(k)*bits,byte=bit/8;
  uint32_t word=row[byte];if(byte+1<rowBytes)word|=uint32_t(row[byte+1])<<8;
  return (word>>(bit%8))&((1u<<bits)-1);
}
std::string f64Certificate(const q::Shape&s,const std::array<q::RawSpan,3>&source,
    MetalBuffer input,MetalBuffer raw) {
  const auto*x=static_cast<const uint16_t*>(input.contents());
  const auto*w=static_cast<const uint8_t*>(source[0].data.view.contents());
  const auto*scales=static_cast<const uint8_t*>(source[1].data.view.contents());
  const auto*biases=static_cast<const uint8_t*>(source[2].data.view.contents());
  const auto*y=static_cast<const float*>(raw.contents());
  uint64_t failed=0;double maximum=0;
  std::array<uint64_t,5>samples{};
  for(uint32_t r=0;r<5;++r)for(uint32_t sample=0;sample<128;++sample) {
    const uint32_t n=uint32_t(uint64_t(sample)*(s.N-1)/127);
    const auto*row=w+uint64_t(n)*s.weightRowStride;
    const auto*sp=reinterpret_cast<const uint16_t*>(scales+uint64_t(n)*s.parameterRowStride);
    const auto*bp=reinterpret_cast<const uint16_t*>(biases+uint64_t(n)*s.parameterRowStride);
    double sum=0,absolute=0,flushed=0;
    for(uint32_t k=0;k<s.K;++k) {
      const uint32_t code=packedCode(row,k,s.bits,q::codeRowBytes(s));
      volatile float product=float(code)*number(sp[k/s.group]);
      const float coefficient=product+number(bp[k/s.group]);
      const double xv=number(x[uint64_t(r)*s.K+k]),value=xv*double(coefficient);
      sum+=value;absolute+=std::abs(value);
      if(xv&&std::abs(xv)<std::numeric_limits<float>::min())flushed+=std::abs(value);
    }
    const double bound=splash::flash::benchmark::f32DotBound(s.K,absolute,flushed);
    const double error=std::abs(double(y[uint64_t(r)*s.N+n])-sum);
    const double ratio=bound?error/bound:(error?INFINITY:0);
    ++samples[r];failed+=!std::isfinite(ratio)||ratio>1;maximum=std::max(maximum,ratio);
  }
  std::ostringstream out;
  out<<"{\"diagnostic_only\":true,\"all_five_rows_sampled\":true,\"per_row_samples\":[";
  for(uint32_t r=0;r<5;++r){if(r)out<<',';out<<samples[r];}
  out<<"],\"bound_failures\":"<<failed<<",\"maximum_ratio\":"<<js(maximum)
    <<",\"unchanged_bound\":\"FlashFloatBoundaryAudit.hpp f32DotBound(K,sumAbs,flushedInputProducts)\""
    <<",\"reference\":\"serial F64 using literal F32 s*packedCode+b reconstruction and original native strides\""
    <<",\"does_not_relax_exact_F32_BF16_or_l2_cosine_gates\":true}";
  return out.str();
}
uint64_t cpu() {
  require(sizeof(FlashAffineParams)==64&&sizeof(CommandTiming)==200,"R5 CPU native ABI");
  uint64_t checks=2;
  const auto reject=[&](auto work) {
    bool refused=false;try{work();}catch(const std::invalid_argument&){refused=true;}
    require(refused,"R5 CPU negative admission accepted");++checks;
  };
  for(uint32_t index=0;index<7;++index) {
    const auto s=q::shape(index);const auto p=q::parameters(s);
    require(q::admissionBytes(s,true)<=q::governorLimit,"R5 CPU shape byte plan");++checks;
    std::array<q::Span,7>spans{};
    const auto coeff=q::coefficientBytes(s);
    const std::array<uint64_t,7>length{q::inputBytes(s),coeff[0],coeff[1],coeff[2],1,q::outputBytes(s),4};
    uintptr_t address=0x100000000ULL;
    for(uint32_t i=0;i<7;++i){spans[i]={address,length[i]};address+=q::rounded(length[i])+16384;}
    q::validate(s,p,spans,true,s.N/8,3,1,64);q::validate(s,p,spans,false,s.N/8,5,1,64);checks+=2;
    for(uint32_t i=0;i<7;++i){auto bad=spans;bad[i].bytes=0;reject([&]{q::validate(s,p,bad,true,s.N/8,3,1,64);});}
    for(uint32_t w:{5u,6u})for(uint32_t r:{0u,1u,2u,3u,4u}) {
      auto bad=spans;bad[w].address=bad[r].address;
      reject([&]{q::validate(s,p,bad,true,s.N/8,3,1,64);});
    }
    for(uint32_t i=0;i<8;++i) {
      auto bad=p;
      switch(i){case 0:bad.rows=4;break;case 1:bad.flags=1;break;
        case 2:bad.selections=2;break;case 3:bad.experts=2;break;
        case 4:--bad.weight_row_stride_bytes;break;case 5:--bad.parameter_row_stride_bytes;break;
        case 6:bad.parameter_expert_stride_bytes=1;break;case 7:bad.weight_row_stride_bytes=UINT64_MAX;break;}
      reject([&]{q::validate(s,bad,spans,true,s.N/8,3,1,64);});
    }
    reject([&]{q::validate(s,p,spans,true,s.N/8-1,3,1,64);});
    reject([&]{q::validate(s,p,spans,true,s.N/8,2,1,64);});
    reject([&]{q::validate(s,p,spans,true,s.N/8,3,2,64);});
    reject([&]{q::validate(s,p,spans,true,s.N/8,3,1,32);});
    std::vector<uint8_t>ownership(uint64_t(5)*s.N,0);
    for(uint32_t gy=0;gy<3;++gy)for(uint32_t gx=0;gx<s.N/8;++gx)
      for(uint32_t sg=0;sg<2;++sg)for(uint32_t n=0;n<4;++n)
        for(uint32_t row=gy*2;row<std::min<uint32_t>(5,gy*2+2);++row)
          ++ownership[uint64_t(row)*s.N+gx*8+sg*4+n];
    for(uint8_t owners:ownership){require(owners==1,"R5 CPU pair/tail ownership");++checks;}
  }
  for(uint32_t bits:{4u,5u,6u})for(uint32_t slot=0;slot<8;++slot)
    for(uint32_t value=0;value<(1u<<bits);++value) {
      std::array<uint8_t,8>packed{};const uint64_t word=uint64_t(value)<<(slot*bits);
      for(uint32_t byte=0;byte<8;++byte)packed[byte]=uint8_t(word>>(8*byte));
      for(uint32_t k=0;k<8;++k){require(packedCode(packed.data(),k,bits,8)==(k==slot?value:0),"R5 CPU packedcross decode");++checks;}
    }
  return checks;
}
std::string pipelineMetadata(const std::string&library,const q::Shape&s) {
  id<MTLDevice>device=MTLCreateSystemDefaultDevice();require(device!=nil,"R5 Metal device");
  NSError*error=nil;
  id<MTLLibrary>lib=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:library.c_str()]] error:&error];
  require(lib!=nil,"R5 component library");
  std::ostringstream out;out<<'[';bool comma=false;
  for(bool paired:{false,true})for(bool tap:{false,true}) {
    const std::string name=q::kernel(s,paired,tap);
    id<MTLFunction>function=[lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    require(function!=nil,"R5 required kernel missing: "+name);
    id<MTLComputePipelineState>pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline!=nil&&pipeline.threadExecutionWidth==32&&pipeline.maxTotalThreadsPerThreadgroup>=64,
      "R5 pipeline SIMD geometry");
    if(comma)out<<',';comma=true;
    out<<"{\"name\":"<<splash::json::quote(name)<<",\"execution_width\":"<<pipeline.threadExecutionWidth
      <<",\"maximum_threads\":"<<pipeline.maxTotalThreadsPerThreadgroup
      <<",\"static_threadgroup_bytes\":"<<pipeline.staticThreadgroupMemoryLength<<'}';
  }
  out<<']';return out.str();
}

std::string runCase(MetalBackend&backend,splash::engine::MemoryGovernor&governor,
    const Options&options,const Source&spec,Audit&audit) {
  const auto&s=spec.shape;const uint64_t admitted=q::admissionBytes(s,!options.inputPacket.empty());
  audit.checkpoint("shape-admission-before-payload");
  auto reservation=governor.tryReserve(admitted);
  audit.must(bool(reservation),"R5 bounded256MiB admission denied before payload");
  const uint64_t before=backend.memoryStats().allocatedBytes;
  std::array<q::RawSpan,3>source;
  for(uint32_t i=0;i<3;++i){source[i]=q::RawSpan::load(backend,spec.files[i],spec.offset[i],spec.length[i]);audit.payloadRead=true;}
  std::array<MetalBuffer,3>native{source[0].data.view,source[1].data.view,source[2].data.view};
  auto input=q::Guarded::allocate(backend,q::inputBytes(s),"R5 five-row BF16 input");
  std::array<Result,2>results;
  for(auto&r:results) {
    r.out=q::Guarded::allocate(backend,q::outputBytes(s),"R5 tapped BF16 outputs");
    r.timed=q::Guarded::allocate(backend,q::outputBytes(s),"R5 untapped measured outputs");
    r.raw=q::Guarded::allocate(backend,q::rawBytes(s),"R5 original result F32 tap");
    r.diag=q::Guarded::allocate(backend,4,"R5 sticky diagnostics");
  }
  std::array<q::Guarded,3>padded;
  padded[0]=q::Guarded::allocate(backend,uint64_t(s.N)*(s.weightRowStride+4),"R5 padded unaligned code",1);
  for(uint32_t i=1;i<3;++i)padded[i]=q::Guarded::allocate(backend,uint64_t(s.N)*(s.parameterRowStride+4),"R5 padded BF16 params",4);
  for(uint32_t n=0;n<s.N;++n)for(uint32_t i=0;i<3;++i) {
    const uint64_t logical=i?q::parameterRowBytes(s):q::codeRowBytes(s);
    const uint64_t oldStride=i?s.parameterRowStride:s.weightRowStride,newStride=oldStride+4;
    std::memcpy(static_cast<uint8_t*>(padded[i].view.contents())+uint64_t(n)*newStride,
      static_cast<const uint8_t*>(native[i].contents())+uint64_t(n)*oldStride,logical);
  }
  auto aliasBacking=q::Guarded::allocate(backend,q::inputBytes(s)+q::outputBytes(s)+32768,"R5 alias-rejection backing");
  pattern(s,input.view,0);
  bool actual=false;std::string inputScope="synthetic independent rows; not model input";
  if(!options.inputPacket.empty()) {
    audit.checkpoint("Root-only-current-input-packet");
    NSDictionary*packet=json(options.inputPacket);
    const std::string projection=packet[@"projection"]?text(packet[@"projection"]):text(packet[@"prefix"]);
    const uint64_t K=packet[@"input_size"]?integer(packet[@"input_size"]):integer(packet[@"K"]);
    audit.must(integer(packet[@"rows"])==5&&K==s.K&&projection==s.prefix&&
      text(packet[@"source_identity_sha256"])==spec.identity,"R5 current packet identity/shape mismatch");
    const fs::path file=text(packet[@"file"]);
    audit.must(fs::file_size(file)==q::inputBytes(s),"R5 packet exact input extent");
    auto capture=q::RawSpan::load(backend,file.string(),0,q::inputBytes(s));
    audit.must(capture.sha==text(packet[@"payload_sha256"]),"R5 packet input digest mismatch");
    std::memcpy(input.view.contents(),capture.data.view.contents(),q::inputBytes(s));
    actual=true;inputScope=packet[@"scope"]?text(packet[@"scope"]):"Root current input packet";
  }
  std::vector<uint8_t>normal(q::inputBytes(s));std::memcpy(normal.data(),input.view.contents(),normal.size());
  const std::string inputSHA=q::hash(input.view.contents(),q::inputBytes(s));
  audit.must(backend.memoryStats().allocatedBytes-before<=admitted,"R5 actual allocations exceed admitted plan");
  reservation->commit();
  const uint64_t allocation=backend.memoryStats().allocatedBytes-before;
  const auto healthy=[&] {
    audit.must(input.clean()&&clean(results[0])&&clean(results[1])&&aliasBacking.clean(),"R5 guard/poison canary changed");
    for(const auto&raw:source)audit.must(raw.immutable(),"R5 immutable native source changed");
    for(const auto&copy:padded)audit.must(copy.clean(),"R5 padded source canary changed");
  };
  std::ostringstream proofs;proofs<<'[';bool comma=false;uint64_t checks=0;
  const auto check=[&](const std::string&name,bool finite,FlashAffineParams params,
      const std::array<MetalBuffer,3>&coefficients) {
    audit.checkpoint("proof:"+name);
    const std::string unchangedInput=q::hash(input.view.contents(),q::inputBytes(s));
    std::array<std::string,3>unchanged;
    for(uint32_t i=0;i<3;++i)unchanged[i]=q::hash(coefficients[i].contents(),coefficients[i].sizeBytes());
    for(auto&r:results)reset(r);
    for(uint32_t i=0;i<2;++i) {
      auto buffers=bindings(input.view,coefficients,results[i]);
      auto g=graph(s,buffers,i==1,true,results[i].raw.view);
      g.p=params;submit(backend,g);audit.GPUExecuted=true;
    }
    const auto c=compare(s,results[0],results[1],finite);
    bool shipping=true;
    for(uint32_t i=0;i<2;++i) {
      const uint32_t diagnostic=*static_cast<const uint32_t*>(results[i].diag.view.contents());
      *static_cast<uint32_t*>(results[i].diag.view.contents())=sticky;
      auto g=graph(s,bindings(input.view,coefficients,results[i],true),i==1,false);
      g.p=params;submit(backend,g);
      shipping&=std::memcmp(results[i].out.view.contents(),results[i].timed.view.contents(),q::outputBytes(s))==0&&
        diagnostic==*static_cast<const uint32_t*>(results[i].diag.view.contents());
    }
    ++checks;
    if(comma)proofs<<',';comma=true;
    proofs<<"{\"case\":"<<splash::json::quote(name)<<",\"shipping_BF16_and_diagnostic_tap_exact\":"
      <<(shipping?"true":"false")<<",\"comparison\":"<<comparisonJSON(c)<<'}';
    audit.must(c.pass&&shipping,"R5 full pair/tail rawF32/BF16/shipping proof failed: "+name);
    healthy();
    audit.must(q::hash(input.view.contents(),q::inputBytes(s))==unchangedInput,"R5 case input mutated");
    for(uint32_t i=0;i<3;++i)audit.must(q::hash(coefficients[i].contents(),coefficients[i].sizeBytes())==unchanged[i],"R5 diagnostic coefficients mutated");
  };
  check(actual?"actual-current-input":"synthetic-not-model-independent-rows",true,q::parameters(s),native);
  const std::string f64=f64Certificate(s,source,input.view,results[0].raw.view);
  // Mandatory tail and malformed-input diagnostics; optional flag enables all
  // thirteen robust input patterns, never prospective unobserved shape claims.
  for(uint32_t mode=1;mode<=12;++mode) {
    if(!options.syntheticCases&&mode!=5&&mode!=6&&mode!=7&&mode!=10&&mode!=11)continue;
    pattern(s,input.view,mode);
    check("synthetic-not-model-pattern-"+std::to_string(mode),mode<5||mode>=8,q::parameters(s),native);
    if(mode>=5&&mode<=7)audit.must((*static_cast<const uint32_t*>(results[0].diag.view.contents())&4u)!=0,"R5 original nonfinite diagnostic absent");
  }
  std::memcpy(input.view.contents(),normal.data(),normal.size());
  const std::array<MetalBuffer,3>paddedViews{padded[0].view,padded[1].view,padded[2].view};
  const auto paddedParams=q::parameters(s,s.weightRowStride+4,s.parameterRowStride+4);
  check("synthetic-stride-padding-poison-and-byte-unaligned-code",true,paddedParams,paddedViews);
  // Fault only an independent copied first parameter row, restore afterwards.
  std::array<std::vector<uint8_t>,2>parameterBackup;
  for(uint32_t i=0;i<2;++i) {
    parameterBackup[i].resize(q::parameterRowBytes(s));
    std::memcpy(parameterBackup[i].data(),padded[i+1].view.contents(),parameterBackup[i].size());
  }
  for(const auto&fault:std::array<std::array<uint16_t,2>,3>{{{{0,0}},{{0x7fc1,0}},{{0x7f80,0}}}}) {
    for(uint32_t i=0;i<2;++i) {
      auto*data=static_cast<uint16_t*>(padded[i+1].view.contents());
      for(uint32_t group=0;group<s.K/s.group;++group)data[group]=fault[i];
    }
    check("synthetic-copied-parameter-fault-"+std::to_string(fault[0]),fault[0]==0,paddedParams,paddedViews);
    for(uint32_t i=0;i<2;++i)std::memcpy(padded[i+1].view.contents(),parameterBackup[i].data(),parameterBackup[i].size());
  }
  for(uint32_t invalid=0;invalid<9;++invalid) {
    auto p=q::parameters(s);
    switch(invalid){case 0:p.rows=0;break;case 1:p.selections=0;break;case 2:p.experts=0;break;
      case 3:--p.input_size;break;case 4:p.output_size=0;break;case 5:p.bits=8;break;
      case 6:p.flags=4;break;case 7:--p.weight_row_stride_bytes;break;case 8:p.parameter_expert_stride_bytes=1;break;}
    check("synthetic-common-invalid-metadata-"+std::to_string(invalid),false,p,native);
    audit.must(*static_cast<const uint32_t*>(results[0].diag.view.contents())==(sticky|2),"R5 common invalid metadata sticky policy");
  }
  auto good=bindings(input.view,native,results[1]);
  const auto hostReject=[&](auto work,const std::string&name) {
    bool refused=false;try{work();}catch(const std::exception&){refused=true;}
    audit.must(refused,"R5 host guard admitted "+name);++checks;healthy();
  };
  for(uint32_t slot=0;slot<7;++slot) {
    auto bad=good;bad[slot]={};
    hostReject([&]{validated(s,bad,q::parameters(s),true);},"short/empty slot"+std::to_string(slot));
  }
  for(const auto groups:std::array<DispatchSize,4>{{{s.N/8-1,3,1},{s.N/8,2,1},{s.N/8,3,2},{s.N/8+1,3,1}}})
    hostReject([&]{validated(s,good,q::parameters(s),true,{},groups);},"omitted/excess host grid");
  hostReject([&]{validated(s,good,q::parameters(s),true,{}, {},{32,1,1});},"thread count");
  for(uint32_t writable:{5u,6u})for(uint32_t sourceSlot:{0u,1u,2u,3u,4u}) {
    auto bad=good;
    if(sourceSlot==0||sourceSlot==4) {
      bad[sourceSlot]=backend.view(aliasBacking.base,64+16384,q::inputBytes(s));
      bad[writable]=backend.view(aliasBacking.base,64,writable==5?q::outputBytes(s):4);
      if(writable==6)bad[writable]=backend.view(aliasBacking.base,64+16384,4);
    } else bad[writable]=backend.view(bad[sourceSlot],0,writable==5?q::outputBytes(s):4);
    hostReject([&]{validated(s,bad,q::parameters(s),true);},"valid-length source writable alias");
  }
  {
    auto bad=good;bad[6]=backend.view(results[1].out.base,results[1].out.offset,4);
    hostReject([&]{validated(s,bad,q::parameters(s),true);},"output diagnostic alias");
  }
  {
    const auto rawAlias=backend.view(source[0].data.base,source[0].data.offset,q::rawBytes(s));
    hostReject([&]{validated(s,good,q::parameters(s),true,rawAlias);},"F32 tap source alias");
  }
  for(uint32_t slot:{0u,2u,3u,5u,6u}) {
    auto bad=good;
    const auto&backing=slot==0?input:(slot==5?results[1].out:(slot==6?results[1].diag:source[slot-1].data));
    bad[slot]=backend.view(backing.base,backing.offset+1,backing.logical);
    hostReject([&]{validated(s,bad,q::parameters(s),true);},"typed alignment");
  }
  std::memcpy(input.view.contents(),normal.data(),normal.size());
  check("normal-recovery-before-timing",true,q::parameters(s),native);
  healthy();
  for(uint32_t i=0;i<2;++i)validated(s,bindings(input.view,native,results[i],true),q::parameters(s),i==1);
  std::array<Graph,2>timed{graph(s,bindings(input.view,native,results[0],true),false,false),
    graph(s,bindings(input.view,native,results[1],true),true,false)};
  std::array<double,2>warm{};std::array<uint64_t,2>warmCalls{};
  std::array<std::vector<CommandTiming>,2>times;std::array<std::vector<uint32_t>,2>positions;
  audit.checkpoint("no-CPU-payload-window-warmup-and18-balanced-whole-projections");
  // No contents(), hash, guard check, source stat, memcpy or CPU tensor access
  // occurs until BOTH >=150ms warmups and ALL36 measured positions complete.
  for(uint32_t i=0;i<2;++i)while(warm[i]<.150) {
    const auto t=submit(backend,timed[i]);
    audit.must(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>0,"R5 invalid GPU warm timestamp");
    warm[i]+=t.gpuSeconds;++warmCalls[i];
  }
  for(uint32_t pair=0;pair<18;++pair)for(uint32_t position=0;position<2;++position) {
    const uint32_t route=(pair&1)?1-position:position;
    times[route].push_back(submit(backend,timed[route]));positions[route].push_back(position);
  }
  for(const auto&route:times)for(const auto&t:route)
    audit.must(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>0&&
      std::isfinite(t.wallSeconds)&&t.wallSeconds>0,"R5 invalid measured command timestamp");
  std::array<std::vector<uint8_t>,2>lastMeasured;
  for(uint32_t i=0;i<2;++i) {
    lastMeasured[i].resize(q::outputBytes(s));
    std::memcpy(lastMeasured[i].data(),results[i].timed.view.contents(),lastMeasured[i].size());
  }
  healthy();
  audit.must(lastMeasured[0]==lastMeasured[1],"R5 actual last measured shipping full BF16 differs");
  audit.must(q::hash(input.view.contents(),q::inputBytes(s))==inputSHA,"R5 timed input changed");
  check("normal-post-timing-shipping-register-taps",true,q::parameters(s),native);
  for(uint32_t i=0;i<2;++i)audit.must(
    std::memcmp(results[i].out.view.contents(),lastMeasured[i].data(),q::outputBytes(s))==0,
    "R5 post-timing probe not coupled to actual last measured BF16 result");
  proofs<<']';
  const auto median=[](const auto&values,bool GPU) {
    std::vector<double>out;for(const auto&t:values)out.push_back((GPU?t.gpuSeconds:t.wallSeconds)*1000);
    std::sort(out.begin(),out.end());return (out[8]+out[9])*.5;
  };
  std::ostringstream out;out<<std::setprecision(17);
  out<<"{\"pass\":true,\"prefix\":"<<splash::json::quote(s.prefix)
    <<",\"K\":"<<s.K<<",\"N\":"<<s.N<<",\"bits\":"<<s.bits<<",\"group_size\":"<<s.group
    <<",\"physical_rows\":5,\"input_is_actual_current_packet\":"<<(actual?"true":"false")
    <<",\"input_scope\":"<<splash::json::quote(inputScope)
    <<",\"input_sha256\":"<<splash::json::quote(inputSHA)
    <<",\"source_identity\":"<<splash::json::quote(spec.identity)
    <<",\"source_manifest_sha256\":"<<splash::json::quote(spec.manifestSHA)
    <<",\"selected_native_payload_bytes\":"<<(spec.length[0]+spec.length[1]+spec.length[2])
    <<",\"selected_span_limit_bytes\":"<<q::selectedSourceLimit
    <<",\"admitted_bytes\":"<<admitted<<",\"actual_owner_allocation_bytes\":"<<allocation
    <<",\"source_immutable\":true,\"checks\":"<<checks<<",\"proofs\":"<<proofs.str()
    <<",\"f64_reference\":"<<f64<<",\"per_row_l2_limit\":0.0001,\"per_row_cosine_min\":0.999999"
    <<",\"timing_executed\":true,\"timed_whole_projection\":true,\"pairs\":18"
    <<",\"CPU_tensor_access_warmup_through_final_timing\":false"
    <<",\"last_measured_shipping_and_post_timing_taps_exact\":true,\"routes\":[";
  for(uint32_t i=0;i<2;++i) {
    if(i)out<<',';
    out<<"{\"name\":"<<splash::json::quote(i?"paired0to3+literal4":"immutable-original-per-row")
      <<",\"warm_GPU_ms\":"<<warm[i]*1000<<",\"warm_calls\":"<<warmCalls[i]
      <<",\"median_GPU_ms\":"<<median(times[i],true)<<",\"median_wall_ms\":"<<median(times[i],false)
      <<",\"GPU_ms\":[";
    for(uint32_t j=0;j<18;++j){if(j)out<<',';out<<times[i][j].gpuSeconds*1000;}
    out<<"],\"position\":[";for(uint32_t j=0;j<18;++j){if(j)out<<',';out<<positions[i][j];}out<<"]}";
  }
  out<<"],\"native_selected_span_sha256\":[";
  for(uint32_t i=0;i<3;++i){if(i)out<<',';out<<splash::json::quote(source[i].sha);}
  out<<"]}";
  audit.ownerPeak=std::max(audit.ownerPeak,backend.memoryStats().peakAllocatedBytes);
  audit.vendorPeak=std::max(audit.vendorPeak,backend.memoryStats().devicePeakAllocatedBytes);
  return out.str();
}
Options parse(int argc,char**argv) {
  require(argc>=5&&std::string_view(argv[1])=="--gpu","R5 explicit Root --gpu required");
  Options result;result.library=argv[2];result.package=argv[3];result.report=argv[4];
  for(int i=5;i<argc;++i) {
    const std::string flag=argv[i];
    if(flag=="--synthetic-cases"){result.syntheticCases=true;continue;}
    require(i+1<argc,"R5 CLI missing value");const std::string value=argv[++i];
    if(flag=="--shape") {
      result.indices.clear();
      if(value=="all")for(uint32_t n=0;n<7;++n)result.indices.push_back(n);
      else {const unsigned long n=std::stoul(value);require(n<7,"R5 CLI shape0..6");result.indices.push_back(uint32_t(n));}
    } else if(flag=="--prefix")result.prefix=value;
    else if(flag=="--input-packet")result.inputPacket=value;
    else throw std::invalid_argument("R5 unknown CLI flag: "+flag);
  }
  require(result.indices.size()==1||(result.inputPacket.empty()&&result.prefix.empty()),
    "R5 all-shapes requires per-shape metadata selection and synthetic input");
  return result;
}
int GPU(const Options&options) {
  Audit audit;audit.report=options.report;
  require(!fs::exists(audit.report)&&!fs::exists(audit.report.string()+".partial")&&
    !fs::exists(audit.report.string()+".checkpoint.json"),"R5 fresh report path");
  auto&cases=audit.completedCases;std::exception_ptr failure;
  id<MTLDevice>auditDevice=nil;
  try {
    audit.checkpoint("frozen-program-before-model-metadata");
    const uint64_t bytes=fs::file_size(options.library);
    audit.must(bytes&&bytes<(64ULL<<20)&&std::strlen(kR5RawOddRowpairLibrarySHA)==64,
      "R5 frozen component program extent/seal");
    std::vector<uint8_t>program(bytes);std::ifstream in(options.library,std::ios::binary);
    in.read(reinterpret_cast<char*>(program.data()),std::streamsize(bytes));
    audit.must(in.gcount()==std::streamsize(bytes)&&q::hash(program.data(),bytes)==kR5RawOddRowpairLibrarySHA,
      "R5 component library program SHA differs");
    @autoreleasepool {
      auditDevice=MTLCreateSystemDefaultDevice();audit.must(auditDevice!=nil,"R5 device unavailable");
      audit.vendorBefore=auditDevice.currentAllocatedSize;
      {
        MetalBackend backend(options.library);audit.backendCreated=true;
        const uint64_t physical=NSProcessInfo.processInfo.physicalMemory;
        const uint64_t hostReserve=std::max<uint64_t>(16ULL<<30,physical/10);
        splash::engine::MemoryGovernor governor(backend,q::governorLimit,hostReserve);
        try {
          for(uint32_t index:options.indices) {
            audit.checkpoint("bounded-metadata-select-shape-"+std::to_string(index));
            const auto source=selectSource(options,q::shape(index));
            const auto metadata=pipelineMetadata(options.library,source.shape);
            std::string result=runCase(backend,governor,options,source,audit);
            result.pop_back();result+=",\"pipeline_metadata\":"+metadata+"}";
            cases.push_back(std::move(result));
            // Persist the complete numeric/timing evidence before the release
            // gate, so a later resource failure cannot discard passed cases.
            audit.checkpoint("completed-shape-evidence-"+std::to_string(index));
            backend.checkHealth();
            const auto caseMemory=backend.refreshMemoryStats();
            const auto caseGovernor=governor.snapshot();
            audit.observedPeak=std::max(audit.observedPeak,caseGovernor.observedResidentBytes);
            audit.vendorPeak=std::max(audit.vendorPeak,caseMemory.devicePeakAllocatedBytes);
            audit.ownerPeak=std::max(audit.ownerPeak,caseMemory.peakAllocatedBytes);
            audit.must(!caseMemory.allocatedBytes&&!caseMemory.sparseResidentBytes&&
              !caseMemory.sparseVirtualBytes&&!caseMemory.pendingSparseUnmaps,
              "R5 case owned or sparse handles leaked");
            audit.must(!caseGovernor.reservedBytes&&!caseGovernor.deniedReservations,
              "R5 case reservation or denial ledger nonzero");
            audit.must(backend.healthy()&&!backend.needsHealthCheck(),
              "R5 case backend unhealthy or outstanding submission");
            audit.must(caseGovernor.observedResidentBytes<=q::governorLimit&&
              caseMemory.deviceCurrentAllocatedBytes<=q::governorLimit&&
              caseMemory.devicePeakAllocatedBytes<=q::governorLimit,
              "R5 sampled process/cache counters exceed256MiB");
          }
        } catch(...) {failure=std::current_exception();}
        try {
          backend.drainSparseUnmaps();
          backend.checkHealth();
        } catch(...) {if(!failure)failure=std::current_exception();}
        const auto memory=backend.refreshMemoryStats();
        const auto snapshot=governor.snapshot();
        audit.finalOwner=memory.allocatedBytes;audit.finalReserved=snapshot.reservedBytes;
        audit.finalObserved=snapshot.observedResidentBytes;audit.denials=snapshot.deniedReservations;
        audit.finalSparseResident=memory.sparseResidentBytes;
        audit.finalSparseVirtual=memory.sparseVirtualBytes;
        audit.finalPendingUnmaps=memory.pendingSparseUnmaps;
        audit.backendHealthy=backend.healthy();
        audit.backendDrained=!backend.needsHealthCheck()&&!memory.pendingSparseUnmaps;
        audit.observedPeak=std::max(audit.observedPeak,snapshot.observedResidentBytes);
        audit.ownerPeak=std::max(audit.ownerPeak,memory.peakAllocatedBytes);
        audit.vendorPeak=std::max(audit.vendorPeak,memory.devicePeakAllocatedBytes);
      }
    }
    // This happens only after MetalBackend/Gov and every case's borrowed view
    // are destroyed and the enclosing autorelease pool has drained.
    audit.backendDestroyed=true;audit.vendorFinal=auditDevice.currentAllocatedSize;
    audit.vendorPeak=std::max({audit.vendorPeak,audit.vendorBefore,audit.vendorFinal});
    audit.teardownSampled=true;
    audit.checkpoint("backend-destroyed-resource-samples");
    if(failure)std::rethrow_exception(failure);
    audit.must(audit.resourceGatePass(),
      "R5 owned/sparse/reservation/health/drain or bounded process/cache resource gate failed");
    std::ostringstream report;
    report<<"{\"schema\":\"R5-raw-odd-rowpair-component-v2\",\"completed\":true,\"pass\":true"
      <<",\"diagnostic_not_canonical_performance\":true,\"worker_integration\":false"
      <<",\"whole_model_qualified\":false,\"observed_shape_count\":7,\"completed_shape_cases\":"<<cases.size()
      <<",\"Gov_budget_bytes\":"<<q::governorLimit<<",\"new_weight_cache_bytes\":0,\"cases\":[";
    for(size_t i=0;i<cases.size();++i){if(i)report<<',';report<<cases[i];}
    report<<"],\"teardown\":"<<audit.teardownJSON()
      <<",\"kernel_AIR_audit_sha256\":"<<splash::json::quote(kR5RawOddRowpairKernelAuditSHA)
      <<",\"provenance\":"<<kR5RawOddRowpairProvenance<<'}';
    atomicJSON(audit.report,report.str());
    std::cout<<"{\"pass\":true,\"report\":"<<splash::json::quote(options.report)<<"}\n";
    return 0;
  } catch(const std::exception&error) {
    audit.reason=error.what();
    std::ostringstream out;
    out<<"{\"schema\":\"R5-raw-odd-rowpair-failure-v2\",\"completed\":false,\"pass\":false,\"stage\":"
      <<splash::json::quote(audit.stage)<<",\"reason\":"<<splash::json::quote(audit.reason)
      <<",\"completed_shape_cases\":"<<cases.size()<<",\"GPU_executed\":"<<(audit.GPUExecuted?"true":"false")
      <<",\"payload_read\":"<<(audit.payloadRead?"true":"false")
      <<",\"cases\":"<<audit.casesJSON()
      <<",\"teardown\":"<<audit.teardownJSON()<<",\"provenance\":"<<kR5RawOddRowpairProvenance<<'}';
    atomicJSON(audit.report.string()+".failure.json",out.str());
    std::cerr<<audit.reason<<'\n';return 1;
  }
}
} // namespace
int main(int argc,char**argv) {
  try {
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test") {
      const uint64_t checks=cpu();
      std::cout<<"{\"pass\":true,\"checks\":"<<checks
        <<",\"observed_shapes\":7,\"rows\":5,\"GPU_work\":false,\"metadata_reads\":0,\"payload_reads\":0,\"device_created\":false}\n";
      return 0;
    }
    if(argc==2&&std::string_view(argv[1])=="--help") {
      std::cout<<"oracle --cpu-self-test\noracle --gpu LIB PACKAGE NEW_REPORT [--shape 0..6|all] [--prefix PREFIX] [--input-packet ROOT_PACKET_JSON] [--synthetic-cases]\n"
        <<"GPU mode is Root-only. Seven observed shapes; default shape0/synthetic input. No worker integration.\n";
      return 0;
    }
    return GPU(parse(argc,argv));
  } catch(const std::exception&error) {std::cerr<<error.what()<<'\n';return 1;}
}
