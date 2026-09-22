// Explicit Root-only single native projection. CPU/help modes return before
// metadata/payload file access, MTL device creation or native MetalBackend.
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
namespace q=raw_q4_rowpair_sep22;
using namespace splash::metal;
constexpr uint32_t sticky=0x80000000u;
void require(bool v,const std::string&s){if(!v)throw std::runtime_error(s);}
float number(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
uint16_t bf16(float x){uint32_t v=std::bit_cast<uint32_t>(x);if((v&0x7f800000u)==0x7f800000u)return uint16_t((v>>16)|((v&0x7fffffu)?0x40u:0u));return uint16_t((v+0x7fffu+((v>>16)&1u))>>16);}
std::string js(double v){if(!std::isfinite(v))return"null";std::ostringstream s;s<<std::setprecision(17)<<v;return s.str();}
std::string text(id x){require([x isKindOfClass:NSString.class],"rowpair metadata string invalid");return [(NSString*)x UTF8String];}
uint64_t integer(id x){require([x isKindOfClass:NSNumber.class]&&[(NSNumber*)x longLongValue]>=0,"rowpair metadata integer invalid");return[(NSNumber*)x unsignedLongLongValue];}
NSDictionary *json(const std::filesystem::path &p){require(std::filesystem::file_size(p)>0&&std::filesystem::file_size(p)<(4ULL<<20),"rowpair metadata file extent invalid");NSData*d=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:p.c_str()]];NSError*e=nil;id x=[NSJSONSerialization JSONObjectWithData:d options:0 error:&e];require(!e&&[x isKindOfClass:NSDictionary.class],"rowpair metadata JSON invalid");return x;}
struct Source {std::array<std::string,3>files;std::array<uint64_t,3>offsets,lengths;std::string manifestSHA;};
Source source(const std::filesystem::path &package){
  const auto p=package/"manifest.json";NSDictionary*m=json(p);
  require(text(m[@"schema"])=="splash-local-qwen4-affine-v1"&&text(m[@"source_identity_sha256"])=="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e","rowpair original native source identity differs");
  NSDictionary*quant=m[@"quantization"];NSString*prefix=@"language_model.model.layers.1.linear_attn.in_proj_qkv";NSDictionary*role=quant[prefix]?:quant;
  require(integer(role[@"bits"])==4&&integer(role[@"group_size"])==64&&text(role[@"mode"])=="affine","rowpair original layer1 QKV Q4/G64 differs");
  NSDictionary*tensors=m[@"tensors"];Source s;uint32_t i=0;
  for(NSString*suf in @[@"weight",@"scales",@"biases"]){NSDictionary*t=tensors[[prefix stringByAppendingFormat:@".%@",suf]];NSArray*sh=t[@"shape"];require([sh isKindOfClass:NSArray.class]&&sh.count==2&&integer(sh[0])==q::kN&&integer(sh[1])==(i?40:320),"rowpair layer1 QKV tensor shape differs");require(text(t[@"dtype"])==(i?"BF16":"U32"),"rowpair source dtype differs");s.offsets[i]=integer(t[@"offset"]);s.lengths[i]=integer(t[@"length"]);require(s.lengths[i]==(i?q::kParameters:q::kWeights),"rowpair source selected span length differs");const std::filesystem::path shard=text(t[@"shard"]);require(!shard.is_absolute()&&shard.string().find("..") == std::string::npos,"rowpair native source shard path invalid");s.files[i]=(package/shard).string();++i;}
  NSData*d=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:p.c_str()]];s.manifestSHA=q::hash(d.bytes,d.length);return s;
}
struct Graph {FlashAffineParams params{};ComputeDispatch dispatch;};
Graph graph(const std::array<MetalBuffer,7>&buffers,bool paired,bool tap,MetalBuffer raw={},FlashAffineParams params=q::parameters(),DispatchSize groups={0,0,0}){
  Graph g;g.params=params;g.dispatch.pipelineName=paired?(tap?"raw_q4_rowpair_sep22_candidate_probe":"raw_q4_rowpair_sep22_timed"):(tap?"raw_q4_rowpair_sep22_control_probe":"flash_affine_mlx_qmv_f32xsum_v1_q4_g64");
  for(uint32_t i=0;i<7;++i)g.dispatch.buffers.push_back({i,buffers[i]});if(tap)g.dispatch.buffers.push_back({8,raw});
  g.dispatch.threadgroups=groups.x==0?DispatchSize{1280,paired?2u:4u,1}:groups;g.dispatch.threadsPerThreadgroup={64,1,1};return g;
}
CommandTiming submit(MetalBackend&backend,Graph&g){g.dispatch.bytes={{7,&g.params,sizeof(g.params)}};return backend.submitCommand({&g.dispatch,1});}
void validated(const std::array<MetalBuffer,7>&buffers,const FlashAffineParams&p,bool paired,MetalBuffer raw={},DispatchSize groups={1280,0,1}){
  std::array<q::Span,7>s{};for(size_t i=0;i<7;++i){require(buffers[i]&&buffers[i].storage()==BufferStorage::Shared&&buffers[i].contents(),"rowpair requires actual Shared owned views");s[i]={reinterpret_cast<uintptr_t>(buffers[i].contents()),buffers[i].sizeBytes()};}
  if(groups.y==0)groups.y=paired?2:4;q::validate(p,s,groups.x,groups.y,groups.z,64,paired,{reinterpret_cast<uintptr_t>(raw.contents()),raw.sizeBytes()});
}
struct Result {q::Guarded out,diag,raw,timed;};
void reset(Result&r){r.out.poison();r.raw.poison();r.timed.poison();*static_cast<uint32_t*>(r.diag.view.contents())=sticky;}
bool clean(const Result&r){return r.out.clean()&&r.diag.clean()&&r.raw.clean()&&r.timed.clean();}
std::array<MetalBuffer,7> buffers(MetalBuffer input,const std::array<q::RawSpan,3>&raw,const Result&r,bool timed=false){return{input,raw[0].data.view,raw[1].data.view,raw[2].data.view,input,timed?r.timed.view:r.out.view,r.diag.view};}
struct Comparison {uint64_t words=0,differences=0,rawDifferences=0,nonfinite=0;std::array<double,4>l2{},cos{};bool pass=true;};
Comparison comparison(const Result&a,const Result&b,bool finite){
  Comparison c;c.words=q::kRows*q::kN;
  const auto*x=static_cast<const uint16_t*>(a.out.view.contents());const auto*y=static_cast<const uint16_t*>(b.out.view.contents());
  c.differences=std::memcmp(x,y,q::kOutput)?std::inner_product(x,x+c.words,y,uint64_t{0},std::plus<>(),[](uint16_t u,uint16_t v){return uint64_t(u!=v);}):0;
  for(uint64_t i=0;i<c.words;++i){uint32_t u,v;std::memcpy(&u,static_cast<const uint8_t*>(a.raw.view.contents())+i*4,4);std::memcpy(&v,static_cast<const uint8_t*>(b.raw.view.contents())+i*4,4);c.rawDifferences+=u!=v;}
  c.pass=c.differences==0&&c.rawDifferences==0&&*static_cast<const uint32_t*>(a.diag.view.contents())==*static_cast<const uint32_t*>(b.diag.view.contents())&&clean(a)&&clean(b);
  for(uint32_t r=0;r<4;++r){double e=0,nx=0,ny=0,dot=0;for(uint32_t n=0;n<q::kN;++n){const double u=number(x[uint64_t(r)*q::kN+n]),v=number(y[uint64_t(r)*q::kN+n]);if(!std::isfinite(u)||!std::isfinite(v)){++c.nonfinite;continue;}e+=(u-v)*(u-v);nx+=u*u;ny+=v*v;dot+=u*v;}c.l2[r]=nx?std::sqrt(e/nx):(e?INFINITY:0);c.cos[r]=nx&&ny?dot/std::sqrt(nx*ny):(nx==ny?1:0);if(finite)c.pass&=std::isfinite(c.l2[r])&&c.l2[r]<=1e-4&&c.cos[r]>=.999999;}
  if(finite)c.pass&=c.nonfinite==0;return c;
}
void write(std::ostream&o,const Comparison&c){o<<"{\"pass\":"<<(c.pass?"true":"false")<<",\"bf16_words\":"<<c.words<<",\"bf16_differences\":"<<c.differences<<",\"raw_F32_bit_differences\":"<<c.rawDifferences<<",\"nonfinite_words\":"<<c.nonfinite<<",\"per_row_l2\":[";for(uint32_t i=0;i<4;++i){if(i)o<<',';o<<js(c.l2[i]);}o<<"],\"per_row_cosine\":[";for(uint32_t i=0;i<4;++i){if(i)o<<',';o<<js(c.cos[i]);}o<<"]}";}
void inputPattern(MetalBuffer input,uint32_t mode){auto*x=static_cast<uint16_t*>(input.contents());for(uint32_t r=0;r<4;++r)for(uint32_t k=0;k<q::kK;++k){float v=float(int((k*37+r*13)%257)-128)/128.0f;if(mode==1)v=float(int((k*37)%257)-128)/128.0f;if(mode==2)v=0;if(mode==3)v=(k&1)?-1.0f:1.0f;if(mode==4)v=std::ldexp(v,40);x[uint64_t(r)*q::kK+k]=bf16(v);}if(mode==5)x[0]=0x7fc1;if(mode==6)x[q::kK+17]=0x7f80;if(mode==7)x[2*q::kK+33]=0xff80;if(mode==8)for(uint32_t i=0;i<4*q::kK;++i)x[i]=uint16_t((i%2?0x8000:0)|(1u+(i%127)));if(mode==9)for(uint32_t i=0;i<4*q::kK;++i)x[i]=uint16_t(i%2?0x8000:0);}
struct F64 {uint64_t samples=0,failed=0;double maxRatio=0;};
F64 certificate(const std::array<q::RawSpan,3>&source,MetalBuffer input,MetalBuffer raw){
  F64 f;const auto*x=static_cast<const uint16_t*>(input.contents());const auto*w=static_cast<const uint8_t*>(source[0].data.view.contents());const auto*s=static_cast<const uint16_t*>(source[1].data.view.contents());const auto*b=static_cast<const uint16_t*>(source[2].data.view.contents());const auto*out=static_cast<const float*>(raw.contents());
  for(uint32_t r=0;r<4;++r)for(uint32_t sample=0;sample<128;++sample){const uint32_t n=uint32_t(uint64_t(sample)*(q::kN-1)/127);double sum=0,abs=0,flushed=0;
    for(uint32_t k=0;k<q::kK;++k){const uint32_t code=(w[uint64_t(n)*1280+k/2]>>((k&1)*4))&15;volatile float product=float(code)*number(s[uint64_t(n)*40+k/64]);const float coefficient=product+number(b[uint64_t(n)*40+k/64]);const double xv=number(x[uint64_t(r)*q::kK+k]),v=xv*double(coefficient);sum+=v;abs+=std::abs(v);if(xv!=0&&std::abs(xv)<std::numeric_limits<float>::min())flushed+=std::abs(v);}
    // Unchanged generic benchmark bound, explicitly diagnostic only for the
    // factored QMV producer. It is not a newly widened QMV certificate.
    const double error=std::abs(double(out[uint64_t(r)*q::kN+n])-sum),bound=splash::flash::benchmark::f32DotBound(q::kK,abs,flushed);const double ratio=bound?error/bound:(error?INFINITY:0);++f.samples;f.failed+=!std::isfinite(ratio)||ratio>1;f.maxRatio=std::max(f.maxRatio,ratio);
  }
  return f;
}
uint64_t cpu(){
  require(sizeof(CommandTiming)==200&&sizeof(FlashAffineParams)==64,"rowpair native ABI differs");uint64_t checks=2;
  std::array<q::Span,7>s{};uintptr_t addr=0x100000000ULL;const std::array<uint64_t,7>sizes{q::kInput,q::kWeights,q::kParameters,q::kParameters,q::kInput,q::kOutput,4};for(size_t i=0;i<7;++i){s[i]={addr,sizes[i]};addr+=q::rounded(sizes[i])+16384;}
  const auto p=q::parameters();q::validate(p,s,1280,2,1,64,true);q::validate(p,s,1280,4,1,64,false);checks+=2;
  const auto reject=[&](auto test){bool bad=false;try{test();}catch(const std::invalid_argument&){bad=true;}require(bad,"rowpair CPU negative guard admitted");++checks;};
  for(uint32_t field=0;field<12;++field){auto bad=p;switch(field){case 0:bad.rows=3;break;case 1:bad.selections=2;break;case 2:bad.experts=2;break;case 3:bad.input_size=2559;break;case 4:bad.output_size=10232;break;case 5:bad.bits=5;break;case 6:bad.group_size=32;break;case 7:bad.flags=1;break;case 8:bad.weight_row_stride_bytes=1279;break;case 9:bad.parameter_row_stride_bytes=79;break;case 10:bad.parameter_expert_stride_bytes=1;break;case 11:bad.weight_row_stride_bytes=UINT64_MAX;break;}reject([&]{q::validate(bad,s,1280,2,1,64,true);});}
  for(size_t i=0;i<7;++i){auto bad=s;bad[i].bytes=0;reject([&]{q::validate(p,bad,1280,2,1,64,true);});}
  for(size_t w:{5u,6u})for(size_t r:{0u,1u,2u,3u,4u}){auto bad=s;bad[w].address=bad[r].address;reject([&]{q::validate(p,bad,1280,2,1,64,true);});}
  reject([&]{q::validate(p,s,1280,1,1,64,true);});reject([&]{q::validate(p,s,1279,2,1,64,true);});reject([&]{q::validate(p,s,1281,2,1,64,true);});reject([&]{q::validate(p,s,1280,2,2,64,true);});reject([&]{q::validate(p,s,1280,2,1,128,true);});
  std::vector<uint8_t>owners(uint64_t(q::kRows)*q::kN);for(uint32_t gy=0;gy<2;++gy)for(uint32_t gx=0;gx<1280;++gx)for(uint32_t simd=0;simd<2;++simd)for(uint32_t row=0;row<4;++row)for(uint32_t inputRow=0;inputRow<2;++inputRow)++owners[uint64_t(gy*2+inputRow)*q::kN+gx*8+simd*4+row];for(uint8_t n:owners){require(n==1,"rowpair CPU output ownership differs");++checks;}
  for(uint32_t packed=0;packed<65536;++packed)for(uint32_t j=0;j<4;++j){const uint32_t mask=15u<<(4*j);require((uint16_t(packed)&mask)==(((packed>>(4*j))&15u)<<(4*j)),"rowpair cached integer mask differs");++checks;}
  return checks;
}
std::string pipelineInfo(const char*path){id<MTLDevice>d=MTLCreateSystemDefaultDevice();require(d!=nil,"rowpair Metal missing");NSError*e=nil;id<MTLLibrary>l=[d newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&e];require(l!=nil,"rowpair component library missing");std::ostringstream o;o<<'[';bool comma=false;for(const char*n:{"flash_affine_mlx_qmv_f32xsum_v1_q4_g64","raw_q4_rowpair_sep22_timed","raw_q4_rowpair_sep22_control_probe","raw_q4_rowpair_sep22_candidate_probe"}){id<MTLFunction>f=[l newFunctionWithName:[NSString stringWithUTF8String:n]];require(f!=nil,"rowpair expected entry missing");id<MTLComputePipelineState>p=[d newComputePipelineStateWithFunction:f error:&e];require(p!=nil&&p.threadExecutionWidth==32&&p.maxTotalThreadsPerThreadgroup>=64,"rowpair pipeline execution geometry unsupported");if(comma)o<<',';comma=true;o<<"{\"name\":"<<splash::json::quote(n)<<",\"execution_width\":"<<p.threadExecutionWidth<<",\"maximum_threads\":"<<p.maxTotalThreadsPerThreadgroup<<",\"static_tg_memory_bytes\":"<<p.staticThreadgroupMemoryLength<<'}';}o<<']';return o.str();}
bool gpu(const char*library,const std::filesystem::path&package,const std::filesystem::path&report,const char*inputMeta){
  require(!std::filesystem::exists(report)&&!std::filesystem::exists(report.string()+".partial"),"rowpair report must be fresh");
  const uint64_t libraryBytes=std::filesystem::file_size(library);require(libraryBytes>0&&libraryBytes<(64ULL<<20),"rowpair code library extent invalid");std::vector<uint8_t>libraryData(libraryBytes);std::ifstream libraryFile(library,std::ios::binary);libraryFile.read(reinterpret_cast<char*>(libraryData.data()),std::streamsize(libraryBytes));require(libraryFile.gcount()==std::streamsize(libraryBytes)&&q::hash(libraryData.data(),libraryBytes)==kRawQ4RowpairLibrarySHA,"rowpair frozen component code library differs");
  const Source spec=source(package);const auto metadata=pipelineInfo(library);MetalBackend backend(library);const uint64_t before=backend.memoryStats().allocatedBytes;
  const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);require(physical>reserve,"rowpair host reserve unavailable");splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);const uint64_t admitted=64ULL<<20;auto reservation=governor.tryReserve(admitted);require(bool(reservation),"rowpair selected-source/taps/guards admission denied");
  std::array<q::RawSpan,3>raw;for(uint32_t i=0;i<3;++i)raw[i]=q::RawSpan::load(backend,spec.files[i],spec.offsets[i],spec.lengths[i]);
  auto input=q::Guarded::allocate(backend,q::kInput,"rowpair BF16 R4 original input");
  std::array<Result,2>r;for(auto&a:r){a.out=q::Guarded::allocate(backend,q::kOutput,"rowpair tapped production BF16");a.diag=q::Guarded::allocate(backend,4,"rowpair sticky");a.raw=q::Guarded::allocate(backend,q::kRaw,"rowpair original register F32 taps");a.timed=q::Guarded::allocate(backend,q::kOutput,"rowpair untapped BF16");}
  // Diagnostic-only padded-row/unaligned-byte views. Native immutable spans
  // remain separate and untouched. Every allocation is under the64MiB plan.
  std::array<q::Guarded,3>alternate;
  alternate[0]=q::Guarded::allocate(backend,uint64_t(q::kN)*1284,"rowpair diagnostic padded Q4 rows",1);
  for(uint32_t i=1;i<3;++i)alternate[i]=q::Guarded::allocate(backend,uint64_t(q::kN)*84,"rowpair diagnostic padded BF16 parameter rows",4);
  for(uint32_t n=0;n<q::kN;++n)for(uint32_t i=0;i<3;++i){const uint64_t nativeStride=i?80:1280,testStride=i?84:1284;std::memcpy(static_cast<uint8_t*>(alternate[i].view.contents())+uint64_t(n)*testStride,static_cast<const uint8_t*>(raw[i].data.view.contents())+uint64_t(n)*nativeStride,nativeStride);}
  auto aliasBacking=q::Guarded::allocate(backend,q::kRaw,"rowpair diagnostics-only overlapping view backing");
  bool actual=false;std::string inputScope="deterministic synthetic R4 BF16 projection inputs",inputSHA;
  inputPattern(input.view,0);
  if(inputMeta){NSDictionary*m=json(inputMeta);require(text(m[@"schema"])=="splash-raw-q4-rowpair-current-register-input-v1"&&text(m[@"projection"])=="language_model.model.layers.1.linear_attn.in_proj_qkv"&&integer(m[@"rows"])==4&&integer(m[@"input_size"])==2560&&text(m[@"source_identity_sha256"])=="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e","rowpair actual input identity differs");std::filesystem::path f=text(m[@"file"]);require(std::filesystem::file_size(f)==q::kInput,"rowpair actual input file extent differs");q::RawSpan captured=q::RawSpan::load(backend,f.string(),0,q::kInput);require(captured.sha==text(m[@"payload_sha256"]),"rowpair current register input SHA differs");std::memcpy(input.view.contents(),captured.data.view.contents(),q::kInput);actual=true;inputScope=text(m[@"scope"]);}
  inputSHA=q::hash(input.view.contents(),q::kInput);std::vector<uint8_t>normalInput(q::kInput);std::memcpy(normalInput.data(),input.view.contents(),q::kInput);
  require(backend.memoryStats().allocatedBytes>=before&&backend.memoryStats().allocatedBytes-before<=admitted,"rowpair actual allocations exceed reserved selected-source plan");reservation->commit();const uint64_t allocation=backend.memoryStats().allocatedBytes-before;
  const auto healthy=[&]{require(input.clean()&&clean(r[0])&&clean(r[1]),"rowpair input/result canary changed");for(const auto&a:raw)require(a.immutable(),"rowpair original immutable native source differs");for(const auto&a:alternate)require(a.clean(),"rowpair diagnostic source view canary differs");};
  uint64_t proofChecks=0;bool accepted=true;std::ostringstream proofs;proofs<<'[';bool comma=false;
  const auto check=[&](const std::string&name,bool finite,FlashAffineParams params=q::parameters(),DispatchSize oldGroups={1280,4,1},DispatchSize newGroups={1280,2,1},std::array<MetalBuffer,3>override={}){
    const auto caseInputSHA=q::hash(input.view.contents(),q::kInput);
    std::array<std::string,3>overrideSHA{};
    if(override[0])for(uint32_t i=0;i<3;++i)overrideSHA[i]=q::hash(override[i].contents(),override[i].sizeBytes());
    for(auto&a:r)reset(a);
    const auto selected=[&](uint32_t i,bool timed=false){auto result=buffers(input.view,raw,r[i],timed);if(override[0])for(uint32_t plane=0;plane<3;++plane)result[plane+1]=override[plane];return result;};
    auto a=graph(selected(0),false,true,r[0].raw.view,params,oldGroups),b=graph(selected(1),true,true,r[1].raw.view,params,newGroups);submit(backend,a);submit(backend,b);Comparison c=comparison(r[0],r[1],finite);accepted&=c.pass;++proofChecks;
    bool witnesses=true;
    for(uint32_t i=0;i<2;++i){const uint32_t beforeDiag=*static_cast<const uint32_t*>(r[i].diag.view.contents());*static_cast<uint32_t*>(r[i].diag.view.contents())=sticky;auto w=graph(selected(i,true),i==1,false,{},params,i?newGroups:oldGroups);submit(backend,w);witnesses&=std::memcmp(r[i].out.view.contents(),r[i].timed.view.contents(),q::kOutput)==0&&beforeDiag==*static_cast<const uint32_t*>(r[i].diag.view.contents());}
    accepted&=witnesses;++proofChecks;
    if(comma)proofs<<',';comma=true;proofs<<"{\"case\":"<<splash::json::quote(name)<<",\"shipping_BF16_and_diagnostic_tap_parity\":"<<(witnesses?"true":"false")<<",\"comparison\":";write(proofs,c);proofs<<'}';healthy();require(q::hash(input.view.contents(),q::kInput)==caseInputSHA,"rowpair GPU changed immutable case input");if(override[0])for(uint32_t i=0;i<3;++i)require(q::hash(override[i].contents(),override[i].sizeBytes())==overrideSHA[i],"rowpair GPU changed diagnostic source fixture");return c.pass&&witnesses;
  };
  check(actual?"actual-current-register-input":"synthetic-independent-rows",true);
  const F64 f64=certificate(raw,input.view,r[0].raw.view);
  // Witnesses compare original/candidate tapped production BF16 to SHIPPING
  // untapped bodies. No independently recomputed dot is used as a witness.
  for(uint32_t i=0;i<2;++i){auto b=buffers(input.view,raw,r[i],true);validated(b,q::parameters(),i==1);auto g=graph(b,i==1,false);submit(backend,g);const bool exact=std::memcmp(r[i].out.view.contents(),r[i].timed.view.contents(),q::kOutput)==0;accepted&=exact;++proofChecks;require(exact,"rowpair tap does not reproduce respective shipping output");}
  for(uint32_t mode=1;mode<=9;++mode){inputPattern(input.view,mode);check("synthetic-pattern-"+std::to_string(mode),mode<5||mode>=8);if(mode>=5&&mode<=7)require(*static_cast<const uint32_t*>(r[0].diag.view.contents())&4u,"rowpair original nonfinite input diagnostic missing");}
  for(uint32_t row=0;row<4;++row){constexpr std::array<uint32_t,4>order{2,0,3,1};std::memcpy(static_cast<uint8_t*>(input.view.contents())+uint64_t(row)*q::kK*2,normalInput.data()+uint64_t(order[row])*q::kK*2,q::kK*2);}
  check("permuted-normal-input-rows",true);
  std::memcpy(input.view.contents(),normalInput.data(),q::kInput);
  std::array<MetalBuffer,3>alternateViews{alternate[0].view,alternate[1].view,alternate[2].view};
  const auto padded=q::parameters(1284,84);
  const auto alternateHash=[&]{return std::array<std::string,3>{q::hash(alternate[0].view.contents(),alternate[0].logical),q::hash(alternate[1].view.contents(),alternate[1].logical),q::hash(alternate[2].view.contents(),alternate[2].logical)};};
  const auto altOriginalSHA=alternateHash();
  check("original-strides-padded-and-byte-unaligned-view",true,padded,{1280,4,1},{1280,2,1},alternateViews);
  require(alternateHash()==altOriginalSHA,"rowpair padded source views changed");
  std::array<std::array<uint16_t,40>,2>parameterBackup{};
  for(uint32_t i=0;i<2;++i)std::memcpy(parameterBackup[i].data(),alternate[i+1].view.contents(),80);
  const std::array<std::array<uint16_t,2>,7>faultParameters{{{0,0},{0x8000,0x8000},{0x3e80,0xbe00},{0x0001,0},{0x0080,0},{0x7fc1,0},{0x7f80,0}}};
  for(uint32_t mode=0;mode<faultParameters.size();++mode){for(uint32_t i=0;i<2;++i){auto*p=static_cast<uint16_t*>(alternate[i+1].view.contents());for(uint32_t g=0;g<40;++g)p[g]=faultParameters[mode][i];}check("diagnostic-copy-scale-bias-fault-"+std::to_string(mode),mode<5,padded,{1280,4,1},{1280,2,1},alternateViews);if(mode>=5)require(*static_cast<const uint32_t*>(r[0].diag.view.contents())&4u,"rowpair original nonfinite parameter diagnostic missing");for(uint32_t i=0;i<2;++i)std::memcpy(alternate[i+1].view.contents(),parameterBackup[i].data(),80);require(alternateHash()==altOriginalSHA,"rowpair deliberate source fault did not restore diagnostic copy");}
  // Both producers reject these invalid params before reading any tensor.
  for(uint32_t i=0;i<10;++i){auto p=q::parameters();switch(i){case 0:p.rows=0;break;case 1:p.selections=0;break;case 2:p.experts=0;break;case 3:p.input_size=2559;break;case 4:p.output_size=0;break;case 5:p.bits=5;break;case 6:p.group_size=32;break;case 7:p.flags=4;break;case 8:p.weight_row_stride_bytes=1279;break;case 9:p.parameter_expert_stride_bytes=1;break;}check("gpu-invalid-common-params-"+std::to_string(i),false,p);require(*static_cast<uint32_t*>(r[0].diag.view.contents())==(sticky|2),"rowpair invalid parameter diagnostic differs");}
  check("gpu-extra-groups",true,q::parameters(),{1283,5,2},{1283,3,2});
  check("gpu-partial-row-pair",false,q::parameters(),{1280,2,1},{1280,1,1});
  check("gpu-partial-column-groups",false,q::parameters(),{640,4,1},{640,2,1});
  auto good=buffers(input.view,raw,r[1]);
  for(uint32_t i=0;i<9;++i){auto bad=good;DispatchSize dims{1280,2,1};if(i<7)bad[i]={};if(i==7)bad[5]=bad[0];if(i==8)dims.y=1;bool rejected=false;try{validated(bad,q::parameters(),true,{},dims);}catch(const std::exception&){rejected=true;}require(rejected,"rowpair actual host alias/extent/partial guard admitted");++proofChecks;healthy();}
  // Valid-length aliases exercise the alias check itself, not an earlier
  // short-extent failure. None is submitted to a no-alias Metal entry.
  for(uint32_t w:{5u,6u})for(uint32_t sourceSlot:{0u,1u,2u,3u,4u}){auto bad=good;if(sourceSlot==0||sourceSlot==4){bad[sourceSlot]=backend.view(aliasBacking.base,64+16384,q::kInput);bad[w]=backend.view(aliasBacking.base,w==5?64:64+16384,w==5?q::kOutput:4);}else bad[w]=backend.view(bad[sourceSlot],0,w==5?q::kOutput:4);bool rejected=false;try{validated(bad,q::parameters(),true);}catch(const std::exception&){rejected=true;}require(rejected,"rowpair actual writable/read-only alias guard admitted");++proofChecks;}
  {auto bad=good;bad[6]=backend.view(r[1].out.base,r[1].out.offset,4);bool rejected=false;try{validated(bad,q::parameters(),true);}catch(const std::exception&){rejected=true;}require(rejected,"rowpair actual output/diagnostic alias guard admitted");++proofChecks;}
  {bool rejected=false;try{validated(good,q::parameters(),true,backend.view(raw[0].data.base,raw[0].data.offset,q::kRaw));}catch(const std::exception&){rejected=true;}require(rejected,"rowpair actual F32 tap/native source alias guard admitted");++proofChecks;}
  {auto bad=good;bad[5]=backend.view(r[1].out.base,r[1].out.offset+1,q::kOutput);bool rejected=false;try{validated(bad,q::parameters(),true);}catch(const std::exception&){rejected=true;}require(rejected,"rowpair actual BF16 output misalignment admitted");++proofChecks;}
  healthy();require(aliasBacking.clean(),"rowpair alias-rejection backing canary changed");
  std::memcpy(input.view.contents(),normalInput.data(),q::kInput);check("normal-restored-before-timing",true);healthy();
  std::array<Graph,2>timed{graph(buffers(input.view,raw,r[0],true),false,false),graph(buffers(input.view,raw,r[1],true),true,false)};
  std::array<double,2>warm{};std::array<uint64_t,2>warmCalls{};std::array<std::vector<CommandTiming>,2>times;std::array<std::vector<uint32_t>,2>positions;bool timing=false;
  if(accepted){
    // From this point until all measured positions finish, no CPU tensor,
    // diagnostic, hash, guard or immutable-source access is permitted.
    for(uint32_t i=0;i<2;++i)while(warm[i]<.150){const auto t=submit(backend,timed[i]);require(t.gpuSeconds>0&&std::isfinite(t.gpuSeconds),"rowpair invalid warm GPU timing");warm[i]+=t.gpuSeconds;++warmCalls[i];}
    for(uint32_t pair=0;pair<18;++pair)for(uint32_t pos=0;pos<2;++pos){const uint32_t i=(pair&1)?1-pos:pos;times[i].push_back(submit(backend,timed[i]));positions[i].push_back(pos);}
    timing=true;
    // Payload access resumes only after every warmed/timed position completed.
    healthy();require(q::hash(input.view.contents(),q::kInput)==inputSHA,"rowpair timed input differs");require(std::memcmp(r[0].timed.view.contents(),r[1].timed.view.contents(),q::kOutput)==0,"rowpair timed full BF16 differs");
    // Capture the actual last measured results BEFORE check() resets or
    // resubmits either untapped producer. These snapshots couple post-timing
    // register probes to the real measured calls, rather than new replay calls.
    std::array<std::vector<uint8_t>,2>lastMeasured;
    for(uint32_t i=0;i<2;++i){lastMeasured[i].resize(q::kOutput);std::memcpy(lastMeasured[i].data(),r[i].timed.view.contents(),q::kOutput);}
    check("normal-after-timing",true);
    for(uint32_t i=0;i<2;++i)require(std::memcmp(r[i].out.view.contents(),lastMeasured[i].data(),q::kOutput)==0,"rowpair post-timing probe differs from actual last measured result");
  }
  proofs<<']';
  const auto median=[](const auto&v,bool gpu)->double{std::vector<double>x;for(const auto&t:v)x.push_back((gpu?t.gpuSeconds:t.wallSeconds)*1000);if(x.empty())return std::numeric_limits<double>::quiet_NaN();std::sort(x.begin(),x.end());return(x[8]+x[9])/2;};
  std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":\"splash-raw-q4-rowpair-single-projection-v1\",\"completed\":true,\"pass\":"<<(accepted?"true":"false")<<",\"GPU_executed\":true,\"whole_model_qualified\":false,\"worker_integration\":false,\"scope\":\"one original layer1 QKV R4/K2560/N10240/Q4G64 projection\",\"input_is_actual_current_capture\":"<<(actual?"true":"false")<<",\"input_scope\":"<<splash::json::quote(inputScope)<<",\"input_sha256\":"<<splash::json::quote(inputSHA)<<",\"source_manifest_sha256\":"<<splash::json::quote(spec.manifestSHA)<<",\"native_selected_source_payload_bytes\":14745600,\"admitted_bytes\":"<<admitted<<",\"actual_allocation_bytes\":"<<allocation<<",\"new_normal_workspace_bytes\":0,\"new_weight_cache_bytes\":0,\"pipeline_metadata\":"<<metadata<<",\"cpu_checks\":"<<cpu()<<",\"GPU_proof_checks\":"<<proofChecks<<",\"raw_F32_and_BF16_exact_required\":true,\"per_row_relative_l2_limit\":0.0001,\"per_row_cosine_min\":0.999999,\"F64_diagnostic_only\":true,\"F64_samples\":"<<f64.samples<<",\"F64_diagnostic_bound_failures\":"<<f64.failed<<",\"F64_maximum_bound_ratio\":"<<js(f64.maxRatio)<<",\"F64_diagnostic_policy\":\"independent serial F64 source F32 coefficients; unchanged FlashFloatBoundaryAudit.hpp f32DotBound(K,sumAbs,flushedInputProducts). Sampled diagnostic only: factored raw QMV is not a generic coefficient-dot certificate. Failures remain explicit and do not override raw-F32/BF16 equality or original L2/cosine gates.\",\"timing_executed\":"<<(timing?"true":"false")<<",\"CPU_tensor_access_from_warmup_to_final_timing\":false,\"pairs\":18,\"proofs\":"<<proofs.str()<<",\"routes\":[";
  for(uint32_t i=0;i<2;++i){if(i)out<<',';out<<"{\"name\":"<<splash::json::quote(i?"paired":"original shipping AIR")<<",\"warm_GPU_ms\":"<<warm[i]*1000<<",\"warm_calls\":"<<warmCalls[i]<<",\"median_GPU_ms\":"<<js(median(times[i],true))<<",\"median_wall_ms\":"<<js(median(times[i],false))<<",\"GPU_ms\":[";for(size_t j=0;j<times[i].size();++j){if(j)out<<',';out<<times[i][j].gpuSeconds*1000;}out<<"],\"wall_ms\":[";for(size_t j=0;j<times[i].size();++j){if(j)out<<',';out<<times[i][j].wallSeconds*1000;}out<<"],\"position\":[";for(size_t j=0;j<positions[i].size();++j){if(j)out<<',';out<<positions[i][j];}out<<"]}";}
  out<<"],\"native_span_sha256\":[";for(uint32_t i=0;i<3;++i){if(i)out<<',';out<<splash::json::quote(raw[i].sha);}out<<"],\"provenance\":"<<kRawQ4RowpairProvenance;
  for(auto &g:timed){g.dispatch.buffers.clear();g.dispatch.bytes.clear();}
  for(auto &b:good)b={};
  for(auto &b:alternateViews)b={};
  for(auto &a:alternate)a={};
  aliasBacking={};
  for(auto &a:r){a.out={};a.diag={};a.raw={};a.timed={};}
  for(auto &a:raw){a.data={};a.file.reset();}
  input={};
  const uint64_t after=backend.memoryStats().allocatedBytes;
  require(after==before,"rowpair selected buffers did not tear down cleanly");
  out<<",\"teardown_allocated_bytes\":"<<after<<",\"teardown_clean\":true}";
  std::ofstream file(report.string()+".partial",std::ios::binary);file<<out.str()<<'\n';file.close();require(bool(file),"rowpair report write failed");std::filesystem::rename(report.string()+".partial",report);std::cout<<"{\"pass\":"<<(accepted?"true":"false")<<",\"timing_executed\":"<<(timing?"true":"false")<<",\"report\":"<<splash::json::quote(report.string())<<"}\n";return accepted;
}
} // namespace
int main(int argc,char**argv){
  try{
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){const auto n=cpu();std::cout<<"{\"pass\":true,\"checks\":"<<n<<",\"GPU_work\":false,\"payload_reads\":0,\"metadata_reads\":0,\"device_created\":false}\n";return 0;}
    if(argc==2&&std::string_view(argv[1])=="--help"){std::cout<<"oracle --cpu-self-test\noracle --gpu COMPONENT_METALLIB ORIGINAL_PACKAGE NEW_REPORT_JSON [CURRENT_INPUT_METADATA_JSON]\nGPU mode is Root-only. Default input is explicitly synthetic.\n";return 0;}
    require((argc==5||argc==6)&&std::string_view(argv[1])=="--gpu","explicit Root --gpu required");@autoreleasepool{return gpu(argv[2],argv[3],argv[4],argc==6?argv[5]:nullptr)?0:2;}
  }catch(const std::exception&e){
    std::cerr<<e.what()<<'\n';
    if(argc>=5&&std::string_view(argv[1])=="--gpu"){
      const std::filesystem::path failure=std::string(argv[4])+".failure.json";
      if(!std::filesystem::exists(argv[4])&&!std::filesystem::exists(failure)){
        std::ofstream f(failure,std::ios::binary);
        f<<"{\"schema\":\"splash-raw-q4-rowpair-GPU-mode-failure-v1\",\"completed\":false,\"pass\":false,\"GPU_mode_requested\":true,\"GPU_execution_complete\":false,\"earlier_GPU_or_payload_milestones_not_asserted\":true,\"worker_integration\":false,\"reason\":"<<splash::json::quote(e.what())<<",\"provenance\":"<<kRawQ4RowpairProvenance<<"}\n";
      }
    }
    return 1;
  }
}
