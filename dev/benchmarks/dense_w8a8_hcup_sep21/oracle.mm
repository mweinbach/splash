// Root-only standalone component oracle. --preview/--cpu-self-test create no GPU
// and read no payloads. --run reads only activated input, normalized plane and
// original F32 coefficients for the single pinned layer0 HC-UP role.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "metal/CommandGraph.hpp"
#include "engine/Json.hpp"
#include "abi.hpp"
#include "quantization.hpp"
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace {
using namespace splash::metal;
namespace q = splash::dense_w8a8;
namespace hq = splash::hcup_w8a8;
constexpr uint32_t R=2048,K=320,N=10240,H=2560;
constexpr uint64_t guard=64;
constexpr uint16_t sentinel=0x7fc1;
constexpr uint32_t sticky=0x40000000, floatGuard=0x7fc10001, integerGuard=0x6e7152a3;
constexpr double l2Limit=.02,cosineLimit=.9998;
const std::string role="language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up";
const std::string downRole="language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down";
const std::string sourcePin="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
const std::string manifestPin="edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0";
const std::string f32Pin="86c57dda081aebfdd7642a7f23496b6b45ecb996a45dbf1e74df9d2d3c8030da";
const std::string bf16Pin="2403b3e90d9452756dbaaac29da866510babc95a4027146d7f34896760ad92bb";
const std::string inputPin="a31906c1fdc06155147b07b6658b26ae7e1fa246b861e0fb0304c4d408d2846f";
const std::string normPin="fb402a313a5f18053671675c43608111370334ed6113e16dfda11d2916963b69";
const std::string rawPin="0091bfea9c293cc9d97898fe1c84f40f771b8428864b767c4263b1cdb04b5787";
void require(bool ok,const std::string &message) { if(!ok) throw std::runtime_error(message); }
float number(uint16_t value) { return q::bf16Number(value); }
uint16_t bf16(float value) { return q::bf16Bits(value); }
std::string hash(const void *p,uint64_t bytes) {
  require(bytes<=UINT32_MAX,"SHA256 bounded extent exceeded");
  unsigned char digest[CC_SHA256_DIGEST_LENGTH]; require(CC_SHA256(p,CC_LONG(bytes),digest),"SHA256 failed");
  std::ostringstream out; for(auto v:digest) out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(v); return out.str();
}
void readExact(const std::string &path,const MetalBuffer &buffer,const std::string &expected) {
  require(std::filesystem::file_size(path)==buffer.sizeBytes(),"payload extent mismatch: "+path);
  std::ifstream file(path,std::ios::binary); require(bool(file),"payload open failed: "+path);
  file.read(static_cast<char *>(buffer.contents()),std::streamsize(buffer.sizeBytes())); require(bool(file),"payload read failed");
  require(hash(buffer.contents(),buffer.sizeBytes())==expected,"payload SHA256 mismatch: "+path);
}
NSDictionary *json(const std::string &path) {
  NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data!=nil,"metadata open failed: "+path); NSError *error=nil;
  id object=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(!error&&[object isKindOfClass:[NSDictionary class]],"invalid metadata JSON"); return object;
}
std::string string(NSDictionary *object,NSString *key) {
  id value=object[key]; require([value isKindOfClass:[NSString class]],"missing metadata string"); return [value UTF8String];
}
uint64_t integer(NSDictionary *object,NSString *key) {
  id value=object[key]; require([value isKindOfClass:[NSNumber class]],"missing metadata integer");
  const double d=[value doubleValue]; require(d>=0&&d<=double(UINT64_MAX)&&d==std::floor(d),"invalid metadata integer"); return [value unsignedLongLongValue];
}
NSDictionary *entry(NSDictionary *document,NSString *key,const std::string &projection,const std::string &format="") {
  id values=document[key]; require([values isKindOfClass:[NSArray class]],"missing metadata entries");
  NSDictionary *result=nil;
  for(id value:values) {
    require([value isKindOfClass:[NSDictionary class]],"invalid metadata entry");
    if(string(value,@"projection")==projection&&(format.empty()||string(value,@"format")==format)) {
      require(!result,"duplicate metadata role/format"); result=value;
    }
  }
  require(result!=nil,"missing pinned metadata role/format"); return result;
}
struct Fixture { std::string inputPath,normPath,f32Path; };
Fixture metadata(const std::string &capturePath,const std::string &storePath) {
  NSDictionary *capture=json(capturePath),*store=json(storePath);
  require(string(capture,@"schema")=="splash-prefill4k-dense-fixtures-v1"&&[capture[@"actual_activations"] boolValue],"requires actual captured activations");
  require(string(store,@"schema")=="splash-local-affine-operands-v1"&&integer(store,@"alignment_bytes")==16384,"operand store schema/alignment mismatch");
  require(string(store,@"source_identity_sha256")==sourcePin&&string(store,@"weights_manifest_fingerprint")==manifestPin,"model source identity mismatch");
  require(string(store,@"math_version_sha256")=="533b04ed7539612cddb5c74bd90d492ba5135ba286e912e9b1dc914d97f85ef6","operand math version mismatch");
  NSDictionary *up=entry(capture,@"cases",role),*down=entry(capture,@"cases",downRole),*f32=entry(store,@"entries",role,"F32");
  require(integer(up,@"rows")==R&&integer(up,@"input_size")==K&&integer(up,@"output_size")==N&&
      integer(down,@"rows")==R&&integer(down,@"input_size")==N&&integer(down,@"output_size")==K,"HC capture shape mismatch");
  require(string(up,@"input_sha256")==inputPin&&string(down,@"input_sha256")==normPin&&
      string(up,@"weights_sha256")==bf16Pin&&string(up,@"expected_sha256")==rawPin,"captured HC sequence identities mismatch");
  id shape=f32[@"shape"]; require([shape isKindOfClass:[NSArray class]]&&[shape count]==2&&[shape[0] unsignedLongLongValue]==N&&[shape[1] unsignedLongLongValue]==K,"F32 coefficient shape mismatch");
  require(string(f32,@"operand_math")=="original-affine-contractoff-f32-coefficients-row-major-v1"&&
      integer(f32,@"logical_bytes")==uint64_t(N)*K*4&&integer(f32,@"allocated_bytes")==uint64_t(N)*K*4&&
      integer(f32,@"offset_bytes")==0&&string(f32,@"payload_sha256")==f32Pin&&string(f32,@"file")=="operand-512.bin","F32 coefficient type/math/extent identity mismatch");
  NSDictionary *source=f32[@"source"]; require([source isKindOfClass:[NSDictionary class]],"missing F32 coefficient source geometry");
  require(integer(source,@"experts")==1&&integer(source,@"bits")==5&&integer(source,@"group_size")==64&&
      integer(source,@"input_size")==K&&integer(source,@"output_size")==N&&integer(source,@"weight_row_stride_bytes")==200&&
      integer(source,@"weight_expert_stride_bytes")==2048000&&integer(source,@"parameter_row_stride_bytes")==10&&
      integer(source,@"parameter_expert_stride_bytes")==102400,"F32 original affine source geometry mismatch");
  return {string(up,@"input_file"),string(down,@"input_file"),(std::filesystem::path(storePath).parent_path()/"operand-512.bin").string()};
}
struct Error {
  uint64_t elements=0,mismatches=0,nonfinite=0; double squared=0,reference=0,actual=0,product=0,maxAbs=0;
  void add(double a,double b,bool same=true) {
    ++elements; mismatches+=!same; if(!std::isfinite(a)||!std::isfinite(b)){++nonfinite;return;}
    const double d=a-b; squared+=d*d; reference+=b*b; actual+=a*a; product+=a*b; maxAbs=std::max(maxAbs,std::abs(d));
  }
  double l2()const { return reference?std::sqrt(squared/reference):(squared?INFINITY:0); }
  double cosine()const { return actual&&reference?std::clamp(product/std::sqrt(actual*reference),-1.,1.):actual==reference?1:0; }
  bool passed()const { return !nonfinite&&l2()<=l2Limit&&cosine()>=cosineLimit; }
  void write(std::ostream &out)const {
    out<<"{\"elements\":"<<elements<<",\"bf16_mismatches\":"<<mismatches<<",\"nonfinite\":"<<nonfinite
       <<",\"relative_l2\":"<<l2()<<",\"cosine\":"<<cosine()<<",\"max_abs\":"<<maxAbs<<'}';
  }
};
std::vector<uint16_t> words(const MetalBuffer &buffer) { const auto *p=static_cast<const uint16_t *>(buffer.contents()); return {p,p+buffer.sizeBytes()/2}; }
Error compare(const std::vector<uint16_t> &a,const std::vector<uint16_t> &b) {
  require(a.size()==b.size(),"comparison extent mismatch"); Error e;
  for(uint64_t i=0;i<a.size();++i)e.add(number(a[i]),number(b[i]),a[i]==b[i]); return e;
}
struct Operands {
  MetalBuffer input,norm,f32,weights,qaBase,qa,qwBase,qw,saBase,sa,swBase,sw,integerBase,integerDot;
  q::QuantError coefficientError; uint64_t f32Subnormal=0,bf16Subnormal=0,inputSubnormal=0,normSubnormal=0;
  std::vector<std::pair<MetalBuffer,std::string>> immutable;
  static MetalBuffer guarded(MetalBackend &backend,MetalBuffer &base,uint64_t bytes,uint32_t width) {
    base=backend.allocateBuffer(bytes+2*guard*width,BufferStorage::Shared);return backend.view(base,guard*width,bytes);
  }
  Operands(MetalBackend &backend,const Fixture &fixture) {
    input=backend.allocateBuffer(uint64_t(R)*K*2,BufferStorage::Shared);norm=backend.allocateBuffer(uint64_t(R)*N*2,BufferStorage::Shared);
    f32=backend.allocateBuffer(uint64_t(N)*K*4,BufferStorage::Shared);weights=backend.allocateBuffer(uint64_t(N)*K*2,BufferStorage::Shared);
    qa=guarded(backend,qaBase,uint64_t(R)*K,1);qw=guarded(backend,qwBase,uint64_t(N)*K,1);
    sa=guarded(backend,saBase,uint64_t(R)*4,4);sw=guarded(backend,swBase,uint64_t(N)*4,4);
    integerDot=guarded(backend,integerBase,uint64_t(R)*N*4,4);
    readExact(fixture.inputPath,input,inputPin);readExact(fixture.normPath,norm,normPin);readExact(fixture.f32Path,f32,f32Pin);
    const auto *source=static_cast<const float *>(f32.contents());auto *derived=static_cast<uint16_t *>(weights.contents());
    for(uint64_t i=0;i<uint64_t(N)*K;++i) {
      require(std::isfinite(source[i]),"nonfinite original F32 coefficient");derived[i]=bf16(source[i]);
      f32Subnormal+=std::fpclassify(source[i])==FP_SUBNORMAL;bf16Subnormal+=(derived[i]&0x7f80)==0&&(derived[i]&0x7f);
    }
    require(hash(weights.contents(),weights.sizeBytes())==bf16Pin,"derived BF16 worker coefficients differ from captured coefficient identity");
    std::memset(qaBase.contents(),0x80,qaBase.sizeBytes());std::memset(qwBase.contents(),0x80,qwBase.sizeBytes());
    std::fill_n(static_cast<uint32_t *>(saBase.contents()),saBase.sizeBytes()/4,floatGuard);
    std::fill_n(static_cast<uint32_t *>(swBase.contents()),swBase.sizeBytes()/4,floatGuard);
    std::fill_n(static_cast<uint32_t *>(integerBase.contents()),integerBase.sizeBytes()/4,integerGuard);
    auto *codes=static_cast<int8_t *>(qw.contents());auto *scales=static_cast<float *>(sw.contents());
    for(uint32_t row=0;row<N;++row)hq::quantizeF32(source+uint64_t(row)*K,K,codes+uint64_t(row)*K,scales[row],coefficientError);
    const auto check=[&](const MetalBuffer &b,uint64_t &count){for(auto value:words(b)){require(std::isfinite(number(value)),"nonfinite captured activation");count+=(value&0x7f80)==0&&(value&0x7f);}};
    check(input,inputSubnormal);check(norm,normSubnormal);const auto *x=static_cast<const uint16_t *>(input.contents());
    for(uint32_t row=0;row<R;++row)(void)q::symmetricScale(x+uint64_t(row)*K,K);
    for(const auto &b:{input,norm,f32,weights,qw,sw})immutable.emplace_back(b,hash(b.contents(),b.sizeBytes()));
  }
  static void canary(const MetalBuffer &base,uint64_t count,uint32_t width,uint32_t value) {
    for(uint64_t i=0;i<guard;++i){bool valid;
      if(width==1){const auto *p=static_cast<const uint8_t *>(base.contents());valid=p[i]==value&&p[guard+count+i]==value;}
      else if(width==2){const auto *p=static_cast<const uint16_t *>(base.contents());valid=p[i]==value&&p[guard+count+i]==value;}
      else{const auto *p=static_cast<const uint32_t *>(base.contents());valid=p[i]==value&&p[guard+count+i]==value;}
      require(valid,"operand/output guard changed");}
  }
  void guards()const {canary(qaBase,qa.sizeBytes(),1,0x80);canary(qwBase,qw.sizeBytes(),1,0x80);canary(saBase,R,4,floatGuard);canary(swBase,N,4,floatGuard);canary(integerBase,uint64_t(R)*N,4,integerGuard);}
  void unchanged()const {for(const auto &[b,digest]:immutable)require(hash(b.contents(),b.sizeBytes())==digest,"immutable source/coefficients/scales changed");}
};
struct Output {
  MetalBuffer rawBase,raw,mixedBase,mixed,diagnostics,gatesBase,gates,productsBase,products,sumsBase,sums;
  explicit Output(MetalBackend &backend) {
    raw=Operands::guarded(backend,rawBase,uint64_t(R)*N*2,2);mixed=Operands::guarded(backend,mixedBase,uint64_t(R)*H*2,2);
    diagnostics=backend.allocateBuffer(64,BufferStorage::Shared);reset();
  }
  void stages(MetalBackend &backend) {
    gates=Operands::guarded(backend,gatesBase,uint64_t(R)*N*2,2);products=Operands::guarded(backend,productsBase,uint64_t(R)*N*2,2);sums=Operands::guarded(backend,sumsBase,uint64_t(R)*N*2,2);
    for(auto &b:{gatesBase,productsBase,sumsBase})std::fill_n(static_cast<uint16_t *>(b.contents()),b.sizeBytes()/2,sentinel);
  }
  void reset(){for(auto &b:{rawBase,mixedBase})std::fill_n(static_cast<uint16_t *>(b.contents()),b.sizeBytes()/2,sentinel);std::memset(diagnostics.contents(),0,diagnostics.sizeBytes());*static_cast<uint32_t *>(diagnostics.contents())=sticky;}
  void guards()const {
    Operands::canary(rawBase,uint64_t(R)*N,2,sentinel);Operands::canary(mixedBase,uint64_t(R)*H,2,sentinel);
    if(gates)for(const auto &b:{gatesBase,productsBase,sumsBase})Operands::canary(b,uint64_t(R)*N,2,sentinel);
    require(*static_cast<const uint32_t *>(diagnostics.contents())==sticky,"shader diagnostics changed");
  }
};
struct Variant {
  uint32_t groups=0;std::string name;std::array<Error,5> quality;Error sourceFP64,dequantizedFP64,quantizationFP64;
  uint64_t integerSamples=0,identityElements=0,scaleElements=0,codeElements=0,lateScaleSubnormalEnvelope=0;
  bool passed=false;std::vector<uint16_t> raw,mixed;std::vector<double> gpu,wall,baseGPU,baseWall,speedup;
  uint32_t ab=0,ba=0,warmAB=0,warmBA=0;double warmGPU=0,warmBaseGPU=0;
  std::vector<std::array<uint32_t,2>> positions;
};
CommandGraph graph(const Operands &o,const Output &b,uint32_t groups,uint32_t repeat,bool probe=false) {
  CommandGraph g;for(uint32_t r=0;r<repeat;++r){
    if(groups){
      g.add("dense_w8a8_bf16_to_i8_t256",{o.input,o.qa,o.sa,b.diagnostics},DenseW8A8QuantizeParams{R,K},{R,1,1},{256,1,1});
      g.add("dense_w8a8_m128_n64_sg"+std::to_string(groups)+(probe?"_probe":""),{o.qa,o.qw,b.raw,o.integerDot,o.sa,o.sw,b.diagnostics},FlashDenseCacheParams{R,K,N,0,N,128,64,0},{N/64,R/128,1},{groups*32,1,1});
    }else g.add("flash_dense_cache_m32_n128",{o.input,o.weights,b.raw,b.diagnostics},FlashDenseCacheParams{R,K,N,0,N,32,128,0},{N/128,R/32,1},{128,1,1});
    const FlashHCParams p{R,H,4,0,1e-6f,0,0,0};
    if(probe)g.add("hcup_post_probe",{o.norm,b.raw,b.mixed,b.gates,b.products,b.sums,b.diagnostics},p,{10,R,1},{256,1,1});
    else g.add("flash_hc_mix",{o.norm,b.raw,b.mixed},p,{40,R,1},{64,1,1});
  }return g;
}
void certify(Variant &v,const Operands &o) {
  const auto *x=static_cast<const uint16_t *>(o.input.contents());const auto *w=static_cast<const float *>(o.f32.contents());
  const auto *a=static_cast<const int8_t *>(o.qa.contents()),*b=static_cast<const int8_t *>(o.qw.contents());
  const auto *sa=static_cast<const float *>(o.sa.contents()),*sw=static_cast<const float *>(o.sw.contents());const auto *dots=static_cast<const int32_t *>(o.integerDot.contents());
  for(uint32_t row=0;row<R;++row){
    require(std::bit_cast<uint32_t>(sa[row])==std::bit_cast<uint32_t>(q::symmetricScale(x+uint64_t(row)*K,K)),"GPU activation scale differs from CPU rowmax");++v.scaleElements;
    for(uint32_t k=0;k<K;++k){require(a[uint64_t(row)*K+k]==q::symmetricCode(number(x[uint64_t(row)*K+k]),sa[row])&&a[uint64_t(row)*K+k]!=-128,"GPU activation RNE I8 code differs from CPU");++v.codeElements;}
  }
  for(uint32_t row=0;row<R;++row)for(uint32_t column=0;column<N;++column){
    const uint64_t at=uint64_t(row)*N+column;require(std::abs(int64_t(dots[at]))<=int64_t(K)*16129,"I32 dot exceeds certified bound");
    const float first=float(dots[at])*sa[row],scaled=first*sw[column];const uint16_t expected=bf16(scaled);++v.identityElements;
    if(v.raw[at]!=expected){
      // Explicit FTZ envelope: only a truly subnormal F32 intermediate may
      // account for a zero result. Normal values still require byte identity.
      const bool subnormal=std::fpclassify(first)==FP_SUBNORMAL||std::fpclassify(scaled)==FP_SUBNORMAL;
      require(subnormal&&(v.raw[at]&0x7fff)==0&&std::abs(double(number(expected)))<=double(std::numeric_limits<float>::min()),"full late-scale BF16 identity failed outside stated FTZ envelope");++v.lateScaleSubnormalEnvelope;
    }
  }
  for(uint32_t ri=0;ri<8;++ri)for(uint32_t ci=0;ci<32;++ci){
    const uint32_t row=uint32_t(uint64_t(ri)*(R-1)/7),column=uint32_t(uint64_t(ci)*(N-1)/31);int64_t exact=0;double source=0,dequantized=0,absolute=0;
    for(uint32_t k=0;k<K;++k){const uint64_t ai=uint64_t(row)*K+k,bi=uint64_t(column)*K+k;exact+=int64_t(a[ai])*b[bi];source+=double(number(x[ai]))*w[bi];dequantized+=(double(a[ai])*sa[row])*(double(b[bi])*sw[column]);absolute+=std::abs((double(a[ai])*sa[row])*(double(b[bi])*sw[column]));}
    require(exact==dots[uint64_t(row)*N+column],"sampled exact I32 dot failed");++v.integerSamples;
    const double late=double(exact)*sa[row]*sw[column];require(std::abs(dequantized-late)<=1e-12*std::max(1.,absolute),"FP64 dequantized decomposition failed");
    const float first=float(exact)*sa[row],scaled=first*sw[column];const double u=std::numeric_limits<float>::epsilon()/2,gamma=3*u/(1-3*u);
    const double envelope=gamma*absolute+std::abs(double(scaled)-double(number(bf16(scaled))))+3*std::numeric_limits<float>::min();
    require(std::abs(double(number(v.raw[uint64_t(row)*N+column]))-late)<=envelope,"sampled late-scale FP64 rounding envelope failed");
    const double observed=number(v.raw[uint64_t(row)*N+column]);v.sourceFP64.add(observed,source);v.dequantizedFP64.add(observed,dequantized);v.quantizationFP64.add(dequantized,source);
  }
}
double median(std::vector<double> v){require(!v.empty(),"empty timing vector");std::sort(v.begin(),v.end());const auto i=v.size()/2;return v.size()%2?v[i]:(v[i-1]+v[i])/2;}
void array(std::ostream &out,const std::vector<double> &v){out<<'[';for(size_t i=0;i<v.size();++i){if(i)out<<',';out<<v[i];}out<<']';}
void cpuTest(){
  uint64_t checks=0;for(uint32_t w=0;w<65536;++w)if(std::isfinite(number(uint16_t(w)))){require(bf16(number(uint16_t(w)))==w,"finite BF16 roundtrip failed");++checks;}
  float source[]{1.0f,2.4999f/127.0f,2.5001f/127.0f};int8_t codes[3];float scale;q::QuantError e;hq::quantizeF32(source,3,codes,scale,e);
  require(codes[1]==2&&codes[2]==3&&bf16(source[1])==bf16(source[2]),"direct-F32 fit silently lost coefficient precision");++checks;
  require(uint64_t(K)*16129==5161280&&uint64_t(K)*16129<INT32_MAX,"I32 bound failed");++checks;
  std::cout<<"{\"cpu_self_test\":\"passed\",\"checks\":"<<checks<<",\"gpu_created\":false,\"payloads_read\":false}\n";
}
}
int main(int argc,char **argv){try{
  bool run=false,preview=false;std::string library="build/dense-w8a8-hcup-sep21/hcup.metallib",capture="build/prefill4k-dense/actual-activations-v1/manifest.json",store="install/local-models/Flash-Next-operands-v1/manifest.json",outPath;
  uint32_t requested=10,repeat=4;double warm=150;std::vector<uint32_t> groups{4,8};
  for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--cpu-self-test"){cpuTest();return 0;}if(arg=="--run"){run=true;continue;}if(arg=="--preview"){preview=true;continue;}
    require(i+1<argc,"missing CLI value");const std::string value=argv[++i];if(arg=="--library")library=value;else if(arg=="--capture-manifest")capture=value;else if(arg=="--operand-manifest")store=value;else if(arg=="--out")outPath=value;
    else if(arg=="--samples")requested=uint32_t(std::stoul(value));else if(arg=="--repeat")repeat=uint32_t(std::stoul(value));else if(arg=="--warm-ms")warm=std::stod(value);else if(arg=="--groups"){groups.clear();std::stringstream s(value);std::string word;while(std::getline(s,word,',')){const auto g=std::stoul(word);require(g==4||g==8,"groups must be 4 or 8");groups.push_back(uint32_t(g));}std::sort(groups.begin(),groups.end());groups.erase(std::unique(groups.begin(),groups.end()),groups.end());}else throw std::runtime_error("unknown CLI argument: "+arg);
  }
  require(run||preview,"payload/GPU work requires explicit --run");require(!groups.empty()&&requested>=2&&requested<=100&&repeat&&repeat<=64&&std::isfinite(warm)&&warm>=150&&warm<=1000,"invalid groups/sample/repeat/warm controls");
  const Fixture fixture=metadata(capture,store);const uint32_t count=uint32_t(groups.size()),strata=2*count,samples=(requested+strata-1)/strata*strata;
  if(preview){std::cout<<"{\"preview_only\":true,\"gpu_created\":false,\"payloads_read\":false,\"numerical_alternative\":true,\"model_qualified\":false,\"projection\":"<<splash::json::quote(role)<<",\"coefficient_source\":\"F32_original_operand512_direct_I8_fit\",\"actual_control\":\"BF16_cached_m32n128_sg4_wholeK_plus_actual_HC_mix\",\"effective_balanced_samples\":"<<samples<<",\"payload_read_count\":3}\n";return 0;}
  MetalBackend backend(library);Operands operands(backend,fixture);std::vector<Variant> variants;std::vector<Output> outputs;
  Variant control;control.name="actual_bf16_m32n128_sg4_plus_hc_mix";variants.push_back(control);for(auto g:groups){Variant v;v.groups=g;v.name="f32_fitted_w8a8_m128n64_sg"+std::to_string(g)+"_plus_hc_mix";variants.push_back(v);}for(size_t i=0;i<variants.size();++i)outputs.emplace_back(backend);
  std::array<std::vector<uint16_t>,3> baselineStages;
  for(size_t i=0;i<variants.size();++i){auto &v=variants[i];auto &b=outputs[i];auto normal=graph(operands,b,v.groups,1);(void)backend.submitCommand(normal.dispatches());b.guards();operands.guards();v.raw=words(b.raw);v.mixed=words(b.mixed);
    b.reset();(void)backend.submitCommand(normal.dispatches());b.guards();require(!compare(words(b.raw),v.raw).mismatches&&!compare(words(b.mixed),v.mixed).mismatches,"deterministic complete output changed");
    if(!i)require(hash(v.raw.data(),v.raw.size()*2)==rawPin,"actual worker raw control differs from complete captured output SHA");
    b.stages(backend);auto probe=graph(operands,b,v.groups,1,true);b.reset();(void)backend.submitCommand(probe.dispatches());b.guards();operands.guards();
    require(!compare(words(b.raw),v.raw).mismatches&&!compare(words(b.mixed),v.mixed).mismatches,"integer/stage probe changed normal complete output");
    const std::array<MetalBuffer,3> stageBuffers{b.gates,b.products,b.sums};if(!i)for(size_t j=0;j<3;++j)baselineStages[j]=words(stageBuffers[j]);
    v.quality[0]=compare(v.raw,variants[0].raw);for(size_t j=0;j<3;++j)v.quality[j+1]=compare(words(stageBuffers[j]),baselineStages[j]);v.quality[4]=compare(v.mixed,variants[0].mixed);
    v.passed=std::all_of(v.quality.begin(),v.quality.end(),[](const Error &e){return e.passed();});if(i)certify(v,operands);v.positions.resize(count);
  }
  const std::string integerBefore=hash(operands.integerDot.contents(),operands.integerDot.sizeBytes());
  const std::string activationBefore=hash(operands.qa.contents(),operands.qa.sizeBytes()),scaleBefore=hash(operands.sa.contents(),operands.sa.sizeBytes());
  std::vector<CommandGraph> commands;for(size_t i=0;i<variants.size();++i)commands.push_back(graph(operands,outputs[i],variants[i].groups,repeat));
  // No CPU payload, output, probe, diagnostics or guard access during GPU warm
  // or timings. AB/BA pairs and candidate positions are fully balanced.
  for(size_t i=1;i<variants.size();++i){auto &v=variants[i];uint32_t rounds=0;while(v.warmGPU<warm||v.warmBaseGPU<warm){require(++rounds<=10000,"GPU warm timestamps unavailable");
    const auto a=backend.submitCommand(commands[0].dispatches()),b=backend.submitCommand(commands[i].dispatches()),c=backend.submitCommand(commands[i].dispatches()),d=backend.submitCommand(commands[0].dispatches());
    require(a.gpuSeconds>0&&b.gpuSeconds>0&&c.gpuSeconds>0&&d.gpuSeconds>0,"GPU timestamps unavailable");v.warmGPU+=(b.gpuSeconds+c.gpuSeconds)*1000;v.warmBaseGPU+=(a.gpuSeconds+d.gpuSeconds)*1000;++v.warmAB;++v.warmBA;
  }}
  for(uint32_t sample=0;sample<samples;++sample)for(uint32_t position=0;position<count;++position){const size_t i=1+(position+(sample/2)%count)%count;auto &v=variants[i];CommandTiming a,b;
    if(!(sample%2)){a=backend.submitCommand(commands[0].dispatches());b=backend.submitCommand(commands[i].dispatches());++v.ab;}else{b=backend.submitCommand(commands[i].dispatches());a=backend.submitCommand(commands[0].dispatches());++v.ba;}++v.positions[position][sample%2];
    require(a.gpuSeconds>0&&b.gpuSeconds>0,"GPU timestamp missing in timed pair");v.gpu.push_back(b.gpuSeconds*1000/repeat);v.wall.push_back(b.wallSeconds*1000/repeat);v.baseGPU.push_back(a.gpuSeconds*1000/repeat);v.baseWall.push_back(a.wallSeconds*1000/repeat);v.speedup.push_back(a.gpuSeconds/b.gpuSeconds);
  }
  for(size_t i=0;i<variants.size();++i){auto &v=variants[i];outputs[i].guards();require(!compare(words(outputs[i].raw),v.raw).mismatches&&!compare(words(outputs[i].mixed),v.mixed).mismatches,"timed complete output changed");if(i){require(v.ab==samples/2&&v.ba==samples/2,"AB/BA balance failed");for(auto p:v.positions)require(p[0]==samples/strata&&p[1]==samples/strata,"position/order balance failed");}}
  operands.guards();operands.unchanged();require(hash(operands.integerDot.contents(),operands.integerDot.sizeBytes())==integerBefore,"normal timing wrote untimed I32 probe");
  require(hash(operands.qa.contents(),operands.qa.sizeBytes())==activationBefore&&hash(operands.sa.contents(),operands.sa.sizeBytes())==scaleBefore,"timed activation codes/scales differ from complete CPU-certified conversion");
  std::ostringstream report;report<<std::setprecision(12)<<"{\"experiment\":\"dense_w8a8_hcup_sep21_v1\",\"numerical_alternative\":true,\"model_qualified\":false,\"production_overlay\":false,\"payload_read_count\":3,\"derived_bf16_coefficients_sha_exact\":true,\"captured_raw_control_sha_exact\":true,\"guards_passed\":true,\"immutable_sources_coefficients_passed\":true,\"timed_probe_writes\":false,\"activation_converter_and_HC_mix_in_timing\":true,\"cpu_payload_access_in_warm_or_timing\":false,\"rows\":"<<R<<",\"input_size\":"<<K<<",\"output_size\":"<<N<<",\"integer_magnitude_bound\":"<<uint64_t(K)*16129<<",\"effective_balanced_samples\":"<<samples<<",\"repeat\":"<<repeat<<",\"quality_preregister\":{\"full_raw_gate_product_streamsum_mixed_relative_l2_max\":"<<l2Limit<<",\"cosine_min\":"<<cosineLimit<<"},\"subnormal_counts\":{\"source_f32_coefficients\":"<<operands.f32Subnormal<<",\"derived_bf16_coefficients\":"<<operands.bf16Subnormal<<",\"activated_input\":"<<operands.inputSubnormal<<",\"normalized_hyper\":"<<operands.normSubnormal<<"},\"source_f32_coefficient_dequantization_error\":";operands.coefficientError.writeJSON(report);report<<",\"variants\":[";
  const std::array<std::string,5> names{"raw_dot","gate","product","sequential_stream_sum","mixed"};
  for(size_t i=1;i<variants.size();++i){const auto &v=variants[i];if(i>1)report<<',';report<<"{\"name\":"<<splash::json::quote(v.name)<<",\"component_quality_passed\":"<<(v.passed?"true":"false")<<",\"median_complete_gpu_ms\":"<<median(v.gpu)<<",\"median_complete_wall_ms\":"<<median(v.wall)<<",\"paired_control_gpu_ms\":"<<median(v.baseGPU)<<",\"paired_median_speedup\":"<<median(v.speedup)<<",\"warm_candidate_gpu_ms\":"<<v.warmGPU<<",\"warm_control_gpu_ms\":"<<v.warmBaseGPU<<",\"warm_ab_pairs\":"<<v.warmAB<<",\"warm_ba_pairs\":"<<v.warmBA<<",\"timed_ab_pairs\":"<<v.ab<<",\"timed_ba_pairs\":"<<v.ba<<",\"integer_samples_exact\":"<<v.integerSamples<<",\"full_late_scale_identity_elements\":"<<v.identityElements<<",\"late_scale_subnormal_FTZ_envelope_elements\":"<<v.lateScaleSubnormalEnvelope<<",\"activation_scales_exact\":"<<v.scaleElements<<",\"activation_RNE_codes_exact\":"<<v.codeElements<<",\"full_component_errors\":{";
    for(size_t j=0;j<5;++j){if(j)report<<',';report<<splash::json::quote(names[j])<<':';v.quality[j].write(report);}report<<"},\"sampled_original_F32_source_FP64_error\":";v.sourceFP64.write(report);report<<",\"sampled_dequantized_FP64_error\":";v.dequantizedFP64.write(report);report<<",\"sampled_source_dequantized_FP64_quantization_error\":";v.quantizationFP64.write(report);report<<",\"complete_gpu_ms\":";array(report,v.gpu);report<<",\"complete_wall_ms\":";array(report,v.wall);report<<",\"paired_control_gpu_ms_samples\":";array(report,v.baseGPU);report<<",\"paired_control_wall_ms_samples\":";array(report,v.baseWall);report<<",\"paired_speedup_samples\":";array(report,v.speedup);report<<'}';
  }report<<"]}\n";if(outPath.empty())std::cout<<report.str();else{std::ofstream out(outPath);require(bool(out),"report file open failed");out<<report.str();}
  for(size_t i=1;i<variants.size();++i)std::cerr<<variants[i].name<<" gpu="<<median(variants[i].gpu)<<"ms speedup="<<median(variants[i].speedup)<<" quality="<<variants[i].passed<<'\n';return 0;
}catch(const std::exception &e){std::cerr<<"HC-UP W8A8 oracle failed: "<<e.what()<<'\n';return 1;}}
