// Private Root-run dense coefficient-I8 screen. CPU exit opens no payload/device.
#include "dev/benchmarks/dense_i8_decode_sep21/cache.hpp"
#include "dev/benchmarks/dense_i8_decode_sep21/abi.hpp"
#include "flash/FlashAffine.hpp"
#include "flash/FlashDenseSmallRows.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

namespace {
using namespace splash::flash;
using namespace splash::metal;
namespace cache=dense_i8_decode_sep21;
namespace precision=splash::flash::dense_i8_decode::precision;
constexpr uint32_t kSticky=0x80000000u;
void require(bool value,const std::string &reason){if(!value)throw std::runtime_error(reason);}
std::string text(NSString *value) {
  require([value isKindOfClass:[NSString class]],"dense fixture text field invalid");return value.UTF8String;
}
uint64_t integer(NSNumber *value) {
  require([value isKindOfClass:[NSNumber class]] && value.longLongValue>=0,"dense fixture integer field invalid");
  return value.unsignedLongLongValue;
}
void digestString(const std::string &value) {
  require(value.size()==64 && value.find_first_not_of("0123456789abcdef")==std::string::npos,"dense fixture SHA256 invalid");
}
std::string finiteJSON(double value){if(!std::isfinite(value))return "null";std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();}
uint32_t envNumber(const char *name,uint32_t fallback,uint32_t maximum) {
  const char *raw=std::getenv(name);if(!raw)return fallback;const std::string value(raw);size_t used=0;
  require(!value.empty() && std::all_of(value.begin(),value.end(),[](char c){return c>='0' && c<='9';}),std::string("invalid ")+name);
  const auto result=std::stoul(value,&used);require(used==value.size() && result>=1 && result<=maximum,std::string("invalid ")+name);return uint32_t(result);
}
struct FileSpec {std::string path,sha;uint64_t logical=0,allocated=0;};
struct SpanSpec {std::string path,dtype;uint64_t offset=0,length=0;std::vector<uint64_t> shape;};
struct Fixture {
  std::string projection;uint32_t k=0,n=0,capturedRows=0,bits=0,group=0;
  FileSpec f32,bf16,input,expected;
  std::array<SpanSpec,3> raw;
  uint64_t weightRow=0,weightExpert=0,parameterRow=0,parameterExpert=0;
};
FileSpec operand(NSDictionary *entry,const char *kind,uint64_t logical,bool optional=false) {
  const auto prefix=std::string(kind)+"_weights_";
  NSString *fileKey=[NSString stringWithUTF8String:(prefix+"file").c_str()];
  if(optional && !entry[fileKey])return {};
  const auto field=[&](const char *suffix){return entry[[NSString stringWithUTF8String:(prefix+suffix).c_str()]];};
  FileSpec result{text(field("file")),text(field("sha256")),integer(field("logical_bytes")),integer(field("allocated_bytes"))};
  digestString(result.sha);require(result.logical==logical && result.allocated>=logical && result.allocated%16384==0,"dense saved operand extent invalid");return result;
}
std::vector<Fixture> loadFixtures(const char *path) {
  const uint64_t bytes=std::filesystem::file_size(path);require(bytes>0 && bytes<(4ULL<<20),"dense metadata manifest too large");
  NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];NSError *error=nil;
  NSDictionary *root=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(root && !error && [root isKindOfClass:[NSDictionary class]],"dense fixture JSON invalid");
  require(text(root[@"schema"])=="splash-private-dense-i8-f32-row-fit-fixtures-sep21-v1","dense fixture schema differs");
  require(text(root[@"source_identity_sha256"])=="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e" &&
      text(root[@"weights_manifest_fingerprint"])=="edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0","dense fixture source identity differs");
  NSArray *cases=root[@"cases"];require([cases isKindOfClass:[NSArray class]] && cases.count>0 && cases.count<=32,"dense fixture role count invalid");
  std::vector<Fixture> result;
  const char *selected=std::getenv("DENSE_I8_DECODE_SEP21_PROJECTION");
  for(NSDictionary *entry in cases) {
    Fixture f;f.projection=text(entry[@"projection"]);
    if(selected && f.projection!=selected)continue;
    require((f.projection.find(".linear_attn.")!=std::string::npos || f.projection.find(".self_attn.")!=std::string::npos ||
        f.projection.find(".ple.")!=std::string::npos) && f.projection.find("mtp")==std::string::npos &&
        f.projection.find("router")==std::string::npos && f.projection.find("lm_head")==std::string::npos &&
        f.projection.find(".hc.")==std::string::npos,"dense initial role excluded");
    const auto k=integer(entry[@"input_size"]),n=integer(entry[@"output_size"]),r=integer(entry[@"captured_rows"]);
    require(k>0 && k<=32768 && k%32==0 && n>0 && n<=32768 && n%64==0 && r==2048,"dense captured role geometry invalid");
    f.k=uint32_t(k);f.n=uint32_t(n);f.capturedRows=uint32_t(r);f.bits=uint32_t(integer(entry[@"source_bits"]));f.group=uint32_t(integer(entry[@"source_group_size"]));
    f.f32=operand(entry,"f32",n*k*4);f.bf16=operand(entry,"bf16",n*k*2,true);
    f.input={text(entry[@"input_file"]),text(entry[@"input_sha256"]),r*k*2,r*k*2};
    f.expected={text(entry[@"captured_expected_file"]),text(entry[@"captured_expected_sha256"]),r*n*2,r*n*2};
    digestString(f.input.sha);digestString(f.expected.sha);
    f.weightRow=integer(entry[@"source_weight_row_stride_bytes"]);f.weightExpert=integer(entry[@"source_weight_expert_stride_bytes"]);
    f.parameterRow=integer(entry[@"source_parameter_row_stride_bytes"]);f.parameterExpert=integer(entry[@"source_parameter_expert_stride_bytes"]);
    NSDictionary *raw=entry[@"raw_source_tensors"];uint32_t index=0;
    for(NSString *name in @[@"weight",@"scales",@"biases"]) {
      NSDictionary *tensor=raw[name];SpanSpec s{text(tensor[@"file"]),text(tensor[@"dtype"]),integer(tensor[@"offset_bytes"]),integer(tensor[@"length_bytes"]),{}};
      for(NSNumber *dim in tensor[@"shape"])s.shape.push_back(integer(dim));
      require(s.shape.size()==2 && s.shape[0]==n && s.length>0 && s.length<=n*k*4,"dense raw tensor span geometry invalid");f.raw[index++]=std::move(s);
    }
    require(f.raw[0].dtype=="U32" && f.raw[1].dtype=="BF16" && f.raw[2].dtype=="BF16" &&
        f.raw[0].length>=n*f.weightRow && f.raw[1].length>=n*f.parameterRow && f.raw[2].length>=n*f.parameterRow &&
        (f.bits==4 || f.bits==5 || f.bits==6 || f.bits==8) && (f.group==64 || f.group==128),"dense original raw affine metadata invalid");
    result.push_back(std::move(f));
  }
  require(!result.empty(),"dense projection selection has no fixture");return result;
}
struct Loaded final {
  std::shared_ptr<cache::ReadonlyMapping> mapping;MetalBuffer base,view;FileSpec spec;
  static Loaded map(MetalBackend &backend,const FileSpec &spec,const char *label) {
    Loaded result;result.spec=spec;result.mapping=std::make_shared<cache::ReadonlyMapping>(spec.path,spec.allocated);
    require(cache::hash(result.mapping->address(),spec.allocated)==spec.sha,"dense certified source/capture payload SHA differs");
    result.mapping->requireUnchanged();result.base=backend.wrapSharedMemory(result.mapping->address(),spec.allocated,result.mapping,label);
    result.view=backend.view(result.base,0,spec.logical);return result;
  }
  bool immutable()const{mapping->requireUnchanged();return cache::hash(base.contents(),spec.allocated)==spec.sha;}
};
struct Comparison final {
  uint64_t elements=0,mismatches=0,nonfinite=0;double l2=0,cosine=1,maxAbs=0;
  bool guard()const{return !nonfinite && std::isfinite(l2) && l2<=precision::kMaximumRelativeL2 && cosine>=precision::kMinimumCosine;}
  void write(std::ostream &out)const {
    out<<"{\"elements\":"<<elements<<",\"bf16_mismatches\":"<<mismatches<<",\"nonfinite_pairs\":"<<nonfinite
        <<",\"relative_l2\":"<<finiteJSON(l2)<<",\"cosine\":"<<finiteJSON(cosine)<<",\"max_abs\":"<<finiteJSON(maxAbs)<<'}';
  }
};
Comparison compare(MetalBuffer a,MetalBuffer b,uint64_t elements) {
  require(a.sizeBytes()>=elements*2 && b.sizeBytes()>=elements*2,"dense output comparison extent invalid");
  Comparison c;c.elements=elements;double error=0,norma=0,normb=0,dot=0;
  const auto *aa=static_cast<const uint16_t *>(a.contents()),*bb=static_cast<const uint16_t *>(b.contents());
  for(uint64_t i=0;i<elements;++i) {
    c.mismatches+=aa[i]!=bb[i];const double x=precision::number(aa[i]),y=precision::number(bb[i]);
    if(!std::isfinite(x)||!std::isfinite(y)){++c.nonfinite;continue;}
    error+=(x-y)*(x-y);norma+=x*x;normb+=y*y;dot+=x*y;c.maxAbs=std::max(c.maxAbs,std::abs(x-y));
  }
  c.l2=norma?std::sqrt(error/norma):(error?std::numeric_limits<double>::infinity():0);
  c.cosine=norma&&normb?dot/std::sqrt(norma*normb):(norma==normb?1:0);return c;
}
struct Variant {uint32_t m,n,sg;const char *suffix;};
constexpr std::array<Variant,4> variants{{{8,64,4,"m8_n64_sg4"},{8,128,4,"m8_n128_sg4"},{16,64,4,"m16_n64_sg4"},{8,64,2,"m8_n64_sg2"}}};
struct Proof final {
  uint64_t samples=0,failed=0;double minOriginal=std::numeric_limits<double>::infinity(),maxError=0;
  std::vector<std::string> firstFailures;
  bool pass()const{return samples>0 && !failed;}
  void write(std::ostream &out)const {
    out<<"{\"pass\":"<<(pass()?"true":"false")<<",\"samples\":"<<samples<<",\"failed\":"<<failed
        <<",\"minimum_abs_original_f64\":"<<finiteJSON(minOriginal)<<",\"maximum_abs_error_from_original_f64\":"<<finiteJSON(maxError)
        <<",\"sample_policy\":\"first/last actual row; 512 uniform columns, N64/N128 tile edges, first/last and smallest audited absolute outputs\",\"first_failures\":[";
    for(size_t i=0;i<firstFailures.size();++i){if(i)out<<',';out<<firstFailures[i];}out<<"]}";
  }
};
Proof projectionProof(const Fixture &f,const float *source,const cache::CoefficientCache &fitted,
    MetalBuffer input,MetalBuffer raw,uint32_t rows) {
  Proof proof;const auto *x=static_cast<const uint16_t *>(input.contents());const auto *audit=static_cast<const float *>(raw.contents());
  const auto *codes=static_cast<const int8_t *>(fitted.codes.view.contents());const auto *scales=static_cast<const float *>(fitted.scales.view.contents());
  std::set<uint32_t> sampledRows{0,rows-1};
  for(uint32_t row:sampledRows) {
    std::set<uint32_t> columns{0,f.n-1};
    for(uint32_t sample=0;sample<512;++sample)columns.insert(uint32_t(uint64_t(sample)*(f.n-1)/511));
    for(uint32_t start=0;start<f.n;start+=64){columns.insert(start);columns.insert(std::min(start+63,f.n-1));}
    std::vector<std::pair<float,uint32_t>> nearZero;nearZero.reserve(f.n);
    for(uint32_t n=0;n<f.n;++n)nearZero.emplace_back(std::abs(audit[uint64_t(row)*f.n+n]),n);
    const uint32_t count=std::min<uint32_t>(8,f.n);
    std::partial_sort(nearZero.begin(),nearZero.begin()+count,nearZero.end(),[](const auto &a,const auto &b){return a.first<b.first;});
    for(uint32_t i=0;i<count;++i)columns.insert(nearZero[i].second);
    for(uint32_t n:columns) {
      const auto certificate=precision::certifyProjection(x+uint64_t(row)*f.k,source+uint64_t(n)*f.k,
          codes+uint64_t(n)*f.k,f.k,scales[n],audit[uint64_t(row)*f.n+n]);
      ++proof.samples;proof.failed+=!certificate.pass();proof.minOriginal=std::min(proof.minOriginal,std::abs(certificate.originalF64));
      proof.maxError=std::max(proof.maxError,certificate.absoluteErrorFromOriginal);
      if(!certificate.pass() && proof.firstFailures.size()<16) {
        std::ostringstream detail;detail<<"{\"row\":"<<row<<",\"n\":"<<n<<",\"certificate\":";certificate.write(detail);detail<<'}';proof.firstFailures.push_back(detail.str());
      }
    }
  }
  return proof;
}
void names(std::ostream &out,std::span<const ComputeDispatch> commands) {
  out<<'[';for(size_t i=0;i<commands.size();++i){if(i)out<<',';out<<splash::json::quote(commands[i].pipelineName);}out<<']';
}
void times(std::ostream &out,const std::vector<CommandTiming> &timings,bool gpu) {
  out<<'[';for(size_t i=0;i<timings.size();++i){if(i)out<<',';out<<(gpu?timings[i].gpuSeconds:timings[i].wallSeconds)*1000;}out<<']';
}
std::vector<uint32_t> rowSelection() {
  const char *raw=std::getenv("DENSE_I8_DECODE_SEP21_ROWS");if(!raw)return {1,4,8,16};
  std::stringstream parser(raw);std::string field;std::vector<uint32_t> rows;
  while(std::getline(parser,field,',')) {
    require(field=="1"||field=="4"||field=="8"||field=="16","dense row selector must be 1/4/8/16");
    const uint32_t row=uint32_t(std::stoul(field));require(std::find(rows.begin(),rows.end(),row)==rows.end(),"dense duplicate row selection");rows.push_back(row);
  }
  require(!rows.empty(),"dense empty row selection");return rows;
}
struct Candidate final {
  uint32_t index=0;cache::Guarded output,raw,diagnostic;CommandGraph timed,audit;
  Comparison initial,final,bf16Comparison,capturedComparison;Proof initialProof,finalProof;
  std::vector<std::string> failures;std::vector<CommandTiming> f32Times,bf16Times,rawTimes,candidateTimes;
  bool eligible=false,auditTimedExact=false;std::string outputSHA;
};
struct RowCase final {
  uint32_t rows=0,paddedRows=0;MetalBuffer input,expected;FlashFloatDenseSmallRowsTile tile{};
  cache::Guarded f32Output,bf16Output,rawOutput,f32Diagnostic,bf16Diagnostic,rawDiagnostic;
  CommandGraph f32Graph,bf16Graph,rawGraph;std::vector<Candidate> candidates;Comparison rawVsF32,f32VsCaptured,bf16VsF32;
};
void failure(Candidate &candidate,const std::string &reason) {
  if(std::find(candidate.failures.begin(),candidate.failures.end(),reason)==candidate.failures.end())candidate.failures.push_back(reason);
}
void candidateGraph(CommandGraph &graph,MetalBackend &backend,const Fixture &f,const cache::CoefficientCache &fitted,
    const RowCase &row,const Candidate &c,const Variant &v,bool audit) {
  DenseI8Params params{row.rows,f.k,f.n,v.m,v.n,0,0,0};
  graph.add(std::string("dense_i8_decode_sep21_")+v.suffix+(audit?"_audit":""),
      {row.input,fitted.codes.view,fitted.scales.view,c.output.view,c.diagnostic.view},params,
      {f.n/v.n,(row.rows+v.m-1)/v.m,1},{v.sg*32u,1,1});
  if(audit) {
    auto d=graph.dispatches().back();d.buffers.push_back({6,c.raw.view});
    // Byte storage stays owned by this graph; append the audit binding directly
    // through a standalone retained list at the call site instead.
    (void)d;
  }
  (void)backend;
}
uint64_t planned(const Fixture &f,const std::vector<uint32_t> &rows,uint32_t count) {
  uint64_t total=f.f32.allocated+f.bf16.allocated+f.input.allocated+f.expected.allocated+
      cache::CoefficientCache::plannedBytes(f.n,f.k)+2*cache::rounded(16ULL*f.k*2);
  for(const auto &span:f.raw)total+=cache::guardedBytes(span.length);
  for(uint32_t r:rows) {
    total+=3*(cache::guardedBytes(uint64_t(r)*f.n*2)+cache::guardedBytes(4));
    total+=uint64_t(count)*(cache::guardedBytes(uint64_t(r)*f.n*2)+cache::guardedBytes(uint64_t(r)*f.n*4)+cache::guardedBytes(4));
  }
  return total;
}
bool runFixture(MetalBackend &backend,const Fixture &f,const std::vector<uint32_t> &rowCounts,
    const std::vector<uint32_t> &active,uint32_t pairs,bool strict,std::ostream &out) {
  const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);
  require(physical>reserve,"dense host reserve unavailable");splash::engine::MemoryGovernor governor(backend,physical-reserve,reserve);
  const uint64_t admitted=planned(f,rowCounts,uint32_t(active.size()));auto admission=governor.tryReserve(admitted);
  require(bool(admission),"dense role weights/cache/captures/guards/workspace governor admission denied");
  const uint64_t before=backend.memoryStats().allocatedBytes;
  auto original=Loaded::map(backend,f.f32,"private dense original certified F32 coefficient matrix");
  auto captured=Loaded::map(backend,f.input,"private dense actual prefill BF16 capture");
  auto expected=Loaded::map(backend,f.expected,"private dense captured prefill BF16 outputs diagnostic only");
  std::unique_ptr<Loaded> bf16;
  if(!f.bf16.path.empty())bf16=std::make_unique<Loaded>(Loaded::map(backend,f.bf16,"private dense existing BF16 coefficient cache control"));
  const auto *source=static_cast<const float *>(original.view.contents());const auto *input=static_cast<const uint16_t *>(captured.view.contents());
  for(uint32_t r=0;r<16;++r)for(uint32_t k=0;k<f.k;++k)require(std::isfinite(precision::number(input[uint64_t(r)*f.k+k])),"dense actual BF16 capture contains nonfinite input");
  const auto fitStart=std::chrono::steady_clock::now();auto fitted=cache::CoefficientCache::fit(backend,source,f.n,f.k);
  const double fitCPUms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fitStart).count();
  const auto coefficientProof=precision::certifyCoefficients(source,static_cast<const int8_t *>(fitted.codes.view.contents()),
      static_cast<const float *>(fitted.scales.view.contents()),f.n,f.k);
  const std::string codesSHA=cache::hash(fitted.codes.view.contents(),fitted.codes.logical);
  const std::string scalesSHA=cache::hash(fitted.scales.view.contents(),fitted.scales.logical);
  std::array<cache::RawSpan,3> spans;
  std::array<FlashTensor,3> rawTensors;
  for(uint32_t plane=0;plane<3;++plane) {
    spans[plane]=cache::RawSpan::load(backend,f.raw[plane].path,f.raw[plane].offset,f.raw[plane].length);
    rawTensors[plane]={spans[plane].storage.view,plane ? FlashDType::BF16 : FlashDType::U32,f.raw[plane].shape,f.raw[plane].length};
  }
  FlashAffineProjection rawProjection{&rawTensors[0],&rawTensors[1],&rawTensors[2],1,f.n,f.k,f.bits,f.group,
      f.weightRow,f.weightExpert,f.parameterRow,f.parameterExpert};
  uint64_t reconstructionFailures=0;
  for(uint32_t sample=0;sample<260;++sample) {
    const uint64_t at=uint64_t(sample)*(uint64_t(f.n)*f.k-1)/259;const uint32_t n=uint32_t(at/f.k),k=uint32_t(at%f.k);
    const auto *packed=static_cast<const std::byte *>(spans[0].storage.view.contents())+uint64_t(n)*f.weightRow;
    const uint32_t code=unpackAffineCode(std::span<const std::byte>(packed,size_t(f.weightRow)),f.bits,k);
    const auto parameter=uint64_t(n)*f.parameterRow+uint64_t(k/f.group)*2;
    uint16_t scaleBits,biasBits;std::memcpy(&scaleBits,static_cast<const uint8_t *>(spans[1].storage.view.contents())+parameter,2);
    std::memcpy(&biasBits,static_cast<const uint8_t *>(spans[2].storage.view.contents())+parameter,2);
    volatile float product=float(code)*precision::number(scaleBits);const float reconstructed=product+precision::number(biasBits);
    reconstructionFailures+=std::bit_cast<uint32_t>(reconstructed)!=std::bit_cast<uint32_t>(source[at]);
  }
  require(!reconstructionFailures,"selected original raw coefficients do not match certified F32 saved matrix");
  FlashFloatDenseSmallRowsWorkspace f32Workspace(backend,f.k);FlashDenseSmallRowsWorkspace bf16Workspace(backend,f.k);
  FlashTensor f32Tensor{original.view,FlashDType::F32,{f.n,f.k},uint64_t(f.n)*f.k*4};
  FlashTensor bf16Tensor;if(bf16)bf16Tensor={bf16->view,FlashDType::BF16,{f.n,f.k},uint64_t(f.n)*f.k*2};
  std::vector<RowCase> shapes;
  for(uint32_t rows:rowCounts) {
    RowCase row;row.rows=rows;row.input=backend.view(captured.view,0,uint64_t(rows)*f.k*2);row.expected=backend.view(expected.view,0,uint64_t(rows)*f.n*2);
    row.tile=flashFloatDenseSmallRowsPolicy(f.projection,rows,f.n,f.k,f.bits,f.group).value_or(
        rows<=8 ? FlashFloatDenseSmallRowsTile::M8N64 : FlashFloatDenseSmallRowsTile::M16N64);
    const uint32_t tileM=uint32_t(row.tile)>=2?16:8;row.paddedRows=(rows+tileM-1)/tileM*tileM;
    row.f32Output=cache::Guarded::allocate(backend,uint64_t(rows)*f.n*2,"dense original F32 cached output guard");
    row.bf16Output=cache::Guarded::allocate(backend,uint64_t(rows)*f.n*2,"dense existing BF16 cached output guard");
    row.rawOutput=cache::Guarded::allocate(backend,uint64_t(rows)*f.n*2,"dense original raw affine output guard");
    row.f32Diagnostic=cache::Guarded::allocate(backend,4,"dense F32 control diagnostic guard");row.bf16Diagnostic=cache::Guarded::allocate(backend,4,"dense BF16 control diagnostic guard");row.rawDiagnostic=cache::Guarded::allocate(backend,4,"dense raw affine diagnostic guard");
    *static_cast<uint32_t *>(row.f32Diagnostic.view.contents())=kSticky;*static_cast<uint32_t *>(row.bf16Diagnostic.view.contents())=kSticky;*static_cast<uint32_t *>(row.rawDiagnostic.view.contents())=kSticky;
    addFloatDenseSmallRows(backend,row.f32Graph,row.input,f32Tensor,row.f32Output.view,row.f32Diagnostic.view,rows,f32Workspace,row.tile);
    if(bf16)addDenseBF16SmallRows(backend,row.bf16Graph,row.input,bf16Tensor,row.bf16Output.view,row.bf16Diagnostic.view,rows,bf16Workspace,static_cast<FlashDenseSmallRowsTile>(row.tile));
    addAffine(row.rawGraph,row.input,rawProjection,row.rawOutput.view,row.rawDiagnostic.view,rows);
    for(uint32_t index:active) {
      Candidate c;c.index=index;c.output=cache::Guarded::allocate(backend,uint64_t(rows)*f.n*2,"dense fitted I8 output guard");
      c.raw=cache::Guarded::allocate(backend,uint64_t(rows)*f.n*4,"dense fitted I8 raw scaled F32 audit guard");c.diagnostic=cache::Guarded::allocate(backend,4,"dense fitted I8 diagnostic guard");
      *static_cast<uint32_t *>(c.diagnostic.view.contents())=kSticky;
      candidateGraph(c.timed,backend,f,fitted,row,c,variants[index],false);candidateGraph(c.audit,backend,f,fitted,row,c,variants[index],true);
      row.candidates.push_back(std::move(c));
    }
    shapes.push_back(std::move(row));
  }
  const auto controlHealthy=[&](const RowCase &r) {
    return *static_cast<const uint32_t *>(r.f32Diagnostic.view.contents())==kSticky && *static_cast<const uint32_t *>(r.rawDiagnostic.view.contents())==kSticky &&
        (!bf16 || *static_cast<const uint32_t *>(r.bf16Diagnostic.view.contents())==kSticky) && r.f32Output.clean() && r.rawOutput.clean() && r.f32Diagnostic.clean() && r.rawDiagnostic.clean() &&
        (!bf16 || (r.bf16Output.clean() && r.bf16Diagnostic.clean())) && fitted.guardsClean();
  };
  const auto candidateHealthy=[&](const Candidate &c){return c.output.clean() && c.raw.clean() && c.diagnostic.clean() && *static_cast<const uint32_t *>(c.diagnostic.view.contents())==kSticky && fitted.guardsClean();};
  const auto auditDispatch=[&](const Candidate &c){auto d=c.audit.dispatches().front();d.buffers.push_back({6,c.raw.view});return d;};
  for(auto &row:shapes) {
    (void)backend.submitCommand(row.f32Graph.dispatches());(void)backend.submitCommand(row.rawGraph.dispatches());if(bf16)(void)backend.submitCommand(row.bf16Graph.dispatches());
    require(controlHealthy(row),"dense stock control diagnostics/guards failed");
    row.rawVsF32=compare(row.f32Output.view,row.rawOutput.view,uint64_t(row.rows)*f.n);row.f32VsCaptured=compare(row.expected,row.f32Output.view,uint64_t(row.rows)*f.n);
    if(bf16)row.bf16VsF32=compare(row.f32Output.view,row.bf16Output.view,uint64_t(row.rows)*f.n);
    require(!row.rawVsF32.nonfinite && row.rawVsF32.l2<=0.0001,"dense raw affine versus F32 cache sanity guard failed");
    for(auto &c:row.candidates) {
      if(f.n%variants[c.index].n){failure(c,"role output does not divide candidate column tile");continue;}
      try {
        const auto audit=auditDispatch(c);(void)backend.submitCommand(std::span<const ComputeDispatch>(&audit,1));
        c.initial=compare(row.f32Output.view,c.output.view,uint64_t(row.rows)*f.n);c.initialProof=projectionProof(f,source,fitted,row.input,c.raw.view,row.rows);
        c.outputSHA=cache::hash(c.output.view.contents(),c.output.logical);(void)backend.submitCommand(c.timed.dispatches());
        c.auditTimedExact=cache::hash(c.output.view.contents(),c.output.logical)==c.outputSHA;
        if(!coefficientProof.pass())failure(c,"all-coefficient quantization census failed");
        if(!c.initial.guard())failure(c,"full BF16 output exceeds preregistered .02 L2/.9998 cosine/finite guard");
        if(!c.initialProof.pass())failure(c,"raw scaled F32 weighted original-coefficient F64 quantization envelope failed");
        if(!c.auditTimedExact)failure(c,"audit and timed BF16 output differs");
        if(!candidateHealthy(c))failure(c,"candidate diagnostics/guard failed before timing");
        if(strict && c.initial.mismatches)failure(c,"strict original F32-control BF16 output differs");
        c.eligible=c.failures.empty();
      }catch(const std::exception &e){failure(c,std::string("initial qualification: ")+e.what());}
    }
  }
  // Rotation spans every requested row shape and coefficient geometry. Each
  // matched group balances candidate/control order; no operand/output scans,
  // hash work or CPU comparisons occur between the timed GPU commands.
  const uint32_t variantsPerShape=uint32_t(active.size()),jobs=uint32_t(shapes.size())*variantsPerShape;
  for(uint32_t pair=0;pair<pairs;++pair)for(uint32_t order=0;order<jobs;++order) {
    const uint32_t job=(pair+order)%jobs;auto &row=shapes[job/variantsPerShape];auto &c=row.candidates[job%variantsPerShape];if(!c.eligible)continue;
    const uint32_t categories=bf16?4:3;
    try {
      for(uint32_t step=0;step<categories;++step) {
        const uint32_t category=(pair+job+step)%categories;
        if(!category)c.f32Times.push_back(backend.submitCommand(row.f32Graph.dispatches()));
        else if(category==1)c.rawTimes.push_back(backend.submitCommand(row.rawGraph.dispatches()));
        else if(category==2)c.candidateTimes.push_back(backend.submitCommand(c.timed.dispatches()));
        else c.bf16Times.push_back(backend.submitCommand(row.bf16Graph.dispatches()));
      }
    }catch(const std::exception &e){failure(c,std::string("matched timing: ")+e.what());c.eligible=false;}
  }
  bool pass=coefficientProof.pass();
  for(auto &row:shapes) {
    require(controlHealthy(row),"dense timed stock control diagnostic/guard failed");
    for(auto &c:row.candidates) {
      if(!c.initialProof.samples){pass=false;continue;}
      try {
        const auto audit=auditDispatch(c);(void)backend.submitCommand(std::span<const ComputeDispatch>(&audit,1));
        c.final=compare(row.f32Output.view,c.output.view,uint64_t(row.rows)*f.n);c.finalProof=projectionProof(f,source,fitted,row.input,c.raw.view,row.rows);
        c.bf16Comparison=bf16?compare(row.bf16Output.view,c.output.view,uint64_t(row.rows)*f.n):Comparison{};
        c.capturedComparison=compare(row.expected,c.output.view,uint64_t(row.rows)*f.n);
        if(!c.final.guard())failure(c,"final complete BF16 output guard failed");if(!c.finalProof.pass())failure(c,"final weighted-F64 raw audit failed");
        if(!candidateHealthy(c))failure(c,"final candidate diagnostics/guard failed");
        if(strict && c.final.mismatches)failure(c,"final strict original-control BF16 output differs");
        if(cache::hash(c.output.view.contents(),c.output.logical)!=c.outputSHA)failure(c,"candidate output changed across matched timing replay");
      }catch(const std::exception &e){failure(c,std::string("final replay: ")+e.what());}
      pass=pass && c.failures.empty();
    }
  }
  const auto finalCoefficientProof=precision::certifyCoefficients(source,static_cast<const int8_t *>(fitted.codes.view.contents()),static_cast<const float *>(fitted.scales.view.contents()),f.n,f.k);
  bool immutable=original.immutable() && captured.immutable() && expected.immutable() && (!bf16 || bf16->immutable()) &&
      codesSHA==cache::hash(fitted.codes.view.contents(),fitted.codes.logical) && scalesSHA==cache::hash(fitted.scales.view.contents(),fitted.scales.logical) && fitted.guardsClean();
  for(const auto &span:spans)immutable=immutable && span.immutable();
  const uint64_t allocated=backend.memoryStats().allocatedBytes;
  require(allocated>=before && allocated-before<=admitted,"dense component allocations exceeded governor plan");admission->commit();pass=pass && immutable && finalCoefficientProof.pass();
  out<<"{\"projection\":"<<splash::json::quote(f.projection)<<",\"input_size\":"<<f.k<<",\"output_size\":"<<f.n
      <<",\"source_bits\":"<<f.bits<<",\"source_group_size\":"<<f.group<<",\"numerical_alternative\":true,\"activation_quantization\":false"
      <<",\"input_scope\":\"first R actual BF16 rows of captured2048-token prefill; not live decode states\""
      <<",\"input_payload_sha256\":"<<splash::json::quote(f.input.sha)<<",\"original_f32_weight_sha256\":"<<splash::json::quote(f.f32.sha)
      <<",\"original_f32_cache_logical_bytes\":"<<f.f32.logical<<",\"fitted_i8_scale_logical_bytes\":"<<fitted.codes.logical+fitted.scales.logical
      <<",\"planned_bytes\":"<<admitted<<",\"actual_allocated_bytes\":"<<allocated-before<<",\"host_reserve_bytes\":"<<reserve
      <<",\"coefficient_fit_cpu_ms_excluded_from_timing\":"<<fitCPUms<<",\"coefficient_census_before\":";coefficientProof.write(out);
  out<<",\"coefficient_census_after\":";finalCoefficientProof.write(out);
  out<<",\"raw_source_reconstruction_samples\":260,\"raw_source_reconstruction_f32_bit_failures\":"<<reconstructionFailures
      <<",\"whole_raw_shard_checksums_claimed\":false,\"raw_selected_span_sha256\":[";
  for(uint32_t i=0;i<3;++i){if(i)out<<',';out<<splash::json::quote(spans[i].initialSHA);}
  out<<"],\"all_sources_codes_scales_and_guards_immutable\":"<<(immutable?"true":"false")<<",\"cases\":[";
  bool first=true;
  for(const auto &row:shapes)for(const auto &c:row.candidates) {
    if(!first)out<<',';first=false;const auto &v=variants[c.index];
    out<<"{\"rows\":"<<row.rows<<",\"variant\":"<<c.index+1<<",\"tile_m\":"<<v.m<<",\"tile_n\":"<<v.n<<",\"sg\":"<<v.sg
        <<",\"candidate_activation_padding_commands\":0,\"stock_cache_control_padded_rows\":"<<row.paddedRows
        <<",\"stock_padding_commands_included_in_control_timing\":true,\"raw_qmv_f32_control\":true,\"raw_scope\":\"original selected affine tensor spans only\""
        <<",\"screen_pass\":"<<(c.failures.empty() && c.initialProof.pass() && c.finalProof.pass()?"true":"false")
        <<",\"numerical_alternative\":true,\"model_semantics_qualified\":false,\"mtp_acceptance_qualified\":false"
        <<",\"live_decode_activation_qualified\":false,\"strict_full_bf16_requested\":"<<(strict?"true":"false")
        <<",\"strict_full_bf16_exact\":"<<(!c.initial.mismatches&&!c.final.mismatches&&c.initialProof.samples?"true":"false")
        <<",\"producer_guard\":{\"maximum_relative_l2\":0.02,\"minimum_cosine\":0.9998},\"initial_vs_f32\":";c.initial.write(out);
    out<<",\"final_vs_f32\":";c.final.write(out);out<<",\"candidate_vs_bf16_control\":";c.bf16Comparison.write(out);
    out<<",\"candidate_vs_captured_prefill_output_diagnostic\":";c.capturedComparison.write(out);
    out<<",\"raw_control_vs_f32\":";row.rawVsF32.write(out);out<<",\"f32_control_vs_captured_prefill_output_diagnostic\":";row.f32VsCaptured.write(out);
    out<<",\"bf16_control_vs_f32\":";row.bf16VsF32.write(out);out<<",\"initial_weighted_f64_raw_certificate\":";c.initialProof.write(out);
    out<<",\"final_weighted_f64_raw_certificate\":";c.finalProof.write(out);
    out<<",\"audit_timed_bf16_exact\":"<<(c.auditTimedExact?"true":"false")<<",\"original_f32_control_pipelines\":";names(out,row.f32Graph.dispatches());
    out<<",\"existing_bf16_control_pipelines\":";names(out,row.bf16Graph.dispatches());out<<",\"original_raw_control_pipelines\":";names(out,row.rawGraph.dispatches());
    out<<",\"candidate_pipelines\":";names(out,c.timed.dispatches());out<<",\"f32_gpu_ms\":";times(out,c.f32Times,true);
    out<<",\"f32_wall_ms\":";times(out,c.f32Times,false);out<<",\"bf16_gpu_ms\":";times(out,c.bf16Times,true);out<<",\"bf16_wall_ms\":";times(out,c.bf16Times,false);
    out<<",\"raw_gpu_ms\":";times(out,c.rawTimes,true);out<<",\"raw_wall_ms\":";times(out,c.rawTimes,false);
    out<<",\"candidate_gpu_ms\":";times(out,c.candidateTimes,true);out<<",\"candidate_wall_ms\":";times(out,c.candidateTimes,false);out<<",\"failures\":[";
    for(size_t i=0;i<c.failures.size();++i){if(i)out<<',';out<<splash::json::quote(c.failures[i]);}out<<"]}";
  }
  out<<"],\"pass\":"<<(pass?"true":"false")<<'}';return pass;
}
void cpuSelfTest() {
  cache::cpuSelfTest();precision::cpuSelfTest();require(sizeof(DenseI8Params)==32,"dense candidate parameter ABI differs");
  for(const auto &v:variants)require(v.m==8 || v.m==16,"dense tile inventory invalid");
}
} // namespace

int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if(argc==2 && std::string_view(argv[1])=="--cpu-self-test") {
        cpuSelfTest();std::cout<<"{\"pass\":true,\"gpu_work\":false,\"model_or_capture_payload_reads\":false,\"dense_i8_gpu_qualification\":\"pending\"}\n";return 0;
      }
      require(argc==5 && std::string_view(argv[1])=="--gpu","usage: dense-i8-oracle --gpu METALLIB FIXTURE_MANIFEST NEW_REPORT");
      require(!std::filesystem::exists(argv[4]),"choose fresh dense report path");
      const auto fixtures=loadFixtures(argv[3]);const auto rows=rowSelection();const uint32_t pairs=envNumber("DENSE_I8_DECODE_SEP21_PAIRS",4,32);
      const char *variant=std::getenv("DENSE_I8_DECODE_SEP21_VARIANT");const uint32_t selected=variant?envNumber("DENSE_I8_DECODE_SEP21_VARIANT",1,4)-1:UINT32_MAX;
      const bool strict=std::getenv("DENSE_I8_DECODE_SEP21_STRICT")!=nullptr;std::vector<uint32_t> active;
      for(uint32_t i=0;i<4;++i)if(selected==UINT32_MAX || selected==i)active.push_back(i);
      require(setenv("SPLASH_FLASH_QMV_F32","1",1)==0,"cannot enable original raw QMV F32 control");
      MetalBackend backend(argv[2]);std::ofstream out(argv[4]);require(bool(out),"cannot create dense report");
      out<<std::setprecision(17)<<"{\"schema\":\"splash-private-dense-i8-actual-capture-component-v1\",\"numerical_alternative\":true"
          <<",\"model_semantics_qualified\":false,\"mtp_acceptance_qualified\":false,\"whole_worker_selector_qualified\":false"
          <<",\"whole_model_loaded\":false,\"pairs\":"<<pairs<<",\"timing_scope\":\"warm balanced rotating actual projection commands; stock F32/BF16 cache padding included, source raw QMV F32 route included, candidate unchanged BF16 input with no padding; conversion/certificates/hashes excluded\",\"roles\":[";
      bool pass=true,first=true;
      for(const auto &f:fixtures) {
        if(!first)out<<',';first=false;std::ostringstream role;
        try {const bool accepted=runFixture(backend,f,rows,active,pairs,strict,role);pass=pass&&accepted;out<<role.str();}
        catch(const std::exception &e){pass=false;out<<"{\"projection\":"<<splash::json::quote(f.projection)<<",\"pass\":false,\"failure\":"<<splash::json::quote(e.what())<<'}';}
        out.flush();
      }
      out<<"],\"pass\":"<<(pass?"true":"false")<<"}\n";out.flush();require(bool(out),"dense report write failed");backend.stop();
      std::cout<<"{\"pass\":"<<(pass?"true":"false")<<",\"report\":"<<splash::json::quote(argv[4])<<"}\n";return pass?0:2;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
  }
}
