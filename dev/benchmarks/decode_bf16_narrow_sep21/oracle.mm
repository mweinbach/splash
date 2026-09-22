// Root-only selected-role screen. No backend or payload reads in CPU mode.
#include "KernelParams.hpp"
#include "flash/FlashAffine.hpp"
#include "flash/FlashDenseSmallRows.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "flash/FlashDecodeBF16DensePolicy.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashFloatDenseCache.h"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "dev/benchmarks/FlashFloatBoundaryAudit.hpp"
#include "dev/benchmarks/dense_i8_decode_sep21/cache.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash;
using namespace splash::flash;
using namespace splash::metal;
using namespace splash::flash::benchmark;
using dense_i8_decode_sep21::ReadonlyMapping;
using dense_i8_decode_sep21::RawSpan;
constexpr uint64_t kAlignment = 16384;
constexpr uint32_t kSticky = 0x40000000u;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
std::string stringField(NSDictionary *object, const char *name) {
  id value = object[[NSString stringWithUTF8String:name]];
  require([value isKindOfClass:NSString.class], "missing/invalid fixture string");
  return std::string(static_cast<NSString *>(value).UTF8String);
}
uint64_t integerField(NSDictionary *object, const char *name) {
  id value = object[[NSString stringWithUTF8String:name]];
  require([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(), "missing/invalid fixture integer");
  const double number = static_cast<NSNumber *>(value).doubleValue;
  require(std::isfinite(number) && number >= 0 && number <= (1ULL << 40) && std::floor(number) == number, "fixture integer exceeds bound");
  return static_cast<NSNumber *>(value).unsignedLongLongValue;
}
bool boolField(NSDictionary *object,const char *name) {
  id value=object[[NSString stringWithUTF8String:name]];
  require([value isKindOfClass:NSNumber.class]&&CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID(),"missing/invalidfixturecapturetag");
  return static_cast<NSNumber *>(value).boolValue;
}
uint32_t parseDecimal(const char *raw, uint32_t limit) {
  require(raw && *raw, "empty decimal argument"); uint64_t value = 0;
  for (const char *p = raw; *p; ++p) { require(*p >= '0' && *p <= '9', "nondecimal argument"); value = value * 10 + unsigned(*p - '0'); require(value <= limit, "decimal argument exceeds bound"); }
  return uint32_t(value);
}
NSDictionary *loadMetadata(const char *path) {
  require(std::filesystem::file_size(path) <= 2ULL << 20, "fixture metadata exceeds2MiB");
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]]; require(data, "cannot load fixture metadata");
  NSError *error = nil; id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(!error && [object isKindOfClass:NSDictionary.class], "fixture metadata must be object"); return object;
}
uint64_t rounded(uint64_t bytes) { return (bytes + kAlignment - 1) & ~(kAlignment - 1); }
std::string hash(const void *bytes, uint64_t length) { return dense_i8_decode_sep21::hash(bytes, length); }
std::string fileHash(const char *path) {
  std::ifstream input(path,std::ios::binary);require(bool(input),"cannotreadartifactprovenance");
  CC_SHA256_CTX context{};CC_SHA256_Init(&context);std::array<char,65536> data{};
  while(input){input.read(data.data(),data.size());if(input.gcount())CC_SHA256_Update(&context,data.data(),CC_LONG(input.gcount()));}
  require(input.eof(),"artifactprovenancereadfailed");std::array<uint8_t,32> result{};CC_SHA256_Final(result.data(),&context);
  constexpr char digits[]="0123456789abcdef";std::string out;for(uint8_t value:result){out+=digits[value>>4];out+=digits[value&15];}return out;
}
struct Guarded {
  MetalBuffer base, view; uint64_t logical = 0;
  Guarded(MetalBackend &backend, uint64_t bytes) : logical(bytes) {
    require(bytes && bytes < (1ULL << 30), "guarded scratch extent invalid");
    base = backend.allocateBuffer(rounded(bytes) + 2 * kAlignment, BufferStorage::Shared, "BF16 narrow guarded source/output/tap/pad");
    view = backend.view(base, kAlignment, bytes); clear();
  }
  void clear() { std::memset(base.contents(), 0xa5, base.sizeBytes()); }
  bool guards() const {
    const auto *bytes = static_cast<const uint8_t *>(base.contents());
    for (uint64_t i = 0; i < kAlignment; ++i) if (bytes[i] != 0xa5) return false;
    for (uint64_t i = kAlignment + logical; i < base.sizeBytes(); ++i) if (bytes[i] != 0xa5) return false;
    return true;
  }
};
struct Operand {
  std::shared_ptr<ReadonlyMapping> mapping; FlashTensor tensor; std::string initialSHA;
  Operand(MetalBackend &backend, const std::string &path, uint64_t allocated, uint64_t logical,
          const std::string &expected, FlashDType dtype, uint32_t n, uint32_t k) {
    require(allocated >= logical && allocated <= (1ULL << 30), "operand byte extent invalid");
    mapping = std::make_shared<ReadonlyMapping>(path, allocated);
    initialSHA = hash(mapping->address(), allocated); require(initialSHA == expected, "operand payload hash differs");
    tensor = {backend.wrapSharedMemory(mapping->address(), allocated, mapping, "verified selected BF16/F32 coefficient mapping"), dtype, {n,k}, logical};
  }
  void unchanged() const { mapping->requireUnchanged(); require(hash(mapping->address(), tensor.buffer.sizeBytes()) == initialSHA, "selected operand mutated"); }
};
struct RawSource {
  std::array<RawSpan, 3> source; std::array<FlashTensor, 3> tensors; FlashAffineProjection projection;
  RawSource(MetalBackend &backend, NSDictionary *entry, uint32_t n, uint32_t k) {
    NSDictionary *spans = entry[@"raw_source_tensors"]; require([spans isKindOfClass:NSDictionary.class], "raw source tensor metadata absent");
    const std::array<const char *,3> fields{"weight","scales","biases"};
    for (uint32_t plane = 0; plane < 3; ++plane) {
      NSDictionary *span = spans[[NSString stringWithUTF8String:fields[plane]]]; require([span isKindOfClass:NSDictionary.class], "raw tensor span absent");
      const auto length = integerField(span, "length_bytes");
      source[plane] = RawSpan::load(backend, stringField(span,"file"), integerField(span,"offset_bytes"), length);
      tensors[plane] = {source[plane].storage.view, plane ? FlashDType::BF16 : FlashDType::U32, {}, length};
    }
    projection = {&tensors[0],&tensors[1],&tensors[2],1,n,k,uint32_t(integerField(entry,"source_bits")),uint32_t(integerField(entry,"source_group_size")),
      integerField(entry,"source_weight_row_stride_bytes"),integerField(entry,"source_weight_expert_stride_bytes"),
      integerField(entry,"source_parameter_row_stride_bytes"),integerField(entry,"source_parameter_expert_stride_bytes")};
  }
  void unchanged() const { for (const auto &span : source) require(span.immutable(), "selected raw affine source mutated"); }
};
float rawCoefficient(const FlashAffineProjection &p, uint32_t n, uint32_t k) {
  const auto *weights = static_cast<const uint8_t *>(p.weights->buffer.contents()) + uint64_t{n} * p.weightRowStrideBytes;
  const uint64_t bit = uint64_t{k} * p.bits; const uint32_t shift = bit % 8;
  uint32_t code = weights[bit/8]; if (shift+p.bits > 8) code |= uint32_t{weights[bit/8+1]} << 8;
  code = (code >> shift) & ((1u << p.bits)-1);
  const auto *scales = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(p.scales->buffer.contents()) + uint64_t{n}*p.parameterRowStrideBytes);
  const auto *biases = reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(p.biases->buffer.contents()) + uint64_t{n}*p.parameterRowStrideBytes);
  const float product = float(code)*number(scales[k/p.groupSize]); return product + number(biases[k/p.groupSize]);
}
struct Reference { double dot=0, absolute=0, bound=0; uint16_t rounded=0; };
std::vector<Reference> references(const uint16_t *input, const uint16_t *weights, uint32_t rows, uint32_t n, uint32_t k) {
  std::vector<Reference> result(uint64_t{rows}*n);
  for (uint32_t column=0; column<n; ++column) for (uint32_t row=0; row<rows; ++row) {
    double sum=0, correction=0, absolute=0;
    for (uint32_t j=0; j<k; ++j) {
      const double term=double(number(input[uint64_t{row}*k+j]))*number(weights[uint64_t{column}*k+j]);
      require(std::isfinite(term), "nonfinite BF16 reference product");
      const double next=sum+term; correction += std::abs(sum)>=std::abs(term) ? (sum-next)+term : (term-next)+sum; sum=next; absolute+=std::abs(term);
    }
    auto &cell=result[uint64_t{row}*n+column]; cell.dot=sum+correction; cell.absolute=absolute;
    cell.bound=f32DotBound(k,absolute); cell.rounded=bf16Double(cell.dot);
  }
  return result;
}
struct Audit {
  uint64_t cells=0, bf16OldMismatch=0, rawF32OldMismatch=0, boundFailures=0, strictFailures=0, signFailures=0, exceptional=0;
  double maxBoundRatio=0;
  void write(std::ostream &out) const {
    out << "{\"cells\":"<<cells<<",\"bf16_old_producer_mismatches\":"<<bf16OldMismatch<<",\"raw_f32_old_dot_mismatches\":"<<rawF32OldMismatch
        <<",\"f64_absolute_bound_failures\":"<<boundFailures<<",\"f64_strict_sensitive_bf16_failures\":"<<strictFailures
        <<",\"sign_failures\":"<<signFailures<<",\"exceptional_cells\":"<<exceptional<<",\"max_absolute_bound_ratio\":"<<maxBoundRatio<<'}';
  }
};
void coefficientDifference(std::ostream &out,const std::vector<uint16_t> &output,const std::vector<Reference> &ref) {
  uint64_t mismatch=0,nonfinite=0,sign=0;uint32_t ulp=0;double squared=0,magnitude=0;
  for(size_t i=0;i<output.size();++i){mismatch+=output[i]!=ref[i].rounded;const double value=number(output[i]);
    nonfinite+=!std::isfinite(value);if(!std::isfinite(value))continue;ulp=std::max(ulp,bf16ULP(output[i],ref[i].rounded));
    const double delta=value-number(ref[i].rounded);squared+=delta*delta;magnitude+=double(number(ref[i].rounded))*number(ref[i].rounded);
    sign+=ref[i].dot!=0&&value!=0&&std::signbit(ref[i].dot)!=std::signbit(value);}
  out<<"{\"reference\":\"full compensatedF64 dot on cachedBF16coefficients; raw/F32 control coefficientrounding may differ, not aqualificationgate\",\"cells\":"<<output.size()
     <<",\"bf16_mismatches\":"<<mismatch<<",\"max_bf16_ulp\":"<<ulp<<",\"nonfinite\":"<<nonfinite<<",\"sign_differences\":"<<sign<<",\"relative_l2\":"<<std::sqrt(squared/std::max(1e-300,magnitude))<<'}';
}
Audit audit(const uint16_t *output, const float *tap, const uint16_t *old, const float *oldTap, const std::vector<Reference> &ref) {
  Audit result; result.cells=ref.size();
  for (uint64_t i=0; i<ref.size(); ++i) {
    result.bf16OldMismatch += output[i]!=old[i]; result.rawF32OldMismatch += std::bit_cast<uint32_t>(tap[i])!=std::bit_cast<uint32_t>(oldTap[i]);
    const bool finite=std::isfinite(tap[i])&&std::isfinite(number(output[i])); result.exceptional+=!finite; if (!finite) continue;
    const double delta=std::abs(double(tap[i])-ref[i].dot); result.boundFailures+=delta>ref[i].bound;
    result.maxBoundRatio=std::max(result.maxBoundRatio,delta/std::max(1e-300,ref[i].bound));
    if (std::abs(ref[i].dot)<=32*ref[i].bound || std::abs(ref[i].dot)<std::numeric_limits<float>::min()) result.strictFailures+=output[i]!=ref[i].rounded;
    result.signFailures+=ref[i].dot!=0 && number(output[i])!=0 && std::signbit(ref[i].dot)!=std::signbit(number(output[i]));
  }
  return result;
}
double median(std::vector<double> data) { require(!data.empty(), "timings empty"); std::sort(data.begin(),data.end()); return data.size()%2 ? data[data.size()/2] : .5*(data[data.size()/2-1]+data[data.size()/2]); }
struct Route {
  std::string name; uint32_t m,n,sg; bool candidate=false, qualified=false;
  Guarded output,padded,tap,diag; CommandGraph timed,probe; Audit numerical;
  std::vector<uint16_t> golden; std::vector<double> gpu,wall;
  Route(MetalBackend &backend,std::string label,uint32_t rows,uint32_t outputs,uint32_t inputs,uint32_t tm,uint32_t tn,uint32_t ts,bool isCandidate)
      :name(std::move(label)),m(tm),n(tn),sg(ts),candidate(isCandidate),output(backend,uint64_t{rows}*outputs*2),padded(backend,uint64_t{(rows+tm-1)/tm*tm}*inputs*2),
       tap(backend,uint64_t{rows}*outputs*4),diag(backend,4) {}
  void reset() { output.clear(); padded.clear(); tap.clear(); diag.clear(); std::memcpy(diag.view.contents(),&kSticky,4); }
  void check() const {
    uint32_t status; std::memcpy(&status,diag.view.contents(),4);
    require(status==kSticky && output.guards()&&padded.guards()&&tap.guards()&&diag.guards(),"diagnostic/guard failure");
  }
  void snapshot(uint32_t rows,uint32_t outputs) { const auto *p=static_cast<const uint16_t *>(output.view.contents()); golden.assign(p,p+uint64_t{rows}*outputs); }
  void addTime(CommandTiming timing) { require(std::isfinite(timing.gpuSeconds)&&timing.gpuSeconds>0&&std::isfinite(timing.wallSeconds)&&timing.wallSeconds>0,"invalid positive timing"); gpu.push_back(timing.gpuSeconds);wall.push_back(timing.wallSeconds); }
};
void padGraph(CommandGraph &graph,Route &route,MetalBuffer input,uint32_t rows,uint32_t outputs,uint32_t inputs) {
  const FlashDenseSmallRowsParams params{rows,(rows+route.m-1)/route.m*route.m,inputs,outputs,0,outputs,route.m,64};
  graph.add("flash_dense_small_rows_pad",{input,route.padded.view,route.diag.view},params,{(uint64_t{params.padded_rows}*inputs+255)/256,1,1},{256,1,1});
}
void cachedGraph(Route &route,MetalBuffer input,MetalBuffer weights,uint32_t rows,uint32_t outputs,uint32_t inputs,bool stock) {
  padGraph(route.timed,route,input,rows,outputs,inputs); padGraph(route.probe,route,input,rows,outputs,inputs);
  const FlashDenseSmallRowsParams params{rows,(rows+route.m-1)/route.m*route.m,inputs,outputs,0,outputs,route.m,route.n};
  const auto suffix="m"+std::to_string(route.m)+"_n"+std::to_string(route.n)+"_s"+std::to_string(route.sg);
  route.timed.add(stock ? "flash_dense_small_rows_m"+std::to_string(route.m)+"_n"+std::to_string(route.n) : "sep21_bf16_"+suffix,
      {route.padded.view,weights,route.output.view,route.diag.view},params,{outputs/route.n,params.padded_rows/route.m,1},{route.sg*32,1,1});
  route.probe.add("sep21_bf16_probe_"+suffix,{route.padded.view,weights,route.output.view,route.diag.view,route.tap.view},params,
      {outputs/route.n,params.padded_rows/route.m,1},{route.sg*32,1,1});
}
void checkPadding(const Route &route,const uint16_t *input,uint32_t rows,uint32_t k) {
  const auto *words=static_cast<const uint16_t *>(route.padded.view.contents());
  for(uint64_t i=0;i<uint64_t{rows}*k;++i) require(words[i]==input[i],"BF16 padding changed source bits");
  for(uint64_t i=uint64_t{rows}*k;i<route.padded.logical/2;++i) require(words[i]==0,"padding not positivezero");
}
void usage() { std::cout<<"oracle --gpu METALLIB FIXTURES_JSON NEW_REPORT [--case N] [--rows 1|4|8|16] [--pairs 3..18]\n"
  <<"CPU mode --cpu-self-test creates no backend/model/payloadreads. Default case0, rows4, pairs9.\n"
  <<"Exact stock BF16 output parity gates candidate timing; full F64 bounds/strict failures remain separate.\n"
  <<"Root env SPLASH_FLASH_QMV_F32=1 required for active rawcontrol. Fixtures are actual prefill slices, not live decode.\n"; }
void cpuTest() {
  require(sizeof(CommandTiming)==200,"normal timing ABI mismatch");
  require(sizeof(FlashDenseSmallRowsParams)==32,"shader param ABI mismatch");
  const uint16_t input[]{bf16(1),bf16(-2),bf16(3),bf16(4)},weights[]{bf16(2),bf16(1),bf16(-1),bf16(.5f)};
  const auto ref=references(input,weights,1,1,4); require(ref[0].dot==-1 && ref[0].rounded==bf16(-1),"F64 compensated reference golden");
  const float tap=-1; const uint16_t out=bf16(-1); require(!audit(&out,&tap,&out,&tap,ref).boundFailures,"reference audit golden");
  std::array<std::array<uint32_t,3>,3> positions{};
  for(uint32_t cycle=0;cycle<9;++cycle)for(uint32_t at=0;at<3;++at)++positions[(at+cycle)%3][at];
  for(const auto &route:positions)for(uint32_t count:route)require(count==3,"compactedqualifiedtimingpositionimbalance");
  std::cout<<"{\"cpu_self_test_pass\":true,\"gpu_work\":false,\"model_payload_bytes_read\":0,\"command_timing_abi_bytes\":200}\n";
}
} // namespace

int main(int argc,char **argv) {
  @autoreleasepool { try {
    static_assert(sizeof(CommandTiming)==200);
    if(argc==2&&std::string_view(argv[1])=="--help"){usage();return 0;}
    if(argc==2&&std::string_view(argv[1])=="--cpu-self-test"){cpuTest();return 0;}
    require(argc>=5&&std::string_view(argv[1])=="--gpu","explicitRootGPUmode required; see--help");
    uint32_t index=0,rows=4,pairs=9;
    for(int i=5;i<argc;i+=2){require(i+1<argc,"missingoptionvalue");const std::string_view flag=argv[i];
      if(flag=="--case")index=parseDecimal(argv[i+1],1000);else if(flag=="--rows")rows=parseDecimal(argv[i+1],16);else if(flag=="--pairs")pairs=parseDecimal(argv[i+1],18);else require(false,"unknownoption");}
    require(rows==1||rows==4||rows==8||rows==16,"rowsmustbe1/4/8/16");require(pairs>=3,"atleast3balancedpairs");
    require(!std::filesystem::exists(argv[4]),"reportmustbenew");require(bf16_decode::switchValue("SPLASH_FLASH_QMV_F32"),"active rawF32QMVcontrol flagrequired");
    NSDictionary *metadata=loadMetadata(argv[3]);NSArray *cases=metadata[@"cases"];require([cases isKindOfClass:NSArray.class]&&index<cases.count,"fixturecaseindexinvalid");
    NSDictionary *entry=cases[index];const auto prefix=stringField(entry,"projection");const uint32_t n=integerField(entry,"output_size"),k=integerField(entry,"input_size");
    require(boolField(entry,"inputs_are_actual_prefill_capture_slices")&&!boolField(entry,"live_decode_activation_capture"),"fixtureisnotattestedactualprefillcapture");
    bf16_decode::Policy admitted;admitted.enabled=true;
    require(bf16_decode::geometry(admitted,prefix,rows,n,k),"excluded/unknownrolegeometry");
    require(prefix.find("hyper_connection")==std::string::npos,"HC excludedinitialscreen");
    const uint64_t capturedRows=integerField(entry,"captured_rows");require(capturedRows>=rows&&capturedRows<=8192,"capturedrowextentinvalid");
    MetalBackend backend(argv[2]); const uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);
    engine::MemoryGovernor governor(backend,physical-reserve,reserve);
    uint64_t planned=integerField(entry,"bf16_weights_allocated_bytes")+integerField(entry,"f32_weights_allocated_bytes")+(64ULL<<20);
    NSDictionary *rawMeta=entry[@"raw_source_tensors"];for(NSString *field in @[@"weight",@"scales",@"biases"])planned+=integerField(rawMeta[field],"length_bytes")+2*kAlignment;
    auto reservation=governor.tryReserve(planned);require(bool(reservation),"governor deniedselectedrole/scratch");const auto allocationBefore=backend.memoryStats().allocatedBytes;
    Operand bf(backend,stringField(entry,"bf16_weights_file"),integerField(entry,"bf16_weights_allocated_bytes"),uint64_t{n}*k*2,stringField(entry,"bf16_weights_sha256"),FlashDType::BF16,n,k);
    Operand f32(backend,stringField(entry,"f32_weights_file"),integerField(entry,"f32_weights_allocated_bytes"),uint64_t{n}*k*4,stringField(entry,"f32_weights_sha256"),FlashDType::F32,n,k);
    RawSource raw(backend,entry,n,k);
    auto capture=std::make_shared<ReadonlyMapping>(stringField(entry,"input_file"),capturedRows*k*2);
    const auto inputCaptureSHA=hash(capture->address(),capturedRows*k*2);require(inputCaptureSHA==stringField(entry,"input_sha256"),"capturedinputhashdiffers");
    Guarded input(backend,uint64_t{rows}*k*2);std::memcpy(input.view.contents(),capture->address(),input.logical);const auto inputSHA=hash(input.view.contents(),input.logical);
    Guarded validationOutput(backend,uint64_t{rows}*n*2),validationDiag(backend,4);
    CommandGraph validated;
    addAffine(validated,input.view,raw.projection,validationOutput.view,validationDiag.view,rows);
    const auto *bfWeights=static_cast<const uint16_t *>(bf.tensor.buffer.contents());const auto *f32Weights=static_cast<const float *>(f32.tensor.buffer.contents());
    uint64_t roundedDifferent=0;
    for(uint32_t column=0;column<n;++column)for(uint32_t j=0;j<k;++j){const uint64_t at=uint64_t{column}*k+j;const auto source=rawCoefficient(raw.projection,column,j);
      require(std::isfinite(source)&&std::bit_cast<uint32_t>(source)==std::bit_cast<uint32_t>(f32Weights[at]),"savedF32coefficientdiffersfromrawsource");
      require(bfWeights[at]==bf16(source),"savedBF16coefficientisnotexactroundedF32");roundedDifferent+=number(bfWeights[at])!=source;}
    const uint32_t tm=rows<=8?8:16,tn=n>=1024?128:64;
    std::vector<std::unique_ptr<Route>> routes;routes.push_back(std::make_unique<Route>(backend,"oldBF16",rows,n,k,tm,tn,4,false));cachedGraph(*routes.back(),input.view,bf.tensor.buffer,rows,n,k,true);
    for(uint32_t columns:{32u,64u})for(uint32_t groups:{1u,2u,4u}){const auto name="bf16_m"+std::to_string(tm)+"_n"+std::to_string(columns)+"_s"+std::to_string(groups);
      routes.push_back(std::make_unique<Route>(backend,name,rows,n,k,tm,columns,groups,true));cachedGraph(*routes.back(),input.view,bf.tensor.buffer,rows,n,k,false);}
    routes.push_back(std::make_unique<Route>(backend,"rawActiveQuantized",rows,n,k,tm,64,4,false));addAffine(routes.back()->timed,input.view,raw.projection,routes.back()->output.view,routes.back()->diag.view,rows);
    Route *rawRoute=routes.back().get();
    FlashFloatDenseSmallRowsWorkspace floatWorkspace(backend);
    std::memset(floatWorkspace.paddedInput().contents(),0xa5,floatWorkspace.paddedInput().sizeBytes());
    const bool f32Selective=bf16_decode::switchValue("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE");
    const auto floatTile=rows<2 ? std::optional<FlashFloatDenseSmallRowsTile>{} : f32Selective
        ? flashFloatDenseSmallRowsPolicy(prefix,rows,n,k,raw.projection.bits,raw.projection.groupSize)
        : std::optional<FlashFloatDenseSmallRowsTile>{rows<=8 ? FlashFloatDenseSmallRowsTile::M8N64 : FlashFloatDenseSmallRowsTile::M16N64};
    if(floatTile){routes.push_back(std::make_unique<Route>(backend,"currentF32Selected",rows,n,k,tm,64,4,false));auto &route=*routes.back();
      if(flashQSAOutF32N32Enabled()&&flashQSAOutF32N32Geometry(prefix,rows,n,k,raw.projection.bits,raw.projection.groupSize)){
        FlashFloatDenseSmallRowsParams p{rows,(rows+7)/8*8,k,n,0,n,8,64};route.timed.add("flash_float_dense_small_rows_pad",{input.view,floatWorkspace.paddedInput(),route.diag.view},p,{(uint64_t{p.padded_rows}*k+255)/256,1,1},{256,1,1});p.tile_outputs=32;
        route.timed.add("flash_qsa_out_f32_n32_m8_n32_s4",{floatWorkspace.paddedInput(),f32.tensor.buffer,route.output.view,route.diag.view},p,{n/32,p.padded_rows/8,1},{128,1,1});
      }else addFloatDenseSmallRows(backend,route.timed,input.view,f32.tensor,route.output.view,route.diag.view,rows,floatWorkspace,*floatTile);}
    require(backend.memoryStats().allocatedBytes-allocationBefore<=planned,"selectedroleexceedsadmission");reservation->commit();
    const auto ref=references(static_cast<const uint16_t *>(input.view.contents()),bfWeights,rows,n,k);
    auto &old=*routes[0];old.reset();(void)backend.submitCommand(old.timed.dispatches());old.check();old.snapshot(rows,n);checkPadding(old,static_cast<const uint16_t *>(input.view.contents()),rows,k);
    old.reset();(void)backend.submitCommand(old.probe.dispatches());old.check();require(!std::memcmp(old.output.view.contents(),old.golden.data(),old.output.logical),"oldBF16probeoutputdiffersfromshipping");
    const auto *oldTap=static_cast<const float *>(old.tap.view.contents());old.numerical=audit(static_cast<const uint16_t *>(old.output.view.contents()),oldTap,old.golden.data(),oldTap,ref);old.qualified=!old.numerical.boundFailures&&!old.numerical.signFailures&&!old.numerical.exceptional;
    std::vector<float> savedOldTap(oldTap,oldTap+uint64_t{rows}*n);
    for(size_t i=1;i<routes.size();++i){auto &route=*routes[i];route.reset();(void)backend.submitCommand(route.timed.dispatches());route.check();route.snapshot(rows,n);
      if(route.candidate){checkPadding(route,static_cast<const uint16_t *>(input.view.contents()),rows,k);route.reset();(void)backend.submitCommand(route.probe.dispatches());route.check();
        require(!std::memcmp(route.output.view.contents(),route.golden.data(),route.output.logical),"candidatetapprojectiondiffersfromtimed");
        route.numerical=audit(static_cast<const uint16_t *>(route.output.view.contents()),static_cast<const float *>(route.tap.view.contents()),old.golden.data(),savedOldTap.data(),ref);
        route.qualified=old.qualified&&!route.numerical.bf16OldMismatch&&!route.numerical.boundFailures&&!route.numerical.signFailures&&!route.numerical.exceptional;
      }else route.qualified=true;
    }
    // Poison all producer scratch; replay must reconstruct qualified outputs.
    for(auto &owned:routes)if(owned->qualified){auto &route=*owned;route.reset();(void)backend.submitCommand(route.timed.dispatches());route.check();require(!std::memcmp(route.output.view.contents(),route.golden.data(),route.output.logical),"poisonedreplayoutputdiffers");}
    std::vector<Route *> eligible;for(auto &route:routes)if(route->qualified)eligible.push_back(route.get());
    require(!eligible.empty(),"noeligiblecontrol/candidate");
    double warmGPUSeconds=0;uint64_t warmCalls=0;uint32_t warmCycles=0;
    const auto warmStart=std::chrono::steady_clock::now();
    // Real GPU work, not a count of tiny launches, drives clock warmup. No
    // model-buffer readback/reset occurs between this warmup and final timing.
    while(warmGPUSeconds<.150){
      require(warmCycles<10000,"GPUwarmupdidnotreachbounded150ms");
      for(size_t at=0;at<eligible.size();++at){auto &route=*eligible[(at+warmCycles)%eligible.size()];
        const auto timing=backend.submitCommand(route.timed.dispatches());
        require(std::isfinite(timing.gpuSeconds)&&timing.gpuSeconds>0,"invalidGPUwarmupduration");
        warmGPUSeconds+=timing.gpuSeconds;++warmCalls;
      }
      ++warmCycles;
    }
    const double warmWallSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-warmStart).count();
    const uint32_t balancedCycles=uint32_t((pairs+eligible.size()-1)/eligible.size()*eligible.size());
    for(uint32_t cycle=0;cycle<balancedCycles;++cycle)for(size_t at=0;at<eligible.size();++at){auto &route=*eligible[(at+cycle)%eligible.size()];route.addTime(backend.submitCommand(route.timed.dispatches()));}
    bool any=false;for(auto &owned:routes){auto &route=*owned;route.check();if(route.qualified){require(!std::memcmp(route.output.view.contents(),route.golden.data(),route.output.logical),"finaltimedoutputdiffers");any|=route.candidate;}}
    if(floatTile){
      uint32_t floatM=(*floatTile==FlashFloatDenseSmallRowsTile::M16N64||*floatTile==FlashFloatDenseSmallRowsTile::M16N128)?16:8;
      if(flashQSAOutF32N32Enabled()&&flashQSAOutF32N32Geometry(prefix,rows,n,k,raw.projection.bits,raw.projection.groupSize))floatM=8;
      const uint64_t used=uint64_t{(rows+floatM-1)/floatM*floatM}*k*2;
      const auto *prepared=static_cast<const uint16_t *>(floatWorkspace.paddedInput().contents());
      require(!std::memcmp(prepared,input.view.contents(),input.logical),"F32controlpadchangedBF16inputbits");
      for(uint64_t i=input.logical/2;i<used/2;++i)require(prepared[i]==0,"F32controlpaddingnotpositivezero");
      const auto *bytes=static_cast<const uint8_t *>(floatWorkspace.paddedInput().contents());
      for(uint64_t i=used;i<floatWorkspace.paddedInput().sizeBytes();++i)require(bytes[i]==0xa5,"F32controlunusedpaddingcanaryoverwritten");
    }
    require(input.guards()&&hash(input.view.contents(),input.logical)==inputSHA,"inputmutated");bf.unchanged();f32.unchanged();raw.unchanged();capture->requireUnchanged();require(hash(capture->address(),capturedRows*k*2)==inputCaptureSHA,"capturemutated");
    std::ofstream report(argv[4]);require(bool(report),"cannotcreateRootreport");report<<std::setprecision(17);
    report<<"{\"schema\":\"splash-private-bf16-narrow-selected-role-screen-sep21-v2\",\"pass\":"<<(any?"true":"false")
      <<",\"actual_prefill_capture_slices\":true,\"live_decode_activation_qualified\":false,\"model_quality_qualified\":false"
      <<",\"model_loaded\":false,\"weight_requantization\":false,\"activation_quantization\":false"
      <<",\"command_timing_abi_bytes\":"<<sizeof(CommandTiming)<<",\"metallib_sha256\":"<<json::quote(fileHash(argv[2]))
      <<",\"oracle_sha256\":"<<json::quote(fileHash(argv[0]))<<",\"fixture_metadata_sha256\":"<<json::quote(fileHash(argv[3]))
      <<",\"bf16_operand_sha256\":"<<json::quote(bf.initialSHA)<<",\"f32_operand_sha256\":"<<json::quote(f32.initialSHA)
      <<",\"f32_control_selective\":"<<(f32Selective?"true":"false")<<",\"requested_timing_cycles\":"<<pairs<<",\"balanced_timing_cycles\":"<<balancedCycles<<",\"eligible_timed_routes\":"<<eligible.size()
      <<",\"warmup_gpu_seconds\":"<<warmGPUSeconds<<",\"warmup_wall_seconds\":"<<warmWallSeconds<<",\"warmup_commands\":"<<warmCalls<<",\"warmup_rotations\":"<<warmCycles
      <<",\"projection\":"<<json::quote(prefix)<<",\"fixture_case\":"<<index<<",\"rows\":"<<rows<<",\"input_size\":"<<k<<",\"output_size\":"<<n
      <<",\"source_bits\":"<<raw.projection.bits<<",\"source_group_size\":"<<raw.projection.groupSize<<",\"rounded_bf16_coefficients_differ_from_f32\":"<<roundedDifferent
      <<",\"captured_input_sha256\":"<<json::quote(inputCaptureSHA)<<",\"input_slice_sha256\":"<<json::quote(inputSHA)<<",\"input_slice_base_mod128\":"<<(reinterpret_cast<uintptr_t>(input.view.contents())%128)
      <<",\"baseline_f64_strict_pass\":"<<(!old.numerical.strictFailures&&old.qualified?"true":"false")
      <<",\"timing_gate\":\"exact stock cached-BF16 output parity plus full F64 absolute/finite/sign bounds and guards/sticky/inputimmutability; baseline strict F64 failures remainfailed independent axis\""
      <<",\"timing_scope\":\"atleast150ms cumulativeGPUwork then positionbalanced compactqualifiedroute rotations, noCPUmodelbufferreadbacks/resetfromwarmuptoendtiming; eachBF16timedchain includes stockpad and oneprojection; untimedF32tap/fullF64/sourcevalidation/hashes excluded\",\"routes\":[";
    for(size_t i=0;i<routes.size();++i){if(i)report<<',';const auto &route=*routes[i];report<<"{\"name\":"<<json::quote(route.name)<<",\"candidate\":"<<(route.candidate?"true":"false")<<",\"timing_allowed\":"<<(route.qualified?"true":"false")<<",\"numerical\":";route.numerical.write(report);
      uint64_t rawMismatch=0;for(size_t j=0;j<route.golden.size();++j)rawMismatch+=route.golden[j]!=rawRoute->golden[j];report<<",\"raw_control_bf16_output_mismatches\":"<<rawMismatch<<",\"full_f64_bf16_coefficient_reference_output_difference\":";coefficientDifference(report,route.golden,ref);
      if(!route.gpu.empty()){report<<",\"median_gpu_seconds\":"<<median(route.gpu)<<",\"median_wall_seconds\":"<<median(route.wall)<<",\"samples\":[";for(size_t j=0;j<route.gpu.size();++j){if(j)report<<',';report<<"{\"gpu_seconds\":"<<route.gpu[j]<<",\"wall_seconds\":"<<route.wall[j]<<'}';}report<<']';}report<<'}';}
    report<<"]}\n";require(bool(report),"reportwritefailed");std::cout<<"{\"report\":"<<json::quote(argv[4])<<",\"qualified_candidate\":"<<(any?"true":"false")<<",\"model_loaded\":false}\n";return any?0:2;
  }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;} }
}
