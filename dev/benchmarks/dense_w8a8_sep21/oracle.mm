// Numerical-alternative standalone W8A8 dense prefill. Only Root may invoke --run.
// Exactly one requested actual role is read; no sidecar/cache is written.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "metal/CommandGraph.hpp"
#include "abi.hpp"
#include "quantization.hpp"
#include "flash/FlashPrefillDenseTiles.hpp"
#include "engine/Json.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::metal;
constexpr uint16_t sentinel = 0x7fc1;
constexpr uint32_t sticky = 0x40000000;
constexpr uint64_t guard = 64;
void require(bool ok, const std::string &what) { if (!ok) throw std::runtime_error(what); }
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
uint16_t bf16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
struct Error {
  uint64_t elements = 0, mismatches = 0, nonfinite = 0, signFlips = 0, nearZero = 0;
  double maxAbs = 0, squaredError = 0, squaredReference = 0, squaredActual = 0, product = 0, nearZeroAbs = 0;
  void add(uint16_t actual, uint16_t expected) {
    ++elements; mismatches += actual != expected;
    const double a = number(actual), b = number(expected);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; return; }
    const double delta = a - b;
    maxAbs = std::max(maxAbs, std::abs(delta)); squaredError += delta * delta;
    squaredReference += b * b; squaredActual += a * a; product += a * b;
    signFlips += a && b && std::signbit(a) != std::signbit(b);
    if (std::abs(b) <= 1e-3) { ++nearZero; nearZeroAbs = std::max(nearZeroAbs,std::abs(delta)); }
  }
  double relativeL2() const { return std::sqrt(squaredError / std::max(1e-30, squaredReference)); }
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"bf16_mismatches\":" << mismatches
        << ",\"nonfinite\":" << nonfinite << ",\"max_abs\":" << maxAbs << ",\"relative_l2\":" << relativeL2()
        << ",\"cosine\":" << product / std::sqrt(std::max(1e-30,squaredReference*squaredActual))
        << ",\"sign_flips\":" << signFlips << ",\"near_zero_elements\":" << nearZero
        << ",\"near_zero_max_abs\":" << nearZeroAbs << '}';
  }
};
std::string sha256(const void *p, uint64_t bytes) {
  require(bytes <= UINT32_MAX,"SHA256 input exceeds bound"); unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  require(CC_SHA256(p,CC_LONG(bytes),digest),"SHA256 failed"); std::ostringstream out;
  for (unsigned char v : digest) out << std::hex << std::setfill('0') << std::setw(2) << unsigned(v);
  return out.str();
}
void readExact(const std::string &path, void *p, uint64_t bytes, const std::string &digest) {
  require(std::filesystem::file_size(path) == bytes,"fixture extent mismatch: " + path);
  std::ifstream in(path,std::ios::binary); require(bool(in),"fixture open failed");
  in.read(static_cast<char *>(p),std::streamsize(bytes)); require(bool(in),"fixture read failed");
  require(digest.empty() || sha256(p,bytes) == digest,"fixture SHA256 mismatch: " + path);
}
struct Shape {
  uint32_t rows = 2048, k = 0, n = 0;
  std::string projection, weightPath, weightSHA, inputPath, inputSHA, expectedPath, expectedSHA;
};
std::vector<Shape> loadFixtures(const std::string &path, const std::string &filter) {
  NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data != nil,"manifest read failed"); NSError *error = nil;
  NSDictionary *document = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(!error && [document isKindOfClass:[NSDictionary class]],"invalid fixture JSON");
  std::vector<Shape> shapes;
  for (NSDictionary *entry in document[@"cases"]) {
    const auto str = [&](NSString *key) { id v = entry[key]; require([v isKindOfClass:[NSString class]],"missing manifest string"); return std::string([v UTF8String]); };
    Shape s; s.rows = [entry[@"rows"] unsignedIntValue]; s.k = [entry[@"input_size"] unsignedIntValue]; s.n = [entry[@"output_size"] unsignedIntValue];
    s.projection = str(@"projection"); s.weightPath = str(@"weights_file"); s.weightSHA = str(@"weights_sha256");
    s.inputPath = str(@"input_file"); s.inputSHA = str(@"input_sha256"); s.expectedPath = str(@"expected_file"); s.expectedSHA = str(@"expected_sha256");
    require(s.rows == 2048 && s.k && s.k <= 32768 && !(s.k%32) && s.n && s.n <= 32768 && !(s.n%64),"unsupported captured shape");
    if (filter.empty() || s.projection.find(filter) != std::string::npos) shapes.push_back(s);
  }
  require(!shapes.empty(),"empty fixture selection"); return shapes;
}
DispatchSize traversal(uint32_t rows, uint32_t columns, uint32_t mode) {
  if (!mode) return {columns,rows,1}; if (mode == 1) return {rows,columns,1};
  const uint32_t width = 1u << (mode - 1); return {uint64_t(columns)*width,(rows+width-1)/width,1};
}
namespace qw = splash::dense_w8a8;
constexpr uint8_t quantGuard = 0x80;
constexpr uint32_t floatGuard = 0x7fc10001, integerGuard = 0x6e7152a3;
constexpr double relativeL2Limit = .02, cosineLimit = .9998;
std::vector<uint32_t> groupsList(std::string value) {
  std::stringstream in(value); std::string word; std::vector<uint32_t> result;
  while (std::getline(in,word,',')) { const auto v = std::stoul(word); require(v == 4 || v == 8,"groups must be 4 or 8"); result.push_back(uint32_t(v)); }
  require(!result.empty(),"empty groups"); std::sort(result.begin(),result.end()); result.erase(std::unique(result.begin(),result.end()),result.end()); return result;
}
struct Variant {
  std::string name, pipeline, outputSHA; uint32_t groups = 0, mode = 0; bool quantized = false;
  std::vector<double> gpu, wall, pairedBaselineGPU, pairedBaselineWall, pairedSpeedup;
  std::vector<std::array<uint32_t,2>> positionOrderCounts;
  uint32_t ab = 0, ba = 0, warmAB = 0, warmBA = 0;
  double warmGPU = 0, warmBaselineGPU = 0, warmWall = 0;
  Error baselineError, sourceOracleError, decompOracleError, sourceQuantizedOracleError;
  uint64_t integerSamples = 0, integerMismatches = 0, identityElements = 0, identityMismatches = 0;
  bool qualityPassed = false;
};
std::vector<Variant> variants(const Shape &s,const std::vector<uint32_t> &groups) {
  const auto plan = splash::flash::flashPrefillDenseTilePolicy(s.rows,s.n,s.k); require(bool(plan),"missing production plan");
  Variant selected; selected.name = "selected_production_bf16"; selected.groups = plan.simdGroups; selected.mode = uint32_t(plan.traversal);
  selected.pipeline = "flash_dense_cache_prefill_m128_n64_sg" + std::to_string(plan.simdGroups);
  std::vector<Variant> choices{selected};
  for (uint32_t g : groups) { Variant v; v.quantized = true; v.groups = g; v.mode = selected.mode; v.pipeline = "dense_w8a8_m128_n64_sg" + std::to_string(g); v.name = v.pipeline; choices.push_back(v); }
  return choices;
}
struct Buffers {
  MetalBuffer input, weights, quantInputBase, quantInput, quantWeightsBase, quantWeights;
  MetalBuffer inputScalesBase, inputScales, weightScalesBase, weightScales;
  MetalBuffer integerBase, integerDot, outputBase, output, diagnostics;
  uint64_t elements = 0; std::string inputSHA, weightSHA, quantWeightSHA, weightScalesSHA;
  qw::QuantError coefficientError;
  static MetalBuffer guarded(MetalBackend &backend,MetalBuffer &base,uint64_t bytes,uint32_t width) {
    base = backend.allocateBuffer(bytes+2*guard*width,BufferStorage::Shared); return backend.view(base,guard*width,bytes);
  }
  Buffers(MetalBackend &backend,const Shape &s) {
    input = backend.allocateBuffer(uint64_t(s.rows)*s.k*2,BufferStorage::Shared);
    weights = backend.allocateBuffer(uint64_t(s.n)*s.k*2,BufferStorage::Shared);
    quantInput = guarded(backend,quantInputBase,uint64_t(s.rows)*s.k,1);
    quantWeights = guarded(backend,quantWeightsBase,uint64_t(s.n)*s.k,1);
    inputScales = guarded(backend,inputScalesBase,uint64_t(s.rows)*4,4);
    weightScales = guarded(backend,weightScalesBase,uint64_t(s.n)*4,4);
    elements = uint64_t(s.rows)*s.n; integerDot = guarded(backend,integerBase,elements*4,4); allocateOutput(backend);
    std::memset(quantInputBase.contents(),quantGuard,quantInputBase.sizeBytes()); std::memset(quantWeightsBase.contents(),quantGuard,quantWeightsBase.sizeBytes());
    std::fill_n(static_cast<uint32_t *>(inputScalesBase.contents()),inputScalesBase.sizeBytes()/4,floatGuard);
    std::fill_n(static_cast<uint32_t *>(weightScalesBase.contents()),weightScalesBase.sizeBytes()/4,floatGuard);
    std::fill_n(static_cast<uint32_t *>(integerBase.contents()),integerBase.sizeBytes()/4,integerGuard);
    if (s.inputPath.empty()) std::fill_n(static_cast<uint16_t *>(input.contents()),input.sizeBytes()/2,bf16(.125f));
    else readExact(s.inputPath,input.contents(),input.sizeBytes(),s.inputSHA);
    if (s.weightPath.empty()) std::fill_n(static_cast<uint16_t *>(weights.contents()),weights.sizeBytes()/2,bf16(.015625f));
    else readExact(s.weightPath,weights.contents(),weights.sizeBytes(),s.weightSHA);
    const auto *w = static_cast<const uint16_t *>(weights.contents()); auto *q = static_cast<int8_t *>(quantWeights.contents()); auto *scales = static_cast<float *>(weightScales.contents());
    for (uint32_t row = 0; row < s.n; ++row) qw::quantizeRow(w+uint64_t(row)*s.k,s.k,q+uint64_t(row)*s.k,scales[row],coefficientError);
    // Finite source input and representable positive quantization scales are
    // prerequired. This CPU check never writes activation codes into the GPU path.
    const auto *x = static_cast<const uint16_t *>(input.contents());
    for (uint32_t row = 0; row < s.rows; ++row) (void)qw::symmetricScale(x+uint64_t(row)*s.k,s.k);
    inputSHA = sha256(input.contents(),input.sizeBytes()); weightSHA = sha256(weights.contents(),weights.sizeBytes());
    quantWeightSHA = sha256(quantWeights.contents(),quantWeights.sizeBytes()); weightScalesSHA = sha256(weightScales.contents(),weightScales.sizeBytes());
  }
  void allocateOutput(MetalBackend &backend) {
    output = guarded(backend,outputBase,elements*2,2); diagnostics = backend.allocateBuffer(64,BufferStorage::Shared); reset();
  }
  void reset() {
    std::fill_n(static_cast<uint16_t *>(outputBase.contents()),outputBase.sizeBytes()/2,sentinel);
    std::memset(diagnostics.contents(),0,diagnostics.sizeBytes()); *static_cast<uint32_t *>(diagnostics.contents()) = sticky;
  }
  static void canary(const MetalBuffer &base,uint64_t elements,uint32_t width,uint32_t value) {
    for (uint64_t i = 0; i < guard; ++i) {
      if (width == 1) { const auto *p = static_cast<const uint8_t *>(base.contents()); require(p[i] == value && p[guard+elements+i] == value,"I8 operand/converter guard changed"); }
      else if (width == 2) { const auto *p = static_cast<const uint16_t *>(base.contents()); require(p[i] == value && p[guard+elements+i] == value,"BF16 output guard changed"); }
      else { const auto *p = static_cast<const uint32_t *>(base.contents()); require(p[i] == value && p[guard+elements+i] == value,"F32 scale/I32 probe guard changed"); }
    }
  }
  void guards() const {
    canary(outputBase,elements,2,sentinel); canary(quantInputBase,quantInput.sizeBytes(),1,quantGuard); canary(quantWeightsBase,quantWeights.sizeBytes(),1,quantGuard);
    canary(inputScalesBase,inputScales.sizeBytes()/4,4,floatGuard); canary(weightScalesBase,weightScales.sizeBytes()/4,4,floatGuard); canary(integerBase,elements,4,integerGuard);
    require(*static_cast<const uint32_t *>(diagnostics.contents()) == sticky,"shader diagnostics changed");
  }
  std::vector<uint16_t> result() const { const auto *p = static_cast<const uint16_t *>(output.contents()); return {p,p+elements}; }
  void immutable() const {
    require(sha256(input.contents(),input.sizeBytes()) == inputSHA && sha256(weights.contents(),weights.sizeBytes()) == weightSHA &&
        sha256(quantWeights.contents(),quantWeights.sizeBytes()) == quantWeightSHA && sha256(weightScales.contents(),weightScales.sizeBytes()) == weightScalesSHA,"immutable BF16 source/I8 coefficients/F32 weight scales changed");
  }
};
CommandGraph graphFor(const Variant &v,const Buffers &b,const Shape &s,uint32_t repeat,bool probe = false) {
  CommandGraph graph;
  for (uint32_t r = 0; r < repeat; ++r) {
    const FlashDenseCacheParams p{s.rows,s.k,s.n,0,s.n,128,64,v.mode};
    if (v.quantized) {
      const DenseW8A8QuantizeParams q{s.rows,s.k};
      graph.add("dense_w8a8_bf16_to_i8_t256",{b.input,b.quantInput,b.inputScales,b.diagnostics},q,{s.rows,1,1},{256,1,1});
      graph.add(v.pipeline+(probe ? "_probe" : ""),{b.quantInput,b.quantWeights,b.output,b.integerDot,b.inputScales,b.weightScales,b.diagnostics},p,
          traversal(s.rows/128,s.n/64,v.mode),{v.groups*32u,1,1});
    } else graph.add(v.pipeline,{b.input,b.weights,b.output,b.diagnostics},p,traversal(s.rows/128,s.n/64,v.mode),{v.groups*32u,1,1});
  }
  return graph;
}
struct ActivationCertificate {
  qw::QuantError error; uint64_t scaleElements = 0, scaleMismatches = 0, codeElements = 0, codeMismatches = 0, forbiddenMinus128 = 0;
  void write(std::ostream &out) const {
    out << "{\"scale_elements\":" << scaleElements << ",\"scale_bit_mismatches\":" << scaleMismatches << ",\"code_elements\":" << codeElements
        << ",\"rne_code_mismatches\":" << codeMismatches << ",\"forbidden_minus128_codes\":" << forbiddenMinus128 << ",\"source_fp64_dequantization_error\":";
    error.writeJSON(out); out << '}';
  }
};
ActivationCertificate activationCertificate(const Buffers &b,const Shape &s) {
  ActivationCertificate c; const auto *x = static_cast<const uint16_t *>(b.input.contents()); const auto *q = static_cast<const int8_t *>(b.quantInput.contents());
  const auto *scales = static_cast<const float *>(b.inputScales.contents());
  for (uint32_t row = 0; row < s.rows; ++row) {
    const float expected = qw::symmetricScale(x+uint64_t(row)*s.k,s.k), actual = scales[row]; ++c.scaleElements;
    c.scaleMismatches += std::bit_cast<uint32_t>(actual) != std::bit_cast<uint32_t>(expected);
    require(std::isfinite(actual) && actual > 0,"invalid GPU activation scale");
    for (uint32_t k = 0; k < s.k; ++k) {
      const uint64_t i = uint64_t(row)*s.k+k; const float source = number(x[i]); ++c.codeElements;
      c.codeMismatches += q[i] != qw::symmetricCode(source,actual); c.forbiddenMinus128 += q[i] == -128;
      c.error.add(double(source),double(q[i])*actual,q[i]);
    }
  }
  require(!c.scaleMismatches && !c.codeMismatches && !c.forbiddenMinus128,"GPU activation rowmax/F32 scale/RNE I8 certificate failed"); return c;
}
void integerAndDecompositionCertificate(Variant &v,const Buffers &b,const Shape &s,const std::vector<uint16_t> &actual) {
  const auto *x = static_cast<const uint16_t *>(b.input.contents()), *w = static_cast<const uint16_t *>(b.weights.contents());
  const auto *a = static_cast<const int8_t *>(b.quantInput.contents()), *q = static_cast<const int8_t *>(b.quantWeights.contents());
  const auto *sa = static_cast<const float *>(b.inputScales.contents()), *sw = static_cast<const float *>(b.weightScales.contents());
  const auto *dot = static_cast<const int32_t *>(b.integerDot.contents());
  const int64_t bound = int64_t(s.k)*127*127; require(bound < INT32_MAX,"integer accumulator overflow bound failed");
  for (uint32_t row = 0; row < s.rows; ++row) for (uint32_t column = 0; column < s.n; ++column) {
    const uint64_t i = uint64_t(row)*s.n+column; require(std::abs(int64_t(dot[i])) <= bound,"integer probe exceeded exact accumulation bound");
    const float scaledInput = float(dot[i])*sa[row], scaled = scaledInput*sw[column];
    ++v.identityElements; v.identityMismatches += actual[i] != bf16(scaled);
  }
  require(!v.identityMismatches,"full late-F32 scaling/BF16 identity certificate failed");
  for (uint32_t ri = 0; ri < 8; ++ri) for (uint32_t ci = 0; ci < 32; ++ci) {
    const uint32_t row = uint32_t(uint64_t(ri)*(s.rows-1)/7), column = uint32_t(uint64_t(ci)*(s.n-1)/31); int64_t integer = 0; double source = 0, dequantized = 0;
    for (uint32_t k = 0; k < s.k; ++k) {
      const uint64_t ai = uint64_t(row)*s.k+k, bi = uint64_t(column)*s.k+k;
      integer += int64_t(a[ai])*q[bi]; source += double(number(x[ai]))*number(w[bi]);
      dequantized += (double(a[ai])*sa[row])*(double(q[bi])*sw[column]);
    }
    ++v.integerSamples; v.integerMismatches += integer != dot[uint64_t(row)*s.n+column];
    const double late = double(integer)*sa[row]*sw[column];
    require(std::abs(dequantized-late) <= 1e-10*std::max(1.,std::abs(late)),"FP64 integer-dot/dequantized-product decomposition identity failed");
    v.sourceOracleError.add(actual[uint64_t(row)*s.n+column],bf16(float(source)));
    v.decompOracleError.add(actual[uint64_t(row)*s.n+column],bf16(float(late)));
    v.sourceQuantizedOracleError.add(bf16(float(late)),bf16(float(source)));
  }
  require(!v.integerMismatches && !v.decompOracleError.nonfinite && v.decompOracleError.relativeL2() <= .004,"exact integer dot or sampled FP64 decomp certificate failed");
}
uint64_t boundaryTests(MetalBackend &backend,const std::vector<Variant> &choices) {
  Shape s; s.rows = 2048; s.k = 32; s.n = 320; Buffers b(backend,s); uint64_t checks = 0;
  for (const auto &v : choices) {
    b.reset(); auto graph = graphFor(v,b,s,1,v.quantized); (void)backend.submitCommand(graph.dispatches()); b.guards(); const auto actual = b.result();
    require(std::all_of(actual.begin(),actual.end(),[](uint16_t x) { return x == bf16(.0625f); }),"synthetic integer max-code/late-scale identity failed");
    if (v.quantized) {
      (void)activationCertificate(b,s); const auto *dot = static_cast<const int32_t *>(b.integerDot.contents());
      require(std::all_of(dot,dot+b.elements,[](int32_t x) { return x == 32*127*127; }),"synthetic exact signed integer dot failed");
    }
    ++checks;
    for (uint32_t which = 0; which < 9; ++which) {
      FlashDenseCacheParams p{s.rows,s.k,s.n,0,s.n,128,64,v.mode}; DispatchSize threads{v.groups*32u,1,1};
      switch (which) { case 0: p.rows = 0; break; case 1: p.rows = 2049; break; case 2: p.input_size = 31; break; case 3: p.output_begin = 321; break;
      case 4: p.output_count = 32; break; case 5: p.tile_rows = 64; break; case 6: p.reserved = 5; break; case 7: p.tile_outputs = 128; break; default: threads.x /= 2; break; }
      b.reset(); CommandGraph bad;
      if (v.quantized) bad.add(v.pipeline,{b.quantInput,b.quantWeights,b.output,b.integerDot,b.inputScales,b.weightScales,b.diagnostics},p,{1,1,1},threads);
      else bad.add(v.pipeline,{b.input,b.weights,b.output,b.diagnostics},p,{1,1,1},threads);
      (void)backend.submitCommand(bad.dispatches()); require(*static_cast<const uint32_t *>(b.diagnostics.contents()) == (sticky|2),"invalid matmul parameters did not set sticky error");
      const auto *all = static_cast<const uint16_t *>(b.outputBase.contents()); require(std::all_of(all,all+b.outputBase.sizeBytes()/2,[](uint16_t x) { return x == sentinel; }),"invalid matmul wrote output"); ++checks;
    }
  }
  b.immutable(); return checks;
}
Error compare(const std::vector<uint16_t> &actual,const std::vector<uint16_t> &expected) {
  require(actual.size() == expected.size(),"output extent mismatch"); Error e;
  for (uint64_t i = 0; i < actual.size(); ++i) e.add(actual[i],expected[i]); return e;
}
Error scalarOracle(const Buffers &b,const Shape &s,const std::vector<uint16_t> &actual) {
  const auto *x = static_cast<const uint16_t *>(b.input.contents()), *w = static_cast<const uint16_t *>(b.weights.contents()); Error e;
  for (uint32_t ri = 0; ri < 8; ++ri) for (uint32_t ci = 0; ci < 32; ++ci) {
    const uint32_t row = uint32_t(uint64_t(ri)*(s.rows-1)/7), column = uint32_t(uint64_t(ci)*(s.n-1)/31);
    double dot = 0; for (uint32_t k = 0; k < s.k; ++k) dot += double(number(x[uint64_t(row)*s.k+k])) * number(w[uint64_t(column)*s.k+k]);
    e.add(actual[uint64_t(row)*s.n+column],bf16(float(dot)));
  }
  return e;
}
double median(std::vector<double> values) {
  require(!values.empty(),"empty timings"); std::sort(values.begin(),values.end()); const auto m = values.size()/2;
  return values.size()%2 ? values[m] : (values[m-1]+values[m])/2;
}
void array(std::ostream &out,const std::vector<double> &values) {
  out << '['; for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; } out << ']';
}
void cpuSelfTest() {
  uint64_t checks = 0;
  for (uint32_t word = 0; word < 65536; ++word) if (std::isfinite(number(uint16_t(word)))) { require(bf16(number(uint16_t(word))) == word,"BF16 finite round-trip failed"); ++checks; }
  for (uint32_t k : {2560u,6144u,32768u}) { require(uint64_t(k)*16129 < INT32_MAX,"I32 bound failed"); ++checks; }
  for (uint32_t count : {1u,2u}) for (uint32_t cycles = 1; cycles <= 8; ++cycles) {
    std::vector<std::vector<std::array<uint32_t,2>>> counts(count,std::vector<std::array<uint32_t,2>>(count));
    for (uint32_t sample = 0; sample < 2*count*cycles; ++sample) for (uint32_t position = 0; position < count; ++position)
      ++counts[(position+(sample/2)%count)%count][position][sample%2];
    for (const auto &candidate : counts) for (const auto &position : candidate) require(position[0] == cycles && position[1] == cycles,"balanced position/order stratum failed"); ++checks;
  }
  std::cout << "{\"cpu_self_test\":\"passed\",\"checks\":" << checks << ",\"gpu_created\":false,\"payloads_read\":false}\n";
}
} // namespace
int main(int argc,char **argv) {
  try {
    bool run = false, preview = false; std::string library = "build/dense-w8a8-sep21/w8a8.metallib", fixture = "build/prefill-dense-sep21/actual-captures.json", output, filter = "linear_attn.in_proj_qkv";
    auto groups = groupsList("4,8"); uint32_t requestedSamples = 10, repeat = 4; double warmMilliseconds = 150;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i]; if (arg == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if (arg == "--run") { run = true; continue; } if (arg == "--preview") { preview = true; continue; }
      require(i+1 < argc,"missing CLI value"); const std::string value = argv[++i];
      if (arg == "--library") library = value; else if (arg == "--fixture-manifest") fixture = value; else if (arg == "--out") output = value;
      else if (arg == "--projection-filter") filter = value; else if (arg == "--groups") groups = groupsList(value);
      else if (arg == "--samples") requestedSamples = uint32_t(std::stoul(value)); else if (arg == "--repeat") repeat = uint32_t(std::stoul(value));
      else if (arg == "--warm-ms") warmMilliseconds = std::stod(value); else throw std::runtime_error("unknown argument: " + arg);
    }
    require(run || preview,"GPU/model-payload work requires explicit --run"); require(requestedSamples >= 2 && requestedSamples <= 100 && repeat && repeat <= 64,"invalid sample/repeat count");
    require(std::isfinite(warmMilliseconds) && warmMilliseconds >= 150 && warmMilliseconds <= 1000,"GPU-only warm minimum must be 150..1000 ms per candidate and matched control");
    const auto shapes = loadFixtures(fixture,filter); require(shapes.size() == 1,"bounded primitive requires exactly one captured role per run"); const auto &s = shapes[0];
    require(s.projection.ends_with(".linear_attn.in_proj_qkv") || s.projection.ends_with(".linear_attn.in_proj_z") || s.projection.ends_with(".linear_attn.out_proj") || s.projection.ends_with(".self_attn.q_proj"),"only QKV,Z,GDNout,QSAq roles are authorized");
    require(s.k == 2560 || s.k == 6144,"unexpected role K"); auto choices = variants(s,groups); const uint32_t count = uint32_t(choices.size()-1), strata = 2*count;
    const uint32_t samples = (requestedSamples+strata-1)/strata*strata;
    if (preview) {
      std::cout << "{\"preview_only\":true,\"gpu_created\":false,\"payloads_read\":false,\"numerical_alternative\":true,\"projection\":" << splash::json::quote(s.projection)
          << ",\"requested_samples\":" << requestedSamples << ",\"effective_balanced_samples\":" << samples << ",\"minimum_gpu_warm_ms_each_candidate_and_matched_control\":" << warmMilliseconds << ",\"variants\":[";
      for (size_t i = 0; i < choices.size(); ++i) { if (i) std::cout << ','; std::cout << splash::json::quote(choices[i].name); }
      std::cout << "]}\n"; return 0;
    }
    MetalBackend backend(library); const uint64_t boundaries = boundaryTests(backend,choices); Buffers operands(backend,s); std::vector<Buffers> buffers; buffers.reserve(choices.size());
    for (size_t i = 0; i < choices.size(); ++i) { Buffers b = operands; b.allocateOutput(backend); buffers.push_back(std::move(b)); }
    std::vector<uint16_t> captured(uint64_t(s.rows)*s.n); readExact(s.expectedPath,captured.data(),captured.size()*2,s.expectedSHA);
    std::vector<uint16_t> baseline; std::vector<std::vector<uint16_t>> warmed; ActivationCertificate activation;
    for (size_t i = 0; i < choices.size(); ++i) {
      auto &v = choices[i]; auto &b = buffers[i]; auto warm = graphFor(v,b,s,1); (void)backend.submitCommand(warm.dispatches()); b.guards(); const auto actual = b.result();
      b.reset(); (void)backend.submitCommand(warm.dispatches()); b.guards(); require(!compare(b.result(),actual).mismatches,"deterministic complete output changed");
      if (!i) { baseline = actual; require(!compare(actual,captured).mismatches,"selected production differs from captured output"); }
      v.baselineError = compare(actual,baseline); v.outputSHA = sha256(actual.data(),actual.size()*2);
      if (v.quantized) {
        activation = activationCertificate(b,s); auto probe = graphFor(v,b,s,1,true); b.reset(); (void)backend.submitCommand(probe.dispatches()); b.guards();
        require(!compare(b.result(),actual).mismatches,"probe output differs from no-probe scaled candidate"); integerAndDecompositionCertificate(v,b,s,actual);
        v.qualityPassed = !v.baselineError.nonfinite && v.baselineError.relativeL2() <= relativeL2Limit &&
            v.baselineError.product/std::sqrt(std::max(1e-30,v.baselineError.squaredReference*v.baselineError.squaredActual)) >= cosineLimit;
      } else {
        v.sourceOracleError = scalarOracle(b,s,actual);
        require(!v.sourceOracleError.nonfinite && v.sourceOracleError.relativeL2() <= .004,"captured BF16 control failed sampled original FP64 oracle");
        v.qualityPassed = true;
      }
      warmed.push_back(actual); v.positionOrderCounts.resize(count);
    }
    const std::string integerBefore = sha256(operands.integerDot.contents(),operands.integerDot.sizeBytes());
    std::vector<CommandGraph> commands; commands.reserve(choices.size()); for (size_t i = 0; i < choices.size(); ++i) commands.push_back(graphFor(choices[i],buffers[i],s,repeat));
    // From here until every timing pair is complete, no CPU operand/output/
    // diagnostic/guard access occurs. Warmup is also balanced AB/BA.
    for (size_t i = 1; i < choices.size(); ++i) {
      auto &v = choices[i]; uint32_t rounds = 0;
      while (v.warmGPU < warmMilliseconds || v.warmBaselineGPU < warmMilliseconds) {
        require(++rounds <= 10000,"GPU warmup did not accumulate usable timestamps");
        const auto a = backend.submitCommand(commands[0].dispatches()), b = backend.submitCommand(commands[i].dispatches());
        const auto c = backend.submitCommand(commands[i].dispatches()), d = backend.submitCommand(commands[0].dispatches());
        require(a.gpuSeconds > 0 && b.gpuSeconds > 0 && c.gpuSeconds > 0 && d.gpuSeconds > 0,"GPU timestamps unavailable during minimum warmup");
        ++v.warmAB; ++v.warmBA; v.warmGPU += (b.gpuSeconds+c.gpuSeconds)*1000; v.warmBaselineGPU += (a.gpuSeconds+d.gpuSeconds)*1000;
        v.warmWall += (a.wallSeconds+b.wallSeconds+c.wallSeconds+d.wallSeconds)*1000;
      }
    }
    for (uint32_t sample = 0; sample < samples; ++sample) {
      const uint32_t rotation = (sample/2)%count, order = sample%2;
      for (uint32_t position = 0; position < count; ++position) {
        const size_t i = 1+(position+rotation)%count; CommandTiming baseTime, candidateTime;
        if (!order) { baseTime = backend.submitCommand(commands[0].dispatches()); candidateTime = backend.submitCommand(commands[i].dispatches()); ++choices[i].ab; }
        else { candidateTime = backend.submitCommand(commands[i].dispatches()); baseTime = backend.submitCommand(commands[0].dispatches()); ++choices[i].ba; }
        ++choices[i].positionOrderCounts[position][order]; const double baseGPU = baseTime.gpuSeconds*1000/repeat, gpu = candidateTime.gpuSeconds*1000/repeat;
        choices[i].gpu.push_back(gpu); choices[i].wall.push_back(candidateTime.wallSeconds*1000/repeat); choices[i].pairedBaselineGPU.push_back(baseGPU);
        choices[i].pairedBaselineWall.push_back(baseTime.wallSeconds*1000/repeat); choices[i].pairedSpeedup.push_back(baseGPU/gpu); choices[0].gpu.push_back(baseGPU); choices[0].wall.push_back(baseTime.wallSeconds*1000/repeat);
      }
    }
    for (size_t i = 0; i < choices.size(); ++i) {
      buffers[i].guards(); require(!compare(buffers[i].result(),warmed[i]).mismatches,"timed full output differs from qualified output");
      if (i) { require(choices[i].ab == samples/2 && choices[i].ba == samples/2,"AB/BA order imbalance"); for (const auto &position : choices[i].positionOrderCounts) require(position[0] == samples/strata && position[1] == samples/strata,"candidate position/order imbalance"); }
    }
    (void)activationCertificate(operands,s); operands.immutable(); require(sha256(operands.integerDot.contents(),operands.integerDot.sizeBytes()) == integerBefore,"normal timed candidate modified untimed I32 probe buffer");
    std::ostringstream report; report << std::setprecision(10) << "{\"experiment\":\"dense_w8a8_sep21_v1\",\"numerical_alternative\":true,\"full_model_quality_qualified\":false,\"production_overlay\":false,"
        << "\"weight_quantization\":\"symmetric_per_output_row_signed_i8_f32_scale\",\"activation_quantization\":\"gpu_per_input_rowmax_rne_signed_i8_f32_scale\","
        << "\"integer_destination\":\"i32\",\"whole_k_integer_bound\":" << uint64_t(s.k)*16129 << ",\"activation_converter_in_every_timed_projection\":true,\"probe_i32_writes_in_timing\":false,"
        << "\"quality_preregister\":{\"full_output_relative_l2_max\":" << relativeL2Limit << ",\"full_output_cosine_min\":" << cosineLimit << "},"
        << "\"boundary_checks\":" << boundaries << ",\"captured_control_exact\":true,\"guards_passed\":true,\"immutable_coefficients_passed\":true,\"probe_immutable_during_timing\":true,"
        << "\"cpu_operand_output_access_in_warm_or_timing\":false,\"requested_samples\":" << requestedSamples << ",\"effective_balanced_samples\":" << samples << ",\"repeat\":" << repeat
        << ",\"minimum_gpu_warm_ms_each_candidate_and_matched_control\":" << warmMilliseconds << ",\"projection\":" << splash::json::quote(s.projection) << ",\"rows\":" << s.rows << ",\"input_size\":" << s.k << ",\"output_size\":" << s.n
        << ",\"source_weight_sha256\":" << splash::json::quote(s.weightSHA) << ",\"source_input_sha256\":" << splash::json::quote(s.inputSHA) << ",\"weight_quantization_error\":"; operands.coefficientError.writeJSON(report);
    report << ",\"activation_certificate\":"; activation.write(report); report << ",\"variants\":[";
    for (size_t i = 0; i < choices.size(); ++i) {
      const auto &v = choices[i]; if (i) report << ',';
      report << "{\"name\":" << splash::json::quote(v.name) << ",\"simdgroups\":" << v.groups << ",\"traversal_mode\":" << v.mode << ",\"w8a8\":" << (v.quantized ? "true" : "false")
          << ",\"numerical_alternative\":" << (v.quantized ? "true" : "false") << ",\"preregistered_component_quality_passed\":" << (v.qualityPassed ? "true" : "false") << ",\"full_output_sha256\":" << splash::json::quote(v.outputSHA)
          << ",\"integer_dot_samples\":" << v.integerSamples << ",\"integer_dot_mismatches\":" << v.integerMismatches << ",\"full_late_scale_identity_elements\":" << v.identityElements << ",\"full_late_scale_identity_mismatches\":" << v.identityMismatches
          << ",\"median_gpu_ms\":" << median(v.gpu) << ",\"median_wall_ms\":" << median(v.wall) << ",\"paired_median_speedup\":" << (i ? median(v.pairedSpeedup) : 1)
          << ",\"warm_candidate_gpu_ms\":" << v.warmGPU << ",\"warm_matched_control_gpu_ms\":" << v.warmBaselineGPU << ",\"warm_ab_pairs\":" << v.warmAB << ",\"warm_ba_pairs\":" << v.warmBA << ",\"warm_wall_ms\":" << v.warmWall
          << ",\"timed_ab_pairs\":" << v.ab << ",\"timed_ba_pairs\":" << v.ba << ",\"candidate_position_order_counts\":[";
      for (size_t position = 0; position < v.positionOrderCounts.size(); ++position) { if (position) report << ','; report << "{\"position\":" << position << ",\"ab\":" << v.positionOrderCounts[position][0] << ",\"ba\":" << v.positionOrderCounts[position][1] << '}'; }
      report << "],\"full_baseline_error\":"; v.baselineError.write(report); report << ",\"sampled_source_fp64_error\":"; v.sourceOracleError.write(report);
      report << ",\"sampled_quantized_decomp_fp64_error\":"; v.decompOracleError.write(report); report << ",\"sampled_source_quantization_fp64_error\":"; v.sourceQuantizedOracleError.write(report);
      report << ",\"gpu_ms\":"; array(report,v.gpu); report << ",\"wall_ms\":"; array(report,v.wall); report << ",\"paired_baseline_gpu_ms\":"; array(report,v.pairedBaselineGPU);
      report << ",\"paired_baseline_wall_ms\":"; array(report,v.pairedBaselineWall); report << ",\"paired_speedup\":"; array(report,v.pairedSpeedup); report << '}';
    }
    report << "]}\n"; if (output.empty()) std::cout << report.str(); else { std::ofstream out(output); require(bool(out),"report open failed"); out << report.str(); }
    for (size_t i = 1; i < choices.size(); ++i) std::cerr << choices[i].name << " gpu=" << median(choices[i].gpu) << "ms speedup=" << median(choices[i].pairedSpeedup) << " quality=" << choices[i].qualityPassed << '\n';
    return 0;
  } catch (const std::exception &e) { std::cerr << "dense W8A8 oracle failed: " << e.what() << '\n'; return 1; }
}
