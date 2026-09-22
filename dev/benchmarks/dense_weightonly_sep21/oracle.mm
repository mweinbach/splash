// Numerical-alternative standalone weight-only I8 prefill. Only Root may invoke --run.
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
#include <limits>
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
constexpr uint32_t floatGuard = 0x7fc10001;
constexpr double relativeL2Limit = .02, cosineLimit = .9998;
std::vector<uint32_t> groupsList(std::string value) {
  std::stringstream in(value); std::string word; std::vector<uint32_t> result;
  while (std::getline(in,word,',')) { const auto v = std::stoul(word); require(v == 4 || v == 8,"groups must be 4 or 8"); result.push_back(uint32_t(v)); }
  require(!result.empty(),"empty groups"); std::sort(result.begin(),result.end()); result.erase(std::unique(result.begin(),result.end()),result.end()); return result;
}
struct FloatError {
  uint64_t elements = 0, nonfinite = 0; double maxAbs = 0, squaredError = 0, squaredReference = 0, squaredActual = 0, product = 0;
  void add(double actual,double expected) {
    ++elements; if (!std::isfinite(actual) || !std::isfinite(expected)) { ++nonfinite; return; }
    const double d = actual-expected; maxAbs = std::max(maxAbs,std::abs(d)); squaredError += d*d; squaredReference += expected*expected; squaredActual += actual*actual; product += actual*expected;
  }
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"nonfinite\":" << nonfinite << ",\"max_abs\":" << maxAbs
        << ",\"relative_l2\":" << std::sqrt(squaredError/std::max(1e-30,squaredReference)) << ",\"cosine\":" << product/std::sqrt(std::max(1e-30,squaredActual*squaredReference)) << '}';
  }
};
double gamma(uint32_t depth,double unit) { const double v = depth*unit; require(v < 1,"invalid gamma bound"); return v/(1-v); }
double outwardNorm(double value,uint32_t depth) {
  require(std::isfinite(value) && value >= 0,"invalid finite positive norm");
  if (!value) return 0;
  return std::nextafter(value/(1-gamma(depth,std::ldexp(1.,-53))),std::numeric_limits<double>::infinity());
}
struct Variant {
  std::string name, pipeline, outputSHA; uint32_t groups = 0, mode = 0, m = 128, n = 64; bool quantized = false;
  std::vector<double> gpu, wall, pairedBaselineGPU, pairedBaselineWall, pairedSpeedup;
  std::vector<std::array<uint32_t,2>> positionOrderCounts;
  uint32_t ab = 0, ba = 0, warmAB = 0, warmBA = 0; double warmGPU = 0, warmBaselineGPU = 0, warmWall = 0;
  Error baselineError, sourceOracleError, decompOracleError, sourceQuantizedOracleError;
  FloatError rawF64Error, residualIdentityError;
  uint64_t rawSamples = 0, rawBoundFailures = 0, residualFailures = 0, identityElements = 0, identityMismatches = 0;
  double maxRawBoundRatio = 0, maxResidualBoundRatio = 0;
  bool qualityPassed = false;
};
std::vector<Variant> variants(const Shape &s,const std::vector<uint32_t> &groups,bool outputTile) {
  const auto plan = splash::flash::flashPrefillDenseTilePolicy(s.rows,s.n,s.k); require(bool(plan),"missing production plan");
  Variant selected; selected.name = "selected_production_bf16"; selected.groups = plan.simdGroups; selected.mode = uint32_t(plan.traversal);
  selected.pipeline = "flash_dense_cache_prefill_m128_n64_sg" + std::to_string(plan.simdGroups);
  std::vector<Variant> choices{selected};
  for (uint32_t g : groups) { Variant v; v.quantized = true; v.groups = g; v.mode = selected.mode; v.pipeline = "dense_weightonly_m128_n64_sg" + std::to_string(g); v.name = v.pipeline; choices.push_back(v); }
  if (outputTile) { Variant v; v.quantized = true; v.groups = 8; v.m = 64; v.n = 128; v.mode = selected.mode; v.pipeline = "dense_weightonly_m64_n128_sg8"; v.name = v.pipeline; choices.push_back(v); }
  return choices;
}
struct Buffers {
  MetalBuffer input, weights, quantWeightsBase, quantWeights, weightScalesBase, weightScales, rawBase, rawDot, outputBase, output, diagnostics;
  uint64_t elements = 0, inputSubnormal = 0, coefficientSubnormal = 0, scaleSubnormal = 0;
  std::string inputSHA, weightSHA, quantWeightSHA, weightScalesSHA; qw::QuantError coefficientError;
  static MetalBuffer guarded(MetalBackend &backend,MetalBuffer &base,uint64_t bytes,uint32_t width) {
    base = backend.allocateBuffer(bytes+2*guard*width,BufferStorage::Shared); return backend.view(base,guard*width,bytes);
  }
  Buffers(MetalBackend &backend,const Shape &s) {
    input = backend.allocateBuffer(uint64_t(s.rows)*s.k*2,BufferStorage::Shared); weights = backend.allocateBuffer(uint64_t(s.n)*s.k*2,BufferStorage::Shared);
    quantWeights = guarded(backend,quantWeightsBase,uint64_t(s.n)*s.k,1); weightScales = guarded(backend,weightScalesBase,uint64_t(s.n)*4,4);
    elements = uint64_t(s.rows)*s.n; rawDot = guarded(backend,rawBase,elements*4,4); allocateOutput(backend);
    std::memset(quantWeightsBase.contents(),quantGuard,quantWeightsBase.sizeBytes()); std::fill_n(static_cast<uint32_t *>(weightScalesBase.contents()),weightScalesBase.sizeBytes()/4,floatGuard);
    std::fill_n(static_cast<uint32_t *>(rawBase.contents()),rawBase.sizeBytes()/4,floatGuard);
    if (s.inputPath.empty()) std::fill_n(static_cast<uint16_t *>(input.contents()),input.sizeBytes()/2,bf16(.125f)); else readExact(s.inputPath,input.contents(),input.sizeBytes(),s.inputSHA);
    if (s.weightPath.empty()) std::fill_n(static_cast<uint16_t *>(weights.contents()),weights.sizeBytes()/2,bf16(.015625f)); else readExact(s.weightPath,weights.contents(),weights.sizeBytes(),s.weightSHA);
    const auto *w = static_cast<const uint16_t *>(weights.contents()); auto *q = static_cast<int8_t *>(quantWeights.contents()); auto *scales = static_cast<float *>(weightScales.contents());
    for (uint32_t row = 0; row < s.n; ++row) {
      qw::quantizeRow(w+uint64_t(row)*s.k,s.k,q+uint64_t(row)*s.k,scales[row],coefficientError);
      scaleSubnormal += std::fpclassify(scales[row]) == FP_SUBNORMAL;
    }
    for (uint64_t i = 0; i < weights.sizeBytes()/2; ++i) coefficientSubnormal += !(w[i]&0x7f80) && (w[i]&127);
    const auto *x = static_cast<const uint16_t *>(input.contents());
    for (uint64_t i = 0; i < input.sizeBytes()/2; ++i) {
      const float a = number(x[i]); require(std::isfinite(a),"nonfinite BF16 source input"); inputSubnormal += !(x[i]&0x7f80) && (x[i]&127);
      require(std::abs(double(a))*127 <= std::numeric_limits<float>::max(),"BF16 source input/I8 product exceeds F32 finite envelope");
    }
    for (uint32_t row = 0; row < s.rows; ++row) {
      double absoluteSum = 0;
      for (uint32_t k = 0; k < s.k; ++k) absoluteSum += std::abs(double(number(x[uint64_t(row)*s.k+k])));
      // Covers every possible signed-I8 partial sum and its worst sequential
      // F32 rounding growth, including an opaque reduction's cancellation.
      require(std::nextafter(127*outwardNorm(absoluteSum,s.k+8)*(1+gamma(s.k-1,std::ldexp(1.,-24))),
          std::numeric_limits<double>::infinity()) <= std::numeric_limits<float>::max(),
          "BF16 source row/I8 partial accumulation exceeds finite F32 envelope");
    }
    inputSHA = sha256(input.contents(),input.sizeBytes()); weightSHA = sha256(weights.contents(),weights.sizeBytes()); quantWeightSHA = sha256(quantWeights.contents(),quantWeights.sizeBytes()); weightScalesSHA = sha256(weightScales.contents(),weightScales.sizeBytes());
  }
  void allocateOutput(MetalBackend &backend) {
    output = guarded(backend,outputBase,elements*2,2); diagnostics = backend.allocateBuffer(64,BufferStorage::Shared); reset();
  }
  void reset() {
    std::fill_n(static_cast<uint16_t *>(outputBase.contents()),outputBase.sizeBytes()/2,sentinel); std::memset(diagnostics.contents(),0,diagnostics.sizeBytes()); *static_cast<uint32_t *>(diagnostics.contents()) = sticky;
  }
  static void canary(const MetalBuffer &base,uint64_t elements,uint32_t width,uint32_t value) {
    for (uint64_t i = 0; i < guard; ++i) {
      if (width == 1) { const auto *p = static_cast<const uint8_t *>(base.contents()); require(p[i] == value && p[guard+elements+i] == value,"I8 guard changed"); }
      else if (width == 2) { const auto *p = static_cast<const uint16_t *>(base.contents()); require(p[i] == value && p[guard+elements+i] == value,"BF16 output guard changed"); }
      else { const auto *p = static_cast<const uint32_t *>(base.contents()); require(p[i] == value && p[guard+elements+i] == value,"F32 scale/raw probe guard changed"); }
    }
  }
  void guards() const {
    canary(outputBase,elements,2,sentinel); canary(quantWeightsBase,quantWeights.sizeBytes(),1,quantGuard); canary(weightScalesBase,weightScales.sizeBytes()/4,4,floatGuard); canary(rawBase,elements,4,floatGuard);
    require(*static_cast<const uint32_t *>(diagnostics.contents()) == sticky,"shader diagnostics changed");
  }
  std::vector<uint16_t> result() const { const auto *p = static_cast<const uint16_t *>(output.contents()); return {p,p+elements}; }
  void immutable() const {
    require(sha256(input.contents(),input.sizeBytes()) == inputSHA && sha256(weights.contents(),weights.sizeBytes()) == weightSHA && sha256(quantWeights.contents(),quantWeights.sizeBytes()) == quantWeightSHA && sha256(weightScales.contents(),weightScales.sizeBytes()) == weightScalesSHA,"BF16 source/input/I8 coefficient/F32 scale changed");
  }
};
CommandGraph graphFor(const Variant &v,const Buffers &b,const Shape &s,uint32_t repeat,bool probe = false) {
  CommandGraph graph;
  for (uint32_t r = 0; r < repeat; ++r) {
    const FlashDenseCacheParams p{s.rows,s.k,s.n,0,s.n,v.m,v.n,v.mode};
    if (v.quantized) graph.add(v.pipeline+(probe ? "_probe" : ""),{b.input,b.quantWeights,b.output,b.rawDot,b.weightScales,b.diagnostics},p,traversal(s.rows/v.m,s.n/v.n,v.mode),{v.groups*32u,1,1});
    else graph.add(v.pipeline,{b.input,b.weights,b.output,b.diagnostics},p,traversal(s.rows/v.m,s.n/v.n,v.mode),{v.groups*32u,1,1});
  }
  return graph;
}
void rawAndResidualCertificate(Variant &v,const Buffers &b,const Shape &s,const std::vector<uint16_t> &actual) {
  const auto *x = static_cast<const uint16_t *>(b.input.contents()), *w = static_cast<const uint16_t *>(b.weights.contents()); const auto *q = static_cast<const int8_t *>(b.quantWeights.contents());
  const auto *scales = static_cast<const float *>(b.weightScales.contents()), *raw = static_cast<const float *>(b.rawDot.contents());
  for (uint32_t row = 0; row < s.rows; ++row) for (uint32_t column = 0; column < s.n; ++column) {
    const uint64_t i = uint64_t(row)*s.n+column; require(std::isfinite(raw[i]),"nonfinite raw F32 probe"); ++v.identityElements; v.identityMismatches += actual[i] != bf16(raw[i]*scales[column]);
  }
  require(!v.identityMismatches,"full raw-F32/late-weight-scale/BF16 identity failed");
  for (uint32_t ri = 0; ri < 8; ++ri) for (uint32_t ci = 0; ci < 32; ++ci) {
    const uint32_t row = uint32_t(uint64_t(ri)*(s.rows-1)/7), column = uint32_t(uint64_t(ci)*(s.n-1)/31); const double scale = scales[column];
    double source = 0, quantized = 0, residual = 0, sourceAbs = 0, quantizedAbs = 0, residualAbs = 0, subnormalAbs = 0;
    for (uint32_t k = 0; k < s.k; ++k) {
      const uint64_t ai = uint64_t(row)*s.k+k, bi = uint64_t(column)*s.k+k; const double a = number(x[ai]), original = number(w[bi]), code = q[bi];
      const double sp = a*original, qp = a*code, rp = a*(original-code*scale); source += sp; quantized += qp; residual += rp;
      sourceAbs += std::abs(sp); quantizedAbs += std::abs(qp); residualAbs += std::abs(rp);
      if (!(x[ai]&0x7f80) && (x[ai]&127)) subnormalAbs += std::abs(qp);
    }
    const double rawActual = raw[uint64_t(row)*s.n+column];
    // BF16 A and I8 B products are exact F32 for normal finite products.
    // Worst K-1 sequential F32 additions conservatively cover opaque MPP tree.
    // Add complete denormal-product and gradual-underflow envelopes explicitly.
    const double g32 = gamma(s.k-1,std::ldexp(1.,-24)), g64 = gamma(s.k,std::ldexp(1.,-53));
    const double rawBound = std::nextafter((g32+g64)*outwardNorm(quantizedAbs,s.k+8) +
        (1+g32)*(outwardNorm(subnormalAbs,s.k+8)+double(s.k+1)*std::numeric_limits<float>::min()),
        std::numeric_limits<double>::infinity());
    const double rawDelta = std::abs(rawActual-quantized); ++v.rawSamples; v.rawBoundFailures += rawDelta > rawBound; v.maxRawBoundRatio = std::max(v.maxRawBoundRatio,rawDelta/std::max(1e-300,rawBound)); v.rawF64Error.add(rawActual,quantized);
    const double reconstructed = quantized*scale+residual;
    const double residualBound = std::nextafter(gamma(2*s.k+16,std::ldexp(1.,-53))*
        outwardNorm(sourceAbs+std::abs(scale)*quantizedAbs+residualAbs,s.k+8)+std::numeric_limits<double>::min()*(s.k+1),
        std::numeric_limits<double>::infinity());
    const double residualDelta = std::abs(source-reconstructed); v.residualFailures += residualDelta > residualBound; v.maxResidualBoundRatio = std::max(v.maxResidualBoundRatio,residualDelta/std::max(1e-300,residualBound)); v.residualIdentityError.add(reconstructed,source);
    const double dequantized = quantized*scale;
    v.sourceOracleError.add(actual[uint64_t(row)*s.n+column],bf16(float(source))); v.decompOracleError.add(actual[uint64_t(row)*s.n+column],bf16(float(dequantized))); v.sourceQuantizedOracleError.add(bf16(float(dequantized)),bf16(float(source)));
  }
  require(!v.rawBoundFailures && !v.residualFailures && !v.decompOracleError.nonfinite,"sampled raw F32/F64 source-residual certificate failed");
}
uint64_t boundaryTests(MetalBackend &backend,const std::vector<Variant> &choices) {
  Shape s; s.rows = 2048; s.k = 32; s.n = 384; Buffers b(backend,s); uint64_t checks = 0;
  for (const auto &v : choices) {
    b.reset(); auto graph = graphFor(v,b,s,1,v.quantized); (void)backend.submitCommand(graph.dispatches()); b.guards(); const auto actual = b.result();
    require(std::all_of(actual.begin(),actual.end(),[](uint16_t x) { return x == bf16(.0625f); }),"synthetic BF16A/I8B scaled identity failed");
    if (v.quantized) { const auto *raw = static_cast<const float *>(b.rawDot.contents()); require(std::all_of(raw,raw+b.elements,[](float x) { return x == 32*.125f*127; }),"synthetic raw dot wrong"); } ++checks;
    for (uint32_t which = 0; which < 9; ++which) {
      FlashDenseCacheParams p{s.rows,s.k,s.n,0,s.n,v.m,v.n,v.mode}; DispatchSize threads{v.groups*32u,1,1};
      switch (which) { case 0: p.rows = 0; break; case 1: p.rows = 2049; break; case 2: p.input_size = 31; break; case 3: p.output_begin = 385; break; case 4: p.output_count = v.n/2; break; case 5: p.tile_rows = v.m/2; break; case 6: p.reserved = 5; break; case 7: p.tile_outputs = v.n*2; break; default: threads.x /= 2; break; }
      b.reset(); CommandGraph bad;
      if (v.quantized) bad.add(v.pipeline,{b.input,b.quantWeights,b.output,b.rawDot,b.weightScales,b.diagnostics},p,{1,1,1},threads);
      else bad.add(v.pipeline,{b.input,b.weights,b.output,b.diagnostics},p,{1,1,1},threads);
      (void)backend.submitCommand(bad.dispatches()); require(*static_cast<const uint32_t *>(b.diagnostics.contents()) == (sticky|2),"invalid matmul did not set sticky error"); const auto *all = static_cast<const uint16_t *>(b.outputBase.contents()); require(std::all_of(all,all+b.outputBase.sizeBytes()/2,[](uint16_t x) { return x == sentinel; }),"invalid matmul wrote output"); ++checks;
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
  for (int code = -127; code <= 127; ++code) { require(number(bf16(float(code))) == code,"signed I8 code not exactly BF16 representable"); ++checks; }
  for (uint32_t word = 0; word < 65536; ++word) if (std::isfinite(number(uint16_t(word)))) { require(bf16(number(uint16_t(word))) == word,"BF16 finite round-trip failed"); ++checks; }
    for (uint32_t count : {1u,2u,3u}) for (uint32_t cycles = 1; cycles <= 8; ++cycles) {
    std::vector<std::vector<std::array<uint32_t,2>>> counts(count,std::vector<std::array<uint32_t,2>>(count));
    for (uint32_t sample = 0; sample < 2*count*cycles; ++sample) for (uint32_t position = 0; position < count; ++position) ++counts[(position+(sample/2)%count)%count][position][sample%2];
    for (const auto &candidate : counts) for (const auto &position : candidate) require(position[0] == cycles && position[1] == cycles,"balanced stratum failed"); ++checks;
  }
  std::cout << "{\"cpu_self_test\":\"passed\",\"checks\":" << checks << ",\"all_symmetric_i8_exact_bf16\":true,\"gpu_created\":false,\"payloads_read\":false}\n";
}
} // namespace
int main(int argc,char **argv) {
  try {
    bool run = false, preview = false, outputTile = false; std::string library = "build/dense-weightonly-sep21/weightonly.metallib", fixture = "build/prefill-dense-sep21/actual-captures.json", output, filter = "linear_attn.in_proj_qkv";
    auto groups = groupsList("4,8"); uint32_t requestedSamples = 10, repeat = 4; double warmMilliseconds = 150;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i]; if (arg == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if (arg == "--output-tile") { outputTile = true; continue; }
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
    require(s.k == 2560 || s.k == 6144,"unexpected role K"); auto choices = variants(s,groups,outputTile); const uint32_t count = uint32_t(choices.size()-1), strata = 2*count;
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
    std::vector<uint16_t> baseline; std::vector<std::vector<uint16_t>> warmed;
    for (size_t i = 0; i < choices.size(); ++i) {
      auto &v = choices[i]; auto &b = buffers[i]; auto warm = graphFor(v,b,s,1); (void)backend.submitCommand(warm.dispatches()); b.guards(); const auto actual = b.result();
      b.reset(); (void)backend.submitCommand(warm.dispatches()); b.guards(); require(!compare(b.result(),actual).mismatches,"deterministic complete output changed");
      if (!i) { baseline = actual; require(!compare(actual,captured).mismatches,"selected production differs from captured output"); }
      v.baselineError = compare(actual,baseline); v.outputSHA = sha256(actual.data(),actual.size()*2);
      if (v.quantized) {
        auto probe = graphFor(v,b,s,1,true); b.reset(); (void)backend.submitCommand(probe.dispatches()); b.guards();
        require(!compare(b.result(),actual).mismatches,"probe output differs from no-probe scaled candidate"); rawAndResidualCertificate(v,b,s,actual);
        v.qualityPassed = !v.baselineError.nonfinite && v.baselineError.relativeL2() <= relativeL2Limit &&
            v.baselineError.product/std::sqrt(std::max(1e-30,v.baselineError.squaredReference*v.baselineError.squaredActual)) >= cosineLimit;
      } else {
        v.sourceOracleError = scalarOracle(b,s,actual);
        require(!v.sourceOracleError.nonfinite && v.sourceOracleError.relativeL2() <= .004,"captured BF16 control failed sampled original FP64 oracle");
        v.qualityPassed = true;
      }
      warmed.push_back(actual); v.positionOrderCounts.resize(count);
    }
    const std::string rawBefore = sha256(operands.rawDot.contents(),operands.rawDot.sizeBytes());
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
    operands.immutable(); require(sha256(operands.rawDot.contents(),operands.rawDot.sizeBytes()) == rawBefore,"normal timed candidate modified untimed raw F32 probe buffer");
    std::ostringstream report; report << std::setprecision(10) << "{\"experiment\":\"dense_weightonly_sep21_v1\",\"numerical_alternative\":true,\"full_model_quality_qualified\":false,\"production_overlay\":false,"
        << "\"activation_original_bf16_unchanged\":true,\"activation_quantizer\":false,\"weight_quantization\":\"symmetric_per_output_row_signed_i8_f32_scale\",\"destination_f32\":true,\"whole_k\":true,"
        << "\"all_symmetric_i8_codes_exact_bf16\":true,\"probe_raw_f32_writes_in_timing\":false,\"quality_preregister\":{\"full_output_relative_l2_max\":" << relativeL2Limit << ",\"full_output_cosine_min\":" << cosineLimit << "},"
        << "\"boundary_checks\":" << boundaries << ",\"captured_control_exact\":true,\"guards_passed\":true,\"immutable_coefficients_passed\":true,\"probe_immutable_during_timing\":true,"
        << "\"cpu_operand_output_access_in_warm_or_timing\":false,\"requested_samples\":" << requestedSamples << ",\"effective_balanced_samples\":" << samples << ",\"repeat\":" << repeat
        << ",\"minimum_gpu_warm_ms_each_candidate_and_matched_control\":" << warmMilliseconds << ",\"projection\":" << splash::json::quote(s.projection) << ",\"rows\":" << s.rows << ",\"input_size\":" << s.k << ",\"output_size\":" << s.n
        << ",\"source_weight_sha256\":" << splash::json::quote(s.weightSHA) << ",\"source_input_sha256\":" << splash::json::quote(s.inputSHA)
        << ",\"source_input_bf16_subnormal\":" << operands.inputSubnormal << ",\"source_coefficient_bf16_subnormal\":" << operands.coefficientSubnormal << ",\"f32_weight_scale_subnormal\":" << operands.scaleSubnormal
        << ",\"weight_quantization_error\":"; operands.coefficientError.writeJSON(report); report << ",\"variants\":[";
    for (size_t i = 0; i < choices.size(); ++i) {
      const auto &v = choices[i]; if (i) report << ',';
      report << "{\"name\":" << splash::json::quote(v.name) << ",\"simdgroups\":" << v.groups << ",\"traversal_mode\":" << v.mode << ",\"weightonly_i8\":" << (v.quantized ? "true" : "false")
          << ",\"numerical_alternative\":" << (v.quantized ? "true" : "false") << ",\"preregistered_component_quality_passed\":" << (v.qualityPassed ? "true" : "false") << ",\"full_output_sha256\":" << splash::json::quote(v.outputSHA)
          << ",\"raw_f32_fp64_samples\":" << v.rawSamples << ",\"raw_bound_failures\":" << v.rawBoundFailures << ",\"source_residual_failures\":" << v.residualFailures << ",\"max_raw_bound_ratio\":" << v.maxRawBoundRatio << ",\"max_residual_bound_ratio\":" << v.maxResidualBoundRatio << ",\"full_late_scale_identity_elements\":" << v.identityElements << ",\"full_late_scale_identity_mismatches\":" << v.identityMismatches
          << ",\"median_gpu_ms\":" << median(v.gpu) << ",\"median_wall_ms\":" << median(v.wall) << ",\"paired_median_speedup\":" << (i ? median(v.pairedSpeedup) : 1)
          << ",\"warm_candidate_gpu_ms\":" << v.warmGPU << ",\"warm_matched_control_gpu_ms\":" << v.warmBaselineGPU << ",\"warm_ab_pairs\":" << v.warmAB << ",\"warm_ba_pairs\":" << v.warmBA << ",\"warm_wall_ms\":" << v.warmWall
          << ",\"timed_ab_pairs\":" << v.ab << ",\"timed_ba_pairs\":" << v.ba << ",\"candidate_position_order_counts\":[";
      for (size_t position = 0; position < v.positionOrderCounts.size(); ++position) { if (position) report << ','; report << "{\"position\":" << position << ",\"ab\":" << v.positionOrderCounts[position][0] << ",\"ba\":" << v.positionOrderCounts[position][1] << '}'; }
      report << "],\"full_baseline_error\":"; v.baselineError.write(report); report << ",\"raw_f32_fp64_error\":"; v.rawF64Error.write(report); report << ",\"source_residual_fp64_identity_error\":"; v.residualIdentityError.write(report); report << ",\"sampled_source_fp64_error\":"; v.sourceOracleError.write(report);
      report << ",\"sampled_quantized_decomp_fp64_error\":"; v.decompOracleError.write(report); report << ",\"sampled_source_quantization_fp64_error\":"; v.sourceQuantizedOracleError.write(report);
      report << ",\"gpu_ms\":"; array(report,v.gpu); report << ",\"wall_ms\":"; array(report,v.wall); report << ",\"paired_baseline_gpu_ms\":"; array(report,v.pairedBaselineGPU);
      report << ",\"paired_baseline_wall_ms\":"; array(report,v.pairedBaselineWall); report << ",\"paired_speedup\":"; array(report,v.pairedSpeedup); report << '}';
    }
    report << "]}\n"; if (output.empty()) std::cout << report.str(); else { std::ofstream out(output); require(bool(out),"report open failed"); out << report.str(); }
    for (size_t i = 1; i < choices.size(); ++i) std::cerr << choices[i].name << " gpu=" << median(choices[i].gpu) << "ms speedup=" << median(choices[i].pairedSpeedup) << " quality=" << choices[i].qualityPassed << '\n';
    return 0;
  } catch (const std::exception &e) { std::cerr << "dense weight-only oracle failed: " << e.what() << '\n'; return 1; }
}
