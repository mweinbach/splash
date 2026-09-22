// CPU/source compilation is safe independently. Only Root may invoke --run.
// Captured operand payloads are read, converted and mapped only after --run.
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashDenseCache.h"
#include "flash/FlashPrefillDenseTiles.hpp"
#include "engine/Json.hpp"
#include <algorithm>
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
uint16_t halfBits(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value), sign = (bits >> 16) & 0x8000;
  const uint32_t exponent = (bits >> 23) & 255, mantissa = bits & 0x7fffff;
  if (exponent == 255) return uint16_t(sign | 0x7c00 | (mantissa ? 0x200 : 0));
  if (exponent >= 143) return uint16_t(sign | 0x7c00);
  if (exponent <= 101) return uint16_t(sign);
  if (exponent >= 113) {
    uint32_t fraction = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fff;
    fraction += remainder > 0x1000 || (remainder == 0x1000 && (fraction & 1));
    return uint16_t(sign | (((exponent - 112) << 10) + fraction));
  }
  const uint32_t shift = 126 - exponent, full = mantissa | 0x800000;
  uint32_t fraction = full >> shift;
  const uint32_t remainder = full & ((1u << shift) - 1), halfway = 1u << (shift - 1);
  fraction += remainder > halfway || (remainder == halfway && (fraction & 1));
  return uint16_t(sign | fraction);
}
float halfNumber(uint16_t value) {
  const uint32_t sign = uint32_t(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 31, fraction = value & 1023;
  if (exponent == 31) return std::bit_cast<float>(sign | 0x7f800000 | (fraction << 13));
  if (!exponent) {
    if (!fraction) return std::bit_cast<float>(sign);
    int power = -14;
    while (!(fraction & 1024)) { fraction <<= 1; --power; }
    return std::bit_cast<float>(sign | (uint32_t(power + 127) << 23) | ((fraction & 1023) << 13));
  }
  return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (fraction << 13));
}
struct Conversion {
  uint64_t elements = 0, changed = 0, sourceSubnormal = 0, targetSubnormal = 0;
  uint64_t underflow = 0, overflow = 0, nonfinite = 0;
  double maxAbs = 0, squaredError = 0, squaredReference = 0;
  void add(uint16_t source, uint16_t target) {
    ++elements;
    const float a = number(source), b = halfNumber(target);
    sourceSubnormal += !(source & 0x7f80) && (source & 127);
    targetSubnormal += !(target & 0x7c00) && (target & 1023);
    underflow += a != 0 && b == 0;
    overflow += std::isfinite(a) && !std::isfinite(b);
    nonfinite += !std::isfinite(a);
    changed += std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b);
    if (std::isfinite(a) && std::isfinite(b)) {
      const double delta = double(b) - a;
      maxAbs = std::max(maxAbs, std::abs(delta));
      squaredError += delta * delta; squaredReference += double(a) * a;
    }
  }
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"changed\":" << changed
        << ",\"source_bf16_subnormal\":" << sourceSubnormal << ",\"target_half_subnormal\":" << targetSubnormal
        << ",\"underflow_to_zero\":" << underflow << ",\"overflow\":" << overflow
        << ",\"source_nonfinite\":" << nonfinite << ",\"max_abs\":" << maxAbs
        << ",\"relative_l2\":" << std::sqrt(squaredError / std::max(1e-30, squaredReference))
        << ",\"exact_numeric_conversion\":" << (!changed ? "true" : "false") << '}';
  }
};
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
struct Variant {
  std::string name, pipeline; uint32_t m = 0, n = 0, groups = 0, mode = 0;
  bool half = false, selected = false, original = false;
  std::vector<double> gpu, wall, pairedBaselineGPU, pairedBaselineWall, pairedSpeedup;
  Error baselineError, oracleError; std::string outputSHA;
};
std::vector<Variant> variants(const Shape &s, bool includeM256, bool allTraversal, bool halfOnly) {
  const auto plan = splash::flash::flashPrefillDenseTilePolicy(s.rows,s.n,s.k);
  require(bool(plan),"captured role lacks selected dense plan");
  Variant baseline; baseline.name = "selected_bf16"; baseline.pipeline = "flash_dense_cache_prefill_m128_n64_sg" + std::to_string(plan.simdGroups);
  baseline.m = 128; baseline.n = 64; baseline.groups = plan.simdGroups; baseline.mode = uint32_t(plan.traversal); baseline.selected = true;
  std::vector<Variant> result{baseline};
  Variant original; original.name = "original_bf16"; original.original = true;
  original.m = s.k == 6144 && s.n == 2560 ? 64 : s.n >= 1024 ? 32 : 16;
  original.n = original.m >= 32 ? 128 : 64; original.groups = original.m >= 64 ? 8 : 4;
  original.pipeline = "flash_dense_cache_m" + std::to_string(original.m) + "_n" + std::to_string(original.n); result.push_back(original);
  for (uint32_t groups : {4u,8u}) {
    Variant v = baseline; v.selected = false; v.half = true; v.groups = groups;
    v.pipeline = "dense_final_half_m128_n64_sg" + std::to_string(groups);
    v.name = v.pipeline + "_traversal" + std::to_string(v.mode); result.push_back(v);
  }
  if (includeM256) for (bool half : {false,true}) {
    if (halfOnly && !half) continue;
    for (const auto &[n,groups] : {std::pair{32u,4u},std::pair{32u,8u},std::pair{64u,8u}}) {
      for (uint32_t mode = 0; mode <= 4; ++mode) {
        if (!allTraversal && mode != uint32_t(plan.traversal)) continue;
        Variant v; v.m = 256; v.n = n; v.groups = groups; v.mode = mode; v.half = half;
        v.pipeline = std::string("dense_final_") + (half ? "half" : "bf16") + "_m256_n" + std::to_string(n) + "_sg" + std::to_string(groups);
        v.name = v.pipeline + "_traversal" + std::to_string(mode); result.push_back(v);
      }
    }
  }
  return result;
}
struct Buffers {
  MetalBuffer input, weights, halfInputBase, halfInput, halfWeights, outputBase, output, diagnostics;
  uint64_t elements = 0; std::string inputSHA, weightSHA, halfWeightSHA;
  Conversion inputConversion, weightConversion;
  Buffers() = default;
  Buffers(MetalBackend &backend, const Shape &s, bool convertHalf = true) {
    input = backend.allocateBuffer(uint64_t(s.rows)*s.k*2,BufferStorage::Shared);
    weights = backend.allocateBuffer(uint64_t(s.n)*s.k*2,BufferStorage::Shared);
    halfInputBase = backend.allocateBuffer(input.sizeBytes()+guard*4,BufferStorage::Shared);
    halfInput = backend.view(halfInputBase,guard*2,input.sizeBytes());
    halfWeights = backend.allocateBuffer(convertHalf ? weights.sizeBytes() : 2,BufferStorage::Shared);
    elements = uint64_t(s.rows)*s.n; allocateOutput(backend);
    readExact(s.inputPath,input.contents(),input.sizeBytes(),s.inputSHA);
    readExact(s.weightPath,weights.contents(),weights.sizeBytes(),s.weightSHA);
    if (convertHalf) {
      auto *converted = static_cast<uint16_t *>(halfWeights.contents());
      const auto *source = static_cast<const uint16_t *>(weights.contents());
      for (uint64_t i = 0; i < weights.sizeBytes()/2; ++i) { converted[i] = halfBits(number(source[i])); weightConversion.add(source[i],converted[i]); }
      const auto *x = static_cast<const uint16_t *>(input.contents());
      for (uint64_t i = 0; i < input.sizeBytes()/2; ++i) inputConversion.add(x[i],halfBits(number(x[i])));
    } else std::memset(halfWeights.contents(),0,halfWeights.sizeBytes());
    inputSHA = sha256(input.contents(),input.sizeBytes()); weightSHA = sha256(weights.contents(),weights.sizeBytes());
    halfWeightSHA = sha256(halfWeights.contents(),halfWeights.sizeBytes());
    std::fill_n(static_cast<uint16_t *>(halfInputBase.contents()),halfInputBase.sizeBytes()/2,sentinel);
  }
  void allocateOutput(MetalBackend &backend) {
    outputBase = backend.allocateBuffer((elements+guard*2)*2,BufferStorage::Shared);
    output = backend.view(outputBase,guard*2,elements*2);
    diagnostics = backend.allocateBuffer(64,BufferStorage::Shared); reset();
  }
  void reset() {
    std::fill_n(static_cast<uint16_t *>(outputBase.contents()),outputBase.sizeBytes()/2,sentinel);
    std::memset(diagnostics.contents(),0,diagnostics.sizeBytes()); *static_cast<uint32_t *>(diagnostics.contents()) = sticky;
  }
  void guards() const {
    const auto *p = static_cast<const uint16_t *>(outputBase.contents());
    const auto *h = static_cast<const uint16_t *>(halfInputBase.contents());
    for (uint64_t i = 0; i < guard; ++i) {
      require(p[i] == sentinel && p[guard+elements+i] == sentinel,"output guard changed");
      require(h[i] == sentinel && h[guard+halfInput.sizeBytes()/2+i] == sentinel,"conversion guard changed");
    }
    require(*static_cast<const uint32_t *>(diagnostics.contents()) == sticky,"shader diagnostics changed");
  }
  std::vector<uint16_t> result() const {
    const auto *p = static_cast<const uint16_t *>(output.contents()); return {p,p+elements};
  }
  void immutable() const {
    require(sha256(input.contents(),input.sizeBytes()) == inputSHA && sha256(weights.contents(),weights.sizeBytes()) == weightSHA &&
        sha256(halfWeights.contents(),halfWeights.sizeBytes()) == halfWeightSHA,"immutable operands changed");
  }
  uint64_t certifyGPUConversion(Conversion *actual = nullptr) const {
    const auto *x = static_cast<const uint16_t *>(input.contents());
    const auto *h = static_cast<const uint16_t *>(halfInput.contents());
    uint64_t mismatches = 0;
    for (uint64_t i = 0; i < input.sizeBytes()/2; ++i) {
      mismatches += h[i] != halfBits(number(x[i]));
      if (actual) actual->add(x[i],h[i]);
    }
    return mismatches;
  }
};
CommandGraph graphFor(const Variant &v, const Buffers &b, const Shape &s, uint32_t repeat) {
  CommandGraph graph;
  for (uint32_t r = 0; r < repeat; ++r) {
    if (v.half) {
      require(b.input.sizeBytes()/2 <= UINT32_MAX,"converter count exceeds ABI");
      const uint32_t count = uint32_t(b.input.sizeBytes()/2);
      graph.add("dense_final_bf16_to_half",{b.input,b.halfInput,b.diagnostics},count,{(count+255u)/256u,1,1},{256,1,1});
    }
    require(!(s.n%v.n) && !(s.rows%v.m),"variant requires complete whole-K tiles");
    const FlashDenseCacheParams p{s.rows,s.k,s.n,0,s.n,v.m,v.n,v.mode};
    graph.add(v.pipeline,{v.half ? b.halfInput : b.input,v.half ? b.halfWeights : b.weights,b.output,b.diagnostics},p,
        traversal(s.rows/v.m,s.n/v.n,v.mode),{v.groups*32u,1,1});
  }
  return graph;
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
  uint64_t checks = 0; Conversion exhaustive;
  for (uint32_t word = 0; word < 65536; ++word) {
    const float source = number(uint16_t(word)); if (!std::isfinite(source)) continue;
    const uint16_t h = halfBits(source); exhaustive.add(uint16_t(word),h);
    const _Float16 native = static_cast<_Float16>(source); uint16_t nativeBits; std::memcpy(&nativeBits,&native,2);
    require(h == nativeBits,"HALF RNE differs from native CPU conversion");
    require(bf16(source) == word,"BF16 finite round-trip failed"); ++checks;
  }
  for (uint32_t word = 0; word < 65536; ++word) {
    const uint16_t h = uint16_t(word); if ((h & 0x7c00) == 0x7c00) continue;
    require(halfBits(halfNumber(h)) == h,"HALF finite round-trip failed"); ++checks;
  }
  std::cout << "{\"cpu_self_test\":\"passed\",\"checks\":" << checks << ",\"exhaustive_finite_bf16_to_half\":";
  exhaustive.write(std::cout); std::cout << "}\n";
}
} // namespace

int main(int argc,char **argv) {
  try {
    bool run = false, preview = false, includeM256 = false, allTraversal = false, halfOnly = false, bf16Only = false;
    std::string library = "build/dense-final-sep21/dense-final.metallib", fixture = "build/prefill-dense-sep21/actual-captures.json", output, filter;
    uint32_t samples = 10, repeat = 4;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if (arg == "--run") { run = true; continue; } if (arg == "--preview") { preview = true; continue; }
      if (arg == "--include-m256") { includeM256 = true; continue; } if (arg == "--all-traversal") { allTraversal = true; continue; }
      if (arg == "--m256-half-only") { halfOnly = true; includeM256 = true; continue; }
      if (arg == "--bf16-only") { bf16Only = true; includeM256 = true; continue; }
      require(i+1 < argc,"missing CLI value"); const std::string value = argv[++i];
      if (arg == "--library") library = value; else if (arg == "--fixture-manifest") fixture = value;
      else if (arg == "--out") output = value; else if (arg == "--projection-filter") filter = value;
      else if (arg == "--samples") samples = uint32_t(std::stoul(value)); else if (arg == "--repeat") repeat = uint32_t(std::stoul(value));
      else throw std::runtime_error("unknown argument: " + arg);
    }
    require(run || preview,"GPU/model-payload work requires explicit --run; use --preview or --cpu-self-test");
    require(samples >= 2 && samples <= 100 && !(samples%2) && repeat && repeat <= 64,"samples must be even 2..100 and repeat 1..64");
    const auto shapes = loadFixtures(fixture,filter);
    if (preview) {
      std::cout << "{\"preview_only\":true,\"gpu_created\":false,\"payloads_read\":false,\"cases\":[";
      for (size_t i = 0; i < shapes.size(); ++i) {
        if (i) std::cout << ','; std::cout << "{\"projection\":" << splash::json::quote(shapes[i].projection) << ",\"variants\":[";
        auto choices = variants(shapes[i],includeM256,allTraversal,halfOnly);
        if (bf16Only) choices.erase(std::remove_if(choices.begin(),choices.end(),[](const Variant &v) { return v.half; }),choices.end());
        for (size_t j = 0; j < choices.size(); ++j) { if (j) std::cout << ','; std::cout << splash::json::quote(choices[j].name); }
        std::cout << "]}";
      }
      std::cout << "]}\n"; return 0;
    }
    MetalBackend backend(library); std::ostringstream report; report << std::setprecision(10)
        << "{\"experiment\":\"dense_final_sep21_v1\",\"full_model_quality_qualified\":false,"
        << "\"whole_k\":true,\"destination_f32\":true,\"output_bf16\":true,\"half_input_gpu_conversion_in_timing\":true,"
        << "\"half_weight_cpu_conversion_outside_timing\":true,\"timing\":\"warmed balanced AB BA pairs with rotated candidate order\","
        << "\"cpu_operand_reads_writes_between_timed_calls\":false,\"samples\":" << samples << ",\"repeat\":" << repeat << ",\"cases\":[";
    bool first = true;
    for (const auto &s : shapes) {
      Buffers operands(backend,s,!bf16Only); auto choices = variants(s,includeM256,allTraversal,halfOnly); std::vector<Buffers> buffers;
      const bool halfFallback = operands.inputConversion.overflow || operands.weightConversion.overflow ||
          operands.inputConversion.nonfinite || operands.weightConversion.nonfinite;
      if (halfFallback || bf16Only) choices.erase(std::remove_if(choices.begin(),choices.end(),[](const Variant &v) { return v.half; }),choices.end());
      const bool halfExecuted = std::any_of(choices.begin(),choices.end(),[](const Variant &v) { return v.half; });
      buffers.reserve(choices.size());
      for (size_t i = 0; i < choices.size(); ++i) { Buffers b = operands; b.allocateOutput(backend); buffers.push_back(std::move(b)); }
      std::vector<uint16_t> captured(uint64_t(s.rows)*s.n); readExact(s.expectedPath,captured.data(),captured.size()*2,s.expectedSHA);
      std::vector<uint16_t> baseline; std::vector<std::vector<uint16_t>> warmed;
      for (size_t i = 0; i < choices.size(); ++i) {
        auto &v = choices[i]; auto &b = buffers[i]; auto warm = graphFor(v,b,s,1);
        (void)backend.submitCommand(warm.dispatches()); b.guards(); const auto actual = b.result();
        b.reset(); (void)backend.submitCommand(warm.dispatches()); b.guards();
        require(compare(b.result(),actual).mismatches == 0,"candidate deterministic output changed");
        if (!i) baseline = actual;
        if (v.selected || v.original) require(compare(actual,captured).mismatches == 0,"production variant differs from captured model output");
        v.baselineError = compare(actual,baseline); v.oracleError = scalarOracle(b,s,actual);
        require(!v.oracleError.nonfinite && v.oracleError.relativeL2() <= .004,"sampled FP64 numerical oracle failed: " + v.name);
        v.outputSHA = sha256(actual.data(),actual.size()*2); warmed.push_back(actual);
      }
      std::vector<CommandGraph> commands; commands.reserve(choices.size());
      for (size_t i = 0; i < choices.size(); ++i) commands.push_back(graphFor(choices[i],buffers[i],s,repeat));
      // Finish all CPU qualification before the final GPU-only warmup. This
      // avoids giving the first timed call the CPU-readback residency penalty.
      for (uint32_t round = 0; round < 2; ++round) for (size_t i = 0; i < choices.size(); ++i)
        (void)backend.submitCommand(commands[(i+round)%choices.size()].dispatches());
      // No output/operand/guard reads or writes occur inside this complete timing block.
      for (uint32_t sample = 0; sample < samples; ++sample) {
        for (size_t j = 0; j+1 < choices.size(); ++j) {
          const size_t i = 1+(j+sample)% (choices.size()-1); CommandTiming baseTime, candidateTime;
          if (!((sample+i)&1)) {
            baseTime = backend.submitCommand(commands[0].dispatches()); candidateTime = backend.submitCommand(commands[i].dispatches());
          } else {
            candidateTime = backend.submitCommand(commands[i].dispatches()); baseTime = backend.submitCommand(commands[0].dispatches());
          }
          const double baseGPU = baseTime.gpuSeconds*1000/repeat, gpu = candidateTime.gpuSeconds*1000/repeat;
          choices[i].gpu.push_back(gpu); choices[i].wall.push_back(candidateTime.wallSeconds*1000/repeat);
          choices[i].pairedBaselineGPU.push_back(baseGPU); choices[i].pairedBaselineWall.push_back(baseTime.wallSeconds*1000/repeat);
          choices[i].pairedSpeedup.push_back(baseGPU/gpu); choices[0].gpu.push_back(baseGPU); choices[0].wall.push_back(baseTime.wallSeconds*1000/repeat);
        }
      }
      for (size_t i = 0; i < choices.size(); ++i) {
        buffers[i].guards(); require(compare(buffers[i].result(),warmed[i]).mismatches == 0,"timed output changed from qualified warmup");
      }
      Conversion gpuInputConversion; uint64_t converterRNEBitMismatches = 0;
      if (halfExecuted) converterRNEBitMismatches = operands.certifyGPUConversion(&gpuInputConversion);
      operands.immutable();
      if (!first) report << ','; first = false; report << "{\"projection\":" << splash::json::quote(s.projection)
          << ",\"rows\":" << s.rows << ",\"input_size\":" << s.k << ",\"output_size\":" << s.n
          << ",\"weight_sha256\":" << splash::json::quote(s.weightSHA) << ",\"input_sha256\":" << splash::json::quote(s.inputSHA)
          << ",\"input_conversion\":"; operands.inputConversion.write(report); report << ",\"weight_conversion\":"; operands.weightConversion.write(report);
      report << ",\"gpu_input_conversion\":"; gpuInputConversion.write(report);
      report << ",\"gpu_converter_rne_bit_mismatches\":" << converterRNEBitMismatches
          << ",\"gpu_input_converter_executed\":" << (halfExecuted ? "true" : "false")
          << ",\"half_fallback_for_overflow_or_nonfinite\":" << (halfFallback ? "true" : "false") << ",\"variants\":[";
      for (size_t i = 0; i < choices.size(); ++i) {
        const auto &v = choices[i]; if (i) report << ',';
        const bool exact = !v.baselineError.mismatches && (!v.half || (!gpuInputConversion.changed && !operands.weightConversion.changed && !converterRNEBitMismatches));
        report << "{\"name\":" << splash::json::quote(v.name) << ",\"tile_rows\":" << v.m << ",\"tile_outputs\":" << v.n
            << ",\"simdgroups\":" << v.groups << ",\"traversal_mode\":" << v.mode << ",\"half_operands\":" << (v.half ? "true" : "false")
            << ",\"exact_operand_and_output_certificate\":" << (exact ? "true" : "false") << ",\"numerical_alternative\":" << (exact ? "false" : "true")
            << ",\"full_output_sha256\":" << splash::json::quote(v.outputSHA) << ",\"median_gpu_ms\":" << median(v.gpu)
            << ",\"median_wall_ms\":" << median(v.wall) << ",\"paired_median_speedup\":" << (i ? median(v.pairedSpeedup) : 1)
            << ",\"baseline_error\":"; v.baselineError.write(report); report << ",\"sampled_fp64_error\":"; v.oracleError.write(report);
        report << ",\"gpu_ms\":"; array(report,v.gpu); report << ",\"wall_ms\":"; array(report,v.wall);
        report << ",\"paired_baseline_gpu_ms\":"; array(report,v.pairedBaselineGPU);
        report << ",\"paired_baseline_wall_ms\":"; array(report,v.pairedBaselineWall); report << ",\"paired_speedup\":"; array(report,v.pairedSpeedup); report << '}';
      }
      report << "]}";
      std::cerr << "dense final qualified " << s.projection << " baseline=" << median(choices[0].gpu) << "ms" << '\n';
    }
    report << "]}\n";
    if (output.empty()) std::cout << report.str(); else { std::ofstream out(output); require(bool(out),"report open failed"); out << report.str(); }
    return 0;
  } catch (const std::exception &e) { std::cerr << "dense final oracle failed: " << e.what() << '\n'; return 1; }
}
