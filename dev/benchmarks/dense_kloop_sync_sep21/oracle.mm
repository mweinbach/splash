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
  std::string name, pipeline; uint32_t groups = 0, mode = 0, block = 0, frequency = 0;
  bool selected = false, whole = false;
  std::vector<double> gpu, wall, pairedBaselineGPU, pairedBaselineWall, pairedSpeedup;
  Error baselineError, wholeGroupError, loopNoSyncError, oracleError;
  std::string outputSHA;
};
std::vector<uint32_t> list(std::string value) {
  std::stringstream in(value); std::string word; std::vector<uint32_t> result;
  while (std::getline(in,word,',')) {
    size_t used = 0; const auto v = std::stoul(word,&used);
    require(used == word.size() && v <= UINT32_MAX,"invalid list"); result.push_back(uint32_t(v));
  }
  require(!result.empty(),"empty list"); std::sort(result.begin(),result.end());
  result.erase(std::unique(result.begin(),result.end()),result.end()); return result;
}
std::vector<Variant> variants(const Shape &s,const std::vector<uint32_t> &groups,
    const std::vector<uint32_t> &blocks,const std::vector<uint32_t> &frequencies) {
  const auto plan = splash::flash::flashPrefillDenseTilePolicy(s.rows,s.n,s.k);
  require(bool(plan),"captured role lacks selected dense plan");
  Variant selected; selected.name = "selected_production_bf16"; selected.pipeline = "flash_dense_cache_prefill_m128_n64_sg" + std::to_string(plan.simdGroups);
  selected.groups = plan.simdGroups; selected.mode = uint32_t(plan.traversal); selected.selected = true; selected.whole = true;
  std::vector<Variant> result{selected};
  for (uint32_t g : groups) {
    Variant v = selected; v.selected = false; v.groups = g;
    v.pipeline = "dense_kloop_bf16_m128_n64_sg" + std::to_string(g) + "_whole";
    v.name = v.pipeline + "_traversal" + std::to_string(v.mode); result.push_back(v);
  }
  for (uint32_t block : blocks) for (uint32_t g : groups) {
    auto fs = frequencies; fs.push_back(0); std::sort(fs.begin(),fs.end()); fs.erase(std::unique(fs.begin(),fs.end()),fs.end());
    for (uint32_t frequency : fs) {
      Variant v; v.groups = g; v.mode = uint32_t(plan.traversal); v.block = block; v.frequency = frequency;
      v.pipeline = "dense_kloop_bf16_m128_n64_sg" + std::to_string(g) + "_k" + std::to_string(block) + "_sync" + std::to_string(frequency);
      v.name = v.pipeline + "_traversal" + std::to_string(v.mode); result.push_back(v);
    }
  }
  return result;
}
struct Buffers {
  MetalBuffer input, weights, outputBase, output, diagnostics;
  uint64_t elements = 0; std::string inputSHA, weightSHA;
  Buffers(MetalBackend &backend,const Shape &s) {
    input = backend.allocateBuffer(uint64_t(s.rows)*s.k*2,BufferStorage::Shared);
    weights = backend.allocateBuffer(uint64_t(s.n)*s.k*2,BufferStorage::Shared);
    elements = uint64_t(s.rows)*s.n; allocateOutput(backend);
    if (s.inputPath.empty()) std::fill_n(static_cast<uint16_t *>(input.contents()),input.sizeBytes()/2,bf16(.125f));
    else readExact(s.inputPath,input.contents(),input.sizeBytes(),s.inputSHA);
    if (s.weightPath.empty()) std::fill_n(static_cast<uint16_t *>(weights.contents()),weights.sizeBytes()/2,bf16(.015625f));
    else readExact(s.weightPath,weights.contents(),weights.sizeBytes(),s.weightSHA);
    inputSHA = sha256(input.contents(),input.sizeBytes()); weightSHA = sha256(weights.contents(),weights.sizeBytes());
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
    for (uint64_t i = 0; i < guard; ++i) require(p[i] == sentinel && p[guard+elements+i] == sentinel,"output guard changed");
    require(*static_cast<const uint32_t *>(diagnostics.contents()) == sticky,"shader diagnostics changed");
  }
  std::vector<uint16_t> result() const {
    const auto *p = static_cast<const uint16_t *>(output.contents()); return {p,p+elements};
  }
  void immutable() const {
    require(sha256(input.contents(),input.sizeBytes()) == inputSHA && sha256(weights.contents(),weights.sizeBytes()) == weightSHA,"immutable BF16 operands changed");
  }
};
CommandGraph graphFor(const Variant &v,const Buffers &b,const Shape &s,uint32_t repeat) {
  CommandGraph graph;
  for (uint32_t r = 0; r < repeat; ++r) {
    require(!(s.n%64) && !(s.rows%128),"variant requires complete row/output tiles");
    const FlashDenseCacheParams p{s.rows,s.k,s.n,0,s.n,128,64,v.mode};
    graph.add(v.pipeline,{b.input,b.weights,b.output,b.diagnostics},p,
        traversal(s.rows/128,s.n/64,v.mode),{v.groups*32u,1,1});
  }
  return graph;
}
uint64_t boundaryTests(MetalBackend &backend,const std::vector<Variant> &choices) {
  Shape s; s.rows = 2048; s.k = 288; s.n = 320; Buffers b(backend,s); uint64_t checks = 0;
  for (const auto &v : choices) {
    // K288 exercises a masked tail for every configured K block. The exact
    // constant product prevents a tolerance from hiding tail/mask errors.
    const FlashDenseCacheParams valid{s.rows,s.k,s.n,64,128,128,64,3};
    b.reset(); CommandGraph graph;
    graph.add(v.pipeline,{b.input,b.weights,b.output,b.diagnostics},valid,traversal(16,2,3),{v.groups*32u,1,1});
    (void)backend.submitCommand(graph.dispatches()); b.guards(); const auto actual = b.result();
    const uint16_t expected = bf16(float(s.k)*.125f*.015625f);
    for (uint32_t r = 0; r < s.rows; ++r) for (uint32_t c = 0; c < s.n; ++c)
      require(actual[uint64_t(r)*s.n+c] == (c >= 64 && c < 192 ? expected : sentinel),"masked K tail/partial output interval failed: " + v.name);
    ++checks;
    for (uint32_t which = 0; which < 9; ++which) {
      auto p = valid; DispatchSize grid{1,1,1}, threads{v.groups*32u,1,1};
      switch (which) {
      case 0: p.rows = 0; break; case 1: p.rows = 2049; break;
      case 2: p.input_size = 31; break; case 3: p.output_begin = 321; break;
      case 4: p.output_count = 32; break; case 5: p.tile_rows = 64; break;
      case 6: p.reserved = 5; break; case 7: p.tile_outputs = 128; break;
      default: threads.x /= 2; break;
      }
      b.reset(); CommandGraph invalid;
      invalid.add(v.pipeline,{b.input,b.weights,b.output,b.diagnostics},p,grid,threads);
      (void)backend.submitCommand(invalid.dispatches());
      require(*static_cast<const uint32_t *>(b.diagnostics.contents()) == (sticky|2),"invalid shader parameters did not set sticky error: " + v.name);
      const auto *words = static_cast<const uint16_t *>(b.outputBase.contents());
      require(std::all_of(words,words+b.outputBase.sizeBytes()/2,[](uint16_t v) { return v == sentinel; }),"invalid shader parameters wrote output");
      ++checks;
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
  for (uint32_t word = 0; word < 65536; ++word) if (std::isfinite(number(uint16_t(word)))) {
    require(bf16(number(uint16_t(word))) == word,"BF16 finite round-trip failed"); ++checks;
  }
  for (uint32_t k : {32u,96u,256u,288u,512u,1024u,2560u,6144u,10240u})
    for (uint32_t block : {256u,512u,1024u}) for (uint32_t frequency : {0u,1u,2u,4u}) {
      std::vector<uint32_t> visits(k); const uint32_t full = k/block, tail = k%block;
      for (uint32_t step = 0; step < full; ++step) for (uint32_t i = 0; i < block; ++i) ++visits[step*block+i];
      for (uint32_t i = 0; i < tail; ++i) ++visits[full*block+i];
      require(std::all_of(visits.begin(),visits.end(),[](uint32_t v) { return v == 1; }),"K-loop partition duplicated or omitted a K element");
      const uint32_t steps = full+(tail != 0); uint32_t barriers = 0;
      for (uint32_t step = 0; step < steps; ++step) barriers += frequency && !(step%frequency);
      require(!frequency || barriers == (steps+frequency-1)/frequency,"barrier schedule mismatch"); ++checks;
    }
  std::cout << "{\"cpu_self_test\":\"passed\",\"checks\":" << checks << ",\"gpu_created\":false,\"payloads_read\":false}\n";
}
} // namespace

int main(int argc,char **argv) {
  try {
    bool run = false, preview = false;
    std::string library = "build/dense-kloop-sync-sep21/kloop.metallib", fixture = "build/prefill-dense-sep21/actual-captures.json", output, filter;
    auto groups = list("4,8"), blocks = list("256,512,1024"), frequencies = list("1,2,4");
    uint32_t samples = 10, repeat = 4;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if (arg == "--run") { run = true; continue; } if (arg == "--preview") { preview = true; continue; }
      require(i+1 < argc,"missing CLI value"); const std::string value = argv[++i];
      if (arg == "--library") library = value; else if (arg == "--fixture-manifest") fixture = value;
      else if (arg == "--out") output = value; else if (arg == "--projection-filter") filter = value;
      else if (arg == "--groups") groups = list(value); else if (arg == "--blocks") blocks = list(value);
      else if (arg == "--sync-frequencies") frequencies = list(value);
      else if (arg == "--samples") samples = uint32_t(std::stoul(value)); else if (arg == "--repeat") repeat = uint32_t(std::stoul(value));
      else throw std::runtime_error("unknown argument: " + arg);
    }
    require(run || preview,"GPU/model-payload work requires explicit --run; use --preview or --cpu-self-test");
    require(samples >= 2 && samples <= 100 && !(samples%2) && repeat && repeat <= 64,"samples must be even 2..100 and repeat 1..64");
    for (uint32_t v : groups) require(v == 4 || v == 8,"groups must be 4 or 8");
    for (uint32_t v : blocks) require(v == 256 || v == 512 || v == 1024,"blocks must be 256,512,1024");
    for (uint32_t v : frequencies) require(v == 0 || v == 1 || v == 2 || v == 4,"frequency must be 0,1,2,4");
    const auto shapes = loadFixtures(fixture,filter);
    if (preview) {
      std::cout << "{\"preview_only\":true,\"gpu_created\":false,\"payloads_read\":false,\"cases\":[";
      for (size_t i = 0; i < shapes.size(); ++i) {
        if (i) std::cout << ','; std::cout << "{\"projection\":" << splash::json::quote(shapes[i].projection) << ",\"variants\":[";
        const auto choices = variants(shapes[i],groups,blocks,frequencies);
        for (size_t j = 0; j < choices.size(); ++j) { if (j) std::cout << ','; std::cout << splash::json::quote(choices[j].name); }
        std::cout << "]}";
      }
      std::cout << "]}\n"; return 0;
    }
    MetalBackend backend(library);
    const uint64_t boundaryChecks = boundaryTests(backend,variants(shapes[0],groups,blocks,frequencies));
    std::ostringstream report; report << std::setprecision(10)
        << "{\"experiment\":\"dense_kloop_sync_sep21_v1\",\"full_model_quality_qualified\":false,"
        << "\"operands_original_bf16\":true,\"destination_f32\":true,\"output_bf16\":true,\"conversion_changed\":false,"
        << "\"opaque_mpp_reduction_order_proven\":false,\"barrier_memory_flag\":\"mem_none\",\"boundary_checks\":" << boundaryChecks << ','
        << "\"timing\":\"warmed balanced AB BA pairs with rotated candidate order\","
        << "\"cpu_operand_reads_writes_between_timed_calls\":false,\"samples\":" << samples << ",\"repeat\":" << repeat << ",\"cases\":[";
    bool first = true;
    for (const auto &s : shapes) {
      Buffers operands(backend,s); auto choices = variants(s,groups,blocks,frequencies); std::vector<Buffers> buffers;
      buffers.reserve(choices.size());
      for (size_t i = 0; i < choices.size(); ++i) { Buffers b = operands; b.allocateOutput(backend); buffers.push_back(std::move(b)); }
      std::vector<uint16_t> captured(uint64_t(s.rows)*s.n); readExact(s.expectedPath,captured.data(),captured.size()*2,s.expectedSHA);
      std::vector<uint16_t> baseline; std::vector<std::vector<uint16_t>> warmed;
      std::map<uint32_t,std::vector<uint16_t>> wholeReferences;
      std::map<std::pair<uint32_t,uint32_t>,std::vector<uint16_t>> loopReferences;
      for (size_t i = 0; i < choices.size(); ++i) {
        auto &v = choices[i]; auto &b = buffers[i]; auto warm = graphFor(v,b,s,1);
        (void)backend.submitCommand(warm.dispatches()); b.guards(); const auto actual = b.result();
        b.reset(); (void)backend.submitCommand(warm.dispatches()); b.guards();
        require(compare(b.result(),actual).mismatches == 0,"candidate deterministic output changed");
        if (!i) baseline = actual;
        if (v.selected) require(compare(actual,captured).mismatches == 0,"selected production differs from captured model output");
        if (v.whole && !v.selected) wholeReferences[v.groups] = actual;
        if (!v.whole) {
          require(wholeReferences.contains(v.groups),"same-group whole-K reference missing");
          v.wholeGroupError = compare(actual,wholeReferences.at(v.groups));
          const auto key = std::pair{v.block,v.groups};
          if (!v.frequency) loopReferences[key] = actual;
          require(loopReferences.contains(key),"same-block no-sync reference missing");
          v.loopNoSyncError = compare(actual,loopReferences.at(key));
        }
        v.baselineError = compare(actual,baseline); v.oracleError = scalarOracle(b,s,actual);
        require(!v.oracleError.nonfinite && v.oracleError.relativeL2() <= .004,"sampled FP64 numerical oracle failed: " + v.name);
        v.outputSHA = sha256(actual.data(),actual.size()*2); warmed.push_back(actual);
      }
      std::vector<CommandGraph> commands; commands.reserve(choices.size());
      for (size_t i = 0; i < choices.size(); ++i) commands.push_back(graphFor(choices[i],buffers[i],s,repeat));
      // All CPU qualification finishes before the GPU-only warming/timing span.
      for (uint32_t round = 0; round < 2; ++round) for (size_t i = 0; i < choices.size(); ++i)
        (void)backend.submitCommand(commands[(i+round)%choices.size()].dispatches());
      for (uint32_t sample = 0; sample < samples; ++sample) {
        for (size_t j = 0; j+1 < choices.size(); ++j) {
          const size_t i = 1+(j+sample)%(choices.size()-1); CommandTiming baseTime, candidateTime;
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
      operands.immutable();
      if (!first) report << ','; first = false; report << "{\"projection\":" << splash::json::quote(s.projection)
          << ",\"rows\":" << s.rows << ",\"input_size\":" << s.k << ",\"output_size\":" << s.n
          << ",\"weight_sha256\":" << splash::json::quote(s.weightSHA) << ",\"input_sha256\":" << splash::json::quote(s.inputSHA)
          << ",\"captured_control_exact\":true,\"guards_passed\":true,\"immutable_operands_passed\":true,\"variants\":[";
      for (size_t i = 0; i < choices.size(); ++i) {
        const auto &v = choices[i]; if (i) report << ','; const bool exact = !v.baselineError.mismatches;
        const uint32_t steps = v.whole ? 1 : (s.k+v.block-1)/v.block;
        report << "{\"name\":" << splash::json::quote(v.name) << ",\"tile_rows\":128,\"tile_outputs\":64"
            << ",\"simdgroups\":" << v.groups << ",\"traversal_mode\":" << v.mode << ",\"k_block\":" << v.block
            << ",\"k_tail\":" << (v.whole ? 0 : s.k%v.block) << ",\"sync_frequency\":" << v.frequency
            << ",\"barriers_per_threadgroup\":" << (v.frequency ? (steps+v.frequency-1)/v.frequency : 0)
            << ",\"full_bf16_parity_with_captured\":" << (exact ? "true" : "false")
            << ",\"same_group_whole_k_parity\":" << (!v.wholeGroupError.mismatches ? "true" : "false")
            << ",\"same_loop_sync_parity\":" << (!v.loopNoSyncError.mismatches ? "true" : "false")
            << ",\"numerical_alternative\":" << (exact ? "false" : "true")
            << ",\"full_output_sha256\":" << splash::json::quote(v.outputSHA) << ",\"median_gpu_ms\":" << median(v.gpu)
            << ",\"median_wall_ms\":" << median(v.wall) << ",\"paired_median_speedup\":" << (i ? median(v.pairedSpeedup) : 1)
            << ",\"baseline_error\":"; v.baselineError.write(report); report << ",\"same_group_whole_k_error\":"; v.wholeGroupError.write(report);
        report << ",\"same_loop_no_sync_error\":"; v.loopNoSyncError.write(report); report << ",\"sampled_fp64_error\":"; v.oracleError.write(report);
        report << ",\"gpu_ms\":"; array(report,v.gpu); report << ",\"wall_ms\":"; array(report,v.wall);
        report << ",\"paired_baseline_gpu_ms\":"; array(report,v.pairedBaselineGPU);
        report << ",\"paired_baseline_wall_ms\":"; array(report,v.pairedBaselineWall); report << ",\"paired_speedup\":"; array(report,v.pairedSpeedup); report << '}';
      }
      report << "]}";
      std::cerr << "dense K-loop qualified " << s.projection << " baseline=" << median(choices[0].gpu) << "ms" << '\n';
    }
    report << "]}\n";
    if (output.empty()) std::cout << report.str(); else { std::ofstream out(output); require(bool(out),"report open failed"); out << report.str(); }
    return 0;
  } catch (const std::exception &e) { std::cerr << "dense K-loop oracle failed: " << e.what() << '\n'; return 1; }
}
