// Root-only GPU pilot; --cpu-self-test creates no Metal device.
// Reuse independent packed-code, BF16, guarded-buffer and CPU-dot utilities.
#define main splashPrivateOriginalFloatOracleMain
#include "flash_float_dense_cache_oracle.mm"
#undef main
#include "FlashAffineInt8Code.h"
#include "metal/abi/FlashFloatDenseCache.h"
#include <tuple>

namespace {
bool int8CodeOverlaps(MetalBuffer a, MetalBuffer b) {
  const uintptr_t aa = reinterpret_cast<uintptr_t>(a.contents());
  const uintptr_t bb = reinterpret_cast<uintptr_t>(b.contents());
  require(aa && bb, "INT8 pilot requires CPU-visible buffers");
  return aa <= bb ? uint64_t(bb - aa) < a.sizeBytes() : uint64_t(aa - bb) < b.sizeBytes();
}
void int8CodeRequireBytes(MetalBuffer b, uint64_t bytes) {
  require(b && b.contents() && b.sizeBytes() >= bytes && bytes,
          "INT8 pilot buffer is missing or short");
}
uint64_t int8CodeSourceExtent(uint32_t rows, uint64_t stride, uint64_t rowBytes) {
  require(rows && rowBytes && stride >= rowBytes &&
          uint64_t(rows - 1) <= (UINT64_MAX - rowBytes) / stride,
          "INT8 pilot source byte extent overflows");
  return uint64_t(rows - 1) * stride + rowBytes;
}
void int8CodeRequireSource(const FlashTensor &tensor, uint64_t bytes) {
  require(tensor.logicalBytes >= bytes && tensor.shape.size() == 2,
          "INT8 pilot source logical extent or rank is invalid");
  int8CodeRequireBytes(tensor.buffer, bytes);
}
FlashAffineInt8CodeParams int8CodeParams(const FlashAffineProjection &p,
                                        uint32_t rows, uint32_t tile) {
  require(p.experts == 1 && p.inputSize && p.inputSize <= 32768 && p.inputSize % 64 == 0 &&
          p.outputSize && p.outputSize % 64 == 0 && p.groupSize == 64 &&
          (p.bits == 4 || p.bits == 5 || p.bits == 6 || p.bits == 8) &&
          p.weights && p.scales && p.biases && p.weights->dtype == FlashDType::U32 &&
          p.scales->dtype == FlashDType::BF16 && p.biases->dtype == FlashDType::BF16 &&
          rows && rows <= 16 && tile < 4,
          "INT8 code pilot requires original G64 affine rank2 source and R1..16");
  const uint32_t m = kTileRows[tile], n = tile % 2 ? 128 : 64;
  require(p.outputSize % n == 0, "INT8 code pilot output tile is not aligned");
  require(p.weightRowStrideBytes >= (uint64_t(p.inputSize) * p.bits + 7) / 8 &&
          p.parameterRowStrideBytes >= uint64_t(p.inputSize / 64) * 2 &&
          p.parameterRowStrideBytes % 2 == 0, "INT8 pilot source strides are invalid");
  int8CodeRequireSource(*p.weights, int8CodeSourceExtent(p.outputSize,
      p.weightRowStrideBytes, (uint64_t(p.inputSize) * p.bits + 7) / 8));
  for (const auto *t : {p.scales, p.biases})
    int8CodeRequireSource(*t, int8CodeSourceExtent(p.outputSize,
        p.parameterRowStrideBytes, uint64_t(p.inputSize / 64) * 2));
  return {rows, (rows + m - 1) / m * m, p.inputSize, p.outputSize, p.bits, 64, m, n,
          p.weightRowStrideBytes, p.parameterRowStrideBytes};
}
struct Int8CodeCache final {
  MetalBuffer base, view;
  uint64_t bytes;
  CommandTiming initialization;
  Int8CodeCache(MetalBackend &backend, const FlashAffineProjection &p, MetalBuffer diagnostics)
      : bytes(uint64_t(p.outputSize) * p.inputSize) {
    const auto params = int8CodeParams(p, 1, 0);
    base = backend.allocateBuffer(bytes + 128, BufferStorage::Shared,
                                  "private-exact-centered-int8-source-codes-with-guards");
    std::memset(base.contents(), 0xa7, base.sizeBytes());
    view = backend.view(base, 64, bytes);
    CommandGraph graph;
    graph.add("flash_affine_int8code_expand", {p.weights->buffer, view, diagnostics},
              params, {(bytes + 255) / 256, 1, 1}, {256, 1, 1});
    initialization = backend.submitCommand(graph.dispatches());
    check();
  }
  void check() const {
    const auto *b = static_cast<const uint8_t *>(base.contents());
    for (uint64_t i = 0; i < 64; ++i)
      require(b[i] == 0xa7 && b[bytes + 64 + i] == 0xa7,
              "INT8 conversion wrote beyond exact code view");
  }
  uint64_t sourceSamples(const FlashAffineProjection &p) const {
    const uint64_t samples = std::min<uint64_t>(bytes, 32768);
    const auto *actual = static_cast<const int8_t *>(view.contents());
    for (uint64_t i = 0; i < samples; ++i) {
      const uint64_t index = i < 2 ? (i ? bytes - 1 : 0) : randomWord(i + 0x18c0de) % bytes;
      const uint32_t n = uint32_t(index / p.inputSize), k = uint32_t(index % p.inputSize);
      const auto *packed = static_cast<const std::byte *>(p.weights->buffer.contents()) +
          uint64_t(n) * p.weightRowStrideBytes;
      const int expected = int(code(packed, p.bits, k)) - int(1u << (p.bits - 1));
      require(int(actual[index]) == expected, "GPU centered code cache changed an original code");
    }
    return samples;
  }
};
void int8CodeGraph(CommandGraph &graph, const FlashAffineProjection &p,
    MetalBuffer codes, MetalBuffer input, MetalBuffer padded, MetalBuffer sums,
    MetalBuffer output, MetalBuffer diagnostics, uint32_t rows, uint32_t tile) {
  const auto params = int8CodeParams(p, rows, tile);
  int8CodeRequireBytes(codes, uint64_t(p.outputSize) * p.inputSize);
  int8CodeRequireBytes(input, uint64_t(rows) * p.inputSize * 2);
  int8CodeRequireBytes(padded, uint64_t(params.padded_rows) * p.inputSize * 2);
  int8CodeRequireBytes(sums, uint64_t(params.padded_rows) * (p.inputSize / 64) * 4);
  int8CodeRequireBytes(output, uint64_t(rows) * p.outputSize * 2);
  int8CodeRequireBytes(diagnostics, 4);
  const std::array immutable{p.weights->buffer, p.scales->buffer, p.biases->buffer, codes, input};
  const std::array writable{padded, sums, output, diagnostics};
  for (const auto &a : immutable)
    for (const auto &b : writable)
      require(!int8CodeOverlaps(a, b), "INT8 pilot writable view aliases immutable source/input");
  for (size_t a = 0; a < writable.size(); ++a)
    for (size_t b = a + 1; b < writable.size(); ++b)
      require(!int8CodeOverlaps(writable[a], writable[b]), "INT8 pilot writable views overlap");
  const FlashFloatDenseSmallRowsParams padding{rows, params.padded_rows, p.inputSize,
      p.outputSize, 0, p.outputSize, params.tile_rows, params.tile_outputs};
  graph.add("flash_float_dense_small_rows_pad", {input, padded, diagnostics}, padding,
            {(uint64_t(params.padded_rows) * p.inputSize + 255) / 256, 1, 1});
  graph.add("flash_affine_int8code_group_sums", {padded, sums, diagnostics}, params,
            {params.padded_rows, p.inputSize / 64, 1}, {32, 1, 1});
  graph.add("flash_affine_int8code_m" + std::to_string(params.tile_rows) + "_n" +
      std::to_string(params.tile_outputs),
      {padded, codes, p.scales->buffer, p.biases->buffer, sums, output, diagnostics},
      params, {p.outputSize / params.tile_outputs, params.padded_rows / params.tile_rows, 1},
      {128, 1, 1});
}
uint32_t int8CodeHostRejections(MetalBackend &backend, const FlashAffineProjection &p,
    MetalBuffer codes, MetalBuffer input, MetalBuffer padded, MetalBuffer sums,
    MetalBuffer output, MetalBuffer diagnostics) {
  uint32_t result = 0;
  auto reject = [&](FlashAffineProjection source, MetalBuffer c, MetalBuffer x,
      MetalBuffer pad, MetalBuffer sx, MetalBuffer y, MetalBuffer d, uint32_t rows, uint32_t tile) {
    bool caught = false;
    try { CommandGraph g; int8CodeGraph(g, source, c, x, pad, sx, y, d, rows, tile); }
    catch (const std::exception &) { caught = true; }
    require(caught, "private INT8 host accepted invalid arguments");
    ++result;
  };
  reject(p,codes,input,padded,sums,output,diagnostics,0,0);
  reject(p,codes,input,padded,sums,output,diagnostics,17,0);
  reject(p,codes,input,padded,sums,output,diagnostics,1,4);
  reject(p,{},input,padded,sums,output,diagnostics,1,0);
  reject(p,codes,{},padded,sums,output,diagnostics,1,0);
  reject(p,codes,input,padded,sums,input,diagnostics,1,0);
  reject(p,codes,input,input,sums,output,diagnostics,1,0);
  reject(p,codes,input,padded,padded,output,diagnostics,1,0);
  reject(p,codes,input,padded,sums,output,backend.view(input,0,4),1,0);
  reject(p,codes,input,padded,sums,output,{},1,0);
  auto invalid = p; invalid.groupSize = 128;
  reject(invalid,codes,input,padded,sums,output,diagnostics,1,0);
  invalid = p; invalid.bits = 3;
  reject(invalid,codes,input,padded,sums,output,diagnostics,1,0);
  invalid = p; invalid.parameterRowStrideBytes = 1;
  reject(invalid,codes,input,padded,sums,output,diagnostics,1,0);
  invalid = p; invalid.parameterRowStrideBytes = UINT64_MAX - 1;
  reject(invalid,codes,input,padded,sums,output,diagnostics,1,0);
  invalid = p; invalid.weightRowStrideBytes = UINT64_MAX;
  reject(invalid,codes,input,padded,sums,output,diagnostics,1,0);
  auto shortSource = *p.weights; shortSource.logicalBytes = 1;
  invalid = p; invalid.weights = &shortSource;
  reject(invalid,codes,input,padded,sums,output,diagnostics,1,0);
  auto wrongRank = *p.scales; wrongRank.shape.push_back(1);
  invalid = p; invalid.scales = &wrongRank;
  reject(invalid,codes,input,padded,sums,output,diagnostics,1,0);
  reject(p,backend.view(codes,0,1),input,padded,sums,output,diagnostics,1,0);
  return result;
}
void int8CodeCPUSelfTest() {
  uint64_t checks = 0;
  for (uint32_t bits : {4u,5u,6u,8u}) {
    const int center = 1 << (bits - 1);
    for (int q = 0; q < 1 << bits; ++q) {
      const int8_t c = int8_t(q - center);
      require(int(c) + center == q, "centered INT8 changed an unsigned source code");
      ++checks;
      for (float s : {-14.25f,-.00390625f,0.0f,.5f,32.0f})
        for (float b : {-128.0f,-.00390625f,0.0f,64.0f,256.0f}) {
          const float original = float(q) * s + b;
          const float factored = float(c) * s + (b + float(center) * s);
          require(original == factored, "exact affine centering trap changed coefficient");
          ++checks;
        }
    }
  }
  require(int(int8_t(0 - 128)) == -128 && int(int8_t(255 - 128)) == 127,
          "Q8 INT8 endpoint -128 was incorrectly rejected");
  ++checks;
  require(int8CodeSourceExtent(2,64,32) == 96 &&
          int8CodeSourceExtent(1,UINT64_MAX,32) == 32,
          "INT8 source extent valid boundary failed");
  checks += 2;
  for (const auto &[rows,stride,rowBytes] :
       std::array<std::tuple<uint32_t,uint64_t,uint64_t>,5>{{
          {0,64,32},{2,16,32},{2,UINT64_MAX,32},
          {UINT32_MAX,UINT64_MAX/2,32},{2,64,0}}}) {
    bool caught=false;
    try { (void)int8CodeSourceExtent(rows,stride,rowBytes); }
    catch(const std::exception &) { caught=true; }
    require(caught,"INT8 source extent accepted invalid boundary");
    ++checks;
  }
  std::cout << "{\"pass\":true,\"int8_code_cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
}
}

int main(int argc, char **argv) {
  @autoreleasepool {
    std::vector<std::string> records;
    bool allPass = true;
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") {
        cpuSelfTest(); int8CodeCPUSelfTest(); return 0;
      }
      require(argc == 4, "usage: flash-affine-int8code-oracle METALLIB PACKAGE REPORT_JSON | --cpu-self-test");
      const auto rowsCases = list("FLASH_INT8CODE_ROWS", {1,4,8,16}, 16);
      const auto tiles = list("FLASH_INT8CODE_TILES", {0,1,2,3}, 3);
      const auto repetitions = list("FLASH_INT8CODE_REPEATS", {6}, 24);
      require(repetitions.size() == 1 && repetitions[0] >= 2, "INT8 repeats must be one value 2..24");
      for (const auto rows : rowsCases) require(rows, "INT8 row count is zero");
      const auto selected = parsePrefixes(std::getenv("FLASH_INT8CODE_PREFIXES") ?
          std::getenv("FLASH_INT8CODE_PREFIXES") :
          "language_model.lm_head,language_model.model.layers.0.linear_attn.in_proj_qkv");
      MetalBackend backend(argv[1]);
      auto weights = FlashWeights::load(backend, argv[2]);
      auto diagnostics = backend.allocateBuffer(4, BufferStorage::Shared, "private INT8 diagnostics");
      auto *status = static_cast<uint32_t *>(diagnostics.contents());
      for (const auto &prefix : selected) {
        const auto &p = weights.projection(prefix);
        (void)int8CodeParams(p,1,0);
        *status = kSticky;
        Int8CodeCache codes(backend,p,diagnostics);
        require(*status == kSticky, "INT8 cache conversion changed sticky diagnostic");
        const auto sourceChecks = codes.sourceSamples(p);
        FlashFloatDenseCache floatCache(backend,weights,prefix);
        const auto coefficientChecks = coefficientSamples(p,floatCache.tensor(prefix));
        FlashFloatDenseSmallRowsWorkspace workspace(backend);
        Guarded groupSums(backend,uint64_t(16) * (p.inputSize/64) * 2);
        for (const uint32_t rows : rowsCases) {
          std::vector<uint16_t> hostInput(uint64_t(rows)*p.inputSize);
          for (size_t i=0;i<hostInput.size();++i)
            hostInput[i]=bf16(float(int32_t(randomWord(i+0xc0ffee)%2047)-1023)/1024.0f);
          if (const char *path=std::getenv("FLASH_INT8CODE_INPUT_BF16")) {
            std::ifstream in(path,std::ios::binary);
            require(bool(in),"cannot read private real BF16 activation fixture");
            in.read(reinterpret_cast<char *>(hostInput.data()),std::streamsize(hostInput.size()*2));
            require(in.gcount()==std::streamsize(hostInput.size()*2),"BF16 fixture is short");
          }
          Guarded input(backend,hostInput.size());input.load(hostInput);
          Guarded raw(backend,uint64_t(rows)*p.outputSize);
          Guarded fp32(backend,uint64_t(rows)*p.outputSize);
          Guarded candidate(backend,uint64_t(rows)*p.outputSize);
          const auto rejectionCount=int8CodeHostRejections(backend,p,codes.view,input.view,
              workspace.paddedInput(),groupSums.view,candidate.view,diagnostics);
          const auto cpu=reference(p,hostInput,rows,19);
          CommandGraph rawGraph;
          addAffine(rawGraph,input.view,p,raw.view,diagnostics,rows);
          for (const uint32_t tile : tiles) {
            clearPadding(workspace.paddedInput());groupSums.clear();candidate.clear();
            *status=kSticky;
            CommandGraph floatGraph,int8Graph;
            floatCache.addSmallRows(floatGraph,prefix,input.view,fp32.view,diagnostics,rows,
                workspace,static_cast<FlashFloatDenseSmallRowsTile>(tile));
            int8CodeGraph(int8Graph,p,codes.view,input.view,workspace.paddedInput(),groupSums.view,
                candidate.view,diagnostics,rows,tile);
            for (const CommandGraph *g : {&rawGraph,&floatGraph,&int8Graph})
              (void)backend.submitCommand(g->dispatches());
            candidate.check();fp32.check();raw.check();input.check();codes.check();groupSums.check(false);
            require(*status==kSticky,"INT8 warmup changed sticky diagnostic");
            const std::vector<uint16_t> first(candidate.values().begin(),candidate.values().end());
            std::array<Times,2> vsRawTimes,vsFloatTimes;
            timed(backend,{&rawGraph,&int8Graph},repetitions[0],vsRawTimes);
            timed(backend,{&floatGraph,&int8Graph},repetitions[0],vsFloatTimes);
            candidate.check();fp32.check();raw.check();input.check();codes.check();groupSums.check(false);
            require(*status==kSticky && compare(candidate.values(),first).mismatch==0 &&
                    compare(input.values(),hostInput).mismatch==0,"INT8 repeated output/input/diagnostic changed");
            checkPadding(workspace.paddedInput(),hostInput,p.inputSize,rows,kTileRows[tile]);
            const auto eraw=compare(candidate.values(),raw.values());
            const auto efloat=compare(candidate.values(),fp32.values());
            const auto ecpu=compareSample(candidate.values(),cpu,cpu.serial,p.outputSize,rows);
            const auto esimd=compareSample(candidate.values(),cpu,cpu.simd32,p.outputSize,rows);
            const bool pass=!eraw.nonfinite&&!efloat.nonfinite&&!ecpu.nonfinite&&!esimd.nonfinite&&
                eraw.relativeL2()<1e-4&&efloat.relativeL2()<1e-4&&
                std::min(ecpu.relativeL2(),esimd.relativeL2())<1e-4;
            allPass=allPass&&pass;
            std::ostringstream record;
            record<<std::setprecision(12)<<"{\"prefix\":"<<splash::json::quote(prefix)
              <<",\"N\":"<<p.outputSize<<",\"K\":"<<p.inputSize<<",\"bits\":"<<p.bits
              <<",\"group_size\":64,\"rows\":"<<rows<<",\"tile\":"<<splash::json::quote(kTileNames[tile])
              <<",\"strict_accuracy_pass\":"<<(pass?"true":"false")
              <<",\"source_code_samples_exact\":"<<sourceChecks
              <<",\"code_cache_bytes\":"<<codes.bytes<<",\"code_cache_guard_bytes\":128"
              <<",\"f32_cache_bytes\":"<<floatCache.tensor(prefix).logicalBytes
              <<",\"coefficient_checks\":";coefficientChecks.write(record);
            record<<",\"host_rejections\":"<<rejectionCount
              <<",\"cache_conversion_gpu_seconds\":"<<codes.initialization.gpuSeconds
              <<",\"raw_dispatches\":"<<rawGraph.dispatches().size()
              <<",\"raw_dispatch_pipelines\":[";
            for (size_t i=0;i<rawGraph.dispatches().size();++i) {
              if(i) record<<',';
              record<<splash::json::quote(rawGraph.dispatches()[i].pipelineName);
            }
            record<<"],\"candidate_dispatches\":"<<int8Graph.dispatches().size()
              <<",\"candidate_paired_with_raw\":";vsRawTimes[1].write(record);
            record<<",\"raw_timing\":";vsRawTimes[0].write(record);
            record<<",\"speedup_vs_raw_gpu\":"<<median(vsRawTimes[0].gpu)/median(vsRawTimes[1].gpu)
              <<",\"candidate_paired_with_f32\":";vsFloatTimes[1].write(record);
            record<<",\"f32_timing\":";vsFloatTimes[0].write(record);
            record<<",\"speedup_vs_f32_gpu\":"<<median(vsFloatTimes[0].gpu)/median(vsFloatTimes[1].gpu)
              <<",\"vs_raw\":";eraw.write(record);
            record<<",\"vs_f32_cache\":";efloat.write(record);
            record<<",\"vs_cpu_serial_sample\":";ecpu.write(record);
            record<<",\"vs_cpu_simd32_sample\":";esimd.write(record);
            record<<",\"output_and_code_guards_pass\":true,\"input_unchanged\":true,\"deterministic\":true}";
            records.push_back(record.str());
            std::cerr<<"INT8 source codes "<<prefix<<" R"<<rows<<" "<<kTileNames[tile]
              <<" strict="<<pass<<" raw_speedup="<<median(vsRawTimes[0].gpu)/median(vsRawTimes[1].gpu)<<'\n';
          }
        }
      }
      std::ostringstream report;
      report<<std::setprecision(12)<<"{\"pass\":"<<(allPass?"true":"false")
        <<",\"completed_screen\":true,\"operator\":\"private exact-centered-affine-code INT8 MPP G64\""
        <<",\"numerical_semantics\":\"source unsigned q maps exactly to signed int8 c=q-center; original BF16 x; BF16 x INT8 G64 MPP/F32 dot; F32 SF*dot+(bias+center*SF)*F32sum_x; F32 group accumulation; BF16 output; reduction factorization is declared numerical alternative\""
        <<",\"activation_quantization\":false,\"weight_requantization\":false"
        <<",\"strict_relative_l2_tolerance\":0.0001,\"source_identity\":"<<splash::json::quote(weights.sourceIdentity())
        <<",\"manifest_fingerprint\":"<<splash::json::quote(weights.manifestFingerprint())
        <<",\"loaded_metallib_sha256\":"<<splash::json::quote(digestHex(backend.metallibSha256()))
        <<",\"device\":"<<splash::json::quote(backend.capabilities().deviceName)
        <<",\"qmv_f32_environment\":"<<splash::json::quote(std::getenv("SPLASH_FLASH_QMV_F32")?std::getenv("SPLASH_FLASH_QMV_F32"):"unset")
        <<",\"expert_qmv_environment\":"<<splash::json::quote(std::getenv("SPLASH_FLASH_EXPERT_QMV")?std::getenv("SPLASH_FLASH_EXPERT_QMV"):"unset")
        <<",\"raw_affine_route_semantics\":"<<splash::json::quote(flashAffineSemantics())
        <<",\"original_models_modified\":false,\"cases\":";writeRecords(report,records);report<<'}';
      writeReport(argv[3],report.str());
      return allPass?0:2;
    } catch(const std::exception &e) {
      if(argc==4) {
        std::ostringstream report;report<<"{\"pass\":false,\"error\":"<<splash::json::quote(e.what())<<",\"completed_cases\":";
        writeRecords(report,records);report<<'}';writeReport(argv[3],report.str());
      }
      std::cerr<<"INT8 affine code oracle failure: "<<e.what()<<'\n';
      return 1;
    }
  }
}
