// Private exact batched-GDN oracle. --cpu-only returns before construction of
// any MetalBackend/device; GPU execution belongs exclusively to Root.
#import <Metal/Metal.h>
#define main original_flash_gdn_oracle_main
#include "../tests/engine/flash_gdn_metal_test.mm"
#undef main
#include "flash_gdn_batch_ilp.hpp"
#include "engine/Json.hpp"
#include "BatchGdnOracleBuildProvenance.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <numeric>
#include <sstream>

namespace {
static_assert(sizeof(splash::metal::CommandTiming) == 200,
              "Recompile every backend/oracle object when timing ABI changes");
constexpr uint64_t kBatchGuard = 128;
constexpr uint8_t kBatchPoison = 0xa5;
constexpr std::array<GdnBatchIlpTile, 2> kBatchTiles{{{32, 32, 8}, {32, 16, 8}}};

std::string batchHex(const std::array<uint8_t, 32> &digest) {
  std::ostringstream out;
  for (uint8_t b : digest)
    out << std::hex << std::setfill('0') << std::setw(2) << uint32_t(b);
  return out.str();
}

std::string batchFileHash(const char *path) {
  std::ifstream in(path, std::ios::binary);
  require(bool(in), "cannot hash oracle provenance file");
  CC_SHA256_CTX context{};
  CC_SHA256_Init(&context);
  std::array<char, 65536> block;
  while (in) {
    in.read(block.data(), block.size());
    if (in.gcount())
      CC_SHA256_Update(&context, block.data(), CC_LONG(in.gcount()));
  }
  require(in.eof(), "provenance file read failed");
  std::array<uint8_t, 32> digest{};
  CC_SHA256_Final(digest.data(), &context);
  return batchHex(digest);
}

void batchValidTiming(const splash::metal::CommandTiming &timing) {
  for (double value : {timing.gpuSeconds, timing.wallSeconds})
    require(std::isfinite(value) && value > 1e-9 && value < 600,
            "reject invalid, implausible or ABI-corrupted command duration");
}

double batchMedian(std::vector<double> values) {
  require(!values.empty(), "no timing samples");
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  return n % 2 ? values[n / 2]
               : (values[n / 2 - 1] + values[n / 2]) * 0.5;
}

std::vector<uint32_t> batchCsv(const char *name, const char *fallback) {
  std::stringstream stream(std::getenv(name) ? std::getenv(name) : fallback);
  std::vector<uint32_t> result;
  std::string word;
  while (std::getline(stream, word, ',')) {
    size_t used = 0;
    const auto value = std::stoul(word, &used);
    require(used == word.size() && value <= 2048, "invalid oracle list");
    result.push_back(uint32_t(value));
  }
  require(!result.empty(), "empty oracle list");
  return result;
}

std::array<uint32_t, 4> batchRows(uint32_t slots, uint32_t rows,
                                 uint32_t pattern) {
  std::array<uint32_t, 4> result{};
  for (uint32_t slot = 0; slot < slots; ++slot)
    result[slot] = pattern == 0 ? rows
        : slot == 0 ? rows
        : slot == 1 ? std::max(1u, rows - 1)
        : slot == 2 ? std::max(1u, rows / 2) : 1;
  if (pattern >= 2 && slots >= 2)
    result[slots / 2] = 0;
  return result;
}

void batchChangeInputs(Reference &seed, uint32_t step) {
  Random random;
  random.state ^= uint64_t(step + 1) * 0x987b315cabULL;
  auto fill = [&](std::vector<uint16_t> &values, float scale) {
    for (auto &value : values)
      value = bits(random.value(scale));
  };
  fill(seed.qkv, .8f);
  fill(seed.z, 2.f);
  fill(seed.a, 1.5f);
  fill(seed.b, 2.f);
  seed.b[0] = bits(-6.84375f);
}

std::array<MetalBuffer FlashGDNBuffers::*, 9> batchFields() {
  return {&FlashGDNBuffers::qkv, &FlashGDNBuffers::z,
          &FlashGDNBuffers::a, &FlashGDNBuffers::b,
          &FlashGDNBuffers::mixed, &FlashGDNBuffers::decay,
          &FlashGDNBuffers::beta, &FlashGDNBuffers::recurrentRows,
          &FlashGDNBuffers::output};
}

constexpr std::array<uint32_t, 9> kBatchWidths{C, V, H, H, C, H, H, V, V};
constexpr std::array<uint32_t, 9> kBatchElementBytes{2, 2, 2, 2, 2, 4, 2, 2, 2};

struct BatchFixture {
  FlashTensor convolution, aLog, timeBias, norm;
  FlashGDNBuffers flat;
  std::array<FlashGDNState, 4> states;
  std::array<uint32_t, 4> rows;
  uint32_t slots, rowStride;
  bool inactiveHasState;

  BatchFixture(MetalBackend &backend, const Reference &seed,
               std::array<uint32_t, 4> counts, bool paddedInactive)
      : convolution(tensor(backend, seed.convolution, {C, 4, 1}, "batch conv")),
        aLog(tensor(backend, seed.aLog, {H}, "batch A log")),
        timeBias(tensor(backend, seed.timeBias, {H}, "batch dt bias")),
        norm(tensor(backend, seed.norm, {K}, "batch direct norm")),
        rows(counts), slots(seed.lanes), rowStride(seed.rows),
        inactiveHasState(paddedInactive) {
    const auto fields = batchFields();
    for (size_t field = 0; field < fields.size(); ++field)
      flat.*fields[field] = buffer(backend, uint64_t(slots) * rowStride *
          kBatchWidths[field] * kBatchElementBytes[field] + kBatchGuard,
          "independent padded batch row buffer");
    flat.diagnostics = buffer(backend, 4 + kBatchGuard, "batch diagnostics");
    for (uint32_t slot = 0; slot < slots; ++slot) {
      if (!rows[slot] && !inactiveHasState)
        continue;
      const uint64_t padding = kBatchGuard * (slot + 1);
      auto &state = states[slot];
      state.convolution = buffer(backend, flashGDNConvolutionLaneBytes() +
          padding, "independent batch convolution state");
      state.recurrent = buffer(backend, flashGDNRecurrentLaneBytes() + padding,
          "independent batch recurrent state");
      state.convolutionLaneStrideBytes = flashGDNConvolutionLaneBytes() + padding;
      state.recurrentLaneStrideBytes = flashGDNRecurrentLaneBytes() + padding;
    }
    resetState(seed);
    loadInputs(seed);
    clearScratch();
  }

  FlashGDNWeights weights() const {
    return {&convolution, &aLog, &timeBias, &norm};
  }

  std::array<GdnBatchIlpLane, 4> lanes() const {
    std::array<GdnBatchIlpLane, 4> result;
    for (uint32_t slot = 0; slot < slots; ++slot)
      result[slot] = {states[slot], rows[slot]};
    return result;
  }

  void resetState(const Reference &seed) {
    for (uint32_t slot = 0; slot < slots; ++slot) {
      const auto &state = states[slot];
      if (!state.recurrent)
        continue;
      std::memset(state.convolution.contents(), kBatchPoison,
                  state.convolution.sizeBytes());
      std::memset(state.recurrent.contents(), kBatchPoison,
                  state.recurrent.sizeBytes());
      if (!rows[slot])
        continue;
      std::memcpy(state.convolution.contents(),
                  seed.history.data() + uint64_t(slot) * 3 * C,
                  flashGDNConvolutionLaneBytes());
      std::memcpy(state.recurrent.contents(),
                  seed.recurrent.data() + uint64_t(slot) * StateElements,
                  flashGDNRecurrentLaneBytes());
    }
  }

  void loadInputs(const Reference &seed) {
    const auto fields = batchFields();
    const std::array<const std::vector<uint16_t> *, 4> inputs{
        &seed.qkv, &seed.z, &seed.a, &seed.b};
    for (size_t index = 0; index < inputs.size(); ++index) {
      const auto &target = flat.*fields[index];
      std::memset(target.contents(), kBatchPoison, target.sizeBytes());
      std::memcpy(target.contents(), inputs[index]->data(),
                  inputs[index]->size() * sizeof(uint16_t));
    }
  }

  void clearScratch() {
    const auto fields = batchFields();
    for (size_t index = 4; index < fields.size(); ++index) {
      const auto &target = flat.*fields[index];
      std::memset(target.contents(), kBatchPoison, target.sizeBytes());
    }
    std::memset(flat.diagnostics.contents(), kBatchPoison,
                flat.diagnostics.sizeBytes());
    *static_cast<uint32_t *>(flat.diagnostics.contents()) = 0;
  }

  FlashGDNBuffers laneViews(MetalBackend &backend, uint32_t slot) const {
    auto result = flat;
    const auto fields = batchFields();
    for (size_t index = 0; index < fields.size(); ++index) {
      const uint64_t rowBytes = uint64_t(kBatchWidths[index]) *
                                 kBatchElementBytes[index];
      result.*fields[index] = backend.view(flat.*fields[index],
          uint64_t(slot) * rowStride * rowBytes, rows[slot] * rowBytes);
    }
    result.diagnostics = backend.view(flat.diagnostics, 0, 4);
    return result;
  }
};

void batchEqualBuffer(const MetalBuffer &a, const MetalBuffer &b,
                        const char *label) {
  require(bool(a) == bool(b), std::string(label) + " presence differs");
  if (!a)
    return;
  require(a.sizeBytes() == b.sizeBytes(), std::string(label) + " size differs");
  if (!std::memcmp(a.contents(), b.contents(), a.sizeBytes()))
    return;
  const auto *x = static_cast<const uint8_t *>(a.contents());
  const auto *y = static_cast<const uint8_t *>(b.contents());
  uint64_t index = 0;
  while (index < a.sizeBytes() && x[index] == y[index])
    ++index;
  throw std::runtime_error(std::string(label) + " changed byte " +
                            std::to_string(index));
}

void batchExact(const BatchFixture &candidate, const BatchFixture &control) {
  const auto fields = batchFields();
  for (size_t index = 4; index < fields.size(); ++index)
    batchEqualBuffer(candidate.flat.*fields[index], control.flat.*fields[index],
                       "batch intermediate/output/tail/guard");
  batchEqualBuffer(candidate.flat.diagnostics, control.flat.diagnostics,
                     "batch diagnostics/guard");
  require(!*static_cast<const uint32_t *>(candidate.flat.diagnostics.contents()),
          "batch valid fixture set diagnostics");
  for (uint32_t slot = 0; slot < candidate.slots; ++slot) {
    batchEqualBuffer(candidate.states[slot].convolution,
        control.states[slot].convolution, "batch convolution history/padding");
    batchEqualBuffer(candidate.states[slot].recurrent,
        control.states[slot].recurrent, "batch recurrent F32 state/padding");
  }
}

void batchPoisonBytes(const MetalBuffer &target, uint64_t begin, uint64_t end,
                       const char *label) {
  const auto *data = static_cast<const uint8_t *>(target.contents());
  for (uint64_t index = begin; index < end; ++index)
    if (data[index] != kBatchPoison)
      throw std::runtime_error(std::string(label) + " overwritten byte " +
                                std::to_string(index));
}

void batchPreserved(const BatchFixture &fixture, const Reference &seed) {
  const auto fields = batchFields();
  const std::array<const std::vector<uint16_t> *, 4> inputs{
      &seed.qkv, &seed.z, &seed.a, &seed.b};
  for (size_t index = 0; index < inputs.size(); ++index) {
    const auto &target = fixture.flat.*fields[index];
    const uint64_t bytes = inputs[index]->size() * 2;
    require(!std::memcmp(target.contents(), inputs[index]->data(), bytes),
            "batch modified projected input bytes");
    batchPoisonBytes(target, bytes, target.sizeBytes(), "input guard");
  }
  for (const auto &[target, values] :
       std::array<std::pair<const FlashTensor *, const std::vector<uint16_t> *>, 4>{{
         {&fixture.convolution, &seed.convolution}, {&fixture.aLog, &seed.aLog},
         {&fixture.timeBias, &seed.timeBias}, {&fixture.norm, &seed.norm}}})
    require(!std::memcmp(target->buffer.contents(), values->data(),
                        values->size() * 2), "batch modified readonly weights");
  for (size_t index = 4; index < fields.size(); ++index) {
    const auto &target = fixture.flat.*fields[index];
    const uint64_t rowBytes = uint64_t(kBatchWidths[index]) * kBatchElementBytes[index];
    for (uint32_t slot = 0; slot < fixture.slots; ++slot)
      batchPoisonBytes(target, (uint64_t(slot) * fixture.rowStride +
          fixture.rows[slot]) * rowBytes, uint64_t(slot + 1) *
          fixture.rowStride * rowBytes, "inactive/padded scratch rows");
    batchPoisonBytes(target, uint64_t(fixture.slots) * fixture.rowStride * rowBytes,
                       target.sizeBytes(), "scratch guard");
  }
  for (uint32_t slot = 0; slot < fixture.slots; ++slot) {
    const auto &state = fixture.states[slot];
    if (!state.recurrent)
      continue;
    batchPoisonBytes(state.convolution, fixture.rows[slot] ?
        flashGDNConvolutionLaneBytes() : 0, state.convolution.sizeBytes(),
        "history padding or inactive state");
    batchPoisonBytes(state.recurrent, fixture.rows[slot] ?
        flashGDNRecurrentLaneBytes() : 0, state.recurrent.sizeBytes(),
        "recurrent padding or inactive state");
  }
  batchPoisonBytes(fixture.flat.diagnostics, 4,
                       fixture.flat.diagnostics.sizeBytes(), "diagnostics guard");
}

CommandGraph batchControlGraph(MetalBackend &backend, const BatchFixture &fixture) {
  CommandGraph result;
  for (uint32_t slot = 0; slot < fixture.slots; ++slot)
    if (fixture.rows[slot])
      addGDNStagedPrefill(result, fixture.weights(), fixture.laneViews(backend, slot),
          fixture.states[slot], fixture.rows[slot], 1,
          FlashGDNStageTile::Values16Time16);
  return result;
}

CommandGraph batchCandidateGraph(MetalBackend &backend,
                                  const BatchFixture &fixture, GdnBatchIlpTile tile) {
  CommandGraph result;
  const auto lanes = fixture.lanes();
  addGdnBatchIlp(backend, result, fixture.weights(), fixture.flat,
      std::span(lanes.data(), fixture.slots), fixture.rowStride, tile);
  return result;
}

std::vector<splash::metal::ComputeDispatch> batchRecurrenceDispatches(
    const CommandGraph &graph) {
  std::vector<splash::metal::ComputeDispatch> result;
  for (const auto &dispatch : graph.dispatches())
    if (dispatch.pipelineName.find("staged_v16_t16") != std::string::npos ||
        dispatch.pipelineName.find("private_gdn_batch_") != std::string::npos)
      result.push_back(dispatch);
  require(!result.empty(), "oracle cannot identify recurrence dispatch");
  return result;
}

uint32_t batchRejectsInvalid(MetalBackend &backend) {
  Reference seed(7, 4, false);
  BatchFixture fixture(backend, seed, {4, 3, 2, 1}, false);
  const auto originalLanes = fixture.lanes();
  uint32_t checks = 0;
  auto reject = [&](auto function) {
    CommandGraph graph;
    graph.add("host_only_preexisting_dispatch", {}, {1, 1, 1});
    bool rejected = false;
    try { function(graph); }
    catch (const std::invalid_argument &) { rejected = true; }
    catch (const splash::metal::MetalBackendError &) { rejected = true; }
    require(rejected && graph.dispatches().size() == 1 &&
        graph.dispatches()[0].pipelineName == "host_only_preexisting_dispatch",
        "invalid batched GDN metadata changed its existing graph");
    ++checks;
  };
  auto invoke = [&](CommandGraph &graph, const FlashGDNWeights &weights,
                    const FlashGDNBuffers &flat,
                    std::span<const GdnBatchIlpLane> lanes, uint32_t stride,
                    GdnBatchIlpTile tile = {32, 32, 8}, float epsilon = 1e-6f) {
    addGdnBatchIlp(backend, graph, weights, flat, lanes, stride, tile, epsilon);
  };
  reject([&](auto &g) { invoke(g, fixture.weights(), fixture.flat, {}, 7); });
  reject([&](auto &g) {
    std::array<GdnBatchIlpLane, 5> lanes;
    std::copy(originalLanes.begin(), originalLanes.end(), lanes.begin());
    invoke(g, fixture.weights(), fixture.flat, lanes, 7);
  });
  reject([&](auto &g) { invoke(g, fixture.weights(), fixture.flat, originalLanes, 0); });
  reject([&](auto &g) { invoke(g, fixture.weights(), fixture.flat, originalLanes, 3); });
  reject([&](auto &g) { invoke(g, fixture.weights(), fixture.flat, originalLanes, 4097); });
  reject([&](auto &g) { invoke(g, fixture.weights(), fixture.flat, originalLanes, 7,
      {32, 32, 8}, std::numeric_limits<float>::quiet_NaN()); });
  reject([&](auto &g) { invoke(g, fixture.weights(), fixture.flat, originalLanes, 7,
      {32, 32, 8}, 0.f); });
  for (GdnBatchIlpTile tile : std::array<GdnBatchIlpTile, 3>{{{16, 32, 8}, {32, 8, 8}, {32, 32, 4}}})
    reject([&](auto &g) { invoke(g, fixture.weights(), fixture.flat, originalLanes, 7, tile); });
  reject([&](auto &g) {
    auto lanes = originalLanes; lanes[3].rows = 8;
    invoke(g, fixture.weights(), fixture.flat, lanes, 7);
  });
  for (bool convolution : {false, true})
    reject([&](auto &g) {
      auto lanes = originalLanes;
      if (convolution) lanes[2].state.convolution = {};
      else lanes[2].state.recurrent = {};
      invoke(g, fixture.weights(), fixture.flat, lanes, 7);
    });
  reject([&](auto &g) {
    auto lanes = originalLanes; lanes[2].state = lanes[0].state;
    invoke(g, fixture.weights(), fixture.flat, lanes, 7);
  });
  reject([&](auto &g) {
    auto lanes = originalLanes;
    lanes[2].state.recurrent = backend.view(lanes[0].state.recurrent, 4,
        flashGDNRecurrentLaneBytes());
    invoke(g, fixture.weights(), fixture.flat, lanes, 7);
  });
  reject([&](auto &g) {
    auto lanes = originalLanes;
    lanes[2].state.recurrentLaneStrideBytes = 3;
    invoke(g, fixture.weights(), fixture.flat, lanes, 7);
  });
  reject([&](auto &g) {
    auto flat = fixture.flat; flat.output = flat.qkv;
    invoke(g, fixture.weights(), flat, originalLanes, 7);
  });
  reject([&](auto &g) {
    auto flat = fixture.flat;
    flat.qkv = backend.view(flat.qkv, 0, 2);
    invoke(g, fixture.weights(), flat, originalLanes, 7);
  });
  reject([&](auto &g) {
    auto weights = fixture.weights(); weights.norm = nullptr;
    invoke(g, weights, fixture.flat, originalLanes, 7);
  });
  {
    const std::array<GdnBatchIlpLane, 4> masked{};
    CommandGraph graph;
    graph.add("host_only_preexisting_dispatch", {}, {1, 1, 1});
    invoke(graph, {}, {}, masked, 7);
    require(graph.dispatches().size() == 1,
            "all-masked valid request appended GPU work");
    ++checks;
  }
  return checks;
}

void batchCpuOnly() {
  cpuOnly();
  uint64_t checks = 0;
  const std::array<GdnBatchIlpLane, 4> masked{};
  for (uint32_t slots = 1; slots <= 4; ++slots) {
    require(validateGdnBatchIlpGeometry(std::span(masked.data(), slots), 4096) == 0,
            "masked geometry requires synthetic state");
    ++checks;
  }
  auto rejectGeometry = [&](auto function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid CPU geometry/missing live states accepted");
    ++checks;
  };
  rejectGeometry([&] { (void)validateGdnBatchIlpGeometry({}, 7); });
  rejectGeometry([&] {
    const std::array<GdnBatchIlpLane, 5> invalid{};
    (void)validateGdnBatchIlpGeometry(invalid, 7);
  });
  for (uint32_t stride : {0u, 4097u})
    rejectGeometry([&] { (void)validateGdnBatchIlpGeometry(masked, stride); });
  for (float epsilon : {0.f, -1.f, float(NAN), float(INFINITY)})
    rejectGeometry([&] { (void)validateGdnBatchIlpGeometry(masked, 7, {}, epsilon); });
  for (GdnBatchIlpTile tile : std::array<GdnBatchIlpTile, 3>{{
           {16, 32, 8}, {32, 8, 8}, {32, 32, 4}}})
    rejectGeometry([&] { validateGdnBatchIlpTile(tile); });
  for (uint32_t rows : {1u, 8u, 2049u})
    rejectGeometry([&] {
      auto invalid = masked; invalid[2].rows = rows;
      (void)validateGdnBatchIlpGeometry(invalid, 7);
    });
  for (auto tile : kBatchTiles) {
    validateGdnBatchIlpTile(tile);
    std::array<uint32_t, 128> visits{};
    for (uint32_t group = 0; group < 128 / tile.values; ++group)
      for (uint32_t sg = 0; sg < tile.simds; ++sg)
        for (uint32_t row = 0; row < tile.values / tile.simds; ++row)
          ++visits[group * tile.values + row * tile.simds + sg];
    for (auto count : visits) { require(count == 1, "ILP key/value ownership differs"); ++checks; }
    for (uint32_t slots = 1; slots <= 4; ++slots)
      for (uint32_t maximum : {1u, 7u, 17u, 33u, 128u, 257u, 512u, 2048u})
        for (uint32_t pattern = 0; pattern <= 3; ++pattern) {
          const auto rows = batchRows(slots, maximum, pattern);
          const uint32_t stride = maximum + 3;
          uint64_t actual = 0, visited = 0;
          for (uint32_t slot = 0; slot < slots; ++slot) {
            actual += rows[slot];
            for (uint32_t begin = 0; begin < rows[slot]; begin += tile.time)
              visited += std::min(tile.time, rows[slot] - begin);
            require(rows[slot] <= stride && uint64_t(slot) * stride + rows[slot] <=
                    uint64_t(slots) * stride, "ragged real-row packing escapes extent");
            ++checks;
          }
          require(actual == visited, "ILP temporal blocks omit or invent rows"); ++checks;
        }
  }
  Reference first(7, 4, false);
  const auto original = first.qkv;
  batchChangeInputs(first, 1);
  require(first.qkv != original, "carried sequence inputs did not change"); ++checks;
  for (double invalid : {0., -1., 1e-10, 1e-314, double(NAN), double(INFINITY), 601.})
    for (bool gpu : {false, true}) {
      splash::metal::CommandTiming t;
      t.gpuSeconds = gpu ? invalid : .01; t.wallSeconds = gpu ? .01 : invalid;
      bool rejected = false;
      try { batchValidTiming(t); } catch (const std::runtime_error &) { rejected = true; }
      require(rejected, "invalid timing was accepted"); ++checks;
    }
  splash::metal::CommandTiming valid;
  valid.gpuSeconds = 1e-5; valid.wallSeconds = 1e-4;
  batchValidTiming(valid); ++checks;
  std::cout << "{\"pass\":true,\"batch_layout_timing_cpu_checks\":" << checks
      << ",\"command_timing_size_bytes\":" << sizeof(valid)
      << ",\"metal_backend_constructions\":0,\"gpu_commands\":0}\n";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-only") {
        batchCpuOnly(); return 0;
      }
      if (argc != 3)
        throw std::invalid_argument("usage: flash-gdn-batch-ilp-oracle METALLIB REPORT_JSON | --cpu-only");
      const auto rowsList = batchCsv("GDN_BATCH_ILP_ROWS", "17,33,128,512");
      const auto slotsList = batchCsv("GDN_BATCH_ILP_SLOTS", "1,2,3,4");
      const auto coldList = batchCsv("GDN_BATCH_ILP_COLD", "0,1");
      const auto extremeList = batchCsv("GDN_BATCH_ILP_EXTREMES", "0,1");
      const auto patterns = batchCsv("GDN_BATCH_ILP_PATTERNS", "0,1,2,3");
      const auto pairsList = batchCsv("GDN_BATCH_ILP_PAIRS", "6");
      require(pairsList.size() == 1 && pairsList[0], "invalid matched pair count");
      const uint32_t pairs = pairsList[0];
      const std::string filter = std::getenv("GDN_BATCH_ILP_FILTER") ?
          std::getenv("GDN_BATCH_ILP_FILTER") : "";
      const bool recurrenceBench = !std::getenv("GDN_BATCH_ILP_RECURRENCE_BENCH") ||
          std::string(std::getenv("GDN_BATCH_ILP_RECURRENCE_BENCH")) != "0";
      const bool shaderValidation = std::getenv("MTL_SHADER_VALIDATION") &&
          std::string(std::getenv("MTL_SHADER_VALIDATION")) != "0";
      const std::string executableHash = batchFileHash(argv[0]);
      MetalBackend backend(argv[1]);
      const std::string libraryHash = batchHex(backend.metallibSha256());
      std::ostringstream provenance;
      provenance << "{\"executable_file_sha256\":" << splash::json::quote(executableHash)
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(libraryHash)
          << ",\"command_timing_size_bytes\":" << sizeof(splash::metal::CommandTiming)
          << ",\"build_provenance\":" << kBatchGdnOracleBuildProvenance << '}';
      std::cerr << provenance.str() << '\n';
      std::ofstream preGpu(std::string(argv[2]) + ".provenance.json");
      require(bool(preGpu), "cannot write pre-GPU provenance");
      preGpu << provenance.str() << '\n'; preGpu.close();
      const uint32_t rejectionChecks = batchRejectsInvalid(backend);
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      NSError *error = nil;
      id<MTLLibrary> library = [device newLibraryWithURL:
          [NSURL fileURLWithPath:@(argv[1])] error:&error];
      require(device && library, "batched ILP library preflight failed");
      std::vector<std::string> records, resources;
      uint32_t skippedDebugVariants = 0;
      for (auto tile : kBatchTiles) {
        const auto name = gdnBatchIlpName(tile);
        if (!filter.empty() && name.find(filter) == std::string::npos)
          continue;
        error = nil;
        id<MTLFunction> function = [library newFunctionWithName:@(name.c_str())];
        require(function, "batch ILP library missing requested function");
        id<MTLComputePipelineState> pipeline = [device
            newComputePipelineStateWithFunction:function error:&error];
        const bool supported = pipeline && pipeline.threadExecutionWidth == 32 &&
            pipeline.maxTotalThreadsPerThreadgroup >= tile.simds * 32 &&
            pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength;
        std::ostringstream resource;
        resource << "{\"tile\":" << splash::json::quote(name)
            << ",\"requested_threads\":" << tile.simds * 32
            << ",\"pipeline_max_threads\":" << (pipeline ? pipeline.maxTotalThreadsPerThreadgroup : 0)
            << ",\"pipeline_tgm_bytes\":" << (pipeline ? pipeline.staticThreadgroupMemoryLength : 0)
            << ",\"device_tgm_limit_bytes\":" << device.maxThreadgroupMemoryLength
            << ",\"shader_validation\":" << (shaderValidation ? "true" : "false")
            << ",\"supported\":" << (supported ? "true" : "false")
            << ",\"pipeline_creation_error\":" << splash::json::quote(pipeline ? "" :
                (error.localizedDescription.UTF8String ?: "pipeline creation failed")) << '}';
        resources.push_back(resource.str()); std::cerr << resource.str() << '\n';
        if (!supported) {
          require(shaderValidation, "normal batched ILP resource limit exceeded");
          ++skippedDebugVariants; continue;
        }
        for (uint32_t maximum : rowsList) for (uint32_t slots : slotsList)
          for (uint32_t cold : coldList) for (uint32_t extremes : extremeList)
            for (uint32_t pattern : patterns) {
              require(maximum && maximum <= 2048 && slots && slots <= 4 &&
                  cold <= 1 && extremes <= 1 && pattern <= 3, "invalid batch fixture geometry");
              if (slots == 1 && pattern >= 2)
                continue;
              const auto counts = batchRows(slots, maximum, pattern);
              const uint32_t actualRows = std::accumulate(counts.begin(), counts.end(), 0u);
              const uint32_t live = uint32_t(std::count_if(counts.begin(), counts.begin() + slots,
                  [](uint32_t count) { return count != 0; }));
              Reference seed(maximum + 3, slots, cold);
              if (extremes) {
                for (uint32_t h = 0; h < H; ++h)
                  seed.aLog[h] = bits(std::array<float, 4>{-8, -4, 4, 8}[h % 4]);
                for (uint32_t d = 0; d < K; ++d)
                  seed.norm[d] = bits(std::array<float, 4>{0, -1, .5f, 2}[d % 4]);
              }
              BatchFixture candidate(backend, seed, counts, pattern == 3);
              BatchFixture control(backend, seed, counts, pattern == 3);
              auto candidateGraph = batchCandidateGraph(backend, candidate, tile);
              auto controlGraph = batchControlGraph(backend, control);
              require(candidateGraph.dispatches().size() == 3 * live + 1 &&
                  controlGraph.dispatches().size() == 4 * live,
                  "batch graph dispatch counts differ from direct-state design");
              for (uint32_t step = 0; step < 2; ++step) {
                batchChangeInputs(seed, step);
                candidate.loadInputs(seed); control.loadInputs(seed);
                candidate.clearScratch(); control.clearScratch();
                batchValidTiming(backend.submitCommand(controlGraph.dispatches()));
                batchValidTiming(backend.submitCommand(candidateGraph.dispatches()));
                batchExact(candidate, control);
                batchPreserved(candidate, seed); batchPreserved(control, seed);
              }
              std::vector<double> cGPU, nGPU, cWall, nWall;
              for (uint32_t pair = 0; pair < pairs + 2; ++pair) {
                candidate.resetState(seed); control.resetState(seed);
                candidate.clearScratch(); control.clearScratch();
                splash::metal::CommandTiming c, n;
                if (pair % 2) {
                  n = backend.submitCommand(candidateGraph.dispatches());
                  c = backend.submitCommand(controlGraph.dispatches());
                } else {
                  c = backend.submitCommand(controlGraph.dispatches());
                  n = backend.submitCommand(candidateGraph.dispatches());
                }
                batchValidTiming(c); batchValidTiming(n);
                batchExact(candidate, control);
                if (pair >= 2) {
                  cGPU.push_back(c.gpuSeconds); nGPU.push_back(n.gpuSeconds);
                  cWall.push_back(c.wallSeconds); nWall.push_back(n.wallSeconds);
                }
              }
              std::vector<double> cRecGPU, nRecGPU, cRecWall, nRecWall;
              if (recurrenceBench) {
                const auto cRec = batchRecurrenceDispatches(controlGraph);
                const auto nRec = batchRecurrenceDispatches(candidateGraph);
                require(cRec.size() == live && nRec.size() == 1,
                    "recurrence-only benchmark did not select exact live dispatches");
                for (uint32_t pair = 0; pair < pairs + 2; ++pair) {
                  candidate.resetState(seed); control.resetState(seed);
                  splash::metal::CommandTiming c, n;
                  if (pair % 2) { n = backend.submitCommand(nRec); c = backend.submitCommand(cRec); }
                  else { c = backend.submitCommand(cRec); n = backend.submitCommand(nRec); }
                  batchValidTiming(c); batchValidTiming(n);
                  batchExact(candidate, control);
                  if (pair >= 2) {
                    cRecGPU.push_back(c.gpuSeconds); nRecGPU.push_back(n.gpuSeconds);
                    cRecWall.push_back(c.wallSeconds); nRecWall.push_back(n.wallSeconds);
                  }
                }
              }
              batchPreserved(candidate, seed); batchPreserved(control, seed);
              std::ostringstream out;
              out << std::setprecision(12) << "{\"tile\":" << splash::json::quote(name)
                  << ",\"maximum_real_rows\":" << maximum << ",\"input_rows_stride\":" << seed.rows
                  << ",\"slots\":" << slots << ",\"live_lanes\":" << live
                  << ",\"actual_rows_total\":" << actualRows << ",\"lane_rows\":[";
              for (uint32_t slot = 0; slot < slots; ++slot) {
                if (slot) out << ','; out << counts[slot];
              }
              out << "],\"cold\":" << cold << ",\"extremes\":" << extremes
                  << ",\"pattern\":" << pattern << ",\"inactive_present_state\":"
                  << (pattern == 3 ? "true" : "false")
                  << ",\"candidate_dispatches\":" << candidateGraph.dispatches().size()
                  << ",\"control_dispatches\":" << controlGraph.dispatches().size()
                  << ",\"pairs\":" << pairs << ",\"control_gpu_seconds\":" << batchMedian(cGPU)
                  << ",\"candidate_gpu_seconds\":" << batchMedian(nGPU)
                  << ",\"gpu_speedup\":" << batchMedian(cGPU) / batchMedian(nGPU)
                  << ",\"control_wall_seconds\":" << batchMedian(cWall)
                  << ",\"candidate_wall_seconds\":" << batchMedian(nWall)
                  << ",\"candidate_actual_rows_per_gpu_second\":" << actualRows / batchMedian(nGPU)
                  << ",\"two_changed_input_carried_sequences_bit_exact\":true"
                     ",\"all_intermediates_output_f32_state_history_padding_bit_exact\":true"
                     ",\"inactive_rows_states_and_all_guards_untouched\":true"
                     ",\"projected_inputs_weights_unchanged\":true";
              if (recurrenceBench)
                out << ",\"recurrence_control_gpu_seconds\":" << batchMedian(cRecGPU)
                    << ",\"recurrence_candidate_gpu_seconds\":" << batchMedian(nRecGPU)
                    << ",\"recurrence_gpu_speedup\":" << batchMedian(cRecGPU) / batchMedian(nRecGPU)
                    << ",\"recurrence_control_wall_seconds\":" << batchMedian(cRecWall)
                    << ",\"recurrence_candidate_wall_seconds\":" << batchMedian(nRecWall);
              out << '}'; records.push_back(out.str());
              std::cerr << name << " max_rows=" << maximum << " slots=" << slots
                  << " live=" << live << " pattern=" << pattern << " exact; speedup="
                  << batchMedian(cGPU) / batchMedian(nGPU) << '\n';
            }
      }
      require(!records.empty(), "filter selected no supported batch fixtures");
      std::ofstream report(argv[2]); require(bool(report), "cannot write batch report");
      report << "{\"pass\":true,\"executable_file_sha256\":" << splash::json::quote(executableHash)
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(libraryHash)
          << ",\"command_timing_size_bytes\":" << sizeof(splash::metal::CommandTiming)
          << ",\"build_provenance\":" << kBatchGdnOracleBuildProvenance
          << ",\"host_preflight_rejection_checks\":" << rejectionChecks
          << ",\"control\":\"per-live-lane current staged V16/T16 with independent F32 state\""
             ",\"error_limit\":\"zero bytes for every intermediate/output/state/history/padding\""
             ",\"timing_validation\":\"finite;1e-9<gpu/wall<600seconds\""
             ",\"rows_normalization\":\"sum of supplied real lane rows only\""
             ",\"timing_scope\":\"complete graph; optional recurrence-only command separately\""
             ",\"warmups_per_route\":2,\"alternating_matched_pairs\":true"
          << ",\"skipped_debug_variants\":" << skippedDebugVariants << ",\"resources\":[";
      for (size_t i = 0; i < resources.size(); ++i) { if (i) report << ','; report << resources[i]; }
      report << "],\"cases\":[";
      for (size_t i = 0; i < records.size(); ++i) { if (i) report << ','; report << records[i]; }
      report << "]}\n";
      return 0;
    } catch (const std::exception &e) {
      std::cerr << "flash-gdn-batch-ilp-oracle: " << e.what() << '\n';
      return 1;
    }
  }
}
