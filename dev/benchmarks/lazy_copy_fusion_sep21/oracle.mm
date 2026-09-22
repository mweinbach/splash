// PRIVATE qualification only. --cpu-only returns before Metal construction.
// Frozen fixture contains an independent scalar oracle; its main is renamed.
#import <Metal/Metal.h>
#define main original_flash_gdn_fixture_main
#include "gdn_fixture.hpp"
#undef main
#include "flash/FlashGDNLazyRollback.hpp"
#include "metal/abi/FlashGDNLazyRollback.h"
#include "engine/Json.hpp"
#include "LazyCopyFusionOracleBuildProvenance.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>

namespace {
static_assert(sizeof(splash::metal::CommandTiming) == 200);
constexpr uint8_t lazyPoison = 0xa5;
constexpr uint64_t lazyGuard = 128;

std::string lazyHex(const std::array<uint8_t, 32> &digest) {
  std::ostringstream out;
  for (uint8_t byte : digest)
    out << std::hex << std::setfill('0') << std::setw(2) << uint32_t(byte);
  return out.str();
}
std::string lazyHash(const char *path) {
  std::ifstream in(path, std::ios::binary);
  require(bool(in), "cannot open provenance file");
  CC_SHA256_CTX context{}; CC_SHA256_Init(&context);
  std::array<char, 65536> block;
  while (in) {
    in.read(block.data(), block.size());
    if (in.gcount()) CC_SHA256_Update(&context, block.data(), CC_LONG(in.gcount()));
  }
  require(in.eof(), "provenance read failed");
  std::array<uint8_t, 32> result{}; CC_SHA256_Final(result.data(), &context);
  return lazyHex(result);
}
void lazyValidTiming(const splash::metal::CommandTiming &timing) {
  for (double value : {timing.gpuSeconds, timing.wallSeconds})
    require(std::isfinite(value) && value > 1e-9 && value < 600,
            "reject implausible or timing-ABI-corrupted command duration");
}
std::vector<uint32_t> lazyCsv(const char *name, const char *fallback) {
  std::stringstream stream(std::getenv(name) ? std::getenv(name) : fallback);
  std::vector<uint32_t> values; std::string word;
  while (std::getline(stream, word, ',')) {
    size_t used = 0; const auto value = std::stoul(word, &used);
    require(used == word.size() && value <= 1024, "invalid oracle list");
    values.push_back(uint32_t(value));
  }
  require(!values.empty(), "empty oracle list"); return values;
}
void lazyExactBytes(const void *got, const void *wanted, uint64_t bytes,
                    const char *label) {
  if (!std::memcmp(got, wanted, bytes)) return;
  const auto *a = static_cast<const uint8_t *>(got);
  const auto *b = static_cast<const uint8_t *>(wanted);
  uint64_t offset = 0; while (offset < bytes && a[offset] == b[offset]) ++offset;
  throw std::runtime_error(std::string(label) + " differs at byte " + std::to_string(offset));
}
std::vector<uint8_t> lazyBytes(const MetalBuffer &buffer) {
  require(bool(buffer) && buffer.contents(), "missing addressable oracle view");
  const auto *begin = static_cast<const uint8_t *>(buffer.contents());
  return {begin, begin + buffer.sizeBytes()};
}
void lazyPoisonTail(const MetalBuffer &buffer, uint64_t begin, uint64_t end) {
  const auto *bytes = static_cast<const uint8_t *>(buffer.contents());
  for (uint64_t i = begin; i < end; ++i)
    if (bytes[i] != lazyPoison)
      throw std::runtime_error("lazy GDN wrote guard/padding byte " + std::to_string(i));
}
void lazyPoisonStates(Fixture &fixture) {
  for (uint32_t lane = 0; lane < fixture.state.recurrent.sizeBytes() / fixture.recurrentStride; ++lane) {
    std::memset(static_cast<uint8_t *>(fixture.state.convolution.contents()) +
        lane * fixture.convStride + flashGDNConvolutionLaneBytes(), lazyPoison, lazyGuard);
    std::memset(static_cast<uint8_t *>(fixture.state.recurrent.contents()) +
        lane * fixture.recurrentStride + flashGDNRecurrentLaneBytes(), lazyPoison, lazyGuard);
  }
}
void lazyReset(Fixture &fixture, const Reference &seed) {
  for (uint32_t lane = 0; lane < seed.lanes; ++lane) {
    std::memcpy(static_cast<uint8_t *>(fixture.state.convolution.contents()) + lane * fixture.convStride,
        seed.history.data() + uint64_t(lane) * 3 * C, flashGDNConvolutionLaneBytes());
    std::memcpy(static_cast<uint8_t *>(fixture.state.recurrent.contents()) + lane * fixture.recurrentStride,
        seed.recurrent.data() + uint64_t(lane) * StateElements, flashGDNRecurrentLaneBytes());
  }
  lazyPoisonStates(fixture);
  *static_cast<uint32_t *>(fixture.buffers.diagnostics.contents()) = 0;
}
void lazyChangedInputs(Reference &seed, uint32_t sequence) {
  Random random; random.state ^= uint64_t(sequence + 1) * 0xd31583b14aULL;
  auto fill = [&](std::vector<uint16_t> &values, float scale) {
    for (auto &value : values) value = bits(random.value(scale));
  };
  fill(seed.qkv, .8f); fill(seed.z, 2.f); fill(seed.a, 1.5f); fill(seed.b, 2.f);
  seed.b[0] = bits(-6.84375f);
}
void lazyLoad(Fixture &fixture, const Reference &seed) {
  copy(fixture.buffers.qkv, seed.qkv); copy(fixture.buffers.z, seed.z);
  copy(fixture.buffers.a, seed.a); copy(fixture.buffers.b, seed.b);
}
void lazyCompareStates(const Fixture &candidate, const Fixture &reference) {
  require(candidate.state.convolution.sizeBytes() == reference.state.convolution.sizeBytes() &&
      candidate.state.recurrent.sizeBytes() == reference.state.recurrent.sizeBytes(),
      "oracle state extent mismatch");
  lazyExactBytes(candidate.state.convolution.contents(), reference.state.convolution.contents(),
      candidate.state.convolution.sizeBytes(), "lazy convolution state/padding");
  lazyExactBytes(candidate.state.recurrent.contents(), reference.state.recurrent.contents(),
      candidate.state.recurrent.sizeBytes(), "lazy FP32 state/padding");
}
void lazyCompareVerify(const Fixture &candidate, const Fixture &reference,
                         const FlashGDNLazyRollback *lazy = nullptr) {
  const auto prepared = lazy ? lazy->savedBuffers() : candidate.buffers;
  for (const auto &[a, b] : std::array<std::pair<MetalBuffer, MetalBuffer>, 5>{{
      {prepared.mixed, reference.buffers.mixed},
      {prepared.decay, reference.buffers.decay},
      {prepared.beta, reference.buffers.beta},
      {candidate.buffers.recurrentRows, reference.buffers.recurrentRows},
      {candidate.buffers.output, reference.buffers.output}}})
    lazyExactBytes(a.contents(), b.contents(), a.sizeBytes(), "lazy verify intermediate/output");
  lazyCompareStates(candidate, reference);
  require(!*static_cast<const uint32_t *>(candidate.buffers.diagnostics.contents()) &&
      !*static_cast<const uint32_t *>(reference.buffers.diagnostics.contents()),
      "valid lazy/captured GDN fixture set diagnostics");
}
void lazyInputsPreserved(const Fixture &fixture, const Reference &seed) {
  const std::array<std::pair<MetalBuffer, const std::vector<uint16_t> *>, 8> checks{{
      {fixture.buffers.qkv, &seed.qkv}, {fixture.buffers.z, &seed.z},
      {fixture.buffers.a, &seed.a}, {fixture.buffers.b, &seed.b},
      {fixture.convolution.buffer, &seed.convolution}, {fixture.aLog.buffer, &seed.aLog},
      {fixture.timeBias.buffer, &seed.timeBias}, {fixture.norm.buffer, &seed.norm}}};
  for (const auto &[buffer, values] : checks)
    lazyExactBytes(buffer.contents(), values->data(), values->size() * 2,
                   "lazy projected input/immutable weights");
}

struct LazyEager {
  MetalBuffer tape;
  uint64_t rowStride, laneStride;
  uint32_t rows, lanes;
  LazyEager(MetalBackend &backend, uint32_t r, uint32_t b)
      : rowStride(flashGDNRecurrentLaneBytes() + lazyGuard),
        laneStride(uint64_t(r - 1) * rowStride + lazyGuard), rows(r), lanes(b) {
    if (r > 1) {
      tape = buffer(backend, uint64_t(b) * laneStride, "guarded eager prefix snapshots");
      std::memset(tape.contents(), lazyPoison, tape.sizeBytes());
    }
  }
  void encode(CommandGraph &graph, const Fixture &fixture) const {
    addGDNFusedCaptured(graph, fixture.weights(), fixture.buffers, fixture.state,
        FlashGDNCapture{tape, rowStride, laneStride, rows - 1, 512}, rows, lanes);
  }
  void guards() const {
    if (rows == 1) return;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      for (uint32_t row = 0; row < rows - 1; ++row)
        lazyPoisonTail(tape, lane * laneStride + row * rowStride + flashGDNRecurrentLaneBytes(),
            lane * laneStride + (row + 1) * rowStride);
      lazyPoisonTail(tape, lane * laneStride + (rows - 1) * rowStride, (lane + 1) * laneStride);
    }
  }
  const void *prefixState(const Fixture &full, uint32_t lane, uint32_t kept) const {
    if (!kept || kept == rows)
      return static_cast<const uint8_t *>(full.state.recurrent.contents()) + lane * full.recurrentStride;
    return static_cast<const uint8_t *>(tape.contents()) + lane * laneStride + (kept - 1) * rowStride;
  }
};

void lazyReferencePrefix(Fixture &expected, const Fixture &full, const LazyEager &eager,
                          const Reference &seed, std::span<const uint32_t> kept) {
  std::memcpy(expected.state.convolution.contents(), full.state.convolution.contents(),
              full.state.convolution.sizeBytes());
  std::memcpy(expected.state.recurrent.contents(), full.state.recurrent.contents(),
              full.state.recurrent.sizeBytes());
  for (uint32_t lane = 0; lane < seed.lanes; ++lane) {
    if (!kept[lane] || kept[lane] == seed.rows) continue;
    std::memcpy(static_cast<uint8_t *>(expected.state.recurrent.contents()) + lane * expected.recurrentStride,
                eager.prefixState(full, lane, kept[lane]), flashGDNRecurrentLaneBytes());
    auto *history = reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(
        expected.state.convolution.contents()) + lane * expected.convStride);
    // Independent concatenation oracle, not the rollback implementation's helper.
    for (uint32_t outRow = 0; outRow < 3; ++outRow) {
      const uint32_t concatenated = kept[lane] + outRow;
      const uint16_t *source = concatenated < 3
          ? seed.history.data() + (uint64_t(lane) * 3 + concatenated) * C
          : seed.qkv.data() + (uint64_t(lane) * seed.rows + concatenated - 3) * C;
      std::memcpy(history + outRow * C, source, uint64_t(C) * 2);
    }
  }
}

FlashGDNBuffers lazyLaneViews(MetalBackend &backend, const Fixture &fixture,
                               uint32_t rowCount, uint32_t lane) {
  auto result = fixture.buffers;
  const std::array<MetalBuffer FlashGDNBuffers::*, 9> fields{
      &FlashGDNBuffers::qkv, &FlashGDNBuffers::z, &FlashGDNBuffers::a, &FlashGDNBuffers::b,
      &FlashGDNBuffers::mixed, &FlashGDNBuffers::decay, &FlashGDNBuffers::beta,
      &FlashGDNBuffers::recurrentRows, &FlashGDNBuffers::output};
  constexpr std::array<uint32_t, 9> widths{C, V, H, H, C, H, H, V, V};
  for (size_t i = 0; i < fields.size(); ++i) {
    const uint64_t bytes = uint64_t(rowCount) * widths[i] * (i == 5 ? 4 : 2);
    result.*fields[i] = backend.view(fixture.buffers.*fields[i], lane * bytes, bytes);
  }
  return result;
}
FlashGDNState lazyLaneState(MetalBackend &backend, const Fixture &fixture, uint32_t lane) {
  return {backend.view(fixture.state.convolution, lane * fixture.convStride, fixture.convStride),
          backend.view(fixture.state.recurrent, lane * fixture.recurrentStride, fixture.recurrentStride),
          fixture.convStride, fixture.recurrentStride};
}
void lazyContinuation(MetalBackend &backend, Fixture &candidate, Fixture &expected,
                       const Reference &seed, std::span<const uint32_t> kept,
                       uint32_t sequence, uint32_t continuationRows) {
  Reference future(continuationRows, seed.lanes, true);
  future.convolution = seed.convolution; future.aLog = seed.aLog;
  future.timeBias = seed.timeBias; future.norm = seed.norm;
  lazyChangedInputs(future, sequence + 20 + continuationRows);
  Fixture actual(backend, future), control(backend, future);
  actual.state = candidate.state; control.state = expected.state;
  for (auto *fixture : {&actual, &control})
    for (const auto &out : {fixture->buffers.mixed, fixture->buffers.decay, fixture->buffers.beta,
                            fixture->buffers.recurrentRows, fixture->buffers.output})
      std::memset(out.contents(), lazyPoison, out.sizeBytes());
  CommandGraph a, c;
  for (uint32_t lane = 0; lane < seed.lanes; ++lane) if (kept[lane]) {
    addGDNFused(a, actual.weights(), lazyLaneViews(backend, actual, continuationRows, lane),
        lazyLaneState(backend, actual, lane), continuationRows, 1, FlashGDNFusion::PersistentHead512);
    addGDNFused(c, control.weights(), lazyLaneViews(backend, control, continuationRows, lane),
        lazyLaneState(backend, control, lane), continuationRows, 1, FlashGDNFusion::PersistentHead512);
  }
  if (!c.empty()) {
    lazyValidTiming(backend.submitCommand(c.dispatches()));
    lazyValidTiming(backend.submitCommand(a.dispatches()));
  }
  lazyCompareVerify(actual, control);
  lazyInputsPreserved(actual, future); lazyInputsPreserved(control, future);
}

struct LazySavedCopy {
  std::array<MetalBuffer, 6> buffers;
  std::array<std::vector<uint8_t>, 6> bytes;
  explicit LazySavedCopy(const FlashGDNLazyRollback &lazy) {
    const auto saved = lazy.savedBuffers();
    buffers = {saved.mixed, saved.decay, saved.beta, saved.qkv,
                lazy.initialRecurrent(), lazy.initialConvolution()};
    for (size_t i = 0; i < buffers.size(); ++i) bytes[i] = lazyBytes(buffers[i]);
  }
  void unchanged() const {
    for (size_t i = 0; i < buffers.size(); ++i)
      lazyExactBytes(buffers[i].contents(), bytes[i].data(), bytes[i].size(),
                     "saved prepared arrays/initial snapshots through rollback");
  }
};

// Selection changes only while constructing a new diagnostic instance. Existing
// records must retain their selected route across later environment changes.
std::unique_ptr<FlashGDNLazyRollback> fusionRecord(MetalBackend &backend,
    uint32_t rows, uint32_t lanes, bool selected) {
  constexpr const char *key = "SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21";
  const char *old = std::getenv(key);
  const std::optional<std::string> previous = old ? std::optional<std::string>(old) : std::nullopt;
  require(setenv(key, selected ? "1" : "0", 1) == 0, "cannot select constructor route");
  try {
    auto record = std::make_unique<FlashGDNLazyRollback>(backend, rows, lanes);
    if (previous) require(setenv(key, previous->c_str(), 1) == 0, "cannot restore constructor env");
    else require(unsetenv(key) == 0, "cannot clear constructor env");
    require(record->copyFusionEnabled() == selected, "constructor did not capture fusion route");
    return record;
  } catch (...) {
    if (previous) (void)setenv(key, previous->c_str(), 1);
    else (void)unsetenv(key);
    throw;
  }
}

void fusionDestination(Fixture &fixture, FlashGDNLazyRollback &record,
    uint32_t rows, uint32_t lanes) {
  if (rows == 1) return;
  fixture.buffers.qkv = record.rawQKVDestination(rows, lanes);
  require(fixture.buffers.qkv.sizeBytes() == uint64_t(rows) * lanes * C * 2,
          "direct raw QKV destination is not the exact active compact extent");
}

void fusionArenasExact(const FlashGDNLazyRollback &candidate,
    const FlashGDNLazyRollback &baseline) {
  const auto a = candidate.arenaBuffers(), b = baseline.arenaBuffers();
  require(a.size() == b.size(), "fusion changed arena count");
  for (size_t i = 0; i < a.size(); ++i) {
    require(a[i].sizeBytes() == b[i].sizeBytes(), "fusion changed arena capacity/guards");
    lazyExactBytes(a[i].contents(), b[i].contents(), a[i].sizeBytes(),
                   "entire baseline/candidate arena including inactive capacity and guards");
  }
}

void fusionInitialExact(const FlashGDNLazyRollback &record, const Reference &seed) {
  if (seed.rows == 1) return;
  for (uint32_t lane = 0; lane < seed.lanes; ++lane) {
    lazyExactBytes(static_cast<const uint8_t *>(record.initialRecurrent().contents()) +
        lane * record.initialRecurrentLaneStrideBytes(),
        seed.recurrent.data() + uint64_t(lane) * StateElements,
        flashGDNRecurrentLaneBytes(), "initial FP32 snapshot");
    lazyExactBytes(static_cast<const uint8_t *>(record.initialConvolution().contents()) +
        lane * record.initialConvolutionLaneStrideBytes(),
        seed.history.data() + uint64_t(lane) * 3 * C,
        flashGDNConvolutionLaneBytes(), "pre-carry initial BF16 history snapshot");
  }
}

void fusionVerifyExact(const Fixture &candidate, const Fixture &baseline,
    const FlashGDNLazyRollback &a, const FlashGDNLazyRollback &b, uint32_t rows,
    bool valid) {
  const auto preparedA = rows > 1 ? a.savedBuffers() : candidate.buffers;
  const auto preparedB = rows > 1 ? b.savedBuffers() : baseline.buffers;
  for (const auto &[x, y] : std::array<std::pair<MetalBuffer, MetalBuffer>, 8>{{
      {preparedA.qkv, preparedB.qkv}, {preparedA.mixed, preparedB.mixed},
      {preparedA.decay, preparedB.decay}, {preparedA.beta, preparedB.beta},
      {candidate.buffers.recurrentRows, baseline.buffers.recurrentRows},
      {candidate.buffers.output, baseline.buffers.output},
      {candidate.buffers.diagnostics, baseline.buffers.diagnostics},
      {candidate.buffers.z, baseline.buffers.z}}}) {
    require(x.sizeBytes() == y.sizeBytes(), "fusion verify compared mismatched extents");
    lazyExactBytes(x.contents(), y.contents(), x.sizeBytes(),
                   "baseline/candidate raw/prepared/output/diagnostics");
  }
  lazyCompareStates(candidate, baseline); fusionArenasExact(a, b);
  require(a.canariesIntact() && b.canariesIntact(), "fusion damaged tape guards");
  if (valid) require(!*static_cast<const uint32_t *>(candidate.buffers.diagnostics.contents()),
                    "valid fusion fixture set diagnostics");
}

void fusionGraphShape(const CommandGraph &graph, uint32_t rows, uint32_t lanes,
    bool selected) {
  require(graph.dispatches().size() == (rows == 1 || selected ? 2 : lanes + 3),
          "unexpected verify graph extent");
  uint64_t copies = 0, fusedCarry = 0;
  for (const auto &dispatch : graph.dispatches()) {
    copies += dispatch.pipelineName == "flash_gdn_lazy_copy";
    fusedCarry += dispatch.pipelineName == "private_gdn_lazy_snapshot_convolution_carry_sep21";
  }
  require(copies == (rows == 1 || selected ? 0 : lanes + 1),
          "raw/history standalone copies were not exactly removed");
  require(fusedCarry == (rows > 1 && selected ? 1 : 0),
          "snapshot carry route was not selected only for candidate multirow");
}

uint64_t fusionHostNegatives(MetalBackend &backend) {
  Reference seed(4, 2, false); Fixture fixture(backend, seed);
  auto baseline = fusionRecord(backend, 16, 4, false);
  auto candidate = fusionRecord(backend, 16, 4, true);
  auto foreign = fusionRecord(backend, 16, 4, true);
  uint64_t checks = 0;
  {
    constexpr const char *key = "SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21";
    const char *raw = std::getenv(key);
    const std::optional<std::string> previous = raw ? std::optional<std::string>(raw) : std::nullopt;
    const auto before = backend.memoryStats().allocatedBytes;
    for (const char *invalid : {"", "2", "01", "true", "-1"}) {
      require(setenv(key, invalid, 1) == 0, "cannot set negative selector");
      bool rejected = false;
      try { FlashGDNLazyRollback invalidRecord(backend, 1, 1); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected && backend.memoryStats().allocatedBytes == before,
              "invalid constructor selector accepted or allocated"); ++checks;
    }
    if (previous) require(setenv(key, previous->c_str(), 1) == 0, "cannot restore selector after negatives");
    else require(unsetenv(key) == 0, "cannot clear selector after negatives");
  }
  auto rejectsGraph = [&](auto operation) {
    CommandGraph graph; graph.add("existing_host_only_dispatch", {}, {1, 1, 1});
    bool rejected = false;
    try { operation(graph); }
    catch (const std::invalid_argument &) { rejected = true; }
    catch (const std::logic_error &) { rejected = true; }
    catch (const splash::metal::MetalBackendError &) { rejected = true; }
    require(rejected && graph.dispatches().size() == 1 &&
        graph.dispatches()[0].pipelineName == "existing_host_only_dispatch",
        "rejected fusion operation mutated caller graph"); ++checks;
  };
  auto rejectsDestination = [&](auto operation) {
    bool rejected = false;
    try { operation(); }
    catch (const std::invalid_argument &) { rejected = true; }
    catch (const std::logic_error &) { rejected = true; }
    catch (const splash::metal::MetalBackendError &) { rejected = true; }
    require(rejected, "invalid raw destination request accepted"); ++checks;
  };
  for (uint32_t rows : {0u, 1u, 17u, UINT32_MAX})
    rejectsDestination([&] { (void)candidate->rawQKVDestination(rows, 2); });
  for (uint32_t lanes : {0u, 5u, UINT32_MAX})
    rejectsDestination([&] { (void)candidate->rawQKVDestination(4, lanes); });
  rejectsDestination([&] { (void)baseline->rawQKVDestination(4, 2); });
  require(!baseline->copyFusionEnabled() && candidate->copyFusionEnabled(),
          "later constructor changed an existing route"); ++checks;
  auto exact = candidate->rawQKVDestination(4, 2);
  require(exact.sameView(candidate->rawQKVDestination(4, 2)), "raw destination not stable"); ++checks;
  for (uint32_t rows : {0u, 17u})
    rejectsGraph([&](auto &g) { (void)candidate->begin(g, fixture.weights(), fixture.buffers, fixture.state, rows, 2); });
  for (uint32_t lanes : {0u, 5u})
    rejectsGraph([&](auto &g) { (void)candidate->begin(g, fixture.weights(), fixture.buffers, fixture.state, 4, lanes); });
  for (float epsilon : {0.f, -1.f, float(NAN), float(INFINITY)}) {
    rejectsGraph([&](auto &g) {
      auto b = fixture.buffers; b.qkv = exact;
      (void)candidate->begin(g, fixture.weights(), b, fixture.state, 4, 2, epsilon);
    });
  }
  rejectsGraph([&](auto &g) {
    (void)candidate->begin(g, fixture.weights(), fixture.buffers, fixture.state, 4, 2);
  }); // A separate ordinary QKV view is not the candidate contract.
  rejectsGraph([&](auto &g) {
    auto b = fixture.buffers; b.qkv = foreign->rawQKVDestination(4, 2);
    (void)candidate->begin(g, fixture.weights(), b, fixture.state, 4, 2);
  });
  for (const auto &wrong : std::array<MetalBuffer, 4>{
      backend.view(exact, 2, exact.sizeBytes() - 2),
      backend.view(exact, 0, exact.sizeBytes() - 2),
      candidate->rawQKVDestination(4, 1), candidate->rawQKVDestination(8, 2)})
    rejectsGraph([&](auto &g) {
      auto b = fixture.buffers; b.qkv = wrong;
      (void)candidate->begin(g, fixture.weights(), b, fixture.state, 4, 2);
    });
  rejectsGraph([&](auto &g) {
    auto b = fixture.buffers; b.qkv = exact; b.output = backend.view(exact, 0, b.output.sizeBytes());
    (void)candidate->begin(g, fixture.weights(), b, fixture.state, 4, 2);
  });
  rejectsGraph([&](auto &g) {
    auto b = fixture.buffers; b.qkv = exact;
    auto s = fixture.state; s.recurrent = {};
    (void)candidate->begin(g, fixture.weights(), b, s, 4, 2);
  });
  rejectsGraph([&](auto &g) {
    auto b = fixture.buffers; b.qkv = exact;
    auto s = fixture.state; s.convolutionLaneStrideBytes = 3;
    (void)candidate->begin(g, fixture.weights(), b, s, 4, 2);
  });
  for (const auto &tape : candidate->arenaBuffers()) {
    rejectsGraph([&](auto &g) {
      auto b = fixture.buffers; b.qkv = exact;
      // This deliberately overlaps another input/output/tape extent; rejection
      // must happen before appending any copy or verify work.
      b.z = backend.view(tape, 0, std::min(tape.sizeBytes(), b.z.sizeBytes()));
      (void)candidate->begin(g, fixture.weights(), b, fixture.state, 4, 2);
    });
  }
  auto direct = fixture.buffers; direct.qkv = exact;
  CommandGraph held;
  const auto ticket = candidate->begin(held, fixture.weights(), direct, fixture.state, 4, 2);
  fusionGraphShape(held, 4, 2, true); ++checks;
  rejectsDestination([&] { (void)candidate->rawQKVDestination(4, 2); });
  rejectsGraph([&](auto &g) { (void)candidate->begin(g, fixture.weights(), direct, fixture.state, 4, 2); });
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> k{4, 5}; candidate->commit(g, ticket, k); });
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 1> k{4}; candidate->commit(g, ticket, k); });
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> k{4, 4}; candidate->commit(g, ticket + 1, k); });
  candidate->abort(ticket); require(!candidate->pending(), "abort retained a pending trial"); ++checks;
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> k{4, 4}; candidate->commit(g, ticket, k); });
  CommandGraph replacement;
  const auto newer = candidate->begin(replacement, fixture.weights(), direct, fixture.state, 4, 2);
  require(newer != ticket, "replacement ticket reused generation"); ++checks;
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> k{4, 4}; candidate->commit(g, ticket, k); });
  candidate->abort(newer);
  CommandGraph a, b;
  const auto localTicket = candidate->begin(a, fixture.weights(), direct, fixture.state, 4, 2);
  auto other = fixture.buffers; other.qkv = foreign->rawQKVDestination(4, 2);
  const auto foreignTicket = foreign->begin(b, fixture.weights(), other, fixture.state, 4, 2);
  require(localTicket != foreignTicket, "separate owners reused ticket"); ++checks;
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> k{4, 4}; candidate->commit(g, foreignTicket, k); });
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> k{4, 4}; foreign->commit(g, localTicket, k); });
  candidate->abort(localTicket); foreign->abort(foreignTicket);
  auto one = fusionRecord(backend, 1, 2, true);
  require(one->allocationBytes() == 0 && one->canariesIntact(), "R1 candidate allocated tape"); ++checks;
  rejectsDestination([&] { (void)one->rawQKVDestination(2, 2); });
  require(candidate->canariesIntact() && baseline->canariesIntact() && foreign->canariesIntact(),
          "host-only negatives damaged guards"); ++checks;
  return checks;
}

uint64_t fusionCpuCarryChecks() {
  uint64_t checks = 0;
  // Bit patterns include signed zero, subnormals, infinities and NaN payloads.
  // Carry is a copy contract, so every uint16_t pattern must be preserved.
  constexpr std::array<uint16_t, 8> special{0x0000, 0x8000, 0x0001, 0x8001,
      0x7f80, 0xff80, 0x7fc1, 0xffff};
  for (uint32_t rows = 1; rows <= 16; ++rows) for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const uint64_t padded = uint64_t(3) * C + 64;
    std::vector<uint16_t> original(lanes * padded, 0xa5a5), candidate = original;
    std::vector<uint16_t> raw(uint64_t(rows) * lanes * C), snapshot(uint64_t(3) * lanes * C);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      for (uint32_t r = 0; r < 3; ++r) for (uint32_t c = 0; c < C; ++c)
        original[lane * padded + r * C + c] = uint16_t((c + r * 871 + lane * 293) ^ special[c % 8]);
      for (uint32_t r = 0; r < rows; ++r) for (uint32_t c = 0; c < C; ++c)
        raw[(uint64_t(lane) * rows + r) * C + c] = uint16_t((c + r * 137 + lane * 941) ^ special[(c + r) % 8]);
    }
    candidate = original;
    for (uint32_t lane = 0; lane < lanes; ++lane) for (uint32_t channel = 0; channel < C; ++channel) {
      std::array<uint16_t, 3> old{};
      for (uint32_t r = 0; r < 3; ++r) {
        old[r] = candidate[lane * padded + r * C + channel];
        snapshot[(uint64_t(lane) * 3 + r) * C + channel] = old[r];
      }
      for (uint32_t r = 0; r < 3; ++r) {
        const uint32_t source = rows + r;
        candidate[lane * padded + r * C + channel] = source < 3 ? old[source]
            : raw[(uint64_t(lane) * rows + source - 3) * C + channel];
      }
    }
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(std::memcmp(snapshot.data() + uint64_t(lane) * 3 * C,
          original.data() + lane * padded, flashGDNConvolutionLaneBytes()) == 0,
          "CPU fused snapshot lost pre-carry bytes"); ++checks;
      for (uint32_t out = 0; out < 3; ++out) {
        const uint32_t source = rows + out;
        const auto *wanted = source < 3 ? original.data() + lane * padded + source * C
            : raw.data() + (uint64_t(lane) * rows + source - 3) * C;
        require(std::memcmp(candidate.data() + lane * padded + out * C, wanted, C * 2) == 0,
                "CPU snapshot carry differs from independent concatenation"); ++checks;
      }
      for (uint32_t c = 3 * C; c < padded; ++c) {
        require(candidate[lane * padded + c] == 0xa5a5, "CPU carry changed padded stride"); ++checks;
      }
    }
    for (uint32_t kept = 1; kept <= rows; ++kept) {
      const auto prefix = *flashMTPConvolutionPrefix(kept);
      std::array<uint32_t, 3> got{}, wanted{};
      for (uint32_t i = 0; i < prefix.oldRows; ++i) got[i] = prefix.oldBegin + i;
      for (uint32_t i = 0; i < prefix.inputRows; ++i)
        got[prefix.destinationInputBegin + i] = 3 + prefix.inputBegin + i;
      for (uint32_t i = 0; i < 3; ++i) wanted[i] = kept + i;
      require(got == wanted, "CPU retained history differs from concatenation"); ++checks;
    }
  }
  return checks;
}

void fusionCpuOnly() {
  cpuOnly(); uint64_t checks = fusionCpuCarryChecks();
  for (const auto &[rows, lanes] : std::array<std::pair<uint32_t, uint32_t>, 6>{{
      {0, 1}, {17, 1}, {1, 0}, {1, 5}, {UINT32_MAX, 1}, {1, UINT32_MAX}}}) {
    bool rejected = false;
    try { (void)FlashGDNLazyRollback::plannedBytes(rows, lanes); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid CPU planner geometry accepted"); ++checks;
  }
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    require(FlashGDNLazyRollback::plannedBytes(1, lanes) == 0, "R1 CPU planner allocated tape"); ++checks;
    for (uint32_t rows = 2; rows <= 16; ++rows) {
      const uint64_t payload = uint64_t(lanes) * (flashGDNRecurrentLaneBytes() +
          flashGDNConvolutionLaneBytes() + uint64_t(rows) * (uint64_t(C) * 4 + uint64_t(H) * 6));
      const auto planned = FlashGDNLazyRollback::plannedBytes(rows, lanes);
      require(planned >= payload && planned <= payload + 6 * 16384,
              "six-plane CPU planner omitted payload/guards or overcounted"); ++checks;
    }
  }
  for (double invalid : {0., -1., 1e-10, 1e-314, double(NAN), double(INFINITY), 601.}) {
    splash::metal::CommandTiming timing; timing.gpuSeconds = invalid; timing.wallSeconds = .01;
    bool rejected = false;
    try { lazyValidTiming(timing); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "corrupted timing accepted"); ++checks;
  }
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
      << ",\"command_timing_size_bytes\":200,\"metal_backend_constructions\":0,\"gpu_commands\":0"
         ",\"scope\":\"independent snapshot-carry bit preservation and frozen CPU planner/reference; GPU qualification pending\"}\n";
}

struct FusionRunCounters {
  uint64_t retained = 0, continuations = 0, aborts = 0, verifyGraphs = 0,
      baselineCopies = 0, candidateCopies = 0, partialCommitGraphs = 0;
};

void fusionSubmit(MetalBackend &backend, const CommandGraph &graph) {
  require(!graph.empty(), "cannot submit empty qualification graph");
  lazyValidTiming(backend.submitCommand(graph.dispatches()));
}

std::string fusionCase(MetalBackend &backend, uint32_t rows, uint32_t lanes,
    uint32_t cold, uint32_t extremes) {
  Reference seed(rows, lanes, cold);
  if (extremes) {
    for (uint32_t h = 0; h < H; ++h) seed.aLog[h] = bits(std::array<float, 4>{-8, -4, 4, 8}[h % 4]);
    for (uint32_t d = 0; d < K; ++d) seed.norm[d] = bits(std::array<float, 4>{0, -1, .5f, 2}[d % 4]);
  }
  Fixture actual(backend, seed), baseline(backend, seed), eagerFull(backend, seed),
      expected(backend, seed), expectedBase(backend, seed);
  auto base = fusionRecord(backend, 16, 4, false), candidate = fusionRecord(backend, 16, 4, true);
  require(base->allocationBytes() == candidate->allocationBytes(), "fusion changed admitted tape bytes");
  fusionDestination(actual, *candidate, rows, lanes);
  LazyEager eager(backend, rows, lanes); FusionRunCounters counts;
  for (uint32_t sequence = 0; sequence < 3; ++sequence) {
    lazyChangedInputs(seed, sequence); lazyLoad(actual, seed); lazyLoad(baseline, seed); lazyLoad(eagerFull, seed);
    lazyReset(eagerFull, seed); CommandGraph captured; eager.encode(captured, eagerFull);
    fusionSubmit(backend, captured); eager.guards();
    auto trial = [&](std::span<const uint32_t> kept, bool abort) {
      lazyReset(actual, seed); lazyReset(baseline, seed); lazyReset(expected, seed); lazyReset(expectedBase, seed);
      CommandGraph a, b;
      const auto baseTicket = base->begin(b, baseline.weights(), baseline.buffers, baseline.state, rows, lanes);
      const auto candidateTicket = candidate->begin(a, actual.weights(), actual.buffers, actual.state, rows, lanes);
      fusionGraphShape(a, rows, lanes, true); fusionGraphShape(b, rows, lanes, false);
      if ((sequence + counts.verifyGraphs) % 2) { fusionSubmit(backend, a); fusionSubmit(backend, b); }
      else { fusionSubmit(backend, b); fusionSubmit(backend, a); }
      counts.verifyGraphs += 2; counts.baselineCopies += rows > 1 ? lanes + 1 : 0;
      fusionVerifyExact(actual, baseline, *candidate, *base, rows, true);
      lazyCompareVerify(actual, eagerFull, rows > 1 ? candidate.get() : nullptr);
      lazyCompareVerify(baseline, eagerFull, rows > 1 ? base.get() : nullptr);
      fusionInitialExact(*candidate, seed); fusionInitialExact(*base, seed);
      std::optional<LazySavedCopy> savedA, savedB;
      if (rows > 1) { savedA.emplace(*candidate); savedB.emplace(*base); }
      lazyReferencePrefix(expected, eagerFull, eager, seed, kept);
      lazyReferencePrefix(expectedBase, eagerFull, eager, seed, kept);
      if (abort) {
        const auto priorA = lazyBytes(actual.state.recurrent), priorB = lazyBytes(actual.state.convolution);
        candidate->abort(candidateTicket); base->abort(baseTicket);
        require(!candidate->pending() && !base->pending(), "completed abort left pending ownership");
        lazyExactBytes(actual.state.recurrent.contents(), priorA.data(), priorA.size(), "abort restored/promoted recurrent state");
        lazyExactBytes(actual.state.convolution.contents(), priorB.data(), priorB.size(), "abort restored/promoted history");
        lazyCompareStates(actual, baseline); fusionArenasExact(*candidate, *base);
        if (savedA) { savedA->unchanged(); savedB->unchanged(); }
        ++counts.aborts; return;
      }
      CommandGraph commitA, commitB;
      candidate->commit(commitA, candidateTicket, kept); base->commit(commitB, baseTicket, kept);
      const bool partial = std::any_of(kept.begin(), kept.end(), [&](uint32_t k) { return k && k != rows; });
      require(commitA.dispatches().size() == (partial ? 2 : 0) &&
          commitB.dispatches().size() == commitA.dispatches().size(),
          "commit work changed full/terminal/partial fastpath");
      if (partial) {
        fusionSubmit(backend, commitB); fusionSubmit(backend, commitA); ++counts.partialCommitGraphs;
      }
      lazyCompareStates(actual, baseline); lazyCompareStates(actual, expected);
      fusionArenasExact(*candidate, *base);
      if (savedA) { savedA->unchanged(); savedB->unchanged(); }
      lazyInputsPreserved(actual, seed); lazyInputsPreserved(baseline, seed);
      for (uint32_t continuationRows : {1u, 2u, 4u}) {
        lazyContinuation(backend, actual, expected, seed, kept, sequence, continuationRows);
        lazyContinuation(backend, baseline, expectedBase, seed, kept, sequence, continuationRows);
        lazyCompareStates(actual, baseline); lazyCompareStates(expected, expectedBase);
        ++counts.continuations;
      }
      if (savedA) { savedA->unchanged(); savedB->unchanged(); }
      require(candidate->canariesIntact() && base->canariesIntact(), "continuation damaged saved tape guards");
      ++counts.retained;
    };
    std::array<uint32_t, 4> kept{};
    for (uint32_t retained = 0; retained <= rows; ++retained) {
      std::fill_n(kept.begin(), lanes, retained); trial(std::span(kept.data(), lanes), false);
    }
    if (lanes > 1) for (uint32_t pattern = 0; pattern < 2; ++pattern) {
      for (uint32_t lane = 0; lane < lanes; ++lane)
        kept[lane] = (lane + pattern) % 3 == 0 ? 0 : (lane + pattern) % 3 == 1 ? 1 : rows;
      trial(std::span(kept.data(), lanes), false);
    }
    std::fill_n(kept.begin(), lanes, rows); trial(std::span(kept.data(), lanes), true);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      std::memcpy(seed.history.data() + uint64_t(lane) * 3 * C,
          static_cast<const uint8_t *>(eagerFull.state.convolution.contents()) + lane * eagerFull.convStride,
          flashGDNConvolutionLaneBytes());
      std::memcpy(seed.recurrent.data() + uint64_t(lane) * StateElements,
          static_cast<const uint8_t *>(eagerFull.state.recurrent.contents()) + lane * eagerFull.recurrentStride,
          flashGDNRecurrentLaneBytes());
    }
  }
  const auto ca = candidate->counters(), cb = base->counters();
  require(ca.layer_trial_graphs_built == cb.layer_trial_graphs_built &&
      ca.commit_calls == cb.commit_calls && ca.aborted_trial_calls == cb.aborted_trial_calls &&
      ca.partial_replay_graphs_built == cb.partial_replay_graphs_built,
      "fusion changed semantic lifecycle counters");
  std::ostringstream out; out << "{\"rows\":" << rows << ",\"lanes\":" << lanes
      << ",\"cold\":" << cold << ",\"extremes\":" << extremes
      << ",\"changed_carried_sequences\":3,\"retained_prefix_cases\":" << counts.retained
      << ",\"ordinary_future_continuation_calls\":" << counts.continuations
      << ",\"completed_abort_cases\":" << counts.aborts
      << ",\"verify_graphs\":" << counts.verifyGraphs
      << ",\"baseline_standalone_copy_dispatches\":" << counts.baselineCopies
      << ",\"candidate_standalone_copy_dispatches\":" << counts.candidateCopies
      << ",\"partial_commit_graph_pairs\":" << counts.partialCommitGraphs
      << ",\"allocation_bytes_each\":" << candidate->allocationBytes()
      << ",\"exact_raw_prepared_output_history_f32state_padding\":true"
         ",\"all_arena_capacity_and_guards_exact\":true"
         ",\"independent_captured_prefix_and_future_continuation_exact\":true"
         ",\"saved_operands_initial_snapshots_immutable\":true} ";
  return out.str();
}

std::string fusionNumericCase(MetalBackend &backend, uint32_t mutation, uint16_t pattern) {
  Reference seed(4, 2, false);
  switch (mutation) {
  case 0: seed.qkv[0] = pattern; break;
  case 1: seed.history[0] = pattern; break;
  case 2: seed.z[0] = pattern; break;
  case 3: seed.a[0] = pattern; break;
  case 4: seed.b[0] = pattern; break;
  case 5: seed.convolution[0] = pattern; break;
  case 6: seed.norm[0] = pattern; break;
  case 7: seed.recurrent[0] = number(pattern); break;
  default: throw std::invalid_argument("unknown numeric mutation");
  }
  Fixture actual(backend, seed), baseline(backend, seed);
  auto a = fusionRecord(backend, 16, 4, true), b = fusionRecord(backend, 16, 4, false);
  fusionDestination(actual, *a, 4, 2); lazyLoad(actual, seed);
  lazyReset(actual, seed); lazyReset(baseline, seed);
  CommandGraph ga, gb;
  const auto ta = a->begin(ga, actual.weights(), actual.buffers, actual.state, 4, 2);
  const auto tb = b->begin(gb, baseline.weights(), baseline.buffers, baseline.state, 4, 2);
  fusionSubmit(backend, gb); fusionSubmit(backend, ga);
  fusionVerifyExact(actual, baseline, *a, *b, 4, false);
  fusionInitialExact(*a, seed); fusionInitialExact(*b, seed);
  const uint32_t diagnostics = *static_cast<const uint32_t *>(actual.buffers.diagnostics.contents());
  // Original sigmoid/softplus behavior can map some infinite inputs to finite
  // outputs. Preserve that diagnostic behavior rather than inventing a broader
  // candidate rejection rule. NaN/Inf that propagates is still flagged by the
  // unchanged native numeric check and compared byte-for-byte above.
  LazySavedCopy savedA(*a), savedB(*b);
  const std::array<uint32_t, 2> kept{2, 4}; CommandGraph ca, cb;
  a->commit(ca, ta, kept); b->commit(cb, tb, kept);
  fusionSubmit(backend, cb); fusionSubmit(backend, ca);
  lazyCompareStates(actual, baseline); fusionArenasExact(*a, *b);
  lazyExactBytes(actual.buffers.diagnostics.contents(), baseline.buffers.diagnostics.contents(), 4,
                 "nonfinite partial replay diagnostics");
  savedA.unchanged(); savedB.unchanged(); lazyInputsPreserved(actual, seed); lazyInputsPreserved(baseline, seed);
  std::ostringstream out; out << "{\"mutation\":" << mutation << ",\"bf16_pattern\":" << pattern
      << ",\"diagnostics\":" << diagnostics
      << ",\"nonfinite_input\":" << (((pattern & 0x7f80) == 0x7f80) ? "true" : "false")
      << ",\"native_nonfinite_diagnostic\":" << ((diagnostics & FlashGDNNonFinite) ? "true" : "false")
      << ",\"verify_partial_replay_initial_raw_output_state_guards_byte_exact\":true} ";
  return out.str();
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    std::string reportPath, stage = "arguments", executableHash, libraryHash;
    bool reportAllowed = false;
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-only") { fusionCpuOnly(); return 0; }
      if (argc != 3) throw std::invalid_argument("usage: lazy-copy-fusion-oracle METALLIB FRESH_REPORT_JSON | --cpu-only");
      reportPath = argv[2];
      { std::ifstream previous(reportPath); require(!previous.good(), "report already exists"); }
      reportAllowed = true;
      const auto rowsList = lazyCsv("GDN_LAZY_FUSION_ROWS", "1,2,3,4,8,16");
      const auto lanesList = lazyCsv("GDN_LAZY_FUSION_LANES", "1,2,3,4");
      const auto coldList = lazyCsv("GDN_LAZY_FUSION_COLD", "0,1");
      const auto extremeList = lazyCsv("GDN_LAZY_FUSION_EXTREMES", "0,1");
      const auto numericList = lazyCsv("GDN_LAZY_FUSION_NUMERIC", "1");
      require(numericList.size() == 1 && numericList[0] <= 1, "numeric flag must be 0 or 1");
      for (uint32_t rows : rowsList) require(rows && rows <= 16, "invalid pre-Metal row list");
      for (uint32_t lanes : lanesList) require(lanes && lanes <= 4, "invalid pre-Metal lane list");
      for (uint32_t cold : coldList) require(cold <= 1, "invalid pre-Metal cold list");
      for (uint32_t extremes : extremeList) require(extremes <= 1, "invalid pre-Metal extreme list");
      // Reject selector typos before any Metal construction, including inherited
      // stale flags. The constructor itself is also tested for immutable choice.
      if (const char *flag = std::getenv("SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21"))
        require(std::string(flag) == "0" || std::string(flag) == "1", "invalid inherited fusion flag");
      executableHash = lazyHash(argv[0]); stage = "Metal construction";
      MetalBackend backend(argv[1]); libraryHash = lazyHex(backend.metallibSha256());
      std::ostringstream provenance; provenance << "{\"executable_file_sha256\":" << splash::json::quote(executableHash)
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(libraryHash)
          << ",\"command_timing_size_bytes\":200,\"build_provenance\":" << kLazyCopyFusionOracleBuildProvenance << '}';
      std::ofstream early(reportPath + ".provenance.json"); require(bool(early), "cannot write early provenance");
      early << provenance.str() << '\n'; early.close();
      stage = "host metadata ownership negatives"; const auto hostChecks = fusionHostNegatives(backend);
      std::vector<std::string> cases, numericCases;
      for (uint32_t rows : rowsList) for (uint32_t lanes : lanesList)
        for (uint32_t cold : coldList) for (uint32_t extremes : extremeList) {
          require(rows && rows <= 16 && lanes && lanes <= 4 && cold <= 1 && extremes <= 1, "invalid case geometry");
          // R8/R16 singleton verifier is supported; native joint verifier rows
          // per lane remain1..4. Do not imply wider native batch admission.
          if (lanes > 1 && rows > 4) continue;
          stage = "strict case rows=" + std::to_string(rows) + " lanes=" + std::to_string(lanes) +
              " cold=" + std::to_string(cold) + " extremes=" + std::to_string(extremes);
          cases.push_back(fusionCase(backend, rows, lanes, cold, extremes));
          std::cerr << stage << " byte exact\n";
        }
      require(!cases.empty(), "no strict cases selected");
      if (numericList[0]) for (uint32_t mutation = 0; mutation < 8; ++mutation)
        for (uint16_t pattern : std::array<uint16_t, 8>{0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x7f80, 0xff80, 0x7fc1}) {
          stage = "numeric mutation=" + std::to_string(mutation) + " pattern=" + std::to_string(pattern);
          numericCases.push_back(fusionNumericCase(backend, mutation, pattern));
        }
      std::ofstream report(reportPath); require(bool(report), "cannot write report");
      report << "{\"pass\":true,\"executable_file_sha256\":" << splash::json::quote(executableHash)
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(libraryHash)
          << ",\"command_timing_size_bytes\":200,\"build_provenance\":" << kLazyCopyFusionOracleBuildProvenance
          << ",\"host_metadata_ownership_lifecycle_rejection_checks\":" << hostChecks
          << ",\"control\":\"frozen runtime FlashGDNLazyRollback constructor selection0 plus independent captured Persistent512 prefix\""
             ",\"candidate\":\"exact active own RawQKV destination plus pre-carry tight history snapshot fused into carry\""
             ",\"error_limit\":\"zero differing bytes; BF16 inputs intermediates outputs history; F32 recurrent snapshots state; all inactive arena capacity lane padding diagnostics guards\""
             ",\"timing_qualification\":false,\"gpu_executed\":true,\"model_payloads_loaded\":false"
             ",\"whole_model_logits_or_qsa_cache_qualified\":false"
             ",\"scope\":\"synthetic layer projected inputs immutable coefficients persistent GDN state and ordinary future continuation only\",\"cases\":[";
      for (size_t i = 0; i < cases.size(); ++i) { if (i) report << ','; report << cases[i]; }
      report << "],\"numeric_cases\":[";
      for (size_t i = 0; i < numericCases.size(); ++i) { if (i) report << ','; report << numericCases[i]; }
      report << "]}\n"; return 0;
    } catch (const std::exception &error) {
      std::cerr << "lazy-copy-fusion-oracle: " << stage << ": " << error.what() << '\n';
      if (reportAllowed) {
        std::ofstream failure(reportPath);
        if (failure) failure << "{\"pass\":false,\"stage\":" << splash::json::quote(stage)
            << ",\"error\":" << splash::json::quote(error.what())
            << ",\"executable_file_sha256\":" << splash::json::quote(executableHash)
            << ",\"loaded_metallib_sha256\":" << splash::json::quote(libraryHash)
            << ",\"whole_model_logits_or_qsa_cache_qualified\":false}\n";
      }
      return 1;
    }
  }
}
