// PRIVATE qualification only. --cpu-only returns before Metal construction.
#import <Metal/Metal.h>
#define main original_flash_gdn_oracle_main
#include "../tests/engine/flash_gdn_metal_test.mm"
#undef main
#include "flash_gdn_lazy_rollback.hpp"
#include "flash_gdn_lazy_rollback.h"
#include "engine/Json.hpp"
#include "LazyGdnOracleBuildProvenance.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <numeric>
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
double lazyMedian(std::vector<double> values) {
  require(!values.empty(), "missing timing pairs");
  std::sort(values.begin(), values.end()); const auto n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) * .5;
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
                         const GdnLazyRollback *lazy = nullptr) {
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
  explicit LazySavedCopy(const GdnLazyRollback &lazy) {
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
void lazyInitialSnapshot(const GdnLazyRollback &lazy, const CommandGraph &begin,
                          const Fixture &candidate, const Reference &seed) {
  uint64_t snapshotStride = 0;
  for (const auto &dispatch : begin.dispatches())
    if (dispatch.pipelineName == "private_gdn_lazy_verify_sg16") {
      require(dispatch.bytes.size() == 1 &&
          dispatch.bytes[0].sizeBytes == sizeof(GdnLazyVerifyParams), "lazy verify ABI changed");
      GdnLazyVerifyParams params{};
      std::memcpy(&params, dispatch.bytes[0].data, sizeof(params));
      snapshotStride = params.snapshot_lane_stride_bytes;
    }
  require(snapshotStride >= flashGDNRecurrentLaneBytes() && snapshotStride % 4 == 0,
          "lazy verify initial snapshot stride unavailable/invalid");
  for (uint32_t lane = 0; lane < seed.lanes; ++lane) {
    lazyExactBytes(static_cast<const uint8_t *>(lazy.initialRecurrent().contents()) + lane * snapshotStride,
        seed.recurrent.data() + uint64_t(lane) * StateElements,
        flashGDNRecurrentLaneBytes(), "initial once-only FP32 snapshot");
    lazyExactBytes(static_cast<const uint8_t *>(lazy.initialConvolution().contents()) +
        lane * flashGDNConvolutionLaneBytes(), seed.history.data() + uint64_t(lane) * 3 * C,
        flashGDNConvolutionLaneBytes(), "initial convolution snapshot");
  }
  const auto saved = lazy.savedBuffers();
  for (const auto &[a, b] : std::array<std::pair<MetalBuffer, MetalBuffer>, 4>{{
      {saved.qkv, candidate.buffers.qkv}, {saved.mixed, candidate.buffers.mixed},
      {saved.decay, candidate.buffers.decay}, {saved.beta, candidate.buffers.beta}}}) {
    require(a.sizeBytes() >= b.sizeBytes(), "lazy prepared save too short");
    lazyExactBytes(a.contents(), b.contents(), b.sizeBytes(), "saved actual prepared operands");
  }
}

uint32_t lazyHostNegatives(MetalBackend &backend) {
  Reference seed(4, 2, false); Fixture fixture(backend, seed);
  GdnLazyRollback lazy(backend, 16, 4);
  uint32_t checks = 0;
  auto rejectsGraph = [&](auto operation) {
    CommandGraph graph; graph.add("existing_host_only_dispatch", {}, {1, 1, 1});
    bool rejected = false;
    try { operation(graph); }
    catch (const std::invalid_argument &) { rejected = true; }
    catch (const std::logic_error &) { rejected = true; }
    catch (const splash::metal::MetalBackendError &) { rejected = true; }
    require(rejected && graph.dispatches().size() == 1 &&
        graph.dispatches()[0].pipelineName == "existing_host_only_dispatch",
        "invalid lazy lifecycle/metadata modified caller graph"); ++checks;
  };
  for (uint32_t rows : {0u, 17u})
    rejectsGraph([&](auto &g) { (void)lazy.begin(g, fixture.weights(), fixture.buffers, fixture.state, rows, 2); });
  for (uint32_t lanes : {0u, 5u})
    rejectsGraph([&](auto &g) { (void)lazy.begin(g, fixture.weights(), fixture.buffers, fixture.state, 4, lanes); });
  rejectsGraph([&](auto &g) {
    auto state = fixture.state; state.recurrent = {};
    (void)lazy.begin(g, fixture.weights(), fixture.buffers, state, 4, 2);
  });
  rejectsGraph([&](auto &g) {
    auto b = fixture.buffers; b.output = b.qkv;
    (void)lazy.begin(g, fixture.weights(), b, fixture.state, 4, 2);
  });
  CommandGraph held;
  const auto ticket = lazy.begin(held, fixture.weights(), fixture.buffers, fixture.state, 4, 2);
  rejectsGraph([&](auto &g) { (void)lazy.begin(g, fixture.weights(), fixture.buffers, fixture.state, 4, 2); });
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> keep{4, 5}; lazy.commit(g, ticket, keep); });
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 1> keep{4}; lazy.commit(g, ticket, keep); });
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> keep{4, 4}; lazy.commit(g, ticket + 1, keep); });
  lazy.abort(ticket);
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> keep{4, 4}; lazy.commit(g, ticket, keep); });
  CommandGraph replacement;
  const auto newer = lazy.begin(replacement, fixture.weights(), fixture.buffers, fixture.state, 4, 2);
  require(newer != ticket, "replacement lazy ticket reused generation"); ++checks;
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> keep{4, 4}; lazy.commit(g, ticket, keep); });
  lazy.abort(newer);
  GdnLazyRollback foreign(backend, 4, 2);
  CommandGraph a, b;
  const auto localTicket = lazy.begin(a, fixture.weights(), fixture.buffers, fixture.state, 4, 2);
  const auto foreignTicket = foreign.begin(b, fixture.weights(), fixture.buffers, fixture.state, 4, 2);
  require(localTicket != foreignTicket, "different owners reused lazy ticket"); ++checks;
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> keep{4, 4}; lazy.commit(g, foreignTicket, keep); });
  rejectsGraph([&](auto &g) { const std::array<uint32_t, 2> keep{4, 4}; foreign.commit(g, localTicket, keep); });
  lazy.abort(localTicket); foreign.abort(foreignTicket);
  GdnLazyRollback one(backend, 1, 2);
  require(one.allocationBytes() == 0 && one.canariesIntact(),
          "R1-only lazy arena allocated a rollback tape"); ++checks;
  require(lazy.canariesIntact(), "host-only lazy validation damaged guarded arena"); ++checks;
  return checks;
}

void lazyCpuOnly() {
  cpuOnly(); uint64_t checks = 0;
  for (const auto &[rows, lanes] : std::array<std::pair<uint32_t, uint32_t>, 6>{{
      {0, 1}, {17, 1}, {1, 0}, {1, 5}, {UINT32_MAX, 1}, {1, UINT32_MAX}}}) {
    bool rejected = false;
    try { (void)GdnLazyRollback::plannedBytes(rows, lanes); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid CPU arena geometry accepted"); ++checks;
  }
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    require(GdnLazyRollback::plannedBytes(1, lanes) == 0,
            "R1-only CPU planner allocated snapshots"); ++checks;
    for (uint32_t rows = 2; rows <= 16; ++rows) {
      const uint64_t tight = flashGDNRecurrentLaneBytes();
      const uint64_t payload = uint64_t(lanes) * (tight + flashGDNConvolutionLaneBytes() +
          uint64_t(rows) * (uint64_t(C) * 4 + uint64_t(H) * 6));
      const auto planned = GdnLazyRollback::plannedBytes(rows, lanes);
      require(planned >= payload && planned <= payload + 6 * 16384,
              "lazy six-plane CPU planner omitted payload/guards or overcounted"); ++checks;
    }
  }
  for (uint32_t rows = 1; rows <= 16; ++rows)
    for (uint32_t kept = 1; kept <= rows; ++kept) {
      const auto prefix = *flashMTPConvolutionPrefix(kept);
      std::array<uint32_t, 3> got{}, wanted{};
      for (uint32_t i = 0; i < prefix.oldRows; ++i) got[i] = prefix.oldBegin + i;
      for (uint32_t i = 0; i < prefix.inputRows; ++i)
        got[prefix.destinationInputBegin + i] = 3 + prefix.inputBegin + i;
      for (uint32_t i = 0; i < 3; ++i) wanted[i] = kept + i;
      require(got == wanted, "retained convolution prefix differs from concatenation"); ++checks;
    }
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    std::array<uint8_t, 128 * 128> visits{};
    for (uint32_t sg = 0; sg < 16; ++sg)
      for (uint32_t value = 0; value < 8; ++value)
        for (uint32_t lane = 0; lane < 32; ++lane)
          for (uint32_t key = 0; key < 4; ++key)
            ++visits[(value * 16 + sg) * 128 + lane * 4 + key];
    for (auto count : visits) { require(count == 1, "SG16 recurrence ownership differs"); ++checks; }
    for (uint32_t rows : {1u, 4u, 8u, 16u}) {
      const uint64_t tight = flashGDNRecurrentLaneBytes();
      const uint64_t old = uint64_t(lanes) * (rows - 1) * tight;
      const uint64_t fresh = uint64_t(lanes) * (tight + flashGDNConvolutionLaneBytes() +
          uint64_t(rows) * (uint64_t(C) * 4 + uint64_t(H) * 6));
      require(fresh > 0 && (rows < 4 || fresh < old), "lazy snapshot/save memory bound differs"); ++checks;
      for (uint32_t slot = 0; slot < lanes; ++slot)
        for (uint32_t row = 0; row < rows; ++row) {
          require((uint64_t(slot) * rows + row) * C + C <= uint64_t(lanes) * rows * C,
                  "saved compact raw/prepared row escapes allocation"); ++checks;
        }
    }
  }
  for (double invalid : {0., -1., 1e-10, 1e-314, double(NAN), double(INFINITY), 601.}) {
    splash::metal::CommandTiming t; t.gpuSeconds = invalid; t.wallSeconds = .01;
    bool rejected = false; try { lazyValidTiming(t); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "timing corruption accepted"); ++checks;
  }
  std::cout << "{\"pass\":true,\"lazy_rollback_cpu_checks\":" << checks
      << ",\"command_timing_size_bytes\":200,\"metal_backend_constructions\":0,\"gpu_commands\":0}\n";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-only") { lazyCpuOnly(); return 0; }
      if (argc != 3) throw std::invalid_argument("usage: flash-gdn-lazy-rollback-oracle METALLIB REPORT_JSON | --cpu-only");
      const auto rowsList = lazyCsv("GDN_LAZY_ROWS", "1,4,8,16");
      const auto lanesList = lazyCsv("GDN_LAZY_LANES", "1,2,3,4");
      const auto coldList = lazyCsv("GDN_LAZY_COLD", "0,1");
      const auto extremeList = lazyCsv("GDN_LAZY_EXTREMES", "0,1");
      const auto pairList = lazyCsv("GDN_LAZY_PAIRS", "6");
      require(pairList.size() == 1 && pairList[0], "invalid timing pair count");
      const uint32_t pairs = pairList[0];
      const std::string executableHash = lazyHash(argv[0]);
      MetalBackend backend(argv[1]);
      const std::string libraryHash = lazyHex(backend.metallibSha256());
      std::ostringstream provenance;
      provenance << "{\"executable_file_sha256\":" << splash::json::quote(executableHash)
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(libraryHash)
          << ",\"command_timing_size_bytes\":200,\"build_provenance\":"
          << kLazyGdnOracleBuildProvenance << '}';
      std::cerr << provenance.str() << '\n';
      std::ofstream early(std::string(argv[2]) + ".provenance.json");
      require(bool(early), "cannot write pre-GPU provenance"); early << provenance.str() << '\n'; early.close();
      const auto hostChecks = lazyHostNegatives(backend);
      std::vector<std::string> cases;
      for (uint32_t rows : rowsList) for (uint32_t lanes : lanesList)
        for (uint32_t cold : coldList) for (uint32_t extremes : extremeList) {
          require(rows && rows <= 16 && lanes && lanes <= 4 && cold <= 1 && extremes <= 1,
                  "invalid lazy oracle geometry");
          if (lanes > 1 && rows > 4) continue; // Real joint windows remain1..4.
          Reference seed(rows, lanes, cold);
          if (extremes) {
            for (uint32_t h = 0; h < H; ++h) seed.aLog[h] = bits(std::array<float, 4>{-8, -4, 4, 8}[h % 4]);
            for (uint32_t d = 0; d < K; ++d) seed.norm[d] = bits(std::array<float, 4>{0, -1, .5f, 2}[d % 4]);
          }
          Fixture actual(backend, seed), eagerFull(backend, seed), expected(backend, seed);
          LazyEager eager(backend, rows, lanes);
          GdnLazyRollback lazy(backend, 16, 4);
          require(lazy.maximumRows() == 16 && lazy.maximumLanes() == 4, "lazy constructor capacity changed");
          uint32_t retainedCases = 0, continuationCalls = 0;
          uint64_t fullCommitDispatches = 0, verifyDispatches = 0, partialCommitDispatches = 0;
          for (uint32_t sequence = 0; sequence < 2; ++sequence) {
            // Sequence two carries the previous fully accepted verifier state.
            lazyChangedInputs(seed, sequence); lazyLoad(actual, seed); lazyLoad(eagerFull, seed);
            lazyReset(actual, seed); lazyReset(eagerFull, seed); lazyReset(expected, seed);
            CommandGraph captured; eager.encode(captured, eagerFull);
            lazyValidTiming(backend.submitCommand(captured.dispatches())); eager.guards();
            for (uint32_t retained = 1; retained <= rows; ++retained) {
              lazyReset(actual, seed);
              CommandGraph verify;
              const auto ticket = lazy.begin(verify, actual.weights(), actual.buffers, actual.state, rows, lanes);
              verifyDispatches = verify.dispatches().size();
              lazyValidTiming(backend.submitCommand(verify.dispatches()));
              lazyCompareVerify(actual, eagerFull, rows > 1 ? &lazy : nullptr);
              std::optional<LazySavedCopy> saved;
              if (rows > 1) { lazyInitialSnapshot(lazy, verify, eagerFull, seed); saved.emplace(lazy); }
              std::array<uint32_t, 4> kept{};
              for (uint32_t lane = 0; lane < lanes; ++lane) kept[lane] = retained;
              lazyReferencePrefix(expected, eagerFull, eager, seed, std::span(kept.data(), lanes));
              CommandGraph commit;
              lazy.commit(commit, ticket, std::span(kept.data(), lanes));
              if (retained == rows) {
                require(commit.empty(), "full acceptance appended rollback GPU work");
                fullCommitDispatches += commit.dispatches().size();
              } else {
                require(commit.dispatches().size() == 2, "partial rollback did not append exactly recurrence/history");
                partialCommitDispatches += commit.dispatches().size();
                lazyValidTiming(backend.submitCommand(commit.dispatches()));
              }
              lazyCompareStates(actual, expected);
              if (saved) saved->unchanged();
              require(lazy.canariesIntact(), "lazy verifier/rollback overwrote arena canaries");
              lazyInputsPreserved(actual, seed); lazyInputsPreserved(eagerFull, seed);
              for (uint32_t count : {1u, 2u}) {
                lazyContinuation(backend, actual, expected, seed, std::span(kept.data(), lanes), sequence, count);
                ++continuationCalls;
              }
              if (saved) saved->unchanged();
              ++retainedCases;
            }
            if (lanes > 1) {
              // Include both accepted and discarded lanes; keep0 is terminal
              // and must retain the full verifier state rather than restore0.
              lazyReset(actual, seed); CommandGraph verify;
              const auto ticket = lazy.begin(verify, actual.weights(), actual.buffers, actual.state, rows, lanes);
              lazyValidTiming(backend.submitCommand(verify.dispatches())); lazyCompareVerify(actual, eagerFull, rows > 1 ? &lazy : nullptr);
              std::optional<LazySavedCopy> saved; if (rows > 1) saved.emplace(lazy);
              std::array<uint32_t, 4> kept{};
              for (uint32_t lane = 0; lane < lanes; ++lane)
                kept[lane] = lane == 1 ? 0 : lane % 2 ? 1 : rows;
              lazyReferencePrefix(expected, eagerFull, eager, seed, std::span(kept.data(), lanes));
              CommandGraph commit; lazy.commit(commit, ticket, std::span(kept.data(), lanes));
              if (!commit.empty()) lazyValidTiming(backend.submitCommand(commit.dispatches()));
              lazyCompareStates(actual, expected); if (saved) saved->unchanged();
              for (uint32_t count : {1u, 2u}) {
                lazyContinuation(backend, actual, expected, seed, std::span(kept.data(), lanes), sequence, count);
                ++continuationCalls;
              }
              require(lazy.canariesIntact(), "joint terminal/mixed prefix overwrote guards");
              ++retainedCases;
            }
            for (uint32_t lane = 0; lane < lanes; ++lane) {
              std::memcpy(seed.history.data() + uint64_t(lane) * 3 * C,
                  static_cast<const uint8_t *>(eagerFull.state.convolution.contents()) + lane * eagerFull.convStride,
                  flashGDNConvolutionLaneBytes());
              std::memcpy(seed.recurrent.data() + uint64_t(lane) * StateElements,
                  static_cast<const uint8_t *>(eagerFull.state.recurrent.contents()) + lane * eagerFull.recurrentStride,
                  flashGDNRecurrentLaneBytes());
            }
          }
          std::vector<double> cGPU, nGPU, cWall, nWall;
          for (uint32_t pair = 0; pair < pairs + 2; ++pair) {
            lazyReset(actual, seed); lazyReset(eagerFull, seed);
            CommandGraph cGraph, nGraph; eager.encode(cGraph, eagerFull);
            const auto ticket = lazy.begin(nGraph, actual.weights(), actual.buffers, actual.state, rows, lanes);
            splash::metal::CommandTiming c, n;
            if (pair % 2) { n = backend.submitCommand(nGraph.dispatches()); c = backend.submitCommand(cGraph.dispatches()); }
            else { c = backend.submitCommand(cGraph.dispatches()); n = backend.submitCommand(nGraph.dispatches()); }
            lazyValidTiming(c); lazyValidTiming(n); lazyCompareVerify(actual, eagerFull, rows > 1 ? &lazy : nullptr);
            std::array<uint32_t, 4> full{}; std::fill_n(full.begin(), lanes, rows);
            CommandGraph commit; lazy.commit(commit, ticket, std::span(full.data(), lanes));
            require(commit.empty(), "timed full acceptance appended hidden rollback");
            if (pair >= 2) { cGPU.push_back(c.gpuSeconds); nGPU.push_back(n.gpuSeconds);
                cWall.push_back(c.wallSeconds); nWall.push_back(n.wallSeconds); }
          }
          require(lazy.canariesIntact(), "timed lazy verify overwrote guards"); eager.guards();
          const uint64_t eagerMaxTape = uint64_t(4) * 15 * flashGDNRecurrentLaneBytes();
          std::ostringstream row; row << std::setprecision(12)
              << "{\"rows\":" << rows << ",\"lanes\":" << lanes << ",\"cold\":" << cold
              << ",\"extremes\":" << extremes << ",\"changed_carried_sequences\":2"
              << ",\"retained_prefix_cases\":" << retainedCases << ",\"continuation_calls\":" << continuationCalls
              << ",\"verify_dispatches\":" << verifyDispatches << ",\"eager_verify_dispatches\":2"
              << ",\"full_commit_appended_dispatches\":" << fullCommitDispatches
              << ",\"partial_commit_appended_dispatches_total\":" << partialCommitDispatches
              << ",\"lazy_max16_b4_allocation_bytes\":" << lazy.allocationBytes()
              << ",\"old_max16_b4_tight_recurrent_tape_bytes\":" << eagerMaxTape
              << ",\"row1_bypasses_snapshots\":" << (rows == 1 ? "true" : "false")
              << ",\"pairs\":" << pairs << ",\"eager_gpu_seconds\":" << lazyMedian(cGPU)
              << ",\"lazy_gpu_seconds\":" << lazyMedian(nGPU)
              << ",\"verify_fullcommit_gpu_speedup\":" << lazyMedian(cGPU) / lazyMedian(nGPU)
              << ",\"eager_wall_seconds\":" << lazyMedian(cWall)
              << ",\"lazy_wall_seconds\":" << lazyMedian(nWall)
              << ",\"verify_fullcommit_wall_speedup\":" << lazyMedian(cWall) / lazyMedian(nWall)
              << ",\"full_verify_bf_intermediates_output_f32state_history_exact\":true"
                 ",\"every_retained_prefix_and_continuation_exact\":true"
                 ",\"prepared_and_initial_snapshots_immutable\":true"
                 ",\"zero_retained_terminal_lanes_untouched\":true"
                 ",\"input_weight_padding_guards_unchanged\":true}";
          cases.push_back(row.str());
          std::cerr << "lazy rows=" << rows << " lanes=" << lanes << " cold=" << cold
              << " extremes=" << extremes << " exact; speedup=" << lazyMedian(cGPU) / lazyMedian(nGPU) << '\n';
        }
      require(!cases.empty(), "no lazy cases selected");
      std::ofstream report(argv[2]); require(bool(report), "cannot write lazy report");
      report << "{\"pass\":true,\"executable_file_sha256\":" << splash::json::quote(executableHash)
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(libraryHash)
          << ",\"command_timing_size_bytes\":200,\"build_provenance\":" << kLazyGdnOracleBuildProvenance
          << ",\"host_only_rejection_lifecycle_checks\":" << hostChecks
          << ",\"control\":\"current captured SG16 Persistent512; captures rows-minus-one\""
             ",\"error_limit\":\"zero bytes for BF intermediates/output and F32 state/history/padding\""
             ",\"timing_scope\":\"full verify plus zero-dispatch full commit; excludes CPU resets/comparisons\""
             ",\"timing_validation\":\"finite;1e-9<gpu/wall<600seconds\""
             ",\"warmups\":2,\"alternating_matched_pairs\":true,\"cases\":[";
      for (size_t i = 0; i < cases.size(); ++i) { if (i) report << ','; report << cases[i]; }
      report << "]}\n"; return 0;
    } catch (const std::exception &error) {
      std::cerr << "flash-gdn-lazy-rollback-oracle: " << error.what() << '\n'; return 1;
    }
  }
}
