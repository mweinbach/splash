// PRIVATE synthetic native joint-wrapper qualification. --cpu-only constructs
// no MetalBackend and submits no GPU command. Compile/link/seal separately.
#import <Metal/Metal.h>
#define main original_batch_gdn_fixture_main
#include "gdn_fixture.hpp"
#undef main
#include "flash/FlashBatchVerifyGDN.hpp"
#include "flash/FlashGDNLazyRollback.hpp"
#include "metal/abi/FlashForward.h"
#include "engine/Json.hpp"
#include "LazyCopyFusionBatchBuildProvenance.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>

namespace {
static_assert(sizeof(splash::metal::CommandTiming) == 200);
constexpr uint64_t BatchGuard = 128;
constexpr uint8_t BatchPoison = 0xa5;
constexpr uint64_t BatchConvStride = kFlashBatchVerifyGDNConvolutionRowStrideBytes;
constexpr uint64_t BatchRecStride = kFlashBatchVerifyGDNRecurrentRowStrideBytes;
static_assert(BatchConvStride == 65536 && BatchRecStride == 3145728);

void batchExact(const void *a, const void *b, uint64_t bytes, const char *label) {
  if (!std::memcmp(a, b, bytes)) return;
  const auto *x = static_cast<const uint8_t *>(a), *y = static_cast<const uint8_t *>(b);
  uint64_t offset = 0; while (offset < bytes && x[offset] == y[offset]) ++offset;
  throw std::runtime_error(std::string(label) + " differs at byte " + std::to_string(offset));
}
void batchGuard(const MetalBuffer &b, uint64_t begin, uint64_t end) {
  const auto *p = static_cast<const uint8_t *>(b.contents());
  for (uint64_t i = begin; i < end; ++i)
    require(p[i] == BatchPoison, "joint wrapper wrote guard or packed padding");
}
std::string batchHex(const std::array<uint8_t, 32> &digest) {
  std::ostringstream out;
  for (uint8_t byte : digest) out << std::hex << std::setfill('0') << std::setw(2) << uint32_t(byte);
  return out.str();
}
std::string batchHash(const char *path) {
  std::ifstream in(path, std::ios::binary); require(bool(in), "cannot open executable provenance");
  CC_SHA256_CTX c{}; CC_SHA256_Init(&c); std::array<char, 65536> block;
  while (in) { in.read(block.data(), block.size());
    if (in.gcount()) CC_SHA256_Update(&c, block.data(), CC_LONG(in.gcount())); }
  require(in.eof(), "executable provenance read failed");
  std::array<uint8_t, 32> digest{}; CC_SHA256_Final(digest.data(), &c); return batchHex(digest);
}
void batchSubmit(MetalBackend &backend, const CommandGraph &graph) {
  require(!graph.empty(), "empty batch qualification submission");
  const auto t = backend.submitCommand(graph.dispatches());
  for (double value : {t.gpuSeconds, t.wallSeconds})
    require(std::isfinite(value) && value > 1e-9 && value < 600, "invalid 200-byte command timing ABI");
}
std::unique_ptr<FlashGDNLazyRollback> batchRecord(MetalBackend &backend, bool fusion) {
  constexpr const char *key = "SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21";
  const char *old = std::getenv(key);
  const std::optional<std::string> prior = old ? std::optional<std::string>(old) : std::nullopt;
  require(!setenv(key, fusion ? "1" : "0", 1), "cannot select joint oracle record");
  try {
    auto r = std::make_unique<FlashGDNLazyRollback>(backend, 4, 4);
    if (prior) require(!setenv(key, prior->c_str(), 1), "cannot restore selector");
    else require(!unsetenv(key), "cannot clear selector");
    require(r->copyFusionEnabled() == fusion, "record did not retain constructor selection"); return r;
  } catch (...) { if (prior) (void)setenv(key, prior->c_str(), 1); else (void)unsetenv(key); throw; }
}
void batchChangedInputs(Reference &r, uint32_t sequence) {
  Random random; random.state ^= uint64_t(sequence + 1) * 0xa321a0b491ULL;
  for (auto &[values, scale] : std::array<std::pair<std::vector<uint16_t> *, float>, 4>{{
      {&r.qkv, .8f}, {&r.z, 2.f}, {&r.a, 1.5f}, {&r.b, 2.f}}})
    for (auto &x : *values) x = bits(random.value(scale));
  r.b[0] = bits(-6.84375f);
}

struct BatchFixture {
  Fixture work;
  std::vector<FlashGDNState> requests;
  uint32_t lanes;
  BatchFixture(MetalBackend &backend, const Reference &r) : work(backend, r), lanes(r.lanes) {
    work.convStride = BatchConvStride; work.recurrentStride = BatchRecStride;
    work.state = {buffer(backend, uint64_t(lanes) * BatchConvStride + BatchGuard, "guarded native packed convolution"),
        buffer(backend, uint64_t(lanes) * BatchRecStride + BatchGuard, "guarded native packed recurrence"),
        BatchConvStride, BatchRecStride};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      requests.push_back({buffer(backend, flashGDNConvolutionLaneBytes() + BatchGuard, "separate request convolution"),
          buffer(backend, flashGDNRecurrentLaneBytes() + BatchGuard, "separate request recurrence"),
          flashGDNConvolutionLaneBytes() + BatchGuard, flashGDNRecurrentLaneBytes() + BatchGuard});
    reset(r);
  }
  void reset(const Reference &r) {
    for (const auto &b : {work.state.convolution, work.state.recurrent})
      std::memset(b.contents(), BatchPoison, b.sizeBytes());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      auto &s = requests[lane];
      std::memset(s.convolution.contents(), BatchPoison, s.convolution.sizeBytes());
      std::memset(s.recurrent.contents(), BatchPoison, s.recurrent.sizeBytes());
      std::memcpy(s.convolution.contents(), r.history.data() + uint64_t(lane) * 3 * C, flashGDNConvolutionLaneBytes());
      std::memcpy(s.recurrent.contents(), r.recurrent.data() + uint64_t(lane) * StateElements, flashGDNRecurrentLaneBytes());
    }
    load(r); *static_cast<uint32_t *>(work.buffers.diagnostics.contents()) = 0;
  }
  void load(const Reference &r) {
    copy(work.buffers.qkv, r.qkv); copy(work.buffers.z, r.z); copy(work.buffers.a, r.a); copy(work.buffers.b, r.b);
  }
  void guards() const {
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      batchGuard(work.state.convolution, lane * BatchConvStride + flashGDNConvolutionLaneBytes(),
          (lane + 1) * BatchConvStride);
      batchGuard(requests[lane].convolution, flashGDNConvolutionLaneBytes(), requests[lane].convolution.sizeBytes());
      batchGuard(requests[lane].recurrent, flashGDNRecurrentLaneBytes(), requests[lane].recurrent.sizeBytes());
    }
    batchGuard(work.state.convolution, uint64_t(lanes) * BatchConvStride, work.state.convolution.sizeBytes());
    batchGuard(work.state.recurrent, uint64_t(lanes) * BatchRecStride, work.state.recurrent.sizeBytes());
  }
};
void batchStatesExact(const BatchFixture &a, const BatchFixture &b) {
  for (const auto &[x, y] : std::array<std::pair<MetalBuffer, MetalBuffer>, 2>{{
      {a.work.state.convolution, b.work.state.convolution}, {a.work.state.recurrent, b.work.state.recurrent}}})
    batchExact(x.contents(), y.contents(), x.sizeBytes(), "full packed state and padding");
  for (uint32_t lane = 0; lane < a.lanes; ++lane)
    for (const auto &[x, y] : std::array<std::pair<MetalBuffer, MetalBuffer>, 2>{{
        {a.requests[lane].convolution, b.requests[lane].convolution},
        {a.requests[lane].recurrent, b.requests[lane].recurrent}}})
      batchExact(x.contents(), y.contents(), x.sizeBytes(), "separate request state and guards");
  a.guards(); b.guards();
}
void batchArenasExact(const FlashGDNLazyRollback &a, const FlashGDNLazyRollback &b) {
  const auto x = a.arenaBuffers(), y = b.arenaBuffers(); require(x.size() == 6 && y.size() == 6, "not six retained arenas");
  for (size_t i = 0; i < x.size(); ++i) {
    require(x[i].sizeBytes() == y[i].sizeBytes(), "arena geometry changed");
    batchExact(x[i].contents(), y[i].contents(), x[i].sizeBytes(), "all six arenas including inactive bytes and guards");
  }
  require(a.canariesIntact() && b.canariesIntact(), "lazy arena guard damaged");
}
void batchInitialExact(const FlashGDNLazyRollback &r, const Reference &seed) {
  for (uint32_t lane = 0; lane < seed.lanes; ++lane) {
    batchExact(static_cast<const uint8_t *>(r.initialConvolution().contents()) + lane * flashGDNConvolutionLaneBytes(),
        seed.history.data() + uint64_t(lane) * 3 * C, flashGDNConvolutionLaneBytes(), "initial tight history");
    batchExact(static_cast<const uint8_t *>(r.initialRecurrent().contents()) + lane * flashGDNRecurrentLaneBytes(),
        seed.recurrent.data() + uint64_t(lane) * StateElements, flashGDNRecurrentLaneBytes(), "initial F32 state");
  }
}
struct BatchSaved {
  std::vector<MetalBuffer> buffers;
  std::vector<std::vector<uint8_t>> bytes;
  explicit BatchSaved(const FlashGDNLazyRollback &r) : buffers(r.arenaBuffers()) {
    for (const auto &b : buffers) { const auto *p = static_cast<const uint8_t *>(b.contents()); bytes.emplace_back(p, p + b.sizeBytes()); }
  }
  void unchanged() const { for (size_t i = 0; i < buffers.size(); ++i)
    batchExact(buffers[i].contents(), bytes[i].data(), bytes[i].size(), "arena bytes through replay and continuation"); }
};
void batchWorkExact(const FlashGDNBuffers &a, const FlashGDNBuffers &b) {
  for (const auto &[x, y] : std::array<std::pair<MetalBuffer, MetalBuffer>, 10>{{
      {a.qkv,b.qkv}, {a.z,b.z}, {a.a,b.a}, {a.b,b.b}, {a.mixed,b.mixed},
      {a.decay,b.decay}, {a.beta,b.beta}, {a.recurrentRows,b.recurrentRows}, {a.output,b.output}, {a.diagnostics,b.diagnostics}}}) {
    require(x.sizeBytes() == y.sizeBytes(), "work extent mismatch"); batchExact(x.contents(), y.contents(), x.sizeBytes(), "raw/prepared/output/diagnostics");
  }
}
void batchInputsPreserved(const Fixture &f, const Reference &seed) {
  for (const auto &[b, v] : std::array<std::pair<MetalBuffer, const std::vector<uint16_t> *>, 8>{{
      {f.buffers.qkv,&seed.qkv}, {f.buffers.z,&seed.z}, {f.buffers.a,&seed.a}, {f.buffers.b,&seed.b},
      {f.convolution.buffer,&seed.convolution}, {f.aLog.buffer,&seed.aLog}, {f.timeBias.buffer,&seed.timeBias}, {f.norm.buffer,&seed.norm}}})
    batchExact(b.contents(), v->data(), v->size() * 2, "immutable projected inputs or weights");
}
void batchShape(const CommandGraph &g, uint32_t lanes, bool fused, bool sentinel = false) {
  require(g.dispatches().size() == uint64_t(fused ? 4 * lanes + 2 : 5 * lanes + 3) + sentinel, "unexpected native wrapper graph size");
  uint32_t words = 0, copies = 0, carries = 0;
  for (const auto &d : g.dispatches()) {
    words += d.pipelineName == "flash_forward_copy_words"; copies += d.pipelineName == "flash_gdn_lazy_copy";
    carries += d.pipelineName == "private_gdn_lazy_snapshot_convolution_carry_sep21";
  }
  require(words == 4 * lanes && copies == (fused ? 0 : lanes + 1) && carries == uint32_t(fused), "pack/scatter or copy-elimination graph mismatch");
}
void batchCopy(CommandGraph &g, const MetalBuffer &input, const MetalBuffer &output, uint64_t bytes) {
  require(bytes % 4 == 0, "copy must preserve native word ABI"); const auto words = bytes / 4;
  g.add("flash_forward_copy_words", {input, output}, FlashForwardCopyParams{words}, {(words + 255) / 256, 1, 1});
}
void batchScatterPartial(CommandGraph &g, MetalBackend &backend, const BatchFixture &f, std::span<const uint32_t> kept, uint32_t rows) {
  for (uint32_t lane = 0; lane < f.lanes; ++lane) if (kept[lane] && kept[lane] != rows) {
    batchCopy(g, backend.view(f.work.state.convolution, lane * BatchConvStride, flashGDNConvolutionLaneBytes()),
        f.requests[lane].convolution, flashGDNConvolutionLaneBytes());
    batchCopy(g, backend.view(f.work.state.recurrent, lane * BatchRecStride, flashGDNRecurrentLaneBytes()),
        f.requests[lane].recurrent, flashGDNRecurrentLaneBytes());
  }
}

struct BatchEager {
  Fixture full;
  MetalBuffer tape;
  uint64_t rowStride, laneStride;
  BatchEager(MetalBackend &backend, const Reference &seed) : full(backend, seed),
      rowStride(flashGDNRecurrentLaneBytes() + BatchGuard), laneStride((seed.rows - 1) * rowStride + BatchGuard) {
    tape = buffer(backend, seed.lanes * laneStride, "independent guarded eager prefix states");
    std::memset(tape.contents(), BatchPoison, tape.sizeBytes());
    CommandGraph g; addGDNFusedCaptured(g, full.weights(), full.buffers, full.state,
        FlashGDNCapture{tape,rowStride,laneStride,seed.rows-1,512}, seed.rows, seed.lanes); batchSubmit(backend,g);
    for (uint32_t lane = 0; lane < seed.lanes; ++lane) {
      for (uint32_t row = 0; row < seed.rows - 1; ++row)
        batchGuard(tape, lane * laneStride + row * rowStride + flashGDNRecurrentLaneBytes(), lane * laneStride + (row + 1) * rowStride);
      batchGuard(tape, lane * laneStride + (seed.rows - 1) * rowStride, (lane + 1) * laneStride);
    }
  }
  void expected(BatchFixture &out, const Reference &seed, std::span<const uint32_t> kept) const {
    for (uint32_t lane = 0; lane < seed.lanes; ++lane) {
      const uint32_t count = kept[lane] ? kept[lane] : seed.rows;
      const void *state = count == seed.rows
          ? static_cast<const uint8_t *>(full.state.recurrent.contents()) + lane * full.recurrentStride
          : static_cast<const uint8_t *>(tape.contents()) + lane * laneStride + (count - 1) * rowStride;
      std::memcpy(static_cast<uint8_t *>(out.work.state.recurrent.contents()) + lane * BatchRecStride, state, flashGDNRecurrentLaneBytes());
      std::memcpy(out.requests[lane].recurrent.contents(), state, flashGDNRecurrentLaneBytes());
      for (uint32_t row = 0; row < 3; ++row) {
        const uint32_t concatenated = count + row;
        const auto *source = concatenated < 3 ? seed.history.data() + (uint64_t(lane) * 3 + concatenated) * C
            : seed.qkv.data() + (uint64_t(lane) * seed.rows + concatenated - 3) * C;
        std::memcpy(static_cast<uint8_t *>(out.work.state.convolution.contents()) + lane * BatchConvStride + row * C * 2, source, C * 2);
        std::memcpy(static_cast<uint8_t *>(out.requests[lane].convolution.contents()) + row * C * 2, source, C * 2);
      }
    }
  }
};
FlashGDNBuffers batchLaneBuffers(MetalBackend &backend, const Fixture &f, uint32_t rows, uint32_t lane) {
  auto b = f.buffers;
  const std::array<MetalBuffer FlashGDNBuffers::*,9> fields{&FlashGDNBuffers::qkv,&FlashGDNBuffers::z,
      &FlashGDNBuffers::a,&FlashGDNBuffers::b,&FlashGDNBuffers::mixed,&FlashGDNBuffers::decay,
      &FlashGDNBuffers::beta,&FlashGDNBuffers::recurrentRows,&FlashGDNBuffers::output};
  constexpr std::array<uint32_t,9> widths{C,V,H,H,C,H,H,V,V};
  for (size_t i = 0; i < fields.size(); ++i) { const uint64_t bytes = uint64_t(rows) * widths[i] * (i == 5 ? 4 : 2);
    b.*fields[i] = backend.view(f.buffers.*fields[i], lane * bytes, bytes); } return b;
}
void batchContinuation(MetalBackend &backend, BatchFixture &actual, BatchFixture &base, BatchFixture &expected,
    const Reference &seed, std::span<const uint32_t> kept, uint32_t sequence) {
  Reference future(1, seed.lanes, false); future.convolution = seed.convolution; future.aLog = seed.aLog;
  future.timeBias = seed.timeBias; future.norm = seed.norm; batchChangedInputs(future,sequence+40);
  Fixture a(backend,future), b(backend,future), c(backend,future);
  for (auto *f : {&a,&b,&c}) for (const auto &out : {f->buffers.mixed,f->buffers.decay,f->buffers.beta,f->buffers.recurrentRows,f->buffers.output})
    std::memset(out.contents(),BatchPoison,out.sizeBytes());
  CommandGraph ga,gb,gc;
  for (uint32_t lane = 0; lane < seed.lanes; ++lane) if (kept[lane]) {
    addGDNFused(ga,a.weights(),batchLaneBuffers(backend,a,1,lane),actual.requests[lane],1,1,FlashGDNFusion::PersistentHead512);
    addGDNFused(gb,b.weights(),batchLaneBuffers(backend,b,1,lane),base.requests[lane],1,1,FlashGDNFusion::PersistentHead512);
    addGDNFused(gc,c.weights(),batchLaneBuffers(backend,c,1,lane),expected.requests[lane],1,1,FlashGDNFusion::PersistentHead512);
  }
  if (!gc.empty()) { batchSubmit(backend,gc); batchSubmit(backend,gb); batchSubmit(backend,ga); }
  batchWorkExact(a.buffers,b.buffers); batchWorkExact(a.buffers,c.buffers);
  batchStatesExact(actual,base); batchStatesExact(actual,expected);
  batchInputsPreserved(a,future); batchInputsPreserved(b,future); batchInputsPreserved(c,future);
}

uint64_t batchHostNegatives(MetalBackend &backend) {
  Reference seed(4,2,false); BatchFixture f(backend,seed);
  auto own = batchRecord(backend,true), foreign = batchRecord(backend,true);
  const auto exact = own->rawQKVDestination(4,2); uint64_t checks = 0;
  auto rejects = [&](const FlashGDNBuffers &buffers, const FlashGDNWeights &weights,
                     const std::vector<FlashGDNState> &requests) {
    CommandGraph g; g.add("existing_host_only_dispatch",{}, {1,1,1}); uint64_t ticket = 0; bool failed = false;
    try { addBatchVerifyGDN(g,backend,weights,buffers,requests,f.work.state,{}, {},4,4,1e-6f,own.get(),&ticket); }
    catch (const std::invalid_argument &) { failed = true; }
    catch (const std::logic_error &) { failed = true; }
    catch (const splash::metal::MetalBackendError &) { failed = true; }
    require(failed && !ticket && !own->pending() && g.dispatches().size() == 1 &&
        g.dispatches()[0].pipelineName == "existing_host_only_dispatch", "wrapper rejection added pack work or altered sentinel/ticket"); ++checks;
  };
  auto good = f.work.buffers; good.qkv = exact;
  auto rawAllocation = own->arenaBuffers().at(2);
  for (const auto &wrong : std::array<MetalBuffer,6>{f.work.buffers.qkv,foreign->rawQKVDestination(4,2),
      backend.view(rawAllocation,4,exact.sizeBytes()),backend.view(exact,0,exact.sizeBytes()-4),
      own->rawQKVDestination(3,2),own->rawQKVDestination(4,3)}) {
    auto bad = good; bad.qkv = wrong; rejects(bad,f.work.weights(),f.requests);
  }
  { auto bad = good; bad.output = backend.view(exact,0,bad.output.sizeBytes()); rejects(bad,f.work.weights(),f.requests); }
  { auto bad = good; bad.mixed = exact; rejects(bad,f.work.weights(),f.requests); }
  { auto bad = good; bad.z = backend.view(exact,0,bad.z.sizeBytes()); rejects(bad,f.work.weights(),f.requests); }
  { auto requests = f.requests; requests[0].convolution = backend.view(exact,0,flashGDNConvolutionLaneBytes()); rejects(good,f.work.weights(),requests); }
  { auto weight = f.work.norm; weight.buffer = backend.view(exact,0,weight.logicalBytes);
    auto weights = f.work.weights(); weights.norm = &weight; rejects(good,weights,f.requests); }
  for (const auto &arena : own->arenaBuffers()) { auto bad = good;
    bad.z = backend.view(arena,0,std::min(arena.sizeBytes(),bad.z.sizeBytes())); rejects(bad,f.work.weights(),f.requests); }
  // The precise work0/own-RawQKV pair is accepted; every other alias above is
  // rejected before ANY pack dispatch. This host graph is never submitted.
  CommandGraph accepted; accepted.add("existing_host_only_dispatch",{}, {1,1,1}); uint64_t ticket = 0;
  addBatchVerifyGDN(accepted,backend,f.work.weights(),good,f.requests,f.work.state,{}, {},4,4,1e-6f,own.get(),&ticket);
  batchShape(accepted,2,true,true); require(ticket && own->pending(),"exact own RawQKV rejected");
  require(accepted.dispatches()[0].pipelineName == "existing_host_only_dispatch","accepted wrapper overwrote sentinel");
  own->abort(ticket); require(own->canariesIntact() && foreign->canariesIntact(),"host metadata checks altered guards"); return checks+1;
}

std::string batchCase(MetalBackend &backend, uint32_t rows, uint32_t lanes, bool cold) {
  Reference seed(rows,lanes,cold); BatchFixture actual(backend,seed),base(backend,seed),expected(backend,seed);
  auto a = batchRecord(backend,true), b = batchRecord(backend,false);
  actual.work.buffers.qkv = a->rawQKVDestination(rows,lanes);
  require(a->allocationBytes() == b->allocationBytes(),"joint fusion changed arena admission");
  uint64_t verifyGraphs = 0, partialCommits = 0, fullCommits = 0, continuationCases = 0;
  for (uint32_t sequence = 0; sequence < 2; ++sequence) {
    batchChangedInputs(seed,sequence); BatchEager eager(backend,seed);
    for (uint32_t cohort = 0; cohort < 3; ++cohort) {
      std::vector<uint32_t> kept(lanes,rows);
      if (cohort) for (uint32_t lane = 0; lane < lanes; ++lane) kept[lane] = (lane+sequence+cohort-1)%(rows+1);
      actual.reset(seed); base.reset(seed); expected.reset(seed);
      CommandGraph ga,gb; uint64_t ta = 0,tb = 0;
      addBatchVerifyGDN(ga,backend,actual.work.weights(),actual.work.buffers,actual.requests,actual.work.state,{}, {},rows,4,1e-6f,a.get(),&ta);
      addBatchVerifyGDN(gb,backend,base.work.weights(),base.work.buffers,base.requests,base.work.state,{}, {},rows,4,1e-6f,b.get(),&tb);
      batchShape(ga,lanes,true); batchShape(gb,lanes,false);
      if ((sequence+cohort)%2) { batchSubmit(backend,ga); batchSubmit(backend,gb); }
      else { batchSubmit(backend,gb); batchSubmit(backend,ga); }
      verifyGraphs += 2;
      batchWorkExact(a->savedBuffers(),b->savedBuffers()); batchWorkExact(a->savedBuffers(),eager.full.buffers);
      batchStatesExact(actual,base); batchArenasExact(*a,*b); batchInitialExact(*a,seed); batchInitialExact(*b,seed);
      require(!*static_cast<const uint32_t *>(actual.work.buffers.diagnostics.contents()),"valid native wrapper set diagnostics");
      std::vector<uint32_t> all(lanes,rows); eager.expected(expected,seed,all); batchStatesExact(actual,expected);
      BatchSaved savedA(*a),savedB(*b);
      CommandGraph ca,cb; a->commit(ca,ta,kept); b->commit(cb,tb,kept);
      bool partial = false; uint32_t partialLanes = 0;
      for (uint32_t n : kept) if (n && n != rows) { partial = true; ++partialLanes; }
      require(ca.dispatches().size() == (partial?2u:0u) && cb.dispatches().size() == (partial?2u:0u),"full/terminal no-replay commit violated");
      batchScatterPartial(ca,backend,actual,kept,rows); batchScatterPartial(cb,backend,base,kept,rows);
      require(ca.dispatches().size() == (partial?2u+2*partialLanes:0u),"partial native scatter extent wrong");
      if (partial) { batchSubmit(backend,cb); batchSubmit(backend,ca); ++partialCommits; } else ++fullCommits;
      eager.expected(expected,seed,kept); batchStatesExact(actual,base); batchStatesExact(actual,expected);
      batchArenasExact(*a,*b); savedA.unchanged(); savedB.unchanged();
      batchInputsPreserved(actual.work,seed); batchInputsPreserved(base.work,seed);
      batchContinuation(backend,actual,base,expected,seed,kept,sequence+cohort); ++continuationCases;
      savedA.unchanged(); savedB.unchanged(); batchArenasExact(*a,*b);
      require(!a->pending() && !b->pending(),"wrapper trial retained pending ownership");
    }
  }
  std::ostringstream out; out << "{\"rows\":" << rows << ",\"lanes\":" << lanes << ",\"cold\":" << (cold?"true":"false")
      << ",\"verify_graphs\":" << verifyGraphs << ",\"partial_commit_scatter_cases\":" << partialCommits
      << ",\"no_replay_commit_cases\":" << fullCommits << ",\"ordinary_future_continuation_cases\":" << continuationCases
      << ",\"raw_prepared_initial_packed_request_states_all_six_arenas_guards_byte_exact\":true}"; return out.str();
}
void batchCpuOnly() {
  cpuOnly(); uint64_t checks = 0;
  for (uint32_t rows : {2u,3u,4u}) for (uint32_t lanes : {2u,3u,4u}) {
    require(uint64_t(rows)*lanes*C*2 == uint64_t(rows)*lanes*20480,"CPU native raw shape mismatch");
    require(BatchConvStride >= flashGDNConvolutionLaneBytes() && BatchRecStride == flashGDNRecurrentLaneBytes(),"CPU native packed strides mismatch");
    const auto planned = FlashGDNLazyRollback::plannedBytes(4,4);
    require(planned >= uint64_t(4)*(flashGDNRecurrentLaneBytes()+flashGDNConvolutionLaneBytes()+uint64_t(4)*(C*4+H*6)),"CPU retained planner undercounted");
    checks += 3;
    for (uint32_t kept = 0; kept <= rows; ++kept) {
      const auto count = kept?kept:rows;
      for (uint32_t row = 0; row < 3; ++row) require(count+row < 3+rows,"CPU retained concatenation out of bounds");
      ++checks;
    }
  }
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
      << ",\"command_timing_size_bytes\":200,\"metal_backend_constructions\":0,\"gpu_commands\":0"
         ",\"native_wrapper_gpu_qualified\":false,\"whole_target_logits_qualified\":false"
         ",\"scope\":\"CPU scalar fixture and native wrapper shape/planner contracts only; GPU wrapper gates pending\"}\n";
}
} // namespace

int main(int argc,char **argv) {
  @autoreleasepool {
    std::string reportPath,stage="arguments",executableHash,libraryHash; bool reportAllowed=false;
    try {
      if (argc==2 && std::string(argv[1])=="--cpu-only") { batchCpuOnly(); return 0; }
      require(argc==3,"usage: lazy-copy-fusion-batch-oracle FROZEN_METALLIB FRESH_REPORT_JSON | --cpu-only");
      reportPath=argv[2]; { std::ifstream prior(reportPath); require(!prior.good(),"report already exists"); } reportAllowed=true;
      if (const char *flag=std::getenv("SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21"))
        require(std::string(flag)=="0" || std::string(flag)=="1","invalid inherited constructor selector");
      executableHash=batchHash(argv[0]); stage="Metal construction"; MetalBackend backend(argv[1]); libraryHash=batchHex(backend.metallibSha256());
      { std::ofstream p(reportPath+".provenance.json"); require(bool(p),"cannot write early provenance");
        p << "{\"executable_file_sha256\":" << splash::json::quote(executableHash) << ",\"loaded_metallib_sha256\":"
          << splash::json::quote(libraryHash) << ",\"build_provenance\":" << kLazyCopyFusionBatchBuildProvenance << "}\n"; }
      stage="native wrapper host alias preflight"; const auto negatives=batchHostNegatives(backend); std::vector<std::string> cases;
      for (uint32_t rows : {2u,3u,4u}) for (uint32_t lanes : {2u,3u,4u}) for (bool cold : {false,true}) {
        stage="native wrapper rows="+std::to_string(rows)+" lanes="+std::to_string(lanes)+" cold="+std::to_string(cold);
        cases.push_back(batchCase(backend,rows,lanes,cold)); std::cerr << stage << " byte exact\n";
      }
      std::ofstream report(reportPath); require(bool(report),"cannot write batch report");
      report << "{\"pass\":true,\"gpu_executed\":true,\"model_payloads_loaded\":false,\"timing_qualification\":false"
        << ",\"executable_file_sha256\":" << splash::json::quote(executableHash) << ",\"loaded_metallib_sha256\":" << splash::json::quote(libraryHash)
        << ",\"command_timing_size_bytes\":200,\"build_provenance\":" << kLazyCopyFusionBatchBuildProvenance
        << ",\"host_wrapper_alias_preflight_checks\":" << negatives
        << ",\"whole_target_logits_qualified\":false,\"qsa_or_ple_qualified\":false"
           ",\"scope\":\"actual frozen addBatchVerifyGDN wrapper; native separate request pack/scatter; literal projected inputs; exact GDN prefix replay and ordinary continuation only\",\"cases\":[";
      for (size_t i=0;i<cases.size();++i) { if(i) report << ','; report << cases[i]; } report << "]}\n"; return 0;
    } catch (const std::exception &e) {
      std::cerr << "lazy-copy-fusion-batch-oracle: " << stage << ": " << e.what() << '\n';
      if (reportAllowed) { std::ofstream report(reportPath); if(report) report << "{\"pass\":false,\"stage\":" << splash::json::quote(stage)
          << ",\"error\":" << splash::json::quote(e.what()) << ",\"executable_file_sha256\":" << splash::json::quote(executableHash)
          << ",\"loaded_metallib_sha256\":" << splash::json::quote(libraryHash) << ",\"whole_target_logits_qualified\":false}\n"; } return 1;
    }
  }
}
