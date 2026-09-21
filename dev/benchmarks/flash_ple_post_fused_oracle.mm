// Isolated Root-run GPU oracle. --cpu-self-test does not create a Metal backend.
// Compare the qualified PLE post-projection graph and the fused graph without
// weakening any BF16, history, prefix-restore, or diagnostic boundary.
#include "flash/FlashPLEPostFused.hpp"
#include "engine/Json.hpp"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::flash;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::CommandTiming;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
constexpr uint16_t kCanary = 0x7fc1;
constexpr uint32_t kSticky = 0x40000000;
constexpr uint64_t kGuardWords = 32;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
uint16_t bf16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u)
    return uint16_t((bits >> 16) | ((bits & 0x7fffffu) ? 0x40u : 0u));
  return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
uint64_t randomWord(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
float seeded(uint64_t index, uint64_t seed, float scale) {
  return float(int32_t(randomWord(index + seed) % 2049) - 1024) * (scale / 1024.0f);
}

std::vector<uint32_t> list(const char *name, std::initializer_list<uint32_t> fallback,
                           uint32_t maximum) {
  const char *raw = std::getenv(name);
  if (!raw) return std::vector<uint32_t>(fallback);
  std::vector<uint32_t> result;
  std::stringstream stream(raw);
  std::string item;
  while (std::getline(stream, item, ',')) {
    require(!item.empty() && item.front() != '-', std::string("invalid ") + name);
    size_t used = 0;
    const auto value = std::stoul(item, &used);
    require(used == item.size() && value <= maximum, std::string("invalid ") + name);
    result.push_back(uint32_t(value));
  }
  require(!result.empty(), std::string("empty ") + name);
  return result;
}
uint32_t single(const char *name, uint32_t fallback, uint32_t maximum) {
  const auto values = list(name, {fallback}, maximum);
  require(values.size() == 1, std::string(name) + " requires one value");
  return values[0];
}

struct SHA256 final {
  CC_SHA256_CTX context{};
  SHA256() { CC_SHA256_Init(&context); }
  void add(const void *data, uint64_t bytes) {
    const auto *next = static_cast<const std::byte *>(data);
    while (bytes) {
      const auto count = CC_LONG(std::min<uint64_t>(bytes, UINT32_MAX));
      CC_SHA256_Update(&context, next, count);
      next += count;
      bytes -= count;
    }
  }
  void add(const MetalBuffer &buffer) {
    require(buffer && buffer.contents(), "hash input is not CPU-visible");
    add(buffer.contents(), buffer.sizeBytes());
  }
  std::string finish() {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    std::ostringstream out;
    for (auto byte : digest)
      out << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
    return out.str();
  }
};
std::string hexDigest(const std::array<uint8_t, 32> &digest) {
  std::ostringstream out;
  for (auto byte : digest)
    out << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
  return out.str();
}

struct Guarded final {
  MetalBuffer base, view;
  uint64_t words;
  Guarded(MetalBackend &backend, uint64_t count) : words(count) {
    require(count > 0 && count <= (UINT64_MAX / 2 - 2 * kGuardWords),
            "invalid guarded buffer extent");
    base = backend.allocateBuffer((count + 2 * kGuardWords) * 2,
      BufferStorage::Shared, "PLE post oracle guarded buffer");
    view = backend.view(base, kGuardWords * 2, count * 2);
    clear();
  }
  void clear() {
    std::fill_n(static_cast<uint16_t *>(base.contents()), words + 2 * kGuardWords, kCanary);
  }
  void load(std::span<const uint16_t> values) {
    require(values.size() == words, "guarded input extent differs");
    std::memcpy(view.contents(), values.data(), values.size_bytes());
  }
  void check(bool finite = true) const {
    const auto *data = static_cast<const uint16_t *>(base.contents());
    for (uint64_t i = 0; i < kGuardWords; ++i)
      require(data[i] == kCanary && data[kGuardWords + words + i] == kCanary,
              "PLE post output guard overwritten");
    if (finite)
      for (uint64_t i = 0; i < words; ++i)
        if (!std::isfinite(number(data[kGuardWords + i])))
          throw std::runtime_error("PLE post output unwritten or nonfinite at element " +
                                   std::to_string(i));
  }
  std::span<const uint16_t> values() const {
    return {static_cast<const uint16_t *>(view.contents()), size_t(words)};
  }
};
uint64_t compareBytes(const MetalBuffer &actual, const MetalBuffer &expected,
                      const char *name) {
  require(actual.sizeBytes() == expected.sizeBytes(), std::string(name) + " extent differs");
  if (std::memcmp(actual.contents(), expected.contents(), actual.sizeBytes()) == 0)
    return actual.sizeBytes();
  const auto *a = static_cast<const uint8_t *>(actual.contents());
  const auto *b = static_cast<const uint8_t *>(expected.contents());
  for (uint64_t i = 0; i < actual.sizeBytes(); ++i)
    if (a[i] != b[i])
      throw std::runtime_error(std::string(name) + " differs at byte " +
        std::to_string(i) + " actual=" + std::to_string(a[i]) +
        " expected=" + std::to_string(b[i]));
  throw std::runtime_error(std::string(name) + " memcmp mismatch without differing byte");
}
uint64_t compareWords(const MetalBuffer &actual, std::span<const uint16_t> expected,
                      const char *name) {
  require(actual.sizeBytes() == expected.size_bytes(), std::string(name) + " extent differs");
  if (std::memcmp(actual.contents(), expected.data(), expected.size_bytes()) == 0)
    return expected.size_bytes();
  const auto *a = static_cast<const uint16_t *>(actual.contents());
  for (uint64_t i = 0; i < expected.size(); ++i)
    if (a[i] != expected[i])
      throw std::runtime_error(std::string(name) + " differs at BF16 element " +
        std::to_string(i) + " actual=" + std::to_string(a[i]) +
        " expected=" + std::to_string(expected[i]));
  throw std::runtime_error(std::string(name) + " comparison failed");
}

struct ResultBuffers final {
  Guarded nk, nq, gv, nc, state, ple, injected, diagnostics;
  FlashPLEPostScratch scratch;
  ResultBuffers(MetalBackend &backend, FlashPLEGeometry g)
    : nk(backend, uint64_t(g.lanes) * g.rows * g.width * g.streams),
      nq(backend, nk.words), gv(backend, nk.words), nc(backend, nk.words),
      state(backend, uint64_t(g.lanes) * 9 * g.width * g.streams),
      ple(backend, nk.words), injected(backend, nk.words), diagnostics(backend, 2),
      scratch{nk.view, nq.view, gv.view, nc.view} {}
  void resetState(std::span<const uint16_t> before) {
    state.load(before);
    *static_cast<uint32_t *>(diagnostics.view.contents()) = kSticky;
  }
  void check(uint32_t expectedDiagnostics = kSticky) const {
    for (const auto *buffer : {&nk, &nq, &gv, &nc, &state, &ple, &injected})
      buffer->check();
    diagnostics.check(false);
    require(*static_cast<const uint32_t *>(diagnostics.view.contents()) == expectedDiagnostics,
            "PLE post changed sticky diagnostics");
  }
};
uint64_t compareResults(const ResultBuffers &actual, const ResultBuffers &expected) {
  uint64_t bytes = 0;
  const std::array<std::pair<const Guarded *, const Guarded *>, 8> pairs{{
    {&actual.nk, &expected.nk}, {&actual.nq, &expected.nq},
    {&actual.gv, &expected.gv}, {&actual.nc, &expected.nc},
    {&actual.state, &expected.state}, {&actual.ple, &expected.ple},
    {&actual.injected, &expected.injected}, {&actual.diagnostics, &expected.diagnostics}
  }};
  const std::array<const char *, 8> names{"normalized_keys", "normalized_queries",
    "gated_values", "normalized_convolution", "convolution_state", "ple_output",
    "injected_output", "diagnostics"};
  for (size_t i = 0; i < pairs.size(); ++i)
    bytes += compareBytes(pairs[i].first->view, pairs[i].second->view, names[i]);
  return bytes;
}

FlashTensor makeNorm(MetalBackend &backend, uint32_t width, FlashDType dtype,
                      NormConvention convention, uint64_t seed) {
  const uint64_t bytes = uint64_t(width) * (dtype == FlashDType::F32 ? 4 : 2);
  auto buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, "PLE synthetic gamma");
  for (uint32_t i = 0; i < width; ++i) {
    const float x = seeded(i, seed, convention == NormConvention::OnePlusWeight ? .3f : 1.3f);
    if (dtype == FlashDType::BF16)
      static_cast<uint16_t *>(buffer.contents())[i] = bf16(x);
    else {
      float value = x + seeded(i, seed + 13, .00073f);
      if (i < 4) {
        constexpr std::array<float, 4> traps{0x1p-10f, -1.0f + 0x1p-12f,
                                           0x1p-20f, -1.0f + 0x1p-20f};
        value = traps[i];
      }
      static_cast<float *>(buffer.contents())[i] = value;
    }
  }
  return {buffer, dtype, {width}, bytes};
}
FlashPLEWeights syntheticWeights(MetalBackend &backend, FlashPLEGeometry g,
                                 NormConvention convention, uint32_t layout) {
  const uint32_t hyperWidth = g.width * g.streams;
  require(layout <= 3, "unknown norm dtype layout");
  const std::array<std::array<FlashDType, 3>, 4> layouts{{
    {FlashDType::BF16, FlashDType::BF16, FlashDType::BF16},
    {FlashDType::F32, FlashDType::F32, FlashDType::F32},
    {FlashDType::BF16, FlashDType::F32, FlashDType::BF16},
    {FlashDType::F32, FlashDType::BF16, FlashDType::F32}
  }};
  FlashPLEWeights weights;
  weights.normConvention = convention;
  weights.normKey = makeNorm(backend, hyperWidth, layouts[layout][0], convention, 0x11a);
  weights.normQuery = makeNorm(backend, hyperWidth, layouts[layout][1], convention, 0x213);
  weights.normConvolution = makeNorm(backend, hyperWidth, layouts[layout][2], convention, 0x339);
  const uint64_t bytes = uint64_t(hyperWidth) * 4 * 2;
  auto convolution = backend.allocateBuffer(bytes, BufferStorage::Shared, "PLE synthetic convolution");
  for (uint64_t i = 0; i < uint64_t(hyperWidth) * 4; ++i)
    static_cast<uint16_t *>(convolution.contents())[i] = bf16(seeded(i, 0x487, .22f));
  weights.convolution = {convolution, FlashDType::BF16, {hyperWidth, 4, 1}, bytes};
  return weights;
}

struct Inputs final {
  MetalBuffer hyper, keys, values, mask, tokens, history, beforeState;
  std::vector<uint16_t> stateValues;
  Inputs(MetalBackend &backend, FlashPLEGeometry g, bool nonzero, bool masked,
          uint64_t seed = 0) {
    const uint64_t rows = uint64_t(g.lanes) * g.rows;
    const uint64_t hw = uint64_t(g.width) * g.streams;
    hyper = backend.allocateBuffer(rows * hw * 2, BufferStorage::Shared, "PLE synthetic hyper input");
    keys = backend.allocateBuffer(rows * hw * 2, BufferStorage::Shared, "PLE synthetic projected keys");
    values = backend.allocateBuffer(rows * g.width * 2, BufferStorage::Shared, "PLE synthetic projected values");
    for (uint64_t i = 0; i < rows * hw; ++i) {
      static_cast<uint16_t *>(hyper.contents())[i] = bf16(seeded(i, seed + 0x151, .87f));
      static_cast<uint16_t *>(keys.contents())[i] = bf16(seeded(i, seed + 0x263, .95f));
    }
    // Include signed zero and zero-norm rows in addition to signed finite values.
    if (rows >= 4) {
      std::fill_n(static_cast<uint16_t *>(keys.contents()) + hw, hw, 0);
      std::fill_n(static_cast<uint16_t *>(hyper.contents()) + 2 * hw, hw, 0x8000);
    }
    for (uint64_t i = 0; i < rows * g.width; ++i)
      static_cast<uint16_t *>(values.contents())[i] = bf16(seeded(i, seed + 0x375, .72f));
    if (masked) {
      mask = backend.allocateBuffer(rows * 4, BufferStorage::Shared, "PLE synthetic mask");
      for (uint64_t i = 0; i < rows; ++i)
        static_cast<uint32_t *>(mask.contents())[i] = i % 3 == 0 ? 0 : (i % 2 ? 1 : 0x80000000);
    }
    tokens = backend.allocateBuffer(rows * 8, BufferStorage::Shared, "PLE restore synthetic tokens");
    for (uint64_t i = 0; i < rows; ++i)
      static_cast<int64_t *>(tokens.contents())[i] = i % 7 == 3 ? g.eosToken :
        int64_t(randomWord(i + seed + 0x397) % g.vocabularySize);
    history = backend.allocateBuffer(uint64_t(g.lanes) * 16, BufferStorage::Shared,
                                      "PLE restore synthetic history");
    for (uint32_t lane = 0; lane < g.lanes; ++lane) {
      static_cast<int64_t *>(history.contents())[uint64_t(lane) * 2] = nonzero ? 100 + lane : g.eosToken;
      static_cast<int64_t *>(history.contents())[uint64_t(lane) * 2 + 1] = nonzero ? 200 + lane : g.eosToken;
    }
    stateValues.resize(uint64_t(g.lanes) * 9 * hw);
    for (uint64_t i = 0; i < stateValues.size(); ++i)
      stateValues[i] = nonzero ? bf16(seeded(i, seed + 0x489, .44f)) : 0;
    beforeState = backend.allocateBuffer(stateValues.size() * 2, BufferStorage::Shared,
                                         "PLE restore convolution snapshot");
    std::memcpy(beforeState.contents(), stateValues.data(), stateValues.size() * 2);
  }
  std::string hash(const FlashPLEWeights &w) const {
    SHA256 sha;
    for (const auto &b : {hyper, keys, values, tokens, history, beforeState,
                         w.normKey.buffer, w.normQuery.buffer,
                         w.normConvolution.buffer, w.convolution.buffer}) sha.add(b);
    if (mask) sha.add(mask);
    return sha.finish();
  }
};
void addBaseline(CommandGraph &graph, const FlashPLEWeights &w, const Inputs &in,
                  const ResultBuffers &out, FlashPLEGeometry g) {
  addPLEPostProject(graph, w, in.hyper, in.keys, in.values, out.scratch,
                    out.state.view, out.ple.view, out.diagnostics.view, g, in.mask);
  addPLEInject(graph, in.hyper, out.ple.view, out.injected.view, g);
}
void addCandidate(CommandGraph &graph, const FlashPLEWeights &w, const Inputs &in,
                   const ResultBuffers &out, FlashPLEGeometry g) {
  addPLEPostProjectFused(graph, w, in.hyper, in.keys, in.values, out.scratch,
    out.state.view, out.ple.view, out.injected.view, out.diagnostics.view, g, in.mask);
}

uint64_t qualifyInplace(MetalBackend &backend, const FlashPLEWeights &w,
                        const Inputs &in, const ResultBuffers &separate,
                        FlashPLEGeometry g) {
  Guarded hyperA(backend, separate.injected.words), hyperB(backend, separate.injected.words);
  const std::span<const uint16_t> original{
    static_cast<const uint16_t *>(in.hyper.contents()), size_t(in.hyper.sizeBytes() / 2)};
  hyperA.load(original); hyperB.load(original);
  ResultBuffers outA(backend, g), outB(backend, g);
  outA.resetState(in.stateValues); outB.resetState(in.stateValues);
  CommandGraph graphA, graphB;
  addPLEPostProject(graphA, w, hyperA.view, in.keys, in.values, outA.scratch,
    outA.state.view, outA.ple.view, outA.diagnostics.view, g, in.mask);
  addPLEInject(graphA, hyperA.view, outA.ple.view, hyperA.view, g);
  addPLEPostProjectFused(graphB, w, hyperB.view, in.keys, in.values, outB.scratch,
    outB.state.view, outB.ple.view, hyperB.view, outB.diagnostics.view, g, in.mask);
  (void)backend.submitCommand(graphA.dispatches());
  (void)backend.submitCommand(graphB.dispatches());
  hyperA.check(); hyperB.check();
  // Populate the ordinary owned result views only after the actual in-place
  // graph, allowing the same full scratch/state/diagnostics comparison helper.
  std::memcpy(outA.injected.view.contents(), hyperA.view.contents(), hyperA.view.sizeBytes());
  std::memcpy(outB.injected.view.contents(), hyperB.view.contents(), hyperB.view.sizeBytes());
  outA.check(); outB.check();
  uint64_t bytes = compareResults(outB, outA);
  bytes += compareResults(outA, separate);
  return bytes;
}

std::vector<std::string> qualifyMaskedNumeric(MetalBackend &backend,
                                             const FlashPLEWeights &w) {
  FlashPLEGeometry g;
  g.rows = 1; g.lanes = 1;
  std::vector<std::string> records;
  for (uint16_t value : {uint16_t(0x7fc0), uint16_t(0x7f80)}) {
    Inputs in(backend, g, false, true, 0x811);
    static_cast<uint16_t *>(in.values.contents())[0] = value;
    require(*static_cast<uint32_t *>(in.mask.contents()) == 0,
            "masked numeric fixture unexpectedly enabled");
    ResultBuffers a(backend, g), b(backend, g);
    a.resetState(in.stateValues); b.resetState(in.stateValues);
    CommandGraph ga, gb;
    addBaseline(ga, w, in, a, g); addCandidate(gb, w, in, b, g);
    const auto hash = in.hash(w);
    (void)backend.submitCommand(ga.dispatches());
    (void)backend.submitCommand(gb.dispatches());
    a.check(kSticky | kFlashPLEInvalidNumeric);
    b.check(kSticky | kFlashPLEInvalidNumeric);
    const auto bytes = compareResults(b, a);
    require(in.hash(w) == hash, "masked numeric case changed immutable inputs");
    std::ostringstream record;
    record << "{\"name\":" << splash::json::quote(value == 0x7fc0 ? "masked_nan_value" : "masked_infinite_value")
           << ",\"compared_bytes\":" << bytes
           << ",\"diagnostics\":" << (kSticky | kFlashPLEInvalidNumeric)
           << ",\"outputs_finite\":true,\"pass\":true}";
    records.push_back(record.str());
  }
  return records;
}

struct Packet final {
  FlashPLEWeights weights;
  MetalBuffer hyper, keys, values;
  FlashPLEPostScratch scratch;
  MetalBuffer state, ple, injected, diagnostics, mask;
  FlashPLEGeometry geometry;
};
void addPacket(CommandGraph &graph, const Packet &p) {
  addPLEPostProjectFused(graph, p.weights, p.hyper, p.keys, p.values, p.scratch,
    p.state, p.ple, p.injected, p.diagnostics, p.geometry, p.mask);
}
uint32_t hostRejections(MetalBackend &backend, const FlashPLEWeights &w,
                        const Inputs &in, const ResultBuffers &out, FlashPLEGeometry g) {
  const Packet valid{w, in.hyper, in.keys, in.values, out.scratch, out.state.view,
    out.ple.view, out.injected.view, out.diagnostics.view, in.mask, g};
  uint32_t rejected = 0;
  const auto reject = [&](auto mutate) {
    Packet p = valid;
    mutate(p);
    CommandGraph graph;
    bool caught = false;
    try { addPacket(graph, p); }
    catch (const std::invalid_argument &) { caught = true; }
    require(caught, "invalid fused PLE host parameters accepted");
    require(graph.empty(), "invalid fused PLE host call appended a partial graph");
    ++rejected;
  };
  reject([](Packet &p) { p.geometry.rows = 0; });
  reject([](Packet &p) { p.geometry.lanes = 0; });
  reject([](Packet &p) { p.geometry.width = 0; });
  reject([](Packet &p) { p.geometry.streams = 0; });
  reject([](Packet &p) { p.geometry.streams = 9; });
  reject([](Packet &p) { p.geometry.epsilon = 0; });
  reject([](Packet &p) { p.geometry.epsilon = -1; });
  reject([](Packet &p) { p.geometry.epsilon = std::numeric_limits<float>::quiet_NaN(); });
  reject([](Packet &p) { p.geometry.epsilon = std::numeric_limits<float>::infinity(); });
  reject([](Packet &p) { p.geometry.vocabularySize = 0; });
  reject([](Packet &p) { p.geometry.eosToken = p.geometry.vocabularySize; });
  reject([](Packet &p) { p.geometry.lanes = UINT32_MAX; p.geometry.rows = 2; });
  reject([](Packet &p) { p.geometry.width = UINT32_MAX; p.geometry.streams = 2; });
  reject([](Packet &p) { p.hyper = {}; });
  reject([](Packet &p) { p.keys = {}; });
  reject([](Packet &p) { p.values = {}; });
  reject([](Packet &p) { p.state = {}; });
  reject([](Packet &p) { p.ple = {}; });
  reject([](Packet &p) { p.injected = {}; });
  reject([](Packet &p) { p.diagnostics = {}; });
  reject([&](Packet &p) { p.hyper = backend.view(p.hyper, 0, p.hyper.sizeBytes() - 2); });
  reject([&](Packet &p) { p.keys = backend.view(p.keys, 0, p.keys.sizeBytes() - 2); });
  reject([&](Packet &p) { p.values = backend.view(p.values, 0, p.values.sizeBytes() - 2); });
  reject([&](Packet &p) { p.state = backend.view(p.state, 0, p.state.sizeBytes() - 2); });
  reject([&](Packet &p) { p.ple = backend.view(p.ple, 0, p.ple.sizeBytes() - 2); });
  reject([&](Packet &p) { p.injected = backend.view(p.injected, 0, p.injected.sizeBytes() - 2); });
  reject([&](Packet &p) { p.diagnostics = backend.view(p.diagnostics, 0, 2); });
  reject([](Packet &p) { p.scratch.normalizedKeys = {}; });
  reject([](Packet &p) { p.scratch.normalizedQueries = p.scratch.normalizedKeys; });
  reject([](Packet &p) { p.scratch.gatedValues = p.hyper; });
  reject([](Packet &p) { p.scratch.normalizedConvolution = p.keys; });
  reject([](Packet &p) { p.ple = p.hyper; });
  reject([](Packet &p) { p.state = p.hyper; });
  reject([](Packet &p) { p.injected = p.ple; });
  reject([](Packet &p) { p.injected = p.scratch.normalizedQueries; });
  reject([](Packet &p) { p.weights.normKey.dtype = FlashDType::I64; });
  reject([](Packet &p) { p.weights.normQuery.logicalBytes = 1; });
  reject([](Packet &p) { p.weights.normConvolution.shape[0] += 1; });
  reject([](Packet &p) { p.weights.convolution.dtype = FlashDType::F32; });
  reject([](Packet &p) { p.weights.convolution.shape[1] = 3; });
  reject([](Packet &p) { p.weights.normConvention = static_cast<NormConvention>(255); });
  if (in.mask)
    reject([&](Packet &p) { p.mask = backend.view(p.mask, 0, p.mask.sizeBytes() - 2); });
  return rejected;
}

std::vector<uint16_t> prefixState(std::span<const uint16_t> before,
                                  const MetalBuffer &normalized,
                                  FlashPLEGeometry g, std::span<const uint32_t> kept) {
  const uint64_t hw = uint64_t(g.width) * g.streams;
  require(before.size() == uint64_t(g.lanes) * 9 * hw && kept.size() == g.lanes,
          "CPU prefix state input extent differs");
  const auto *norm = static_cast<const uint16_t *>(normalized.contents());
  std::vector<uint16_t> result(before.size());
  for (uint32_t lane = 0; lane < g.lanes; ++lane) {
    require(kept[lane] <= g.rows, "CPU prefix retained count exceeds rows");
    for (uint32_t row = 0; row < 9; ++row) {
      const uint64_t timeline = uint64_t(kept[lane]) + row;
      const auto *source = timeline < 9 ?
        before.data() + (uint64_t(lane) * 9 + timeline) * hw :
        norm + (uint64_t(lane) * g.rows + timeline - 9) * hw;
      std::memcpy(result.data() + (uint64_t(lane) * 9 + row) * hw, source, hw * 2);
    }
  }
  return result;
}
std::vector<int64_t> prefixHistory(const Inputs &in, FlashPLEGeometry g,
                                  std::span<const uint32_t> kept) {
  const auto *before = static_cast<const int64_t *>(in.history.contents());
  const auto *tokens = static_cast<const int64_t *>(in.tokens.contents());
  std::vector<int64_t> result(uint64_t(g.lanes) * 2);
  for (uint32_t lane = 0; lane < g.lanes; ++lane)
    for (uint32_t slot = 0; slot < 2; ++slot) {
      const uint64_t timeline = uint64_t(kept[lane]) + slot;
      result[uint64_t(lane) * 2 + slot] = timeline < 2 ?
        before[uint64_t(lane) * 2 + timeline] :
        tokens[uint64_t(lane) * g.rows + timeline - 2];
    }
  return result;
}
struct PrefixQualification final { uint32_t windows = 0; uint64_t bytes = 0; };
PrefixQualification qualifyPrefixes(MetalBackend &backend, const FlashPLEWeights &w,
                                     const Inputs &in, const ResultBuffers &control,
                                     const ResultBuffers &candidate, FlashPLEGeometry g) {
  auto keptBuffer = backend.allocateBuffer(uint64_t(g.lanes) * 4, BufferStorage::Shared,
                                           "PLE prefix retained counts");
  Guarded historyA(backend, uint64_t(g.lanes) * 8), historyB(backend, uint64_t(g.lanes) * 8);
  Guarded stateA(backend, control.state.words), stateB(backend, control.state.words);
  Guarded diagA(backend, 2), diagB(backend, 2);
  FlashPLEGeometry continuationGeometry = g;
  continuationGeometry.rows = 4;
  Inputs continuation(backend, continuationGeometry, true, bool(in.mask), 0x771);
  ResultBuffers continuationA(backend, continuationGeometry), continuationB(backend, continuationGeometry);
  CommandGraph continuationGraphA, continuationGraphB;
  addBaseline(continuationGraphA, w, continuation, continuationA, continuationGeometry);
  addCandidate(continuationGraphB, w, continuation, continuationB, continuationGeometry);
  std::vector<uint32_t> kept(g.lanes);
  PrefixQualification result;
  for (uint32_t retain = 0; retain <= std::min(g.rows, 16u); ++retain) {
    // Different retained counts in neighboring lanes also check lane strides.
    for (uint32_t lane = 0; lane < g.lanes; ++lane)
      kept[lane] = (retain + lane) % (std::min(g.rows, 16u) + 1);
    std::memcpy(keptBuffer.contents(), kept.data(), kept.size() * 4);
    historyA.clear(); historyB.clear(); stateA.clear(); stateB.clear();
    *static_cast<uint32_t *>(diagA.view.contents()) = kSticky;
    *static_cast<uint32_t *>(diagB.view.contents()) = kSticky;
    CommandGraph restoreA, restoreB;
    addPLERestorePrefix(restoreA, in.history, in.tokens, in.beforeState, control.nc.view,
      keptBuffer, historyA.view, stateA.view, diagA.view, g);
    addPLERestorePrefix(restoreB, in.history, in.tokens, in.beforeState, candidate.nc.view,
      keptBuffer, historyB.view, stateB.view, diagB.view, g);
    (void)backend.submitCommand(restoreA.dispatches());
    (void)backend.submitCommand(restoreB.dispatches());
    stateA.check(); stateB.check(); historyA.check(false); historyB.check(false);
    diagA.check(false); diagB.check(false);
    require(*static_cast<uint32_t *>(diagA.view.contents()) == kSticky &&
            *static_cast<uint32_t *>(diagB.view.contents()) == kSticky,
            "PLE prefix restore changed sticky diagnostics");
    result.bytes += compareBytes(stateB.view, stateA.view, "restored convolution state");
    result.bytes += compareBytes(historyB.view, historyA.view, "restored token history");
    const auto expectedState = prefixState(in.stateValues, control.nc.view, g, kept);
    result.bytes += compareWords(stateA.view, expectedState, "CPU prefix convolution state");
    const auto expectedHistory = prefixHistory(in, g, kept);
    require(std::memcmp(historyA.view.contents(), expectedHistory.data(),
                        expectedHistory.size() * 8) == 0, "CPU prefix token history differs");
    continuationA.resetState(expectedState);
    continuationB.resetState(expectedState);
    // Use the actual restored views so the continuation validates the retained
    // state, rather than sharing only the independent host reference.
    std::memcpy(continuationA.state.view.contents(), stateA.view.contents(), stateA.view.sizeBytes());
    std::memcpy(continuationB.state.view.contents(), stateB.view.contents(), stateB.view.sizeBytes());
    (void)backend.submitCommand(continuationGraphA.dispatches());
    (void)backend.submitCommand(continuationGraphB.dispatches());
    continuationA.check(); continuationB.check();
    result.bytes += compareResults(continuationB, continuationA);
    ++result.windows;
  }
  return result;
}

double median(std::vector<double> values) {
  require(!values.empty(), "empty timing samples");
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) * .5;
}
struct Times final {
  std::vector<double> gpu, wall;
  void add(CommandTiming t) {
    require(std::isfinite(t.gpuSeconds) && t.gpuSeconds > 0 &&
            std::isfinite(t.wallSeconds) && t.wallSeconds > 0, "invalid command timing");
    gpu.push_back(t.gpuSeconds); wall.push_back(t.wallSeconds);
  }
  void write(std::ostream &out) const {
    out << "{\"samples\":" << gpu.size() << ",\"median_gpu_seconds\":" << median(gpu)
        << ",\"median_wall_seconds\":" << median(wall) << ",\"gpu_seconds\":[";
    for (size_t i = 0; i < gpu.size(); ++i) out << (i ? "," : "") << gpu[i];
    out << "]}";
  }
};

void cpuSelfTest() {
  uint64_t checks = 0;
  for (uint32_t word = 0; word < 65536; ++word)
    if (std::isfinite(number(uint16_t(word)))) {
      require(bf16(number(uint16_t(word))) == word, "BF16 finite round-trip differs");
      ++checks;
    }
  require(bf16(std::bit_cast<float>(uint32_t(0x3f808000))) == 0x3f80 &&
          bf16(std::bit_cast<float>(uint32_t(0x3f818000))) == 0x3f82,
          "BF16 nearest-even rounding differs");
  checks += 2;
  for (uint32_t lanes = 1; lanes <= 4; ++lanes)
    for (uint32_t rows : {1u, 4u, 8u, 16u, 128u, 512u, 2048u})
      for (uint32_t kept = 0; kept <= std::min(rows, 16u); ++kept)
        for (uint32_t lane = 0; lane < lanes; ++lane)
          for (uint32_t historyRow = 0; historyRow < 9; ++historyRow) {
            const uint64_t timeline = uint64_t(kept) + historyRow;
            const uint64_t source = timeline < 9 ? uint64_t(lane) * 9 + timeline :
              uint64_t(lane) * rows + timeline - 9;
            require(source < uint64_t(lanes) * (timeline < 9 ? 9 : rows),
                    "prefix history source exceeds lane extent");
            ++checks;
          }
  SHA256 hash;
  hash.add("abc", 3);
  require(hash.finish() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA256 self-test differs");
  ++checks;
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
            << ",\"gpu_commands\":0,\"metal_backend_created\":false}\n";
}
void writeReport(const std::filesystem::path &path, const std::string &value) {
  std::ofstream out(path);
  require(bool(out), "cannot open PLE post fused report");
  out << value << '\n';
  out.close();
  require(bool(out), "cannot write PLE post fused report");
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    std::vector<std::string> records;
    std::string metadata;
    std::filesystem::path reportPath;
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") {
        cpuSelfTest();
        return 0;
      }
      if (argc != 3 && !(argc == 5 && std::string(argv[3]) == "--package"))
        throw std::invalid_argument("usage: flash-ple-post-fused-oracle METALLIB REPORT_JSON [--package PACKAGE] | --cpu-self-test");
      reportPath = argv[2];
      const auto rowsList = list("FLASH_PLE_POST_ROWS", {1, 4, 8, 16, 128, 512, 2048}, 2048);
      const auto lanesList = list("FLASH_PLE_POST_LANES", {1, 2, 3, 4}, 4);
      const auto norms = list("FLASH_PLE_POST_NORMS", {0, 1}, 1);
      const auto layouts = list("FLASH_PLE_POST_GAMMA_LAYOUTS", {0, 1, 2, 3}, 3);
      const auto histories = list("FLASH_PLE_POST_HISTORY", {0, 1}, 1);
      const auto masks = list("FLASH_PLE_POST_MASK", {0, 1}, 1);
      const bool full = single("FLASH_PLE_POST_FULL", 0, 1) != 0;
      const bool prefixes = single("FLASH_PLE_POST_PREFIX", 1, 1) != 0;
      const bool inplace = single("FLASH_PLE_POST_INPLACE", 1, 1) != 0;
      const bool maskedNumeric = single("FLASH_PLE_POST_MASKED_NUMERIC", 1, 1) != 0;
      const uint32_t repeats = single("FLASH_PLE_POST_REPEATS", 5, 100);
      const uint32_t warmup = single("FLASH_PLE_POST_WARMUP", 2, 100);
      const auto widths = list("FLASH_PLE_POST_WIDTHS", {32, 65, 128, 513, 768, 1025, 2560, 2563, 4097}, 65536);
      const uint32_t streams = single("FLASH_PLE_POST_STREAMS", 4, 8);
      require(repeats && warmup && streams, "timing and geometry values must be positive");
      for (auto width : widths) require(width > 0, "widths must be positive");
      for (auto rows : rowsList) require(rows > 0, "rows must be positive");
      for (auto lanes : lanesList) require(lanes > 0, "lanes must be positive");
      const std::string filter = std::getenv("FLASH_PLE_POST_FILTER") ?
        std::getenv("FLASH_PLE_POST_FILTER") : "";
      MetalBackend backend(argv[1]);
      std::optional<FlashWeights> package;
      std::optional<FlashPLEWeights> checkpointPost;
      if (argc == 5) {
        require(widths.size() == 1 && widths[0] == 2560 && streams == 4,
                "checkpoint requires WIDTHS=2560/STREAMS=4");
        package.emplace(FlashWeights::load(backend, argv[4]));
        checkpointPost.emplace(FlashPLEWeights::fromWeights(*package));
      }
      std::ostringstream meta;
      meta << "\"scope\":\"PLE post-projection plus injection only; affine projections, ngram hash and gather excluded\""
           << ",\"comparison\":\"bit-exact BF16 scratch/output/state and sticky U32 diagnostics\""
           << ",\"timing\":\"uninstrumented whole command graphs; state reset excluded; alternating paired order\""
           << ",\"immutable_inputs_preserved\":true,\"guard_bytes_per_side\":" << kGuardWords * 2
           << ",\"metallib_sha256\":" << splash::json::quote(hexDigest(backend.metallibSha256()))
           << ",\"fixture\":" << splash::json::quote(package ? "checkpoint post tensors with synthetic projected inputs" : "synthetic signed finite tensors")
           << ",\"shader_validation_environment\":" << splash::json::quote(std::getenv("MTL_SHADER_VALIDATION") ? std::getenv("MTL_SHADER_VALIDATION") : "unset");
      if (package)
        meta << ",\"source_identity\":" << splash::json::quote(package->sourceIdentity())
             << ",\"manifest_fingerprint\":" << splash::json::quote(package->manifestFingerprint());
      metadata = meta.str();
      uint64_t ordinal = 0;
      for (uint32_t width : widths)
        for (uint32_t rows : rowsList)
        for (size_t li = 0; li < lanesList.size(); ++li)
          for (size_t ni = 0; ni < norms.size(); ++ni)
            for (size_t di = 0; di < layouts.size(); ++di)
              for (size_t hi = 0; hi < histories.size(); ++hi)
                for (size_t mi = 0; mi < masks.size(); ++mi) {
                  if (checkpointPost && (ni != 0 || di != 0)) continue;
                  // Compact pairwise sweep. Set FULL=1 for the entire filtered
                  // product; every requested row count still appears by default.
                  if (!full && width != 2560 && rows != 4 && !std::getenv("FLASH_PLE_POST_ROWS")) continue;
                  if (!full && ((!checkpointPost && (ni != li % norms.size() || di != li % layouts.size())) ||
                      hi != (li / 2) % histories.size() || mi != (li + 1) % masks.size())) continue;
                  FlashPLEGeometry g;
                  g.rows = rows; g.lanes = lanesList[li]; g.width = width; g.streams = streams;
                  const auto norm = norms[ni] == 0 ? NormConvention::OnePlusWeight : NormConvention::DirectGamma;
                  const uint32_t layout = layouts[di];
                  const bool history = histories[hi] != 0, masked = masks[mi] != 0;
                  const std::string name = "width" + std::to_string(width) + "-rows" + std::to_string(rows) + "-lanes" +
                    std::to_string(g.lanes) + "-norm" + std::to_string(norms[ni]) +
                    "-gamma" + std::to_string(layout) + "-history" + std::to_string(history) +
                    "-mask" + std::to_string(masked);
                  if (!filter.empty() && name.find(filter) == std::string::npos) continue;
                  const auto w = checkpointPost ? *checkpointPost : syntheticWeights(backend, g, norm, layout);
                  Inputs in(backend, g, history, masked);
                  ResultBuffers control(backend, g), candidate(backend, g);
                  const auto beforeHash = in.hash(w);
                  const uint32_t rejections = hostRejections(backend, w, in, candidate, g);
                  CommandGraph controlGraph, candidateGraph;
                  addBaseline(controlGraph, w, in, control, g);
                  addCandidate(candidateGraph, w, in, candidate, g);
                  control.resetState(in.stateValues); candidate.resetState(in.stateValues);
                  (void)backend.submitCommand(controlGraph.dispatches());
                  (void)backend.submitCommand(candidateGraph.dispatches());
                  control.check(); candidate.check();
                  uint64_t comparedBytes = compareResults(candidate, control);
                  const uint64_t inplaceBytes = inplace ? qualifyInplace(backend, w, in, control, g) : 0;
                  PrefixQualification prefix;
                  if (prefixes) prefix = qualifyPrefixes(backend, w, in, control, candidate, g);
                  for (uint32_t i = 0; i < warmup; ++i) {
                    control.resetState(in.stateValues); candidate.resetState(in.stateValues);
                    (void)backend.submitCommand(controlGraph.dispatches());
                    (void)backend.submitCommand(candidateGraph.dispatches());
                  }
                  Times a, b;
                  for (uint32_t i = 0; i < repeats; ++i) {
                    control.resetState(in.stateValues); candidate.resetState(in.stateValues);
                    if (i % 2 == 0) {
                      a.add(backend.submitCommand(controlGraph.dispatches()));
                      b.add(backend.submitCommand(candidateGraph.dispatches()));
                    } else {
                      b.add(backend.submitCommand(candidateGraph.dispatches()));
                      a.add(backend.submitCommand(controlGraph.dispatches()));
                    }
                  }
                  control.check(); candidate.check();
                  comparedBytes += compareResults(candidate, control);
                  require(in.hash(w) == beforeHash, "PLE post mutated immutable input/weight/snapshot bytes");
                  std::ostringstream record;
                  record << std::setprecision(12) << "{\"name\":" << splash::json::quote(name)
                    << ",\"width\":" << width << ",\"streams\":" << streams
                    << ",\"rows_per_lane\":" << rows << ",\"lanes\":" << g.lanes
                    << ",\"norm_convention\":" << splash::json::quote(w.normConvention == NormConvention::OnePlusWeight ? "one_plus_weight" : "direct_gamma")
                    << ",\"gamma_layout\":" << layout << ",\"nonzero_history\":" << (history ? "true" : "false")
                    << ",\"masked\":" << (masked ? "true" : "false")
                    << ",\"host_rejections\":" << rejections << ",\"compared_bytes\":" << comparedBytes
                    << ",\"inplace_injection_qualified\":" << (inplace ? "true" : "false")
                    << ",\"inplace_compared_bytes\":" << inplaceBytes
                    << ",\"prefix_windows\":" << prefix.windows << ",\"prefix_compared_bytes\":" << prefix.bytes
                    << ",\"input_sha256\":" << splash::json::quote(beforeHash)
                    << ",\"baseline_dispatches\":" << controlGraph.dispatches().size()
                    << ",\"candidate_dispatches\":" << candidateGraph.dispatches().size()
                    << ",\"baseline\":";
                  a.write(record); record << ",\"candidate\":"; b.write(record);
                  record << ",\"gpu_speedup\":" << median(a.gpu) / median(b.gpu)
                         << ",\"wall_speedup\":" << median(a.wall) / median(b.wall) << ",\"pass\":true}";
                  records.push_back(record.str());
                  std::cout << "PASS " << ++ordinal << ' ' << name << " gpu_speedup="
                            << median(a.gpu) / median(b.gpu) << '\n' << std::flush;
                }
      require(!records.empty(), "PLE post filter selected no cases");
      std::vector<std::string> diagnosticRecords;
      if (maskedNumeric) {
        FlashPLEGeometry diagnosticGeometry;
        const auto w = checkpointPost ? *checkpointPost :
          syntheticWeights(backend, diagnosticGeometry, NormConvention::OnePlusWeight, 2);
        diagnosticRecords = qualifyMaskedNumeric(backend, w);
      }
      std::ostringstream report;
      report << std::setprecision(12) << "{\"pass\":true,\"cases\":" << records.size() << ','
             << metadata << ",\"results\":[";
      for (size_t i = 0; i < records.size(); ++i) report << (i ? "," : "") << records[i];
      report << "],\"masked_numeric_results\":[";
      for (size_t i = 0; i < diagnosticRecords.size(); ++i)
        report << (i ? "," : "") << diagnosticRecords[i];
      report << "]}";
      writeReport(reportPath, report.str());
      return 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      if (!reportPath.empty()) {
        std::ostringstream failure;
        failure << "{\"pass\":false,\"error\":" << splash::json::quote(error.what())
                << ",\"completed_cases\":" << records.size();
        if (!metadata.empty()) failure << ',' << metadata;
        failure << ",\"results\":[";
        for (size_t i = 0; i < records.size(); ++i) failure << (i ? "," : "") << records[i];
        failure << "]}";
        try { writeReport(reportPath, failure.str()); } catch (...) {}
      }
      return 1;
    }
  }
}
