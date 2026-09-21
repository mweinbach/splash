#include "FlashHCDownF32Candidate.hpp"
#include "FlashHCDownF32CandidateABI.h"
#include "metal/abi/FlashFloatDenseCache.h"
#include <array>
#include <limits>
#include <string>

namespace splash::flash::candidate {
namespace {
void require(bool condition, const char *reason) {
  if (!condition) throw std::invalid_argument(std::string("HC down F32 candidate: ") + reason);
}
void bytes(const metal::MetalBuffer &buffer, uint64_t count) {
  require(buffer && buffer.contents() && buffer.sizeBytes() >= count, "invalid Shared byte extent");
}
bool overlaps(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto x = reinterpret_cast<uintptr_t>(a.contents()), y = reinterpret_cast<uintptr_t>(b.contents());
  require(x && y && x <= std::numeric_limits<uintptr_t>::max() - a.sizeBytes() &&
      y <= std::numeric_limits<uintptr_t>::max() - b.sizeBytes(), "invalid address extent");
  return x < y + b.sizeBytes() && y < x + a.sizeBytes();
}
FlashHCFusedMatrix matrix(const FlashAffineProjection &p) {
  return {p.inputSize, p.outputSize, p.bits, p.groupSize,
      p.weightRowStrideBytes, p.parameterRowStrideBytes};
}
void source(const FlashAffineProjection &p, uint32_t outputs) {
  require(p.experts == 1 && p.inputSize == 10240 && p.outputSize == outputs &&
      (p.bits == 4 || p.bits == 5 || p.bits == 6 || p.bits == 8) &&
      (p.groupSize == 32 || p.groupSize == 64 || p.groupSize == 128) &&
      p.weights && p.scales && p.biases && p.parameterRowStrideBytes % 2 == 0 &&
      p.weights->dtype == FlashDType::U32 && p.scales->dtype == FlashDType::BF16 &&
      p.biases->dtype == FlashDType::BF16, "invalid original HC projection");
  const uint64_t packed = (10240 * p.bits + 7) / 8, parameters = 10240 / p.groupSize * 2;
  constexpr auto top = std::numeric_limits<uint64_t>::max();
  require(p.weightRowStrideBytes >= packed && p.parameterRowStrideBytes >= parameters &&
      p.weightRowStrideBytes <= (top - packed) / (outputs - 1) &&
      p.parameterRowStrideBytes <= (top - parameters) / (outputs - 1), "invalid/overflowing source strides");
  const uint64_t weightBytes = uint64_t{outputs - 1} * p.weightRowStrideBytes + packed;
  const uint64_t parameterBytes = uint64_t{outputs - 1} * p.parameterRowStrideBytes + parameters;
  require(p.weights->logicalBytes >= weightBytes && p.scales->logicalBytes >= parameterBytes &&
      p.biases->logicalBytes >= parameterBytes, "source logical extent differs");
  bytes(p.weights->buffer, weightBytes); bytes(p.scales->buffer, parameterBytes);
  bytes(p.biases->buffer, parameterBytes);
}
HCDownF32CandidateParams parameters(const FlashAffineProjection &down,
    const FlashAffineProjection *injection, uint32_t rows, uint32_t n,
    uint32_t parts, bool debug) {
  source(down, 320); if (injection) source(*injection, 4);
  require(rows == 4 || rows == 8 || rows == 16, "rows must be4/8/16");
  HCDownF32CandidateParams p{};
  p.literal.rows = rows; p.literal.width = 2560; p.literal.streams = 4;
  p.literal.lowrank = 320; p.literal.has_injection = injection != nullptr;
  p.literal.simdgroups = 4; p.literal.down = matrix(down);
  p.literal.injection = matrix(injection ? *injection : down);
  p.padded_rows = (rows + 7) / 8 * 8; p.tile_outputs = n;
  p.partitions = parts; p.write_debug = debug; return p;
}
void outputs(metal::MetalBuffer normalized, metal::MetalBuffer activated,
    metal::MetalBuffer gates, metal::MetalBuffer diagnostic, uint32_t rows,
    bool injection, const HCDownF32Debug &debug) {
  bytes(normalized, uint64_t{rows} * 10240 * 2); bytes(activated, uint64_t{rows} * 320 * 2);
  bytes(diagnostic, 4); require(!overlaps(normalized, activated) && !overlaps(diagnostic, normalized) &&
      !overlaps(diagnostic, activated), "mutable/input overlap");
  if (injection) {
    bytes(gates, uint64_t{rows} * 4 * 2);
    for (const auto &b : {normalized, activated, diagnostic}) require(!overlaps(gates, b), "injection output overlap");
  }
  require(bool(debug.rawBF16) == bool(debug.rawF32), "paired debug planes are required");
  if (debug.rawBF16) {
    bytes(debug.rawBF16, uint64_t{rows} * 324 * 2); bytes(debug.rawF32, uint64_t{rows} * 324 * 4);
    require(!overlaps(debug.rawBF16, debug.rawF32), "debug planes overlap");
    for (const auto &b : {normalized, activated, diagnostic}) {
      require(!overlaps(debug.rawBF16, b) && !overlaps(debug.rawF32, b), "debug output overlap");
    }
    if (injection) require(!overlaps(debug.rawBF16, gates) && !overlaps(debug.rawF32, gates), "debug/gate overlap");
  }
}
void immutable(const metal::MetalBuffer &weight, metal::MetalBuffer activated,
    metal::MetalBuffer gates, metal::MetalBuffer diagnostic, bool injection,
    const HCDownF32Debug &debug) {
  for (const auto &b : {activated, diagnostic}) require(!overlaps(weight, b), "writer aliases immutable source");
  if (injection) require(!overlaps(weight, gates), "injection writer aliases immutable source");
  if (debug.rawBF16) require(!overlaps(weight, debug.rawBF16) && !overlaps(weight, debug.rawF32), "debug writer aliases immutable source");
}
} // namespace

void addHCDownLiteralWitness(metal::MetalBackend &backend, metal::CommandGraph &graph, metal::MetalBuffer normalized,
    const FlashAffineProjection &down, const FlashAffineProjection *injection,
    metal::MetalBuffer activated, metal::MetalBuffer gates,
    metal::MetalBuffer diagnostic, uint32_t rows, const HCDownF32Debug &debug) {
  require(debug.rawBF16 && debug.rawF32, "literal witness requires raw debug planes");
  const auto p = parameters(down, injection, rows, 32, 1, true);
  outputs(normalized, activated, gates, diagnostic, rows, injection != nullptr, debug);
  const auto &inj = injection ? *injection : down;
  for (const auto &b : {down.weights->buffer, down.scales->buffer, down.biases->buffer,
      inj.weights->buffer, inj.scales->buffer, inj.biases->buffer}) {
    immutable(b, activated, gates, diagnostic, injection != nullptr, debug);
    (void)backend.view(b, 0, b.sizeBytes());
  }
  for (const auto &b : {normalized, activated, diagnostic, debug.rawBF16, debug.rawF32})
    (void)backend.view(b, 0, b.sizeBytes());
  if (injection) (void)backend.view(gates, 0, gates.sizeBytes());
  graph.add("flash_hc_down_f32_literal_witness", {normalized, down.weights->buffer,
      down.scales->buffer, down.biases->buffer, inj.weights->buffer, inj.scales->buffer,
      inj.biases->buffer, activated, injection ? gates : activated, diagnostic,
      debug.rawBF16, debug.rawF32}, p, {injection ? 81u : 80u, rows, 1}, {128, 1, 1});
}

void addHCDownF32Candidate(metal::MetalBackend &backend, metal::CommandGraph &graph,
    metal::MetalBuffer normalized, const FlashTensor &cached,
    const FlashAffineProjection &down, const FlashAffineProjection *injection,
    metal::MetalBuffer activated, metal::MetalBuffer gates,
    metal::MetalBuffer diagnostic, uint32_t rows, HCDownF32Mode mode,
    const HCDownF32Scratch &scratch, const HCDownF32Debug &debug) {
  (void)hcDownF32ModeName(mode);
  const uint32_t n = mode == HCDownF32Mode::M8N64 ? 64 : 32;
  const uint32_t parts = mode == HCDownF32Mode::M8N32Split4 ? 4 : mode == HCDownF32Mode::M8N32Split8 ? 8 : 1;
  const auto p = parameters(down, injection, rows, n, parts, bool(debug.rawBF16));
  require(cached.dtype == FlashDType::F32 && cached.shape == std::vector<uint64_t>{320, 10240} &&
      cached.logicalBytes == 320ULL * 10240 * 4, "original F32 cache geometry differs");
  bytes(cached.buffer, cached.logicalBytes);
  outputs(normalized, activated, gates, diagnostic, rows, injection != nullptr, debug);
  immutable(cached.buffer, activated, gates, diagnostic, injection != nullptr, debug);
  const auto &inj = injection ? *injection : down;
  for (const auto &b : {down.weights->buffer, down.scales->buffer, down.biases->buffer,
      inj.weights->buffer, inj.scales->buffer, inj.biases->buffer}) {
    immutable(b, activated, gates, diagnostic, injection != nullptr, debug);
    (void)backend.view(b, 0, b.sizeBytes());
  }
  (void)backend.view(normalized, 0, uint64_t{rows} * 10240 * 2);
  (void)backend.view(cached.buffer, 0, cached.logicalBytes);
  for (const auto &b : {activated, diagnostic}) (void)backend.view(b, 0, b.sizeBytes());
  if (injection) (void)backend.view(gates, 0, gates.sizeBytes());
  if (debug.rawBF16) for (const auto &b : {debug.rawBF16, debug.rawF32})
    (void)backend.view(b, 0, b.sizeBytes());
  if (mode == HCDownF32Mode::LiteralRowReuse) {
    graph.add("flash_hc_down_f32_literal_reuse_r" + std::to_string(rows),
        {normalized, cached.buffer, inj.weights->buffer, inj.scales->buffer, inj.biases->buffer,
         activated, injection ? gates : activated, diagnostic,
         debug.rawBF16 ? debug.rawBF16 : activated, debug.rawF32 ? debug.rawF32 : diagnostic},
        p, {injection ? 81u : 80u, 1, 1}, {128, 1, 1});
    return;
  }
  bytes(scratch.padded, uint64_t{p.padded_rows} * 10240 * 2);
  bytes(scratch.partials, uint64_t{p.padded_rows} * parts * 320 * 4);
  for (const auto &temporary : {scratch.padded, scratch.partials}) {
    for (const auto &b : {normalized, activated, diagnostic, cached.buffer})
      require(!overlaps(temporary, b), "candidate scratch overlap");
    if (injection) require(!overlaps(temporary, gates), "scratch/gate overlap");
    for (const auto &b : {down.weights->buffer, down.scales->buffer, down.biases->buffer,
        inj.weights->buffer, inj.scales->buffer, inj.biases->buffer})
      require(!overlaps(temporary, b), "scratch aliases immutable original source");
    immutable(temporary, debug.rawBF16 ? debug.rawBF16 : activated,
        gates, diagnostic, injection != nullptr, debug);
  }
  require(!overlaps(scratch.padded, scratch.partials), "candidate scratch planes overlap");
  (void)backend.view(scratch.padded, 0, scratch.padded.sizeBytes());
  (void)backend.view(scratch.partials, 0, scratch.partials.sizeBytes());
  const FlashFloatDenseSmallRowsParams pad{rows, p.padded_rows, 10240, 320, 0, 320, 8, 64};
  graph.add("flash_float_dense_small_rows_pad", {normalized, scratch.padded, diagnostic}, pad,
      {(uint64_t{p.padded_rows} * 10240 + 255) / 256, 1, 1}, {256, 1, 1});
  graph.add("flash_hc_down_f32_mpp_m8_n" + std::to_string(n),
      {scratch.padded, cached.buffer, scratch.partials, diagnostic}, p,
      {320 / n, p.padded_rows / 8, parts}, {128, 1, 1});
  graph.add("flash_hc_down_f32_fold_activate", {scratch.partials, activated, diagnostic,
      debug.rawBF16 ? debug.rawBF16 : activated, debug.rawF32 ? debug.rawF32 : diagnostic},
      p, {(uint64_t{rows} * 320 + 255) / 256, 1, 1}, {256, 1, 1});
  if (injection) graph.add("flash_hc_down_f32_literal_injection",
      {normalized, inj.weights->buffer, inj.scales->buffer, inj.biases->buffer, gates, diagnostic,
       debug.rawBF16 ? debug.rawBF16 : activated, debug.rawF32 ? debug.rawF32 : diagnostic},
      p, {1, rows, 1}, {128, 1, 1});
}
} // namespace splash::flash::candidate
