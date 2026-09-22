// Root-run one-layer primitive qualification. --cpu-self-test creates no device.
#include "flash/FlashInt8ExpertStoreMetadata.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashMoE.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashMoEDirectA.h"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#include "prefill4k_allrows_qmv_one_layer.hpp"
#include "prefill4k_allrows_qmv_probe.h"
#include "prefill4k_allrows_qmv_reference.hpp"
#include "flash/FlashGatheredI8QMV.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace {
using namespace splash::flash;
using namespace splash::metal;
namespace reference = splash::flash::gathered_i8_qmv_reference;
namespace one = splash::flash::qmv_one_layer;
void require(bool value, const char *why) { if (!value) throw std::runtime_error(why); }
uint32_t numeric(const char *raw, uint32_t low, uint32_t high) {
  std::string value(raw); size_t end = 0;
  require(!value.empty() && std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }),
      "invalid unsigned CLI value");
  const auto result = std::stoul(value, &end);
  require(end == value.size() && result >= low && result <= high, "CLI value outside frozen bounds");
  return uint32_t(result);
}
template<class T> std::vector<T> fixture(const std::string &path, uint64_t elements) {
  require(elements && elements * sizeof(T) <= 2 << 20, "fixture exceeds bounded2MiB");
  require(std::filesystem::file_size(path) == elements * sizeof(T), "fixture byte extent differs");
  std::vector<T> values(elements);
  std::ifstream in(path, std::ios::binary); in.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(T));
  require(bool(in) && in.peek() == std::char_traits<char>::eof(), "fixture changed or read failed");
  return values;
}
struct Guard {
  MetalBuffer allocation; uint64_t bytes;
  bool valid() const {
    const auto *data = static_cast<const uint8_t *>(allocation.contents());
    return std::all_of(data, data + 64, [](uint8_t x) { return x == 0x5a; }) &&
        std::all_of(data + bytes + 64, data + bytes + 128, [](uint8_t x) { return x == 0x5a; });
  }
};
MetalBuffer guarded(MetalBackend &backend, uint64_t bytes, std::vector<Guard> &guards) {
  auto base = backend.allocateBuffer(bytes + 128, BufferStorage::Shared, "one layer QMV guarded buffer");
  std::memset(base.contents(), 0x5a, base.sizeBytes());
  auto view = backend.view(base, 64, bytes);
  std::memset(view.contents(), 0xa5, bytes);
  guards.push_back({base, bytes}); return view;
}
template<class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values,
                                     std::vector<Guard> &guards) {
  auto view = guarded(backend, values.size() * sizeof(T), guards);
  std::memcpy(view.contents(), values.data(), view.sizeBytes()); return view;
}
void guardScratch(MetalBackend &backend, FlashMoEBlockedScratch &scratch, std::vector<Guard> &guards) {
  for (auto *b : {&scratch.buckets.counts, &scratch.buckets.offsets, &scratch.buckets.routeMap,
      &scratch.buckets.canonicalToPacked, &scratch.buckets.packedInputs, &scratch.buckets.jobOffsets,
      &scratch.buckets.jobCount, &scratch.buckets.tileJobs, &scratch.packedActivated, &scratch.scatteredDown})
    *b = guarded(backend, b->sizeBytes(), guards);
}
void clear(MetalBuffer diag) { *static_cast<uint32_t *>(diag.contents()) = 0x80000000u; }
uint32_t diagnostic(MetalBuffer diag) { return *static_cast<uint32_t *>(diag.contents()); }
std::string finiteJSON(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out; out << std::setprecision(17) << value; return out.str();
}
void mppChain(CommandGraph &graph, const one::OneLayerPayload &layer,
    MetalBuffer hidden, MetalBuffer ids, const FlashMoEBlockedScratch &s, MetalBuffer diag, uint32_t rows) {
  constexpr auto tile = FlashMoEBlockedTile::M16N64;
  addMoEBlockedPack(graph, hidden, ids, s, diag, rows, tile, 10);
  const uint32_t routes = rows * 10, jobs = moEBucketJobCapacity(rows, 10, 16);
  const FlashInt8ExpertStoreParams p{rows, 10, routes, jobs, 16, 512, 0, 0};
  graph.add("flash_int8_expert_store_gate_up_m16_n64", {s.buckets.packedInputs,
      layer.codes[0], layer.scales[0], layer.codes[1], layer.scales[1], layer.ranks,
      s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount, s.packedActivated, diag},
      p, {10, std::min(routes, jobs), 1}, {128, 1, 1});
  const FlashMoEBlockedDownParams poison{{rows, 10, 640, 2560, 512, 0, 0, 0,
      320, uint64_t{2560} * 320, 20, uint64_t{2560} * 20}, routes, jobs, 16, 0};
  graph.add("flash_moe_blocked_poison_excluded_routes", {s.buckets.canonicalToPacked, s.scatteredDown, diag},
      poison, {10, routes, 1}, {256, 1, 1});
  graph.add("flash_moe_direct_a_prepare_down", {s.packedActivated, s.buckets.offsets, s.packedActivated, diag},
      FlashMoEDirectAPrepareParams{routes, 640, 63, 0}, {routes + 63, 1, 1}, {256, 1, 1});
  graph.add("flash_int8_expert_store_down_scatter_m16_n64", {s.packedActivated,
      layer.codes[2], layer.scales[2], layer.ranks, s.buckets.offsets, s.buckets.tileJobs,
      s.buckets.jobCount, s.buckets.routeMap, s.scatteredDown, diag}, p,
      {40, std::min(routes, jobs), 1}, {128, 1, 1});
}
void qmvChain(CommandGraph &graph, const one::OneLayerPayload &layer,
    MetalBuffer hidden, MetalBuffer ids, MetalBuffer activated, MetalBuffer down,
    MetalBuffer diag, uint32_t rows, uint32_t columns) {
  const std::string c = std::to_string(columns);
  const FlashGatheredI8QMVParams p{rows, 10, 512, 0};
  graph.add("flash_gathered_i8_qmv_gate_up_sg4_c" + c, {hidden,
      layer.codes[0], layer.scales[0], layer.codes[1], layer.scales[1], layer.ranks, ids, activated, diag},
      p, {160 / columns, rows, 10}, {128, 1, 1});
  graph.add("flash_gathered_i8_qmv_down_sg4_c" + c, {activated,
      layer.codes[2], layer.scales[2], layer.ranks, ids, down, diag},
      p, {640 / columns, rows, 10}, {128, 1, 1});
}
struct Outputs { MetalBuffer dot, scaled, bf16, diag; };
Outputs outputs(MetalBackend &b, uint64_t elements, std::vector<Guard> &g) {
  return {guarded(b, elements * 4, g), guarded(b, elements * 4, g), guarded(b, elements * 2, g), guarded(b, 4, g)};
}
struct Summary {
  uint64_t dots = 0, boundFailures = 0, strictSensitive = 0, strictFailures = 0;
  uint64_t signFailures = 0, exceptional = 0, bf16ReferenceMismatches = 0;
  uint32_t maxBF16ULPs = 0; double maxBoundRatio = 0;
  std::vector<std::string> failures;
  bool pass() const { return !boundFailures && !strictFailures && !signFailures && !exceptional; }
  bool finiteBoundSignPass() const { return !boundFailures && !signFailures && !exceptional; }
  void json(std::ostream &o) const {
    o << "{\"dots\":" << dots << ",\"bound_failures\":" << boundFailures
      << ",\"strict_sensitive_dots\":" << strictSensitive << ",\"strict_bf16_failures\":" << strictFailures
      << ",\"sign_failures\":" << signFailures << ",\"exceptional_dots\":" << exceptional
      << ",\"bf16_f64_reference_mismatches\":" << bf16ReferenceMismatches
      << ",\"max_bf16_ulp\":" << maxBF16ULPs << ",\"max_absolute_bound_ratio\":" << finiteJSON(maxBoundRatio)
      << ",\"regular_finite_primitive_pass\":" << (pass() ? "true" : "false") << ",\"first_failures\":[";
    for (size_t i = 0; i < failures.size(); ++i) { if (i) o << ','; o << failures[i]; } o << "]}";
  }
};
// A separate diagnostic replay policy; the frozen primitive checker is unchanged.
bool timingPolicy(bool allowBaselineStrictFailure, bool candidatesStrictPass,
                  bool baselineStrictPass, bool baselineFiniteBoundSignPass,
                  bool commonChecksPass) {
  return candidatesStrictPass && commonChecksPass &&
      (baselineStrictPass || (allowBaselineStrictFailure && baselineFiniteBoundSignPass));
}
Summary evaluate(const one::OneLayerPayload &layer, uint32_t plane, uint32_t rows,
    const std::vector<int64_t> &ids, MetalBuffer input, const Outputs &capture, uint32_t variant) {
  const uint32_t n = plane == 2 ? 2560 : 640, k = plane == 2 ? 640 : 2560;
  const auto *x = static_cast<const uint16_t *>(input.contents());
  const auto *codes = static_cast<const int8_t *>(layer.codes[plane].contents());
  const auto *scales = static_cast<const float *>(layer.scales[plane].contents());
  const auto *ranks = static_cast<const uint32_t *>(layer.ranks.contents());
  const auto *dots = static_cast<const float *>(capture.dot.contents()), *scaled = static_cast<const float *>(capture.scaled.contents());
  const auto *bf = static_cast<const uint16_t *>(capture.bf16.contents());
  Summary result;
  for (uint32_t route = 0; route < rows * 10; ++route) {
    require(ids[route] >= 0 && ids[route] < 512, "reference input IDs malformed");
    const uint32_t rank = ranks[ids[route]];
    for (uint32_t column = 0; column < n; ++column) {
      const uint64_t index = uint64_t{route} * n + column, row = uint64_t{rank} * n + column;
      const auto ref = reference::reference({x + uint64_t{plane == 2 ? route : route / 10} * k, k},
          {codes + row * k, k}, scales[row], 32, variant == 0);
      const auto report = reference::assess(ref, dots[index], scaled[index], bf[index], diagnostic(capture.diag));
      ++result.dots; result.boundFailures += !report.dotWithinBound || !report.scaledWithinBound;
      result.strictSensitive += report.strictSensitive;
      result.strictFailures += report.strictSensitive && !report.strictBF16Pass;
      result.signFailures += !report.signMatches; result.exceptional += ref.exceptional;
      result.bf16ReferenceMismatches += !report.bf16Exact;
      result.maxBF16ULPs = std::max(result.maxBF16ULPs, report.bf16ULPs);
      if (ref.dotAbsoluteBound) result.maxBoundRatio = std::max(result.maxBoundRatio, report.dotAbsoluteError / ref.dotAbsoluteBound);
      if ((!report.regularFinitePrimitivePass || !report.signMatches) && result.failures.size() < 16) {
        std::ostringstream f; f << std::setprecision(17) << "{\"route\":" << route << ",\"column\":" << column
          << ",\"reference_dot\":" << finiteJSON(ref.dot) << ",\"observed_dot\":" << finiteJSON(dots[index])
          << ",\"dot_abs_bound\":" << finiteJSON(ref.dotAbsoluteBound) << ",\"reference_scaled\":" << finiteJSON(ref.scaled)
          << ",\"observed_scaled\":" << finiteJSON(scaled[index]) << ",\"scaled_abs_bound\":" << finiteJSON(ref.scaledAbsoluteBound)
          << ",\"reference_bf16\":" << ref.bf16 << ",\"observed_bf16\":" << bf[index]
          << ",\"strict_sensitive\":" << (report.strictSensitive ? "true" : "false")
          << ",\"finite_capture\":" << (report.finiteCapture ? "true" : "false") << '}';
        result.failures.push_back(f.str());
      }
    }
  }
  return result;
}
void cpuTest() {
  require(sizeof(FlashQMVProbeParams) == 32 && sizeof(FlashGatheredI8QMVParams) == 16,
      "probe/shared ABI differs");
  require(gathered_i8_qmv::geometry(16).expertDownBytes == 819200, "bounded geometry differs");
  std::vector<uint16_t> x(640, 0); std::vector<int8_t> c(640, 1);
  x[0] = reference::bf16FromF64(0x1p100); x[1] = 0x3f80; x[2] = reference::bf16FromF64(-0x1p100);
  auto ref = reference::reference(x, c, 1); auto report = reference::assess(ref, 0, 0, 0, 0);
  require(report.dotWithinBound && !report.regularFinitePrimitivePass, "broad bound bypassed strict cancellation gate");
  require(!timingPolicy(false, true, false, true, true), "default timed baseline strict failure");
  require(timingPolicy(true, true, false, true, true), "explicit diagnostic-only comparison was blocked");
  require(!timingPolicy(true, false, false, true, true), "diagnostic option relaxed candidate numerical gate");
  require(!timingPolicy(true, true, false, false, true), "diagnostic option relaxed baseline bound/sign gate");
  require(!timingPolicy(true, true, false, true, false), "diagnostic option relaxed guards/stages/parity gate");
  require(timingPolicy(false, true, true, true, true), "default blocked fully qualified primitive");
  Summary baseline; baseline.strictFailures = 1;
  const auto retainedFailures = baseline.strictFailures;
  require(!baseline.pass() && baseline.finiteBoundSignPass() &&
      timingPolicy(true, true, baseline.pass(), baseline.finiteBoundSignPass(), true) &&
      baseline.strictFailures == retainedFailures && !baseline.pass(), "diagnostic replay erased baseline strict failure");
  std::cout << "{\"cpu_checks\":\"passed\",\"gpu_work\":false,\"one_layer_only\":true,\"numerical_qualification\":\"pending\"}\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") { cpuTest(); return 0; }
    require(argc >= 7 && std::string_view(argv[1]) == "--gpu", "usage: oracle --gpu metallib fullStore layer rows report [--hidden bf16] [--ids i64] [--pairs1..9] [--allow-baseline-strict-failure]");
    const uint32_t layerIndex = numeric(argv[4], 0, 47), rows = numeric(argv[5], 1, 16);
    const std::filesystem::path output(argv[6]);
    std::string hiddenPath, idsPath; uint32_t pairs = 3; bool allowBaselineStrictFailure = false;
    for (int i = 7; i < argc; ++i) {
      const std::string option(argv[i]);
      if (option == "--allow-baseline-strict-failure") {
        require(!allowBaselineStrictFailure, "duplicate diagnostic timing option");
        allowBaselineStrictFailure = true; continue;
      }
      require(i + 1 < argc, "CLI option missing value");
      if (option == "--hidden") hiddenPath = argv[i + 1]; else if (option == "--ids") idsPath = argv[i + 1];
      else if (option == "--pairs") pairs = numeric(argv[i + 1], 1, 9); else throw std::invalid_argument("unknown oracle option");
      ++i;
    }
    std::vector<uint16_t> hidden(uint64_t{rows} * 2560); std::vector<int64_t> ids(uint64_t{rows} * 10);
    for (uint32_t row = 0; row < rows; ++row) {
      double squares = 0;
      for (uint32_t k = 0; k < 2560; ++k) { const double x = double(int32_t((k * 173 + row * 31) % 211) - 105); squares += x * x; }
      const double normalization = std::sqrt(squares / 2560);
      for (uint32_t k = 0; k < 2560; ++k) hidden[row * 2560 + k] = reference::bf16FromF64(double(int32_t((k * 173 + row * 31) % 211) - 105) / normalization);
      for (uint32_t slot = 0; slot < 10; ++slot) ids[row * 10 + slot] = (slot * 53 + (row % 2 ? row * 73 : 0)) % 512;
    }
    if (!hiddenPath.empty()) hidden = fixture<uint16_t>(hiddenPath, hidden.size());
    if (!idsPath.empty()) ids = fixture<int64_t>(idsPath, ids.size());
    require(gathered_i8_qmv::fixtureIDDiagnostics(ids, rows) == 0, "valid unique I64 IDs required for finite primitive gate");
    const auto metadata = loadFlashInt8ExpertStoreMetadata(argv[3],
        "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
        "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0", NormConvention::OnePlusWeight);
    require(metadata.identitySha256 == "ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1", "Full512 certificate differs");
    ::setenv("SPLASH_FLASH_MOE_Q4X8", "1", 1); ::setenv("SPLASH_FLASH_MOE_DIRECT_A", "1", 1);
    @autoreleasepool {
      MetalBackend backend(argv[2]);
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory, reserve = 4ULL << 30;
      require(physical > reserve, "physical memory cannot protect host reserve");
      splash::engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      // One readonly layer plus generous bounded scratch allowance; no source model.
      const uint64_t planned = one::oneLayerPlannedBytes(metadata.layers[layerIndex]) + (128ULL << 20);
      auto reservation = governor.tryReserve(planned); require(bool(reservation), "one-layer primitive memory admission refused");
      const uint64_t before = backend.memoryStats().allocatedBytes;
      auto layer = one::OneLayerPayload::load(backend, metadata, layerIndex);
      const auto immutableHashes = layer.immutableHashes();
      std::vector<Guard> guards;
      auto x = upload(backend, hidden, guards), idBuffer = upload(backend, ids, guards);
      auto scratch = allocateMoEBlockedScratch(backend, rows, 10); guardScratch(backend, scratch, guards);
      std::array<MetalBuffer, 3> diag{guarded(backend, 4, guards), guarded(backend, 4, guards), guarded(backend, 4, guards)};
      std::array<MetalBuffer, 2> act{guarded(backend, uint64_t{rows} * 10 * 640 * 2, guards), guarded(backend, uint64_t{rows} * 10 * 640 * 2, guards)};
      std::array<MetalBuffer, 2> down{guarded(backend, uint64_t{rows} * 10 * 2560 * 2, guards), guarded(backend, uint64_t{rows} * 10 * 2560 * 2, guards)};
      auto canonicalActivation = guarded(backend, uint64_t{rows} * 10 * 640 * 2, guards);
      std::array<CommandGraph, 3> chain;
      mppChain(chain[0], layer, x, idBuffer, scratch, diag[0], rows);
      qmvChain(chain[1], layer, x, idBuffer, act[0], down[0], diag[1], rows, 1);
      qmvChain(chain[2], layer, x, idBuffer, act[1], down[1], diag[2], rows, 2);
      clear(diag[0]); (void)backend.submitCommand(chain[0].dispatches());
      CommandGraph unpack;
      unpack.add("flash_qmv_probe_unpack_activation", {scratch.packedActivated, scratch.buckets.canonicalToPacked, canonicalActivation, diag[0]},
          FlashQMVProbeParams{rows,10,2560,640,0,0,512,0}, {3,rows * 10,1}, {256,1,1});
      (void)backend.submitCommand(unpack.dispatches());
      std::array<std::array<Outputs, 3>, 3> captures;
      std::array<std::array<Summary, 3>, 3> summaries;
      bool finiteGate = true, commonChecksPass = true, candidatesStrictPass = true;
      bool baselineStrictPass = true, baselineFiniteBoundSignPass = true;
      for (uint32_t plane = 0; plane < 3; ++plane) {
        const uint32_t n = plane == 2 ? 2560 : 640, k = plane == 2 ? 640 : 2560;
        const auto sourceInput = plane == 2 ? canonicalActivation : x;
        for (uint32_t variant = 0; variant < 3; ++variant) {
          auto &c = captures[plane][variant]; c = outputs(backend, uint64_t{rows} * 10 * n, guards); clear(c.diag);
          CommandGraph probe;
          const FlashQMVProbeParams p{rows,10,k,n,uint32_t(plane == 2),variant,512,0};
          if (variant == 0)
            probe.add("flash_qmv_probe_mpp_m16", {plane == 2 ? scratch.packedActivated : scratch.buckets.packedInputs,
                layer.codes[plane],layer.scales[plane],layer.ranks,scratch.buckets.offsets,
                scratch.buckets.tileJobs,scratch.buckets.jobCount,scratch.buckets.routeMap,
                c.dot,c.scaled,c.bf16,c.diag}, p, {n/64, rows*10, 1}, {128,1,1});
          else probe.add("flash_qmv_probe_c" + std::to_string(variant),
              {sourceInput,layer.codes[plane],layer.scales[plane],layer.ranks,idBuffer,c.dot,c.scaled,c.bf16,c.diag},
              p, {n/(4*variant),rows,10}, {128,1,1});
          (void)backend.submitCommand(probe.dispatches());
          summaries[plane][variant] = evaluate(layer, plane, rows, ids, sourceInput, c, variant);
          finiteGate &= summaries[plane][variant].pass();
          if (variant == 0) {
            baselineStrictPass &= summaries[plane][variant].pass();
            baselineFiniteBoundSignPass &= summaries[plane][variant].finiteBoundSignPass();
          } else candidatesStrictPass &= summaries[plane][variant].pass();
          commonChecksPass &= diagnostic(c.diag) == 0x80000000u;
        }
      }
      uint64_t c1c2DotMismatches = 0, c1c2ScaledMismatches = 0, c1c2BF16Mismatches = 0;
      for (uint32_t plane = 0; plane < 3; ++plane) {
        const uint64_t elements = uint64_t{rows} * 10 * (plane == 2 ? 2560 : 640);
        const auto &a = captures[plane][1], &b = captures[plane][2];
        for (uint64_t i = 0; i < elements; ++i) {
          c1c2DotMismatches += static_cast<const uint32_t *>(a.dot.contents())[i] != static_cast<const uint32_t *>(b.dot.contents())[i];
          c1c2ScaledMismatches += static_cast<const uint32_t *>(a.scaled.contents())[i] != static_cast<const uint32_t *>(b.scaled.contents())[i];
          c1c2BF16Mismatches += static_cast<const uint16_t *>(a.bf16.contents())[i] != static_cast<const uint16_t *>(b.bf16.contents())[i];
        }
      }
      finiteGate &= !c1c2DotMismatches && !c1c2ScaledMismatches && !c1c2BF16Mismatches;
      commonChecksPass &= !c1c2DotMismatches && !c1c2ScaledMismatches && !c1c2BF16Mismatches;
      // The shipping fused gate/up must match the qualified standalone BF16
      // SwiGLU stage on its tapped rounded projections; CPU exp is no oracle.
      std::array<uint64_t, 3> swigluStageMismatches{};
      std::array<MetalBuffer, 3> expectedAct;
      for (uint32_t variant = 0; variant < 3; ++variant) {
        clear(diag[variant]); (void)backend.submitCommand(chain[variant].dispatches());
        expectedAct[variant] = guarded(backend, uint64_t{rows} * 10 * 640 * 2, guards);
        CommandGraph swiglu;
        addSiLUMultiply(swiglu, captures[0][variant].bf16, captures[1][variant].bf16,
            expectedAct[variant], diag[variant], rows, 640, 10);
        (void)backend.submitCommand(swiglu.dispatches());
        const auto actual = variant ? act[variant - 1] : canonicalActivation;
        for (uint64_t i = 0; i < uint64_t{rows} * 10 * 640; ++i)
          swigluStageMismatches[variant] += static_cast<const uint16_t *>(actual.contents())[i] !=
              static_cast<const uint16_t *>(expectedAct[variant].contents())[i];
        finiteGate &= swigluStageMismatches[variant] == 0 && diagnostic(diag[variant]) == 0x80000000u;
        commonChecksPass &= swigluStageMismatches[variant] == 0 && diagnostic(diag[variant]) == 0x80000000u;
      }
      uint64_t c1c2ActivatedMismatches = 0, c1c2DownMismatches = 0;
      for (uint64_t i = 0; i < uint64_t{rows} * 10 * 640; ++i)
        c1c2ActivatedMismatches += static_cast<const uint16_t *>(act[0].contents())[i] != static_cast<const uint16_t *>(act[1].contents())[i];
      for (uint64_t i = 0; i < uint64_t{rows} * 10 * 2560; ++i)
        c1c2DownMismatches += static_cast<const uint16_t *>(down[0].contents())[i] != static_cast<const uint16_t *>(down[1].contents())[i];
      finiteGate &= !c1c2ActivatedMismatches && !c1c2DownMismatches;
      commonChecksPass &= !c1c2ActivatedMismatches && !c1c2DownMismatches;
      bool canaries = std::all_of(guards.begin(), guards.end(), [](const Guard &g) { return g.valid(); });
      bool inputsImmutable = !std::memcmp(x.contents(), hidden.data(), x.sizeBytes()) && !std::memcmp(idBuffer.contents(), ids.data(), idBuffer.sizeBytes());
      finiteGate &= canaries && inputsImmutable;
      commonChecksPass &= canaries && inputsImmutable;
      finiteGate &= commonChecksPass;
      const uint64_t after = backend.memoryStats().allocatedBytes;
      require(after >= before && after - before <= planned, "one-layer oracle exceeded its reservation");
      reservation->commit();
      std::array<std::vector<CommandTiming>, 3> timings;
      bool timingAllowed = timingPolicy(allowBaselineStrictFailure, candidatesStrictPass,
          baselineStrictPass, baselineFiniteBoundSignPass, commonChecksPass);
      const bool diagnosticTimingUsed = timingAllowed && !baselineStrictPass;
      bool replayChecksPass = true;
      // Numerical gate precedes timed replay. Never select tolerance after result.
      if (timingAllowed) {
        for (uint32_t variant = 0; variant < 3; ++variant) {
          clear(diag[variant]); (void)backend.submitCommand(chain[variant].dispatches());
          replayChecksPass &= diagnostic(diag[variant]) == 0x80000000u;
        }
        for (uint32_t pair = 0; pair < pairs; ++pair)
          for (uint32_t step = 0; step < 3; ++step) {
            const uint32_t variant = (pair + step) % 3; clear(diag[variant]);
            timings[variant].push_back(backend.submitCommand(chain[variant].dispatches()));
            replayChecksPass &= diagnostic(diag[variant]) == 0x80000000u;
          }
      }
      std::array<uint64_t, 3> postReplayStageMismatches{};
      uint64_t postReplayC1C2DownMismatches = 0;
      const auto *inverse = static_cast<const uint32_t *>(scratch.buckets.canonicalToPacked.contents());
      const auto *packedAct = static_cast<const uint16_t *>(scratch.packedActivated.contents());
      for (uint32_t route = 0; route < rows * 10; ++route) {
        const uint32_t packed = inverse[route];
        if (packed >= rows * 10) { replayChecksPass = false; continue; }
        for (uint32_t column = 0; column < 640; ++column) {
          const uint64_t index = uint64_t{route} * 640 + column;
          postReplayStageMismatches[0] += packedAct[uint64_t{packed} * 640 + column] != static_cast<const uint16_t *>(expectedAct[0].contents())[index];
          for (uint32_t variant = 1; variant < 3; ++variant)
            postReplayStageMismatches[variant] += static_cast<const uint16_t *>(act[variant - 1].contents())[index] != static_cast<const uint16_t *>(expectedAct[variant].contents())[index];
        }
      }
      for (uint64_t i = 0; i < uint64_t{rows} * 10 * 2560; ++i)
        postReplayC1C2DownMismatches += static_cast<const uint16_t *>(down[0].contents())[i] != static_cast<const uint16_t *>(down[1].contents())[i];
      replayChecksPass &= !postReplayStageMismatches[0] && !postReplayStageMismatches[1] &&
          !postReplayStageMismatches[2] && !postReplayC1C2DownMismatches;
      canaries &= std::all_of(guards.begin(), guards.end(), [](const Guard &g) { return g.valid(); });
      inputsImmutable &= !std::memcmp(x.contents(), hidden.data(), x.sizeBytes()) && !std::memcmp(idBuffer.contents(), ids.data(), idBuffer.sizeBytes());
      finiteGate &= canaries && inputsImmutable && replayChecksPass;
      commonChecksPass &= canaries && inputsImmutable && replayChecksPass;
      timingAllowed &= commonChecksPass;
      layer.checkImmutableHashes(immutableHashes);
      std::ofstream report(output); require(bool(report), "cannot create oracle report"); report << std::setprecision(17);
      report << "{\"schema\":\"splash-private-one-layer-qmv-primitive-v1\",\"primitive_pass\":" << (finiteGate ? "true" : "false")
        << ",\"allow_baseline_strict_failure_requested\":" << (allowBaselineStrictFailure ? "true" : "false")
        << ",\"baseline_strict_pass\":" << (baselineStrictPass ? "true" : "false")
        << ",\"baseline_finite_bound_sign_pass\":" << (baselineFiniteBoundSignPass ? "true" : "false")
        << ",\"candidates_strict_pass\":" << (candidatesStrictPass ? "true" : "false")
        << ",\"common_checks_pass\":" << (commonChecksPass ? "true" : "false")
        << ",\"post_replay_checks_pass\":" << (replayChecksPass ? "true" : "false")
        << ",\"post_replay_stage_bf16_mismatches_mpp_c1_c2\":[" << postReplayStageMismatches[0] << ',' << postReplayStageMismatches[1] << ',' << postReplayStageMismatches[2] << ']'
        << ",\"post_replay_c1_c2_down_bf16_mismatches\":" << postReplayC1C2DownMismatches
        << ",\"timing_allowed\":" << (timingAllowed ? "true" : "false")
        << ",\"diagnostic_timing_used\":" << (diagnosticTimingUsed ? "true" : "false")
        << ",\"diagnostic_timing_pass\":" << (diagnosticTimingUsed && timingAllowed ? "true" : "false")
        << ",\"timing_policy\":" << splash::json::quote(diagnosticTimingUsed ? "explicit baseline-strict-failure diagnostic comparison; baseline has not passed strict F64/BF16 gate" : "default fail-closed all-variant primitive policy")
        << ",\"model_quality_qualified\":false,\"real_normalized_decode_activation_qualified\":false,\"layer\":" << layerIndex << ",\"rows\":" << rows
        << ",\"mapped_layer_bytes\":" << metadata.layers[layerIndex].bytes << ",\"one_shared_readonly_layer\":true,\"source_manifest_sha256\":" << splash::json::quote(metadata.sourceManifestSha256)
        << ",\"full512_identity\":" << splash::json::quote(metadata.identitySha256)
        << ",\"layer_payload_sha256\":" << splash::json::quote(metadata.layers[layerIndex].sha256)
        << ",\"hidden_sha256\":" << splash::json::quote(one::detail::hash(hidden.data(), hidden.size() * 2))
        << ",\"ids_sha256\":" << splash::json::quote(one::detail::hash(ids.data(), ids.size() * 8))
        << ",\"hidden_policy\":" << splash::json::quote(hiddenPath.empty() ? "synthetic normalized BF16; source-backed immutable Full512 weights" : "caller-provided BF16 fixture; provenance not attested by oracle")
        << ",\"ids_policy\":" << splash::json::quote(idsPath.empty() ? "synthetic unique concentrated/spread original I64 IDs" : "caller-provided I64 fixture")
        << ",\"bound_policy\":\"compensated F64 approximate dot with explicitF64 uncertainty; sumAbs products; QMV gamma(ceilK32+31); opaqueMPPcontrol separateconservative gamma(K); plus product error and finalscale FP32 rounding; strictnearzero BF16/sign gate; exceptional neverautoqualify\""
        << ",\"input_immutable\":" << (inputsImmutable ? "true" : "false") << ",\"canaries\":" << (canaries ? "true" : "false")
        << ",\"weights_immutable\":true,\"c1_c2_fp32_dot_mismatches\":" << c1c2DotMismatches << ",\"c1_c2_scaled_fp32_mismatches\":" << c1c2ScaledMismatches
        << ",\"c1_c2_bf16_mismatches\":" << c1c2BF16Mismatches
        << ",\"swiglu_stage_bf16_mismatches_mpp_c1_c2\":[" << swigluStageMismatches[0] << ',' << swigluStageMismatches[1] << ',' << swigluStageMismatches[2] << ']'
        << ",\"c1_c2_activated_bf16_mismatches\":" << c1c2ActivatedMismatches
        << ",\"c1_c2_chain_down_bf16_mismatches\":" << c1c2DownMismatches << ",\"projection_reports\":[";
      for (uint32_t plane = 0; plane < 3; ++plane) { if (plane) report << ','; report << "{\"plane\":" << plane << ",\"variants\":[";
        for (uint32_t v = 0; v < 3; ++v) { if (v) report << ','; report << "{\"variant\":" << splash::json::quote(v ? v == 1 ? "C1" : "C2" : "MPP16") << ",\"metrics\":"; summaries[plane][v].json(report); report << '}'; } report << "]}"; }
      report << "],\"timing_scope\":\"complete gateUp+SwiGLU+down; MPP includes bucket prelude and downprepare; primitive probes/F64 checks/hashes excluded; warm alternating matched replays only after selected explicit timingpolicy; diagnostic baseline timing does not assert strict F64 parity\",\"timings\":[";
      for (uint32_t v = 0; v < 3; ++v) { if (v) report << ','; report << "{\"variant\":" << splash::json::quote(v ? v == 1 ? "C1" : "C2" : "MPP16") << ",\"samples\":[";
        for (size_t i = 0; i < timings[v].size(); ++i) { if (i) report << ','; report << "{\"wall_ms\":" << timings[v][i].wallSeconds * 1000 << ",\"gpu_ms\":" << timings[v][i].gpuSeconds * 1000 << '}'; } report << "]}"; }
      report << "]}\n"; backend.stop();
      std::cout << "{\"primitive_pass\":" << (finiteGate ? "true" : "false") << ",\"timing_allowed\":" << (timingAllowed ? "true" : "false")
        << ",\"diagnostic_timing_pass\":" << (diagnosticTimingUsed && timingAllowed ? "true" : "false")
        << ",\"model_quality_qualified\":false,\"report\":" << splash::json::quote(output.string()) << "}\n";
      return timingAllowed ? 0 : 2;
    }
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
