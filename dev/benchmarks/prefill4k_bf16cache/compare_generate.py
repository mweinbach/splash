#!/usr/bin/env python3
"""Build/plan a private persisted-INT8 Top128 versus lossless-BF16 comparator.

Generation, planning and --cpu-self-test submit no GPU work. Root alone uses
``run --run`` after arranging the shared GPU/memory slot.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUILD = Path("build/prefill4k-bf16-int8-compare")
DEFAULT_PARENT = Path("build/prefill4k-bf16cache")
DEFAULT_STORE = Path("install/local-models/Flash-Next-int8-experts-top128-v1")
PLAN = Path("build/release/flash/hot-expert-plan128.json")


def replacement(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError(f"BF16/INT8 comparator source drift: {before!r}")
    return source.replace(before, after)


SUPPORT = r'''
uint64_t casePlannedBytes(uint32_t rows) {
  const auto rounded = [](uint64_t bytes) { return (bytes + 16383) & ~uint64_t(16383); };
  const uint64_t routes = uint64_t(rows) * kSelections;
  uint64_t bytes = 2 * flashMoEBlockedWorkspacePlannedBytes(rows, kSelections);
  for (const uint64_t extent : {uint64_t(rows) * 2560 * 2, routes * 8, routes * 2,
      uint64_t(rows) * 2560 * 2, uint64_t(rows) * 2, uint64_t(4)}) bytes += rounded(extent);
  // Original scratch outputs may coexist briefly with guarded replacements.
  // Admit both sets and each independently rounded output canary.
  for (const uint64_t extent : {(routes + 63) * 640 * 2 + kGuardBytes,
      routes * 2560 * 2 + kGuardBytes, uint64_t(rows) * 2560 * 2 + kGuardBytes})
    bytes += 2 * rounded(extent);
  return bytes;
}
uint64_t exactMissSlices(const std::array<FlashMoEBlockedScratch, 2> &scratch,
    const ref::Packed &packed, std::span<const int64_t> ids, std::span<const uint32_t> hot) {
  uint64_t misses = 0;
  for (uint32_t expert = 0; expert < 512; ++expert) {
    if (std::binary_search(hot.begin(), hot.end(), expert)) continue;
    for (uint32_t row = packed.offsets[expert]; row < packed.offsets[expert + 1]; ++row) {
      const uint64_t offset = uint64_t(row) * 640 * 2;
      require(std::memcmp(static_cast<const uint8_t *>(scratch[0].packedActivated.contents()) + offset,
          static_cast<const uint8_t *>(scratch[1].packedActivated.contents()) + offset, 640 * 2) == 0,
          "shared Direct-A Q4 miss activation differs between cache variants");
    }
  }
  for (uint32_t route = 0; route < ids.size(); ++route) {
    if (std::binary_search(hot.begin(), hot.end(), uint32_t(ids[route]))) continue;
    const uint64_t offset = uint64_t(route) * 2560 * 2;
    require(std::memcmp(static_cast<const uint8_t *>(scratch[0].scatteredDown.contents()) + offset,
        static_cast<const uint8_t *>(scratch[1].scatteredDown.contents()) + offset, 2560 * 2) == 0,
        "shared Direct-A Q4 miss down scatter differs between cache variants");
    ++misses;
  }
  return misses;
}
void checkStoreRanks(const FlashInt8ExpertStore &store, const FlashInt8ExpertStoreMetadata &metadata,
    std::span<const MetalBuffer> buffers) {
  require(buffers.size() == 96, "store immutable base/rank list differs");
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const auto hot = store.selectedExpertIDs(layer);
    const auto &expected = metadata.layers[layer].selectedIDs;
    require(std::equal(hot.begin(), hot.end(), expected.begin(), expected.end()), "store selected IDs differ");
    require(buffers[layer * 2].sizeBytes() == metadata.layers[layer].bytes, "mapped layer extent differs");
    const auto &rankBuffer = buffers[layer * 2 + 1];
    require(rankBuffer.sizeBytes() >= 512 * 4, "compact rank map extent differs");
    const auto *ranks = static_cast<const uint32_t *>(rankBuffer.contents());
    for (uint32_t expert = 0; expert < 512; ++expert) {
      const auto it = std::lower_bound(hot.begin(), hot.end(), expert);
      const uint32_t rank = it != hot.end() && *it == expert ? uint32_t(it - hot.begin()) : UINT32_MAX;
      require(ranks[expert] == rank, "original expert ID-to-compact-rank map differs");
    }
    const auto *padding = static_cast<const uint8_t *>(rankBuffer.contents()) + 512 * 4;
    require(std::all_of(padding, static_cast<const uint8_t *>(rankBuffer.contents()) + rankBuffer.sizeBytes(),
        [](uint8_t value) { return value == 0xff; }), "immutable rank padding differs");
  }
}
std::string digestBuffer(const MetalBuffer &buffer, uint64_t bytes = 0) {
  if (!bytes) bytes = buffer.sizeBytes();
  require(buffer.contents() && bytes <= buffer.sizeBytes(), "SHA256 buffer extent differs");
  CC_SHA256_CTX context{};
  require(CC_SHA256_Init(&context), "SHA256 initialization failed");
  const auto *cursor = static_cast<const uint8_t *>(buffer.contents());
  while (bytes) {
    const CC_LONG chunk = CC_LONG(std::min<uint64_t>(bytes, 1ULL << 30));
    require(CC_SHA256_Update(&context, cursor, chunk), "SHA256 update failed");
    cursor += chunk; bytes -= chunk;
  }
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  require(CC_SHA256_Final(digest.data(), &context), "SHA256 finalization failed");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (uint8_t byte : digest) { result += hex[byte >> 4]; result += hex[byte & 15]; }
  return result;
}
void names(std::ostream &out, std::span<const ComputeDispatch> dispatches) {
  out << '[';
  for (size_t i = 0; i < dispatches.size(); ++i) {
    if (i) out << ',';
    out << splash::json::quote(dispatches[i].pipelineName);
  }
  out << ']';
}
void profile(std::ostream &out, const std::vector<splash::metal::DispatchTiming> &values) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << "{\"pipeline\":" << splash::json::quote(values[i].pipelineName)
        << ",\"gpu_ms\":" << values[i].gpuSeconds * 1000 << '}';
  }
  out << ']';
}
struct Replay final {
  std::array<CommandGraph, 2> graphs;
  std::array<MetalBuffer, 2> activated, down, output;
  std::array<std::array<std::string, 3>, 2> expectedHashes;
  MetalBuffer diagnostic;
  std::vector<Guard> guards;
  uint32_t rows = 0;
  void run(MetalBackend &backend) const {
    for (uint32_t i = 0; i < 2; ++i) {
      // A missing producer must not pass by leaving a previous output intact.
      std::memset(activated[i].contents(), 0xa5, activated[i].sizeBytes());
      std::memset(down[i].contents(), 0xa5, down[i].sizeBytes());
      std::memset(output[i].contents(), 0xa5, output[i].sizeBytes());
      (void)backend.submitCommand(graphs[i].dispatches());
      require(digestBuffer(activated[i], uint64_t(rows) * kSelections * 640 * 2) == expectedHashes[i][0] &&
          digestBuffer(down[i], uint64_t(rows) * kSelections * 2560 * 2) == expectedHashes[i][1] &&
          digestBuffer(output[i], uint64_t(rows) * 2560 * 2) == expectedHashes[i][2],
          "retained graph output differs from its own variant after owners were destroyed");
      require(*static_cast<const uint32_t *>(diagnostic.contents()) == kSticky,
          "retained graph diagnostic differs");
      for (const auto &guard : guards) require(guard.clean(), "retained graph canary changed");
    }
  }
};
'''

MAIN = r'''
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      cpuSelfTest();
      if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") {
        std::cout << "{\"pass\":true,\"gpu_work\":false,\"checks\":[\"Q4_coefficients\",\"BF16_RNE\",\"job_bounds\"]}\n";
        return 0;
      }
      require(argc == 5, "usage: bf16-int8-compare COMBINED_METALLIB PACKAGE INT8_TOP128_STORE REPORT_JSON");
      require(!std::getenv("FLASH_EXPERT_CACHE_REQUIRE_EXACT"),
          "cross-variant exact-output policy is invalid for persisted INT8 requantization");
      require(!std::filesystem::exists(argv[4]), "choose a fresh report path");
      const uint32_t pairs = envNumber("FLASH_EXPERT_CACHE_PAIRS", 4, 32);
      const uint32_t rows = envNumber("FLASH_EXPERT_CACHE_ROWS", 2048, 8192);
      const uint32_t m = envNumber("FLASH_EXPERT_CACHE_TILE", 32, 64);
      require(m == 16 || m == 32 || m == 64, "tile must be16/32/64");
      require(m != 64 || rows >= 1024, "M64 policy requires at least1024 rows");
      const char *selectedPrefix = std::getenv("FLASH_EXPERT_CACHE_PREFIX");
      const std::string prefix = selectedPrefix ? selectedPrefix : "language_model.model.layers.0.mlp.switch_mlp";
      constexpr std::string_view head = "language_model.model.layers.", tail = ".mlp.switch_mlp";
      require(prefix.starts_with(head) && prefix.ends_with(tail), "selected prefix is outside trunk MoE");
      const std::string field = prefix.substr(head.size(), prefix.size() - head.size() - tail.size());
      size_t consumed = 0; const auto parsedLayer = std::stoul(field, &consumed);
      require(consumed == field.size() && parsedLayer < 48, "selected layer differs");
      const uint32_t layer = uint32_t(parsedLayer);
      require(setenv("SPLASH_FLASH_MOE_Q4X8", "1", 1) == 0 &&
          setenv("SPLASH_FLASH_MOE_DIRECT_A", "1", 1) == 0 &&
          setenv("SPLASH_FLASH_MOE_M64", "1", 1) == 0,
          "cannot force matched Direct-A control policy");
      const std::string pipelineInfo = pipelineMetadata(argv[1]);
      MetalBackend backend(argv[1]);
      auto weights = FlashWeights::load(backend, argv[2]);
      const auto storeMetadata = loadFlashInt8ExpertStoreMetadata(argv[3], weights.sourceIdentity(),
          weights.manifestFingerprint(), weights.normConvention());
      for (const auto &entry : storeMetadata.layers)
        require(entry.selectedIDs.size() == 128, "comparator requires current persisted Top128 inventory");
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      require(physical > reserve, "insufficient physical RAM");
      splash::engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      const uint64_t storePlanned = FlashInt8ExpertStore::plannedBytes(weights, argv[3]);
      auto storeReservation = governor.tryReserve(storePlanned);
      require(bool(storeReservation), "real governor denied INT8 Top128 BEFORE readonly mapping");
      auto store = std::make_unique<FlashInt8ExpertStore>(backend, weights, argv[3]);
      require(store->actualAllocatedBytes() <= storePlanned && store->mappedBytes() == storeMetadata.totalBytes,
          "persisted INT8 mapped/accounted bytes differ from admitted plan");
      storeReservation->commit();
      std::vector<uint32_t> hotIDs;
      const char *raw = std::getenv("FLASH_EXPERT_CACHE_HOT_IDS");
      require(raw && *raw, "provide actual frequency-ranked Top128 original expert IDs");
      std::stringstream parser(raw); std::string token;
      while (std::getline(parser, token, ',')) {
        size_t used = 0; const auto id = std::stoul(token, &used);
        require(used == token.size() && id < 512, "invalid fixed hot ID list");
        hotIDs.push_back(uint32_t(id));
      }
      const auto storedIDs = store->selectedExpertIDs(layer);
      require(hotIDs.size() == 128 && std::equal(hotIDs.begin(), hotIDs.end(), storedIDs.begin(), storedIDs.end()),
          "BF16 cache and current INT8 Top128 original expert IDs MUST be identical");
      const uint64_t bf16Planned = FlashExpertDenseCache::plannedBytes(weights, prefix, hotIDs);
      auto bf16Reservation = governor.tryReserve(bf16Planned);
      require(bool(bf16Reservation), "real governor denied BF16 Hot128 BEFORE conversion");
      auto cache = std::make_unique<FlashExpertDenseCache>(backend, weights, prefix, hotIDs);
      require(cache->actualAllocatedBytes() <= bf16Planned, "BF16 cache exceeds admitted plan");
      bf16Reservation->commit();
      const auto hashes = exactCoefficients(weights, prefix, *cache);
      const auto bf16RanksHash = digestBuffer(cache->expertRanks());
      const auto *bf16Ranks = static_cast<const uint32_t *>(cache->expertRanks().contents());
      for (uint32_t expert = 0; expert < 512; ++expert) {
        const auto found = std::lower_bound(hotIDs.begin(), hotIDs.end(), expert);
        const uint32_t rank = found != hotIDs.end() && *found == expert ? uint32_t(found - hotIDs.begin()) : UINT32_MAX;
        require(bf16Ranks[expert] == rank, "BF16 original expert ID-to-compact-rank map differs");
      }
      std::vector<MetalBuffer> sourceBuffers;
      std::vector<std::string> sourceHashes;
      for (const char *plane : {"gate_proj", "up_proj", "down_proj"}) {
        const auto &p = weights.projection(prefix + "." + plane);
        for (const auto &buffer : {p.weights->buffer, p.scales->buffer, p.biases->buffer}) {
          sourceBuffers.push_back(buffer); sourceHashes.push_back(digestBuffer(buffer));
        }
      }
      auto immutable = store->immutableWeightBuffers();
      checkStoreRanks(*store, storeMetadata, immutable);
      std::vector<std::string> rankHashes;
      for (uint32_t index = 0; index < 48; ++index) rankHashes.push_back(digestBuffer(immutable[index * 2 + 1]));
      std::ofstream out(argv[4]); require(bool(out), "cannot create report");
      out << std::setprecision(12)
          << "{\"schema\":\"prefill4k-persisted-int8-top128-vs-lossless-bf16-hot128-v1\",\"pass\":true"
          << ",\"source_identity\":" << splash::json::quote(weights.sourceIdentity())
          << ",\"manifest_identity\":" << splash::json::quote(weights.manifestFingerprint())
          << ",\"prefix\":" << splash::json::quote(prefix) << ",\"layer\":" << layer
          << ",\"control_policy\":" << splash::json::quote(kFlashInt8ExpertStoreSemantics)
          << ",\"candidate_policy\":\"lossless-original-Q4-to-BF16-coefficients-whole-K-all-valid-jobs-Direct-A-misses\""
          << ",\"cross_variant_numerical_difference_expected\":true,\"cross_variant_bit_exact_required\":false"
          << ",\"model_quality_qualified\":false,\"identical_original_expert_ids\":true,\"hot_experts\":128"
          << ",\"original_expert_ids\":[";
      for (size_t i = 0; i < hotIDs.size(); ++i) { if (i) out << ','; out << hotIDs[i]; }
      const auto initialMemory = governor.snapshot();
      out << "],\"pairs\":" << pairs
          << ",\"timing_scope\":\"complete pack/gate-up/down/combine chain, warm alternating matched commands; constructors, hashes, comparisons and optional individual-dispatch replays excluded\""
          << ",\"int8_store\":{\"identity_sha256\":" << splash::json::quote(store->identitySha256())
          << ",\"plan_sha256\":" << splash::json::quote(store->planSha256())
          << ",\"planned_admitted_bytes\":" << storePlanned << ",\"mapped_bytes\":" << store->mappedBytes()
          << ",\"actual_allocated_bytes\":" << store->actualAllocatedBytes()
          << ",\"constructed_once\":true,\"constructor_all_payload_checksums_verified\":true}"
          << ",\"bf16_cache\":{\"identity_sha256\":" << splash::json::quote(cache->identitySha256())
          << ",\"planned_admitted_bytes\":" << bf16Planned << ",\"actual_allocated_bytes\":" << cache->actualAllocatedBytes()
          << ",\"stored_coefficients_exact\":true,\"initialization_gpu_ms\":" << cache->initializationTiming().gpuSeconds * 1000
          << ",\"coefficient_sha256\":[";
      for (size_t i = 0; i < hashes.size(); ++i) { if (i) out << ','; out << splash::json::quote(hashes[i]); }
      out << "]},\"initial_memory\":{\"original_actual_allocated_bytes\":" << weights.actualAllocatedBytes()
          << ",\"original_gpu_mapped_bytes\":" << weights.pleSSDStorageStats().gpuMappedBytes
          << ",\"original_disk_only_payload_bytes\":" << weights.pleSSDStorageStats().diskOnlyPayloadBytes
          << ",\"engine_limit_bytes\":" << initialMemory.limitBytes
          << ",\"host_reserve_bytes\":" << reserve << ",\"host_available_bytes\":" << initialMemory.hostAvailableBytes
          << ",\"observed_resident_bytes\":" << initialMemory.observedResidentBytes << '}'
          << ",\"pipeline_metadata\":" << pipelineInfo << ",\"cases\":[";
      Replay replay; bool first = true;
      for (const char *pattern : {"all-hot", "mixed", "spread"}) {
        const char *selectedPattern = std::getenv("FLASH_EXPERT_CACHE_PATTERN");
        if (selectedPattern && std::string_view(selectedPattern) != pattern) continue;
        if (!first) out << ','; first = false;
        replay = runCase(backend, weights, prefix, rows, m, pattern, pairs, *cache, *store, layer, governor, out);
        if (std::getenv("FLASH_EXPERT_CACHE_INPUT")) break;
      }
      require(!first, "selected route pattern is unknown");
      require(exactCoefficients(weights, prefix, *cache) == hashes, "immutable BF16 coefficient bytes changed");
      require(digestBuffer(cache->expertRanks()) == bf16RanksHash, "BF16 compact rank map changed");
      checkStoreRanks(*store, storeMetadata, immutable);
      out << "],\"post_run_int8_payload_sha256\":[";
      for (uint32_t index = 0; index < 48; ++index) {
        const auto actual = digestBuffer(immutable[index * 2]);
        require(actual == storeMetadata.layers[index].sha256, "readonly persisted INT8 payload changed");
        require(digestBuffer(immutable[index * 2 + 1]) == rankHashes[index], "INT8 compact rank map changed");
        if (index) out << ','; out << splash::json::quote(actual);
      }
      out << "],\"source_operand_sha256\":[";
      for (size_t i = 0; i < sourceBuffers.size(); ++i) {
        require(digestBuffer(sourceBuffers[i]) == sourceHashes[i], "original source affine bytes changed");
        if (i) out << ','; out << splash::json::quote(sourceHashes[i]);
      }
      const auto memory = backend.memoryStats();
      out << "],\"source_and_store_hashes_unchanged\":true,\"bf16_coefficients_still_exact\":true"
          << ",\"final_memory\":{\"allocated_bytes\":" << memory.allocatedBytes
          << ",\"peak_allocated_bytes\":" << memory.peakAllocatedBytes
          << ",\"device_peak_allocated_bytes\":" << memory.devicePeakAllocatedBytes << '}';
      immutable.clear(); sourceBuffers.clear(); cache.reset(); store.reset(); weights = FlashWeights();
      replay.run(backend);
      out << ",\"retained_graph_mapping_lifetime\":{\"pass\":true,\"int8_store_destroyed\":true,\"bf16_cache_destroyed\":true,\"source_weights_destroyed\":true,\"both_variants_replayed\":true}}\n";
      out.flush(); require(bool(out), "report write failed");
      std::cout << "{\"pass\":true,\"model_quality_qualified\":false,\"report\":" << splash::json::quote(argv[4]) << "}\n";
      return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  }
}
'''


def generate(destination, parent):
    source = (parent / "oracle.mm").read_text()
    source = replacement(source, '#include "flash/FlashExpertDenseCache.hpp"',
                         '#include "flash/FlashExpertDenseCache.hpp"\n#include "flash/FlashInt8ExpertStore.hpp"\n#include "flash/FlashInt8ExpertStoreMetadata.hpp"\n#include <memory>')
    source = replacement(source, '  for (const auto &name : names) {',
                         '  for (uint32_t m : {16u,32u,64u})\n    for (const char *phase : {"gate_up","gate_up_miss_direct","down_scatter","down_miss_direct"})\n      names.push_back(std::string("flash_int8_expert_store_") + phase + "_m" + std::to_string(m) + (m ==64 ? "_n64_sg8" : "_n64"));\n  for (const auto &name : names) {')
    source = replacement(source,
                         '  require(result.relativeL2 <= 0.01 && result.cosine >= 0.9999,\n      "whole-K expert cache numerical comparison failed (1% relative L2 / .9999 cosine bounds)");',
                         '  // INT8 requantization is a numerical alternative; metrics are observations, not model-quality qualification.')
    source = replacement(source,
                         '        << ",\\\"relative_l2\\\":" << relativeL2 << ",\\\"cosine\\\":" << cosine << ",\\\"max_abs\\\":" << maxAbs << \'}\';',
                         '        << ",\\\"relative_l2\\\":";\n    if (std::isfinite(relativeL2)) out << relativeL2; else out << "null";\n    out << ",\\\"relative_l2_defined\\\":" << (std::isfinite(relativeL2) ? "true" : "false")\n        << ",\\\"cosine\\\":" << cosine << ",\\\"max_abs\\\":" << maxAbs << \'}\';')
    source = replacement(source, 'void runCase(', SUPPORT + '\nReplay runCase(')
    source = replacement(source, '             const FlashExpertDenseCache &cache, std::ostream &out) {',
                         '             const FlashExpertDenseCache &cache, const FlashInt8ExpertStore &store,\n             uint32_t layer, splash::engine::MemoryGovernor &governor, std::ostream &out) {')
    source = replacement(source, '  const auto input = upload(backend, hidden, "Q4x8 hidden");',
                         '  const uint64_t casePlanned = casePlannedBytes(rows);\n  auto caseReservation = governor.tryReserve(casePlanned);\n  require(bool(caseReservation), "real governor denied comparator workspace BEFORE allocation");\n  const uint64_t caseBefore = backend.memoryStats().allocatedBytes;\n  const auto input = upload(backend, hidden, "Q4x8 hidden");')
    source = replacement(source, '  const auto route = upload(backend, std::vector<uint16_t>(ids.size(), bf16(0.1f)), "Q4x8 route weights");',
                         '  std::vector<uint16_t> routingWeights(ids.size());\n  for (size_t i = 0; i < routingWeights.size(); ++i) routingWeights[i] = bf16(float(i % kSelections + 1) / 55.0f);\n  const auto route = upload(backend, routingWeights, "matched unequal canonical route weights");')
    source = replacement(source, '  const auto candidate = graphs[1].dispatches();',
                         '  const uint64_t caseAllocated = splash::metal::allocationDelta(caseBefore, backend.memoryStats().allocatedBytes);\n  require(caseAllocated <= casePlanned, "comparator workspace exceeds admitted plan");\n  caseReservation->commit();\n  const auto candidate = graphs[1].dispatches();')
    control_start = source.index('      addMoEBlockedGateUp(graphs[i], gate, up, scratch[i], diagnostic, rows, tile);')
    control_end = source.index('    } else {', control_start)
    source = source[:control_start] + '''      store.addGateUp(graphs[i], layer, scratch[i], diagnostic, rows, tile);
      store.addDownScatter(graphs[i], layer, scratch[i], diagnostic, rows, tile);
      const auto dispatches = graphs[i].dispatches();
      require(std::count_if(dispatches.begin(), dispatches.end(), [](const ComputeDispatch &d) {
          return d.pipelineName.starts_with("flash_int8_expert_store_");
        }) == 4, "control must use current persisted INT8 hit/Direct-A-miss producers");
''' + source[control_end:]
    source = replacement(source,
                         '  if (!wholeKJobs || std::getenv("FLASH_EXPERT_CACHE_REQUIRE_EXACT")) require(!activation.mismatches && !downComparison.mismatches && !combined.mismatches,\n      "miss/tail-only producer must be bit-exact to current Q4x8");',
                         '  // Cross-variant equality is deliberately not required: control uses requantized INT8.')
    source = replacement(source, '  std::array<std::vector<CommandTiming>, 2> timing;',
                         '''  std::array<std::array<std::string, 3>, 2> warmHashes;
  for (uint32_t i = 0; i < 2; ++i) warmHashes[i] = {
      digestBuffer(scratch[i].packedActivated, uint64_t(rows) * kSelections * 640 * 2),
      digestBuffer(scratch[i].scatteredDown), digestBuffer(output[i])};
  const auto stable = [&](uint32_t i) {
    require(digestBuffer(scratch[i].packedActivated, uint64_t(rows) * kSelections * 640 * 2) == warmHashes[i][0] &&
        digestBuffer(scratch[i].scatteredDown) == warmHashes[i][1] && digestBuffer(output[i]) == warmHashes[i][2],
        "own-variant expert outputs changed after matched timed/profiled commands");
  };
  std::array<std::vector<CommandTiming>, 2> timing;''')
    timed_start = source.index('  if (std::getenv("FLASH_EXPERT_CACHE_REQUIRE_EXACT")) {')
    timed_end = source.index('  const uint32_t activeExperts', timed_start)
    source = source[:timed_start] + '''  stable(0); stable(1);
  const auto finalActivation = compareNumerical(scratch[0].packedActivated, scratch[1].packedActivated,
      uint64_t(rows) * kSelections * 640);
  const auto finalDown = compareNumerical(scratch[0].scatteredDown, scratch[1].scatteredDown,
      uint64_t(rows) * kSelections * 2560);
  const auto finalCombined = compareNumerical(output[0], output[1], uint64_t(rows) * 2560);
  const uint64_t exactMissRoutes = exactMissSlices(scratch, packed, ids, cache.selectedExpertIDs());
  std::array<std::vector<splash::metal::DispatchTiming>, 2> phases;
  if (std::getenv("FLASH_EXPERT_COMPARE_PROFILE_PHASES")) {
    backend.setDispatchProfiling(true);
    for (uint32_t i = 0; i < 2; ++i) {
      (void)backend.submitCommand(graphs[i].dispatches());
      phases[i] = backend.takeDispatchProfile(); healthy(); stable(i);
    }
    backend.setDispatchProfiling(false);
  }
''' + source[timed_end:]
    source = replacement(source,
                         '  out << ",\\\"candidate_wall_ms\\\":"; times(out, timing[1], false); out << \'}\';',
                         r'''  out << ",\"candidate_wall_ms\":"; times(out, timing[1], false);
  out << ",\"workspace_planned_admitted_bytes\":" << casePlanned
      << ",\"workspace_actual_allocated_bytes\":" << caseAllocated
      << ",\"q4_miss_routes_bit_exact\":" << exactMissRoutes;
  out << ",\"input_bf16_sha256\":" << splash::json::quote(digestBuffer(input))
      << ",\"route_ids_sha256\":" << splash::json::quote(digestBuffer(expertIDs))
      << ",\"route_weights_sha256\":" << splash::json::quote(digestBuffer(route))
      << ",\"control_pipelines\":"; names(out, graphs[0].dispatches());
  out << ",\"candidate_pipelines\":"; names(out, graphs[1].dispatches());
  out << ",\"final_activation\":"; finalActivation.write(out);
  out << ",\"final_down\":"; finalDown.write(out);
  out << ",\"final_combine\":"; finalCombined.write(out);
  out << ",\"output_sha256\":{";
  for (uint32_t i = 0; i < 2; ++i) {
    if (i) out << ',';
    out << splash::json::quote(i == 0 ? "int8_control" : "bf16_candidate") << ":{\"activation\":"
        << splash::json::quote(digestBuffer(scratch[i].packedActivated, uint64_t(rows) * kSelections * 640 * 2))
        << ",\"down\":" << splash::json::quote(digestBuffer(scratch[i].scatteredDown))
        << ",\"combine\":" << splash::json::quote(digestBuffer(output[i])) << '}';
  }
  out << "},\"phase_replay_scope\":\"optional individual-dispatch GPU commands excluded from full-chain timing\""
      << ",\"control_phase_replay\":"; profile(out, phases[0]);
  out << ",\"candidate_phase_replay\":"; profile(out, phases[1]);
  out << '}';
  Replay replay; replay.graphs = std::move(graphs); replay.output = output;
  replay.diagnostic = diagnostic; replay.guards = std::move(guards); replay.rows = rows;
  for (uint32_t i = 0; i < 2; ++i) {
    replay.activated[i] = scratch[i].packedActivated; replay.down[i] = scratch[i].scatteredDown;
    replay.expectedHashes[i] = warmHashes[i];
  }
  return replay;''')
    main_start = source.index('int main(int argc, char **argv) {')
    source = source[:main_start] + MAIN
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "oracle.mm").write_text(source)
    (destination / "generation.json").write_text(json.dumps({
        "scope": "private source-copy comparator; generation performs no GPU work or payload scans",
        "parent_oracle_sha256": sha(parent / "oracle.mm"),
        "parent_candidate_air_sha256": sha(parent / "candidate.air"),
        "bridge_sha256": sha(ROOT / "dev/benchmarks/prefill4k_bf16cache/bridge.hpp"),
        "cross_variant_bit_exact_required": False,
        "bf16_coefficient_exact_checks_retained": True,
    }, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_run(a):
    build = (ROOT / a.build).resolve()
    store = (ROOT / a.store).resolve()
    plan = json.loads((ROOT / PLAN).read_text())
    manifest = json.loads((store / "manifest.json").read_text())
    ids = plan["selected_experts"][a.layer]
    if (len(ids) != 128 or ids != sorted(set(ids))
            or plan["selected_experts"] != manifest["selected_experts"]):
        raise ValueError("The actual Top128 frequency plan and persisted INT8 original IDs differ")
    if not 1 <= a.rows <= 8192 or not 1 <= a.pairs <= 32 or (a.tile == 64 and a.rows < 1024):
        raise ValueError("Unsupported comparator row/pair/tile geometry")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SPLASH_FLASH_", "FLASH_EXPERT_CACHE_", "FLASH_EXPERT_COMPARE_"))}
    controls = {"SPLASH_FLASH_PLE_SSD_STREAMING": "1", "SPLASH_FLASH_MOE_Q4X8": "1",
                "SPLASH_FLASH_MOE_DIRECT_A": "1", "SPLASH_FLASH_MOE_M64": "1",
                "FLASH_EXPERT_CACHE_HOT_IDS": ",".join(map(str, ids)),
                "FLASH_EXPERT_CACHE_PREFIX": f"language_model.model.layers.{a.layer}.mlp.switch_mlp",
                "FLASH_EXPERT_CACHE_ROWS": str(a.rows), "FLASH_EXPERT_CACHE_TILE": str(a.tile),
                "FLASH_EXPERT_CACHE_PAIRS": str(a.pairs)}
    if a.pattern:
        controls["FLASH_EXPERT_CACHE_PATTERN"] = a.pattern
    if a.profile_phases:
        controls["FLASH_EXPERT_COMPARE_PROFILE_PHASES"] = "1"
    env.update(controls)
    report = a.report.resolve()
    witness = report.with_suffix(report.suffix + ".invocation.json")
    if report.exists() or witness.exists():
        raise ValueError("Choose a fresh comparator report/provenance path")
    command = [str(build / "oracle"), str(build / "splash.metallib"),
               str(ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1"), str(store), str(report)]
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps({"scope": "full-chain INT8 Top128 versus lossless BF16 Hot128 primitive; model quality unqualified",
        "gpu_requested": a.run, "gpu_completed": False, "command": command, "controls": controls,
        "identical_original_expert_ids": ids, "cross_variant_bit_exact_required": False,
        "binary_sha256": sha(build / "oracle"), "metallib_sha256": sha(build / "splash.metallib"),
        "frequency_plan_sha256": sha(ROOT / PLAN), "int8_manifest_sha256": sha(store / "manifest.json"),
        "int8_constructor_maps_and_checksums_all_48_layers": True,
        "int8_planned_allocation_bytes": manifest["planned_allocation_bytes"],
        "int8_mapped_bytes": manifest["total_bytes"],
        "bf16_layer_logical_coefficient_bytes": 128 * (640 * 2560 * 2 + 2560 * 640) * 2,
        "bf16_full_48_layer_logical_coefficient_bytes": 48 * 128 * (640 * 2560 * 2 + 2560 * 640) * 2,
        "generation_qualified": False}, indent=2) + "\n")
    print(json.dumps({"gpu_requested": a.run, "gpu_completed": False, "command": command, "witness": str(witness)}))
    if a.run:
        subprocess.run(command, env=env, check=True, cwd=ROOT)
        completed = json.loads(witness.read_text())
        completed["gpu_completed"] = json.loads(report.read_text()).get("pass") is True
        completed["report_sha256"] = sha(report)
        witness.write_text(json.dumps(completed, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    g = commands.add_parser("generate", help="copy ready parent oracle and create isolated comparator source")
    g.add_argument("destination", type=Path)
    g.add_argument("--parent-build", type=Path, default=DEFAULT_PARENT)
    r = commands.add_parser("run", help="write provenance only unless Root explicitly adds --run")
    r.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    r.add_argument("--store", type=Path, default=DEFAULT_STORE)
    r.add_argument("--layer", type=int, choices=range(48), default=0)
    r.add_argument("--rows", type=int, default=2048)
    r.add_argument("--tile", type=int, choices=(16, 32, 64), default=32)
    r.add_argument("--pairs", type=int, default=4)
    r.add_argument("--pattern", choices=("all-hot", "mixed", "spread"))
    r.add_argument("--profile-phases", action="store_true", help="additional separate dispatch replays after normal timing")
    r.add_argument("--report", type=Path, required=True)
    r.add_argument("--run", action="store_true")
    a = p.parse_args()
    if a.command == "generate":
        generate(a.destination, a.parent_build)
    else:
        plan_run(a)


if __name__ == "__main__":
    main()
