// Device-free hybrid checks against copied SG2 policy, actual Store pipeline
// hooks, and actual compiled bucket job bounds. Raw hit-parameter construction
// is qualified by the sealed source witness, not a fabricated allRows hook.
// No Store, model, Metal backend, allocator or payload is created/read.
#include "worker_bridge.hpp"
#include "dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>

namespace splash::flash {
std::string combinedSG2TailPipelineForCPU(bool gate);
std::string combinedSG2TailFallbackPipelineForCPU(const char *phase, uint32_t tileRows);
bool combinedSG2TailEligibleForCPU(uint32_t rows, uint32_t tileRows, bool verification);
} // namespace splash::flash

namespace tail = splash::flash::adaptive_expert_tail_sg2k128_sep21;
namespace fixed = splash::flash::fixed_sg2_prefill_sep21;
namespace store = splash::flash;
using splash::flash::FlashMoEBlockedTile;

namespace {
constexpr const char *kTailFlag = "SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21";
constexpr const char *kSG2Flag = "SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT";
constexpr std::string_view kTailMarker =
    ";private-prefill-moe-sg2-k128-m16-tail-validle16-original-m32-jobs-r2048-exact-f32-bf16-sep21-v1";
constexpr std::string_view kSG2Marker =
    ";private-prefill-moe-sg2-k128-sep21-main-nonverification-r2048-m32-original-i8-f32lateScale-bf16-boundaries-v1";
constexpr std::array<uint32_t, 6> kSelections{0, 1, 6, 7, 8, UINT32_MAX};
constexpr std::array<uint32_t, 7> kTiles{0, 8, 16, 32, 64, 128, UINT32_MAX};
size_t checks = 0;

void require(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
template <typename F> void rejects(F &&operation, const char *message) {
  bool caught = false;
  try { operation(); } catch (const std::invalid_argument &) { caught = true; }
  require(caught, message);
}
void set(const char *key, const char *value) {
  if ((value ? setenv(key, value, 1) : unsetenv(key)) != 0)
    throw std::runtime_error("Cannot set CPU combined-policy environment");
}
std::string_view expectedName(bool gate, uint32_t selected, bool enabled) {
  if (selected != 7) return {};
  if (enabled) return gate ? "adaptive_expert_tail_sg2k128_sep21_gate_up_m16_tail" :
      "adaptive_expert_tail_sg2k128_sep21_down_scatter_m16_tail";
  return gate ? "prefill_moe_sep21_memory_fixed_gate_up_m32_n64_k128_sg2" :
      "prefill_moe_sep21_memory_fixed_down_scatter_m32_n64_k128_sg2";
}

void parserTests() {
  require(!tail::parseSwitch(nullptr) && !tail::parseSwitch("0") && tail::parseSwitch("1"),
      "Combined tail parser missing/0/1 golden differs");
  for (const char *raw : {"", "00", "01", "+1", "-0", "-1", "2", "7", "true", "false",
       "yes", " 1", "1 ", "0\n", "1\n", "0\t", "1\t", "0x1", "1.0"})
    rejects([&] { (void)tail::parseSwitch(raw); }, "Malformed combined-tail flag accepted");
  require(fixed::parseSelection(nullptr) == 0 && fixed::parseSelection("0") == 0 &&
      fixed::parseSelection("7") == 7, "Inherited SG2 parser missing/0/7 golden differs");
  for (const char *raw : {"", "00", "07", "+7", "-7", "1", "6", "8", "true", "false",
       " 7", "7 ", "0\n", "7\n", "0x7", "7.0"})
    rejects([&] { (void)fixed::parseSelection(raw); }, "Malformed inherited SG2 selection accepted");
}

size_t explicitPolicyTests() {
  size_t cases = 0;
  for (uint32_t selected : kSelections)
    for (bool enabled : {false, true}) {
      for (bool gate : {false, true}) {
        ++cases;
        require(std::string_view(tail::producerNameFor(gate, selected, enabled)) == expectedName(gate, selected, enabled),
            "Combined explicit producer inventory/control matrix differs");
        require(std::string_view(fixed::producerNameFor(gate, selected)) == expectedName(gate, selected, false),
            "Inherited explicit SG2 producer API changed");
      }
      require(std::string_view(tail::markerFor(selected, enabled)) ==
          (selected == 7 && enabled ? kTailMarker : std::string_view()),
          "Combined marker changed disabled/inactive identity or missed active policy");
      require(std::string_view(fixed::markerFor(selected)) ==
          (selected == 7 ? kSG2Marker : std::string_view()), "Inherited SG2 marker changed");
      require(fixed::producerThreadsFor(selected) == (selected == 7 ? 64u : 0u),
          "Combined producer changed inherited 64-thread policy");
    }
  require(kTailMarker != kSG2Marker, "Combined tail marker does not identify its distinct execution policy");
  return cases;
}

size_t eligibilityTests() {
  size_t cases = 0;
  for (uint32_t rows = 0; rows <= 8193; ++rows)
    for (uint32_t tileRows : kTiles)
      for (bool verification : {false, true})
        for (uint32_t selected : kSelections) {
          ++cases;
          const bool expected = selected == 7 && rows == 2048 && tileRows == 32 && !verification;
          require(fixed::eligibleFor(rows, static_cast<FlashMoEBlockedTile>(tileRows), verification, selected) == expected,
              "Combined route widened main/nonverification/R2048/M32 eligibility");
        }
  for (uint32_t tileRows : kTiles)
    for (bool verification : {false, true})
      require(!fixed::eligibleFor(UINT32_MAX, static_cast<FlashMoEBlockedTile>(tileRows), verification, 7),
          "Extreme row count became combined-prefill eligible");
  static_assert(fixed::eligibleFor(2048, FlashMoEBlockedTile::M32N64, false, 7));
  static_assert(!fixed::eligibleFor(2048, FlashMoEBlockedTile::M32N64, true, 7));
  static_assert(!fixed::eligibleFor(16, FlashMoEBlockedTile::M16N64, false, 7));
  static_assert(!fixed::eligibleFor(4096, FlashMoEBlockedTile::M64N64, false, 7));
  return cases;
}

void checkCapacity(uint32_t rows, uint32_t selections, uint32_t tileRows) {
  // Independently bound sum_e ceil(count[e]/M), without reading any counts.
  const uint32_t routes = rows * selections;
  const uint32_t expected = routes / tileRows + (routes % tileRows ? 1u : 0u) + 511;
  require(store::moEBucketJobCapacity(rows, selections, tileRows) == expected,
      "Actual compiled bucket capacity differs from original matrix-job bound");
}
size_t capacityTests() {
  static_assert(std::is_standard_layout_v<FlashInt8ExpertStoreParams>);
  static_assert(std::is_trivially_copyable_v<FlashInt8ExpertStoreParams>);
  static_assert(sizeof(FlashInt8ExpertStoreParams) == 32);
  static_assert(alignof(FlashInt8ExpertStoreParams) == alignof(uint32_t));
  static_assert(offsetof(FlashInt8ExpertStoreParams, rows) == 0);
  static_assert(offsetof(FlashInt8ExpertStoreParams, selections) == 4);
  static_assert(offsetof(FlashInt8ExpertStoreParams, route_capacity) == 8);
  static_assert(offsetof(FlashInt8ExpertStoreParams, job_capacity) == 12);
  static_assert(offsetof(FlashInt8ExpertStoreParams, tile_rows) == 16);
  static_assert(offsetof(FlashInt8ExpertStoreParams, stored_experts) == 20);
  static_assert(offsetof(FlashInt8ExpertStoreParams, scale_group_size) == 24);
  static_assert(offsetof(FlashInt8ExpertStoreParams, reserved) == 28);
  static_assert(sizeof(FlashInt8ExpertStoreGateParams) == 128);
  static_assert(sizeof(FlashInt8ExpertStoreDownParams) == 96);
  // Static ABI golden only. The overlay witness checks the actual raw methods'
  // hit initializers, producer buffers, grids and copied preparatory commands.
  constexpr FlashInt8ExpertStoreParams canonical{2048, 10, 20480, 1151, 32, 512, 0, 0};
  constexpr std::array<uint32_t, 8> canonicalWords{2048, 10, 20480, 1151, 32, 512, 0, 0};
  static_assert(canonical.tile_rows == 32 && canonical.stored_experts == 512 &&
      canonical.scale_group_size == 0 && canonical.reserved == 0);
  require(std::memcmp(&canonical, canonicalWords.data(), sizeof(canonical)) == 0,
      "Static canonical R2048 M32 producer ABI golden differs");
  require(store::moEBucketJobCapacity(2048, 10, 32) == canonical.job_capacity,
      "M16 math descriptor changed canonical original-M32 job capacity");
  size_t cases = 0;
  for (uint32_t rows = 1; rows <= 8192; ++rows)
    for (uint32_t selections = 1; selections <= 10; ++selections)
      for (uint32_t tileRows : {16u, 32u, 64u}) { ++cases; checkCapacity(rows, selections, tileRows); }
  for (const auto &geometry : std::array<std::array<uint32_t, 3>, 7>{{
      {0, 10, 32}, {8193, 10, 32}, {UINT32_MAX, 10, 32}, {2048, 0, 32},
      {2048, 11, 32}, {2048, 10, 12}, {2048, 10, UINT32_MAX}}})
    rejects([&] { (void)store::moEBucketJobCapacity(geometry[0], geometry[1], geometry[2]); },
        "Actual compiled bucket job bound accepted invalid geometry");
  return cases;
}

void runtimeTests(uint32_t selected, bool enabled) {
  require(fixed::selection() == selected && fixed::requested() == (selected == 7),
      "Frozen SG2 control changed after combining tail policy");
  require(tail::requested() == enabled, "Frozen tail flag is coupled to SG2 selection or mutable");
  require(std::string_view(fixed::marker()) == (selected == 7 ? kSG2Marker : std::string_view()),
      "Frozen inherited SG2 identity changed");
  require(std::string_view(tail::markerFor(fixed::selection(), tail::requested())) ==
      (selected == 7 && enabled ? kTailMarker : std::string_view()), "Frozen combined identity marker differs");
  require(fixed::producerThreads() == (selected == 7 ? 64u : 0u), "Frozen combined launch threads differ");
  for (bool gate : {false, true}) {
    const auto expected = expectedName(gate, selected, enabled);
    require(std::string_view(fixed::producerName(gate)) == expected, "Actual copied SG2 producer wrapper differs");
    require(store::combinedSG2TailPipelineForCPU(gate) == expected,
        "Actual Store producer wrapper differs from shared frozen controls");
  }
  // These helpers implement unchanged decode, verification and other-row
  // fallback producer names; no FlashForward/model is needed to inspect them.
  for (uint32_t tileRows : {16u, 32u, 64u})
    for (const char *phase : {"gate_up", "down_scatter", "gate_up_miss_direct", "gate_up_miss_staged",
         "down_miss_direct", "down_miss_staged"}) {
      const std::string original = std::string("flash_int8_expert_store_") + phase + "_m" +
          std::to_string(tileRows) + (tileRows == 64 ? "_n64_sg8" : "_n64");
      require(store::combinedSG2TailFallbackPipelineForCPU(phase, tileRows) == original,
          "Combined tail flag changed original M16/M32/M64/miss fallback pipeline");
    }
  for (uint32_t rows : {0u, 1u, 16u, 255u, 1024u, 2047u, 2048u, 2049u, 4096u, 8192u, UINT32_MAX})
    for (uint32_t tileRows : kTiles)
      for (bool verification : {false, true}) {
        const bool expected = selected == 7 && rows == 2048 && tileRows == 32 && !verification;
        require(fixed::eligible(rows, static_cast<FlashMoEBlockedTile>(tileRows), verification) == expected,
            "Copied runtime eligibility differs from frozen main-prefill scope");
        require(store::combinedSG2TailEligibleForCPU(rows, tileRows, verification) == expected,
            "Actual Store runtime eligibility differs across translation units");
      }
  for (uint32_t rows : {1u, 16u, 256u, 1024u, 2048u, 8192u})
    for (uint32_t tileRows : {16u, 32u, 64u}) checkCapacity(rows, 10, tileRows);
}

void freezeTests(std::string_view mode) {
  // Hybrid retains Q4 for small/verification rows; no all-row-I8 flag is needed.
  set("SPLASH_FLASH_ALLROWS_FULL512_TARGET", "0");
  set("SPLASH_FLASH_ALLROWS_GATHERED_MPP", "0");
  uint32_t selected = 0;
  bool enabled = false;
  if (mode == "--freeze00" || mode == "--freeze01" || mode == "--freeze70" || mode == "--freeze71") {
    selected = mode[8] == '7' ? 7 : 0;
    enabled = mode[9] == '1';
    set(kSG2Flag, selected == 7 ? "7" : "0"); set(kTailFlag, enabled ? "1" : "0");
  } else if (mode == "--missing00") {
    set(kSG2Flag, nullptr); set(kTailFlag, nullptr);
  } else if (mode == "--retry00" || mode == "--retry71") {
    set(kSG2Flag, "bad"); set(kTailFlag, "bad");
    rejects([] { (void)fixed::selection(); }, "Malformed first SG2 request did not throw");
    rejects([] { (void)tail::requested(); }, "Malformed first independent tail request did not throw");
    selected = mode == "--retry71" ? 7 : 0; enabled = selected == 7;
    set(kSG2Flag, selected == 7 ? "7" : "0"); set(kTailFlag, enabled ? "1" : "0");
  } else if (mode == "--sg2-first") {
    set(kSG2Flag, "7"); set(kTailFlag, "bad");
    require(fixed::selection() == 7, "SG2 first-use incorrectly parsed independent tail flag");
    set(kSG2Flag, "0"); set(kTailFlag, "1"); selected = 7; enabled = true;
  } else if (mode == "--tail-first") {
    set(kSG2Flag, "bad"); set(kTailFlag, "1");
    require(tail::requested(), "Tail first-use incorrectly parsed independent SG2 flag");
    set(kSG2Flag, "7"); set(kTailFlag, "0"); selected = 7; enabled = true;
  } else throw std::invalid_argument("Unknown combined policy freeze mode");
  runtimeTests(selected, enabled);
  // Mutate each control separately, then malformed/missing controls together.
  // Store hooks detect accidentally separate frozen statics in other TUs.
  set(kSG2Flag, selected == 7 ? "0" : "7"); runtimeTests(selected, enabled);
  set(kTailFlag, enabled ? "0" : "1"); runtimeTests(selected, enabled);
  set(kSG2Flag, "bad"); set(kTailFlag, "bad"); runtimeTests(selected, enabled);
  set(kSG2Flag, nullptr); set(kTailFlag, nullptr); runtimeTests(selected, enabled);
  std::cout << "{\"kind\":\"hybrid_sg2k128_m16_tail_worker_policy_cpu\",\"mode\":\"" << mode
      << "\",\"frozen_sg2_selection\":" << selected << ",\"frozen_tail_enabled\":" << (enabled ? "true" : "false")
      << ",\"checks\":" << checks << ",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2) { freezeTests(argv[1]); return 0; }
    if (argc != 1) throw std::invalid_argument("Use no arguments or a documented combined policy freeze mode");
    set("SPLASH_FLASH_ALLROWS_FULL512_TARGET", "0");
    set("SPLASH_FLASH_ALLROWS_GATHERED_MPP", "0");
    parserTests();
    const auto producerCases = explicitPolicyTests();
    const auto eligibilityCases = eligibilityTests();
    const auto capacityCases = capacityTests();
    runtimeTests(fixed::parseSelection(std::getenv(kSG2Flag)), tail::parseSwitch(std::getenv(kTailFlag)));
    std::cout << "{\"kind\":\"hybrid_sg2k128_m16_tail_worker_policy_cpu\",\"producer_cases\":" << producerCases
        << ",\"eligibility_cases\":" << eligibilityCases << ",\"job_capacity_cases\":" << capacityCases
        << ",\"frozen_sg2_selection\":" << fixed::selection() << ",\"frozen_tail_enabled\":" << (tail::requested() ? "true" : "false")
        << ",\"parameter_abi_bytes\":32,\"r2048_m32_job_capacity\":1151,\"active_producer_threads\":64,"
           "\"allrows_full512_target_required\":false,\"actual_raw_hit_params_qualification\":\"sealed_source_witness\","
           "\"checks\":" << checks << ",\"gpu_work\":false,\"payload_reads\":false,\"allocator_used\":false,\"pass\":true}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Hybrid SG2K128/M16-tail CPU policy check failed: " << error.what() << '\n';
    return 1;
  }
}
