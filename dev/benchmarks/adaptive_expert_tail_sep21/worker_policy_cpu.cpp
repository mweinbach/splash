// CPU-only qualification of the sealed whole-worker pipeline policy.
// The extern hooks call the actual copied FlashInt8ExpertStore helpers. No
// Store, FlashForward, model, MetalBackend, buffer or payload is constructed.
#include "worker_bridge.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

namespace splash::flash {
// Define these only in the copied Store, after its anonymous namespace ends.
// The explicit policy hook uses the same helper as the actual runtime wrapper;
// the runtime hook calls that actual wrapper with its first-use frozen flag.
std::string adaptiveExpertTailPipelineForCPU(const char *, uint32_t, bool);
std::string adaptiveExpertTailPipelineRuntimeForCPU(const char *, uint32_t);
FlashInt8ExpertStoreParams adaptiveExpertTailParamsForCPU(uint32_t, uint32_t, uint32_t);
uint32_t adaptiveExpertTailLaunchForCPU(const FlashInt8ExpertStoreParams &);
} // namespace splash::flash

namespace policy = splash::flash::adaptive_expert_tail_sep21;
namespace store = splash::flash;

namespace {
constexpr const char *kFlag = "SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SEP21";
constexpr std::string_view kMarker =
    ";private-native-m32-jobs-m16-tail-validle16-sg4-exact-f32-bf16-sep21-v1";
constexpr std::array<const char *, 14> kPhases{
    "gate_up", "down_scatter", "gate_up_miss", "down_miss",
    "gate_up_miss_direct", "gate_up_miss_staged", "down_miss_direct", "down_miss_staged",
    "gate_up_extra", "down_scatter_extra", "gate_upX", "gate_up_missm32", "", "unknown"};
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
void setFlag(const char *value) {
  if ((value ? setenv(kFlag, value, 1) : unsetenv(kFlag)) != 0)
    throw std::runtime_error("Cannot set CPU policy-test environment");
}
std::string expectedPipeline(const char *phase, uint32_t tileRows, bool enabled) {
  const std::string_view name(phase);
  if (enabled && tileRows == 32 && (name == "gate_up" || name == "down_scatter"))
    return std::string("adaptive_expert_tail_sep21_") + phase + "_m16_tail";
  return std::string("flash_int8_expert_store_") + phase + "_m" + std::to_string(tileRows) +
      (tileRows == 64 ? "_n64_sg8" : "_n64");
}

void parserTests() {
  require(!policy::parseSwitch(nullptr) && !policy::parseSwitch("0") && policy::parseSwitch("1"),
      "Missing/0/1 parser golden differs");
  for (const char *raw : {"", "00", "01", "+1", "-0", "-1", "2", "true", "false", "yes",
       " 1", "1 ", "0\n", "1\n", "0\t", "1\t", "0x1", "1.0"})
    rejects([&] { (void)policy::parseSwitch(raw); }, "Malformed adaptive-tail flag was accepted");
}

size_t pipelineTests() {
  size_t cases = 0;
  for (uint32_t tileRows : {16u, 32u, 64u})
    for (const char *phase : kPhases)
      for (bool enabled : {false, true}) {
        ++cases;
        const std::string_view name(phase);
        const bool selected = enabled && tileRows == 32 && (name == "gate_up" || name == "down_scatter");
        require(policy::selected(name, tileRows, enabled) == selected, "Hit-only M32 selection differs");
        const std::string expected = expectedPipeline(phase, tileRows, enabled);
        require(policy::pipeline(phase, tileRows, enabled) == expected, "Bridge pipeline inventory differs");
        require(store::adaptiveExpertTailPipelineForCPU(phase, tileRows, enabled) == expected,
            "Actual Store pipeline helper differs from independent inventory");
      }
  for (uint32_t tileRows : {0u, 1u, 8u, 15u, 17u, 31u, 33u, 63u, 65u, UINT32_MAX})
    for (bool enabled : {false, true}) {
      require(!policy::selected("gate_up", tileRows, enabled), "Unsupported tile became selected");
      rejects([&] { (void)policy::pipeline("gate_up", tileRows, enabled); }, "Bridge accepted unsupported tile");
      rejects([&] { (void)store::adaptiveExpertTailPipelineForCPU("gate_up", tileRows, enabled); },
          "Actual Store helper accepted unsupported tile");
    }
  require(policy::marker(false).empty() && policy::marker(true) == kMarker,
      "Disabled identity changed or enabled identity marker differs");
  require(policy::marker(true) != policy::marker(false), "Adaptive policy lacks a distinct runtime marker");
  return cases;
}

void checkParams(uint32_t rows, uint32_t selections, uint32_t tileRows) {
  // Independent integer bound: sum ceil(bucket[e]/M) <= ceil(routes/M)+511.
  const uint32_t routes = rows * selections;
  const uint32_t jobs = routes / tileRows + (routes % tileRows != 0 ? 1u : 0u) + 511;
  const std::array<uint32_t, 8> expected{rows, selections, routes, jobs, tileRows, 512, 0, 0};
  const auto actual = store::adaptiveExpertTailParamsForCPU(rows, selections, tileRows);
  require(std::memcmp(&actual, expected.data(), sizeof(actual)) == 0,
      "Actual Store parameter bytes/job capacity differ from original ABI");
  const uint32_t expectedLaunch = rows < 256 ? std::min(routes, jobs) : jobs;
  require(store::adaptiveExpertTailLaunchForCPU(actual) == expectedLaunch,
      "Actual Store job launch differs from original launch bound");
}

size_t parameterTests() {
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
  size_t cases = 0;
  for (uint32_t rows = 1; rows <= 8192; ++rows)
    for (uint32_t selections = 1; selections <= 10; ++selections)
      for (uint32_t tileRows : {16u, 32u, 64u}) {
        ++cases;
        checkParams(rows, selections, tileRows);
      }
  const auto r2048 = store::adaptiveExpertTailParamsForCPU(2048, 10, 32);
  require(r2048.route_capacity == 20480 && r2048.job_capacity == 1151 && r2048.tile_rows == 32,
      "Adaptive M16 descriptor changed original 2K M32 jobs/parameters");
  const auto r1024 = store::adaptiveExpertTailParamsForCPU(1024, 10, 32);
  require(r1024.route_capacity == 10240 && r1024.job_capacity == 831 && r1024.tile_rows == 32,
      "Adaptive M16 descriptor changed original 1K M32 jobs/parameters");
  for (const auto &geometry : std::array<std::array<uint32_t, 3>, 10>{{
      {0, 10, 32}, {8193, 10, 32}, {UINT32_MAX, 10, 32},
      {2048, 0, 32}, {2048, 11, 32}, {2048, UINT32_MAX, 32},
      {2048, 10, 0}, {2048, 10, 12}, {2048, 10, 33}, {2048, 10, UINT32_MAX}}})
    rejects([&] { (void)store::adaptiveExpertTailParamsForCPU(geometry[0], geometry[1], geometry[2]); },
        "Actual Store parameter helper accepted unsupported geometry");
  return cases;
}

void runtimeTests(bool expected) {
  require(policy::requested() == expected, "Frozen requested policy differs from expected state");
  require(policy::marker(policy::requested()) == (expected ? kMarker : std::string_view()),
      "Frozen runtime identity marker differs");
  for (uint32_t tileRows : {16u, 32u, 64u})
    for (const char *phase : kPhases)
      require(store::adaptiveExpertTailPipelineRuntimeForCPU(phase, tileRows) == expectedPipeline(phase, tileRows, expected),
          "Actual Store runtime wrapper differs from frozen hit-only pipeline policy");
  // Confirm original parameter/job ABI survives each frozen policy state.
  for (uint32_t rows : {1u, 16u, 255u, 256u, 1024u, 2048u, 8192u})
    for (uint32_t tileRows : {16u, 32u, 64u}) checkParams(rows, 10, tileRows);
}

void freezeTests(std::string_view mode) {
  bool expected = false;
  if (mode == "--freeze0") setFlag("0");
  else if (mode == "--freeze1") { setFlag("1"); expected = true; }
  else if (mode == "--missing0") setFlag(nullptr);
  else if (mode == "--retry0" || mode == "--retry1") {
    setFlag("bad");
    rejects([] { (void)policy::requested(); }, "Invalid first request failed to throw");
    expected = mode == "--retry1";
    setFlag(expected ? "1" : "0");
  } else throw std::invalid_argument("Unknown CPU policy freeze mode");
  runtimeTests(expected);
  // Freeze in this translation unit, mutate the environment, then inspect the
  // actual Store translation unit to detect a separate policy-static instance.
  setFlag(expected ? "0" : "1"); runtimeTests(expected);
  setFlag(""); runtimeTests(expected);
  setFlag(nullptr); runtimeTests(expected);
  std::cout << "{\"kind\":\"adaptive_expert_tail_worker_policy_cpu\",\"mode\":\"" << mode
      << "\",\"enabled\":" << (expected ? "true" : "false") << ",\"checks\":" << checks
      << ",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2) { freezeTests(argv[1]); return 0; }
    if (argc != 1) throw std::invalid_argument("Use no arguments, --freeze0, --freeze1, --missing0, --retry0 or --retry1");
    parserTests();
    const size_t pipelineCases = pipelineTests();
    const size_t parameterCases = parameterTests();
    runtimeTests(policy::parseSwitch(std::getenv(kFlag)));
    std::cout << "{\"kind\":\"adaptive_expert_tail_worker_policy_cpu\",\"pipeline_cases\":" << pipelineCases
        << ",\"parameter_cases\":" << parameterCases << ",\"rows_checked\":\"1..8192\",\"selections_checked\":\"1..10\","
           "\"tile_rows_checked\":[16,32,64],\"parameter_abi_bytes\":32,\"r2048_m32_job_capacity\":1151,"
           "\"r1024_m32_job_capacity\":831,\"frozen_enabled\":" << (policy::requested() ? "true" : "false")
        << ",\"checks\":" << checks << ",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Adaptive-tail worker CPU policy check failed: " << error.what() << '\n';
    return 1;
  }
}
