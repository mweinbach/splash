#include "dev/benchmarks/moe_pointwise_sep21/bridge.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
namespace policy = splash::flash::pointwise_sep21;
using Route = policy::CombineRoute;

uint64_t checks = 0;

void require(bool condition, const std::string &message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

const char *routeName(Route route) {
  switch (route) {
  case Route::Original: return "Original";
  case Route::SimdSlots: return "SimdSlots";
  case Route::WholeRow: return "WholeRow";
  }
  return "invalid enum";
}

void checkRoute(uint32_t rows, uint32_t width, uint32_t experts,
                uint32_t selections, bool enabled, Route expected) {
  const Route observed = policy::combineRoute(rows, width, experts,
                                               selections, enabled);
  require(observed == expected,
          "combine rows=" + std::to_string(rows) +
              " width=" + std::to_string(width) +
              " experts=" + std::to_string(experts) +
              " selections=" + std::to_string(selections) +
              " enabled=" + std::to_string(enabled) +
              " expected=" + routeName(expected) +
              " observed=" + routeName(observed));
}

void parserChecks() {
  require(!policy::parseSwitch(nullptr), "unset switch must be disabled");
  require(!policy::parseSwitch("0"), "zero switch must be disabled");
  require(policy::parseSwitch("1"), "one switch must be enabled");
  constexpr std::array rejected{
      "", " ", "\t", "\n", "00", "01", "10", "11", "+0", "+1",
      "-0", "-1", "2", "255", "true", "false", "TRUE", "FALSE",
      "on", "off", "yes", "no", " 0", " 1", "0 ", "1 ", "0\n", "1\n"};
  for (const char *value : rejected) {
    bool threw = false;
    try {
      (void)policy::parseSwitch(value);
    } catch (...) {
      threw = true;
    }
    require(threw, "invalid switch must throw: [" + std::string(value) + "]");
  }
  // A rejected parse must not alter later direct parses.
  require(policy::parseSwitch("1"), "direct parse changed after rejection");
  require(!policy::parseSwitch("0"), "direct parse changed after rejection");
}

struct RowCase {
  uint32_t rows;
  Route combine;
  bool poison;
};

constexpr std::array rowCases{
    RowCase{0, Route::Original, false},
    RowCase{1, Route::SimdSlots, false},
    RowCase{4, Route::SimdSlots, false},
    RowCase{16, Route::SimdSlots, false},
    RowCase{17, Route::Original, false},
    RowCase{255, Route::Original, false},
    RowCase{256, Route::WholeRow, true},
    RowCase{2048, Route::WholeRow, true},
    RowCase{8192, Route::WholeRow, true},
    RowCase{8193, Route::Original, false},
    RowCase{std::numeric_limits<uint32_t>::max(), Route::Original, false}};

void routeChecks() {
  constexpr std::array<uint32_t, 6> widths{
      0, 1, 2559, 2560, 2561, std::numeric_limits<uint32_t>::max()};
  constexpr std::array<uint32_t, 7> experts{
      0, 1, 10, 511, 512, 513, std::numeric_limits<uint32_t>::max()};
  constexpr std::array<uint32_t, 6> selections{
      0, 1, 9, 10, 11, std::numeric_limits<uint32_t>::max()};
  for (const auto &row : rowCases) {
    for (const uint32_t width : widths) {
      for (const uint32_t expertCount : experts) {
        for (const uint32_t selectionCount : selections) {
          const bool canonical = width == 2560 && expertCount == 512 &&
                                 selectionCount == 10;
          checkRoute(row.rows, width, expertCount, selectionCount, true,
                     canonical ? row.combine : Route::Original);
          checkRoute(row.rows, width, expertCount, selectionCount, false,
                     Route::Original);
        }
      }
    }
  }
}

void poisonChecks() {
  for (const auto &row : rowCases) {
    for (uint32_t selections = 0; selections <= 11; ++selections) {
      const bool expected = row.poison && selections >= 1 && selections <= 10;
      require(policy::poisonEnabled(row.rows, selections, true) == expected,
              "poison rows=" + std::to_string(row.rows) +
                  " selections=" + std::to_string(selections));
      require(!policy::poisonEnabled(row.rows, selections, false),
              "disabled poison must retain original route");
    }
    require(!policy::poisonEnabled(row.rows,
                                  std::numeric_limits<uint32_t>::max(), true),
            "maximum uint32 selections must not enable poison");
    require(!policy::poisonEnabled(row.rows,
                                  std::numeric_limits<uint32_t>::max(), false),
            "disabled maximum uint32 selections must not enable poison");
  }
}

void markerChecks() {
  require(policy::marker(false).empty(), "disabled marker must be empty");
  constexpr std::string_view expected =
      ";private-moe-pointwise-exact-bf16-simdslots-r1to16-ctarow-r256plus-poison-sg1-r256plus-sep21-v1";
  require(policy::marker(true) == expected,
          "enabled marker must identify exact phase-separated pointwise policy");
}

void freezeChecks(const char *environmentKey, bool initiallyEnabled) {
  require(setenv(environmentKey, initiallyEnabled ? "1" : "0", 1) == 0,
          "cannot set initial switch");
  require(policy::requested() == initiallyEnabled,
          "requested did not parse initial environment value");
  require(setenv(environmentKey, "invalid-after-first-parse", 1) == 0,
          "cannot mutate switch to invalid value");
  require(policy::requested() == initiallyEnabled,
          "requested reparsed invalid environment value after first call");
  require(setenv(environmentKey, initiallyEnabled ? "0" : "1", 1) == 0,
          "cannot invert switch");
  require(policy::requested() == initiallyEnabled,
          "requested changed after valid switch inversion");
  require(unsetenv(environmentKey) == 0, "cannot unset switch");
  require(policy::requested() == initiallyEnabled,
          "requested changed after environment removal");
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::string_view mode = "default";
    if (argc == 1) {
      parserChecks();
      routeChecks();
      poisonChecks();
      markerChecks();
    } else {
      require(argc == 2, "usage: policy_cpu [--freeze0|--freeze1]");
      mode = argv[1];
      require(mode == "--freeze0" || mode == "--freeze1",
              "usage: policy_cpu [--freeze0|--freeze1]");
      freezeChecks("SPLASH_FLASH_MOE_POINTWISE_SEP21", mode == "--freeze1");
    }
    std::cout << "{\"pass\":true,\"gpu_work\":false,\"checks\":" << checks
              << ",\"mode\":\"" << mode << "\"}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pointwise policy CPU checks failed: " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "pointwise policy CPU checks failed: unknown exception\n";
    return 1;
  }
}
