#include "worker_bridge.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace splash::flash::gdn_wy_sep21;

namespace {
bool digest(std::string_view value) {
  return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string_view::npos;
}
uint64_t round16k(uint64_t bytes) { return ((bytes + 16383) / 16384) * 16384; }
}

int main(int argc, char **argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (argc > 2) throw std::invalid_argument("CPU policy accepts one mode");
    if (mode == "--freeze0" || mode == "--freeze1") {
      const bool want = mode == "--freeze1";
      setenv(kFlag, want ? "1" : "0", 1);
      setenv("SPLASH_FLASH_GDN_STAGED", "1", 1);
      if (requested() != want) throw std::runtime_error("WY initial selector mismatch");
      setenv(kFlag, want ? "0" : "1", 1);
      setenv("SPLASH_FLASH_GDN_STAGED", "0", 1);
      if (requested() != want) throw std::runtime_error("WY selector changed after freeze");
      std::cout << "WY strict selector freeze passed\n";
      return 0;
    }
    if (mode == "--invalid" || mode == "--missing-staged") {
      setenv(kFlag, mode == "--invalid" ? "bad" : "1", 1);
      setenv("SPLASH_FLASH_GDN_STAGED", mode == "--invalid" ? "1" : "0", 1);
      bool rejected = false;
      try { (void)requested(); } catch (const std::invalid_argument &) { rejected = true; }
      if (!rejected) throw std::runtime_error("WY invalid selector/dependency accepted");
      std::cout << "WY invalid selector/dependency rejected before backend\n";
      return 0;
    }
    if (!mode.empty()) throw std::invalid_argument("Unknown CPU policy mode");
    uint64_t checks = 0;
    const auto require = [&](bool value, const char *why) {
      if (!value) throw std::runtime_error(why);
      ++checks;
    };
    require(std::string_view(kFlag) == "SPLASH_FLASH_GDN_PREFILL_WY_SEP21", "WY flag spelling");
    for (const char *staged : std::initializer_list<const char *>{nullptr, "0", "1", "bad"}) {
      require(!detail::parseRequested(nullptr, staged), "WY missing must be disabled");
      require(!detail::parseRequested("0", staged), "WY zero must preserve inherited policy");
    }
    require(detail::parseRequested("1", "1"), "WY one plus staged one must enable");
    for (const char *bad : {"", "true", "false", " 1", "1 ", "01", "-1", "2", "\n1"}) {
      bool rejected = false;
      try { (void)detail::parseRequested(bad, "1"); } catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "WY selector is not strict 0|1");
    }
    for (const char *staged : std::initializer_list<const char *>{nullptr, "", "0", "true", "01", " 1", "1 ", "2"}) {
      bool rejected = false;
      try { (void)detail::parseRequested("1", staged); } catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "WY enabled without exact staged=1");
    }
    for (uint32_t rows = 0; rows <= 2112; ++rows)
      for (uint32_t lanes = 0; lanes <= 33; ++lanes)
        for (bool selected : {false, true}) for (bool verify : {false, true})
          require(eligible(rows, lanes, selected, verify) ==
              (selected && !verify && lanes == 1 && rows >= 64 && rows <= 2048),
              "WY eligibility escapes singleton nonverification rows64..2048");
    for (uint32_t rows : {0u, 1u, 63u, 64u, 65u, 2047u, 2048u, 2049u, 4096u,
                         std::numeric_limits<uint32_t>::max()})
      for (bool selected : {false, true}) {
        const bool provisioned = selected && rows >= 64;
        require(provisioning(rows, selected) == provisioned, "WY provisioning threshold differs");
        require(plannedBytes(rows, selected) == (provisioned ? 167133184ULL : 0ULL),
                "WY physical fixed2048/1 planner differs");
      }
    require(kCoefficientsBytes == 64ULL * 48 * (3 * 32 * 128 + 32 * 32 + 32) * 4,
            "WY coefficient layout differs");
    require(kSnapshotBytes == 48ULL * 128 * 128 * 4, "WY snapshot layout differs");
    require(kFlagsBytes == 48ULL * 4, "WY guard flag layout differs");
    require(kCoefficientsBytes + kSnapshotBytes + kFlagsBytes == 167116992ULL,
            "WY logical ledger differs");
    require(round16k(kCoefficientsBytes) + round16k(kSnapshotBytes) + round16k(kFlagsBytes) == kArenaBytes &&
            kArenaBytes == 167133184ULL, "WY physical allocation ledger differs");
    require(kCoefficientsBytes % 16384 == 0 && kSnapshotBytes % 16384 == 0 &&
            kArenaBytes - kCoefficientsBytes - kSnapshotBytes - kFlagsBytes == 16192,
            "WY alignment and unused tail differ");
    for (std::string_view hash : {kCandidateSourceSHA256, kNativeFallbackSourceSHA256,
                                 kSnapshotSourceSHA256, kGuardProofSHA256})
      require(digest(hash), "WY static identity source pin is not SHA256");
    require(!std::string_view(kPolicy).empty() && !std::string_view(kMarker).empty(),
            "WY policy/route marker absent");
    require(std::string(selectionMarker(true)).find(kMarker) != std::string::npos &&
            std::string(selectionMarker(false)) != std::string(selectionMarker(true)),
            "WY selected marker differs");
    const std::string baseA(64, 'a'), baseB(64, 'b');
    require(numericalIdentity(baseA, false) == baseA && numericalIdentity(baseB, false) == baseB,
            "WY zero changes inherited native/FMA numerical identity");
    const auto activeA = numericalIdentity(baseA, true), activeB = numericalIdentity(baseB, true);
    require(digest(activeA) && digest(activeB) && activeA != activeB && activeA != baseA,
            "WY active numerical identity fails to bind base/policy");
    require(numericalIdentity(baseA, true) == activeA && numericalIdentity(baseA, false) == baseA,
            "WY numerical identity is not deterministic");
    require(digest(workspaceIdentity()) && workspaceIdentity() == workspaceIdentity(),
            "WY fixed workspace identity absent or unstable");
    std::cout << "{\"gdn_wy_worker_cpu_policy\":\"passed\",\"checks\":" << checks
              << ",\"workspace_rows\":2048,\"workspace_lanes\":1,\"logical_bytes\":167116992"
              << ",\"workspace_planned_bytes\":" << kArenaBytes
              << ",\"numerical_identity_enabled\":\"" << activeA
              << "\",\"workspace_identity\":\"" << workspaceIdentity()
              << "\",\"gpu_work\":false,\"payload_reads\":false,\"worker_invoked\":false}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
