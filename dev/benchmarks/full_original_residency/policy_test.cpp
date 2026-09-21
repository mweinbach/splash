#include "Policy.hpp"

#include <iostream>
#include <string_view>

using namespace splash::flash::private_full_residency;

int main() {
  uint64_t checks = 0;
  const auto require = [&](bool result) {
    ++checks;
    if (!result) throw std::runtime_error("private full residency policy failed");
  };
  const auto rejects = [&](auto &&body) {
    bool rejected = false;
    try { body(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected);
  };
  require(!parseSwitch(nullptr));
  require(!parseSwitch("0"));
  require(parseSwitch("1"));
  for (const char *invalid : {"", "00", "01", " 1", "1 ", "2", "true", "-1"})
    rejects([&] { (void)parseSwitch(invalid); });
  for (bool full : {false, true}) for (bool text : {false, true}) {
    if (full && text) rejects([&] { validateFlags(full, text); });
    else { validateFlags(full, text); require(true); }
  }
  validateGeometry(kSource, kLayout, kOriginalBases, kOriginalBytes, 48, 512, 2560);
  require(true);
  rejects([&] { validateGeometry("wrong", kLayout, kOriginalBases, kOriginalBytes, 48, 512, 2560); });
  rejects([&] { validateGeometry(kSource, "wrong", kOriginalBases, kOriginalBytes, 48, 512, 2560); });
  for (uint64_t count : {0ULL, 13ULL, 20ULL, 22ULL, 3748ULL})
    rejects([&] { validateGeometry(kSource, kLayout, count, kOriginalBytes, 48, 512, 2560); });
  for (uint64_t bytes : {0ULL, kOriginalBytes - 1, kOriginalBytes + 1})
    rejects([&] { validateGeometry(kSource, kLayout, kOriginalBases, bytes, 48, 512, 2560); });
  rejects([&] { validateGeometry(kSource, kLayout, kOriginalBases, kOriginalBytes, 47, 512, 2560); });
  rejects([&] { validateGeometry(kSource, kLayout, kOriginalBases, kOriginalBytes, 48, 511, 2560); });
  rejects([&] { validateGeometry(kSource, kLayout, kOriginalBases, kOriginalBytes, 48, 512, 2559); });
  for (bool valid : {false, true}) for (bool growth : {false, true})
    for (bool normal : {false, true})
      for (uint64_t headroom : {0ULL, kOriginalBytes, kOriginalBytes + kMarginBytes - 1,
           kOriginalBytes + kMarginBytes, kOriginalBytes + kMarginBytes + 1, UINT64_MAX})
        require(hostAllowed(valid, growth, normal, headroom) ==
            (valid && growth && normal && headroom >= kOriginalBytes + kMarginBytes));
  std::cout << "{\"valid\":true,\"gpu_work\":false,\"checks\":" << checks << "}\n";
}
