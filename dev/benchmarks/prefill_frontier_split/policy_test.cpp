#include "Diagnostic.hpp"

#include <iostream>

using namespace splash::flash::private_prefill_frontier;

int main() {
  uint64_t checks = 0;
  auto require = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("frontier CPU policy failed"); };
  require(!parseSwitch(nullptr)); require(!parseSwitch("0")); require(parseSwitch("1"));
  for (const char *raw : {"", "00", "01", "2", "true", " 1", "1 "}) {
    bool rejected = false;
    try { (void)parseSwitch(raw); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected);
  }
  for (bool enabled : {false, true}) for (bool verify : {false, true})
    for (uint32_t rows = 0; rows <= 2049; ++rows)
      for (uint32_t begin : {0u, 1u, 127u, 128u, 8192u})
        require(eligible(enabled, rows, begin, verify) ==
            (enabled && !verify && rows == 128 && begin == 0));
  require(sizeof(splash::metal::CommandTiming) == 200);
  std::cout << "{\"valid\":true,\"gpu_work\":false,\"checks\":" << checks << "}\n";
}
