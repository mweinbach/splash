#include "flash/FlashGatheredMPP.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  using namespace splash::flash::gathered_mpp;
  const auto check = [](bool condition) {
    if (!condition) throw std::runtime_error("row-cap policy self-test failed");
  };
  unsetenv(kFlag);
  unsetenv(kMaximumRowsFlag);
  check(!requested() && requestedMaximumRows() == 4);
  for (const auto cap : {1u, 2u, 4u, 8u, 16u}) {
    setenv(kMaximumRowsFlag, std::to_string(cap).c_str(), 1);
    check(requestedMaximumRows() == cap);
    for (const auto rows : {1u, 2u, 3u, 4u, 8u, 15u, 16u, 17u, 2048u, 8192u}) {
      check((rows <= requestedMaximumRows()) == (rows <= cap));
    }
  }
  for (const auto invalid : {"", "0", "3", "5", "32", "04", " 4", "4 ", "-1", "4294967296"}) {
    setenv(kMaximumRowsFlag, invalid, 1);
    bool rejected = false;
    try { (void)requestedMaximumRows(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected);
  }
  unsetenv(kMaximumRowsFlag);
  setenv(kFlag, "1", 1);
  check(requested() && requestedMaximumRows() == 4);
  setenv(kFlag, "0", 1);
  check(!requested() && requestedMaximumRows() == 4);
  const auto one = geometry(1), sixteen = geometry(16);
  check(one.gateColumnGroups == 10 && one.downColumnGroups == 40);
  check(sixteen.gateColumnGroups == 10 && sixteen.downColumnGroups == 40);
  unsetenv(kFlag);
  std::cout << "{\"row_cap_policy_cpu_pass\":true,\"gpu_work\":false,\"payload_bytes_read\":0}\n";
}
