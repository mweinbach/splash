#include "Policy.hpp"
#include "metal/MetalBackend.hpp"
#include <array>
#include <iostream>
#include <limits>
using namespace splash::metal::private_idle_residency;
int main() {
  uint64_t checks = 0;
  auto require = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("idle refresh CPU policy failed"); };
  require(!parseSwitch(nullptr)); require(!parseSwitch("0")); require(parseSwitch("1"));
  for (const char *raw : {"", "00", "01", "2", "true", " 1", "1 "}) {
    bool rejected = false;
    try { (void)parseSwitch(raw); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected);
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (bool enabled : {false,true}) for (bool set : {false,true})
    for (bool stopping : {false,true}) for (bool outstanding : {false,true})
      for (double last : {-1.0,0.0,1.0,100.0,nan,inf})
        for (double now : {-1.0,0.0,1.0,6.0,6.0001,105.0,105.0001,200.0,nan,inf})
          require(shouldRefresh(enabled,set,stopping,outstanding,last,now) ==
              (enabled && set && !stopping && !outstanding && std::isfinite(last) &&
              std::isfinite(now) && last > 0.0 && now-last > 5.0));
  require(!shouldRefresh(true,true,false,false,10.0,15.0));
  require(shouldRefresh(true,true,false,false,10.0,15.0001));
  require(!shouldRefresh(true,true,false,true,10.0,100.0));
  require(sizeof(splash::metal::CommandTiming) == 200);
  std::cout << "{\"valid\":true,\"gpu_work\":false,\"checks\":" << checks << "}\n";
}
