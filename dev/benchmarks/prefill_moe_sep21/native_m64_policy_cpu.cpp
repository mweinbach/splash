#include "flash/FlashMoEBlocked.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
using namespace splash::flash;
int main() {
  if (setenv("SPLASH_FLASH_MOE_M64","1",1) ||setenv("SPLASH_FLASH_MOE_Q4X8","1",1) ||
      setenv("SPLASH_FLASH_MOE_DIRECT_A","1",1)) return 1;
  const auto require=[](bool value) { if (!value) throw std::runtime_error("private native M64 row policy failed"); };
  require(flashMoEBlockedTile(1,false) ==FlashMoEBlockedTile::M16N64);
  require(flashMoEBlockedTile(512,false) ==FlashMoEBlockedTile::M16N64);
  require(flashMoEBlockedTile(1024,false) ==FlashMoEBlockedTile::M32N64);
  require(flashMoEBlockedTile(2048,false) ==FlashMoEBlockedTile::M64N64);
  require(flashMoEBlockedTile(4096,false) ==FlashMoEBlockedTile::M64N64);
  require(flashMoEBlockedTile(8192,false) ==FlashMoEBlockedTile::M64N64);
  require(flashMoEBlockedTile(2048,true) ==FlashMoEBlockedTile::M32N64);
  const auto planned=flashMoEBlockedWorkspacePlannedBytes(2048,10,16384);
  constexpr uint64_t routes=2048 *10, operands=routes +63, jobs=(routes +7) /8 +511;
  const std::array<uint64_t,10> sizes{512 *4,513 *4,routes *4,routes *4,operands *2560 *2,
    513 *4,4,jobs *8,operands *640 *2,routes *2560 *2};
  uint64_t expected=0;
  for (uint64_t size :sizes) expected +=(size +16383) &~uint64_t(16383);
  require(planned ==expected);
  std::cout <<"{\"pass\":true,\"gpu_work\":false,\"native_m64_at_2048\":true,\"direct_a_padding_rows\":63,\"unchanged_workspace_planned_bytes\":" <<planned <<"}\n";
}
