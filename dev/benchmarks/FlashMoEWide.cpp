#include "FlashMoEWide.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace splash::flash::candidate {

std::vector<metal::ComputeDispatch> moEWideDispatches(
    std::span<const metal::ComputeDispatch> source, bool wideGateUp,
    bool wideDown) {
  if (!wideGateUp && !wideDown)
    throw std::invalid_argument("Flash MoE wide candidate must select a phase");
  std::vector<metal::ComputeDispatch> result(source.begin(), source.end());
  unsigned foundGate = 0, foundDown = 0;
  for (auto &dispatch : result) {
    const bool gate = dispatch.pipelineName.starts_with("flash_moe_q4x8_gate_up_");
    const bool down = dispatch.pipelineName.starts_with("flash_moe_q4x8_down_scatter_");
    if (!gate && !down) continue;
    if (gate) ++foundGate;
    if (down) ++foundDown;
    const auto &name = dispatch.pipelineName;
    const bool supported = name.ends_with("_m8_n64") ||
        name.ends_with("_m16_n64") || name.ends_with("_m32_n64");
    if (!supported || dispatch.threadgroups.x != (gate ? 10u : 40u) ||
        dispatch.threadgroups.z != 1 || dispatch.threadsPerThreadgroup.x != 128 ||
        dispatch.threadsPerThreadgroup.y != 1 ||
        dispatch.threadsPerThreadgroup.z != 1)
      throw std::invalid_argument("Flash MoE wide source geometry differs");
    if ((gate && !wideGateUp) || (down && !wideDown)) continue;
    dispatch.pipelineName.replace(0, std::strlen("flash_moe_q4x8"), "flash_moe_wide");
    dispatch.pipelineName.replace(dispatch.pipelineName.size() - 2, 2, "128");
    dispatch.threadgroups.x /= 2;
  }
  if (foundGate != 1 || foundDown != 1)
    throw std::invalid_argument("Flash MoE wide expects one gate/up and one down phase");
  return result;
}

} // namespace splash::flash::candidate
