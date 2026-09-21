#include "FlashInt8PersistentJobs.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"

#include <array>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace splash::flash::candidate {
namespace {
void require(bool value, const char *reason) {
  if (!value) throw std::invalid_argument(reason);
}
template<class P> P params(const metal::ComputeDispatch &d, uint32_t index) {
  require(d.bytes.size() == 1 && d.bytes[0].index == index &&
          d.bytes[0].sizeBytes == sizeof(P) && d.bytes[0].data,
          "persistent INT8 source parameter binding differs");
  require(d.buffers.size() == index,
          "persistent INT8 source buffer binding count differs");
  for (uint32_t i = 0; i < index; ++i)
    require(d.buffers[i].index == i, "persistent INT8 source buffer index differs");
  P p;
  std::memcpy(&p, d.bytes[0].data, sizeof(P));
  return p;
}
uint32_t tile(std::string_view name) {
  if (name.ends_with("_m16_n64")) return 16;
  if (name.ends_with("_m32_n64")) return 32;
  if (name.ends_with("_m64_n64_sg8")) return 64;
  throw std::invalid_argument("persistent INT8 source tile unsupported");
}
void geometry(const metal::ComputeDispatch &d, uint32_t m, uint32_t rows,
              uint32_t selections, uint32_t routes, uint32_t jobs, bool gate) {
  require(rows && rows <= 8192 && selections && selections <= 10 &&
          routes == rows * selections && jobs == (routes + m - 1) / m + 511 &&
          d.threadgroups.x == (gate ? 10u : 40u) && d.threadgroups.y == jobs &&
          d.threadgroups.z == 1 && d.threadsPerThreadgroup.x == (m == 64 ? 256u : 128u) &&
          d.threadsPerThreadgroup.y == 1 && d.threadsPerThreadgroup.z == 1,
          "persistent INT8 source geometry differs");
}
}

std::vector<metal::ComputeDispatch> persistentInt8JobDispatches(
    std::span<const metal::ComputeDispatch> source, uint32_t gateGridY,
    uint32_t downGridY, bool selectGate, bool selectDown) {
  require(selectGate || selectDown, "persistent INT8 must select a phase");
  require(gateGridY && gateGridY <= 128 && downGridY && downGridY <= 128,
          "persistent INT8 grids must be1..128");
  std::vector<metal::ComputeDispatch> result(source.begin(), source.end());
  std::array<unsigned, 4> phases{};
  for (auto &d : result) {
    constexpr std::string_view prefix = "flash_int8_expert_store_";
    if (!d.pipelineName.starts_with(prefix)) continue;
    const std::string_view phase = std::string_view(d.pipelineName).substr(prefix.size());
    const bool gateMiss = phase.starts_with("gate_up_miss_direct_");
    const bool downMiss = phase.starts_with("down_miss_direct_");
    const bool gateHit = !gateMiss && phase.starts_with("gate_up_m");
    const bool downHit = phase.starts_with("down_scatter_m");
    if (!gateMiss && !downMiss && !gateHit && !downHit) {
      require(!phase.starts_with("gate_up_miss_") && !phase.starts_with("down_miss_"),
              "persistent INT8 candidate requires the qualified direct Q4 miss route");
      continue;
    }
    const uint32_t m = tile(d.pipelineName);
    const bool gate = gateHit || gateMiss;
    if (gateHit || downHit) {
      ++phases[gateHit ? 0 : 2];
      const auto p = params<FlashInt8ExpertStoreParams>(d, gateHit ? 11 : 10);
      require(p.tile_rows == m && p.stored_experts && p.stored_experts <= 128 &&
              !p.scale_group_size && !p.reserved,
              "persistent INT8 hit source policy differs");
      geometry(d,m,p.rows,p.selections,p.route_capacity,p.job_capacity,gate);
    } else if (gateMiss) {
      ++phases[1];
      const auto p = params<FlashInt8ExpertStoreGateParams>(d,13);
      require(p.blocked.tile_rows == m && p.stored_experts && p.stored_experts <= 128 &&
              p.flags == 1 && !p.reserved0 && !p.reserved1 && !p.blocked.reserved &&
              p.blocked.affine.input_size == 2560 && p.blocked.affine.output_size == 640,
              "persistent INT8 gate miss source policy differs");
      geometry(d,m,p.blocked.affine.rows,p.blocked.affine.selections,
               p.blocked.route_capacity,p.blocked.job_capacity,true);
    } else {
      ++phases[3];
      const auto p = params<FlashInt8ExpertStoreDownParams>(d,11);
      require(p.blocked.tile_rows == m && p.stored_experts && p.stored_experts <= 128 &&
              p.flags == 1 && !p.reserved0 && !p.reserved1 && !p.blocked.reserved &&
              p.blocked.affine.input_size == 640 && p.blocked.affine.output_size == 2560,
              "persistent INT8 down miss source policy differs");
      geometry(d,m,p.blocked.affine.rows,p.blocked.affine.selections,
               p.blocked.route_capacity,p.blocked.job_capacity,false);
    }
    if ((gate && !selectGate) || (!gate && !selectDown)) continue;
    d.pipelineName.replace(0,prefix.size(),"flash_int8_expert_persistent_");
    d.threadgroups.y = std::min<uint64_t>(d.threadgroups.y,gate ? gateGridY : downGridY);
  }
  require(phases == std::array<unsigned,4>{1,1,1,1},
          "persistent INT8 expects exactly one complete v5 hit/miss producer chain");
  return result;
}
} // namespace splash::flash::candidate
