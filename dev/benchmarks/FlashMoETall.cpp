#include "FlashMoETall.hpp"

#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace splash::flash::candidate {
namespace {
template<class T> T sourceParams(const metal::ComputeDispatch &dispatch,
                                 uint32_t binding) {
  if (dispatch.bytes.size() != 1 || dispatch.bytes[0].index != binding ||
      dispatch.bytes[0].sizeBytes != sizeof(T) || !dispatch.bytes[0].data)
    throw std::invalid_argument("Flash MoE tall parameter ABI differs");
  T value;
  std::memcpy(&value, dispatch.bytes[0].data, sizeof(value));
  return value;
}
void sourceMatrix(const metal::ComputeDispatch &dispatch, bool gate,
                  uint32_t tile, uint32_t capacity) {
  const std::string suffix = "_m" + std::to_string(tile) + "_n64";
  if ((tile != 8 && tile != 16 && tile != 32) ||
      !dispatch.pipelineName.ends_with(suffix) ||
      dispatch.threadgroups.x != (gate ? 10u : 40u) ||
      dispatch.threadgroups.y != capacity || dispatch.threadgroups.z != 1 ||
      dispatch.threadsPerThreadgroup.x != 128 ||
      dispatch.threadsPerThreadgroup.y != 1 ||
      dispatch.threadsPerThreadgroup.z != 1)
    throw std::invalid_argument("Flash MoE tall source matrix geometry differs");
}
} // namespace

uint32_t moETallJobCapacity(uint32_t rows, uint32_t selections) {
  if (!rows || rows > kFlashMoEBucketMaximumRows || !selections ||
      selections > kFlashMoEBucketMaximumSelections)
    throw std::invalid_argument("Flash MoE tall job geometry differs");
  return (rows * selections + 63) / 64 + 511;
}

MoETallDispatches::MoETallDispatches(
    std::span<const metal::ComputeDispatch> source, uint32_t simdgroups)
    : dispatches_(source.begin(), source.end()) {
  if (simdgroups != 4 && simdgroups != 8)
    throw std::invalid_argument("Flash MoE tall SIMD groups must be4 or8");
  uint32_t foundPrefix = 0, foundJobs = 0, foundGate = 0, foundDown = 0;
  uint32_t coherentRows = 0, coherentSelections = 0;
  auto own = [&](metal::ComputeDispatch &dispatch, const auto &params) {
    payloads_.emplace_back(sizeof(params));
    std::memcpy(payloads_.back().data(), &params, sizeof(params));
    dispatch.bytes[0].data = payloads_.back().data();
  };
  auto geometry = [&](uint32_t rows, uint32_t selections, uint32_t tile,
                      uint32_t capacity) {
    const uint32_t tallCapacity = moETallJobCapacity(rows, selections);
    if ((tile != 8 && tile != 16 && tile != 32) ||
        capacity != (rows * selections + tile - 1) / tile + 511)
      throw std::invalid_argument("Flash MoE tall source job extent differs");
    if (coherentRows && (rows != coherentRows || selections != coherentSelections))
      throw std::invalid_argument("Flash MoE tall source phase extents differ");
    coherentRows = rows; coherentSelections = selections;
    return tallCapacity;
  };
  for (auto &dispatch : dispatches_) {
    const bool prefix = dispatch.pipelineName == "flash_moe_bucket_job_prefix";
    const bool jobs = dispatch.pipelineName == "flash_moe_bucket_jobs";
    if (prefix || jobs) {
      auto p = sourceParams<FlashMoEBucketParams>(dispatch, 5);
      const uint32_t capacity = geometry(p.rows, p.selections, p.tile_rows, p.job_capacity);
      if (p.width != 2560 || p.experts != 512 || p.routes != p.rows * p.selections ||
          p.reserved || dispatch.threadsPerThreadgroup.x != 256 ||
          dispatch.threadsPerThreadgroup.y != 1 || dispatch.threadsPerThreadgroup.z != 1 ||
          dispatch.threadgroups.x != (prefix ? 1u : (p.job_capacity + 255) / 256) ||
          dispatch.threadgroups.y != 1 || dispatch.threadgroups.z != 1)
        throw std::invalid_argument("Flash MoE tall source job geometry differs");
      p.tile_rows = 64; p.job_capacity = capacity;
      dispatch.pipelineName = prefix ? "flash_moe_tall_bucket_job_prefix" : "flash_moe_tall_bucket_jobs";
      dispatch.threadgroups.x = prefix ? 1u : (capacity + 255) / 256;
      own(dispatch, p);
      if (prefix) ++foundPrefix; else ++foundJobs;
    } else if (dispatch.pipelineName.starts_with("flash_moe_q4x8_gate_up_")) {
      auto p = sourceParams<FlashMoEBlockedGateParams>(dispatch, 12);
      const uint32_t capacity = geometry(p.affine.rows, p.affine.selections, p.tile_rows, p.job_capacity);
      sourceMatrix(dispatch, true, p.tile_rows, p.job_capacity);
      p.tile_rows = 64; p.job_capacity = capacity;
      dispatch.pipelineName = "flash_moe_tall_gate_up_m64_n64_sg" + std::to_string(simdgroups);
      dispatch.threadgroups.y = capacity; dispatch.threadsPerThreadgroup.x = simdgroups * 32;
      own(dispatch, p); ++foundGate;
    } else if (dispatch.pipelineName.starts_with("flash_moe_q4x8_down_scatter_")) {
      auto p = sourceParams<FlashMoEBlockedDownParams>(dispatch, 10);
      const uint32_t capacity = geometry(p.affine.rows, p.affine.selections, p.tile_rows, p.job_capacity);
      sourceMatrix(dispatch, false, p.tile_rows, p.job_capacity);
      p.tile_rows = 64; p.job_capacity = capacity;
      dispatch.pipelineName = "flash_moe_tall_down_scatter_m64_n64_sg" + std::to_string(simdgroups);
      dispatch.threadgroups.y = capacity; dispatch.threadsPerThreadgroup.x = simdgroups * 32;
      own(dispatch, p); ++foundDown;
    }
  }
  if (foundPrefix != 1 || foundJobs != 1 || foundGate != 1 || foundDown != 1)
    throw std::invalid_argument("Flash MoE tall expects one complete blocked expert chain");
}

} // namespace splash::flash::candidate
