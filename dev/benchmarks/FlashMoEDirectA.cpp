#include "FlashMoEDirectA.hpp"
#include "FlashMoEDirectAABI.h"

#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"

#include <cstring>
#include <stdexcept>
#include <string_view>

namespace splash::flash::candidate {
namespace {
void require(bool value, const char *reason) {
  if (!value) throw std::invalid_argument(reason);
}

template<class P> P parameters(const metal::ComputeDispatch &dispatch,
                              uint32_t index) {
  require(dispatch.bytes.size() == 1 && dispatch.bytes.front().index == index &&
          dispatch.bytes.front().sizeBytes == sizeof(P) && dispatch.bytes.front().data,
          "Flash direct A source parameter binding differs");
  P result;
  std::memcpy(&result, dispatch.bytes.front().data, sizeof(P));
  return result;
}

const metal::MetalBuffer &buffer(const metal::ComputeDispatch &dispatch,
                                uint32_t index) {
  require(dispatch.buffers.size() > index && dispatch.buffers[index].index == index &&
          bool(dispatch.buffers[index].buffer),
          "Flash direct A source buffer binding differs");
  return dispatch.buffers[index].buffer;
}

bool overlaps(const metal::MetalBuffer &left, const metal::MetalBuffer &right) {
  if (left.sameView(right)) return true;
  const auto a = reinterpret_cast<uintptr_t>(left.contents());
  const auto b = reinterpret_cast<uintptr_t>(right.contents());
  if (!a || !b) return false;
  return a <= b ? b - a < left.sizeBytes() : a - b < right.sizeBytes();
}

uint32_t tile(std::string_view name) {
  if (name.ends_with("_m8_n64")) return 8;
  if (name.ends_with("_m16_n64")) return 16;
  if (name.ends_with("_m32_n64")) return 32;
  if (name.ends_with("_m64_n64_sg8")) return 64;
  throw std::invalid_argument("Flash direct A unsupported source tile");
}

void geometry(const metal::ComputeDispatch &dispatch, uint32_t m, uint32_t nGroups,
              uint32_t routes, uint32_t jobs) {
  require(m == 8 || m == 16 || m == 32 || m == 64,
          "Flash direct A source M differs");
  require(routes && routes <= kFlashMoEBucketMaximumRows * 10 &&
          jobs == (routes + m - 1) / m + 511 &&
          dispatch.threadgroups.x == nGroups && dispatch.threadgroups.y == jobs &&
          dispatch.threadgroups.z == 1 &&
          dispatch.threadsPerThreadgroup.x == (m == 64 ? 256u : 128u) &&
          dispatch.threadsPerThreadgroup.y == 1 &&
          dispatch.threadsPerThreadgroup.z == 1,
          "Flash direct A source geometry differs");
}
} // namespace

MoEDirectACommands::MoEDirectACommands(std::span<const metal::ComputeDispatch> source,
                                      metal::MetalBuffer preparedDown,
                                      bool directGateUp, bool directDown,
                                      bool inplaceDown) {
  require(directGateUp || directDown, "Flash direct A must select at least one phase");
  unsigned foundPack = 0, foundGate = 0, foundDown = 0;
  uint32_t packedRoutes = 0;
  metal::MetalBuffer packedInput, activated;
  for (const auto &original : source) {
    auto dispatch = original;
    if (dispatch.pipelineName == "flash_moe_bucket_pack") {
      ++foundPack;
      const auto p = parameters<FlashMoEBucketParams>(dispatch, 5);
      packedRoutes = p.routes;
      require(foundPack == 1 && foundGate == 0 && foundDown == 0,
              "Flash direct A source pack order differs");
      require(p.rows && p.rows <= kFlashMoEBucketMaximumRows && p.selections &&
              p.selections <= 10 && p.routes == p.rows * p.selections &&
              p.width == 2560 && p.experts == 512 && !p.tile_rows &&
              !p.job_capacity && !p.reserved && dispatch.threadgroups.x == p.routes &&
              dispatch.threadgroups.y == 1 && dispatch.threadgroups.z == 1 &&
              dispatch.threadsPerThreadgroup.x == 256 &&
              dispatch.threadsPerThreadgroup.y == 1 &&
              dispatch.threadsPerThreadgroup.z == 1,
              "Flash direct A source pack geometry differs");
      if (directGateUp) {
        require(buffer(dispatch, 3).sizeBytes() >=
                    uint64_t(p.routes + kFlashMoEDirectAPaddingRows) * 2560 * 2,
                "Flash direct A packed input lacks 63 guard rows");
        dispatch.pipelineName = "flash_moe_direct_a_pack";
        dispatch.threadgroups.x += kFlashMoEDirectAPaddingRows;
      }
      packedInput = buffer(dispatch, 3);
    } else if (dispatch.pipelineName.starts_with("flash_moe_q4x8_gate_up_")) {
      ++foundGate;
      const auto p = parameters<FlashMoEBlockedGateParams>(dispatch, 12);
      require(foundPack == 1 && foundGate == 1 && foundDown == 0,
              "Flash direct A source gate order differs");
      const uint32_t m = tile(dispatch.pipelineName);
      geometry(dispatch, m, 10, p.route_capacity, p.job_capacity);
      require(p.tile_rows == m && p.route_capacity == packedRoutes &&
              p.affine.input_size == 2560 && p.affine.output_size == 640 &&
              p.affine.experts == 512 && p.affine.rows * p.affine.selections == packedRoutes,
              "Flash direct A source gate parameters differ");
      require(buffer(dispatch, 0).sameView(packedInput),
              "Flash direct A source gate does not consume pack output");
      activated = buffer(dispatch, 10);
      if (directGateUp) {
        require(buffer(dispatch, 0).sizeBytes() >=
                    uint64_t(packedRoutes + kFlashMoEDirectAPaddingRows) * 2560 * 2,
                "Flash direct A gate input lacks guard rows");
        dispatch.pipelineName.replace(0, std::strlen("flash_moe_q4x8"),
                                      "flash_moe_direct_a");
      }
    } else if (dispatch.pipelineName.starts_with("flash_moe_q4x8_down_scatter_")) {
      ++foundDown;
      const auto p = parameters<FlashMoEBlockedDownParams>(dispatch, 10);
      require(foundPack == 1 && foundGate == 1 && foundDown == 1,
              "Flash direct A source down order differs");
      const uint32_t m = tile(dispatch.pipelineName);
      geometry(dispatch, m, 40, p.route_capacity, p.job_capacity);
      require(p.tile_rows == m && p.route_capacity == packedRoutes &&
              p.affine.input_size == 640 && p.affine.output_size == 2560 &&
              p.affine.experts == 512 && p.affine.rows * p.affine.selections == packedRoutes,
              "Flash direct A source down parameters differ");
      require(buffer(dispatch, 0).sameView(activated),
              "Flash direct A source down does not consume gate output");
      if (directDown) {
        require(preparedDown && preparedDown.sizeBytes() >=
                    uint64_t(packedRoutes + kFlashMoEDirectAPaddingRows) * 640 * 2,
                "Flash direct A down operand lacks guard rows");
        for (const auto &binding : dispatch.buffers) {
          if (binding.index == 0 && inplaceDown) {
            require(preparedDown.sameView(binding.buffer),
                    "Flash direct A in-place preparation must use the exact source view");
            continue;
          }
          require(!overlaps(preparedDown, binding.buffer),
                  "Flash direct A down preparation aliases a source binding");
        }
        preparation_.add("flash_moe_direct_a_prepare_down",
            {buffer(dispatch, 0), buffer(dispatch, 4), preparedDown, buffer(dispatch, 9)},
            FlashMoEDirectAPrepareParams{packedRoutes, 640,
                                       kFlashMoEDirectAPaddingRows, 0},
            {packedRoutes + kFlashMoEDirectAPaddingRows, 1, 1}, {256, 1, 1});
        dispatches_.push_back(preparation_.dispatches().back());
        dispatch.buffers[0].buffer = preparedDown;
        dispatch.pipelineName.replace(0, std::strlen("flash_moe_q4x8"),
                                      "flash_moe_direct_a");
      }
    }
    dispatches_.push_back(std::move(dispatch));
  }
  require(foundPack == 1 && foundGate == 1 && foundDown == 1,
          "Flash direct A requires exactly one ordered pack/gate/down chain");
}

} // namespace splash::flash::candidate
