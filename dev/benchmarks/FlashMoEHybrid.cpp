#include "FlashMoEHybrid.hpp"

#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace splash::flash::candidate {
namespace {
constexpr uint64_t kGuardBytes = 64;
void require(bool value, const char *reason) {
  if (!value) throw std::invalid_argument(reason);
}
template<class T> T sourceParams(const metal::ComputeDispatch &dispatch,
                                 uint32_t binding) {
  require(dispatch.bytes.size() == 1 && dispatch.bytes[0].index == binding &&
      dispatch.bytes[0].sizeBytes == sizeof(T) && dispatch.bytes[0].data,
      "Flash MoE hybrid parameter ABI differs");
  T value;
  std::memcpy(&value,dispatch.bytes[0].data,sizeof(value));
  return value;
}
metal::MetalBuffer guarded(metal::MetalBackend &backend, uint64_t bytes,
                           metal::MetalBuffer &allocation) {
  allocation = backend.allocateBuffer(bytes+kGuardBytes,metal::BufferStorage::Shared,
      "Flash hybrid private guarded jobs");
  std::memset(allocation.contents(),0xa5,bytes);
  std::memset(static_cast<std::byte *>(allocation.contents())+bytes,0x5a,kGuardBytes);
  return backend.view(allocation,0,bytes);
}
}

bool MoEHybridJobList::canariesClean() const {
  const std::array<uint64_t,3> bytes{513*4,4,uint64_t{capacity}*8};
  for (size_t i=0; i<bytes.size(); ++i) {
    const auto *tail = static_cast<const uint8_t *>(allocations[i].contents())+bytes[i];
    if (!std::all_of(tail,tail+kGuardBytes,[](uint8_t value) { return value == 0x5a; }))
      return false;
  }
  return true;
}

MoEHybridDispatches::MoEHybridDispatches(
    metal::MetalBackend &backend, std::span<const metal::ComputeDispatch> source) {
  uint32_t rows=0,selections=0,sourceTile=0,sourceCapacity=0;
  uint32_t prefixes=0,jobs=0,gates=0,downs=0;
  for (const auto &dispatch : source) {
    if (dispatch.pipelineName != "flash_moe_bucket_job_prefix") continue;
    const auto p=sourceParams<FlashMoEBucketParams>(dispatch,5);
    require(!rows,"Flash hybrid duplicate job prefix");
    require(p.rows && p.rows<=kFlashMoEBucketMaximumRows && p.selections &&
        p.selections<=10 && p.width==2560 && p.experts==512 &&
        p.routes==p.rows*p.selections && !p.reserved &&
        (p.tile_rows==16 || p.tile_rows==32 || p.tile_rows==64) &&
        p.job_capacity==(p.routes+p.tile_rows-1)/p.tile_rows+511,
        "Flash hybrid source geometry differs");
    rows=p.rows; selections=p.selections; sourceTile=p.tile_rows; sourceCapacity=p.job_capacity;
  }
  require(rows,"Flash hybrid source has no job prefix");
  const uint32_t routes=rows*selections;
  for (uint32_t i=0; i<3; ++i) {
    auto &list=lists_[i];
    list.tile=16u<<i;
    list.capacity=(routes+list.tile-1)/list.tile+511;
    // Cold M16 and M32 experts each generate at most one job. This bound is
    // independent of dynamic GPU counts and avoids launching inactive tails.
    list.launchCapacity=list.tile<64 ? std::min(512u,list.capacity) : list.capacity;
    list.offsets=guarded(backend,513*4,list.allocations[0]);
    list.count=guarded(backend,4,list.allocations[1]);
    list.jobs=guarded(backend,uint64_t{list.capacity}*8,list.allocations[2]);
  }
  auto own = [&](metal::ComputeDispatch &dispatch, const auto &params) {
    payloads_.emplace_back(sizeof(params));
    std::memcpy(payloads_.back().data(),&params,sizeof(params));
    dispatch.bytes[0].data=payloads_.back().data();
  };
  auto coherent = [&](uint32_t r,uint32_t s,uint32_t m,uint32_t capacity) {
    require(r==rows && s==selections && m==sourceTile && capacity==sourceCapacity,
        "Flash hybrid source phase geometry differs");
  };
  for (const auto &original : source) {
    const bool prefix=original.pipelineName=="flash_moe_bucket_job_prefix";
    const bool emit=original.pipelineName=="flash_moe_bucket_jobs";
    const bool gate=original.pipelineName.starts_with("flash_moe_q4x8_gate_up_");
    const bool down=original.pipelineName.starts_with("flash_moe_q4x8_down_scatter_");
    if (!prefix && !emit && !gate && !down) {
      dispatches_.push_back(original); continue;
    }
    if (prefix) ++prefixes;
    if (emit) ++jobs;
    if (gate) ++gates;
    if (down) ++downs;
    for (const auto &list : lists_) {
      auto dispatch=original;
      if (prefix || emit) {
        auto p=sourceParams<FlashMoEBucketParams>(original,5);
        coherent(p.rows,p.selections,p.tile_rows,p.job_capacity);
        require(dispatch.buffers.size()==5 && dispatch.threadsPerThreadgroup.x==256 &&
            dispatch.threadsPerThreadgroup.y==1 && dispatch.threadsPerThreadgroup.z==1,
            "Flash hybrid source job bindings differ");
        p.tile_rows=list.tile; p.job_capacity=list.capacity;
        dispatch.pipelineName=prefix ? "flash_moe_hybrid_bucket_job_prefix" : "flash_moe_hybrid_bucket_jobs";
        dispatch.threadgroups={prefix ? 1u : (list.capacity+255)/256,1,1};
        if (prefix) {
          dispatch.buffers[2].buffer=list.offsets; dispatch.buffers[3].buffer=list.count;
        } else {
          dispatch.buffers[1].buffer=list.offsets; dispatch.buffers[2].buffer=list.count;
          dispatch.buffers[3].buffer=list.jobs;
        }
        own(dispatch,p);
      } else {
        const std::string expected=std::string("flash_moe_q4x8_")+(gate ? "gate_up" : "down_scatter")+
            "_m"+std::to_string(sourceTile)+"_n64"+(sourceTile==64 ? "_sg8" : "");
        require(original.pipelineName==expected && dispatch.threadgroups.x==(gate ? 10u : 40u) &&
            dispatch.threadgroups.y==sourceCapacity && dispatch.threadgroups.z==1 &&
            dispatch.threadsPerThreadgroup.x==(sourceTile==64 ? 256u : 128u) &&
            dispatch.threadsPerThreadgroup.y==1 && dispatch.threadsPerThreadgroup.z==1 &&
            dispatch.buffers.size()==(gate ? 12u : 10u),
            "Flash hybrid source matrix geometry differs");
        dispatch.pipelineName=std::string("flash_moe_q4x8_")+(gate ? "gate_up" : "down_scatter")+
            "_m"+std::to_string(list.tile)+"_n64"+(list.tile==64 ? "_sg8" : "");
        dispatch.threadgroups.y=list.launchCapacity;
        dispatch.threadsPerThreadgroup.x=list.tile==64 ? 256u : 128u;
        if (gate) {
          auto p=sourceParams<FlashMoEBlockedGateParams>(original,12);
          coherent(p.affine.rows,p.affine.selections,p.tile_rows,p.job_capacity);
          p.tile_rows=list.tile; p.job_capacity=list.capacity;
          dispatch.buffers[8].buffer=list.jobs; dispatch.buffers[9].buffer=list.count;
          own(dispatch,p);
        } else {
          auto p=sourceParams<FlashMoEBlockedDownParams>(original,10);
          coherent(p.affine.rows,p.affine.selections,p.tile_rows,p.job_capacity);
          p.tile_rows=list.tile; p.job_capacity=list.capacity;
          dispatch.buffers[5].buffer=list.jobs; dispatch.buffers[6].buffer=list.count;
          own(dispatch,p);
        }
      }
      dispatches_.push_back(std::move(dispatch));
    }
  }
  require(prefixes==1 && jobs==1 && gates==1 && downs==1,
      "Flash hybrid source needs one complete expert chain");
}

} // namespace splash::flash::candidate
