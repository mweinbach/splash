#pragma once
#include "flash/FlashMoEBlocked.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"
#include "metal/abi/FlashMoEBuckets.h"
#include <array>
#include <deque>
#include <cstring>
#include <stdexcept>

namespace splash::flash::prefill4k_int8tiles {
class HitCommands {
 public:
  metal::CommandGraph preparation;
  std::vector<metal::ComputeDispatch> commands;
  std::array<metal::MetalBuffer,3> allocations;
  metal::MetalBuffer offsets,count,jobs;
  std::deque<std::array<std::byte,sizeof(FlashInt8ExpertStoreParams)>> payloads;
  uint32_t capacity =0,tile =0;
  HitCommands() =default;
  HitCommands(const HitCommands &) =delete;
  HitCommands &operator=(const HitCommands &) =delete;
  HitCommands(HitCommands &&) =default;
  HitCommands &operator=(HitCommands &&) =default;
  HitCommands(metal::MetalBackend &backend,std::span<const metal::ComputeDispatch> source,
      uint32_t rows,uint32_t candidateTile,uint32_t sg,bool relaxed =false) :tile(candidateTile) {
    if ((tile !=32 && tile !=64 && tile !=128) ||
        (tile ==32 && sg !=4) || (tile ==64 && sg !=8) || (tile ==128 && sg !=8 && sg !=16))
      throw std::invalid_argument("Hit tile32/64/128 and valid SIMD grouping required");
    if (!rows || rows >8192) throw std::invalid_argument("Hit row capacity unsupported");
    capacity =(rows *10 +tile -1) /tile +511;
    const auto guarded =[&](uint32_t i,uint64_t logical) {
      allocations[i] =backend.allocateBuffer(logical +64,metal::BufferStorage::Shared,"private hit-only tile jobs");
      std::memset(allocations[i].contents(),0xa5,logical);
      std::memset(static_cast<std::byte *>(allocations[i].contents()) +logical,0x5a,64);
      return backend.view(allocations[i],0,logical);
    };
    offsets =guarded(0,513 *4); count =guarded(1,4); jobs =guarded(2,uint64_t(capacity) *8);
    bool prepared =false,gateSeen =false,downSeen =false;
    metal::MetalBuffer bucketOffsets;
    for (const auto &original :source) {
      auto d =original;
      const bool gate =d.pipelineName.starts_with("flash_int8_expert_store_gate_up_m") &&
          !d.pipelineName.starts_with("flash_int8_expert_store_gate_up_miss_");
      const bool down =d.pipelineName.starts_with("flash_int8_expert_store_down_scatter_");
      if (!gate && !down) { commands.push_back(d); continue; }
      if (d.bytes.size() !=1 || d.bytes[0].sizeBytes !=sizeof(FlashInt8ExpertStoreParams) || !d.bytes[0].data)
        throw std::invalid_argument("Hit source inline parameter differs");
      FlashInt8ExpertStoreParams p; std::memcpy(&p,d.bytes[0].data,sizeof(p));
      if (p.rows !=rows || p.selections !=10 || p.route_capacity !=rows *10 || p.scale_group_size || p.reserved ||
          p.job_capacity !=(p.route_capacity +p.tile_rows -1) /p.tile_rows +511)
        throw std::invalid_argument("Hit source geometry differs");
      if (gate) {
        if (gateSeen || downSeen) throw std::invalid_argument("Hit gate order differs");
        gateSeen =true;
        const auto &ranks =d.buffers[5].buffer;
        bucketOffsets =d.buffers[6].buffer;
        // The immediately preceding original bucket producer owns its counts.
        const metal::MetalBuffer *counts =nullptr;
        for (const auto &sourceDispatch :source)
          if (sourceDispatch.pipelineName =="flash_moe_bucket_job_prefix") counts =&sourceDispatch.buffers[0].buffer;
        if (!counts) throw std::invalid_argument("Hit list requires original validated counts");
        const FlashMoEBucketParams params{rows,10,2560,512,rows *10,tile,capacity,0};
        preparation.add("prefill4k_int8tiles_hit_prefix",{*counts,bucketOffsets,ranks,offsets,count,d.buffers[10].buffer},
            params,{1,1,1},{256,1,1});
        preparation.add("prefill4k_int8tiles_hit_jobs",{bucketOffsets,offsets,count,jobs,d.buffers[10].buffer},
            params,{(capacity +255) /256,1,1},{256,1,1});
        commands.insert(commands.end(),preparation.dispatches().begin(),preparation.dispatches().end()); prepared =true;
        d.buffers[7].buffer =jobs; d.buffers[8].buffer =count;
      } else {
        if (!prepared || downSeen || !d.buffers[4].buffer.sameView(bucketOffsets))
          throw std::invalid_argument("Hit down order/offsets differ");
        downSeen =true; d.buffers[5].buffer =jobs; d.buffers[6].buffer =count;
      }
      p.tile_rows =tile; p.job_capacity =capacity;
      payloads.emplace_back(); std::memcpy(payloads.back().data(),&p,sizeof(p));
      d.bytes[0].data =payloads.back().data();
      d.pipelineName =std::string(relaxed ? "prefill4k_int8relaxed_" :"prefill4k_int8tiles_") +(gate ? "gate_up" :"down_scatter") +"_m" +
          std::to_string(tile) +"_n64_sg" +std::to_string(sg);
      d.threadgroups.y =capacity; d.threadsPerThreadgroup.x =sg *32;
      commands.push_back(d);
    }
    if (!gateSeen || !downSeen) throw std::invalid_argument("Hit source missing unique producers");
  }
  bool canaries() const {
    const std::array<uint64_t,3> logical{513 *4,4,uint64_t(capacity) *8};
    for (uint32_t i =0; i <3; ++i) {
      const auto *tail =static_cast<const uint8_t *>(allocations[i].contents()) +logical[i];
      for (uint32_t j =0; j <64; ++j) if (tail[j] !=0x5a) return false;
    }
    return true;
  }
};
} // namespace splash::flash::prefill4k_int8tiles
