#pragma once
#include "dev/benchmarks/prefill4k_q4coded/params.h"
#include "metal/CommandGraph.hpp"
#include <array>
#include <deque>
#include <cstring>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace splash::flash::prefill4k_q4coded {
class Commands {
 public:
  metal::CommandGraph sums;
  std::vector<metal::ComputeDispatch> commands;
  std::array<metal::MetalBuffer,5> allocations,values;
  std::deque<std::vector<std::byte>> payloads;
  uint32_t rows =0;
  Commands() =default;
  Commands(const Commands &) =delete;
  Commands &operator=(const Commands &) =delete;
  Commands(Commands &&) =default;
  Commands &operator=(Commands &&) =default;
  Commands(metal::MetalBackend &backend,std::span<const metal::ComputeDispatch> source,uint32_t rowCount,bool all =false) :rows(rowCount) {
    if (!rows || rows >8192) throw std::invalid_argument("Q4coded row count unsupported");
    for (uint32_t i =0; i <2; ++i) {
      const uint64_t bytes =uint64_t(rows) *10 *(i ? 10 :40) *4;
      allocations[i] =backend.allocateBuffer(bytes +64,metal::BufferStorage::Shared,"Q4coded guarded input sums");
      std::fill_n(static_cast<float *>(allocations[i].contents()),bytes /4,std::numeric_limits<float>::quiet_NaN());
      std::memset(static_cast<std::byte *>(allocations[i].contents()) +bytes,0x5a,64);
      values[i] =backend.view(allocations[i],0,bytes);
    }
    for (uint32_t i =2; i <5; ++i) {
      const uint64_t bytes =uint64_t(rows) *10 *(i ==4 ? 2560 :640) *4;
      allocations[i] =backend.allocateBuffer(bytes +64,metal::BufferStorage::Shared,"Q4coded guarded linear audit");
      std::fill_n(static_cast<float *>(allocations[i].contents()),bytes /4,std::numeric_limits<float>::quiet_NaN());
      std::memset(static_cast<std::byte *>(allocations[i].contents()) +bytes,0x5a,64);
      values[i] =backend.view(allocations[i],0,bytes);
    }
    bool gateSeen =false,downSeen =false;
    for (const auto &original :source) {
      auto d =original;
      const bool gate =d.pipelineName.starts_with("flash_int8_expert_store_gate_up_miss_direct_");
      const bool down =d.pipelineName.starts_with("flash_int8_expert_store_down_miss_direct_");
      const bool hit =(d.pipelineName.starts_with("flash_int8_expert_store_gate_up_m") && !d.pipelineName.starts_with("flash_int8_expert_store_gate_up_miss_")) || d.pipelineName.starts_with("flash_int8_expert_store_down_scatter_");
      if (all && hit) continue;
      if (!gate && !down) { commands.push_back(d);continue; }
      if (d.bytes.size() !=1 || !d.bytes[0].data) throw std::invalid_argument("Q4coded source inline bytes differ");
      const uint32_t i =down ? 1 :0;
      const uint32_t expectedBytes =gate ? sizeof(FlashInt8ExpertStoreGateParams) :sizeof(FlashInt8ExpertStoreDownParams);
      if (d.bytes[0].sizeBytes !=expectedBytes || d.bytes[0].index !=(gate ? 13u :11u))
        throw std::invalid_argument("Q4coded original miss ABI differs");
      uint32_t capacity =0;
      if (gate) {
        Prefill4KQ4CodedGateParams p{};std::memcpy(&p.source,d.bytes[0].data,sizeof(p.source));p.selector =all;
        if (p.source.blocked.tile_rows !=32 || p.source.flags !=1 || p.source.blocked.affine.rows !=rows || gateSeen || downSeen)
          throw std::invalid_argument("Q4coded requires unique M32 Direct-A gate miss");
        capacity =p.source.blocked.job_capacity;gateSeen =true;
        payloads.emplace_back(sizeof(p));std::memcpy(payloads.back().data(),&p,sizeof(p));
      } else {
        Prefill4KQ4CodedDownParams p{};std::memcpy(&p.source,d.bytes[0].data,sizeof(p.source));p.selector =all;
        if (p.source.blocked.tile_rows !=32 || p.source.flags !=1 || p.source.blocked.affine.rows !=rows || !gateSeen || downSeen)
          throw std::invalid_argument("Q4coded requires unique M32 Direct-A down miss");
        capacity =p.source.blocked.job_capacity;downSeen =true;
        payloads.emplace_back(sizeof(p));std::memcpy(payloads.back().data(),&p,sizeof(p));
      }
      const Prefill4KQ4CodedSumParams sumParams{rows *10,down ? 640u :2560u,down ? 10u :40u,0};
      sums.add("prefill4k_q4coded_input_sums",{d.buffers[0].buffer,d.buffers[down ? 4 :7].buffer,values[i],d.buffers[down ? 9 :11].buffer},
          sumParams,{(uint64_t(sumParams.routes) *sumParams.groups +255) /256,1,1},{256,1,1});
      commands.push_back(sums.dispatches().back());
      d.buffers.push_back({down ? 11u :13u,values[i]});
      if (down) d.buffers.push_back({12,values[4]});
      else { d.buffers.push_back({14,values[2]});d.buffers.push_back({15,values[3]}); }
      d.bytes[0] ={down ? 13u :16u,payloads.back().data(),payloads.back().size()};
      d.pipelineName =gate ? "prefill4k_q4coded_gate_up_m32_n64" :"prefill4k_q4coded_down_scatter_m32_n64";
      d.threadgroups ={gate ? 10u :40u,capacity,1};d.threadsPerThreadgroup ={gate ? 128u :32u,1,1};commands.push_back(d);
    }
    if (!gateSeen || !downSeen) throw std::invalid_argument("Q4coded source missing sorted expert miss producers");
  }
  bool canaries() const {
    for (uint32_t i =0; i <5; ++i) {
      const uint64_t bytes =uint64_t(rows) *10 *(i <2 ? (i ? 10 :40) : i ==4 ? 2560 :640) *4;
      const auto *tail =static_cast<const uint8_t *>(allocations[i].contents()) +bytes;
      for (uint32_t j =0; j <64; ++j) if (tail[j] !=0x5a) return false;
    }
    return true;
  }
};
} // namespace splash::flash::prefill4k_q4coded
