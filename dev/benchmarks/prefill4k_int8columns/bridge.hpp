#pragma once
#include "flash/FlashMoEBlocked.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"
#include <array>
#include <cstring>
#include <stdexcept>

namespace splash::flash::prefill4k_int8columns {
// No allocation, job compaction or parameter mutation. Source graph owns the
// inline parameters; every original bucket and Q4 miss dispatch stays intact.
class HitCommands {
 public:
  std::vector<metal::ComputeDispatch> commands;
  uint32_t width=0,sg=0;
  HitCommands()=default;
  HitCommands(HitCommands &&)=default;
  HitCommands &operator=(HitCommands &&)=default;
  HitCommands(const HitCommands &)=delete;
  HitCommands &operator=(const HitCommands &)=delete;
  HitCommands(std::span<const metal::ComputeDispatch> source,uint32_t rows,
      uint32_t n,uint32_t groups,const std::array<metal::MetalBuffer,3> &linear,
      bool audit=false) :width(n),sg(groups) {
    if (!rows ||rows >8192 ||(n !=128 &&n !=256) ||groups !=n /16)
      throw std::invalid_argument("wide-column geometry must be M32N128SG8 or M32N256SG16");
    bool gateSeen=false,downSeen=false;
    metal::MetalBuffer offsets,jobs,count;
    for (const auto &original :source) {
      auto d=original;
      const bool gate=d.pipelineName.starts_with("flash_int8_expert_store_gate_up_m") &&
          !d.pipelineName.starts_with("flash_int8_expert_store_gate_up_miss_");
      const bool down=d.pipelineName.starts_with("flash_int8_expert_store_down_scatter_");
      if (!gate &&!down) { commands.push_back(d);continue; }
      if (d.bytes.size() !=1 ||d.bytes[0].sizeBytes !=sizeof(FlashInt8ExpertStoreParams) ||!d.bytes[0].data)
        throw std::invalid_argument("wide-column source parameter ABI differs");
      FlashInt8ExpertStoreParams p;std::memcpy(&p,d.bytes[0].data,sizeof(p));
      if (p.rows !=rows ||p.selections !=10 ||p.route_capacity !=rows *10 ||
          p.tile_rows !=32 ||p.job_capacity !=(rows *10 +31) /32 +511 ||
          !p.stored_experts ||p.stored_experts >512 ||p.scale_group_size ||p.reserved ||
          d.threadgroups.y !=p.job_capacity ||d.threadgroups.z !=1)
        throw std::invalid_argument("wide-column requires unchanged native M32 jobs");
      if (gate) {
        if (gateSeen ||downSeen ||d.buffers.size() !=11 ||d.bytes[0].index !=11 ||d.threadgroups.x !=10)
          throw std::invalid_argument("wide-column gate ABI/order differs");
        gateSeen=true;offsets=d.buffers[6].buffer;jobs=d.buffers[7].buffer;count=d.buffers[8].buffer;
        d.buffers.push_back({12,linear[0]});d.buffers.push_back({13,linear[1]});
      } else {
        if (!gateSeen ||downSeen ||d.buffers.size() !=10 ||d.bytes[0].index !=10 ||d.threadgroups.x !=40 ||
            !d.buffers[4].buffer.sameView(offsets) ||!d.buffers[5].buffer.sameView(jobs) ||!d.buffers[6].buffer.sameView(count))
          throw std::invalid_argument("wide-column down ABI/jobs/order differs");
        downSeen=true;d.buffers.push_back({11,linear[2]});
      }
      d.pipelineName=std::string(audit ? "prefill4k_int8columnsaudit_" :"prefill4k_int8columns_") +
          (gate ? "gate_up" :"down_scatter") +"_m32_n" +std::to_string(n) +"_sg" +std::to_string(groups);
      d.threadgroups.x=((gate ? 640u :2560u) +n -1) /n;
      d.threadsPerThreadgroup={uint64_t(groups) *32,1,1};commands.push_back(d);
    }
    if (!gateSeen ||!downSeen) throw std::invalid_argument("wide-column source producer missing");
    if (commands.size() !=source.size()) throw std::logic_error("wide-column unexpectedly added dispatches");
  }
};
} // namespace splash::flash::prefill4k_int8columns
