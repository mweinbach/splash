#include "FlashInt8PartitionedJobs.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"

#include <array>
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
          d.bytes[0].sizeBytes == sizeof(P) && d.bytes[0].data && d.buffers.size() == index,
          "partitioned INT8 source parameter/buffer bindings differ");
  for (uint32_t i = 0; i < index; ++i)
    require(d.buffers[i].index == i && d.buffers[i].buffer,
            "partitioned INT8 source buffer binding invalid");
  P p; std::memcpy(&p,d.bytes[0].data,sizeof(P)); return p;
}
void bytes(const metal::MetalBuffer &b,uint64_t size) {
  require(b && b.sizeBytes() >= size,"partitioned INT8 scratch/source extent too small");
}
bool overlap(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  if(a.sameView(b))return true;
  const auto x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());
  if(!x || !y)return false;
  return x<=y ? y-x<a.sizeBytes() : x-y<b.sizeBytes();
}
}

Int8PartitionedJobScratch allocateInt8PartitionedJobs(metal::MetalBackend &backend,
    uint32_t rows,uint32_t selections,uint32_t tileRows,uint32_t storedExperts) {
  const uint32_t h=flashInt8JobPartitionHitCapacity(rows,selections,tileRows,storedExperts);
  const uint32_t m=flashInt8JobPartitionMissCapacity(rows,selections,tileRows,storedExperts);
  require(h && m,"partitioned INT8 allocation geometry invalid");
  Int8PartitionedJobScratch result;
  result.hitCapacity=h; result.missCapacity=m;
  result.hits=backend.allocateBuffer(uint64_t(h)*8,metal::BufferStorage::Shared,"private INT8 hit jobs");
  result.misses=backend.allocateBuffer(uint64_t(m)*8,metal::BufferStorage::Shared,"private Q4 miss jobs");
  result.counts=backend.allocateBuffer(8,metal::BufferStorage::Shared,"private INT8/Q4 job counts");
  result.hitCount=backend.view(result.counts,0,4); result.missCount=backend.view(result.counts,4,4);
  return result;
}

PartitionedInt8Commands::PartitionedInt8Commands(std::span<const metal::ComputeDispatch> source,
                                                const Int8PartitionedJobScratch &s) {
  std::array<unsigned,4> phases{};
  std::array<uint32_t,6> common{};
  metal::MetalBuffer originalJobs,originalCount,ranks,offsets,diagnostics;
  for(const auto &original:source) {
    auto d=original;
    constexpr std::string_view prefix="flash_int8_expert_store_";
    if(!d.pipelineName.starts_with(prefix)) {dispatches_.push_back(std::move(d));continue;}
    const auto name=std::string_view(d.pipelineName).substr(prefix.size());
    const bool gateMiss=name.starts_with("gate_up_miss_direct_");
    const bool downMiss=name.starts_with("down_miss_direct_");
    const bool gateHit=!gateMiss && name.starts_with("gate_up_m");
    const bool downHit=name.starts_with("down_scatter_m");
    if(!gateMiss && !downMiss && !gateHit && !downHit) {dispatches_.push_back(std::move(d));continue;}
    uint32_t rows=0,selections=0,routeCapacity=0,tile=0,hot=0,jobCapacity=0;
    uint32_t jobsIndex=0,countIndex=0,ranksIndex=0,offsetIndex=0,diagnosticsIndex=0;
    if(gateHit || downHit) {
      const auto p=params<FlashInt8ExpertStoreParams>(d,gateHit?11:10);
      require(!p.reserved && !p.scale_group_size,"partitioned INT8 hit policy differs");
      rows=p.rows; selections=p.selections; routeCapacity=p.route_capacity;
      tile=p.tile_rows;hot=p.stored_experts;jobCapacity=p.job_capacity;
      jobsIndex=gateHit?7:5;countIndex=gateHit?8:6;ranksIndex=gateHit?5:3;
      offsetIndex=gateHit?6:4;diagnosticsIndex=gateHit?10:9;
      ++phases[gateHit?0:2];
    } else if(gateMiss) {
      const auto p=params<FlashInt8ExpertStoreGateParams>(d,13);
      require(p.flags==1 && !p.reserved0 && !p.reserved1 && !p.blocked.reserved,
              "partitioned INT8 gate miss must retain direct Q4 producer");
      rows=p.blocked.affine.rows;selections=p.blocked.affine.selections;routeCapacity=p.blocked.route_capacity;
      tile=p.blocked.tile_rows;hot=p.stored_experts;jobCapacity=p.blocked.job_capacity;
      jobsIndex=8;countIndex=9;ranksIndex=12;offsetIndex=7;diagnosticsIndex=11;++phases[1];
    } else {
      const auto p=params<FlashInt8ExpertStoreDownParams>(d,11);
      require(p.flags==1 && !p.reserved0 && !p.reserved1 && !p.blocked.reserved,
              "partitioned INT8 down miss must retain direct Q4 producer");
      rows=p.blocked.affine.rows;selections=p.blocked.affine.selections;routeCapacity=p.blocked.route_capacity;
      tile=p.blocked.tile_rows;hot=p.stored_experts;jobCapacity=p.blocked.job_capacity;
      jobsIndex=5;countIndex=6;ranksIndex=10;offsetIndex=4;diagnosticsIndex=9;++phases[3];
    }
    require(flashInt8JobPartitionSourceCapacity(rows,selections,tile,hot) &&
            routeCapacity==rows*selections && jobCapacity==flashInt8JobPartitionSourceCapacity(rows,selections,tile,hot),
            "partitioned INT8 source capacity differs");
    const bool gate=gateHit || gateMiss,hit=gateHit || downHit;
    require(d.threadgroups.x==(gate?10u:40u) && d.threadgroups.y==jobCapacity && d.threadgroups.z==1 &&
            d.threadsPerThreadgroup.x==(tile==64?256u:128u) &&
            d.threadsPerThreadgroup.y==1 && d.threadsPerThreadgroup.z==1,
            "partitioned INT8 source launch differs");
    const std::array<uint32_t,6> metadata{rows,selections,routeCapacity,tile,hot,jobCapacity};
    if(gateHit) {
      require(phases==std::array<unsigned,4>{1,0,0,0},"partitioned INT8 chain must start with one gate hit");
      common=metadata;
      originalJobs=d.buffers[jobsIndex].buffer;originalCount=d.buffers[countIndex].buffer;
      ranks=d.buffers[ranksIndex].buffer;offsets=d.buffers[offsetIndex].buffer;diagnostics=d.buffers[diagnosticsIndex].buffer;
      require(s.hitCapacity==flashInt8JobPartitionHitCapacity(rows,selections,tile,hot) &&
              s.missCapacity==flashInt8JobPartitionMissCapacity(rows,selections,tile,hot),
              "partitioned INT8 scratch metadata differs");
      bytes(s.hits,uint64_t(s.hitCapacity)*8);bytes(s.misses,uint64_t(s.missCapacity)*8);bytes(s.counts,8);
      bytes(s.hitCount,4);bytes(s.missCount,4);
      require(s.hitCount.contents()==s.counts.contents() &&
              static_cast<const uint8_t *>(s.missCount.contents())==static_cast<const uint8_t *>(s.counts.contents())+4,
              "partitioned INT8 count views differ from owned count allocation");
      bytes(originalJobs,uint64_t(jobCapacity)*8);bytes(originalCount,4);bytes(ranks,2048);bytes(offsets,2052);bytes(diagnostics,4);
      const std::array<metal::MetalBuffer,3> writable{s.hits,s.misses,s.counts};
      for(size_t a=0;a<writable.size();++a) {
        for(size_t b=a+1;b<writable.size();++b) require(!overlap(writable[a],writable[b]),"partitioned INT8 outputs alias");
        for(const auto &input:{originalJobs,originalCount,ranks,offsets,diagnostics})
          require(!overlap(writable[a],input),"partitioned INT8 output aliases source metadata");
        for(const auto &node:source)
          for(const auto &binding:node.buffers)
            require(!overlap(writable[a],binding.buffer),"partitioned INT8 output aliases original producer binding");
      }
      preparation_.add("flash_int8_job_partition",
          {originalJobs,originalCount,ranks,offsets,s.hits,s.misses,s.counts,diagnostics},
          FlashInt8JobPartitionParams{rows,selections,routeCapacity,tile,hot,jobCapacity,
                                     s.hitCapacity,s.missCapacity,0,0,0,0},
          {1,1,1},{256,1,1});
      dispatches_.push_back(preparation_.dispatches().back());
    } else {
      require(common==metadata && d.buffers[jobsIndex].buffer.sameView(originalJobs) &&
              d.buffers[countIndex].buffer.sameView(originalCount) && d.buffers[ranksIndex].buffer.sameView(ranks) &&
              d.buffers[offsetIndex].buffer.sameView(offsets) && d.buffers[diagnosticsIndex].buffer.sameView(diagnostics),
              "partitioned INT8 phases do not share stable job ownership");
    }
    d.buffers[jobsIndex].buffer=hit?s.hits:s.misses;
    d.buffers[countIndex].buffer=hit?s.hitCount:s.missCount;
    d.threadgroups.y=hit?s.hitCapacity:s.missCapacity;
    dispatches_.push_back(std::move(d));
  }
  require(phases==std::array<unsigned,4>{1,1,1,1},"partitioned INT8 needs one complete v5 hit/miss chain");
}
} // namespace splash::flash::candidate
