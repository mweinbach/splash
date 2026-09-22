#pragma once
#include "source_identity.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashMoEBuckets.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::flash::compact_native_r5_verify_sep22 {
inline constexpr const char* kFlag="SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22";
inline constexpr const char* kAIR="50002976851cd0bf2cf0f133c0164ccf177dfc1e7493b28563ba8fdbd1bb7ac3";
inline constexpr const char* kScope="singleton main physical R5 verification only; per-layer full-chain graph construction, not GPU completion";
inline constexpr const char* kMarker=";private-singleton-main-R5-integer-only-original-M16-sourceSha256=";
inline std::atomic<uint64_t> graphCalls{0},graphRows{0};
inline bool parse(const char*value) {
  if(!value||std::string_view(value)=="0")return false;
  if(std::string_view(value)=="1")return true;
  throw std::invalid_argument("R5 integer verify selector must be canonical0/1");
}
inline bool requested() {
  const bool now=parse(std::getenv(kFlag));static const bool frozen=now;
  if(now!=frozen)throw std::logic_error("R5 integer verify selector changed after construction");return frozen;
}
inline bool env(const char*name){return parse(std::getenv(name));}
inline void validateDependencies() {
  if(!requested())return;
  for(const char*name:{"SPLASH_FLASH_ALLROWS_FULL512_TARGET","SPLASH_FLASH_BLOCKED_MOE","SPLASH_FLASH_MOE_DIRECT_A","SPLASH_FLASH_MOE_Q4X8","SPLASH_FLASH_ALLROWS_GATHERED_MPP"})
    if(!env(name))throw std::invalid_argument(std::string("R5 integer verify requires ")+name+"=1");
  const char*cap=std::getenv("SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS");
  if(!cap||std::string_view(cap)!="4")throw std::invalid_argument("R5 integer verify preserves gathered cap exactly4");
}
inline void validateDepth(bool mtp,uint32_t depth,bool explicitDepth) {
  validateDependencies();if(requested()&&(!mtp||depth!=4||!explicitDepth))throw std::invalid_argument("R5 integer verify requires genuine explicit singleton MTP depth4");
}
inline constexpr bool eligibleFor(uint32_t rows,bool verification,bool singleton,bool enabled)noexcept{return enabled&&verification&&singleton&&rows==5;}
inline bool eligible(uint32_t rows,bool verification,bool singleton){return eligibleFor(rows,verification,singleton,requested());}
inline void validateConstruction(uint32_t verifyRows,bool full512,bool blocked) {
  validateDependencies();if(requested()&&(verifyRows!=5||!full512||!blocked))throw std::invalid_argument("R5 integer verify requires owned five-row tape/Full512/blocked source");
}
inline std::string marker(){return requested()?std::string(kMarker)+kSourceIdentitySha256:std::string{};}
inline void addSetup(metal::CommandGraph&graph,metal::MetalBuffer hidden,metal::MetalBuffer ids,
    const FlashMoEBlockedScratch&scratch,metal::MetalBuffer diag,uint32_t rows,FlashMoEBlockedTile tile,
    bool verification,bool singleton) {
  validateDependencies();if(!eligible(rows,verification,singleton)||tile!=FlashMoEBlockedTile::M16N64)throw std::invalid_argument("R5 integer setup received excluded caller");
  metal::CommandGraph original;addMoEBlockedPack(original,hidden,ids,scratch,diag,rows,tile,10);
  if(original.dispatches().size()!=6)throw std::logic_error("R5 original pack setup topology changed");
  const FlashMoEBucketParams packParams{5,10,2560,512,50,0,0,0},planParams{5,10,2560,512,50,16,515,0};
  const std::array<const char*,6> names{"flash_moe_bucket_histogram","flash_moe_bucket_prefix","flash_moe_bucket_stable_map","flash_moe_direct_a_pack","flash_moe_bucket_job_prefix","flash_moe_bucket_jobs"};
  const std::array<uint64_t,6> grids{512,1,512,113,1,3};
  const std::array<std::vector<metal::MetalBuffer>,6> bindings{{{ids,scratch.buckets.counts,scratch.buckets.canonicalToPacked,diag},
    {scratch.buckets.counts,scratch.buckets.offsets,diag},{ids,scratch.buckets.counts,scratch.buckets.offsets,scratch.buckets.routeMap,scratch.buckets.canonicalToPacked,diag},
    {hidden,scratch.buckets.offsets,scratch.buckets.routeMap,scratch.buckets.packedInputs,diag},{scratch.buckets.counts,scratch.buckets.offsets,scratch.buckets.jobOffsets,scratch.buckets.jobCount,diag},
    {scratch.buckets.offsets,scratch.buckets.jobOffsets,scratch.buckets.jobCount,scratch.buckets.tileJobs,diag}}};
  for(uint32_t index=0;index<6;++index) {
    const auto&d=original.dispatches()[index];const auto expected=index<4?packParams:planParams;
    if(d.pipelineName!=names[index]||d.threadgroups.x!=grids[index]||d.threadgroups.y!=1||d.threadgroups.z!=1||d.threadsPerThreadgroup.x!=256||d.threadsPerThreadgroup.y!=1||d.threadsPerThreadgroup.z!=1||d.buffers.size()!=bindings[index].size())throw std::logic_error("R5 original setup descriptor geometry changed");
    if(d.bytes.size()!=1||d.bytes[0].index!=bindings[index].size()||d.bytes[0].sizeBytes!=32||!d.bytes[0].data||std::memcmp(d.bytes[0].data,&expected,32))throw std::logic_error("R5 original setup packet changed");
    for(uint32_t slot=0;slot<bindings[index].size();++slot)if(d.buffers[slot].index!=slot||!d.buffers[slot].buffer.sameView(bindings[index][slot]))throw std::logic_error("R5 original setup source bindings changed");
  }
  graph.add("expert_r5_compact_native_sep22_plan",{ids,scratch.buckets.counts,scratch.buckets.offsets,scratch.buckets.routeMap,scratch.buckets.canonicalToPacked,scratch.buckets.jobOffsets,scratch.buckets.tileJobs,scratch.buckets.jobCount,diag},planParams,{1,1,1},{256,1,1});
  const auto&pack=original.dispatches()[3];std::vector<metal::MetalBuffer> packBindings;
  for(const auto&binding:pack.buffers)packBindings.push_back(binding.buffer);
  graph.add(pack.pipelineName,std::move(packBindings),packParams,pack.threadgroups,pack.threadsPerThreadgroup);
}
inline void recordCompletedGraph(uint32_t rows){if(rows!=5||!requested())throw std::logic_error("R5 full-chain counter caller changed");graphCalls.fetch_add(1,std::memory_order_relaxed);graphRows.fetch_add(5,std::memory_order_relaxed);}
}
