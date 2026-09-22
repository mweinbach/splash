#pragma once
#include "flash/FlashQSA.hpp"
#include "flash/FlashQSAMPP.hpp"
#include "dev/benchmarks/prefill4k_attention/bulk.hpp"
#include "metal/abi/FlashQSAFast.h"
#include "../kernel/PackedV.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace qsa_online_packed_v_host {
inline constexpr uint32_t rows=2048,ordinaryRows=128,partitions=4,pairs=18;
inline constexpr uint64_t page=16384,guard=16384,reservation=1ULL<<30;
inline constexpr uint64_t packedBytes=2097152,quotientBytes=50331648,roundedBytes=25165824;
inline constexpr QSAOnlinePackedVParams packParams{2048,2,256,0};
static_assert(sizeof(FlashQSAFastParams)==64 && sizeof(QSAOnlinePackedVParams)==16);
inline constexpr uint64_t charge(uint64_t logical) {return ((logical+page-1)&~(page-1))+2*guard;}
struct PlannedOwner final {std::string name;uint64_t logical=0,charged=0;};
inline std::vector<PlannedOwner> ownerPlan(uint32_t capacity) {
  std::vector<PlannedOwner> result;
  const auto add=[&](std::string name,uint64_t n){result.push_back({std::move(name),n,charge(n)});};
  const auto input=[&](const std::string &name,uint32_t n){
    for(auto width:{12288u,512u,512u,640u})add(name+" projected",uint64_t(n)*width*2);
    add(name+" output",uint64_t(n)*6144*2);add(name+" diagnostic",4);
    for(auto width:{256u,256u,128u,128u})add(name+" norm maximumF32",width*4);
  };
  for(uint32_t variant=0;variant<2;++variant){const std::string n="variant"+std::to_string(variant);
    input(n,rows);input(n+" temporary future",ordinaryRows);
    for(auto width:{512u,512u,128u})add(n+" cache",uint64_t(capacity)*width*2);
    add(n+" pooled",uint64_t((capacity+3)/4)*128*2);add(n+" positions",uint64_t(capacity)*8);
    add(n+" preparedQ",uint64_t(rows)*6144*2);add(n+" preparedIndexQ",uint64_t(rows)*512*2);
    add(n+" selection",uint64_t(rows)*512*4);add(n+" statistics",uint64_t(rows)*24*4*2*4);
    add(n+" numerators",uint64_t(rows)*24*4*256*4);
  }
  add("shared ordinaryQ",uint64_t(ordinaryRows)*6144*2);add("shared ordinaryIndexQ",uint64_t(ordinaryRows)*512*2);
  add("shared blockScores",uint64_t(ordinaryRows)*((capacity+3)/4)*4);add("shared selection",uint64_t(ordinaryRows)*512*4);
  add("shared attentionScores",uint64_t(ordinaryRows)*24*2051*4);add("shared probabilities",uint64_t(ordinaryRows)*24*2051*2);
  add("shared ordinary statistics",uint64_t(ordinaryRows)*24*32*2*4);add("shared ordinary numerators",uint64_t(ordinaryRows)*24*32*256*4);
  add("shared live quotient",quotientBytes);add("shared live rounded",roundedBytes);add("packedV",packedBytes);
  return result;
}
inline uint64_t plannedBytes(uint32_t capacity){uint64_t total=0;for(const auto &owner:ownerPlan(capacity))total+=owner.charged;return total;}
} // namespace qsa_online_packed_v_host
