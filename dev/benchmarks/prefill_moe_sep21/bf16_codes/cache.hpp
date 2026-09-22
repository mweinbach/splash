#pragma once

// Root-GPU-mode-only persistent cache of exact integer coefficients. F32 row
// scales deliberately remain external; no scale participates in this cache.
#include "dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::flash::prefill_moe_bf16_codes {
inline constexpr uint64_t kElementsPerPlane=838860800ULL;
inline constexpr uint64_t kLogicalBytesPerPlane=kElementsPerPlane*2;
inline constexpr uint64_t kGuardBytes=64;
inline constexpr uint64_t kAlignment=16384;
inline constexpr uint64_t kAllocatedBytesPerPlane=
    (kLogicalBytesPerPlane+kGuardBytes+kAlignment-1)&~(kAlignment-1);

inline uint16_t integerBF16(int value) {
  if (value<-128 || value>127) throw std::invalid_argument("I8 integer outside exact cache range");
  if (!value) return 0;
  const uint32_t magnitude=uint32_t(value<0 ? -value : value);
  const uint32_t exponent=std::bit_width(magnitude)-1;
  return uint16_t((value<0 ? 0x8000u : 0u) | ((127u+exponent)<<7) |
      ((magnitude<<(7u-exponent))&127u));
}
inline std::array<uint16_t,256> conversionLookup() {
  std::array<uint16_t,256> table{};
  for (uint32_t raw=0;raw<256;++raw) {
    const int value=int(std::bit_cast<int8_t>(uint8_t(raw)));
    table[raw]=integerBF16(value);
  }
  return table;
}

class ExactCodeCache final {
 public:
  std::array<metal::MetalBuffer,3> allocations,codes;
  static constexpr uint64_t plannedBytes() { return 3*kAllocatedBytesPerPlane; }
  static constexpr uint64_t logicalBytes() { return 3*kLogicalBytesPerPlane; }
  static constexpr uint64_t words() { return 3*kElementsPerPlane; }

  // Caller independently reserves plannedBytes() before this method. Build and
  // CPU self-tests never call it or inspect a persisted coefficient payload.
  static ExactCodeCache convert(metal::MetalBackend &backend,
      const qmv_one_layer::OneLayerPayload &source) {
    if (kAllocatedBytesPerPlane>backend.capabilities().maxBufferLengthBytes)
      throw std::invalid_argument("exact BF16 code cache exceeds Metal buffer limit");
    const auto lookup=conversionLookup();ExactCodeCache result;
    const uint64_t before=backend.memoryStats().allocatedBytes;
    for (uint32_t plane=0;plane<3;++plane) {
      if (!source.codes[plane].contents() || source.codes[plane].sizeBytes()!=kElementsPerPlane)
        throw std::invalid_argument("exact code cache source extent differs");
      result.allocations[plane]=backend.allocateBuffer(kAllocatedBytesPerPlane,
          metal::BufferStorage::Shared,"private exact unscaled BF16 I8 integer codes");
      result.codes[plane]=backend.view(result.allocations[plane],0,kLogicalBytesPerPlane);
      auto *destination=static_cast<uint16_t *>(result.codes[plane].contents());
      const auto *original=static_cast<const uint8_t *>(source.codes[plane].contents());
      for (uint64_t i=0;i<kElementsPerPlane;++i) destination[i]=lookup[original[i]];
      std::memset(static_cast<uint8_t *>(result.allocations[plane].contents())+
          kLogicalBytesPerPlane,0x5a,kAllocatedBytesPerPlane-kLogicalBytesPerPlane);
    }
    const uint64_t after=backend.memoryStats().allocatedBytes;
    if (after<before || after-before>plannedBytes())
      throw std::runtime_error("exact BF16 code cache exceeded its independent admission");
    return result;
  }

  bool guardsClean() const {
    return std::all_of(allocations.begin(),allocations.end(),[](const auto &allocation) {
      if (!allocation.contents() || allocation.sizeBytes()!=kAllocatedBytesPerPlane) return false;
      const auto *tail=static_cast<const uint8_t *>(allocation.contents())+kLogicalBytesPerPlane;
      return std::all_of(tail,tail+kAllocatedBytesPerPlane-kLogicalBytesPerPlane,
          [](uint8_t value){return value==0x5a;});
    });
  }

  // Independent of the integer lookup encoder: F32 represents all signed I8
  // integers exactly, whose low sixteen mantissa bits are zero. Check EVERY
  // word before and after the root-controlled timing scope, without row scales.
  uint64_t verifyEveryWord(const qmv_one_layer::OneLayerPayload &source) const {
    uint64_t mismatches=0;
    for (uint32_t plane=0;plane<3;++plane) {
      if (!codes[plane].contents() || codes[plane].sizeBytes()!=kLogicalBytesPerPlane ||
          !source.codes[plane].contents() || source.codes[plane].sizeBytes()!=kElementsPerPlane)
        throw std::invalid_argument("exact cache verification extent differs");
      const auto *actual=static_cast<const uint16_t *>(codes[plane].contents());
      const auto *original=static_cast<const int8_t *>(source.codes[plane].contents());
      for (uint64_t i=0;i<kElementsPerPlane;++i) {
        const uint16_t expected=uint16_t(std::bit_cast<uint32_t>(float(original[i]))>>16);
        mismatches+=actual[i]!=expected;
      }
    }
    return mismatches;
  }
};

struct Variant final {
  const char *suffix;
  uint32_t m,sg,k;
  bool staticExtent,differentReduction;
};
inline constexpr std::array<Variant,4> variants{{
    {"m32_n64_sg4",32,4,0,false,false},
    {"m32_n64_sg2",32,2,0,false,false},
    {"m64_n64_sg8",64,8,0,false,false},
    {"m32_n64_k128_sg2",32,2,128,true,true}
}};
inline std::string pipelineName(const Variant &v,bool gate) {
  return std::string("prefill_moe_sep21_bf16_codes_")+
      (gate ? "gate_up_" : "down_scatter_")+v.suffix;
}

class CacheCommands final {
 public:
  std::vector<metal::ComputeDispatch> commands;
  CacheCommands(std::span<const metal::ComputeDispatch> source,uint32_t rows,
      const Variant &variant,const qmv_one_layer::OneLayerPayload &layer,
      const ExactCodeCache &cache) {
    bool gateSeen=false,downSeen=false;
    for (const auto &original:source) {
      auto dispatch=original;
      const bool gate=dispatch.pipelineName.starts_with("flash_int8_expert_store_gate_up_m");
      const bool down=dispatch.pipelineName.starts_with("flash_int8_expert_store_down_scatter_m");
      if (!gate && !down) {commands.push_back(dispatch);continue;}
      if (dispatch.bytes.size()!=1 || dispatch.bytes[0].sizeBytes!=sizeof(FlashInt8ExpertStoreParams) ||
          !dispatch.bytes[0].data)
        throw std::invalid_argument("BF16 code candidate parameter ABI differs");
      FlashInt8ExpertStoreParams p{};std::memcpy(&p,dispatch.bytes[0].data,sizeof(p));
      if (p.rows!=rows || p.selections!=10 || p.route_capacity!=rows*10 ||
          p.tile_rows!=variant.m || p.job_capacity!=(rows*10+variant.m-1)/variant.m+511 ||
          p.stored_experts!=512 || p.scale_group_size || p.reserved ||
          dispatch.threadgroups.y!=p.job_capacity || dispatch.threadgroups.z!=1)
        throw std::invalid_argument("BF16 code candidate native bucket geometry differs");
      if (gate) {
        if (gateSeen || downSeen || dispatch.buffers.size()!=11 || dispatch.bytes[0].index!=11 ||
            dispatch.threadgroups.x!=10 || !dispatch.buffers[1].buffer.sameView(layer.codes[0]) ||
            !dispatch.buffers[2].buffer.sameView(layer.scales[0]) ||
            !dispatch.buffers[3].buffer.sameView(layer.codes[1]) ||
            !dispatch.buffers[4].buffer.sameView(layer.scales[1]))
          throw std::invalid_argument("BF16 code candidate gate/scales/source order differs");
        gateSeen=true;dispatch.buffers[1].buffer=cache.codes[0];dispatch.buffers[3].buffer=cache.codes[1];
      } else {
        if (!gateSeen || downSeen || dispatch.buffers.size()!=10 || dispatch.bytes[0].index!=10 ||
            dispatch.threadgroups.x!=40 || !dispatch.buffers[1].buffer.sameView(layer.codes[2]) ||
            !dispatch.buffers[2].buffer.sameView(layer.scales[2]))
          throw std::invalid_argument("BF16 code candidate down/scales/source order differs");
        downSeen=true;dispatch.buffers[1].buffer=cache.codes[2];
      }
      dispatch.pipelineName=pipelineName(variant,gate);
      dispatch.threadsPerThreadgroup={uint64_t(variant.sg)*32,1,1};
      commands.push_back(std::move(dispatch));
    }
    if (!gateSeen || !downSeen || commands.size()!=source.size())
      throw std::invalid_argument("BF16 code candidate native producers missing");
  }
};

inline void cpuSelfTest() {
  const auto lookup=conversionLookup();
  for (uint32_t raw=0;raw<256;++raw) {
    const int8_t value=std::bit_cast<int8_t>(uint8_t(raw));
    const uint32_t exact=std::bit_cast<uint32_t>(float(value));
    if ((exact&65535u)!=0 || lookup[raw]!=uint16_t(exact>>16))
      throw std::logic_error("BF16 unscaled I8 conversion lookup differs");
  }
  if (ExactCodeCache::plannedBytes()!=5033213952ULL || ExactCodeCache::logicalBytes()!=5033164800ULL ||
      ExactCodeCache::words()!=2516582400ULL || lookup[0]!=0 || lookup[128]!=0xc300)
    throw std::logic_error("BF16 exact-code cache accounting/golden differs");
  for (const auto &v:variants)
    if ((v.m!=32 && v.m!=64) || pipelineName(v,true)==pipelineName(v,false) ||
        (v.k && (2560%v.k || 640%v.k)))
      throw std::logic_error("BF16 code candidate inventory differs");
}
} // namespace splash::flash::prefill_moe_bf16_codes
