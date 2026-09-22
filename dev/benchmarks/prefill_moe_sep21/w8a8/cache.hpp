#pragma once

// Numerical prefill experiment. Original I8 coefficient and F32 row-scale
// views stay immutable. Only quantized activation rows/scales and audits exist.
#include "dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::flash::prefill_moe_w8a8 {
inline constexpr uint64_t kAlignment=16384,kGuardBytes=64;
struct QuantParams final {uint32_t rows,width,reserved0,reserved1;};
static_assert(sizeof(QuantParams)==16);

class Workspace final {
 public:
  std::array<metal::MetalBuffer,2> quantized,activationScale;
  std::array<metal::MetalBuffer,3> scaledAudit,integerAudit;
  std::array<metal::MetalBuffer,10> allocations;
  std::array<uint64_t,10> logical{};
  uint32_t rows=0,routes=0,paddedRoutes=0;
  static std::array<uint64_t,10> sizes(uint32_t rows) {
    if (rows<1024 || rows>2048) throw std::invalid_argument("W8A8 bounded row extent unsupported");
    const uint64_t routes=uint64_t(rows)*10,padded=routes+63;
    return {padded*2560,padded*640,padded*4,padded*4,
        routes*640*4,routes*640*4,routes*2560*4,
        routes*640*4,routes*640*4,routes*2560*4};
  }
  static uint64_t plannedBytes(uint32_t rows) {
    uint64_t total=0;
    for (uint64_t size:sizes(rows)) total+=(size+kGuardBytes+kAlignment-1)&~(kAlignment-1);
    return total;
  }
  static Workspace allocate(metal::MetalBackend &backend,uint32_t rows) {
    Workspace result;result.rows=rows;result.routes=rows*10;result.paddedRoutes=result.routes+63;
    result.logical=sizes(rows);std::array<metal::MetalBuffer,10> views;
    const uint64_t before=backend.memoryStats().allocatedBytes;
    for (uint32_t i=0;i<views.size();++i) {
      const uint64_t bytes=(result.logical[i]+kGuardBytes+kAlignment-1)&~(kAlignment-1);
      result.allocations[i]=backend.allocateBuffer(bytes,metal::BufferStorage::Shared,
          "private W8A8 governed activation/scale/projection audit");
      views[i]=backend.view(result.allocations[i],0,result.logical[i]);
      std::memset(result.allocations[i].contents(),0xa5,result.logical[i]);
      std::memset(static_cast<uint8_t *>(result.allocations[i].contents())+result.logical[i],
          0x5a,bytes-result.logical[i]);
    }
    result.quantized={views[0],views[1]};result.activationScale={views[2],views[3]};
    result.scaledAudit={views[4],views[5],views[6]};result.integerAudit={views[7],views[8],views[9]};
    const uint64_t after=backend.memoryStats().allocatedBytes;
    if (after<before || after-before>plannedBytes(rows))
      throw std::runtime_error("W8A8 activation/audit workspace exceeded independent admission");
    return result;
  }
  bool guardsClean() const {
    for (uint32_t i=0;i<allocations.size();++i) {
      if (!allocations[i].contents() || allocations[i].sizeBytes()<logical[i]+kGuardBytes) return false;
      const auto *begin=static_cast<const uint8_t *>(allocations[i].contents())+logical[i];
      if (!std::all_of(begin,begin+allocations[i].sizeBytes()-logical[i],
          [](uint8_t value){return value==0x5a;})) return false;
    }
    return true;
  }
};

struct Variant final {uint32_t sg;};
inline constexpr std::array<Variant,2> variants{{{4},{2}}};
inline std::string pipelineName(const Variant &v,bool gate,bool audit=false) {
  return std::string("prefill_moe_sep21_w8a8_")+
      (gate ? "gate_up_" : "down_scatter_")+"m32_n64_sg"+std::to_string(v.sg)+
      (audit ? "_audit" : "");
}

class QuantizedCommands final {
 public:
  metal::CommandGraph quantizers;
  std::vector<metal::ComputeDispatch> commands,auditCommands;
  QuantizedCommands(std::span<const metal::ComputeDispatch> source,uint32_t rows,
      const Variant &variant,const qmv_one_layer::OneLayerPayload &layer,
      const Workspace &workspace) {
    bool gateSeen=false,downSeen=false;
    for (const auto &original:source) {
      auto dispatch=original;
      const bool gate=dispatch.pipelineName.starts_with("flash_int8_expert_store_gate_up_m32_");
      const bool down=dispatch.pipelineName.starts_with("flash_int8_expert_store_down_scatter_m32_");
      if (!gate && !down) {commands.push_back(dispatch);auditCommands.push_back(dispatch);continue;}
      if (dispatch.bytes.size()!=1 || dispatch.bytes[0].sizeBytes!=sizeof(FlashInt8ExpertStoreParams) ||
          !dispatch.bytes[0].data)
        throw std::invalid_argument("W8A8 native producer parameter ABI differs");
      FlashInt8ExpertStoreParams p{};std::memcpy(&p,dispatch.bytes[0].data,sizeof(p));
      if (p.rows!=rows || p.selections!=10 || p.route_capacity!=rows*10 || p.tile_rows!=32 ||
          p.job_capacity!=(rows*10+31)/32+511 || p.stored_experts!=512 ||
          p.scale_group_size || p.reserved || dispatch.threadgroups.y!=p.job_capacity)
        throw std::invalid_argument("W8A8 native M32 bucket geometry differs");
      const uint32_t activation=gate ? 0 : 1,width=gate ? 2560 : 640;
      const uint32_t diagnostic=gate ? 10 : 9;
      if ((gate && (gateSeen || downSeen || dispatch.buffers.size()!=11 || dispatch.bytes[0].index!=11 ||
          !dispatch.buffers[1].buffer.sameView(layer.codes[0]) ||
          !dispatch.buffers[2].buffer.sameView(layer.scales[0]) ||
          !dispatch.buffers[3].buffer.sameView(layer.codes[1]) ||
          !dispatch.buffers[4].buffer.sameView(layer.scales[1]))) ||
          (down && (!gateSeen || downSeen || dispatch.buffers.size()!=10 || dispatch.bytes[0].index!=10 ||
          !dispatch.buffers[1].buffer.sameView(layer.codes[2]) ||
          !dispatch.buffers[2].buffer.sameView(layer.scales[2]))))
        throw std::invalid_argument("W8A8 immutable source/scales/producer order differs");
      if (gate) gateSeen=true;else downSeen=true;
      const std::string quantizer=gate ? "prefill_moe_sep21_w8a8_quantize_gate_t256" :
          "prefill_moe_sep21_w8a8_quantize_down_t128";
      quantizers.add(quantizer,{dispatch.buffers[0].buffer,workspace.quantized[activation],
          workspace.activationScale[activation],dispatch.buffers[diagnostic].buffer},
          QuantParams{workspace.paddedRoutes,width,0,0},{workspace.paddedRoutes,1,1},
          {gate ? 256u : 128u,1,1});
      const auto quantize=quantizers.dispatches().back();commands.push_back(quantize);auditCommands.push_back(quantize);
      dispatch.pipelineName=pipelineName(variant,gate);
      dispatch.buffers[0].buffer=workspace.quantized[activation];
      dispatch.buffers.push_back({gate ? 12u : 11u,workspace.activationScale[activation]});
      dispatch.threadsPerThreadgroup={uint64_t(variant.sg)*32,1,1};commands.push_back(dispatch);
      dispatch.pipelineName=pipelineName(variant,gate,true);
      if (gate) {
        dispatch.buffers.push_back({13,workspace.scaledAudit[0]});
        dispatch.buffers.push_back({14,workspace.scaledAudit[1]});
        dispatch.buffers.push_back({15,workspace.integerAudit[0]});
        dispatch.buffers.push_back({16,workspace.integerAudit[1]});
      } else {
        dispatch.buffers.push_back({12,workspace.scaledAudit[2]});
        dispatch.buffers.push_back({13,workspace.integerAudit[2]});
      }
      auditCommands.push_back(std::move(dispatch));
    }
    if (!gateSeen || !downSeen || commands.size()!=source.size()+2 || auditCommands.size()!=commands.size())
      throw std::invalid_argument("W8A8 quantizer/projection dispatch inventory differs");
  }
};

inline void cpuSelfTest() {
  if (sizeof(QuantParams)!=16 || Workspace::plannedBytes(2048)<695047544ULL ||
      Workspace::plannedBytes(2048)>(700ULL<<20))
    throw std::logic_error("W8A8 workspace accounting/ABI differs");
  for (const auto &v:variants)
    if ((v.sg!=4 && v.sg!=2) || pipelineName(v,true)==pipelineName(v,false) ||
        pipelineName(v,true,true)==pipelineName(v,true))
      throw std::logic_error("W8A8 candidate inventory differs");
}
} // namespace splash::flash::prefill_moe_w8a8
