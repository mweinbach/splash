#pragma once
#include "dev/benchmarks/prefill_moe_sep21/w8a8/cache.hpp"

namespace splash::flash::prefill_moe_w8a8_wide {
using Workspace=prefill_moe_w8a8::Workspace;
using QuantParams=prefill_moe_w8a8::QuantParams;
struct Variant final {uint32_t m,n,sg;bool wide;};
inline constexpr std::array<Variant,3> variants{{
    {32,64,2,false},{32,128,4,true},{16,128,2,true}
}};
inline std::string pipelineName(const Variant &v,bool gate,bool audit=false) {
  return std::string(v.wide ? "prefill_moe_sep21_w8a8_wide_" : "prefill_moe_sep21_w8a8_")+
      (gate ? "gate_up_" : "down_scatter_")+"m"+std::to_string(v.m)+"_n"+std::to_string(v.n)+
      "_sg"+std::to_string(v.sg)+(audit ? "_audit" : "");
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
      const bool gate=dispatch.pipelineName.starts_with("flash_int8_expert_store_gate_up_m");
      const bool down=dispatch.pipelineName.starts_with("flash_int8_expert_store_down_scatter_m");
      if (!gate && !down) {commands.push_back(dispatch);auditCommands.push_back(dispatch);continue;}
      if (dispatch.bytes.size()!=1 || dispatch.bytes[0].sizeBytes!=sizeof(FlashInt8ExpertStoreParams) || !dispatch.bytes[0].data)
        throw std::invalid_argument("wide W8A8 native producer parameter ABI differs");
      FlashInt8ExpertStoreParams p{};std::memcpy(&p,dispatch.bytes[0].data,sizeof(p));
      if (p.rows!=rows || p.selections!=10 || p.route_capacity!=rows*10 || p.tile_rows!=variant.m ||
          p.job_capacity!=(rows*10+variant.m-1)/variant.m+511 || p.stored_experts!=512 ||
          p.scale_group_size || p.reserved || dispatch.threadgroups.y!=p.job_capacity)
        throw std::invalid_argument("wide W8A8 native M16/M32 bucket geometry differs");
      const uint32_t activation=gate ? 0 : 1,width=gate ? 2560 : 640,diagnostic=gate ? 10 : 9;
      if ((gate && (gateSeen || downSeen || dispatch.buffers.size()!=11 || dispatch.bytes[0].index!=11 ||
          !dispatch.buffers[1].buffer.sameView(layer.codes[0]) || !dispatch.buffers[2].buffer.sameView(layer.scales[0]) ||
          !dispatch.buffers[3].buffer.sameView(layer.codes[1]) || !dispatch.buffers[4].buffer.sameView(layer.scales[1]))) ||
          (down && (!gateSeen || downSeen || dispatch.buffers.size()!=10 || dispatch.bytes[0].index!=10 ||
          !dispatch.buffers[1].buffer.sameView(layer.codes[2]) || !dispatch.buffers[2].buffer.sameView(layer.scales[2]))))
        throw std::invalid_argument("wide W8A8 immutable source/scales/producer order differs");
      if (gate) gateSeen=true;else downSeen=true;
      quantizers.add(gate ? "prefill_moe_sep21_w8a8_quantize_gate_t256" : "prefill_moe_sep21_w8a8_quantize_down_t128",
          {dispatch.buffers[0].buffer,workspace.quantized[activation],workspace.activationScale[activation],
           dispatch.buffers[diagnostic].buffer},QuantParams{workspace.paddedRoutes,width,0,0},
          {workspace.paddedRoutes,1,1},{gate ? 256u : 128u,1,1});
      const auto quantize=quantizers.dispatches().back();commands.push_back(quantize);auditCommands.push_back(quantize);
      dispatch.pipelineName=pipelineName(variant,gate);dispatch.buffers[0].buffer=workspace.quantized[activation];
      dispatch.buffers.push_back({gate ? 12u : 11u,workspace.activationScale[activation]});
      dispatch.threadgroups.x=(gate ? 640u : 2560u)/variant.n;
      dispatch.threadsPerThreadgroup={uint64_t(variant.sg)*32,1,1};commands.push_back(dispatch);
      dispatch.pipelineName=pipelineName(variant,gate,true);
      if (gate) {
        dispatch.buffers.push_back({13,workspace.scaledAudit[0]});dispatch.buffers.push_back({14,workspace.scaledAudit[1]});
        dispatch.buffers.push_back({15,workspace.integerAudit[0]});dispatch.buffers.push_back({16,workspace.integerAudit[1]});
      } else {
        dispatch.buffers.push_back({12,workspace.scaledAudit[2]});dispatch.buffers.push_back({13,workspace.integerAudit[2]});
      }
      auditCommands.push_back(std::move(dispatch));
    }
    if (!gateSeen || !downSeen || commands.size()!=source.size()+2 || auditCommands.size()!=commands.size())
      throw std::invalid_argument("wide W8A8 quantizer/projection dispatch inventory differs");
  }
};
inline void cpuSelfTest() {
  prefill_moe_w8a8::cpuSelfTest();
  for (const auto &v:variants)
    if ((v.m!=16 && v.m!=32) || (v.n!=64 && v.n!=128) || (v.sg!=2 && v.sg!=4) ||
        pipelineName(v,true)==pipelineName(v,false)) throw std::logic_error("wide W8A8 variant inventory differs");
  if ((20480+15)/16+511!=1791 || (20480+7)/8+511!=3071)
    throw std::logic_error("wide W8A8 native M16 job capacity differs");
}
} // namespace splash::flash::prefill_moe_w8a8_wide
