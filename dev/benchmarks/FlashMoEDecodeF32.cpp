#include "FlashMoEDecodeF32.hpp"
#include "FlashMoEDecodeF32ABI.h"
#include "metal/abi/FlashMoEBlocked.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace splash::flash::candidate {
namespace {
void require(bool value,const char *reason) {
  if (!value) throw std::invalid_argument(reason);
}
void rowsGuard(uint32_t rows) {
  require(rows && rows<=16,"Private decode MPP requires1..16 real rows");
}
void aligned(const FlashAffineProjection &p) {
  require(p.experts==512 && p.bits==4 && p.groupSize==64 &&
      ((p.inputSize==2560 && p.outputSize==640) || (p.inputSize==640 && p.outputSize==2560)) &&
      p.weights && p.weights->buffer.contents() &&
      reinterpret_cast<uintptr_t>(p.weights->buffer.contents())%4==0 &&
      p.weightRowStrideBytes%4==0 && p.weightExpertStrideBytes%4==0,
      "Private decode MPP requires original aligned Q4/G64 expert planes");
}
template<class T> T params(const metal::ComputeDispatch &dispatch,uint32_t binding) {
  require(dispatch.bytes.size()==1 && dispatch.bytes[0].index==binding &&
      dispatch.bytes[0].sizeBytes==sizeof(T) && dispatch.bytes[0].data,
      "Private decode MPP source parameter ABI differs");
  T p; std::memcpy(&p,dispatch.bytes[0].data,sizeof(p)); return p;
}
std::vector<metal::MetalBuffer> buffers(const metal::ComputeDispatch &dispatch) {
  std::vector<metal::MetalBuffer> out;
  for (const auto &binding : dispatch.buffers) {
    require(binding.index==out.size(),"Private decode MPP source binding order differs");
    out.push_back(binding.buffer);
  }
  return out;
}
}

void addMoEDecodeF32GateUp(metal::CommandGraph &graph,
    const FlashAffineProjection &gate,const FlashAffineProjection &up,
    const FlashMoEBlockedScratch &scratch,metal::MetalBuffer gateTap,
    metal::MetalBuffer upTap,metal::MetalBuffer diagnostics,uint32_t rows) {
  rowsGuard(rows); aligned(gate); aligned(up);
  const uint64_t bytes=uint64_t{rows}*10*640*2;
  require(gateTap && upTap && gateTap.sizeBytes()>=bytes && upTap.sizeBytes()>=bytes &&
      !gateTap.sameView(upTap),"Private decode MPP insufficient/disjoint tap buffers");
  metal::CommandGraph validated;
  addMoEBlockedGateUp(validated,gate,up,scratch,diagnostics,rows,FlashMoEBlockedTile::M8N64);
  require(validated.dispatches().size()==1,"Private decode MPP gate source dispatch differs");
  const auto &source=validated.dispatches()[0];
  auto bound=buffers(source);
  for (const auto &buffer : bound)
    require(!gateTap.sameView(buffer) && !upTap.sameView(buffer),
        "Private decode MPP tap/source alias");
  require(!gateTap.sameView(scratch.buckets.routeMap) && !upTap.sameView(scratch.buckets.routeMap),
      "Private decode MPP tap/map alias");
  bound.push_back(scratch.buckets.routeMap); bound.push_back(gateTap); bound.push_back(upTap);
  const auto p=params<FlashMoEBlockedGateParams>(source,12);
  graph.add("flash_moe_decode_f32_gate_up_m8_n64",std::move(bound),p,
      {10,std::min(rows*10,p.job_capacity),1},{128,1,1});
}

void addMoEDecodeBF16GateUpTaps(metal::CommandGraph &graph,
    const FlashAffineProjection &gate,const FlashAffineProjection &up,
    const FlashMoEBlockedScratch &scratch,metal::MetalBuffer gateTap,
    metal::MetalBuffer upTap,metal::MetalBuffer diagnostics,uint32_t rows) {
  rowsGuard(rows); aligned(gate); aligned(up);
  const uint64_t bytes=uint64_t{rows}*10*640*2;
  require(gateTap && upTap && gateTap.sizeBytes()>=bytes && upTap.sizeBytes()>=bytes &&
      !gateTap.sameView(upTap),"Private decode MPP insufficient/disjoint tap buffers");
  metal::CommandGraph validated;
  addMoEBlockedGateUp(validated,gate,up,scratch,diagnostics,rows,FlashMoEBlockedTile::M8N64);
  require(validated.dispatches().size()==1,"Private decode MPP gate source dispatch differs");
  const auto &source=validated.dispatches()[0];
  auto bound=buffers(source);
  for (const auto &buffer : bound)
    require(!gateTap.sameView(buffer) && !upTap.sameView(buffer),
        "Private decode MPP tap/source alias");
  require(!gateTap.sameView(scratch.buckets.routeMap) && !upTap.sameView(scratch.buckets.routeMap),
      "Private decode MPP tap/map alias");
  bound.push_back(scratch.buckets.routeMap); bound.push_back(gateTap); bound.push_back(upTap);
  const auto p=params<FlashMoEBlockedGateParams>(source,12);
  graph.add("flash_moe_decode_bf16_gate_up_m8_n64",std::move(bound),p,
      {10,std::min(rows*10,p.job_capacity),1},{128,1,1});
}

void addMoEDecodeF32Down(metal::CommandGraph &graph,
    const FlashAffineProjection &down,const FlashMoEBlockedScratch &scratch,
    metal::MetalBuffer diagnostics,uint32_t rows) {
  rowsGuard(rows); aligned(down);
  metal::CommandGraph validated;
  addMoEBlockedDownScatter(validated,down,scratch,diagnostics,rows,FlashMoEBlockedTile::M8N64);
  require(validated.dispatches().size()==2,"Private decode MPP down source dispatch differs");
  const auto &poison=validated.dispatches()[0], &source=validated.dispatches()[1];
  const auto p=params<FlashMoEBlockedDownParams>(source,10);
  graph.add(poison.pipelineName,buffers(poison),params<FlashMoEBlockedDownParams>(poison,3),
      poison.threadgroups,poison.threadsPerThreadgroup);
  graph.add("flash_moe_decode_f32_down_scatter_m8_n64",buffers(source),p,
      {40,std::min(rows*10,p.job_capacity),1},{128,1,1});
}

void addMoEDecodeF32CoefficientSample(metal::CommandGraph &graph,
    const FlashAffineProjection &p,uint32_t expert,uint32_t outputBegin,
    uint32_t inputBegin,metal::MetalBuffer output,metal::MetalBuffer diagnostics,bool bf16Operand) {
  aligned(p);
  require(expert<512 && outputBegin%64==0 && inputBegin%64==0 &&
      outputBegin<=p.outputSize-64 && inputBegin<=p.inputSize-64 &&
      output && output.sizeBytes()>=4096*4 && diagnostics && diagnostics.sizeBytes()>=4,
      "Private decode MPP coefficient sample geometry differs");
  for (const auto &source : {p.weights->buffer,p.scales->buffer,p.biases->buffer,diagnostics})
    require(!output.sameView(source),"Private decode coefficient output/source alias");
  const FlashMoEDecodeF32CoefficientParams parameters{{1,1,p.inputSize,p.outputSize,512,4,64,0,
      p.weightRowStrideBytes,p.weightExpertStrideBytes,p.parameterRowStrideBytes,p.parameterExpertStrideBytes},
      expert,outputBegin,inputBegin,0};
  graph.add(bf16Operand ? "flash_moe_decode_bf16_coefficient_sample" : "flash_moe_decode_f32_coefficient_sample",{p.weights->buffer,p.scales->buffer,
      p.biases->buffer,output,diagnostics},parameters,{16,1,1},{256,1,1});
}
}
