#pragma once
// Root-run untimed safety harness for the private vector kernels. Including or
// CPU-building this file creates no device and reads no model payload.
#include "flash/FlashGatheredI8QMV.hpp"
#include "prefill4k_allrows_qmv_one_layer.hpp"
#include "metal/CommandGraph.hpp"
#include <array>
#include <cstring>
#include <sstream>
#include <vector>

namespace splash::flash::gemv_decode_sep21_safety {
using metal::MetalBuffer;
using metal::MetalBackend;
using metal::CommandGraph;
struct VariantReport {
  uint32_t lanes=0;
  bool invalidGatePoison=false,invalidHiddenScanned=false,invalidDownPoison=false;
  bool duplicateOwnership=false,nonfinitePositiveZeroPair=false,corruptRankPoison=false;
  bool fixturesImmutable=true;
  std::array<uint32_t,8>diagnostics{};
  bool pass() const {
    return invalidGatePoison&&invalidHiddenScanned&&invalidDownPoison&&duplicateOwnership&&
        nonfinitePositiveZeroPair&&corruptRankPoison&&fixturesImmutable;
  }
};
struct Report {
  std::array<VariantReport,2>variants;
  bool pass() const { return variants[0].pass()&&variants[1].pass(); }
  std::string json() const {
    std::ostringstream out;
    out<<"{\"safety_pass\":"<<(pass()?"true":"false")<<",\"variants\":[";
    for (uint32_t v=0;v<2;++v) {
      if(v)out<<','; const auto &r=variants[v];
      out<<"{\"lanes\":"<<r.lanes<<",\"invalid_gate_poison\":"<<(r.invalidGatePoison?"true":"false")
          <<",\"invalid_hidden_scanned\":"<<(r.invalidHiddenScanned?"true":"false")
          <<",\"invalid_down_poison\":"<<(r.invalidDownPoison?"true":"false")
          <<",\"duplicate_separate_ownership\":"<<(r.duplicateOwnership?"true":"false")
          <<",\"nonfinite_positive_zero_bitwise_pair\":"<<(r.nonfinitePositiveZeroPair?"true":"false")
          <<",\"corrupt_rank_poison\":"<<(r.corruptRankPoison?"true":"false")
          <<",\"fixtures_immutable\":"<<(r.fixturesImmutable?"true":"false")<<",\"diagnostics\":[";
      for(uint32_t i=0;i<r.diagnostics.size();++i){if(i)out<<',';out<<r.diagnostics[i];}out<<"]}";
    }
    out<<"]}"; return out.str();
  }
};
inline bool poison(MetalBuffer values,uint64_t begin,uint64_t count) {
  const auto *bits=static_cast<const uint16_t *>(values.contents());
  for(uint64_t i=0;i<count;++i)if(bits[begin+i]!=0x7fc0u)return false;
  return true;
}
inline bool finite(MetalBuffer values) {
  const auto *bits=static_cast<const uint16_t *>(values.contents());
  for(uint64_t i=0;i<values.sizeBytes()/2;++i)if((bits[i]&0x7f80u)==0x7f80u)return false;
  return true;
}
template<class Allocate,class T>
MetalBuffer upload(Allocate &allocate,const std::vector<T> &values) {
  auto buffer=allocate(uint64_t(values.size())*sizeof(T));
  std::memcpy(buffer.contents(),values.data(),buffer.sizeBytes()); return buffer;
}
template<class Allocate>
Report run(MetalBackend &backend,const qmv_one_layer::OneLayerPayload &layer,uint32_t rows,
    const std::vector<uint16_t> &hidden,const std::vector<int64_t> &ids,Allocate allocate) {
  if(!rows||rows>16||hidden.size()!=uint64_t{rows}*2560||ids.size()!=uint64_t{rows}*10)
    throw std::invalid_argument("vector GEMV safety fixture geometry differs");
  const uint64_t routes=uint64_t{rows}*10,actElements=routes*640,downElements=routes*2560;
  const FlashGatheredI8QMVParams p{rows,10,512,0};
  auto x=upload(allocate,hidden),idBuffer=upload(allocate,ids);
  std::vector<int64_t>invalidIDs(ids.size(),-1),duplicateIDs=ids;duplicateIDs[1]=duplicateIDs[0];
  auto invalid=upload(allocate,invalidIDs),duplicates=upload(allocate,duplicateIDs);
  auto malformedHidden=hidden,zeroHidden=hidden;
  malformedHidden[0]=0x7fc0u;malformedHidden[1]=0x7f80u;malformedHidden[2]=0xff80u;
  zeroHidden[0]=0;zeroHidden[1]=0;zeroHidden[2]=0;
  auto badX=upload(allocate,malformedHidden),zeroX=upload(allocate,zeroHidden);
  std::vector<uint16_t>badIntermediate(actElements,0x7fc0u);
  auto badAct=upload(allocate,badIntermediate);
  std::vector<uint32_t>corruptRanks(layer.ranks.sizeBytes()/4);
  std::memcpy(corruptRanks.data(),layer.ranks.contents(),layer.ranks.sizeBytes());
  if(ids[0]<0||ids[0]>=512)throw std::invalid_argument("vector GEMV safety requires original valid IDs");
  corruptRanks[uint32_t(ids[0])]=512;auto corrupt=upload(allocate,corruptRanks);
  Report result;
  for(uint32_t v=0;v<2;++v) {
    const uint32_t lanes=v?16:32,outputs=128/lanes;
    const std::string prefix=v?"gemv_decode_sep21_v4_l16_o8":"gemv_decode_sep21_v4_l32_o4";
    auto &r=result.variants[v];r.lanes=lanes;
    auto diag=allocate(4),act=allocate(actElements*2),down=allocate(downElements*2);
    auto reset=[&]{*static_cast<uint32_t *>(diag.contents())=0x80000000u;
      std::memset(act.contents(),0xa5,act.sizeBytes());std::memset(down.contents(),0xa5,down.sizeBytes());};
    auto diagnostic=[&]{return *static_cast<const uint32_t *>(diag.contents());};
    auto gate=[&](CommandGraph &g,MetalBuffer input,MetalBuffer routesBuffer,MetalBuffer ranks) {
      g.add(prefix+"_gate_up",{input,layer.codes[0],layer.scales[0],layer.codes[1],layer.scales[1],
          ranks,routesBuffer,act,diag},p,{640/outputs,rows,10},{128,1,1});
    };
    auto lower=[&](CommandGraph &g,MetalBuffer input,MetalBuffer routesBuffer,MetalBuffer ranks) {
      g.add(prefix+"_down",{input,layer.codes[2],layer.scales[2],ranks,routesBuffer,down,diag},
          p,{2560/outputs,rows,10},{128,1,1});
    };
    reset();CommandGraph invalidGate;gate(invalidGate,x,invalid,layer.ranks);
    (void)backend.submitCommand(invalidGate.dispatches());r.diagnostics[0]=diagnostic();
    r.invalidGatePoison=poison(act,0,actElements)&&r.diagnostics[0]==0x80000001u;
    reset();CommandGraph invalidHidden;gate(invalidHidden,badX,invalid,layer.ranks);
    (void)backend.submitCommand(invalidHidden.dispatches());r.diagnostics[1]=diagnostic();
    r.invalidHiddenScanned=poison(act,0,actElements)&&r.diagnostics[1]==0x80000005u;
    reset();CommandGraph invalidDown;lower(invalidDown,badAct,invalid,layer.ranks);
    (void)backend.submitCommand(invalidDown.dispatches());r.diagnostics[2]=diagnostic();
    r.invalidDownPoison=poison(down,0,downElements)&&r.diagnostics[2]==0x80000005u;
    reset();CommandGraph normal;gate(normal,x,idBuffer,layer.ranks);lower(normal,act,idBuffer,layer.ranks);
    (void)backend.submitCommand(normal.dispatches());const bool cleanFinite=finite(act)&&finite(down)&&diagnostic()==0x80000000u;
    std::array<uint16_t,640>cleanRouteAct;std::array<uint16_t,2560>cleanRouteDown;
    std::memcpy(cleanRouteAct.data(),act.contents(),640*2);std::memcpy(cleanRouteDown.data(),down.contents(),2560*2);
    reset();CommandGraph duplicate;gate(duplicate,x,duplicates,layer.ranks);lower(duplicate,act,duplicates,layer.ranks);
    (void)backend.submitCommand(duplicate.dispatches());r.diagnostics[3]=diagnostic();
    r.duplicateOwnership=cleanFinite&&finite(act)&&finite(down)&&
        !std::memcmp(cleanRouteAct.data(),act.contents(),640*2)&&
        !std::memcmp(cleanRouteAct.data(),static_cast<const uint16_t *>(act.contents())+640,640*2)&&
        !std::memcmp(cleanRouteDown.data(),down.contents(),2560*2)&&
        !std::memcmp(cleanRouteDown.data(),static_cast<const uint16_t *>(down.contents())+2560,2560*2)&&r.diagnostics[3]==0x80000001u;
    reset();CommandGraph nonfinite;gate(nonfinite,badX,idBuffer,layer.ranks);lower(nonfinite,act,idBuffer,layer.ranks);
    (void)backend.submitCommand(nonfinite.dispatches());r.diagnostics[4]=diagnostic();
    std::vector<uint16_t>savedAct(actElements),savedDown(downElements);
    std::memcpy(savedAct.data(),act.contents(),act.sizeBytes());std::memcpy(savedDown.data(),down.contents(),down.sizeBytes());
    reset();CommandGraph positiveZero;gate(positiveZero,zeroX,idBuffer,layer.ranks);lower(positiveZero,act,idBuffer,layer.ranks);
    (void)backend.submitCommand(positiveZero.dispatches());r.diagnostics[5]=diagnostic();
    r.nonfinitePositiveZeroPair=!std::memcmp(savedAct.data(),act.contents(),act.sizeBytes())&&
        !std::memcmp(savedDown.data(),down.contents(),down.sizeBytes())&&r.diagnostics[4]==0x80000004u&&r.diagnostics[5]==0x80000000u;
    reset();CommandGraph corruptGate;gate(corruptGate,x,idBuffer,corrupt);
    (void)backend.submitCommand(corruptGate.dispatches());r.diagnostics[6]=diagnostic();
    const bool corruptGatePass=poison(act,0,640)&&r.diagnostics[6]==0x80000001u;
    reset();CommandGraph corruptDown;lower(corruptDown,badAct,idBuffer,corrupt);
    (void)backend.submitCommand(corruptDown.dispatches());r.diagnostics[7]=diagnostic();
    // Other valid routes deliberately contain nonfinite A and retain bit4;
    // corrupt route0 is excluded and poisons with bit5 without reading its A.
    r.corruptRankPoison=corruptGatePass&&poison(down,0,2560)&&r.diagnostics[7]==0x80000005u;
    r.fixturesImmutable=!std::memcmp(hidden.data(),x.contents(),x.sizeBytes())&&
        !std::memcmp(ids.data(),idBuffer.contents(),idBuffer.sizeBytes())&&
        !std::memcmp(invalidIDs.data(),invalid.contents(),invalid.sizeBytes())&&
        !std::memcmp(duplicateIDs.data(),duplicates.contents(),duplicates.sizeBytes())&&
        !std::memcmp(malformedHidden.data(),badX.contents(),badX.sizeBytes())&&
        !std::memcmp(zeroHidden.data(),zeroX.contents(),zeroX.sizeBytes())&&
        !std::memcmp(badIntermediate.data(),badAct.contents(),badAct.sizeBytes())&&
        !std::memcmp(corruptRanks.data(),corrupt.contents(),corrupt.sizeBytes());
  }
  return result;
}
} // namespace splash::flash::gemv_decode_sep21_safety
