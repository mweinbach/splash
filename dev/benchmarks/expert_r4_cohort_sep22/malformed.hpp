#pragma once
// Included and CPU-compiled without invoking GPU/model operations. Root alone
// calls run() in the admitted one-layer oracle, outside shipping timings.
#include "abi.hpp"
#include "prefill4k_allrows_qmv_one_layer.hpp"
#include "metal/CommandGraph.hpp"
#include <array>
#include <cstring>
#include <sstream>
#include <vector>
namespace splash::flash::expert_r4_cohort_sep22_safety {
using metal::MetalBackend;using metal::MetalBuffer;using metal::CommandGraph;
struct Plan {MetalBuffer counts,offsets,map,inverse,jobs,count,stages;};
struct Case {std::string name;bool pass=false;uint32_t diagnostic=0;};
struct Report {
  std::vector<Case>cases;
  bool pass()const{return !cases.empty()&&std::all_of(cases.begin(),cases.end(),[](const Case&c){return c.pass;});}
  std::string json()const {std::ostringstream o;o<<"{\"pass\":"<<(pass()?"true":"false")<<",\"cases\":[";
    for(size_t i=0;i<cases.size();++i){if(i)o<<',';o<<"{\"name\":\""<<cases[i].name<<"\",\"pass\":"<<(cases[i].pass?"true":"false")<<",\"diagnostic\":"<<cases[i].diagnostic<<'}';}
    o<<"]}";return o.str();}
};
inline bool overlaps(MetalBuffer a,MetalBuffer b) {
  const auto aa=reinterpret_cast<uintptr_t>(a.contents()),bb=reinterpret_cast<uintptr_t>(b.contents());
  return aa>=bb?aa-bb<b.sizeBytes():bb-aa<a.sizeBytes();
}
inline void requireView(MetalBuffer b,uint64_t bytes,uint32_t align) {
  if(!b||b.storage()!=metal::BufferStorage::Shared||!b.contents()||b.sizeBytes()<bytes||reinterpret_cast<uintptr_t>(b.contents())%align)
    throw std::invalid_argument("R4 cohort safety view extent/storage/alignment differs");
}
inline void validate(const qmv_one_layer::OneLayerPayload&layer,const Plan&p,MetalBuffer h,MetalBuffer ids,MetalBuffer act,MetalBuffer down,MetalBuffer diag) {
  requireView(h,4*2560*2,8);requireView(ids,40*8,8);requireView(act,40*640*2,8);requireView(down,40*2560*2,8);requireView(diag,4,4);
  const std::array<MetalBuffer,7>meta{p.counts,p.offsets,p.map,p.inverse,p.jobs,p.count,p.stages};
  const std::array<uint64_t,7>sizes{512*4,513*4,40*4,40*4,40*8,4,513*4};
  for(uint32_t i=0;i<7;++i)requireView(meta[i],sizes[i],i==4?8:4);
  const std::array<MetalBuffer,12>views{h,ids,act,down,diag,p.counts,p.offsets,p.map,p.inverse,p.jobs,p.count,p.stages};
  for(uint32_t i=0;i<views.size();++i) {
    if(overlaps(views[i],layer.base)||overlaps(views[i],layer.ranks))throw std::invalid_argument("R4 cohort safety aliases immutable payload/ranks");
    for(uint32_t j=i+1;j<views.size();++j)if(overlaps(views[i],views[j]))throw std::invalid_argument("R4 cohort safety operand/metadata aliases");
  }
}
template<class Allocate,class T>MetalBuffer upload(Allocate&a,const std::vector<T>&v) {
  auto b=a(v.size()*sizeof(T));std::memcpy(b.contents(),v.data(),b.sizeBytes());return b;
}
template<class Allocate>Plan plan(Allocate&a) {
  return {a(512*4),a(513*4),a(40*4),a(40*4),a(40*8),a(4),a(513*4)};
}
inline bool poison(MetalBuffer b,uint64_t start,uint64_t count) {
  const auto*v=static_cast<const uint16_t*>(b.contents());for(uint64_t i=0;i<count;++i)if(v[start+i]!=0x7fc0)return false;return true;
}
inline bool finite(MetalBuffer b) {
  const auto*v=static_cast<const uint16_t*>(b.contents());for(uint64_t i=0;i<b.sizeBytes()/2;++i)if((v[i]&0x7f80u)==0x7f80u)return false;return true;
}
inline bool marker(const Plan&p,bool audit,bool down) {
  const auto*s=static_cast<const uint32_t*>(p.stages.contents());return s[0]==7&&s[1]==kExpertR4MetaReady&&s[4]==kExpertR4GateReady&&
      (!down||s[5]==kExpertR4DownReady)&&s[2]==(audit?kExpertR4GateCTAs:0)&&s[3]==(audit&&down?kExpertR4DownCTAs:0);
}
inline void addPlan(CommandGraph&g,const qmv_one_layer::OneLayerPayload&l,MetalBuffer h,MetalBuffer ids,const Plan&p,MetalBuffer diag,FlashExpertR4CohortParams params) {
  g.add("expert_r4_cohort_sep22_plan",{h,ids,l.ranks,p.counts,p.offsets,p.map,p.inverse,p.jobs,p.count,p.stages,diag},params,{1,1,1},{256,1,1});
}
inline void addGate(CommandGraph&g,const qmv_one_layer::OneLayerPayload&l,MetalBuffer h,MetalBuffer ids,const Plan&p,MetalBuffer act,MetalBuffer diag,FlashExpertR4CohortParams params,bool audit=false,bool partial=false) {
  g.add(audit?"expert_r4_cohort_sep22_gate_up_n16_audit":"expert_r4_cohort_sep22_gate_up_n16",{h,l.codes[0],l.scales[0],l.codes[1],l.scales[1],l.ranks,ids,p.offsets,p.map,p.inverse,p.jobs,p.count,p.stages,act,diag},params,{partial?1u:40u,partial?1u:10u,1},{128,1,1});
}
inline void addDown(CommandGraph&g,const qmv_one_layer::OneLayerPayload&l,MetalBuffer act,MetalBuffer ids,const Plan&p,MetalBuffer down,MetalBuffer diag,FlashExpertR4CohortParams params,bool audit=false) {
  g.add(audit?"expert_r4_cohort_sep22_down_n16_audit":"expert_r4_cohort_sep22_down_n16",{act,l.codes[2],l.scales[2],l.ranks,ids,p.offsets,p.map,p.inverse,p.jobs,p.count,p.stages,down,diag},params,{160,10,1},{128,1,1});
}
template<class Allocate>
Report run(MetalBackend&backend,const qmv_one_layer::OneLayerPayload&layer,const std::vector<uint16_t>&hidden,const std::vector<int64_t>&ids,Allocate allocate) {
  if(hidden.size()!=4*2560||ids.size()!=40)throw std::invalid_argument("R4 cohort safety needs exact R4 fixtures");
  const FlashExpertR4CohortParams params{4,10,512,16,10,7,0,0};
  auto p=plan(allocate);
  auto h=upload(allocate,hidden),id=upload(allocate,ids),act=allocate(40*640*2),down=allocate(40*2560*2),diag=allocate(4);
  validate(layer,p,h,id,act,down,diag);
  auto reset=[&]{*static_cast<uint32_t*>(diag.contents())=0x80000000u;std::memset(act.contents(),0xa5,act.sizeBytes());std::memset(down.contents(),0xa5,down.sizeBytes());};
  auto diagnostic=[&]{return *static_cast<const uint32_t*>(diag.contents());};
  auto untouched=[](MetalBuffer b){const auto*v=static_cast<const uint8_t*>(b.contents());return std::all_of(v,v+b.sizeBytes(),[](uint8_t x){return x==0xa5;});};
  Report report;
  reset();CommandGraph normal;addPlan(normal,layer,h,id,p,diag,params);addGate(normal,layer,h,id,p,act,diag,params,true);addDown(normal,layer,act,id,p,down,diag,params,true);
  (void)backend.submitCommand(normal.dispatches());
  std::vector<uint8_t>savedAct(act.sizeBytes()),savedDown(down.sizeBytes());std::memcpy(savedAct.data(),act.contents(),savedAct.size());std::memcpy(savedDown.data(),down.contents(),savedDown.size());
  report.cases.push_back({"normal_audit_complete",marker(p,true,true)&&finite(act)&&finite(down)&&diagnostic()==0x80000000u,diagnostic()});
  reset();CommandGraph shipping;addPlan(shipping,layer,h,id,p,diag,params);addGate(shipping,layer,h,id,p,act,diag,params);addDown(shipping,layer,act,id,p,down,diag,params);
  (void)backend.submitCommand(shipping.dispatches());
  report.cases.push_back({"shipping_audit_literal_stage_match",marker(p,false,true)&&!std::memcmp(savedAct.data(),act.contents(),savedAct.size())&&!std::memcmp(savedDown.data(),down.contents(),savedDown.size())&&diagnostic()==0x80000000u,diagnostic()});
  for(auto b:{p.counts,p.offsets,p.map,p.inverse,p.jobs,p.count,p.stages})std::memset(b.contents(),0xa5,b.sizeBytes());
  reset();CommandGraph missing;addGate(missing,layer,h,id,p,act,diag,params);addDown(missing,layer,act,id,p,down,diag,params);
  (void)backend.submitCommand(missing.dispatches());report.cases.push_back({"missing_plan_fails_before_operands",untouched(act)&&untouched(down)&&diagnostic()==0x80000002u,diagnostic()});
  reset();CommandGraph reorder;addPlan(reorder,layer,h,id,p,diag,params);addDown(reorder,layer,act,id,p,down,diag,params);
  (void)backend.submitCommand(reorder.dispatches());report.cases.push_back({"down_before_gate_fails_closed",untouched(down)&&diagnostic()==0x80000002u,diagnostic()});
  reset();CommandGraph partial;addPlan(partial,layer,h,id,p,diag,params);addGate(partial,layer,h,id,p,act,diag,params,false,true);addDown(partial,layer,act,id,p,down,diag,params);
  (void)backend.submitCommand(partial.dispatches());report.cases.push_back({"partial_gate_grid_rejected",untouched(act)&&untouched(down)&&diagnostic()==0x80000002u,diagnostic()});
  reset();auto wrong=params;wrong.epoch=8;CommandGraph epoch;addPlan(epoch,layer,h,id,p,diag,params);addGate(epoch,layer,h,id,p,act,diag,wrong);
  (void)backend.submitCommand(epoch.dispatches());report.cases.push_back({"wrong_metadata_epoch_rejected",untouched(act)&&diagnostic()==0x80000002u,diagnostic()});
  reset();CommandGraph cleanPlan;addPlan(cleanPlan,layer,h,id,p,diag,params);(void)backend.submitCommand(cleanPlan.dispatches());
  auto*map=static_cast<uint32_t*>(p.map.contents());map[0]=UINT_MAX;CommandGraph corrupt;addGate(corrupt,layer,h,id,p,act,diag,params);(void)backend.submitCommand(corrupt.dispatches());
  report.cases.push_back({"corrupt_live_route_map_rejected",untouched(act)&&diagnostic()==0x80000002u,diagnostic()});
  std::vector<int64_t>invalidIds(40,-1);auto invalid=upload(allocate,invalidIds);
  reset();CommandGraph invalidGate;addPlan(invalidGate,layer,h,invalid,p,diag,params);addGate(invalidGate,layer,h,invalid,p,act,diag,params,true);
  (void)backend.submitCommand(invalidGate.dispatches());report.cases.push_back({"invalid_gate_poison_without_early_numeric_bit",poison(act,0,40*640)&&marker(p,true,false)&&diagnostic()==0x80000001u,diagnostic()});
  auto badHidden=hidden,positiveZero=hidden;badHidden[3*2560]=0x7fc0;badHidden[3*2560+1]=0x7f80;badHidden[3*2560+2]=0xff80;
  positiveZero[3*2560]=0;positiveZero[3*2560+1]=0;positiveZero[3*2560+2]=0;
  auto badH=upload(allocate,badHidden),zeroH=upload(allocate,positiveZero);
  reset();CommandGraph invalidHidden;addPlan(invalidHidden,layer,badH,invalid,p,diag,params);addGate(invalidHidden,layer,badH,invalid,p,act,diag,params,true);
  (void)backend.submitCommand(invalidHidden.dispatches());report.cases.push_back({"all_invalid_still_inspects_fourth_hidden_row",poison(act,0,40*640)&&diagnostic()==0x80000005u,diagnostic()});
  std::vector<uint16_t>nanInput(40*640,0x7fc0);auto arbitrary=upload(allocate,nanInput);
  // A legal completed empty gate is required, then down poisons without reading
  // arbitrary excluded A; source review supplies the no-read proof separately.
  CommandGraph invalidDown;addDown(invalidDown,layer,arbitrary,invalid,p,down,diag,params,true);(void)backend.submitCommand(invalidDown.dispatches());
  report.cases.push_back({"invalid_down_poison_excluded_inputs",poison(down,0,40*2560)&&marker(p,true,true)&&diagnostic()==0x80000005u,diagnostic()});
  reset();CommandGraph nonfinite;addPlan(nonfinite,layer,badH,id,p,diag,params);addGate(nonfinite,layer,badH,id,p,act,diag,params,true);addDown(nonfinite,layer,act,id,p,down,diag,params,true);
  (void)backend.submitCommand(nonfinite.dispatches());const uint32_t badDiag=diagnostic();std::memcpy(savedAct.data(),act.contents(),savedAct.size());std::memcpy(savedDown.data(),down.contents(),savedDown.size());
  reset();CommandGraph zero;addPlan(zero,layer,zeroH,id,p,diag,params);addGate(zero,layer,zeroH,id,p,act,diag,params,true);addDown(zero,layer,act,id,p,down,diag,params,true);
  (void)backend.submitCommand(zero.dispatches());report.cases.push_back({"nonfinite_positive_zero_bitwise_pair",badDiag==0x80000004u&&diagnostic()==0x80000000u&&!std::memcmp(savedAct.data(),act.contents(),savedAct.size())&&!std::memcmp(savedDown.data(),down.contents(),savedDown.size()),diagnostic()});
  reset();CommandGraph clean;addPlan(clean,layer,h,id,p,diag,params);addGate(clean,layer,h,id,p,act,diag,params,true);addDown(clean,layer,act,id,p,down,diag,params,true);(void)backend.submitCommand(clean.dispatches());
  std::array<uint16_t,640>routeAct;std::array<uint16_t,2560>routeDown;std::memcpy(routeAct.data(),act.contents(),640*2);std::memcpy(routeDown.data(),down.contents(),2560*2);
  auto duplicateIds=ids;for(uint32_t row=0;row<4;++row){duplicateIds[row*10]=ids[0];duplicateIds[row*10+1]=ids[0];}auto duplicate=upload(allocate,duplicateIds);
  reset();CommandGraph dup;addPlan(dup,layer,h,duplicate,p,diag,params);addGate(dup,layer,h,duplicate,p,act,diag,params,true);addDown(dup,layer,act,duplicate,p,down,diag,params,true);(void)backend.submitCommand(dup.dispatches());
  const bool dupPair=!std::memcmp(routeAct.data(),act.contents(),640*2)&&!std::memcmp(routeAct.data(),static_cast<const uint16_t*>(act.contents())+640,640*2)&&!std::memcmp(routeDown.data(),down.contents(),2560*2)&&!std::memcmp(routeDown.data(),static_cast<const uint16_t*>(down.contents())+2560,2560*2);
  const uint32_t*counts=static_cast<const uint32_t*>(p.counts.contents());
  report.cases.push_back({"duplicate_ownership_and_split_cohorts",dupPair&&finite(act)&&finite(down)&&counts[uint32_t(ids[0])]>=8&&diagnostic()==0x80000001u,diagnostic()});
  auto corruptLayer=layer;std::vector<uint32_t>ranks(layer.ranks.sizeBytes()/4);std::memcpy(ranks.data(),layer.ranks.contents(),layer.ranks.sizeBytes());ranks[uint32_t(ids[0])]=512;corruptLayer.ranks=upload(allocate,ranks);
  reset();CommandGraph rank;addPlan(rank,corruptLayer,h,id,p,diag,params);addGate(rank,corruptLayer,h,id,p,act,diag,params,true);addDown(rank,corruptLayer,act,id,p,down,diag,params,true);(void)backend.submitCommand(rank.dispatches());
  report.cases.push_back({"corrupt_rank_excluded_before_coefficients",poison(act,0,640)&&poison(down,0,2560)&&diagnostic()==0x80000005u,diagnostic()});
  bool aliasRejected=false;try{auto alias=p;alias.counts=h;validate(layer,alias,h,id,act,down,diag);}catch(const std::invalid_argument&){aliasRejected=true;}
  report.cases.push_back({"host_metadata_operand_alias_rejected",aliasRejected,0});
  report.cases.push_back({"all_original_fixtures_immutable",!std::memcmp(hidden.data(),h.contents(),h.sizeBytes())&&!std::memcmp(ids.data(),id.contents(),id.sizeBytes())&&!std::memcmp(invalidIds.data(),invalid.contents(),invalid.sizeBytes())&&!std::memcmp(duplicateIds.data(),duplicate.contents(),duplicate.sizeBytes())&&!std::memcmp(badHidden.data(),badH.contents(),badH.sizeBytes())&&!std::memcmp(positiveZero.data(),zeroH.contents(),zeroH.sizeBytes())&&!std::memcmp(nanInput.data(),arbitrary.contents(),arbitrary.sizeBytes())&&!std::memcmp(ranks.data(),corruptLayer.ranks.contents(),corruptLayer.ranks.sizeBytes()),0});
  return report;
}
} // namespace splash::flash::expert_r4_cohort_sep22_safety
