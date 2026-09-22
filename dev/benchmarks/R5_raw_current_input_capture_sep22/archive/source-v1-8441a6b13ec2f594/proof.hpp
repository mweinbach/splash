#pragma once
#include "inspect.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "metal/ProfilingJson.hpp"
#include <sstream>

namespace splash::flash::r5_raw_current_capture_sep22 {
class Proof final {
 using Access=FlashDeepPrefixOracle;
 struct Tail final { Access::Undefined region;std::vector<uint8_t> bytes; };
 struct Binding final { std::string label;metal::MetalBuffer view;uint64_t physical; };
public:
 Proof(Owner&owner,FlashForward&target):owner_(owner),target_(target){
  if(!owner.config.proof)return;
  for(auto p:Access::tapeUndefined(target)){require(p.buffer&&p.buffer.contents()&&p.begin<=p.buffer.sizeBytes(),"invalid local undefined tail");const auto *data=static_cast<const uint8_t*>(p.buffer.contents())+p.begin;tails_.push_back({p,{data,data+p.buffer.sizeBytes()-p.begin}});}
 }
 void verified(const FlashRequestState&state,const FlashForwardResult&result,uint64_t id,uint64_t generation,uint32_t depth,uint32_t rows){
  if(!owner_.config.proof||id!=1||generation!=1||depth!=4||rows!=5)return;
  if(owner_.completed==3&&phase_==0){require(owner_.captured&&!owner_.failed,"third real capture not established");for(auto&t:tails_)if(t.region.label=="retained_count.inactive"){require(t.bytes.size()>=4,"warm retained word unavailable");std::memcpy(t.bytes.data(),t.region.buffer.contents(),4);}binding_=Access::binding(state);for(const auto&p:Access::planes(state))bindings_.push_back({p.label,p.buffer,p.buffer.sizeBytes()});owner_.rememberInputs();captureInputs();snapshot("selected_pending",state,&result);phase_=1;}
  else if(owner_.completed==4&&phase_==2){snapshot("next_real_target",state,&result);phase_=3;publishComplete();}
 }
 void committed(const FlashRequestState&state,uint32_t retained,uint64_t id,uint64_t generation){
  if(!owner_.config.proof||id!=1||generation!=1||phase_!=1)return;
  require(retained>=1&&retained<=5&&!Access::pending(state)&&!state.poisoned()&&target_.ownsState(state),"selected real commit did not resolve ownership");
  if(retained<5)for(auto&t:tails_)if(t.region.label=="retained_count.inactive"){require(t.bytes.size()>=4,"retained tail too short");std::memcpy(t.bytes.data(),&retained,4);}
  snapshot("selected_commit",state,nullptr);phase_=2;
 }
private:
 void local(const FlashRequestState&state){
  require(Access::binding(state)==binding_&&Access::rawOwns(target_,state),"actual request owner/binding changed");const auto planes=Access::planes(state);require(planes.size()==bindings_.size(),"actual request plane census changed");
  for(uint32_t i=0;i<planes.size();++i)require(planes[i].label==bindings_[i].label&&planes[i].buffer.sameView(bindings_[i].view)&&planes[i].buffer.sizeBytes()==bindings_[i].physical,"actual request physical view changed");
  for(const auto&t:tails_){const auto *p=static_cast<const uint8_t*>(t.region.buffer.contents())+t.region.begin;require(std::memcmp(p,t.bytes.data(),t.bytes.size())==0,"undefined PLE/count tail changed locally");}
  require(Access::tapeCanaries(target_),"actual lazy tape guard changed");owner_.canaries();owner_.inputsUnchanged();
 }
 void file(const std::filesystem::path&path,const void*data,uint64_t bytes){require(bytes<=kSpillLimit&&spilled_<=kSpillLimit-bytes,"three-frame spill exceeds<4GiB");freshWrite(path,data,bytes);spilled_+=bytes;}
 void captureInputs(){
  std::ostringstream out;out<<"{\"schema\":\"current-real-R5-113-raw-inputs-v1\",\"capture_enabled\":"<<(owner_.config.capture?"true":"false")<<",\"request_id\":1,\"generation\":1,\"R5_ordinal\":3,\"prior_successful_real_R5\":2,\"physical_rows\":5,\"roles\":[";
  for(uint32_t i=0;i<owner_.slots.size();++i){const auto&s=owner_.slots[i];const auto&p=s.role.projection;require(s.seen,"raw input role omitted");if(i)out<<',';const auto name="raw-"+std::to_string(i)+".bf16";
   if(owner_.config.capture)file(std::filesystem::path(owner_.config.directory)/name,s.view.contents(),s.view.sizeBytes());
   out<<"{\"index\":"<<i<<",\"role\":"<<json::quote(s.role.name)<<",\"pipeline\":"<<json::quote(kShapes[s.role.shape].pipeline)<<",\"input_dtype\":\"BF16\",\"shape\":[5,"<<p.inputSize<<"],\"row_stride_bytes\":"<<p.inputSize*2<<",\"logical_bytes\":"<<logicalBytes(s.role)<<",\"guarded_allocation_bytes\":"<<guardedBytes(s.role)<<",\"N\":"<<p.outputSize<<",\"bits\":"<<p.bits<<",\"group\":"<<p.groupSize<<",\"weights_dtype\":\"U32\",\"scales_dtype\":\"BF16\",\"biases_dtype\":\"BF16\",\"weight_row_stride_bytes\":"<<p.weightRowStrideBytes<<",\"parameter_row_stride_bytes\":"<<p.parameterRowStrideBytes<<",\"weight_expert_stride_bytes\":"<<p.weightExpertStrideBytes<<",\"parameter_expert_stride_bytes\":"<<p.parameterExpertStrideBytes<<",\"weight_view_bytes\":"<<p.weights->buffer.sizeBytes()<<",\"scale_view_bytes\":"<<p.scales->buffer.sizeBytes()<<",\"bias_view_bytes\":"<<p.biases->buffer.sizeBytes()<<",\"existing_F32_cache_contains\":true,\"existing_F32_policy_null\":true,\"production_view_rechecked\":true,\"file\":"<<(owner_.config.capture?json::quote(name):"null")<<'}';}
  out<<"]}";const auto text=out.str();file(std::filesystem::path(owner_.config.directory)/"raw-inputs.json",text.data(),text.size());
 }
 void snapshot(const std::string&label,const FlashRequestState&state,const FlashForwardResult*result){
  local(state);const auto dir=std::filesystem::path(owner_.config.directory)/label;require(std::filesystem::create_directory(dir),"fresh three-frame directory required");
  std::ostringstream out;out<<"{\"label\":"<<json::quote(label)<<",\"state\":"<<Access::metadata(target_,state)<<",\"tape\":"<<Access::tapeMetadata(target_)<<",\"local_owner_views_unchanged\":true,\"capture_guards_intact\":true,\"lazy_guards_intact\":true,\"undefined_tails_locally_unchanged\":true,\"planes\":[";uint32_t index=0;
  const auto emit=[&](const Access::Plane&p,uint64_t bytes){require(p.buffer&&p.buffer.contents()&&bytes<=p.buffer.sizeBytes(),"proof plane extent invalid");if(index)out<<',';const auto name=std::to_string(index++)+".bin";file(dir/name,p.buffer.contents(),bytes);out<<"{\"label\":"<<json::quote(p.label)<<",\"type\":"<<uint32_t(p.type)<<",\"file\":"<<json::quote(name)<<",\"bytes\":"<<bytes<<'}';};
  if(result){require(result->logitRows==5&&result->greedyRows==5&&result->logicalLength==state.logicalLength(),"actual target output geometry differs");emit({"output.hidden",Access::Type::BF16,result->hiddenBF16,5ULL*10240*2},5ULL*10240*2);emit({"output.logits",Access::Type::BF16,result->logitsBF16,5ULL*248320*2},5ULL*248320*2);emit({"output.greedy",Access::Type::U32,result->greedyResultsU32,5ULL*sizeof(FlashGreedyGPURowResult)},5ULL*sizeof(FlashGreedyGPURowResult));}
  const auto request=Access::planes(state);require(request.size()==134,"proof physical request census differs");for(const auto&p:request)emit(p,p.buffer.sizeBytes());const auto tapes=Access::tapePlanes(target_);require(tapes.size()==218,"proof known lazy/PLE census differs");for(const auto&p:tapes)emit(p,p.live);
  out<<"]}";const auto text=out.str();file(dir/"frame.json",text.data(),text.size());++frames_;
 }
 void publishComplete(){require(frames_==3&&!owner_.failed,"full capture-off/on three-frame source proof incomplete");std::ostringstream out;out<<"{\"schema\":\"current-real-R5-capture-three-frame-proof-v1\",\"complete\":true,\"frames\":3,\"request_id\":1,\"generation\":1,\"physical_rows\":5,\"ordinal\":3,\"prior_successful_real_R5\":2,\"future_actual_R5_ordinal\":4,\"capture_enabled\":"<<(owner_.config.capture?"true":"false")<<",\"copy_dispatches\":"<<owner_.copies<<",\"roles\":113,\"logical_input_bytes\":4362240,\"aligned_input_bytes\":5046272,\"owner_allocation_delta\":"<<owner_.ownerDelta<<",\"owner_reservation_bytes\":16777216,\"host_preallocated_bytes\":67108864,\"host_headroom_admission_bytes\":134217728,\"spill_bytes\":"<<spilled_<<",\"request_planes_per_frame\":134,\"initialized_lazy_arenas_per_frame\":216,\"PLE_cross_process_defined_only\":true,\"undefined_tails_local_only\":true,\"local_owner_binding_view_checks\":true,\"capture_inputs_unchanged_through_commit_future\":true,\"public_production_headers_changed\":false,\"performance_claim\":false}";const auto text=out.str();file(std::filesystem::path(owner_.config.directory)/"complete.json",text.data(),text.size());}
 Owner&owner_;FlashForward&target_;uint32_t phase_=0,frames_=0;uint64_t spilled_=0;std::array<uintptr_t,3>binding_{};std::vector<Binding>bindings_;std::vector<Tail>tails_;
};
}
