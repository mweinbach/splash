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
 struct CampaignPlan final {uint32_t capacity=0;uint64_t request=0,tapes=0,localTails=0,outputs=0,inputs=0,metadata=0,total=0;bool operator==(const CampaignPlan&)const=default;};
 static constexpr uint64_t metadataFileLimit=1ULL<<20,metadataFiles=5,maximumFiles=3*(134+218)+6+113+metadataFiles;
 static uint64_t checkedAdd(uint64_t a,uint64_t b){require(a<=kSpillLimit&&b<=kSpillLimit&&a<=kSpillLimit-b,"whole capture campaign byte sum exceeds<4GiB");return a+b;}
 static uint64_t checkedTimes(uint64_t a,uint64_t b){require(!a||b<=kSpillLimit/a,"whole capture campaign byte product exceeds<4GiB");return a*b;}
 static CampaignPlan sourcePlan(uint32_t capacity){
  CampaignPlan p;p.capacity=capacity;p.request=FlashForward::requestStateBytes(capacity);
  p.tapes=checkedAdd(checkedTimes(36,FlashGDNLazyRollback::plannedBytes(5,1)),16+9ULL*10240*2);
  const auto physicalVerification=FlashForward::verificationWorkspaceBytes(5);require(physicalVerification>=p.tapes,"source verification/tape planner mismatch");p.localTails=physicalVerification-p.tapes;
  p.outputs=5ULL*10240*2+5ULL*248320*2+5ULL*sizeof(FlashGreedyGPURowResult);p.inputs=kLogicalBytes;
  // Five bounded JSON headers plus4KiB per possible payload/header file.
  // Undefined PLE/count tails are reserved conservatively, kept LOCAL only.
  p.metadata=checkedAdd(checkedTimes(metadataFiles,metadataFileLimit),checkedTimes(maximumFiles,4096));
  p.total=checkedAdd(checkedAdd(checkedTimes(3,p.request),checkedTimes(3,checkedAdd(p.tapes,p.localTails))),checkedAdd(checkedTimes(2,p.outputs),checkedAdd(p.inputs,p.metadata)));return p;
 }
 Proof(Owner&owner,FlashForward&target,CampaignPlan plan):owner_(owner),target_(target),plan_(plan){
  if(!owner.config.proof)return;
  require(plan_==sourcePlan(plan_.capacity),"capture campaign source plan changed/forged");
  for(auto p:Access::tapeUndefined(target)){require(p.buffer&&p.buffer.contents()&&p.begin<=p.buffer.sizeBytes(),"invalid local undefined tail");const auto *data=static_cast<const uint8_t*>(p.buffer.contents())+p.begin;tails_.push_back({p,{data,data+p.buffer.sizeBytes()-p.begin}});}
 }
 void verified(const FlashRequestState&state,const FlashForwardResult&result,uint64_t id,uint64_t generation,uint32_t depth,uint32_t rows){
  if(!owner_.config.proof||id!=1||generation!=1||depth!=4||rows!=5)return;
  if(owner_.completed==3&&phase_==0){require(owner_.captured&&!owner_.failed,"third real capture not established");actualPreflight(state,result);for(auto&t:tails_)if(t.region.label=="retained_count.inactive"){require(t.bytes.size()>=4,"warm retained word unavailable");std::memcpy(t.bytes.data(),t.region.buffer.contents(),4);}binding_=Access::binding(state);for(const auto&p:Access::planes(state))bindings_.push_back({p.label,p.buffer,p.buffer.sizeBytes()});owner_.rememberInputs();captureInputs();snapshot("selected_pending",state,&result);phase_=1;}
  else if(owner_.completed==4&&phase_==2){snapshot("next_real_target",state,&result);phase_=3;publishComplete();}
 }
 void committed(const FlashRequestState&state,uint32_t retained,uint64_t id,uint64_t generation){
  if(!owner_.config.proof||id!=1||generation!=1||phase_!=1)return;
  require(retained>=1&&retained<=5&&!Access::pending(state)&&!state.poisoned()&&target_.ownsState(state),"selected real commit did not resolve ownership");
  if(retained<5)for(auto&t:tails_)if(t.region.label=="retained_count.inactive"){require(t.bytes.size()>=4,"retained tail too short");std::memcpy(t.bytes.data(),&retained,4);}
  snapshot("selected_commit",state,nullptr);phase_=2;
 }
private:
 void actualPreflight(const FlashRequestState&state,const FlashForwardResult&result){
  require(!preflight_&&spilled_==0&&state.capacity()==plan_.capacity,"whole campaign preflight must precede first export");
  const auto request=Access::planes(state),tapes=Access::tapePlanes(target_);require(request.size()==134&&tapes.size()==218,"whole campaign actual plane inventory differs");uint64_t requestBytes=0,tapeBytes=0,tailBytes=0,inputBytes=0;
  for(const auto&p:request){require(p.buffer&&p.buffer.contents()&&p.live<=p.buffer.sizeBytes()&&p.label.size()<=256,"whole campaign actual request extent/header invalid");requestBytes=checkedAdd(requestBytes,p.buffer.sizeBytes());}
  for(uint32_t i=0;i<tapes.size();++i){const auto&p=tapes[i];require(p.buffer&&p.buffer.contents()&&p.live<=p.buffer.sizeBytes()&&p.label.size()<=256&&(i>=216||p.live==p.buffer.sizeBytes()),"whole campaign full initialized lazy/definedPLE extent invalid");tapeBytes=checkedAdd(tapeBytes,p.live);}
  require(tails_.size()==3,"whole campaign local PLE/count tail inventory differs");for(const auto&t:tails_){require(t.region.buffer&&t.region.begin<=t.region.buffer.sizeBytes()&&t.bytes.size()==t.region.buffer.sizeBytes()-t.region.begin,"whole campaign local tail extent invalid");tailBytes=checkedAdd(tailBytes,t.bytes.size());}
  require(owner_.slots.size()==113,"whole campaign input role inventory differs");for(const auto&s:owner_.slots){require(s.seen&&s.role.name.size()<=256&&(!owner_.config.capture||(s.view&&s.view.contents()&&s.view.sizeBytes()==logicalBytes(s.role))),"whole campaign actual input slot extent/header invalid");inputBytes=checkedAdd(inputBytes,logicalBytes(s.role));}
  require(result.logitRows==5&&result.greedyRows==5&&result.logicalLength==state.logicalLength()&&result.hiddenBF16&&result.hiddenBF16.contents()&&result.hiddenBF16.sizeBytes()>=5ULL*10240*2&&result.logitsBF16&&result.logitsBF16.contents()&&result.logitsBF16.sizeBytes()>=5ULL*248320*2&&result.greedyResultsU32&&result.greedyResultsU32.contents()&&result.greedyResultsU32.sizeBytes()>=5ULL*sizeof(FlashGreedyGPURowResult),"whole campaign actual Verify5 output inventory invalid");
  require(requestBytes==plan_.request&&tapeBytes==plan_.tapes&&tailBytes==plan_.localTails&&inputBytes==plan_.inputs,"whole campaign actual/source extent census differs");
  actualTotal_=checkedAdd(checkedAdd(checkedTimes(3,requestBytes),checkedTimes(3,checkedAdd(tapeBytes,tailBytes))),checkedAdd(checkedTimes(2,plan_.outputs),checkedAdd(inputBytes,plan_.metadata)));require(actualTotal_==plan_.total&&actualTotal_<=kSpillLimit,"whole actual three-frame campaign bound exceeds<4GiB");preflight_=true;
 }
 void local(const FlashRequestState&state){
  require(Access::binding(state)==binding_&&Access::rawOwns(target_,state),"actual request owner/binding changed");const auto planes=Access::planes(state);require(planes.size()==bindings_.size(),"actual request plane census changed");
  for(uint32_t i=0;i<planes.size();++i)require(planes[i].label==bindings_[i].label&&planes[i].buffer.sameView(bindings_[i].view)&&planes[i].buffer.sizeBytes()==bindings_[i].physical,"actual request physical view changed");
  for(const auto&t:tails_){const auto *p=static_cast<const uint8_t*>(t.region.buffer.contents())+t.region.begin;require(std::memcmp(p,t.bytes.data(),t.bytes.size())==0,"undefined PLE/count tail changed locally");}
  require(Access::tapeCanaries(target_),"actual lazy tape guard changed");owner_.canaries();owner_.inputsUnchanged();
 }
 void file(const std::filesystem::path&path,const void*data,uint64_t bytes){require(preflight_&&actualTotal_==plan_.total&&files_<maximumFiles,"WHOLE campaign preflight/file bound required before export");require(bytes<=actualTotal_&&spilled_<=actualTotal_-bytes,"three-frame spill exceeds whole preflight bound");if(path.extension()==".json")require(bytes<=metadataFileLimit&&metadataWritten_<metadataFiles,"campaign metadata/header bound exceeded");freshWrite(path,data,bytes);spilled_+=bytes;++files_;if(path.extension()==".json")++metadataWritten_;}
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
 Owner&owner_;FlashForward&target_;CampaignPlan plan_;uint32_t phase_=0,frames_=0;uint64_t spilled_=0,actualTotal_=0,files_=0,metadataWritten_=0;bool preflight_=false;std::array<uintptr_t,3>binding_{};std::vector<Binding>bindings_;std::vector<Tail>tails_;
};
}
