// Oracle-clone-only READONLY access. Existing worker methods/bodies are intact.
namespace splash::flash {
std::vector<r5_raw_current_capture_sep22::Role> FlashDeepPrefixOracle::captureInventory(const FlashForward &target){
 using namespace r5_raw_current_capture_sep22;
 require(target.impl_!=nullptr,"capture inventory target unavailable");auto &s=*target.impl_;std::lock_guard lock(s.mutex);
 require(s.maximumVerifyRows==5&&s.maximumRows>=5&&s.descriptor.layers==48&&!s.descriptor.pleLayerIndices.empty()&&s.cacheFloat&&s.selectiveFloat&&s.floatDenseCache&&s.lazyGDN&&flashAffineFastEnabled(),"capture requires unchanged current selective F32/R5/lazy policy");
 require(s.weights.sourceIdentity()=="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e","capture original trained model source differs");
 std::vector<Role> out;out.reserve(kRoles);std::array<uint32_t,7> counts{};uint64_t logical=0,aligned=0;
 const auto add=[&](const std::string &name){const auto&p=s.weights.projection(name);uint32_t shape=uint32_t(kShapes.size());for(uint32_t i=0;i<kShapes.size();++i){const auto&t=kShapes[i];if(p.inputSize==t.K&&p.outputSize==t.N&&p.bits==t.bits&&p.groupSize==t.group){shape=i;break;}}if(shape==kShapes.size())return;
  require(p.experts==1&&p.weights&&p.scales&&p.biases&&p.weights->dtype==FlashDType::U32&&p.scales->dtype==FlashDType::BF16&&p.biases->dtype==FlashDType::BF16,"capture original quantdtype invalid");
  require(p.weights->buffer&&p.scales->buffer&&p.biases->buffer&&p.weights->buffer.contents()&&p.scales->buffer.contents()&&p.biases->buffer.contents(),"capture original shared quantized views unavailable");
  const uint64_t packed=uint64_t{p.inputSize}*p.bits/8,parameter=uint64_t{p.inputSize/p.groupSize}*2;
  require(p.inputSize%p.groupSize==0&&p.weightRowStrideBytes==packed&&p.parameterRowStrideBytes==parameter&&p.weightExpertStrideBytes>=uint64_t{p.outputSize}*packed&&p.parameterExpertStrideBytes>=uint64_t{p.outputSize}*parameter,"capture original stride census differs");
  require(p.weights->logicalBytes==uint64_t{p.outputSize}*packed&&p.scales->logicalBytes==uint64_t{p.outputSize}*parameter&&p.biases->logicalBytes==uint64_t{p.outputSize}*parameter&&p.weights->buffer.sizeBytes()>=p.weights->logicalBytes&&p.scales->buffer.sizeBytes()>=p.scales->logicalBytes&&p.biases->buffer.sizeBytes()>=p.biases->logicalBytes,"capture source plane extent invalid");
  require(s.floatDenseCache->contains(name)&&!flashFloatDenseSmallRowsPolicy(name,5,p.outputSize,p.inputSize,p.bits,p.groupSize),"capture role is not existing cache-member/null-F32 raw branch");
  Role role{name,p,shape};logical+=logicalBytes(role);aligned+=guardedBytes(role);++counts[shape];out.push_back(std::move(role));};
 for(uint32_t layer=0;layer<48;++layer){const auto p="language_model.model.layers."+std::to_string(layer);if(s.descriptor.layerKinds[layer]==FlashLayerKind::GatedDeltaNet){add(p+".linear_attn.in_proj_qkv");add(p+".linear_attn.in_proj_z");add(p+".linear_attn.out_proj");}else{add(p+".self_attn.q_proj");add(p+".self_attn.o_proj");}if(layer==s.descriptor.pleLayerIndices.front())add(p+".ple.key_proj");}
 require(out.size()==kRoles&&logical==kLogicalBytes&&aligned==kAlignedBytes,"capture exact113-role/byte inventory differs");for(uint32_t i=0;i<counts.size();++i)require(counts[i]==kShapes[i].calls,"capture seven observed shape/format inventory differs");return out;
}
std::vector<FlashDeepPrefixOracle::Plane> FlashDeepPrefixOracle::tapePlanes(const FlashForward &target){
  if(!target.impl_||!target.impl_->lazyGDN)throw std::logic_error("full tape inspection requires lazy GDN");
  auto &s=*target.impl_;std::lock_guard lock(s.mutex);std::vector<Plane> out;
  const std::array<Type,6> types{Type::F32,Type::BF16,Type::BF16,Type::BF16,Type::F32,Type::BF16};
  const std::array<const char *,6> names{"initial_state","initial_history","raw_qkv","mixed","decay","beta"};
  for(uint32_t layer=0;layer<48;++layer)if(s.lazyGDNRecords[layer]){
    auto &r=*s.lazyGDNRecords[layer];const auto arenas=r.arenaBuffers();
    if(r.maximumRows()!=5||r.maximumLanes()!=1||arenas.size()!=6)throw std::logic_error("tape arena shape differs");
    for(uint32_t i=0;i<6;++i)out.push_back({"gdn."+std::to_string(layer)+"."+names[i],types[i],arenas[i],arenas[i].sizeBytes()});
  }
  if(out.size()!=216)throw std::logic_error("tape inventory differs");
  out.push_back({"ple.initial_history.defined",Type::I64,s.backend.view(s.beforePLEHistory,0,16),16});
  out.push_back({"ple.initial_convolution.defined",Type::BF16,s.backend.view(s.beforePLEConvolution,0,9ULL*10240*2),9ULL*10240*2});
  return out;
}
std::vector<FlashDeepPrefixOracle::Undefined> FlashDeepPrefixOracle::tapeUndefined(const FlashForward &target){
  if(!target.impl_)throw std::logic_error("tape target unavailable");auto &s=*target.impl_;std::lock_guard lock(s.mutex);
  return {{"ple.initial_history.inactive",s.beforePLEHistory,16},{"ple.initial_convolution.inactive",s.beforePLEConvolution,9ULL*10240*2},{"retained_count.inactive",s.retainedCount,0}};
}
std::string FlashDeepPrefixOracle::tapeMetadata(const FlashForward &target){
  if(!target.impl_)throw std::logic_error("tape target unavailable");auto &s=*target.impl_;std::lock_guard lock(s.mutex);
  std::ostringstream out;out<<"{\"lazy\":"<<(s.lazyGDN?"true":"false")<<",\"maximum_verify_rows\":"<<s.maximumVerifyRows<<",\"pending_rows\":"<<s.pendingRows<<",\"pending_begin\":"<<s.pendingBegin<<",\"active_records\":";
  uint32_t active=0;for(const auto &r:s.lazyGDNRecords)if(r&&r->pending())++active;
  out<<active<<",\"cross_process_scope\":\"all216 known physical lazy arenas including inactive/padding; PLE defined ranges; undefined PLE/count tails checked unchanged locally\"}";return out.str();
}
bool FlashDeepPrefixOracle::tapeCanaries(const FlashForward &target){
  if(!target.impl_)return false;auto &s=*target.impl_;std::lock_guard lock(s.mutex);
  for(const auto &r:s.lazyGDNRecords)if(r&&!r->canariesIntact())return false;return true;
}
uint32_t FlashDeepPrefixOracle::retainedValue(const FlashForward &target){
  if(!target.impl_||!target.impl_->retainedCount.contents())throw std::logic_error("retained buffer unavailable");uint32_t value=0;std::memcpy(&value,target.impl_->retainedCount.contents(),4);return value;
}
bool FlashDeepPrefixOracle::rawOwns(const FlashForward &target,const FlashRequestState &request){return target.impl_&&request.impl_&&request.impl_->owner==target.impl_->owner&&request.impl_->capacity==target.impl_->capacity;}
std::array<uint64_t,3> FlashDeepPrefixOracle::tapeStatus(const FlashForward &target){
  if(!target.impl_)throw std::logic_error("tape target unavailable");auto &s=*target.impl_;std::lock_guard lock(s.mutex);uint64_t active=0;for(const auto &r:s.lazyGDNRecords)if(r&&r->pending())++active;return{s.pendingRows,s.pendingBegin,active};
}
}
