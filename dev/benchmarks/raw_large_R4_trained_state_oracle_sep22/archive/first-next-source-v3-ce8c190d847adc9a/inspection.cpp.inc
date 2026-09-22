// Oracle-clone-only READONLY access. Existing worker methods/bodies are intact.
namespace splash::flash {
std::vector<FlashDeepPrefixOracle::Plane> FlashDeepPrefixOracle::tapePlanes(const FlashForward &target){
  if(!target.impl_||!target.impl_->lazyGDN)throw std::logic_error("full tape inspection requires lazy GDN");
  auto &s=*target.impl_;std::lock_guard lock(s.mutex);std::vector<Plane> out;
  const std::array<Type,6> types{Type::F32,Type::BF16,Type::BF16,Type::BF16,Type::F32,Type::BF16};
  const std::array<const char *,6> names{"initial_state","initial_history","raw_qkv","mixed","decay","beta"};
  for(uint32_t layer=0;layer<48;++layer)if(s.lazyGDNRecords[layer]){
    auto &r=*s.lazyGDNRecords[layer];const auto arenas=r.arenaBuffers();
    if(r.maximumRows()!=4||r.maximumLanes()!=1||arenas.size()!=6)throw std::logic_error("tape arena shape differs");
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
