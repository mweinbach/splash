// Private oracle-only metadata checks against actual owned R5 scratch views.
// No graph here is submitted; no operand contents are read or modified.
namespace splash::flash {
uint32_t FlashDeepPrefixOracle::actualR5GuardProbes(FlashForward&target){
  if(!target.impl_||!target.impl_->int8ExpertStore)throw std::logic_error("R5 owned Store unavailable");
  auto&p=*target.impl_;const auto baseline=p.blockedScratch;
  auto input=p.bf(Scratch::Mixed,5,2560),ids=p.view(Scratch::ExpertIDs,5ULL*10*8),diag=p.view(Scratch::Diagnostics,4);
  const bool candidate=compact_native_r5_verify_sep22::requested();uint32_t cases=0;
  const auto check=[&](const FlashMoEBlockedScratch&s,metal::MetalBuffer x,metal::MetalBuffer i,metal::MetalBuffer d){
    const auto submits=p.backend.submissionCount();bool oldRejected=false,newRejected=false;metal::CommandGraph oldGraph,newGraph;
    try{addMoEBlockedPack(oldGraph,x,i,s,d,5,FlashMoEBlockedTile::M16N64,10);}catch(const std::invalid_argument&){oldRejected=true;}catch(const std::logic_error&){oldRejected=true;}
    if(candidate){try{compact_native_r5_verify_sep22::addSetup(newGraph,x,i,s,d,5,FlashMoEBlockedTile::M16N64,true,true);}catch(const std::invalid_argument&){newRejected=true;}catch(const std::logic_error&){newRejected=true;}}
    if(!oldRejected||(candidate&&(!newRejected||!newGraph.empty()))||p.backend.submissionCount()!=submits)throw std::runtime_error("actual R5 old/candidate negative decision differs");++cases;
  };
  {metal::CommandGraph oldGraph,newGraph;addMoEBlockedPack(oldGraph,input,ids,baseline,diag,5,FlashMoEBlockedTile::M16N64,10);
    if(oldGraph.dispatches().size()!=6)throw std::logic_error("actual R5 original canonical setup count differs");
    if(candidate){compact_native_r5_verify_sep22::addSetup(newGraph,input,ids,baseline,diag,5,FlashMoEBlockedTile::M16N64,true,true);if(newGraph.dispatches().size()!=2)throw std::logic_error("actual R5 candidate canonical setup count differs");}
  }
  check(baseline,{},ids,diag);check(baseline,input,{},diag);check(baseline,input,ids,{});
  {auto s=baseline;s.buckets.counts={};check(s,input,ids,diag);}
  {auto s=baseline;s.buckets.routeMap={};check(s,input,ids,diag);}
  {auto s=baseline;s.buckets.counts=p.backend.view(s.buckets.counts,0,2044);check(s,input,ids,diag);}
  {auto s=baseline;s.buckets.counts=p.backend.view(s.buckets.offsets,0,2048);check(s,input,ids,diag);}
  check(baseline,input,ids,p.backend.view(input,0,4));
  check(baseline,input,p.backend.view(baseline.buckets.packedInputs,0,400),diag);
  return cases;
}
}
