// Actual owned-buffer guard probes. Metadata graph construction only: no submit.
namespace splash::flash {
uint32_t FlashDeepPrefixOracle::actualBundleGuardProbes(FlashForward &target){
  if(!target.impl_||!target.impl_->int8ExpertStore)throw std::logic_error("guard Store unavailable");
  auto &p=*target.impl_;auto &store=*p.int8ExpertStore;
  auto input=p.bf(Scratch::Mixed,4,2560),ids=p.view(Scratch::ExpertIDs,4ULL*10*8),diag=p.view(Scratch::Diagnostics,4);
  const auto baseline=p.blockedScratch;uint32_t cases=0;
  const auto check=[&](const FlashMoEBlockedScratch &scratch,metal::MetalBuffer x,metal::MetalBuffer tokenIDs,metal::MetalBuffer d,uint32_t rows,uint32_t selections){
    bool oldRejected=false,newRejected=false;metal::CommandGraph oldGraph,newGraph;
    try{store.addCompactNativeR4VerifyPack(oldGraph,0,x,tokenIDs,scratch,d,rows,selections);store.addCompactNativeR4VerifyGateUp(oldGraph,0,scratch,d,rows,selections);store.addCompactNativeR4VerifyDown(oldGraph,0,scratch,d,rows,selections);}catch(const std::invalid_argument&){oldRejected=true;}catch(const std::logic_error&){oldRejected=true;}
    try{store.addCompactNativeR4VerifyChain(newGraph,0,x,tokenIDs,scratch,d,rows,selections);}catch(const std::invalid_argument&){newRejected=true;}catch(const std::logic_error&){newRejected=true;}
    if(!oldRejected||!newRejected||oldRejected!=newRejected||!newGraph.empty())throw std::runtime_error("actual old/new guard decision differs");++cases;
  };
  check(baseline,input,ids,diag,1,10);check(baseline,input,ids,diag,4,9);
  check(baseline,input,ids,p.backend.view(input,0,4),4,10);
  check(baseline,input,p.backend.view(baseline.buckets.packedInputs,0,320),diag,4,10);
  {auto s=baseline;s.buckets.counts=p.backend.view(s.buckets.offsets,0,2048);check(s,input,ids,diag,4,10);}
  {auto s=baseline;s.scatteredDown=p.backend.view(s.packedActivated,0,4ULL*10*2560*2);check(s,input,ids,diag,4,10);}
  const auto immutable=store.immutableWeightBuffers();if(immutable.empty())throw std::logic_error("actual mapped source unavailable");
  check(baseline,p.backend.view(immutable[0],0,4ULL*2560*2),ids,diag,4,10);
  check(baseline,input,ids,p.backend.view(immutable[0],0,4),4,10);
  return cases;
}
}
