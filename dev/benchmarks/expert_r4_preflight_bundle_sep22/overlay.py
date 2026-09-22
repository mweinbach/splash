"""CPU guard-only overlay. Parent public native producers remain literal."""
PRIVATE='dev/benchmarks/expert_r4_preflight_bundle_sep22'
CHANGED={'runtime/flash/FlashInt8ExpertStore.hpp','runtime/flash/FlashInt8ExpertStore.mm',
         'runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm'}


def replace(text,before,after):
    if text.count(before)!=1:
        raise ValueError('Preflight overlay anchor drift: '+before[:100])
    return text.replace(before,after)


def native_suffix(text,name,next_name,start):
    block=text[text.index('void FlashInt8ExpertStore::'+name+'('):text.index('void FlashInt8ExpertStore::'+next_name+'(')]
    suffix=block[block.index(start):]
    if not suffix.rstrip().endswith('}'):
        raise ValueError('Original producer suffix has unexpected end')
    return suffix.rstrip()[:-1].rstrip()


def transform(relative,text):
    if relative in CHANGED:
        text=f'#include "{PRIVATE}/policy.hpp"\n'+text
    if relative=='runtime/flash/FlashInt8ExpertStore.hpp':
        text=replace(text,'private:\n  struct Impl;', '''  [[nodiscard]] compact_r4_preflight_sep22::Counters compactR4PreflightCounters() const;
  void addCompactNativeR4VerifyChain(metal::CommandGraph &graph,uint32_t layer,
      metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &scratch,
      metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections=10) const;
private:
  struct Impl;''')
    if relative=='runtime/flash/FlashInt8ExpertStore.mm':
        gu=native_suffix(text,'addGateUp','addDownScatter','  const auto p = allRowsParams(')
        down=native_suffix(text,'addDownScatter','addFixedSG2PrefillGateUp','  const uint32_t routes = rows * selections;')
        text=f'#include "{PRIVATE}/guard.hpp"\n#include <type_traits>\n'+text
        text=replace(text,'  const bool compactR4Verify = compact_native_r4_verify_sep22::requested();', '''  const bool compactPreflight = compact_r4_preflight_sep22::requested();
  mutable std::atomic<uint64_t> compactPreflightCalls{0},compactPreflightRows{0};
  const bool compactR4Verify = compact_native_r4_verify_sep22::requested();''')
        text=replace(text,'    numericalIdentity = hash(derivative.data(), derivative.size());', '''    if (compactPreflight)
      derivative += std::string("CPU_only_compact_R4_preflight_policy=")+compact_r4_preflight_sep22::marker()+"\\n";
    numericalIdentity = hash(derivative.data(), derivative.size());''')
        methods='''compact_r4_preflight_sep22::Counters FlashInt8ExpertStore::compactR4PreflightCounters() const {
  if(!impl_)fail("compact R4 preflight Store disposed");
  if(compact_r4_preflight_sep22::requested()!=impl_->compactPreflight)fail("compact R4 preflight flag changed");
  return {impl_->compactPreflight,impl_->compactPreflightCalls.load(std::memory_order_relaxed),impl_->compactPreflightRows.load(std::memory_order_relaxed)};
}
void FlashInt8ExpertStore::addCompactNativeR4VerifyChain(metal::CommandGraph &graph,uint32_t index,
    metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &scratch,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if(!compactNativeR4VerifyEnabled()||rows!=4||selections!=10)fail("compact verifier only canonical R4/S10");
  if(!compactR4PreflightCounters().enabled)fail("compact R4 complete preflight requires frozen flag1");
  const auto &sourceLayer=impl_->layer(index);
  if(impl_->metadata.layers[index].selectedIDs.size()!=512)fail("compact verifier requires Full512 inventory");
  // This type has lexical method scope. It cannot escape or be constructed by
  // a caller, and it is never cached. Copies bind exact allocation/view refs.
  struct CapturedViews final {
    const FlashMoEBlockedScratch s;
    const metal::MetalBuffer input,ids,diag;
    const std::remove_cvref_t<decltype(sourceLayer)> source;
    const uint32_t rows,selections;
    const FlashMoEBlockedTile tile;
  };
  // Capture before admission: no caller scratch/source is consulted by append.
  const CapturedViews captured{scratch,input,ids,diagnostics,sourceLayer,rows,selections,FlashMoEBlockedTile::M16N64};
  compact_r4_preflight_sep22::validateComplete(captured.s,captured.input,captured.ids,captured.source.ranks,captured.diag,captured.rows,captured.selections,
      [](const auto &s,auto d,uint32_t r,uint32_t n){allRowsScratch(s,d,r,FlashMoEBlockedTile::M16N64,n);},
      [](const auto &b,uint64_t n){requireBytes(b,n);},
      [](const auto &a,const auto &b){disjoint(a,b);},
      [&](const auto &b){impl_->immutableDisjoint(b);});
  // Noncopyable, nonmovable, single-consume token is constructed ONLY after
  // admission. Its const snapshot reference cannot outlive this method.
  struct ValidatedBundle final {
    const void *const storeEpoch;
    const void *const layerEpoch;
    metal::CommandGraph *const graphOwner;
    const uint32_t layerIndex;
    const CapturedViews &v;
    bool consumed=false;
    ValidatedBundle(const void *owner,const void *selected,metal::CommandGraph *g,uint32_t layer,const CapturedViews &views)
        :storeEpoch(owner),layerEpoch(selected),graphOwner(g),layerIndex(layer),v(views){}
    ValidatedBundle(const ValidatedBundle &)=delete;
    ValidatedBundle &operator=(const ValidatedBundle &)=delete;
    ValidatedBundle(ValidatedBundle &&)=delete;
    ValidatedBundle &operator=(ValidatedBundle &&)=delete;
  };
  static_assert(!std::is_copy_constructible_v<ValidatedBundle> && !std::is_move_constructible_v<ValidatedBundle>);
  ValidatedBundle bundle{impl_.get(),&sourceLayer,&graph,index,captured};
  // Only this private lexical appender can consume the admitted token. Epoch,
  // graph and immutable source-view identities are rechecked at consumption.
  const auto appendValidated=[&](ValidatedBundle &b) {
    if(b.consumed||b.storeEpoch!=impl_.get()||b.layerEpoch!=&sourceLayer||b.graphOwner!=&graph||b.layerIndex!=index||
        b.v.rows!=4||b.v.selections!=10||b.v.tile!=FlashMoEBlockedTile::M16N64||
        !b.v.source.base.sameView(sourceLayer.base)||!b.v.source.ranks.sameView(sourceLayer.ranks))
      fail("compact R4 admitted bundle owner/epoch/source changed");
    for(uint32_t plane=0;plane<3;++plane)
      if(!b.v.source.codes[plane].sameView(sourceLayer.codes[plane])||!b.v.source.scales[plane].sameView(sourceLayer.scales[plane]))
        fail("compact R4 admitted immutable source view changed");
    if(!compactNativeR4VerifyEnabled()||!compactR4PreflightCounters().enabled)
      fail("compact R4 admitted frozen policy changed");
    b.consumed=true;
    const auto &s=b.v.s;const auto &layer=b.v.source;const auto diagnostics=b.v.diag;
    const auto rows=b.v.rows,selections=b.v.selections;const auto tile=b.v.tile;
    graph.add("expert_r4_compact_native_sep22_plan",{b.v.ids,s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,
        s.buckets.jobOffsets,s.buckets.tileJobs,s.buckets.jobCount,diagnostics},FlashMoEBucketParams{4,10,2560,512,40,16,514,0},{1,1,1},{256,1,1});
    graph.add("flash_moe_direct_a_pack",{b.v.input,s.buckets.offsets,s.buckets.routeMap,s.buckets.packedInputs,diagnostics},
        FlashMoEBucketParams{4,10,2560,512,40,0,0,0},{103,1,1},{256,1,1});
    impl_->compactR4PlanCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4PlanRows.fetch_add(rows,std::memory_order_relaxed);
    { // Original public gate/up producer suffix, literal and unmodified.
GATE_SUFFIX
    }
    impl_->compactR4GateCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4GateRows.fetch_add(rows,std::memory_order_relaxed);
    { // Original public down producer suffix, literal and unmodified.
DOWN_SUFFIX
    }
    impl_->compactR4DownCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactR4DownRows.fetch_add(rows,std::memory_order_relaxed);
  };
  appendValidated(bundle);
  impl_->compactPreflightCalls.fetch_add(1,std::memory_order_relaxed);impl_->compactPreflightRows.fetch_add(rows,std::memory_order_relaxed);
}
'''.replace('GATE_SUFFIX',gu).replace('DOWN_SUFFIX',down)
        text=replace(text,'} // namespace splash::flash',methods+'} // namespace splash::flash')
    if relative=='runtime/flash/FlashForward.cpp':
        text=replace(text,'      compact_native_r4_verify_sep22::implementationMarker() +','''      compact_native_r4_verify_sep22::implementationMarker() +
      compact_r4_preflight_sep22::marker() +''')
        text=replace(text,'''    if (compactR4Verify) {
      impl_->int8ExpertStore->addCompactNativeR4VerifyPack''','''    if (compactR4Verify && compact_r4_preflight_sep22::requested()) {
      impl_->int8ExpertStore->addCompactNativeR4VerifyChain(graph,layer,mixed,ids,impl_->blockedScratch,diag,rows,kSelections);
    } else if (compactR4Verify) {
      impl_->int8ExpertStore->addCompactNativeR4VerifyPack''')
    if relative=='runtime/flash/FlashWorker.mm':
        text=replace(text,'      compact_native_r4_verify_sep22::validateDependencies(','''      compact_r4_preflight_sep22::validateDependencies(compact_native_r4_verify_sep22::requested());
      compact_native_r4_verify_sep22::validateDependencies(''')
        text=replace(text,'  const auto compactR4Graphs=persistedExperts ? persistedExperts->compactNativeR4VerifyCounters() : compact_native_r4_verify_sep22::Counters{};', '''  const auto compactR4Graphs=persistedExperts ? persistedExperts->compactNativeR4VerifyCounters() : compact_native_r4_verify_sep22::Counters{};
  const auto compactPreflight=persistedExperts ? persistedExperts->compactR4PreflightCounters() : compact_r4_preflight_sep22::Counters{};''')
        text=replace(text,'      << R"(,"gdn_verification_storage":{"lazy_enabled":)"', '''      << R"(,"compact_r4_preflight":{"schema":"CPU-only-unique13view-StoreLayerGraph-stack-bundle-v1","scope":"singleton physicalR4 verification; graph construction not completion","requested":)"
      <<(compact_r4_preflight_sep22::requested()?"true":"false")<<R"(,"enabled":)"<<(compactPreflight.enabled?"true":"false")
      <<R"(,"source_identity_sha256":)"<<json::quote(compact_r4_preflight_sep22::kPreflightSourceIdentitySha256)
      <<R"(,"guard_preflight_graph_calls":)"<<compactPreflight.calls<<R"(,"guard_preflight_graph_rows":)"<<compactPreflight.rows
      <<R"(,"unique_logical_views":13,"immutable_spans":96,"GPU_allocation_bytes_added":0,"shader_math_changed":false,"original_public_native_API_guards_unchanged":true,"whole_state_qualified":false})"
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
    return text
