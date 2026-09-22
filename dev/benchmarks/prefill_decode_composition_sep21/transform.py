"""Private phase transforms; no model payload or backend operations."""
from pathlib import Path
PRIVATE='dev/benchmarks/prefill_decode_composition_sep21'
def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Phase frozen anchor differs: '+old[:120])
    return text.replace(old,new,1)
def function(text,start):
    begin=text.index(start);brace=text.index('{',begin);depth=1;end=brace+1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    return text[begin:end]
def transform(relative,text,hybrid):
    if relative in ('runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm',
                    'runtime/flash/FlashFloatDenseCache.cpp','runtime/flash/FlashWeights.mm'):
        text=f'#include "{PRIVATE}/phase.hpp"\n'+text
    if relative=='runtime/flash/FlashWeights.hpp':
        text=once(text,'  [[nodiscard]] FlashOriginalTextResidencySelection checkedOriginalTextResidency() const;',
            '  [[nodiscard]] FlashOriginalTextResidencySelection checkedOriginalTextResidency() const;\n  [[nodiscard]] FlashOriginalTextResidencySelection checkedHybridTargetExpertResidency() const;')
        text=once(text,'  struct Impl;\n  std::unique_ptr<Impl> impl_;',
            '  [[nodiscard]] static FlashWeights loadPhaseOriginalQ4(metal::MetalBackend &backend, const std::filesystem::path &root, bool verifyPayloadHashes);\n  struct Impl;\n  std::unique_ptr<Impl> impl_;')
    elif relative=='runtime/flash/FlashWeights.mm':
        marker='FlashWeights FlashWeights::load('
        old=function((hybrid/relative).read_text(),marker)
        old=old.replace('FlashWeights::load(','FlashWeights::loadPhaseOriginalQ4(',1)
        text=once(text,'                               bool verifyPayloadHashes) {\n  @autoreleasepool {',
            '                               bool verifyPayloadHashes) {\n  if(phase_q4_sep21::requested())return loadPhaseOriginalQ4(backend,directory,verifyPayloadHashes);\n  @autoreleasepool {')
        text=once(text,marker,old+'\n\n'+marker)
        getter=function((hybrid/relative).read_text(),'FlashOriginalTextResidencySelection FlashWeights::checkedHybridTargetExpertResidency() const')
        text=once(text,'FlashOriginalTextResidencySelection FlashWeights::checkedOriginalTextResidency() const {',getter+'\n\nFlashOriginalTextResidencySelection FlashWeights::checkedOriginalTextResidency() const {')
    elif relative=='runtime/flash/FlashFloatDenseCache.cpp':
        old='''  return FlashDenseCache::defaultPrefixes(weights, includeVocabularyHead);
}'''
        new='''  if(!phase_q4_sep21::requested())return FlashDenseCache::defaultPrefixes(weights,includeVocabularyHead);
  if(includeVocabularyHead)throw std::invalid_argument("phase profile requires original code head");
  const auto names=normalize(FlashDenseCache::defaultPrefixes(weights,false));
  std::vector<std::string> selected;uint64_t bytes=0;
  for(const auto &name:names){
    const auto role=policyRole(name);
    if(role.starts_with("ple.") && !name.starts_with("language_model.model.layers.1.ple."))continue;
    const auto &p=weights.projection(name);bool keep=false;
    for(const uint32_t rows:{4u,8u,9u,16u})keep=keep || bool(flashFloatDenseSmallRowsPolicy(name,rows,p.outputSize,p.inputSize,p.bits,p.groupSize));
    if(!keep)continue;
    if(p.experts!=1)throw std::invalid_argument("phase F32 union selected expert bank");
    bytes=plus(bytes,rounded(product(product(p.outputSize,p.inputSize),4)));selected.push_back(name);
  }
  if(selected.size()!=296 || bytes!=12097945600ULL)
    throw std::invalid_argument("phase F32 union must contain296maps/12097945600bytes");
  return selected;
}'''
        text=once(text,old,new)
    elif relative=='runtime/flash/FlashForward.hpp':
        marker='  // output budget. Correction/bonus output is not a consumed verify input.'
        text=once(text,marker,'  [[nodiscard]] FlashForwardResult forwardDecode(FlashRequestState &state, std::span<const uint32_t> token, bool captureHidden = false);\n'+marker)
        text=once(text,'bool returnAllLogits, bool captureHidden, bool verification);',
            'bool returnAllLogits, bool captureHidden, bool verification, bool decodePhase = false);')
    elif relative=='runtime/flash/FlashForward.cpp':
        text=once(text,'#include <optional>','#include <optional>\n#include <set>')
        text=once(text,'  std::unique_ptr<FlashFloatDenseCache> floatDenseCache;',
            '  std::unique_ptr<FlashFloatDenseCache> floatDenseCache;\n  std::set<std::string> prefillF32SelectorPrefixes;')
        text=once(text,'    descriptor.validate();',
            '''    descriptor.validate();
    if(phase_q4_sep21::requested()) {
      const auto names=FlashDenseCache::defaultPrefixes(weights,false);
      prefillF32SelectorPrefixes.insert(names.begin(),names.end());
      if(names.size()!=508 || prefillF32SelectorPrefixes.size()!=508)
        throw std::invalid_argument("phase Prefill requires original508 selector membership");
    }''')
        text=once(text,'    if (floatDenseCache && rows >= 2 && rows <= 16 && floatDenseCache->contains(prefix)) {',
            '''    const bool originalPrefillF32Member=phase_q4_sep21::requested() && singletonMain && !verification &&
        prefillF32SelectorPrefixes.contains(prefix);
    if (floatDenseCache && rows >= 2 && rows <= 16 &&
        (floatDenseCache->contains(prefix) || originalPrefillF32Member)) {''')
        oldHC=function(text,'  void hc(')
        newHC=once(oldHC,'bool injection, bool normalizedReady = false)',
            'bool injection, bool normalizedReady = false, bool prefillSelector = false)')
        newHC=newHC.replace('view(Scratch::Diagnostics, 4), rows);',
            'view(Scratch::Diagnostics, 4), rows, false, prefillSelector);')
        if newHC.count('rows, false, prefillSelector);')!=3:
            raise ValueError('Expected all three generic HC projections to propagate Prefill selector')
        text=once(text,oldHC,newHC)
        for call in ('    impl_->hc(graph, prefix + ".attn_hyper_connection", rows, true, normalizedReady);',
                     '    impl_->hc(graph, prefix + ".mlp_hyper_connection", rows, true, normalizedReady);',
                     '  impl_->hc(graph, "language_model.model.hyper_connection_mixer", rows, false, normalizedReady);'):
            text=once(text,call,call.replace('normalizedReady);','normalizedReady, phase_q4_sep21::requested() && !verification && !decodePhase);'))
        marker='FlashForwardResult FlashForward::verify('
        entry='''FlashForwardResult FlashForward::forwardDecode(FlashRequestState &request,
    std::span<const uint32_t> token,bool captureHidden) {
  if(token.size()!=1)throw std::invalid_argument("explicit Decode requires one token");
  return forwardImpl(request,token,false,captureHidden,false,true);
}

'''
        text=once(text,marker,entry+marker)
        text=once(text,'                                             bool verification) {',
            '                                             bool verification, bool decodePhase) {')
        text=once(text,'    if (allRowsInt8Target) {','    if (allRowsInt8Target || phase_q4_sep21::requested()) {')
        old='''    const bool gatheredMPP = impl_->allRowsInt8Target && impl_->int8ExpertStore &&
        impl_->int8ExpertStore->gatheredMPPEnabled() && rows <= impl_->int8ExpertStore->gatheredMPPMaximumRows();
    const bool blocked = impl_->blockMoE && (impl_->allRowsInt8Target || rows >= 256);'''
        new='''    const bool phasePrefillI8=phase_q4_sep21::requested() && !verification && !decodePhase;
    const bool useI8=impl_->allRowsInt8Target || phasePrefillI8;
    const auto phase=verification?phase_q4_sep21::Phase::Verify:decodePhase?phase_q4_sep21::Phase::Decode:phase_q4_sep21::Phase::Prefill;
    const bool gatheredMPP=useI8 && impl_->int8ExpertStore && impl_->int8ExpertStore->gatheredMPPEnabled() &&
        rows<=impl_->int8ExpertStore->gatheredMPPMaximumRows();
    const bool blocked=impl_->blockMoE && (useI8 || (!verification && !decodePhase && rows>=256));'''
        text=once(text,old,new)
        text=once(text,'const auto tile = impl_->allRowsInt8Target && rows < 256','const auto tile = useI8 && rows < 256')
        text=once(text,'    if (!batchSharedExpertFused(graph, mlp + ".shared_expert", mixed,',
            '    phase_q4_sep21::record(phase,useI8,rows);\n    if (!batchSharedExpertFused(graph, mlp + ".shared_expert", mixed,')
        text=once(text,'  if (impl_->denseCache) append(impl_->denseCache->persistedWeightBuffers());',
            '''  if (impl_->denseCache) append(dense_w8a8_residency_sep21::bf16PersistentOperands(*impl_->denseCache,
      bool(impl_->denseW8Cache),impl_->denseW8Requested,dense_w8a8_residency_sep21::requested()));''')
        text=once(text,'  if (impl_->denseW8Cache) append(impl_->denseW8Cache->immutableWeightBuffers());',
            '''  if (dense_w8a8_residency_sep21::includeDerived(bool(impl_->denseW8Cache),impl_->denseW8Requested,dense_w8a8_residency_sep21::requested()))
    append(impl_->denseW8Cache->immutableWeightBuffers());''')
        text=once(text,'  if (impl_->floatDenseCache) append(impl_->floatDenseCache->persistedWeightBuffers());',
            '  if (impl_->floatDenseCache) append(phase_q4_sep21::persistentF32(impl_->weights,*impl_->floatDenseCache));')
    elif relative=='runtime/flash/FlashWorker.mm':
        text=once(text,'      std::signal(SIGPIPE, SIG_IGN);',
            '''      if(!phase_q4_sep21::requested())throw std::invalid_argument("private phase worker requires PREFILL_I8_DECODE_Q4=1");
      (void)dense_w8a8_residency_sep21::requested();
      if(!environmentSwitch("SPLASH_FLASH_DENSE_W8A8_RESIDENCY_PRUNE_SEP21") ||
          !environmentSwitch("SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT") ||
          !environmentSwitch("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT"))
        throw std::invalid_argument("private phase worker requires saved/expert residency and BF16 lease pruning");
      for(const char *name:{"SPLASH_FLASH_ALLROWS_FULL512_TARGET","SPLASH_FLASH_BATCH","SPLASH_FLASH_BATCH_PREFILL", "SPLASH_FLASH_BATCH_MTP",
          "SPLASH_FLASH_BATCH_MTP_PREFILL","SPLASH_FLASH_GDN_BATCH_ILP","SPLASH_FLASH_GPU_PREFILL_COPY","SPLASH_FLASH_MTP_ADAPTIVE",
          "SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE","SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT","SPLASH_FLASH_DENSE_SMALL_ROWS"})
        if(environmentSwitch(name))throw std::invalid_argument(std::string("private phase worker requires ")+name+"=0");
      for(const char *name:{"SPLASH_FLASH_ALLROWS_GATHERED_MPP","SPLASH_FLASH_DENSE_CACHE","SPLASH_FLASH_FLOAT_DENSE_CACHE",
          "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE","SPLASH_FLASH_QMV_F32","SPLASH_FLASH_EXPERT_QMV","SPLASH_FLASH_INT8_HEAD",
          "SPLASH_FLASH_HC_UP_F32_MPP","SPLASH_FLASH_BLOCKED_MOE","SPLASH_FLASH_PLE_SSD_STREAMING",
          "SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21","SPLASH_FLASH_GDN_PREFILL_FMA_SEP21","SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21",
          "SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21","SPLASH_FLASH_GDN_AB_MERGE_SEP21","SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21"})
        if(!environmentSwitch(name))throw std::invalid_argument(std::string("private phase worker requires ")+name+"=1");
      if(std::getenv("SPLASH_FLASH_HOT_EXPERT_PLAN"))throw std::invalid_argument("phase worker forbids competing hot expert policy");
      if(gathered_mpp::requestedMaximumRows()!=4)throw std::invalid_argument("phase Prefill gatheredMPP requires physical cap4");
      if(fixed_sg2_prefill_sep21::selection()!=7)throw std::invalid_argument("phase Prefill requires fixedSG2variant7");
      std::signal(SIGPIPE, SIG_IGN);''')
        text=once(text,'      if (!environmentSwitch("SPLASH_FLASH_ALLROWS_FULL512_TARGET") ||',
            '      if (!phase_q4_sep21::requested() ||')
        text=once(text,'      if (capacity > descriptor.maximumContextTokens)',
            '''      if(prefillRows!=2048 || capacity>16384 || singletonMTP.maximumDepth!=3)
        throw std::invalid_argument("phase worker requires arena2048/depth3/context at most16384");
      if (capacity > descriptor.maximumContextTokens)''')
        text=once(text,'      const auto result = forward_.forward(*request.state, token);',
            '      const auto result = forward_.forwardDecode(*request.state, token);')
        text=once(text,'                              : forward_.forward(*request.state, inputs, false, true);',
            '                              : forward_.forwardDecode(*request.state, inputs, true);')
        text=once(text,'"scope":"PRIVATE all target rows including decode/verifier; signed INT8 late row scaling; original trained MTP retained"',
            '"scope":"PRIVATE every singleton Prefill row uses Full512-I8; explicit Decode and singleton Verify use originalQ4; original trained MTP retained"')
        text=once(text,'      << R"(,"target_all_rows_full512":true,"original_target_gpu_omitted":true)"',
            '''      << R"(,"target_all_rows_full512":false,"original_target_gpu_omitted":false,"target_hybrid_phase":true,"true_batch_graph_supported":false,"universal_original_q4_parity_claim":false)"
      << R"(,"target_phase_policy_schema":)" << json::quote(phase_q4_sep21::schema)
      << R"(,"target_phase_policy":)" << json::quote(phase_q4_sep21::policy)
      << R"(,"phase_f32_backing_count":296,"phase_f32_backing_bytes":12097945600)"''')
        text=once(text,'      << R"(,"phase_f32_backing_count":296,"phase_f32_backing_bytes":12097945600)"',
            '''      << R"(,"phase_f32_backing_count":296,"phase_f32_backing_bytes":12097945600,"phase_prefill_f32_selector_membership_count":508)"
      << R"(,"phase_prefill_f32_selector_policy":"parent508 membership retained; missing null-policy coefficients select original RAW;296 qualified backing maps unchanged")"''')
        old='''json::quote(prefill_qsa_twopass_sep21::numericalIdentity(dense_w8a8_sep21::numericalIdentity(
          gdn_prefill_fma_sep21::numericalIdentity(persistedExperts->numericalIdentitySha256(),
              gdn_prefill_fma_sep21::requested()),forward_.kernelRoutes()),prefill_qsa_twopass_sep21::requested()))'''
        text=once(text,old,'json::quote(phase_q4_sep21::identity('+old[len('json::quote('):-1]+'))')
        stage='''  bool hybridExpertRequested = false, hybridExpertAdded = false;
  FlashOriginalTextResidencySelection hybridExpertSelection;
  engine::MemoryGovernorSnapshot hybridExpertHost{};
  std::string hybridExpertFailureReason;'''
        text=once(text,'  std::string originalFailureReason;', '  std::string originalFailureReason;\n'+stage)
        text=once(text,'      savedResidency.originalRequested = originalTextResidencyRequested;',
            '      savedResidency.originalRequested = originalTextResidencyRequested;\n      savedResidency.hybridExpertRequested=environmentSwitch("SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT");')
        text=once(text,'      << R"(,"scope":)" << json::quote(savedResidency_.originalAdded',
            '      << R"(,"scope":)" << json::quote(savedResidency_.hybridExpertAdded?"saved I8/dense operands plus25 existing qualified originalQ4 expert owners; all296F32backing retained/118persistent":savedResidency_.originalAdded')
        addition='''        if(savedResidency.hybridExpertRequested) {
          savedResidency.hybridExpertSelection=weights.checkedHybridTargetExpertResidency();
          governor.setPressure(pressure.value());savedResidency.hybridExpertHost=governor.snapshot();
          const auto &host=savedResidency.hybridExpertHost;
          if(flashOriginalResidencyHostAllowed(host.hostMeasurementValid,host.growthAllowed,
              host.pressure==engine::MemoryPressure::Normal && host.systemPressure==engine::MemoryPressure::Normal,
              host.hostHeadroomBytes,savedResidency.hybridExpertSelection.mappedBytes)) {
            operands.insert(operands.end(),savedResidency.hybridExpertSelection.buffers.begin(),savedResidency.hybridExpertSelection.buffers.end());
            savedResidency.hybridExpertAdded=true;
          }else savedResidency.hybridExpertFailureReason="host reserve protected; existing expert owners require unchanged2GiBheadroom";
        }
'''
        text=once(text,'        if (originalTextResidencyRequested) {',addition+'        if (originalTextResidencyRequested) {')
        text=once(text,'        if (savedResidency.originalAdded && !savedResidencyLease)',
            '''        if(savedResidency.hybridExpertAdded && savedResidencyLease &&
            (savedResidencyLease.bufferCount()!=dense_w8a8_residency_sep21::expectedHybridOwners(
                dense_w8a8_sep21::requiresCache(prefillRows),dense_w8a8_sep21::requested(),true) ||
             savedResidencyLease.byteCount()!=dense_w8a8_residency_sep21::expectedHybridBytes(
                dense_w8a8_sep21::requiresCache(prefillRows),dense_w8a8_sep21::requested(),true)))
          throw std::logic_error("phase composite registered owner census differs");
        if(savedResidency.hybridExpertAdded && !savedResidencyLease)savedResidency.hybridExpertFailureReason=savedResidency.failureReason;
        if (savedResidency.originalAdded && !savedResidencyLease)''')
        counters='''  out << R"(,"phase_f32_persistent_residency":{"retained_owner_count":118,"retained_owner_bytes":3247964160,"transient_only_owner_count":178,"transient_only_owner_bytes":8849981440,"all_backing_retained":true,"all_backing_charged":true})";
  out << R"(,"target_phase_graph_counters":{"scope":"graph construction, not GPU completion","enabled":true)";
  const std::array<const char *,3> phaseNames{"prefill","decode","verify"};
  for(uint32_t i=0;i<3;++i){const auto &c=phase_q4_sep21::counters[i];
    out << ',' << json::quote(phaseNames[i]) << R"(:{"i8_gate_up_graph_calls":)" << c.i8Calls.load()
        << R"(,"i8_gate_up_graph_rows":)" << c.i8Rows.load() << R"(,"i8_down_graph_calls":)" << c.i8Calls.load()
        << R"(,"i8_down_graph_rows":)" << c.i8Rows.load() << R"(,"q4_gate_up_graph_calls":)" << c.q4Calls.load()
        << R"(,"q4_gate_up_graph_rows":)" << c.q4Rows.load() << R"(,"q4_down_graph_calls":)" << c.q4Calls.load()
        << R"(,"q4_down_graph_rows":)" << c.q4Rows.load() << '}';}
  out << '}';
  out << R"(,"hybrid_q4_expert_residency":{"requested":)" << (savedResidency_.hybridExpertRequested?"true":"false")
      << R"(,"added_to_composite":)" << (savedResidency_.hybridExpertAdded?"true":"false")
      << R"(,"active":)" << (savedResidency_.hybridExpertAdded && savedResidencyLease_ && healthy?"true":"false")
      << R"(,"selected_owner_count":)" << savedResidency_.hybridExpertSelection.buffers.size()
      << R"(,"selected_owner_bytes":)" << savedResidency_.hybridExpertSelection.mappedBytes
      << R"(,"registered_composite_owner_count":)" << savedResidencyLease_.bufferCount()
      << R"(,"registered_composite_owner_bytes":)" << savedResidencyLease_.byteCount()
      << R"(,"host_measurement_valid":)" << (savedResidency_.hybridExpertHost.hostMeasurementValid?"true":"false")
      << R"(,"host_available_bytes":)" << savedResidency_.hybridExpertHost.hostAvailableBytes
      << R"(,"host_reserve_bytes":)" << savedResidency_.hybridExpertHost.hostReserveBytes
      << R"(,"host_headroom_bytes":)" << savedResidency_.hybridExpertHost.hostHeadroomBytes
      << R"(,"growth_allowed":)" << (savedResidency_.hybridExpertHost.growthAllowed?"true":"false")
      << R"(,"required_host_headroom_bytes":71510786048,"new_weight_backing_bytes":0)"
      << R"(,"failure_reason":)" << json::quote(savedResidency_.hybridExpertFailureReason)
      << R"(,"physical_pinning_verified":false,"backing_already_charged":true})";
'''
        text=once(text,'  out << R"(},"batch_decode":{"enabled":)"',
            "  out << '}';\n"+counters+'  out << R"(,"batch_decode":{"enabled":)"')
    return text
