"""Strict private singleton-only transformation; qualified component sources are unchanged."""
INCLUDE='#include "dev/benchmarks/prefill_qsa_twopass_sep21/worker_bridge.hpp"\n'
def once(text,old,new):
    if text.count(old)!=1:raise ValueError(f'Expected exactly one source anchor: {old[:100]!r}')
    return text.replace(old,new,1)
def transform(relative,text):
    if relative=='runtime/flash/FlashForward.cpp':
        text=INCLUDE+text
        text=once(text,'  std::optional<prefill4k::BulkExactWorkspace> bulkQSAWorkspace;',
            '  std::optional<prefill4k::BulkExactWorkspace> bulkQSAWorkspace;\n  std::optional<prefill_qsa_twopass_sep21::Workspace> twoPassQSAWorkspace;')
        text=once(text,'    uint32_t gdnSlot = 0;',
            '    if (prefill_qsa_twopass_sep21::plannedExtraBytes(maximumRows,prefill_qsa_twopass_sep21::requested())) {\n'
            '      if (!bulkQSAWorkspace) throw std::logic_error("private QSA needs admitted original bulk prefix");\n'
            '      twoPassQSAWorkspace.emplace(backend,bulkQSAWorkspace->prepared);\n'
            '    }\n    uint32_t gdnSlot = 0;')
        text=once(text,'  if (dense_w8a8_sep21::requiresCache(maximumRows))\n    total += dense_w8a8_sep21::Workspace::plannedBytes();',
            '  total += prefill_qsa_twopass_sep21::plannedExtraBytes(maximumRows,prefill_qsa_twopass_sep21::requested());\n'
            '  if (dense_w8a8_sep21::requiresCache(maximumRows))\n    total += dense_w8a8_sep21::Workspace::plannedBytes();')
        text=once(text,'  return std::string(flashAffineSemantics()) +',
            '  return std::string(flashAffineSemantics()) +\n      std::string(prefill_qsa_twopass_sep21::selectionMarker(prefill_qsa_twopass_sep21::requested())) +')
        text=once(text,'  const uint32_t begin = static_cast<uint32_t>(state.length);',
            '  const uint32_t begin = static_cast<uint32_t>(state.length);\n'
            '  const bool privateTwoPassQSA = prefill_qsa_twopass_sep21::mainEligible(\n'
            '      begin,rows,verification,true,prefill_qsa_twopass_sep21::requested());\n'
            '  if (privateTwoPassQSA) prefill_qsa_twopass_sep21::recordForward();')
        text=once(text,'        prefill4k::addBulkExactQSA(impl_->backend, graph, bulkInputs, state.qsa[layer],\n'
            '            impl_->qsaWorkspace, impl_->qsaFastWorkspace, *impl_->bulkQSAWorkspace, begin, rows,\n'
            '            impl_->bulkQSAPrefillSG8);',
            '        if (privateTwoPassQSA) {\n'
            '          if (!impl_->twoPassQSAWorkspace) throw std::logic_error("private QSA arena missing after admission");\n'
            '          prefill4k::addTwoPassQSA(impl_->backend,graph,bulkInputs,state.qsa[layer],\n'
            '              impl_->qsaWorkspace,impl_->qsaFastWorkspace,impl_->twoPassQSAWorkspace->qualified,\n'
            '              begin,rows,verification,true); // Only Root-qualified packed-V route.\n'
            '          prefill_qsa_twopass_sep21::recordLayer();\n'
            '        } else {\n'
            '          prefill4k::addBulkExactQSA(impl_->backend, graph, bulkInputs, state.qsa[layer],\n'
            '              impl_->qsaWorkspace, impl_->qsaFastWorkspace, *impl_->bulkQSAWorkspace, begin, rows,\n'
            '              impl_->bulkQSAPrefillSG8);\n'
            '        }')
        text=once(text,'  const auto &blocked = impl_->blockedScratch;',
            '  if (impl_->twoPassQSAWorkspace) {\n'
            '    const auto &two = *impl_->twoPassQSAWorkspace;\n'
            '    for (const auto &buffer : {two.arena,two.qualified.prepared.queries,\n'
            '        two.qualified.prepared.indexQueries,two.qualified.prepared.selectedBlocks,\n'
            '        two.qualified.packedQueries,two.qualified.scoresAndProbabilities,two.qualified.rawAttention}) reject(buffer);\n'
            '  }\n  const auto &blocked = impl_->blockedScratch;')
    elif relative=='runtime/flash/FlashWorker.mm':
        text=INCLUDE+text
        text=once(text,'      (void)prefill_hc_inject_norm_sep21::requested(); // Freeze exact prefill selector before paths/metadata/backend.',
            '      (void)prefill_qsa_twopass_sep21::requested(); // Freeze strict numerical selector before paths/metadata/backend.\n'
            '      (void)prefill_hc_inject_norm_sep21::requested(); // Freeze exact prefill selector before paths/metadata/backend.')
        old='json::quote(dense_w8a8_sep21::numericalIdentity(\n          gdn_prefill_fma_sep21::numericalIdentity(persistedExperts->numericalIdentitySha256(),\n              gdn_prefill_fma_sep21::requested()),forward_.kernelRoutes()))'
        new='json::quote(prefill_qsa_twopass_sep21::numericalIdentity(dense_w8a8_sep21::numericalIdentity(\n          gdn_prefill_fma_sep21::numericalIdentity(persistedExperts->numericalIdentitySha256(),\n              gdn_prefill_fma_sep21::requested()),forward_.kernelRoutes()),prefill_qsa_twopass_sep21::requested()))'
        text=once(text,old,new)
        text=once(text,'      << R"(,"prefill_hc_inject_norm_enabled":)"',
            '      << R"(,"prefill_qsa_twopass_enabled":)" << (prefill_qsa_twopass_sep21::requested() ? "true" : "false")\n'
            '      << R"(,"prefill_qsa_twopass_policy":)" << json::quote(std::string(prefill_qsa_twopass_sep21::kPolicy))\n'
            '      << R"(,"prefill_qsa_twopass_shader_sha256":)" << json::quote(prefill_qsa_twopass_sep21::kShaderSHA)\n'
            '      << R"(,"prefill_qsa_twopass_host_sha256":)" << json::quote(prefill_qsa_twopass_sep21::kHostSHA)\n'
            '      << R"(,"prefill_qsa_twopass_added_workspace_bytes":)" << prefill_qsa_twopass_sep21::plannedExtraBytes(prefillRows_,prefill_qsa_twopass_sep21::requested())\n'
            '      << R"(,"prefill_qsa_twopass_numerical_change":)" << (prefill_qsa_twopass_sep21::requested() ? "true" : "false")\n'
            '      << R"(,"prefill_qsa_twopass_scope":"singleton main fresh2048 nonverification; packed-V only; decoder/verify/batch/MTP unchanged")"\n'
            '      << R"(,"prefill_hc_inject_norm_enabled":)"')
        text=once(text,'      << R"(,"memory_pressure":)"',
            '      << R"(,"prefill_qsa_twopass":{"enabled":)" << (prefill_qsa_twopass_sep21::requested() ? "true" : "false")\n'
            '      << R"(,"encoded_forwards":)" << prefill_qsa_twopass_sep21::encodedCounters().forwards.load(std::memory_order_relaxed)\n'
            '      << R"(,"encoded_attention_layers":)" << prefill_qsa_twopass_sep21::encodedCounters().attentionLayers.load(std::memory_order_relaxed)\n'
            '      << R"(,"constructed_arenas":)" << prefill_qsa_twopass_sep21::encodedCounters().constructedArenas.load(std::memory_order_relaxed)\n'
            '      << R"(,"constructed_arena_bytes":)" << prefill_qsa_twopass_sep21::encodedCounters().constructedArenaBytes.load(std::memory_order_relaxed)\n'
            '      << R"(,"counter_scope":"constructed owners and encoded graphs, not completed commands","counters_excluded_from_identity":true})"\n'
            '      << R"(,"memory_pressure":)"')
    elif relative=='dev/benchmarks/prefill4k_attribution.mm':
        text=INCLUDE+text
        text=once(text,'      (void)prefill_hc_inject_norm_sep21::requested(); // Freeze exact prefill HC before backend/model.',
            '      (void)prefill_qsa_twopass_sep21::requested(); // Freeze strict numerical QSA before backend/model.\n'
            '      (void)prefill_hc_inject_norm_sep21::requested(); // Freeze exact prefill HC before backend/model.')
    return text
