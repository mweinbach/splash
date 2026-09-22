"""Strict private overlay. This never changes a numerical identity expression."""
INCLUDE = '#include "dev/benchmarks/prefill_hc_inject_norm_sep21/worker_bridge.hpp"\n'
def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one transform anchor: {old[:100]!r}")
    return text.replace(old, new, 1)
def transform(relative, text):
    if relative == "runtime/flash/FlashForward.cpp":
        text = INCLUDE + text
        text = once(text, '      std::string(dense_w8a8_sep21::selectionMarker(impl_->denseW8Requested)) +',
            '      std::string(prefill_hc_inject_norm_sep21::selectionMarker(prefill_hc_inject_norm_sep21::requested())) +\n'
            '      std::string(dense_w8a8_sep21::selectionMarker(impl_->denseW8Requested)) +')
        text = once(text, '  const uint32_t begin = static_cast<uint32_t>(state.length);',
            '  const uint32_t begin = static_cast<uint32_t>(state.length);\n'
            '  const bool privatePrefillHC = prefill_hc_inject_norm_sep21::mainEligible(\n'
            '      rows, verification, true, prefill_hc_inject_norm_sep21::requested());\n'
            '  if (privatePrefillHC) prefill_hc_inject_norm_sep21::recordForward();')
        text = once(text, '    const std::string mlpNorm = prefix + ".mlp_hyper_connection.hc_norm.weight";\n'
            '    if (impl_->fuseHC && rows <= 32) {',
            '    const std::string mlpNorm = prefix + ".mlp_hyper_connection.hc_norm.weight";\n'
            '    if (privatePrefillHC) {\n'
            '      prefill_hc_inject_norm_sep21::addPrivatePrefillHCInjectNormSep21(graph, hyper, branch,\n'
            '          bf(Scratch::HCInjectionWeights, 4), impl_->weights.tensor(mlpNorm), hyper,\n'
            '          bf(Scratch::HCNormalized, kHyper), diag, hcGeometry, impl_->weights.normConvention(mlpNorm));\n'
            '      prefill_hc_inject_norm_sep21::recordAttentionToMlp();\n'
            '      normalizedReady = true;\n'
            '    } else if (impl_->fuseHC && rows <= 32) {')
        text = once(text, '    normalizedReady = impl_->fuseHC && rows <= 32 && !nextHasPLE;',
            '    const bool nextIsTerminalMixer = layer + 1 == impl_->descriptor.layers;\n'
            '    const bool privatePrefillNextNorm =\n'
            '        prefill_hc_inject_norm_sep21::nextNormEligible(privatePrefillHC, nextHasPLE, nextIsTerminalMixer);\n'
            '    if (privatePrefillHC && nextHasPLE) prefill_hc_inject_norm_sep21::recordPLEExcluded();\n'
            '    if (privatePrefillHC && nextIsTerminalMixer) prefill_hc_inject_norm_sep21::recordTerminalExcluded();\n'
            '    normalizedReady = (impl_->fuseHC && rows <= 32 && !nextHasPLE) || privatePrefillNextNorm;')
        text = once(text, '      addHCFusedInjectNorm(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4),\n'
            '          impl_->weights.tensor(nextNorm), hyper, bf(Scratch::HCNormalized, kHyper), diag,\n'
            '          hcGeometry, impl_->weights.normConvention(nextNorm));',
            '      if (privatePrefillNextNorm) {\n'
            '        prefill_hc_inject_norm_sep21::addPrivatePrefillHCInjectNormSep21(graph, hyper, branch,\n'
            '            bf(Scratch::HCInjectionWeights, 4), impl_->weights.tensor(nextNorm), hyper,\n'
            '            bf(Scratch::HCNormalized, kHyper), diag, hcGeometry, impl_->weights.normConvention(nextNorm));\n'
            '        prefill_hc_inject_norm_sep21::recordMlpToNext();\n'
            '      } else {\n'
            '        addHCFusedInjectNorm(graph, hyper, branch, bf(Scratch::HCInjectionWeights, 4),\n'
            '            impl_->weights.tensor(nextNorm), hyper, bf(Scratch::HCNormalized, kHyper), diag,\n'
            '            hcGeometry, impl_->weights.normConvention(nextNorm));\n'
            '      }')
    elif relative == "runtime/flash/FlashWorker.mm":
        text = INCLUDE + text
        text = once(text, '      (void)dense_w8a8_sep21::requested(); // Freeze strict selector before paths, metadata or backend.',
            '      (void)prefill_hc_inject_norm_sep21::requested(); // Freeze exact prefill selector before paths/metadata/backend.\n'
            '      (void)dense_w8a8_sep21::requested(); // Freeze strict selector before paths, metadata or backend.')
        text = once(text, '      << R"(,"dense_w8a8_prefill_enabled":)"',
            '      << R"(,"prefill_hc_inject_norm_enabled":)" << (prefill_hc_inject_norm_sep21::requested() ? "true" : "false")\n'
            '      << R"(,"prefill_hc_inject_norm_policy":)" << json::quote(std::string(prefill_hc_inject_norm_sep21::kPolicy))\n'
            '      << R"(,"prefill_hc_inject_norm_numerical_change":false,"prefill_hc_inject_norm_added_workspace_bytes":0)"\n'
            '      << R"(,"dense_w8a8_prefill_enabled":)"')
        text = once(text, '      << R"(,"memory_pressure":)"',
            '      << R"(,"prefill_hc_inject_norm":{"enabled":)" << (prefill_hc_inject_norm_sep21::requested() ? "true" : "false")\n'
            '      << R"(,"encoded_forwards":)" << prefill_hc_inject_norm_sep21::encodedCounters().forwards.load(std::memory_order_relaxed)\n'
            '      << R"(,"encoded_attention_to_mlp":)" << prefill_hc_inject_norm_sep21::encodedCounters().attentionToMlp.load(std::memory_order_relaxed)\n'
            '      << R"(,"encoded_mlp_to_next":)" << prefill_hc_inject_norm_sep21::encodedCounters().mlpToNext.load(std::memory_order_relaxed)\n'
            '      << R"(,"encoded_ple_exclusions":)" << prefill_hc_inject_norm_sep21::encodedCounters().pleExcluded.load(std::memory_order_relaxed)\n'
            '      << R"(,"encoded_terminal_exclusions":)" << prefill_hc_inject_norm_sep21::encodedCounters().terminalExcluded.load(std::memory_order_relaxed)\n'
            '      << R"(,"counter_scope":"encoded graphs, not completed commands","counters_excluded_from_identity":true})"\n'
            '      << R"(,"memory_pressure":)"')
    elif relative == "dev/benchmarks/prefill4k_attribution.mm":
        text = INCLUDE + text
        text = once(text, '      (void)dense_w8a8_sep21::requested(); // Freeze strict dense selector before backend/model.',
            '      (void)prefill_hc_inject_norm_sep21::requested(); // Freeze exact prefill HC before backend/model.\n'
            '      (void)dense_w8a8_sep21::requested(); // Freeze strict dense selector before backend/model.')
    return text
