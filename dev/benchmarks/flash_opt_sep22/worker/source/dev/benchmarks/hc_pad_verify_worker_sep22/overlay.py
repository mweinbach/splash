#!/usr/bin/env python3
"""Exact VerifyR4 HC scheduling overlay; apply only to a new frozen tree."""
from pathlib import Path

PRIVATE = 'dev/benchmarks/hc_pad_verify_worker_sep22'
CHANGED = {'runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm'}


def once(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError('HC overlay source anchor changed: ' + before[:100])
    return text.replace(before, after)


def transform(relative, text):
    if relative not in CHANGED:
        return text
    text = '#include "' + PRIVATE + '/worker_bridge.hpp"\n' + text
    if relative == 'runtime/flash/FlashForward.cpp':
        text = once(text, '  const bool hcUpF32 = flashHCUpF32MPPEnabled();',
                    '  const bool hcPadVerifyR4 = hc_pad_verify_sep22::requested();\n'
                    '  const bool hcUpF32 = flashHCUpF32MPPEnabled();')
        text = once(text, '    descriptor.validate();',
                    '    descriptor.validate();\n'
                    '    hc_pad_verify_sep22::validateDependencies(hcUpF32,fuseHC,cacheFloat);\n'
                    '    if(hcPadVerifyR4 && maximumRows<8)\n'
                    '      throw std::invalid_argument("HC padding requires an admitted eight-row HCDown scratch view");')
        text = once(text, '    // Resource policy is fixed for both selector values. Source BF16 operands\n',
                    '''    if(hcPadVerifyR4) {
      const auto checkHC=[&](const std::string &prefix,bool injection) {
        const auto &d=weights.projection(prefix+".input_mix_weight_down");
        const auto &u=weights.projection(prefix+".input_mix_weight_up");
        const auto *i=injection ? &weights.projection(prefix+".block_inject_weight") : nullptr;
        if(!floatDenseCache || !floatDenseWorkspace || !floatDenseCache->contains(prefix+".input_mix_weight_up") ||
            !flashHCUpF32MPPGeometry(prefix+".input_mix_weight_up",4,10240,320) ||
            !supportsHCFused(d,u,i,{4,kWidth,4,static_cast<float>(descriptor.normEpsilon)}))
          throw std::invalid_argument("HC padding requires all97 original fused-down/cached-F32-up bindings");
        const auto &t=floatDenseCache->tensor(prefix+".input_mix_weight_up");
        if(t.dtype!=FlashDType::F32 || t.shape!=std::vector<uint64_t>{10240,320} || t.logicalBytes!=13107200ULL)
          throw std::invalid_argument("HC padding original cached up view differs");
      };
      for(uint32_t layer=0;layer<48;++layer) {
        const auto prefix="language_model.model.layers."+std::to_string(layer);
        checkHC(prefix+".attn_hyper_connection",true);checkHC(prefix+".mlp_hyper_connection",true);
      }
      checkHC("language_model.model.hyper_connection_mixer",false);
    }
    // Resource policy is fixed for both selector values. Source BF16 operands
''')
        text = once(text, '          uint32_t rows, bool injection, bool normalizedReady = false) {',
                    '          uint32_t rows, bool injection, bool normalizedReady = false, bool verification = false) {\n'
                    '    hc_pad_verify_sep22::requireFrozen();')
        text = once(text, '    const auto down = bf(Scratch::HCDown, rows, 320);',
                    '    const bool padVerify = hcPadVerifyR4 && hc_pad_verify_sep22::eligible(verification,rows,maximumRows) &&\n'
                    '        hcUpF32 && fuseHC && floatDenseCache && floatDenseWorkspace &&\n'
                    '        flashHCUpF32MPPGeometry(prefix+".input_mix_weight_up",rows,10240,320) &&\n'
                    '        floatDenseCache->contains(prefix+".input_mix_weight_up");\n'
                    '    const auto down = bf(Scratch::HCDown, padVerify ? 8 : rows, 320);')
        before = '      addHCFusedDown(graph, normalized, downProjection, injectionProjection, down,\n'
        after = '''      if(padVerify) {
        hc_pad_verify_sep22::addChain(backend,graph,normalized,downProjection,injectionProjection,down,
            injection ? bf(Scratch::HCInjectionWeights,rows,4) : metal::MetalBuffer{},
            floatDenseCache->tensor(prefix+".input_mix_weight_up"),bf(Scratch::Mixed,rows,kWidth),diagnostics,
            *floatDenseWorkspace,geometry,verification,maximumRows);
        hcUpCounters.recordAttempt(rows);hcUpCounters.recordEligible(rows);hcUpCounters.recordCachedEncoded(rows);
        return;
      }
      addHCFusedDown(graph, normalized, downProjection, injectionProjection, down,
'''
        text = once(text, before, after)
        for prefix, injection in [('prefix + ".attn_hyper_connection"', 'true'),
                                  ('prefix + ".mlp_hyper_connection"', 'true'),
                                  ('"language_model.model.hyper_connection_mixer"', 'false')]:
            before = 'impl_->hc(graph, ' + prefix + ', rows, ' + injection + ', normalizedReady);'
            text = once(text, before, before[:-2] + ', verification);')
        text = once(text, '      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +',
                    '      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +\n'
                    '      (impl_->hcPadVerifyR4 ? hc_pad_verify_sep22::marker : "") +')
    else:
        text = once(text, '      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.',
                    '      hc_pad_verify_sep22::validateDependencies(environmentSwitch("SPLASH_FLASH_HC_UP_F32_MPP"),\n'
                    '          environmentSwitch("SPLASH_FLASH_FUSE_HC"),environmentSwitch("SPLASH_FLASH_FLOAT_DENSE_CACHE"));\n'
                    '      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.')
        # kernelRoutes and these immutable values participate in identity; no
        # changing graph counter is inserted in that object.
        text = once(text, '      << R"(,"target_all_rows_full512":true,"original_target_gpu_omitted":true)"',
                    '      << R"(,"hc_pad_verify_r4_enabled":)"<<(hc_pad_verify_sep22::requested()?"true":"false")\n'
                    '      << R"(,"hc_pad_verify_r4_policy":)"<<json::quote(hc_pad_verify_sep22::policy)\n'
                    '      << R"(,"hc_pad_verify_r4_source_certificate":)"<<json::quote(hc_pad_verify_sep22::certificate)\n'
                    '      << R"(,"target_all_rows_full512":true,"original_target_gpu_omitted":true)"')
        text = once(text, '      << R"(,"gdn_verification_storage":{"lazy_enabled":)"',
                    '      << R"(,"hc_pad_verify_r4_route_counters":{"scope":"main VerifyR4 graph construction; notGPU completion","graph_calls":)"\n'
                    '      <<hc_pad_verify_sep22::graphCalls.load(std::memory_order_relaxed)<<R"(,"graph_rows":)"\n'
                    '      <<hc_pad_verify_sep22::graphRows.load(std::memory_order_relaxed)<<R"(,"padding_dispatches_saved":)"\n'
                    '      <<hc_pad_verify_sep22::savedPaddingDispatches.load(std::memory_order_relaxed)<<\'}\'\n'
                    '      << R"(,"gdn_verification_storage":{"lazy_enabled":)"')
    return text


if __name__ == '__main__':
    # Source-only dry transform, no writes to the specified parent.
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    base = (root / args.base).resolve()
    for relative in sorted(CHANGED):
        transform(relative, (base / 'source' / relative).read_text())
    print(json.dumps({'pass': True, 'source_transform_only': True, 'gpu_executed': False,
                      'parent_files_written': False, 'changed_paths': sorted(CHANGED)}))
