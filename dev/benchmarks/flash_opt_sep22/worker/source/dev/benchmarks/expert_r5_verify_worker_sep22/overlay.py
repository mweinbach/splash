PRIVATE='dev/benchmarks/expert_r5_verify_worker_sep22'
def once(s,old,new):
    if s.count(old)!=1:raise ValueError('R5 overlay anchor changed:'+old[:100])
    return s.replace(old,new)
def transform(path,s):
    if path not in ('runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm'):return s
    s='#include "'+PRIVATE+'/policy.hpp"\n'+s
    if path.endswith('FlashForward.cpp'):
        s=once(s,'    descriptor.validate();','    descriptor.validate();\n    compact_native_r5_verify_sep22::validateConstruction(maximumVerifyRows,allRowsInt8Target,blockMoE);')
        s=once(s,'      raw_q4_verify_sep22::marker() +','      raw_q4_verify_sep22::marker() +\n      compact_native_r5_verify_sep22::marker() +')
        s=once(s,'      addMoEBlockedPack(graph, mixed, ids, impl_->blockedScratch, diag, rows, tile);','''      const bool compactR5=impl_->allRowsInt8Target&&impl_->int8ExpertStore&&tile==FlashMoEBlockedTile::M16N64&&compact_native_r5_verify_sep22::eligible(rows,verification,true);
      if(compactR5)
        compact_native_r5_verify_sep22::addSetup(graph,mixed,ids,impl_->blockedScratch,diag,rows,tile,verification,true);
      else
        addMoEBlockedPack(graph, mixed, ids, impl_->blockedScratch, diag, rows, tile);''')
        s=once(s,'          impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);','          impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);\n          if(compactR5)compact_native_r5_verify_sep22::recordCompletedGraph(rows);')
    else:
        s=once(s,'      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.','''      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.
      const auto earlyR5Depth=environmentSingletonMTPPolicy();
      compact_native_r5_verify_sep22::validateDepth(environmentSwitch("SPLASH_FLASH_MTP"),earlyR5Depth.maximumDepth,earlyR5Depth.explicitOverride);''')
        s=once(s,'      const auto singletonMTP = environmentSingletonMTPPolicy();','      const auto singletonMTP = environmentSingletonMTPPolicy();\n      compact_native_r5_verify_sep22::validateDepth(mtpEnabled,singletonMTP.maximumDepth,singletonMTP.explicitOverride);')
        s=once(s,'      << R"(,"gdn_verification_storage":{"lazy_enabled":)"','''      << R"(,"compact_native_r5_verify":{"schema":"singleton-main-R5-integer-only-original-native-M16-six-stage-v1","requested":)"
      <<(compact_native_r5_verify_sep22::requested()?"true":"false")<<R"(,"enabled":)"<<(compact_native_r5_verify_sep22::requested()?"true":"false")
      <<R"(,"source_identity_sha256":)"<<json::quote(compact_native_r5_verify_sep22::kSourceIdentitySha256)
      <<R"(,"planner_AIR_sha256":)"<<json::quote(compact_native_r5_verify_sep22::kAIR)<<R"(,"scope":)"<<json::quote(compact_native_r5_verify_sep22::kScope)
      <<R"(,"graph_calls":)"<<compact_native_r5_verify_sep22::graphCalls.load(std::memory_order_relaxed)
      <<R"(,"graph_rows":)"<<compact_native_r5_verify_sep22::graphRows.load(std::memory_order_relaxed)
      <<R"(,"dispatches_per_layer":6,"GPU_allocation_bytes_added":0,"maximum_verify_rows":5,"whole_state_qualified":false})"
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
    return s
