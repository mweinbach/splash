"""Two private C++ changes; public headers and all other consumers unchanged."""
PRIVATE='dev/benchmarks/raw_q4_verify_worker_sep22'
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('rawQ4 overlay anchor drift:'+a[:100])
    return s.replace(a,b)
def transform(path,s):
    if path not in ['runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm']:return s
    s='#include "'+PRIVATE+'/policy.hpp"\n'+s
    if path.endswith('FlashForward.cpp'):
        s=once(s,'  const bool selectiveFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE");','  const bool selectiveFloat = fusionEnabled("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE");\n  const bool rawQ4Rowpair = raw_q4_verify_sep22::requested();')
        s=once(s,'    descriptor.validate();','    descriptor.validate();\n    raw_q4_verify_sep22::validateDependencies();\n    raw_q4_verify_sep22::validateInventory(weights);')
        s=once(s,'      floatDenseWorkspace = std::make_unique<FlashFloatDenseSmallRowsWorkspace>(backend);','      floatDenseWorkspace = std::make_unique<FlashFloatDenseSmallRowsWorkspace>(backend);\n      raw_q4_verify_sep22::validateCachePresence(weights,*floatDenseCache);')
        s=once(s,'        else\n          addAffine(graph, input, projection, output, diagnostics, rows);','''        else {
          if (rawQ4Rowpair && raw_q4_verify_sep22::selected(prefix,rows,verification,singletonMain,projection))
            raw_q4_verify_sep22::add(graph,prefix,input,projection,output,diagnostics,rows,verification,singletonMain);
          else
            addAffine(graph, input, projection, output, diagnostics, rows);
        }''')
        s=once(s,'      guard_hc_fast_composite_sep22::marker() +','      guard_hc_fast_composite_sep22::marker() +\n      raw_q4_verify_sep22::marker() +')
    else:
        s=once(s,'      (void)adaptive_expert_tail_sg2k128_sep21::requested(); // Freeze tail control before paths/metadata/backend.','      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.\n      (void)adaptive_expert_tail_sg2k128_sep21::requested(); // Freeze tail control before paths/metadata/backend.')
        s=once(s,'      << R"(,"gdn_verification_storage":{"lazy_enabled":)"','''      << R"(,"raw_q4_rowpair_verify":{"schema":"mainGDNQKV-rawQ4-VerifyR4-rowpair-v1","requested":)"
      <<(raw_q4_verify_sep22::requested()?"true":"false")<<R"(,"source_identity_sha256":)"<<json::quote(raw_q4_verify_sep22::kSourceIdentitySha256)
      <<R"(,"scope":)"<<json::quote(raw_q4_verify_sep22::kScope)<<R"(,"qualified_candidate_AIR_sha256":)"<<json::quote(raw_q4_verify_sep22::kAIR)
      <<R"(,"graph_calls":)"<<raw_q4_verify_sep22::graphCalls.load(std::memory_order_relaxed)
      <<R"(,"graph_rows":)"<<raw_q4_verify_sep22::graphRows.load(std::memory_order_relaxed)
      <<R"(,"expected_main_roles_per_VerifyR4":26,"GPU_allocation_bytes_added":0,"counter_scope":"process graph construction, not GPU completion","whole_state_qualified":false})"
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
    return s
