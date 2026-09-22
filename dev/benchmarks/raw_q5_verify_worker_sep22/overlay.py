"""Only two private C++ consumers; preserve the qualified Q4 branch and all public headers."""
PRIVATE='dev/benchmarks/raw_q5_verify_worker_sep22'
changes=[]
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('rawQ5 overlay anchor drift:'+a[:100])
    changes.append((a,b))
    return s.replace(a,b)
def transform(path,s):
    if path not in ['runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm']:return s
    s='#include "'+PRIVATE+'/policy.hpp"\n'+s
    if path.endswith('FlashForward.cpp'):
        s=once(s,'  const bool rawQ4Rowpair = raw_q4_verify_sep22::requested();','  const bool rawQ4Rowpair = raw_q4_verify_sep22::requested();\n  const bool rawQ5Rowpair = raw_q5_verify_sep22::requested();')
        s=once(s,'    raw_q4_verify_sep22::validateInventory(weights);','    raw_q4_verify_sep22::validateInventory(weights);\n    raw_q5_verify_sep22::validateDependencies();\n    raw_q5_verify_sep22::validateInventory(weights);')
        s=once(s,'      raw_q4_verify_sep22::validateCachePresence(weights,*floatDenseCache);','      raw_q4_verify_sep22::validateCachePresence(weights,*floatDenseCache);\n      raw_q5_verify_sep22::validateCachePresence(weights,*floatDenseCache);')
        s=once(s,'            raw_q4_verify_sep22::add(graph,prefix,input,projection,output,diagnostics,rows,verification,singletonMain);','            raw_q4_verify_sep22::add(graph,prefix,input,projection,output,diagnostics,rows,verification,singletonMain);\n          else if (rawQ5Rowpair && raw_q5_verify_sep22::selected(prefix,rows,verification,singletonMain,projection))\n            raw_q5_verify_sep22::add(graph,prefix,input,projection,output,diagnostics,rows,verification,singletonMain);')
        s=once(s,'      raw_q4_verify_sep22::marker() +','      raw_q4_verify_sep22::marker() +\n      raw_q5_verify_sep22::marker() +')
    else:
        s=once(s,'      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.','      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.\n      raw_q5_verify_sep22::validateDependencies(); // Freeze strict child rowpair before paths/model/backend.')
        s=once(s,'      << R"(,"gdn_verification_storage":{"lazy_enabled":)"','''      << R"(,"raw_q5_rowpair_verify":{"schema":"mainGDNout-rawQ5-VerifyR4-rowpair-v1","requested":)"
      <<(raw_q5_verify_sep22::requested()?"true":"false")<<R"(,"source_identity_sha256":)"<<json::quote(raw_q5_verify_sep22::kSourceIdentitySha256)
      <<R"(,"scope":)"<<json::quote(raw_q5_verify_sep22::kScope)<<R"(,"qualified_candidate_AIR_sha256":)"<<json::quote(raw_q5_verify_sep22::kAIR)
      <<R"(,"graph_calls":)"<<raw_q5_verify_sep22::graphCalls.load(std::memory_order_relaxed)
      <<R"(,"graph_rows":)"<<raw_q5_verify_sep22::graphRows.load(std::memory_order_relaxed)
      <<R"(,"expected_main_roles_per_VerifyR4":36,"GPU_allocation_bytes_added":0,"counter_scope":"process graph construction, not GPU completion","whole_state_qualified":false})"
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
    return s
