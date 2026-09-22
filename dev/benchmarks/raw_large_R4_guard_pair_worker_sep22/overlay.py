"""Only two private implementation TUs change; legacy Q4 remains first."""
PRIVATE = 'dev/benchmarks/raw_large_R4_guard_pair_worker_sep22'
CHANGED_PATHS = ('runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm')


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('raw large R4 overlay anchor drift: ' + before[:100])
    return source.replace(before, after)


def transform(path, source):
    if path not in CHANGED_PATHS:
        return source
    source = '#include "' + PRIVATE + '/policy.hpp"\n' + source
    if path.endswith('FlashForward.cpp'):
        source = once(source,
            '  const bool rawQ4Rowpair = raw_q4_verify_sep22::requested();',
            '  const bool rawQ4Rowpair = raw_q4_verify_sep22::requested();\n'
            '  const bool rawLargeR4GuardPair = raw_large_r4_guard_pair_sep22::requested();')
        source = once(source,
            '  std::unique_ptr<FlashFloatDenseSmallRowsWorkspace> floatDenseWorkspace;',
            '  std::unique_ptr<FlashFloatDenseSmallRowsWorkspace> floatDenseWorkspace;\n'
            '  std::optional<raw_large_r4_guard_pair_sep22::QualifiedRoles> rawLargeR4Roles;')
        source = once(source,
            '    raw_q4_verify_sep22::validateDependencies();',
            '    raw_q4_verify_sep22::validateDependencies();\n'
            '    raw_large_r4_guard_pair_sep22::validateDependencies();')
        source = once(source,
            '      raw_q4_verify_sep22::validateCachePresence(weights,*floatDenseCache);',
            '      raw_q4_verify_sep22::validateCachePresence(weights,*floatDenseCache);\n'
            '      if (rawLargeR4GuardPair) rawLargeR4Roles.emplace(weights, *floatDenseCache);')
        source = once(source,
            '            raw_q4_verify_sep22::add(graph,prefix,input,projection,output,diagnostics,rows,verification,singletonMain);\n'
            '          else\n'
            '            addAffine(graph, input, projection, output, diagnostics, rows);',
            '            raw_q4_verify_sep22::add(graph,prefix,input,projection,output,diagnostics,rows,verification,singletonMain);\n'
            '          else if (rawLargeR4GuardPair && rawLargeR4Roles &&\n'
            '              rawLargeR4Roles->selected(projection, rows, verification, singletonMain))\n'
            '            rawLargeR4Roles->add(graph, input, projection, output, diagnostics, rows, verification, singletonMain);\n'
            '          else\n'
            '            addAffine(graph, input, projection, output, diagnostics, rows);')
        source = once(source,
            '      raw_q4_verify_sep22::marker() +',
            '      raw_q4_verify_sep22::marker() +\n'
            '      raw_large_r4_guard_pair_sep22::marker() +')
    else:
        source = once(source,
            '      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.',
            '      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.\n'
            '      raw_large_r4_guard_pair_sep22::validateDependencies(); // Freeze child before paths/model/backend.')
        source = once(source,
            '      << R"(,"gdn_verification_storage":{"lazy_enabled":)"',
            '''      << R"(,"raw_large_R4_guard_pair":{"schema":"main-RAW-large-VerifyR4-guard-pair-v1","requested":)"
      << (raw_large_r4_guard_pair_sep22::requested() ? "true" : "false")
      << R"(,"source_identity_sha256":)" << json::quote(raw_large_r4_guard_pair_sep22::kSourceIdentitySha256)
      << R"(,"scope":)" << json::quote(raw_large_r4_guard_pair_sep22::kScope)
      << R"(,"authenticated_RAW_roles":)" << raw_large_r4_guard_pair_sep22::authenticatedRawRoles.load(std::memory_order_relaxed)
      << R"(,"authenticated_new_roles":)" << raw_large_r4_guard_pair_sep22::authenticatedNewRoles.load(std::memory_order_relaxed)
      << R"(,"graph_calls":)" << raw_large_r4_guard_pair_sep22::graphCalls.load(std::memory_order_relaxed)
      << R"(,"graph_rows":)" << raw_large_r4_guard_pair_sep22::graphRows.load(std::memory_order_relaxed)
      << R"(,"excluded_context_graph_calls":)" << raw_large_r4_guard_pair_sep22::excludedContextGraphCalls.load(std::memory_order_relaxed)
      << R"(,"expected_RAW_roles_per_VerifyR4":113,"legacy_Q4_precedence_roles":26,"expected_new_calls_per_actual_H3":87,"physical_rows":4,"GPU_allocation_bytes_added":0,"new_residency_or_backing":false,"numerical_policy_changed":false,"counter_scope":"successful process graph construction, not GPU completion; exclusions count refused add attempts","whole_state_qualified":false})"
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
    return source
