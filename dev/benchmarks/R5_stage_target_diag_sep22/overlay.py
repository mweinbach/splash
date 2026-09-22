PRIVATE='dev/benchmarks/R5_stage_target_diag_sep22'
def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Current R5 diagnostic source anchor changed: '+old[:100])
    return text.replace(old,new)
def transform(path,text):
    if path not in ('runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm'):return text
    text='#include "'+PRIVATE+'/bridge.hpp"\n'+text
    if path.endswith('FlashForward.cpp'):
        text=once(text,'    timing = impl_->backend.submitCommand(graph.dispatches());\n    uint32_t status = 0;\n','    r5_stage_target_diag_sep22::recordGraph(graph,verification,rows,begin);\n    timing = impl_->backend.submitCommand(graph.dispatches());\n    uint32_t status = 0;\n')
    else:
        text=once(text,'      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.','      r5_stage_target_diag_sep22::validateStartup(); // Private diagnostic flag freezes before paths/model/backend.\n      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.')
        text=once(text,'  Transport &transport_;','  r5_stage_target_diag_sep22::Diagnostic stageDiagnostic_{r5_stage_target_diag_sep22::Config::fromEnvironment()};\n  Transport &transport_;')
        text=once(text,'  const auto verified = depth ? forward_.verify(*request.state, inputs)\n                              : forward_.forward(*request.state, inputs, false, true);','  r5_stage_target_diag_sep22::ScopedVerifyCookie diagnosticScope(stageDiagnostic_,id,generation,depth,static_cast<uint32_t>(inputs.size()),governor_,backend_);\n  const auto verified = depth ? forward_.verify(*request.state, inputs)\n                              : forward_.forward(*request.state, inputs, false, true);\n  diagnosticScope.complete();')
    return text
