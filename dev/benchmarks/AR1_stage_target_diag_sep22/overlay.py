"""Private Worker-only profile scope; never changes Forward or any graph."""
PRIVATE = "dev/benchmarks/AR1_stage_target_diag_sep22"


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Current standard AR1 source anchor drift: " + before[:100])
    return text.replace(before, after)


def transform(path, text):
    if path != "runtime/flash/FlashWorker.mm":
        return text
    text = '#include "' + PRIVATE + '/bridge.hpp"\n' + text
    text = once(text,
        "      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.",
        "      ar1_stage_target_diag_sep22::validateStartup(); // Private diagnostic only, default Off before model/backend.\n"
        "      raw_q4_verify_sep22::validateDependencies(); // Freeze strict rowpair before paths/model/backend.")
    text = once(text, "  Transport &transport_;",
        "  ar1_stage_target_diag_sep22::Diagnostic ar1Diagnostic_{ar1_stage_target_diag_sep22::Config::fromEnvironment()};\n"
        "  Transport &transport_;")
    text = once(text,
        "      const auto result = forward_.forward(*request.state, token);\n"
        '      traceRequestCommand("autoregressive_decode", "target_trunk", {id, generation}, 1, result.timing);',
        "      ar1_stage_target_diag_sep22::ScopedAR1 diagnosticScope(ar1Diagnostic_,id,generation,\n"
        "          !request.mtpState.has_value(),request.promptOffset,request.frame.promptTokens.size(),\n"
        "          request.state->logicalLength(),governor_,backend_);\n"
        "      const auto result = forward_.forward(*request.state, token);\n"
        "      diagnosticScope.complete(result.logicalLength,result.logitRows);\n"
        '      traceRequestCommand("autoregressive_decode", "target_trunk", {id, generation}, 1, result.timing);')
    return text
