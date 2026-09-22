#!/usr/bin/env python3
"""Prepare private Worker-only native emission clocks; no hashes/GPU/payload reads."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HEADER = "dev/benchmarks/current_batch_sep22/NativeLifecycleTrace.hpp"


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("qualified Worker source anchor differs: " + before[:90])
    return source.replace(before, after, 1)


def transform(source):
    source = once(source, '#include "flash/FlashRequestCommandTrace.hpp"',
        '#include "flash/FlashRequestCommandTrace.hpp"\n#include "' + HEADER + '"')
    source = once(source, 'FlashMTPTeacherBulkForward *teacherBulk = nullptr, uint32_t teacherBulkQAPauseMilliseconds = 0)',
        'FlashMTPTeacherBulkForward *teacherBulk = nullptr, uint32_t teacherBulkQAPauseMilliseconds = 0,\n'
        '         std::unique_ptr<batch_clock_sep22::Trace> nativeLifecycleTrace = {})')
    source = once(source, '        requestCommandTrace_(std::move(requestCommandTrace)),',
        '        requestCommandTrace_(std::move(requestCommandTrace)),\n'
        '        nativeLifecycleTrace_(std::move(nativeLifecycleTrace)),')
    source = once(source, '  std::unique_ptr<FlashRequestCommandTrace> requestCommandTrace_;',
        '  std::unique_ptr<FlashRequestCommandTrace> requestCommandTrace_;\n'
        '  std::unique_ptr<batch_clock_sep22::Trace> nativeLifecycleTrace_;')
    source = once(source, '        words_((weights.descriptor().vocabularySize + 31) / 32) {}',
        '        words_((weights.descriptor().vocabularySize + 31) / 32) {\n'
        '    if (nativeLifecycleTrace_) nativeLifecycleTrace_->setSourceIdentity(weights_.sourceIdentity());\n  }')
    source = once(source, '    return transport_.failed() ? 2 : 0;',
        '    if (nativeLifecycleTrace_) nativeLifecycleTrace_->finish();\n'
        '    return transport_.failed() ? 2 : 0;')
    source = once(source, '    transport_.send(wire::DoneEvent{id, reason, static_cast<uint32_t>(request.frame.promptTokens.size()),',
        '''    if (nativeLifecycleTrace_)
      nativeLifecycleTrace_->record("done", instance_, id, request.generation,
          std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count(),
          request.emitted, static_cast<uint32_t>(request.frame.promptTokens.size()),
          reason == wire::FinishReason::Length ? "length" :
          reason == wire::FinishReason::Stop ? "stop" : "cancelled");
    transport_.send(wire::DoneEvent{id, reason, static_cast<uint32_t>(request.frame.promptTokens.size()),''')
    source = once(source, '    if (!request.firstToken) { request.firstToken = now; ttft_.append(milliseconds(request.arrived, now)); }',
        '    const bool firstNativeEmission = !request.firstToken;\n'
        '    if (!request.firstToken) { request.firstToken = now; ttft_.append(milliseconds(request.arrived, now)); }')
    source = once(source, '    request.lastToken = now;\n    transport_.send(wire::TokensEvent{request.frame.requestId, request.emitted,',
        '''    request.lastToken = now;
    if (nativeLifecycleTrace_ && firstNativeEmission)
      nativeLifecycleTrace_->record("first_emission", instance_, request.frame.requestId,
          request.generation,
          std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count(),
          static_cast<uint32_t>(tokens.size()), static_cast<uint32_t>(request.frame.promptTokens.size()));
    transport_.send(wire::TokensEvent{request.frame.requestId, request.emitted,''')
    source = once(source, '      auto requestCommandTrace = FlashRequestCommandTrace::fromEnvironment();',
        '      auto nativeLifecycleTrace = batch_clock_sep22::Trace::fromEnvironment();\n'
        '      auto requestCommandTrace = FlashRequestCommandTrace::fromEnvironment();')
    source = once(source, '  out << R"(,"request_command_trace":{"enabled":)"',
        '''  if (nativeLifecycleTrace_)
    out << R"(,"native_lifecycle_timestamps_sep22":{"enabled":true,"schema":"splash-native-lifecycle-trace-sep22-v1","clock_kind":"native-worker-std-steady-nanoseconds-v1","clock_domain":)"
        << json::quote(nativeLifecycleTrace_->clockDomain())
        << R"(,"native_worker_pid":)" << nativeLifecycleTrace_->workerPid()
        << R"(,"native_worker_start_token_ns":)" << nativeLifecycleTrace_->workerStartTokenNs()
        << R"(,"native_worker_instance_id":)"
        << instance_ << R"(,"source_identity_sha256":)" << json::quote(weights_.sourceIdentity())
        << R"(,"capture":"original first-emission and Done timing boundaries","record_writes":"after serving stops","kernel_math_changed":false})";
  out << R"(,"request_command_trace":{"enabled":)"''')
    return once(source, 'std::move(idleMaintenanceFailureReason), teacherBulk?&*teacherBulk:nullptr,teacherBulkQAPauseMilliseconds);',
        'std::move(idleMaintenanceFailureReason), teacherBulk?&*teacherBulk:nullptr,teacherBulkQAPauseMilliseconds,\n'
        '                    std::move(nativeLifecycleTrace));')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5")
    parser.add_argument("--output", type=Path, default=ROOT / "build/current-batch-native-clock-sep22-v4")
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    if output.exists() or ROOT / "build" not in output.parents:
        raise ValueError("choose a fresh private output under build")
    relative = "runtime/flash/FlashWorker.mm"
    original = (parent / "source" / relative).read_text()
    modified = transform(original)
    worker = output / "source" / relative
    worker.parent.mkdir(parents=True)
    worker.write_text(modified)
    header = output / "source" / HEADER
    header.parent.mkdir(parents=True)
    header.write_text((ROOT / HEADER).read_text())
    witness = {"schema": "splash-current-batch-native-clock-source-plan-sep22-v1",
        "metadata_only": True, "parent": str(parent), "changed_parent_source_paths": [relative],
        "new_header": HEADER, "gpu_executed": False, "model_payload_bytes_read": 0,
        "hashes_computed": 0, "kernel_changes": [], "wire_protocol_changes": [],
        "default_enabled": False, "default_route_has_no_trace_allocation": True,
        "timestamp_marker_flag": "SPLASH_FLASH_NATIVE_LIFECYCLE_TIMESTAMPS_SEP22",
        "buffered_host_metadata_maximum_records": 4096,
        "measured_emission_boundaries_write_files": False,
        "clock_scope": "original Worker Clock::now at first emitTokens and Done; integer steady nanoseconds",
        "extra_native_trace_records_per_successful_request": 2,
        "stop_flush_complete_footer_required": True,
        "build_needed": "compile this Worker TU with private source root first and qualified parent includes/flags; link it in place of the parent Worker object, retaining every other parent object and exact metallib; root verifies/seals artifacts",
        "GPU_qualification_complete": False, "requires_root_build_and_runtime_proof": True}
    (output / "source-plan.json").write_text(json.dumps(witness, indent=2) + "\n")
    print(json.dumps(witness))


if __name__ == "__main__":
    main()
