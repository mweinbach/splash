#!/usr/bin/env python3
"""CPU re-audit of the completed, registered Teacher/V3 reports only.

This preserves the frozen 22 tasks, graders and execution policy. The old
Teacher has no HC profile, so each registered report uses its own frozen
status/coverage/ownership gates. Candidate snapshots cannot select legacy
gates: every snapshot is bound to its report's registered numerical identity
before the original comparator runs. No inference or tensor captures are read.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
COMPACT = ROOT / 'build/compact-native-r4-verify-teacher-sep22-worker-v1b'
V3 = ROOT / 'build/hc-pad-compact-r4-verify-teacher-sep22-worker-v3'
REPORTS = ROOT / 'build/release/flash'
PARENT_DERIVATIVE = 'b87448342df3b3a9ae1642379b8aab513bb208ff234e552bcf70efc26a82c09d'
CANDIDATE_DERIVATIVE = '315e69146b20501aad1203bfb8ab4304e24534c19f8d9d42261e395f33ed9c0e'
REGISTRY = (
    ('sep21-teacher-bulk-ab-qsa-model-and-quality-v1-3.semantic.json',
     PARENT_DERIVATIVE,
     {'splash-flash': '5229511c716af33d163d2f077411cf54d39241925daf99e156cc7f8be532bebc',
      'splash.metallib': 'bb09bf88bb53a8b6e9bfc5254068c16942bb0913784ff1c672d810f13c6eb6f0'}),
    ('sep22-compact-hc-pad-teacher-mtp3-model-and-quality-v3-3.semantic.json',
     CANDIDATE_DERIVATIVE,
     {'splash-flash': '6f87fb12c896420e03391122ef054418c8b854a5a7104e69bf4881c8053cddf4',
      'splash.metallib': '390e67bb04e81c1aa291013f2de16fbca27bbf7771819b17af95aa7be4c88177'}),
)
REGISTERED_HELPER = '362895b6b28c5010dcc025ccb3bec3a488d51a96462f4cf2a79609d769063f8c'
OUTPUT = REPORTS / 'sep22-teacher-vs-compact-hc-fast-v3-frozen22-comparison-v1.json'
RECEIPT = REPORTS / 'sep22-teacher-vs-compact-hc-fast-v3-frozen22-source-receipt-v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshots(report):
    yield report['initial_status']
    yield report['final_status']
    for case in report['cases']:
        yield case['status_before']
        yield case['status_after']


def main():
    helper_path = V3 / 'source/dev/benchmarks/hc_pad_verify_worker_sep22/semantic_quality.py'
    if digest(helper_path) != REGISTERED_HELPER:
        raise ValueError('Unknown V3 admission helper')
    old_path = COMPACT / 'source/dev/benchmarks/expert_r4_compact_verify_worker_sep22/semantic_quality.py'
    original = 'source/dev/benchmarks/prefill4k_attribution_quality.py'
    if digest(V3 / original) != digest(COMPACT / original):
        raise ValueError('Original frozen 22-task runner differs')
    loaded_reports = []
    for name, derivative, artifacts in REGISTRY:
        report = json.loads((REPORTS / name).read_text())
        if report['runtime_file_sha256'] != artifacts:
            raise ValueError('Registered runtime artifact differs: ' + name)
        if report['store_witness']['target_numerical_derivative_sha256'] != derivative:
            raise ValueError('Registered derivative witness differs: ' + name)
        for status in snapshots(report):
            if status['identity']['target_numerical_derivative_sha256'] != derivative:
                raise ValueError('Snapshot cannot select a different report gate: ' + name)
            if derivative == PARENT_DERIVATIVE and ('hc_pad_verify_r4_enabled' in status['identity'] or 'compact_native_r4_verify' in status):
                raise ValueError('Registered legacy snapshot has candidate feature metadata')
        loaded_reports.append(report)
    old = load('_completed_v3_old_compact_adapter', old_path).load(build=COMPACT)
    current = load('_completed_v3_mandatory_hc_adapter', helper_path).load(V3)
    functions = {PARENT_DERIVATIVE: {k: getattr(old, k) for k in ('gate_status', 'coverage', 'ownership_policy')},
                 CANDIDATE_DERIVATIVE: {k: getattr(current, k) for k in ('gate_status', 'coverage', 'ownership_policy')}}

    def selected(status):
        derivative = status.get('identity', {}).get('target_numerical_derivative_sha256')
        if derivative not in functions:
            raise ValueError('Unknown saved status identity')
        return functions[derivative]

    def gate_status(status, *args, **kwargs):
        return selected(status)['gate_status'](status, *args, **kwargs)

    def coverage(before, after, *args, **kwargs):
        if before['identity']['target_numerical_derivative_sha256'] != after['identity']['target_numerical_derivative_sha256']:
            raise ValueError('Saved coverage identities differ')
        return selected(before)['coverage'](before, after, *args, **kwargs)

    def ownership(status, *args, **kwargs):
        return selected(status)['ownership_policy'](status, *args, **kwargs)

    current.gate_status, current.coverage, current.ownership_policy = gate_status, coverage, ownership
    result = current.compare(SimpleNamespace(reports=[REPORTS / x[0] for x in REGISTRY],
                                            output=OUTPUT, allow_runtime_change=True))
    if result:
        raise ValueError('Registered report evidence re-audit failed')
    comparison = json.loads(OUTPUT.read_text())
    original_parent = json.loads((REPORTS / 'sep21-teacher-bulk-ab-qsa-model-and-quality-v1.json').read_text())
    model = json.loads((REPORTS / 'sep22-compact-hc-pad-teacher-mtp3-model-and-quality-v3.json').read_text())
    a = [x for x in original_parent['waves'] if not x['warmup']]
    b = [x for x in model['waves'] if not x['warmup']]
    if len(a) != len(b) or len(a) != 3:
        raise ValueError('Expected the registered three measured coding trials')
    coding = []
    for left, right in zip(a, b):
        if len(left['records']) != 1 or len(right['records']) != 1:
            raise ValueError('Expected one actual coding record per trial')
        coding.append({'trial': right['trial'], 'text_equal': left['records'][0]['text'] == right['records'][0]['text'],
                       'usage_equal': left['records'][0]['usage'] == right['records'][0]['usage'],
                       'cycles_equal': left['native_counter_delta']['mtp.verification_cycles'] == right['native_counter_delta']['mtp.verification_cycles'],
                       'accepted_drafts_equal': left['native_counter_delta']['metrics.accepted_draft_tokens'] == right['native_counter_delta']['metrics.accepted_draft_tokens']})
    receipt = {'schema': 'registered-completed-HC-fast-V3-frozen22-source-audit-v1',
               'CPU_only': True, 'GPU_work': False, 'model_or_tensor_capture_payload_reads': False,
               'generated_answer_regrading_and_coding_equality_explicitly_authorized_by_Root': True,
               'V3_tree_modified': False, 'mandatory_helper_sha256': REGISTERED_HELPER,
               'original_22_runner_sha256': digest(V3 / original),
               'audit_program_sha256': digest(__file__),
               'report_input_sha256': {name: digest(REPORTS / name) for name, _, _ in REGISTRY},
               'registered_runtime_artifacts_and_all_snapshot_derivatives_bound': True,
               'legacy_dispatch_scope': 'ONLY exact registered Teacher runtime and derivative; all candidate snapshots prevalidated to candidate derivative',
               'comparison_report': str(OUTPUT), 'comparison_sha256': digest(OUTPUT),
               'valid_evidence': comparison['valid'], 'no_new_task_regressions': comparison['no_new_task_regressions'],
               'all_tasks_pass': comparison['all_tasks_pass'],
               'generation_differences': comparison['comparisons'][0]['generation_differences'],
               'coding_trials': coding, 'all_three_coding_texts_exact': all(x['text_equal'] for x in coding),
               'launch_metadata_drift_is_separate': True}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('valid_evidence', 'no_new_task_regressions', 'all_tasks_pass', 'generation_differences', 'all_three_coding_texts_exact')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
