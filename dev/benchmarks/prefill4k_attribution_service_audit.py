#!/usr/bin/env python3
"""CPU-only audit of paired teacher-cache HTTP quality/lifecycle reports."""
import argparse
import copy
import json
import math
from pathlib import Path


def get(document, path):
    for key in path.split('.'):
        if not isinstance(document, dict):
            return None
        document = document.get(key)
    return document


def finite(document):
    if isinstance(document, float):
        if not math.isfinite(document) or 0 < abs(document) < 1e-100:
            raise ValueError('Nonfinite or implausibly tiny numeric metadata')
    elif isinstance(document, dict):
        for value in document.values():
            finite(value)
    elif isinstance(document, list):
        for value in document:
            finite(value)


def normal_body(body):
    result = copy.deepcopy(body)
    # Tool call IDs vary with server request IDs; the named function and its
    # exact argument string and tool result remain part of the comparison.
    for message in result.get('messages', []):
        if 'tool_call_id' in message:
            message['tool_call_id'] = '<generated-tool-id>'
        for call in message.get('tool_calls', []):
            call.pop('id', None)
    return result


def normal_output(record):
    result = {name: record.get(name) for name in ['text', 'reasoning_text', 'finish_reason', 'usage', 'http_status', 'content_type', 'model_ids']}
    calls = copy.deepcopy(record.get('tool_calls', []))
    for call in calls:
        call.pop('id', None)
    result['tool_calls'] = calls
    result['request_body'] = normal_body(record['request_body'])
    return result


def compare_outputs(baseline, candidate):
    left = {case['id']: case for case in baseline['cases']}
    right = {case['id']: case for case in candidate['cases']}
    if left.keys() != right.keys():
        raise ValueError('Qualification case coverage differs')
    results = []
    for case_id, case in left.items():
        if case_id in ['cancellation', 'deadline']:
            continue
        peer = right[case_id]
        if len(case['records']) != len(peer['records']):
            raise ValueError('Qualification response count differs')
        for index, (a, b) in enumerate(zip(case['records'], peer['records'], strict=True)):
            results.append({'case': case_id, 'record': index, 'normalized_output_and_request_equal': normal_output(a) == normal_output(b)})
    return results


def audit_status(status, selected):
    errors = []
    if get(status, 'mtp.teacher_cache_only_requested') is not selected:
        errors.append('Effective selector is not the requested value')
    route = get(status, 'identity.mtp_teacher_priming_route')
    expected = 'mtp-teacher-cache-only-original-preparation-pooling-no-attention-or-mlp-v1' if selected else 'mtp-full-forward-none-logits-v1'
    if route != expected:
        errors.append('Actual teacher route identity differs')
    if get(status, 'mtp.singleton_maximum_draft_tokens') != 3:
        errors.append('Current depth-3 policy differs')
    for name in ['metal.healthy', 'memory_audit.valid', 'transport.ready']:
        if get(status, name) is not True:
            errors.append(f'{name} is not true')
    if get(status, 'scheduler.command_in_flight') is not False:
        errors.append('Native GPU work has not completed')
    for name in ['scheduler.active_requests', 'scheduler.prefilling', 'scheduler.decoding', 'scheduler.waiting_mask']:
        if get(status, name) != 0:
            errors.append(f'{name} is not idle')
    if get(status, 'metrics.metal_failures') != 0:
        errors.append('Metal failures are nonzero')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    baseline, candidate = [json.loads(path.read_text()) for path in [a.baseline, a.candidate]]
    errors = []
    for name, report, selected in [('baseline', baseline, False), ('candidate', candidate, True)]:
        finite(report)
        if not report.get('completed') or not report.get('pass') or len(report.get('cases', [])) != 8:
            errors.append(f'{name} did not pass all eight complete cases')
        for case in report['cases']:
            if case.get('status') != 'passed':
                errors.append(f"{name} {case['id']} is not passed")
            for which in ['status_before', 'status_after']:
                errors.extend(f"{name} {case['id']} {which}: {error}" for error in audit_status(case[which], selected))
        for which in ['initial_status', 'final_status']:
            errors.extend(f'{name} {which}: {error}' for error in audit_status(report[which], selected))
    if baseline['settings']['nonce'] != candidate['settings']['nonce']:
        errors.append('Paired quality nonce differs')
    comparisons = compare_outputs(baseline, candidate)
    if len(comparisons) != 9 or not all(row['normalized_output_and_request_equal'] for row in comparisons):
        errors.append('Nine non-control normalized responses did not all match')
    prime = []
    for case in candidate['cases']:
        before, after = case['status_before'], case['status_after']
        old, new = [get(status, 'mtp.teacher_cache_only_priming_calls') for status in [before, after]]
        command_old, command_new = [get(status, 'mtp.head_priming.host_command_subphases.timed_commands') for status in [before, after]]
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in [old, new, command_old, command_new]):
            errors.append(f"{case['id']} teacher/command counter is unavailable")
            continue
        delta, command_delta = new - old, command_new - command_old
        if delta < 0 or delta > command_delta:
            errors.append(f"{case['id']} selected call counter exceeds actual teacher commands")
        prime.append({'case': case['id'], 'cache_only_successful_calls': delta, 'all_teacher_commands': command_delta,
            'eligible_requests': get(after, 'mtp.eligible_requests') - get(before, 'mtp.eligible_requests')})
    total = sum(row['cache_only_successful_calls'] for row in prime)
    if total == 0:
        errors.append('Candidate had no completed cache-only teacher calls')
    for case in baseline['cases']:
        if get(case['status_after'], 'mtp.teacher_cache_only_priming_calls') != 0:
            errors.append('Off control unexpectedly used teacher cache priming')
    result = {'schema': 'splash-teacher-cache-service-audit-v1', 'valid': not errors, 'cpu_only_audit': True,
        'source_baseline': str(a.baseline), 'source_candidate': str(a.candidate), 'errors': errors,
        'all_numeric_metadata_finite': True, 'eight_quality_lifecycle_cases_complete': not any('eight' in error for error in errors),
        'normalized_non_control_response_comparisons': comparisons, 'teacher_call_deltas': prime,
        'candidate_successful_teacher_cache_calls': total, 'time_dependent_control_outputs_excluded': True,
        'throughput_benchmark': False, 'whole_model_2k_256_positive_control_still_required': True}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'valid': result['valid'], 'selected_calls': total, 'errors': errors}))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
