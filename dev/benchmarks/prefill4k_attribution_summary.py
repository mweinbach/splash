#!/usr/bin/env python3
"""Summarize current diagnostic full-target/teacher-prime traces; no GPU work."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


class Classifier:
    def __init__(self):
        self.inside_moe = False
        self.shared = False
        self.routes = 0

    @staticmethod
    def dense(name):
        return name.startswith(('flash_affine', 'flash_dense', 'flash_float_dense', 'flash_int8_head'))

    def family(self, dispatch):
        name = dispatch['pipeline']
        if name == 'flash_moe_route':
            self.inside_moe, self.shared = True, False
            self.routes += 1
            return 'moe'
        if name == 'flash_moe_combine':
            self.inside_moe = self.shared = False
            return 'moe'
        if self.inside_moe and not self.shared and dispatch['threadgroups'][2] > 1 and (
                name.startswith('flash_affine') or name == 'flash_moe_silu_multiply'):
            return 'moe'
        if self.inside_moe and self.dense(name):
            self.shared = True
        if self.shared and (self.dense(name) or name == 'flash_moe_silu_multiply'):
            return 'shared_expert'
        if name.startswith('flash_shared_expert'):
            return 'shared_expert'
        for prefix, value in [('flash_gdn', 'gdn'), ('flash_qsa', 'qsa'), ('flash_ple', 'ple'),
                ('flash_hc', 'hc'), ('flash_forward_hc', 'hc'), ('flash_greedy', 'greedy')]:
            if name.startswith(prefix):
                return value
        if name == 'flash_affine_embedding':
            return 'embedding'
        if name == 'flash_forward_copy_words':
            return 'copies'
        if name.startswith(('flash_moe', 'flash_expert', 'flash_int8_expert')):
            return 'moe'
        return 'dense' if self.dense(name) else 'other'


def check_numbers(value):
    if isinstance(value, float):
        if not math.isfinite(value) or (0 < abs(value) < 1e-100):
            raise ValueError(f'Nonfinite or implausibly tiny numerical metadata: {value!r}')
    elif isinstance(value, dict):
        for item in value.values():
            check_numbers(item)
    elif isinstance(value, list):
        for item in value:
            check_numbers(item)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    report = json.loads(a.report.read_text())
    check_numbers(report)
    checked_floats = 0
    checked_dispatches = 0
    max_duration_error = 0.0
    trace = a.report.with_suffix(a.report.suffix + '.trace.jsonl')
    phases = defaultdict(lambda: {'commands': 0, 'rows': 0, 'command_gpu_seconds': 0.0,
        'sampled_dispatch_gpu_seconds': 0.0, 'families': defaultdict(float), 'pipelines': defaultdict(lambda: {'calls': 0, 'gpu_seconds': 0.0}), 'metadata_complete': True})
    for line in trace.read_text().splitlines():
        x = json.loads(line)
        check_numbers(x)
        classifier = Classifier()
        value = phases[x['phase']]
        c = x['command']
        value['commands'] += 1
        value['rows'] += x['rows']
        value['command_gpu_seconds'] += c['gpu_seconds']
        value['metadata_complete'] &= not c['dispatch_metadata_truncated'] and len(c['dispatches']) == c['dispatch_count']
        for dispatch in c['dispatches']:
            family = classifier.family(dispatch)
            if not dispatch['timestamps_valid']:
                continue
            elapsed = dispatch['gpu_seconds']
            if elapsed <= 0 or elapsed > c['gpu_seconds'] * 1.1:
                raise ValueError('Invalid dispatch duration')
            duration_error = abs(elapsed - (dispatch['calibrated_end_seconds'] - dispatch['calibrated_start_seconds']))
            if duration_error > 1e-8:
                raise ValueError('Calibrated timestamp and duration disagree')
            max_duration_error = max(max_duration_error, duration_error)
            checked_dispatches += 1
            value['families'][family] += elapsed
            value['sampled_dispatch_gpu_seconds'] += elapsed
            item = value['pipelines'][dispatch['pipeline']]
            item['calls'] += 1
            item['gpu_seconds'] += elapsed
    for value in phases.values():
        value['pipelines'] = dict(sorted(value['pipelines'].items(), key=lambda x: -x[1]['gpu_seconds']))
    result = {'schema': 'splash-prefill4k-attribution-summary-v1',
        'diagnostic_counter_timings_only': report['profile_mode'] in ['stage', 'dispatch'],
        'classification_scope': 'Pipeline names plus gathered selection z geometry; head generic affine expert kernels count as routed MoE',
        'numeric_metadata_verified': True, 'checked_dispatches': checked_dispatches, 'max_calibrated_duration_error_seconds': max_duration_error,
        'source_report': str(a.report), 'prompt_tokens': report['prompt_tokens'], 'measured_trials': report['measured_trials'],
        'policy_routes': report['kernel_routes'], 'phases': phases,
        'normal_forward_call_prefill_tokens_per_second':
            report['prompt_tokens'] * report['measured_trials'] / (report['target_forward_call_seconds'] + report['prime_forward_call_seconds'])
            if report['profile_mode'] == 'normal' else None,
        'target_gpu_seconds': report['target_gpu_seconds'], 'prime_gpu_seconds': report['prime_gpu_seconds'],
        'target_forward_call_seconds': report['target_forward_call_seconds'], 'prime_forward_call_seconds': report['prime_forward_call_seconds'],
        'final_logits_sha256': report['final_logits_sha256'], 'first_tokens': report['first_tokens']}
    output = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.write_text(output)
    else:
        print(output)


if __name__ == '__main__':
    main()
