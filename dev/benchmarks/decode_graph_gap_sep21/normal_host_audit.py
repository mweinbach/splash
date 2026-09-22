#!/usr/bin/env python3
"""CPU-only warm-wave latency accounting from saved exact benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def analyze(path):
    data = json.loads(path.read_text())
    groups = {}
    for wave in data['waves']:
        if wave['warmup'] or not wave['valid']:
            continue
        before = wave['status_before']['model_timing']['decode']
        after = wave['status_after']['model_timing']['decode']
        h0, h1 = before['host_command_subphases'], after['host_command_subphases']
        gpu = after['total_gpu_ms'] - before['total_gpu_ms']
        forward = after['forward_host_wall_ms'] - before['forward_host_wall_ms']
        client = wave['client_first_content_to_last_done_wave_seconds'] * 1000
        tokens = wave['native_post_first_emission_tokens_sum']
        prep = h1['preparation_ms'] - h0['preparation_ms']
        encode = h1['encoding_ms'] - h0['encoding_ms']
        sample = {'trial': wave['trial'], 'emitted_post_first_tokens': tokens,
                  'decode_gpu_ms': gpu, 'forward_host_wall_ms': forward,
                  'client_decode_ms': client, 'submission_preparation_ms': prep,
                  'submission_encoding_ms': encode,
                  'all_non_gpu_removed_conditional_tokens_per_second': tokens * 1000 / gpu,
                  'preparation_encoding_removed_conditional_tokens_per_second': tokens * 1000 / (client - prep - encode)}
        groups.setdefault((wave['mtp_setting'], wave['http_width']), []).append(sample)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'groups': [{'mtp_setting': key[0], 'http_width': key[1], 'warm_samples': samples,
                        'medians': {k: median(s[k] for s in samples)
                                    for k in samples[0] if k != 'trial'}}
                       for key, samples in groups.items()]}


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--output', required=True)
    args = args.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit('fresh output required')
    root = Path(__file__).resolve().parents[3]
    paths = [root / 'build/release/flash/sep21-gathered-standard-mtp3-batch-matrix-v1.json',
             root / 'build/release/flash/sep21-pointwise-model-coding-v1.json']
    result = {'schema': 'splash-sep21-normal-decode-host-audit-v1',
              'gpu_executed': False,
              'scope': 'saved warm exact coding 2048/256 waves; aggregate exact post-first emissions',
              'limitations': ['Conditional removal accounts are not achieved rates.',
                              'Normal preparation/encoding durations are distinct from graph-construction time.',
                              'All-non-GPU removal assumes unchanged GPU math and acceptance; no attainable-peak claim.',
                              'Per-dispatch stage verifier profiling changes encoder boundaries and cannot establish the normal host gap.'],
              'reports': [analyze(p) for p in paths]}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'pass': True, 'gpu_executed': False, 'output': str(output)}))


if __name__ == '__main__':
    main()
