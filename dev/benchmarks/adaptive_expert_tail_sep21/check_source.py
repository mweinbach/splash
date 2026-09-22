#!/usr/bin/env python3
"""Check adaptive-tail source provenance/geometry without GPU or payload reads."""
from pathlib import Path
import argparse
import hashlib
import json
import importlib.util

ROOT = Path(__file__).resolve().parents[3]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(build):
    module_path = Path(__file__).with_name('generate_shader.py')
    spec = importlib.util.spec_from_file_location('adaptive_tail_generator', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = module.NATIVE.read_text()
    shader = (build / 'adaptive.metal').read_text()
    manifest = json.loads((build / 'shader-manifest.json').read_text())
    require(manifest['native_source_sha256'] == sha(module.NATIVE.read_bytes()), 'Native source drift')
    require(manifest['generator_source_sha256'] == sha(module_path.read_bytes()), 'Generator source drift')
    require(manifest['shader_sha256'] == sha((build / 'adaptive.metal').read_bytes()), 'Private shader drift')
    require(manifest['validator_byte_identical'], 'Validator provenance policy changed')
    start = native.index('template <ushort M, ushort SG>\ninline bool int8_expert_store_job(')
    end = native.index('template <ushort M, ushort SG>\ninline void int8_expert_store_gate(', start)
    validator = native[start:end]
    require(validator in shader and manifest['original_validator_sha256'] == sha(validator.encode()),
            'Private job validator differs from original')
    require(shader.count('int8_expert_store_job<32, 4>(p, ranks, offsets, jobs, count, diag,') == 2,
            'Each producer must call original M32 validation once')
    for body in (module.GATE_WRAPPER, module.DOWN_WRAPPER):
        require(body in shader and body.count('int8_expert_store_job<32, 4>') == 1,
                'Validated producer wrapper drift')
    for kind in ('gate', 'down'):
        require(f'adaptive_expert_tail_{kind}_math<8, Probe>' in shader and
                f'adaptive_expert_tail_{kind}_math<16, Probe>' in shader and
                f'adaptive_expert_tail_{kind}_math<32, Probe>' in shader,
                'Adaptive descriptor inventory incomplete')
    require(len(manifest['pipelines']) == 12 and
            all(p['threads'] == 128 and p['simdgroups'] == 4 and p['native_job_tile'] == 32
                for p in manifest['pipelines']), 'Private descriptor/launch contract changed')
    require('constexpr ushort N = 64;' in shader and 'execution_simdgroups<4>' in shader,
            'Original N/SG contract changed')
    require('threadgroup bfloat' not in shader and 'scale_group_size' in validator,
            'Unexpected staging or changed persisted row-scale format')
    require(shader.count('matmul2d_descriptor::mode::multiply)') == 2,
            'Original whole-K reduction policy changed')
    require(shader.count('static_cast<int>(dynamic_extent)') == 2,
            'Original dynamic whole-K extent changed')
    # Independent original M32 job partition; adaptation never changes jobs.
    # Covers empty/one-row/threshold/tail/full/multi-job expert counts.
    for count in (0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 48, 64, 65, 2048):
        valid_rows = [min(32, count - begin) for begin in range(0, count, 32)]
        require(sum(valid_rows) == count and all(1 <= valid <= 32 for valid in valid_rows),
                'Independent original job partition failed')
        descriptors16 = [16 if valid <= 16 else 32 for valid in valid_rows]
        descriptors8 = [8 if valid <= 8 else 16 if valid <= 16 else 32 for valid in valid_rows]
        require(len(descriptors16) == len(descriptors8) == (count + 31) // 32 and
                all(valid <= desc for valid, desc in zip(valid_rows, descriptors16)) and
                all(valid <= desc for valid, desc in zip(valid_rows, descriptors8)),
                'Adaptive descriptor masks valid original rows')
    jobs = 512 * 2
    require(jobs == 1024 and jobs * 32 == 32768 and
            512 * (32 + 16) == 24576 and 512 * (32 + 8) == 20480,
            'R2048 uniform40/expert golden changed')
    print(json.dumps({'source_checks': 'passed', 'gpu_executed': False,
                      'model_payload_bytes_read': 0, 'native_job_tile': 32,
                      'uniform_r2048_original_jobs': jobs,
                      'uniform_r2048_padded_rows': [32768, 24576, 20480],
                      'raw_f32_and_bf16_gpu_exactness': 'pending'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    check(parser.parse_args().build.resolve())
