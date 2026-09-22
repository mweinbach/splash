#!/usr/bin/env python3
"""Prepare a bounded one-layer three-variant right-only M32 screen; --run opts into GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/prefill-moe-sep21-right-only')
    parser.add_argument('--store', type=Path, default=ROOT / 'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1')
    parser.add_argument('--layer', type=int, choices=range(48), default=0)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--pattern', choices=('hit-concentrated', 'hit-spread', 'spread-all'), default='spread-all')
    parser.add_argument('--variant', type=int, choices=range(1, 4), help='Screen one candidate; omit for all three')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--ids', type=Path)
    parser.add_argument('--normalized', action='store_true', help='Use synthetic hidden values divided by 74 instead of the inherited 512')
    parser.add_argument('--strict', action='store_true', help='Require byte-identical complete BF16 chain before timing')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Execute only in the exclusive root GPU window')
    args = parser.parse_args()
    if not 1024 <= args.rows <= 2048 or not 1 <= args.pairs <= 32 or bool(args.input) != bool(args.ids):
        raise ValueError('One-layer oracle requires R1024..2048, pairs1..32, paired raw input/IDs')
    if args.normalized and args.input:
        raise ValueError('--normalized applies only to synthetic input')

    build = args.build.resolve()
    store = args.store.resolve()
    report = args.report.resolve()
    witness = Path(str(report) + '.invocation.json')
    if report.exists() or witness.exists():
        raise ValueError('Choose fresh one-layer report and witness paths')
    manifest_path = store / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('selected_experts') != [list(range(512)) for _ in range(48)]:
        raise ValueError('One-layer oracle requires canonical persisted Full512 inventory')

    controls = {
        'SPLASH_FLASH_MOE_Q4X8': '1',
        'SPLASH_FLASH_MOE_DIRECT_A': '1',
        'SPLASH_FLASH_MOE_M64': '1',
        'FLASH_INT8_STORE_PAIRS': str(args.pairs),
        'FLASH_INT8_STORE_PATTERN': args.pattern,
        'FLASH_INT8_STORE_MAX_RL2': '0.001',
        'FLASH_INT8_STORE_MIN_COSINE': '0.999999',
    }
    if args.variant:
        controls['PREFILL_MOE_SEP21_VARIANT'] = str(args.variant)
    if args.strict:
        controls['PREFILL_MOE_SEP21_STRICT'] = '1'
    if args.normalized:
        controls['PREFILL_MOE_SEP21_NORMALIZED'] = '1'
    if args.input:
        controls['FLASH_INT8_STORE_INPUT'] = str(args.input.resolve())
        controls['FLASH_INT8_STORE_ROUTE_IDS'] = str(args.ids.resolve())
    env = {key: value for key, value in os.environ.items() if not key.startswith(
        ('SPLASH_FLASH_', 'FLASH_INT8_STORE_', 'PREFILL4K_', 'PREFILL_MOE_'))}
    env.update(controls)
    command = [str(build / 'oracle'), '--gpu', str(build / 'splash.metallib'),
               str(store), str(args.layer), str(args.rows), str(report)]
    provenance = {
        'gpu_execution_requested': args.run,
        'command': command,
        'controls': controls,
        'binary_sha256': sha(build / 'oracle'),
        'metallib_sha256': sha(build / 'splash.metallib'),
        'oracle_source_sha256': sha(build / 'oracle.mm'),
        'generator_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/right_only/generate.py'),
        'runner_source_sha256': sha(Path(__file__)),
        'makefile_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/right_only/Makefile'),
        'inherited_baseline_source_sha256': sha(ROOT / 'build/prefill4k-int8tiles/oracle.mm'),
        'inherited_one_layer_adapter_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/one_layer/generate.py'),
        'native_adapter_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/native_m64/generate.py'),
        'one_layer_loader_source_sha256': sha(ROOT / 'dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp'),
        'variant_adapter_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/bridge.hpp'),
        'precision_source_sha256': sha(ROOT / 'dev/benchmarks/prefill4k_int8columns/precision.hpp'),
        'right_only_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/right_only.metal'),
        'store_manifest_sha256': sha(manifest_path),
        'stored_experts_per_layer': 512,
        'selected_variants': [args.variant] if args.variant else list(range(1, 4)),
        'one_readonly_layer_bytes': 2524446720,
        'additional_scratch_reservation_bytes': 1 << 30,
        'host_reserve_minimum_bytes': 16 << 30,
        'host_reserve_fraction': 0.1,
        'full_model_loaded': False,
        'all_48_payloads_loaded': False,
        'production_store_constructor_used': False,
        'native_job_tile': 32,
        'native_m32_jobs_and_parameters_unchanged': True,
        'additional_hit_list_dispatches': 0,
        'additional_prefix_dispatches': 0,
        'private_gathered_shaders_used': False,
        'persisted_i8_and_f32_scales_unchanged': True,
        'i8_to_bf16_register_conversion_exact': True,
        'left_operand': 'unchanged device BF16 A with bounded logical K and valid-row extents',
        'right_operand': 'cooperative BF16 B registers filled losslessly from unchanged signed-I8 codes',
        'mpp_destination': 'F32 multiply-accumulate initialized to zero',
        'logical_job_shape': 'M32N64; sequential N32 column halves; one SIMD group',
        'variants': {
            '1': {'descriptor_k': 64, 'gate_blocks': 40, 'down_blocks': 10},
            '2': {'descriptor_k': 128, 'gate_blocks': 20, 'down_blocks': 5},
            '3': {'descriptor_k': 256, 'gate_blocks': 10, 'down_blocks': 3,
                  'down_tail': 'K128 device-A extent; cooperative B zero-masked for k>=640'},
        },
        'register_register_k32_restriction_applies': False,
        'original_f32_post_dot_scales_and_bf16_swiglu_boundaries_preserved': True,
        'private_airs': ['build/prefill-moe-sep21/right_only.air'],
        'private_memory_register_and_gathered_airs_used': False,
        'synthetic_input_divisor': None if args.input else (74 if args.normalized else 512),
        'full_bf16_error_and_equality_measured_before_timing': True,
        'strict_full_bf16_equality_required': args.strict,
        'max_relative_l2': 0.001,
        'min_cosine': 0.999999,
        'rejected_candidates_reported_without_timing': True,
        'numerical_alternatives_may_time_after_guard_pass': not args.strict,
        'model_quality_qualified': False,
        'provenance_reads': 'code, binary/metallib hashes and store manifest only; no input, ID or model payload reads',
        'adapter_boundary': 'original Full512 native M32 jobs over the certified one-layer loader; production 48-layer constructor excluded',
        'scope': 'matched complete expert chain; device-A/cooperative-B real-K64/K128/K256 candidates; one shared immutable 2,524,446,720-byte layer and fixtures; warm rotating paired GPU command timings',
    }
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'gpu_execution_requested': args.run, 'command': command, 'witness': str(witness)}))
    if args.run:
        subprocess.run(command, env=env, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
