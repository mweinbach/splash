#!/usr/bin/env python3
"""Prepare bounded private adaptive-tail qualification; --run is root GPU only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/adaptive-expert-tail-sep21')
    parser.add_argument('--store', type=Path, default=ROOT / 'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1')
    parser.add_argument('--layer', type=int, choices=range(48), default=0)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--pattern', choices=('hit-concentrated', 'hit-spread', 'spread-all'), default='spread-all')
    parser.add_argument('--variant', type=int, choices=(1, 2), help='1: M16 tails; 2: M8 tails; default: both')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--ids', type=Path)
    policies = parser.add_mutually_exclusive_group()
    policies.add_argument('--normalized', action='store_true', help='Default: true row RMS 1 before BF16 rounding')
    policies.add_argument('--inherited', action='store_true', help='Frozen /512 BF16 synthetic fixture')
    policies.add_argument('--divisor74', action='store_true', help='Approximate /74 BF16 synthetic fixture')
    parser.add_argument('--strict', action='store_true', default=True, help='Mandatory complete BF16 and raw-F32 exactness')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Execute only during the exclusive root GPU window')
    args = parser.parse_args()
    if not 1024 <= args.rows <= 2048 or not 2 <= args.pairs <= 32 or args.pairs % 2:
        raise ValueError('Private oracle requires rows 1024..2048 and even pairs 2..32')
    if bool(args.input) != bool(args.ids):
        raise ValueError('Raw BF16 hidden and I64 IDs must be supplied together')
    if args.input and (args.normalized or args.inherited or args.divisor74):
        raise ValueError('Synthetic input policies cannot be applied to raw input')
    build, store, report = args.build.resolve(), args.store.resolve(), args.report.resolve()
    witness = Path(str(report) + '.invocation.json')
    if report.exists() or witness.exists():
        raise ValueError('Choose fresh report and witness paths')
    manifest_path = store / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('selected_experts') != [list(range(512)) for _ in range(48)]:
        raise ValueError('Private oracle requires canonical persisted Full512 inventory')
    subprocess.run([str(ROOT / '.venv/bin/python'), str(Path(__file__).with_name('check_source.py')),
                    str(build)], cwd=ROOT, check=True)
    source_manifest_path = build / 'shader-manifest.json'
    source_manifest = json.loads(source_manifest_path.read_text())
    policy = 'inherited512' if args.inherited else 'divisor74' if args.divisor74 else 'row-rms'
    controls = {
        'SPLASH_FLASH_MOE_Q4X8': '1', 'SPLASH_FLASH_MOE_DIRECT_A': '1',
        'FLASH_INT8_STORE_PAIRS': str(args.pairs), 'FLASH_INT8_STORE_PATTERN': args.pattern,
        'FLASH_INT8_STORE_MAX_RL2': '0.001', 'FLASH_INT8_STORE_MIN_COSINE': '0.999999',
        'PREFILL_MOE_NATIVE_M16_INPUT_POLICY': policy,
    }
    if args.variant:
        controls['ADAPTIVE_EXPERT_TAIL_SEP21_VARIANT'] = str(args.variant)
    if args.input:
        controls['FLASH_INT8_STORE_INPUT'] = str(args.input.resolve())
        controls['FLASH_INT8_STORE_ROUTE_IDS'] = str(args.ids.resolve())
    env = {key: value for key, value in os.environ.items() if not key.startswith(
        ('SPLASH_FLASH_', 'FLASH_INT8_STORE_', 'PREFILL4K_', 'PREFILL_MOE_', 'ADAPTIVE_EXPERT_TAIL_'))}
    env.update(controls)
    command = [str(build / 'oracle'), '--gpu', str(build / 'splash.metallib'), str(store),
               str(args.layer), str(args.rows), str(report)]
    files = {
        'binary': build / 'oracle', 'metallib': build / 'splash.metallib',
        'oracle_source': build / 'oracle.mm', 'private_shader': build / 'adaptive.metal',
        'private_shader_manifest': source_manifest_path,
        'oracle_generator': Path(__file__).with_name('generate_oracle.py'),
        'shader_generator': Path(__file__).with_name('generate_shader.py'),
        'source_checker': Path(__file__).with_name('check_source.py'),
        'runner': Path(__file__), 'makefile': Path(__file__).with_name('Makefile'),
        'native_m16_generator': ROOT / 'dev/benchmarks/prefill_moe_sep21/native_m16/generate.py',
        'native_m64_generator': ROOT / 'dev/benchmarks/prefill_moe_sep21/native_m64/generate.py',
        'inherited_oracle': ROOT / 'build/prefill4k-int8tiles/oracle.mm',
        'one_layer_loader': ROOT / 'dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp',
        'shipping_kernel': ROOT / 'runtime/metal/kernels/shared/flash_int8_expert_store.metal',
        'store_manifest': manifest_path,
    }
    provenance = {
        'gpu_execution_requested': args.run, 'command': command, 'controls': controls,
        'sha256': {name: sha(path) for name, path in files.items()},
        'private_shader_source_manifest': source_manifest,
        'one_readonly_layer_bytes': 2524446720, 'additional_scratch_reservation_bytes': 3 << 30,
        'host_reserve_minimum_bytes': 16 << 30, 'host_reserve_fraction': 0.1,
        'full_model_loaded': False, 'all_48_payloads_loaded': False,
        'production_store_constructor_used': False, 'model_quality_qualified': False,
        'strict_full_bf16_equality_required': True, 'raw_f32_bit_equality_and_finiteness_required': True,
        'scaled_bf16_bit_equality_and_finiteness_required': True,
        'copied_m32_probe_full_chain_qualification_against_native_m32_control_required': True,
        'private_pipeline_variants': ['m16-tail', 'm8-tail'], 'producer_threads': 128,
        'original_m32_jobs_params_grids_preserved': True, 'additional_dispatches': 0,
        'probe_dispatches_excluded_from_timing': True, 'minimum_gpu_warm_ms_per_variant_and_control': 100,
        'balanced_positions_and_alternating_pair_order': True, 'cpu_buffer_reads_inside_timed_loop': False,
        'synthetic_input_policy': None if args.input else policy,
        'normalized_target_row_rms': 1 if not args.input and policy == 'row-rms' else None,
        'bf16_rounding_follows_normalization': not args.input and policy == 'row-rms',
        'provenance_reads': 'code, binaries, metallib, source metadata and manifest only; no payload or raw fixture reads',
        'scope': 'complete original-M32-job expert chain over one certified shared Full512 layer',
    }
    if args.rows == 2048 and args.pattern == 'spread-all' and not args.input:
        provenance.update({'rows_per_expert': 40, 'active_m32_jobs': 1024,
                           'original_m32_job_capacity': 1151, 'original_padded_rows': 32768})
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'gpu_execution_requested': args.run, 'command': command, 'witness': str(witness)}))
    if args.run:
        subprocess.run(command, env=env, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
