#!/usr/bin/env python3
"""Prepare a one-layer exact unscaled BF16 integer-code screen; --run opts into GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[4]
VARIANTS = (
    ('m32_n64_sg4', 32, 4, 0),
    ('m32_n64_sg2', 32, 2, 0),
    ('m64_n64_sg8', 64, 8, 0),
    ('m32_n64_k128_sg2', 32, 2, 128),
)
ALIGNMENT = 16384
TAIL_GUARD_BYTES = 64
PLANE_LOGICAL_BYTES = 512 * 2560 * 640 * 2
PLANE_ALLOCATED_BYTES = (PLANE_LOGICAL_BYTES + TAIL_GUARD_BYTES + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
CACHE_LOGICAL_BYTES = 3 * PLANE_LOGICAL_BYTES
CACHE_ALLOCATED_BYTES = 3 * PLANE_ALLOCATED_BYTES
LAYER_BYTES = 2524446720
SCRATCH_RESERVATION_BYTES = 1 << 30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/prefill-moe-sep21-bf16-codes')
    parser.add_argument('--store', type=Path, default=ROOT / 'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1')
    parser.add_argument('--layer', type=int, choices=range(48), default=0)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--pattern', choices=('hit-concentrated', 'hit-spread', 'spread-all'), default='spread-all')
    parser.add_argument('--variant', type=int, choices=range(1, 5), help='Screen one candidate; omit for all four')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--ids', type=Path)
    parser.add_argument('--normalized', action='store_true', help='Use synthetic hidden values divided by 74 instead of the inherited 512')
    parser.add_argument('--strict', action='store_true', help='Require byte-identical complete BF16 chain before timing')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Execute only in the exclusive root GPU window')
    args = parser.parse_args()
    if not 1024 <= args.rows <= 2048 or not 1 <= args.pairs <= 32 or bool(args.input) != bool(args.ids):
        raise ValueError('BF16-code oracle requires R1024..2048, pairs1..32, paired raw input/IDs')
    if args.normalized and args.input:
        raise ValueError('--normalized applies only to synthetic input')

    build = args.build.resolve()
    store = args.store.resolve()
    report = args.report.resolve()
    witness = Path(str(report) + '.invocation.json')
    if report.exists() or witness.exists():
        raise ValueError('Choose fresh BF16-code report and witness paths')
    manifest_path = store / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('selected_experts') != [list(range(512)) for _ in range(48)]:
        raise ValueError('BF16-code oracle requires canonical persisted Full512 inventory')

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
        controls['PREFILL_MOE_BF16_CODES_VARIANT'] = str(args.variant)
    if args.strict:
        controls['PREFILL_MOE_BF16_CODES_STRICT'] = '1'
    if args.normalized:
        controls['PREFILL_MOE_BF16_CODES_NORMALIZED'] = '1'
    if args.input:
        controls['FLASH_INT8_STORE_INPUT'] = str(args.input.resolve())
        controls['FLASH_INT8_STORE_ROUTE_IDS'] = str(args.ids.resolve())
    env = {key: value for key, value in os.environ.items() if not key.startswith(
        ('SPLASH_FLASH_', 'FLASH_INT8_STORE_', 'PREFILL4K_', 'PREFILL_MOE_'))}
    env.update(controls)
    command = [str(build / 'oracle'), '--gpu', str(build / 'splash.metallib'),
               str(store), str(args.layer), str(args.rows), str(report)]
    selected = [args.variant] if args.variant else list(range(1, 5))
    provenance = {
        'gpu_execution_requested': args.run,
        'command': command,
        'controls': controls,
        'binary_sha256': sha(build / 'oracle'),
        'metallib_sha256': sha(build / 'splash.metallib'),
        'oracle_source_sha256': sha(build / 'oracle.mm'),
        'candidate_source_sha256': sha(build / 'candidate.metal'),
        'generator_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/bf16_codes/generate.py'),
        'kernel_generator_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/bf16_codes/generate_kernel.py'),
        'cache_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/bf16_codes/cache.hpp'),
        'kernel_template_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/memory.metal'),
        'runner_source_sha256': sha(Path(__file__)),
        'makefile_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/bf16_codes/Makefile'),
        'inherited_baseline_source_sha256': sha(ROOT / 'build/prefill4k-int8tiles/oracle.mm'),
        'native_adapter_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/native_m64/generate.py'),
        'one_layer_loader_source_sha256': sha(ROOT / 'dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp'),
        'store_manifest_sha256': sha(manifest_path),
        'stored_experts_per_layer': 512,
        'selected_variants': selected,
        'variants': [
            {'variant': index, 'candidate_suffix': VARIANTS[index - 1][0],
             'native_job_tile': VARIANTS[index - 1][1], 'simdgroups': VARIANTS[index - 1][2],
             'fixed_reduction_k': VARIANTS[index - 1][3]}
            for index in selected
        ],
        'one_readonly_i8_layer_bytes': LAYER_BYTES,
        'one_layer_rank_allocation_bytes': ALIGNMENT,
        'bf16_code_cache_planes': 3,
        'bf16_code_cache_logical_bytes': CACHE_LOGICAL_BYTES,
        'bf16_code_cache_allocated_bytes': CACHE_ALLOCATED_BYTES,
        'bf16_code_cache_tail_guard_bytes_per_plane': TAIL_GUARD_BYTES,
        'bf16_code_cache_allocation_alignment': ALIGNMENT,
        'cache_admission_required_before_allocation': True,
        'additional_scratch_reservation_bytes': SCRATCH_RESERVATION_BYTES,
        'total_bounded_planned_bytes': LAYER_BYTES + ALIGNMENT + SCRATCH_RESERVATION_BYTES + CACHE_ALLOCATED_BYTES,
        'host_reserve_minimum_bytes': 16 << 30,
        'host_reserve_fraction': 0.1,
        'full_model_loaded': False,
        'all_48_payloads_loaded': False,
        'production_store_constructor_used': False,
        'code_cache_semantics': 'exact unscaled signed-I8 integers in BF16; source F32 row-scale views retained',
        'conversion_lookup_words': 256,
        'code_cache_every_word_independent_verification_required_before_and_after_timing': True,
        'conversion_and_verification_excluded_from_complete_chain_timing': True,
        'native_jobs_match_corresponding_i8_control': True,
        'additional_hit_list_dispatches': 0,
        'additional_prefix_dispatches': 0,
        'memory_or_register_air_dependencies': False,
        'persisted_i8_and_f32_scales_unchanged': True,
        'source_scales_incorporated_into_cached_codes': False,
        'synthetic_input_divisor': None if args.input else (74 if args.normalized else 512),
        'full_bf16_error_and_equality_measured_before_timing': True,
        'strict_full_bf16_equality_required': args.strict,
        'max_relative_l2': 0.001,
        'min_cosine': 0.999999,
        'rejected_candidates_reported_without_timing': True,
        'model_quality_qualified': False,
        'full_worker_hot64_cache_experiment_requires_complete_chain_speedup_over': 1.25,
        'full_worker_hot64_cache_experiment_qualified': False,
        'provenance_reads': 'code, binary/metallib hashes and store manifest only; no input, ID or model payload reads',
        'scope': 'bounded exact unscaled BF16-code primitive; matched complete native M32/M64 expert chains over one shared immutable layer and fixtures',
    }
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'gpu_execution_requested': args.run, 'command': command, 'witness': str(witness)}))
    if args.run:
        subprocess.run(command, env=env, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
