#!/usr/bin/env python3
"""Source checks and private variant7-vs-adaptive qualification; root GPU only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
SCOPE = 'SG2 fixed K128 and Static=true for BOTH M32 and adaptive M16; per-K loop/order unchanged'
BASELINE_GATE = 'prefill_moe_sep21_memory_fixed_gate_up_m32_n64_k128_sg2'
BASELINE_DOWN = 'prefill_moe_sep21_memory_fixed_down_scatter_m32_n64_k128_sg2'
PREFIX = 'adaptive_expert_tail_sg2k128_sep21_'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_check(build):
    generator_path = Path(__file__).with_name('generate_shader.py')
    shader_path = build / 'adaptive.metal'
    shader = shader_path.read_text()
    manifest = json.loads((build / 'shader-manifest.json').read_text())
    source_path = Path(manifest['source_path'])
    baseline = source_path.read_text()
    require(manifest['source_sha256'] == sha(source_path), 'Frozen variant7 source drift')
    require(manifest['generator_source_sha256'] == sha(generator_path), 'Combined shader generator drift')
    require(manifest['entry_generator_sha256'] == sha(manifest['entry_generator_path']),
            'Inherited adaptive wrapper generator drift')
    require(manifest['shader_sha256'] == sha(shader_path), 'Combined shader drift')
    require(manifest['validator_byte_identical'], 'Original job validator policy changed')
    start = baseline.index('template <ushort M, ushort SG, ushort K = 0, bool Static = false>\ninline bool prefill_moe_sep21_memory_job(')
    end = baseline.index('template <ushort M, ushort SG, ushort K = 0, bool Static = false>\ninline void prefill_moe_sep21_memory_gate(', start)
    validator = baseline[start:end]
    require(validator in shader and manifest['original_validator_sha256'] ==
            hashlib.sha256(validator.encode()).hexdigest(), 'Original variant7 job validator differs')
    require(manifest['scope_policy'] == SCOPE, 'Shared SG2/static-K128 policy changed')
    pipelines = manifest['pipelines']
    expected = {PREFIX + kind + tile + probe for kind in ('gate_up_', 'down_scatter_')
                for tile in ('m32_control', 'm16_tail') for probe in ('', '_probe')}
    require(len(pipelines) == 8 and {p['name'] for p in pipelines} == expected,
            'Combined private pipeline inventory changed')
    require(all(p['threads'] == 64 and p['simdgroups'] == 2 and p['native_job_tile'] == 32
                and p['fixed_k'] == 128 and p['static_full_row_extent'] for p in pipelines),
            'Both control and candidate require SG2/K128 and original M32 jobs')
    require(shader.count('prefill_moe_sep21_memory_job<32, 2>(p, ranks, offsets, jobs, count, diag,') == 2,
            'Each producer must validate original M32/SG2 ownership once')
    require(BASELINE_GATE not in shader and BASELINE_DOWN not in shader,
            'Private shader must not redefine the existing variant7 baseline exports')
    require('constexpr ushort N = 64;' in shader and 'execution_simdgroups<SG>' in shader and
            shader.count('constexpr ushort SG = 2, K = 128;') == 2 and
            shader.count('constexpr bool Static = true;') == 2,
            'Shared N64/SG2/static-K128 descriptor changed')
    require('threadgroup bfloat' not in shader and 'scale_group_size' in validator,
            'Unexpected staging or changed persisted row-scale format')
    require('matmul2d_descriptor::mode::multiply_accumulate' in shader and
            'for (uint k = 0; k < 2560; k += K)' in shader and
            'for (uint k = 0; k < 640; k += K)' in shader,
            'Original fixed-K reduction loop inventory changed')
    require('prefill_moe_sep21_memory_job<16' not in shader and
            'prefill_moe_sep21_memory_job<8' not in shader and
            'execution_simdgroups<4>' not in shader,
            'Changed jobs or introduced unmatched SIMD-group count')
    oracle = (build / 'oracle.mm').read_text()
    require('d.threadsPerThreadgroup.x=64;' in oracle and BASELINE_GATE in oracle and
            BASELINE_DOWN in oracle and 'fixedCommands(graphs[0].dispatches(),rows)' in oracle and
            'fixedCommands(graphs[1].dispatches(),rows)' in oracle,
            'Both native scratch graphs must become variant7 controls before qualification')
    # Independent original-job ownership golden: the descriptor choice changes
    # no job begin/end, capacity, route count, expert offset or dispatch grid.
    for count in (0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 48, 64, 65, 2048):
        valid_rows = [min(32, count - begin) for begin in range(0, count, 32)]
        descriptors = [16 if valid <= 16 else 32 for valid in valid_rows]
        require(sum(valid_rows) == count and len(valid_rows) == (count + 31) // 32 and
                all(1 <= valid <= desc for valid, desc in zip(valid_rows, descriptors)),
                'Independent original M32 job/tail descriptor partition failed')
    require(512 * 2 == 1024 and 512 * (32 + 16) == 24576,
            'R2048 uniform40/expert golden changed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/adaptive-expert-tail-sg2k128-sep21')
    parser.add_argument('--source-check', action='store_true', help='CPU/code metadata only; no payload reads')
    parser.add_argument('--store', type=Path, default=ROOT / 'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1')
    parser.add_argument('--layer', type=int, choices=range(48), default=0)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--pattern', choices=('hit-concentrated', 'hit-spread', 'spread-all'), default='spread-all')
    parser.add_argument('--variant', type=int, choices=(1,), help='1: shared SG2/K128 control vs M16 tail')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--ids', type=Path)
    policies = parser.add_mutually_exclusive_group()
    policies.add_argument('--normalized', action='store_true', help='Default: true row RMS 1 before BF16 rounding')
    policies.add_argument('--inherited', action='store_true', help='Frozen /512 BF16 synthetic fixture')
    policies.add_argument('--divisor74', action='store_true', help='Approximate /74 BF16 synthetic fixture')
    parser.add_argument('--strict', action='store_true', default=True, help='Mandatory full BF16 and raw-F32 exactness')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--run', action='store_true', help='Exclusive root GPU qualification window only')
    args = parser.parse_args()
    build = args.build.resolve()
    source_manifest = source_check(build)
    if args.source_check:
        require(not args.run and not args.report, '--source-check cannot execute or write a GPU report')
        print(json.dumps({'source_checks': 'passed', 'gpu_executed': False,
                          'model_payload_bytes_read': 0, 'native_job_tile': 32,
                          'producer_threads': 64, 'producer_simdgroups': 2, 'fixed_k': 128,
                          'uniform_r2048_original_jobs': 1024,
                          'uniform_r2048_padded_rows': [32768, 24576],
                          'raw_f32_and_bf16_gpu_exactness': 'pending'}))
        return
    require(args.report is not None, '--report is required for an invocation witness')
    require(1024 <= args.rows <= 2048 and 2 <= args.pairs <= 32 and not args.pairs % 2,
            'Private oracle requires rows 1024..2048 and even pairs 2..32')
    require(bool(args.input) == bool(args.ids), 'Raw BF16 hidden and I64 IDs must be supplied together')
    require(not args.input or not (args.normalized or args.inherited or args.divisor74),
            'Synthetic input policies cannot be applied to raw input')
    store, report = args.store.resolve(), args.report.resolve()
    witness = Path(str(report) + '.invocation.json')
    require(not report.exists() and not witness.exists(), 'Choose fresh report and witness paths')
    manifest_path = store / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('selected_experts') == [list(range(512)) for _ in range(48)],
            'Private oracle requires canonical persisted Full512 inventory')
    policy = 'inherited512' if args.inherited else 'divisor74' if args.divisor74 else 'row-rms'
    controls = {
        'SPLASH_FLASH_MOE_Q4X8': '1', 'SPLASH_FLASH_MOE_DIRECT_A': '1',
        'FLASH_INT8_STORE_PAIRS': str(args.pairs), 'FLASH_INT8_STORE_PATTERN': args.pattern,
        'FLASH_INT8_STORE_MAX_RL2': '0.001', 'FLASH_INT8_STORE_MIN_COSINE': '0.999999',
        'PREFILL_MOE_NATIVE_M16_INPUT_POLICY': policy,
    }
    if args.variant:
        controls['ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21_VARIANT'] = str(args.variant)
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
        'private_shader_manifest': build / 'shader-manifest.json',
        'oracle_generator': Path(__file__).with_name('generate_oracle.py'),
        'shader_generator': Path(__file__).with_name('generate_shader.py'),
        'runner_and_source_checker': Path(__file__), 'makefile': Path(__file__).with_name('Makefile'),
        'parent_oracle_generator': Path(__file__).resolve().parents[1] / 'generate_oracle.py',
        'native_m16_generator': ROOT / 'dev/benchmarks/prefill_moe_sep21/native_m16/generate.py',
        'native_m64_generator': ROOT / 'dev/benchmarks/prefill_moe_sep21/native_m64/generate.py',
        'inherited_oracle': ROOT / 'build/prefill4k-int8tiles/oracle.mm',
        'one_layer_loader': ROOT / 'dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp',
        'shipping_kernel': ROOT / 'runtime/metal/kernels/shared/flash_int8_expert_store.metal',
        'existing_variant7_shader': Path(source_manifest['source_path']),
        'existing_variant7_air': build / 'variant7.air', 'store_manifest': manifest_path,
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
        'copied_m32_probe_full_chain_qualification_against_existing_variant7_required': True,
        'private_pipeline_variants': ['sg2-k128-m16-tail'], 'producer_threads': 64,
        'producer_simdgroups': 2, 'fixed_k': 128, 'static_full_tiles': True,
        'control_pipelines': [BASELINE_GATE, BASELINE_DOWN],
        'original_m32_jobs_params_grids_preserved': True, 'additional_dispatches': 0,
        'probe_dispatches_excluded_from_timing': True, 'minimum_gpu_warm_ms_per_variant_and_control': 100,
        'balanced_positions_and_alternating_pair_order': True, 'cpu_buffer_reads_inside_timed_loop': False,
        'synthetic_input_policy': None if args.input else policy,
        'normalized_target_row_rms': 1 if not args.input and policy == 'row-rms' else None,
        'bf16_rounding_follows_normalization': not args.input and policy == 'row-rms',
        'provenance_reads': 'code, binaries, metallib, source metadata and manifest only; no payload or raw fixture reads',
        'scope': 'existing variant7 SG2/static-K128 vs adaptive M16 within original M32 jobs; one certified shared Full512 layer',
    }
    if args.rows == 2048 and args.pattern == 'spread-all' and not args.input:
        provenance.update({'rows_per_expert': 40, 'active_m32_jobs': 1024,
                           'original_m32_job_capacity': 1151, 'original_padded_rows': 32768,
                           'adaptive_m16_padded_rows': 24576})
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'gpu_execution_requested': args.run, 'command': command, 'witness': str(witness)}))
    if args.run:
        subprocess.run(command, env=env, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
