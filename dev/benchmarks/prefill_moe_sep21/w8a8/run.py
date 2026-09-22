#!/usr/bin/env python3
"""Prepare a bounded W8A8 numerical-alternative screen; --run opts into GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[4]
VARIANTS = ('m32_n64_sg4', 'm32_n64_sg2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workspace_memory(rows):
    """Derive Workspace::sizes/plannedBytes from the agreed guarded extents."""
    routes = rows * 10
    padded = routes + 63
    sizes = (padded * 2560, padded * 640, padded * 4, padded * 4,
             routes * 640 * 4, routes * 640 * 4, routes * 2560 * 4,
             routes * 640 * 4, routes * 640 * 4, routes * 2560 * 4)
    allocated = [((size + 64 + 16384 - 1) & ~(16384 - 1)) for size in sizes]
    return {'logical_allocation_bytes': list(sizes), 'guarded_allocation_bytes': allocated,
            'logical_bytes': sum(sizes), 'planned_bytes': sum(allocated),
            'tail_guard_bytes_per_allocation': 64, 'allocation_alignment': 16384,
            'derivation': 'Workspace::sizes/plannedBytes in cache.hpp; ten guarded allocations at this row extent'}


def measured_rms_fields(value, prefix=''):
    """Copy measured RMS statistics from report metadata, without reading fixtures."""
    fields = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f'{prefix}.{key}' if prefix else key
            if 'rms' in key.lower() and isinstance(child, (float, int)) and not isinstance(child, bool):
                fields[path] = child
            fields.update(measured_rms_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.update(measured_rms_fields(child, f'{prefix}[{index}]'))
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/prefill-moe-sep21-w8a8')
    parser.add_argument('--store', type=Path, default=ROOT / 'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1')
    parser.add_argument('--layer', type=int, choices=range(48), default=0)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--pattern', choices=('hit-concentrated', 'hit-spread', 'spread-all'), default='spread-all')
    parser.add_argument('--variant', type=int, choices=(1, 2), help='Screen one candidate; omit for both')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--ids', type=Path)
    parser.add_argument('--strict', action='store_true', help='Require byte-identical complete BF16 chain before timing')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Execute only in the exclusive root GPU window')
    args = parser.parse_args()
    if not 1024 <= args.rows <= 2048 or not 1 <= args.pairs <= 32 or bool(args.input) != bool(args.ids):
        raise ValueError('W8A8 oracle requires R1024..2048, pairs1..32, paired raw input/IDs')

    build = args.build.resolve()
    store = args.store.resolve()
    report = args.report.resolve()
    witness = Path(str(report) + '.invocation.json')
    if report.exists() or witness.exists():
        raise ValueError('Choose fresh W8A8 report and witness paths')
    manifest_path = store / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('selected_experts') != [list(range(512)) for _ in range(48)]:
        raise ValueError('W8A8 oracle requires canonical persisted Full512 inventory')

    controls = {
        'SPLASH_FLASH_MOE_Q4X8': '1',
        'SPLASH_FLASH_MOE_DIRECT_A': '1',
        'SPLASH_FLASH_MOE_M64': '1',
        'FLASH_INT8_STORE_PAIRS': str(args.pairs),
        'FLASH_INT8_STORE_PATTERN': args.pattern,
        'FLASH_INT8_STORE_MAX_RL2': '0.05',
        'FLASH_INT8_STORE_MIN_COSINE': '0.9985',
    }
    if args.variant:
        controls['PREFILL_MOE_W8A8_VARIANT'] = str(args.variant)
    if args.strict:
        controls['PREFILL_MOE_W8A8_STRICT'] = '1'
    if args.input:
        controls['FLASH_INT8_STORE_INPUT'] = str(args.input.resolve())
        controls['FLASH_INT8_STORE_ROUTE_IDS'] = str(args.ids.resolve())
    env = {key: value for key, value in os.environ.items() if not key.startswith(
        ('SPLASH_FLASH_', 'FLASH_INT8_STORE_', 'PREFILL4K_', 'PREFILL_MOE_'))}
    env.update(controls)
    command = [str(build / 'oracle'), '--gpu', str(build / 'splash.metallib'),
               str(store), str(args.layer), str(args.rows), str(report)]
    selected = [args.variant] if args.variant else [1, 2]
    routes = args.rows * 10
    memory = workspace_memory(args.rows)
    provenance = {
        'gpu_execution_requested': args.run,
        'command': command,
        'controls': controls,
        'binary_sha256': sha(build / 'oracle'),
        'metallib_sha256': sha(build / 'splash.metallib'),
        'oracle_source_sha256': sha(build / 'oracle.mm'),
        'candidate_source_sha256': sha(build / 'candidate.metal'),
        'generator_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/w8a8/generate.py'),
        'cache_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/w8a8/cache.hpp'),
        'precision_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/w8a8/precision.hpp'),
        'kernel_generator_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/w8a8/generate_kernel.py'),
        'kernel_template_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/memory.metal'),
        'quantizer_abi_source_sha256': sha(build / 'candidate.metal'),
        'expert_store_abi_source_sha256': sha(ROOT / 'runtime/metal/abi/FlashInt8ExpertStore.h'),
        'bucket_abi_source_sha256': sha(ROOT / 'runtime/metal/abi/FlashMoEBuckets.h'),
        'runner_source_sha256': sha(Path(__file__)),
        'makefile_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/w8a8/Makefile'),
        'readme_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/w8a8/README.md'),
        'inherited_baseline_source_sha256': sha(ROOT / 'build/prefill4k-int8tiles/oracle.mm'),
        'native_adapter_source_sha256': sha(ROOT / 'dev/benchmarks/prefill_moe_sep21/native_m64/generate.py'),
        'one_layer_loader_source_sha256': sha(ROOT / 'dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp'),
        'store_manifest_sha256': sha(manifest_path),
        'stored_experts_per_layer': 512,
        'selected_variants': selected,
        'variants': [{'variant': index, 'candidate_suffix': VARIANTS[index - 1],
                      'native_job_tile': 32, 'simdgroups': 4 if index == 1 else 2,
                      'whole_dynamic_k': True} for index in selected],
        'one_readonly_i8_layer_bytes': 2524446720,
        'one_layer_rank_allocation_bytes': 16384,
        'baseline_scratch_reservation_bytes': 1 << 30,
        'quant_and_audit_admission_required_before_allocation': True,
        'quant_and_audit_memory_plan': memory,
        'total_bounded_planned_bytes': 2524446720 + 16384 + (1 << 30) + memory['planned_bytes'],
        'routes': routes,
        'quantized_a_rows_including_pad': routes + 63,
        'quantized_a_gate_shape': [routes + 63, 2560],
        'quantized_a_down_shape': [routes + 63, 640],
        'f32_row_scale_lengths': [routes + 63, routes + 63],
        'scaled_f32_audit_shapes': {'gate': [routes, 640], 'up': [routes, 640], 'down_canonical': [routes, 2560]},
        'raw_i32_audit_shapes': {'gate': [routes, 640], 'up': [routes, 640], 'down_canonical': [routes, 2560]},
        'host_reserve_minimum_bytes': 16 << 30,
        'host_reserve_fraction': 0.1,
        'full_model_loaded': False,
        'all_48_payloads_loaded': False,
        'production_store_constructor_used': False,
        'large_converted_coefficient_cache_used': False,
        'persisted_i8_b_and_f32_row_scales_unchanged': True,
        'synthetic_fixture_normalization': None if args.input else 'true per-row RMS normalization with final BF16 rounding',
        'true_row_rms_normalized_fixture': not bool(args.input),
        'input_row_rms_min_and_max_required_before_timing': True,
        'input_row_rms_report_fields': ['hidden_row_rms_min', 'hidden_row_rms_max'],
        'measured_input_rms_fields': None,
        'quantizer': 'per-row maxabs BF16 A -> signed I8; round-to-nearest-even; clamp [-127,127]; F32 scale=maxabs/127',
        'zero_row_scale': 1,
        'nonfinite_input_sanitized_value': 0,
        'nonfinite_diagnostic_bit': 4,
        'quantizer_pipelines': ['prefill_moe_sep21_w8a8_quantize_gate_t256', 'prefill_moe_sep21_w8a8_quantize_down_t128'],
        'additional_gpu_quantization_commands': 2,
        'gpu_quantization_included_in_complete_chain_timing': True,
        'native_m32_jobs_and_parameters_unchanged': True,
        'additional_hit_list_dispatches': 0,
        'independent_cpu_every_quantized_i8_and_scale_check_required_before_timing': True,
        'sampled_exact_raw_i32_cpu_dot_check_required_before_timing': True,
        'sampled_scaled_f32_cpu_dot_check_required_before_timing': True,
        'sampled_f64_original_bf16_a_i8_b_row_scale_quantization_envelope_required_before_timing': True,
        'audit_kernels_excluded_from_complete_chain_timing': True,
        'full_bf16_error_and_equality_measured_before_timing': True,
        'strict_full_bf16_equality_required': args.strict,
        'preregistered_max_relative_l2': 0.05,
        'preregistered_min_cosine': 0.9985,
        'numerical_alternative': True,
        'rejected_candidates_reported_without_timing': True,
        'semantic_quality_qualified': False,
        'mtp_acceptance_qualified': False,
        'model_quality_qualified': False,
        'provenance_reads': 'code, binary/metallib hashes and store manifest only; no input, ID or model payload reads',
        'scope': 'bounded W8A8 numerical alternative; complete native M32 expert chains including both GPU A quantizers over one shared immutable I8 layer and fixtures',
    }
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'gpu_execution_requested': args.run, 'command': command, 'witness': str(witness)}))
    if args.run:
        result = subprocess.run(command, env=env, check=False, cwd=ROOT)
        if report.exists():
            provenance['measured_input_rms_fields'] = measured_rms_fields(json.loads(report.read_text()))
            witness.write_text(json.dumps(provenance, indent=2) + '\n')
        result.check_returncode()


if __name__ == '__main__':
    main()
