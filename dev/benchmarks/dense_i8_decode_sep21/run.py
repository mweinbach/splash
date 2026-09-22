#!/usr/bin/env python3
"""Prepare a saved-fixture dense I8 decode primitive; --run opts into GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'dev/benchmarks/dense_i8_decode_sep21'
VARIANTS = (
    ('m8_n64_sg4', 8, 64, 4),
    ('m8_n128_sg4', 8, 128, 4),
    ('m16_n64_sg4', 16, 64, 4),
    ('m8_n64_sg2', 8, 64, 2),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_rows(values):
    rows = []
    for value in values:
        for token in value.split(','):
            if token not in ('1', '4', '8', '16'):
                raise ValueError('Dense I8 rows must be selected from 1,4,8,16')
            row = int(token)
            if row in rows:
                raise ValueError('Dense I8 row selections must be unique')
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build/dense-i8-decode-sep21')
    parser.add_argument('--host', type=Path, default=ROOT / 'build/prefill4k-wide-fullcache', help='Host source/object provenance matching the build')
    parser.add_argument('--cases', type=Path, required=True, help='Certified fixture manifest JSON; payloads are not read during preparation')
    parser.add_argument('--projection', help='Select an exact projection name from the fixture manifest')
    parser.add_argument('--rows', nargs='+', default=['1', '4', '8', '16'], metavar='R', help='Unique row shapes 1,4,8,16, as CSV or separate values')
    parser.add_argument('--pairs', type=int, default=4, help='Balanced timing repetitions, 1..32')
    parser.add_argument('--variant', type=int, choices=range(1, 5), help='Screen one candidate; omit for all four')
    parser.add_argument('--strict', action='store_true', help='Require byte-identical complete BF16 output before timing')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Execute only in the exclusive root GPU window')
    args = parser.parse_args()
    rows = selected_rows(args.rows)
    if not 1 <= args.pairs <= 32:
        raise ValueError('Dense I8 timing pairs must be 1..32')
    build = args.build.resolve()
    host = args.host.resolve()
    cases = args.cases.resolve()
    report = args.report.resolve()
    witness = Path(str(report) + '.invocation.json')
    if report.exists() or witness.exists():
        raise ValueError('Choose fresh dense I8 report and witness paths')
    manifest = json.loads(cases.read_text())
    if not isinstance(manifest, dict) or manifest.get('schema') != 'splash-private-dense-i8-f32-row-fit-fixtures-sep21-v1':
        raise ValueError('Dense I8 oracle requires the certified saved-role fixture manifest schema')
    entries = manifest.get('cases')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Dense I8 fixture manifest must contain cases')
    manifest_case_count = len(entries)
    for entry in entries:
        projection = entry.get('projection', '')
        if not isinstance(projection, str) or not any(role in projection for role in ('.linear_attn.', '.self_attn.', '.ple.')):
            raise ValueError('Dense I8 fixture role must be attention, GDN or PLE')
        if not entry.get('inputs_are_actual_prefill_capture_slices') or entry.get('live_decode_activation_capture'):
            raise ValueError('Dense I8 screen requires certified prefill capture slices')
    if args.projection is not None:
        entries = [entry for entry in entries if entry['projection'] == args.projection]
        if not entries:
            raise ValueError('--projection must exactly match a fixture manifest case')

    controls = {'DENSE_I8_DECODE_SEP21_ROWS': ','.join(str(row) for row in rows),
                'DENSE_I8_DECODE_SEP21_PAIRS': str(args.pairs), 'SPLASH_FLASH_QMV_F32': '1'}
    if args.projection is not None:
        controls['DENSE_I8_DECODE_SEP21_PROJECTION'] = args.projection
    if args.variant:
        controls['DENSE_I8_DECODE_SEP21_VARIANT'] = str(args.variant)
    if args.strict:
        controls['DENSE_I8_DECODE_SEP21_STRICT'] = '1'
    env = {key: value for key, value in os.environ.items() if not key.startswith(
        ('SPLASH_FLASH_', 'FLASH_INT8_STORE_', 'PREFILL4K_', 'PREFILL_MOE_', 'DENSE_I8_DECODE_'))}
    env.update(controls)
    command = [str(build / 'oracle'), '--gpu', str(build / 'splash.metallib'), str(cases), str(report)]
    selected = [args.variant] if args.variant else list(range(1, 5))
    provenance = {
        'gpu_execution_requested': args.run,
        'command': command,
        'controls': controls,
        'row_shapes': rows,
        'pairs': args.pairs,
        'selected_variants': selected,
        'variants': [{'variant': index, 'candidate_prefix': 'dense_i8_decode_sep21_',
                      'candidate_suffix': VARIANTS[index - 1][0], 'tile_m': VARIANTS[index - 1][1],
                      'tile_n': VARIANTS[index - 1][2], 'simdgroups': VARIANTS[index - 1][3]}
                     for index in selected],
        'binary_sha256': sha(build / 'oracle'),
        'metallib_sha256': sha(build / 'splash.metallib'),
        'oracle_source_sha256': sha(SOURCE / 'oracle.mm'),
        'cache_source_sha256': sha(SOURCE / 'cache.hpp'),
        'precision_source_sha256': sha(SOURCE / 'precision.hpp'),
        'candidate_source_sha256': sha(SOURCE / 'candidate.metal'),
        'abi_source_sha256': sha(SOURCE / 'abi.hpp'),
        'runner_source_sha256': sha(Path(__file__)),
        'makefile_source_sha256': sha(SOURCE / 'Makefile'),
        'readme_sha256': sha(SOURCE / 'README.md'),
        'float_control_source_sha256': sha(host / 'source/runtime/flash/FlashFloatDenseCache.hpp'),
        'bf16_control_source_sha256': sha(host / 'source/runtime/flash/FlashDenseSmallRows.hpp'),
        'raw_control_source_sha256': sha(host / 'source/runtime/flash/FlashAffine.hpp'),
        'float_control_object_sha256': sha(host / 'host/FlashFloatDenseCache.o'),
        'bf16_control_object_sha256': sha(host / 'host/FlashDenseSmallRows.o'),
        'raw_control_object_sha256': sha(host / 'host/FlashAffine.o'),
        'fixture_manifest_sha256': sha(cases),
        'fixture_manifest_schema': manifest.get('schema'),
        'fixture_manifest_metadata_keys': sorted(manifest),
        'fixture_manifest_case_count': manifest_case_count,
        'selected_projection': args.projection,
        'fixture_cases_metadata': [
            {key: entry.get(key) for key in ('projection', 'input_size', 'output_size', 'captured_rows',
                                           'source_bits', 'source_group_size', 'input_sha256',
                                           'f32_weights_sha256', 'bf16_weights_sha256', 'captured_expected_sha256')}
            for entry in entries
        ],
        'fixture_prepare_source_sha256_from_manifest': manifest.get('prepare_source_sha256'),
        'fixture_payload_hashes_recomputed_for_provenance': False,
        'weight_payload_read_for_provenance': False,
        'captured_input_payload_read_for_provenance': False,
        'captured_expected_payload_read_for_provenance': False,
        'optional_captured_expected_comparison_available': any(entry.get('captured_expected_file') for entry in entries),
        'captured_expected_hash_verification_root_runtime_only': True,
        'source_matrix_origin': 'one actual saved F32 N-by-K role from Flash-Next-operands-v1 per case',
        'activation_fixture_origin': 'first actual R rows of captured BF16 input from a prefill-2K execution',
        'live_decode_activation_fixture': False,
        'eligible_role_categories': ['attention', 'GDN', 'PLE'],
        'excluded_role_categories': ['vocabulary', 'router', 'HC down', 'MTP'],
        'role_and_shape_metadata_certification_required_before_payload_loading': True,
        'runtime_coefficient_quantization': 'F32 row maxabs/127 floored to minimum normal F32 for nonzero rows; signed-I8 RNE clamp [-127,127]',
        'zero_coefficient_row_codes': 0,
        'zero_coefficient_row_scale': 1,
        'activation_quantization': False,
        'candidate_activation_padding': False,
        'candidate_dot': 'whole-K unchanged BF16 A times requantized I8 B; F32 accumulation, one post-dot row scale, one BF16 cast',
        'stock_float_dense_small_rows_control': True,
        'optional_stock_bf16_small_rows_control_if_manifest_provides_matrix': True,
        'control_input_padding_and_helper_dispatch_cost_included_in_timing': True,
        'original_raw_affine_add_affine_control_included': True,
        'raw_qmv_f32_control_enabled': True,
        'r1_shipping_raw_route_constructed_on_captured_input': True,
        'raw_selected_source_span_count_per_case': 3,
        'raw_selected_span_hashes_required_before_and_after_timing': True,
        'whole_raw_shard_checksums_claimed': False,
        'full_raw_model_payload_read': False,
        'original_affine_coefficient_reconstruction_samples': 260,
        'original_affine_reconstruction_f32_bit_match_required_before_timing': True,
        'raw_vs_f32_max_relative_l2': 0.0001,
        'all_memory_governed_before_allocation': True,
        'memory_plan_source': 'cache.hpp; shape-derived original F32 and optional BF16 weights, captured inputs, I8 codes/scales, stock pad workspaces, outputs/audits and guards',
        'new_i8_code_memory_formula': 'round_up(N*K + 64, 16384)',
        'new_f32_row_scale_memory_formula': 'round_up(N*4 + 64, 16384)',
        'stock_max_row_pad_workspaces': 2,
        'stock_max_row_pad_logical_bytes_per_workspace_formula': '16*K*2',
        'host_reserve_minimum_bytes': 16 << 30,
        'host_reserve_fraction': 0.1,
        'full_model_loaded': False,
        'balanced_warm_category_rotation_required': True,
        'timing_categories': ['original raw affine QMV F32', 'stock F32 SmallRows',
                              'optional stock BF16 SmallRows', 'candidate BF16-by-I8'],
        'row_shapes_and_variants_rotated': True,
        'cpu_scans_between_gpu_timed_commands': False,
        'coefficient_conversion_excluded_from_timing': True,
        'initial_final_immutability_and_guard_checks_outside_timing': True,
        'weighted_f64_original_coefficient_quantization_envelope_required_before_timing': True,
        'weighted_f64_audit_sample_policy': 'at most 512 uniform columns plus every tile boundary, first/last and near-zero actual-output columns; first/last active rows',
        'coefficient_census_all_values_required_before_timing': True,
        'full_bf16_error_and_equality_measured_before_timing': True,
        'strict_full_bf16_equality_required': args.strict,
        'preregistered_max_relative_l2': 0.02,
        'preregistered_min_cosine': 0.9998,
        'numerical_alternative': True,
        'rejected_candidates_reported_without_timing': True,
        'semantic_quality_qualified': False,
        'mtp_acceptance_qualified': False,
        'live_decode_qualified': False,
        'model_quality_qualified': False,
        'provenance_reads': 'code, code-object/binary/metallib hashes and fixture manifest metadata only; no weight or captured input payload reads',
        'scope': 'saved-fixture dense I8 primitive at R1/R4/R8/R16; original raw route and complete cache control helper costs; captured prefill inputs, not live decode or end-to-end qualification',
    }
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'gpu_execution_requested': args.run, 'command': command, 'witness': str(witness)}))
    if args.run:
        subprocess.run(command, env=env, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
