#!/usr/bin/env python3
"""Write a bounded I8-LUT invocation plan; --run is for the root GPU owner only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
VARIANTS=[(f'm32_n64_k{k}_sg{sg}_{dtype}',k,sg,dtype)
          for k in (128,256) for sg in (2,4) for dtype in ('bf16','i8')]


def sha_if_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def bounded_json_if_file(path):
    if not path.is_file():return None
    if not 0<path.stat().st_size<=1<<20:raise ValueError(f'bounded metadata size differs: {path}')
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build/prefill-i8-lut-sep21')
    parser.add_argument('--pack-dir',type=Path,required=True)
    parser.add_argument('--rows',type=int,choices=(2048,),default=2048)
    parser.add_argument('--pairs',type=int,default=4)
    parser.add_argument('--pattern',choices=('both','spread-all','hit-concentrated'),default='both')
    parser.add_argument('--variant',type=int,choices=range(1,9))
    parser.add_argument('--input',type=Path)
    parser.add_argument('--ids',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--run',action='store_true',help='Run only in the exclusive root-owned GPU window')
    args=parser.parse_args()
    if not 2<=args.pairs<=32 or args.pairs%2 or bool(args.input)!=bool(args.ids):
        raise ValueError('pairs must be even and2..32; raw input and route IDs must be paired')
    build=args.build.resolve();pack=args.pack_dir.resolve();report=args.report.resolve()
    witness=Path(str(report)+'.invocation.json')
    if report.exists() or witness.exists():
        raise ValueError('choose fresh report and invocation witness paths')
    command=[str(build/'oracle'),'--gpu',str(build/'splash.metallib'),str(pack),str(args.rows),str(report)]
    controls={'SPLASH_FLASH_MOE_Q4X8':'1','SPLASH_FLASH_MOE_DIRECT_A':'1','PREFILL_I8_LUT_PAIRS':str(args.pairs)}
    if args.pattern!='both':controls['PREFILL_I8_LUT_PATTERN']=args.pattern
    if args.variant:controls['PREFILL_I8_LUT_VARIANT']=str(args.variant)
    if args.input:
        controls['PREFILL_I8_LUT_INPUT']=str(args.input.resolve());controls['PREFILL_I8_LUT_IDS']=str(args.ids.resolve())
    env={key:value for key,value in os.environ.items() if not key.startswith(
        ('SPLASH_FLASH_','FLASH_INT8_STORE_','PREFILL4K_','PREFILL_MOE_','PREFILL_I8_LUT_'))}
    env.update(controls)
    selected=[args.variant] if args.variant else list(range(1,9))
    pack_manifest=bounded_json_if_file(pack/'manifest.json')
    pack_certificate=bounded_json_if_file(pack/'certificate.json')
    source_files=['generate.py','run.py','cache.hpp','component.mk','pack.py','prepare.py','verify_sdk.py','candidate.metal']
    provenance={
        'schema':'prefill-i8-lut-sep21-invocation-v1',
        'gpu_execution_requested':args.run,'gpu_owner':'root only','command':command,'controls':controls,
        'source_sha256':{name:sha_if_file(HERE/name) for name in source_files},
        'built_sha256':{name:sha_if_file(build/name) for name in ('oracle','oracle.mm','candidate.metal','splash.metallib')},
        'inherited_source_sha256':{name:sha_if_file(ROOT/name) for name in (
            'dev/benchmarks/prefill_moe_sep21/native_m64/generate.py',
            'dev/benchmarks/prefill_moe_sep21/memory.metal',
            'dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp',
            'build/prefill4k-int8tiles/oracle.mm')},
        'sdk_compile_proof_sha256':sha_if_file(build/'sdk-compile.json'),
        'sdk_compile_proof':bounded_json_if_file(build/'sdk-compile.json'),
        'pack_manifest_sha256':sha_if_file(pack/'manifest.json'),
        'pack_certificate_sha256':sha_if_file(pack/'certificate.json'),
        'original_q4_id_inputs':pack_manifest.get('original_q4_id_inputs') if pack_manifest else None,
        'pack_source_i8_layer':pack_manifest.get('source_i8_layer') if pack_manifest else None,
        'pack_origin_certificate':pack_certificate,
        'selected_variants':[{'variant':i,'suffix':VARIANTS[i-1][0],'fixed_k':VARIANTS[i-1][1],
            'simdgroups':VARIANTS[i-1][2],'threadgroup_b_dtype':VARIANTS[i-1][3]} for i in selected],
        'comparison_controls':['whole_sg4','fixed_k128_sg2'],
        'candidate_storage':'cooperative threadgroup staged B',
        'cooperative_register_b_sdk_probe':'rejected for SG2/SG4; input cooperative tensors require one SIMD group',
        'q4_ids_bytes_per_g64':32,'i8_lut_bytes_per_g64':16,'compressed_codes_bytes_per_g64':48,
        'source_saved_i8_bytes_per_g64':64,'late_f32_row_scales_bound_from_original_store':True,
        'strict_every_source_code_reconstruction_before_after':True,
        'preregistration_revision':'before_gpu_same_descriptor_and_current_best',
        'strict_raw_f32_scaled_f32_scaled_bf16_and_full_chain_against_same_descriptor_uncompressed_before_timing':True,
        'strict_complete_bf16_chain_against_current_best_before_timing':True,
        'sg2_k128_also_raw_scaled_f32_and_boundary_bf16_exact_current_best':True,
        'whole_sg4_is_honest_raw_scaled_boundary_and_full_chain_contrast_only':True,
        'rejected_candidates_timed':False,'strict_equality_optional':False,
        'dot_audits_down_input':'same original whole-SG4 saved activated BF16 boundary',
        'candidate_full_chain_down_input':'candidate saved activated BF16 boundary',
        'synthetic_input':'per-row true RMS normalization before BF16 rounding' if not args.input else 'caller-provided BF16',
        'patterns': ['spread-all','hit-concentrated'] if args.pattern=='both' else [args.pattern],
        'pairs_per_control':args.pairs,'warm_gpu_minimum_ms_per_arm':150,
        'timing_order':'balanced alternating control/candidate pairs against both controls',
        'timing_scopes':['full_chain','gate_up_only','down_only_common_original_bf16_activation'],
        'timing_control_sets':[['whole_sg4','fixed_k128_sg2'],['same_descriptor_uncompressed','fixed_k128_sg2']],
        'untimed_gpu_malformed_cases':['ID -1','ID512','duplicate ID','rank UINTMAX','rank512',
            'job expert512','job row_begin OOB','jobcount OOB','terminal offset OOB','canonical route OOB'],
        'cpu_diagnostic_touches_reads_scans_hashes_between_timed_submissions':False,
        'audit_checks_payload_reconstruction_and_hashes_in_gpu_timing':False,
        'one_layer_only':True,'all_48_layer_payloads_loaded':False,'production_metadata_loader_used':False,
        'scratch_and_audit_admission_bytes':5<<30,'host_reserve_minimum_bytes':16<<30,'host_reserve_fraction':0.1,
        'model_quality_qualified':False,'full_worker_qualified':False,
        'provenance_reads':'source/build files and bounded JSON only; no model, compressed IDs/LUT, input or route payload reads',
    }
    if args.run:
        for path in (build/'oracle',build/'splash.metallib',pack/'manifest.json',pack/'certificate.json'):
            if not path.is_file():raise ValueError(f'root run prerequisite missing: {path}')
    witness.parent.mkdir(parents=True,exist_ok=True);witness.write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'gpu_execution_requested':args.run,'command':command,'witness':str(witness)}))
    if args.run:subprocess.run(command,env=env,check=True,cwd=ROOT)


if __name__=='__main__':main()
