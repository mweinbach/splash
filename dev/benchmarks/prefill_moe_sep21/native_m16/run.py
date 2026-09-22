#!/usr/bin/env python3
"""Prepare a bounded shipping M32/M16 comparison; only root may add --run."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[4]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build/prefill-moe-sep21-native-m16')
    parser.add_argument('--store',type=Path,default=ROOT/'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1')
    parser.add_argument('--layer',type=int,choices=range(48),default=0)
    parser.add_argument('--rows',type=int,default=2048)
    parser.add_argument('--pairs',type=int,default=4)
    parser.add_argument('--pattern',choices=('hit-concentrated','hit-spread','spread-all'),default='spread-all')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--ids',type=Path)
    policies=parser.add_mutually_exclusive_group()
    policies.add_argument('--normalized',action='store_true',help='Default: normalize each row to true RMS 1 before BF16 rounding')
    policies.add_argument('--inherited',action='store_true',help='Reproduce the original frozen /512 BF16 fixture')
    policies.add_argument('--divisor74',action='store_true',help='Reproduce the one_layer approximate /74 fixture')
    parser.add_argument('--strict',action='store_true',default=True,help='Always required: complete BF16 equality before timing')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--run',action='store_true',help='Execute only during the exclusive root GPU window')
    args=parser.parse_args()
    if not 1024<=args.rows<=2048 or not 1<=args.pairs<=32 or bool(args.input)!=bool(args.ids):
        raise ValueError('Native oracle requires R1024..2048, pairs1..32, paired raw input/IDs')
    if args.input and (args.normalized or args.inherited or args.divisor74):
        raise ValueError('Synthetic input policies cannot be applied to raw input')
    build=args.build.resolve();store=args.store.resolve();report=args.report.resolve()
    witness=Path(str(report)+'.invocation.json')
    if report.exists() or witness.exists():
        raise ValueError('Choose fresh native report and witness paths')
    manifest_path=store/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('selected_experts')!=[list(range(512)) for _ in range(48)]:
        raise ValueError('Native oracle requires canonical persisted Full512 inventory')
    policy='inherited512' if args.inherited else 'divisor74' if args.divisor74 else 'row-rms'
    controls={
        'SPLASH_FLASH_MOE_Q4X8':'1','SPLASH_FLASH_MOE_DIRECT_A':'1',
        'FLASH_INT8_STORE_PAIRS':str(args.pairs),'FLASH_INT8_STORE_PATTERN':args.pattern,
        'FLASH_INT8_STORE_MAX_RL2':'0.001','FLASH_INT8_STORE_MIN_COSINE':'0.999999',
        'PREFILL_MOE_NATIVE_M16_INPUT_POLICY':policy}
    if args.input:
        controls['FLASH_INT8_STORE_INPUT']=str(args.input.resolve())
        controls['FLASH_INT8_STORE_ROUTE_IDS']=str(args.ids.resolve())
    env={key:value for key,value in os.environ.items() if not key.startswith(
        ('SPLASH_FLASH_','FLASH_INT8_STORE_','PREFILL4K_','PREFILL_MOE_'))}
    env.update(controls)
    command=[str(build/'oracle'),'--gpu',str(build/'splash.metallib'),str(store),
        str(args.layer),str(args.rows),str(report)]
    witness.parent.mkdir(parents=True,exist_ok=True)
    provenance={
        'gpu_execution_requested':args.run,'command':command,'controls':controls,
        'binary_sha256':sha(build/'oracle'),'metallib_sha256':sha(build/'splash.metallib'),
        'oracle_source_sha256':sha(build/'oracle.mm'),
        'generator_source_sha256':sha(Path(__file__).with_name('generate.py')),
        'runner_source_sha256':sha(Path(__file__)),
        'makefile_source_sha256':sha(Path(__file__).with_name('Makefile')),
        'inherited_baseline_source_sha256':sha(ROOT/'build/prefill4k-int8tiles/oracle.mm'),
        'native_adapter_source_sha256':sha(ROOT/'dev/benchmarks/prefill_moe_sep21/native_m64/generate.py'),
        'one_layer_loader_source_sha256':sha(ROOT/'dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp'),
        'frozen_store_source_sha256':sha(ROOT/'build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1/source/runtime/flash/FlashInt8ExpertStore.mm'),
        'shipping_kernel_source_sha256':sha(ROOT/'runtime/metal/kernels/shared/flash_int8_expert_store.metal'),
        'store_manifest_sha256':sha(manifest_path),
        'one_readonly_layer_bytes':2524446720,'additional_scratch_reservation_bytes':1<<30,
        'host_reserve_minimum_bytes':16<<30,'host_reserve_fraction':0.1,
        'full_model_loaded':False,'all_48_payloads_loaded':False,'production_store_constructor_used':False,
        'native_job_tiles':[32,16],'producer_simdgroups':[4,4],
        'independent_original_job_lists':True,'coefficients_scales_and_bf16_boundaries_shared':True,
        'additional_hit_list_dispatches':0,'additional_prefix_dispatches':0,'private_shaders_used':False,
        'strict_full_bf16_equality_required':True,'full_comparisons_retained_on_mismatch':True,
        'synthetic_input_policy':None if args.input else policy,
        'normalized_input_target_row_rms':1 if not args.input and policy=='row-rms' else None,
        'bf16_rounding_follows_normalization':not args.input and policy=='row-rms',
        'model_quality_qualified':False,
        'provenance_reads':'source, binaries, metallib and manifest only; no model payload, hidden or ID reads',
        'adapter_boundary':'shipping Full512 M16/M32 native dispatches over existing certified one-layer loader; production 48-layer constructor excluded',
        'scope':'matched complete expert chain; shared immutable layer/fixtures; warm alternating paired command timings',
    }
    if args.rows==2048 and args.pattern=='spread-all' and not args.input:
        provenance.update({'rows_per_expert':40,'expected_active_jobs':[1024,1536],
            'expected_padded_matrix_rows':[32768,24576],'expected_job_capacities':[1151,1791],
            'scratch_job_capacity':3071,'candidate_padding_reduction':0.25})
    witness.write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'gpu_execution_requested':args.run,'command':command,'witness':str(witness)}))
    if args.run:
        subprocess.run(command,env=env,check=True,cwd=ROOT)


if __name__=='__main__':
    main()
