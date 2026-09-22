#!/usr/bin/env python3
"""Prepare Root-exclusive low-SIMD and exact register-code expert screens."""
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
    parser.add_argument('--build', type=Path, default=ROOT / 'build/prefill-moe-sep21')
    parser.add_argument('--store', type=Path, default=ROOT / 'build/prefill4k-fullcache-artifacts/int8-experts-frequency256-v1')
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--layer', type=int, choices=(0,24,47), default=0)
    parser.add_argument('--variant', type=int, choices=range(1,12))
    parser.add_argument('--pattern', choices=('hit-concentrated','hit-spread','miss-only','mixed','spread-all'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--ids', type=Path)
    parser.add_argument('--strict', action='store_true', help='Require byte-identical complete BF16 chain')
    parser.add_argument('--edge', action='store_true')
    parser.add_argument('--phases', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.rows <=8192 or not 1 <= args.pairs <=32 or bool(args.input) != bool(args.ids):
        raise ValueError('Private expert shape, pairing, or raw input/ID pair unsupported')
    build=args.build.resolve();store=args.store.resolve();report=args.report.resolve()
    witness=Path(str(report)+'.invocation.json')
    if report.exists() or witness.exists():
        raise ValueError('Choose fresh private expert report/provenance')
    manifest=json.loads((store/'manifest.json').read_text())
    counts={len(ids) for ids in manifest['selected_experts']}
    if len(counts) !=1 or next(iter(counts)) not in (32,64,128,256):
        raise ValueError('Original-source primitive supports only partial uniform stores; use the bounded one-layer or original-target-omitted worker for Full512')
    count=next(iter(counts))
    controls={
        'SPLASH_FLASH_PLE_SSD_STREAMING':'1','SPLASH_FLASH_MOE_Q4X8':'1',
        'SPLASH_FLASH_MOE_DIRECT_A':'1','SPLASH_FLASH_MOE_M64':'1',
        'FLASH_INT8_STORE_ROWS':str(args.rows),'FLASH_INT8_STORE_TILE':'32',
        'FLASH_INT8_STORE_LAYERS':str(args.layer),'FLASH_INT8_STORE_PAIRS':str(args.pairs),
        'FLASH_INT8_STORE_MAX_RL2':'0.001','FLASH_INT8_STORE_MIN_COSINE':'0.999999'}
    if args.pattern: controls['FLASH_INT8_STORE_PATTERN']=args.pattern
    if args.variant: controls['PREFILL_MOE_SEP21_VARIANT']=str(args.variant)
    if args.strict: controls['PREFILL_MOE_SEP21_STRICT']='1'
    if args.edge: controls['PREFILL4K_INT8_HIT_EDGE']='1'
    if args.phases: controls['PREFILL_MOE_SEP21_PHASES']='1'
    if args.input:
        controls['FLASH_INT8_STORE_INPUT']=str(args.input.resolve())
        controls['FLASH_INT8_STORE_ROUTE_IDS']=str(args.ids.resolve())
    env={key:value for key,value in os.environ.items() if not key.startswith(('SPLASH_FLASH_','FLASH_INT8_STORE_','PREFILL4K_','PREFILL_MOE_'))}
    env.update(controls)
    command=[str(build/'oracle'),str(build/'splash.metallib'),str(ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1'),str(store),str(report)]
    witness.parent.mkdir(parents=True,exist_ok=True)
    witness.write_text(json.dumps({
        'gpu_executed':args.run,'command':command,'controls':controls,
        'binary_sha256':sha(build/'oracle'),'metallib_sha256':sha(build/'splash.metallib'),
        'store_manifest_sha256':sha(store/'manifest.json'),'stored_experts_per_layer':count,
        'memory_source_sha256':sha(ROOT/'dev/benchmarks/prefill_moe_sep21/memory.metal'),
        'register_source_sha256':sha(ROOT/'dev/benchmarks/prefill_moe_sep21/register.metal'),
        'oracle_source_sha256':sha(build/'oracle.mm'),
        'persisted_i8_and_f32_scales_unchanged':True,'i8_to_bf16_register_conversion_exact':True,
        'native_m32_jobs_and_parameters_unchanged':True,'additional_prefix_dispatches':0,
        'full_bf16_error_and_equality_measured_before_timing':True,'strict_full_bf16_equality_required':args.strict,
        'numerical_alternatives_reported_per_variant':True,'model_quality_qualified':False,
        'register_sdk_tile':'M32/M16 N32K32; N64 uses sequential column halves; logical K64/K128 groups 2/4 K32 substeps',
        'scope':'complete expert chain with unchanged buckets/Q4 misses/scatter/combine; warm rotating paired commands; CPU checks/scans excluded'
    },indent=2)+'\n')
    print(json.dumps({'gpu_executed':args.run,'command':command,'witness':str(witness)}))
    if args.run:subprocess.run(command,env=env,check=True,cwd=ROOT)


if __name__=='__main__':
    main()
