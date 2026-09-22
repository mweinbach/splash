#!/usr/bin/env python3
"""Prepare a bounded one-layer native M32/M64 invocation; --run opts into GPU."""
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
    parser.add_argument('--build', type=Path, default=ROOT / 'build/prefill-moe-sep21-native-m64')
    parser.add_argument('--store', type=Path, default=ROOT / 'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1')
    parser.add_argument('--layer', type=int, choices=range(48), default=0)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--pattern', choices=('hit-concentrated', 'hit-spread', 'spread-all'), default='spread-all')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--ids', type=Path)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not 1024 <= args.rows <= 2048 or not 1 <= args.pairs <= 32 or bool(args.input) != bool(args.ids):
        raise ValueError('Native oracle requires R1024..2048, pairs1..32, paired raw input/IDs')
    build=args.build.resolve();store=args.store.resolve();report=args.report.resolve()
    witness=Path(str(report)+'.invocation.json')
    if report.exists() or witness.exists():
        raise ValueError('Choose fresh native report and witness paths')
    manifest=json.loads((store/'manifest.json').read_text())
    if manifest.get('selected_experts') != [list(range(512)) for _ in range(48)]:
        raise ValueError('Native oracle requires canonical persisted Full512 inventory')
    controls={
        'SPLASH_FLASH_MOE_Q4X8':'1','SPLASH_FLASH_MOE_DIRECT_A':'1','SPLASH_FLASH_MOE_M64':'1',
        'FLASH_INT8_STORE_PAIRS':str(args.pairs),'FLASH_INT8_STORE_PATTERN':args.pattern,
        'FLASH_INT8_STORE_MAX_RL2':'0.001','FLASH_INT8_STORE_MIN_COSINE':'0.999999'}
    if args.strict:controls['PREFILL_MOE_NATIVE_M64_STRICT']='1'
    if args.input:
        controls['FLASH_INT8_STORE_INPUT']=str(args.input.resolve())
        controls['FLASH_INT8_STORE_ROUTE_IDS']=str(args.ids.resolve())
    env={k:v for k,v in os.environ.items() if not k.startswith(
        ('SPLASH_FLASH_','FLASH_INT8_STORE_','PREFILL4K_','PREFILL_MOE_'))}
    env.update(controls)
    command=[str(build/'oracle'),'--gpu',str(build/'splash.metallib'),str(store),
        str(args.layer),str(args.rows),str(report)]
    witness.parent.mkdir(parents=True,exist_ok=True)
    witness.write_text(json.dumps({
        'gpu_execution_requested':args.run,'command':command,'controls':controls,
        'binary_sha256':sha(build/'oracle'),'metallib_sha256':sha(build/'splash.metallib'),
        'oracle_source_sha256':sha(build/'oracle.mm'),
        'inherited_baseline_source_sha256':sha(ROOT/'build/prefill4k-int8tiles/oracle.mm'),
        'one_layer_loader_source_sha256':sha(ROOT/'dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp'),
        'store_manifest_sha256':sha(store/'manifest.json'),
        'one_readonly_layer_bytes':2524446720,'full_model_loaded':False,'all_48_payloads_loaded':False,
        'native_job_tiles':[32,64],'producer_simdgroups':[4,8],
        'additional_hit_list_dispatches':0,'private_shaders_used':False,
        'strict_full_bf16_equality_required':args.strict,'model_quality_qualified':False,
        'adapter_boundary':'original Full512 store native dispatches over existing certified one-layer loader; production 48-layer constructor excluded',
        'scope':'matched complete expert chain; shared immutable layer/fixtures; warm alternating paired GPU command timings'
    },indent=2)+'\n')
    print(json.dumps({'gpu_execution_requested':args.run,'command':command,'witness':str(witness)}))
    if args.run:subprocess.run(command,env=env,check=True,cwd=ROOT)


if __name__=='__main__':
    main()
