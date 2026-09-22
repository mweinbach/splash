#!/usr/bin/env python3
"""Root-only optional GPU launch; default is sealed provenance preparation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'build/prefill-m64-low-sg-sep21')
    p.add_argument('--store',type=Path,default=ROOT/'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1')
    p.add_argument('--layer',type=int,choices=range(48),default=0)
    p.add_argument('--rows',type=int,default=2048)
    p.add_argument('--pairs',type=int,default=8)
    p.add_argument('--pattern',choices=['spread-all','hit-concentrated','hit-spread'],default='spread-all')
    p.add_argument('--input',type=Path);p.add_argument('--ids',type=Path)
    p.add_argument('--source-check',action='store_true')
    p.add_argument('--report',type=Path);p.add_argument('--run',action='store_true')
    a=p.parse_args();build=a.build.resolve();seal=json.loads((build/'input-seal.json').read_text())
    for name,expected in seal['sha256'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Sealed code/input changed: '+name)
    if a.source_check:
        if a.run or a.report:raise ValueError('Source check submits no GPU and writes no report')
        print(json.dumps(dict(source_check='passed',inputs=len(seal['sha256']),gpu_executed=False,payload_bytes_read=0)))
        return
    if a.report is None:raise ValueError('--report is required')
    if not 1024<=a.rows<=2048 or not 4<=a.pairs<=32 or a.pairs%2:raise ValueError('Requires rows1024..2048 and even pairs4..32')
    if bool(a.input)!=bool(a.ids):raise ValueError('Raw input and IDs must be supplied together')
    report=a.report.resolve();witness=Path(str(report)+'.invocation.json')
    if report.exists() or witness.exists():raise ValueError('Choose a fresh report and witness')
    store=a.store.resolve();manifest=json.loads((store/'manifest.json').read_text())
    if manifest.get('selected_experts')!=[list(range(512)) for _ in range(48)]:raise ValueError('Requires certified Full512 metadata inventory')
    controls={'SPLASH_FLASH_MOE_Q4X8':'1','SPLASH_FLASH_MOE_DIRECT_A':'1','SPLASH_FLASH_MOE_M64':'1',
              'FLASH_INT8_STORE_PAIRS':str(a.pairs),'FLASH_INT8_STORE_PATTERN':a.pattern,
              'FLASH_INT8_STORE_MAX_RL2':'.001','FLASH_INT8_STORE_MIN_COSINE':'.999999',
              'PREFILL_MOE_NATIVE_M16_INPUT_POLICY':'row-rms'}
    if a.input:
        controls.update(FLASH_INT8_STORE_INPUT=str(a.input.resolve()),FLASH_INT8_STORE_ROUTE_IDS=str(a.ids.resolve()))
    env={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_FLASH_','FLASH_INT8_STORE_','PREFILL4K_','PREFILL_MOE_','ADAPTIVE_EXPERT_TAIL_'))}
    env.update(controls)
    cmd=[str(build/'oracle'),'--gpu',str(build/'splash.metallib'),str(store),str(a.layer),str(a.rows),str(report)]
    provenance=dict(schema='private-native-m64-low-sg-invocation-v1',gpu_execution_requested=a.run,command=cmd,controls=controls,
                    input_seal_sha256=sha(build/'input-seal.json'),input_seal=seal,
                    store_manifest_sha256=sha(store/'manifest.json'),payload_reads_by_runner=0,
                    one_readonly_layer_bytes=2524446720,additional_scratch_reservation_bytes=3<<30,
                    host_reserve_minimum_bytes=16<<30,host_reserve_fraction=.1,
                    control_native_job_tile=32,candidate_native_job_tile=64,uniform_r2048_active_jobs=[1024,512],
                    control='qualified SG2 fixedK128 M32 with M16 adaptive tail',candidate_sg=[4,2],
                    raw_f32_scaled_bf16_and_full_chain_exact_required=True,
                    same_original_source_coefficients=True,original63row_pad=True,valid_rows_masked_stores=True,
                    native_job_generation_cost_timed=True,minimum_gpu_warm_ms=100,cpu_buffer_reads_in_timed_loop=False,
                    model_quality_qualified=False,full_model_loaded=False,all48layers_loaded=False)
    witness.parent.mkdir(parents=True,exist_ok=True);witness.write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(command=cmd,witness=str(witness),gpu_execution_requested=a.run)))
    if a.run:subprocess.run(cmd,env=env,cwd=ROOT,check=True)


if __name__=='__main__':main()
