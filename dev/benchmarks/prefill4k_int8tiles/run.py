#!/usr/bin/env python3
"""Prepare/run Root-exclusive same-coefficient INT8 hit tile screens."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT =Path(__file__).resolve().parents[3]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p =argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tile',type=int,choices=(32,64,128),default=64)
    p.add_argument('--sg',type=int,choices=(4,8,16))
    p.add_argument('--rows',type=int,default=2048)
    p.add_argument('--pairs',type=int,default=4)
    p.add_argument('--layer',type=int,choices=(0,24,47),default=0)
    p.add_argument('--relaxed',action='store_true')
    p.add_argument('--edge',action='store_true')
    p.add_argument('--pattern',choices=('hit-concentrated','hit-spread','miss-only','mixed','spread-all'))
    p.add_argument('--store',type=Path,default=ROOT /'build/prefill4k-fullcache-artifacts/int8-experts-frequency256-v1')
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    p.add_argument('--sweep',action='store_true',help='One constructor/all seven variants with shared fixture/control graphs')
    a =p.parse_args()
    sg =a.sg if a.sg else 4 if a.tile ==32 else 8
    if (a.tile ==32 and sg !=4) or (a.tile ==64 and sg !=8) or not 1 <=a.rows <=8192:
        raise ValueError('Unsupported hit tile/SIMD/row geometry')
    build =ROOT /('build/prefill4k-int8tiles-sweep' if a.sweep else 'build/prefill4k-int8tiles')
    source =a.store.resolve();report =a.report.resolve()
    manifest =json.loads((source /'manifest.json').read_text())
    count =len(manifest['selected_experts'][0])
    if count not in (32,64,128,256):
        raise ValueError('Finite synthetic miss patterns require partial uniform stored inventory')
    controls ={'SPLASH_FLASH_PLE_SSD_STREAMING':'1','SPLASH_FLASH_MOE_Q4X8':'1','SPLASH_FLASH_MOE_DIRECT_A':'1','SPLASH_FLASH_MOE_M64':'1',
        'FLASH_INT8_STORE_ROWS':str(a.rows),'FLASH_INT8_STORE_TILE':str(64 if a.rows >=4096 else 32 if a.rows >=1024 else 16),
        'FLASH_INT8_STORE_LAYERS':str(a.layer),'FLASH_INT8_STORE_PAIRS':str(a.pairs),
        'PREFILL4K_INT8_HIT_TILE':str(a.tile),'PREFILL4K_INT8_HIT_SG':str(sg)}
    if a.pattern:controls['FLASH_INT8_STORE_PATTERN'] =a.pattern
    if a.relaxed:controls['PREFILL4K_INT8_HIT_RELAXED'] ='1'
    if a.edge:controls['PREFILL4K_INT8_HIT_EDGE'] ='1'
    env ={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_FLASH_','FLASH_INT8_STORE_','PREFILL4K_'))};env.update(controls)
    command =[str(build /'oracle'),str(build /'splash.metallib'),str(ROOT /'install/local-models/Flash-Next-oQ4e-mtp-v1'),str(source),str(report)]
    witness =report.with_suffix(report.suffix +'.invocation.json')
    if report.exists() or witness.exists():raise ValueError('Choose fresh hit tile report/provenance')
    witness.parent.mkdir(parents=True,exist_ok=True)
    witness.write_text(json.dumps({'gpu_executed':a.run,'command':command,'controls':controls,'store_manifest_sha256':sha(source /'manifest.json'),
        'stored_experts_per_layer':count,'binary_sha256':sha(build /'oracle'),'metallib_sha256':sha(build /'splash.metallib'),
        'same_persisted_coefficients_and_scales':True,'q4_miss_geometry_unchanged':True,'strict_full_bf16_equality_required':True,
        'relaxed_precision':a.relaxed,'actual_code_cancellation_fixture':a.edge,'scope':'complete expert chain including compact hit-list prefix/emission; constructor/scans excluded'},indent=2)+'\n')
    print(json.dumps({'gpu_executed':a.run,'command':command,'witness':str(witness)}))
    if a.run:subprocess.run(command,env=env,check=True,cwd=ROOT)

if __name__ =='__main__':main()
