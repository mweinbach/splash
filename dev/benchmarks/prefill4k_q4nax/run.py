#!/usr/bin/env python3
"""Root-exclusive once-rounded BF16 coefficient register-NAX screens."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT /'build/prefill4k-q4nax')
    p.add_argument('--relaxed',action='store_true');p.add_argument('--all-experts',action='store_true')
    p.add_argument('--rows',type=int,default=2048);p.add_argument('--pairs',type=int,default=4)
    p.add_argument('--layer',type=int,choices=(0,24,47),default=0)
    p.add_argument('--pattern',choices=('hit-concentrated','hit-spread','miss-only','mixed','spread-all'))
    p.add_argument('--edge',action='store_true');p.add_argument('--report',type=Path,required=True);p.add_argument('--run',action='store_true')
    a=p.parse_args();build=a.build.resolve();store=ROOT /'build/prefill4k-fullcache-artifacts/int8-experts-frequency256-v1';report=a.report.resolve()
    if not 1 <=a.rows <=8192 or not 1 <=a.pairs <=32:raise ValueError('Unsupported NAX rows/pairs')
    controls={'SPLASH_FLASH_PLE_SSD_STREAMING':'1','SPLASH_FLASH_MOE_Q4X8':'1','SPLASH_FLASH_MOE_DIRECT_A':'1','SPLASH_FLASH_MOE_M64':'1',
        'FLASH_INT8_STORE_ROWS':str(a.rows),'FLASH_INT8_STORE_TILE':'32','FLASH_INT8_STORE_LAYERS':str(a.layer),'FLASH_INT8_STORE_PAIRS':str(a.pairs),
        'PREFILL4K_Q4NAX_DIAGNOSTIC':str(report)+'.diagnostic.json'}
    if a.pattern:controls['FLASH_INT8_STORE_PATTERN']=a.pattern
    if a.all_experts:controls['PREFILL4K_Q4CODED_ALL']='1'
    if a.edge:controls['PREFILL4K_INT8_HIT_EDGE']='1'
    if a.relaxed:controls['PREFILL4K_Q4NAX_RELAXED']='1'
    env={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_FLASH_','FLASH_INT8_STORE_','PREFILL4K_'))};env.update(controls)
    command=[str(build /'oracle'),str(build /'splash.metallib'),str(ROOT /'install/local-models/Flash-Next-oQ4e-mtp-v1'),str(store),str(report)]
    witness=report.with_suffix(report.suffix +'.invocation.json')
    if report.exists() or witness.exists() or Path(controls['PREFILL4K_Q4NAX_DIAGNOSTIC']).exists():raise ValueError('Choose fresh NAX report/provenance/diagnostic')
    witness.parent.mkdir(parents=True,exist_ok=True)
    audit_source=(build /'audit.metal').read_text()
    unique_audit_helpers='q4naxaudit_gate' in audit_source and 'q4naxaudit_down' in audit_source
    witness.write_text(json.dumps({'gpu_executed':a.run,'command':command,'controls':controls,'binary_sha256':sha(build /'oracle'),'metallib_sha256':sha(build /'splash.metallib'),
        'same_once_rounded_bf16_coefficients':True,'changed_k16_mma_grouping':True,'relaxed_precision':a.relaxed,'model_quality_qualified':False,
        'audit_helpers_uniquely_named':unique_audit_helpers,'audit_source_sha256':sha(build /'audit.metal'),'oracle_source_sha256':sha(build /'oracle.mm'),
        'diagnostic_path':controls['PREFILL4K_Q4NAX_DIAGNOSTIC'],
        'scope':'full expert chain; no sum dispatches/TGM/barriers; four independent single-SIMD M16N32 quadrants; separate F32 linear/FP64 source-BF16 audit'},indent=2)+'\n')
    print(json.dumps({'gpu_executed':a.run,'command':command,'witness':str(witness)}))
    if a.run:subprocess.run(command,env=env,check=True,cwd=ROOT)
if __name__ =='__main__':main()
