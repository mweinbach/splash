#!/usr/bin/env python3
"""Prepare a Root-exclusive wide-column INT8 expert screen."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'build/prefill4k-int8columns')
    p.add_argument('--store',type=Path,default=ROOT/'build/prefill4k-fullcache-artifacts/int8-experts-frequency256-v1')
    p.add_argument('--rows',type=int,default=2048);p.add_argument('--pairs',type=int,default=4)
    p.add_argument('--layer',type=int,choices=(0,24,47),default=0)
    p.add_argument('--pattern',choices=('hit-concentrated','hit-spread','miss-only','mixed','spread-all'))
    p.add_argument('--edge',action='store_true');p.add_argument('--report',type=Path,required=True);p.add_argument('--run',action='store_true')
    a=p.parse_args();build=a.build.resolve();store=a.store.resolve();report=a.report.resolve()
    if not 1 <=a.rows <=8192 or not 1 <=a.pairs <=32:raise ValueError('Unsupported wide-column rows/pairs')
    manifest=json.loads((store/'manifest.json').read_text());counts={len(ids) for ids in manifest['selected_experts']}
    if len(counts)!=1 or next(iter(counts)) not in (32,64,128,256):raise ValueError('Synthetic miss patterns require partial uniform expert inventory')
    controls={'SPLASH_FLASH_PLE_SSD_STREAMING':'1','SPLASH_FLASH_MOE_Q4X8':'1','SPLASH_FLASH_MOE_DIRECT_A':'1','SPLASH_FLASH_MOE_M64':'1',
        'FLASH_INT8_STORE_ROWS':str(a.rows),'FLASH_INT8_STORE_TILE':'32','FLASH_INT8_STORE_LAYERS':str(a.layer),'FLASH_INT8_STORE_PAIRS':str(a.pairs)}
    if a.pattern:controls['FLASH_INT8_STORE_PATTERN']=a.pattern
    if a.edge:controls['PREFILL4K_INT8_HIT_EDGE']='1'
    env={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_FLASH_','FLASH_INT8_STORE_','PREFILL4K_'))};env.update(controls)
    command=[str(build/'oracle'),str(build/'splash.metallib'),str(ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1'),str(store),str(report)]
    witness=Path(str(report)+'.invocation.json')
    if report.exists() or witness.exists():raise ValueError('Choose fresh wide-column report/provenance')
    witness.parent.mkdir(parents=True,exist_ok=True)
    witness.write_text(json.dumps({'gpu_executed':a.run,'command':command,'controls':controls,'binary_sha256':sha(build/'oracle'),'metallib_sha256':sha(build/'splash.metallib'),
        'store_manifest_sha256':sha(store/'manifest.json'),'stored_experts_per_layer':next(iter(counts)),
        'candidate_source_sha256':sha(build/'candidate.metal'),'audit_source_sha256':sha(build/'audit.metal'),'oracle_source_sha256':sha(build/'oracle.mm'),
        'same_persisted_i8_and_f32_scales':True,'native_m32_jobs_and_parameters_unchanged':True,'additional_prefix_dispatches':0,
        'variants':[{'m':32,'n':128,'sg':8,'gate_ctas':5,'down_ctas':20},{'m':32,'n':256,'sg':16,'gate_ctas':3,'gate_final_columns':128,'down_ctas':10}],
        'full_bf16_equality_required_before_timing':True,'separate_raw_f32_fp64_absolute_audit_before_timing':True,'normalized_synthetic_input':not a.edge,
        'single_constructor':True,'model_quality_qualified':False,
        'scope':'complete expert chain; unchanged buckets/Q4 misses/scatter/combine; warm alternating paired commands; audit/certification and source/store scans excluded'},indent=2)+'\n')
    print(json.dumps({'gpu_executed':a.run,'command':command,'witness':str(witness)}))
    if a.run:subprocess.run(command,env=env,check=True,cwd=ROOT)
if __name__=='__main__':main()
