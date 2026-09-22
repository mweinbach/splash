#!/usr/bin/env python3
"""Verify the seal and print a Root-only command; --run executes the serialized GPU job."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build/prefill-qsa-twopass-sep21')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=8)
    parser.add_argument('--packed-v',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();build=args.build.resolve();report=args.report.resolve();source=build/'source'
    if report.exists():raise ValueError('Report must be fresh')
    if args.pairs<4 or args.pairs>32 or args.pairs%4:raise ValueError('Pairs must be a multiple of4 within4..32')
    seal=json.loads((build/'cpu-seal.json').read_text())
    if not seal['pass'] or seal['gpu_executed']:raise ValueError('Invalid CPU seal')
    if sha(build/'source-manifest.json')!=seal['source_manifest_sha256']:raise ValueError('Source manifest changed')
    manifest=json.loads((build/'source-manifest.json').read_text())
    for relative,digest in manifest['source_sha256'].items():
        if sha(source/relative)!=digest:raise ValueError(f'Frozen source changed: {relative}')
    for relative,digest in seal['artifact_sha256'].items():
        if sha(build/relative)!=digest:raise ValueError(f'Artifact changed: {relative}')
    for path,digest in seal['toolchain_metadata_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError(f'Toolchain metadata changed: {path}')
    settings={'SPLASH_FLASH_QSA_F32':'1','SPLASH_FLASH_QSA_MPP':'1','SPLASH_FLASH_QSA_ROW_TILES':'1',
              'PREFILL_QSA_TWOPASS_ROOT_GPU':'1','PREFILL_QSA_TWOPASS_PAIRS':str(args.pairs),
              'PREFILL_QSA_TWOPASS_PACKED_V':str(int(args.packed_v))}
    command=[str(build/'oracle'),str(build/'splash.metallib'),str(report)]
    print(json.dumps({'prepared':True,'GPU_executed':args.run,'Root_only_GPU_execution':True,
                      'command':command,'environment':settings,'CPU_seal_sha256':sha(build/'cpu-seal.json')}),flush=True)
    if args.run:
        report.parent.mkdir(parents=True,exist_ok=True)
        env={k:v for k,v in os.environ.items() if not k.startswith(('SPLASH_FLASH_QSA_','FLASH_QSA_MPP_','PREFILL_QSA_TWOPASS_'))}
        env.update(settings);subprocess.run(command,env=env,check=True)
if __name__=='__main__':main()
