#!/usr/bin/env python3
"""Compile isolated current host closure only; no model/device or GPU calls."""
import argparse, concurrent.futures, hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=ROOT/'build/batch-prefill-twopass-restored-sep22-worker-v1');a=p.parse_args();b=a.build.resolve();m=json.loads((b/'overlay-manifest.json').read_text());base=Path(m['base']);s=json.loads((base/'compiled-cpu-seal.json').read_text())
 if sha(base/'compiled-cpu-seal.json')!=m['base_seal_sha256']:raise ValueError('parent seal drift')
 for e in m['files']:
  if sha(b/'source'/e['path'])!=e['sha256']:raise ValueError('frozen source drift:'+e['path'])
 flags=s['compiler_flags'];flags=[x for x in flags if not x.startswith('-I')];flags+=['-I'+str(b/'source'),'-I'+str(b/'source/runtime'),'-I'+str(b/'source/dev/benchmarks/prefill4k_attention')]
 cxx=['xcrun','-sdk','macosx','clang++',*flags]
 def compile_one(e):
  dest=b/e['object'];dest.parent.mkdir(parents=True,exist_ok=True)
  cmd=[*cxx,'-MMD','-MP','-c',str(b/'source'/e['source']),'-o',str(dest)]
  result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
  if result.returncode:raise RuntimeError(e['source']+'\n'+result.stderr)
  return {'path':e['object'],'source':e['source'],'sha256':sha(dest),'command':cmd}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:objects=list(pool.map(compile_one,m['rebuild']))
 linked=[b/e['path']for e in objects]+[b/e['path']for e in m['Core']]
 if len(linked)!=54:raise ValueError('exact54 objects required')
 cmd=[*cxx,*map(str,linked),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'splash-flash')]
 subprocess.run(cmd,cwd=ROOT,check=True)
 cpu=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],capture_output=True,text=True,check=True);data=json.loads(cpu.stdout)
 if not data['valid']or data['gpu_work']or len(data['checks'])!=49:raise ValueError('Worker49 CPU gate failed')
 result={'schema':'batch-packedV-twopass-current50host-cpu-build-v1','pass':True,'objects':objects,'link_command':cmd,'binary_sha256':sha(b/'splash-flash'),'metallib_sha256':sha(b/'splash.metallib'),'cpu':data,'GPU_executed':False,'model_payload_read':False,'not_final_source_quality_seal':True}
 (b/'compiled-build.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'pass':True,'objects':len(linked),'binary_sha256':result['binary_sha256'],'GPU_work':False}))
if __name__=='__main__':main()
