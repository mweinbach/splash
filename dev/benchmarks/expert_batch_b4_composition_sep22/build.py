#!/usr/bin/env python3
"""CPU-only fresh current50-host/private-header build and exact76+2 AIR linkage."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[3];PRIVATE=Path('dev/benchmarks/expert_batch_b4_composition_sep22')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args();b=a.build.resolve();m=json.loads((b/'overlay-manifest.json').read_text());flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(b/'source'),'-I'+str(b/'source/runtime'),'-I'+str(b/'source/dev/benchmarks/prefill4k_attention')];cxx=['xcrun','-sdk','macosx','clang++',*flags]
 for r in m['files']:
  if sha(b/'source'/r['path'])!=r['sha256']:raise ValueError('source drift:'+r['path'])
 for r in m['frozen_inputs']:
  if sha(b/r['path'])!=r['sha256']:raise ValueError('frozen input drift:'+r['path'])
 def one(r):
  obj=b/r['object'];obj.parent.mkdir(parents=True,exist_ok=True);cmd=[*cxx,'-MMD','-MP','-c',str(b/'source'/r['source']),'-o',str(obj)];p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
  if p.returncode:raise RuntimeError(r['source']+'\n'+p.stderr)
  return {'path':r['object'],'source':r['source'],'sha256':sha(obj),'compile_command':cmd}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:objects=list(pool.map(one,m['rebuild']))
 if len(objects)!=50 or len(m['Core'])!=4:raise ValueError('exact50+4 closure required')
 link=[*cxx,*[str(b/r['path'])for r in objects],*[str(b/r['path'])for r in m['Core']],'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'splash-flash')];subprocess.run(link,cwd=ROOT,check=True)
 current_airs=[str(b/r['path'])for r in m['frozen_inputs']if r['category']=='currentFloatAIR'];baseline=b/'current76-baseline.metallib';subprocess.run(['xcrun','-sdk','macosx','metallib',*current_airs,'-o',str(baseline)],cwd=ROOT,check=True)
 if sha(baseline)!='1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf':raise ValueError('CURRENT76 exact1e34 recipe failed')
 airs=[str(b/r['path'])for r in m['frozen_inputs']if r['category']in ('currentFloatAIR','integerAIR')];metal=['xcrun','-sdk','macosx','metallib',*airs,'-o',str(b/'splash.metallib')];subprocess.run(metal,cwd=ROOT,check=True)
 tests=[]
 for name,src,frameworks,modes in [('policy-cpu','policy_cpu.cpp',[],['--missing','--freeze0','--freeze1','--retry0','--retry1']),('bqsa-policy-cpu','bqsa_policy_cpu.mm',['-framework','Foundation','-framework','Metal','-framework','IOKit'],['--missing','--freeze0','--freeze1','--retry1'])]:
  subprocess.run([*cxx,str(b/'source'/PRIVATE/src),*frameworks,'-o',str(b/name)],cwd=ROOT,check=True)
  for mode in modes:
   p=subprocess.run([str(b/name),mode],capture_output=True,text=True,check=True);tests.append(json.loads(p.stdout))
 p=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],capture_output=True,text=True,check=True);worker=json.loads(p.stdout)
 if not worker.get('valid')or worker.get('gpu_work')or len(worker['checks'])!=49:raise ValueError('Worker49 CPU failure')
 (b/'cpu-self-test.json').write_text(json.dumps({'policy':tests,'Worker':worker,'GPU_work':False},indent=2)+'\n')
 artifact=['splash-flash','splash.metallib','policy-cpu','bqsa-policy-cpu'];seal={'schema':'current-B4BQSA-plus-exact-integer-50host-CPUbuild-v1','pass':True,'source_identity_sha256':m['source_identity_sha256'],'BQSA_source_policy_sha256':m['BQSA_source_policy_sha256'],'objects':objects,'artifacts':[{'path':n,'sha256':sha(b/n)}for n in artifact],'link_command':link,'metallib_link_command':metal,'compiler_flags':flags,'source_manifest_sha256':sha(b/'overlay-manifest.json'),'current76_baseline_sha256':sha(baseline),'GPU_work':False,'payload_reads':False,'whole_state_quality_timing_qualified':False};(b/'compiled-build.json').write_text(json.dumps(seal,indent=2)+'\n');print(json.dumps({'pass':True,'source_identity_sha256':m['source_identity_sha256'],'objects':54,'AIRs':len(airs),'GPU_work':False}))
if __name__=='__main__':main()
