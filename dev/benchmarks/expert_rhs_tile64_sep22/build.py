#!/usr/bin/env python3
"""CPU-only compiler/source seal. Model payloads and GPU are Root-only."""
from pathlib import Path
import argparse,hashlib,json,shlex,shutil,subprocess
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PARENT=ROOT/'build/compact-r4-preflight-bundle-sep22-worker-v1'
PRIVATE=Path('dev/benchmarks/expert_rhs_tile64_sep22')
ORIGINAL_AIR_SHA='a0cd35e03daf13324d0308c8b4d8cee1d6d932989be6cbdc0429e6ec471d05c2'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(argv):
 result=subprocess.run(list(map(str,argv)),cwd=ROOT,capture_output=True,text=True)
 if result.returncode:print(result.stderr);result.check_returncode()
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'build/expert-rhs-tile64-sep22-component-v1');p.add_argument('--reuse-air-from',type=Path);a=p.parse_args();out=a.output.resolve()
 if out.exists() or ROOT/'build' not in out.parents:raise ValueError('Fresh private build required')
 manifest=json.loads((PARENT/'overlay-manifest.json').read_text());seal=json.loads((PARENT/'cpu-source-witness.json').read_text())
 if not seal['pass']:raise ValueError('Sealed current parent required')
 shutil.copytree(PARENT/'source',out/'source')
 for r in manifest['files']:
  if sha(out/'source'/r['path'])!=r['sha256']:raise ValueError('Parent program source drift:'+r['path'])
 dest=out/'source'/PRIVATE;dest.mkdir(parents=True,exist_ok=True)
 for path in HERE.iterdir():
  if path.is_file() and path.suffix in ('.hpp','.metal','.mm','.py','.md','.json'):shutil.copy2(path,dest/path.name)
 helper=Path('dev/benchmarks/prefill4k_allrows_qmv_one_layer.hpp')
 shutil.copy2(ROOT/helper,out/'source'/helper)
 reused=[]
 for r in seal['compiled_objects']:
  if Path(r['path']).name=='FlashWorker.o':continue
  src=PARENT/r['path']
  if sha(src)!=r['sha256']:raise ValueError('Parent host drift')
  d=out/'reused'/r['path'];d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,d);reused.append({'path':str(d.relative_to(out)),'sha256':sha(d),'source':str(src)})
 for r in manifest['frozen_inputs']:
  if '/core/' not in '/'+r['path']:continue
  src=PARENT/r['path']
  if sha(src)!=r['sha256']:raise ValueError('Parent Core drift')
  d=out/'reused'/r['path'];d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,d);reused.append({'path':str(d.relative_to(out)),'sha256':sha(d),'source':str(src)})
 if len(reused)!=53:raise ValueError('Exactly current53 nonWorker+Core closure required')
 original=next(PARENT/r['path'] for r in manifest['frozen_inputs'] if Path(r['path']).name.endswith('flash_gathered_mpp.air'))
 if sha(original)!=ORIGINAL_AIR_SHA:raise ValueError('Immutable original gathered AIR required')
 shutil.copy2(original,out/'original-gathered.air')
 metal=['xcrun','-sdk','macosx','metal','-std=metal4.1','-O3','-mmacosx-version-min=27.0',f'-I{out}/source/runtime',f'-I{dest}','-c',dest/'candidate.metal','-o',out/'candidate.air']
 if a.reuse_air_from:
  prior=a.reuse_air_from.resolve();priorseal=json.loads((prior/'CPU_READY.json').read_text());art={r['path']:r['sha256'] for r in priorseal['artifacts']}
  for name in ('candidate.metal','abi.hpp'):
   if sha(dest/name)!=sha(prior/'source'/PRIVATE/name):raise ValueError('Reused shader source differs')
  for name in ('candidate.air','candidate.metallib'):
   if sha(prior/name)!=art[name]:raise ValueError('Reused audited shader artifact drift')
   shutil.copy2(prior/name,out/name)
  link=priorseal['metallib_command'];metal=priorseal['metal_command']
 else:
  run(metal);link=['xcrun','-sdk','macosx','metallib',out/'original-gathered.air',out/'candidate.air','-o',out/'candidate.metallib'];run(link)
 flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1',f'-I{out}/source',f'-I{out}/source/runtime',f'-I{dest}']
 cc=['xcrun','-sdk','macosx','clang++',*flags,'-MMD','-MP','-c',dest/'oracle.mm','-o',out/'oracle.o'];run(cc)
 ld=['xcrun','-sdk','macosx','clang++',*flags,out/'oracle.o',*[out/x['path'] for x in reused],'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',out/'oracle'];run(ld)
 cpu=json.loads(run([out/'oracle','--cpu-self-test']).stdout)
 if cpu.get('GPU_work') is not False or cpu.get('payload_reads') is not False:raise ValueError('Pure CPU selftest required')
 dfile=out/'oracle.d';tokens=shlex.split(dfile.read_text().replace('\\\n',' ').splitlines()[0].split(':',1)[1]);deps=[]
 for token in tokens:
  path=Path(token).resolve()
  if ROOT in path.parents:
   if out/'source' not in path.parents:raise ValueError('External program source escaped frozen closure')
   deps.append({'path':str(path.relative_to(out)),'sha256':sha(path)})
 sources=[{'path':str(path.relative_to(out/'source')),'sha256':sha(path)} for path in sorted((out/'source').rglob('*')) if path.is_file()]
 parts={name:sha(dest/name) for name in ('candidate.metal','abi.hpp','oracle.mm','packing.hpp','PLAN.md')};parts.update({'parent_manifest':sha(PARENT/'overlay-manifest.json'),'original_AIR':ORIGINAL_AIR_SHA,'scope':'Root-only-bounded-oneexpert-synthetic-ranks0-R4-RHS-K64permutation-no-worker-v1'})
 identity=hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 artifacts=[{'path':name,'sha256':sha(out/name)} for name in ('oracle','oracle.o','candidate.air','candidate.metallib','original-gathered.air')]
 result={'schema':'bounded-expert-RHS-tile64-CPU-source-compiled-seal-v1','pass':True,'source_identity_sha256':identity,'identity_parts':parts,'parent':str(PARENT),'sources':sources,'reused53':reused,'compiler_dependencies':deps,'artifacts':artifacts,'compile_command':list(map(str,cc)),'link_command':list(map(str,ld)),'metal_command':list(map(str,metal)),'metallib_command':list(map(str,link)),'CPU_selftest':cpu,'GPU_work':False,'model_capture_payload_reads_or_hashes':False,'Root_coefficient_fixture_bytes':4930560,'whole_model_qualification':False,'Root_GPU_component_proof_pending':True,'AIR_independent_arithmetic_review_pending':True}
 (out/'CPU_READY.json').write_text(json.dumps(result,indent=2)+'\n')
 command={'schema':'Root-only-bounded-expert-RHS-tile64-component-command-v1','Root_GPU_only':True,'requires_independent_source_AIR_and_oracle_review':True,'argv':[str(out/'oracle'),str(out/'candidate.metallib'),str(ROOT/'build/prefill4k-fullcache-artifacts/int8-experts-all512-v1'),str(ROOT/'build/release/flash/sep22-expert-rhs-tile64-singleexpert-v1.json'),'0','0'],'scope':'Rootpread oneoriginalexpert4.93056MB/syntheticuniqueIDs-rank0; notFull512routing orwholemodel speed','CPU_READY_sha256':sha(out/'CPU_READY.json'),'artifact_pins':artifacts}
 (out/'root-command.json').write_text(json.dumps(command,indent=2)+'\n');print(json.dumps({'built':str(out),'source_identity':identity,'CPU_checks':cpu['checks'],'artifacts':artifacts,'GPU_work':False}))
if __name__=='__main__':main()
