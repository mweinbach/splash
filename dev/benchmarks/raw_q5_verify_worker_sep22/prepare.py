#!/usr/bin/env python3
"""CPU-only narrow rowpair worker; immutable qualified AIR reuse, no data reads."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;PRIVATE='dev/benchmarks/raw_q5_verify_worker_sep22'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(c):subprocess.run(c,cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args();b=a.build.resolve();base=ROOT/'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2';component=ROOT/'build/raw-q5-rowpair-sep22-component-v1'
 if b.exists():raise ValueError('fresh rowpair worker output required')
 seal=json.loads((base/'compiled-cpu-seal.json').read_text());native=json.loads((base/'Root-rawQ4-native-qualified.json').read_text());overlay=json.loads((base/'overlay-manifest.json').read_text())
 if not seal['pass']or not native['pass']or not native['qualification_complete']or sha(base/'splash-flash')!='663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438'or sha(base/'splash.metallib')!='7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8':raise ValueError('qualified current parent drift')
 air=component/'candidate.air';airSHA='99d57fd78957cfcc68ecc30d426c754b0dd702e42a32acef0008155234482a69'
 if sha(air)!=airSHA:raise ValueError('qualified immutable shipping candidate AIR differs')
 if sha(base/'Root-rawQ4-native-qualified.json')!='4d4ce0cee4e9c40351b443ec069162eb5f0098d0e2a392809bba0dd06d4c8818':raise ValueError('Root current parent26/54 receipt differs')
 if native['frames']!=26 or native['repeated_frames']!=54 or not native['backend_destroyed'] or native['rawQ4_calls']!=130 or native['rawQ4_rows']!=520:raise ValueError('completed current parent context proof missing')
 for record in overlay['files']:
  if sha(base/'source'/record['path'])!=record['sha256']:raise ValueError('sealed parent source drift:'+record['path'])
 for r in seal['compiled_objects']+seal['artifacts']:
  if sha(base/r['path'])!=r['sha256']:raise ValueError('actual current parent artifact drift:'+r['path'])
 shutil.copytree(base,b)
 for n in ['overlay-manifest.json','compiled-cpu-seal.json','Root-rawQ4-native-qualified.json']:(b/n).rename(b/('Q4-qualified-parent-'+n))
 own=b/'source'/PRIVATE;shutil.copytree(HERE,own,ignore=shutil.ignore_patterns('__pycache__','*.pyc','semantic_quality.py','test_semantic_quality.py'))
 sourcePolicy=b/'source/dev/benchmarks/raw_q5_rowpair_sep22';sourcePolicy.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'dev/benchmarks/raw_q5_rowpair_sep22/policy.hpp',sourcePolicy/'policy.hpp')
 spec=importlib.util.spec_from_file_location('rowpairOverlay',HERE/'overlay.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 journal=[]
 for n in ['runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm']:
  old=(base/'source'/n).read_text();mod.changes=[];new=mod.transform(n,old);restored=new
  for previous,replacement in reversed(mod.changes):
   if restored.count(replacement)!=1:raise ValueError('Q5 inverse overlay occurrence differs:'+n)
   restored=restored.replace(replacement,previous)
  restored=restored.removeprefix('#include "'+PRIVATE+'/policy.hpp"\n')
  if restored!=old:raise ValueError('Q5 hook journal fails exact parent restoration:'+n)
  (b/'source'/n).write_text(new);journal.append({'path':n,'parent_sha256':sha(base/'source'/n),'new_sha256':sha(b/'source'/n),'only_private_rowpair_selector_wrapper_status_change':True,'restores_parent_byte_exact':True,'replacements':mod.changes})
 parts={n:sha(own/n)for n in ['policy.hpp','overlay.py','policy_cpu.cpp','prepare.py','PLAN.json']};parts.update({'parent_source_identity':native['source_identity_sha256'],'parent_compiled_seal_sha256':sha(base/'compiled-cpu-seal.json'),'parent_current_native_receipt_sha256':sha(base/'Root-rawQ4-native-qualified.json'),'qualified_candidate_AIR_sha256':airSHA,'scope':'MAIN GDN output36 only; singleton VerifyR4 actualNULLtileRAW Q5G128 K6144N2560; allotherpaths unchanged; originalbacking'})
 identity=hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(',',':')).encode()).hexdigest();(own/'source_identity.hpp').write_text('#pragma once\nnamespace splash::flash::raw_q5_verify_sep22 {inline constexpr char kSourceIdentitySha256[]='+json.dumps(identity)+';}\n')
 flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(b/'source'),'-I'+str(b/'source/runtime'),'-I'+str(b/'source/dev/benchmarks/prefill4k_attention')]
 link=(b/'link-inputs.mk').read_text();names=next(l for l in link.splitlines()if l.startswith('REBUILD_NAMES :=')).split(':=',1)[1].split();srcs={m.group(1):m.group(2)for m in re.finditer(r'^SRC_(\S+) := \$\(BUILD\)/source/(.*)$',link,re.M)};cores=[b/t.removeprefix('$(BUILD)/')for t in next(l for l in link.splitlines()if l.startswith('CORE :=')).split(':=',1)[1].split()];objects=[b/'host'/(n+'.o')for n in names]
 if len(names)!=50 or len(cores)!=4:raise ValueError('exact50host/Core4 closure required')
 commands=[];census=[];opins=[]
 for name,obj in zip(names,objects):
  old=base/'host'/(name+'.o');source=b/'source'/srcs[name];c=['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(source)];r=subprocess.run(c,cwd=ROOT,check=True,capture_output=True,text=True);deps=r.stdout.replace('\\\n',' ').split();consumer=any(x.endswith('/raw_q5_verify_worker_sep22/policy.hpp')for x in deps)
  if consumer!=(name in ['FlashForward','FlashWorker']):raise ValueError('unexpected private header consumer:'+name)
  census.append({'object':name,'source':str(source),'rowpair_header_consumer':consumer,'dependencies':deps})
  if consumer:c=['xcrun','-sdk','macosx','clang++',*flags,'-MMD','-MP','-c',str(source),'-o',str(obj)];commands.append(c);run(c)
  elif sha(obj)!=sha(old):raise ValueError('unaffected host object changed')
  opins.append({'path':str(obj.relative_to(b)),'sha256':sha(obj),'parent_sha256':sha(old),'changed_for_private_header':consumer})
 for obj in cores:
  old=base/obj.relative_to(b)
  if sha(obj)!=sha(old):raise ValueError('Core4 changed')
  opins.append({'path':str(obj.relative_to(b)),'sha256':sha(obj),'parent_sha256':sha(old),'changed_for_private_header':False})
 # Reproduce exact ordered complete Q4 parent link before adding the single Q5 AIR.
 airs=[]
 for l in link.splitlines():
  if l.startswith('AIRS :=')or l.startswith('AIRS +='):airs += [b/t.removeprefix('$(BUILD)/')for t in l.split('=',1)[1].split()]
 airs.append(b/'hc-pad.air');airs.append(b/'rawQ4-qualified.air');baseline=['xcrun','-sdk','macosx','metallib',*map(str,airs),'-o',str(b/'baseline-relinked.metallib')];commands.append(baseline);run(baseline)
 if sha(b/'baseline-relinked.metallib')!=native['newlib_sha256']:raise ValueError('original ordered full AIR relink is not exact754')
 shutil.copy2(air,b/'rawQ5-qualified.air');c=['xcrun','-sdk','macosx','metallib',*map(str,airs),str(b/'rawQ5-qualified.air'),'-o',str(b/'splash.metallib')];commands.append(c);run(c)
 c=['xcrun','-sdk','macosx','clang++',*flags,*map(str,objects+cores),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'splash-flash')];commands.append(c);run(c)
 c=['xcrun','-sdk','macosx','clang++',*flags,str(own/'policy_cpu.cpp'),*map(str,[o for o,n in zip(objects,names)if n!='FlashWorker']+cores),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'rowpair-policy-CPU')];commands.append(c);run(c)
 cpu=subprocess.run([str(b/'rowpair-policy-CPU')],cwd=ROOT,check=True,text=True,capture_output=True);wc=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],cwd=ROOT,check=True,text=True,capture_output=True);(b/'policy-CPU.json').write_text(cpu.stdout);(b/'Worker-CPU.json').write_text(wc.stdout)
 env=dict(os.environ)
 for k in list(env):
  if k.startswith('SPLASH_'):del env[k]
 for k in ['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22','SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22','SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22','SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22','SPLASH_FLASH_ALLROWS_FULL512_TARGET','SPLASH_FLASH_QMV_F32','SPLASH_FLASH_FLOAT_DENSE_CACHE','SPLASH_FLASH_FLOAT_DENSE_SELECTIVE']:env[k]='1'
 env['SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21']='0';env['SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22']='1';env['SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22']='1';dependency=[]
 for missing in [None,*[k for k in env if k!='SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22'and k.startswith('SPLASH_')and env[k]=='1']]:
  e=dict(env)
  if missing:e[missing]='0'
  r=subprocess.run([str(b/'rowpair-policy-CPU'),'--dependency'],cwd=ROOT,env=e,capture_output=True,text=True)
  if (r.returncode==0)!=(missing is None):raise ValueError('strict dependency guard failed:'+str(missing))
  dependency.append({'missing':missing,'accepted':r.returncode==0})
 for value in ['0','1']:
  e=dict(env);e['SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22']=value;r=subprocess.run([str(b/'rowpair-policy-CPU'),'--lifetime'],cwd=ROOT,env=e,check=True,capture_output=True,text=True);dependency.append({'lifetime_initial':value,'mutation_rejected':json.loads(r.stdout)['pass']})
 e=dict(env);e.pop('SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22');
 for k in list(e):
  if k.startswith('SPLASH_'):del e[k]
 r=subprocess.run([str(b/'rowpair-policy-CPU'),'--dependency'],cwd=ROOT,env=e,check=True,capture_output=True,text=True);dependency.append({'default0_no_parent_dependencies_required':json.loads(r.stdout)['pass']})
 # Public headers and all nonchanged sources preserve the parent bytes.
 unchanged=[]
 for p in sorted((base/'source').rglob('*')):
  if p.is_file()and str(p.relative_to(base/'source'))not in ['runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm']:
   if sha(p)!=sha(b/'source'/p.relative_to(base/'source')):raise ValueError('unrelated source changed:'+str(p))
   unchanged.append(str(p.relative_to(base/'source')))
 art=[{'path':n,'sha256':sha(b/n)}for n in ['splash-flash','splash.metallib','rowpair-policy-CPU']]
 mf={'schema':'rawQ4Q5-GDN36-VerifyR4-narrow-worker-source-v1','base':str(base),'parent_source_identity':native['source_identity_sha256'],'source_identity_sha256':identity,'identity_parts':parts,'files':[{'path':str(p.relative_to(b/'source')),'sha256':sha(p)}for p in sorted((b/'source').rglob('*'))if p.is_file()],'changed_paths':['runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm'],'source_hook_journal':journal,'actual50TU_header_census':census,'private_header_consumers':['FlashForward','FlashWorker'],'compiled_objects':opins,'Core4_unchanged':True,'normal_GPU_allocation_bytes_added':0,'public_headers_changed':False,'same_binary_flag0_parent_addAffine_Q4_unchanged':True,'NULLtile_RAW_branch_only':True,'all_other_contexts_roles_excluded':True,'eligible_main_GDN_output_roles':36,'eligible_parent_main_GDN_QKV_roles':26,'original_ordered_AIR_relink_exact754':True,'qualified_candidate_AIR_sha256':airSHA,'new_AIR_count':1,'Metal_math_compiles':0,'parent_qualified_candidate_AIR_sha256':'4a671caf5641a451f5d60d5d2295b9064e4d78bf23effa74c5b86eaeeaec1003','new_combined_Root_native_receipt_required':'Root-rawQ4Q5-native-qualified.json','compiler_commands':commands,'artifacts':art,'CPU_policy':json.loads(cpu.stdout),'CPU_Worker':json.loads(wc.stdout),'dependency_cases':dependency,'GPU_work':False,'model_capture_payload_reads':0,'whole_state_qualified':False,'Root_independent_review_pending':True};(b/'overlay-manifest.json').write_text(json.dumps(mf,indent=2)+'\n')
 cp={'schema':'rawQ4Q5-current-narrow-worker-compiled-source-closure-v1','pass':True,'source_identity_sha256':identity,'compiled_objects':opins,'artifacts':art,'current_nonWorker_closure_count':53,'actual50TU_header_census':census,'GPU_work':False,'new_GPU_allocation_bytes':0,'whole_state_qualified':False};(b/'compiled-cpu-seal.json').write_text(json.dumps(cp,indent=2)+'\n')
 print(json.dumps({'CPU_complete':True,'build':str(b),'source_identity_sha256':identity,'worker_sha256':sha(b/'splash-flash'),'library_sha256':sha(b/'splash.metallib'),'CPU_policy':mf['CPU_policy'],'CPU_Worker':mf['CPU_Worker'],'private_consumers':['FlashForward','FlashWorker'],'GPU_work':False,'payload_reads':0}))
if __name__=='__main__':main()
