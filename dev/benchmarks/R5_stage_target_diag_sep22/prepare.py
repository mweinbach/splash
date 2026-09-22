#!/usr/bin/env python3
"""CPU-only source/host composition over frozen current fixed4/R5; no inference."""
import argparse,difflib,hashlib,importlib.util,json,pathlib,re,shutil,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];HERE=pathlib.Path(__file__).resolve().parent
BASE=ROOT/'build/R5-integer-currentQ4-fixed4-sep22-worker-v2';PRIVATE=HERE.relative_to(ROOT)
EXE='6f7e22a2ca9c0c9728bf356391d2bde17c4e9bc90ab6295e625cac15e7987d68';LIB='dc1ab6f9178aac706bb408601fb734e9d508fb5c6c491732bc6ec4e36e6287e6'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--build',type=pathlib.Path,required=True);a=p.parse_args();b=a.build.resolve()
 if b.exists() or ROOT/'build' not in b.parents:raise ValueError('Fresh private build required')
 seal=json.loads((BASE/'compiled-cpu-seal.json').read_text());parent=json.loads((BASE/'overlay-manifest.json').read_text())
 if not seal['pass'] or sha(BASE/'splash-flash')!=EXE or sha(BASE/'splash.metallib')!=LIB:raise ValueError('Current qualified parent drift')
 for record in seal['compiled_objects']:
  if sha(BASE/record['path'])!=record['sha256']:raise ValueError('Current object drift')
 for record in parent['files']:
  if sha(BASE/'source'/record['path'])!=record['sha256']:raise ValueError('Current source/header drift')
 shutil.copytree(BASE,b);own=b/'source'/PRIVATE;shutil.copytree(HERE,own)
 for n in ['CPU_READY.json','compiled-cpu-seal.json','overlay-manifest.json']:(b/n).rename(b/('normal-parent-'+n))
 spec=importlib.util.spec_from_file_location('_privateR5stageOverlay',HERE/'overlay.py');overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
 journal=[]
 for rel in ['runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm']:
  old=(BASE/'source'/rel).read_text();new=overlay.transform(rel,old);(b/'source'/rel).write_text(new)
  edits=[];before,after=old.splitlines(keepends=True),new.splitlines(keepends=True)
  for tag,i,j,x,y in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
   if tag!='equal':edits.append({'old_start':i,'old_end':j,'new_start':x,'new_end':y,'old_lines':before[i:j],'new_lines':after[x:y]})
  restored=list(after)
  for edit in reversed(edits):restored[edit['new_start']:edit['new_end']]=edit['old_lines']
  if restored!=before:raise ValueError('Normal floating body inverse differs')
  journal.append({'path':rel,'parent_sha256':sha(BASE/'source'/rel),'sha256':sha(b/'source'/rel),'inverse_literal_exact':True,'edits':edits})
 flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(b/'source/runtime'),'-I'+str(b/'source'),'-I'+str(b/'source/dev/benchmarks/prefill4k_attention')]
 link=(b/'link-inputs.mk').read_text();names=next(x for x in link.splitlines() if x.startswith('REBUILD_NAMES :=')).split(':=')[1].split();srcs={m.group(1):m.group(2) for m in re.finditer(r'^SRC_(\S+) := \$\(BUILD\)/source/(.*)$',link,re.M)}
 cores=[b/t.removeprefix('$(BUILD)/') for t in next(x for x in link.splitlines() if x.startswith('CORE :=')).split(':=')[1].split()];objs=[b/'host'/(n+'.o') for n in names]
 if len(names)!=50 or len(cores)!=4:raise ValueError('Current50host/Core4 closure required')
 commands=[];census=[];pins=[]
 for n,obj in zip(names,objs):
  src=b/'source'/srcs[n];dep=subprocess.run(['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(src)],cwd=ROOT,capture_output=True,text=True,check=True).stdout.replace('\\\n',' ').split();consumer=any(x.endswith('/R5_stage_target_diag_sep22/bridge.hpp') for x in dep)
  if consumer!=(n in ('FlashForward','FlashWorker')):raise ValueError('Private diagnostic header consumer drift:'+n)
  census.append({'object':n,'diagnostic_consumer':consumer,'dependencies':dep})
  if consumer:
   c=['xcrun','-sdk','macosx','clang++',*flags,'-MMD','-MP','-c',str(src),'-o',str(obj)];commands.append(c);subprocess.run(c,cwd=ROOT,check=True)
  elif sha(obj)!=sha(BASE/obj.relative_to(b)):raise ValueError('Unaffected host changed')
  pins.append({'path':str(obj.relative_to(b)),'sha256':sha(obj),'parent_sha256':sha(BASE/obj.relative_to(b)),'changed':consumer})
 for obj in cores:
  if sha(obj)!=sha(BASE/obj.relative_to(b)):raise ValueError('Core4 changed')
  pins.append({'path':str(obj.relative_to(b)),'sha256':sha(obj),'parent_sha256':sha(BASE/obj.relative_to(b)),'changed':False})
 c=['xcrun','-sdk','macosx','clang++',*flags,*map(str,objs+cores),'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/'splash-flash')];commands.append(c);subprocess.run(c,cwd=ROOT,check=True)
 if sha(b/'splash.metallib')!=LIB:raise ValueError('Diagnostic changed shader library')
 result=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],cwd=ROOT,capture_output=True,text=True,check=True)
 r={'schema':'current-fixed4-R5-StagePerDispatch-private-host-diagnostic-CPU-v1','pass':True,'GPU_work':False,'model_token_capture_or_response_payload_reads':0,'parent':str(BASE),'parent_exe_sha256':EXE,'library_sha256':LIB,'exe_sha256':sha(b/'splash-flash'),'changed_paths':[x['path'] for x in journal],'private_header_consumers':['FlashForward','FlashWorker'],'actual50TU_census':census,'compiled54_objects':pins,'Core4_unchanged':True,'new_floating_or_integer_AIR_compiles':0,'whole_normal_runtime_or_quality_qualified':False,'profiling_diagnostic_not_canonical_performance':True,'CPU_self_test':json.loads(result.stdout),'journal':journal,'compiler_commands':commands,'files':[{'path':str(x.relative_to(b/'source')),'sha256':sha(x)} for x in sorted((b/'source').rglob('*')) if x.is_file()]}
 for n in ['compiled-cpu-seal.json','CPU_READY.json','overlay-manifest.json']:(b/n).write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'build':str(b),'CPU_READY_sha256':sha(b/'CPU_READY.json'),'exe_sha256':r['exe_sha256'],'library_sha256':LIB,'GPU_work':False}))
if __name__=='__main__':main()
