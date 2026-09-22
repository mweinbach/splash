#!/usr/bin/env python3
"""Fresh strict-full oracle library reauth; no model/input/export payload access."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

p=argparse.ArgumentParser();p.add_argument('--base',default='build/trunkverify-exact-candidate-sep22-v3');p.add_argument('--worker',required=True);p.add_argument('--build',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[3];base=(root/a.base).resolve();worker=(root/a.worker).resolve();build=(root/a.build).resolve()
if build.exists():raise SystemExit('fresh oracle output required')
m=json.loads((base/'manifest.json').read_text());s=json.loads((worker/'compiled-cpu-seal.json').read_text())
if m['role']!='candidate' or len(m['objects'])!=53 or not s['pass'] or s['gpu_executed']:raise SystemExit('strict full candidate and CPU worker seal required')
artifacts={x['path']:x['sha256'] for x in s['artifacts']}
newlib=artifacts['splash.metallib']
if newlib!='390e67bb04e81c1aa291013f2de16fbca27bbf7771819b17af95aa7be4c88177':raise SystemExit('HC-fast library profile mismatch')
oldworker=Path(m['worker'])
for path in (oldworker/'source/runtime/flash').glob('*'):
 if path.is_file() and path.suffix in ['.hpp','.h','.cpp','.mm']:
  if path.read_bytes()!=(worker/'source/runtime/flash'/path.name).read_bytes():raise SystemExit('native source/header operation changed: '+path.name)
# Source/object SHA is authorized; no library/model/tensor/capture hash here.
oldseal=json.loads((oldworker/'compiled-cpu-seal.json').read_text());oldart={x['path']:x['sha256'] for x in oldseal['artifacts']}
newobjs={re.sub(r'^(?:\d+-)+','',Path(k).stem):v for k,v in artifacts.items() if k.endswith('.o')}
oldobjs={re.sub(r'^(?:\d+-)+','',Path(k).stem):v for k,v in oldart.items() if k.endswith('.o')}
if oldobjs!=newobjs or artifacts['splash-flash']!=oldart['splash-flash']:raise SystemExit('native object/worker identity changed')
shutil.copytree(base,build)
for stale in ['run-root-compare.sh','root-compare-command.json']:(build/stale).unlink(missing_ok=True)
old=str(base);new=str(build)
m['objects']=[{**x,'fast_reauth_object_source':x['path'],'path':x['path'].replace(old,new)} for x in m['objects']]
m['headers']=[{**x,'path':x['path'].replace(old,new)} for x in m['headers']]
for h in m['headers']:
 if hashlib.sha256(Path(h['path']).read_bytes()).hexdigest()!=h['sha256']:raise SystemExit('readonly header changed')
for obj in m['objects']:
 if Path(obj['path']).read_bytes()!=Path(obj['fast_reauth_object_source']).read_bytes():raise SystemExit('copied oracle object changed')
shutil.copyfile(worker/'splash.metallib',build/'splash.metallib')
m.update({'worker':str(worker),'metallib_sha256':newlib,'HC_fast_worker_executable_sha256':artifacts['splash-flash'],
 'HC_fast_original_53_objects_and_all_headers_unchanged':True,'HC_fast_only_library_recipe_reauth':True,
 'HC_fast_source_profile':json.loads((worker/'HC-fast-source-profile.json').read_text()),
 'parent_failed_oracle_untouched':str(base),'strict_full_qualification_unchanged':True,
 'Root_97role_fast_component_report':'build/release/flash/sep22-hc-pad-producer-r4-97chain-fast-v8.json',
 'Root_whole_trunk_verify_qualification_pending':True})
command=[x.replace(old,new) for x in m['compiler_command']];m['compiler_command']=command
summary={k:v for k,v in m.items() if k not in ['objects','header_census','headers','cpu','compiler_command']}
(build/'provenance.json').write_text(json.dumps(m,indent=2)+'\n')
(build/'TrunkVerifyBuildProvenance.hpp').write_text('#pragma once\ninline constexpr const char *kPrefillExactProvenancePath='+json.dumps(str(build/'provenance.json'))+';\ninline constexpr const char *kPrefillExactBuildProvenance=R"META('+json.dumps(summary,sort_keys=True)+')META";\n')
subprocess.run(command,cwd=root,check=True)
cpu=subprocess.check_output([str(build/'oracle'),'--cpu-only'],cwd=root,text=True);m['cpu']=json.loads(cpu)
(build/'cpu-self-test.json').write_text(cpu);(build/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'pass':True,'build':str(build),'strict_both_features_required':True,'operations_headers_53objects_unchanged':True,'new_library_pin':newlib,'cpu':m['cpu'],'GPU_started':False,'Root_new_oracle_pin_required':True}))
