#!/usr/bin/env python3
"""CPU-only diagnostic relink; original qualification artifacts are unchanged."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser();p.add_argument('--base',default='build/trunkverify-exact-candidate-sep22-v3');p.add_argument('--build',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[3];base=(root/a.base).resolve();build=(root/a.build).resolve()
if build.exists():raise SystemExit('fresh diagnostic output required')
m=json.loads((base/'manifest.json').read_text())
if m['role']!='candidate' or len(m['objects'])!=53:raise SystemExit('exact candidate closure required')
shutil.copytree(base,build)
for stale in ['run-root-compare.sh','root-compare-command.json']:
    (build/stale).unlink(missing_ok=True)
private=build/'source/dev/benchmarks/trunk_verify_exact_sep22';oracle=private/'oracle.mm';s=oracle.read_text()
old='''  require(selected("SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22")==bool(SPLASH_VERIFY_CANDIDATE),"compact role mismatch");
  require(selected("SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22")==bool(SPLASH_VERIFY_CANDIDATE),"HC-pad role mismatch");'''
new='''  const bool compact=selected("SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22"),hc=selected("SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22");
  require(compact!=hc,"diagnostic requires exactly one feature enabled");'''
if s.count(old)!=1:raise SystemExit('strict full-policy anchor drift')
s=s.replace(old,new)
role='require(exporting!=bool(SPLASH_VERIFY_CANDIDATE),"binary role mismatch");'
if s.count(role)!=1:raise SystemExit('role anchor drift')
s=s.replace(role,role+'require(!exporting,"diagnostic comparator cannot export");')
s=s.replace('<<(exporting?"false":"true")','<<"false"')
s=s.replace('\\"scope\\":\\"TRUNKVerify exact boundedmatrix only\\"','\\"diagnostic_run\\":true,\\"diagnostic_feature\\":'+'''"<<json::quote(selected("SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22")?"compact_only":"hc_only")<<"'''+',\\"scope\\":\\"one-feature diagnosis against complete control; not combined qualification\\"')
s=s.replace('\\"pass\\":false,\\"qualification_complete\\":false,\\"stage\\":','\\"pass\\":false,\\"qualification_complete\\":false,\\"diagnostic_run\\":true,\\"role\\":\\"compare\\",\\"stage\\":')
oracle.write_text(s)
old_path=str(base);new_path=str(build)
m['objects']=[{**x,'diagnostic_reused_from':x['path'],'path':x['path'].replace(old_path,new_path)} for x in m['objects']]
m['headers']=[{**x,'path':x['path'].replace(old_path,new_path)} for x in m['headers']]
m['role']='diagnostic';m['qualification_complete']=False;m['diagnostic_only']=True;m['full_candidate_base']=str(base)
m['source_or_header_or_library_operation_changes']=False;m['strict_full_qualification_unchanged']=True
command=[x.replace(old_path,new_path) for x in m['compiler_command']];m['compiler_command']=command
(build/'provenance.json').write_text(json.dumps(m,indent=2)+'\n')
summary={k:v for k,v in m.items() if k not in ['objects','header_census','headers','cpu','compiler_command']}
(build/'TrunkVerifyBuildProvenance.hpp').write_text('#pragma once\ninline constexpr const char *kPrefillExactProvenancePath='+json.dumps(str(build/'provenance.json'))+';\ninline constexpr const char *kPrefillExactBuildProvenance=R"META('+json.dumps(summary,sort_keys=True)+')META";\n')
subprocess.run(command,cwd=root,check=True)
cpu=subprocess.check_output([str(build/'oracle'),'--cpu-only'],cwd=root,text=True);m['cpu']=json.loads(cpu)
(build/'cpu-self-test.json').write_text(cpu);(build/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'pass':True,'build':str(build),'diagnostic_only':True,'operations_headers_objects_library_reused':True,'cpu':m['cpu'],'GPU_started':False,'Root_artifact_pin_required':True}))
