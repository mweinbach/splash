#!/usr/bin/env python3
"""Fresh oracle-only relink; guarded header/hook/object closure stays byte-equal."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import hashlib

p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--build',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[3];here=Path(__file__).resolve().parent
base=(root/a.base).resolve();build=(root/a.build).resolve()
if build.exists():raise SystemExit('fresh output required')
m=json.loads((base/'manifest.json').read_text());shutil.copytree(base,build)
private=build/'source/dev/benchmarks/trunk_verify_exact_sep22'
for name in ['inspect.hpp','inspection.cpp.inc']:
 if (here/name).read_bytes()!=(private/name).read_bytes():raise SystemExit('header/hook changed; full rebuild required')
shutil.copyfile(here/'oracle.mm',private/'oracle.mm');shutil.copyfile(here/'relink.py',private/'relink.py')
old=str(base);new=str(build)
m['objects']=[{**x,'prepared_object_source':x['path'],'path':x['path'].replace(old,new)} for x in m['objects']]
m['headers']=[{**x,'path':x['path'].replace(old,new)} for x in m['headers']]
for h in m['headers']:
 if hashlib.sha256(Path(h['path']).read_bytes()).hexdigest()!=h['sha256']:raise SystemExit('copied header drift')
m['prepared_header_and_object_closure_reused_from']=str(base);m['fresh_oracle_only_relink']=True
command=[x.replace(old,new) for x in m['compiler_command']]
m['compiler_command']=command
summary={k:v for k,v in m.items() if k not in ['objects','header_census','headers','cpu','compiler_command']}
(build/'provenance.json').write_text(json.dumps(m,indent=2)+'\n')
(build/'TrunkVerifyBuildProvenance.hpp').write_text('#pragma once\ninline constexpr const char *kPrefillExactProvenancePath='+json.dumps(str(build/'provenance.json'))+';\ninline constexpr const char *kPrefillExactBuildProvenance=R"META('+json.dumps(summary,sort_keys=True)+')META";\n')
subprocess.run(command,cwd=root,check=True)
cpu=subprocess.check_output([str(build/'oracle'),'--cpu-only'],cwd=root,text=True);m['cpu']=json.loads(cpu)
(build/'cpu-self-test.json').write_text(cpu);(build/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'pass':True,'build':str(build),'role':m['role'],'objects':len(m['objects']),'header_TU_census':len(m['header_census']),'cpu':m['cpu'],'GPU_started':False,'Root_artifact_pin_required':True}))
