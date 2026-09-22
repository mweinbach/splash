#!/usr/bin/env python3
"""Private CPU-only actual-projection replay build. Never loads device/model/data."""
import argparse,hashlib,json,shlex,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
 if out.exists():raise ValueError('fresh private component directory required')
 parent=ROOT/'build/batch-prefill-restored-teacher-clock-sep22-worker-v1';packed=ROOT/'build/batch-prefill-twopass-restored-sep22-worker-v3'
 seal=json.loads((parent/'compiled-cpu-seal.json').read_text());newseal=json.loads((packed/'compiled-cpu-seal.json').read_text())
 assert sha(packed/'compiled-cpu-seal.json')=='50f4a427e9847428ca0ee26af3e7b091a39f4d124ae736e518b4b619991cc291'
 out.mkdir(parents=True);(out/'source').mkdir()
 sources={'component_oracle.mm':HERE/'diagnostic_oracle.mm','component_helpers.mm':ROOT/'dev/benchmarks/prefill_qsa_twopass_sep21/oracle.mm','base_oracle.mm':ROOT/'build/prefill-qsa-twopass-sep21/source/experiment/base_oracle.mm','batch_policy.hpp':packed/'source/dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp'}
 for dest,src in sources.items():
  if dest=='component_helpers.mm':
   helpers=src.read_text();assert helpers.count('\nint main(int argc,char **argv) {')==1
   (out/'source'/dest).write_text(helpers.split('\nint main(int argc,char **argv) {')[0]+'\n')
  else:shutil.copy2(src,out/'source'/dest)
 objects=[]
 object_hashes={e['object']:e['sha256']for e in seal['compiled_objects']};object_hashes.update({e['path']:e['sha256']for e in seal['core_objects']})
 source_hashes={e['path']:e['sha256']for e in seal['source_files']}
 for rel,h in object_hashes.items():
  if not rel.endswith('.o')or rel=='host/FlashWorker.o':continue
  assert sha(parent/rel)==h;objects.append({'path':str(parent/rel),'sha256':h})
 assert len(objects)==53
 assert sha(parent/'splash.metallib')=='1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf'
 flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1']
 flags+=['-I'+str(q)for q in [out/'source',parent/'source',parent/'source/runtime',parent/'source/dev/benchmarks/prefill4k_attention',parent/'source/dev/benchmarks/prefill_qsa_twopass_sep21']]
 compile_cmd=['xcrun','-sdk','macosx','clang++',*flags,'-MMD','-MP','-c',str(out/'source/component_oracle.mm'),'-o',str(out/'component-oracle.o')]
 subprocess.run(compile_cmd,check=True)
 link_cmd=['xcrun','-sdk','macosx','clang++',*flags,str(out/'component-oracle.o'),*[r['path']for r in objects],'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(out/'actual-batch-qsa-component-oracle')]
 subprocess.run(link_cmd,check=True)
 cpu=json.loads(subprocess.check_output([str(out/'actual-batch-qsa-component-oracle'),'--cpu-self-test'],text=True));assert cpu['pass']and cpu['gpu_commands']==0
 tokens=shlex.split((out/'component-oracle.d').read_text().replace('\\\n',' ').splitlines()[0].split(':',1)[1]);deps=[]
 for token in tokens:
  q=Path(token).resolve()
  if parent/'source'in q.parents:
   rel=q.relative_to(parent/'source').as_posix();assert rel in source_hashes and sha(q)==source_hashes[rel];deps.append({'path':str(q),'sha256':sha(q)})
  elif out/'source'in q.parents:deps.append({'path':str(q),'sha256':sha(q)})
  elif ROOT in q.parents:raise AssertionError('unsealed live source dependency:'+str(q))
 d={'schema':'actual-batch-qsa-diagnostic-CPU-closure-v1','diagnostic_only_no_warm_or_timing':True,'derived_gpu_math_from_frozen_replay_v7':True,'pass':True,'CPU_READY':True,'GPU_executed':False,'model_or_capture_payload_read':False,'whole_GPU_qualified':False,'parent_seal_sha256':sha(parent/'compiled-cpu-seal.json'),'packed_policy_seal_sha256':sha(packed/'compiled-cpu-seal.json'),'library':{'path':str(parent/'splash.metallib'),'sha256':sha(parent/'splash.metallib')},'component_helpers_transform':'only omit original standalone main; all numeric/helper definitions literal exact prefix','source_inputs':[{'path':str(src),'sha256':sha(src),'frozen_path':str(out/'source'/dest),'frozen_sha256':sha(out/'source'/dest)}for dest,src in sources.items()],'objects':objects,'actual_dependencies':deps,'compile_command':compile_cmd,'link_command':link_cmd,'binary_sha256':sha(out/'actual-batch-qsa-component-oracle'),'oracle_object_sha256':sha(out/'component-oracle.o'),'CPU_checks':cpu,'numerical_envelopes':'literal unchanged qualified component helpers; additionally each real BF16 row gated','Root_remaining':['capture manifest actual provenance/hash/extent','Root-only GPU replay all numeric gates','current-header whole-model HEAD/MAIN/state/sem22/performance qualification separately']}
 (out/'CPU_READY.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'path':str(out),'binary_sha256':d['binary_sha256'],'CPU_READY_sha256':sha(out/'CPU_READY.json'),'CPU':cpu}))
if __name__=='__main__':main()
