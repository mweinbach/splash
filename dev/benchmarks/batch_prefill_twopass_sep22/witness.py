#!/usr/bin/env python3
"""CPU/source closure proof only. Never executes model/device/capture commands."""
import argparse,hashlib,importlib.util,json,os,shlex,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=ROOT/'build/batch-prefill-twopass-restored-sep22-worker-v2');a=p.parse_args();b=a.build.resolve();m=json.loads((b/'overlay-manifest.json').read_text());built=json.loads((b/'compiled-build.json').read_text());base=Path(m['base']);checks=[]
 def require(value,label):
  if not value:raise ValueError(label)
  checks.append(label)
 for e in m['files']:require(sha(b/'source'/e['path'])==e['sha256'],'source:'+e['path'])
 spec=importlib.util.spec_from_file_location('frozenBatchPrepare',b/'machinery/prepare.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 for file,fn in [('FlashBatchPrefill.cpp',t.transform_batch),('FlashWorker.mm',t.transform_worker)]:require(fn((base/'source/runtime/flash'/file).read_text())==(b/'source/runtime/flash'/file).read_text(),'literal bounded transform:'+file)
 for e in m['files']:
  if e.get('new')or e.get('changed'):continue
  require((base/'source'/e['path']).read_bytes()==(b/'source'/e['path']).read_bytes(),'all other baseline source literal:'+e['path'])
 require((base/'source/runtime/flash/FlashForward.cpp').read_bytes()==(b/'source/runtime/flash/FlashForward.cpp').read_bytes(),'entireB1Forward exact')
 require(sha(b/'splash.metallib')==m['metallib_sha256']=='1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf','exact restoredlibrary')
 artifacts={};dependencies={};known={e['path']for e in m['files']}
 for e in built['objects']:
  require(sha(b/e['path'])==e['sha256'],'object:'+e['path']);artifacts[e['path']]=e['sha256'];owned=[]
  tokens=shlex.split((b/e['path']).with_suffix('.d').read_text().replace('\\\n',' ').splitlines()[0].split(':',1)[1])
  for token in tokens:
   path=Path(token);path=(ROOT/path).resolve()if not path.is_absolute()else path.resolve()
   if b/'source'in path.parents:
    rel=path.relative_to(b/'source').as_posix();require(rel in known,'manifested dependency:'+e['path']+':'+rel);owned.append(rel)
   elif ROOT in path.parents:raise ValueError('live/parent source dependency:'+str(path))
  dependencies[e['path']]=sorted(set(owned))
 for e in m['Core']:require(sha(b/e['path'])==e['sha256'],'Core:'+e['path']);artifacts[e['path']]=e['sha256']
 require(len(artifacts)==54 and len(dependencies)==50,'actual50host/Core4 closure')
 raw=(b/'source/dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp').read_bytes().replace(m['source_policy_sha256'].encode(),b'BATCH_SOURCE_SHA_PLACEHOLDER')
 require(hashlib.sha256(raw+(base/'source/runtime/flash/FlashBatchPrefill.cpp').read_bytes()+(base/'source/dev/benchmarks/prefill_qsa_twopass_sep21/twopass.cpp').read_bytes()).hexdigest()==m['source_policy_sha256'],'static sourcepolicy material bound')
 worker=(b/'source/runtime/flash/FlashWorker.mm').read_text();require(worker.index('batch_prefill_twopass_sep22::requested();')<worker.index('std::filesystem::canonical(argv[2])'),'strict requested beforeconfig/path/backend')
 status=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],text=True,capture_output=True,check=True);cpu=json.loads(status.stdout);require(cpu['valid']and not cpu['gpu_work']and len(cpu['checks'])==49,'actualWorker49CPU')
 with tempfile.TemporaryDirectory(prefix='batchTwoPassPolicyCPU-')as temp:
  binary=Path(temp)/'policy-cpu';subprocess.run(['xcrun','-sdk','macosx','clang++','-std=c++20','-O3','-Wno-deprecated-declarations','-I'+str(b/'source'),'-I'+str(b/'source/runtime'),'-I'+str(b/'source/dev/benchmarks/prefill4k_attention'),'-I'+str(HERE),str(HERE/'policy_cpu.cpp'),'-o',str(binary)],check=True)
  policy=json.loads(subprocess.check_output([str(binary)],text=True));require(policy['pass']and not policy['GPU_work'],'full eligibility/arena/strictCPU policy')
 # These refusals occur before canonicalizing an intentionally nonexistent package.
 env={k:v for k,v in os.environ.items()if not k.startswith('SPLASH_FLASH_')};guards=[]
 for name,value,error in [('badFlag','2','must be 0 or 1'),('missingDependency','1','requires SPLASH_FLASH_BATCH_PREFILL=1')]:
  run=subprocess.run([str(b/'splash-flash'),'serve-flash-native','/nonexistent-private-batch-twopass-cpu-test','16384','auto'],env={**env,'SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22':value},capture_output=True,text=True)
  require(run.returncode==3 and error in run.stderr,'compiledprepath refusal:'+name);guards.append({'case':name,'stderr':run.stderr.strip()})
 for name in ['build.py','witness.py','policy_cpu.cpp','PLAN.md','CAPTURE_ORACLE.md']:
  path=b/'machinery'/name;path.write_bytes((HERE/name).read_bytes());artifacts['machinery/'+name]=sha(path)
 for name in ['splash-flash','splash.metallib','machinery/prepare.py','machinery/policy.hpp','overlay-manifest.json','compiled-build.json']:artifacts[name]=sha(b/name)
 data={'schema':'current-batch-packedV-twoPass-CPU-source-closure-seal-v1','pass':True,'GPU_executed':False,'model_capture_payload_read':False,'whole_GPU_qualified':False,'source_manifest_sha256':sha(b/'overlay-manifest.json'),'source_sha256':{e['path']:e['sha256']for e in m['files']},'artifact_sha256':artifacts,'source_policy_sha256':m['source_policy_sha256'],'all50host_rebuilt':True,'Core4_only_reused':True,'objects':54,'actual_dependencies':dependencies,'CPU_worker':cpu,'CPU_policy':policy,'compiled_early_refusals':guards,'checks':checks,'Root_actual_capture_QSA_and_quality_proofs_still_required':True}
 (b/'compiled-cpu-seal.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'objects':54,'CPU_policy':policy,'sealSHA':sha(b/'compiled-cpu-seal.json'),'GPU_work':False}))
if __name__=='__main__':main()
