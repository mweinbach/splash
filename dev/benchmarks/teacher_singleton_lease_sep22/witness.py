#!/usr/bin/env python3
"""CPU-only frozen source, dependency, startup and saved-quality witness."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,shlex,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[3]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--build',type=Path,default=ROOT/'build/teacher-singleton-lease-sep22-worker-v1');a=ap.parse_args();b=a.build.resolve();mp=b/'overlay-manifest.json';m=json.loads(mp.read_text());base=Path(m['base']);checks=[]
 def require(ok,name):
  if not ok:raise ValueError(name)
  checks.append(name)
 records={r['path']:r for r in m['files']}
 for rel,r in records.items():require(sha((b/'source'/rel).read_bytes())==r['sha256'],'source:'+rel)
 spec=importlib.util.spec_from_file_location('privateLeaseTransform',b/'machinery/prepare.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 changed=[]
 for rel,r in records.items():
  if r.get('new'):continue
  before=(base/'source'/rel).read_bytes();after=(b/'source'/rel).read_bytes()
  require(sha(before)==r['parent_sha256'],'parentSourcePin:'+rel)
  require(t.transform(rel,before.decode()).encode()==after,'exactBoundedTransform:'+rel)
  if before!=after:changed.append(rel)
 require(sorted(changed)==['runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm'],'OnlyGetterAndWorkerSourceChanged')
 # Removing getter owner selection and new include makes ENTIRE Forward byte-identical.
 before=(base/'source/runtime/flash/FlashForward.cpp').read_text();after=(b/'source/runtime/flash/FlashForward.cpp').read_text()
 after=after.removeprefix('#include "dev/benchmarks/teacher_singleton_lease_sep22/policy.hpp"\n')
 after=after.replace('teacher_singleton_lease_sep22::persistentBF16(*impl_->denseCache)','impl_->denseCache->persistedWeightBuffers()')
 after=after.replace('teacher_singleton_lease_sep22::persistentF32(impl_->weights,*impl_->floatDenseCache)','impl_->floatDenseCache->persistedWeightBuffers()')
 require(after==before,'EntireForwardArithmeticConstructorPresenceByteIdenticalAfterOnlyGetterNormalization')
 before=(base/'source/runtime/flash/FlashWorker.mm').read_text();after=(b/'source/runtime/flash/FlashWorker.mm').read_text()
 start='      << R"(,"target_numerical_derivative_sha256":)"';end='      << R"(,"prefill_qsa_twopass_enabled":)"'
 require(before[before.index(start):before.index(end,before.index(start))]==after[after.index(start):after.index(end,after.index(start))],'NumericalIdentityBodyByteIdentical')
 for name in ['FlashFloatDenseCache.cpp','FlashDenseCache.cpp','FlashWeights.mm','FlashWeights.hpp','FlashForward.hpp','FlashMTP.cpp','FlashMTP.hpp','FlashBatchForward.cpp','FlashBatchPrefill.cpp']:
  require((base/'source/runtime/flash'/name).read_bytes()==(b/'source/runtime/flash'/name).read_bytes(),'AllBackingAndBatchSourceUnchanged:'+name)
 header=(b/'source/dev/benchmarks/teacher_singleton_lease_sep22/policy.hpp').read_bytes();raw=header.replace(m['source_policy_sha256'].encode(),b'SOURCE_SHA_PLACEHOLDER');old=(b/'source/dev/benchmarks/dense_w8a8_residency_sep21/worker_bridge.hpp').read_bytes()
 require(sha(raw+(base/'source/runtime/flash/FlashForward.cpp').read_bytes()+old)==m['source_policy_sha256'],'NewResourceSourcePolicyMaterialBound')
 require(sha((b/'splash.metallib').read_bytes())==m['metallib_sha256'],'ExactParentMetallib')
 require(len(m['rebuild'])==50 and len(m['frozen_core'])==4,'All50HostAnd4CoreClosure')
 artifact={}
 for r in m['frozen_core']:
  require(sha((b/r['path']).read_bytes())==r['sha256'],'OriginalCore:'+r['path']);artifact[r['path']]=r['sha256']
 deps={}
 for r in m['rebuild']:
  obj=b/'host'/(r['object']+'.o');dp=obj.with_suffix('.d');require(obj.exists()and dp.exists(),'ObjectAndActualDependencies:'+r['object']);artifact[obj.relative_to(b).as_posix()]=sha(obj.read_bytes())
  tokens=shlex.split(dp.read_text().replace('\\\n',' ').splitlines()[0].split(':',1)[1]);owned=[]
  for token in tokens:
   p=Path(token);p=p.resolve()if p.is_absolute()else(ROOT/p).resolve()
   if b/'source'in p.parents:
    rel=p.relative_to(b/'source').as_posix();require(rel in records,'FrozenManifestedDependency:'+r['object']+':'+rel);owned.append(rel)
   elif ROOT in p.parents:raise ValueError('LiveRepoDependency:'+str(p))
  deps[obj.relative_to(b).as_posix()]=sorted(set(owned))
 actual=subprocess.run(['make','-pqn','-rR','-f',str(b/'machinery/worker.mk'),f'BUILD={b}',str(b/'splash-flash')],cwd=ROOT,capture_output=True,text=True)
 line=[s for s in actual.stdout.splitlines()if s.startswith(str(b/'splash-flash')+':')];require(len(line)==1,'ActualLinkSingleTarget')
 objs={Path(p).resolve().relative_to(b).as_posix()for p in shlex.split(line[0].split(':',1)[1])if p.endswith('.o')}
 require(len(objs)==54 and objs==set(artifact),'ActualLinkExactly54SealedObjects')
 cpu=subprocess.run([str(b/'splash-flash'),'--cpu-self-test'],capture_output=True,text=True,check=True);cpuJSON=json.loads(cpu.stdout);require(cpuJSON['valid']and len(cpuJSON['checks'])==49,'Worker49CPUSelfChecks')
 env=json.loads((b/'root-mtp-environment.json').read_text());ambient={k:v for k,v in os.environ.items()if not k.startswith('SPLASH_FLASH_')};guards=[]
 cases=[('newValid',{},'manifest could not be opened'),('oldFlag0',{'SPLASH_FLASH_TEACHER_SINGLETON_LEASE_PRUNE_SEP22':'0'},'manifest could not be opened'),('invalidFlag',{'SPLASH_FLASH_TEACHER_SINGLETON_LEASE_PRUNE_SEP22':'2'},'must be 0 or 1'),('missingSaved',{'SPLASH_FLASH_SAVED_OPERANDS_RESIDENT':'0'},'requires SPLASH_FLASH_SAVED_OPERANDS_RESIDENT=1'),('missingW8',{'SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21':'0'},'requires SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21=1'),('missingF32',{'SPLASH_FLASH_FLOAT_DENSE_CACHE':'0'},'requires SPLASH_FLASH_FLOAT_DENSE_CACHE=1'),('missingAllrows',{'SPLASH_FLASH_ALLROWS_FULL512_TARGET':'0'},'requires SPLASH_FLASH_ALLROWS_FULL512_TARGET=1'),('forbiddenOriginalText',{'SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT':'1'},'forbids SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT=1'),('forbiddenPhase',{'SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21':'1'},'forbids SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21=1'),('forbiddenIdle',{'SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE':'1'},'forbids SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=1')]
 with tempfile.TemporaryDirectory(prefix='teacherLeaseCPU-')as temp:
  tempRoot=Path(temp).resolve();package=tempRoot/'package';package.mkdir();store=tempRoot/'store';store.mkdir();(package/'config.json').write_bytes((ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1/config.json').read_bytes())
  for label,changes,marker in cases:
   result=subprocess.run([str(b/'splash-flash'),'serve-flash-native',str(package),'16384','auto'],env={**ambient,**env,**changes,'SPLASH_FLASH_INT8_EXPERT_STORE':str(store)},capture_output=True,text=True)
   require(result.returncode==3 and marker in result.stderr,'CompiledPreBackendGuard:'+label);guards.append({'case':label,'stderr':result.stderr.strip()})
 sys.path.insert(0,str(ROOT));from dev.benchmarks import teacher_singleton_lease_quality as q
 require(q.SOURCE_SHA==m['source_policy_sha256'],'StrictQualityHelperBindsExactPolicySource')
 report=json.loads((ROOT/'build/release/flash/sep21-teacher-bulk-ab-qsa-model-and-quality-v1.json').read_text());status=copy.deepcopy(report['server_runs'][0]['final_status'])
 require(not q.status_errors(status),'LegacyQualifiedParentStatusAcceptedUnchanged')
 status['identity'].update({'teacher_singleton_lease_profile':q.PROFILE,'teacher_singleton_lease_source_sha256':q.SOURCE_SHA,'teacher_singleton_lease_source_policy':q.SOURCE_POLICY,'teacher_singleton_lease_enabled':True})
 lease=status['saved_operands_residency'];lease.update({'requested_view_count':807,'requested_view_bytes':131005546496,'registered_base_allocation_count':807,'registered_base_allocation_bytes':131005546496})
 status['teacher_singleton_lease']={'requested':True,'retained_F32_owner_count':118,'retained_F32_owner_bytes':3247964160,'omitted_F32_owner_count':390,'omitted_F32_owner_bytes':11143741440,'retained_BF16_owner_count':425,'retained_BF16_owner_bytes':4692377600,'omitted_BF16_owner_count':84,'omitted_BF16_owner_bytes':3774873600,'derived_W8_owner_count':168,'derived_W8_owner_bytes':1890975744,'all_backing_retained':True,'all_backing_charged':True,'selection_only':True,'qualification_scope':q.SCOPE}
 require(not q.status_errors(status),'New807SavedParentFixtureAccepted')
 fixture=b/'quality-saved-status-fixture.json';fixture.write_text(json.dumps(status,indent=2)+'\n');tamper=[]
 for group,keys in {'identity':['teacher_singleton_lease_profile','teacher_singleton_lease_source_sha256','teacher_singleton_lease_source_policy','teacher_singleton_lease_enabled','target_numerical_derivative_sha256','target_all_rows_full512','original_target_gpu_omitted','dense_w8a8_prefill_enabled','dense_w8a8_immutable_buffer_count'], 'saved_operands_residency':['requested','active','registered_base_allocation_count','registered_base_allocation_bytes','backing_already_charged','physical_pinning_verified'],'persisted_operands':['f32_tensors','f32_mapped_payload_bytes','bf16_tensors','bf16_mapped_payload_bytes','store_manifest_sha256'],'teacher_singleton_lease':list(status['teacher_singleton_lease'])}.items():
  for key in keys:
   bad=copy.deepcopy(status);v=bad[group][key];bad[group][key]=not v if type(v)is bool else v+1 if type(v)is int else v+'bad'
   require(bool(q.status_errors(bad)),'ExactProfileSingleFieldTamper:'+group+'.'+key);tamper.append(group+'.'+key)
 inactive=copy.deepcopy(status);inactive['identity'].update({'teacher_singleton_lease_profile':None,'teacher_singleton_lease_source_sha256':None,'teacher_singleton_lease_source_policy':None,'teacher_singleton_lease_enabled':False});inactive['teacher_singleton_lease']['requested']=False
 require(not q.status_errors(inactive),'Flag0InactiveResourceProfileAccepted')
 old=report['server_runs'][0]['final_status']['saved_operands_residency'];require(old['registered_base_allocation_count']==1281 and old['registered_base_allocation_bytes']==145924161536,'Flag0LegacySavedFixture1281Retained')
 for name in ['splash-flash','splash.metallib','quality-saved-status-fixture.json','source-census-plan.json','root-model-command.txt','root-mtp-environment.json','machinery/prepare.py','machinery/worker.mk','machinery/policy.hpp','machinery/census_plan.py']:
  artifact[name]=sha((b/name).read_bytes())
 for name in ['prefill4k_attribution_quality.py','teacher_singleton_lease_quality.py','splash_tuning_sep21.py']:
  rel='review-tools/'+name;p=ROOT/'dev/benchmarks'/name;(b/rel).parent.mkdir(parents=True,exist_ok=True);(b/rel).write_bytes(p.read_bytes());artifact[rel]=sha(p.read_bytes())
 for name in ['witness.py','README.md']:
  rel='machinery/'+name;data=(Path(__file__).parent/name).read_bytes();(b/rel).write_bytes(data);artifact[rel]=sha(data)
 for source,dest in [(ROOT/'dev/tests/flash/test_teacher_singleton_lease_quality.py','review-tools/test_teacher_singleton_lease_quality.py'),(b/'quality-cpu-tests.log','quality-cpu-tests.log'),(b/'CPU-synthetic807-saved22.json','CPU-synthetic807-saved22.json'),(b/'CPU-synthetic807-saved22-comparison.json','CPU-synthetic807-saved22-comparison.json')]:
  if source.exists():
   if source!=b/dest:(b/dest).write_bytes(source.read_bytes())
   artifact[dest]=sha(source.read_bytes())
 require((b/'quality-cpu-tests.log').exists()and 'Ran 32 tests' in (b/'quality-cpu-tests.log').read_text()and (b/'quality-cpu-tests.log').read_text().rstrip().endswith('OK'),'32QualityCPUtestsPass')
 comparison=json.loads((b/'CPU-synthetic807-saved22-comparison.json').read_text())
 require(comparison['valid']and comparison['no_new_task_regressions']and comparison['cpu_only'] is True,'Saved22StrictFormalComparisonCPUOnlyPass')
 data={'schema':'pure-I8-teacher-lease807-compiled-cpu-seal-v1','pass':True,'GPU_executed':False,'model_payload_bytes_read':0,'source_manifest_sha256':sha(mp.read_bytes()),'source_sha256':{rel:r['sha256']for rel,r in records.items()},'artifact_sha256':artifact,'effective_objects':54,'all50HostRecompiled':True,'compiler_dependency_closure':deps,'compiled_pre_backend_guards':guards,'CPU_worker':cpuJSON,'strict_quality_single_field_tampers':tamper,'all_Prefill_Decode_Verify_batch_arithmetic_backing_constructors_and_planning_unchanged':True,'numerical_derivative_changed':False,'new807LeaseOnly':True,'physical_pinning_or_hardware_schedule_claimed':False,'GPU_numerical_semantic_performance_qualified':False,'checks':checks}
 (b/'compiled-cpu-seal.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'sources':len(records),'objects':54,'startupGuards':len(guards),'strictTampers':len(tamper),'sealSHA':sha((b/'compiled-cpu-seal.json').read_bytes())}))
if __name__=='__main__':main()
