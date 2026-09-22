#!/usr/bin/env python3
"""Freeze resource-only pure-I8 teacher worker; CPU source/object work only."""
from pathlib import Path
import argparse,hashlib,json,re,shlex,subprocess
ROOT=Path(__file__).resolve().parents[3];PRIVATE=Path('dev/benchmarks/teacher_singleton_lease_sep22');OLD=ROOT/'build/dense-w8a8-residency-prune-sep21-worker-v1'
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def once(text,before,after):
 if text.count(before)!=1:raise ValueError('source anchor differs:'+before[:90])
 return text.replace(before,after,1)
def transform(rel,text):
 if rel not in ['runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm']:return text
 text='#include "dev/benchmarks/teacher_singleton_lease_sep22/policy.hpp"\n'+text
 if rel.endswith('FlashForward.cpp'):
  text=once(text,'  if (impl_->denseCache) append(impl_->denseCache->persistedWeightBuffers());','  if (impl_->denseCache) append(teacher_singleton_lease_sep22::persistentBF16(*impl_->denseCache));')
  text=once(text,'  if (impl_->floatDenseCache) append(impl_->floatDenseCache->persistedWeightBuffers());','  if (impl_->floatDenseCache) append(teacher_singleton_lease_sep22::persistentF32(impl_->weights,*impl_->floatDenseCache));')
  return text
 text=once(text,'      (void)adaptive_expert_tail_sg2k128_sep21::requested();', '''      const bool teacherSingletonLeasePrune=teacher_singleton_lease_sep22::requested();
      if(teacherSingletonLeasePrune){
        for(const char *required:{"SPLASH_FLASH_SAVED_OPERANDS_RESIDENT","SPLASH_FLASH_FLOAT_DENSE_CACHE","SPLASH_FLASH_FLOAT_DENSE_SELECTIVE","SPLASH_FLASH_DENSE_CACHE","SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21","SPLASH_FLASH_ALLROWS_FULL512_TARGET"})
          if(!environmentSwitch(required))throw std::invalid_argument(std::string("teacher lease-only profile requires ")+required+"=1");
        for(const char *excluded:{"SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT","SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT","SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21","SPLASH_FLASH_PHASE_Q4_SAVED_ONLY_RESIDENCY_SEP22","SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE"})
          if(environmentSwitch(excluded))throw std::invalid_argument(std::string("teacher lease-only profile forbids ")+excluded+"=1");
      }
      (void)adaptive_expert_tail_sg2k128_sep21::requested();''')
 text=once(text,'        savedResidency.requestedViews = operands.size();', '''        savedResidency.requestedViews = operands.size();
        if(teacherSingletonLeasePrune){
          const auto actual=forward.persistedOperandStatus();
          if(actual.bf16Tensors!=509 || actual.f32Tensors!=508 || actual.bf16PayloadBytes!=8467251200ULL || actual.f32PayloadBytes!=14391705600ULL)
            throw std::logic_error("teacher lease-only actual full508/509 backing differs");
          if(savedResidency.requestedViews!=807 || savedResidency.requestedViewBytes!=131005546496ULL)
            throw std::logic_error("teacher lease-only selected807 owner census differs");
          for(size_t i=0;i<operands.size();++i)for(size_t j=0;j<i;++j)
            if(operands[i].sameView(operands[j]))throw std::logic_error("teacher lease-only duplicate full owner");
        }''')
 text=once(text,'        if (savedResidency.originalAdded && !savedResidencyLease)', '''        if(teacherSingletonLeasePrune && savedResidencyLease &&
            (savedResidencyLease.bufferCount()!=807 || savedResidencyLease.byteCount()!=131005546496ULL))
          throw std::logic_error("teacher lease-only registered807 unique owner census differs");
        if (savedResidency.originalAdded && !savedResidencyLease)''')
 # Add immutable resource fields to identity with explicit null/false flag0; numerical identity remains untouched.
 anchor='      << R"(,"target_numerical_derivative_sha256":)"'
 text=once(text,anchor,'''      << R"(,"teacher_singleton_lease_profile":)" << (teacher_singleton_lease_sep22::requested()?json::quote(teacher_singleton_lease_sep22::profile):"null")
      << R"(,"teacher_singleton_lease_source_sha256":)" << (teacher_singleton_lease_sep22::requested()?json::quote(teacher_singleton_lease_sep22::sourceSHA):"null")
      << R"(,"teacher_singleton_lease_source_policy":)" << (teacher_singleton_lease_sep22::requested()?json::quote(teacher_singleton_lease_sep22::policy):"null")
      << R"(,"teacher_singleton_lease_enabled":)" << (teacher_singleton_lease_sep22::requested()?"true":"false")
'''+anchor)
 anchor='      << R"(,"original_text_residency":{"requested":)"'
 text=once(text,anchor,'''      << R"(,"teacher_singleton_lease":{"requested":)" << (teacher_singleton_lease_sep22::requested()?"true":"false")
      << R"(,"retained_F32_owner_count":)" << (teacher_singleton_lease_sep22::requested()?118:508)
      << R"(,"retained_F32_owner_bytes":)" << (teacher_singleton_lease_sep22::requested()?3247964160ULL:14391705600ULL)
      << R"(,"omitted_F32_owner_count":)" << (teacher_singleton_lease_sep22::requested()?390:0)
      << R"(,"omitted_F32_owner_bytes":)" << (teacher_singleton_lease_sep22::requested()?11143741440ULL:0)
      << R"(,"retained_BF16_owner_count":)" << (teacher_singleton_lease_sep22::requested()?425:509)
      << R"(,"retained_BF16_owner_bytes":)" << (teacher_singleton_lease_sep22::requested()?4692377600ULL:8467251200ULL)
      << R"(,"omitted_BF16_owner_count":)" << (teacher_singleton_lease_sep22::requested()?84:0)
      << R"(,"omitted_BF16_owner_bytes":)" << (teacher_singleton_lease_sep22::requested()?3774873600ULL:0)
      << R"(,"derived_W8_owner_count":168,"derived_W8_owner_bytes":1890975744,"all_backing_retained":true,"all_backing_charged":true,"selection_only":true,"qualification_scope":"B1; batch math/capabilities unchanged; transient direct-binding owners remain charged; physical pinning not promised"})"
'''+anchor)
 return text
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,default=ROOT/'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5');ap.add_argument('--output',type=Path,default=ROOT/'build/teacher-singleton-lease-sep22-worker-v1');a=ap.parse_args();base=a.base.resolve();out=a.output.resolve()
 if out.exists() or ROOT/'build'not in out.parents:raise ValueError('Fresh private output required')
 mp=base/'overlay-manifest.json';sp=base/'compiled-cpu-seal.json';parent=json.loads(mp.read_text());seal=json.loads(sp.read_text())
 if not seal['pass']or sha(mp.read_bytes())!=seal['source_manifest_sha256']:raise ValueError('Sealed Teacher parent required')
 for rel,digest in seal['source_sha256'].items():
  if sha((base/'source'/rel).read_bytes())!=digest:raise ValueError('Parent source drift:'+rel)
 for rel,digest in seal['artifact_sha256'].items():
  if sha((base/rel).read_bytes())!=digest:raise ValueError('Parent artifact drift:'+rel)
 oldmp=OLD/'overlay-manifest.json';old=json.loads(oldmp.read_text());oldRecords={r['path']:r for r in old['files']};bridge='dev/benchmarks/dense_w8a8_residency_sep21/worker_bridge.hpp';oldData=(OLD/'source'/bridge).read_bytes()
 if sha(oldData)!=oldRecords[bridge]['overlay_sha256']:raise ValueError('Audited BF16 selector drift')
 header=(ROOT/PRIVATE/'policy.hpp').read_bytes();material=header+(base/'source/runtime/flash/FlashForward.cpp').read_bytes()+oldData
 policySHA=sha(material);header=header.replace(b'SOURCE_SHA_PLACEHOLDER',policySHA.encode())
 files=[]
 for r in parent['files']:
  rel=r['path'];data=(base/'source'/rel).read_bytes();new=transform(rel,data.decode()).encode();write(out/'source'/rel,new);files.append({'path':rel,'sha256':sha(new),'parent_sha256':sha(data),'changed':new!=data})
 for rel,data in [(PRIVATE/'policy.hpp',header),(Path(bridge),oldData)]:
  write(out/'source'/rel,data);files.append({'path':rel.as_posix(),'sha256':sha(data),'new':True})
 result=subprocess.run(['make','-pqn','-rR','-f',str(base/'machinery/worker.mk'),f'BUILD={base}',str(base/'splash-flash')],cwd=ROOT,capture_output=True,text=True,timeout=30)
 if result.returncode not in (0,1):raise ValueError('Actual make closure failed')
 lines=[l for l in result.stdout.splitlines()if l.startswith(str(base/'splash-flash')+':')]
 if len(lines)!=1:raise ValueError('Ambiguous link')
 objs=[Path(x).resolve()for x in shlex.split(lines[0].split(':',1)[1])if x.endswith('.o')]
 corelines=re.findall(r'^CORE\s*:=\s*(.*)$',result.stdout,re.M)
 if len(corelines)!=1:raise ValueError('Core classification')
 cores={Path(x).resolve()for x in shlex.split(corelines[0])}
 if len(objs)!=54 or len(set(objs))!=54 or len(cores)!=4:raise ValueError('Actual54 closure required')
 names={r['path']for r in files};aliases={'teacher_bulk':'dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp','Prefill4kQSABulk':'dev/benchmarks/prefill4k_attention/bulk.cpp','Prefill4kQSACoalesced':'dev/benchmarks/prefill4k_attention/coalesced.cpp'};rebuild=[];frozen=[]
 for obj in objs:
  if obj in cores:
   rel=Path('reused/core')/obj.name;data=obj.read_bytes();write(out/rel,data);frozen.append({'path':rel.as_posix(),'sha256':sha(data),'parent_path':str(obj)});continue
  stem=re.sub(r'^\d{3}-','',obj.stem);src=aliases.get(stem)or next((x for x in names if Path(x).stem==stem and x.endswith(('.cpp','.mm'))),None)
  if not src:raise ValueError('Unresolved TU:'+str(obj))
  rebuild.append({'object':obj.stem,'source':src,'parent_object_sha256':sha(obj.read_bytes())})
 write(out/'splash.metallib',(base/'splash.metallib').read_bytes())
 link='REBUILD_NAMES := '+' '.join(r['object']for r in rebuild)+'\nCORE := '+' '.join('$(BUILD)/'+r['path']for r in frozen)+'\n'
 for r in rebuild:link+='SRC_'+r['object']+' := $(BUILD)/source/'+r['source']+'\n'
 write(out/'link-inputs.mk',link.encode())
 for name in ['prepare.py','worker.mk','policy.hpp','census_plan.py']:write(out/'machinery'/name,(ROOT/PRIVATE/name).read_bytes())
 plan=ROOT/'build/release/flash/sep22-pure-I8-teacher-lease-only-source-census-plan-v1.json';write(out/'source-census-plan.json',plan.read_bytes())
 manifest={'schema':'private-pure-I8-teacher-parent508-lease-only807-v1','base':str(base),'base_source_manifest_sha256':sha(mp.read_bytes()),'base_compiled_cpu_seal_sha256':sha(sp.read_bytes()),'metallib_sha256':sha((base/'splash.metallib').read_bytes()),'files':files,'rebuild':rebuild,'frozen_core':frozen,'all50HostTUs_rebuilt':True,'only_source_body_changes':['FlashForward.cpp cachedOperandsOnly','FlashWorker.mm startup guards/lease census/static resource identity'], 'source_policy_sha256':policySHA,'source_policy_material':'unsubstituted new policy.hpp + exact parent FlashForward.cpp + sealed old BF16 selector','expected_owner_count':807,'expected_owner_bytes':131005546496,'all508F32_and509BF16_backing_preserved':True,'numerical_derivative_changed':False,'allocation_planner_reservations_changed':False,'GPU_executed':False,'model_payload_bytes_read':0,'GPU_qualified':False}
 write(out/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
 # Prepare an exact fresh Root canonical command, keeping every Parent environment flag.
 cmd=shlex.split((base/'root-model-command.txt').read_text());cmd[cmd.index('--binary')+1]=str(out/'splash-flash');cmd[cmd.index('--output')+1]=str(ROOT/'build/release/flash/sep22-pure-I8-teacher-lease-only-model-and-quality-v1.json');cmd+=['--env','SPLASH_FLASH_TEACHER_SINGLETON_LEASE_PRUNE_SEP22=1'];write(out/'root-model-command.txt',(shlex.join(cmd)+'\n').encode())
 env={};i=0
 while i<len(cmd):
  if cmd[i]=='--env':k,v=cmd[i+1].split('=',1);env[k]=v;i+=1
  i+=1
 write(out/'root-mtp-environment.json',(json.dumps(env,indent=2)+'\n').encode());print(json.dumps({'build':str(out),'sources':len(files),'host_rebuild':len(rebuild),'policySHA':policySHA}))
if __name__=='__main__':main()
