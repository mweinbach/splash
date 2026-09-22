#!/usr/bin/env python3
"""CPU-only current-header oracle freeze; no model/token/export payload reads."""
import argparse,hashlib,json,os,re,shlex,shutil,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OLD=ROOT/'build/batchverify-exact-compact-sep22-v13'
PRIVATE=Path('dev/benchmarks/expert_batch_b4_composition_sep22/qa')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(s,a,b):
 if s.count(a)!=1:raise ValueError('source anchor changed:'+a[:90])
 return s.replace(a,b,1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker',type=Path,default=ROOT/'build/integer-b4-twopass-composed-sep22-worker-v2');ap.add_argument('--build',type=Path,default=ROOT/'build/batchverify-current-b4-integer-composed-sep22-v3');a=ap.parse_args();worker=a.worker.resolve();b=a.build.resolve()
 if b.exists()or ROOT/'build'not in b.parents:raise ValueError('fresh private build required')
 mp=worker/'overlay-manifest.json';m=json.loads(mp.read_text());sp=worker/'compiled-cpu-seal.json';seal=json.loads(sp.read_text())
 if not seal['pass']:raise ValueError('Current CPU-sealed worker required')
 # Inspect the actual fresh compiler closure, never an opaque historical host link.
 compiled=json.loads((worker/'compiled-build.json').read_text());host=m['rebuild']
 if len(host)!=50 or len({x['object']for x in host})!=50:raise ValueError('actual current50TU census required')
 for rec in m['files']:
  if sha(worker/'source'/rec['path'])!=rec['sha256']:raise ValueError('current source drift:'+rec['path'])
 if sha(worker/'splash-flash')!='ca7bc795bfb42b50949f1eaf1f035620ec3bf1b861ab397ed9a166ccc3202e68' or sha(worker/'splash.metallib')!='74af1228995f38890df035555b2000018899783fd46ad3a12f8a835c833231c8':raise ValueError('Final current v2 artifact required')
 b.mkdir(parents=True);shutil.copytree(worker/'source',b/'source');private=b/'source'/PRIVATE;private.mkdir(parents=True)
 for name in ['inspect.hpp','batch_inspection.cpp.inc','forward_inspection.cpp.inc','oracle.mm','transform_oracle.py','build.py']:shutil.copyfile(HERE/name,private/name)
 # Current original Core objects are exactly the original Core ancestry of v13;
 # use frozen original bodies, not live repo sources or historical host objects.
 oldmeta=json.loads((OLD/'provenance.json').read_text());oldcores={Path(x['inherited_path']).name:x for x in oldmeta['objects']if Path(x['inherited_path']).name in {Path(x['path']).name for x in m['Core']}}
 if len(oldcores)!=4:raise ValueError('exact four original Core ancestry records required')
 core_sources={'MetalBackend':'runtime/metal/MetalBackend.mm','DeviceCapabilities':'runtime/metal/DeviceCapabilities.cpp','Protocol':'runtime/engine/Protocol.cpp','MemoryGovernor':'runtime/engine/MemoryGovernor.cpp'}
 pins={x['source']:x['original_sha256']for x in oldmeta['core_source_pins']}
 core_origins=[]
 for rec in m['Core']:
  original=worker/rec['path'];name=Path(rec['path']).stem.split('-',1)[-1];src=core_sources[name];raw=(OLD/'source'/src).read_bytes()
  if rec['sha256']!=oldcores[original.name]['inherited_sha256']or sha(original)!=rec['sha256']:raise ValueError('Current Core ancestry changed:'+name)
  if name=='MetalBackend':raw=raw[:raw.index(b'\nnamespace splash::metal {\nuint64_t MetalBuffer::oracleChargedBytesSep22()')]
  if hashlib.sha256(raw).hexdigest()!=pins[src]:raise ValueError('Frozen original Core body changed:'+name)
  dest=b/'source'/src;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);core_origins.append({'source':src,'original_source_sha256':pins[src],'current_original_object_sha256':rec['sha256']})
 mh=b/'source/runtime/metal/MetalBackend.hpp';original_mh=mh.read_text();anchor='  [[nodiscard]] uint64_t sizeBytes() const noexcept;'
 if original_mh.count(anchor)<1:raise ValueError('Core header metadata anchor')
 mh.write_text(original_mh.replace(anchor,anchor+'\n  [[nodiscard]] uint64_t oracleChargedBytesSep22() const noexcept;\n  [[nodiscard]] uintptr_t oracleOwnerIdentitySep22() const noexcept;',1))
 core=b/'source/runtime/metal/MetalBackend.mm';core.write_text(core.read_text()+'''\nnamespace splash::metal {
uint64_t MetalBuffer::oracleChargedBytesSep22() const noexcept {return impl_&&impl_->allocation?impl_->allocation->bytes:0;}
uintptr_t MetalBuffer::oracleOwnerIdentitySep22() const noexcept {return impl_&&impl_->allocation?reinterpret_cast<uintptr_t>(impl_->allocation.get()):0;}
}\n''')
 fh=b/'source/runtime/flash/FlashForward.hpp';s=fh.read_text();anchor='  friend class FlashBatchForward;'
 if s.count(anchor)!=2:raise ValueError('actual currentForward friendship anchor')
 before,after=s.rsplit(anchor,1);fh.write_text(before+'  friend class FlashDeepPrefixOracle;\n'+anchor+after)
 bh=b/'source/runtime/flash/FlashBatchVerify.hpp';bh.write_text(once(bh.read_text(),'private:\n  struct Impl;','private:\n  friend class FlashDeepPrefixOracle;\n  struct Impl;'))
 for rel,inc in [('runtime/flash/FlashForward.cpp','forward_inspection.cpp.inc'),('runtime/flash/FlashBatchVerify.cpp','batch_inspection.cpp.inc')]:
  p=b/'source'/rel;body=p.read_text();p.write_text('#include "inspect.hpp"\n'+body+'\n'+(private/inc).read_text())
 flags=['-std=c++20','-O3','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-fobjc-arc','-mmacosx-version-min=27.0','-DSPLASH_INT8_EXPERIMENT=1','-I'+str(b/'source'),'-I'+str(b/'source/runtime'),'-I'+str(b/'source/dev/benchmarks/prefill4k_attention'),'-I'+str(private),'-I'+str(b)]
 tasks=[];census=[];objects=[]
 for rec in host:
  name=Path(rec['object']).stem;src=b/'source'/rec['source'];dp=b/'objects'/(name+'.d');obj=b/'objects'/(name+'.o');obj.parent.mkdir(exist_ok=True)
  if name=='FlashWorker':
   deps=subprocess.run(['xcrun','-sdk','macosx','clang++',*flags,'-MM',str(src)],cwd=ROOT,check=True,capture_output=True,text=True).stdout;census.append({'object':name,'source':rec['source'],'excluded_main':True,'preprocess_dependencies':deps});continue
  tasks.append((src,obj,dp));objects.append(obj);census.append({'object':name,'source':rec['source'],'excluded_main':False,'actual_current_TU_recompiled':True})
 for rec in m['Core']:
  name=Path(rec['path']).stem.split('-',1)[-1];obj=b/'objects'/Path(rec['path']).name;tasks.append((b/'source'/core_sources[name],obj,obj.with_suffix('.d')));objects.append(obj)
 if len(objects)!=53 or len(census)!=50:raise ValueError('53effective/50headerconsumer census required')
 def compile_one(t):
  src,obj,dp=t;subprocess.run(['xcrun','-sdk','macosx','clang++',*flags,'-MMD','-MP','-MF',str(dp),'-c',str(src),'-o',str(obj)],cwd=ROOT,check=True)
 with ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(compile_one,tasks))
 shutil.copyfile(worker/'splash.metallib',b/'splash.metallib')
 files=[{'path':str(p.relative_to(b/'source')),'sha256':sha(p)}for p in sorted((b/'source').rglob('*'))if p.is_file()]
 meta={'schema':'current-B4BQSA-plus-integer-batchverify-CPU-oracle-v1','current_worker':str(worker),'current_source_identity':m['source_identity_sha256'],'current_worker_manifest_SHA':sha(mp),'current_worker_seal_SHA':sha(sp),'workerSHA':sha(worker/'splash-flash'),'metallib_sha256':sha(b/'splash.metallib'),'objects':[],'header_census':census,'Core_source_ancestry':core_origins,'sources':files,'all49NonWorkerCurrentTUsRebuilt':True,'current4CoreCloneReadonlyHooksRebuilt':True,'current78AIRLibraryExact':True,'original_worker_FP_method_bodies_literal':True,'no_suffix_strip_or_old_numeric_identity_forgery':True,'GPU_executed':False,'model_operand_response_capture_payload_reads':0,'full_physical_selected_labels':['initial','fresh-r16.pending','fresh-r16.committed','same-r8.future.completed','b2.initial','fresh-r8.pending','fresh-r8.future.completed'],'all18_legacy_campaign_labels_and_arena_output_checks_retained':True,'fullphysical_all18_inherited':False,'whole_state_quality_timing_qualified':False}
 for obj in objects:
  dp=obj.with_suffix('.d');tokens=shlex.split(dp.read_text().replace('\\\n',' ').splitlines()[0].split(':',1)[1]);deps=[]
  for token in tokens:
   p=Path(token).resolve()
   if b/'source'in p.parents:deps.append(str(p.relative_to(b/'source')))
   elif ROOT in p.parents:raise ValueError('live/unsealed repo dependency:'+str(p))
  meta['objects'].append({'path':str(obj.relative_to(b)),'sha256':sha(obj),'dependency_file':str(dp.relative_to(b)),'dependencies':sorted(set(deps))})
 (b/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n');summary={k:v for k,v in meta.items()if k not in ['objects','header_census','sources']}
 (b/'BatchVerifyBuildProvenance.hpp').write_text('#pragma once\ninline constexpr const char *kPrefillExactProvenancePath='+json.dumps(str(b/'provenance.json'))+';\ninline constexpr const char *kPrefillExactBuildProvenance=R"META('+json.dumps(summary,sort_keys=True)+')META";\n')
 cmds=[]
 for role in ['control','candidate']:
  cmd=['xcrun','-sdk','macosx','clang++',*flags,'-Wno-unused-function','-DSPLASH_VERIFY_CANDIDATE='+str(int(role=='candidate')),str(private/'oracle.mm'),*[str(p)for p in objects],'-framework','Foundation','-framework','Metal','-framework','IOKit','-o',str(b/('oracle-'+role))];subprocess.run(cmd,cwd=ROOT,check=True);cmds.append(cmd)
  cpuenv={k:v for k,v in os.environ.items()if not k.startswith('SPLASH_FLASH_')};cpuenv.update({k:'1' for k in ['SPLASH_FLASH_FUSE_GDN','SPLASH_FLASH_GDN_LAZY_ROLLBACK','SPLASH_FLASH_GPU_GREEDY','SPLASH_FLASH_PLE_SSD_STREAMING','SPLASH_FLASH_ALLROWS_FULL512_TARGET','SPLASH_FLASH_BATCH_PREFILL_TWOPASS_SEP22','SPLASH_FLASH_BATCH_PREFILL','SPLASH_FLASH_BATCH_QSA_BULK_PREFILL','SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21','SPLASH_FLASH_QSA_F32','SPLASH_FLASH_QSA_MPP','SPLASH_FLASH_QSA_ROW_TILES','SPLASH_FLASH_QSA_BULK_PREFILL','SPLASH_FLASH_QSA_BULK_PREFILL_SG8','SPLASH_FLASH_MTP','SPLASH_FLASH_BATCH_MTP','SPLASH_FLASH_BATCH_MTP_PREFILL','SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY','SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY']});cpuenv['SPLASH_FLASH_MTP_DRAFT_DEPTH']='3'
  cpu=subprocess.run([str(b/('oracle-'+role)),'--cpu-only'],cwd=ROOT,env=cpuenv,check=True,capture_output=True,text=True);(b/('cpu-self-test-'+role+'.json')).write_text(cpu.stdout);meta['CPU_'+role]=json.loads(cpu.stdout)
 meta['link_commands']=cmds;meta['artifact_sha256']={n:sha(b/n)for n in ['oracle-control','oracle-candidate','splash.metallib','provenance.json','BatchVerifyBuildProvenance.hpp','cpu-self-test-control.json','cpu-self-test-candidate.json']};meta['pass']=True
 (b/'CPU_READY.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'build':str(b),'pass':True,'53_effective_objects':True,'GPU_started':False}))
if __name__=='__main__':main()
