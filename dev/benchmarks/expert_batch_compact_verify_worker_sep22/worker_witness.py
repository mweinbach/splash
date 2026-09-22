#!/usr/bin/env python3
"""Private fixed-batch target worker source/dependency/old-state closure witness."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_batch_compact_verify_worker_sep22')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args();b=a.build.resolve();m=json.loads((b/'overlay-manifest.json').read_text());base=Path(m['base']);clock=Path(m['clock_parent']);parent=json.loads((base/'overlay-manifest.json').read_text());checks=[]
 def need(v,name):
  checks.append({'name':name,'pass':bool(v)})
  if not v:raise ValueError('Batch worker CPU closure failed:'+name)
 need(sha((base/'overlay-manifest.json').read_bytes())==m['parent_manifest_sha256'],'authenticated math parent manifest')
 need(sha((b/'link-inputs.mk').read_bytes())==m['link_make_sha256'],'sealed link make')
 for r in m['files']:need(sha((b/'source'/r['path']).read_bytes())==r['sha256'],'sealed source '+r['path'])
 for r in m['frozen_inputs']:need(sha((b/r['path']).read_bytes())==r['sha256'],'sealed frozen binary '+r['path'])
 spec=importlib.util.spec_from_file_location('batch_prepare',b/'source'/PRIVATE/'worker_prepare.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);changed=[]
 for r in parent['files']:
  rel=r['path'];raw=(base/'source'/rel).read_bytes();need(sha(raw)==(r.get('sha256')or r.get('overlay_sha256')),'math parent source '+rel)
  old=(clock/'source'/rel).read_bytes()if rel in mod.OVERRIDES else raw;actual=(b/'source'/rel).read_bytes();need(actual==mod.transform(rel,old.decode()).encode(),'registered transform only '+rel)
  if actual!=old:changed.append(rel)
 need(sorted(changed)==m['changed_paths']==sorted(mod.CHANGED),'exact four changed host sources after nativeclock overrides')
 need(sha(json.dumps(m['identity_parts'],sort_keys=True,separators=(',',':')).encode())==m['source_identity_sha256'],'canonical source identity')
 need(sha((clock/'compiled-cpu-seal.json').read_bytes())==m['identity_parts']['clock_parent_seal'],'guarded nativeclock ancestry')
 src=b/'source';o=(src/'runtime/flash/FlashBatchVerify.cpp').read_text();store=(src/'runtime/flash/FlashInt8ExpertStore.mm').read_text();worker=(src/'runtime/flash/FlashWorker.mm').read_text()
 for rows in (8,16):
  q=ROOT/f'build/expert-batch-r{rows}-compact-native-sep22-component-v2';qm=json.loads((q/'source-seal.json').read_text());qp=(q/'source/dev/benchmarks/expert_batch_compact_native_sep22/plan.metal').read_bytes();actual=(src/PRIVATE/f'plan-r{rows}.metal').read_bytes();need(actual==qp.replace(b'expert_batch_compact_native_sep22_plan',f'expert_batch_r{rows}_compact_native_sep22_plan'.encode()),f'qualified R{rows} planner export rename only');need(qm['source_identity_sha256']==m['qualified_parallel_source_identities'][str(rows)],f'qualified R{rows} source identity')
 need('compact_native_batch_verify_sep22::eligible(lanes,rows,true)'in o,'actual active lane count and rows4 target selector')
 need('addCompactNativeBatchVerifyPack(graph,layer,mixed,expertIDs,impl_->blockedScratch,diag,flattened,kSelections)'in o,'native physical rows and original scratch')
 need('addGateUp(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);'in store and'addDownScatter(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);'in store,'literal original M16 gate and down methods')
 need('FlashMoEBucketParams{rows,10,2560,512,routes,16,jobs,0},{1,1,1},{256,1,1}'in store and'{routes+63,1,1},{256,1,1}'in store,'native ABI original padding and fixed oneCTA')
 old=(base/'source/runtime/flash/FlashInt8ExpertStore.mm').read_text();first='void FlashInt8ExpertStore::addGateUp(';last='void FlashInt8ExpertStore::addFixedSG2PrefillGateUp(';need(store[store.index(first):store.index(last)]==old[old.index(first):old.index(last)],'every original M16 dispatch/guard/counter unchanged literal')
 old=(base/'source/runtime/flash/FlashBatchVerify.cpp').read_text();suffix='metal::CommandTiming FlashBatchVerify::commitBatch(';need(o[o.index(suffix):]==old[old.index(suffix):],'all batch commit0/null abort lazy owner state ticket source unchanged')
 need(worker.index('compact_native_batch_verify_sep22::validateDependencies(')<worker.index('(void)pointwise_sep21::requested();'),'strict qualified role freeze before model construction')
 need('graph construction not GPU completion'in worker and'base_native_dispatches_per_layer":10'in worker and'additional_gpu_allocation_bytes":0'in worker,'honest perphysicalwidth graph metadata')
 for rel in ('runtime/flash/FlashForward.cpp','runtime/flash/FlashBatchForward.cpp','runtime/flash/FlashBatchPrefill.cpp','runtime/flash/FlashBatchMTPForward.cpp','runtime/flash/FlashMTP.cpp','runtime/flash/FlashMTPDepth.cpp','runtime/flash/FlashInt8Head.cpp','runtime/flash/FlashBF16Q8Head.cpp','dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp'):
  need((src/rel).read_bytes()==(base/'source'/rel).read_bytes(),'all Pref AR singleton R4 head math unchanged '+rel)
 need((src/'runtime/flash/FlashGDNBatchILP.cpp').read_bytes()==(clock/'source/runtime/flash/FlashGDNBatchILP.cpp').read_bytes(),'nativeclock discarded GDN producer guard retained')
 need((src/'dev/benchmarks/current_batch_sep22/NativeLifecycleTrace.hpp').read_bytes()==(clock/'source/dev/benchmarks/current_batch_sep22/NativeLifecycleTrace.hpp').read_bytes(),'native first-emission Done clock retained')
 link=(b/'link-inputs.mk').read_text();names=next(x.split(' := ',1)[1].split()for x in link.splitlines()if x.startswith('REBUILD_NAMES := '));need(len(names)==len(set(names))==50,'all50 nonCore host TU census');need(len(parent['frozen_objects'])==4,'exact4 unchanged Core objects')
 deps=[];objects=[];consumers=[];storeConsumers=[]
 for name in names:
  obj=b/'host'/(name+'.o');dep=obj.with_suffix('.d');need(obj.is_file()and dep.is_file(),'fresh object/dependency '+name);rule=dep.read_text().replace('\\\n','').split('\n',1)[0];paths=[Path(x).resolve()for x in rule.split(': ',1)[1].split()];private=[x for x in paths if ROOT in x.parents];need(all(src in x.parents for x in private),'private-only compiler dependencies '+name)
  if src/PRIVATE/'bridge.hpp'in paths:consumers.append(name)
  if src/'runtime/flash/FlashInt8ExpertStore.hpp'in paths:storeConsumers.append(name)
  deps.extend(str(x.relative_to(src))for x in private);objects.append({'path':str(obj.relative_to(b)),'sha256':sha(obj.read_bytes()),'bytes':obj.stat().st_size})
 need({'FlashWorker','002-FlashInt8ExpertStore','010-FlashBatchVerify'}.issubset(consumers),'actual changed public-header consumers rebuilt')
 artifacts=[]
 for name in ('splash-flash','splash.metallib','compact-r8-plan.air','compact-r16-plan.air','policy-cpu'):
  path=b/name;need(path.is_file()and path.stat().st_size>0,'complete artifact '+name);artifacts.append({'path':name,'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size})
 out={'schema':'compact-native-batch-target-verify-CPUclosure-v1','pass':True,'source_identity_sha256':m['source_identity_sha256'],'host_tus_rebuilt':len(names),'new_bridge_consumers':consumers,'actual_Store_header_consumers':storeConsumers,'private_compiler_dependencies':sorted(set(deps)),'compiled_objects':objects,'artifacts':artifacts,'checks':checks,'additional_gpu_allocation_bytes':0,'gpu_work':False,'model_or_capture_payload_reads':False,'whole_model_quality_qualified':False,'batch_state_qualified':False};(b/'cpu-source-witness.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'host_tus_rebuilt':len(names),'new_bridge_consumers':len(consumers),'Store_header_consumers':len(storeConsumers),'output':str(b/'cpu-source-witness.json')}))
if __name__=='__main__':main()
