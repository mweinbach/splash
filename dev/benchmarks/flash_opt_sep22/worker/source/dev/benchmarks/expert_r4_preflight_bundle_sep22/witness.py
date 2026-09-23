#!/usr/bin/env python3
"""CPU compiled/source closure and unchanged public native producer witness."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_r4_preflight_bundle_sep22')
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);a=p.parse_args();b=a.build.resolve();m=json.loads((b/'overlay-manifest.json').read_text());base=Path(m['base']);parent=json.loads((base/'overlay-manifest.json').read_text());checks=[]
 def need(value,name):
  checks.append({'name':name,'pass':bool(value)})
  if not value:raise ValueError('CPU bundle source witness failed:'+name)
 need(sha((base/'overlay-manifest.json').read_bytes())==m['parent_manifest_sha256'],'admitted compact parent manifest')
 need(sha((base/'READY.json').read_bytes())==m['parent_READY_sha256'],'admitted compact parent READY')
 need(sha((b/'link-inputs.mk').read_bytes())==m['link_make_sha256'],'full inherited host link mapping')
 need(sha(json.dumps(m['identity_parts'],sort_keys=True,separators=(',',':')).encode())==m['source_identity_sha256'],'canonical new CPU policy identity')
 for r in m['files']:need(sha((b/'source'/r['path']).read_bytes())==r['sha256'],'sealed program source '+r['path'])
 for r in m['frozen_inputs']:need(sha((b/r['path']).read_bytes())==r['sha256'],'frozen original library/Core/AIR '+r['path'])
 need((b/'splash.metallib').read_bytes()==(base/'splash.metallib').read_bytes(),'exact original shader library no shader rebuild')
 spec=importlib.util.spec_from_file_location('cpu_bundle_overlay',b/'source'/PRIVATE/'overlay.py');overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay);changed=[]
 for r in parent['files']:
  rel=r['path'];original=(base/'source'/rel).read_bytes();need(sha(original)==r['sha256'],'parent original '+rel);actual=(b/'source'/rel).read_bytes();need(actual==overlay.transform(rel,original.decode()).encode(),'registered CPU-only transform '+rel)
  if actual!=original:changed.append(rel)
 need(sorted(changed)==m['changed_paths']==sorted(overlay.CHANGED),'exact four optional host sources changed')
 src=b/'source';old=(base/'source/runtime/flash/FlashInt8ExpertStore.mm').read_text();new=(src/'runtime/flash/FlashInt8ExpertStore.mm').read_text()
 first='void FlashInt8ExpertStore::addGateUp(';last='void FlashInt8ExpertStore::addFixedSG2PrefillGateUp('
 need(new[new.index(first):new.index(last)]==old[old.index(first):old.index(last)],'public native GU/down fullvalidation entiremethods unchanged')
 guardstart='void requireBytes(';guardend='FlashInt8ExpertStoreParams allRowsParams('
 need(new[new.index(guardstart):new.index(guardend)]==old[old.index(guardstart):old.index(guardend)],'all original requireBytes/disjoint/allRowsScratch guards unchanged')
 chain=new[new.index('void FlashInt8ExpertStore::addCompactNativeR4VerifyChain('):]
 for role,nxt,start in [('addGateUp','addDownScatter','  const auto p = allRowsParams('),('addDownScatter','addFixedSG2PrefillGateUp','  const uint32_t routes = rows * selections;')]:
  suffix=overlay.native_suffix(old,role,nxt,start);need(suffix in chain,'private append original literal suffix '+role)
 need(chain.index('const CapturedViews captured{')<chain.index('::validateComplete(')<chain.index('ValidatedBundle bundle{'),'capture-before-validation-before-token')
 need('ValidatedBundle(const ValidatedBundle &)=delete;'in chain and'ValidatedBundle(ValidatedBundle &&)=delete;'in chain,'noncopyable nonmovable private lexical token')
 need(chain.index('if(b.consumed||')<chain.index('b.consumed=true;')<chain.index('graph.add("expert_r4'),'single consume/epoch/layer/graph/policy rejection before append')
 need(chain.count('appendValidated(bundle);')==1,'one lexical consumption no token escape or cache')
 need('const auto &s=b.v.s;const auto &layer=b.v.source;const auto diagnostics=b.v.diag;'in chain and'const auto rows=b.v.rows,selections=b.v.selections;const auto tile=b.v.tile;'in chain,'append binds only const captured view/parameter copies')
 fwd=(src/'runtime/flash/FlashForward.cpp').read_text();need('if (compactR4Verify && compact_r4_preflight_sep22::requested()) {'in fwd,'same explicit singleton physicalR4 verification selector')
 for rel in ['runtime/flash/FlashBatchForward.cpp','runtime/flash/FlashBatchVerify.cpp','runtime/flash/FlashBatchPrefill.cpp','runtime/flash/FlashMTP.cpp','runtime/flash/FlashMTPDepth.cpp','runtime/flash/FlashInt8Head.cpp','runtime/flash/FlashBF16Q8Head.cpp','dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp']:
  need((src/rel).read_bytes()==(base/'source'/rel).read_bytes(),'other callers/math/state unchanged '+rel)
 names=next(line.split(' := ',1)[1].split()for line in(b/'link-inputs.mk').read_text().splitlines()if line.startswith('REBUILD_NAMES := '));need(len(names)==len(set(names))==50,'all50 host header consumers fresh')
 objects=[];deps=[];consumers=[]
 for name in names:
  obj=b/'host'/(name+'.o');dep=obj.with_suffix('.d');need(obj.is_file()and dep.is_file(),'fresh object/dependency '+name);rule=dep.read_text().replace('\\\n','').split('\n',1)[0];paths=[Path(x).resolve()for x in rule.split(': ',1)[1].split()];project=[x for x in paths if ROOT in x.parents];need(all(src in x.parents for x in project),'no live dependency '+name)
  if src/PRIVATE/'policy.hpp'in paths:consumers.append(name)
  deps.extend(str(x.relative_to(src))for x in project);objects.append({'path':str(obj.relative_to(b)),'sha256':sha(obj.read_bytes())})
 need({'FlashForward','FlashWorker','002-FlashInt8ExpertStore','004-FlashBatchPrefill','008-FlashBatchForward','010-FlashBatchVerify'}.issubset(consumers),'all six actual Store header consumers new policy')
 artifacts=[{'path':name,'sha256':sha((b/name).read_bytes())}for name in ['splash-flash','splash.metallib','guard-decision-cpu','preflight-policy-cpu']]
 report={'schema':'CPU-only-local-R4-preflight-compiled-source-witness-v1','pass':True,'source_identity_sha256':m['source_identity_sha256'],'checks':checks,'host_TUs_rebuilt':50,'actual_policy_header_consumers':consumers,'private_compiler_dependencies':sorted(set(deps)),'compiled_objects':objects,'artifacts':artifacts,'shader_library_byte_equal_parent':True,'GPU_work':False,'model_capture_payload_reads':False,'new_GPU_allocation_bytes':0,'whole_state_quality_performance_qualified':False}
 out=b/'cpu-source-witness.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'output':str(out),'GPU_work':False}))
if __name__=='__main__':main()
