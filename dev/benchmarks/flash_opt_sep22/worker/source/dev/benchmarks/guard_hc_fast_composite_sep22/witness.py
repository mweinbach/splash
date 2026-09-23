#!/usr/bin/env python3
"""CPU source/host/AIR closure; no old whole-proof inheritance."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[3];PRIVATE=Path('dev/benchmarks/guard_hc_fast_composite_sep22')
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args();b=a.build.resolve();m=json.loads((b/'overlay-manifest.json').read_text());base=Path(m['base']);hc=Path(m['HC_origin']);parent=json.loads((base/'overlay-manifest.json').read_text());checks=[]
 def need(value,name):
  checks.append({'name':name,'pass':bool(value)})
  if not value:raise ValueError('Composite CPU closure failed:'+name)
 for r in m['files']:need(sha((b/'source'/r['path']).read_bytes())==r['sha256'],'sealed source '+r['path'])
 for r in m['frozen_inputs']:need(sha((b/r['path']).read_bytes())==r['sha256'],'frozen Core/originalAIR/integer/FASTdown '+r['path'])
 need(sha(json.dumps(m['identity_parts'],sort_keys=True,separators=(',',':')).encode())==m['source_identity_sha256'],'canonical new composite source identity')
 need(sha((base/'overlay-manifest.json').read_bytes())==m['identity_parts']['guard_parent_manifest'],'registered b32 guard origin manifest')
 need(sha((b/'link-inputs.mk').read_bytes())==m['link_make_sha256'],'exact host/shader input mapping')
 spec=importlib.util.spec_from_file_location('composite_overlay',b/'source'/PRIVATE/'overlay.py');overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
 spec=importlib.util.spec_from_file_location('registeredHC_overlay',b/'source/dev/benchmarks/hc_pad_verify_worker_sep22/overlay.py');hcoverlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(hcoverlay);changed=[]
 for r in parent['files']:
  rel=r['path'];old=(base/'source'/rel).read_bytes();actual=(b/'source'/rel).read_bytes();need(sha(old)==r['sha256'],'b32 original source '+rel);need(actual==overlay.transform(rel,old.decode(),hcoverlay).encode(),'registered HC+composite transform only '+rel)
  if old!=actual:changed.append(rel)
 need(sorted(changed)==m['changed_paths']==sorted(overlay.CHANGED),'only Forward and Worker source hooks changed')
 for rel in ['runtime/flash/FlashInt8ExpertStore.hpp','runtime/flash/FlashInt8ExpertStore.mm','runtime/flash/FlashBatchForward.cpp','runtime/flash/FlashBatchVerify.cpp','runtime/flash/FlashBatchPrefill.cpp','runtime/flash/FlashMTP.cpp','runtime/flash/FlashMTPDepth.cpp','dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp']:
  need((b/'source'/rel).read_bytes()==(base/'source'/rel).read_bytes(),'guard/publicAPI/othermath/state unchanged '+rel)
 need((b/'source/dev/benchmarks/hc_pad_verify_worker_sep22/worker_bridge.hpp').read_bytes()==(hc/'source/dev/benchmarks/hc_pad_verify_worker_sep22/worker_bridge.hpp').read_bytes(),'HC original97 guard/producer bridge exact')
 need(sha((b/'hc-pad.air').read_bytes())==m['HC_down_AIR_sha256']=='7cc3d642e22bbf90fb95f6c524af232a75fe6c8d99d7385a5e51f7ee572eeefa','exact registered FASTdown original recipe AIR')
 fwd=(b/'source/runtime/flash/FlashForward.cpp').read_text();worker=(b/'source/runtime/flash/FlashWorker.mm').read_text()
 need('guard_hc_fast_composite_sep22::validate(compact_native_r4_verify_sep22::requested(),'in fwd and'guard_hc_fast_composite_sep22::validate(compact_native_r4_verify_sep22::requested(),'in worker,'direct Forward constructor and Worker strictthreeflag selection')
 need('impl_->hc(graph, prefix + ".attn_hyper_connection", rows, true, normalizedReady, verification);'in fwd and'impl_->hc(graph, prefix + ".mlp_hyper_connection", rows, true, normalizedReady, verification);'in fwd,'explicit main verify context HC callsites')
 need('if (compactR4Verify && compact_r4_preflight_sep22::requested()) {'in fwd,'complete guard preflight branch preserved')
 need('whole_composite_state_qualified":false'in worker,'pending new exe+library wholeproof not oldV3 inheritance')
 names=next(x.split(' := ',1)[1].split()for x in(b/'link-inputs.mk').read_text().splitlines()if x.startswith('REBUILD_NAMES := '));need(len(names)==len(set(names))==50,'all50 host TU census');objects=[];deps=[];consumers=[]
 for name in names:
  obj=b/'host'/(name+'.o');dep=obj.with_suffix('.d');need(obj.is_file()and dep.is_file(),'fresh object/dependency '+name);rule=dep.read_text().replace('\\\n','').split('\n',1)[0];paths=[Path(x).resolve()for x in rule.split(': ',1)[1].split()];project=[x for x in paths if ROOT in x.parents];need(all(b/'source'in x.parents for x in project),'all project compiler deps private '+name)
  if b/'source'/PRIVATE/'policy.hpp'in paths:consumers.append(name)
  deps.extend(str(x.relative_to(b/'source'))for x in project);objects.append({'path':str(obj.relative_to(b)),'sha256':sha(obj.read_bytes())})
 need(set(consumers)=={'FlashForward','FlashWorker'},'composite only Forward Worker policy consumers')
 artifacts=[{'path':name,'sha256':sha((b/name).read_bytes())}for name in ['splash-flash','splash.metallib','hc-pad.air']]
 report={'schema':'guard-C1-HCfastV8-current-host-source-closure-v1','pass':True,'source_identity_sha256':m['source_identity_sha256'],'checks':checks,'host_TUs_rebuilt':50,'current_nonWorker_closure_count':53,'policy_consumers':consumers,'private_compiler_dependencies':sorted(set(deps)),'compiled_objects':objects,'artifacts':artifacts,'new_GPU_allocation_bytes':0,'GPU_work':False,'model_capture_payload_reads':False,'oldV3_full_proof_reused':False,'whole_composite_state_qualified':False};out=b/'compiled-cpu-seal.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'output':str(out),'GPU_work':False}))
if __name__=='__main__':main()
