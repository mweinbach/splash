#!/usr/bin/env python3
"""Compose frozen guard C++ and registered FAST HC AIR; CPU only."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/guard_hc_fast_composite_sep22')
GUARD_ID='b32c030bff181fb65e546685282650e1db5636a5dfa466dcf8a745c66931fad5'
HC_AIR='7cc3d642e22bbf90fb95f6c524af232a75fe6c8d99d7385a5e51f7ee572eeefa'
HC_LEAF='92680570764ecfe797fe7db566146d1be1f6e51246a73b197a423a722d1feb61'
def sha(data):return hashlib.sha256(data).hexdigest()
def write(p,data):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT/'build/compact-r4-preflight-bundle-sep22-worker-v1');p.add_argument('--HC',type=Path,default=ROOT/'build/hc-pad-compact-r4-verify-teacher-sep22-worker-v3');p.add_argument('--output',type=Path,default=ROOT/'build/guard-HC-fast-composite-sep22-worker-v1');a=p.parse_args();base=a.base.resolve();hc=a.HC.resolve();out=a.output.resolve()
 if out.exists()or ROOT/'build'not in out.parents:raise ValueError('Fresh private output required')
 parent=json.loads((base/'overlay-manifest.json').read_text());ready=json.loads((base/'READY.json').read_text());hcready=json.loads((hc/'READY.json').read_text());profile=json.loads((hc/'HC-fast-source-profile.json').read_text())
 if parent['source_identity_sha256']!=GUARD_ID or not ready.get('ready_for_Root_guard_state_qualification'):raise ValueError('Registered guard origin required')
 for r in ready['receipts']+ready['artifacts']:
  if sha((base/r['path']).read_bytes())!=r['sha256']:raise ValueError('Guard program/artifact drift')
 if not hcready.get('active_leaf_V8_fresh97PASS')or profile['active_leaf_manifest_sha256']!=HC_LEAF or profile['HC_down_AIR_sha256']!=HC_AIR or sha((hc/'hc-pad.air').read_bytes())!=HC_AIR:raise ValueError('Exact registered FASTV8 DOWN AIR required')
 leaf=Path(profile['active_leaf']);leafm=json.loads((leaf/'manifest.json').read_text())
 if sha((leaf/'manifest.json').read_bytes())!=HC_LEAF or leafm.get('private_math_recipe')!='Metal4.1/O3/default-fast/original089-baseline-match' or not leafm.get('separate_down_and_safe_up_probe_translation_units') or leafm.get('candidate_AIR_sha256')!=HC_AIR:raise ValueError('Registered original compiler math recipe required')
 spec=importlib.util.spec_from_file_location('composite_overlay',ROOT/PRIVATE/'overlay.py');overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
 hcspec=importlib.util.spec_from_file_location('fast_hc_overlay',hc/'source/dev/benchmarks/hc_pad_verify_worker_sep22/overlay.py');hcoverlay=importlib.util.module_from_spec(hcspec);hcspec.loader.exec_module(hcoverlay)
 files=[];paths=set()
 for r in parent['files']:
  rel=r['path'];data=(base/'source'/rel).read_bytes()
  if sha(data)!=r['sha256']:raise ValueError('Guard source drift:'+rel)
  result=overlay.transform(rel,data.decode(),hcoverlay).encode();write(out/'source'/rel,result);files.append({'path':rel,'sha256':sha(result),'parent_sha256':sha(data),'changed':data!=result});paths.add(rel)
 for r in json.loads((base/'semantic-extra-source-seal.json').read_text())['sources']:
  rel=r['path'];data=(base/'source'/rel).read_bytes()
  if sha(data)!=r['sha256']:raise ValueError('Guard helper drift')
  write(out/'source'/rel,data);files.append({'path':rel,'sha256':sha(data),'parent_semantic_copy':True});paths.add(rel)
 sources=['dev/benchmarks/hc_pad_verify_worker_sep22/worker_bridge.hpp','dev/benchmarks/hc_pad_verify_worker_sep22/overlay.py','dev/benchmarks/hc_pad_verify_worker_sep22/policy_cpu.cpp','dev/benchmarks/hc_pad_producer_sep22/abi.hpp','dev/benchmarks/hc_pad_producer_sep22/candidate.metal']
 hcm=json.loads((hc/'overlay-manifest.json').read_text());hcrecords={r['path']:r['sha256']for r in hcm['files']}
 for rel in sources:
  data=(hc/'source'/rel).read_bytes()
  if sha(data)!=hcrecords[rel]:raise ValueError('Registered HC source drift:'+rel)
  if rel in paths:raise ValueError('Unexpected HC source collision:'+rel)
  write(out/'source'/rel,data);files.append({'path':rel,'sha256':sha(data),'registered_HC_copy':True});paths.add(rel)
 names=('policy.hpp','overlay.py','prepare.py','worker.mk','witness.py')
 for name in names:
  rel=PRIVATE/name;data=(ROOT/rel).read_bytes();write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'new':True})
 parts={name:sha((ROOT/PRIVATE/name).read_bytes())for name in names};parts.update({'guard_parent_manifest':sha((base/'overlay-manifest.json').read_bytes()),'guard_source_identity':GUARD_ID,'HC_overlay':sha((hc/'source/dev/benchmarks/hc_pad_verify_worker_sep22/overlay.py').read_bytes()),'HC_bridge':sha((hc/'source/dev/benchmarks/hc_pad_verify_worker_sep22/worker_bridge.hpp').read_bytes()),'HC_fast_leaf_manifest':HC_LEAF,'HC_fast_down_AIR':HC_AIR,'scope':'singleton-VerifyR4-C1-Bundle1-HCfastV8-original-guard-epoch-and-H97-semantic-retained-v1'})
 identity=sha(json.dumps(parts,sort_keys=True,separators=(',',':')).encode());rel=PRIVATE/'source_identity.hpp';data=f'#pragma once\nnamespace splash::flash::guard_hc_fast_composite_sep22 {{inline constexpr char kCompositeSourceIdentitySha256[]="{identity}";}}\n'.encode();write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'generated':True})
 frozen=[]
 for r in parent['frozen_inputs']:
  if r['path']=='splash.metallib':continue
  data=(base/r['path']).read_bytes()
  if sha(data)!=r['sha256']:raise ValueError('Original Core/AIR drift')
  write(out/r['path'],data);frozen.append({'path':r['path'],'sha256':sha(data),'source':str(base/r['path'])})
 original=Path(parent['base']);src=original/'compact-plan.air';data=src.read_bytes();rel=Path('reused/air/compact-qualified.air');write(out/rel,data);frozen.append({'path':str(rel),'sha256':sha(data),'source':str(src)})
 src=hc/'hc-pad.air';data=src.read_bytes();rel=Path('hc-pad.air');write(out/rel,data);frozen.append({'path':str(rel),'sha256':sha(data),'source':str(src)})
 link=(base/'link-inputs.mk').read_text()+'AIRS += $(BUILD)/reused/air/compact-qualified.air\n';write(out/'link-inputs.mk',link.encode())
 manifest={'schema':'guard-C1-HCfastV8-VerifyR4-composite-source-v1','base':str(base),'HC_origin':str(hc),'guard_source_identity':GUARD_ID,'HC_fast_leaf_manifest_sha256':HC_LEAF,'HC_down_AIR_sha256':HC_AIR,'source_identity_sha256':identity,'identity_parts':parts,'files':files,'changed_paths':sorted(overlay.CHANGED),'rebuild':parent['rebuild'],'fresh_host_TUs':50,'frozen_inputs':frozen,'link_make_sha256':sha(link.encode()),'new_public_store_API_shape_changed':False,'resource_lease':'original1281/not-private807','original76AIR_plus_one_integer_plus_FASTdown_only':True,'new_GPU_allocation_bytes':0,'GPU_work':False,'model_capture_payload_reads':False,'Root_OLD_HC_full_proof_not_reused_as_new_composite_proof':True,'whole_composite_state_qualified':False}
 write(out/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps({'prepared':str(out),'sources':len(files),'source_identity_sha256':identity,'GPU_work':False,'HC_shader_compile':False}))
if __name__=='__main__':main()
