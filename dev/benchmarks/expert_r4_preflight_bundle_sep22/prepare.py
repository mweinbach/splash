#!/usr/bin/env python3
"""CPU guard-only private source preparation; freeze original shader library."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_r4_preflight_bundle_sep22')


def sha(data):return hashlib.sha256(data).hexdigest()


def write(p,data):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--parent',type=Path,default=ROOT/'build/compact-native-r4-verify-teacher-sep22-worker-v1b')
 p.add_argument('--output',type=Path,default=ROOT/'build/compact-r4-preflight-bundle-sep22-worker-v1')
 a=p.parse_args();base=a.parent.resolve();out=a.output.resolve()
 if out.exists()or ROOT/'build'not in out.parents:raise ValueError('Fresh private output required')
 parent=json.loads((base/'overlay-manifest.json').read_text());ready=json.loads((base/'READY.json').read_text())
 if not ready.get('ready_for_root_qualification')or parent['source_identity_sha256']!='8ed20bad23072a34a50bb0516b058e991ae396c7e8b88900193d487034735a79':raise ValueError('Admitted original compact parent required')
 for r in ready['compiled_artifacts']+ready['receipts']:
  if sha((base/r['path']).read_bytes())!=r['sha256']:raise ValueError('Compact parent CPU artifact drift:'+r['path'])
 spec=importlib.util.spec_from_file_location('preflight_overlay',ROOT/PRIVATE/'overlay.py');overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
 files=[]
 for r in parent['files']:
  rel=r['path'];data=(base/'source'/rel).read_bytes()
  if sha(data)!=r['sha256']:raise ValueError('Compact parent source drift:'+rel)
  result=overlay.transform(rel,data.decode()).encode();write(out/'source'/rel,result)
  files.append({'path':rel,'sha256':sha(result),'parent_sha256':sha(data),'changed':data!=result})
 semantic=json.loads((base/'semantic-source-seal.json').read_text())
 paths={r['path']for r in files}
 for r in semantic['sources']:
  rel=r['path'];data=(base/'source'/rel).read_bytes()
  if sha(data)!=r['sha256']:raise ValueError('Parent semantic source drift:'+rel)
  if rel not in paths:write(out/'source'/rel,data);files.append({'path':rel,'sha256':sha(data),'parent_semantic_copy':True});paths.add(rel)
 names=('policy.hpp','guard.hpp','overlay.py','prepare.py','worker.mk','witness.py','PLAN.md','baseline_extract.py','guard_decision_cpu.cpp','policy_cpu.cpp')
 for name in names:
  rel=PRIVATE/name;data=(ROOT/rel).read_bytes();write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'new':True})
 parts={name:sha((ROOT/PRIVATE/name).read_bytes())for name in names}
 parts.update({'parent_manifest':sha((base/'overlay-manifest.json').read_bytes()),'parent_semantic_seal':sha((base/'semantic-source-seal.json').read_bytes()),'original_compact_source_identity':parent['source_identity_sha256'],'original_metallib':sha((base/'splash.metallib').read_bytes()),'scope':'CPU-only-one13view-preflight-R4-native6stage-private-noncopyable-singleconsume-StoreLayerGraph-local-bundle-v1'})
 identity=sha(json.dumps(parts,sort_keys=True,separators=(',',':')).encode());rel=PRIVATE/'source_identity.hpp'
 data=f'#pragma once\nnamespace splash::flash::compact_r4_preflight_sep22 {{inline constexpr char kPreflightSourceIdentitySha256[]="{identity}";}}\n'.encode();write(out/'source'/rel,data);files.append({'path':str(rel),'sha256':sha(data),'generated':True})
 link=(base/'link-inputs.mk').read_bytes();write(out/'link-inputs.mk',link)
 frozen=[]
 for r in parent['frozen_inputs']:
  data=(base/r['path']).read_bytes()
  if sha(data)!=r['sha256']:raise ValueError('Original frozen input drift')
  write(out/r['path'],data);frozen.append({'path':r['path'],'sha256':sha(data),'source':str(base/r['path'])})
 # Original integer planner is already linked into the frozen parent library.
 # No AIR, shader compiler invocation or new pipeline occurs in this profile.
 lib=(base/'splash.metallib').read_bytes();write(out/'splash.metallib',lib)
 frozen.append({'path':'splash.metallib','sha256':sha(lib),'source':str(base/'splash.metallib')})
 # Generate old guards from the admitted parent's source, never model values.
 extractor_spec=importlib.util.spec_from_file_location('preflight_baseline_extract',ROOT/PRIVATE/'baseline_extract.py');extractor=importlib.util.module_from_spec(extractor_spec);extractor_spec.loader.exec_module(extractor)
 baseline,records=extractor.build((base/'source/runtime/flash/FlashInt8ExpertStore.mm').read_text(),(base/'source/runtime/flash/FlashMoEBuckets.cpp').read_text())
 rel=PRIVATE/'baseline_generated.hpp';write(out/'source'/rel,baseline.encode());files.append({'path':str(rel),'sha256':sha(baseline.encode()),'generated_verbatim_old_guards':True})
 manifest={'schema':'CPU-only-compactR4-complete-local-preflight-source-v1','base':str(base),'parent_manifest_sha256':parts['parent_manifest'],'parent_READY_sha256':sha((base/'READY.json').read_bytes()),'parent_semantic_seal_sha256':parts['parent_semantic_seal'],'original_compact_source_identity':parent['source_identity_sha256'],'source_identity_sha256':identity,'identity_parts':parts,'files':files,'rebuild':parent['rebuild'],'host_TUs':50,'changed_paths':sorted(overlay.CHANGED),'frozen_inputs':frozen,'link_make_sha256':sha(link),'baseline_guard_extraction_records':records,'all_original_public_native_producers_fullvalidation_unchanged':True,'shader_math_workspace_changed':False,'additional_GPU_allocation_bytes':0,'GPU_work':False,'model_capture_payload_reads':False,'whole_state_quality_performance_qualified':False}
 write(out/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
 print(json.dumps({'prepared':str(out),'source_identity_sha256':identity,'sources':len(files),'GPU_work':False,'shader_rebuild':False}))


if __name__=='__main__':main()
