#!/usr/bin/env python3
"""Seal FIRST R4/N16 component sources and existing binary inputs; no GPU/model IO."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/expert_r4_cohort_sep22')
def sha(data):return hashlib.sha256(data).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(d)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,default=ROOT/'build/prefill-qsa-twopass-sep21-worker-v1');p.add_argument('--output',type=Path,default=ROOT/'build/expert-r4-cohort-n16-sep22-component-v1');a=p.parse_args();parent=a.parent.resolve();out=a.output.resolve()
 if out.exists()or ROOT/'build' not in out.parents:raise ValueError('NEW private component output required')
 pm=parent/'overlay-manifest.json';manifest=json.loads(pm.read_text());records=[]
 def source(rel,data,**meta):write(out/'source'/rel,data);records.append({'path':str(rel),'sha256':sha(data),'bytes':len(data),**meta})
 for r in manifest['files']:
  data=(parent/'source'/r['path']).read_bytes()
  if sha(data)!=r['overlay_sha256']:raise ValueError('Parent frozen source drift: '+r['path'])
  source(Path(r['path']),data,parent_source=True)
 for path in sorted((ROOT/PRIVATE).iterdir()):
  if path.is_file()and path.suffix in ('.hpp','.h','.metal','.mm','.mk','.py','.md','.json'):source(PRIVATE/path.name,path.read_bytes(),component_source=True)
 for name in ('prefill4k_allrows_qmv_oracle.mm','prefill4k_allrows_qmv_reference.hpp','prefill4k_allrows_qmv_one_layer.hpp','prefill4k_allrows_qmv_probe.h','prefill4k_allrows_gathered_mpp_probe.metal'):
  source(Path('dev/benchmarks')/name,(ROOT/'dev/benchmarks'/name).read_bytes(),frozen_oracle_helper=True)
 # C++ baseline helper references this historical header/guard pair. They are
 # closed copied dependencies, not an additional producer or live include path.
 for name in ('FlashGatheredI8QMV.hpp',):
  source(Path('runtime/flash')/name,(ROOT/'build/prefill4k-allrows-qmv-c2-component/source/runtime/flash'/name).read_bytes(),frozen_oracle_helper=True)
 source(Path('runtime/metal/MetalBackend.mm'),(ROOT/'runtime/metal/MetalBackend.mm').read_bytes(),fresh_unmodified_backend_snapshot=True)
 probe=(ROOT/'dev/benchmarks/prefill4k_allrows_gathered_mpp_probe.metal').read_text().replace('"prefill4k_allrows_gathered_mpp.metal"','"metal/kernels/shared/flash_gathered_mpp.metal"')
 source(PRIVATE/'parent_gathered_probe.metal',probe.encode(),parent_arithmetic_probe=True)
 identity_parts={n:sha((ROOT/PRIVATE/n).read_bytes())for n in ('abi.hpp','kernels.metal','quality.hpp','PREREGISTRATION.md')}
 identity_parts['backend_snapshot']=sha((ROOT/'runtime/metal/MetalBackend.mm').read_bytes());identity_parts['parent_manifest']=sha(pm.read_bytes());identity_parts['schema']='R4N16-balanced-char4-one-partial-XOR32-cohort4-slots10-D27D12-NKplus31-u23-FTZ4lambda-v1'
 identity=sha(json.dumps(identity_parts,sort_keys=True,separators=(',',':')).encode())
 source(PRIVATE/'source_identity.hpp',f'#pragma once\ninline constexpr char kExpertR4SourceIdentitySha256[]="{identity}";\n'.encode(),generated_identity=True)
 inputs={'COHORT_FROZEN_HOST':[],'COHORT_FROZEN_AIRS':[]};links=[]
 def freeze(path,rel,cat):
  data=path.read_bytes();write(out/rel,data);inputs[cat].append(rel.as_posix());links.append({'private_path':rel.as_posix(),'source_path':str(path),'sha256':sha(data),'bytes':len(data),'category':cat})
 for line in (parent/'link-inputs.mk').read_text().splitlines():
  cat,raw=line.split(' := ',1)
  for token in raw.split():
   if not token.startswith('$(BUILD)/'):raise ValueError('Unsealed parent link token')
   rel=Path(token[9:]);path=parent/rel
   if path.name in ('FlashWorker.o','MetalBackend.o','flash_gathered_mpp.air'):continue
   freeze(path,Path('reused/parent')/rel,'COHORT_FROZEN_AIRS'if cat=='AIRS'else'COHORT_FROZEN_HOST')
 for path in sorted((parent/'host').glob('*.o')):
  if path.name!='FlashWorker.o':freeze(path,Path('reused/parent-host')/path.name,'COHORT_FROZEN_HOST')
 # Include every directly linked parent AIR not already represented in its list.
 for path in sorted(parent.glob('*.air')):
  if path.name!='flash_gathered_mpp.air':freeze(path,Path('reused/parent-air')/path.name,'COHORT_FROZEN_AIRS')
 freeze(ROOT/'build/prefill4k-allrows-qmv-one-layer/probe.air',Path('reused/old-bucket-probe.air'),'COHORT_FROZEN_AIRS')
 mk='\n'.join(cat+' := '+' '.join('$(COHORT_BUILD)/'+n for n in names)for cat,names in inputs.items())+'\n';write(out/'component-inputs.mk',mk.encode())
 seal={'schema':'splash-first-expert-r4-n16-cpu-source-seal-v1','parent_build':str(parent),'parent_manifest_sha256':sha(pm.read_bytes()),'source_identity_sha256':identity,'identity_parts':identity_parts,'sources':records,'link_inputs':links,'link_make_sha256':sha(mk.encode()),'gpu_work':False,'model_payload_reads':False,'capture_payload_reads':False,'model_or_capture_hashes':False}
 write(out/'source-seal.json',(json.dumps(seal,indent=2)+'\n').encode());print(json.dumps({'sealed':str(out),'source_identity_sha256':identity,'sources':len(records),'binary_inputs':len(links),'gpu_work':False}))
if __name__=='__main__':main()
