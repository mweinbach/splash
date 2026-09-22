#!/usr/bin/env python3
"""Current source rebase for B4-only packedV prefill plus exact integer batch verify."""
from pathlib import Path
import argparse,hashlib,json
from integer_transform import transform as integer_transform
ROOT=Path(__file__).resolve().parents[3];PRIVATE=Path('dev/benchmarks/expert_batch_b4_composition_sep22');BQSA=Path('dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp')
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,b):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def once(s,a,b):
 if s.count(a)!=1:raise ValueError('composition source anchor differs:'+a[:100])
 return s.replace(a,b)
def policy_transform(s):
 s=once(s,'batch-real2or4-allfresh2048-existing-packedV-twopass-v1','batch-real4-MTP3-allfresh2048-existing-packedV-twopass-v2')
 s=once(s,'whole real2/4 cohort fresh2048 only;','whole actual real4 MTP3 cohort fresh2048 only;')
 a=s.index('inline bool requested()');b=s.index('constexpr bool eligible',a)
 s=s[:a]+'''enum class SwitchState:uint8_t {Missing,Disabled,Enabled};
inline SwitchState state(const char *raw) {
 if(!raw)return SwitchState::Missing;
 if(std::string_view(raw)=="0")return SwitchState::Disabled;
 if(std::string_view(raw)=="1")return SwitchState::Enabled;
 throw std::invalid_argument(std::string(flag)+" must be exactly missing, 0 or 1");
}
inline bool requested() {
 static const SwitchState frozen=[] {
  const auto value=state(std::getenv(flag));
  if(value==SwitchState::Enabled) {
   for(const char *dep:{"SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_BATCH_QSA_BULK_PREFILL",
    "SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21","SPLASH_FLASH_QSA_F32","SPLASH_FLASH_QSA_MPP",
    "SPLASH_FLASH_QSA_ROW_TILES","SPLASH_FLASH_QSA_BULK_PREFILL","SPLASH_FLASH_QSA_BULK_PREFILL_SG8",
    "SPLASH_FLASH_MTP","SPLASH_FLASH_BATCH_MTP","SPLASH_FLASH_BATCH_MTP_PREFILL",
    "SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY","SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY"}) {
    const char *raw=std::getenv(dep);
    if(!raw||std::string_view(raw)!="1")throw std::invalid_argument(std::string(flag)+"=1 requires "+dep+"=1");
   }
   const char *depth=std::getenv("SPLASH_FLASH_MTP_DRAFT_DEPTH");
   if(!depth||std::string_view(depth)!="3")throw std::invalid_argument(std::string(flag)+"=1 requires MTP draft depth exactly3");
  }
  return value;
 }();
 if(state(std::getenv(flag))!=frozen)throw std::logic_error(std::string(flag)+" changed after policy freeze");
 return frozen==SwitchState::Enabled;
}
'''+s[b:]
 s=once(s,'selected && (lanes == 2 || lanes == 4) && rows == 2048 && allFresh','selected && lanes == 4 && rows == 2048 && allFresh')
 s=once(s,'d87b4a84305ab37d690bf44a742bfbd9fb8737234419dc0edc1326e552a1516a','NEW_B4_POLICY_SOURCE_SHA')
 return s

def transform(rel,s):
 s=integer_transform(rel,s)
 if rel==str(BQSA):s=policy_transform(s)
 if rel=='runtime/flash/FlashBatchPrefill.cpp':s=once(s,';private-batch-real2or4-allfresh2048-existing-packedV-twopass-v1',';private-batch-real4-MTP3-allfresh2048-existing-packedV-twopass-v2')
 if rel=='runtime/flash/FlashInt8ExpertStore.hpp':s=once(s,'  [[nodiscard]] bool compactNativeBatchVerifyEnabled() const;','  [[nodiscard]] const std::string &compactNativeBatchVerifyNumericParent() const;\n  [[nodiscard]] bool compactNativeBatchVerifyEnabled() const;')
 if rel=='runtime/flash/FlashInt8ExpertStore.mm':
  s=once(s,'  std::string numericalIdentity;','  std::string numericalIdentity;\n  std::string compactNumericParentIdentity;')
  s=once(s,'    if (compactBatchVerify)\n      derivative +=','    compactNumericParentIdentity = hash(derivative.data(), derivative.size());\n    if (compactBatchVerify)\n      derivative +=')
  s=once(s,'bool FlashInt8ExpertStore::compactNativeBatchVerifyEnabled() const {','const std::string &FlashInt8ExpertStore::compactNativeBatchVerifyNumericParent() const {\n  if(!impl_)fail("compact batch numerical parent Store disposed");\n  return impl_->compactNumericParentIdentity;\n}\nbool FlashInt8ExpertStore::compactNativeBatchVerifyEnabled() const {')
 if rel=='runtime/flash/FlashWorker.mm':
  s=once(s,'      << R"(,"batch_prefill_twopass_numerical_identity":)"','      << R"(,"batch_prefill_twopass_numerical_parent_routes":)" << json::quote(forward_.kernelRoutes())\n      << R"(,"batch_prefill_twopass_numerical_identity":)"')
  s=once(s,'<<R"(,"source_identity_sha256":)"<<json::quote(compact_native_batch_verify_sep22::kSourceIdentitySha256)','<<R"(,"source_identity_sha256":)"<<json::quote(compact_native_batch_verify_sep22::kSourceIdentitySha256)\n      <<R"(,"target_base_numeric_parent_sha256":)"<<(persistedExperts?json::quote(persistedExperts->compactNativeBatchVerifyNumericParent()):"null")\n      <<R"(,"target_numeric_parent_sha256":)"<<(persistedExperts?json::quote(prefill_qsa_twopass_sep21::numericalIdentity(dense_w8a8_sep21::numericalIdentity(\n          gdn_prefill_fma_sep21::numericalIdentity(persistedExperts->compactNativeBatchVerifyNumericParent(),\n              gdn_prefill_fma_sep21::requested()),forward_.kernelRoutes()),prefill_qsa_twopass_sep21::requested())):"null")')
 return s

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=ROOT/'build/batch-prefill-twopass-restored-sep22-worker-v3');p.add_argument('--output',type=Path,required=True);a=p.parse_args();base=a.base.resolve();out=a.output.resolve();target=ROOT/'build/compact-native-batch-verify-clock-sep22-worker-v1c'
 if out.exists()or ROOT/'build'not in out.parents:raise ValueError('fresh private output required')
 parent=json.loads((base/'overlay-manifest.json').read_text());seal=json.loads((base/'compiled-cpu-seal.json').read_text());expected={'splash-flash':'640c4af9d23aa048a5b3e0d8b1fd1acc02a925d6b32accd09ae2849ac09701fa','splash.metallib':'1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf'}
 for n,v in expected.items():
  if sha((base/n).read_bytes())!=v:raise ValueError('CURRENT parent artifact mismatch:'+n)
 helper_names=['bridge.hpp','policy_cpu.cpp','counter_audit.py','integer_transform.py','prepare.py','build.py','witness.py','bqsa_policy_cpu.mm','adapter.py','test_adapter.py']
 policy_before=(base/'source'/BQSA).read_text();policy_material=policy_transform(policy_before).encode()+sha((base/'compiled-cpu-seal.json').read_bytes()).encode()+sha((ROOT/PRIVATE/'integer_transform.py').read_bytes()).encode();policy_sha=sha(policy_material)
 parts={'current_parent_manifest':sha((base/'overlay-manifest.json').read_bytes()),'current_parent_seal':sha((base/'compiled-cpu-seal.json').read_bytes()),'new_B4_policy_sha':policy_sha,'qualified_integer_source':json.loads((target/'overlay-manifest.json').read_text())['source_identity_sha256'],'scope':'B4onlyMTP3fresh2048-pref-plus-batch-target-rows4-active2or4-currentHeader-float76plusinteger2-v1','private_sources':{n:sha((ROOT/PRIVATE/n).read_bytes())for n in helper_names}}
 identity=sha(json.dumps(parts,sort_keys=True,separators=(',',':')).encode());files=[];changed=[]
 for rec in parent['files']:
  rel=rec['path'];raw=(base/'source'/rel).read_bytes()
  if sha(raw)!=rec['sha256']:raise ValueError('CURRENT source drift:'+rel)
  new=transform(rel,raw.decode()).replace('NEW_B4_POLICY_SOURCE_SHA',policy_sha).encode();write(out/'source'/rel,new);files.append({'path':rel,'sha256':sha(new),'parent_sha256':sha(raw),'changed':new!=raw})
  if new!=raw:changed.append(rel)
 for n in helper_names:
  rel=PRIVATE/n;raw=(ROOT/rel).read_bytes();write(out/'source'/rel,raw);files.append({'path':str(rel),'sha256':sha(raw),'new':True})
 rel=PRIVATE/'source_identity.hpp';raw=f'#pragma once\nnamespace splash::flash::compact_native_batch_verify_sep22 {{inline constexpr char kSourceIdentitySha256[]="{identity}";}}\n'.encode();write(out/'source'/rel,raw);files.append({'path':str(rel),'sha256':sha(raw),'generated':True})
 frozen=[]
 for rec in parent['Core']:
  raw=(base/rec['path']).read_bytes()
  if sha(raw)!=rec['sha256']:raise ValueError('CURRENT Core drift')
  write(out/rec['path'],raw);frozen.append({'path':rec['path'],'sha256':sha(raw),'category':'Core'})
 target_frozen=json.loads((target/'overlay-manifest.json').read_text())['frozen_inputs'];original_air_digests={r['sha256']for r in target_frozen if r['path'].endswith('.air')}
 restored=Path(parent['base']);rs=json.loads((restored/'compiled-cpu-seal.json').read_text());airs=[Path(x)for x in rs['metallib_link_command']if x.endswith('.air')]
 if len(airs)!=76:raise ValueError('Current restored exact76 AIRs required')
 for i,path in enumerate(airs):
  raw=path.read_bytes();digest=sha(raw)
  if path.name=='wide-dense-prefill.air':
   if digest!='bfe2157b084eb36a8781a3b3831a001ce1729c2e6c4fcdf740d3815afd35c81f':raise ValueError('CURRENT wideDense leaf drift')
  elif digest not in original_air_digests:raise ValueError('Original75 AIR input drift')
  rel=Path('air')/f'{i:03d}-{path.name}';write(out/rel,raw);frozen.append({'path':str(rel),'sha256':sha(raw),'original_path':str(path),'category':'currentFloatAIR'})
 for n,digest in [('compact-r8-plan.air','8c5dded43eb45df4ba37516288599fb9662888d18a418aeb59eec151185b8662'),('compact-r16-plan.air','d65c187dca32c8d4e5e2214eafd88b428c464ebe67ed08abe01797d04edfd2c0')]:
  raw=(target/n).read_bytes()
  if sha(raw)!=digest:raise ValueError('Qualified integer AIR drift')
  write(out/'air'/n,raw);frozen.append({'path':'air/'+n,'sha256':sha(raw),'category':'integerAIR'})
 for n in ['plan-r8.metal','plan-r16.metal','abi.hpp']:
  rel=PRIVATE/n;raw=(target/'source/dev/benchmarks/expert_batch_compact_verify_worker_sep22'/n).read_bytes();write(out/'source'/rel,raw);files.append({'path':str(rel),'sha256':sha(raw),'qualified_source_literal':True})
 m={'schema':'current-B4BQSA-plus-exact-batchinteger-source-v1','base':str(base),'files':files,'rebuild':parent['rebuild'],'Core':parent['Core'],'frozen_inputs':frozen,'source_identity_sha256':identity,'BQSA_source_policy_sha256':policy_sha,'identity_parts':parts,'changed_paths':sorted(changed),'new_GPU_allocation_beyond_current_parent':0,'GPU_work':False,'model_operand_response_capture_payload_reads':False,'whole_state_quality_timing_qualified':False};write(out/'overlay-manifest.json',(json.dumps(m,indent=2)+'\n').encode());print(json.dumps({'prepared':str(out),'sources':len(files),'fresh_host_TUs':len(m['rebuild']),'current_Core':len(m['Core']),'AIRs':len(frozen)-4,'source_identity':identity,'B4policy':policy_sha,'GPU_work':False}))
if __name__=='__main__':main()
