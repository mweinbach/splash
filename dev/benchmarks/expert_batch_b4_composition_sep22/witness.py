#!/usr/bin/env python3
"""CPU-only current header/source/float76+integer2 compiled dependency audit."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parents[3];PRIVATE=Path('dev/benchmarks/expert_batch_b4_composition_sep22')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);a=p.parse_args();b=a.build.resolve();m=json.loads((b/'overlay-manifest.json').read_text());c=json.loads((b/'compiled-build.json').read_text());parent=Path(m['base']);checks=[]
 def req(v,k):
  checks.append({'check':k,'pass':bool(v)})
  if not v:raise ValueError('composition witness:'+k)
 for r in m['files']:req(sha((b/'source'/r['path']).read_bytes())==r['sha256'],'sealed source '+r['path'])
 for r in m['frozen_inputs']:req(sha((b/r['path']).read_bytes())==r['sha256'],'sealed original/current AIRCore '+r['path'])
 req(c['source_identity_sha256']==m['source_identity_sha256'],'compiled source identity');req(sha(json.dumps(m['identity_parts'],sort_keys=True,separators=(',',':')).encode())==m['source_identity_sha256'],'canonical own source tuple')
 req(len(c['objects'])==50 and len(m['Core'])==4,'actual50+4 census');deps=[];store=[];BQSA=[]
 for r in c['objects']:
  obj=b/r['path'];req(sha(obj.read_bytes())==r['sha256'],'compiled fresh '+r['path']);rule=obj.with_suffix('.d').read_text().replace('\\\n','').split('\n',1)[0];paths=[Path(x).resolve()for x in rule.split(': ',1)[1].split()];project=[x for x in paths if ROOT in x.parents];req(all(b/'source'in x.parents for x in project),'all private .d '+r['path']);deps.extend(str(x.relative_to(b/'source'))for x in project)
  if b/'source/runtime/flash/FlashInt8ExpertStore.hpp'in paths:store.append(r['path'])
  if b/'source/dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp'in paths:BQSA.append(r['path'])
 req(len(store)==6,'actual6Store consumers rebuilt');req(len([r for r in m['frozen_inputs']if r['category']=='currentFloatAIR'])==76,'current original76 with currentwideLeaf');req(len([r for r in m['frozen_inputs']if r['category']=='integerAIR'])==2,'only qualifiedinteger2 added')
 src=b/'source';s=(src/'runtime/flash/FlashInt8ExpertStore.mm').read_text();old=(parent/'source/runtime/flash/FlashInt8ExpertStore.mm').read_text();first='void FlashInt8ExpertStore::addGateUp(';last='void FlashInt8ExpertStore::addFixedSG2PrefillGateUp(';req(s[s.index(first):s.index(last)]==old[old.index(first):old.index(last)],'all5floatstageGU-down-body binding literal CURRENT')
 s=(src/'runtime/flash/FlashBatchVerify.cpp').read_text();old=(parent/'source/runtime/flash/FlashBatchVerify.cpp').read_text();first='metal::CommandTiming FlashBatchVerify::commitBatch(';req(s[s.index(first):]==old[old.index(first):],'allcommit0-null/abort/owner/ticket literal CURRENT');req('eligible(lanes,rows,true)'in s,'actualrows4andactive2or4 target only')
 policy=(src/'dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp').read_text();req('selected && lanes == 4 && rows == 2048 && allFresh'in policy,'BQSA actual4 only');req('SPLASH_FLASH_MTP_DRAFT_DEPTH'in policy and'BATCH_MTP_TEACHER_CACHE_ONLY'in policy,'BQSA early MTP3deps')
 for rel in ['runtime/flash/FlashForward.cpp','runtime/flash/FlashBatchForward.cpp','runtime/flash/FlashBatchMTPForward.cpp','runtime/flash/FlashMTP.cpp','runtime/flash/FlashMTP.hpp','runtime/flash/FlashGDNBatchILP.cpp','dev/benchmarks/prefill_qsa_twopass_sep21/twopass.cpp','dev/benchmarks/prefill_qsa_twopass_sep21/candidate.metal']:
  req((src/rel).read_bytes()==(parent/'source'/rel).read_bytes(),'CURRENT prefill/head/AR/numericbody '+rel)
 worker=(src/'runtime/flash/FlashWorker.mm').read_text();req('batch_prefill_twopass_numerical_parent_routes'in worker and'target_numeric_parent_sha256'in worker,'explicit raw numeric parent separateexecdisplay')
 for r in c['artifacts']:req(sha((b/r['path']).read_bytes())==r['sha256'],'compiled artifact '+r['path'])
 out={'schema':'current-B4only-BQSA-integer-composition-CPUclosure-v1','pass':True,'source_identity_sha256':m['source_identity_sha256'],'BQSA_source_policy_sha256':m['BQSA_source_policy_sha256'],'checks':checks,'actual_host_TUs':50,'Core':4,'currentFloatAIR':76,'integerAIR':2,'actual_Store_consumers':store,'actual_BQSA_consumers':BQSA,'private_dependencies':sorted(set(deps)),'artifacts':c['artifacts'],'GPU_work':False,'payload_reads':False,'whole_state_quality_timing_qualified':False};(b/'compiled-cpu-seal.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'checks':len(checks),'Store_consumers':len(store),'BQSA_consumers':len(BQSA),'GPU_work':False}))
if __name__=='__main__':main()
