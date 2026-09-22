#!/usr/bin/env python3
"""Saved JSON/source-only lease census; no model payloads or backend."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest_names(names):return hashlib.sha256(('\n'.join(sorted(names))+'\n').encode()).hexdigest()
def main():
 report=ROOT/'build/release/flash/sep21-teacher-bulk-ab-qsa-model-and-quality-v1.json'
 operands=ROOT/'install/local-models/Flash-Next-operands-v1/manifest.json'
 parentfloat=BASE/'source/runtime/flash/FlashFloatDenseCache.cpp'
 source=parentfloat.read_text()
 table=re.findall(r'\{"([^"]+)",\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\}',source)
 assert len(table)==29
 policy={(role,int(n),int(k),int(bits),int(group)):[int(a),int(b),int(c),int(d)] for role,n,k,bits,group,a,b,c,d in table}
 entries=json.loads(operands.read_text())['entries']
 f32=[e for e in entries if e['format']=='F32']; bf16=[e for e in entries if e['format']=='BF16']
 assert len(f32)==508 and len(bf16)==509
 def role(name):
  if name=='language_model.model.hyper_connection_mixer.input_mix_weight_up':return 'hc_up'
  m=re.fullmatch(r'language_model.model.layers.(\d+).(.+)',name)
  if not m:return ''
  r=m[2]
  if r in ['attn_hyper_connection.input_mix_weight_up','mlp_hyper_connection.input_mix_weight_up']:return 'hc_up'
  return r
 kept=[]; union=[]
 for e in f32:
  name=e['projection'];s=e['source']
  v=policy.get((role(name),s['output_size'],s['input_size'],s['bits'],s['group_size']))
  if v and any(t>=0 for t in v):union.append(e)
  if v and v[0]>=0:kept.append(e)
 assert len(kept)==118 and sum(e['allocated_bytes'] for e in kept)==3247964160
 assert len(union)==296 and sum(e['allocated_bytes'] for e in union)==12097945600
 omitted_bf16=[]
 for e in bf16:
  m=re.fullmatch(r'language_model.model.layers.(\d+).(.+)',e['projection'])
  if not m:continue
  l=int(m[1]);r=m[2]
  if (l%4!=3 and r in ['linear_attn.in_proj_qkv','linear_attn.in_proj_z']) or (l%4==3 and r=='self_attn.q_proj'):omitted_bf16.append(e)
 assert len(omitted_bf16)==84 and sum(e['allocated_bytes'] for e in omitted_bf16)==3774873600
 initial=json.loads(report.read_text())['server_runs'][0]['initial_status']
 lease=initial['saved_operands_residency'];status=initial['persisted_operands']
 assert status['f32_tensors']==508 and status['bf16_tensors']==509
 assert sum(e['allocated_bytes'] for e in f32)==status['f32_mapped_payload_bytes']==14391705600
 assert sum(e['allocated_bytes'] for e in bf16)==status['bf16_mapped_payload_bytes']==8467251200
 w8=initial['identity'];assert w8['dense_w8a8_immutable_buffer_count']==168 and w8['dense_w8a8_fixed_cache_planned_bytes']==1890975744
 full512_count=lease['registered_base_allocation_count']-508-509-168
 full512_bytes=lease['registered_base_allocation_bytes']-14391705600-8467251200-1890975744
 assert full512_count==96 and full512_bytes==121174228992
 f32_removed=14391705600-3247964160; bf16_kept=8467251200-3774873600
 target_count=425+118+96+168;target_bytes=bf16_kept+3247964160+full512_bytes+1890975744
 assert target_count==807 and target_bytes==131005546496
 result={'schema':'pure-I8-teacher-singleton-lease-source-census-plan-v1','CPU_only':True,'GPU_executed':False,'model_payload_bytes_read':0,'model_payload_hashed':False,'build_authorized_or_performed':False,
 'base':str(BASE.relative_to(ROOT)), 'inputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [report,operands,parentfloat,BASE/'source/runtime/flash/FlashForward.cpp',BASE/'source/runtime/flash/FlashDenseCache.cpp',BASE/'source/runtime/flash/FlashWorker.mm',ROOT/'dev/benchmarks/dense_w8a8_residency_sep21/worker_bridge.hpp']},
 'parent':{'BF16_owners':509,'BF16_bytes':8467251200,'F32_actual_backing_owners':508,'F32_actual_backing_bytes':14391705600,'F32_planner_with_diagnostics_bytes':14391721984,'Full512_owners':96,'Full512_bytes':full512_bytes,'derived_W8_owners_already_included':168,'derived_W8_bytes':1890975744,'lease_owners':1281,'lease_bytes':145924161536},
 'proposed':{'BF16_persistent_owners':425,'BF16_persistent_bytes':bf16_kept,'F32_persistent_owners':118,'F32_persistent_bytes':3247964160,'Full512_owners':96,'Full512_bytes':full512_bytes,'derived_W8_owners':168,'derived_W8_bytes':1890975744,'lease_owners':target_count,'lease_bytes':target_bytes,'omitted_BF16_owners':84,'omitted_BF16_bytes':3774873600,'omitted_F32_owners':390,'omitted_F32_bytes':f32_removed,'omitted_total_owners':474,'omitted_total_bytes':14918615040,'F32_positive_union_transient_owners':178,'F32_positive_union_transient_bytes':8849981440,'F32_null_policy_backing_transient_owners':212,'F32_null_policy_backing_transient_bytes':2293760000},
 'membership':{'kept_F32_names':sorted(e['projection'] for e in kept),'omitted_BF16_names':sorted(e['projection'] for e in omitted_bf16),'F32_all_names_sha256':digest_names(e['projection'] for e in f32),'kept_F32_names_sha256':digest_names(e['projection'] for e in kept)},
 'implementation':{'new_flag':'SPLASH_FLASH_TEACHER_SINGLETON_LEASE_PRUNE_SEP22','flag0':'Original Parent getter/lease selection byte-identical','flag1':'Select copied persistent owner vectors only; require full508 F32 backing and full509 BF16 status; fixed R4 118 exact full-view membership and old audited BF1684 selection; retain W8 168 already present', 'source_changes':['FlashForward.cpp cachedOperandsOnly selector ONLY','new private header: strict parse, plan and selection validation','FlashWorker.mm early profile guard/static resource identity/startup unique census and lease status'], 'headers_existing_changed':False, 'minimum_rebuild':['FlashForward.o','FlashWorker.o'],'all_changed_header_consumers_rebuild_required':True,'old_phase_F32_selector_reusable_unmodified':False,'old_phase_selector_reason':'Its guard requires296; Parent actual full508 must remain and requires an independent strict508 guard','numerical_derivative_changed':False,'new_static_resource_profile_required':True,'batch_arena_flags_changed':False,'new_shader':False,'new_Core_API':False,'lease_phase_swapping':False,'backing_freed_or_unmapped':False,'planner_reservations_decreased':False,'original_Q4_loader_views_restored':False},
 'strict_profile_guards':['parse strict0/1 before backend','flag1 source-bound only to exact qualified I8 teacher Parent, not PhaseQ4','SAVED_OPERANDS_RESIDENT1, cacheFloat1, cacheDense1, DenseW8 selected1, Full512 allrows1 complete targets; no original-text/Q4-composite residency','check all508/509 saved full-view mappings and expected store identity before persistent selection','validate each118 selected F32 owner membership/shape/bytes/storage and no duplicate owner','audited84 BF16 source exact full-view membership; old cache collections remain unchanged','validate168 derived W8 owners already present, union807/131005546496 unique owners at startup','B1-qualified resource profile; no batch capability or batch arithmetic disabled silently','new quality status helper must bind resource profile/source SHA and807 census; old qualification branches unchanged'],
 'preserved':['all508 F32 constructor/plannedBytes/contains/tensor and original NULL-policy RAW branch','all509 BF16 backing/tensor/fallbacks','full I8 Prefill/Decode/Verify and trained MTP math','all kernels/metallib and command graph construction','all allocations/ledger/reservations/governor; fresh normal admission still required','all batch math/capabilities; B1 throughput qualification only','frozen22 semantic plan/graders and budget/actual API/pair counter definitions'],
 'qualification_required':['CPU saved metadata census + strict tamper tests + frozen source proof that arithmetic bodies/constructors unchanged','CPU old-profile flag0 and new-profile startup negatives before backend','fresh Root normal B1 canonical1warm3trial plus frozen22 semantic suite; actual resource admission/census recorded','performance/semantics not inherited from phase-saved-only result; no physical-pinning or schedule mechanism claim'], 'status':'SOURCE_CENSUS_PLAN_READY_NOT_BUILT'}
 out=ROOT/'build/release/flash/sep22-pure-I8-teacher-lease-only-source-census-plan-v1.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'artifact':str(out.relative_to(ROOT)),'sha256':sha(out),'parent':result['parent'],'proposed':result['proposed'],'status':result['status']},indent=2))
if __name__=='__main__':main()
