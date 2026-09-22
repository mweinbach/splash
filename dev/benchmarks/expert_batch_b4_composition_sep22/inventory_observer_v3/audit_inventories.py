#!/usr/bin/env python3
"""Additive Root-small-metadata inventory audit. Never opens reports, .bin or spills."""
from pathlib import Path
import argparse,hashlib,json,re
MAX_BYTES=262144

def require(v,m):
 if not v:raise ValueError(m)
def same(a,b):
 if type(a)is not type(b):return False
 if type(a)is dict:return set(a)==set(b)and all(same(a[k],b[k])for k in a)
 if type(a)is list:return len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
 return a==b
def is_sha(v):return type(v)is str and re.fullmatch('[0-9a-f]{64}',v)is not None
def positive(v):return type(v)is int and v>0

def audit_one(value,expected):
 e=[]
 def check(v,m):
  if not v:e.append(m)
 check(value.get('schema')=='Root-extracted-current-batch-native-frame-inventory-v1'and value.get('Root_extracted_metadata_only')is True,'Root small-metadata extraction envelope required')
 check(value.get('pairPASS')is True,'Root paired comparison PASS unavailable')
 check(is_sha(value.get('export_report_sha256'))and is_sha(value.get('compare_report_sha256')),'externally recorded paired report digest metadata unavailable')
 m=value.get('complete_manifest',{});check(m.get('schema')=='batchverify-export-v1'and m.get('complete')is True,'complete successful native manifest unavailable')
 common=m.get('common',{});check(same(common,value.get('pair_common')),'whole paired typed common tuple differs')
 for k,v in expected['common_fixed'].items():check(same(common.get(k),v),'current source common field differs:'+k)
 check(same(common.get('numeric_flags'),expected['numeric_flags']),'full47flag typed tuple changed/masked/missing')
 raw=common.get('BQSA_numeric_parent_routes_raw');check(type(raw)is str and bool(raw),'raw immutable Forward getter unavailable');check(is_sha(common.get('target_raw_numeric_parent_SHA')),'raw Store numerical parent unavailable')
 producer=m.get('producer',{});w=producer.get('worker',{});checks=producer.get('checks',{})
 for k,v in expected['worker_fixed'].items():check(same(w.get(k),v),'sealed worker/FP/header/AIR parent provenance differs:'+k)
 check(same(producer.get('BQSA_numeric_parent_routes_raw'),raw)and same(producer.get('kernel_routes'),raw),'producer raw getter route differs/marker stripped')
 check(same(producer.get('target_numeric_parent_SHA'),common.get('target_raw_numeric_parent_SHA')),'producer RAW parent differs, do not substitute overall wrapped hash')
 check(same(producer.get('BQSA_policy_SHA'),common.get('BQSA_source_policy_SHA')),'producer policy differs')
 for k,v in expected['campaign_counts'].items():check(same(checks.get(k),v),'actual campaign counts differ:'+k)
 check(type(checks.get('local_pointer_owner_redzone_checks'))is int and checks['local_pointer_owner_redzone_checks']>0,'native local owner/redzone checks unavailable')
 check(checks.get('source_replacement_guard_runtime_proved')is True and checks.get('foreign_trunk_rejection_runtime_proved')is False and checks.get('worker_deadline_callback_proved')is False,'native scope/foreign/deadline proof misclaimed')
 selected=common.get('selected_checkpoints');check(type(selected)is list and same(selected,checks.get('selected_full_checkpoints'))and selected in expected['partition_order'],'selected fullphysical labels/order differs')
 frames=m.get('frames',[]);files=value.get('full_inventory',[]);check(type(frames)is list and type(files)is list,'complete frame/file inventory unavailable')
 if type(frames)is not list or type(files)is not list:return None,e
 labels=[f.get('label')for f in frames if type(f)is dict];names=[f.get('name')for f in files if type(f)is dict]
 check(len(frames)==len(labels)==len(set(labels))==value.get('frames')==expected['base_frames_per_campaign']+len(selected),'exact156base plus actual selected fullphysical frames required, no capped progress list')
 check(len(files)==len(names)==len(set(names))==len(frames),'binary-file inventory count differs')
 filemap={f.get('name'):f.get('bytes')for f in files if type(f)is dict}
 for f in frames:
  if type(f)is not dict:e.append('nonliteral frame metadata');continue
  label=f.get('label');check(type(label)is str and filemap.get(str(label)+'.bin')==f.get('bytes'),'full binary name/byte extent differs')
  check(positive(f.get('bytes'))and positive(f.get('planes'))and type(f.get('live_bytes'))is int and 0<=f.get('live_bytes',-1)<=f.get('bytes',-1)and is_sha(f.get('sha256')),'invalid exact frame/plane extent or recorded comparator hash')
 check(all(positive(x)for x in filemap.values()),'invalid file extent')
 spill=m.get('spill_bytes');check(type(spill)is int and 0<spill<=4<<30 and all(type(x)is int for x in filemap.values())and spill==sum(filemap.values()),'bounded complete exact spill inventory sum differs')
 labelset=set(labels);seen=[x for x in expected['legacy18']if x+'.max16-arena'in labelset];check(len(seen)==18,'all18 actual legacy max16 backing checkpoints required')
 check(all(x+'.max16-arena'in labelset for x in expected['added3']),'all3 new actual trajectory checkpoints required')
 whole=[f for f in frames if type(f)is dict and str(f.get('label')).endswith('.whole-state-and-batch-tapes')];observed=[f['label'].removesuffix('.whole-state-and-batch-tapes')for f in whole];check(same(observed,selected),'selected whole-state/tape categories omitted/extra')
 for f in whole:check(f.get('planes')==expected['full_frame_planes'][f['label'].removesuffix('.whole-state-and-batch-tapes')],'full134perowned-request +complete batch/tape plane count differs')
 check(positive(value.get('bytes_compared'))and value.get('bytes_compared',0)>=spill and positive(value.get('planes_compared')),'actual full-byte/completed-plane paired comparison unavailable')
 result={'selected':selected,'frames':len(frames),'binary_files':len(files),'spill_bytes':spill,'bytes_compared':value.get('bytes_compared'),'planes_compared':value.get('planes_compared'),'legacy18_seen':seen,'whole_state_frames':[{'label':x['label'],'planes':x['planes'],'bytes':x['bytes'],'live_bytes':x['live_bytes']}for x in whole],'campaign_counts':{k:checks.get(k)for k in expected['campaign_counts']},'paired_report_sha256':{'export':value.get('export_report_sha256'),'compare':value.get('compare_report_sha256')},'common':common,'teardown_evidence':'Root pairedPASS with externally recorded report pins; complete manifest publication after native backend scope destruction per frozen Source. Envelope has no direct after-destructor allocation snapshot.','completed_campaign_reservation_snapshot_not_required_zero':True}
 return result,e

def audit_many(values,expected):
 results=[];errors=[];seen=[];reference=None
 for i,value in enumerate(values):
  result,e=audit_one(value,expected);errors.extend('inventory'+str(i)+': '+x for x in e)
  if result is None:continue
  if reference is None:reference=result['common']
  else:
   a={k:v for k,v in reference.items()if k!='selected_checkpoints'};b={k:v for k,v in result['common'].items()if k!='selected_checkpoints'}
   if not same(a,b):errors.append('Across partitions actual fixed common/root/raw numeric tuple differs')
  if result['selected']in seen:errors.append('Duplicate selected partition, not four distinct pairs')
  seen.append(result['selected']);results.append(result)
 complete=len(results)==4 and len(seen)==4 and all(x in seen for x in expected['partition_order'])and not errors
 return {'schema':'current-BQSA4-integer-Root-small-inventory-audit-v3','pass':not errors,'status':'complete'if complete else'partial','all_four_pairs_complete':complete,'pairs_checked':len(results),'required_pairs':4,'missing_selected_partitions':[x for x in expected['partition_order']if x not in seen],'selected7_fullphysical_qualified':complete,'all18_fullphysical_qualified':False,'trained_head_service_lifecycle_task_performance_qualified':False,'teardown_allocation_claim':'No after-destruction reserved-zero snapshot exists in this envelope; completed campaign snapshots can retain separately held1.704GB host reserve. Successful Root pairPASS/backend scope Source evidence is tracked separately.','errors':errors,'parts':results,'GPU_started':False,'native_report_spill_payload_read':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('--inventories',type=Path,nargs='+',required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--require-all-four',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'NEW output required');require(hashlib.sha256(a.expected.read_bytes()).hexdigest()==a.expected_sha256,'external Source expectation pin drift');values=[]
 for path in a.inventories:require(path.stat().st_size<=MAX_BYTES,'authorized smallRoot inventories only, no large reports/spills');values.append(json.loads(path.read_text()))
 result=audit_many(values,json.loads(a.expected.read_text()));result['input_metadata_paths']=[str(x)for x in a.inventories];a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['pass','status','all_four_pairs_complete','pairs_checked','missing_selected_partitions','errors','GPU_started']}))
 if not result['pass']or a.require_all_four and not result['all_four_pairs_complete']:raise SystemExit(2)
if __name__=='__main__':main()
