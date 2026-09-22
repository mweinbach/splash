#!/usr/bin/env python3
"""Audit small Root-extracted QA metadata; never open native report/spill/tensors."""
from pathlib import Path
import argparse,hashlib,json,re
MAX_METADATA_BYTES=262144

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def same(a,b):
 if type(a)is not type(b):return False
 if type(a)is dict:return set(a)==set(b)and all(same(a[k],b[k])for k in a)
 if type(a)is list:return len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
 return a==b
def errors(receipt,expected):
 e=[]
 if receipt.get('Root_extracted_metadata_only')is not True:return ['Root-extracted metadata schema marker required; do not pass native report']
 def check(v,m):
  if not v:e.append(m)
 for key in ['native_final_seal_sha256','native_library_sha256','worker_source_identity_sha256','BQSA_source_policy_sha256']:
  check(same(receipt.get(key),expected[key]),'source/artifact binding changed:'+key)
 role=receipt.get('role');check(role in ('export','compare'),'unknown native role')
 common=receipt.get('common',{});check(type(common)is dict,'native common tuple unavailable')
 for key,value in expected['common_fixed'].items():check(same(common.get(key),value),'native common field changed:'+key)
 flags=common.get('numeric_flags');check(same(flags,expected['numeric_flags']),'entire current47flag numeric tuple changed/masked/missing')
 check(type(common.get('BQSA_numeric_parent_routes_raw'))is str and bool(common.get('BQSA_numeric_parent_routes_raw')),'raw Forward parent routes unavailable')
 check(type(common.get('target_raw_numeric_parent_SHA'))is str and re.fullmatch('[0-9a-f]{64}',common.get('target_raw_numeric_parent_SHA',''))is not None,'raw persisted numerical parent unavailable')
 check(receipt.get('pass')is True and receipt.get('partition_complete')is True and receipt.get('backend_destroyed')is True,'successful native terminal/teardown required')
 checks=receipt.get('campaign_checks',{})
 for key,value in expected['campaign_counts'].items():check(same(checks.get(key),value),'actual campaign count differs:'+key)
 labels=receipt.get('observed_legacy_checkpoints');check(type(labels)is list and len(labels)==18 and len(set(labels))==18 and set(labels)==set(expected['legacy18']),'all18 actual legacy checkpoint labels required, no omission/duplication')
 full=checks.get('selected_full_checkpoints');requested=common.get('selected_checkpoints');check(same(full,requested),'selected whole physical labels differ from requested actual replay')
 check(type(full)is list and 1<=len(full)<=2 and all(x in expected['selected7']for x in full),'new selected fullphysical scope invalid')
 check(receipt.get('all18_fullphysical_qualified')is False,'selected proof must not claim all18 fullphysical')
 check(receipt.get('head_service_quality_performance_qualified')is False,'native proof cannot inherit head/service/task/speed')
 if role=='compare':
  prior=receipt.get('control_common');check(same(prior,common),'paired roles whole typed common tuple differ (no normalization)')
  check(receipt.get('matched_export_frame_count')==receipt.get('frame_count'),'paired frame count differs')
  check(type(receipt.get('bytes_compared'))is int and receipt['bytes_compared']>0 and type(receipt.get('planes_compared'))is int and receipt['planes_compared']>0,'actual byte/plane comparison evidence unavailable')
 return e

def main():
 p=argparse.ArgumentParser();p.add_argument('--metadata',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 require(not a.output.exists(),'NEW observer receipt output required')
 require(a.metadata.stat().st_size<=MAX_METADATA_BYTES,'Root-extracted small metadata only; large native reports/spills forbidden')
 require(sha(a.expected)==a.expected_sha256,'external source observer expected pin differs')
 expected=json.loads(a.expected.read_text());metadata=json.loads(a.metadata.read_text());err=errors(metadata,expected);result={'schema':'current-BQSA4-integer-native-smallmetadata-observer-v1','pass':not err,'errors':err,'native_payload_or_report_read':False,'GPU_started':False,'expected_source_sha256':a.expected_sha256,'current_full_numeric_flag_count':len(expected['numeric_flags']),'all18logical_checkpoint_labels_checked':True,'all18fullphysical_qualified':False};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 if err:raise SystemExit(2)
if __name__=='__main__':main()
