#!/usr/bin/env python3
"""Root tiny current-native stage receipt from completed small observer evidence only."""
from pathlib import Path
import argparse,hashlib,json,re
SOURCE='edb3b1496296dad42908eb98fec4dc965ddfaaf98316eeadad1390cb5babcf94'
WORKER='ca7bc795bfb42b50949f1eaf1f035620ec3bf1b861ab397ed9a166ccc3202e68'
LIB='74af1228995f38890df035555b2000018899783fd46ad3a12f8a835c833231c8'
BQSA='b191ef4f7636d7700560c46789cdbec1322f93524ae63c93c3fef34d4ed25d07'
WORKER_SEAL='32338e764d23627d0bfb74f12393c9aa84a0e4aaa37b2ec1da520c3c381f7461'
NATIVE_SEAL='80d7618e4b002a077ddfc2e9f801e6f5944e5bf40d5bfb61f77b29a5162b55d5'
ROOT_AUTHENTICATED_AUDIT_SHA='82d09a379e05820b0666a220004169e55be9afe9001aa1f95de34b442085dbe6'
LABELS=['initial','fresh-r16.pending','fresh-r16.committed','same-r8.future.completed','b2.initial','fresh-r8.pending','fresh-r8.future.completed']

def require(v,m):
 if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make(audit,audit_path,audit_sha):
 require(audit_sha==ROOT_AUTHENTICATED_AUDIT_SHA,'only exact Root-authenticated current observer82d admission permitted')
 require(sha(audit_path)==ROOT_AUTHENTICATED_AUDIT_SHA,'observed source metadata bytes differ from exact Root-authenticated current audit')
 require(json.dumps(audit,sort_keys=True)==json.dumps(json.loads(Path(audit_path).read_text()),sort_keys=True),'passed audit object differs from exact authenticated source metadata')
 require(audit.get('schema')=='current-BQSA4-integer-Root-small-inventory-audit-v3','fresh formula-correct current observer required; old generic native stages excluded')
 require(audit.get('pass')is True and audit.get('status')=='complete'and audit.get('all_four_pairs_complete')is True and audit.get('pairs_checked')==4 and audit.get('missing_selected_partitions')==[] and audit.get('errors')==[],'ALL4distinct current paired comparisons mandatory')
 require(audit.get('all18_fullphysical_qualified')is False and audit.get('selected7_fullphysical_qualified')is True and audit.get('trained_head_service_lifecycle_task_performance_qualified')is False,'native scope cannot inherit all18fullphysical/head/service/task/performance')
 parts=audit.get('parts');require(type(parts)is list and len(parts)==4,'complete paired metadata parts required');observed=[]
 for p in parts:
  require(p.get('campaign_counts')=={'actual_verify_commands':17,'actual_commit_commands':16,'invalid_operation_checks':13},'actual full campaign counts required')
  require(len(p.get('legacy18_seen',[]))==18 and len(set(p['legacy18_seen']))==18,'all18 legacy checkpoint evidence required')
  observed.extend(p['selected']);require(p['frames']==156+len(p['selected']),'current exact frame count formula required')
  c=p['common'];require(c['BQSA_source_policy_SHA']==BQSA and len(c['numeric_flags'])==47 and c['capacity']==4096 and c['prefill_rows']==2048,'current numerical source/47flag geometry required')
 require(observed==LABELS,'exact selected7 current fullphysical order required')
 return {'schema':'current-BQSA4-integer-target-native-selected7-stage-admission-v1','pass':True,'stage_admission_only':True,'Root_native_GPU_evidence_observed':True,'factory_itself_GPU_started':False,'observer_receipt':str(audit_path),'observer_receipt_sha256':audit_sha,'worker_source_identity_sha256':SOURCE,'worker_sha256':WORKER,'metallib_sha256':LIB,'worker_compiled_seal_sha256':WORKER_SEAL,'native_QA_final_seal_sha256':NATIVE_SEAL,'BQSA_source_policy_sha256':BQSA,'target_underlying_overall_numerical_parent_sha256':'b87448342df3b3a9ae1642379b8aab513bb208ff234e552bcf70efc26a82c09d','raw_numerical_parent_SHA':parts[0]['common']['target_raw_numeric_parent_SHA'],'raw_numerical_parent_routes':parts[0]['common']['BQSA_numeric_parent_routes_raw'],'all47_literal_numeric_flags':parts[0]['common']['numeric_flags'],'proof_capacity':4096,'canonical_prefill_rows_per_lane':2048,'paired_partitions':4,'all18_logical_output_complete_max16_MoE_and_rejection_checks_qualified':True,'selected7_full134_state_and_batch_tapes_qualified':True,'selected_fullphysical_checkpoints':LABELS,'all18_fullphysical_qualified':False,'matched_frames_per_partition':[p['frames']for p in parts],'compared_bytes_including_repeated_checks':sum(p['bytes_compared']for p in parts),'compared_planes_including_replayed_campaign_overlap':sum(p['planes_compared']for p in parts),'counts_per_replayed_partition':{'verify':17,'commit':16,'rejected_API':13},'teardown_evidence':'Root pairedPASS/external report pins and complete manifest publication after native backend scope destruction; no post-dtor reserved-zero snapshot claimed. Campaign reserved hostcopy may remain1.704GB before its destructor.','trained_head_cache_future_qualified':False,'actual_worker_cancel_deadline_reuse_qualified':False,'ORIGINAL22_B2_B4_MTP_tasks_qualified':False,'service16K_or_standard_profile_qualified':False,'normal_canonical_native_decode_prefill_performance_qualified':False,'no_other_profile_or_old23f15_stage_admitted':True,'input_evidence_only_smallRootMetadata':True,'native_report_or_spill_or_model_or_response_payload_read':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('--observer-receipt',type=Path,required=True);p.add_argument('--observer-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--factory-source-sha256',required=True);a=p.parse_args();require(re.fullmatch('[0-9a-f]{64}',a.factory_source_sha256)is not None and sha(Path(__file__))==a.factory_source_sha256,'executed frozen factory Source differs from external self-source pin');require(not a.output.exists(),'NEW native stage receipt output required');require(a.observer_receipt.stat().st_size<=262144,'small observer metadata only');require(re.fullmatch('[0-9a-f]{64}',a.observer_receipt_sha256)is not None and sha(a.observer_receipt)==a.observer_receipt_sha256,'external current observer receipt digest differs');result=make(json.loads(a.observer_receipt.read_text()),a.observer_receipt,a.observer_receipt_sha256);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'output':str(a.output),'stage_admission_only':True,'GPU_started':False}))
if __name__=='__main__':main()
