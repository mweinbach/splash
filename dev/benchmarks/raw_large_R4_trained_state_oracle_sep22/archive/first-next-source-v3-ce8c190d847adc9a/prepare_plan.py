#!/usr/bin/env python3
"""CPU source/interface validation only; no compiler or payload access."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    ast.parse((HERE/'prepare_plan.py').read_text())
    files=['contracts.hpp','interfaces.hpp','PLAN.md','prepare_plan.py']
    proof={'schema':'raw-large-R4-genuine-trained-three-boundary-SOURCE-plan-v1','source_plan_only':True,'files':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files},'control_source':'162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a','candidate_ready_required_before_clone':True,'Root_scope_review_required_before_compile':True,'native_cookie':{'request':1,'generation':1,'depth':3,'physical_rows':4,'ordinal':3,'prior_successful_R4_calls':2},'snapshot_boundaries':['third_pending_verify4','third_commit_head_truncate_fold_resolved','next_actual_target_window'],'main_physical_planes':134,'initialized_physical_lazy_arenas':216,'trained_head_physical_planes':5,'worker_owned_carry_and_offsets_included':True,'new_backend_owners':0,'input_copy_dispatches':0,'raw_input_exports':0,'whole_source_and_actual_preflight_before_first_payload_read_or_export':True,'policy_binding_pending_Root_measured_winner':True,'no_old_W8_on_profile_binding':True,'GPU_work':False,'compiler_invocations':0,'model_token_tensor_capture_profile_actual_report_reads_or_hashes':0,'ordinary26frame_or_full_task_or_performance_qualification_claimed':False}
    (HERE/'SOURCE_PLAN.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'source_plan':str(HERE/'SOURCE_PLAN.json'),'source_plan_sha256':hashlib.sha256((HERE/'SOURCE_PLAN.json').read_bytes()).hexdigest(),'compiler_invocations':0,'GPU_work':False,'payload_reads_or_hashes':0}))


if __name__=='__main__':main()
