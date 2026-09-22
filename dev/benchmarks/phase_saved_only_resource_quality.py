"""Resource-only phase startup profiles; no change to sampling/task coverage."""
from dev.benchmarks import qualify_flash_http as http
PROFILE = 'phase-saved-only-startup807-existing-q4-backing-direct-transient-v1'
SOURCE_POLICY = ('explicit saved-only flag1 requires Q4expert lease0; flag0 preserves original '
                 'composite guard; mathematical derivative unchanged; all backing and reservations retained')
def status_errors(status):
    profile=http.get_path(status,'identity.phase_resource_profile')
    if profile is None or profile=='phase-q4-expert-composite-startup832-v1':return []
    if profile!=PROFILE:return ['Unknown explicit phase resource profile']
    requirements={
        'identity.phase_resource_variant_source_policy':SOURCE_POLICY,
        'saved_operands_residency.requested':True,
        'saved_operands_residency.active':True,
        'saved_operands_residency.request_succeeded':True,
        'saved_operands_residency.registered_base_allocation_count':807,
        'saved_operands_residency.registered_base_allocation_bytes':131005546496,
        'saved_operands_residency.backing_already_charged':True,
        'saved_operands_residency.physical_pinning_verified':False,
        'hybrid_q4_expert_residency.requested':False,
        'hybrid_q4_expert_residency.added_to_composite':False,
        'hybrid_q4_expert_residency.active':False,
        'hybrid_q4_expert_residency.selected_owner_count':0,
        'hybrid_q4_expert_residency.selected_owner_bytes':0,
        'hybrid_q4_expert_residency.registered_composite_owner_count':807,
        'hybrid_q4_expert_residency.registered_composite_owner_bytes':131005546496,
        'identity.phase_prefill_f32_selector_membership_count':508,
        'persisted_operands.f32_tensors':296,
        'persisted_operands.f32_mapped_payload_bytes':12097945600,
        'phase_f32_persistent_residency.retained_owner_count':118,
        'phase_f32_persistent_residency.retained_owner_bytes':3247964160,
        'phase_f32_persistent_residency.transient_only_owner_count':178,
        'phase_f32_persistent_residency.transient_only_owner_bytes':8849981440,
        'phase_f32_persistent_residency.all_backing_retained':True,
        'phase_f32_persistent_residency.all_backing_charged':True,
        'ple_storage.gpu_mapped_original_bytes':74317889536,
    }
    return ['Saved-only numerical phase resource inventory differs: '+path
            for path,value in requirements.items()if not http.same_json(http.get_path(status,path),value)]
