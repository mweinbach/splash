"""Exact resource-only pure-I8 teacher lease profile; frozen task grading unchanged."""
from dev.benchmarks import qualify_flash_http as http
PROFILE='pure-I8-teacher-parent508-startup-persistent807-r4F32118-omitBF1684-v1'
SOURCE_SHA='f8897f688bdad5ec8467b9f26c0fc0ba35e91b2c1e138248372c8c3d16279da0'
SOURCE_POLICY=('startup lease selection only; preserve actual508F32/509BF16 backing and math; fixed R4 F32118 persistent; '
 'BF16 W8 source84 direct-transient; W8derived168 already included; all backing/governor reservations retained; '
 'B1 qualification only; batch math/capabilities unchanged')
NUMERICAL_DERIVATIVE='b87448342df3b3a9ae1642379b8aab513bb208ff234e552bcf70efc26a82c09d'
SCOPE='B1; batch math/capabilities unchanged; transient direct-binding owners remain charged; physical pinning not promised'
IDENTITY_FIELDS=('identity.teacher_singleton_lease_profile','identity.teacher_singleton_lease_source_sha256',
 'identity.teacher_singleton_lease_source_policy','identity.teacher_singleton_lease_enabled')
OWNERSHIP_FIELDS=(*IDENTITY_FIELDS,'teacher_singleton_lease.requested',
 'teacher_singleton_lease.retained_F32_owner_count','teacher_singleton_lease.retained_F32_owner_bytes',
 'teacher_singleton_lease.omitted_F32_owner_count','teacher_singleton_lease.omitted_F32_owner_bytes',
 'teacher_singleton_lease.retained_BF16_owner_count','teacher_singleton_lease.retained_BF16_owner_bytes',
 'teacher_singleton_lease.omitted_BF16_owner_count','teacher_singleton_lease.omitted_BF16_owner_bytes',
 'teacher_singleton_lease.derived_W8_owner_count','teacher_singleton_lease.derived_W8_owner_bytes',
 'teacher_singleton_lease.all_backing_retained','teacher_singleton_lease.all_backing_charged',
 'teacher_singleton_lease.selection_only','teacher_singleton_lease.qualification_scope')
def present(status):
 return any(http.get_path(status,p) is not None for p in IDENTITY_FIELDS) or http.get_path(status,'teacher_singleton_lease') is not None
def status_errors(status):
 if not present(status):return []
 active=http.get_path(status,'identity.teacher_singleton_lease_enabled')
 if active is False:
  requirements={p:None for p in IDENTITY_FIELDS[:3]}
  requirements.update({'identity.teacher_singleton_lease_enabled':False,'teacher_singleton_lease.requested':False})
 elif active is True:
  requirements={
   'identity.teacher_singleton_lease_profile':PROFILE,'identity.teacher_singleton_lease_source_sha256':SOURCE_SHA,
   'identity.teacher_singleton_lease_source_policy':SOURCE_POLICY,'identity.teacher_singleton_lease_enabled':True,
   'identity.target_numerical_derivative_sha256':NUMERICAL_DERIVATIVE,'identity.target_all_rows_full512':True,
   'identity.original_target_gpu_omitted':True,'identity.target_hybrid_phase':None,
   'saved_operands_residency.requested':True,'saved_operands_residency.active':True,
   'saved_operands_residency.request_succeeded':True,'saved_operands_residency.requested_view_count':807,
   'saved_operands_residency.requested_view_bytes':131005546496,
   'saved_operands_residency.registered_base_allocation_count':807,
   'saved_operands_residency.registered_base_allocation_bytes':131005546496,
   'saved_operands_residency.backing_already_charged':True,'saved_operands_residency.physical_pinning_verified':False,
   'original_text_residency.requested':False,'original_text_residency.added_to_union':False,
   'persisted_operands.bf16_tensors':509,'persisted_operands.bf16_mapped_payload_bytes':8467251200,
   'persisted_operands.f32_tensors':508,'persisted_operands.f32_mapped_payload_bytes':14391705600,
   'persisted_operands.store_manifest_sha256':'433e8a0ea5150fc063b7ccd02fc91191ba08032640ec2fd5d63cff5ece129512',
   'identity.dense_w8a8_prefill_enabled':True,'identity.dense_w8a8_projection_count':84,
   'identity.dense_w8a8_immutable_buffer_count':168,'identity.dense_w8a8_fixed_cache_planned_bytes':1890975744,
   'teacher_singleton_lease.requested':True,'teacher_singleton_lease.retained_F32_owner_count':118,
   'teacher_singleton_lease.retained_F32_owner_bytes':3247964160,'teacher_singleton_lease.omitted_F32_owner_count':390,
   'teacher_singleton_lease.omitted_F32_owner_bytes':11143741440,'teacher_singleton_lease.retained_BF16_owner_count':425,
   'teacher_singleton_lease.retained_BF16_owner_bytes':4692377600,'teacher_singleton_lease.omitted_BF16_owner_count':84,
   'teacher_singleton_lease.omitted_BF16_owner_bytes':3774873600,'teacher_singleton_lease.derived_W8_owner_count':168,
   'teacher_singleton_lease.derived_W8_owner_bytes':1890975744,'teacher_singleton_lease.all_backing_retained':True,
   'teacher_singleton_lease.all_backing_charged':True,'teacher_singleton_lease.selection_only':True,
   'teacher_singleton_lease.qualification_scope':SCOPE,
  }
 else:return ['Teacher singleton lease enabled flag is not a literal boolean']
 return ['Teacher singleton lease resource profile differs: '+p for p,v in requirements.items()
         if not http.same_json(http.get_path(status,p),v)]
