"""Saved actual22 accounting with synthetic lease-only metadata; CPU only."""
import contextlib,copy,io,json,tempfile,unittest
from types import SimpleNamespace
from pathlib import Path
from dev.benchmarks import prefill4k_attribution_quality as quality
from dev.benchmarks import teacher_singleton_lease_quality as lease
class TeacherSingletonLeaseSavedFixtureTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.saved=json.loads((quality.ROOT/'build/release/flash/sep21-teacher-bulk-ab-qsa-model-and-quality-v1-3.semantic.json').read_text())
  cls.plan=quality.read_plan(Path(cls.saved['plan']));cls.cases={c['id']:c for c in cls.plan['cases']}
  cls.example=json.loads((quality.ROOT/'build/teacher-singleton-lease-sep22-worker-v1/quality-saved-status-fixture.json').read_text())
 def adapt(self,status):
  status=copy.deepcopy(status)
  for p in lease.IDENTITY_FIELDS:status['identity'][p.split('.')[1]]=self.example['identity'][p.split('.')[1]]
  status['teacher_singleton_lease']=copy.deepcopy(self.example['teacher_singleton_lease'])
  for k in ['requested_view_count','requested_view_bytes','registered_base_allocation_count','registered_base_allocation_bytes']:
   status['saved_operands_residency'][k]=self.example['saved_operands_residency'][k]
  return status
 def test_all22_saved_actual_cache_budget_and_teacher_proofs_unchanged(self):
  self.assertEqual(len(self.saved['cases']),22)
  self.assertEqual(self.plan['content_sha256'],'a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac')
  for row in self.saved['cases']:
   case=self.cases[row['id']];old=[row[k]for k in ['status_before','status_after']];new=[self.adapt(s)for s in old]
   for status in new:
    self.assertEqual(lease.status_errors(status),[],row['id'])
    self.assertEqual(quality.gate_status(status,self.plan,self.saved['store_witness']),[],row['id'])
   old_details,old_errors=quality.coverage(*old,case);new_details,new_errors=quality.coverage(*new,case)
   self.assertEqual(new_errors,old_errors,row['id']);self.assertEqual(new_details,old_details,row['id'])
   self.assertEqual(quality.execution_policy(old[0]),quality.execution_policy(new[0]))
   self.assertNotEqual(quality.ownership_policy(old[0]),quality.ownership_policy(new[0]))
 def test_phase_and_wrong_numeric_identity_cannot_alias_profile(self):
  row=self.saved['cases'][0];status=self.adapt(row['status_before'])
  for group,k,v in [('identity','target_hybrid_phase',True),('identity','target_numerical_derivative_sha256','a'*64),('persisted_operands','f32_tensors',296),('persisted_operands','f32_mapped_payload_bytes',12097945600),('identity','teacher_singleton_lease_source_sha256','b'*64),('saved_operands_residency','registered_base_allocation_count',1281)]:
   with self.subTest(field=k):
    bad=copy.deepcopy(status);bad[group][k]=v
    self.assertTrue(quality.gate_status(bad,self.plan,self.saved['store_witness']))
 def test_resource_profile_does_not_waive_actual_API_budget_or_expert_coverage(self):
  row=self.saved['cases'][0];before,after=[self.adapt(row[k])for k in ['status_before','status_after']];case=self.cases[row['id']]
  self.assertEqual(quality.coverage(before,after,case)[1],[])
  for field in ['teacher_cache_only_priming_calls','eligible_requests']:
   bad=copy.deepcopy(after);bad['mtp'][field]=before['mtp'][field]-1
   self.assertTrue(quality.coverage(before,bad,case)[1],field)
  bad=copy.deepcopy(after);bad['mtp']['singleton_teacher_bulk']['completed_pairs']+=1
  self.assertTrue(quality.coverage(before,bad,case)[1])
  bad=copy.deepcopy(after);bad['persisted_experts']['graph_counters']['large_row_gate_up_graph_rows']+=1
  self.assertTrue(quality.coverage(before,bad,case)[1])
 def test_actual_frozen_request_budget_tamper_rejected_by_formal_compare(self):
  candidate=copy.deepcopy(self.saved);candidate['label']='CPU synthetic807 resource status; actual saved Parent outputs; no new GPU qualification'
  for field in ['initial_status','final_status']:candidate[field]=self.adapt(candidate[field])
  for row in candidate['cases']:
   for field in ['status_before','status_after']:row[field]=self.adapt(row[field])
  candidate['cases'][0]['records'][0]['request_body']['max_completion_tokens']+=1
  with tempfile.TemporaryDirectory()as temp:
   baseline=Path(temp)/'baseline.json';changed=Path(temp)/'changed.json';output=Path(temp)/'out.json'
   baseline.write_text(json.dumps(self.saved));changed.write_text(json.dumps(candidate))
   with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'Actual task request differs from frozen body'):
    quality.compare(SimpleNamespace(reports=[baseline,changed],output=output,allow_runtime_change=True))
 def test_flag0_retains_original_parent_1281_acceptance(self):
  status=copy.deepcopy(self.saved['cases'][0]['status_before']);status['identity'].update(teacher_singleton_lease_profile=None,teacher_singleton_lease_source_sha256=None,teacher_singleton_lease_source_policy=None,teacher_singleton_lease_enabled=False);status['teacher_singleton_lease']={'requested':False}
  self.assertEqual(lease.status_errors(status),[])
  self.assertEqual(status['saved_operands_residency']['registered_base_allocation_count'],1281)
  self.assertEqual(status['saved_operands_residency']['registered_base_allocation_bytes'],145924161536)
  self.assertEqual(quality.gate_status(status,self.plan,self.saved['store_witness']),[])
if __name__=='__main__':unittest.main()
