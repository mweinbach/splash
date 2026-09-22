"""Pure synthetic metadata checks; no models/tokenizers/data/device/source edits."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import unittest

PATH=Path(os.environ.get('B4_COMPOSITION_ADAPTER_SOURCE',str(Path(__file__).with_name('adapter.py'))))
spec=importlib.util.spec_from_file_location('sealed_b4_adapter',PATH)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def binding():
 return {'policy_source_sha256':'1'*64,'policy_text':'whole actual real4 MTP3 cohort fresh2048 only; testfixture',
         'policy_shader_sha256':'2'*64,'policy_host_sha256':'3'*64,
         'integer_source_identity_sha256':'4'*64,'target_execution_child_sha256':'5'*64}
def status(enabled=True,integer=True):
 b=binding();raw='actual-raw-Forward-parent-routes';i={'source':a.MODEL_SOURCE,'loaded_model_layout_sha256':a.MODEL_LAYOUT,
  'engine_instance_id':123,'batch_prefill_twopass_requested':enabled,
  'batch_prefill_twopass_schema':a.BQSA_SCHEMA if enabled else None,
  'batch_prefill_twopass_policy':b['policy_text']if enabled else None,
  'batch_prefill_twopass_source_sha256':b['policy_source_sha256']if enabled else None,
  'batch_prefill_twopass_shader_sha256':b['policy_shader_sha256']if enabled else None,
  'batch_prefill_twopass_host_sha256':b['policy_host_sha256']if enabled else None,
  'batch_prefill_twopass_arena_plan_bytes':a.ARENA_BYTES if enabled else 0,
  'batch_prefill_twopass_numerical_parent_routes':raw,
  'batch_prefill_twopass_numerical_identity':hashlib.sha256('\n'.join((raw,a.BQSA_SCHEMA,b['policy_text'],b['policy_source_sha256'],b['policy_shader_sha256'],b['policy_host_sha256'])).encode()).hexdigest()if enabled else None,
  'batch_prefill_kernel_routes':'old bulk route'+(a.BQSA_MARKER if enabled else ''),
  'target_numerical_derivative_sha256':b['target_execution_child_sha256']if integer else a.TARGET_NUMERIC_PARENT}
 s={'identity':i,'maximum_context_tokens':16384,'batch_prefill':{'maximum_lanes':4,'maximum_real_rows_per_lane':2048},
  'scheduler':{'maximum_prefill_rows':2048,'maximum_batch_prefill_rows_per_lane':2048},
  'memory_governor':{'host_measurement_valid':True,'growth_allowed':True,'denied_reservations':0},
  'batch_prefill_twopass_counters':{'scope':a.BQSA_SCOPE,'constructed_arenas':int(enabled),'constructed_arena_bytes':a.ARENA_BYTES if enabled else 0,'encoded_QSA_lane_calls':0,'encoded_QSA_lane_layer_calls':0,'completed_native_forwards':0},
  'compact_native_batch_verify':{'schema':a.INTEGER_SCHEMA,'scope':a.INTEGER_SCOPE,'requested':integer,'enabled':integer,'source_identity_sha256':b['integer_source_identity_sha256'],'target_numeric_parent_sha256':a.TARGET_NUMERIC_PARENT,'dispatches_per_layer':6,'base_native_dispatches_per_layer':10,'additional_gpu_allocation_bytes':0,'full_model_quality_qualified':False}}
 for r,tg in ((8,2752),(16,3392)):
  s['compact_native_batch_verify']['r'+str(r)]={'physical_rows':r,'planner_threadgroup_bytes':tg,**{role+'_'+kind:0 for role in ('plan','gate','down')for kind in ('graph_calls','graph_rows')}}
 return s
class Adapter(unittest.TestCase):
 def test_valid_fresh_profile(self):
  self.assertEqual(a.status_errors(status(),binding(),'mtp3'),[])
 def test_bqsa_digest_uses_raw_forward_not_display_routes(self):
  s=status();s['identity']['kernel_routes']='different display wrapper';self.assertEqual(a.status_errors(s,binding(),'mtp3'),[])
  s['identity']['batch_prefill_twopass_numerical_parent_routes']='wrong';self.assertTrue(a.status_errors(s,binding(),'mtp3'))
 def test_parent_child_do_not_alias(self):
  s=status();s['compact_native_batch_verify']['target_numeric_parent_sha256']='5'*64;self.assertTrue(a.status_errors(s,binding(),'mtp3'))
  s=status();s['identity']['target_numerical_derivative_sha256']=a.TARGET_NUMERIC_PARENT;self.assertTrue(a.status_errors(s,binding(),'mtp3'))
 def test_b4only_cumulative_counter(self):
  s=status();s['batch_prefill_twopass_counters'].update(encoded_QSA_lane_calls=24,encoded_QSA_lane_layer_calls=24,completed_native_forwards=1);self.assertTrue(a.status_errors(s,binding(),'mtp3'))
 def test_standard_new_flags_disabled(self):
  self.assertTrue(a.status_errors(status(),binding(),'standard'))
  self.assertEqual(a.status_errors(status(False,False),binding(),'standard',False,False),[])
 def test_b4_and_b2_first_window(self):
  for width,expected in ((4,48),(2,0)):
   before=status();after=copy.deepcopy(before);after['batch_prefill_twopass_counters'].update(encoded_QSA_lane_calls=expected,encoded_QSA_lane_layer_calls=expected,completed_native_forwards=int(bool(expected)))
   record,errors=a.coverage(before,after,{'prompt_token_count':2048,'body':{}},width,'mtp3',binding());self.assertEqual(errors,[]);self.assertEqual(record['new_B4_encoded_lane_layer_calls_expected'],expected)
 def test_masked_main_prefill_still_b4_no_integer_verify(self):
  before=status();after=copy.deepcopy(before);after['batch_prefill_twopass_counters'].update(encoded_QSA_lane_calls=48,encoded_QSA_lane_layer_calls=48,completed_native_forwards=1)
  case={'prompt_token_count':2048,'body':{'response_format':{'type':'json_schema'}}};self.assertEqual(a.coverage(before,after,case,4,'mtp3',binding())[1],[])
  for role in ('plan','gate','down'):after['compact_native_batch_verify']['r16'][role+'_graph_calls']=48;after['compact_native_batch_verify']['r16'][role+'_graph_rows']=768
  self.assertTrue(a.coverage(before,after,case,4,'mtp3',binding())[1])
 def test_integer_actual_physical_rows(self):
  s=status()
  for role in ('plan','gate','down'):s['compact_native_batch_verify']['r8'][role+'_graph_calls']=48;s['compact_native_batch_verify']['r8'][role+'_graph_rows']=384
  self.assertEqual(a.integer_errors(s,binding(),True,True),[])
  s['compact_native_batch_verify']['r8']['down_graph_rows']=768;self.assertTrue(a.integer_errors(s,binding(),True))
 def test_decreasing_and_inrun_identity_drift(self):
  before=status();after=copy.deepcopy(before);after['identity']['engine_instance_id']=124;self.assertTrue(a.coverage(before,after,{'prompt_token_count':2048,'body':{}},2,'mtp3',binding())[1])
  errors=[];self.assertIsNone(a.counter_delta({'x':1},{'x':0},'x',errors));self.assertTrue(errors)
 def test_unknown_modes_and_widths(self):
  self.assertTrue(a.coverage(status(),status(),{'prompt_token_count':2048,'body':{}},3,'mtp3',binding())[1])
  self.assertTrue(a.status_errors(status(),binding(),'mtp2'))
if __name__=='__main__':unittest.main()
