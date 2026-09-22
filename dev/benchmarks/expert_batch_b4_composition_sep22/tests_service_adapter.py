"""Synthetic supplemental metadata tests; no runtime/model/data/device work."""
import copy
import hashlib
from pathlib import Path
import unittest
from dev.benchmarks.expert_batch_b4_composition_sep22 import service_adapter as service
from dev.benchmarks.expert_batch_b4_composition_sep22 import test_adapter as base

INNER=Path('build/integer-b4-twopass-composed-sep22-worker-v2/source/dev/benchmarks/expert_batch_b4_composition_sep22/adapter.py').resolve()
def binding():
 b=base.binding();b.update(inner_adapter_path=str(INNER),inner_adapter_sha256=hashlib.sha256(INNER.read_bytes()).hexdigest(),frozen_original22_common_status_and_coverage_required=True);return b

def status():
 s=base.status();s['capabilities']={'mtp':True,'batch_mtp':True,'batch_mtp_prefill':True};s['mtp']={'enabled':True,'singleton_maximum_draft_tokens':3,'teacher_cache_only_requested':True,'batch_teacher_cache_only_requested':True};s['batch_prefill']['enabled']=True;return s
class Supplemental(unittest.TestCase):
 def test_source_matched_prerequisites(self):self.assertEqual(service.status_errors(status(),binding(),'mtp3'),[])
 def test_missing_prerequisite_rejected(self):
  for group,key in [('capabilities','batch_mtp'),('capabilities','batch_mtp_prefill'),('mtp','singleton_maximum_draft_tokens'),('mtp','teacher_cache_only_requested'),('mtp','batch_teacher_cache_only_requested')]:
   s=status();s[group].pop(key);self.assertTrue(service.status_errors(s,binding(),'mtp3'))
 def test_no_common_suite_bypass(self):
  b=binding();b['frozen_original22_common_status_and_coverage_required']=False;self.assertTrue(service.status_errors(status(),b,'mtp3'))
 def test_width2_r16_rejected_and_width4_r8_allowed(self):
  for width,rows,good in [(2,16,False),(4,8,True)]:
   before=status();after=copy.deepcopy(before)
   if width==4:after['batch_prefill_twopass_counters'].update(encoded_QSA_lane_calls=48,encoded_QSA_lane_layer_calls=48,completed_native_forwards=1)
   for role in ('plan','gate','down'):after['compact_native_batch_verify']['r'+str(rows)][role+'_graph_calls']=48;after['compact_native_batch_verify']['r'+str(rows)][role+'_graph_rows']=48*rows
   result,errors=service.coverage(before,after,{'prompt_token_count':2048,'body':{}},width,'mtp3',binding());self.assertEqual(not errors,good)
 def test_inner_source_pin_drift(self):
  b=binding();b['inner_adapter_sha256']='f'*64
  with self.assertRaises(ValueError):service.status_errors(status(),b,'mtp3')
if __name__=='__main__':unittest.main()
