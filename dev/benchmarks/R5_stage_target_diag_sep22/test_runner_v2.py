import unittest
from unittest import mock
from types import SimpleNamespace as S
from server import protocol as wire
from dev.benchmarks.R5_stage_target_diag_sep22 import run_root_v2 as v2
class Tests(unittest.TestCase):
 def result(self):return S(start=S(request_id=1,cache_disposition=wire.CacheDisposition.MISS,matched_prompt_tokens=0),done=S(prompt_tokens=2048,completion_tokens=64),tokens=tuple(range(64)))
 def states(self):return {'metrics':{'prefill_input_tokens':0,'autoregressive_output_tokens':0}},{'metrics':{'prefill_input_tokens':2048,'autoregressive_output_tokens':64}}
 def test_valid_cache_count(self):self.assertEqual(v2.check_cache_and_counts(self.result(),*self.states())['matched_prompt_tokens'],0)
 def test_cachehit(self):r=self.result();r.start.cache_disposition=wire.CacheDisposition.PREFIX_HIT;self.assertRaises(ValueError,v2.check_cache_and_counts,r,*self.states())
 def test_matched(self):r=self.result();r.start.matched_prompt_tokens=1;self.assertRaises(ValueError,v2.check_cache_and_counts,r,*self.states())
 def test_wrong_prompt(self):a,z=self.states();z['metrics']['prefill_input_tokens']=2047;self.assertRaises(ValueError,v2.check_cache_and_counts,self.result(),a,z)
 def test_wrong_output(self):a,z=self.states();z['metrics']['autoregressive_output_tokens']=63;self.assertRaises(ValueError,v2.check_cache_and_counts,self.result(),a,z)
 def test_shortdone(self):r=self.result();r.done.completion_tokens=63;self.assertRaises(ValueError,v2.check_cache_and_counts,r,*self.states())
 def test_group_present(self):
  with mock.patch.object(v2.os,'killpg',return_value=None):self.assertFalse(v2.group_gone(123))
 def test_group_gone(self):
  with mock.patch.object(v2.os,'killpg',side_effect=ProcessLookupError):self.assertTrue(v2.group_gone(123))
 def test_group_denied(self):
  with mock.patch.object(v2.os,'killpg',side_effect=PermissionError):self.assertFalse(v2.group_gone(123))
if __name__=='__main__':unittest.main()
