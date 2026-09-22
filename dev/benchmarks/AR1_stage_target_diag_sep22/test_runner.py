import copy
import unittest
from types import SimpleNamespace as S
from unittest import mock
from server import protocol as wire
from dev.benchmarks.AR1_stage_target_diag_sep22 import run_root as r


class Tests(unittest.TestCase):
    def result(self):
        return S(start=S(request_id=1,cache_disposition=wire.CacheDisposition.MISS,matched_prompt_tokens=0),
                 done=S(prompt_tokens=2048,completion_tokens=64),tokens=tuple(range(64)))
    def states(self):
        def value(end):
            return {'metrics':{'prefill_input_tokens':2048 if end else 0,'autoregressive_output_tokens':64 if end else 0},
                'decode_batches_by_width':{'b1':63 if end else 0,'b2':0,'b3':0,'b4':0},
                'mtp':{'eligible_requests':0,'autoregressive_requests':1 if end else 0,'verification_cycles':0,'drafted_tokens':0,
                       'enabled':False,'singleton_teacher_bulk':{'completed_teacher_commands':0,'requested':False}}}
        return value(False),value(True)
    def test_valid_full_standard(self):
        self.assertEqual(r.cache_counts_and_standard(self.result(),*self.states())['actual_native_AR1_target_calls'],63)
    def test_cachehit_rejected(self):
        value=self.result();value.start.cache_disposition=wire.CacheDisposition.PREFIX_HIT
        self.assertRaises(ValueError,r.cache_counts_and_standard,value,*self.states())
    def test_matched_token_rejected(self):
        value=self.result();value.start.matched_prompt_tokens=1
        self.assertRaises(ValueError,r.cache_counts_and_standard,value,*self.states())
    def test_short_done_rejected(self):
        value=self.result();value.done.completion_tokens=63
        self.assertRaises(ValueError,r.cache_counts_and_standard,value,*self.states())
    def test_real_AR_count_required(self):
        before,after=self.states();after['decode_batches_by_width']['b1']=62
        self.assertRaises(ValueError,r.cache_counts_and_standard,self.result(),before,after)
    def test_batch_substitution_rejected(self):
        before,after=self.states();after['decode_batches_by_width']['b2']=1
        self.assertRaises(ValueError,r.cache_counts_and_standard,self.result(),before,after)
    def test_MTP_state_rejected(self):
        before,after=self.states();after['mtp']['enabled']=True
        self.assertRaises(ValueError,r.cache_counts_and_standard,self.result(),before,after)
    def test_teacher_substitution_rejected(self):
        before,after=self.states();after['mtp']['singleton_teacher_bulk']['requested']=True
        self.assertRaises(ValueError,r.cache_counts_and_standard,self.result(),before,after)
    def test_group_gone(self):
        with mock.patch.object(r.os,'killpg',side_effect=ProcessLookupError):self.assertTrue(r.group_gone(123))
    def test_group_alive(self):
        with mock.patch.object(r.os,'killpg',return_value=None):self.assertFalse(r.group_gone(123))


if __name__=='__main__':unittest.main()
