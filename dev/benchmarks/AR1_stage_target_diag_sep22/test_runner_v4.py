"""CPU-only exact-status and lifecycle tests. No report/payload/GPU reads."""
import io
import unittest
from types import SimpleNamespace as S
from server import protocol as wire
from dev.benchmarks.AR1_stage_target_diag_sep22 import run_root_v4 as r


class FakeProcess:
    def __init__(self):self.stdin=io.BytesIO();self.stdout=io.BytesIO();self.code=None;self.signal_count=0;self.on_wait=None
    def poll(self):return self.code
    def wait(self,timeout):
        if self.on_wait:self.on_wait()
        self.code=0;return 0
    def terminate(self):self.signal_count+=1;self.code=-15
    def kill(self):self.signal_count+=1;self.code=-9


class FakeRuntime:
    def __init__(self,process):self.process=process;self.closed=False
    def close(self):
        self.closed=True
        if self.process.poll() is None:self.process.terminate()


class Tests(unittest.TestCase):
    def result(self):return S(start=S(request_id=1,cache_disposition=wire.CacheDisposition.MISS,matched_prompt_tokens=0),done=S(prompt_tokens=2048,completion_tokens=64),tokens=tuple(range(64)))
    def states(self):
        def value(end):return {'metrics':{'prefill_input_tokens':2048 if end else 0,'autoregressive_output_tokens':64 if end else 0},
          'scheduler':{'decode_batches_by_width':{'b1':63 if end else 0,'b2':0,'b3':0,'b4':0}},
          'mtp':{'enabled':False,'eligible_requests':0,'autoregressive_requests':1 if end else 0,'verification_cycles':0,'drafted_tokens':0,
            'singleton_teacher_bulk':{'completed_teacher_commands':0,'requested':False}}}
        return value(False),value(True)
    def test_current_exact_scheduler_path(self):self.assertEqual(r.cache_counts_and_standard(self.result(),*self.states())['actual_native_AR1_target_calls'],63)
    def test_no_top_level_counter_fallback(self):
        before,after=self.states()
        for value in (before,after):value['decode_batches_by_width']=value.pop('scheduler')['decode_batches_by_width']
        self.assertRaises(KeyError,r.cache_counts_and_standard,self.result(),before,after)
    def test_wrong_count_rejected(self):
        before,after=self.states();after['scheduler']['decode_batches_by_width']['b1']=62
        self.assertRaises(ValueError,r.cache_counts_and_standard,self.result(),before,after)
    def test_EOF_before_runtime_close(self):
        actual=FakeProcess();process=r.TrackedProcess(actual);runtime=FakeRuntime(process)
        outcome=r.close_with_graceful_EOF(runtime,[process])
        self.assertTrue(outcome['stdin_EOF_sent']);self.assertTrue(outcome['exit_observed_before_runtime_close']);self.assertTrue(runtime.closed)
        self.assertFalse(process.terminate_called);self.assertFalse(process.kill_called);self.assertEqual(actual.signal_count,0)
    def test_validation_exception_still_graceful(self):
        actual=FakeProcess();process=r.TrackedProcess(actual);runtime=FakeRuntime(process)
        try:raise ValueError('metadata validation failed after Done')
        except ValueError:pass
        finally:outcome=r.close_with_graceful_EOF(runtime,[process])
        self.assertEqual(outcome['errors'],[]);self.assertFalse(process.terminate_called);self.assertEqual(actual.code,0)
    def test_concurrent_generic_termination_deferred_honestly(self):
        actual=FakeProcess();process=r.TrackedProcess(actual);runtime=FakeRuntime(process);actual.on_wait=process.terminate
        outcome=r.close_with_graceful_EOF(runtime,[process])
        self.assertEqual(process.generic_terminate_requests_deferred,1);self.assertFalse(process.terminate_called)
        self.assertEqual(actual.signal_count,0);self.assertTrue(outcome['runtime_close_after_process_exit'])
    def test_timeout_is_invalid_and_fallback_signal_honest(self):
        actual=FakeProcess();process=r.TrackedProcess(actual);runtime=FakeRuntime(process)
        def timeout():raise TimeoutError('synthetic blocked exit')
        actual.on_wait=timeout
        outcome=r.close_with_graceful_EOF(runtime,[process])
        self.assertTrue(outcome['errors']);self.assertFalse(outcome['exit_observed_before_runtime_close'])
        self.assertTrue(process.terminate_called);self.assertEqual(actual.signal_count,1)
    def test_no_process_cannot_claim_stop(self):
        outcome=r.close_with_graceful_EOF(None,[])
        self.assertFalse(outcome['stdin_EOF_sent']);self.assertFalse(outcome['exit_observed_before_runtime_close'])


if __name__=='__main__':unittest.main()
