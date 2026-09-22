import copy
import unittest
from dev.benchmarks.fixed4_head5_trace_sep22 import probe

def event(sequence,phase='target_verify',role='target_trunk',rows=5):
    return {'schema':'splash-request-command-trace-v1','instrumentation_on':True,'instance_id':7,'lanes':1,
            'submission_expected':True,'profile_present':True,'profiling_mode':'command','hardware_timestamps_valid':True,
            'gpu_hardware_start_mach_seconds':float(sequence),'gpu_hardware_end_mach_seconds':sequence+.01,
            'requests':[{'request_id':13,'generation':2,'input_rows':rows}],'actual_rows':rows,
            'command_sequence':sequence,'phase':phase,'role':role}
class ProbeTests(unittest.TestCase):
    def test_actual_fold5_then_future_Verify5(self):
        data=[event(1,'committed_head_fold','mtp_head'),event(2,'draft_head_chain','mtp_head',1),event(3)]
        saved=copy.deepcopy(data);value=probe.audit_trace(data,7)
        self.assertTrue(value['direct_event_proved']);self.assertEqual(value['committed_head_fold_rows5_events'],1);self.assertEqual(data,saved)
    def test_no_five_fold_is_inconclusive(self):
        value=probe.audit_trace([event(1,'committed_head_fold','mtp_head',4),event(2)],7)
        self.assertFalse(value['direct_event_proved'])
    def test_fold5_without_future_verify_is_inconclusive(self):
        self.assertFalse(probe.audit_trace([event(1,'committed_head_fold','mtp_head')],7)['direct_event_proved'])
    def test_identity_generation_or_instance_drift_fails(self):
        for field,bad in [('request_id',14),('generation',3)]:
            data=[event(1,'committed_head_fold','mtp_head'),event(2)];data[1]['requests'][0][field]=bad
            with self.assertRaises(ValueError):probe.audit_trace(data,7)
        with self.assertRaises(ValueError):probe.audit_trace([event(1)],8)
    def test_sequence_row_profile_and_timestamp_corruption_fails(self):
        for field,bad in [('command_sequence',True),('actual_rows',6),('profile_present',False),('hardware_timestamps_valid',False),('gpu_hardware_end_mach_seconds',.1),('profiling_mode','stage_per_dispatch')]:
            data=[event(1)];data[0][field]=bad
            with self.assertRaises(ValueError):probe.audit_trace(data,7)
        with self.assertRaises(ValueError):probe.audit_trace([event(2),event(1)],7)
    def test_native_resolved_noGPU_restore_is_allowed_but_not_fold_proof(self):
        record=event(2,'target_prefix_restore','target_restore');record.update(submission_expected=False,profile_present=False,event='resolved_without_gpu_submit');record.pop('command_sequence')
        self.assertTrue(probe.audit_trace([event(1,'committed_head_fold','mtp_head'),record,event(3)],7)['direct_event_proved'])
        record['phase']='committed_head_fold';record['role']='mtp_head'
        with self.assertRaises(ValueError):probe.audit_trace([record],7)

if __name__=='__main__':unittest.main()
