"""Independent fake metadata mutations only; no token file or GPU invocation."""
import copy
import math
import unittest
from dev.benchmarks.AR1_stage_target_diag_sep22 import run_root as r


class Tests(unittest.TestCase):
    def valid(self):
        data={'schema':'private-standard-AR1-stage-diagnostic-sep22-v1','diagnostic_valid':True,
          'request_id':1,'generation':1,'ordinary_AR1':True,'MTP_state_present':False,'physical_rows':1,'logit_rows':1,
          'begin':2050,'returned_length':2051,'AR1_ordinal':3,'prior_completed_unprofiled_AR1_calls':2,
          'maximum_samples':1,'samples_attempted':1,'single_whole_target_command':True,'legacy_dispatch_replay':False,
          'profiling_restored_off':True,'stale_profiles_at_activation':0,'profiles_after_target':1,
          'governor_reservation_bytes':64<<20,'reservation_admitted':True,'reservation_released':True,
          'instrumentation_is_performance_perturbation':True,'throughput_baseline':False,'SourceWorld_qualified':False,
          'Forward_graph_snapshot_hook':False,'numeric_inline_ABI_values':None,'tensor_payload_reads':0,'tensor_payload_hashes':0,'token_payload_reads':0,
          'baseline_source_identity_sha256':'162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a','diagnostic_source_id':'syntheticCPUonly',
          'governor_snapshots':{'before_activation':{'reserved_bytes':0},'reservation_held':{'reserved_bytes':64<<20},
            'after_reservation_release':{'reserved_bytes':0,'denied_reservations':0}},
          'raw_command_profile':{'dispatch_count':1,'mode':'stage','status':'complete','encoder_boundaries_altered':True,
            'sampling_barriers':False,'dropped_profiles_before':0,'dispatch_metadata_truncated':False,'dispatches_truncated':False,
            'dispatches_total':1,'dispatches_emitted':1,'command_kernel_timing_valid':True,
            'dispatches':[{'index':0,'pipeline':'synthetic_CPU_only','threadgroups':[1,1,1],'threads_per_threadgroup':[32,1,1],
              'bindings_truncated':False,'bindings_total':2,'bindings':[{'index':0,'size_bytes':64,'inline_bytes':False},{'index':1,'size_bytes':40,'inline_bytes':True}],
              'timestamps_valid':True,'gpu_seconds':.001,'gpu_start_timestamp':1,'gpu_end_timestamp':2}]}}
        return data
    def test_valid(self):self.assertIs(r.audit_profile(self.valid(),'syntheticCPUonly')['ordinary_AR1'],True)
    def test_typed_identity(self):
        value=self.valid();value['request_id']=True
        self.assertRaises(ValueError,r.audit_profile,value,'syntheticCPUonly')
    def test_true_begin_and_return(self):
        for field,value in [('begin',2049),('returned_length',2050),('AR1_ordinal',2),('prior_completed_unprofiled_AR1_calls',1)]:
            data=self.valid();data[field]=value
            self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_ordinary_no_MTP(self):
        data=self.valid();data['MTP_state_present']=True
        self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_unknown_ABI_stays_unknown(self):
        data=self.valid();data['numeric_inline_ABI_values']={'invented':1}
        self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_noncanonical_claim(self):
        data=self.valid();data['throughput_baseline']=True
        self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_no_mode_fallback(self):
        data=self.valid();data['raw_command_profile']['mode']='command'
        self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_drop_or_truncation(self):
        for field,value in [('dropped_profiles_before',1),('dispatch_metadata_truncated',True),('dispatches_truncated',True)]:
            data=self.valid();data['raw_command_profile'][field]=value
            self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_zero_or_mismatched_dispatch_count(self):
        for count in (0,2,4097):
            data=self.valid();data['raw_command_profile']['dispatch_count']=count
            self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_binding_truncation(self):
        data=self.valid();data['raw_command_profile']['dispatches'][0]['bindings_truncated']=True
        self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_duplicate_or_outside_binding(self):
        for index in (0,32):
            data=self.valid();data['raw_command_profile']['dispatches'][0]['bindings'][1]['index']=index
            self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_geometry_positive(self):
        data=self.valid();data['raw_command_profile']['dispatches'][0]['threadgroups']=[0,1,1]
        self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_timestamp_finite(self):
        for seconds in (math.nan,-1):
            data=self.valid();data['raw_command_profile']['dispatches'][0]['gpu_seconds']=seconds
            self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_missing_actual_timestamp(self):
        data=self.valid();data['raw_command_profile']['dispatches'][0]['timestamps_valid']=False
        self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_reservation_released(self):
        data=self.valid();data['governor_snapshots']['after_reservation_release']['reserved_bytes']=64<<20
        self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')
    def test_no_diagnostic_admission(self):
        data=self.valid();data['governor_snapshots']['reservation_held']['reserved_bytes']=0
        self.assertRaises(ValueError,r.audit_profile,data,'syntheticCPUonly')


if __name__=='__main__':unittest.main()
