import copy,unittest
from dev.benchmarks.R5_stage_target_diag_sep22 import run_root as r
def healthy():
 w={'schema':'private-R5-target-stage-diagnostic-sep22-v1','diagnostic_valid':True,'request_id':1,'generation':1,'draft_depth':4,'physical_rows':5,'R5_ordinal':3,'prior_completed_unprofiled_R5_calls':2,'maximum_samples':1,'samples_attempted':1,'profiling_restored_off':True,'stale_profiles_at_activation':0,'profiles_after_target':1,'governor_reservation_bytes':64<<20,'reservation_admitted':True,'reservation_released':True,'tensor_payload_reads':0,'tensor_payload_hashes':0,'legacy_dispatch_replay':False,'single_whole_target_command':True}
 d={'index':0,'pipeline':'test','threadgroups':{'x':1,'y':1,'z':1},'threads_per_threadgroup':{'x':128,'y':1,'z':1},'bindings':[]}
 w['Forward_graph']={'calls':1,'verification':True,'rows':5,'metadata_truncated':False,'dispatch_count':1,'metadata_count':1,'dispatches':[d]}
 w['raw_command_profile']={'mode':'stage','status':'complete','encoder_boundaries_altered':True,'sampling_barriers':False,'dropped_profiles_before':0,'dispatch_metadata_truncated':False,'dispatch_count':1,'dispatches':[{**copy.deepcopy(d),'timestamps_valid':True,'gpu_seconds':.001}]}
 w['governor_snapshots']={'after_reservation_release':{'reserved_bytes':0,'denied_reservations':0}};return w
class Tests(unittest.TestCase):
 def test_valid(self):self.assertEqual(r.audit_profile(healthy()),healthy())
 def reject(self,w):
  with self.assertRaises((ValueError,KeyError,TypeError)):r.audit_profile(w)
 def test_unsupported(self):w=healthy();w['raw_command_profile']['status']='unsupported';self.reject(w)
 def test_missed_cookie(self):w=healthy();w['request_id']=2;self.reject(w)
 def test_missed_third(self):w=healthy();w['R5_ordinal']=2;self.reject(w)
 def test_drops(self):w=healthy();w['raw_command_profile']['dropped_profiles_before']=1;self.reject(w)
 def test_truncation(self):w=healthy();w['Forward_graph']['metadata_truncated']=True;self.reject(w)
 def test_nan(self):w=healthy();w['raw_command_profile']['dispatches'][0]['gpu_seconds']=float('nan');self.reject(w)
 def test_actualbindings(self):w=healthy();w['raw_command_profile']['dispatches'][0]['threadgroups']['x']=2;self.reject(w)
 def test_govresidue(self):w=healthy();w['governor_snapshots']['after_reservation_release']['reserved_bytes']=64;self.reject(w)
 def test_bool_cookie(self):w=healthy();w['request_id']=True;self.reject(w)
 def test_scope(self):w=healthy();w['physical_rows']=4;self.reject(w)
 def test_missing(self):w=healthy();del w['raw_command_profile'];self.reject(w)
if __name__=='__main__':unittest.main()
