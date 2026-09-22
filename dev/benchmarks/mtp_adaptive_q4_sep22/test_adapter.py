"""CPU source/control and fail-closed policy checks; no tokenizer or GPU."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
from dev.benchmarks.mtp_adaptive_q4_sep22 import policy_quality as p, prepare, run

def native(adaptive=True):
    return {'identity':{'worker_semantics':'native-worker5-joint-greedy-mtp3-and-singleton-policy-v4' if adaptive else 'native-worker5-singleton-fixed-depth1..15-fold8-jointcap3-v6'},
            'capabilities':{'mtp':True,'batch_mtp':True},
            'mtp':{'enabled':True,'maximum_draft_tokens':3,'singleton_maximum_draft_tokens':3,
                   'singleton_concurrent_draft_cap':3,'joint_maximum_draft_tokens':3,'teacher_cache_only_requested':True,
                   'singleton_depth_override':None if adaptive else 3,'depth_controller_semantics':p.CONTROLLER if adaptive else None,
                   'policy':p.ADAPTIVE_POLICY if adaptive else p.FIXED_POLICY,
                   'joint_policy':'fixed cap3 shared-budget/EOS bound; true joint head and target; survivors retain independent prefixes',
                   'completed_cycles_by_proposed_depth':[0]*16}}

class AdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=json.loads(prepare.BASE.read_text())
        cls.text,cls.journal=prepare.transform(prepare.DRIVER.read_text())
        cls.temporary=tempfile.TemporaryDirectory();cls.dir=Path(cls.temporary.name)
        cls.path=cls.dir/'tuning.py';cls.path.write_text(cls.text)
        spec=importlib.util.spec_from_file_location('_cpu_adaptive_driver',cls.path)
        cls.driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.driver)
    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()
    def test_inverse_literal_source(self):
        restored=self.text
        for change in reversed(self.journal):restored=restored.replace(change['new'],change['old'])
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(),prepare.DRIVER_SHA)
    def test_unrelated_source_change_rejected(self):
        with self.assertRaises(ValueError):prepare.transform(prepare.DRIVER.read_text().replace('temperature=0','temperature=1')+'\n# foreign edit\n')
    def test_final_profile_depth_is_absent(self):
        args=SimpleNamespace(package=Path('/unused'),environment_overrides={'SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22':'1'})
        def apply(env,defaults):
            for name,value in defaults.items():env.setdefault(name,value)
        with mock.patch.object(self.driver.launcher,'_local_profile_defaults',return_value={'SPLASH_FLASH_MTP_DRAFT_DEPTH':'3','SPLASH_FLASH_MTP':'1'}),mock.patch.object(self.driver.launcher,'_apply_local_profile_defaults',side_effect=apply):
            env,receipt=self.driver.environment_for(args,'adaptive')
        self.assertNotIn('SPLASH_FLASH_MTP_DRAFT_DEPTH',env)
        self.assertEqual(env['SPLASH_FLASH_MTP_ADAPTIVE'],'1')
        self.assertEqual(receipt['intentionally_unset_profile_flags'],['SPLASH_FLASH_MTP_DRAFT_DEPTH'])
        self.assertNotIn('SPLASH_FLASH_MTP_DRAFT_DEPTH',receipt['resolved_flash_environment'])
    def test_explicit_depth_rejected_in_adaptive(self):
        args=SimpleNamespace(package=Path('/unused'),environment_overrides={'SPLASH_FLASH_MTP_DRAFT_DEPTH':'3'})
        with mock.patch.object(self.driver.launcher,'_local_profile_defaults',return_value={'SPLASH_FLASH_MTP_DRAFT_DEPTH':'3'}):
            with self.assertRaises(ValueError):self.driver.environment_for(args,'adaptive')
    def test_fixed_retains_original_depth(self):
        args=SimpleNamespace(package=Path('/unused'),environment_overrides={})
        with mock.patch.object(self.driver.launcher,'_local_profile_defaults',return_value={'SPLASH_FLASH_MTP_DRAFT_DEPTH':'3'}),mock.patch.object(self.driver.launcher,'_apply_local_profile_defaults'):
            env,receipt=self.driver.environment_for(args,'3')
        self.assertEqual(env['SPLASH_FLASH_MTP_DRAFT_DEPTH'],'3')
        self.assertEqual(receipt['intentionally_unset_profile_flags'],[])
    def test_sole_canonical_control_delta(self):
        fixed,env0=prepare.derive(self.base,self.dir,'fixed3',Path('/tmp/fixed.json'))
        adaptive,env1=prepare.derive(self.base,self.dir,'adaptive',Path('/tmp/adaptive.json'))
        self.assertNotIn('SPLASH_FLASH_MTP_DRAFT_DEPTH=3',adaptive)
        self.assertEqual(env1,{**{k:v for k,v in env0.items() if k!='SPLASH_FLASH_MTP_DRAFT_DEPTH'},'SPLASH_FLASH_MTP_ADAPTIVE':'1'})
        for option,wanted in [('--contexts','2048'),('--output-tokens','256'),('--batches','1'),('--warmup','1'),('--trials','3'),('--max-context','16384')]:
            self.assertEqual(adaptive[adaptive.index(option)+1],wanted)
        self.assertEqual(fixed[fixed.index('--semantic-plan')+1],adaptive[adaptive.index('--semantic-plan')+1])
    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):p.policy_errors(native(),'fast')
        with self.assertRaises(ValueError):prepare.derive(self.base,self.dir,'4',Path('/tmp/x.json'))
    def test_both_exact_policy_contracts(self):
        self.assertEqual(p.policy_errors(native(),'adaptive'),[])
        self.assertEqual(p.policy_errors(native(False),'fixed3'),[])
        self.assertTrue(p.policy_errors(native(False),'adaptive'))
    def test_adaptive_override_tamper_rejected(self):
        value=native();value['mtp']['singleton_depth_override']=3
        self.assertTrue(p.policy_errors(value,'adaptive'))
    def test_unknown_controller_and_false_caps_rejected(self):
        for key,bad in [('depth_controller_semantics',p.CONTROLLER+'-changed'),('singleton_maximum_draft_tokens',True),('policy',p.FIXED_POLICY)]:
            value=native();value['mtp'][key]=bad
            self.assertTrue(p.policy_errors(value,'adaptive'))
    def test_depth_overflow_and_non_u64_rejected(self):
        for bad in [True,-1,1.0,2**64]:
            value=native();value['mtp']['completed_cycles_by_proposed_depth'][0]=bad
            self.assertTrue(p.policy_errors(value,'adaptive'))
        value=native();value['mtp']['completed_cycles_by_proposed_depth'][4]=1
        self.assertTrue(p.policy_errors(value,'adaptive'))
    def test_mixed_depth_counts_only_actual_R4(self):
        a=native();b=copy.deepcopy(a);b['mtp']['completed_cycles_by_proposed_depth'][:4]=[3,9,17,21]
        details,errors=p.make_hooks(None,'adaptive')[1](a,b,{},True,'mtp3')
        self.assertEqual(errors,[]);self.assertEqual(details['actual_R4_cycle_delta'],21)
        self.assertEqual(details['actual_singleton_depth_cycle_deltas'][:4],[3,9,17,21])
    def test_decreasing_or_mixed_policy_rejected(self):
        a=native();b=native();a['mtp']['completed_cycles_by_proposed_depth'][1]=2
        self.assertTrue(p.make_hooks(None,'adaptive')[1](a,b,{},True,'mtp3')[1])
        self.assertTrue(p.make_hooks(None,'adaptive')[1](native(),native(False),{},True,'mtp3')[1])
    def test_forbidden_original_semantic_mode(self):
        self.assertTrue(p.make_hooks(None,'adaptive')[0](native(),None,None,'adaptive'))
    def test_command_controls_and_pins_tamper(self):
        argv,env=prepare.derive(self.base,self.dir,'adaptive',Path('/tmp/sep22-adaptive-test-unused-report.json'))
        command={'schema':'qualified-Q4-existing-adaptive-depth-command-v1','policy':'adaptive','worker':str(prepare.WORKER),'cwd':str(prepare.ROOT),
                 'base_command':str(prepare.BASE),'base_command_sha256':prepare.sha(prepare.BASE),'pins':{str(self.path):prepare.sha(self.path)},'argv':argv,'environment':env,'report':'/tmp/sep22-adaptive-test-unused-report.json'}
        self.assertEqual(run.validate(command,prepare),argv)
        bad=copy.deepcopy(command);bad['argv'][bad['argv'].index('--output-tokens')+1]='128'
        with self.assertRaises(ValueError):run.validate(bad,prepare)
        bad=copy.deepcopy(command);bad['environment']['SPLASH_FLASH_QMV_F32']='0'
        with self.assertRaises(ValueError):run.validate(bad,prepare)
        bad=copy.deepcopy(command);bad['pins'][str(self.path)]='0'*64
        with self.assertRaises(ValueError):run.validate(bad,prepare)

    def test_clean_frozen_namespace_old_failure_and_private_fix(self):
        # -I removes cwd/PYTHONPATH package resolution and starts without the
        # normal driver's cached live native-policy module. This reproduces
        # the actual standalone semantic child, not an in-process test import.
        template = '''
import importlib.util, json, sys
from pathlib import Path
path = Path(POLICY_FILE)
spec = importlib.util.spec_from_file_location('_clean_private_policy', path)
policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
runner = policy.load()
frozen = Path(FROZEN_DIRECTORY)
for name in ('qualify_flash_http','prefill_decode_phase_quality','singleton_teacher_bulk_quality','flash_precision_quality'):
    loaded = sys.modules['dev.benchmarks.'+name]
    assert Path(loaded.__file__).resolve() == frozen/(name+'.py'), loaded.__file__
assert 'dev.benchmarks.flash_http_performance' not in sys.modules
value = NATIVE_STATUS
try:
    details, errors = policy.make_hooks(None,'adaptive')[1](value,value,{},True,'mtp3')
except ModuleNotFoundError as error:
    assert EXPECT_FAILURE and error.name == 'dev.benchmarks.flash_http_performance', repr(error)
    print(json.dumps({'old_clean_semantic_import_failure_reproduced':True,'GPU_work':False}))
else:
    assert not EXPECT_FAILURE, 'Old failure not reproduced'
    assert errors == [], errors
    assert details['registered_native_mtp_depth_metadata']['verified'] is True
    assert Path(policy._native_policy_module.__file__).resolve() == policy.NATIVE_POLICY_PATH
    for name in ('qualify_flash_http','prefill_decode_phase_quality','singleton_teacher_bulk_quality','flash_precision_quality'):
        assert Path(sys.modules['dev.benchmarks.'+name].__file__).resolve() == frozen/(name+'.py')
    print(json.dumps({'private_policy_capture_pass':True,'frozen_dependency_paths_preserved':True,'GPU_work':False}))
'''
        frozen=prepare.ROOT/'build/compact-native-r4-verify-teacher-sep22-worker-v1b/source/dev/benchmarks'
        old=prepare.ROOT/'build/mtp-adaptive-Q4-sep22-root-v3/source-at-run/policy_quality.py'
        for path,failure in ((old,True),(Path(p.__file__).resolve(),False)):
            code=template.replace('POLICY_FILE',repr(str(path))).replace('FROZEN_DIRECTORY',repr(str(frozen))).replace('NATIVE_STATUS',repr(native())).replace('EXPECT_FAILURE',repr(failure))
            result=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=self.dir,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('"GPU_work": false',result.stdout)

    def test_native_policy_source_drift_rejected(self):
        with mock.patch.object(p,'NATIVE_POLICY_SHA','0'*64):
            with self.assertRaises(ValueError):p.make_hooks(None,'adaptive')

if __name__=='__main__':unittest.main()
