"""CPU scope/tamper regression tests; no original plans, tokenizer or inference."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from dev.benchmarks.mtp_fixed4_r5_sep22 import policy_quality as p,prepare,postrun_compare as compare

def binding():
    return {'schema':p.BINDING_SCHEMA,'worker':str(p.ROOT/'build/fictional-future-R5-test-only'),
            'source_identity_sha256':'a'*64,'exe_sha256':'b'*64,'library_sha256':'c'*64,
            'metadata_sha256':{name:'d'*64 for name in p.FILES},'native_state_counts':{'frames':9,'planes':123,'bytes':456},
            'native_oracle_sha256':'e'*64,'native_report_sha256':'f'*64,
            'native_allocation_guards':{k:True for k in p.ALLOCATION_GUARDS},'native_source_scope':'MAIN ONLY'}

def native_receipt():
    b=binding()
    return {'schema':p.RECEIPT_SCHEMA,'pass':True,'qualification_complete':True,'Root_GPU_executed':True,
        'source_identity_sha256':b['source_identity_sha256'],'exe_sha256':b['exe_sha256'],'library_sha256':b['library_sha256'],
        'planner_AIR_sha256':p.PLANNER_AIR,'maximum_verify_rows':5,'backend_destroyed':True,
        'state_counts':b['native_state_counts'],'oracle_sha256':b['native_oracle_sha256'],'root_report_sha256':b['native_report_sha256'],
        'allocation_guards':b['native_allocation_guards'],'native_source_scope':'MAIN ONLY','teacher_head_proved':False,
        'memory_admission_scope':'explicit_governor_reservation_and_six_ledger_guards',
        'device_allocation_measurements_present':False,'allocation_owner_ledger':copy.deepcopy(p.OWNER_LEDGER),'original22_qualified':False}
def status(depth=4,h3=0,h4=0):
    hist=[0]*16;hist[3]=h3;hist[4]=h4
    value={'identity':{'worker_semantics':'native-worker5-singleton-fixed-depth1..15-fold8-jointcap3-v6',
                       'kernel_routes':'original'+(p.MARKER+binding()['source_identity_sha256'] if depth==4 else ''),'engine_instance_id':depth},
           'capabilities':{'mtp':True,'batch_mtp':True},
           'mtp':{'enabled':True,'maximum_draft_tokens':depth,'singleton_maximum_draft_tokens':depth,
                  'singleton_depth_override':depth,'singleton_concurrent_draft_cap':3,'joint_maximum_draft_tokens':3,
                  'depth_controller_semantics':None,'teacher_cache_only_requested':True,'policy':p.fixed_policy(depth),
                  'joint_policy':'fixed cap3 shared-budget/EOS bound; true joint head and target; survivors retain independent prefixes',
                  'completed_cycles_by_proposed_depth':hist}}
    if depth==4:value[p.SECTION]={'schema':p.PROFILE_SCHEMA,'requested':True,'enabled':True,
        'source_identity_sha256':binding()['source_identity_sha256'],'planner_AIR_sha256':p.PLANNER_AIR,'scope':p.SCOPE,
        'dispatches_per_layer':6,'GPU_allocation_bytes_added':0,'whole_state_qualified':False,'maximum_verify_rows':5,
        'graph_calls':48*h4,'graph_rows':5*48*h4}
    return value

class Fixed4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=json.loads(prepare.BASE.read_text());cls.text,cls.journal=prepare.transform(prepare.DRIVER.read_text())
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)/'tuning.py';cls.path.write_text(cls.text)
        spec=importlib.util.spec_from_file_location('_fixed4_CPU_driver',cls.path);cls.driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.driver)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_literal_driver_inverse(self):
        text=self.text
        for edit in reversed(self.journal):text=text.replace(edit['new'],edit['old'])
        self.assertEqual(p.hashlib.sha256(text.encode()).hexdigest(),prepare.DRIVER_SHA)
    def test_foreign_source_edit_refused(self):
        with self.assertRaises(ValueError):prepare.transform(prepare.DRIVER.read_text()+'\n# foreign source\n')
    def test_only_canonical_command_differences(self):
        argv,env=prepare.derive(self.base,Path(self.temp.name),binding(),Path('/tmp/binding-test-metadata.json'),'1'*64,Path('/tmp/report-test-metadata.json'))
        self.assertEqual(env,{**self.base['environment'],'SPLASH_FLASH_MTP_DRAFT_DEPTH':'4',p.FLAG:'1'})
        for option,wanted in [('contexts','2048'),('output-tokens','256'),('batches','1'),('warmup','1'),('trials','3'),('max-context','16384'),('mtp','4')]:
            self.assertEqual(argv[argv.index('--'+option)+1],wanted)
        self.assertEqual(argv[argv.index('--semantic-plan')+1],self.base['argv'][self.base['argv'].index('--semantic-plan')+1])
    def test_fixed4_environment_survives_profile_default3(self):
        args=SimpleNamespace(package=Path('/unused'),environment_overrides={p.FLAG:'1'})
        def apply(env,defaults):
            for k,v in defaults.items():env.setdefault(k,v)
        with mock.patch.object(self.driver.launcher,'_local_profile_defaults',return_value={'SPLASH_FLASH_MTP_DRAFT_DEPTH':'3'}),mock.patch.object(self.driver.launcher,'_apply_local_profile_defaults',side_effect=apply):
            env,meta=self.driver.environment_for(args,'4')
        self.assertEqual(env['SPLASH_FLASH_MTP_DRAFT_DEPTH'],'4');self.assertEqual(meta['resolved_flash_environment']['SPLASH_FLASH_MTP_DRAFT_DEPTH'],'4')
    def test_normal_runner_is_bound_through_args(self):
        tree=ast.parse(self.text)
        self.assertIn('args.fixed4_runner.coverage(before, after',self.text)
        for node in ast.walk(tree):
            if isinstance(node,ast.Name):self.assertNotEqual(node.id,'fixed4_runner')
    def test_exact_fixed3_and_fixed4_policy(self):
        self.assertEqual(p.policy_errors(status(3),3),[]);self.assertEqual(p.policy_errors(status(),4),[])
        for key,bad in [('singleton_depth_override',None),('maximum_draft_tokens',3),('singleton_concurrent_draft_cap',4),('depth_controller_semantics','adaptive'),('singleton_maximum_draft_tokens',True)]:
            value=status();value['mtp'][key]=bad;self.assertTrue(p.policy_errors(value,4))
    def test_depth5_or_adaptive_not_supported(self):
        with self.assertRaises(ValueError):p.policy_errors(status(),5)
        value=status();value['mtp']['completed_cycles_by_proposed_depth'][5]=1;self.assertTrue(p.policy_errors(value,4))
    def test_only_single_cap_normalizes_and_inputs_unchanged(self):
        value=status();saved=copy.deepcopy(value);view=p.inherited_status_view(value)
        self.assertEqual(value,saved);expected=copy.deepcopy(value);expected['mtp']['singleton_maximum_draft_tokens']=3;self.assertEqual(view,expected)
        for bad in (True,3,5,None):
            altered=status();altered['mtp']['singleton_maximum_draft_tokens']=bad
            with self.assertRaises(ValueError):p.inherited_status_view(altered)
    def test_unknown_summary_fields_retained(self):
        summary={'mtp.singleton_maximum_draft_tokens':4,'unknown_policy':{'nested':True}}
        value={'actual_execution_policy':summary};normalized=p.normalize_policy_summary(value)
        self.assertEqual(normalized,{'mtp.singleton_maximum_draft_tokens':3,'unknown_policy':{'nested':True}});self.assertEqual(summary['mtp.singleton_maximum_draft_tokens'],4)
        other={'actual_execution_policy':{**summary,'unknown_policy':{'nested':False}}};self.assertNotEqual(normalized,p.normalize_policy_summary(other))
    def test_R5_exact_profile_marker_unknown_fields_refused(self):
        self.assertEqual(p.r5_status_errors(status(),binding()),[])
        for key,bad in [('maximum_verify_rows',4),('graph_rows',1),('planner_AIR_sha256','f'*64),('GPU_allocation_bytes_added',False)]:
            value=status();value[p.SECTION][key]=bad;self.assertTrue(p.r5_status_errors(value,binding()))
        value=status();value[p.SECTION]['unknown_gate']='new';self.assertTrue(p.r5_status_errors(value,binding()))
        value=status();value['identity']['kernel_routes']+=p.MARKER+'f'*64;self.assertTrue(p.r5_status_errors(value,binding()))
    def test_R5_counts_H4_only_with_short_EOS_shapes(self):
        a,z=status(h3=2,h4=1),status(h3=7,h4=4);z['mtp']['completed_cycles_by_proposed_depth'][:3]=[1,9,13]
        details,errors=p.make_hooks(None,binding())[1](a,z,{'body':{}},True,'mtp3')
        self.assertEqual(errors,[]);self.assertEqual(details['R5_expected_calls'],144);self.assertEqual(details['R5_actual_H4_cycles'],3)
        for key in ('graph_calls','graph_rows'):
            bad=copy.deepcopy(z);bad[p.SECTION][key]+=1;self.assertTrue(p.make_hooks(None,binding())[1](a,bad,{'body':{}},True,'mtp3')[1])
    def test_excluded_schema_batch_and_prefill_callers(self):
        hook=p.make_hooks(None,binding())[1]
        for case in ({'compact_scope':'batch'},{'compact_scope':'prefill-only'},{'body':{'response_format':{'type':'json_schema'}}}):
            self.assertEqual(hook(status(),status(),case,True,'mtp3')[1],[])
            self.assertTrue(hook(status(),status(h4=1),case,True,'mtp3')[1])
    def test_counters_monotone_U64(self):
        hook=p.make_hooks(None,binding())[1]
        self.assertTrue(hook(status(h4=2),status(h4=1),{},True,'mtp3')[1])
        for bad in (True,-1,1.0,2**64):
            value=status();value['mtp']['completed_cycles_by_proposed_depth'][0]=bad;self.assertTrue(hook(status(),value,{},True,'mtp3')[1])
    def test_binding_no_absent_or_parent_admission_escape(self):
        self.assertEqual(p.validate_binding(binding()),binding())
        for key in p.BINDING_KEYS:
            value=binding();del value[key]
            with self.assertRaises((ValueError,TypeError)):p.validate_binding(value)
        for key,bad in [('exe_sha256',p.PARENT_EXE),('library_sha256',p.PARENT_LIB),('native_state_counts',{'frames':0}),('native_state_counts',{'frames':True})]:
            value=binding();value[key]=bad
            with self.assertRaises(ValueError):p.validate_binding(value)
    def test_runtime_every_occurrence_and_abbreviation(self):
        worker=Path('/tmp/exact-future-worker')
        self.assertEqual(p.runtime_args(['measure'],worker),['measure','--runtime-build',str(worker)])
        for argv in (['measure','--runtime-b',str(worker)],['measure','--runtime-build'],['measure','--runtime-build',str(worker),'--runtime-build=/tmp/other']):
            with self.assertRaises(ValueError):p.runtime_args(argv,worker)
    def test_plan_path_hash_all_occurrences_and_abbreviations(self):
        original=p.ROOT/'build/release/flash/prefill4k-semantic-plan-v1.json'
        with mock.patch.object(p,'sha',return_value=p.PLAN_FILE_SHA):
            argv=['measure','--plan',str(original)]
            self.assertEqual(p.plan_args(argv),argv)
            for bad in (['measure'],['measure','--pl',str(original)],['measure','--plan'],['measure','--plan',str(original),'--plan=/tmp/foreign']):
                with self.assertRaises(ValueError):p.plan_args(bad)
        with mock.patch.object(p,'sha',return_value='f'*64):
            with self.assertRaises(ValueError):p.plan_args(['measure','--plan',str(original)])
    def test_registered_comparison_summary_caps_do_not_mask_foreign_keys(self):
        class Runner:
            http=None
            gate_status=coverage=ownership_policy=lambda *args:None
        context=compare.Contexts(Runner(),binding());value=status();context.contexts={p.fingerprint(value):4}
        report={'initial_status':value,'actual_execution_policy':{'mtp.singleton_maximum_draft_tokens':4,'unknown_policy':{'enabled':True}}}
        normalized=context.comparison_policy(report)
        self.assertEqual(normalized['unknown_policy'],{'enabled':True})
        bad=copy.deepcopy(report);bad['actual_execution_policy']['mtp.singleton_maximum_draft_tokens']=3
        with self.assertRaises(ValueError):context.comparison_policy(bad)
    def test_missing_fresh_receipt_fails_before_artifact_access(self):
        value=binding()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fictional-binding-source-test.json';path.write_text(json.dumps(value))
            with mock.patch.object(p,'sha',side_effect=lambda candidate:p.hashlib.sha256(path.read_bytes()).hexdigest() if Path(candidate)==path else (_ for _ in ()).throw(FileNotFoundError('No future receipt'))):
                with self.assertRaises(FileNotFoundError):p.authenticate(path,p.hashlib.sha256(path.read_bytes()).hexdigest())
    def test_inherited_gate_errors_coverage_and_ownership_retained(self):
        class Runner:
            http=None
            def gate_status(self,value,*args):
                self.seen=value;return ['original-source-error']
            def coverage(self,*args,**kwargs):return {'original_detail':True},['original-coverage-error']
            def ownership_policy(self,*args):return {'original_owner':True}
        runner=p.install_fixed4(Runner(),binding());value=status();saved=copy.deepcopy(value)
        errors=runner.gate_status(value,None,None);self.assertIn('original-source-error',errors);self.assertEqual(value,saved);self.assertEqual(runner.seen['mtp']['singleton_maximum_draft_tokens'],3)
        details,errors=runner.coverage(value,value,{'body':{}},True,'mtp3');self.assertTrue(details['original_detail']);self.assertIn('original-coverage-error',errors);self.assertTrue(runner.ownership_policy(value)['original_owner'])
    def test_saved_fingerprint_rejects_new_field(self):
        class Runner:
            http=None
            gate_status=coverage=ownership_policy=lambda *args:None
        context=compare.Contexts(Runner(),binding());value=status();context.contexts={p.fingerprint(value):4}
        self.assertEqual(context.find(value),4);bad=copy.deepcopy(value);bad['unknown_field']='drift'
        with self.assertRaises(ValueError):context.find(bad)
    def test_original_comparator_single_edit_inverse(self):
        runner=p.load_parent();source=p.inspect.getsource(runner.compare)
        class Context:
            gate=runner.gate_status;coverage=runner.coverage;ownership=runner.ownership_policy
            comparison_policy=lambda self,r:p.normalize_policy_summary(r)
        journal=compare.install_comparator(runner,Context());self.assertTrue(journal['inverse_exact'])
        self.assertEqual(source.count(compare.EQUALITY),1)
    def test_clean_frozen_namespace_native_helper_and_H4_scope(self):
        code='''
import importlib.util,sys,json
from pathlib import Path
path=Path(POLICY_PATH)
spec=importlib.util.spec_from_file_location('_clean_fixed4',path);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
runner=p.load_parent()
frozen=p.ROOT/'build/compact-native-r4-verify-teacher-sep22-worker-v1b/source/dev/benchmarks'
names=('qualify_flash_http','prefill_decode_phase_quality','singleton_teacher_bulk_quality','flash_precision_quality')
before={n:str(Path(sys.modules['dev.benchmarks.'+n].__file__).resolve()) for n in names}
assert all(Path(v)==frozen/(n+'.py') for n,v in before.items())
value=FAKE_STATUS
details,errors=p.make_hooks(None,FAKE_BINDING)[1](value,value,{'body':{}},True,'mtp3')
assert errors==[],errors
assert details['fixed4_native_policy']['verified']
assert 'dev.benchmarks.flash_http_performance' not in sys.modules
assert Path(p._native.__file__).resolve()==p.NATIVE_PATH
assert before=={n:str(Path(sys.modules['dev.benchmarks.'+n].__file__).resolve()) for n in names}
print(json.dumps({'CPU_only':True,'frozen_namespace_preserved':True}))
'''
        code=code.replace('POLICY_PATH',repr(str(Path(p.__file__).resolve()))).replace('FAKE_STATUS',repr(status())).replace('FAKE_BINDING',repr(binding()))
        result=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=self.temp.name,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('frozen_namespace_preserved',result.stdout)
    def test_native_SHA_drift_refused_even_cached(self):
        p.native_metadata();self.assertIsNotNone(p._native)
        with mock.patch.object(p,'NATIVE_SHA','0'*64):
            with self.assertRaises(ValueError):p.native_metadata()
    def test_six_actual_allocation_guards_exact_names_and_bool_types(self):
        guards={key:True for key in p.ALLOCATION_GUARDS}
        self.assertEqual(p.validate_allocation_guards(guards),guards)
        for key in guards:
            for bad in (False,0,1,None,'true'):
                value=copy.deepcopy(guards);value[key]=bad
                with self.assertRaises(ValueError):p.validate_allocation_guards(value)
            value=copy.deepcopy(guards);del value[key]
            with self.assertRaises(ValueError):p.validate_allocation_guards(value)
        value={**guards,'device_peak_bytes_violations':0}
        with self.assertRaises(ValueError):p.validate_allocation_guards(value)
    def test_native_receipt_main_only_owned_ledger_without_device_claim(self):
        receipt=native_receipt();self.assertEqual(p.validate_native_receipt(receipt,binding()),receipt)
        for key,bad in [('teacher_head_proved',True),('native_source_scope','HEAD AND MAIN'),('device_allocation_measurements_present',True),('device_allocation_measurements_present',0)]:
            value=native_receipt();value[key]=bad
            with self.assertRaises(ValueError):p.validate_native_receipt(value,binding())
        for key in p.OWNER_LEDGER:
            value=native_receipt();del value['allocation_owner_ledger'][key]
            with self.assertRaises(ValueError):p.validate_native_receipt(value,binding())
        for key in ('after_model_destruction_bytes','governor_reserved_bytes','governor_denied_reservations'):
            for bad in (False,1,0.0):
                value=native_receipt();value['allocation_owner_ledger'][key]=bad
                with self.assertRaises(ValueError):p.validate_native_receipt(value,binding())
        value=native_receipt();value['memory_admission_scope']='per_physical_allocation_callback'
        with self.assertRaises(ValueError):p.validate_native_receipt(value,binding())
    def test_legacy_or_unmeasured_device_zero_axes_are_refused(self):
        for key,bad in [('allocation_axes',{'device_current_bytes_violations':0,'device_peak_bytes_violations':0}),('device_current_bytes_violations',0),('physical_device_peak_bytes',0)]:
            value=native_receipt();value[key]=bad
            with self.assertRaises(ValueError):p.validate_native_receipt(value,binding())
        value=binding();value['schema']='Root-registered-fixed4-R5-exact-runtime-binding-v1'
        with self.assertRaises(ValueError):p.validate_binding(value)
    def test_receipt_guard_scope_and_counts_must_match_exact_binding(self):
        for bad in (False,1,None):
            value=native_receipt();value['allocation_guards'][p.ALLOCATION_GUARDS[0]]=bad
            with self.assertRaises(ValueError):p.validate_native_receipt(value,binding())
        value=native_receipt();value['state_counts']['frames']=True
        with self.assertRaises(ValueError):p.validate_native_receipt(value,binding())
        value=native_receipt();del value['allocation_guards'][p.ALLOCATION_GUARDS[0]]
        with self.assertRaises(ValueError):p.validate_native_receipt(value,binding())

if __name__=='__main__':unittest.main()
