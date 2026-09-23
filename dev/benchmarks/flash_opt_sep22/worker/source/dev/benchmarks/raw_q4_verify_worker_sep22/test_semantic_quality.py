"""CPU tests of the new admission/counter increment and unchanged old gates."""
import copy
import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest import mock
from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks.raw_q4_verify_worker_sep22 import semantic_quality as q

class RawTests(unittest.TestCase):
    def snapshot(self, active, cycles=0):
        hist = [0] * 16; hist[3] = cycles
        calls = 26 * cycles if active else 0
        return {'identity': {'kernel_routes': 'original' + (q.FAMILY + q.SOURCE if active else '')},
            q.SECTION: {'schema': q.SCHEMA, 'requested': active, 'source_identity_sha256': q.SOURCE,
                'scope': q.SCOPE, 'qualified_candidate_AIR_sha256': q.AIR,
                'expected_main_roles_per_VerifyR4': 26, 'GPU_allocation_bytes_added': 0,
                'counter_scope': q.COUNTER_SCOPE, 'whole_state_qualified': False,
                'graph_calls': calls, 'graph_rows': 4*calls},
            'compact_r4_preflight': {'guard_preflight_graph_calls': 48*cycles},
            'mtp': {'completed_cycles_by_proposed_depth': hist}}
    def test_exact_active_disabled_and_all_field_types(self):
        for active in (False, True):
            gate, _, _ = q.make_hooks(http, active)
            self.assertEqual(gate(self.snapshot(active, 5), None, None), [])
            for key in self.snapshot(active)[q.SECTION]:
                for bad in (None, True, False, -1, 1.0, 'wrong', 2**64):
                    s=self.snapshot(active,5)
                    if q.same(s[q.SECTION][key],bad): continue
                    s[q.SECTION][key]=bad
                    self.assertTrue(gate(s,None,None), (active,key,bad))
            for routes in ('',q.FAMILY+'f'*64,q.FAMILY+q.SOURCE+q.FAMILY+q.SOURCE):
                if not active and not routes: continue
                s=self.snapshot(active);s['identity']['kernel_routes']=routes
                self.assertTrue(gate(s,None,None))
            self.assertTrue(gate({},None,None))
    def test_actual_H3_26_calls_and_exclusions(self):
        for active in (False,True):
            _,cov,_=q.make_hooks(http,active)
            a,b=self.snapshot(active,2),self.snapshot(active,7)
            b['mtp']['completed_cycles_by_proposed_depth'][1]=10
            self.assertEqual(cov(a,b,{'body':{}})[1],[])
            for key in ('graph_calls','graph_rows'):
                for delta in (-1,1):
                    c=copy.deepcopy(b);c[q.SECTION][key]+=delta
                    self.assertTrue(cov(a,c,{'body':{}})[1])
            for case,mode in (({'compact_scope':'batch'},'mtp3'),({'compact_scope':'prefill-only'},'mtp3'),
                    ({'body':{'response_format':{'type':'json_schema'}}},'mtp3'), ({},'standard')):
                self.assertEqual(cov(self.snapshot(active),self.snapshot(active),case,execution_mode=mode)[1],[])
                self.assertTrue(cov(a,b,case,execution_mode=mode)[1])
            self.assertTrue(cov(b,a,{'body':{}})[1])
            for v in (None,[],[True]*16,[0.0]*16,[0]*15):
                c=copy.deepcopy(b);c['mtp']['completed_cycles_by_proposed_depth']=v
                self.assertTrue(cov(a,c,{'body':{}})[1])
    def test_runtime_path_all_occurrences_and_abbreviations(self):
        build=Path('/tmp/exact-child').resolve()
        self.assertEqual(q.runtime_args(['measure'],build),['measure','--runtime-build',str(build)])
        self.assertEqual(q.runtime_args(['measure','--runtime-build='+str(build)],build),['measure','--runtime-build='+str(build)])
        for rest in (['measure','--runtime-build'],['measure','--runtime-build='],['measure','--runtime-b',str(build)],
                ['measure','--runtime-build','/tmp/other'],['measure','--runtime-build',str(build),'--runtime-build=/tmp/other']):
            with self.assertRaises(ValueError): q.runtime_args(rest,build)
    def test_install_keeps_every_old_error_and_ownership(self):
        import importlib.util
        p=Path('dev/benchmarks/guard_hc_fast_composite_sep22/semantic_quality.py')
        spec=importlib.util.spec_from_file_location('_parent_install',p)
        parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
        class Runner:
            def gate_status(self,*a,**kw):return ['original22/base/teacher/compact/HC error']
            def coverage(self,*a,**kw):return {'old_detail':True},['old_guard_error']
            def ownership_policy(self,s):return {'old_owner':True}
        r=parent.install(Runner(),q.make_hooks(http,True));s=self.snapshot(True)
        self.assertIn('original22/base/teacher/compact/HC error',r.gate_status(s,None,None))
        details,errors=r.coverage(s,s,{'body':{}})
        self.assertTrue(details['old_detail']);self.assertIn('old_guard_error',errors)
        self.assertTrue(r.ownership_policy(s)['old_owner'])
    def test_real_sealed_metadata_both_flags_and_fresh_receipt(self):
        build=Path('build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2')
        for active in (False,True):q.load(build,expected=active,require_state=True)
        actual=q.digest
        for suffix in ('splash-flash','splash.metallib','rawQ4-qualified.air','Root-rawQ4-native-qualified.json'):
            with mock.patch.object(q,'digest',side_effect=lambda p,s=suffix: 'f'*64 if str(p).endswith(s) else actual(p)):
                with self.assertRaises(ValueError):q.authenticate(build,require_state=True)
    def test_private_driver_provenance_anchors_original_sources(self):
        from dev.benchmarks.raw_q4_verify_worker_sep22 import tuning
        with tempfile.TemporaryDirectory() as temp:
            package=Path(temp)
            (package/'config.json').write_text('{}')
            build=Path('build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2').resolve()
            args=SimpleNamespace(binary=build/'splash-flash',package=package,tokenizer=package,semantic_plan=None)
            def checked_source(path):
                self.assertTrue(path.is_file(),str(path))
                return 'a'*64
            with mock.patch.object(tuning,'sha256_file',side_effect=checked_source):
                result=tuning.provenance(args)
            for name in ('flash_context_grid.py','splash_tuning_sep21.py','prefill4k_attribution_quality.py'):
                self.assertIn('dev/benchmarks/'+name,result['source_file_sha256'])
            self.assertIn('dev/benchmarks/raw_q4_verify_worker_sep22/tuning.py',result['source_file_sha256'])

if __name__=='__main__':unittest.main()
