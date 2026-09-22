"""Synthetic metadata-only postrun-dispatch regressions; no actual responses."""
import copy
import unittest
from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks.raw_q4_verify_worker_sep22 import semantic_quality as raw
from dev.benchmarks.raw_q4_verify_worker_sep22.postrun_compare import RecordedFlags, load_parent
from dev.benchmarks.raw_q4_verify_worker_sep22 import test_semantic_quality as snapshots

class PostrunTests(unittest.TestCase):
    def fixture(self, active):
        snapshot=snapshots.RawTests().snapshot
        plan={'content_sha256':'a'*64,'cases':[{'id':str(i),'body':{}} for i in range(22)]}
        report={'schema':'splash-prefill4k-semantic-report-v1','execution_mode':'mtp3','completed':True,
            'full_plan_coverage':True,'strict_cache_graph_coverage_required':True,
            'runtime_file_sha256':{'splash-flash':raw.EXE,'splash.metallib':raw.LIB},
            'plan_content_sha256':plan['content_sha256'],'initial_status':snapshot(active,0),
            'final_status':snapshot(active,22),'store_witness':{},
            'cases':[{'id':str(i),'status_before':snapshot(active,i),'status_after':snapshot(active,i+1)} for i in range(22)]}
        return plan,report
    def dispatcher(self):
        plan,a=self.fixture(False);_,b=self.fixture(True)
        return RecordedFlags(raw,http).bind_reports([a,b],[False,True],plan),plan,a,b
    def test_distinct_flags_dispatch_both_full_reports_and_keyword_boundary(self):
        d,p,a,b=self.dispatcher()
        for active,r in ((False,a),(True,b)):
            self.assertIs(d.flag_for(r['initial_status']),active)
            self.assertEqual(d.status(status=r['initial_status'],plan=p,store={}),[])
            for case in r['cases']:
                self.assertEqual(d.coverage(case['status_before'],case['status_after'],p['cases'][int(case['id'])])[1],[])
    def test_mixed_tampered_flags_markers_sources_and_unregistered_context(self):
        for name in ('requested','source_identity_sha256','qualified_candidate_AIR_sha256','graph_calls'):
            p,a=self.fixture(False);_,b=self.fixture(True)
            b['cases'][5]['status_before'][raw.SECTION][name]=False if name=='requested' else 'bad'
            with self.assertRaises(ValueError):RecordedFlags(raw,http).bind_reports([a,b],[False,True],p)
        d,p,a,b=self.dispatcher()
        self.assertTrue(d.coverage(a['cases'][0]['status_before'],b['cases'][0]['status_after'],p['cases'][0])[1])
        altered=copy.deepcopy(b['initial_status']);altered['unregistered_extra']='changed'
        self.assertTrue(d.status(altered,p,{}));self.assertTrue(d.status({},p,{}))
        with self.assertRaises(ValueError):d.ownership(altered)
    def test_explicit_flags_runtime_and_complete22_are_mandatory(self):
        p,a=self.fixture(False);_,b=self.fixture(True)
        for flags in ([True,True],[False,False],[False],[0,1]):
            with self.assertRaises(ValueError):RecordedFlags(raw,http).bind_reports([a,b],flags,p)
        for key,value in (('runtime_file_sha256',{'splash-flash':'f'*64,'splash.metallib':raw.LIB}),
                ('completed',False),('full_plan_coverage',False),('strict_cache_graph_coverage_required',False)):
            c=copy.deepcopy(b);c[key]=value
            with self.assertRaises(ValueError):RecordedFlags(raw,http).bind_reports([a,c],[False,True],p)
        b['cases'].pop()
        with self.assertRaises(ValueError):RecordedFlags(raw,http).bind_reports([a,b],[False,True],p)
    def test_original_gates_and_grader_identity_remain_intact(self):
        d,p,a,b=self.dispatcher()
        raw_module,parent,runner=load_parent('build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2')
        original_compare,original_grade=runner.compare,runner.grade_record
        old_status=runner.gate_status
        runner.gate_status=lambda *args,**kwargs:['original_base_teacher_compact_guard_HC_error']
        runner=parent.install(runner,(d.status,d.coverage,d.ownership))
        self.assertIn('original_base_teacher_compact_guard_HC_error',runner.gate_status(b['initial_status'],p,{}))
        self.assertIs(runner.compare,original_compare);self.assertIs(runner.grade_record,original_grade)
    def test_wrong_raw_deltas_and_missing_partial_profile_fail(self):
        for mutation in ('missing','partial','counter','marker'):
            p,a=self.fixture(False);_,b=self.fixture(True)
            s=b['cases'][3]['status_after']
            if mutation=='missing':del s[raw.SECTION]
            elif mutation=='partial':s[raw.SECTION]={'requested':True}
            elif mutation=='counter':s[raw.SECTION]['graph_calls']+=26;s[raw.SECTION]['graph_rows']+=104;s['compact_r4_preflight']['guard_preflight_graph_calls']+=48
            else:s['identity']['kernel_routes']='original'
            with self.assertRaises(ValueError):RecordedFlags(raw,http).bind_reports([a,b],[False,True],p)

if __name__=='__main__':unittest.main()
