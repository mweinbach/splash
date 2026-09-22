#!/usr/bin/env python3
"""Root-only additive regrading with one explicitly registered cap difference."""
from __future__ import annotations
import argparse
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from dev.benchmarks.mtp_fixed4_r5_sep22 import policy_quality as p

EQUALITY = "if len({digest(report['actual_execution_policy']) for report in reports}) != 1:"
REGISTERED_EQUALITY = "if len({digest(registered_comparison_policy(report)) for report in reports}) != 1:"

BATCH_PREFIX = 'native-flash-uniform-real-lane-prefill-shared-source-project-v1:'
BATCH_ILP_SUFFIX = ';gdn-batch-prefill-uniform-b2to4-r512plus-v32-t32-sg8-f32-separate-states'

def normalize_registered_identity_pair(identities, source):
    if not isinstance(identities, list) or len(identities) != 2 or not p.is_sha(source):
        raise ValueError('Two exact registered identities and source digest required')
    normalized = copy.deepcopy(identities)
    if not all(isinstance(value, dict) for value in normalized): raise ValueError('Saved identities must be complete dictionaries')
    marker = p.MARKER + source
    suffixes = []
    for index, identity in enumerate(normalized):
        identity.pop('engine_instance_id', None)
        routes, batch = identity.get('kernel_routes'), identity.get('batch_prefill_kernel_routes')
        if not isinstance(routes, str) or not isinstance(batch, str): raise ValueError('Both exact declared route displays required')
        expected_count = index
        for display in (routes, batch):
            if display.count(p.MARKER) != expected_count or display.count(marker) != expected_count:
                raise ValueError('Registered R5 marker count/source differs in route display')
        composition = BATCH_PREFIX + routes
        if batch == composition: suffix = ''
        elif batch == composition + BATCH_ILP_SUFFIX: suffix = BATCH_ILP_SUFFIX
        else: raise ValueError('Batch-prefill route display differs from unchanged source getter')
        suffixes.append(suffix)
        if index:
            # Remove exactly the same authenticated R5 execution marker from
            # both display fields, preserving every prefix/suffix/other key.
            identity['kernel_routes'] = routes.replace(marker, '', 1)
            identity['batch_prefill_kernel_routes'] = batch.replace(marker, '', 1)
    if suffixes[0] != suffixes[1]: raise ValueError('Batch-prefill optional ILP display suffix changed')
    if not p.same(normalized[0], normalized[1]): raise ValueError('Unknown cross-runtime identity drift')
    return normalized

class Contexts:
    def __init__(self,runner,b):
        self.runner,self.b,self.contexts=runner,b,{}
        self.old_gate,self.old_cov,self.old_own=runner.gate_status,runner.coverage,runner.ownership_policy
        self.hooks=p.make_hooks(runner.http,b)
    def bind(self,reports,plan):
        if len(reports)!=2 or plan['content_sha256']!=p.PLAN_CONTENT:raise ValueError('Exact original22 fixed3 then fixed4 pair required')
        frozen={case['id']:case for case in plan['cases']}
        if len(frozen)!=22:raise ValueError('Original22 unique plan required')
        staged={}
        for report,depth in zip(reports,(3,4)):
            runtime={'splash-flash':p.PARENT_EXE,'splash.metallib':p.PARENT_LIB} if depth==3 else {'splash-flash':self.b['exe_sha256'],'splash.metallib':self.b['library_sha256']}
            if (report.get('schema')!='splash-prefill4k-semantic-report-v1' or report.get('execution_mode')!='mtp3'
                    or report.get('runtime_file_sha256')!=runtime or report.get('plan_content_sha256')!=p.PLAN_CONTENT
                    or report.get('completed') is not True or report.get('full_plan_coverage') is not True
                    or report.get('strict_cache_graph_coverage_required') is not True):raise ValueError('Saved report does not bind registered runtime/plan')
            cases=report.get('cases')
            if not isinstance(cases,list) or len(cases)!=22 or {c.get('id') for c in cases}!=set(frozen):raise ValueError('Saved report lacks unique original22 coverage')
            values=[report.get('initial_status')]
            for case in cases:values.extend((case.get('status_before'),case.get('status_after')))
            values.append(report.get('final_status'));previous=None
            for status in values:
                errors=p.policy_errors(status,depth)
                if depth==4:errors+=p.r5_status_errors(status,self.b)
                elif p.SECTION in status or p.MARKER in str(p.get(status,'identity.kernel_routes')):errors.append('Parent runtime unexpectedly declares R5 profile')
                if errors:raise ValueError('Saved fixed-policy declaration invalid: '+'; '.join(errors))
                hist=p.get(status,'mtp.completed_cycles_by_proposed_depth')
                if previous is not None and any(y<x for x,y in zip(previous,hist)):raise ValueError('Saved process depth counters decrease')
                previous=hist;key=p.fingerprint(status)
                if key in staged and staged[key]!=depth:raise ValueError('Conflicting policy ownership')
                staged[key]=depth
            initial=report['initial_status']
            if self.runner.execution_policy(initial)!=report['actual_execution_policy']:raise ValueError('Own actual policy summary differs from actual initial status')
        self.contexts=staged
        for report in reports:
            for case in report['cases']:
                a,z=case['status_before'],case['status_after']
                errors=self.coverage(a,z,frozen[case['id']],True,'mtp3')[1]
                if errors:raise ValueError('Saved full old/new coverage invalid: '+'; '.join(errors))
        normalize_registered_identity_pair([report['initial_status']['identity'] for report in reports], self.b['source_identity_sha256'])
        return self
    def find(self,status):
        key=p.fingerprint(status)
        if key not in self.contexts:raise ValueError('Unregistered complete saved status')
        return self.contexts[key]
    def gate(self,status,plan,store,execution_mode='mtp3'):
        try:depth=self.find(status)
        except (TypeError,ValueError):return ['Unregistered saved fixed-policy status']
        own=p.policy_errors(status,depth)
        if depth==4:
            own+=p.r5_status_errors(status,self.b);status=p.inherited_status_view(status)
        return self.old_gate(status,plan,store,execution_mode)+own
    def coverage(self,a,z,case,require_counters=True,execution_mode='mtp3'):
        try:x,y=self.find(a),self.find(z)
        except (TypeError,ValueError):return {},['Unregistered saved coverage status']
        if x!=y:return {},['Mixed fixed-policy contexts in request']
        details,errors=self.old_cov(a,z,case,require_counters,execution_mode)
        if x==4:
            extra,added=self.hooks[1](a,z,case,require_counters,execution_mode);details.update(extra);errors+=added
        else:errors+=p.native_metadata()(a,z)['errors']
        return details,errors
    def ownership(self,status):
        depth=self.find(status);extra=self.hooks[2](status) if depth==4 else {'registered_singleton_policy':3}
        return {**self.old_own(status),**extra}
    def comparison_policy(self,report):
        depth=self.find(report['initial_status'])
        if not p.same(report['actual_execution_policy'].get('mtp.singleton_maximum_draft_tokens'),depth):raise ValueError('Unregistered policy summary cap')
        return p.normalize_policy_summary(report)

def install_comparator(runner,contexts):
    source=inspect.getsource(runner.compare)
    if source.count(EQUALITY)!=1:raise ValueError('Frozen comparator equality source anchor changed')
    changed=source.replace(EQUALITY,REGISTERED_EQUALITY)
    if changed.count(REGISTERED_EQUALITY)!=1 or changed.replace(REGISTERED_EQUALITY,EQUALITY)!=source:raise ValueError('Comparator inverse differs')
    runner.gate_status=contexts.gate;runner.coverage=contexts.coverage;runner.ownership_policy=contexts.ownership
    runner.__dict__['registered_comparison_policy']=contexts.comparison_policy
    namespace={};exec(compile(changed,str(Path(__file__).resolve())+'::registered-original-compare','exec'),runner.__dict__,namespace)
    runner.compare=namespace['compare']
    return {'original_function_sha256':p.hashlib.sha256(source.encode()).hexdigest(),
            'new_function_sha256':p.hashlib.sha256(changed.encode()).hexdigest(),'inverse_exact':True,
            'sole_edit':'registered cross-report policy equality: singleton cap3/4 normalized; original reconstruction and grading literal'}

def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--binding-sha256',required=True)
    parser.add_argument('--reports',nargs=2,type=Path,required=True);parser.add_argument('--report-sha256',nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('Fresh additive comparison required')
    b=p.authenticate(args.binding,args.binding_sha256);runner=p.load_parent()
    for path,wanted in zip(args.reports,args.report_sha256):
        if not p.is_sha(wanted) or p.sha(path)!=wanted:raise ValueError('Externally registered report SHA differs')
    reports=[runner.http.strict_json(path.read_text()) for path in args.reports]
    for report in reports:runner.strict_numbers(report)
    plan=runner.read_plan(Path(reports[0]['plan']));contexts=Contexts(runner,b).bind(reports,plan)
    install_comparator(runner,contexts)
    return runner.compare(SimpleNamespace(reports=args.reports,output=args.output,allow_runtime_change=True))

if __name__=='__main__':raise SystemExit(main())
