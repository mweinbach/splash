#!/usr/bin/env python3
"""Audit own policy on saved original22 reports; preserve every original grade."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from dev.benchmarks.mtp_adaptive_q4_sep22 import policy_quality as policy

ROOT=policy.ROOT
FROZEN=ROOT/'dev/benchmarks/raw_q4_verify_worker_sep22/postrun_compare.py'
FROZEN_SHA='c414600b66cc7204ae6b6e1e5ed9b0d8cf785d92599f28a7cf01c8fbda2a7ef2'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

class OwnPolicy:
    def __init__(self,raw,http): self.raw,self.http,self.contexts=raw,http,{}
    def bind(self,reports,expected,plan):
        if expected != ['fixed3','adaptive'] or len(reports)!=2: raise ValueError('Exact fixed3 then adaptive pair required')
        frozen={case['id']:case for case in plan['cases']}
        if len(frozen)!=22: raise ValueError('All original22 required')
        staged={}
        for report,wanted in zip(reports,expected):
            if (report.get('schema')!='splash-prefill4k-semantic-report-v1' or report.get('execution_mode')!='mtp3'
                    or report.get('completed') is not True or report.get('full_plan_coverage') is not True
                    or report.get('strict_cache_graph_coverage_required') is not True
                    or report.get('runtime_file_sha256') != {'splash-flash':self.raw.EXE,'splash.metallib':self.raw.LIB}
                    or report.get('plan_content_sha256') != plan['content_sha256']): raise ValueError('Saved report lacks exact completed ownRuntime scope')
            cases=report.get('cases')
            if not isinstance(cases,list) or len(cases)!=22 or {case.get('id') for case in cases} != set(frozen): raise ValueError('All unique original22 required')
            statuses=[report.get('initial_status')]
            for case in cases:
                a,b=case.get('status_before'),case.get('status_after')
                errors=policy.make_hooks(self.http,wanted)[1](a,b,frozen[case['id']],True,'mtp3')[1]
                if errors: raise ValueError('Saved own-policy coverage invalid: '+'; '.join(errors))
                statuses += [a,b]
            statuses.append(report.get('final_status'))
            for status in statuses:
                errors=policy.policy_errors(status,wanted)
                if errors: raise ValueError('Saved own policy invalid: '+'; '.join(errors))
                key=fingerprint(status)
                if key in staged and staged[key]!=wanted: raise ValueError('Conflicting own-policy status')
                staged[key]=wanted
        self.contexts=staged;return self
    def find(self,status):
        key=fingerprint(status)
        if key not in self.contexts: raise ValueError('Unregistered own-policy saved status')
        return self.contexts[key]
    def status(self,status,plan,store,execution_mode='mtp3'):
        try: expected=self.find(status)
        except (ValueError,TypeError): return ['Unregistered own-policy status']
        return policy.make_hooks(self.http,expected)[0](status,plan,store,execution_mode)
    def coverage(self,a,b,case,require_counters=True,execution_mode='mtp3'):
        try: x,y=self.find(a),self.find(b)
        except (ValueError,TypeError): return {},['Unregistered own-policy coverage']
        if x!=y: return {},['Mixed own-policy contexts in request']
        return policy.make_hooks(self.http,x)[1](a,b,case,require_counters,execution_mode)
    def ownership(self,status): return policy.make_hooks(self.http,self.find(status))[2](status)

def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--reports',nargs=2,type=Path,required=True)
    parser.add_argument('--report-sha256',nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError('Fresh comparison required')
    for path,expected in zip(args.reports,args.report_sha256):
        if sha(path)!=expected: raise ValueError('Externally registered saved report differs')
    if sha(FROZEN)!=FROZEN_SHA: raise ValueError('Recorded raw-Q4 comparator drift')
    spec=importlib.util.spec_from_file_location('_adaptive_frozen_raw_compare',FROZEN)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    raw,parent,runner=original.load_parent(policy.WORKER)
    reports=[runner.http.strict_json(path.read_text()) for path in args.reports]
    for report in reports:runner.strict_numbers(report)
    plan=runner.read_plan(Path(reports[0]['plan']))
    if plan['content_sha256']!='a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac': raise ValueError('Exact frozen original22 plan required')
    identities=[dict(report['initial_status']['identity']) for report in reports]
    for identity in identities:
        for field in ('engine_instance_id','worker_semantics'):identity.pop(field,None)
    if identities[0]!=identities[1]:raise ValueError('Unexpected cross-policy identity drift')
    flags=original.RecordedFlags(raw,runner.http).bind_reports(reports,[True,True],plan)
    runner=parent.install(runner,(flags.status,flags.coverage,flags.ownership))
    own=OwnPolicy(raw,runner.http).bind(reports,['fixed3','adaptive'],plan)
    runner=parent.install(runner,(own.status,own.coverage,own.ownership))
    # The literal original comparator preserves caps/context/sampling, all
    # frozen requests, response grading and unrelated runtime/resource gates.
    # Its execution-policy fields are identical; intended controller metadata
    # above is independently authenticated on each complete saved context.
    return runner.compare(SimpleNamespace(reports=args.reports,output=args.output,allow_runtime_change=False))

if __name__=='__main__':raise SystemExit(main())
