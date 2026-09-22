#!/usr/bin/env python3
"""CPU comparison of sealed raw-Q4 reports with distinct recorded flag settings.

This is additive postrun code. It neither edits nor substitutes a measured
report, task plan, grader, shipping source, or frozen measurement adapter.
Actual response reports are read only by Root when this command is executed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

RAW_HELPER_SHA = 'be9f0a262eebd1ea3a67e9b6ace88571f5cbeab3dc10c5dfb36f3405f67d772e'
PARENT_HELPER_SHA = 'c9ca7d8c8d06d99aa54af8fbb708318f72cfbaae5e8f6af156b08b1c545e93a2'

def file_digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def import_pinned(path, expected, name):
    if file_digest(path) != expected: raise ValueError('Frozen comparison dependency drift: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class RecordedFlags:
    """Only complete, explicitly registered report status contexts can dispatch."""
    def __init__(self, raw, http):
        self.raw, self.http, self.registered = raw, http, {}
        self.hooks = {flag: raw.make_hooks(http, flag) for flag in (False, True)}
        self.report_flags = []

    def bind_reports(self, reports, expected_flags, plan):
        if len(reports) != len(expected_flags) or len(reports) < 2:
            raise ValueError('Each complete comparison report requires one explicit expected flag')
        frozen = {case['id']: case for case in plan['cases']}
        if len(frozen) != 22: raise ValueError('Original comparison requires all22 frozen cases')
        staged = {}; declared = []
        for report, flag in zip(reports, expected_flags):
            if type(flag) is not bool: raise ValueError('Recorded expected flag must be Boolean')
            if (report.get('schema') != 'splash-prefill4k-semantic-report-v1'
                    or report.get('execution_mode') != 'mtp3' or report.get('completed') is not True
                    or report.get('full_plan_coverage') is not True
                    or report.get('strict_cache_graph_coverage_required') is not True
                    or report.get('runtime_file_sha256') != {'splash-flash': self.raw.EXE, 'splash.metallib': self.raw.LIB}
                    or report.get('plan_content_sha256') != plan['content_sha256']):
                raise ValueError('Recorded report does not bind the completed exact raw-Q4 runtime/plan')
            cases = report.get('cases')
            if not isinstance(cases, list) or len(cases) != 22 or {case.get('id') for case in cases} != set(frozen):
                raise ValueError('Recorded report lacks all22 unique original cases')
            values = [report.get('initial_status')]
            for case in cases:
                before, after = case.get('status_before'), case.get('status_after')
                values.extend((before, after))
                errors = self.hooks[flag][1](before, after, frozen[case['id']], True, 'mtp3')[1]
                if errors: raise ValueError('Recorded raw-Q4 case context invalid: ' + '; '.join(errors))
            values.append(report.get('final_status'))
            previous = None
            for status in values:
                errors = self.hooks[flag][0](status, plan, report.get('store_witness'), 'mtp3')
                if errors: raise ValueError('Recorded raw-Q4 flag/profile invalid: ' + '; '.join(errors))
                calls = self.http.get_path(status, self.raw.SECTION + '.graph_calls')
                if previous is not None and calls < previous:
                    raise ValueError('Recorded raw-Q4 process calls decreased across saved report statuses')
                previous = calls
                key = fingerprint(status)
                if key in staged and staged[key] is not flag:
                    raise ValueError('A saved status has conflicting recorded flag ownership')
                staged[key] = flag
            declared.append(flag)
        self.registered, self.report_flags = staged, declared
        return self

    def flag_for(self, status):
        key = fingerprint(status)
        if key not in self.registered: raise ValueError('Unregistered saved raw-Q4 report status context')
        return self.registered[key]

    def status(self, status, plan, store, execution_mode='mtp3'):
        try: flag = self.flag_for(status)
        except (ValueError, TypeError): return ['Unregistered saved raw-Q4 report status context']
        return self.hooks[flag][0](status, plan, store, execution_mode)

    def coverage(self, before, after, case, require_counters=True, execution_mode='mtp3'):
        try: a, b = self.flag_for(before), self.flag_for(after)
        except (ValueError, TypeError): return {}, ['Unregistered saved raw-Q4 report coverage context']
        if a is not b: return {}, ['Mixed recorded raw-Q4 flag contexts in one request']
        return self.hooks[a][1](before, after, case, require_counters, execution_mode)

    def ownership(self, status):
        try: flag = self.flag_for(status)
        except (ValueError, TypeError): raise ValueError('Unregistered saved raw-Q4 ownership context')
        return self.hooks[flag][2](status)

def load_parent(build):
    build = Path(build).resolve()
    raw = import_pinned(build / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py',
        RAW_HELPER_SHA, '_postrun_frozen_rawQ4')
    _, origin = raw.authenticate(build, require_state=True)
    parent = import_pinned(origin / 'source/dev/benchmarks/guard_hc_fast_composite_sep22/semantic_quality.py',
        PARENT_HELPER_SHA, '_postrun_frozen_original_composite')
    return raw, parent, parent.load(origin, require_state=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--expected-flags', nargs='+', choices=('0', '1'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError('Choose a fresh additive comparison output')
    raw, parent, runner = load_parent(args.build)
    reports = [runner.http.strict_json(path.read_text()) for path in args.reports]
    for report in reports: runner.strict_numbers(report)
    plan = runner.read_plan(Path(reports[0]['plan']))
    dispatch = RecordedFlags(raw, runner.http).bind_reports(reports,
        [flag == '1' for flag in args.expected_flags], plan)
    runner = parent.install(runner, (dispatch.status, dispatch.coverage, dispatch.ownership))
    # The literal original comparator re-reads reports and audits all old gates,
    # bodies, statuses, request terminals and output grades. Registered status
    # fingerprints reject an intervening status substitution. No runtime-change
    # allowance is used; both report runtime hashes must equal663/754 above.
    return runner.compare(SimpleNamespace(reports=args.reports, output=args.output, allow_runtime_change=False))

if __name__ == '__main__': raise SystemExit(main())
