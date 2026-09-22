#!/usr/bin/env python3
"""Additive Root-only comparison of the registered Q4 and Q4+Q5 reports.

Preparation/tests use synthetic metadata only. Root reads actual completed
reports, their unchanged frozen plan, and outputs when executing main().
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path('/Users/mweinbach/Projects/splash')
Q5_FAMILY = ';private-rawQ5-rowpair-mainGDNout-VerifyR4-sourceSha256='
REGISTRY = {
    'parent': {
        'build': ROOT / 'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2',
        'helper': 'raw_q4_verify_worker_sep22',
        'helper_sha': 'be9f0a262eebd1ea3a67e9b6ace88571f5cbeab3dc10c5dfb36f3405f67d772e',
        'proof': 'Root-rawQ4-native-qualified.json',
        'proof_sha': '4d4ce0cee4e9c40351b443ec069162eb5f0098d0e2a392809bba0dd06d4c8818',
        'source': '162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a',
        'runtime': {'splash-flash': '663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438',
                    'splash.metallib': '7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8'}},
    'combined': {
        'build': ROOT / 'build/rawQ4Q5-GDN-VerifyR4-composite-sep22-worker-v1',
        'helper': 'raw_q5_verify_worker_sep22',
        'helper_sha': 'bcaac55afca904181fb4ce15a8cfac1fc70b3eb1be53bdfe9b7541016455b81f',
        'proof': 'Root-rawQ4Q5-native-qualified.json',
        'proof_sha': '010f1215993ad880046810cbbcaafa11fee5208b5e9e9d849197a2dc80ef79cf',
        'source': '90d42b5ce8edd6c7254b6f9286628a7ffa7a6eede53fa3058eba8b519e9a64d6',
        'runtime': {'splash-flash': '613e4dfe6a9b429cabf8fbd2270b5857dbc601cde6d1b3af10c46fec1b8095c0',
                    'splash.metallib': '8fffe24fd4d99174cc522dcef6090b5b2e72e36f26e5c78afe9df46a118d4246'}}}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Profile:
    def __init__(self, name, runner):
        self.name, self.runner, self.record = name, runner, REGISTRY[name]
        # Capture complete old gates before patching the comparison namespace.
        self.status, self.coverage = runner.gate_status, runner.coverage
        self.ownership, self.execution = runner.ownership_policy, runner.execution_policy


def load_profiles():
    profiles = {}
    for name, record in REGISTRY.items():
        build = record['build']
        path = build / ('source/dev/benchmarks/' + record['helper'] + '/semantic_quality.py')
        if file_digest(path) != record['helper_sha'] or file_digest(build / record['proof']) != record['proof_sha']:
            raise ValueError('Registered profile helper/current native proof drift: ' + name)
        for artifact, expected in record['runtime'].items():
            if file_digest(build / artifact) != expected:
                raise ValueError('Registered profile runtime drift: ' + name)
        spec = importlib.util.spec_from_file_location('_postrun_Q4Q5_' + name, path)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        profiles[name] = Profile(name, helper.load(build, expected=True, require_state=True))
    a, b = profiles['parent'].runner, profiles['combined'].runner
    if a is b or a.compare.__globals__ is b.compare.__globals__:
        raise ValueError('Registered profile comparators must have independent namespaces')
    if a.specs() != b.specs() or len(a.specs()) != 22 or a.grade_record.__code__.co_code != b.grade_record.__code__.co_code or a.compare.__code__.co_code != b.compare.__code__.co_code:
        raise ValueError('Registered profiles changed original22 specs/graders/comparator')
    return profiles


class RecordedProfiles:
    def __init__(self, profiles, http):
        self.profiles, self.http, self.registered = profiles, http, {}

    def bind_reports(self, reports, plan):
        if len(reports) != 2 or set(self.profiles) != {'parent', 'combined'}:
            raise ValueError('Exactly the registered enabled-Q4 parent and combined report are required')
        frozen = {case['id']: case for case in plan['cases']}
        order = [case['id'] for case in plan['cases']]
        if len(frozen) != 22 or len(order) != 22:
            raise ValueError('All22 unique original frozen cases are required')
        staged, policies = {}, []
        for report, name in zip(reports, ('parent', 'combined')):
            profile = self.profiles[name]
            if (report.get('schema') != 'splash-prefill4k-semantic-report-v1'
                    or report.get('execution_mode') != 'mtp3' or report.get('completed') is not True
                    or report.get('full_plan_coverage') is not True
                    or report.get('strict_cache_graph_coverage_required') is not True
                    or report.get('runtime_file_sha256') != profile.record['runtime']
                    or report.get('plan_content_sha256') != plan['content_sha256']):
                raise ValueError('Report does not bind its specific completed runtime/plan: ' + name)
            cases = report.get('cases')
            if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases) or [case.get('id') for case in cases] != order:
                raise ValueError('Report lacks original22 cases in recorded order: ' + name)
            values = [report.get('initial_status')]
            for case in cases:
                records = case.get('records')
                if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
                    raise ValueError('Specific frozen task lacks one actual response record: ' + name)
                before, after = case.get('status_before'), case.get('status_after')
                errors = profile.coverage(before, after, frozen[case['id']], True, 'mtp3')[1]
                if errors:
                    raise ValueError('Specific report coverage invalid: ' + name + ': ' + '; '.join(errors))
                values.extend((before, after))
            values.append(report.get('final_status'))
            previous = None
            for status in values:
                routes = self.http.get_path(status, 'identity.kernel_routes')
                identity = status.get('identity') if isinstance(status, dict) else None
                if not isinstance(status, dict) or (name == 'parent' and ('raw_q5_rowpair_verify' in status or (isinstance(identity, dict) and 'raw_q5_rowpair_verify' in identity) or (isinstance(routes, str) and Q5_FAMILY in routes))):
                    raise ValueError('Unknown/mixed specific runtime status context: ' + name)
                errors = profile.status(status, plan, report.get('store_witness'), 'mtp3')
                if errors:
                    raise ValueError('Specific full report profile invalid: ' + name + ': ' + '; '.join(errors))
                if not self.http.idle(status):
                    raise ValueError('Specific saved report status is not idle: ' + name)
                policy = profile.execution(status)
                if policy != report.get('actual_execution_policy'):
                    raise ValueError('Specific actual execution policy differs: ' + name)
                counters = [self.http.get_path(status, 'raw_q4_rowpair_verify.graph_calls')]
                if name == 'combined': counters.append(self.http.get_path(status, 'raw_q5_rowpair_verify.graph_calls'))
                if any(type(value) is not int or value < 0 or value >= 2**64 for value in counters) or (previous is not None and any(y < x for x, y in zip(previous, counters))):
                    raise ValueError('Specific process graph counters invalid/decreased: ' + name)
                previous = counters
                key = fingerprint(status)
                if key in staged and staged[key] != name:
                    raise ValueError('A saved status has mixed registered runtime ownership')
                staged[key] = name
            policies.append(report['actual_execution_policy'])
        if policies[0] != policies[1]:
            raise ValueError('Registered runtime change cannot change frozen request execution policy')
        self.registered = staged
        return self

    def profile_for(self, status):
        key = fingerprint(status)
        if key not in self.registered:
            raise ValueError('Unregistered saved specific runtime status context')
        return self.profiles[self.registered[key]]

    def status(self, status, plan, store, execution_mode='mtp3'):
        try: profile = self.profile_for(status)
        except (ValueError, TypeError): return ['Unregistered saved specific runtime status context']
        return profile.status(status, plan, store, execution_mode)

    def coverage(self, before, after, case, require_counters=True, execution_mode='mtp3'):
        try: a, b = self.profile_for(before), self.profile_for(after)
        except (ValueError, TypeError): return {}, ['Unregistered saved specific runtime coverage context']
        if a.name != b.name: return {}, ['Mixed registered runtimes in one saved request']
        return a.coverage(before, after, case, require_counters, execution_mode)

    def ownership(self, status):
        return self.profile_for(status).ownership(status)


def install_dispatch(runner, dispatch):
    namespace = runner.compare.__globals__
    for name in ('gate_status', 'coverage', 'ownership_policy'):
        if namespace.get(name) is not getattr(runner, name):
            raise ValueError('Original comparator has an unregistered gate namespace')
    for name, method in [('gate_status', dispatch.status), ('coverage', dispatch.coverage), ('ownership_policy', dispatch.ownership)]:
        namespace[name] = method
        setattr(runner, name, method)
    return runner


def check_report_pins(paths, pins):
    if len(paths) != 2 or len(pins) != 2 or any(not isinstance(pin, str) or re.fullmatch('[0-9a-f]{64}', pin) is None for pin in pins):
        raise ValueError('Each specific completed report requires an external exact SHA')
    for path, pin in zip(paths, pins):
        if file_digest(path) != pin: raise ValueError('Externally registered report digest differs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--reports', nargs=2, type=Path, required=True)
    parser.add_argument('--report-shas', nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    staging = args.output.with_name(args.output.name + '.audit-staging')
    if args.output.exists() or staging.exists(): raise FileExistsError('Fresh additive comparison output required')
    check_report_pins(args.reports, args.report_shas)
    profiles = load_profiles(); runner = profiles['parent'].runner
    reports = [runner.http.strict_json(path.read_text()) for path in args.reports]
    for report in reports: runner.strict_numbers(report)
    plan = runner.read_plan(Path(reports[0]['plan']))
    dispatch = RecordedProfiles(profiles, runner.http).bind_reports(reports, plan)
    original_compare, original_grade = runner.compare, runner.grade_record
    install_dispatch(runner, dispatch)
    if runner.compare is not original_compare or runner.grade_record is not original_grade:
        raise ValueError('Original comparator/grader function identity changed')
    # Only the two exact registered bit-preserving derivatives are admitted.
    # Original compare still audits frozen bodies, old gates, statuses, fresh
    # terminals, actual execution policy and grades every actual output.
    result = runner.compare(SimpleNamespace(reports=args.reports, output=staging, allow_runtime_change=True))
    check_report_pins(args.reports, args.report_shas)
    # Atomic no-clobber publication also refuses a destination created while
    # the original comparator was auditing. Source reports remain untouched.
    os.link(staging, args.output, follow_symlinks=False)
    staging.unlink()
    return result


if __name__ == '__main__': raise SystemExit(main())
