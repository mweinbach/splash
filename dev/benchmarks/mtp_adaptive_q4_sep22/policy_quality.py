#!/usr/bin/env python3
"""Private policy gate on the unchanged authenticated Q4 original22 runner."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / 'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2'
RAW_SHA = 'be9f0a262eebd1ea3a67e9b6ace88571f5cbeab3dc10c5dfb36f3405f67d772e'
CONTROLLER = 'native-mtp-conditional-acceptance-measured-wall-cost-initial-confirm-bidirectional-probe-v2'
NATIVE_POLICY_PATH = ROOT / 'dev/benchmarks/flash_http_performance.py'
NATIVE_POLICY_SHA = 'af0c883e1bb5c787e8cabdc004337172397ccd074c8beecbc454e29bac260389'
_native_policy_module = None
FIELDS = ('identity.worker_semantics', 'mtp.singleton_depth_override',
          'mtp.depth_controller_semantics', 'mtp.policy')
FIXED_POLICY = ('singleton fixed cap3 bounded by output budget; concurrent ready MTP peers or pending admission cap singleton drafts at 3; '
                'joint fixedcap3; greedy unmasked; AR for ineligible requests')
ADAPTIVE_POLICY = 'singleton adaptive depth0..3; joint fixedcap3; greedy unmasked; AR for ineligible requests'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def get(value, path):
    for part in path.split('.'):
        if not isinstance(value, dict) or part not in value: return object()
        value = value[part]
    return value
def same(a, b): return type(a) is type(b) and a == b
def u64(x): return type(x) is int and 0 <= x < 2**64

def native_policy_metadata():
    """Capture the exact helper without resolving it through a frozen package.

    load() calls this after raw.load establishes its existing dev namespace.
    The standalone native helper imports the already available HTTPClient;
    the original runner's frozen dependencies retain their original paths.
    """
    global _native_policy_module
    if sha(NATIVE_POLICY_PATH) != NATIVE_POLICY_SHA: raise ValueError('Exact native policy helper drift')
    if _native_policy_module is None:
        spec = importlib.util.spec_from_file_location('_Q4_private_exact_native_policy', NATIVE_POLICY_PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        _native_policy_module = module
    return _native_policy_module.native_mtp_depth_metadata

def policy_errors(status, expected):
    if expected not in ('fixed3', 'adaptive'): raise ValueError('Unknown registered singleton policy')
    adaptive = expected == 'adaptive'
    wanted = {
        'capabilities.mtp': True, 'capabilities.batch_mtp': True,
        'mtp.enabled': True, 'mtp.maximum_draft_tokens': 3,
        'mtp.singleton_maximum_draft_tokens': 3,
        'mtp.singleton_concurrent_draft_cap': 3, 'mtp.joint_maximum_draft_tokens': 3,
        'mtp.teacher_cache_only_requested': True,
        'mtp.singleton_depth_override': None if adaptive else 3,
        'mtp.depth_controller_semantics': CONTROLLER if adaptive else None,
        'mtp.policy': ADAPTIVE_POLICY if adaptive else FIXED_POLICY,
        'identity.worker_semantics': ('native-worker5-joint-greedy-mtp3-and-singleton-policy-v4'
                                     if adaptive else 'native-worker5-singleton-fixed-depth1..15-fold8-jointcap3-v6'),
        'mtp.joint_policy': 'fixed cap3 shared-budget/EOS bound; true joint head and target; survivors retain independent prefixes',
    }
    errors = ['Registered singleton policy differs: ' + key for key, value in wanted.items()
              if not same(get(status, key), value)]
    histogram = get(status, 'mtp.completed_cycles_by_proposed_depth')
    if not isinstance(histogram, list) or len(histogram) != 16 or not all(u64(x) for x in histogram):
        errors.append('Registered singleton requires sixteen U64 actual-depth counters')
    elif any(histogram[4:]): errors.append('Registered max3 singleton recorded depth above3')
    return errors

def make_hooks(http, expected):
    if expected not in ('fixed3', 'adaptive'): raise ValueError('Unknown registered singleton policy')
    metadata = native_policy_metadata()
    def status(value, plan, store, execution_mode='mtp3'):
        if execution_mode != 'mtp3': return ['Registered singleton policy requires original mtp3 semantic mode']
        return policy_errors(value, expected)
    def coverage(before, after, case, require_counters=True, execution_mode='mtp3'):
        errors = status(before, None, None, execution_mode) + status(after, None, None, execution_mode)
        native_policy = metadata(before, after)
        errors.extend(native_policy['errors'])
        for path in FIELDS:
            if not same(get(before, path), get(after, path)): errors.append('Singleton policy changed during request: ' + path)
        a, b = (get(s, 'mtp.completed_cycles_by_proposed_depth') for s in (before, after))
        if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == 16
                and all(u64(x) for x in a + b) and all(y >= x for x, y in zip(a, b))):
            return {}, errors + ['Singleton actual-depth counters decreased or are invalid']
        return {'registered_singleton_depth_policy': expected,
                'registered_native_mtp_depth_metadata': native_policy,
                'actual_singleton_depth_cycle_deltas': [y-x for x, y in zip(a, b)],
                'actual_R4_cycle_delta': b[3]-a[3],
                'R4_source_gate_scope': 'unchanged raw26/HC97/preflight48/compact48 hooks count histogram3 only'}, errors
    def ownership(value): return {path: get(value, path) for path in FIELDS}
    return status, coverage, ownership

def load(build=WORKER, expected='adaptive'):
    build = Path(build).resolve()
    if build != WORKER.resolve(): raise ValueError('Adaptive policy requires the exact qualified Q4 worker')
    if sha(NATIVE_POLICY_PATH) != NATIVE_POLICY_SHA: raise ValueError('Exact native policy helper drift')
    path = build / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py'
    if sha(path) != RAW_SHA: raise ValueError('Qualified raw-Q4 helper drift')
    spec = importlib.util.spec_from_file_location('_adaptive_frozen_rawQ4', path)
    raw = importlib.util.module_from_spec(spec); spec.loader.exec_module(raw)
    runner = raw.load(build, expected=True, require_state=True)
    hooks = make_hooks(runner.http, expected)
    old_status, old_coverage, old_ownership = runner.gate_status, runner.coverage, runner.ownership_policy
    def gate_status(*args, **kwargs): return old_status(*args, **kwargs) + hooks[0](*args, **kwargs)
    def coverage(*args, **kwargs):
        details, errors = old_coverage(*args, **kwargs); extra, added = hooks[1](*args, **kwargs)
        return {**details, **extra}, errors + added
    def ownership(value): return {**old_ownership(value), **hooks[2](value)}
    runner.gate_status, runner.coverage, runner.ownership_policy = gate_status, coverage, ownership
    return runner

def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--depth-policy', choices=('fixed3', 'adaptive'), required=True)
    args, rest = parser.parse_known_args(argv)
    runner = load(args.build, args.depth_policy)
    if rest and rest[0] == 'measure':
        path = args.build.resolve(); explicit = False
        for i, token in enumerate(rest):
            option = token.split('=', 1)[0]
            if len(option) > 2 and option.startswith('--') and '--runtime-build'.startswith(option) and option != '--runtime-build':
                raise ValueError('Complete --runtime-build required')
            if token == '--runtime-build':
                if i+1 == len(rest): raise ValueError('Missing runtime-build')
                value = rest[i+1]
            elif token.startswith('--runtime-build='): value = token.split('=', 1)[1]
            else: continue
            explicit = True
            if not value or Path(value).resolve() != path: raise ValueError('Every runtime-build must equal663663 worker')
        if not explicit: rest += ['--runtime-build', str(path)]
    return runner.main(rest)

if __name__ == '__main__': raise SystemExit(main())
