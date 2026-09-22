#!/usr/bin/env python3
"""Prepare private adaptive/fixed commands using the same qualified binary."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'build/rawQ4-GDN26-matched-model-sep22-root-v2/root-flag1-command.json'
WORKER = ROOT / 'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2'
DRIVER = WORKER / 'source/dev/benchmarks/raw_q4_verify_worker_sep22/tuning.py'
DRIVER_SHA = '8361f2cad577da735d828e52cd22ef3998ac81192a9941917df1ab688caaa4b2'
ROOT_SUPPLIED_METADATA_PINS = {
    str(ROOT / '.splash-local-profile.json'): '9455441295ebf60c042f14f49ca9a3d1feeeee5a3e83bb8ced6656f42189b39f',
    str(ROOT / 'build/release/flash/prefill4k-semantic-plan-v1.json'): '3fd2bbccf372dd78378929015db04ea347c22d4394c1eea55f39cb5f59a8400c',
}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def patches():
    helper = str(HERE / 'policy_quality.py')
    return [
        ('    else:\n        explicit.update({"SPLASH_FLASH_MTP": "1", "SPLASH_FLASH_MTP_DRAFT_DEPTH": mode})',
         '    elif mode == "adaptive":\n        if "SPLASH_FLASH_MTP_DRAFT_DEPTH" in explicit:\n            raise ValueError("Adaptive mode requires absent explicit draft depth")\n        explicit.update({"SPLASH_FLASH_MTP": "1", "SPLASH_FLASH_MTP_ADAPTIVE": "1"})\n    else:\n        explicit.update({"SPLASH_FLASH_MTP": "1", "SPLASH_FLASH_MTP_DRAFT_DEPTH": mode})'),
        ('    launcher._apply_local_profile_defaults(clean, defaults)\n    flags =',
         '    launcher._apply_local_profile_defaults(clean, defaults)\n    if mode == "adaptive":\n        clean.pop("SPLASH_FLASH_MTP_DRAFT_DEPTH", None)\n    flags ='),
        ('    return clean, {"removed_inherited_flag_names": removed,',
         '    return clean, {"intentionally_unset_profile_flags": ["SPLASH_FLASH_MTP_DRAFT_DEPTH"] if mode == "adaptive" else [],\n                   "removed_inherited_flag_names": removed,'),
        ('["standard", "off", "1", "2", "3", "4", "7", "8", "15"]',
         '["standard", "off", "1", "2", "3", "4", "7", "8", "15", "adaptive"]'),
        ('mode not in ("standard", "3")', 'mode not in ("standard", "3", "adaptive")'),
        ('mode == "3" else "standard"', 'mode in ("3", "adaptive") else "standard"'),
        ('actual["singleton_maximum_draft_tokens"] != int(mode)',
         'actual["singleton_maximum_draft_tokens"] != (3 if mode == "adaptive" else int(mode))'),
        ('                    run["frontend_prompt_witnesses"] = []',
         '                    from dev.benchmarks.mtp_adaptive_q4_sep22.policy_quality import policy_errors, load as policy_load\n                    normal_runner = policy_load(args.binary.parent, "adaptive" if mode == "adaptive" else "fixed3")\n                    errors = policy_errors(initial, "adaptive" if mode == "adaptive" else "fixed3")\n                    if errors: raise ValueError(errors)\n                    run["frontend_prompt_witnesses"] = []'),
        ('                                    errors.extend(policy["errors"])',
         '                                    errors.extend(policy["errors"])\n                                    errors.extend(policy_errors(before, "adaptive" if mode == "adaptive" else "fixed3"))\n                                    errors.extend(policy_errors(after, "adaptive" if mode == "adaptive" else "fixed3"))\n                                    full_details, full_errors = normal_runner.coverage(before, after, {"prompt_token_count": prompt["prompt_tokens"], "compact_scope": "singleton-main", "body": {"max_completion_tokens": budget}}, True, "mtp3")\n                                    errors.extend(full_errors)\n                                    wave["actual_H3_composite_coverage"] = full_details\n                                    wave["actual_singleton_depth_cycle_deltas"] = [y-x for x,y in zip(get_path(before, "mtp.completed_cycles_by_proposed_depth"), get_path(after, "mtp.completed_cycles_by_proposed_depth"))]'),
        ('quality_command = [sys.executable, str(args.binary.parent / "source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py"),',
         'quality_command = [sys.executable, ' + repr(helper) + ','),
        ('"--expected-rowpair", args.environment_overrides["SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22"], "measure",',
         '"--depth-policy", "adaptive" if mode == "adaptive" else "fixed3", "measure",'),
        ('    plan = prepare_plan(args)',
         '    if args.mtp not in (["3"], ["adaptive"]):\n        raise ValueError("Private adapter supports one exact fixed3/adaptive policy")\n    from dev.benchmarks.mtp_adaptive_q4_sep22.policy_quality import load as adaptive_load\n    adaptive_load(args.binary.parent, "adaptive" if args.mtp == ["adaptive"] else "fixed3")\n    plan = prepare_plan(args)'),
        ('    source_paths.append(CANONICAL)',
         '    source_paths.extend(ROOT / "dev/benchmarks/mtp_adaptive_q4_sep22" / name for name in ("policy_quality.py", "prepare.py", "run.py", "postrun_compare.py"))\n    source_paths.append(CANONICAL)'),
    ]

def transform(text):
    journal = []
    for old, new in patches():
        count = text.count(old)
        if not count: raise ValueError('Qualified driver source fragment unavailable: ' + old[:80])
        text = text.replace(old, new); journal.append({'old':old, 'new':new, 'count':count})
    # This source audit authenticates all inverse edits, not just an opcode census.
    restored = text
    for entry in reversed(journal):
        if restored.count(entry['new']) != entry['count']: raise ValueError('Ambiguous inverse adaptive edit')
        restored = restored.replace(entry['new'], entry['old'])
    if hashlib.sha256(restored.encode()).hexdigest() != DRIVER_SHA: raise ValueError('Adaptive driver inverse differs from literal qualified driver')
    ast.parse(text)
    return text, journal

def derive(base, output_dir, policy, report):
    if policy not in ('fixed3', 'adaptive'): raise ValueError('Unknown policy')
    argv = list(base['argv']); argv[2] = str(output_dir / 'tuning.py')
    argv[argv.index('--output') + 1] = str(report)
    argv[argv.index('--port') + 1] = '8047'
    environment = dict(base['environment'])
    if policy == 'adaptive':
        argv[argv.index('--mtp') + 1] = 'adaptive'
        positions = [i for i,t in enumerate(argv) if t == 'SPLASH_FLASH_MTP_DRAFT_DEPTH=3']
        if len(positions) != 1 or argv[positions[0]-1] != '--env': raise ValueError('Canonical sole depth override differs')
        del argv[positions[0]-1:positions[0]+1]
        environment.pop('SPLASH_FLASH_MTP_DRAFT_DEPTH')
        argv += ['--env','SPLASH_FLASH_MTP_ADAPTIVE=1']
        environment['SPLASH_FLASH_MTP_ADAPTIVE'] = '1'
    return argv, environment

def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'build/mtp-adaptive-Q4-sep22-root-v3')
    args = parser.parse_args(argv); out = args.output_dir.resolve(); out.mkdir(exist_ok=False)
    if sha(DRIVER) != DRIVER_SHA: raise ValueError('Qualified driver drift')
    base = json.loads(BASE.read_text()); text, journal = transform(DRIVER.read_text())
    (out/'tuning.py').write_text(text)
    (out/'driver-edit-journal.json').write_text(json.dumps({'base_sha256':DRIVER_SHA,'edits':journal,'inverse_literal_source_exact':True},indent=2)+'\n')
    paths = set(base['pins']) | {str(p) for p in (BASE,DRIVER,out/'tuning.py',out/'driver-edit-journal.json',HERE/'prepare.py',HERE/'run.py',HERE/'policy_quality.py',HERE/'postrun_compare.py',HERE/'test_adapter.py',ROOT/'dev/benchmarks/raw_q4_verify_worker_sep22/postrun_compare.py')}
    pins = {p:sha(p) for p in sorted(paths)}
    # Root supplied these exact digests. Preparation does not open the plan or
    # tokens; Root's launch runner verifies them before model/tokenizer loading.
    pins.update(ROOT_SUPPLIED_METADATA_PINS)
    for policy in ('fixed3','adaptive'):
        report = ROOT/f'build/release/flash/sep22-Q4-existing-{policy}-2K256-original22-v1.json'
        command, environment = derive(base,out,policy,report)
        config = {'schema':'qualified-Q4-existing-adaptive-depth-command-v1','Root_GPU_only':True,
                  'cwd':str(ROOT),'worker':str(WORKER),'base_command':str(BASE),'base_command_sha256':sha(BASE),
                  'policy':policy,'argv':command,'environment':environment,'report':str(report),'pins':pins,
                  'Root_supplied_external_metadata_pins':ROOT_SUPPLIED_METADATA_PINS,
                  'no_kernel_changes':True,'original22_plan_graders_budgets_unchanged':True}
        path=out/f'root-{policy}-command.json';path.write_text(json.dumps(config,indent=2)+'\n')
        script=out/f'run-root-{policy}.sh'
        script.write_text('#!/bin/sh\nset -eu\nexec '+shlex.join([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'run.py'),str(path),sha(path)])+'\n')
        print(json.dumps({'policy':policy,'command':str(path),'sha256':sha(path),'script':str(script),'GPU_work':False}))
    return 0

if __name__ == '__main__': raise SystemExit(main())
