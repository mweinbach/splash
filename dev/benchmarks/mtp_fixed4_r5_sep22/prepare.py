#!/usr/bin/env python3
"""CPU source preparation; a runnable command additionally requires Root binding."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from dev.benchmarks.mtp_fixed4_r5_sep22 import policy_quality as p

HERE=Path(__file__).resolve().parent
BASE=p.ROOT/'build/rawQ4-GDN26-matched-model-sep22-root-v2/root-flag1-command.json'
DRIVER=p.PARENT/'source/dev/benchmarks/raw_q4_verify_worker_sep22/tuning.py'
DRIVER_SHA='8361f2cad577da735d828e52cd22ef3998ac81192a9941917df1ab688caaa4b2'

def patches():
    return [
      ('    parser.add_argument("--semantic-plan", type=Path, help="Run frozen semantic qualification in the same loaded process after measured waves")',
       '    parser.add_argument("--r5-binding", type=lambda value: Path(value).resolve(), required=True)\n    parser.add_argument("--r5-binding-sha256", required=True)\n    parser.add_argument("--semantic-plan", type=Path, help="Run frozen semantic qualification in the same loaded process after measured waves")'),
      ('mode not in ("standard", "3")','mode not in ("standard", "3", "4")'),
      ('mode == "3" else "standard"','mode in ("3", "4") else "standard"'),
      ('    raw.load(args.binary.parent, expected=expected == "1", require_state=True)',
       '    from dev.benchmarks.mtp_fixed4_r5_sep22.policy_quality import load as fixed4_load\n    if expected != "1" or args.mtp != ["4"] or args.environment_overrides.get("SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22") != "1":\n        raise ValueError("Private R5 driver requires exact raw1/R5flag1/fixed4")\n    args.fixed4_runner = fixed4_load(args.r5_binding, args.r5_binding_sha256)'),
      ('                    run["frontend_prompt_witnesses"] = []',
       '                    errors = args.fixed4_runner.fixed4_policy_status(initial, None, None, "mtp3")\n                    if errors: raise ValueError(errors)\n                    run["frontend_prompt_witnesses"] = []'),
      ('                    errors = raw_status(run["final_status"], None, None, "mtp3" if mode in ("3", "4") else "standard")',
       '                    errors = raw_status(run["final_status"], None, None, "mtp3")\n                    errors.extend(args.fixed4_runner.fixed4_policy_status(run["final_status"], None, None, "mtp3"))'),
      ('                                    errors.extend(policy["errors"])',
       '                                    errors.extend(policy["errors"])\n                                    full_details, full_errors = args.fixed4_runner.coverage(before, after, {"prompt_token_count": prompt["prompt_tokens"], "compact_scope": "singleton-main", "body": {"max_completion_tokens": budget}}, True, "mtp3")\n                                    errors.extend(full_errors)\n                                    wave["fixed4_full_original_and_R5_coverage"] = full_details'),
      ('quality_command = [sys.executable, str(args.binary.parent / "source/dev/benchmarks/raw_q4_verify_worker_sep22/semantic_quality.py"),',
       'quality_command = [sys.executable, '+repr(str(HERE/'policy_quality.py'))+','),
      ('"--build", str(args.binary.parent), "--expected-rowpair", args.environment_overrides["SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22"], "measure",',
       '"--binding", str(args.r5_binding), "--binding-sha256", args.r5_binding_sha256, "measure",'),
      ('    source_paths.append(CANONICAL)',
       '    source_paths.extend(ROOT / "dev/benchmarks/mtp_fixed4_r5_sep22" / name for name in ("policy_quality.py", "prepare.py", "run.py", "postrun_compare.py", "test_adapter.py"))\n    source_paths.append(args.r5_binding)\n    source_paths.append(CANONICAL)'),
    ]

def transform(text):
    if p.hashlib.sha256(text.encode()).hexdigest()!=DRIVER_SHA:raise ValueError('Literal qualified driver source required')
    journal=[]
    for old,new in patches():
        count=text.count(old)
        if not count:raise ValueError('Driver source anchor unavailable: '+old[:70])
        text=text.replace(old,new);journal.append({'old':old,'new':new,'count':count})
    restored=text
    for edit in reversed(journal):
        if restored.count(edit['new'])!=edit['count']:raise ValueError('Ambiguous inverse driver edit')
        restored=restored.replace(edit['new'],edit['old'])
    if p.hashlib.sha256(restored.encode()).hexdigest()!=DRIVER_SHA:raise ValueError('Driver inverse differs')
    ast.parse(text);return text,journal

def derive(base,out,b,binding,binding_sha,report):
    argv=list(base['argv']);argv[2]=str(out/'tuning.py')
    argv[argv.index('--binary')+1]=str(Path(b['worker'])/'splash-flash')
    argv[argv.index('--output')+1]=str(report);argv[argv.index('--port')+1]='8048'
    if argv[argv.index('--mtp')+1]!='3':raise ValueError('Canonical fixed3 parent command required')
    argv[argv.index('--mtp')+1]='4';positions=[i for i,x in enumerate(argv) if x=='SPLASH_FLASH_MTP_DRAFT_DEPTH=3']
    if len(positions)!=1 or argv[positions[0]-1]!='--env':raise ValueError('Sole canonical depth declaration required')
    argv[positions[0]]='SPLASH_FLASH_MTP_DRAFT_DEPTH=4'
    argv.extend(['--env',p.FLAG+'=1','--r5-binding',str(binding),'--r5-binding-sha256',binding_sha])
    env=dict(base['environment']);env['SPLASH_FLASH_MTP_DRAFT_DEPTH']='4';env[p.FLAG]='1'
    return argv,env

def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--binding',type=Path);parser.add_argument('--binding-sha256')
    args=parser.parse_args(argv);out=args.output_dir.resolve()
    if p.ROOT/'build' not in out.parents:raise ValueError('New private build output required')
    out.mkdir(exist_ok=False);text,journal=transform(DRIVER.read_text())
    (out/'tuning.py').write_text(text)
    (out/'driver-edit-journal.json').write_text(json.dumps({'base_sha256':DRIVER_SHA,'edits':journal,'inverse_literal_source_exact':True},indent=2)+'\n')
    if args.binding is None:
        if args.binding_sha256 is not None:raise ValueError('Binding path required with digest')
        print(json.dumps({'source_only':True,'driver':str(out/'tuning.py'),'GPU_work':False,'runnable_command_prepared':False}));return 0
    b=p.authenticate(args.binding,args.binding_sha256);base=json.loads(BASE.read_text())
    report=p.ROOT/'build/release/flash/sep22-fixed4-R5-2K256-original22-v1.json'
    command,environment=derive(base,out,b,args.binding.resolve(),args.binding_sha256,report)
    paths=set(base['pins'])|{str(x) for x in (BASE,DRIVER,out/'tuning.py',out/'driver-edit-journal.json',args.binding.resolve())}
    paths.update(str(HERE/name) for name in ('policy_quality.py','prepare.py','run.py','postrun_compare.py','test_adapter.py'))
    worker=Path(b['worker']);paths.update(str(worker/name) for name in (*p.FILES,'splash-flash','splash.metallib','R5-qualified.air'))
    pins={path:p.sha(path) for path in sorted(paths)}
    # Root invokes bound preparation; this is not part of source-only preparation.
    plan=p.ROOT/'build/release/flash/prefill4k-semantic-plan-v1.json'
    if p.sha(plan)!=p.PLAN_FILE_SHA:raise ValueError('Original22 preregistered plan file differs')
    pins[str(plan)]=p.PLAN_FILE_SHA;pins[str(p.ROOT/'.splash-local-profile.json')]=p.sha(p.ROOT/'.splash-local-profile.json')
    value={'schema':'Root-fixed4-R5-canonical-command-v1','cwd':str(p.ROOT),'binding':str(args.binding.resolve()),
           'binding_sha256':args.binding_sha256,'base_command':str(BASE),'base_command_sha256':p.sha(BASE),
           'argv':command,'environment':environment,'report':str(report),'pins':pins}
    path=out/'root-command.json';path.write_text(json.dumps(value,indent=2)+'\n')
    script=out/'run-root.sh';script.write_text('#!/bin/sh\nset -eu\nexec '+shlex.join([str(p.ROOT/'.venv/bin/python'),'-B',str(HERE/'run.py'),str(path),p.sha(path)])+'\n')
    print(json.dumps({'command':str(path),'sha256':p.sha(path),'GPU_work':False}));return 0

if __name__=='__main__':raise SystemExit(main())
