#!/usr/bin/env python3
"""Future Root-only command registration after separately reviewed CPU build.

This source is not executed by the SOURCE-only preparation. It reads supplied
command/CPU metadata and program files, never tokens/captures/actual reports.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOKEN_FILE = ROOT / 'build/release/flash/sep22-fixed4-qualified-http-exact2048-Root-tokens-v1.json'
TOKEN_SHA = '72e5a23f0504ba862d43c22e01e820fc4007b3b939ec4cbc8518aaee499832fb'
TOKEN_U32_SHA = '55cf1a355b4a2c97012c752b87955198ef3bb1f1b992b3fb48d35ff7659f3795'
LIBRARY = 'dc1ab6f9178aac706bb408601fb734e9d508fb5c6c491732bc6ec4e36e6287e6'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--Root-reviewed-capture-build', action='store_true')
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--parent-command', type=Path, required=True)
    parser.add_argument('--exe-sha256', required=True)
    parser.add_argument('--cpu-source-review-sha256', required=True)
    parser.add_argument('--command', type=Path, required=True)
    parser.add_argument('--capture-prefix', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(args.Root_reviewed_capture_build, 'Root reviewed CPU capture build required; SOURCE preparation cannot register execution')
    parent = json.loads(args.parent_command.read_text())
    require(parent['library_sha256'] == LIBRARY and parent['context'] == 2048 and parent['output_tokens'] == 64 and parent['request_id'] == parent['generation'] == parent['requests'] == 1, 'exact current parent native command metadata required')
    require(parent['tokens'] == str(TOKEN_FILE) and parent['tokens_file_sha256'] == TOKEN_SHA and parent['tokens_u32le_sha256'] == TOKEN_U32_SHA, 'exact static Root token witnesses required')
    for digest in (args.exe_sha256, args.cpu_source_review_sha256):
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Root independent artifact/review digest required')
    require(args.build.is_absolute() and args.capture_prefix.is_absolute() and args.report.is_absolute() and args.command.is_absolute(), 'absolute private Root paths required')
    require(args.build != ROOT / 'build/R5-integer-currentQ4-fixed4-sep22-worker-v2' and args.exe_sha256 != '6f7e22a2ca9c0c9728bf356391d2bde17c4e9bc90ab6295e625cac15e7987d68', 'fresh instrumented artifact required')
    command = {k: parent[k] for k in ('package', 'environment', 'context', 'output_tokens', 'request_id', 'generation', 'requests')}
    command.update({'schema': 'Root-current-trained-R5-raw-capture-three-frame-command-v1', 'build': str(args.build), 'exe_sha256': args.exe_sha256, 'library_sha256': LIBRARY, 'Root_cpu_source_review_sha256': args.cpu_source_review_sha256, 'parent_command': str(args.parent_command.resolve()), 'tokens': str(TOKEN_FILE), 'tokens_file_sha256': TOKEN_SHA, 'tokens_u32le_sha256': TOKEN_U32_SHA, 'report': str(args.report), 'performance_claim': False})
    command['environment'] = dict(parent['environment'])
    for name in ('SPLASH_FLASH_DIAG_R5_TARGET_STAGE_SEP22',):
        command['environment'][name] = '0'
    command['environment'].pop('SPLASH_FLASH_DIAG_R5_STAGE_OUTPUT_SEP22', None)
    require(command['environment']['SPLASH_FLASH_MTP'] == '1' and command['environment']['SPLASH_FLASH_MTP_DRAFT_DEPTH'] == '4' and command['environment']['SPLASH_FLASH_COMPACT_NATIVE_R5_VERIFY_SEP22'] == '1', 'exact current trained cap4/R5 profile required')
    for suffix in ('', '.failure.json', '.partial', '.writing'):
        require(not Path(str(args.report) + suffix).exists(), 'fresh proof report required')
    command['arms'] = {}
    for name in ('control', 'candidate'):
        directory, log = Path(str(args.capture_prefix) + '-' + name), Path(str(args.capture_prefix) + '-' + name + '.native.log')
        require(not directory.exists() and not log.exists(), 'fresh per-process capture directory/log required')
        command['arms'][name] = {'directory': str(directory), 'native_log': str(log)}
    runner = args.command.with_suffix('.run.py')
    require(not args.command.exists() and not runner.exists(), 'fresh preregistration required')
    shutil.copyfile(HERE / 'run_root.py', runner)
    command['program_pins'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [runner, *sorted(HERE.glob('*.hpp')), HERE / 'inspection.cpp.inc', HERE / 'overlay.py']}
    args.command.write_text(json.dumps(command, indent=2) + '\n')
    digest = hashlib.sha256(args.command.read_bytes()).hexdigest()
    entry = args.command.with_suffix('.run.sh')
    entry.write_text('#!/bin/sh\nset -eu\nexec ' + shlex.quote(str(ROOT / '.venv/bin/python')) + ' -B ' + shlex.quote(str(runner)) + ' --command ' + shlex.quote(str(args.command)) + ' --command-sha256 ' + shlex.quote(digest) + ' --run-root-gpu\n')
    entry.chmod(0o755)
    print(json.dumps({'command': str(args.command), 'command_sha256': digest, 'entry': str(entry), 'GPU_work': False, 'token_model_capture_actual_report_reads_or_hashes': 0}))


if __name__ == '__main__':
    main()
