#!/usr/bin/env python3
"""Prepare fresh metadata only; no model/fixture/export/report payload access."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--build', required=True)
parser.add_argument('--original-command', required=True)
parser.add_argument('--oracle-sha256', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
build = (root / args.build).resolve()
manifest = json.loads((build / 'manifest.json').read_text())
original_path = (root / args.original_command).resolve()
original = json.loads(original_path.read_text())
if manifest.get('variant') != 'raw-q4q5' or manifest['role'] != 'candidate' or not manifest.get('pass') or original['role'] != 'compare':
    raise SystemExit('fresh CPU-passed rawQ4Q5 candidate and original compare required')
if len(args.oracle_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.oracle_sha256):
    raise SystemExit('Root independent oracle digest required')
command = dict(original)
command.update({
    'schema': 'trunkverify-rawQ4Q5-GDN-current-root-command-v1',
    'Root_GPU_only': True,
    'combined_source_identity_sha256': '90d42b5ce8edd6c7254b6f9286628a7ffa7a6eede53fa3058eba8b519e9a64d6',
    'qualified_worker_executable_sha256': '613e4dfe6a9b429cabf8fbd2270b5857dbc601cde6d1b3af10c46fec1b8095c0',
    'required_rawQ4_calls': 130, 'required_rawQ4_rows': 520,
    'required_rawQ5_calls': 180, 'required_rawQ5_rows': 720,
    'required_bundle_calls': 240, 'required_bundle_rows': 960,
    'required_HC_calls': 485, 'required_HC_rows': 1940,
    'required_HC_padding_dispatches_saved': 485,
    'required_actual_owned_guard_cases': 8,
    'counter_census_before_owned_guard_probes': True,
    'scope': 'new rawQ4Q5 strict flags1 original real26frames54repeats134planes216tapes plus8actual metadata guards; no head/residency/performance proof',
    'oracle_sha256': args.oracle_sha256,
    'metallib_sha256': manifest['metallib_sha256'],
    'inputs_or_export_payload_read_or_hashed': False,
    'original_control_command': str(original_path),
    'environment': dict(original['environment']),
    'argv': list(original['argv']),
})
for flag in (
    'SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22',
    'SPLASH_FLASH_RAW_Q5_ROWPAIR_VERIFY_SEP22',
    'SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22',
    'SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22',
    'SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22',
    'SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22',
):
    command['environment'][flag] = '1'
command['argv'][0] = str(build / 'oracle')
command['argv'][3] = str(build / 'splash.metallib')
report = root / 'build/release/flash/sep22-trunkverify-rawQ4Q5-GDN-current-compare-v1.json'
command['argv'][6] = str(report)
for suffix in ('', '.partial', '.failure.json', '.writing'):
    if Path(str(report) + suffix).exists():
        raise SystemExit('fresh combined report required')
runner = build / 'run-root-rawQ4Q5.py'
shutil.copyfile(Path(__file__).with_name('run_raw_q4q5_root.py'), runner)
command['runner_source_sha256'] = hashlib.sha256(runner.read_bytes()).hexdigest()
path = build / 'Root-rawQ4Q5-native-command.json'
if path.exists():
    raise SystemExit('fresh command registration required')
path.write_text(json.dumps(command, indent=2) + '\n')
digest = hashlib.sha256(path.read_bytes()).hexdigest()
script = build / 'run-root-rawQ4Q5-compare.sh'
script.write_text('#!/bin/sh\nset -eu\nexec ' + shlex.quote(str(root / '.venv/bin/python')) + ' -B ' + shlex.quote(str(runner)) + ' ' + shlex.quote(str(path)) + ' ' + shlex.quote(digest) + '\n')
script.chmod(0o755)
print(json.dumps({'prepared': str(script), 'command': str(path), 'command_metadata_sha256': digest, 'runner_source_sha256': command['runner_source_sha256'], 'report': str(report), 'GPU_started': False, 'input_or_export_payload_reads': 0}))
