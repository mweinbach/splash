#!/usr/bin/env python3
"""Launch current-profile full-target/teacher-prime attribution; dry-run default."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from install import launcher


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build/prefill4k-attribution')
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--mode', choices=['normal', 'command', 'stage', 'dispatch'], default='normal')
    p.add_argument('--rows', type=int, default=2048)
    p.add_argument('--capacity', type=int, default=8192)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--teacher-prime', type=int, choices=[0, 1], default=1)
    teacher = p.add_mutually_exclusive_group()
    teacher.add_argument('--teacher-cache-only', dest='teacher_cache_only', action='store_const', const=True, default=None)
    teacher.add_argument('--full-teacher-prime', dest='teacher_cache_only', action='store_const', const=False)
    p.add_argument('--teacher-oracle', action='store_true', help='Run strict cache/proposal/rollback qualification')
    p.add_argument('--expert-store', type=Path, help='Explicit verified derivative; overrides qualified Top64 default')
    p.add_argument('--env', action='append', default=[], metavar='SPLASH_FLASH_KEY=VALUE', help='Explicit recorded Flash experiment override')
    p.add_argument('--run', action='store_true', help='Execute GPU inference; root serializes all GPU workloads')
    a = p.parse_args()
    package = ROOT / 'install/local-models/Flash-Next-oQ4e-mtp-v1'
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('SPLASH_FLASH_', 'PREFILL4K_'))}
    defaults = launcher._local_profile_defaults(package)
    profile_flags = len(json.loads((ROOT / '.splash-local-profile.json').read_text())['environment'])
    if len(defaults) != profile_flags + 2 or defaults.get('SPLASH_FLASH_MTP_DRAFT_DEPTH') != '3':
        raise RuntimeError('Current qualified depth-3 profile must resolve to its policy flags plus 2 saved paths')
    environment.update(defaults)
    if a.teacher_cache_only is not None:
        environment["SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY"] = "1" if a.teacher_cache_only else "0"
    overrides = {}
    expert_witness = None
    if a.expert_store is not None:
        store = a.expert_store.resolve(strict=True)
        manifest = json.loads((store / 'manifest.json').read_text())
        if manifest.get('source_identity_sha256') != launcher.LOCAL_PROFILE['source_identity_sha256']:
            raise ValueError('Explicit expert store source differs from current package')
        overrides['SPLASH_FLASH_INT8_EXPERT_STORE'] = str(store)
        expert_witness = {'manifest_sha256': sha(store / 'manifest.json'),
            'source_identity': manifest['source_identity_sha256'], 'plan_sha256': manifest['plan_sha256'],
            'expert_counts_by_layer': [len(row) for row in manifest['selected_experts']],
            'mapped_bytes': manifest['total_bytes']}
    for setting in a.env:
        key, separator, value = setting.partition('=')
        if not separator or not re.fullmatch(r'SPLASH_FLASH_[A-Z0-9_]+', key) or '\0' in value:
            raise ValueError('Experiment override must be a Flash environment KEY=VALUE')
        if key in overrides:
            raise ValueError('Duplicate explicit environment override: ' + key)
        overrides[key] = value
    environment.update(overrides)
    controls = {
        'PREFILL4K_ATTRIBUTION_MODE': a.mode,
        'PREFILL4K_ATTRIBUTION_ROWS': str(a.rows),
        'PREFILL4K_ATTRIBUTION_CAPACITY': str(a.capacity),
        'PREFILL4K_ATTRIBUTION_WARMUP': str(a.warmup),
        'PREFILL4K_ATTRIBUTION_REPEATS': str(a.repeats),
        'PREFILL4K_ATTRIBUTION_TEACHER_PRIME': str(a.teacher_prime),
    }
    environment.update(controls)
    build, tokens, report = a.build.resolve(), a.tokens.resolve(), a.report.resolve()
    binary = build / ('prefill4k-teacher-oracle' if a.teacher_oracle else 'prefill4k-attribution')
    command = [str(binary), str(build / 'splash.metallib'), str(package), str(tokens), str(report)]
    provenance = {
        'schema': 'splash-prefill4k-attribution-invocation-v1', 'gpu_executed': bool(a.run),
        'command': command, 'policy_environment': defaults, 'controls': controls,
        'explicit_environment_overrides': overrides,
        'explicit_expert_store_witness': expert_witness,
        'effective_flash_environment': {key: value for key, value in environment.items() if key.startswith('SPLASH_FLASH_')},
        'teacher_cache_only': environment.get('SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY') == '1',
        'teacher_cache_only_override': a.teacher_cache_only, 'inherited_flash_flags_removed': True, 'profile_sha256': sha(ROOT / '.splash-local-profile.json'),
        'binary_sha256': sha(binary), 'metallib_sha256': sha(build / 'splash.metallib'),
        'tokens_json_sha256': sha(tokens),
        'source_sha256': {str(f.relative_to(ROOT)): sha(f) for f in [
            ROOT / 'dev/benchmarks/prefill4k_attribution.mm', ROOT / 'runtime/flash/FlashForward.cpp',
            ROOT / 'dev/benchmarks/prefill4k_attribution_teacher_oracle.mm',
            ROOT / 'runtime/flash/FlashMTP.cpp', ROOT / 'runtime/flash/FlashWorker.mm',
            ROOT / 'runtime/metal/MetalBackend.mm'] if f.exists()},
        'source_witness_scope': 'current working tree; copied private source is identified separately by overlay manifest',
        'private_overlay_manifest_sha256': sha(build / 'overlay-manifest.json') if (build / 'overlay-manifest.json').is_file() else None,
    }
    witness = report.with_suffix(report.suffix + '.invocation.json')
    if witness.exists():
        raise RuntimeError('Choose a fresh invocation/report output')
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'gpu_executed': bool(a.run), 'invocation': str(witness), 'command': command}))
    if a.run:
        subprocess.run(command, env=environment, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
