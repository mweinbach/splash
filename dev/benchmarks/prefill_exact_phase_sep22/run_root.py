#!/usr/bin/env python3
"""Root-owned GPU entry point; each invocation launches only one process."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# Preparation replaces these literals in the frozen runner. Failure handling
# never obtains its publication path from an unchecked command document.
TRUSTED_REPORT = None
TRUSTED_ROLE = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failure(stage, error, returncode=None):
    report = {'schema': 'trunkpref-launch-failure-v1', 'pass': False,
              'qualification_complete': False, 'role': TRUSTED_ROLE,
              'stage': stage, 'error': str(error), 'returncode': returncode,
              'signal': -returncode if returncode is not None and returncode < 0 else None,
              'backend_lifetime': 'see native failure report; unknown if no native report',
              'tensor_payload_bytes_read_by_launcher': 0}
    if TRUSTED_REPORT is None:
        print(json.dumps(report), file=sys.stderr)
        return
    destination = Path(TRUSTED_REPORT + '.launcher-failure.json')
    child_failure = Path(TRUSTED_REPORT + '.failure.json')
    report['native_failure_report_exists'] = child_failure.is_file()
    report['native_failure_report_path'] = str(child_failure)
    try:
        if destination.exists():
            raise RuntimeError('fresh launcher failure path required')
        temporary = destination.with_name(destination.name + '.writing')
        with temporary.open('x') as output:
            json.dump(report, output, indent=2)
            output.write('\n')
        temporary.replace(destination)
    except Exception as publish_error:
        report['failure_publish_error'] = str(publish_error)
        print(json.dumps(report), file=sys.stderr)


def main():
    stage = 'launcher_arguments'
    try:
        if len(sys.argv) != 2:
            raise ValueError('exactly one sealed command path required')
        command_path = Path(sys.argv[1]).resolve()
        stage = 'launcher_command_read'
        command = json.loads(command_path.read_text())
        ready = json.loads((command_path.parent / 'ready-command-seal.json').read_text())
        if digest(command_path) != ready['command_sha256']:
            raise ValueError('sealed command changed')
        if digest(Path(__file__)) != command['runner_sha256']:
            raise ValueError('sealed runner changed')
        stage = 'launcher_artifact_pins'
        for entry in command['artifact_pins']:
            path = Path(entry['path'])
            if digest(path) != entry['sha256']:
                raise ValueError('sealed artifact changed: ' + str(path))
        manifest = json.loads((command_path.parent / 'manifest.json').read_text())
        for name, key in [('oracle', 'oracle_sha256'), ('splash.metallib', 'metallib_sha256')]:
            if digest(command_path.parent / name) != manifest[key]:
                raise ValueError('compiled oracle artifact drift: ' + name)
        stage = 'launcher_role_policy'
        expected_role = 'phase' if command['role'] == 'compare' else 'teacher'
        if command['role'] not in ['export', 'compare'] or manifest['role'] != expected_role:
            raise ValueError('oracle manifest/command role mismatch')
        if command['role'] != TRUSTED_ROLE or command['argv'][-2] != TRUSTED_REPORT:
            raise ValueError('sealed launcher target/command mismatch')
        if command['argv'][1:3] != ['--gpu', command['role']]:
            raise ValueError('oracle argv/command role mismatch')
        phase = command['environment']['SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21']
        allrows = command['environment']['SPLASH_FLASH_ALLROWS_FULL512_TARGET']
        if (phase, allrows) != (('1', '0') if expected_role == 'phase' else ('0', '1')):
            raise ValueError('oracle role/environment mismatch')
        if command['role'] == 'compare':
            stage = 'launcher_export_completed'
            parent = json.loads(Path(command['export_report']).read_text())
            if not parent['pass'] or not parent['backend_destroyed'] or parent['role'] != 'export':
                raise ValueError('successful completed separate-process export required')
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith('SPLASH_FLASH_')}
        environment.update(command['environment'])
        stage = 'launcher_child_launch'
        result = subprocess.run(command['argv'], cwd=command['cwd'], env=environment)
        if result.returncode:
            failure('launcher_child_exit', 'native oracle exited unsuccessfully', result.returncode)
        return result.returncode if result.returncode >= 0 else 128 - result.returncode
    except Exception as error:
        failure(stage, error)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
