#!/usr/bin/env python3
"""Root-only bounded all-seven synthetic run, from a sealed copied program."""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path('/Users/mweinbach/Projects/splash')
SCHEMA = 'Root-R5-odd-raw-guard-specialized-synthetic-all7-command-v1'
INPUT_SCOPE = ('synthetic thirteen robust patterns for each of seven observed '
               'shape/quant combinations; NOT current model activations')


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def group_gone(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def validate_command(command):
    build = pathlib.Path(command['build'])
    report = pathlib.Path(command['report'])
    expected = [str(build / 'oracle'), '--gpu', str(build / 'component.metallib'),
                str(ROOT / 'install/local-models/Flash-Next-oQ4e-mtp-v1'),
                str(report), '--shape', 'all', '--synthetic-cases']
    if (command['schema'] != SCHEMA or command['cwd'] != str(ROOT)
            or ROOT / 'build' not in build.parents
            or command['argv'] != expected
            or command['Root_only_GUV_bytes'] != 256 << 20
            or command['selected_native_coefficients_per_case_max_bytes'] != 64 << 20
            or command['input_scope'] != INPUT_SCOPE):
        raise ValueError('Unknown bounded command or canonical scope')
    if command.get('guard_only_specialization') is not True:
        raise ValueError('Bounded guard-only specialization required')
    if command.get('host_accounting_scope') != 'owned-zero; measured-process-cache-bounded':
        raise ValueError('Fresh Host V2 resource accounting is required')
    if any(type(command.get(key)) is not int or command[key] != 1
           for key in ('baseline_dispatches', 'candidate_dispatches')):
        raise ValueError('Inclusive single-dispatch comparison is required')
    if command.get('minimum_warm_GPU_ms_each') != 150 or command.get('balanced_pairs') != 18:
        raise ValueError('Frozen warm and balanced timing protocol is required')
    return expected, report


def verify_pins(command):
    for name, wanted in command['program_pins'].items():
        if sha(name) != wanted:
            raise ValueError('Program/artifact pin changed: ' + name)


def validate_report(report):
    # The pinned oracle performs all numerical/resource gates. This check
    # additionally refuses an incomplete subset as an all-seven completion.
    if (report.get('pass') is not True or report.get('completed') is not True
            or type(report.get('completed_shape_cases')) is not int
            or report['completed_shape_cases'] != 7
            or not isinstance(report.get('cases'), list)
            or len(report['cases']) != 7
            or report.get('worker_integration') is not False
            or report.get('whole_model_qualified') is not False
            or report.get('Gov_budget_bytes') != 256 << 20):
        raise ValueError('Missing successful complete bounded all-seven report')
    teardown = report.get('teardown')
    required = ('backend_destroyed', 'owned_handles_zero', 'sparse_resources_zero',
                'reserved_zero', 'denials_zero', 'backend_healthy_before_destroy',
                'backend_drained_before_destroy', 'sampled_process_cache_within_budget',
                'resource_gate_pass')
    if not isinstance(teardown, dict) or any(teardown.get(key) is not True for key in required):
        raise ValueError('Missing strict Host V2 owned/bounded-process resource proof')


def atomic_fresh(path, value):
    temporary = pathlib.Path(str(path) + '.partial')
    with temporary.open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')
    os.link(temporary, path)
    temporary.unlink()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--command', type=pathlib.Path, required=True)
    parser.add_argument('--command-sha256', required=True)
    parser.add_argument('--run-root-gpu', action='store_true')
    args = parser.parse_args()
    if not args.run_root_gpu or sha(args.command) != args.command_sha256:
        raise ValueError('Explicit Root GPU and external command pin required')
    command = json.loads(args.command.read_text())
    expected, report = validate_command(command)
    verify_pins(command)
    receipt_path = pathlib.Path(str(report) + '.runner.json')
    for suffix in ('', '.partial', '.checkpoint.json', '.failure.json',
                   '.failure.checkpoint.json', '.cases.json', '.cases.json.partial',
                   '.runner.json', '.runner.json.partial'):
        if pathlib.Path(str(report) + suffix).exists():
            raise FileExistsError('Fresh Root report/evidence required')
    lock = (ROOT / 'build/splash-tuning-gpu.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('SPLASH_', 'FLASH_'))}
    try:
        with pathlib.Path(command['log']).open('x') as output:
            process = subprocess.Popen(expected, cwd=ROOT, env=environment,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            return_code = process.wait()
        disappeared = group_gone(process.pid)
        verify_pins(command)
        reason = None
        if return_code == 0 and disappeared:
            try:
                validate_report(json.loads(report.read_text()))
            except (ValueError, OSError) as error:
                reason = str(error)
        elif return_code:
            reason = 'Pinned oracle exited unsuccessfully'
        else:
            reason = 'Native process group remains after parent RC0'
        passed = return_code == 0 and disappeared and reason is None
        receipt = {'schema': 'Root-R5-odd-raw-guard-specialized-process-completion-v1',
                   'pass': passed, 'oracle_return_code': return_code,
                   'process_group_gone_ESRCH': disappeared,
                   'no_SIGKILL_or_restart': True,
                   'command_sha256': args.command_sha256, 'report': str(report),
                   'synthetic_input': True,
                   'no_worker_or_actual_input_qualification': True,
                   'reason': reason}
        atomic_fresh(receipt_path, receipt)
        print(json.dumps(receipt))
        return return_code if return_code else (0 if passed else 1)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
