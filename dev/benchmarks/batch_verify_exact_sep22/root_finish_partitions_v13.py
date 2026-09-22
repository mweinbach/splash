#!/usr/bin/env python3
"""Root-only serialized continuation of the already pinned V13 campaign."""
import json
import hashlib
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / 'build/release/flash'
BUILD = ROOT / 'build/batchverify-exact-compact-sep22-v13'


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def qualified(data, role, ready, checkpoints):
    require(data['role'] == role, 'report role differs')
    require(data['pass'] and data['partition_complete'] and data['backend_destroyed'], 'incomplete or live report')
    guards = data['allocation']['guards']
    expected_guards = {'target_ledger_matches_workspace', 'target_fits_category_plan',
                       'workspace_plus_state_plan_fits_reservation', 'state_fits_state_plan',
                       'workspace_plus_actual_state_fits_reservation', 'live_backend_delta_fits_reservation'}
    require(set(guards) == expected_guards and all(value is True for value in guards.values()), 'six allocation guards must pass')
    governor = data['allocation']['governor_snapshot']
    require(not governor['denied_reservations'] and governor['growth_allowed'] and governor['host_measurement_valid'], 'governor guard failed')
    checks = data['producer']['checks']
    require((checks['actual_verify_commands'], checks['actual_commit_commands'], checks['invalid_operation_checks']) == (17, 16, 13), 'actual command/control counts differ')
    require(checks['selected_full_checkpoints'] == checkpoints and data['common']['selected_checkpoints'] == checkpoints, 'actual whole checkpoints differ')
    worker = data['producer']['worker']
    require(worker['worker_source_identity_sha256'] == ready['worker_source_identity_sha256'] and
            worker['metallib_sha256'] == ready['metallib_sha256'], 'report source/library identity differs')
    require(0 < data['frames'] and 0 <= data['spill_bytes'] < (4 << 30), 'invalid bounded report')
    require(data['batch_allocation']['partition_spill_preflight_upper_bound'] < (4 << 30), 'partition preflight failed')


def receipt_commands(receipt, ordinal, ready, plan):
    require(receipt['index'] == ordinal and receipt['checkpoints'] == plan['partition_order'][ordinal]['checkpoints'], 'receipt order/checkpoints differ')
    commands = {}
    spill = REPORTS / f'sep22-batchverify-compact-state-p{ordinal:02d}-v2'
    for role in ('export', 'compare'):
        command_path = Path(receipt[role + '_command'])
        script_path = Path(receipt[role + '_script'])
        selected_name = '-'.join(receipt['checkpoints'])
        expected_command = BUILD / f'root-{role}-{selected_name}-command.json'
        require(command_path == expected_command and script_path == expected_command.with_suffix('.sh'), 'unexpected generated command/script path')
        command = load(command_path)
        binary = 'oracle-control' if role == 'export' else 'oracle-candidate'
        report = REPORTS / f'sep22-batchverify-compact-state-p{ordinal:02d}-{role}-v2.json'
        require(Path(receipt[role + '_report']) == report, 'unexpected generated report path')
        require(Path(command['build']) == BUILD and Path(command['cwd']) == ROOT and command['role'] == role and command['Root_GPU_only'], 'command build/cwd/role differs')
        require(command['oracle_sha256'] == ready['artifact_sha256'][binary] and command['metallib_sha256'] == ready['artifact_sha256']['splash.metallib'], 'command artifact pins differ')
        require(command['argv'] == [str(BUILD / binary), '--gpu', role, str(BUILD / 'splash.metallib'),
                                   str(ROOT / 'install/local-models/Flash-Next-oQ4e-mtp-v1'),
                                   str(REPORTS / 'prefill4k-fixture/code2048.tokens.json'),
                                   str(report), str(spill), ','.join(receipt['checkpoints'])], 'command executable/model/input/report/spill/partition differs')
        expected_control = str(Path(receipt['export_report'])) if role == 'compare' else None
        require(command['control_report'] == expected_control, 'command control receipt differs')
        digest = hashlib.sha256(command_path.read_bytes()).hexdigest()
        expected_script = '#!/bin/sh\nset -eu\nexec ' + shlex.quote(str(ROOT / '.venv/bin/python')) + ' -B ' + shlex.quote(str(ROOT / 'dev/benchmarks/batch_verify_exact_sep22/run_root.py')) + ' ' + shlex.quote(str(command_path)) + ' ' + shlex.quote(digest) + '\n'
        require(script_path.read_text() == expected_script, 'script no longer invokes exact sealed command digest')
        commands[role] = command
    return commands, spill


def main():
    require(len(sys.argv) == 2, 'one continuation ordinal required')
    start = int(sys.argv[1])
    require(0 <= start <= 9, 'continuation ordinal outside campaign')
    receipts = load(REPORTS / 'sep22-batchverify-root-partition-receipts-v2.json')
    require(isinstance(receipts, list) and len(receipts) == 10, 'original ten-entry receipt list required')
    ready = load(BUILD / 'CPU_READY.json')
    plan = load(BUILD / 'partition-plan.json')
    prepared = [receipt_commands(receipt, ordinal, ready, plan) for ordinal, receipt in enumerate(receipts)]
    audit_receipts = REPORTS / 'sep22-batchverify-root-partition-audit-input-v2.json'
    audit_output = REPORTS / 'sep22-batchverify-all-state-partitions-audit-v2.json'
    require(not audit_receipts.exists() and not audit_output.exists(), 'fresh audit paths required')
    # A continuation may skip only complete earlier pairs, never missing or
    # failed work. This observes report metadata and cannot restart a role.
    for ordinal, receipt in enumerate(receipts[:start]):
        prior = {role: load(receipt[role + '_report']) for role in ('export', 'compare')}
        for role in prior:
            qualified(prior[role], role, ready, receipt['checkpoints'])
        require(prior['export']['common'] == prior['compare']['common'] and prior['export']['frames'] == prior['compare']['frames'] and
                prior['compare']['bytes_compared'] >= prior['export']['spill_bytes'], f'p{ordinal:02d} earlier pair incomplete')
    for receipt in receipts[start:]:
        ordinal = receipt['index']
        print(f'p{ordinal:02d} starting sequential control/candidate', flush=True)
        data = {}
        for role in ('export', 'compare'):
            report = Path(receipt[role + '_report'])
            require(all(not Path(str(report) + suffix).exists() for suffix in ['', '.partial', '.failure.json', '.writing']), f'fresh role required: {report}')
            subprocess.run(['sh', receipt[role + '_script']], cwd=ROOT, check=True)
            data[role] = load(report)
            qualified(data[role], role, ready, receipt['checkpoints'])
            print(f'p{ordinal:02d} {role} PASS; backend destroyed', flush=True)
        require(data['export']['common'] == data['compare']['common'] and data['export']['frames'] == data['compare']['frames'] and
                data['compare']['bytes_compared'] >= data['export']['spill_bytes'] and data['compare']['planes_compared'] > 0, 'exact completed pair required before spill deletion')
        _, spill = prepared[ordinal]
        require(spill.is_dir() and not spill.is_symlink() and spill.resolve() == spill and spill.parent == REPORTS, 'generated spill directory required')
        manifest = load(spill / 'complete.json')
        require(manifest['complete'] is True and manifest['schema'] == 'batchverify-export-v1' and manifest['common'] == data['export']['common'] and
                manifest['spill_bytes'] == data['export']['spill_bytes'] and len(manifest['frames']) == data['export']['frames'], 'matched completed export manifest required before deletion')
        shutil.rmtree(spill)
        print(f'p{ordinal:02d} exact pair PASS; matched spill removed', flush=True)
    require(not audit_receipts.exists(), 'fresh audit bridge required')
    audit_receipts.write_text(json.dumps({'partitions': receipts}, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'dev/benchmarks/batch_verify_exact_sep22/partition_audit.py'),
                    '--build', str(BUILD), '--receipts', str(audit_receipts),
                    '--output', str(audit_output)],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
