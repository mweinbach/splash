#!/usr/bin/env python3
"""Metadata-only receipt audit after Root executes every bounded partition.

Receipt JSON is {"partitions":[{"export_report":"...", "compare_report":"...",
"export_command":"...", "compare_command":"..."}, ...]}.
No tensor/export/model payload is opened.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', required=True)
    p.add_argument('--receipts', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = (root / a.build).resolve()
    plan = json.loads((build / 'partition-plan.json').read_text())
    ready = json.loads((build / 'CPU_READY.json').read_text())
    receipts = json.loads((root / a.receipts).read_text())['partitions']
    expected = set(plan['required_checkpoints'])
    covered = set()
    common_reference = None
    total_bytes = total_planes = 0
    records = []
    if len(receipts) != len(plan['partition_order']):
        raise SystemExit('all planned partition receipts required')
    for ordinal, receipt in enumerate(receipts):
        control = json.loads((root / receipt['export_report']).read_text())
        candidate = json.loads((root / receipt['compare_report']).read_text())
        for data, role in [(control, 'export'), (candidate, 'compare')]:
            command = json.loads((root / receipt[role + '_command']).read_text())
            binary = 'oracle-control' if role == 'export' else 'oracle-candidate'
            if Path(command['build']) != build or command['oracle_sha256'] != ready['artifact_sha256'][binary]:
                raise SystemExit('receipt command does not pin this exact private oracle')
            if Path(command['argv'][0]) != build / binary or Path(command['argv'][6]) != (root / receipt[role + '_report']).resolve():
                raise SystemExit('receipt command executable/report differs')
            if command['argv'][8].split(',') != data['common']['selected_checkpoints']:
                raise SystemExit('receipt command partition differs')
            if not data['pass'] or not data['partition_complete'] or not data['backend_destroyed'] or data['role'] != role:
                raise SystemExit('incomplete separate-process receipt')
            if data['trained_head_proved'] or data['worker_cancel_deadline_proved'] or data['foreign_trunk_runtime_proved']:
                raise SystemExit('unsupported wider proof claim')
            worker = data['producer']['worker']
            if worker['worker_source_identity_sha256'] != ready['worker_source_identity_sha256']:
                raise SystemExit('receipt worker source differs')
            if worker['metallib_sha256'] != ready['metallib_sha256']:
                raise SystemExit('receipt library differs')
            if not 0 < data['frames'] or not 0 <= data['spill_bytes'] < 4 << 30:
                raise SystemExit('invalid bounded frame receipt')
            if not data['batch_allocation']['partition_spill_preflight_upper_bound'] < 4 << 30:
                raise SystemExit('partition preflight did not pass hard bound')
            checks = data['producer']['checks']
            if checks['actual_verify_commands'] != 17 or checks['actual_commit_commands'] != 16 or checks['invalid_operation_checks'] != 13:
                raise SystemExit('actual command/control campaign incomplete')
            if checks['selected_full_checkpoints'] != data['common']['selected_checkpoints']:
                raise SystemExit('selected whole checkpoint missing from actual campaign')
            governor = data['allocation']['governor_snapshot']
            if governor['denied_reservations'] or not governor['host_measurement_valid'] or not governor['growth_allowed']:
                raise SystemExit('governor guard failed')
        if control['common'] != candidate['common'] or control['frames'] != candidate['frames']:
            raise SystemExit('control/candidate partition identity differs')
        if not candidate['bytes_compared'] >= control['spill_bytes'] or not candidate['planes_compared'] > 0:
            raise SystemExit('no complete actual byte comparisons')
        selected = candidate['common']['selected_checkpoints']
        if selected != plan['partition_order'][ordinal]['checkpoints']:
            raise SystemExit('receipt partition order differs from prepared plan')
        if any(x in covered or x not in expected for x in selected):
            raise SystemExit('duplicate/unexpected full checkpoint')
        covered.update(selected)
        numeric_common = {k: v for k, v in candidate['common'].items() if k != 'selected_checkpoints'}
        if common_reference is None:
            common_reference = numeric_common
        elif common_reference != numeric_common:
            raise SystemExit('model/numeric policy differs across partitions')
        total_bytes += candidate['bytes_compared']
        total_planes += candidate['planes_compared']
        records.append({'index': ordinal, 'checkpoints': selected, 'export_report': receipt['export_report'],
                        'compare_report': receipt['compare_report'], 'bytes_compared': candidate['bytes_compared']})
    if covered != expected:
        raise SystemExit('missing required whole checkpoints')
    result = {'schema': 'batchverify-all-bounded-partitions-metadata-audit-v1', 'pass': True,
              'all_planned_whole_state_checkpoints_qualified': True, 'partitions': records,
              'full_checkpoints': len(covered), 'bytes_compared': total_bytes, 'planes_compared': total_planes,
              'head_or_worker_lifecycle_or_quality_or_performance_qualified': False,
              'foreign_trunk_runtime_proved': False, 'model_or_tensor_payload_reads': 0,
              'common': common_reference, 'build': str(build)}
    out = (root / a.output).resolve()
    if out.exists():
        raise SystemExit('fresh audit output required')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['partitions', 'common']}))


if __name__ == '__main__':
    main()
