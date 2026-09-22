#!/usr/bin/env python3
"""CPU-only closure/readonly-hook witness. Never opens runtime/model payloads."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    build = (root / a.build).resolve()
    ready = json.loads((build / 'CPU_READY.json').read_text())
    worker = Path(ready['worker'])
    overlay = json.loads((worker / 'overlay-manifest.json').read_text())
    checks = []
    def check(name, value):
        checks.append({'name': name, 'pass': bool(value)})
        if not value:
            raise RuntimeError(name)
    check('complete_nonworker_object_census', len(ready['objects']) == 53)
    check('complete_actual_source_TU_census', len(ready['header_census']) == 50)
    check('actual_modified_header_consumers_rebuilt', all(x['excluded_worker_main'] or not x['header_consumer']
        or x['object'] in ready['rebuilt_header_consumers'] for x in ready['header_census']))
    for item in overlay['files']:
        path = worker / 'source' / item['path']
        check('sealed_parent_source:' + item['path'], path.is_file() and sha(path) == item['sha256'])
    for item in ready['objects']:
        check('compiled_artifact:' + Path(item['path']).name, sha(Path(item['path'])) == item['sha256'])
        if not item['recompiled_header_consumer']:
            check('unaffected_object_inherited_literal:' + Path(item['path']).name,
                  sha(Path(item['path'])) == item['inherited_sha256'])
    for item in ready['headers']:
        check('private_inspection_pin:' + Path(item['path']).name, sha(Path(item['path'])) == item['sha256'])
    for name, digest in ready['artifact_sha256'].items():
        check('linked_artifact:' + name, sha(build / name) == digest)
    modified = {'runtime/flash/FlashForward.cpp', 'runtime/flash/FlashForward.hpp',
                'runtime/flash/FlashBatchVerify.cpp', 'runtime/flash/FlashBatchVerify.hpp', 'runtime/metal/MetalBackend.hpp'}
    for item in overlay['files']:
        relative = item['path']
        if relative in modified:
            continue
        check('unmodified_candidate_source_literal:' + relative,
              sha(build / 'source' / relative) == item['sha256'])
    private = build / 'source/dev/benchmarks/batch_verify_exact_sep22'
    for relative, inc in [('runtime/flash/FlashForward.cpp', 'forward_inspection.cpp.inc'),
                          ('runtime/flash/FlashBatchVerify.cpp', 'batch_inspection.cpp.inc')]:
        original = (worker / 'source' / relative).read_text()
        expected = '#include "inspect.hpp"\n' + original + '\n' + (private / inc).read_text()
        check('original_method_bodies_literal:' + relative, (build / 'source' / relative).read_text() == expected)
    original = (worker / 'source/runtime/flash/FlashForward.hpp').read_text()
    before, after = original.rsplit('  friend class FlashBatchForward;', 1)
    expected = before + '  friend class FlashDeepPrefixOracle;\n  friend class FlashBatchForward;' + after
    check('Forward_header_only_friendship', (build / 'source/runtime/flash/FlashForward.hpp').read_text() == expected)
    original = (worker / 'source/runtime/flash/FlashBatchVerify.hpp').read_text()
    expected = original.replace('private:\n  struct Impl;', 'private:\n  friend class FlashDeepPrefixOracle;\n  struct Impl;')
    check('Batch_header_only_friendship', (build / 'source/runtime/flash/FlashBatchVerify.hpp').read_text() == expected)
    original = (worker / 'source/runtime/metal/MetalBackend.hpp').read_text()
    anchor = '  [[nodiscard]] uint64_t sizeBytes() const noexcept;'
    expected = original.replace(anchor, anchor + '\n  // Clone-only metadata: native owner charge and opaque owner identity.\n'
        + '  [[nodiscard]] uint64_t oracleChargedBytesSep22() const noexcept;\n'
        + '  [[nodiscard]] uintptr_t oracleOwnerIdentitySep22() const noexcept;', 1)
    check('MetalBuffer_header_adds_only_const_metadata_methods', (build / 'source/runtime/metal/MetalBackend.hpp').read_text() == expected)
    for pin in ready['core_source_pins']:
        original = (root / pin['source']).read_text()
        check('clean_original_Core_pin:' + pin['source'], sha(root / pin['source']) == pin['original_sha256'])
        actual = (build / 'source' / pin['source']).read_text()
        if pin['source'].endswith('MetalBackend.mm'):
            check('all_original_Core_methods_literal', actual.startswith(original))
            check('charge_getter_reads_actual_native_owner_field', 'return impl_ && impl_->allocation ? impl_->allocation->bytes : 0;' in actual[len(original):])
            check('owner_getter_reads_original_shared_allocation_identity', 'reinterpret_cast<uintptr_t>(impl_->allocation.get())' in actual[len(original):])
        else:
            check('unmodified_original_Core_source:' + pin['source'], actual == original)
    check('all_actual_Core_header_consumers_rebuilt', all(not x['header_consumer'] or any(Path(o['path']).stem.endswith(x['object'])
        and o['recompiled_header_consumer'] for o in ready['objects']) for x in ready['core_header_census']))
    inspection = (private / 'batch_inspection.cpp.inc').read_text()
    oracle = (private / 'oracle.mm').read_text()
    for value in ['b.jobCapacity!=531', '223ULL*2560*2', '223ULL*640*2', '160ULL*2560*2',
                  'r.maximumRows()!=4||r.maximumLanes()!=4', 'bytes!=s.allocatedBytes+guardDelta', 'owners.insert(owner).second']:
        check('required_physical_inventory:' + value, value in inspection)
    for value in ['verifyCommands_==17&&commitCommands_==16&&invalidChecks_==13',
                  'expired lane with nonzero retained', 'moved source verify', 'moved source commit',
                  'governor_.tryReserve(allowance)', 'stateDelta==batchAllocation_.stateAllowance',
                  'bytes<=kLimit-spilled_', 'completed_==frames_.size()']:
        check('required_runtime_gate_present:' + value, value in oracle)
    for value in ['original_storage\\\":\\\"Private', 'p.buffer.sizeBytes()==31232', 'mirrorPlanned=32768',
                  'flash_forward_copy_words', 'ticket.wait()', 'oracleChargedBytesSep22()', 'batchGuardPlannedDelta(batch_)']:
        check('Private_copy_and_charge_gate_present:' + value, value in oracle)
    check('two_role_synthetic_stream_checks_pass', ready['cpu_control']['pass'] and ready['cpu_candidate']['pass']
          and ready['cpu_control']['stream_mutation_rejected'] and ready['cpu_candidate']['stream_mutation_rejected'])
    result = {'schema': 'batchverify-readonly-clone-cpu-source-audit-v1', 'pass': True,
              'checks': len(checks), 'detail': checks, 'gpu_executed': False,
              'model_or_fixture_or_export_or_capture_payload_reads': 0,
              'full_model_state_or_quality_or_performance_qualified': False,
              'not_an_independent_review': True, 'build': str(build)}
    out = (root / a.output).resolve()
    if out.exists():
        raise SystemExit('fresh audit output required')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'detail'}))


if __name__ == '__main__':
    main()
