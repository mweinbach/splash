#!/usr/bin/env python3
"""Source inspection/preparation ONLY: no compiler, backend, or payload access."""
import ast
import difflib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'build/R5-integer-currentQ4-fixed4-sep22-worker-v2/source'


def main():
    spec = importlib.util.spec_from_file_location('privateR5CaptureOverlay', HERE / 'overlay.py')
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    inspection = (HERE / 'inspection.cpp.inc').read_text()
    journal = []
    for rel in ('runtime/flash/FlashForward.hpp', 'runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm'):
        old = (PARENT / rel).read_text()
        new = overlay.transform(rel, old, inspection)
        before, after = old.splitlines(keepends=True), new.splitlines(keepends=True)
        edits = []
        for tag, i, j, x, y in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
            if tag != 'equal':
                edits.append({'old_start': i, 'old_end': j, 'new_start': x, 'new_end': y, 'old_lines': before[i:j], 'new_lines': after[x:y]})
        restored = list(after)
        for edit in reversed(edits):
            restored[edit['new_start']:edit['new_end']] = edit['old_lines']
        if restored != before:
            raise ValueError('literal inverse differs from current parent')
        journal.append({'path': rel, 'literal_inverse_exact': True, 'edits': edits})
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    witness = {
        'schema': 'current-real-R5-raw-input-capture-SOURCE-preparation-v1',
        'source_preparation_pass': True,
        'ready_for_compile': False, 'ready_for_GPU': False,
        'parent_source': str(PARENT),
        'changed_source_paths': [x['path'] for x in journal],
        'public_production_or_frozen_source_edits': False,
        'private_cloned_Forward_friend_only': True,
        'Forward_fields_or_public_API_changed': False,
        'actual50TU_census_still_required_before_future_build': True,
        'planned_recompile': ['FlashForward', 'FlashWorker', 'FlashBatchForward', 'FlashBatchPrefill', 'FlashBatchVerify'],
        'Core4_and_other_host_objects_must_remain_exact': True,
        'new_shader_compiles': 0,
        'current_original_library_required': 'dc1ab6f9178aac706bb408601fb734e9d508fb5c6c491732bc6ec4e36e6287e6',
        'canonical_role_count': 113, 'observed_shape_formats': 7,
        'capture_logical_bytes': 4362240, 'capture_guarded_aligned_bytes': 5046272,
        'owner_Gov_reservation_bytes': 16 << 20, 'preallocated_host_bytes': 64 << 20,
        'extra_host_headroom_admission_bytes': 128 << 20,
        'proof_snapshots': ['selected_pending', 'selected_commit', 'next_real_target'],
        'third_genuine_R5_after_two_successful_uninstrumented_calls': True,
        'captures_before_RAW_producer_after_existing_F32_NULL_decision': True,
        'existing_copy_shader_only_no_new_FP_math': True,
        'GPU_work': False, 'compiler_invocations': 0,
        'model_tensor_token_profile_or_actual_report_reads_or_hashes': 0,
        'whole_model_worker_compile_requires_Root_source_review': True,
        'whole_state_or_head_or_performance_qualified': False,
        'journal': journal,
    }
    out = HERE / 'SOURCE_PREP.json'
    out.write_text(json.dumps(witness, indent=2) + '\n')
    print(json.dumps({k: v for k, v in witness.items() if k != 'journal'}))


if __name__ == '__main__':
    main()
