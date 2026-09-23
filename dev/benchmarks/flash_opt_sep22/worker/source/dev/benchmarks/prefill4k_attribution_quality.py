#!/usr/bin/env python3
"""Freeze, root-run, or audit 22 long-prompt semantic checks of expert inventories.

Planning/comparison use CPU only. Measurement never starts or changes a server.
Generation differences are diagnostics; exact task outcomes determine regressions.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.benchmarks import qualify_flash_http as http
from dev.benchmarks import prefill_decode_phase_quality as phase_quality
from dev.benchmarks import singleton_teacher_bulk_quality as teacher_quality
from dev.benchmarks.flash_precision_quality import tokens_for

SOURCE = 'ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e'
MODEL = 'local/Qwen3.8-Flash-Next-oQ4e-mtp'
TOKENIZER = Path.home() / '.omlx/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp'
NONCE = 'prefill4k-semantic-20260921-v1'
SCHEMA = 'splash-prefill4k-semantic-plan-v1'
EXECUTION_POLICY_FIELDS = ['maximum_context_tokens', 'scheduler.maximum_prefill_rows',
    'scheduler.maximum_batch_prefill_rows_per_lane', 'mtp.singleton_maximum_draft_tokens',
    'mtp.teacher_cache_only_requested', 'ple_storage.ssd_streaming_enabled',
    'idle_residency_maintenance.requested']
STANDARD_MTP_REQUIREMENTS = {
    'capabilities.mtp': False, 'capabilities.batch_mtp': False,
    'capabilities.batch_mtp_prefill': False, 'mtp.enabled': False,
    'mtp.maximum_draft_tokens': 0, 'mtp.singleton_maximum_draft_tokens': 0,
    'mtp.singleton_concurrent_draft_cap': 0, 'mtp.joint_maximum_draft_tokens': 0,
    'mtp.teacher_cache_only_requested': False, 'mtp.depth_controller_semantics': None,
    'mtp.joint_policy': None, 'identity.mtp_semantics': None,
    'identity.mtp_teacher_priming_route': None, 'identity.mtp_attention_route': None,
    'identity.joint_head_semantics': None, 'identity.joint_head_attention_route': None,
    'identity.joint_verifier_semantics': None,
    'identity.joint_head_vocabulary_register': False, 'identity.joint_head_vocabulary_route': None,
    'identity.batch_mtp_prefill_semantics': None, 'identity.batch_mtp_prefill_attention_route': None,
}
STANDARD_MTP_ZERO_COUNTERS = [
    'teacher_cache_only_priming_calls', 'eligible_requests', 'verification_cycles',
    'drafted_tokens', 'matched_proposals', 'accepted_committed_drafts',
    'emitted_accepted_proposals', 'singleton_committed_fold_calls',
    'joint_cohorts_attempted', 'joint_partial_cohorts_before_target',
    'joint_members_dropped_before_commit', 'joint_borrowed_hidden_copied_bytes',
    'joint_head_vocabulary_register_commands', 'joint_head_vocabulary_register_rows',
    'permanent_batch_fallback_requests',
]
STANDARD_MTP_ZERO_ARRAYS = {'accepted_prefix_histogram': 16,
    'completed_cycles_by_proposed_depth': 16, 'joint_head_commands_by_width': 4,
    'joint_target_verifiers_by_width': 4}
STANDARD_MTP_TIMING_GROUPS = ['head_priming', 'head_decode', 'target_verify',
    'prefix_restore', 'batched_head_priming_subset']


def standard_mtp_errors(status):
    """Require an absent head and zero completed MTP work, not just cap zero."""
    requirements = dict(STANDARD_MTP_REQUIREMENTS)
    requirements.update({'mtp.' + key: 0 for key in STANDARD_MTP_ZERO_COUNTERS})
    requirements.update({'mtp.' + key: [0] * count for key, count in STANDARD_MTP_ZERO_ARRAYS.items()})
    requirements.update({'metrics.drafted_tokens': 0, 'metrics.accepted_draft_tokens': 0})
    for group in STANDARD_MTP_TIMING_GROUPS:
        for key in ['forward_host_wall_ms', 'last_gpu_ms', 'last_wall_ms', 'total_gpu_ms', 'total_wall_ms']:
            requirements[f'mtp.{group}.{key}'] = 0
        requirements[f'mtp.{group}.host_command_subphases.timed_commands'] = 0
    errors = []
    for key, expected in requirements.items():
        actual = status
        for part in key.split('.'):
            if not isinstance(actual, dict) or part not in actual:
                actual = object()  # Missing fields do not prove an explicit null head.
                break
            actual = actual[part]
        # Timing durations may serialize as either integer zero or float zero.
        if key.endswith('_ms') and expected == 0 and type(expected) is int:
            matches = type(actual) in [int, float] and math.isfinite(actual) and actual == 0
        else:
            matches = http.same_json(actual, expected)
        if not matches:
            errors.append('Standard execution requires absent/unused MTP: ' + key)
    def unused_timing(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                unused_timing(item, path + '.' + key)
        elif type(value) in [int, float] and (not math.isfinite(value) or value != 0):
            errors.append('Standard execution unexpectedly recorded MTP timing: ' + path)
    for group in STANDARD_MTP_TIMING_GROUPS:
        unused_timing(http.get_path(status, 'mtp.' + group), 'mtp.' + group)
    return errors


def execution_policy(status):
    policy = {key: http.get_path(status, key) for key in EXECUTION_POLICY_FIELDS}
    if phase_quality.phase_present(status):
        for key in ['identity.target_phase_policy_schema', 'identity.target_phase_policy']:
            policy[key] = http.get_path(status, key)
    return policy


def ownership_policy(status):
    # Ownership/residency is intentionally separate from fixed sampling and
    # prompt execution policy when testing a numerical target derivative.
    fields = ['identity.target_all_rows_full512', 'identity.original_target_gpu_omitted',
        'saved_operands_residency.requested', 'saved_operands_residency.request_succeeded',
        'original_text_residency.requested', 'original_text_residency.added_to_union',
        'ple_storage.gpu_mapped_original_bytes', 'persisted_experts.expert_count',
        'persisted_experts.mapped_bytes']
    if phase_quality.phase_present(status):
        fields += ['identity.target_hybrid_phase', *phase_quality.F32_OWNERSHIP_FIELDS]
    if teacher_quality.teacher_bulk_present(status):
        fields += list(teacher_quality.OWNERSHIP_FIELDS)
    return {key: http.get_path(status, key) for key in fields}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')


def checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def strict_numbers(value):
    if isinstance(value, float) and (not math.isfinite(value) or 0 < abs(value) < 1e-100):
        raise ValueError('Nonfinite or implausibly tiny numeric metadata')
    elif isinstance(value, dict):
        for item in value.values():
            strict_numbers(item)
    elif isinstance(value, list):
        for item in value:
            strict_numbers(item)


def spec(identifier, group, question, expected, rows=2048, budget=64, payload='', position='middle', schema=None):
    options = {'stream': True, 'stream_options': {'include_usage': True}}
    if schema:
        options['response_format'] = {'type': 'json_schema', 'json_schema': {'name': identifier, 'strict': True, 'schema': schema}}
    return {'id': identifier, 'group': group, 'target_prompt_tokens': rows, 'question': question,
        'payload': payload, 'payload_position': position, 'expected': expected,
        'body': http.chat_body(MODEL, [{'role': 'user', 'content': ''}], budget, **options)}


def specs():
    rows = []
    for identifier, question, answer in [
        ('multiply', 'Calculate 137 * 29. Reply with only the integer answer.', str(137 * 29)),
        ('signed', 'Calculate (-37 * 19) + (144 / 12). Reply with only the integer answer.', str(-37 * 19 + 12)),
        ('parentheses', 'Calculate 5 * (83 - 17) - 24. Reply with only the integer answer.', str(5 * (83 - 17) - 24)),
        ('inventory', 'Start with 187 units, remove 9 groups of 13 units, then add 26 units. How many remain? Reply with only the integer answer.', str(187 - 9 * 13 + 26)),
    ]:
        rows.append(spec('arithmetic_' + identifier, 'arithmetic', question, {'kind': 'text', 'value': answer}))
    rows += [
        spec('extract_shipment', 'extraction_logic', 'In the RELEVANT LEDGER, return only the shipment_id whose status is queued. Match the record exactly.',
            {'kind': 'text', 'value': 'RX-407'}, payload='RELEVANT LEDGER\nshipment_id=RX-101; units=9; status=sent\nshipment_id=RX-407; units=23; status=queued\nshipment_id=RX-990; units=6; status=sent', position='early'),
        spec('extract_latest_balance', 'extraction_logic', 'In the RELEVANT LEDGER, find account A and return only its balance from the row with the highest version. Return one signed integer.',
            {'kind': 'text', 'value': '-6'}, payload='RELEVANT LEDGER\naccount=A; version=1; balance=17\naccount=B; version=8; balance=400\naccount=A; version=3; balance=-6\naccount=A; version=2; balance=93', position='late'),
        spec('logic_order', 'extraction_logic', 'Ada is before Bo. Cy is after Bo. Return only a JSON array listing these three names in the implied order.',
            {'kind': 'json', 'value': ['Ada', 'Bo', 'Cy']}),
        spec('logic_implication', 'extraction_logic', 'For this fictional problem, every rock is blue and every blue thing is round. Does it necessarily follow that every rock is round? Reply with only yes or no.',
            {'kind': 'text', 'value': 'yes'}),
    ]
    rows += [
        spec('json_arithmetic_schema', 'json', 'Return exactly a JSON object with sum=6+9, difference=6-9, and ok=true if sum is greater than 10. Include only these keys.',
            {'kind': 'json', 'value': {'sum': 15, 'difference': -3, 'ok': True}}, schema={
                'type': 'object', 'properties': {'sum': {'type': 'integer'}, 'difference': {'type': 'integer'}, 'ok': {'type': 'boolean'}},
                'required': ['sum', 'difference', 'ok'], 'additionalProperties': False}, budget=128),
        spec('json_counts_unconstrained', 'json', 'Given records blue=7, green=-2, blue=4, return only a JSON object mapping each color to the sum of its values. Include exactly blue and green.',
            {'kind': 'json', 'value': {'blue': 11, 'green': -2}}, budget=128),
        spec('json_filter_schema', 'json', 'For [-3, 0, 5, 8, -2], return a JSON object with even=the even values in original order and count=their number. Include only those keys.',
            {'kind': 'json', 'value': {'even': [0, 8, -2], 'count': 3}}, schema={
                'type': 'object', 'properties': {'even': {'type': 'array', 'items': {'type': 'integer'}}, 'count': {'type': 'integer'}},
                'required': ['even', 'count'], 'additionalProperties': False}, budget=128),
        spec('json_nested_copy', 'json', 'Return only the following JSON value, preserving every key, type, value and array order: {"meta":{"ticket":"AX-105","active":false},"values":[3,1,3],"checksum":7}. Do not add Markdown.',
            {'kind': 'json', 'value': {'meta': {'ticket': 'AX-105', 'active': False}, 'values': [3, 1, 3], 'checksum': 7}}, rows=2049, budget=128),
    ]
    code_rules = ('Write only Python code defining exactly the requested function. No imports, attributes (including .append), dunder names, '
        'helper functions, annotations, decorators, defaults, I/O or explanations. Use operators, loops or comprehensions and pure builtins only. ')
    code_tasks = [
        ('sum_even', ['numbers'], 'sum_even(numbers): return the sum of even integers, including negative even integers. Empty input returns 0.', [
            ([[]], 0), ([[1, 2, 3, 4]], 6), ([[-6, -3, 0, 8]], 2), ([[2, 2, 2]], 6), ([list(range(-100, 101))], 0)]),
        ('clamp_values', ['numbers', 'low', 'high'], 'clamp_values(numbers, low, high): return a new list with each integer restricted to the inclusive interval [low, high]. Assume low<=high. Preserve order; empty input returns [].', [
            ([[], -1, 1], []), ([[-4, -1, 0, 2, 9], -1, 2], [-1, -1, 0, 2, 2]), ([[7, 7, -7], 3, 3], [3, 3, 3]), ([[0, 5, 10], 0, 10], [0, 5, 10]), ([list(range(-20, 21)), -3, 4], [max(-3, min(4, n)) for n in range(-20, 21)])]),
        ('prefix_sums', ['numbers'], 'prefix_sums(numbers): return a list of inclusive prefix sums in the original order. For [2,-1,4] return [2,1,5]. Empty input returns [].', [
            ([[]], []), ([[2, -1, 4]], [2, 1, 5]), ([[0, 0, 0]], [0, 0, 0]), ([[-3, -2, 7, -4]], [-3, -5, 2, -2]), ([list(range(1, 65))], [i * (i + 1) // 2 for i in range(1, 65)])]),
        ('longest_run', ['numbers'], 'longest_run(numbers): return the length of the longest contiguous run of equal values. Empty input returns 0. For [1,1,2,2,2,1] return 3.', [
            ([[]], 0), ([[1]], 1), ([[1, 1, 2, 2, 2, 1]], 3), ([[-2, -2, -2, -2, 7, 7]], 4), ([[i % 2 for i in range(128)]], 1), ([[5] * 128], 128)]),
        ('merge_counts', ['pairs'], 'merge_counts(pairs): pairs is a list of [string_key, integer_amount]. Return a dictionary mapping each distinct key to the sum of its amounts. Preserve negative amounts; empty input returns {}.', [
            ([[]], {}), ([[['a', 2], ['b', 7], ['a', -5]]], {'a': -3, 'b': 7}), ([[['x', 0], ['x', 0]]], {'x': 0}), ([[['b', -4], ['a', 9], ['b', 4], ['c', -2]]], {'a': 9, 'b': 0, 'c': -2})]),
    ]
    for name, arguments, question, tests in code_tasks:
        rows.append(spec('python_' + name, 'python', code_rules + question,
            {'kind': 'python', 'function': name, 'argument_names': arguments, 'tests': [{'arguments': a, 'output': value} for a, value in tests]}, budget=512))
    copy_payloads = [
        ('copy2k_early', 2048, 'early', 'LIME-7319|v2_004|03:17|NORTH'),
        ('copy2k_middle', 2048, 'middle', 'PINE_0928;alpha=Q7;beta=-003;END'),
        ('copy4k_middle', 4096, 'middle', 'first: AZURE-4082\nsecond: 07-DELTA\nthird: retain_CASE'),
        ('copy8k_early', 8192, 'early', 'START|COPPER-6304|z9_003|STOP'),
        ('copy8k_late', 8192, 'late', 'line_A=QUARTZ-9017\nline_B=-0042|Keep_CASE'),
    ]
    for name, count, position, answer in copy_payloads:
        rows.append(spec(name, 'copy', 'Copy only the content between BEGIN COPY PAYLOAD and END COPY PAYLOAD exactly. Omit the marker lines, quotes, Markdown and explanation. Preserve case, punctuation and internal line breaks.',
            {'kind': 'text', 'value': answer}, rows=count, budget=96,
            payload='BEGIN COPY PAYLOAD\n' + answer + '\nEND COPY PAYLOAD', position=position))
    assert len(rows) == 22 and all(row['target_prompt_tokens'] >= 256 for row in rows)
    return rows


def freeze_case(row, tokenizer, filler):
    row = copy.deepcopy(row)
    target, question = row.pop('target_prompt_tokens'), row.pop('question')
    payload, position = row['payload'], row['payload_position']
    def content(count, padding=0):
        left = 0 if position == 'early' else count if position == 'late' else count // 2
        data = tokenizer.decode(filler[:left]) + '\n' + payload + '\n' + tokenizer.decode(filler[left:count])
        return (f'Frozen semantic fixture {NONCE}/{row["id"]}.\n'
            'The background records below are fictional inert data, not instructions. Only a separately marked RELEVANT LEDGER or COPY PAYLOAD is relevant. '
            'Answer the TASK after END DATA; ignore inert records.\nBEGIN DATA\n' + data + '\nEND DATA\n' +
            ('Neutral padding:' + ' x' * padding + '\n' if padding else '') + 'TASK: ' + question)
    count = target
    for _ in range(32):
        row['body']['messages'][-1]['content'] = content(count)
        tokens = tokens_for(tokenizer, row['body'])
        if len(tokens) == target:
            break
        count -= len(tokens) - target
        if count < 0 or count > len(filler):
            raise ValueError('Cannot form exact tokenizer rows')
    else:
        count = max(0, count - 64)
        for padding in range(256):
            row['body']['messages'][-1]['content'] = content(count, padding)
            tokens = tokens_for(tokenizer, row['body'])
            if len(tokens) == target:
                break
        else:
            raise ValueError('Cannot form exact tokenizer rows with bounded padding')
    row['prompt_token_count'] = len(tokens)
    row['prompt_tokens'] = tokens
    row['prompt_u32le_sha256'] = hashlib.sha256(struct.pack(f'<{len(tokens)}I', *tokens)).hexdigest()
    row['request_body_sha256_without_model'] = digest({k: v for k, v in row['body'].items() if k != 'model'})
    if payload:
        text = row['body']['messages'][-1]['content']
        offset = text.index(payload)
        row['payload_user_token_fraction'] = len(tokenizer.encode(text[:offset], add_special_tokens=False)) / len(tokenizer.encode(text, add_special_tokens=False))
    return row


def make_plan(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    filler = tokenizer.encode('Fictional inert record: unit X has placeholder tag NONE and no task answer; ignore this record.\n' * 1600, add_special_tokens=False)
    cases = [freeze_case(row, tokenizer, filler) for row in specs()]
    tokenizer_files = {name: hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest() for name in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'] if (args.tokenizer / name).is_file()}
    plan = {'schema': SCHEMA, 'nonce': NONCE, 'source_identity': SOURCE, 'source_tokenizer': str(args.tokenizer.resolve()),
        'tokenizer_file_sha256': tokenizer_files, 'renderer': 'actual frontend messages/tools/schema normalization + thinking disabled',
        'case_count': len(cases), 'cases': cases, 'minimum_prompt_tokens': min(case['prompt_token_count'] for case in cases),
        'minimum_context_capacity_tokens': max(case['prompt_token_count'] + case['body']['max_completion_tokens'] for case in cases),
        'common_policy': {'mtp_depth': 3, 'teacher_cache_only': True, 'ple_ssd_streaming': True, 'idle_maintenance': False,
            'temperature': 0, 'reasoning_effort': 'none', 'seed': 0, 'prefix_reuse': 0, 'request_concurrency': 1},
        'quality_scope': '22 fixed synthetic task outcomes; not general model quality, likelihood or numerical source certification',
        'python_scope': 'restricted pure-function fixture evaluator with AST validation and process/resource limits; not general code execution',
        'numeric_source_quality': {'separate_required': True, 'reason': 'Coefficient error certificate and GPU numerical guard evidence are independent of generated task success'},
        'source_logprobs_available': False, 'lifecycle_cases': 'Existing eight-case service suite is required separately'}
    plan['content_sha256'] = digest(plan)
    write_new(args.output, plan)
    print(json.dumps({'plan': str(args.output), 'cases': len(cases), 'prompt_rows': [case['prompt_token_count'] for case in cases], 'content_sha256': plan['content_sha256'], 'gpu_executed': False}))


def read_plan(path):
    plan = http.strict_json(path.read_text())
    old = plan.pop('content_sha256', None)
    if old != digest(plan) or plan.get('schema') != SCHEMA or plan.get('source_identity') != SOURCE or len(plan.get('cases', [])) != 22:
        raise ValueError('Frozen plan identity/hash/cardinality changed')
    plan['content_sha256'] = old
    for case in plan['cases']:
        if case['prompt_token_count'] < 256 or case['prompt_token_count'] != len(case['prompt_tokens']):
            raise ValueError('Plan could qualify only small-row fallback')
    return plan


def store_witness(path, inventory):
    raw = (path / 'manifest.json').read_bytes()
    manifest = http.strict_json(raw)
    ids = manifest['selected_experts']
    if manifest['source_identity_sha256'] != SOURCE or manifest['target_layers'] != 48 or len(ids) != 48 or any(len(row) != inventory or row != sorted(set(row)) for row in ids):
        raise ValueError('Expected source/uniform expert inventory changed')
    if inventory == 512 and any(row != list(range(512)) for row in ids):
        raise ValueError('Full inventory does not cover every original expert')
    effective = ('splash.native-flash-weights-v1\nsource=' + manifest['source_identity_sha256'] +
        '\nmanifest=' + manifest['source_manifest_sha256'] + '\nnorm=one-plus-weight\n')
    return {'inventory_per_layer': inventory, 'expert_count': inventory * 48, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'plan_sha256': manifest['plan_sha256'], 'mapped_bytes': manifest['total_bytes'], 'planned_bytes': manifest['planned_allocation_bytes'],
        'source_identity': manifest['source_identity_sha256'], 'source_manifest': manifest['source_manifest_sha256'],
        'calibrated_norm_convention': 'one-plus-weight', 'effective_weights_layout_sha256': hashlib.sha256(effective.encode()).hexdigest(),
        'path': str(path.resolve()),
        'payload_validation': 'Runtime constructor verifies payload hashes/scales/codes; runner reads only manifest, not full tensor payload'}


def gate_status(status, plan, store, execution_mode='mtp3'):
    errors = http.validate_status(status)
    requirements = {'identity.source': SOURCE, 'identity.loaded_model_layout_sha256': store['effective_weights_layout_sha256'],
        'capabilities.native_route': 'flash-next', 'persisted_experts.enabled': True,
        'persisted_experts.expert_count': store['expert_count'], 'persisted_experts.store_manifest_sha256': store['manifest_sha256'],
        'persisted_experts.selection_plan_sha256': store['plan_sha256'], 'persisted_experts.mapped_bytes': store['mapped_bytes'],
        'persisted_experts.numerical_alternative': True,
        'ple_storage.ssd_streaming_enabled': True, 'idle_residency_maintenance.requested': False,
        'capabilities.prefix_cache': False,
        'persisted_experts.graph_counters.scope': 'graph construction, not GPU completion'}
    if execution_mode == 'mtp3':
        requirements.update({'mtp.singleton_maximum_draft_tokens': 3, 'mtp.teacher_cache_only_requested': True})
    elif execution_mode == 'standard':
        errors.extend(standard_mtp_errors(status))
    else:
        raise ValueError('Unknown semantic execution mode: ' + execution_mode)
    for key, value in requirements.items():
        if not http.same_json(http.get_path(status, key), value):
            errors.append('Actual policy/inventory differs: ' + key)
    if phase_quality.phase_present(status):
        errors.extend(phase_quality.phase_status_errors(status, store))
    if teacher_quality.teacher_bulk_present(status):
        errors.extend(teacher_quality.teacher_bulk_status_errors(status, store))
    if http.get_path(status, 'identity.target_all_rows_full512') is True:
        derivative = store.get('target_numerical_derivative_sha256')
        if not derivative or http.get_path(status, 'identity.target_numerical_derivative_sha256') != derivative:
            errors.append('Explicit all-row target numerical derivative identity differs or is unavailable')
        if store['inventory_per_layer'] != 512 or http.get_path(status, 'identity.original_target_gpu_omitted') is not True:
            errors.append('All-row private target lacks full inventory/original-GPU omission identity')
        omitted = {'ple_storage.all_rows_full512_target': True,
            'ple_storage.target_original_disk_tensor_count': 432,
            'ple_storage.target_original_disk_projection_count': 144,
            'ple_storage.target_original_disk_logical_bytes': 67947724800,
            'ple_storage.gpu_mapped_original_bytes': 6370164736}
        for name, expected in omitted.items():
            if not http.same_json(http.get_path(status, name), expected):
                errors.append('Private original-GPU omission geometry differs: ' + name)
    if http.get_path(status, 'maximum_context_tokens') < plan['minimum_context_capacity_tokens']:
        errors.append('Service capacity cannot complete the 8K task budget')
    return errors


def grade_record(record, case):
    expected = case['expected']
    errors = http.check_answer(record, expected, case['body']['max_completion_tokens'])
    if http.get_path(record, 'usage.prompt_tokens') != case['prompt_token_count']:
        errors.append('Actual rendered prompt count differs from frozen exact rows')
    if http.get_path(record, 'usage.prompt_tokens_details.cached_tokens') != 0:
        errors.append('Request reused prefix tokens')
    text = record.get('text', '')
    if '\ufffd' in text or re.search(r'<\|(?:im_|fim_|endoftext)|<think>|</think>', text):
        errors.append('Response has invalid text or leaked protocol markers')
    details = None
    if expected['kind'] == 'python':
        from dev.benchmarks.prefill4k_attribution_quality_python import grade_function
        properties = copy.deepcopy(expected)
        # This property is stated in the frozen clamp task's request: callers
        # receive a new list, including when values already lie in bounds.
        if case['id'] == 'python_clamp_values':
            properties['require_fresh_result'] = True
        details = grade_function(text, properties)
        errors.extend(details['errors'])
    return errors, details


# Field names are supplied by private overlay instrumentation; these counters
# count graph construction, never hardware work. Completion/idle is separate.
CACHE_COUNTERS = ['gate_up_graph_calls', 'gate_up_graph_rows', 'down_graph_calls', 'down_graph_rows',
    'encoded_hit_dispatches', 'encoded_miss_dispatches', 'full_inventory_graph_calls']


def coverage(before, after, case, require_counters=True, execution_mode='mtp3'):
    errors = []
    chunks = http.get_path(after, 'scheduler.maximum_prefill_rows')
    if not isinstance(chunks, int) or chunks < 256:
        return {}, ['Actual singleton prefill chunk policy cannot exercise large-row cache']
    windows = [min(chunks, case['prompt_token_count'] - start) for start in range(0, case['prompt_token_count'], chunks)]
    large = [count for count in windows if count >= 256]
    prime_calls = http.get_path(after, 'mtp.teacher_cache_only_priming_calls') - http.get_path(before, 'mtp.teacher_cache_only_priming_calls')
    expected_prime = sum((min(count, case['prompt_token_count'] - start - 1) + 127) // 128 for start, count in zip(range(0, case['prompt_token_count'], chunks), windows, strict=True))
    # Strict JSON schema uses AR/constrained cohort, so it has no MTP state.
    eligible = http.get_path(after, 'mtp.eligible_requests') - http.get_path(before, 'mtp.eligible_requests')
    if execution_mode not in ['standard', 'mtp3']:
        raise ValueError('Unknown semantic execution mode: ' + execution_mode)
    if execution_mode == 'standard':
        errors.extend('before: ' + error for error in standard_mtp_errors(before))
        errors.extend('after: ' + error for error in standard_mtp_errors(after))
    expected_eligible = 0 if execution_mode == 'standard' or case.get('body', {}).get('response_format', {}).get('type') == 'json_schema' else 1
    if expected_eligible == 0:
        expected_prime = 0
    teacher_details = {}
    for side, status in [('before', before), ('after', after)]:
        if teacher_quality.teacher_bulk_present(status) and not teacher_quality.teacher_bulk_active(status):
            errors.extend(side + ': ' + error for error in teacher_quality.teacher_bulk_status_errors(status))
    if teacher_quality.teacher_bulk_active(before) or teacher_quality.teacher_bulk_active(after):
        teacher_details, teacher_errors = teacher_quality.teacher_bulk_coverage(
            before, after, windows, case['prompt_token_count'], expected_eligible)
        errors.extend(teacher_errors)
        expected_prime = teacher_details['expected_teacher_commands']
    if eligible != expected_eligible:
        errors.append('Observed MTP eligibility differs from the frozen greedy/constrained body')
    if expected_eligible == 1 and prime_calls != expected_prime:
        errors.append('Successful teacher-cache API command count differs from strict bulk schedule' if teacher_details
                      else 'Successful teacher-cache call count differs from true adjacent-pair windows')
    if expected_eligible == 0 and prime_calls != 0:
        errors.append('Standard/constrained execution unexpectedly used teacher-cache priming')
    if phase_quality.phase_present(before) or phase_quality.phase_present(after):
        phase_details, phase_errors = phase_quality.phase_coverage(before, after, windows)
        errors.extend(phase_errors)
        if expected_eligible == 0 and any(value != 0 for value in
                phase_details['target_phase_graph_counter_deltas']['verify'].values()):
            errors.append('Standard/constrained execution unexpectedly encoded target Verify expert graphs')
        return {'execution_mode': execution_mode, 'actual_main_row_windows': windows,
            'large_cache_row_windows': large, 'teacher_successful_calls': prime_calls,
            'eligible_requests': eligible, 'expected_eligible_requests': expected_eligible,
            'expected_teacher_successful_calls_for_eligible': expected_prime,
            **phase_details, **teacher_details}, errors
    values = {}
    general_values = {}
    all_rows = http.get_path(after, 'identity.target_all_rows_full512') is True
    available = True
    for name in CACHE_COUNTERS:
        general_old, general_new = [http.get_path(status, 'persisted_experts.graph_counters.' + name) for status in [before, after]]
        if all(type(value) is int for value in [general_old, general_new]):
            general_values[name] = general_new - general_old
        selected_name = 'large_row_' + name if all_rows else name
        old, new = [http.get_path(status, 'persisted_experts.graph_counters.' + selected_name) for status in [before, after]]
        if not all(type(value) is int for value in [old, new]):
            available = False
            continue
        values[name] = new - old
    if not available and require_counters:
        errors.append('Private expert graph counters are unavailable; cache-use coverage not proven')
    if available:
        expected_calls, expected_rows = 48 * len(large), 48 * sum(large)
        for name in ['gate_up_graph_calls', 'down_graph_calls']:
            if values[name] != expected_calls:
                errors.append('Cache graph call count differs from all 48 target layers: ' + name)
        for name in ['gate_up_graph_rows', 'down_graph_rows']:
            if values[name] != expected_rows:
                errors.append('Cache graph row count differs from actual large main windows: ' + name)
        full = http.get_path(after, 'persisted_experts.expert_count') == 24576
        if values['encoded_hit_dispatches'] != 2 * expected_calls:
            errors.append('Expected hit projections were not encoded')
        if values['encoded_miss_dispatches'] != (0 if full else 2 * expected_calls):
            errors.append('Encoded miss dispatch count differs from actual inventory policy')
        if values['full_inventory_graph_calls'] != (2 * expected_calls if full else 0):
            errors.append('Full-inventory graph counter differs')
    small_row = {name: general_values[name] - values[name] for name in CACHE_COUNTERS if name in general_values and name in values}
    if all_rows:
        if len(general_values) != len(CACHE_COUNTERS) or any(value < 0 for value in small_row.values()):
            errors.append('All-row general counters are unavailable or precede large-row counters')
        if general_values.get('encoded_miss_dispatches') != 0:
            errors.append('All-row Full512 unexpectedly encoded a raw miss projection')
        expected_scope = 'physical rows>=256; graph construction; excludes tiny prefill/decode/verifier'
        if http.get_path(after, 'persisted_experts.graph_counters.large_row_counter_scope') != expected_scope:
            errors.append('All-row large-prefill graph counter scope differs')
    return {'execution_mode': execution_mode, 'actual_main_row_windows': windows, 'large_cache_row_windows': large, 'teacher_successful_calls': prime_calls,
        'eligible_requests': eligible, 'expected_eligible_requests': expected_eligible, 'expected_teacher_successful_calls_for_eligible': expected_prime,
        'expert_graph_counter_deltas': values, 'expert_graph_counters_available': available,
        'target_all_rows_full512': all_rows, 'general_expert_graph_counter_deltas': general_values,
        'small_row_target_graph_counter_deltas': small_row,
        'small_row_scope': 'decode/verifier and tiny prefill tails; graph construction, not GPU completion',
        'counter_scope': 'graph construction, not GPU completion; fresh idle/completed request counters establish completion separately',
        **teacher_details}, errors


def measure(args):
    if not args.run_root_gpu:
        raise ValueError('Actual model requests require root-exclusive --run-root-gpu')
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme != 'http' or endpoint.hostname not in ['127.0.0.1', 'localhost', '::1'] or not endpoint.port or endpoint.port == 8000:
        raise ValueError('Explicit private loopback endpoint required')
    if args.output.exists():
        raise FileExistsError('Choose a fresh report')
    plan = read_plan(args.plan)
    store = store_witness(args.expert_store, args.inventory)
    if args.target_derivative_sha256:
        if not re.fullmatch(r'[0-9a-f]{64}', args.target_derivative_sha256):
            raise ValueError('Explicit target derivative must be a SHA256 digest')
        store['target_numerical_derivative_sha256'] = args.target_derivative_sha256
    runtime_hashes = {name: hashlib.sha256((args.runtime_build / name).read_bytes()).hexdigest() for name in ['splash-flash', 'splash.metallib']}
    execution_mode = getattr(args, 'execution_mode', 'mtp3')
    actual_common_policy = copy.deepcopy(plan['common_policy'])
    if execution_mode == 'standard':
        actual_common_policy.update(mtp_depth=0, teacher_cache_only=False)
    report = {'schema': 'splash-prefill4k-semantic-report-v1', 'completed': False, 'valid': False, 'label': args.label,
        'execution_mode': execution_mode,
        'plan': str(args.plan.resolve()), 'plan_content_sha256': plan['content_sha256'], 'store_witness': store,
        'base_url': args.base_url, 'model': args.model, 'runtime_file_sha256': runtime_hashes,
        'runtime_witness_scope': 'root-supplied runtime files; HTTP identity/inventory and private graph counters independently checked',
        'cases': [], 'errors': [], 'common_policy': actual_common_policy,
        'frozen_plan_common_policy': plan['common_policy'],
        'quality_scope': plan['quality_scope'], 'numeric_source_quality': plan['numeric_source_quality'],
        'timing_scope': 'observational quality-run timings; not a matched full-output-budget performance benchmark',
        'strict_cache_graph_coverage_required': not args.allow_inferred_cache_coverage}
    write_new(args.output, report)
    runner = None
    try:
        runner = http.Qualification(args, report)
        report['actual_execution_policy'] = execution_policy(runner.initial)
        errors = gate_status(runner.initial, plan, store, execution_mode)
        if errors:
            raise ValueError('; '.join(errors))
        for case in plan['cases']:
            if args.case_id and case['id'] not in args.case_id:
                report['cases'].append({'id': case['id'], 'status': 'skipped', 'reason': 'explicit bounded subset'})
                continue
            before = runner.wait_idle()
            row = {'id': case['id'], 'group': case['group'], 'expected': case['expected'], 'status_before': before, 'errors': []}
            body = copy.deepcopy(case['body']); body['model'] = args.model
            record = runner.client.send('POST', '/v1/chat/completions', body)
            record['expected_model'] = args.model; record['cache_disabled'] = runner.cache_disabled
            record['request_body'] = body
            record['prompt_u32le_sha256'] = case['prompt_u32le_sha256']
            record['request_body_sha256_without_model'] = case['request_body_sha256_without_model']
            errors, details = grade_record(record, case)
            record['task_errors'] = errors
            if details is not None:
                record['python_grading'] = details
            row['records'] = [record]
            row['task_passed'] = not errors
            after = runner.wait_idle(after_snapshot=http.get_path(before, 'status_snapshot.steady_seconds'))
            row['status_after'] = after
            row['errors'].extend(gate_status(after, plan, store, execution_mode))
            if execution_policy(before) != report['actual_execution_policy'] or execution_policy(after) != report['actual_execution_policy']:
                row['errors'].append('Actual execution policy changed during the isolated task')
            row['native_counter_duration_delta'] = http.counter_delta(before, after)
            delta = row['native_counter_duration_delta']
            if delta.get('requests.submitted') != 1 or delta.get('requests.completed') != 1 or delta.get('requests.failed') != 0 or delta.get('requests.cancelled') != 0:
                row['errors'].append('Actual native terminals differ from one completed request')
            if delta.get('metrics.prefill_input_tokens') != case['prompt_token_count']:
                row['errors'].append('Native main prefill count differs from frozen actual prompt count')
            row['cache_use_coverage'], cache_errors = coverage(before, after, case, not args.allow_inferred_cache_coverage, execution_mode)
            row['errors'].extend(cache_errors)
            row['status'] = 'passed' if row['task_passed'] and not row['errors'] else 'failed'
            report['cases'].append(row)
            checkpoint(args.output, report)
            print(json.dumps({'id': row['id'], 'status': row['status'], 'task_errors': errors, 'coverage_errors': row['errors'], 'usage': record.get('usage')}), flush=True)
        report['final_status'] = runner.wait_idle()
        report['completed'] = True
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    report['task_passed_cases'] = sum(row.get('task_passed') is True for row in report['cases'])
    report['passed_cases'] = sum(row['status'] == 'passed' for row in report['cases'])
    report['failed_cases'] = sum(row['status'] == 'failed' for row in report['cases'])
    report['skipped_cases'] = sum(row['status'] == 'skipped' for row in report['cases'])
    report['full_plan_coverage'] = report['completed'] and len(report['cases']) == 22 and report['skipped_cases'] == 0
    report['expert_graph_coverage_complete'] = report['full_plan_coverage'] and all(row.get('cache_use_coverage', {}).get('expert_graph_counters_available') is True for row in report['cases'])
    report['valid'] = report['full_plan_coverage'] and report['expert_graph_coverage_complete'] and not report['errors'] and report['failed_cases'] == 0 and report['passed_cases'] == 22
    strict_numbers(report)
    checkpoint(args.output, report)
    print(json.dumps({'report': str(args.output), 'valid': report['valid'], 'task_passed': report['task_passed_cases'], 'coverage_passed': report['passed_cases'], 'full_plan_coverage': report['full_plan_coverage']}))
    return 0 if report['valid'] else 1


def compare(args):
    reports = [http.strict_json(path.read_text()) for path in args.reports]
    for report in reports:
        strict_numbers(report)
    execution_modes = {report.get('execution_mode', 'mtp3') for report in reports}
    if len(execution_modes) != 1 or not execution_modes.issubset({'standard', 'mtp3'}):
        raise ValueError('All inventories must use the same declared semantic execution mode')
    runtime_changed = len({digest(report['runtime_file_sha256']) for report in reports}) != 1
    if len({digest(report['actual_execution_policy']) for report in reports}) != 1:
        raise ValueError('All inventories must use the same actual execution policy')
    if runtime_changed and not getattr(args, 'allow_runtime_change', False):
        raise ValueError('Intentional numerical-derivative binary comparison requires explicit --allow-runtime-change')
    if len({report['plan_content_sha256'] for report in reports}) != 1:
        raise ValueError('All inventories must use the same frozen task plan')
    baseline = reports[0]
    plan = read_plan(Path(baseline['plan']))
    frozen = {case['id']: case for case in plan['cases']}
    if plan['content_sha256'] != baseline['plan_content_sha256']:
        raise ValueError('Report frozen plan digest differs')
    regraded = []
    report_audits = []
    for report in reports:
        execution_mode = report.get('execution_mode', 'mtp3')
        tasks = {}
        audit_errors = list(report.get('errors', []))
        reconstructed_policy = execution_policy(report['initial_status'])
        if reconstructed_policy != report['actual_execution_policy']:
            audit_errors.append('Actual execution policy summary differs from the saved initial status')
        if len(report['cases']) != 22 or {row['id'] for row in report['cases']} != set(frozen):
            audit_errors.append('Report does not have all22 unique frozen tasks')
        for name in ['initial_status', 'final_status']:
            if name not in report:
                audit_errors.append('Saved terminal status is unavailable: ' + name)
            else:
                audit_errors.extend(gate_status(report[name], plan, report['store_witness'], execution_mode))
                if not http.idle(report[name]):
                    audit_errors.append('Saved run status is not idle: ' + name)
                if execution_policy(report[name]) != reconstructed_policy:
                    audit_errors.append('Actual execution policy changed in saved run status: ' + name)
        for case in report['cases']:
            if case['id'] not in frozen:
                raise ValueError('Report has an unknown task')
            if len(case.get('records', [])) != 1:
                tasks[case['id']] = {'passed': False, 'errors': ['Task has no single actual response']}
                continue
            record = case['records'][0]
            body_hash = digest({k: v for k, v in record['request_body'].items() if k != 'model'})
            if body_hash != frozen[case['id']]['request_body_sha256_without_model']:
                raise ValueError('Actual task request differs from frozen body')
            errors, _ = grade_record(record, frozen[case['id']])
            before, after = case.get('status_before'), case.get('status_after')
            if not isinstance(before, dict) or not isinstance(after, dict):
                audit_errors.append('Task lacks saved before/after status: ' + case['id'])
            else:
                for status in [before, after]:
                    audit_errors.extend(gate_status(status, plan, report['store_witness'], execution_mode))
                    audit_errors.extend(http.validate_status(status, report['initial_status']['identity']))
                    if not http.idle(status):
                        audit_errors.append('Saved case status is not idle: ' + case['id'])
                    if execution_policy(status) != reconstructed_policy:
                        audit_errors.append('Actual execution policy changed in saved task status: ' + case['id'])
                _, coverage_errors = coverage(before, after, frozen[case['id']], True, execution_mode)
                audit_errors.extend(case['id'] + ': ' + message for message in coverage_errors)
                delta = http.counter_delta(before, after)
                if delta.get('requests.submitted') != 1 or delta.get('requests.completed') != 1 or delta.get('requests.failed') != 0 or delta.get('requests.cancelled') != 0:
                    audit_errors.append('Recomputed native terminals differ: ' + case['id'])
                if delta.get('metrics.prefill_input_tokens') != frozen[case['id']]['prompt_token_count']:
                    audit_errors.append('Recomputed main input tokens differ: ' + case['id'])
                old_snapshot = http.get_path(before, 'status_snapshot.steady_seconds')
                new_snapshot = http.get_path(after, 'status_snapshot.steady_seconds')
                if not all(type(value) in [int, float] for value in [old_snapshot, new_snapshot]) or new_snapshot <= old_snapshot:
                    audit_errors.append('Saved completion snapshot is not newer: ' + case['id'])
            tasks[case['id']] = {'passed': not errors, 'errors': errors}
        regraded.append(tasks)
        complete = len(report['cases']) == 22 and set(tasks) == set(frozen)
        evidence_valid = not audit_errors and complete
        passed = sum(task['passed'] for task in tasks.values())
        report_audits.append({'label': report['label'], 'valid': evidence_valid and passed == 22,
            'evidence_valid': evidence_valid, 'full_plan_coverage': complete,
            'task_passed_cases': passed, 'all_tasks_pass': passed == 22,
            'task_failure_ids': [name for name, task in tasks.items() if not task['passed']], 'errors': audit_errors})
    rows = []
    for report_index, candidate in enumerate(reports[1:], 1):
        left = {row['id']: row for row in baseline['cases']}
        right = {row['id']: row for row in candidate['cases']}
        if left.keys() != right.keys():
            raise ValueError('Report coverage differs')
        cases = []
        for name, a in left.items():
            b = right[name]
            ta = a.get('records', [{}])[0].get('text', '')
            tb = b.get('records', [{}])[0].get('text', '')
            left_grade, right_grade = regraded[0][name], regraded[report_index][name]
            cases.append({'id': name, 'baseline_task_passed': left_grade['passed'], 'candidate_task_passed': right_grade['passed'],
                'new_task_regression': left_grade['passed'] and not right_grade['passed'],
                'generation_text_equal_diagnostic_only': ta == tb,
                'baseline_answer': ta, 'candidate_answer': tb,
                'baseline_task_errors': left_grade['errors'], 'candidate_task_errors': right_grade['errors']})
        rows.append({'baseline_label': baseline['label'], 'candidate_label': candidate['label'], 'baseline_inventory': baseline['store_witness']['inventory_per_layer'],
            'candidate_inventory': candidate['store_witness']['inventory_per_layer'], 'baseline_valid': report_audits[0]['valid'], 'candidate_valid': report_audits[report_index]['valid'],
            'baseline_evidence_valid': report_audits[0]['evidence_valid'], 'candidate_evidence_valid': report_audits[report_index]['evidence_valid'],
            'baseline_task_passed_cases': report_audits[0]['task_passed_cases'], 'candidate_task_passed_cases': report_audits[report_index]['task_passed_cases'],
            'both_full_plan_coverage': report_audits[0]['full_plan_coverage'] and report_audits[report_index]['full_plan_coverage'],
            'regraded_all_tasks_pass': len(cases) == 22 and all(case['baseline_task_passed'] and case['candidate_task_passed'] for case in cases),
            'new_task_regressions': sum(case['new_task_regression'] for case in cases),
            'new_task_regression_ids': [case['id'] for case in cases if case['new_task_regression']],
            'persisting_baseline_failure_ids': [case['id'] for case in cases if not case['baseline_task_passed'] and not case['candidate_task_passed']],
            'resolved_baseline_failure_ids': [case['id'] for case in cases if not case['baseline_task_passed'] and case['candidate_task_passed']],
            'ownership_and_residency_difference': {'baseline': ownership_policy(baseline['initial_status']),
                'candidate': ownership_policy(candidate['initial_status']),
                'scope': 'intentional model/resource policy differences; independent of fixed request sampling and task grading'},
            'generation_differences': sum(not case['generation_text_equal_diagnostic_only'] for case in cases), 'cases': cases})
    result = {'schema': 'splash-prefill4k-semantic-comparison-v1', 'cpu_only': True, 'plan_content_sha256': baseline['plan_content_sha256'],
        'execution_mode': next(iter(execution_modes)),
        'runtime_binary_changed': runtime_changed, 'runtime_change_explicitly_allowed': getattr(args, 'allow_runtime_change', False),
        'target_numerical_derivative_identities': [report['store_witness'].get('target_numerical_derivative_sha256') for report in reports],
        'comparisons': rows, 'recomputed_report_audits': report_audits, 'generation_hash_changes_are_not_regressions': True, 'task_outcomes_regraded_from_actual_outputs': True,
        'numeric_source_quality_separate_required': True, 'eight_protocol_lifecycle_cases_separate_required': True,
        'validity_scope': 'complete independently audited comparison evidence; task success and new regressions are separate outcomes',
        'all_tasks_pass': bool(rows) and all(row['regraded_all_tasks_pass'] for row in rows),
        'no_new_task_regressions': bool(rows) and all(row['new_task_regressions'] == 0 for row in rows),
        'full_plan_coverage': bool(rows) and all(row['both_full_plan_coverage'] for row in rows),
        'valid': bool(rows) and all(row['baseline_evidence_valid'] and row['candidate_evidence_valid'] and row['both_full_plan_coverage'] for row in rows)}
    write_new(args.output, result)
    print(json.dumps({'valid': result['valid'], 'all_tasks_pass': result['all_tasks_pass'], 'no_new_task_regressions': result['no_new_task_regressions'],
        'comparisons': len(rows), 'new_task_regressions': [row['new_task_regressions'] for row in rows], 'gpu_executed': False}))
    return 0 if result['valid'] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--tokenizer', type=Path, default=TOKENIZER)
    plan.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('measure')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--expert-store', type=Path, required=True)
    run.add_argument('--runtime-build', type=Path, default=ROOT / 'build/prefill4k-fullcache')
    run.add_argument('--target-derivative-sha256', help='Required explicit identity for private all-row Full512 or Prefill-I8/Decode-Q4 phase target')
    run.add_argument('--inventory', type=int, choices=[64, 128, 256, 512], required=True)
    run.add_argument('--execution-mode', choices=['standard', 'mtp3'], default='mtp3',
                     help='MTP3 retains the frozen priming gates; standard requires absent head and zero MTP work')
    run.add_argument('--base-url', default='http://127.0.0.1:8011')
    run.add_argument('--model', default=MODEL)
    run.add_argument('--label', required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--timeout', type=float, default=240)
    run.add_argument('--cleanup-timeout', type=float, default=30)
    run.add_argument('--deadline-seconds', type=float, default=0.2)
    run.add_argument('--run-root-gpu', action='store_true')
    run.add_argument('--allow-inferred-cache-coverage', action='store_true', help='Explicit incomplete screen; cannot prove actual cache graph coverage')
    run.add_argument('--case-id', action='append', default=[])
    pair = commands.add_parser('compare')
    pair.add_argument('--reports', type=Path, nargs='+', required=True)
    pair.add_argument('--output', type=Path, required=True)
    pair.add_argument('--allow-runtime-change', action='store_true', help='Explicit intentional numerical-derivative binary comparison; common policy still required')
    args = p.parse_args(argv)
    if args.command == 'plan':
        make_plan(args); return 0
    if args.command == 'measure':
        return measure(args)
    return compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
