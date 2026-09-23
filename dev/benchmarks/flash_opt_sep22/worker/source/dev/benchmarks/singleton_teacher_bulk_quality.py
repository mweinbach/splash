"""Strict CPU coverage of the GPU-qualified singleton teacher bulk schedule."""
import re

from dev.benchmarks import qualify_flash_http as http

TEACHER_SCHEMA = 'singleton-teacher-original-cache-bulk2048-prefix128-v1'
TEACHER_SOURCE_SHA = '99b7f20ae3d78ad1b81a4fe47cef2a04c0ac327eea9732174e1d6e86c1bb7e1a'
WORKSPACE_BYTES = 310181888
COUNTERS = ('completed_teacher_commands', 'completed_pairs',
    'completed_original128_cache_prefix_windows', 'completed_original128_cache_prefix_rows',
    'completed_bulk_commands', 'completed_bulk_pairs', 'completed_tail_commands', 'completed_tail_pairs')
OWNERSHIP_FIELDS = ('identity.singleton_teacher_bulk_schema',
    'identity.singleton_teacher_bulk_enabled', 'identity.singleton_teacher_bulk_source_sha256',
    'mtp.singleton_teacher_bulk.requested', 'mtp.singleton_teacher_bulk.maximum_bulk_rows',
    'mtp.singleton_teacher_bulk.planned_workspace_bytes',
    'mtp.singleton_teacher_bulk.allocated_workspace_bytes')


def teacher_bulk_present(status):
    identity = status.get('identity', {}) if isinstance(status, dict) else {}
    return (http.get_path(status, 'mtp.singleton_teacher_bulk') is not None
            or isinstance(identity, dict) and any(name in identity for name in (
                'singleton_teacher_bulk_schema', 'singleton_teacher_bulk_enabled',
                'singleton_teacher_bulk_source_sha256')))


def teacher_bulk_active(status):
    return (http.get_path(status, 'identity.singleton_teacher_bulk_enabled') is True
            or http.get_path(status, 'mtp.singleton_teacher_bulk.requested') is True
            or http.get_path(status, 'identity.singleton_teacher_bulk_schema') is not None
            or http.get_path(status, 'identity.singleton_teacher_bulk_source_sha256') is not None)


def _counter(value):
    return type(value) is int and 0 <= value < 2 ** 64


def teacher_bulk_status_errors(status, store=None):
    active = teacher_bulk_active(status)
    errors = []
    requirements = {
        'identity.singleton_teacher_bulk_enabled': active,
        'identity.singleton_teacher_bulk_schema': TEACHER_SCHEMA if active else None,
        'identity.singleton_teacher_bulk_source_sha256': TEACHER_SOURCE_SHA if active else None,
        'mtp.singleton_teacher_bulk.requested': active,
        'mtp.singleton_teacher_bulk.maximum_bulk_rows': 2048 if active else 0,
        'mtp.singleton_teacher_bulk.planned_workspace_bytes': WORKSPACE_BYTES if active else 0,
        'mtp.singleton_teacher_bulk.allocated_workspace_bytes': WORKSPACE_BYTES if active else 0,
    }
    if active:
        requirements.update({'mtp.teacher_cache_only_requested': True,
            'scheduler.maximum_prefill_rows': 2048})
        if store is not None:
            derivative = store.get('target_numerical_derivative_sha256')
            if (not isinstance(derivative, str) or not re.fullmatch(r'[0-9a-f]{64}', derivative)
                    or http.get_path(status, 'identity.target_numerical_derivative_sha256') != derivative):
                errors.append('Explicit teacher-bulk target numerical derivative identity differs or is unavailable')
    for path, value in requirements.items():
        if not http.same_json(http.get_path(status, path), value):
            errors.append('Strict singleton teacher-bulk profile differs: ' + path)
    values = {}
    for name in COUNTERS:
        value = http.get_path(status, 'mtp.singleton_teacher_bulk.' + name)
        if not _counter(value):
            errors.append('Singleton teacher-bulk counter is unavailable or invalid: ' + name)
        else:
            values[name] = value
            if not active and value != 0:
                errors.append('Inactive singleton teacher bulk unexpectedly recorded work: ' + name)
    if active and len(values) == len(COUNTERS):
        for total, left, right in (('completed_teacher_commands', 'completed_bulk_commands', 'completed_tail_commands'),
                                  ('completed_pairs', 'completed_bulk_pairs', 'completed_tail_pairs')):
            if values[total] != values[left] + values[right]:
                errors.append('Singleton teacher-bulk cumulative totals do not reconcile: ' + total)
        if values['completed_original128_cache_prefix_rows'] != values['completed_pairs']:
            errors.append('Singleton teacher-bulk cumulative original-cache prefix rows differ from pairs')
        actual_calls = http.get_path(status, 'mtp.teacher_cache_only_priming_calls')
        if not _counter(actual_calls) or actual_calls != values['completed_teacher_commands']:
            errors.append('Singleton teacher-bulk actual API calls differ from completed teacher commands')
        bulk_calls, bulk_pairs = values['completed_bulk_commands'], values['completed_bulk_pairs']
        tail_calls, tail_pairs = values['completed_tail_commands'], values['completed_tail_pairs']
        if bulk_pairs % 128 or not 128 * bulk_calls <= bulk_pairs <= 2048 * bulk_calls:
            errors.append('Singleton teacher-bulk cumulative bulk work violates its step budget')
        if not tail_calls <= tail_pairs <= 127 * tail_calls:
            errors.append('Singleton teacher-bulk cumulative tail work violates its original API budget')
        if values['completed_original128_cache_prefix_windows'] != bulk_pairs // 128 + tail_calls:
            errors.append('Singleton teacher-bulk cumulative logical prefix windows differ')
    return errors


def teacher_bulk_schedule(windows, total):
    values = dict.fromkeys(COUNTERS, 0)
    pairs_per_window = []
    start = 0
    for rows in windows:
        pairs = max(0, min(rows, total - start - 1))
        pairs_per_window.append(pairs)
        remaining = pairs
        while remaining:
            complete = min(2048, remaining // 128 * 128)
            count = complete if complete else remaining
            values['completed_teacher_commands'] += 1
            values['completed_pairs'] += count
            values['completed_original128_cache_prefix_windows'] += (count + 127) // 128
            values['completed_original128_cache_prefix_rows'] += count
            kind = 'bulk' if complete else 'tail'
            values['completed_' + kind + '_commands'] += 1
            values['completed_' + kind + '_pairs'] += count
            remaining -= count
        start += rows
    return values, pairs_per_window


def teacher_bulk_coverage(before, after, windows, total, expected_eligible):
    errors = []
    for side, status in (('before', before), ('after', after)):
        errors.extend(side + ': ' + error for error in teacher_bulk_status_errors(status))
        if not teacher_bulk_active(status):
            errors.append(side + ': Strict singleton teacher-bulk route is not active')
    expected, pairs_per_window = teacher_bulk_schedule(windows, total)
    if expected_eligible == 0:
        expected = dict.fromkeys(COUNTERS, 0)
    deltas = {}
    for name in COUNTERS:
        old, new = [http.get_path(status, 'mtp.singleton_teacher_bulk.' + name)
                    for status in (before, after)]
        if not all(_counter(value) for value in (old, new)) or new < old:
            errors.append('Singleton teacher-bulk counter is unavailable, invalid or decreased: ' + name)
            continue
        deltas[name] = new - old
        if deltas[name] != expected[name]:
            errors.append('Singleton teacher-bulk completed schedule differs from true adjacent pairs: ' + name)
    old, new = [http.get_path(status, 'mtp.teacher_cache_only_priming_calls')
                for status in (before, after)]
    actual = new - old if all(_counter(value) for value in (old, new)) and new >= old else None
    if actual != expected['completed_teacher_commands']:
        errors.append('Singleton teacher-bulk actual API call delta differs from completed schedule')
    return {'singleton_teacher_bulk_enabled': True,
        'singleton_teacher_bulk_schema': TEACHER_SCHEMA,
        'singleton_teacher_bulk_source_sha256': TEACHER_SOURCE_SHA,
        'teacher_pair_rows_per_main_window': pairs_per_window,
        'teacher_bulk_counter_deltas': deltas,
        'expected_teacher_commands': expected['completed_teacher_commands'],
        'expected_teacher_pairs': expected['completed_pairs'],
        'expected_original128_cache_prefix_windows': expected['completed_original128_cache_prefix_windows'],
        'expected_original128_cache_prefix_rows': expected['completed_original128_cache_prefix_rows'],
        'teacher_actual_api_commands': actual,
        'teacher_command_scope': 'completed actual arena/API calls; original128 cache prefix windows and pair rows are separate',
    }, errors
