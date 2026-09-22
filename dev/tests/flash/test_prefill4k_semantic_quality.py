"""CPU-only semantic grading and cache-coverage gates; no model/GPU."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from dev.benchmarks import prefill4k_attribution_quality as quality


class SemanticQualityTests(unittest.TestCase):
    def record(self, text, case):
        prompt = case['target_prompt_tokens']
        return {'http_status': 200, 'content_type': 'text/event-stream', 'stream': True, 'done': True,
            'first_content_ms': 1, 'request_ids': ['fixture-id'], 'model_ids': [quality.MODEL], 'expected_model': quality.MODEL,
            'text': text, 'tool_calls': [], 'finish_reason': 'stop', 'reasoning_text': '', 'error_frames': [], 'errors': [],
            'usage': {'prompt_tokens': prompt, 'completion_tokens': 10, 'total_tokens': prompt + 10,
                'prompt_tokens_details': {'cached_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0}}}

    def prepared(self, case):
        case = copy.deepcopy(case)
        case['prompt_token_count'] = case['target_prompt_tokens']
        return case

    def test_cardinality_and_real_arithmetic(self):
        cases = quality.specs()
        self.assertEqual(len(cases), 22)
        self.assertTrue(all(c['target_prompt_tokens'] >= 256 for c in cases))
        answers = {c['id']: c['expected']['value'] for c in cases if c['group'] == 'arithmetic'}
        self.assertEqual(answers, {'arithmetic_multiply': '3973', 'arithmetic_signed': '-691', 'arithmetic_parentheses': '306', 'arithmetic_inventory': '96'})

    def test_number_anywhere_is_not_an_answer(self):
        case = self.prepared(quality.specs()[0])
        self.assertFalse(quality.grade_record(self.record('3973', case), case)[0])
        for text in ['The answer is 3973.', '3973 then 7', '```3973```', 'wrong=0 expected=3973', '3974']:
            self.assertTrue(quality.grade_record(self.record(text, case), case)[0])

    def test_strict_json_types_and_duplicate_keys(self):
        case = self.prepared(next(c for c in quality.specs() if c['id'] == 'json_arithmetic_schema'))
        self.assertFalse(quality.grade_record(self.record('{"ok":true,"difference":-3,"sum":15}', case), case)[0])
        for text in ['{"ok":1,"difference":-3,"sum":15}', '{"ok":true,"difference":-3,"sum":15.0}',
            '{"ok":true,"difference":-3,"sum":15,"extra":0}', '{"ok":true,"difference":-3,"sum":15,"sum":15}',
            '```json\n{"ok":true,"difference":-3,"sum":15}\n```']:
            self.assertTrue(quality.grade_record(self.record(text, case), case)[0])

    def test_copy_preserves_every_inner_character(self):
        case = self.prepared(next(c for c in quality.specs() if c['id'] == 'copy8k_late'))
        expected = case['expected']['value']
        self.assertFalse(quality.grade_record(self.record(expected, case), case)[0])
        for text in [expected.lower(), expected.replace('-0042', '-42'), expected.replace('\n', ' '), expected + ' done']:
            self.assertTrue(quality.grade_record(self.record(text, case), case)[0])

    def test_five_functions_pass_their_heldout_vectors(self):
        sources = {
            'sum_even': 'def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)',
            'clamp_values': 'def clamp_values(numbers, low, high):\n    return [max(low, min(high, n)) for n in numbers]',
            'prefix_sums': 'def prefix_sums(numbers):\n    return [sum(numbers[:i+1]) for i in range(len(numbers))]',
            'longest_run': 'def longest_run(numbers):\n    best = 0\n    run = 0\n    previous = 0\n    for value in numbers:\n        if run and value == previous:\n            run += 1\n        else:\n            run = 1\n        previous = value\n        best = max(best, run)\n    return best',
            'merge_counts': 'def merge_counts(pairs):\n    return {key: sum(value for item, value in pairs if item == key) for key in set(item for item, value in pairs)}',
        }
        for original in quality.specs():
            if original['group'] != 'python':
                continue
            case = self.prepared(original)
            errors, details = quality.grade_record(self.record(sources[case['expected']['function']], case), case)
            self.assertEqual(errors, [], (case['id'], details))
        case = self.prepared(next(c for c in quality.specs() if c['id'] == 'python_longest_run'))
        errors, _ = quality.grade_record(self.record('def longest_run(numbers):\n    return len(numbers)', case), case)
        self.assertTrue(errors)

    def status(self, count, rows, calls, prime, eligible, misses=0):
        return {'scheduler': {'maximum_prefill_rows': 2048}, 'mtp': {'teacher_cache_only_priming_calls': prime, 'eligible_requests': eligible},
            'persisted_experts': {'expert_count': count, 'graph_counters': {
                'gate_up_graph_calls': calls, 'gate_up_graph_rows': rows, 'down_graph_calls': calls, 'down_graph_rows': rows,
                'encoded_hit_dispatches': calls * 2, 'encoded_miss_dispatches': misses, 'full_inventory_graph_calls': calls * 2 if count == 24576 else 0}}}

    def test_full_inventory_actual_long_cache_projection_gate(self):
        case = {'prompt_token_count': 4096}
        before = self.status(24576, 0, 0, 0, 0)
        after = self.status(24576, 48 * 4096, 96, 32, 1)
        coverage, errors = quality.coverage(before, after, case)
        self.assertEqual(errors, [])
        self.assertEqual(coverage['actual_main_row_windows'], [2048, 2048])
        self.assertTrue(quality.coverage(before, self.status(24576, 48 * 4096, 96, 31, 1), case)[1])
        self.assertTrue(quality.coverage(before, self.status(24576, 48 * 4096, 96, 32, 1, 1), case)[1])
        no_counters = copy.deepcopy(after); no_counters['persisted_experts'].pop('graph_counters')
        self.assertTrue(quality.coverage(before, no_counters, case)[1])

    def test_2049_tail_counts_only_large_cached_main_window(self):
        case = {'prompt_token_count': 2049}
        before = self.status(6144, 0, 0, 0, 0)
        after = self.status(6144, 48 * 2048, 48, 16, 1, 96)
        coverage, errors = quality.coverage(before, after, case)
        self.assertEqual(errors, [])
        self.assertEqual(coverage['actual_main_row_windows'], [2048, 1])
        self.assertEqual(coverage['large_cache_row_windows'], [2048])

    def test_missing_teacher_eligibility_cannot_pass_a_greedy_task(self):
        case = {'prompt_token_count': 2048, 'body': {'temperature': 0}}
        before = self.status(24576, 0, 0, 0, 0)
        after = self.status(24576, 48 * 2048, 48, 0, 0)
        self.assertTrue(quality.coverage(before, after, case)[1])
        after = self.status(24576, 48 * 2048, 48, 16, 1)
        self.assertFalse(quality.coverage(before, after, case)[1])
        case['body']['response_format'] = {'type': 'json_schema'}
        after = self.status(24576, 48 * 2048, 48, 0, 0)
        self.assertFalse(quality.coverage(before, after, case)[1])
        after = self.status(24576, 48 * 2048, 48, 1, 0)
        self.assertTrue(quality.coverage(before, after, case)[1])

    def test_published_top256_witness_and_partial_projection_policy(self):
        directory = quality.ROOT / 'build/prefill4k-fullcache-artifacts/int8-experts-frequency256-v1'
        if not (directory / 'manifest.json').is_file():
            self.skipTest('Certified Top256 sidecar is not published in this checkout')
        witness = quality.store_witness(directory, 256)
        self.assertEqual(witness['inventory_per_layer'], 256)
        self.assertEqual(witness['expert_count'], 12288)
        self.assertEqual(witness['mapped_bytes'], 60586721280)
        self.assertEqual(witness['manifest_sha256'], 'ed271c58bd52f5914d0a3601321a2c716e24c8713c9a590fccbaee37fe888e10')
        case = {'prompt_token_count': 2048, 'body': {'temperature': 0}}
        before = self.status(witness['expert_count'], 0, 0, 0, 0)
        after = self.status(witness['expert_count'], 48 * 2048, 48, 16, 1, 96)
        coverage, errors = quality.coverage(before, after, case)
        self.assertEqual(errors, [])
        self.assertEqual(coverage['expert_graph_counter_deltas']['full_inventory_graph_calls'], 0)
        self.assertEqual(coverage['expert_graph_counter_deltas']['encoded_miss_dispatches'], 96)
        no_misses = self.status(witness['expert_count'], 48 * 2048, 48, 16, 1, 0)
        self.assertTrue(quality.coverage(before, no_misses, case)[1])
        wrong_full = copy.deepcopy(after)
        wrong_full['persisted_experts']['graph_counters']['full_inventory_graph_calls'] = 96
        self.assertTrue(quality.coverage(before, wrong_full, case)[1])

    def test_real_top64_failures_remain_visible_in_complete_comparison(self):
        report = quality.ROOT / 'build/release/flash/prefill4k-semantic-prefill4k-top64-qual.json'
        if not report.is_file():
            self.skipTest('Root Top64 semantic generation fixture is unavailable')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'comparison.json'
            self.assertEqual(quality.compare(SimpleNamespace(reports=[report, report], output=output)), 0)
            audit = json.loads(output.read_text())
        self.assertTrue(audit['valid'])
        self.assertTrue(audit['full_plan_coverage'])
        self.assertTrue(audit['no_new_task_regressions'])
        self.assertFalse(audit['all_tasks_pass'])
        row = audit['comparisons'][0]
        self.assertEqual(row['baseline_task_passed_cases'], 19)
        self.assertEqual(row['candidate_task_passed_cases'], 19)
        self.assertEqual(set(row['persisting_baseline_failure_ids']),
            {'arithmetic_signed', 'arithmetic_inventory', 'python_merge_counts'})
        self.assertEqual(row['new_task_regressions'], 0)

    def test_allrows_full512_separates_large_prefill_from_small_target_graphs(self):
        case = {'prompt_token_count': 2048, 'body': {'temperature': 0}}
        before = self.status(24576, 0, 0, 0, 0)
        after = self.status(24576, 48 * (2048 + 8), 48 * 3, 16, 1)
        for status in [before, after]:
            status['identity'] = {'target_all_rows_full512': True}
            counters = status['persisted_experts']['graph_counters']
            counters['large_row_counter_scope'] = 'physical rows>=256; graph construction; excludes tiny prefill/decode/verifier'
            large = self.status(24576, 48 * 2048, 48, 16, 1)['persisted_experts']['graph_counters'] if status is after else self.status(24576, 0, 0, 0, 0)['persisted_experts']['graph_counters']
            counters.update({'large_row_' + name: value for name, value in large.items()})
        coverage, errors = quality.coverage(before, after, case)
        self.assertEqual(errors, [])
        self.assertTrue(coverage['target_all_rows_full512'])
        self.assertEqual(coverage['expert_graph_counter_deltas']['gate_up_graph_calls'], 48)
        self.assertEqual(coverage['general_expert_graph_counter_deltas']['gate_up_graph_calls'], 144)
        self.assertEqual(coverage['small_row_target_graph_counter_deltas']['gate_up_graph_calls'], 96)
        self.assertEqual(coverage['small_row_target_graph_counter_deltas']['gate_up_graph_rows'], 48 * 8)
        broken = copy.deepcopy(after)
        broken['persisted_experts']['graph_counters'].pop('large_row_gate_up_graph_calls')
        self.assertTrue(quality.coverage(before, broken, case)[1])


if __name__ == '__main__':
    unittest.main()
