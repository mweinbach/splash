"""CPU-only integration tests for the restricted generated-function evaluator."""

import os
import subprocess
import sys
import time
import unittest
from unittest import mock

from dev.benchmarks import prefill4k_attribution_quality_python as quality


def task(name="sum_even", arguments=None, tests=None):
    return {
        "function": name,
        "argument_names": arguments if arguments is not None else ["numbers"],
        "tests": tests if tests is not None else [
            {"arguments": [[]], "output": 0},
            {"arguments": [[-4, -3, -2, 0, 2, 2, 3]], "output": -2},
            {"arguments": [[3, 5, 7]], "output": 0},
        ],
    }


class RestrictedPythonGraderTests(unittest.TestCase):
    def test_sum_comprehension_handles_empty_duplicates_and_negatives(self):
        result = quality.grade_function(
            "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)", task())
        self.assertTrue(result["pass"], result["errors"])
        self.assertEqual(result["passed_tests"], 3)
        self.assertEqual(result["applied_limits"]["cpu_seconds"], 1)
        self.assertEqual(result["applied_limits"]["file_bytes"], 0)
        self.assertEqual(result["applied_limits"]["file_descriptors"], 16)
        self.assertEqual(result["applied_limits"]["core_bytes"], 0)
        self.assertGreater(result["memory_monitoring"]["resident_samples"], 0)
        self.assertFalse(result["bounded_execution"]["os_security_sandbox"])

    def test_fenced_loop_function_is_supported(self):
        source = "```python\ndef sum_even(numbers):\n    result = 0\n    for number in numbers:\n        if number % 2 == 0:\n            result += number\n    return result\n```"
        self.assertEqual(quality.grader_errors(source, task()), [])

    def test_dictionary_comprehension_and_string_keys(self):
        expected = task("frequencies", tests=[
            {"arguments": [[]], "output": {}},
            {"arguments": [[-2, 1, -2, 0, 1]], "output": {"-2": 2, "0": 1, "1": 2}},
        ])
        source = "def frequencies(numbers):\n    return {str(n): sum(1 for item in numbers if item == n) for n in sorted(set(numbers))}"
        self.assertEqual(quality.grader_errors(source, expected), [])

    def test_enumerate_slice_comprehension_preserves_first_occurrence(self):
        expected = task("unique", tests=[
            {"arguments": [[]], "output": []},
            {"arguments": [[-1, 2, -1, 0, 2]], "output": [-1, 2, 0]},
        ])
        source = "def unique(numbers):\n    return [n for index, n in enumerate(numbers) if n not in numbers[:index]]"
        self.assertEqual(quality.grader_errors(source, expected), [])

    def test_two_arguments_zip_and_builtin_keyword(self):
        expected = task("dot", arguments=["left", "right"], tests=[
            {"arguments": [[], []], "output": 0},
            {"arguments": [[-2, 3, 3], [4, -1, 2]], "output": -5},
        ])
        source = "def dot(left, right):\n    return sum(a * b for a, b in zip(left, right))"
        self.assertEqual(quality.grader_errors(source, expected), [])
        expected = task("unique_desc", tests=[
            {"arguments": [[-1, 2, -1, 0, 2]], "output": [2, 0, -1]},
        ])
        self.assertEqual(quality.grader_errors(
            "def unique_desc(numbers):\n    return sorted(set(numbers), reverse=True)", expected), [])

    def test_incorrect_but_allowed_function_fails_vectors(self):
        result = quality.grade_function(
            "def sum_even(numbers):\n    return sum(n for n in numbers if n > 0 and n % 2 == 0)", task())
        self.assertFalse(result["pass"])
        self.assertEqual(result["passed_tests"], 2)
        self.assertIn("Test 1", result["errors"][0])

    def test_recursive_exact_json_type_comparison(self):
        cases = [
            ("True", 1), ("1", True), ("1.0", 1), ("1", 1.0),
            ("[True]", [1]), ("{'answer': True}", {"answer": 1}),
        ]
        for expression, expected_output in cases:
            with self.subTest(expression=expression, expected=expected_output):
                expected = task(tests=[{"arguments": [[]], "output": expected_output}])
                self.assertTrue(quality.grader_errors(
                    "def sum_even(numbers):\n    return " + expression, expected))

    def test_json_serialization_cannot_coerce_unsupported_return_types(self):
        for expression in ("(1, 2)", "{1, 2}", "{1: 'answer'}", "range(2)", "sum"):
            with self.subTest(expression=expression):
                errors = quality.grader_errors(
                    "def sum_even(numbers):\n    return " + expression, task())
                self.assertTrue(errors)
                self.assertIn("JSON", errors[0])

    def test_unsafe_and_unsupported_syntax_is_rejected_before_spawn(self):
        sources = [
            "import os\ndef sum_even(numbers):\n    return 0",
            "print('hello')\ndef sum_even(numbers):\n    return 0",
            "def sum_even(numbers):\n    return numbers.count(2)",
            "def sum_even(numbers):\n    return open('/tmp/quality')",
            "def sum_even(numbers):\n    return eval('1')",
            "def sum_even(numbers):\n    return sum_even(numbers)",
            "def sum_even(numbers):\n    return __builtins__",
            "def sum_even(numbers):\n    _hidden = 1\n    return _hidden",
            "def sum_even(numbers):\n    sum = 1\n    return sum",
            "@list\ndef sum_even(numbers):\n    return 0",
            "def sum_even(numbers=[]):\n    return 0",
            "def sum_even(numbers: list) -> int:\n    return 0",
            "def sum_even(*numbers):\n    return 0",
            "def sum_even(numbers, /):\n    return 0",
            "def sum_even(*, numbers):\n    return 0",
            "def sum_even(numbers):\n    def inner():\n        return 0\n    return 0",
            "def sum_even(numbers):\n    class Inner:\n        pass\n    return 0",
            "def sum_even(numbers):\n    return (lambda: 0)()",
            "async def sum_even(numbers):\n    return 0",
            "def sum_even(numbers):\n    global outside\n    return 0",
            "def sum_even(numbers):\n    nonlocal outside\n    return 0",
            "def sum_even(numbers):\n    with numbers:\n        return 0",
            "def sum_even(numbers):\n    try:\n        return 0\n    except:\n        return 0",
            "def sum_even(numbers):\n    raise numbers",
            "def sum_even(numbers):\n    del numbers[0]\n    return 0",
            "def sum_even(numbers):\n    yield 0",
            "def sum_even(numbers):\n    return sum(*numbers)",
            "def sum_even(numbers):\n    return sorted(numbers, **{})",
            "def sum_even(numbers):\n    return dict(__class__=1)",
            "def sum_even(numbers):\n    return {**{}}",
            "def sum_even(numbers):\n    return f'{numbers}'",
            "def sum_even(numbers):\n    match numbers:\n        case list():\n            return 0",
            "def sum_even(numbers):\n    return missing_name",
            "def sum_even(numbers):\n    return b'bytes'",
            "def sum_even(numbers):\n    return 1j",
            "def sum_even(numbers):\n    return ...",
            "def sum_even(numbers):\n    return 1e309",
            "def other(numbers):\n    return 0",
            "def sum_even(numbers):\n    return 0\ndef other(numbers):\n    return 0",
            "Here is your code:\n```python\ndef sum_even(numbers):\n    return 0\n```",
            "def sum_even(numbers):\n    value = 0  # type: int\n    return value",
        ]
        with mock.patch.object(quality.subprocess, "Popen", side_effect=AssertionError("Rejected syntax must not execute")):
            for source in sources:
                with self.subTest(source=source):
                    result = quality.grade_function(source, task())
                    self.assertFalse(result["pass"])
                    self.assertFalse(result["execution_started"])
                    self.assertTrue(result["errors"])

    def test_invalid_task_vectors_and_large_source_do_not_execute(self):
        invalid = [
            task(arguments=["numbers", "numbers"]), task(arguments=["__globals__"]),
            task("open"), task(tests=[]),
            task(tests=[{"arguments": [], "output": 0}]),
            task(tests=[{"arguments": [[]], "output": {1: "number key"}}]),
            task(tests=[{"arguments": [[float("nan")]], "output": 0}]),
            task(tests=[{"arguments": [[]], "output": (1, 2)}]),
            task(tests=[{"arguments": [["x" * quality.MAX_INPUT_BYTES]], "output": 0}]),
        ]
        with mock.patch.object(quality.subprocess, "Popen", side_effect=AssertionError("Invalid task must not execute")):
            for expected in invalid:
                with self.subTest(expected=expected):
                    self.assertTrue(quality.grader_errors("def sum_even(numbers):\n    return 0", expected))
            self.assertTrue(quality.grader_errors("#" * (quality.MAX_SOURCE_BYTES + 1), task()))

    def test_runtime_error_is_a_failed_case(self):
        result = quality.grade_function("def sum_even(numbers):\n    return 1 // 0", task())
        self.assertFalse(result["pass"])
        self.assertTrue(all("ZeroDivisionError" in row["error"] for row in result["results"]))

    def test_correct_return_cannot_hide_input_mutation(self):
        expected = task(tests=[{"arguments": [[-4, -2, 2]], "output": -4}])
        source = "def sum_even(numbers):\n    answer = sum(numbers)\n    numbers[0] = 0\n    return answer"
        result = quality.grade_function(source, expected)
        self.assertFalse(result["pass"])
        self.assertTrue(result["requires_unchanged_inputs"])
        self.assertIn("mutated its input arguments", result["errors"][0])

    def test_fresh_list_contract_rejects_alias_but_accepts_new_equal_list(self):
        expected = task("clamp_values", arguments=["numbers", "low", "high"], tests=[
            {"arguments": [[-2, 0, 2], -3, 3], "output": [-2, 0, 2]},
        ])
        source = "def clamp_values(numbers, low, high):\n    return numbers"
        self.assertEqual(quality.grader_errors(source, expected), [])
        expected["require_fresh_result"] = True
        result = quality.grade_function(source, expected)
        self.assertFalse(result["pass"])
        self.assertTrue(result["requires_fresh_result"])
        self.assertIn("aliases a mutable input object", result["errors"][0])
        self.assertEqual(quality.grader_errors(
            "def clamp_values(numbers, low, high):\n    return [max(low, min(high, n)) for n in numbers]", expected), [])

    def test_infinite_loop_is_cpu_bounded(self):
        started = time.monotonic()
        result = quality.grade_function("def sum_even(numbers):\n    while True:\n        pass", task())
        self.assertFalse(result["pass"])
        self.assertTrue(result["execution_started"])
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertTrue(any("resource limits" in error or "wall limit" in error for error in result["errors"]))

    def test_output_total_is_bounded(self):
        expected = task(tests=[{"arguments": [[]], "output": []}])
        result = quality.grade_function("def sum_even(numbers):\n    return list(range(40000))", expected)
        self.assertFalse(result["pass"])
        self.assertTrue(any("byte limit" in error for error in result["errors"]))

    def test_child_isolated_and_cannot_inherit_environment_or_file_descriptors(self):
        original_popen = subprocess.Popen
        captured = []
        def observed_popen(*args, **kwargs):
            captured.append((args, kwargs))
            return original_popen(*args, **kwargs)
        with mock.patch.object(quality.subprocess, "Popen", side_effect=observed_popen), \
                mock.patch.dict(os.environ, {"PYTHONPATH": "/tmp/should-not-load", "SPLASH_GRADER_TEST_SECRET": "must-not-inherit"}):
            self.assertEqual(quality.grader_errors("def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)", task()), [])
        args, kwargs = captured[0]
        self.assertEqual(args[0][:4], [sys.executable, "-I", "-S", "-c"])
        self.assertEqual(kwargs["env"], {"LANG": "C", "LC_ALL": "C"})
        self.assertTrue(kwargs["close_fds"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertNotIn("preexec_fn", kwargs)
        self.assertNotIn("shell", kwargs)

    def test_parent_enforces_resident_memory_limit_without_large_allocations(self):
        with mock.patch.object(quality, "_memory_reader", return_value=lambda pid: quality.ADDRESS_SPACE_BYTES + 1):
            result = quality.grade_function("def sum_even(numbers):\n    return 0", task())
        self.assertFalse(result["pass"])
        self.assertIn("resident-memory limit", result["errors"][0])
        self.assertGreater(result["memory_monitoring"]["peak_observed_resident_bytes"], quality.ADDRESS_SPACE_BYTES)

    def test_memory_monitor_exception_kills_reaps_and_closes_child(self):
        original_popen = subprocess.Popen
        processes = []
        def observed_popen(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            return process
        def broken_monitor(pid):
            raise OSError("Monitor unavailable")
        with mock.patch.object(quality.subprocess, "Popen", side_effect=observed_popen), \
                mock.patch.object(quality, "_memory_reader", return_value=broken_monitor):
            result = quality.grade_function("def sum_even(numbers):\n    while True:\n        pass", task())
        self.assertFalse(result["pass"])
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(all(pipe.closed for pipe in [processes[0].stdin, processes[0].stdout, processes[0].stderr]))


if __name__ == "__main__":
    unittest.main()
