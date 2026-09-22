#!/usr/bin/env python3
"""CPU-only grading of small generated functions under a strict syntax policy.

This is a constrained evaluator, not an OS security sandbox. Only a validated
function definition is compiled. Its JSON test vectors are evaluated by a new
``python -I -S`` process with a private builtin namespace and resource limits.
No generated module statements, imports, attributes, or arbitrary calls run.
"""

from __future__ import annotations

import ast
import ctypes
import json
import keyword
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


PURE_BUILTINS = (
    "len", "sum", "min", "max", "abs", "range", "sorted", "list", "dict",
    "set", "enumerate", "zip", "all", "any", "int", "bool", "str",
)
MAX_SOURCE_BYTES = 32 * 1024
MAX_AST_NODES = 4096
MAX_INPUT_BYTES = 128 * 1024
MAX_OUTPUT_BYTES = 128 * 1024
MAX_TESTS = 128
MAX_VALUE_NODES = 50_000
MAX_VALUE_DEPTH = 64
MAX_INTEGER_BITS = 4096
CPU_SECONDS = 1
WALL_SECONDS = 2.0
ADDRESS_SPACE_BYTES = 128 * 1024 * 1024
DATA_BYTES = 64 * 1024 * 1024
FILE_DESCRIPTORS = 16
MEMORY_SAMPLE_SECONDS = 0.01

_RESERVED = frozenset(PURE_BUILTINS) | frozenset({
    "eval", "exec", "compile", "open", "input", "print", "globals", "locals",
    "vars", "dir", "getattr", "setattr", "delattr", "hasattr", "type", "object",
    "super", "help", "breakpoint", "classmethod", "staticmethod", "property",
    "bytes", "bytearray", "memoryview", "Exception", "BaseException",
})
_ALLOWED_AST = frozenset({
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Return, ast.Assign, ast.AugAssign, ast.If, ast.For, ast.While,
    ast.Break, ast.Continue, ast.Pass, ast.Expr,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Set,
    ast.Dict, ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.IfExp, ast.Call, ast.keyword,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
    ast.UAdd, ast.USub, ast.Not, ast.Invert, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot,
})
_FENCE = re.compile(r"\A```(?:python|py)?[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t]*\Z")
_IDENTIFIER = re.compile(r"\A[A-Za-z][A-Za-z0-9_]*\Z")


def _identifier(name: Any, *, builtin: bool = False) -> bool:
    return (
        type(name) is str and len(name) <= 128
        and _IDENTIFIER.fullmatch(name) is not None
        and "__" not in name and not keyword.iskeyword(name)
        and (name not in _RESERVED or (builtin and name in PURE_BUILTINS))
    )


def _json_value(value: Any) -> int:
    """Reject non-JSON Python values before JSON can coerce their types."""
    remaining = MAX_VALUE_NODES
    byte_count = 0

    def check(item: Any, depth: int) -> None:
        nonlocal remaining, byte_count
        remaining -= 1
        if remaining < 0 or depth > MAX_VALUE_DEPTH:
            raise ValueError("JSON value exceeds node/depth limit")
        byte_count += 1
        if byte_count > MAX_INPUT_BYTES:
            raise ValueError("JSON value exceeds byte limit")
        kind = type(item)
        if kind in (type(None), bool):
            return
        if kind is int:
            if item.bit_length() > MAX_INTEGER_BITS:
                raise ValueError("JSON integer exceeds bit limit")
            byte_count += len(str(item))
            return
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("JSON value contains a nonfinite number")
            return
        if kind is str:
            if len(item.encode("utf-8")) > MAX_OUTPUT_BYTES:
                raise ValueError("JSON string exceeds byte limit")
            byte_count += len(json.dumps(item, ensure_ascii=True).encode("utf-8"))
            if byte_count > MAX_INPUT_BYTES:
                raise ValueError("JSON value exceeds byte limit")
            return
        if kind is list:
            for child in item:
                check(child, depth + 1)
            return
        if kind is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("JSON object keys must be strings")
                check(key, depth + 1)
                check(child, depth + 1)
            return
        raise ValueError(f"Returned value is not a JSON type: {kind.__name__}")

    check(value, 0)
    return byte_count


def _prepare(text: str, expected: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if type(text) is not str:
        raise ValueError("Generated answer must be text")
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Generated answer exceeds source byte limit")
    if type(expected) is not dict:
        raise ValueError("Function task specification must be an object")
    name, arguments, tests = (
        expected.get("function"), expected.get("argument_names"), expected.get("tests"),
    )
    if not _identifier(name):
        raise ValueError("Task function name is invalid or reserved")
    if (
        type(arguments) is not list or len(arguments) > 32
        or not all(_identifier(arg) for arg in arguments)
        or len(set(arguments)) != len(arguments)
    ):
        raise ValueError("Task argument names are invalid, duplicate, or reserved")
    if type(tests) is not list or not 1 <= len(tests) <= MAX_TESTS:
        raise ValueError("Task must have 1 to 128 JSON test vectors")
    require_fresh_result = expected.get("require_fresh_result", False)
    if type(require_fresh_result) is not bool:
        raise ValueError("require_fresh_result must be a boolean")
    clean_tests = []
    vector_bytes = len(text.encode("utf-8"))
    for index, test in enumerate(tests):
        if type(test) is not dict or "arguments" not in test or "output" not in test:
            raise ValueError(f"Test {index} must contain arguments and output")
        if type(test["arguments"]) is not list or len(test["arguments"]) != len(arguments):
            raise ValueError(f"Test {index} argument count differs from the signature")
        vector_bytes += _json_value(test["arguments"]) + _json_value(test["output"])
        if vector_bytes > MAX_INPUT_BYTES:
            raise ValueError("Evaluation input exceeds byte limit")
        clean_tests.append({"arguments": test["arguments"], "output": test["output"]})

    source = text.strip()
    fence = _FENCE.fullmatch(source)
    if fence:
        source = fence.group(1)
    try:
        tree = ast.parse(source, mode="exec", type_comments=True)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError(f"Generated function does not parse: {type(exc).__name__}") from None
    if len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef:
        raise ValueError("Answer must contain exactly one synchronous function definition")
    function = tree.body[0]
    if function.name != name:
        raise ValueError("Generated function name differs from the requested function")
    signature = function.args
    if (
        function.decorator_list or function.returns is not None
        or getattr(function, "type_params", []) or function.type_comment
        or signature.posonlyargs or signature.kwonlyargs or signature.vararg
        or signature.kwarg or signature.defaults or signature.kw_defaults
        or [arg.arg for arg in signature.args] != arguments
        or any(arg.annotation is not None or arg.type_comment for arg in signature.args)
        or tree.type_ignores
    ):
        raise ValueError("Function signature must use only the requested plain arguments")
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ValueError("Generated function exceeds AST node limit")
    local_names = set(arguments)
    for node in nodes:
        if type(node) not in _ALLOWED_AST:
            raise ValueError(f"Forbidden generated syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise ValueError("Nested function definitions are forbidden")
        if getattr(node, "type_comment", None):
            raise ValueError("Type comments are forbidden")
        if isinstance(node, ast.Name):
            if not _identifier(node.id, builtin=isinstance(node.ctx, ast.Load)):
                raise ValueError("Generated identifier is invalid or reserved")
            if isinstance(node.ctx, ast.Store):
                local_names.add(node.id)
        elif isinstance(node, ast.Constant):
            _json_value(node.value)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in PURE_BUILTINS:
                raise ValueError("Generated calls must name an allowed pure builtin")
        elif isinstance(node, ast.keyword):
            if node.arg is None or not _identifier(node.arg):
                raise ValueError("Expanded or reserved call keywords are forbidden")
        elif isinstance(node, ast.comprehension) and node.is_async:
            raise ValueError("Asynchronous comprehensions are forbidden")
        elif isinstance(node, ast.Dict) and any(key is None for key in node.keys):
            raise ValueError("Dictionary expansions are forbidden")
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in local_names and node.id not in PURE_BUILTINS:
                raise ValueError("Generated function loads an undeclared identifier")
    specification = {"function": name, "argument_names": arguments, "tests": clean_tests,
                     "require_fresh_result": require_fresh_result}
    return source, specification


# This trusted bootstrap receives data only on stdin. The source it compiles has
# already passed _prepare; it contains one plain function definition and no
# executable module statements. The generated namespace cannot access bootstrap
# imports, its output streams, or its original builtin namespace.
_WORKER = r'''
import builtins
import json
import math
import os
import resource
import sys

OUTPUT_LIMIT = 131072
LIMITS = {
    "cpu_seconds": (resource.RLIMIT_CPU, 1),
    "address_space_bytes": (resource.RLIMIT_AS, 134217728),
    "data_bytes": (resource.RLIMIT_DATA, 67108864),
    "file_bytes": (resource.RLIMIT_FSIZE, 0),
    "file_descriptors": (resource.RLIMIT_NOFILE, 16),
    "core_bytes": (resource.RLIMIT_CORE, 0),
}

memory_mode = "absolute-address-space-and-data-limits-plus-parent-resident-monitor"
if sys.platform == "darwin":
    # macOS Python reserves hundreds of GiB of virtual address space. Smaller
    # absolute AS/DATA caps fail even for an empty interpreter. Bound additional
    # virtual mappings here; the parent separately enforces a sampled RSS cap.
    import ctypes
    info = ctypes.create_string_buffer(96)
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                   ctypes.c_void_p, ctypes.c_int]
    libproc.proc_pidinfo.restype = ctypes.c_int
    if libproc.proc_pidinfo(os.getpid(), 4, 0, info, 96) != 96:
        raise RuntimeError("Cannot inspect baseline virtual memory")
    baseline_virtual = int.from_bytes(info.raw[:8], sys.byteorder)
    LIMITS["address_space_bytes"] = (resource.RLIMIT_AS, baseline_virtual + 134217728)
    LIMITS["data_bytes"] = (resource.RLIMIT_DATA, baseline_virtual + 67108864)
    memory_mode = "virtual-headroom-limits-plus-parent-resident-monitor"

def emit(document):
    encoded = json.dumps(document, ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(encoded) > OUTPUT_LIMIT:
        encoded = b'{"ok":false,"error":"Evaluation output exceeds byte limit"}'
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()

def valid_json(value):
    remaining = [50000]
    def check(item, depth):
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 64:
            raise ValueError("Returned JSON value exceeds node/depth limit")
        kind = type(item)
        if kind in (type(None), bool):
            return
        if kind is int:
            if item.bit_length() > 4096:
                raise ValueError("Returned integer exceeds bit limit")
            return
        if kind is float:
            if not math.isfinite(item):
                raise ValueError("Returned value contains a nonfinite number")
            return
        if kind is str:
            if len(item.encode("utf-8")) > OUTPUT_LIMIT:
                raise ValueError("Returned string exceeds byte limit")
            return
        if kind is list:
            for child in item:
                check(child, depth + 1)
            return
        if kind is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("Returned JSON object keys must be strings")
                check(key, depth + 1)
                check(child, depth + 1)
            return
        raise ValueError("Returned value is not a JSON type: " + kind.__name__)
    check(value, 0)

def equal_json(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(equal_json(a, b) for a, b in zip(left, right))
    if type(left) is dict:
        return left.keys() == right.keys() and all(equal_json(left[key], right[key]) for key in left)
    return left == right

def mutable_input_ids(arguments):
    identities = set()
    originals = []
    def visit(value):
        if type(value) is list:
            identities.add(id(value))
            originals.append(value)
            for child in value:
                visit(child)
        elif type(value) is dict:
            identities.add(id(value))
            originals.append(value)
            for child in value.values():
                visit(child)
    # The wrapper arguments list belongs to the runner, not the function.
    for argument in arguments:
        visit(argument)
    # Retain originals through the call so freed/replaced input objects cannot
    # have their IDs recycled into an unrelated newly allocated result.
    return identities, originals

try:
    applied = {}
    for name, (kind, bound) in LIMITS.items():
        resource.setrlimit(kind, (bound, bound))
        if resource.getrlimit(kind) != (bound, bound):
            raise RuntimeError("Resource limit could not be applied: " + name)
        applied[name] = bound
    raw = sys.stdin.buffer.read(131073)
    if len(raw) > 131072:
        raise ValueError("Evaluation input exceeds byte limit")
    payload = json.loads(raw)
    allowed = {name: getattr(builtins, name) for name in (
        "len", "sum", "min", "max", "abs", "range", "sorted", "list", "dict",
        "set", "enumerate", "zip", "all", "any", "int", "bool", "str",
    )}
    namespace = {"__builtins__": allowed}
    exec(compile(payload["source"], "<generated-function>", "exec", dont_inherit=True),
         namespace, namespace)
    function = namespace[payload["specification"]["function"]]
    results = []
    output_bytes = 0
    for index, test in enumerate(payload["specification"]["tests"]):
        try:
            snapshot = json.loads(json.dumps(test["arguments"], ensure_ascii=True,
                                            allow_nan=False, separators=(",", ":")))
            input_ids, original_inputs = mutable_input_ids(test["arguments"])
            actual = function(*test["arguments"])
            if not equal_json(test["arguments"], snapshot):
                raise ValueError("Function mutated its input arguments")
            if payload["specification"]["require_fresh_result"] and id(actual) in input_ids:
                raise ValueError("Result aliases a mutable input object; a fresh result is required")
            valid_json(actual)
            output_bytes += len(json.dumps(actual, ensure_ascii=True, allow_nan=False,
                                          separators=(",", ":")).encode("utf-8"))
            if output_bytes > OUTPUT_LIMIT:
                raise ValueError("Returned outputs exceed total byte limit")
            passed = equal_json(actual, test["output"])
            row = {"index": index, "passed": passed}
            if not passed:
                row["error"] = "Returned JSON type or value differs from expected output"
            results.append(row)
        except BaseException as error:
            message = str(error)[:300] if isinstance(error, ValueError) else type(error).__name__
            results.append({"index": index, "passed": False, "error": "Evaluation failed: " + message})
    emit({"ok": True, "results": results, "applied_limits": applied,
          "returned_output_bytes": output_bytes, "memory_enforcement_mode": memory_mode})
except BaseException as error:
    emit({"ok": False, "error": "Isolated evaluation failed: " + type(error).__name__})
'''


def _memory_reader():
    """Create a native RSS reader without spawning repeated helper processes."""
    if sys.platform == "darwin":
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        function = library.proc_pidinfo
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                             ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_int

        def read(pid: int) -> int | None:
            # proc_taskinfo: six uint64_t followed by twelve int32_t (96 B).
            info = ctypes.create_string_buffer(96)
            if function(pid, 4, 0, info, 96) != 96:
                return None
            return int.from_bytes(info.raw[8:16], sys.byteorder)

        return read
    if sys.platform.startswith("linux"):
        def read(pid: int) -> int | None:
            try:
                with open(f"/proc/{pid}/status", encoding="ascii") as status:
                    for line in status:
                        if line.startswith("VmRSS:"):
                            fields = line.split()
                            if len(fields) == 3 and fields[2] == "kB":
                                return int(fields[1]) * 1024
            except (OSError, ValueError):
                return None
            return None

        return read
    raise OSError("Native resident-memory monitor unavailable on this platform")


def _kill_and_reap(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()


def _exchange(process: subprocess.Popen, payload: bytes, memory_reader):
    """Bound both output pipes while applying the wall and resident watchdogs."""
    deadline = time.monotonic() + WALL_SECONDS
    output = {"stdout": bytearray(), "stderr": bytearray()}
    metadata = {"resident_limit_bytes": ADDRESS_SPACE_BYTES,
                "resident_sample_seconds": MEMORY_SAMPLE_SECONDS,
                "peak_observed_resident_bytes": 0,
                "resident_samples": 0, "polling_can_overshoot": True}
    position = 0
    error = None
    with selectors.DefaultSelector() as selector:
        for pipe, name, event in [(process.stdin, "stdin", selectors.EVENT_WRITE),
                                  (process.stdout, "stdout", selectors.EVENT_READ),
                                  (process.stderr, "stderr", selectors.EVENT_READ)]:
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, event, name)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = "Isolated evaluation exceeded the two-second wall limit"
                break
            resident = memory_reader(process.pid)
            if resident is None:
                if process.poll() is None:
                    error = "Isolated evaluation resident-memory monitor failed"
                    break
            else:
                metadata["resident_samples"] += 1
                metadata["peak_observed_resident_bytes"] = max(
                    metadata["peak_observed_resident_bytes"], resident)
                if resident > ADDRESS_SPACE_BYTES:
                    error = "Isolated evaluation exceeded the resident-memory limit"
                    break
            for key, _ in selector.select(min(MEMORY_SAMPLE_SECONDS, remaining)):
                pipe, name = key.fileobj, key.data
                if name == "stdin":
                    try:
                        position += os.write(pipe.fileno(), payload[position:position + 8192])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        position = len(payload)
                    if position == len(payload):
                        selector.unregister(pipe)
                        pipe.close()
                else:
                    try:
                        chunk = os.read(pipe.fileno(), 8192)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(pipe)
                        pipe.close()
                    elif sum(len(value) for value in output.values()) + len(chunk) > MAX_OUTPUT_BYTES:
                        error = "Isolated evaluation exceeded the output byte limit"
                        break
                    else:
                        output[name].extend(chunk)
            if error:
                break
    if error:
        _kill_and_reap(process)
    else:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            error = "Isolated evaluation exceeded the two-second wall limit"
            _kill_and_reap(process)
        else:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                error = "Isolated evaluation exceeded the two-second wall limit"
                _kill_and_reap(process)
    for pipe in [process.stdin, process.stdout, process.stderr]:
        if not pipe.closed:
            pipe.close()
    return bytes(output["stdout"]), bytes(output["stderr"]), metadata, error


def grade_function(text: str, expected: dict[str, Any]) -> dict[str, Any]:
    """Return grading details; failures are ordinary data, never a passing run.

    ``expected`` contains ``function``, ordered ``argument_names``, and nonempty
    ``tests`` of ``{"arguments": [...], "output": <JSON value>}``. A lone code
    fence is accepted. All vectors share a one-second CPU/two-second wall budget.
    Results require identical JSON Python types recursively (including bool vs
    int and int vs float); tuples, sets, non-string object keys and NaN fail.
    Inputs must remain unchanged. ``require_fresh_result=True`` additionally
    rejects returning any original mutable input object (including nested ones).
    macOS AS/DATA limits bound additional virtual mappings; a native parent RSS
    watchdog polls at 10 ms and can overshoot between samples. This evaluator
    cannot establish general code safety or mathematical purity.
    """
    result: dict[str, Any] = {
        "pass": False, "errors": [], "test_count": 0, "passed_tests": 0,
        "results": [], "cpu_only": True, "execution_started": False,
        "requires_unchanged_inputs": True,
        "bounded_execution": {
            "isolated_python_flags": ["-I", "-S"], "cpu_seconds": CPU_SECONDS,
            "parent_wall_seconds": WALL_SECONDS,
            "resident_limit_bytes": ADDRESS_SPACE_BYTES,
            "address_space_budget_bytes": ADDRESS_SPACE_BYTES, "data_budget_bytes": DATA_BYTES,
            "file_bytes": 0, "file_descriptors": FILE_DESCRIPTORS, "core_bytes": 0,
            "input_bytes": MAX_INPUT_BYTES, "output_bytes": MAX_OUTPUT_BYTES,
            "os_security_sandbox": False,
        },
    }
    try:
        source, specification = _prepare(text, expected)
        result["test_count"] = len(specification["tests"])
        result["requires_fresh_result"] = specification["require_fresh_result"]
        payload = json.dumps({"source": source, "specification": specification},
                             ensure_ascii=True, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_INPUT_BYTES:
            raise ValueError("Evaluation input exceeds byte limit")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        result["errors"] = [str(error)[:400]]
        return result
    try:
        memory_reader = _memory_reader()
        # No preexec_fn: limits are applied by the fresh interpreter before it
        # parses/compiles generated source. Avoid forking parent model state.
        with tempfile.TemporaryDirectory(prefix="splash-function-grade-") as folder:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _WORKER],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=folder, env={"LANG": "C", "LC_ALL": "C"}, close_fds=True,
                start_new_session=True,
            )
            result["execution_started"] = True
            try:
                stdout, stderr, monitoring, error = _exchange(process, payload, memory_reader)
            finally:
                if process.poll() is None:
                    _kill_and_reap(process)
                for pipe in [process.stdin, process.stdout, process.stderr]:
                    if not pipe.closed:
                        pipe.close()
            result["memory_monitoring"] = monitoring
            if error:
                result["errors"] = [error]
                return result
        if process.returncode != 0:
            result["errors"] = [f"Isolated evaluation failed or exceeded resource limits (exit {process.returncode})"]
            return result
        if stderr or len(stdout) > MAX_OUTPUT_BYTES:
            result["errors"] = ["Isolated evaluation produced unexpected stderr or exceeded output byte limit"]
            return result
        document = json.loads(stdout)
        if not isinstance(document, dict) or document.get("ok") is not True:
            result["errors"] = [document.get("error", "Isolated evaluation protocol failed") if isinstance(document, dict) else "Isolated evaluation protocol failed"]
            return result
        rows = document.get("results")
        if (
            type(rows) is not list or len(rows) != result["test_count"]
            or any(type(row) is not dict or type(row.get("index")) is not int
                   or row["index"] != index or type(row.get("passed")) is not bool
                   for index, row in enumerate(rows))
        ):
            result["errors"] = ["Isolated evaluation returned incomplete or malformed test results"]
            return result
        result["results"] = rows
        result["passed_tests"] = sum(row["passed"] for row in rows)
        result["errors"] = [f"Test {row['index']}: {row.get('error', 'incorrect output')}"
                            for row in rows if not row["passed"]]
        result["applied_limits"] = document.get("applied_limits")
        result["memory_enforcement_mode"] = document.get("memory_enforcement_mode")
        result["returned_output_bytes"] = document.get("returned_output_bytes")
        result["pass"] = not result["errors"]
        return result
    except (OSError, ValueError, TypeError, AttributeError, UnicodeError, RecursionError) as error:
        result["errors"] = [f"Isolated evaluator unavailable or malformed: {type(error).__name__}"]
        return result


def grader_errors(text: str, expected: dict[str, Any]) -> list[str]:
    """Integration entry point: empty only when every test vector passed."""
    return grade_function(text, expected)["errors"]
