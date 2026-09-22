"""Private installed-MLX backend reference; CPU preflight never imports MLX.

Flash needs the installed qwen4_exp compatibility model, which is not pristine
upstream MLX-LM. Optional oMLX HC/QSA/eager serving routes are disabled. Original
checkpoint coefficients/config remain untouched. Root alone uses --run-gpu.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time

import flash_mlx_reference as source_reference

APP = source_reference.APP
SITE = APP / "Python/framework-mlx-base/lib/python3.11/site-packages"
PYROOT = APP / "Python/cpython-3.11"
DEFAULT_PROMPT = source_reference.PROJECT / "build/release/flash/prefill4k-fixture/code2048.tokens.json"
CONTROLS = {
    "OMLX_QWEN4_HC_FUSED": "0",
    "OMLX_QWEN4_HC_HYBRID": "0",
    "OMLX_QWEN4_EAGER_DISPATCH": "0",
    "OMLX_QWEN4_QSA_GATHERED_VERIFY": "0",
}


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cpu_preflight(args) -> tuple[list[int], dict, dict]:
    tokens, watched, report = source_reference.preflight(args.model, args.prompt, args.bundle)
    if len(tokens) != args.expected_prompt_tokens:
        raise ValueError(f"expected {args.expected_prompt_tokens} frozen tokens, found {len(tokens)}")
    config = json.loads((args.model / "config.json").read_bytes())
    files = [
        source_reference.VENDOR / "vendor/mlx_vlm/models/qwen4_exp" / name
        for name in ("language.py", "qwen4_exp.py", "config.py", "cache.py", "qsa_fast.py", "hc_fused.py", "hc_projection.py")
    ] + [SITE / rel for rel in (
        "mlx_vlm/models/qwen3_5/language.py", "mlx_vlm/models/qwen3_5/gated_delta.py",
        "mlx_vlm/models/qwen3_5_moe/language.py", "mlx_vlm/models/base.py",
        "mlx_vlm/utils.py", "mlx_lm/models/switch_layers.py", "mlx_lm/models/base.py", "mlx_lm/models/gated_delta.py",
    )]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"installed model dependencies missing: {missing}")
    source_hashes = {str(path): source_reference.sha256(path) for path in files}
    for path in files:
        watched[str(path)] = source_reference.snapshot(path)
    versions = {}
    wanted = {"mlx", "mlx-lm", "mlx-vlm", "numpy", "safetensors", "transformers"}
    for distribution in importlib.metadata.distributions(path=[str(SITE)]):
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name in wanted:
            versions[name] = distribution.version
    report.update({
        "schema": "splash-prefill4k-installed-mlx-backend-reference-v1",
        "reference_label": "installed bundled MLX backend with vendored qwen4_exp compatibility model",
        "pristine_upstream_mlx_lm_model": False,
        "canonical_transformer_quality_qualified": False,
        "normalization_caveat": "Installed Qwen3.5 GDN forward does not call the vendor Qwen4 _normalize_qk override; this is the installed implementation's performance, not proven canonical Flash quality.",
        "preflight_python_executable": sys.executable,
        "planned_python_executable": str(PYROOT / "bin/python3"),
        "installed_distribution_versions": versions,
        "native_mlx_backend_environment": CONTROLS,
        "optional_vendored_gathered_qsa_disabled_in_process": True,
        "installed_model_dependency_sha256": source_hashes,
        "benchmark_sha256": source_reference.sha256(Path(__file__)),
        "expected_prompt_tokens": args.expected_prompt_tokens,
        "vocabulary_size": config["text_config"]["vocab_size"],
        "prefill_chunk_rows": args.prefill_rows,
        "ple_mode": args.ple_mode,
        "dense_ple_table_materialized": args.ple_mode == "resident",
        "warmup_requests": 1,
        "measured_requests": args.samples,
        "maximum_output_tokens": args.max_tokens,
        "temperature": 0,
        "reasoning": "off requested; frozen token IDs supplied without retemplating",
        "prefix_cache_enabled": False,
        "serving_cache_enabled": False,
        "fresh_request_local_cache": True,
        "lm_head_policy": args.lm_head_policy,
        "lm_head_policy_note": "last slices hidden states inside a harness wrapper around the original loaded LM-head module; coefficient/operator unchanged, one final prompt logit row as in Splash",
        "timing_scope": "synchronized model forward including fresh KV/GDN/PLE cache priming and final logits; excludes loading, first warmup, sampling, finite-state validation and reporting; no GPU-only timing claim",
        "official_mlx_release_comparison": {
            "latest_release_verified": "0.32.2",
            "m5_ultra_commit": "2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99",
            "official_v0_32_2_contains_commit": False,
            "installed_binary_source_commit_proven": False,
            "release_source": "https://github.com/ml-explore/mlx/releases/tag/v0.32.2",
            "tag_matmul_source": "https://raw.githubusercontent.com/ml-explore/mlx/v0.32.2/mlx/backend/metal/matmul.cpp",
        },
        "gpu_executed": False,
        "valid": True,
    })
    report["source_verification"]["full_weight_shard_reads_by_preflight"] = False
    source_reference.require_unchanged(watched)
    if "mlx.core" in sys.modules:
        raise RuntimeError("CPU preflight unexpectedly imported MLX core")
    report["cpu_preflight_imported_mlx_core"] = False
    return tokens, watched, report


@contextlib.contextmanager
def sanitize_metadata_without_engine(model_dir: Path):
    """The installed sanitize-before-quantize fix, without importing a serving engine."""
    import safetensors
    original = safetensors.safe_open
    targets = {path.resolve() for path in model_dir.glob("*.safetensors")}

    class Handle:
        def __init__(self, inner):
            self.inner = inner
        def __enter__(self):
            self.inner.__enter__()
            return self
        def __exit__(self, *args):
            return self.inner.__exit__(*args)
        def __getattr__(self, key):
            return getattr(self.inner, key)
        def metadata(self):
            result = self.inner.metadata()
            if isinstance(result, dict) and result.get("format") == "mlx":
                result = dict(result)
                result.pop("format", None)
            return result

    def readonly_metadata(filename, *args, **kwargs):
        handle = original(filename, *args, **kwargs)
        return Handle(handle) if Path(filename).resolve() in targets else handle

    safetensors.safe_open = readonly_metadata
    try:
        yield
    finally:
        safetensors.safe_open = original


def run_gpu(args, tokens, watched, report):
    if not args.run_gpu:
        raise RuntimeError("model/GPU execution needs --run-gpu from Root")
    for name, value in CONTROLS.items():
        os.environ[name] = value
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_vlm
    import mlx_vlm.models
    import mlx_vlm.utils as utils
    from mlx.utils import tree_flatten

    # Register only the compatibility namespace. No serving/performance patcher.
    vendor = source_reference.VENDOR / "vendor/mlx_vlm"
    mlx_vlm.__path__.append(str(vendor))
    mlx_vlm.models.__path__.append(str(vendor / "models"))
    language_module = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
    language_module.configure_ple_runtime(args.model, mode=args.ple_mode)
    language_module.configure_mtp_runtime(args.model, enabled=False)
    if language_module.get_ple_runtime_mode() != args.ple_mode or language_module.get_mtp_runtime().enabled:
        raise RuntimeError("PLE/MTP reference binding differs")
    disabled = {}
    for name in ("_gathered_text_prefill_eligible", "_gathered_text_decode_eligible", "_gathered_text_verify_eligible"):
        previous = getattr(language_module.Qwen4ExpAttention, name)
        disabled[name] = previous.__module__ + "." + previous.__qualname__
        setattr(language_module.Qwen4ExpAttention, name, lambda self, *a, **k: False)
    report["disabled_optional_model_routes"] = disabled
    report["actual_python_executable"] = sys.executable
    report["actual_python_version"] = sys.version
    report["actual_mlx_version"] = importlib.metadata.version("mlx")
    report["gpu_device_info"] = mx.metal.device_info()
    native = [SITE / "mlx/core.cpython-311-darwin.so", SITE / "mlx/lib/libmlx.dylib", SITE / "mlx/lib/mlx.metallib"]
    report["native_mlx_binary_sha256"] = {str(path): source_reference.sha256(path) for path in native}
    deadline = time.perf_counter() + args.timeout_seconds
    def check_deadline():
        if time.perf_counter() >= deadline:
            raise TimeoutError("reference deadline reached between synchronized calls")

    loaded = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr), mx.stream(mx.gpu), source_reference.readonly_loader(utils, mx), sanitize_metadata_without_engine(args.model):
        model = utils.load_model(args.model, lazy=True, strict=True)
        language = model.language_model
        if hasattr(model, "mtp") or language.get_mtp_module() is not None:
            raise RuntimeError("ordinary autoregressive reference unexpectedly attached MTP")
        parameters = tree_flatten(model.parameters())
        report["resident_parameter_bytes"] = sum(value.nbytes for _, value in parameters)
        report["installed_model_class"] = type(model).__module__ + "." + type(model).__name__
        report["installed_language_class"] = type(language).__module__ + "." + type(language).__name__
        mx.eval(model.parameters())
        mx.synchronize()
    report["excluded_load_eval_seconds"] = time.perf_counter() - loaded
    source_reference.require_unchanged(watched)
    removed_compiled_hc = 0
    for _, module in model.named_modules():
        if type(module) is language_module.Qwen4ExpGatedResidual and hasattr(module, "_compiled_forward"):
            delattr(module, "_compiled_forward")
            removed_compiled_hc += 1
    report["disabled_vendor_compiled_singleton_hc_modules"] = removed_compiled_hc

    class LastLogitHead(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
        def __call__(self, hidden, *args, **kwargs):
            return self.original(hidden[:, -1:, :], *args, **kwargs)
    report["original_lm_head_class"] = type(language.lm_head).__module__ + "." + type(language.lm_head).__name__
    if args.lm_head_policy == "last":
        language.lm_head = LastLogitHead(language.lm_head)

    def collect_arrays(value, result, seen):
        if isinstance(value, mx.array):
            if id(value) not in seen:
                seen.add(id(value))
                result.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect_arrays(item, result, seen)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_arrays(item, result, seen)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                collect_arrays(getattr(value, field.name), result, seen)

    def state_arrays(caches):
        result, seen = [], set()
        for cache in caches:
            collect_arrays(cache.state, result, seen)
            collect_arrays(vars(cache), result, seen)
        return result

    def finite_check(logits, caches, full_state):
        arrays = [logits] + (state_arrays(caches) if full_state else [])
        floating = [value for value in arrays if value.dtype in (mx.float16, mx.bfloat16, mx.float32)]
        checks = [mx.all(mx.isfinite(value)) for value in floating]
        mx.eval(checks)
        mx.synchronize()
        if not all(bool(value.item()) for value in checks):
            raise RuntimeError("reference logits/cache contains nonfinite values")
        return {"floating_arrays_checked": len(floating), "floating_bytes_checked": sum(value.nbytes for value in floating), "finite": True}

    trials = []
    for trial in range(args.samples + 1):
        check_deadline()
        caches = language.make_cache()
        if len(caches) != 48:
            raise RuntimeError("Flash reference cache layer count differs")
        report["request_cache_classes"] = [type(cache).__name__ for cache in caches]
        # These are request-scoped text-position fields in the installed adapter.
        language._position_ids = None
        language._rope_deltas = None
        length, calls, output = 0, [], []
        observed = time.perf_counter()
        def forward(ids, phase):
            nonlocal length
            check_deadline()
            with mx.stream(mx.gpu):
                inputs = mx.array([ids], dtype=mx.uint32)
                positions = mx.arange(length, length + len(ids), dtype=mx.int64)[None, :]
                mx.eval(inputs, positions)
                mx.synchronize()
                began = time.perf_counter()
                result = language(inputs, cache=caches, position_ids=positions)
                logits = result.logits[:, -1, :]
                arrays = state_arrays(caches)
                mx.eval(logits, arrays)
                mx.synchronize()
                duration = time.perf_counter() - began
            length += len(ids)
            calls.append({"phase": phase, "input_rows": len(ids), "logical_length": length,
                          "synchronized_wall_seconds": duration, "lm_head_rows": 1 if args.lm_head_policy == "last" else len(ids)})
            finite_check(logits, caches, False)
            return logits

        logits = None
        for offset in range(0, len(tokens), args.prefill_rows):
            logits = forward(tokens[offset:offset + args.prefill_rows], "prefill")
        prefill_state = finite_check(logits, caches, True)
        finished = "length"
        for index in range(args.max_tokens):
            with mx.stream(mx.gpu):
                chosen = mx.argmax(logits, axis=-1)
                mx.eval(chosen)
                mx.synchronize()
                token = int(chosen.item())
            if token < 0 or token >= report["vocabulary_size"]:
                raise RuntimeError("invalid predicted token")
            output.append(token)
            if token in (248044, 248046):
                finished = "stop"
                break
            if index + 1 < args.max_tokens:
                logits = forward([token], "decode")
        final_state = finite_check(logits, caches, True)
        prefill = sum(call["synchronized_wall_seconds"] for call in calls if call["phase"] == "prefill")
        decode = sum(call["synchronized_wall_seconds"] for call in calls if call["phase"] == "decode")
        record = {
            "trial": trial, "warmup": trial == 0, "prefill_tokens": len(tokens), "prefill_seconds": prefill,
            "prefill_tokens_per_second": len(tokens) / prefill,
            "decode_forward_calls": sum(call["phase"] == "decode" for call in calls), "decode_seconds": decode,
            "decode_forward_tokens_per_second": (len(output) - 1) / decode if decode and output else None,
            "output_tokens": output, "output_count": len(output), "requested_output_tokens": args.max_tokens,
            "full_output_budget": len(output) == args.max_tokens, "finish_reason": finished,
            "output_u32le_sha256": hashlib.sha256(struct.pack(f"<{len(output)}I", *output)).hexdigest(),
            "observed_request_seconds_including_validation": time.perf_counter() - observed,
            "prefill_state": prefill_state, "final_state": final_state, "calls": calls,
            "memory": {"active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory(), "peak_bytes": mx.get_peak_memory()},
        }
        trials.append(record)
        print(json.dumps({key: record[key] for key in ("trial", "warmup", "prefill_tokens_per_second", "decode_forward_tokens_per_second", "output_count", "output_u32le_sha256")}), file=sys.stderr, flush=True)
        del caches, logits
        gc.collect()
        source_reference.require_unchanged(watched)
    measured = trials[1:]
    report.update({
        "gpu_executed": True, "valid": True, "trials": trials,
        "median_prefill_tokens_per_second": statistics.median(record["prefill_tokens_per_second"] for record in measured),
        "median_decode_forward_tokens_per_second": statistics.median(record["decode_forward_tokens_per_second"] for record in measured) if all(record["decode_forward_tokens_per_second"] is not None for record in measured) else None,
        "measured_greedy_outputs_identical": len({record["output_u32le_sha256"] for record in measured}) == 1,
        "source_unchanged": True,
    })
    source_reference.require_unchanged(watched)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=source_reference.DEFAULT_MODEL)
    parser.add_argument("--bundle", type=Path, default=source_reference.DEFAULT_BUNDLE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--expected-prompt-tokens", type=int, default=2048)
    parser.add_argument("--prefill-rows", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--ple-mode", choices=("mmap", "resident"), default="mmap")
    parser.add_argument("--lm-head-policy", choices=("last", "all"), default="last")
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-gpu", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.prefill_rows <= 8192 and 1 <= args.expected_prompt_tokens <= 8192 and
            0 <= args.max_tokens <= 256 and 1 <= args.samples <= 9 and args.timeout_seconds > 0):
        parser.error("invalid bounded benchmark dimensions")
    args.model, args.bundle, args.prompt, args.output = (path.expanduser().resolve() for path in (args.model, args.bundle, args.prompt, args.output))
    if args.output.exists() or any(args.output.is_relative_to(path) for path in (args.model, args.bundle)):
        raise ValueError("choose a fresh output outside preserved source/model directories")
    tokens, watched, report = cpu_preflight(args)
    if args.run_gpu:
        report = run_gpu(args, tokens, watched, report)
    write_new(args.output, report)
    print(json.dumps({"valid": report["valid"], "gpu_executed": report["gpu_executed"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
