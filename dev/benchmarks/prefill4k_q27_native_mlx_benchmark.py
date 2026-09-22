#!/usr/bin/env python3
"""Root-only Q27 native-coefficient / stock-MLX benchmark; preflight is CPU-only."""
from __future__ import annotations
import argparse
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import struct
import sys
import time

import prefill4k_q27_mlx_layout as layout
import prefill4k_q27_native_mlx_adapter as adapter

ROOT = layout.ROOT
SITE = layout.SITE
DEFAULT_MODEL = ROOT / "build/prefill4k-q27-native-mlx-v1"
DEFAULT_PROMPT = ROOT / "build/release/flash/prefill4k_q27_quick2048.tokens.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_preflight(args):
    plan = layout.make_plan(args.package)
    tokens = json.loads(args.prompt.read_bytes())
    vocabulary = plan["runtime_geometry"]["vocabularySize"]
    if len(tokens) != 2048 or any(type(value) is not int or not 0 <= value < vocabulary for value in tokens):
        raise ValueError("reference requires exactly2048 original frozen valid token IDs")
    witness_path = args.prompt.with_name(args.prompt.name.replace(".tokens.json", ".metadata.json"))
    witness = json.loads(witness_path.read_bytes())
    u32_sha = hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()
    if (witness["prompt_u32le_sha256"] != u32_sha or witness["tokens_file_sha256"] != sha(args.prompt) or
            witness["token_count"] != 2048 or witness["enable_thinking"] is not False or
            not witness["saved_raw_user_content_sha256_match"] or witness["native_manifest_sha256"] != plan["manifest_sha256"]):
        raise ValueError("frozen original quick-prompt source witness differs")
    paths = [Path(__file__), Path(adapter.__file__), Path(layout.__file__)]
    paths += [SITE / name for name in ("mlx_lm/utils.py", "mlx_lm/models/qwen3_5.py", "mlx_lm/models/gated_delta.py",
                                      "mlx_lm/models/qwen3_next.py", "mlx_lm/models/cache.py")]
    versions = {}
    for distribution in importlib.metadata.distributions(path=[str(SITE)]):
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name in {"mlx", "mlx-lm", "numpy", "safetensors", "transformers"}:
            versions[name] = distribution.version
    ready = (args.model / "config.json").exists()
    converted = adapter.preflight(args.model)[1] if ready else {"status": "full Root-memory conversion queued; source layout and bounded CPU certificates available"}
    if ready and converted["native_source_manifest_sha256"] != plan["manifest_sha256"]:
        raise ValueError("converted MLX coefficient artifact differs from chosen native package source")
    report = {"schema": "splash-private-q27-native-coefficient-stock-mlx-benchmark-v1",
              "reference_label": "native operative Q27 coefficients with stock MLX-LM qwen3_5 operators and mandatory exact native-decay bridge",
              "model_directory": str(args.model), "original_native_package": str(args.package),
              "native_manifest_sha256": plan["manifest_sha256"], "native_source_upstream": plan["upstream"],
              "quantization": plan["quantization"], "quant_code_scale_bias_bit_exact": True,
              "norm_source": "unaltered native operative BF16 gain; no inverse centering/no extra+1",
              "decay_source": "unaltered native operative pre-exponentiated negative F32 a_scale",
              "original_raw_norm_A_log_words_recovered": False,
              "stock_GDN_state_update_kernel_unchanged": True, "native_decay_binding_layers_expected": 48,
              "converted_model_preflight": converted, "ready_to_run": ready,
              "prompt_path": str(args.prompt), "prompt_file_sha256": sha(args.prompt),
              "prompt_u32le_sha256": u32_sha,
              "prompt_token_count": len(tokens), "frontend_prompt_witness": witness,
              "reasoning": "off; exact frozen frontend-normalized tokens used directly",
              "sampling": {"temperature": 0, "top_k": 0, "top_p": 1}, "mtp_enabled": False,
              "prefix_cache_enabled": False, "serving_cache_enabled": False, "fresh_cache_each_trial": True,
              "warmup_requests": 1, "measured_requests": args.samples, "output_tokens_requested": args.max_tokens,
              "prefill_chunk_rows": args.prefill_rows, "lm_head_policy": "last-row-only wrapper around original loaded vocabulary module",
              "installed_distribution_versions": versions, "preflight_python_executable": sys.executable,
              "source_code_sha256": {str(path): sha(path) for path in paths},
              "native_layout_source_code_sha256": plan["source_code_sha256"],
              "timing_scope": "model forward through forced cache/logit evaluation+GPU synchronize; includes fresh cache priming; excludes loading, firstwarmup, input preparation, sampling and finite-state validation",
              "gpu_only_timing_available": False, "gpu_executed": False, "model_loaded": False, "valid": True}
    watched = paths + [args.prompt, witness_path, args.package / "manifest.json", args.package / "tokenizer/config.json"]
    if ready:
        watched += [args.model / name for name in ("config.json", "model.safetensors.index.json", "native-coefficient-manifest.json")]
        watched += list(args.model.glob("*.safetensors")) + [args.model / "native-bridge/gdn-decay.safetensors"]
    report["preserved_source_snapshot"] = {str(path): [path.stat().st_dev, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns] for path in watched}
    if "mlx.core" in sys.modules:
        raise RuntimeError("CPU preflight unexpectedly imported MLX")
    return tokens, report


def unchanged(report):
    for name, before in report["preserved_source_snapshot"].items():
        info = Path(name).stat()
        if [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns] != before:
            raise RuntimeError("benchmark input/model/source changed")


def run_gpu(args, tokens, report):
    if not args.run_gpu or not report["ready_to_run"]:
        raise RuntimeError("Root GPU slot and completed native-coefficient artifact are required")
    import mlx.core as mx
    import mlx.nn as nn
    native = [SITE / "mlx/core.cpython-311-darwin.so", SITE / "mlx/lib/libmlx.dylib", SITE / "mlx/lib/mlx.metallib"]
    report["actual_python_executable"] = sys.executable
    report["actual_python_version"] = sys.version
    report["actual_mlx_version"] = importlib.metadata.version("mlx")
    report["native_mlx_binary_sha256"] = {str(path): sha(path) for path in native}
    report["metal_device_info"] = mx.metal.device_info()
    deadline = time.perf_counter() + args.timeout_seconds
    def guard_time():
        if time.perf_counter() >= deadline:
            raise TimeoutError("reference deadline reached between synchronized calls")

    loaded = time.perf_counter()
    with mx.stream(mx.gpu):
        model, config, handle = adapter.load_model(args.model, run_root_gpu=True, lazy=True)
    report["actual_native_coefficient_bridge"] = handle.report
    if handle.report["native_decay_layers_bound"] != 48:
        raise RuntimeError("all48 native decay layers must bind")

    class LastHead(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
        def __call__(self, hidden):
            return self.original(hidden[:, -1:, :])

    report["original_lm_head_class"] = type(model.language_model.lm_head).__module__ + "." + type(model.language_model.lm_head).__name__
    model.language_model.lm_head = LastHead(model.language_model.lm_head)
    def collect(value, result, seen):
        if isinstance(value, mx.array):
            if id(value) not in seen:
                seen.add(id(value))
                result.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item, result, seen)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item, result, seen)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                collect(getattr(value, field.name), result, seen)
    def cache_arrays(caches):
        arrays, seen = [], set()
        for cache in caches:
            collect(cache.state, arrays, seen)
            collect(vars(cache), arrays, seen)
        return arrays
    def finite(logits, caches, full):
        arrays = [logits] + (cache_arrays(caches) if full else [])
        values = [array for array in arrays if array.dtype in (mx.float16, mx.bfloat16, mx.float32)]
        checks = [mx.all(mx.isfinite(array)) for array in values]
        mx.eval(checks)
        mx.synchronize()
        if not all(bool(value.item()) for value in checks):
            raise RuntimeError("nonfinite reference logits/cache")
        return {"finite": True, "floating_arrays": len(values), "floating_bytes": sum(value.nbytes for value in values)}

    trials = []
    with handle:
        with mx.stream(mx.gpu):
            mx.eval(model.parameters(), handle.arrays)
            mx.synchronize()
        report["excluded_load_eval_seconds"] = time.perf_counter() - loaded
        report["model_loaded"] = True
        for trial in range(args.samples + 1):
            guard_time()
            caches = model.language_model.make_cache()
            if len(caches) != 64:
                raise RuntimeError("Q27 reference needs64 fresh caches")
            classes = [type(cache).__name__ for cache in caches]
            report["request_cache_classes"] = classes
            length, calls, output = 0, [], []
            request_start = time.perf_counter()
            def forward(ids, phase):
                nonlocal length
                guard_time()
                with mx.stream(mx.gpu):
                    inputs = mx.array([ids], dtype=mx.uint32)
                    mx.eval(inputs)
                    mx.synchronize()
                    began = time.perf_counter()
                    logits = model(inputs, cache=caches)[:, -1, :]
                    mx.eval(logits, cache_arrays(caches))
                    mx.synchronize()
                    duration = time.perf_counter() - began
                length += len(ids)
                calls.append({"phase": phase, "input_rows": len(ids), "logical_length": length,
                              "synchronized_wall_seconds": duration, "lm_head_rows": 1})
                finite(logits, caches, False)
                return logits
            for offset in range(0, len(tokens), args.prefill_rows):
                logits = forward(tokens[offset:offset + args.prefill_rows], "prefill")
            prefill_state = finite(logits, caches, True)
            reason = "length"
            for index in range(args.max_tokens):
                with mx.stream(mx.gpu):
                    chosen = mx.argmax(logits, axis=-1)
                    mx.eval(chosen)
                    mx.synchronize()
                    token = int(chosen.item())
                if not 0 <= token < config["text_config"]["vocab_size"]:
                    raise RuntimeError("invalid predicted token")
                output.append(token)
                if token in (248044, 248046):
                    reason = "stop"
                    break
                if index + 1 < args.max_tokens:
                    logits = forward([token], "decode")
            final_state = finite(logits, caches, True)
            prefill = sum(call["synchronized_wall_seconds"] for call in calls if call["phase"] == "prefill")
            decode = sum(call["synchronized_wall_seconds"] for call in calls if call["phase"] == "decode")
            record = {"trial": trial, "warmup": trial == 0, "prompt_tokens": len(tokens), "prefill_seconds": prefill,
                      "prefill_tokens_per_second": len(tokens) / prefill, "decode_seconds": decode,
                      "decode_forward_tokens_per_second": (len(output) - 1) / decode if decode else None,
                      "output_count": len(output), "output_requested": args.max_tokens, "output_full_budget": len(output) == args.max_tokens,
                      "finish_reason": reason, "output_tokens": output,
                      "output_u32le_sha256": hashlib.sha256(struct.pack(f"<{len(output)}I", *output)).hexdigest(),
                      "observed_request_seconds_including_validation": time.perf_counter() - request_start,
                      "prefill_state": prefill_state, "final_state": final_state, "calls": calls,
                      "memory": {"active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory(), "peak_bytes": mx.get_peak_memory()}}
            trials.append(record)
            print(json.dumps({key: record[key] for key in ("trial", "warmup", "prefill_tokens_per_second", "decode_forward_tokens_per_second", "output_count", "output_u32le_sha256")}), file=sys.stderr, flush=True)
            del caches, logits
            gc.collect()
            unchanged(report)
        measured = trials[1:]
        report.update(gpu_executed=True, trials=trials, valid=True,
                      median_prefill_tokens_per_second=statistics.median(record["prefill_tokens_per_second"] for record in measured),
                      median_decode_forward_tokens_per_second=statistics.median(record["decode_forward_tokens_per_second"] for record in measured) if all(record["decode_forward_tokens_per_second"] is not None for record in measured) else None,
                      measured_greedy_outputs_identical=len({record["output_u32le_sha256"] for record in measured}) == 1)
    report["stock_compute_g_restored_after_benchmark"] = True
    unchanged(report)
    report["preserved_sources_unchanged"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=layout.PACKAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--prefill-rows", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-gpu", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.prefill_rows <= 2048 and 1 <= args.max_tokens <= 256 and 1 <= args.samples <= 9 and args.timeout_seconds > 0):
        parser.error("invalid bounded reference dimensions")
    args.package, args.model, args.prompt, args.output = (path.expanduser().resolve() for path in (args.package, args.model, args.prompt, args.output))
    if args.output.exists() or any(args.output.is_relative_to(path) for path in (args.package, args.model)):
        raise ValueError("choose a fresh result outside preserved source and converted model")
    tokens, report = cpu_preflight(args)
    if args.run_gpu:
        report = run_gpu(args, tokens, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"valid": True, "ready_to_run": report["ready_to_run"], "gpu_executed": report["gpu_executed"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
