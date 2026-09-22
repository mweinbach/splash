#!/usr/bin/env python3
"""CPU-only dense QSA reference for captured, prepared BF16 operands.

NPZ input (loaded with allow_pickle=False):
  queries_bf16: uint16[rows,24,256]
  keys_bf16, values_bf16: uint16[capacity,2,256]
  gates_bf16: uint16[rows,24,256]
  query_offset: integer scalar, with 0 <= offset and offset + rows <= 2048
The cache must include every token through offset + rows. Each query sees only
tokens [0, offset + row]. Preparation/RoPE/indexer operations are not reproduced.

A JSON manifest can instead contain query_offset and an "arrays" object. Each
array descriptor is {"file":"relative.bin", "dtype":"<u2", "shape":[...]},
with optional sha256. A .npy file descriptor also verifies dtype and shape.
Candidate-only NPZ/JSON files can be supplied with --candidate. Candidate names:
  candidate_attention_{f32,f64,bf16}, candidate_output_{f32,f64,bf16}
All candidate arrays have shape [rows,24,256]; BF16 means raw uint16 bits.

Reference QK, max-shift softmax and PV use FP64 CPU NumPy/BLAS. Sampled complete
causal rows are independently recomputed with math.fsum for QK, denominator and
PV. FP32-P and globally normalized BF16-P counterfactuals round the same FP64
global probabilities; they do not emulate a particular FP32 softmax reduction.
BF16 conversions explicitly use FP32 then round-to-nearest-even BF16, matching
the storage boundary but retaining possible FP64->FP32->BF16 double rounding.

Output comparisons use a NOMINAL staged BF16 gate model: FP64 libm exp rounded
to FP32, then staged BF16 exp/add/reciprocal/subtraction/multiply. This cannot
assert bit parity with Metal precise::exp. An ideal FP64 sigmoid/output is also
reported. Numerical operator results do not establish generation quality.

Examples:
  .venv/bin/python dev/benchmarks/prefill4k_attention/fp64_reference.py --self-test
  .../fp64_reference.py capture.npz --candidate candidate.npz --report report.json
  .../fp64_reference.py capture.json --comparison-policy fp32-p \
      --write-reference reference.npz --max-relative-l2 0.001 --max-abs 0.01
This script never imports MLX, Metal, Torch, or a GPU backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


HEADS, KV_HEADS, DIM, GROUP = 24, 2, 256, 12
POLICIES = {"fp64": "fp64", "fp32-p": "fp32_p_fp64", "bf16-p": "bf16_p_fp64"}
INPUTS = ("queries_bf16", "keys_bf16", "values_bf16", "gates_bf16")
CANDIDATES = tuple(
    f"candidate_{kind}_{dtype}"
    for kind in ("attention", "output") for dtype in ("f32", "f64", "bf16")
)


def bf16_decode(bits: np.ndarray) -> np.ndarray:
    """Decode BF16 words exactly into FP64; no arithmetic on uint16 operands."""
    words = np.asarray(bits, dtype=np.uint16)
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def bf16_encode(values: np.ndarray) -> np.ndarray:
    """Explicit FP32 conversion followed by BF16 round-to-nearest-even."""
    with np.errstate(over="ignore", invalid="ignore"):
        words = np.asarray(values, dtype=np.float32).view(np.uint32)
    finite = (words & 0x7F800000) != 0x7F800000
    rounded = ((words.astype(np.uint64) + 0x7FFF + ((words >> 16) & 1)) >> 16)
    special = (words >> 16) | np.where((words & 0x7FFFFF) != 0, 0x40, 0)
    return np.where(finite, rounded, special).astype(np.uint16)


def bf16_round(values: np.ndarray) -> np.ndarray:
    return bf16_decode(bf16_encode(values))


def nominal_staged_sigmoid(gates: np.ndarray) -> np.ndarray:
    """Approximation to the production staged BF16 precise-exp gate, see header."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exponential = bf16_round(np.exp(np.abs(gates)))
        denominator = bf16_round(1.0 + exponential)
        tail = bf16_round(1.0 / denominator)
        return np.where(gates < 0.0, tail, bf16_round(1.0 - tail))


def ideal_sigmoid(gates: np.ndarray) -> np.ndarray:
    exp_negative = np.exp(-np.abs(gates))
    return np.where(gates < 0.0, exp_negative / (1.0 + exp_negative),
                    1.0 / (1.0 + exp_negative))


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("array shape must be a nonempty JSON list")
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in value):
        raise ValueError("shape dimensions must be positive integers")
    return tuple(value)


def _descriptor(base: Path, descriptor: Any) -> np.ndarray:
    if not isinstance(descriptor, dict):
        raise ValueError("array descriptor must be an object")
    shape = _shape(descriptor.get("shape"))
    if not isinstance(descriptor.get("dtype"), str):
        raise ValueError("array descriptor requires a numeric dtype string")
    dtype = np.dtype(descriptor["dtype"])
    if dtype.kind not in "uif" or dtype.hasobject:
        raise ValueError("array dtype must be numeric, non-object")
    if "file" not in descriptor or not isinstance(descriptor["file"], str):
        raise ValueError("array descriptor requires file")
    path = base / descriptor["file"]
    if "sha256" in descriptor:
        expected_hash = descriptor["sha256"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"array SHA256 mismatch: {path}")
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"NPY dtype/shape mismatch: {path}")
    else:
        expected_bytes = math.prod(shape) * dtype.itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"binary file extent mismatch: {path}")
        array = np.fromfile(path, dtype=dtype).reshape(shape)
    return array


def load_capture(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load only explicit numeric arrays; NPZ and NPY always forbid pickle."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        metadata: dict[str, Any] = {}
    elif path.suffix.lower() == ".json":
        manifest = json.loads(path.read_text())
        if not isinstance(manifest, dict) or not isinstance(manifest.get("arrays"), dict):
            raise ValueError("JSON input requires an arrays object")
        arrays = {name: _descriptor(path.parent, entry)
                  for name, entry in manifest["arrays"].items()}
        if "query_offset" in manifest:
            if isinstance(manifest["query_offset"], bool):
                raise ValueError("query_offset cannot be boolean")
            arrays["query_offset"] = np.asarray(manifest["query_offset"])
        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
    else:
        raise ValueError("input must be an NPZ archive or JSON manifest")
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype.hasobject or array.dtype.kind not in "uif":
            raise ValueError(f"non-numeric or object array: {name}")
    return arrays, metadata


def validate_capture(arrays: dict[str, np.ndarray]) -> int:
    for name in INPUTS:
        if name not in arrays:
            raise ValueError(f"missing prepared operand: {name}")
        array = arrays[name]
        if array.dtype.kind != "u" or array.dtype.itemsize != 2:
            raise ValueError(f"{name} must contain uint16 BF16 bits")
        if np.any((array & 0x7F80) == 0x7F80):
            raise ValueError(f"{name} contains nonfinite BF16 words")
    q, k, v, g = (arrays[name] for name in INPUTS)
    if q.ndim != 3 or q.shape[1:] != (HEADS, DIM) or q.shape[0] == 0:
        raise ValueError("queries_bf16 must have shape [rows,24,256]")
    if g.shape != q.shape:
        raise ValueError("gates_bf16 shape must match prepared queries")
    if k.ndim != 3 or k.shape[1:] != (KV_HEADS, DIM) or v.shape != k.shape:
        raise ValueError("keys/values must have matching [capacity,2,256] shapes")
    offset_array = arrays.get("query_offset")
    if offset_array is None or offset_array.size != 1 or offset_array.dtype.kind not in "iu":
        raise ValueError("query_offset must be an integer scalar or one-element array")
    offset = int(offset_array.reshape(-1)[0])
    end = offset + q.shape[0]
    if offset < 0 or end > 2048 or k.shape[0] < end:
        raise ValueError("dense append requires 0 <= offset and offset+rows <= min(capacity,2048)")
    for name in CANDIDATES:
        if name not in arrays:
            continue
        array = arrays[name]
        expected = np.dtype("uint16" if name.endswith("bf16") else
                            "float32" if name.endswith("f32") else "float64")
        if array.shape != q.shape or array.dtype.kind != expected.kind or array.dtype.itemsize != expected.itemsize:
            raise ValueError(f"{name} must have dtype {expected} and shape {q.shape}")
    return offset


def _fsum_reference(q: np.ndarray, keys: np.ndarray, values: np.ndarray,
                    head: int, visible: int) -> tuple[np.ndarray, np.ndarray]:
    kv = head // GROUP
    scores = np.asarray([
        math.fsum(float(a) * float(b) for a, b in zip(q[head], keys[t, kv]))
        / math.sqrt(DIM) for t in range(visible)
    ], dtype=np.float64)
    maximum = float(np.max(scores))
    weights = np.asarray([math.exp(float(x) - maximum) for x in scores])
    probabilities = weights / math.fsum(float(x) for x in weights)
    columns = (0, 127, 255)
    output = np.asarray([
        math.fsum(float(p) * float(v) for p, v in zip(probabilities, values[:visible, kv, column]))
        for column in columns
    ])
    return scores, output


def reference(arrays: dict[str, np.ndarray], block_rows: int = 16,
              sampled_audit: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    offset = validate_capture(arrays)
    if not 1 <= block_rows <= 128:
        raise ValueError("block_rows must be in [1,128]")
    q, keys, values, gates = (bf16_decode(arrays[name]) for name in INPUTS)
    rows = q.shape[0]
    attentions = {policy: np.empty(q.shape, dtype=np.float64) for policy in POLICIES.values()}
    sample_rows = set((0, rows // 2, rows - 1)) if sampled_audit else set()
    audits: list[dict[str, Any]] = []
    score_errors, pv_errors = [], []
    for first in range(0, rows, block_rows):
        stop = min(first + block_rows, rows)
        cache_end = offset + stop
        for kv in range(KV_HEADS):
            head_begin = kv * GROUP
            query_block = q[first:stop, head_begin:head_begin + GROUP]
            # Both operands are FP64; dot accumulation is the CPU BLAS reduction.
            scores = (query_block.reshape(-1, DIM) @ keys[:cache_end, kv].T
                      / math.sqrt(DIM)).reshape(stop - first, GROUP, cache_end)
            visible = offset + np.arange(first, stop) + 1
            causal = np.arange(cache_end)[None, None, :] < visible[:, None, None]
            scores = np.where(causal, scores, -np.inf)
            maximum = np.max(scores, axis=-1, keepdims=True)
            weights = np.exp(scores - maximum)
            probability = weights / np.sum(weights, axis=-1, keepdims=True, dtype=np.float64)
            probability_variants = {
                "fp64": probability,
                "fp32_p_fp64": probability.astype(np.float32).astype(np.float64),
                "bf16_p_fp64": bf16_round(probability),
            }
            for policy, probabilities in probability_variants.items():
                attentions[policy][first:stop, head_begin:head_begin + GROUP] = (
                    probabilities.reshape(-1, cache_end) @ values[:cache_end, kv]
                ).reshape(stop - first, GROUP, DIM)
            for row in sorted(sample_rows.intersection(range(first, stop))):
                for head in (0, 11, 23):
                    if head // GROUP != kv:
                        continue
                    count = offset + row + 1
                    compensated_scores, compensated_pv = _fsum_reference(q[row], keys, values, head, count)
                    blas_scores = scores[row - first, head - head_begin, :count]
                    blas_pv = attentions["fp64"][row, head, (0, 127, 255)]
                    score_error = np.abs(blas_scores - compensated_scores)
                    pv_error = np.abs(blas_pv - compensated_pv)
                    score_errors.extend(score_error.tolist())
                    pv_errors.extend(pv_error.tolist())
                    audits.append({
                        "row": row, "head": head, "visible_tokens": count,
                        "columns": [0, 127, 255],
                        "qk_max_abs_blas_vs_fsum": float(np.max(score_error)),
                        "pv_max_abs_blas_vs_fsum": float(np.max(pv_error)),
                        "pv_fsum": compensated_pv.tolist(), "pv_blas": blas_pv.tolist(),
                    })
    sigmoid = ideal_sigmoid(gates)
    staged_sigmoid = nominal_staged_sigmoid(gates)
    outputs: dict[str, np.ndarray] = {
        "ideal_sigmoid_fp64": sigmoid,
        "nominal_staged_sigmoid_bf16": bf16_encode(staged_sigmoid),
    }
    for policy, attention in attentions.items():
        outputs[f"attention_{policy}"] = attention
        outputs[f"attention_{policy}_bf16"] = bf16_encode(attention)
        outputs[f"ideal_output_{policy}"] = attention * sigmoid
        outputs[f"nominal_staged_output_{policy}_bf16"] = bf16_encode(
            bf16_round(attention) * staged_sigmoid)
    audit = {
        "enabled": sampled_audit, "method": "FP64 math.fsum QK; max-shift exp; fsum denominator/PV",
        "qk_elements": len(score_errors), "pv_elements": len(pv_errors),
        "qk_max_abs_blas_vs_fsum": max(score_errors, default=None),
        "pv_max_abs_blas_vs_fsum": max(pv_errors, default=None),
        "samples": audits,
        "qualification": None,
    }
    return outputs, audit


def _ordered_bf16(bits: np.ndarray) -> np.ndarray:
    words = bits.astype(np.int64)
    return np.where((words & 0x8000) != 0, 0x8000 - (words & 0x7FFF), 0x8000 + words)


def _ordered_f32(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        words = values.astype(np.float32).view(np.uint32).astype(np.int64)
    return np.where((words & 0x80000000) != 0,
                    0x80000000 - (words & 0x7FFFFFFF), 0x80000000 + words)


def error_metrics(actual: np.ndarray, expected: np.ndarray,
                  near_zero_abs: float = 1e-5) -> dict[str, Any]:
    """Report absolute/relative/ULP/near-zero errors without an implicit pass."""
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError("comparison arrays must have matching nonempty shapes")
    finite = np.isfinite(actual) & np.isfinite(expected)
    a, e = actual[finite], expected[finite]
    delta = a - e
    reference_norm = float(np.linalg.norm(e))
    actual_norm = float(np.linalg.norm(a))
    delta_norm = float(np.linalg.norm(delta))
    relative_l2 = (delta_norm / reference_norm if reference_norm else
                   0.0 if delta_norm == 0 else None)
    cosine = (float(np.dot(a, e) / (actual_norm * reference_norm))
              if actual_norm and reference_norm else
              1.0 if actual_norm == reference_norm == 0 else None)
    near_zero = np.abs(e) <= near_zero_abs
    expected_bf16, actual_bf16 = bf16_encode(e), bf16_encode(a)
    bf16_ulp = np.abs(_ordered_bf16(actual_bf16) - _ordered_bf16(expected_bf16))
    with np.errstate(over="ignore", invalid="ignore"):
        finite_f32 = np.isfinite(a.astype(np.float32)) & np.isfinite(e.astype(np.float32))
    f32_ulp = np.abs(_ordered_f32(a[finite_f32]) - _ordered_f32(e[finite_f32]))
    nz_error = np.abs(delta[near_zero])
    appreciable = ~near_zero
    result = {
        "elements": int(actual.size), "finite_pairs": int(np.count_nonzero(finite)),
        "actual_nonfinite": int(np.count_nonzero(~np.isfinite(actual))),
        "reference_nonfinite": int(np.count_nonzero(~np.isfinite(expected))),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else None,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
        "relative_l2": relative_l2, "reference_l2": reference_norm,
        "relative_l2_denominator_zero": reference_norm == 0,
        "cosine": cosine,
        "max_relative_error_above_near_zero": (
            float(np.max(np.abs(delta[appreciable] / e[appreciable]))) if np.any(appreciable) else None),
        "bf16_rounded_mismatches": int(np.count_nonzero(actual_bf16 != expected_bf16)),
        "bf16_ulp_max": int(np.max(bf16_ulp)) if bf16_ulp.size else None,
        "bf16_ulp_p99": float(np.percentile(bf16_ulp, 99)) if bf16_ulp.size else None,
        "fp32_ulp_max": int(np.max(f32_ulp)) if f32_ulp.size else None,
        "fp32_ulp_finite_pairs": int(f32_ulp.size),
        "near_zero_abs": near_zero_abs, "near_zero_elements": int(np.count_nonzero(near_zero)),
        "near_zero_max_abs": float(np.max(nz_error)) if nz_error.size else None,
        "near_zero_reference_bf16_zero_actual_nonzero": int(np.count_nonzero(
            near_zero & ((expected_bf16 & 0x7FFF) == 0) & ((actual_bf16 & 0x7FFF) != 0))),
    }
    if np.any(finite):
        absolute = np.where(finite, np.abs(actual - expected), -1.0)
        index = np.unravel_index(int(np.argmax(absolute)), actual.shape)
        result["max_abs_location"] = list(map(int, index))
        result["actual_at_max_abs"] = float(actual[index])
        result["reference_at_max_abs"] = float(expected[index])
    return result


def threshold_decision(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any] | None:
    if not thresholds:
        return None
    checks = {"finite": metrics["actual_nonfinite"] == metrics["reference_nonfinite"] == 0}
    for name, limit in thresholds.items():
        metric_name = {"max_relative_l2": "relative_l2", "max_abs": "max_abs",
                       "min_cosine": "cosine", "max_bf16_ulp": "bf16_ulp_max",
                       "max_near_zero_abs": "near_zero_max_abs"}[name]
        value = metrics[metric_name]
        checks[name] = (True if value is None and name == "max_near_zero_abs" and
                       metrics["near_zero_elements"] == 0 else
                       value is not None and (value >= limit if name == "min_cosine" else value <= limit))
    return {"thresholds": thresholds, "checks": checks, "qualified": all(checks.values()),
            "scope": "operator numerical comparison only; no generation/service qualification"}


def compare_candidates(arrays: dict[str, np.ndarray], outputs: dict[str, np.ndarray],
                       policy: str, near_zero_abs: float,
                       thresholds: dict[str, float]) -> dict[str, Any]:
    suffix = POLICIES[policy]
    comparisons = {}
    for name in CANDIDATES:
        if name not in arrays:
            continue
        actual = bf16_decode(arrays[name]) if name.endswith("bf16") else arrays[name]
        kind = "attention" if name.startswith("candidate_attention") else "output"
        if kind == "attention":
            reference_name = f"attention_{suffix}"
            expected = outputs[reference_name]
        else:
            reference_name = f"nominal_staged_output_{suffix}_bf16"
            expected = bf16_decode(outputs[reference_name])
        metrics = error_metrics(actual, expected, near_zero_abs)
        comparisons[name] = {"reference": reference_name, "metrics": metrics,
                             "qualification": threshold_decision(metrics, thresholds)}
        if kind == "output":
            comparisons[name]["ideal_fp64_sigmoid_comparison"] = error_metrics(
                actual, outputs[f"ideal_output_{suffix}"], near_zero_abs)
    return comparisons


def self_test() -> dict[str, Any]:
    # Tiny fixtures exercise exact BF16 storage, all-head GQA mapping and causality.
    ties = np.asarray([1.0 + 1.0 / 256, 1.0 + 3.0 / 256])
    assert bf16_encode(ties).tolist() == [0x3F80, 0x3F82]
    special = bf16_encode(np.asarray([0.0, -0.0, math.inf, -math.inf, math.nan]))
    assert special[:4].tolist() == [0, 0x8000, 0x7F80, 0xFF80]
    assert special[4] & 0x7FC0 == 0x7FC0
    q = np.zeros((2, HEADS, DIM), dtype=np.float64)
    keys = np.zeros((5, KV_HEADS, DIM), dtype=np.float64)
    values = np.broadcast_to(np.asarray([1., -1., 1., -1., 4.])[:, None, None], keys.shape).copy()
    values[:, 1] *= 2.0
    gates = np.zeros_like(q)
    gates[:, :, 1] = -80.0
    gates[:, :, 2] = 80.0
    capture = {"queries_bf16": bf16_encode(q), "keys_bf16": bf16_encode(keys),
               "values_bf16": bf16_encode(values), "gates_bf16": bf16_encode(gates),
               "query_offset": np.asarray(3, dtype=np.int64)}
    outputs, audit = reference(capture, block_rows=1)
    assert np.all(outputs["attention_fp64"][0] == 0.0)
    assert np.allclose(outputs["attention_fp64"][1, :12], 0.8, rtol=0, atol=1e-15)
    assert np.allclose(outputs["attention_fp64"][1, 12:], 1.6, rtol=0, atol=2e-15)
    staged = bf16_decode(outputs["nominal_staged_sigmoid_bf16"])
    assert staged[0, 0, 0] == 0.5 and staged[0, 0, 1] > 0 and staged[0, 0, 2] == 1
    assert audit["qk_max_abs_blas_vs_fsum"] == 0.0
    perturbed = outputs["attention_fp64"].copy()
    perturbed[0, 0, 0] = 1e-6
    metrics = error_metrics(perturbed, outputs["attention_fp64"])
    assert metrics["near_zero_reference_bf16_zero_actual_nonzero"] == 1
    assert threshold_decision(metrics, {}) is None
    assert not threshold_decision(metrics, {"max_near_zero_abs": 1e-7})["qualified"]
    for invalid_offset in (-1, 2047):
        bad = dict(capture, query_offset=np.asarray(invalid_offset))
        try:
            validate_capture(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid causality/extent accepted")
    # Nonzero Q/K validates BLAS layout and compensated complete-row auditing.
    rng = np.random.default_rng(7185)
    nonzero = dict(capture, queries_bf16=bf16_encode(rng.normal(size=q.shape)),
                   keys_bf16=bf16_encode(rng.normal(size=keys.shape)),
                   values_bf16=bf16_encode(rng.normal(size=values.shape)))
    _, nonzero_audit = reference(nonzero, block_rows=2)
    assert nonzero_audit["qk_max_abs_blas_vs_fsum"] < 1e-12
    assert nonzero_audit["pv_max_abs_blas_vs_fsum"] < 1e-12
    return {"self_test_assertions_completed": True, "gpu_executed": False,
            "scope": "CPU implementation self-test; not model/kernel qualification"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path, nargs="?")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--write-reference", type=Path)
    parser.add_argument("--comparison-policy", choices=tuple(POLICIES), default="fp64")
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--near-zero-abs", type=float, default=1e-5)
    parser.add_argument("--skip-sampled-audit", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    for option in ("max-relative-l2", "max-abs", "min-cosine", "max-bf16-ulp", "max-near-zero-abs"):
        parser.add_argument(f"--{option}", type=float)
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), indent=2, allow_nan=False))
        return 0
    if args.capture is None:
        parser.error("capture is required unless --self-test is used")
    thresholds = {name: getattr(args, name) for name in
                  ("max_relative_l2", "max_abs", "min_cosine", "max_bf16_ulp", "max_near_zero_abs")
                  if getattr(args, name) is not None}
    if not math.isfinite(args.near_zero_abs) or args.near_zero_abs < 0:
        parser.error("near-zero-abs must be finite and nonnegative")
    for name, value in thresholds.items():
        valid_range = -1 <= value <= 1 if name == "min_cosine" else value >= 0
        if not math.isfinite(value) or not valid_range:
            parser.error(f"invalid threshold: {name}")
    arrays, metadata = load_capture(args.capture)
    if args.candidate:
        candidate_arrays, _ = load_capture(args.candidate)
        if not any(name in candidate_arrays for name in CANDIDATES):
            raise ValueError("candidate file contains no recognized candidate arrays")
        for name in CANDIDATES:
            if name in candidate_arrays:
                if name in arrays:
                    raise ValueError(f"candidate supplied twice: {name}")
                arrays[name] = candidate_arrays[name]
    offset = validate_capture(arrays)
    outputs, audit = reference(arrays, args.block_rows, not args.skip_sampled_audit)
    comparisons = compare_candidates(arrays, outputs, args.comparison_policy, args.near_zero_abs, thresholds)
    counterfactuals = {
        policy: error_metrics(outputs[f"attention_{suffix}"], outputs["attention_fp64"], args.near_zero_abs)
        for policy, suffix in POLICIES.items() if policy != "fp64"
    }
    report = {
        "schema": "splash-dense-qsa-fp64-reference-v1", "gpu_executed": False,
        "capture": str(args.capture), "metadata": metadata,
        "shape": list(arrays["queries_bf16"].shape), "query_offset": offset,
        "cache_capacity": int(arrays["keys_bf16"].shape[0]),
        "causality": "each row sees [0,query_offset+row], dense append end<=2048",
        "reference": "FP64 CPU BLAS QK/PV; FP64 max-shift global softmax; scale=1/sqrt(256)",
        "sampled_compensated_audit": audit,
        "probability_counterfactuals": counterfactuals,
        "counterfactual_semantics": "round globally normalized FP64 P to FP32 or FP32->RNE BF16, then FP64 PV; not an FP32 softmax oracle",
        "gate_semantics": "ideal stable FP64 sigmoid separately; nominal staged BF16 exp(abs(gate))/add/reciprocal/sign/subtract/product; exp is FP64 libm rounded to FP32, not Metal precise::exp parity",
        "nominal_gate_vs_ideal": error_metrics(
            bf16_decode(outputs["nominal_staged_sigmoid_bf16"]), outputs["ideal_sigmoid_fp64"], args.near_zero_abs),
        "comparison_policy": args.comparison_policy, "comparisons": comparisons,
        "qualification": None,
        "limitations": ["Prepared-input operator reference only; no preparation/cache/indexer qualification",
                        "CPU BLAS reduction independently audited at sampled complete causal rows, not exact arithmetic at every element",
                        "Staged gate exp approximation requires separate Metal gate-boundary validation",
                        "No generation quality or full-service throughput qualification"],
    }
    if thresholds and comparisons:
        report["qualification"] = {
            "explicit_thresholds": thresholds,
            "qualified": all(value["qualification"]["qualified"] for value in comparisons.values()),
            "scope": "candidate operator numerical comparisons under supplied thresholds only",
        }
    if args.write_reference:
        with args.write_reference.open("wb") as destination:
            np.savez(destination, query_offset=np.asarray(offset, dtype=np.int64), **outputs)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.write_text(serialized)
    print(serialized, end="")
    return 2 if report["qualification"] and not report["qualification"]["qualified"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"FP64 reference input/computation error: {error}", file=sys.stderr)
        raise SystemExit(1)
