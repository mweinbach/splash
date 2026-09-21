"""ROOT-ONLY tiny MLX Metal golden generator; requires explicit --run-gpu.

This script is prepared by CPU-only agents and must be executed only by the
root coordinating serial GPU jobs. It does not load the model or a dense PLE
table: it gathers just the requested checkpoint rows, plus one rounding trap.
Baseline CPU fixtures are copied to a private output directory. Expected raw
binary files are replaced with results of the actual installed MLX GPU ops.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil

import numpy as np

from flash_ple_reference import (
    CheckpointRows, HEAD_DIMENSION, SHARED_WEIGHT_SCALE, bf16, bf16_bits,
    decode_affine_row, shard_for_row,
)


def run_oracle(fixtures, output, checkpoint, *, allow_gpu=False):
    if not allow_gpu:
        raise RuntimeError("GPU oracle execution requires explicit --run-gpu")
    # Keep MLX imports after the explicit gate. Importing this file or running
    # --help never initializes an MLX device or submits commands.
    import mlx.core as mx
    import mlx.nn as nn

    fixtures, output, checkpoint = map(Path, (fixtures, output, checkpoint))
    input_path, output_path = fixtures.resolve(), output.resolve()
    if output_path.is_relative_to(input_path) or input_path.is_relative_to(output_path):
        raise ValueError("oracle output must not overlap the preserved CPU fixtures")
    if output.exists():
        raise FileExistsError("choose a fresh private GPU golden output directory")
    baseline_manifest = json.loads((fixtures / "manifest.json").read_text())
    baseline_names = {entry["name"] for entry in baseline_manifest["tensors"]}
    if not {"hash-expected_gather", "hash-shared_scale"} <= baseline_names:
        raise ValueError("main CPU fixtures must be generated with --checkpoint")
    if not (fixtures / "hash-gather.npz").is_file():
        raise FileNotFoundError("checkpoint CPU gather NPZ fixture is missing")
    # Validate the preserved fixture payloads before labeling any copied input
    # as a reference. Nested post fixtures have independent manifests.
    for manifest_path in (fixtures / "manifest.json",
                          fixtures / "post-fullwidth" / "manifest.json"):
        if not manifest_path.exists():
            continue
        preserved = json.loads(manifest_path.read_text())
        for entry in preserved["tensors"]:
            binary = manifest_path.parent / entry["file"]
            payload = binary.read_bytes()
            if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"CPU fixture integrity mismatch: {binary}")
    shutil.copytree(fixtures, output)
    gpu_generated_files = set()
    gpu_generated_tensors = {}
    source_ref = Path(__file__).with_name("flash_ple_reference.py")
    report = {
        "schema": "splash-flash-ple-mlx-metal-golden-v1",
        "reference": "actual installed MLX GPU operators",
        "gpu_executed": True, "dense_table_materialized": False,
        "mlx_version": importlib.metadata.version("mlx"),
        "cpu_fixture_directory": str(fixtures.resolve()),
        "golden_directory": str(output.resolve()), "cases": [],
        "source_checkpoint_directory": str(checkpoint.resolve()),
        "cpu_fixture_manifest_sha256": hashlib.sha256((fixtures / "manifest.json").read_bytes()).hexdigest(),
        "oracle_generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cpu_source_reference_sha256": hashlib.sha256(source_ref.read_bytes()).hexdigest(),
        "integer_hash_reference": "independent CPU I64 source reference using checkpoint arrays; not recomputed by MLX GPU",
    }

    def read(directory, name, kind, shape):
        raw = np.fromfile(directory / f"{name}.{kind}",
                          dtype={"u16": "<u2", "u32": "<u4", "i64": "<i8"}[kind])
        raw = raw.reshape(shape)
        return ((raw.astype(np.uint32) << 16).view(np.float32)
                if kind == "u16" else raw)

    def host_bf16(array):
        # Call this while the GPU stream is selected, so the cast also follows
        # the explicit GPU stream before the small host synchronization.
        converted = array.astype(mx.float32)
        mx.eval(converted)
        return bf16_bits(np.asarray(converted)).astype("<u2")

    def metrics(actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        a = (actual.astype(np.uint32) << 16).view(np.float32)
        e = (expected.astype(np.uint32) << 16).view(np.float32)
        return {"values": int(actual.size),
                "bf16_bit_mismatches": int(np.count_nonzero(actual != expected)),
                "max_abs_difference": float(np.max(np.abs(a - e), initial=0))}

    def replace(directory, name, actual, case):
        path = directory / f"{name}.u16"
        actual = np.asarray(actual, dtype="<u2")
        if path.exists():
            expected = np.fromfile(path, dtype="<u2").reshape(actual.shape)
            case["cpu_vs_gpu"][name] = metrics(actual, expected)
        path.write_bytes(actual.tobytes(order="C"))
        gpu_generated_files.add(path.resolve())
        gpu_generated_tensors[path.resolve()] = {
            "name": name, "file": path.name, "dtype": "u16",
            "shape": list(actual.shape), "bytes": actual.nbytes,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reference": "actual installed MLX Metal GPU output",
        }

    def update_manifest(directory):
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["cpu_only"] = False
        manifest.pop("gpu_commands", None)
        manifest["gpu_executed"] = True
        manifest["expected_binary_reference"] = "per-tensor provenance; floating expected data is actual MLX Metal GPU"
        manifest["affine_reference_rounding"] = "actual MLX GPU dequantize"
        manifest["expected_operator_reference"] = "actual installed MLX Metal GPU"
        manifest["integer_hash_reference"] = report["integer_hash_reference"]
        listed_files = {entry["file"] for entry in manifest["tensors"]}
        for binary, entry in gpu_generated_tensors.items():
            if binary.parent == directory.resolve() and entry["file"] not in listed_files:
                manifest["tensors"].append(dict(entry))
                listed_files.add(entry["file"])
        for entry in manifest["tensors"]:
            binary = directory / entry["file"]
            payload = binary.read_bytes()
            entry["bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            if binary.resolve() in gpu_generated_files:
                entry["reference"] = "actual installed MLX Metal GPU output"
            elif entry["name"] in {"hash-expected_ids", "hash-expected_history"}:
                entry["reference"] = "independent CPU I64 source reference using checkpoint arrays"
            else:
                entry["reference"] = "preserved CPU fixture input or checkpoint array"
        npz = directory / "hash-gather.npz"
        if npz.exists():
            manifest["auxiliary_artifacts"] = [{
                "file": npz.name, "reference": "actual installed MLX Metal GPU",
                "integer_indices_reference": report["integer_hash_reference"],
                "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            }]
        path.write_text(json.dumps(manifest, indent=2) + "\n")

    source = CheckpointRows(checkpoint)
    manifest = json.loads((output / "manifest.json").read_text())
    geometry = manifest["hash_geometry"]
    ids = read(output, "hash-expected_ids", "i64",
               (geometry["lanes"], geometry["rows"], geometry["heads"]))
    triples = [source.ple_row(*shard_for_row(int(row))) for row in ids.flat]
    words = np.stack([triple[0] for triple in triples]).astype(np.uint32)
    scales = np.stack([triple[1] for triple in triples]).astype(np.float32)
    biases = np.stack([triple[2] for triple in triples]).astype(np.float32)
    affine_variants = {}
    for name, intermediate in (("f32_product", False), ("bf16_product", True)):
        rows = np.stack([
            decode_affine_row(*triple, intermediate_bf16_product=intermediate)
            for triple in triples
        ])
        affine_variants[name] = (bf16_bits(rows), bf16_bits(bf16(rows * SHARED_WEIGHT_SCALE)))
    gather_case = {"name": "checkpoint-gather", "selected_rows": int(ids.size),
                   "cpu_vs_gpu": {}, "affine_reference_variants": {}}
    trap_row, trap_column = 2927653, 32
    trap = source.ple_row(*shard_for_row(trap_row))
    with mx.stream(mx.gpu):
        # MLX accepts NumPy arrays, but rejects NumPy scalar objects nested
        # inside Python lists. Normalize all BF16 host inputs before import.
        array = lambda value: mx.array(np.asarray(value, dtype=np.float32),
                                        dtype=mx.bfloat16)
        dequantized = mx.dequantize(mx.array(words, dtype=mx.uint32),
                                    array(scales), array(biases),
                                    group_size=32, bits=4, mode="affine")
        scaled = dequantized.astype(mx.bfloat16) * array([float(SHARED_WEIGHT_SCALE)])
        raw_dequantized, raw_scaled = host_bf16(dequantized), host_bf16(scaled)
        replace(output, "hash-expected_gather",
                raw_scaled.reshape(geometry["lanes"], geometry["rows"], -1),
                gather_case)
        with np.load(output / "hash-gather.npz", allow_pickle=False) as baseline:
            npz_values = {name: baseline[name] for name in baseline.files}
        npz_values["embedding_bf16"] = raw_scaled.reshape(*ids.shape, HEAD_DIMENSION)
        np.savez(output / "hash-gather.npz", **npz_values)
        for name, (dequant, scaled_variant) in affine_variants.items():
            gather_case["affine_reference_variants"][name] = {
                "dequantized_vs_gpu": metrics(raw_dequantized, dequant),
                "shared_scaled_vs_gpu": metrics(raw_scaled, scaled_variant),
            }
        trap_dequantized = mx.dequantize(
            mx.array(trap[0][None], dtype=mx.uint32), array(trap[1][None]),
            array(trap[2][None]), group_size=32, bits=4, mode="affine")
        trap_scaled = trap_dequantized * array([float(SHARED_WEIGHT_SCALE)])
        raw_trap, raw_trap_scaled = host_bf16(trap_dequantized), host_bf16(trap_scaled)
    def as_float(raw):
        return float((np.uint32(raw) << np.uint32(16)).view(np.float32))
    trap_code = (int(trap[0][trap_column // 8]) >> ((trap_column % 8) * 4)) & 15
    gather_case["trap_point"] = {
        "row": trap_row, "column": trap_column, "code": trap_code,
        "scale": float(trap[1][trap_column // 32]),
        "bias": float(trap[2][trap_column // 32]),
        "gpu_dequantized": as_float(raw_trap[0, trap_column]),
        "gpu_shared_scaled": as_float(raw_trap_scaled[0, trap_column]),
        "f32_product_reference": float(decode_affine_row(*trap)[trap_column]),
        "bf16_product_reference": float(decode_affine_row(
            *trap, intermediate_bf16_product=True)[trap_column]),
    }
    gather_case["checkpoint_payload_bytes_read"] = source.bytes_read
    report["cases"].append(gather_case)

    def post_case(directory):
        manifest = json.loads((directory / "manifest.json").read_text())
        p = manifest["post_geometry"]
        lanes, rows, width, streams = (p[name] for name in ("lanes", "rows", "width", "streams"))
        channels = width * streams
        options = {"dtype": mx.bfloat16}
        case = {"name": str(directory.relative_to(output)) or ".",
                "geometry": p, "cpu_vs_gpu": {}}
        with mx.stream(mx.gpu):
            def load(name, shape):
                return mx.array(read(directory, f"post-{name}", "u16", shape), **options)
            hyper = load("hyper", (lanes, rows, channels))
            key = load("key", (lanes, rows, channels))
            value = load("value", (lanes, rows, width))
            norms = [load(name, (channels,)) for name in ("norm_key", "norm_query", "norm_conv")]
            conv = load("conv", (channels, 4, 1))
            initial_state = load("state_initial", (lanes, 9, channels))
            mask = mx.array(read(directory, "post-mask", "u32", (lanes, rows)).astype(bool))
            def norm(x, weight):
                scale = weight.astype(mx.float32)
                if p["one_plus_weight"]:
                    scale = 1.0 + scale
                groups = x.astype(mx.float32).reshape(*x.shape[:-1], streams, width)
                normalized = mx.fast.rms_norm(groups, None, p["epsilon"])
                return (normalized * scale.reshape(streams, width)).reshape(x.shape).astype(x.dtype)
            def post(hyper, key, value, state, mask):
                keys = norm(key, norms[0]).reshape(*hyper.shape[:-1], streams, width)
                queries = norm(hyper, norms[1]).reshape(*hyper.shape[:-1], streams, width)
                gate_products = keys * queries
                gate_reduced = mx.sum(gate_products, axis=-1, keepdims=True)
                gate_divided = gate_reduced / math.sqrt(width)
                gate_magnitude = mx.maximum(mx.abs(gate_divided), 1e-6)
                gate_root = mx.sqrt(gate_magnitude)
                gate = mx.sign(gate_divided) * gate_root
                gate_probability = mx.sigmoid(gate)
                gated = (gate_probability * value[..., None, :]).reshape(hyper.shape)
                normed = norm(gated, norms[2])
                gated = mx.where(mask[..., None], gated, 0)
                normed = mx.where(mask[..., None], normed, 0)
                joined = mx.concatenate([state, normed], axis=1)
                convolution = mx.conv1d(joined, conv, dilation=3, groups=channels)
                activated = nn.silu(convolution)
                output = gated + activated
                return {"output": output, "state": joined[:, -9:], "gate": gate,
                        "gated": gated, "normalized_conv": normed,
                        "injected": hyper + output,
                        "normalized_keys": keys.reshape(hyper.shape),
                        "normalized_queries": queries.reshape(hyper.shape),
                        "gate_products": gate_products.reshape(hyper.shape),
                        "gate_reduced": gate_reduced,
                        "gate_divided": gate_divided,
                        "gate_magnitude": gate_magnitude,
                        "gate_root": gate_root,
                        "gate_probability": gate_probability,
                        "convolution": convolution, "activated": activated}
            full = post(hyper, key, value, initial_state, mask)
            for name, tensor in full.items():
                replace(directory, f"post-expected_{name}", host_bf16(tensor), case)
            state, pieces = initial_state, []
            for chunk in manifest["chunks"]:
                index, begin, end = (chunk[name] for name in ("index", "begin", "end"))
                replace(directory, f"post-chunk{index}-state_initial", host_bf16(state), case)
                part = post(hyper[:, begin:end], key[:, begin:end], value[:, begin:end],
                            state, mask[:, begin:end])
                replace(directory, f"post-chunk{index}-expected_output", host_bf16(part["output"]), case)
                replace(directory, f"post-chunk{index}-expected_state", host_bf16(part["state"]), case)
                state = part["state"]
                pieces.append(part["output"])
            chunked = mx.concatenate(pieces, axis=1)
            raw_chunked = host_bf16(chunked)
            raw_whole = host_bf16(full["output"])
            replace(directory, "post-expected_output_chunked", raw_chunked, case)
            case["actual_gpu_chunked_vs_whole"] = metrics(raw_chunked, raw_whole)
            case["actual_gpu_final_state_vs_whole"] = metrics(host_bf16(state), host_bf16(full["state"]))
        manifest["post_chunk_outputs_equal_whole"] = case["actual_gpu_chunked_vs_whole"]["bf16_bit_mismatches"] == 0
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        update_manifest(directory)
        return case

    report["cases"].append(post_case(output))
    if (output / "post-fullwidth" / "manifest.json").exists():
        report["cases"].append(post_case(output / "post-fullwidth"))
    (output / "mlx-gpu-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "cases": len(report["cases"]),
                      "report": str(output / "mlx-gpu-report.json")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-gpu", action="store_true",
                        help="explicit root-only authorization for serial GPU oracle work")
    args = parser.parse_args()
    if not args.run_gpu:
        parser.error("pass --run-gpu only from the root coordinating serial GPU jobs")
    run_oracle(args.fixtures, args.output, args.checkpoint, allow_gpu=args.run_gpu)
