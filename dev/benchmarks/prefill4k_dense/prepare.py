#!/usr/bin/env python3
"""CPU-only actual BF16 coefficient fixtures; no MLX or GPU imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path
import random
import struct

ROLES = [
    "language_model.model.layers.0.linear_attn.in_proj_qkv",
    "language_model.model.layers.3.self_attn.q_proj",
    "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down",
    "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up",
    "language_model.model.layers.0.linear_attn.in_proj_z",
    "language_model.model.layers.0.linear_attn.out_proj",
    "language_model.model.layers.3.self_attn.k_proj",
    "language_model.model.layers.3.self_attn.indexer.index_qk_proj",
    "language_model.model.layers.0.mlp.shared_expert.down_proj",
    "language_model.model.layers.1.ple.value_proj",
]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def rounded_bf16(value: float) -> int:
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    if word & 0x7F800000 == 0x7F800000:
        return (word >> 16) | (0x40 if word & 0x7FFFFF else 0)
    return ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF


class Tensors:
    def __init__(self, root: Path):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.local = json.loads((root / "manifest.json").read_text())
        self.files: dict[str, tuple[mmap.mmap, dict, int]] = {}

    def tensor(self, name: str) -> memoryview:
        if name in self.local["tensors"]:
            tensor = self.local["tensors"][name]
            filename = tensor["shard"]
            if filename not in self.files:
                with (self.root / filename).open("rb") as file:
                    mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                self.files[filename] = (mapped, {}, 0)
            mapped = self.files[filename][0]
            return memoryview(mapped)[tensor["offset"]:tensor["offset"] + tensor["length"]]
        filename = self.index[name]
        if filename not in self.files:
            with (self.root / filename).open("rb") as file:
                mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            length = struct.unpack_from("<Q", mapped)[0]
            self.files[filename] = (mapped, json.loads(mapped[8:8 + length]), 8 + length)
        mapped, header, begin = self.files[filename]
        offsets = header[name]["data_offsets"]
        return memoryview(mapped)[begin + offsets[0]:begin + offsets[1]]


def source_samples(source: Tensors, entry: dict, payload: bytes) -> int:
    spec = entry["source"]
    packed = source.tensor(entry["projection"] + ".weight")
    scales = source.tensor(entry["projection"] + ".scales")
    biases = source.tensor(entry["projection"] + ".biases")
    rows, width = entry["shape"]
    rng = random.Random(14371 + rows + width)
    coordinates = [(0, 0), (0, width - 1), (rows - 1, 0), (rows - 1, width - 1)]
    coordinates += [(rng.randrange(rows), rng.randrange(width)) for _ in range(256)]
    for n, k in coordinates:
        bit = k * spec["bits"]
        offset = n * spec["weight_row_stride_bytes"] + bit // 8
        bytes_needed = (bit % 8 + spec["bits"] + 7) // 8
        code = (int.from_bytes(packed[offset:offset + bytes_needed], "little") >> (bit % 8))
        code &= (1 << spec["bits"]) - 1
        parameter = n * spec["parameter_row_stride_bytes"] + (k // spec["group_size"]) * 2
        scale = bf16(struct.unpack_from("<H", scales, parameter)[0])
        bias = bf16(struct.unpack_from("<H", biases, parameter)[0])
        reconstructed = f32(f32(code * scale) + bias)
        if not math.isfinite(reconstructed):
            raise ValueError("nonfinite source coefficient")
        expected = rounded_bf16(reconstructed)
        actual = struct.unpack_from("<H", payload, (n * width + k) * 2)[0]
        if actual != expected:
            raise ValueError(f"BF16 source coefficient differs: {entry['projection']} n={n} k={k}")
    return len(coordinates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operand-store", type=Path,
                        default=Path("install/local-models/Flash-Next-operands-v1"))
    parser.add_argument("--source-model", type=Path,
                        default=Path("install/local-models/Flash-Next-oQ4e-mtp-v1"))
    parser.add_argument("--out", type=Path, default=Path("build/prefill4k-dense/actual-weights.json"))
    parser.add_argument("--roles", default=",".join(ROLES))
    args = parser.parse_args()
    manifest = json.loads((args.operand_store / "manifest.json").read_text())
    source = Tensors(args.source_model)
    if manifest["schema"] != "splash-local-affine-operands-v1":
        raise ValueError("unexpected operand schema")
    if manifest["source_identity_sha256"] != source.local["source_identity_sha256"]:
        raise ValueError("operand source identity differs from current model")
    cases = []
    coefficient_checks = 0
    for role in args.roles.split(","):
        entry = next(e for e in manifest["entries"] if e["projection"] == role and e["format"] == "BF16")
        if entry["operand_math"] != "mlx-affine-f32-reconstruct-rounded-bf16-row-major-v1":
            raise ValueError("unexpected operand coefficient arithmetic")
        path = args.operand_store / entry["file"]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry["payload_sha256"] or len(payload) != entry["allocated_bytes"]:
            raise ValueError("operand payload verification failed")
        if entry["logical_bytes"] != entry["allocated_bytes"]:
            raise ValueError("fixture reader requires an exact matrix extent")
        checks = source_samples(source, entry, payload)
        coefficient_checks += checks
        cases.append({
            "projection": role,
            "input_size": entry["shape"][1], "output_size": entry["shape"][0],
            "weights_file": str(path.resolve()), "weights_sha256": digest,
            "coefficient_source_samples_exact": checks,
            "source_bits": entry["source"]["bits"], "source_group_size": entry["source"]["group_size"],
        })
    result = {
        "schema": "splash-prefill4k-dense-fixtures-v1",
        "gpu_executed": False,
        "source_identity_sha256": manifest["source_identity_sha256"],
        "weights_manifest_fingerprint": manifest["weights_manifest_fingerprint"],
        "actual_activations": False,
        "coefficient_source_samples_exact": coefficient_checks,
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fixture_cases": len(cases), "coefficient_source_samples_exact": coefficient_checks,
                      "manifest": str(args.out), "gpu_executed": False}))


if __name__ == "__main__":
    main()
