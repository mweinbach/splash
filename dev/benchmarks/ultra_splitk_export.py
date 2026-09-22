"""CPU-only source-affine export for the private F32 split-K oracle."""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest_bytes = (args.package / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    tensors = manifest["tensors"]
    quant = manifest["quantization"]
    q = quant.get(args.prefix, quant)
    bits, group = q["bits"], q["group_size"]
    source = {}
    raw = {}
    for suffix in ("weight", "scales", "biases"):
        t = tensors[f"{args.prefix}.{suffix}"]
        assert t["dtype"] == ("U32" if suffix == "weight" else "BF16")
        with (args.package / t["shard"]).open("rb") as stream:
            stream.seek(t["offset"])
            raw[suffix] = stream.read(t["length"])
        assert len(raw[suffix]) == t["length"]
        source[suffix] = dict(t, sha256=hashlib.sha256(raw[suffix]).hexdigest())
    n, groups = source["scales"]["shape"]
    k = groups * group
    assert source["biases"]["shape"] == [n, groups]
    assert bits in (4, 5, 6, 8) and group in (32, 64, 128)
    wstride, pstride = k * bits // 8, groups * 2
    assert len(raw["weight"]) == n * wstride
    assert len(raw["scales"]) == len(raw["biases"]) == n * pstride
    packed = np.frombuffer(raw["weight"], dtype=np.uint8).reshape(n, wstride)
    scales = (np.frombuffer(raw["scales"], dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(n, groups)
    biases = (np.frombuffer(raw["biases"], dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(n, groups)
    shift = (np.arange(k, dtype=np.uint32) * bits) % 8
    byte = (np.arange(k, dtype=np.uint32) * bits) // 8
    cross = shift + bits > 8
    values = np.empty((n, k), dtype="<f4")
    for begin in range(0, n, 32):
        block = packed[begin : begin + 32]
        codes = block[:, byte].astype(np.uint16)
        codes[:, cross] |= block[:, byte[cross] + 1].astype(np.uint16) << 8
        codes = (codes >> shift[None, :]) & ((1 << bits) - 1)
        # Separate ufuncs guarantee original F32 multiply/add staging, no FMA.
        product = np.multiply(codes.astype(np.float32), np.repeat(scales[begin : begin + 32], group, axis=1), dtype=np.float32)
        values[begin : begin + 32] = np.add(product, np.repeat(biases[begin : begin + 32], group, axis=1), dtype=np.float32)
    assert np.isfinite(values).all()
    coefficient_bytes = values.tobytes()
    prefix_bytes = args.prefix.encode()
    source_identity = manifest["source_identity_sha256"].encode("ascii")
    assert len(source_identity) == 64
    header = struct.pack("<8s8I64s", b"ULTRAF32", n, k, bits, group, wstride, pstride, len(prefix_bytes), 0, source_identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        for part in (header, prefix_bytes, raw["weight"], raw["scales"], raw["biases"], coefficient_bytes):
            stream.write(part)
    metadata = {
        "prefix": args.prefix, "n": n, "k": k, "bits": bits, "group_size": group,
        "source_identity_sha256": manifest["source_identity_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "coefficient_sha256": hashlib.sha256(coefficient_bytes).hexdigest(),
        "original_sources": source, "gpu_commands": 0,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "original_sources"}))


if __name__ == "__main__":
    main()
