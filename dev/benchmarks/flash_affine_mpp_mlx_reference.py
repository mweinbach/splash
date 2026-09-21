"""Root-only MLX GPU affine/GEMM goldens; requires explicit --run-gpu.

Imports, --help and --cpu-self-test do not import MLX or initialize a device.
Synthetic one-hot GEMMs distinguish F32 reconstruction from a BF16 multiply
boundary. Optional checkpoint cases read small original affine row crops from
the aligned bundle, never a dense PLE table or an expanded whole model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def bf16_bits(values):
    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    result = (rounded >> np.uint32(16)).astype(np.uint16)
    nonfinite = (words & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    special = ((words >> np.uint32(16)) |
               np.where(words & np.uint32(0x7FFFFF), np.uint32(0x40), np.uint32(0))).astype(np.uint16)
    return np.where(nonfinite, special, result).astype(np.uint16)


def bf16_float(values):
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16(values):
    return bf16_float(bf16_bits(values))


def splitmix64(values):
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def native_inputs(rows, width):
    """Match flash_affine_mpp_oracle.mm's i + 0xc0ffee sequence exactly."""
    random = splitmix64(np.arange(rows * width, dtype=np.uint64) + np.uint64(0xC0FFEE))
    signed = (random % np.uint64(2047)).astype(np.int32) - 1023
    return bf16(signed.astype(np.float32).reshape(rows, width) / np.float32(1024.0))


def pack_codes(codes, bits):
    codes = np.asarray(codes, dtype=np.uint32)
    if codes.ndim != 2 or bits not in (4, 5, 6, 8) or codes.shape[1] * bits % 32:
        raise ValueError("codes must be rank two with a whole-word supported bit width")
    if np.any(codes >= np.uint32(1 << bits)):
        raise ValueError("code exceeds bit width")
    result = np.zeros((codes.shape[0], codes.shape[1] * bits // 32), dtype=np.uint32)
    for channel in range(codes.shape[1]):
        word, shift = divmod(channel * bits, 32)
        result[:, word] |= codes[:, channel] << np.uint32(shift)
        if shift + bits > 32:
            result[:, word + 1] |= codes[:, channel] >> np.uint32(32 - shift)
    return result


def unpack_codes(words, bits, width):
    words = np.asarray(words, dtype=np.uint32)
    if words.ndim != 2 or bits not in (4, 5, 6, 8) or words.shape[1] * 32 != width * bits:
        raise ValueError("packed extent does not match code geometry")
    position = np.arange(width, dtype=np.uint64) * np.uint64(bits)
    index, shift = (position // np.uint64(32)).astype(np.intp), position % np.uint64(32)
    padded = np.pad(words, ((0, 0), (0, 1)))
    combined = (padded[:, index].astype(np.uint64) |
                (padded[:, index + 1].astype(np.uint64) << np.uint64(32)))
    return ((combined >> shift) & np.uint64((1 << bits) - 1)).astype(np.uint32)


def reconstruct(words, scales, biases, bits, group):
    scales, biases = map(lambda value: np.asarray(value, dtype=np.float32), (scales, biases))
    if scales.ndim != 2 or scales.shape != biases.shape or group not in (32, 64, 128):
        raise ValueError("invalid affine coefficient geometry")
    width = scales.shape[1] * group
    codes = unpack_codes(words, bits, width).astype(np.float32)
    sf, bs = (np.repeat(value, group, axis=1) for value in (scales, biases))
    # NumPy's individual float32 ufuncs retain an explicit product boundary.
    # The product is exact in F32 for these BF16 coefficients and integer codes.
    product = np.multiply(codes, sf, dtype=np.float32)
    full_f32 = np.add(product, bs, dtype=np.float32)
    multiply_first_bf16 = np.add(bf16(product), bs, dtype=np.float32)
    return full_f32, bf16(full_f32), bf16(multiply_first_bf16)


def metrics(actual, expected):
    actual, expected = map(lambda value: np.asarray(value, dtype=np.uint16), (actual, expected))
    if actual.shape != expected.shape:
        raise ValueError("BF16 comparison shapes differ")
    a, b = bf16_float(actual).astype(np.float64), bf16_float(expected).astype(np.float64)
    delta = a - b
    finite = np.isfinite(a) & np.isfinite(b)
    ordered = lambda value: np.where(value & 0x8000,
                                     0x8000 - (value.astype(np.int32) & 0x7FFF),
                                     0x8000 + value.astype(np.int32))
    ulp = np.abs(ordered(actual) - ordered(expected))
    return {
        "values": int(actual.size),
        "bf16_bit_mismatches": int(np.count_nonzero(actual != expected)),
        "nonfinite_values": int(np.count_nonzero(~finite)),
        "max_bf16_ulp": int(np.max(ulp[finite], initial=0)),
        "max_abs": float(np.max(np.abs(delta[finite]), initial=0)),
        "relative_l2": float(np.sqrt(np.sum(delta[finite] ** 2) /
                                      max(1e-30, float(np.sum(b[finite] ** 2))))),
    }


class BundleCrop:
    """Verified manifest identity and bounded reads of source affine rows."""

    def __init__(self, package):
        self.directory = Path(package).resolve()
        raw = (self.directory / "manifest.json").read_bytes()
        expected = (self.directory / "manifest.sha256").read_text().strip().split()[0]
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if self.manifest_sha256 != expected:
            raise ValueError("bundle manifest hash mismatch")
        self.manifest = json.loads(raw)
        if self.manifest["schema"] != "splash-local-qwen4-affine-v1":
            raise ValueError("unsupported derived bundle schema")
        self.bytes_read = 0

    def tensor(self, name, rows, dtype):
        tensor = self.manifest["tensors"][name]
        shape = tensor["shape"]
        expected_dtype = {"<u4": "U32", "<u2": "BF16"}[dtype]
        if tensor["dtype"] != expected_dtype or len(shape) != 2 or not 0 < rows <= shape[0]:
            raise ValueError(f"invalid source tensor crop: {name}")
        file = (self.directory / tensor["shard"]).resolve()
        if not file.is_relative_to(self.directory):
            raise ValueError("tensor path leaves derived package")
        row_bytes = shape[1] * np.dtype(dtype).itemsize
        if tensor["length"] != shape[0] * row_bytes or file.stat().st_size < tensor["offset"] + tensor["length"]:
            raise ValueError("source tensor byte extent mismatch")
        mapped = np.memmap(file, mode="r", dtype=dtype, offset=tensor["offset"], shape=(rows, shape[1]))
        result = np.array(mapped, copy=True)
        self.bytes_read += result.nbytes
        return result

    def projection(self, prefix, maximum_rows):
        quantization = self.manifest["quantization"]
        rule = quantization.get(prefix, quantization)
        bits, group = int(rule["bits"]), int(rule["group_size"])
        rows = min(maximum_rows, self.manifest["tensors"][prefix + ".weight"]["shape"][0])
        before = self.bytes_read
        words = self.tensor(prefix + ".weight", rows, "<u4")
        scales = bf16_float(self.tensor(prefix + ".scales", rows, "<u2"))
        biases = bf16_float(self.tensor(prefix + ".biases", rows, "<u2"))
        if words.shape[0] != scales.shape[0] or scales.shape != biases.shape:
            raise ValueError("source affine row geometry mismatch")
        return words, scales, biases, bits, group, self.bytes_read - before


def cpu_self_test():
    count = 0
    words = np.arange(65536, dtype=np.uint16)
    finite = np.isfinite(bf16_float(words))
    assert np.array_equal(bf16_bits(bf16_float(words[finite])), words[finite])
    count += int(np.count_nonzero(finite))
    for bits in (4, 5, 6, 8):
        codes = (splitmix64(np.arange(256, dtype=np.uint64)) % np.uint64(1 << bits)).astype(np.uint32).reshape(2, 128)
        packed = pack_codes(codes, bits)
        assert np.array_equal(unpack_codes(packed, bits, 128), codes)
        count += codes.size
        for group in (32, 64, 128):
            trap = pack_codes(np.full((2, 128), 7, dtype=np.uint32), bits)
            scales = np.full((2, 128 // group), -14.25, dtype=np.float32)
            biases = np.full_like(scales, 128)
            f32, rounded, intermediate = reconstruct(trap, scales, biases, bits, group)
            assert np.all(f32 == 28.25) and np.all(rounded == 28.25) and np.all(intermediate == 28)
            count += f32.size * 3
    # Independent scalar reproduction verifies the native seed and wraparound.
    mask = (1 << 64) - 1
    for index in (0, 1, 31, 32, 1023, 2559, 8191):
        value = (index + 0xC0FFEE + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        assert int(splitmix64(np.uint64(index + 0xC0FFEE))) == value
        count += 1
    error = metrics(bf16_bits([28.25]), bf16_bits([28]))
    assert error["bf16_bit_mismatches"] == 1 and error["max_abs"] == 0.25
    print(json.dumps({"pass": True, "cpu_checks": count + 1, "gpu_executed": False}))


def run_gpu(args):
    if not args.run_gpu:
        raise RuntimeError("MLX GPU execution requires explicit --run-gpu")
    # Import after the gate: module import and CPU helper tests stay GPU-free.
    import mlx.core as mx

    output = Path(args.output).resolve()
    bundle = BundleCrop(args.package) if args.package else None
    if bundle and (output.is_relative_to(bundle.directory) or bundle.directory.is_relative_to(output)):
        raise ValueError("GPU golden directory must not overlap the preserved bundle")
    if output.exists():
        raise FileExistsError("choose a fresh GPU golden output directory")
    output.mkdir(parents=True)
    report = {
        "schema": "splash-flash-affine-mpp-mlx-golden-v1",
        "gpu_executed": True,
        "reference": "actual installed MLX GPU quantized_matmul and dequantize",
        "mlx_version": importlib.metadata.version("mlx"),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rounding_candidates": {
            "f32_coeff": "float32(code)*float32(BF16(scale))+float32(BF16(bias)); no coefficient BF16 round",
            "f32_coeff_bf16": "F32 reconstruction then one final BF16 coefficient round",
            "bf16_product_first": "BF16(code*scale) then add BF16 bias and round to BF16",
        },
        "model_or_table_expanded": False,
        "cases": [],
    }
    if bundle:
        report["source_identity"] = bundle.manifest["source_identity_sha256"]
        report["bundle_manifest_sha256"] = bundle.manifest_sha256

    def host_bits(value):
        as_float = value.astype(mx.float32)
        mx.eval(as_float)
        return bf16_bits(np.asarray(as_float)).astype("<u2")

    def save(directory, name, values, kind, artifacts):
        values = np.asarray(values, dtype={"u16": "<u2", "u32": "<u4", "f32": "<f4"}[kind])
        path = directory / f"{name}.{kind}"
        payload = values.tobytes(order="C")
        path.write_bytes(payload)
        artifacts.append({"name": name, "file": path.name, "dtype": kind,
                          "shape": list(values.shape), "bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest()})

    def case(name, words, scales, biases, bits, group, inputs, metadata):
        width = scales.shape[1] * group
        coefficient_f32, coefficient_bf16, coefficient_intermediate = reconstruct(words, scales, biases, bits, group)
        directory = output / name
        directory.mkdir()
        entry = {"name": name, "rows": inputs.shape[0], "n": words.shape[0], "k": width,
                 "bits": bits, "group_size": group,
                 "negative_scales": int(np.count_nonzero(scales < 0)),
                 "scale_count": int(scales.size), "metadata": metadata, "artifacts": []}
        with mx.stream(mx.gpu):
            array = lambda value: mx.array(np.asarray(value, dtype=np.float32), dtype=mx.bfloat16)
            packed = mx.array(words, dtype=mx.uint32)
            sf, bs, x = array(scales), array(biases), array(inputs)
            dequantized = mx.dequantize(packed, sf, bs, group_size=group, bits=bits, mode="affine")
            quantized = mx.quantized_matmul(x, packed, sf, bs, transpose=True,
                                            group_size=group, bits=bits, mode="affine")
            dequantized_mm = x @ mx.transpose(dequantized)
            rounded_mm = x @ mx.transpose(array(coefficient_bf16))
            intermediate_mm = x @ mx.transpose(array(coefficient_intermediate))
            f32_mm = x.astype(mx.float32) @ mx.transpose(mx.array(coefficient_f32, dtype=mx.float32))
            dequantized_bits = host_bits(dequantized)
            qmm_bits, dqmm_bits = host_bits(quantized), host_bits(dequantized_mm)
            rounded_bits, intermediate_bits = host_bits(rounded_mm), host_bits(intermediate_mm)
            f32_bits = host_bits(f32_mm)
        entry["dequantize_vs_cpu"] = {
            "f32_coeff_bf16": metrics(dequantized_bits, bf16_bits(coefficient_bf16)),
            "bf16_product_first": metrics(dequantized_bits, bf16_bits(coefficient_intermediate)),
        }
        entry["qmm_vs_gpu_matmul"] = {
            "mlx_dequantize": metrics(qmm_bits, dqmm_bits),
            "f32_coeff_bf16": metrics(qmm_bits, rounded_bits),
            "bf16_product_first": metrics(qmm_bits, intermediate_bits),
            "f32_coeff": metrics(qmm_bits, f32_bits),
        }
        if metadata.get("one_hot_trap"):
            entry["trap_observed"] = {
                "qmm_unique_bf16_values": np.unique(bf16_float(qmm_bits)).astype(float).tolist(),
                "dequantize_unique_bf16_values": np.unique(bf16_float(dequantized_bits)).astype(float).tolist(),
                "f32_coeff_expected": 28.25, "bf16_product_first_expected": 28.0,
            }
        for label, values, kind in (
            ("packed_weight", words, "u32"), ("scales", bf16_bits(scales), "u16"),
            ("biases", bf16_bits(biases), "u16"), ("input", bf16_bits(inputs), "u16"),
            ("coefficient_f32", coefficient_f32, "f32"),
            ("coefficient_f32_bf16", bf16_bits(coefficient_bf16), "u16"),
            ("coefficient_product_bf16", bf16_bits(coefficient_intermediate), "u16"),
            ("mlx_dequantize", dequantized_bits, "u16"), ("mlx_qmm", qmm_bits, "u16"),
            ("mlx_dequantized_matmul", dqmm_bits, "u16"),
            ("mlx_f32_coefficient_matmul", f32_bits, "u16"),
            ("mlx_rounded_coefficient_matmul", rounded_bits, "u16"),
            ("mlx_product_bf16_coefficient_matmul", intermediate_bits, "u16"),
        ):
            save(directory, label, values, kind, entry["artifacts"])
        (directory / "manifest.json").write_text(json.dumps(entry, indent=2) + "\n")
        report["cases"].append(entry)
        print(json.dumps({"case": name, "qmm_vs_gpu_matmul": entry["qmm_vs_gpu_matmul"],
                          "trap_observed": entry.get("trap_observed")}))

    if not args.source_only:
        width, outputs = 2560, 640
        for bits in args.bits:
            # Pack one row and repeat it; every chosen activation observes the
            # same coefficient. M32/N640 avoids the single-row QMV route.
            words = np.repeat(pack_codes(np.full((1, width), 7, dtype=np.uint32), bits), outputs, axis=0)
            for group in args.groups:
                scales = np.full((outputs, width // group), -14.25, dtype=np.float32)
                biases = np.full_like(scales, 128)
                for rows in args.rows:
                    inputs = np.zeros((rows, width), dtype=np.float32)
                    inputs[np.arange(rows), np.arange(rows) % width] = 1
                    case(f"trap-q{bits}-g{group}-r{rows}", words, scales, biases, bits, group,
                         inputs, {"one_hot_trap": True, "code": 7, "scale": -14.25, "bias": 128})
    if bundle:
        projections = args.projection or ["language_model.model.layers.0.mlp.shared_expert.gate_proj"]
        for index, prefix in enumerate(projections):
            words, scales, biases, bits, group, source_bytes = bundle.projection(prefix, args.source_columns)
            width = scales.shape[1] * group
            for rows in args.rows:
                case(f"source-{index}-q{bits}-g{group}-r{rows}", words, scales, biases, bits, group,
                     native_inputs(rows, width), {"projection": prefix, "source_bytes_read": source_bytes,
                                                 "input_sequence": "native-oracle-splitmix64-i-plus-0xc0ffee"})
        report["checkpoint_payload_bytes_read"] = bundle.bytes_read
    if not report["cases"]:
        raise ValueError("no golden cases selected")
    report["pass"] = True  # Execution/integrity; numerical differences are measured, not rejected.
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pass": True, "cases": len(report["cases"]), "report": str(output / "report.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true", help="explicit root-only permission to execute MLX GPU ops")
    parser.add_argument("--cpu-self-test", action="store_true")
    parser.add_argument("--output", type=Path, help="fresh golden output directory")
    parser.add_argument("--package", type=Path, help="read-only aligned Flash bundle for optional source crops")
    parser.add_argument("--projection", action="append", help="actual affine source prefix; may repeat")
    parser.add_argument("--source-only", action="store_true", help="omit synthetic traps; requires --package")
    parser.add_argument("--bits", type=int, nargs="+", choices=(4, 5, 6, 8), default=[4, 5, 6, 8])
    parser.add_argument("--groups", type=int, nargs="+", choices=(32, 64, 128), default=[32, 64, 128])
    parser.add_argument("--rows", type=int, nargs="+", choices=(32, 128), default=[32])
    parser.add_argument("--source-columns", type=int, choices=(64, 128, 256, 512), default=64)
    args = parser.parse_args()
    if args.cpu_self_test:
        if args.run_gpu:
            parser.error("--cpu-self-test and --run-gpu are mutually exclusive")
        cpu_self_test()
        return
    if not args.run_gpu:
        parser.error("GPU execution is disabled; use --cpu-self-test or explicitly --run-gpu")
    if not args.output:
        parser.error("--run-gpu requires a fresh --output directory")
    if (args.source_only or args.projection) and not args.package:
        parser.error("source-only/projection selection requires --package")
    if len(set(args.bits)) != len(args.bits) or len(set(args.groups)) != len(args.groups) or len(set(args.rows)) != len(args.rows):
        parser.error("bits, groups and rows must not contain duplicates")
    run_gpu(args)


if __name__ == "__main__":
    main()
