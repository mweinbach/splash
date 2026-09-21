#!/usr/bin/env python3
"""Private CPU audit for exact centered codes and declared F32 alternatives.

Uses only Python's standard library. No Metal device, MLX import, model package,
GPU command, runtime route, or launcher is used. Counterexamples below are
expected: passing this audit does not qualify MPP precision or performance.
"""

from __future__ import annotations

import ctypes
from fractions import Fraction
import json
import math
import random
import struct


UINT64_MAX = (1 << 64) - 1
WIDTHS = (4, 5, 6, 8)


def f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bf16_value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def bf16_word(value: float) -> int:
    word = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    if word & 0x7F800000 == 0x7F800000:
        return (word >> 16) | (0x40 if word & 0x7FFFFF else 0)
    return ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF


def pack_codes(values: list[int], bits: int) -> bytes:
    # Independent bit-at-a-time encoder; no shared GPU/CPU unpacking helper.
    packed = bytearray((len(values) * bits + 7) // 8)
    for k, value in enumerate(values):
        assert 0 <= value < 1 << bits
        for bit in range(bits):
            index = k * bits + bit
            packed[index // 8] |= ((value >> bit) & 1) << (index % 8)
    return bytes(packed)


def unpack_code(packed: bytes, bits: int, k: int) -> int:
    index = k * bits
    byte, shift = divmod(index, 8)
    word = packed[byte]
    if shift + bits > 8:
        word |= packed[byte + 1] << 8
    return (word >> shift) & ((1 << bits) - 1)


def source_extent(rows: int, stride: int, row_bytes: int) -> int:
    if not (rows and row_bytes and stride >= row_bytes):
        raise ValueError("invalid extent")
    if rows - 1 > (UINT64_MAX - row_bytes) // stride:
        raise ValueError("overflow")
    return (rows - 1) * stride + row_bytes


def checked_bf16(value: float) -> float:
    assert math.isfinite(value) and bf16_value(bf16_word(value)) == value
    return value


def centered_group(x: list[float], q: list[int], bits: int, scale: float,
                   bias: float) -> float:
    # These synthetic products/sums are exact dyadics before the epilog. This
    # does not predict an unspecified MPP hardware reduction order.
    center = 1 << (bits - 1)
    partial = f32(math.fsum(v * (code - center) for v, code in zip(x, q)))
    sum_x = f32(math.fsum(x))
    correction = f32(bias + f32(center * scale))
    scaled = f32(partial * scale)
    return f32(scaled + f32(correction * sum_x))


def source_group(x: list[float], q: list[int], scale: float, bias: float) -> float:
    coefficients = [f32(f32(code * scale) + bias) for code in q]
    return f32(math.fsum(v * weight for v, weight in zip(x, coefficients)))


def unsigned_group(x: list[float], q: list[int], scale: float, bias: float) -> float:
    """Variant A: BF16 x original uint8 q dot, then s*dot_q + b*sum_x."""
    partial = f32(math.fsum(v * code for v, code in zip(x, q)))
    sum_x = f32(math.fsum(x))
    return f32(f32(partial * scale) + f32(bias * sum_x))


def reconstructed_group(x: list[float], q: list[int], bits: int, scale: float,
                        bias: float) -> float:
    """Variant B: reconstruct dot_q from centered dot before applying scale."""
    center = 1 << (bits - 1)
    partial = f32(math.fsum(v * (code - center) for v, code in zip(x, q)))
    sum_x = f32(math.fsum(x))
    unsigned_dot = f32(partial + f32(center * sum_x))
    return f32(f32(unsigned_dot * scale) + f32(bias * sum_x))


def main() -> None:
    counts = {"finite_bf16_roundtrips": 0, "centered_codes": 0,
              "packed_code_checks": 0, "exact_affine_groups": 0,
              "extent_checks": 0, "stable_variant_checks": 0,
              "q8_zero_copy_code_checks": 0, "q8_strided_copy_checks": 0,
              "q8_zero_copy_plan_checks": 0}
    for word in range(1 << 16):
        value = bf16_value(word)
        if math.isfinite(value):
            assert bf16_word(value) == word
            counts["finite_bf16_roundtrips"] += 1

    ranges = {}
    rng = random.Random(0x18C0DE)
    for bits in WIDTHS:
        center = 1 << (bits - 1)
        ranges[f"Q{bits}"] = [-center, center - 1]
        for q in range(1 << bits):
            centered = ctypes.c_int8(q - center).value
            assert -128 <= centered <= 127 and centered + center == q
            counts["centered_codes"] += 1
            packed = pack_codes([q] * 64, bits)
            for k in range(64):
                assert unpack_code(packed, bits, k) == q
                counts["packed_code_checks"] += 1
        for length in (64, 128, 320, 1024):
            values = [rng.randrange(1 << bits) for _ in range(length)]
            packed = pack_codes(values, bits)
            for k, q in enumerate(values):
                assert unpack_code(packed, bits, k) == q
                counts["packed_code_checks"] += 1
        x = [Fraction(checked_bf16((k % 17 - 8) / 16)) for k in range(64)]
        q = [rng.randrange(1 << bits) for _ in x]
        for scale in (-14.25, -1 / 256, 0.0, 0.5, 32.0):
            for bias in (-128.0, -1 / 256, 0.0, 64.0, 256.0):
                s, b = Fraction(checked_bf16(scale)), Fraction(checked_bf16(bias))
                source = sum(v * (code * s + b) for v, code in zip(x, q))
                factored = s * sum(v * (code - center) for v, code in zip(x, q))
                factored += (b + center * s) * sum(x)
                assert source == factored
                counts["exact_affine_groups"] += 1

    # For Q8 the shared helper reduces to bit=8*k, byte=k, shift=0 and
    # (row[k] >> 0) & 255. UINT8 tensor operands therefore read exactly the
    # original bytes, regardless of the buffer view's address alignment.
    q8_rows, q8_k = 4, 2560
    original_rows = [[(n * 17 + k * 37) % 256 for k in range(q8_k)]
                     for n in range(q8_rows)]
    q8_bytes = b"".join(pack_codes(row, 8) for row in original_rows)
    for offset in (0, 1, 3, 4, 17, 65):
        storage = bytearray(b"\xA7" * offset + q8_bytes + b"\xA7" * 64)
        untouched = bytes(storage)
        original_view = memoryview(storage)[offset:offset + len(q8_bytes)].toreadonly()
        code_view = original_view[:]
        assert code_view.readonly and code_view.obj is original_view.obj
        address = ctypes.addressof(ctypes.c_uint8.from_buffer(storage)) + offset
        if offset % 2:
            assert address % 2 == 1
        for n, values in enumerate(original_rows):
            row_view = original_view[n * q8_k:(n + 1) * q8_k]
            for k, expected in enumerate(values):
                # Independent U32 little-endian word extraction also verifies
                # every byte lane of the original packed U32 tensor layout.
                word = struct.unpack_from("<I", row_view, (k // 4) * 4)[0]
                word_code = (word >> (8 * (k % 4))) & 255
                assert word_code == unpack_code(row_view, 8, k) == expected
                assert code_view[n * q8_k + k] == expected
                counts["q8_zero_copy_code_checks"] += 1
        assert bytes(storage) == untouched

    # A non-contiguous row stride must retain the dense copy route; a single
    # linear UINT8 view would include source padding between rows.
    for stride in (q8_k + 1, q8_k + 16):
        extent = source_extent(q8_rows, stride, q8_k)
        for offset in (1, 3):
            storage = bytearray(b"\xA7" * (offset + extent + 64))
            for n, values in enumerate(original_rows):
                row_offset = offset + n * stride
                storage[row_offset:row_offset + q8_k] = pack_codes(values, 8)
            untouched = bytes(storage)
            source_view = memoryview(storage)[offset:offset + extent].toreadonly()
            dense_codes = b"".join(bytes(source_view[n * stride:n * stride + q8_k])
                                   for n in range(q8_rows))
            assert bytes(source_view[:q8_rows * q8_k]) != dense_codes
            for n, values in enumerate(original_rows):
                row_view = source_view[n * stride:n * stride + q8_k]
                for k, expected in enumerate(values):
                    assert unpack_code(row_view, 8, k) == expected
                    assert dense_codes[n * q8_k + k] == expected
                    counts["q8_strided_copy_checks"] += 1
            assert bytes(storage) == untouched

    rounded = lambda value: (value + 16383) & ~16383
    head_code_bytes = 248320 * 2560
    assert source_extent(248320, 2560, 2560) == head_code_bytes == 635699200
    zero_copy_plans = {"borrowed_padding": 2 * 16384,
                       "owned_padding": 2 * 16384 + rounded(16 * 2560 * 2),
                       "copied_borrowed_padding": rounded(head_code_bytes) + 2 * 16384,
                       "copied_owned_padding": rounded(head_code_bytes) + 2 * 16384
                       + rounded(16 * 2560 * 2)}
    for key, expected in (("borrowed_padding", 32768), ("owned_padding", 114688),
                          ("copied_borrowed_padding", 635731968),
                          ("copied_owned_padding", 635813888)):
        assert zero_copy_plans[key] == expected
        counts["q8_zero_copy_plan_checks"] += 1

    class Params(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint32) for name in
                    ("rows", "padded_rows", "input_size", "output_size", "bits",
                     "group_size", "tile_rows", "tile_outputs")]
        _fields_ += [("weight_row_stride_bytes", ctypes.c_uint64),
                    ("parameter_row_stride_bytes", ctypes.c_uint64)]

    assert ctypes.sizeof(Params) == 48 and ctypes.alignment(Params) == 8
    assert Params.weight_row_stride_bytes.offset == 32
    assert Params.parameter_row_stride_bytes.offset == 40
    for args, expected in (((2, 64, 32), 96), ((1, UINT64_MAX, 32), 32),
                           ((64, 32, 32), 2048), ((64, 2, 2), 128)):
        assert source_extent(*args) == expected
        counts["extent_checks"] += 1
    overflow_stride = (32 * pow(63, -1, 1 << 64)) & UINT64_MAX
    # The old unchecked expression would ask for just 64 bytes, even though
    # its row-1 offset is enormous. The current checked C++ guard rejects it.
    assert (63 * overflow_stride + 32) & UINT64_MAX == 64
    for args in ((0, 64, 32), (2, 16, 32), (2, UINT64_MAX, 32),
                 ((1 << 32) - 1, UINT64_MAX // 2, 32), (2, 64, 0),
                 (64, overflow_stride, 32)):
        try:
            source_extent(*args)
        except ValueError:
            counts["extent_checks"] += 1
        else:
            raise AssertionError(f"extent accepted {args}")

    alternatives = []
    for bits in WIDTHS:
        q = [0] * 64
        x = [1.0] + [0.0] * 63
        bias = checked_bf16(math.ldexp(1.0, bits - 25))
        source = source_group(x, q, 1.0, bias)
        candidate = centered_group(x, q, bits, 1.0, bias)
        assert source == bias and candidate == 0.0
        assert bf16_word(source) != bf16_word(candidate)
        unsigned = unsigned_group(x, q, 1.0, bias)
        reconstructed = reconstructed_group(x, q, bits, 1.0, bias)
        assert unsigned == source and reconstructed == source
        counts["stable_variant_checks"] += 2
        alternatives.append({"bits": bits, "case": "bias_cancellation",
                             "source": source, "centered": candidate,
                             "uint8_A": unsigned, "reconstructed_B": reconstructed,
                             "relative_error": 1.0})

        x = [1.0, checked_bf16(3 / 256)] + [0.0] * 62
        scale = checked_bf16(math.ldexp(1.0, -23 - bits))
        source = bf16_value(bf16_word(source_group(x, q, scale, 1.0)))
        candidate = bf16_value(bf16_word(centered_group(x, q, bits, scale, 1.0)))
        assert source == 1.015625 and candidate == 1.0078125
        unsigned = bf16_value(bf16_word(unsigned_group(x, q, scale, 1.0)))
        reconstructed = bf16_value(bf16_word(reconstructed_group(x, q, bits, scale, 1.0)))
        assert unsigned == source and reconstructed == source
        counts["stable_variant_checks"] += 2
        alternatives.append({"bits": bits, "case": "bf16_midpoint",
                             "source": source, "centered": candidate,
                             "uint8_A": unsigned, "reconstructed_B": reconstructed,
                             "bf16_ulp_difference": 1})

        x = [1.0, -1.0] + [0.0] * 62
        q = [0, (1 << bits) - 1] + [0] * 62
        bias = checked_bf16(math.ldexp(1.0, bits + 23))
        source = source_group(x, q, 1.0, bias)
        candidate = centered_group(x, q, bits, 1.0, bias)
        assert source == -(1 << bits) and candidate == -((1 << bits) - 1)
        unsigned = unsigned_group(x, q, 1.0, bias)
        reconstructed = reconstructed_group(x, q, bits, 1.0, bias)
        assert unsigned == candidate and reconstructed == candidate
        counts["stable_variant_checks"] += 2
        alternatives.append({"bits": bits, "case": "coefficient_rounding",
                             "source": source, "centered": candidate,
                             "uint8_A": unsigned, "reconstructed_B": reconstructed})

        x = [1.0] + [0.0] * 63
        q = [0] * 64
        scale = checked_bf16(math.ldexp(1.0, 129 - bits))
        assert source_group(x, q, scale, 0.0) == 0.0
        assert math.isnan(centered_group(x, q, bits, scale, 0.0))
        assert unsigned_group(x, q, scale, 0.0) == 0.0
        assert reconstructed_group(x, q, bits, scale, 0.0) == 0.0
        counts["stable_variant_checks"] += 2
        alternatives.append({"bits": bits, "case": "centered_overflow",
                             "source": 0.0, "centered": "NaN",
                             "uint8_A": 0.0, "reconstructed_B": 0.0})

        x = [1.0, checked_bf16(math.ldexp(1.0, -24))] + [0.0] * 62
        q = [0, 1] + [0] * 62
        source = source_group(x, q, 1.0, 0.0)
        unsigned = unsigned_group(x, q, 1.0, 0.0)
        reconstructed = reconstructed_group(x, q, bits, 1.0, 0.0)
        assert unsigned == source == math.ldexp(1.0, -24)
        assert reconstructed == 0.0
        counts["stable_variant_checks"] += 2
        alternatives.append({"bits": bits, "case": "small_q_reconstruction_loss",
                             "source": source, "uint8_A": unsigned,
                             "reconstructed_B": reconstructed})

        x = [checked_bf16(math.ldexp(1.0, 123))] * 64
        q = [0] * 64
        assert source_group(x, q, 1.0, 0.0) == 0.0
        assert math.isnan(unsigned_group(x, q, 1.0, 0.0))
        assert math.isnan(reconstructed_group(x, q, bits, 1.0, 0.0))
        counts["stable_variant_checks"] += 2
        alternatives.append({"bits": bits, "case": "sum_x_overflow",
                             "source": 0.0, "uint8_A": "NaN",
                             "reconstructed_B": "NaN"})

        # Moderate, exactly representable small-q and mixed-sign groups check
        # both proposals without conflating them with hardware reduction tests.
        patterns = ([0] * 64, [1] * 64, [k % 3 for k in range(64)],
                    [0, (1 << bits) - 1, 1 << (bits - 1), 1] * 16)
        for q in patterns:
            for x in ([1.0, -1.0, 0.5, -0.5] * 16,
                      [1.0, 0.5, -0.25, 0.125] * 16):
                for scale in (-14.25, -1 / 256, 0.5, 32.0):
                    for bias in (-128.0, -1 / 256, 0.0, 64.0):
                        source = source_group(x, q, scale, bias)
                        assert unsigned_group(x, q, scale, bias) == source
                        assert reconstructed_group(x, q, bits, scale, bias) == source
                        counts["stable_variant_checks"] += 2

    print(json.dumps({"pass": True, "gpu_commands": 0, "model_packages_loaded": 0,
                      "checks": counts, "centered_ranges": ranges,
                      "host_abi_bytes": ctypes.sizeof(Params),
                      "q8_head_planned_bytes": zero_copy_plans,
                      "expected_f32_alternative_counterexamples": alternatives,
                      "remaining_f32_limits": [
                          "Per-weight F32 coefficient rounding differs from factoring.",
                          "MPP dot and SIMD sum reduction orders are not emulated.",
                          "Reconstructing unsigned dot can cancel a small q contribution.",
                          "Sum/dot overflow can precede a mathematically zero result."],
                      "mpp_hardware_precision_qualified": False}, sort_keys=True))


if __name__ == "__main__":
    main()
