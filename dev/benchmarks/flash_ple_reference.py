"""Independent CPU reference for the inspected Flash-Next PLE checkpoint.

Only selected affine rows are decoded. This module never imports MLX, builds a
dense embedding table, submits GPU work, or changes checkpoint files. Norm
reductions use NumPy float32 and may differ in their reduction tree. Gate
reductions preserve MLX Metal's BF16 local sums and SIMD partition.

Source semantics: local oMLX qwen4_exp/language.py, Qwen4ExpNGramEmbedding,
Qwen4ExpPLELayer and Qwen4ExpRMSNorm. The checkpoint arrays below are stored
I64/BF16 values, rather than regenerated hash parameters.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

EOS = 248044
MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)
HEAD_SIZES = (
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
    20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
    20000153, 20000159, 20000161, 20000171,
)
HEAD_OFFSETS = (
    0, 20000003, 40000026, 60000059, 80000106, 100000165,
    120000228, 140000297, 160000374, 180000455, 200000548,
    220000655, 240000802, 260000955, 280001114, 300001275,
)
SHARED_WEIGHT_SCALE = np.float32(0.00019931793212890625)
TABLE_ROWS = 320001536
PARTS = 128
HEAD_DIMENSION = 160
PREFIX = "language_model.model.layers.1.ple"
_MASK64 = (1 << 64) - 1


def signed_i64(value: int) -> int:
    """Two's-complement conversion after an overflowing I64 operation."""
    value &= _MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def bf16(value):
    """BF16 round-to-nearest, ties-to-even, returned in float32 storage."""
    source = np.asarray(value, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    rounded &= np.uint32(0xFFFF0000)
    # Do not round an existing NaN into infinity. Preserve the sign and make
    # quiet NaN independently of payload truncation.
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    rounded = np.where(nan, bits | np.uint32(0x00400000), rounded)
    return np.asarray(rounded & np.uint32(0xFFFF0000)).view(np.float32)


def bf16_bits(value):
    return (bf16(value).view(np.uint32) >> 16).astype(np.uint16)


def shifted_without_crossing_eos(tokens: Sequence[int], shift: int, eos=EOS):
    """Previous EOS is exclusive, so an EOS token still sees prior tokens."""
    if shift < 0:
        raise ValueError("negative history shift")
    if shift == 0:
        return [int(token) for token in tokens]
    result = []
    previous_eos = -1
    for position, token in enumerate(tokens):
        source = position - shift
        valid = source >= previous_eos + 1 and source >= 0
        result.append(int(tokens[source]) if valid else int(eos))
        if int(token) == eos:
            previous_eos = position
    return result


def ngram_indices(
    tokens: Sequence[int],
    previous_context: Sequence[int] | None = None,
    *,
    multipliers=MULTIPLIERS,
    sizes=HEAD_SIZES,
    offsets=HEAD_OFFSETS,
    eos=EOS,
):
    """Return [tokens, 16] rows and the two-token window for the next chunk."""
    if len(multipliers) != 3 or len(sizes) != 16 or len(offsets) != 16:
        raise ValueError("Flash-Next PLE requires 3 multipliers and 16 heads")
    if any(int(size) <= 0 for size in sizes):
        raise ValueError("head vocabulary sizes must be positive")
    context = [eos, eos] if previous_context is None else list(previous_context)
    if len(context) != 2:
        raise ValueError("PLE history must contain exactly two tokens")
    history = context + [int(token) for token in tokens]
    shifted = [shifted_without_crossing_eos(history, i, eos) for i in range(3)]
    rows = []
    for position in range(2, len(history)):
        products = [
            signed_i64(shifted[i][position] * int(multipliers[i]))
            for i in range(3)
        ]
        bigram = signed_i64(products[0] ^ products[1])
        trigram = signed_i64(bigram ^ products[2])
        # Python's positive-divisor remainder is the source's nonnegative
        # remainder. C/Metal signed `%` needs a negative-result correction.
        rows.append([
            (bigram if head < 8 else trigram) % int(sizes[head])
            + int(offsets[head])
            for head in range(16)
        ])
    return np.asarray(rows, dtype=np.int64).reshape(len(tokens), 16), history[-2:]


def shard_offsets(rows=TABLE_ROWS, parts=PARTS):
    if parts <= 0 or parts > rows:
        raise ValueError("invalid shard count")
    base, remainder = divmod(rows, parts)
    offsets = [0]
    for index in range(parts):
        offsets.append(offsets[-1] + base + (index < remainder))
    return tuple(offsets)


def shard_for_row(row: int, offsets=None):
    offsets = shard_offsets() if offsets is None else offsets
    if row < 0 or row >= offsets[-1]:
        raise IndexError("PLE embedding index is outside the sharded vocabulary")
    shard = bisect.bisect_right(offsets, row) - 1
    return shard, row - offsets[shard]


def decode_affine_row(words, scales, biases, *, dimensions=160, group_size=32,
                      intermediate_bf16_product=False):
    """Decode one raw MLX Q4 row, with the canonical BF16 output boundary."""
    words = np.asarray(words, dtype=np.uint32)
    scales, biases = bf16(scales), bf16(biases)
    if dimensions % group_size or dimensions % 8:
        raise ValueError("invalid affine row geometry")
    if words.size != dimensions // 8:
        raise ValueError("packed Q4 row size mismatch")
    if scales.size != dimensions // group_size or biases.size != scales.size:
        raise ValueError("affine parameter row size mismatch")
    dimension = np.arange(dimensions)
    codes = (words[dimension // 8] >> ((dimension % 8) * 4)) & np.uint32(15)
    values = codes.astype(np.float32) * scales[dimension // group_size]
    if intermediate_bf16_product:
        values = bf16(values)
    values += biases[dimension // group_size]
    return bf16(values)


def gather_embeddings(
    indices,
    row_reader: Callable[[int, int], tuple],
    *,
    dimensions=160,
    group_size=32,
    shared_scale=SHARED_WEIGHT_SCALE,
    offsets=None,
    intermediate_bf16_product=False,
):
    """Gather requested rows through a callback, without allocating table rows."""
    indices = np.asarray(indices, dtype=np.int64)
    output = np.empty((*indices.shape, dimensions), dtype=np.float32)
    scale = bf16(shared_scale)
    for position in np.ndindex(indices.shape):
        shard, local = shard_for_row(int(indices[position]), offsets)
        words, scales, biases = row_reader(shard, local)
        # Shared scaling happens after affine dequantization rounds to BF16.
        output[position] = bf16(
            decode_affine_row(words, scales, biases,
                              dimensions=dimensions, group_size=group_size,
                              intermediate_bf16_product=intermediate_bf16_product)
            * scale
        )
    return output


class CheckpointRows:
    """Bounded CPU reads from safetensors for independent checkpoint fixtures."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.index = json.loads(
            (self.directory / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.headers = {}
        self.reads = 0
        self.bytes_read = 0

    def metadata(self, name):
        path = self.directory / self.index[name]
        if path not in self.headers:
            with path.open("rb") as file:
                length = struct.unpack("<Q", file.read(8))[0]
                self.headers[path] = (length + 8, json.loads(file.read(length)))
        start, header = self.headers[path]
        return path, start, header[name]

    def read(self, name, row=None):
        path, base, entry = self.metadata(name)
        dtypes = {"U32": "<u4", "I64": "<i8", "F32": "<f4", "BF16": "<u2"}
        dtype = np.dtype(dtypes[entry["dtype"]])
        shape = tuple(entry["shape"])
        begin, end = entry["data_offsets"]
        if row is not None:
            if len(shape) != 2 or row < 0 or row >= shape[0]:
                raise IndexError("invalid safetensors row")
            row_bytes = shape[1] * dtype.itemsize
            begin += row * row_bytes
            end = begin + row_bytes
            shape = shape[1:]
        with path.open("rb") as file:
            file.seek(base + begin)
            payload = file.read(end - begin)
        if len(payload) != end - begin:
            raise IOError("short safetensors read")
        self.reads += 1
        self.bytes_read += len(payload)
        result = np.frombuffer(payload, dtype=dtype).reshape(shape)
        if entry["dtype"] == "BF16":
            result = (result.astype(np.uint32) << 16).view(np.float32)
        return result

    def ple_row(self, shard, local):
        prefix = f"{PREFIX}.ple_embedding.ngram_embedding.shards.{shard}"
        return tuple(self.read(f"{prefix}.{name}", local)
                     for name in ("weight", "scales", "biases"))


def grouped_rms_norm(x, weight, hidden_size, *, eps=1e-6, one_plus_weight=True):
    x, weight = bf16(x), bf16(weight)
    if x.shape[-1] % hidden_size or weight.shape != (x.shape[-1],):
        raise ValueError("grouped RMSNorm shape mismatch")
    groups = x.reshape(*x.shape[:-1], -1, hidden_size).astype(np.float32)
    variance = np.mean(groups * groups, axis=-1, keepdims=True, dtype=np.float32)
    inverse = np.float32(1) / np.sqrt(variance + np.float32(eps))
    scale = weight.astype(np.float32)
    if one_plus_weight:
        scale = np.float32(1) + scale
    output = groups * inverse * scale.reshape(-1, hidden_size)
    return bf16(output.reshape(x.shape))


def sigmoid_cpu_bf16(x):
    """The MLX CPU functor, whose float constants promote the tail to F32.

    This is intentionally separate from Metal: the CPU functor uses 1.0f,
    while its Metal counterpart uses integer literal 1. Explicit CPU-only MLX
    fixtures confirm these intermediate boundaries; they are not a GPU oracle.
    """
    x = bf16(x)
    with np.errstate(over="ignore"):
        exponential = bf16(np.exp(np.abs(x)))
    complement = np.float32(1) / (np.float32(1) + exponential)
    return bf16(np.where(x < np.float32(0), complement,
                         np.float32(1) - complement))


def sigmoid_metal_bf16(x):
    """The MLX Metal functor's integer-literal/BF16 arithmetic boundaries.

    Metal 4.1 compile-time type checks confirmed both `1 + bfloat` and
    `1 / (1 + bfloat)` have bfloat result type. Thus exp(abs(x)), denominator,
    reciprocal and the nonnegative complement each round separately to BF16.
    This follows the exact stored Metal functor; it differs from the CPU's
    float-literal promotions and from a single final BF16 float32 sigmoid.
    """
    x = bf16(x)
    with np.errstate(over="ignore"):
        exponential = bf16(np.exp(np.abs(x)))
    denominator = bf16(np.float32(1) + exponential)
    tail = bf16(np.float32(1) / denominator)
    return bf16(np.where(x < np.float32(0), tail,
                         bf16(np.float32(1) - tail)))


def sigmoid_bf16(x):
    """Default to Metal math for native-kernel reference fixtures."""
    return sigmoid_metal_bf16(x)


def _sigmoid_for_mode(mode):
    if mode == "metal":
        return sigmoid_metal_bf16
    if mode == "cpu":
        return sigmoid_cpu_bf16
    raise ValueError("sigmoid mode must be metal or cpu")


def silu_bf16(x, *, sigmoid_mode="metal"):
    """MLX nn.silu is x * sigmoid(x), with BF16 at both boundaries."""
    x = bf16(x)
    return bf16(x * _sigmoid_for_mode(sigmoid_mode)(x))


def mlx_metal_bf16_row_sum(products, *, keepdims=False):
    """Reproduce the inspected MLX Metal BF16 contiguous-row sum schedule.

    MLX 0.32.2's backend/metal/reduce.cpp selects a scalar fold for widths up
    to 64. Larger rows assign four consecutive values to each thread, with
    BF16 after every local addition. Installed reduce_row.h retains that
    accumulator type; bf16_math.h reduces each SIMD group in float32 and then
    converts back to BF16. The final SIMD reduction applies the same boundary.
    The two float32 SIMD reductions use NumPy's mathematical reduction tree.
    """
    products = bf16(products)
    if products.ndim == 0 or products.shape[-1] == 0:
        raise ValueError("BF16 row sum requires a nonempty last dimension")
    width = products.shape[-1]
    shape = products.shape[:-1]
    if width <= 64:
        total = np.zeros(shape, dtype=np.float32)
        for column in range(width):
            total = bf16(total + products[..., column])
    else:
        threads = (32 if width <= 512 else 128 if width <= 1024
                   else min(1024, ((width + 127) // 128) * 32))
        local = np.zeros((*shape, threads), dtype=np.float32)
        thread_columns = np.arange(threads) * 4
        for block in range(0, width, threads * 4):
            for offset in range(4):
                columns = block + thread_columns + offset
                valid = columns < width
                contribution = np.zeros((*shape, threads), dtype=np.float32)
                contribution[..., valid] = products[..., columns[valid]]
                local = bf16(local + contribution)
        groups = local.reshape(*shape, threads // 32, 32)
        partials = bf16(np.sum(groups, axis=-1, dtype=np.float32))
        total = bf16(np.sum(partials, axis=-1, dtype=np.float32))
    return total[..., None] if keepdims else total


@dataclass
class PostResult:
    output: np.ndarray
    gated_values: np.ndarray
    normed_conv_inputs: np.ndarray
    conv_state: np.ndarray
    gate: np.ndarray


def ple_post(
    hidden, key_projection, value_projection, norm_key, norm_query, norm_conv,
    conv_weights, *, hidden_size, hc_count=4, state=None, mask=None,
    eps=1e-6, one_plus_weight=True, sigmoid_mode="metal",
):
    """Key/value projection outputs through PLE gating and causal convolution.

    Inputs are [batch,time,hc_count*hidden_size], except value_projection
    [batch,time,hidden_size]. Convolution weights are canonical MLX
    [hc_count*hidden_size,4,1]. Every non-norm expression retains the source's
    BF16 dtype boundaries. Sigmoid defaults to Metal semantics; choose "cpu"
    only to compare against explicit MLX CPU evaluation. Metal mode follows
    the source's BF16 local accumulation schedule and float32 SIMD reduction
    boundaries; NumPy's float32 SIMD reduction tree remains a CPU reference.
    """
    hidden, key_projection, value_projection = map(
        bf16, (hidden, key_projection, value_projection)
    )
    if hidden.ndim != 3 or hidden.shape[-1] != hc_count * hidden_size:
        raise ValueError("PLE hidden-state geometry mismatch")
    batch, length, channels = hidden.shape
    if key_projection.shape != hidden.shape:
        raise ValueError("PLE key projection geometry mismatch")
    if value_projection.shape != (batch, length, hidden_size):
        raise ValueError("PLE value projection geometry mismatch")
    weights = bf16(conv_weights)
    if weights.shape != (channels, 4, 1):
        raise ValueError("PLE convolution requires [channels,4,1] weights")
    norm_options = {"eps": eps, "one_plus_weight": one_plus_weight}
    keys = grouped_rms_norm(key_projection, norm_key, hidden_size, **norm_options)
    queries = grouped_rms_norm(hidden, norm_query, hidden_size, **norm_options)
    product = bf16(keys * queries).reshape(batch, length, hc_count, hidden_size)
    reduced = (mlx_metal_bf16_row_sum(product, keepdims=True)
               if sigmoid_mode == "metal" else
               bf16(np.sum(product, axis=-1, keepdims=True, dtype=np.float32)))
    gate = bf16(reduced / bf16(math.sqrt(hidden_size)))
    magnitude = bf16(np.maximum(np.abs(gate), bf16(1e-6)))
    gate = bf16(np.sign(gate) * bf16(np.sqrt(magnitude)))
    sigmoid = _sigmoid_for_mode(sigmoid_mode)
    gated = bf16(sigmoid(gate) * value_projection[..., None, :])
    gated = gated.reshape(hidden.shape)
    normed = grouped_rms_norm(gated, norm_conv, hidden_size, **norm_options)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (batch, length):
            raise ValueError("PLE mask must have shape [batch,time]")
        gated = np.where(mask[..., None], gated, np.float32(0))
        normed = np.where(mask[..., None], normed, np.float32(0))
    if state is None:
        state = np.zeros((batch, 9, channels), dtype=np.float32)
    state = bf16(state)
    if state.shape != (batch, 9, channels):
        raise ValueError("PLE convolution state must retain nine rows")
    joined = np.concatenate((state, normed), axis=1)
    convolution = np.zeros(hidden.shape, dtype=np.float32)
    for tap in range(4):
        # Conv1d is cross-correlation, oldest tap first; dilation is three.
        convolution += joined[:, 3 * tap:3 * tap + length] * weights[:, tap, 0]
    convolution = bf16(convolution)
    output = bf16(gated + silu_bf16(convolution, sigmoid_mode=sigmoid_mode))
    return PostResult(output, gated, normed, joined[:, -9:].copy(), gate)


def inject(hidden, ple_output):
    """The decoder adds PLE before attention hyper-connections."""
    hidden, ple_output = bf16(hidden), bf16(ple_output)
    if hidden.shape != ple_output.shape:
        raise ValueError("PLE injection geometry mismatch")
    return bf16(hidden + ple_output)


def write_binary_fixtures(directory, *, checkpoint=None, post_lanes=2,
                          post_rows=19, post_width=32):
    """Write bounded raw little-endian fixtures for a native GPU test driver."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = []

    def write(name, value, kind):
        if kind == "u16":
            raw = bf16_bits(value).astype("<u2")
        else:
            raw = np.asarray(value, dtype={"i64": "<i8", "u32": "<u4"}[kind])
        path = directory / f"{name}.{kind}"
        payload = raw.tobytes(order="C")
        path.write_bytes(payload)
        tensors.append({"name": name, "file": path.name, "dtype": kind,
                        "shape": list(raw.shape), "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()})

    tokens = np.array([[7, 8, EOS, 9, 10, 11, EOS, 12]], dtype=np.int64)
    initial_history = np.array([[EOS, EOS]], dtype=np.int64)
    indices, history = ngram_indices(tokens[0], initial_history[0])
    indices = indices[None]
    for name, value in (
        ("tokens", tokens), ("initial_history", initial_history),
        ("expected_ids", indices), ("expected_history", np.array([history])),
        ("multipliers", MULTIPLIERS), ("sizes", HEAD_SIZES),
        ("offsets", HEAD_OFFSETS),
    ):
        write(f"hash-{name}", value, "i64")
    checkpoint_reads = None
    if checkpoint is not None:
        source = CheckpointRows(checkpoint)
        prefix = PREFIX + ".ple_embedding"
        for name, expected in (("layer_multipliers", MULTIPLIERS),
                               ("ngram_heads_vocab_sizes", HEAD_SIZES),
                               ("ngram_heads_offsets", HEAD_OFFSETS)):
            np.testing.assert_array_equal(source.read(f"{prefix}.{name}"), expected)
        np.testing.assert_array_equal(
            source.read(prefix + ".ngram_embedding.weight_scale"),
            [SHARED_WEIGHT_SCALE],
        )
        embedding = gather_embeddings(indices, source.ple_row)
        write("hash-expected_gather", embedding.reshape(1, 8, 2560), "u16")
        write("hash-shared_scale", [SHARED_WEIGHT_SCALE], "u16")
        np.savez(directory / "hash-gather.npz", tokens=tokens, indices=indices,
                 history=np.array([history]), embedding_bf16=bf16_bits(embedding))
        checkpoint_reads = {"selected_rows": int(indices.size),
                            "payload_bytes": source.bytes_read,
                            "tensor_reads": source.reads}

    if min(post_lanes, post_rows, post_width) <= 0:
        raise ValueError("post fixture dimensions must be positive")
    geometry = {"lanes": post_lanes, "rows": post_rows, "width": post_width,
                "streams": 4,
                "one_plus_weight": True, "epsilon": 1e-6,
                "sigmoid_mode": "metal"}
    rng = np.random.default_rng(219)
    lanes, rows, width, streams = post_lanes, post_rows, post_width, 4
    hyper = bf16(rng.normal(size=(lanes, rows, width * streams)))
    key = bf16(rng.normal(size=hyper.shape))
    value = bf16(rng.normal(size=(lanes, rows, width)))
    norms = [bf16(rng.normal(0, 0.1, width * streams)) for _ in range(3)]
    conv = bf16(rng.normal(0, 0.1, (width * streams, 4, 1)))
    state = bf16(rng.normal(0, 0.1, (lanes, 9, width * streams)))
    mask = np.ones((lanes, rows), dtype=np.uint32)
    mask[0, min(3, rows - 1)] = 0
    if lanes > 1:
        mask[1, min(8, rows - 1)] = mask[1, min(16, rows - 1)] = 0
    full = ple_post(hyper, key, value, *norms, conv, hidden_size=width,
                    hc_count=streams, state=state, mask=mask)
    for name, tensor in (
        ("hyper", hyper), ("key", key), ("value", value),
        ("norm_key", norms[0]), ("norm_query", norms[1]),
        ("norm_conv", norms[2]), ("conv", conv),
        ("state_initial", state), ("expected_output", full.output),
        ("expected_state", full.conv_state), ("expected_gate", full.gate),
        ("expected_gated", full.gated_values),
        ("expected_normalized_conv", full.normed_conv_inputs),
        ("expected_injected", inject(hyper, full.output)),
    ):
        write(f"post-{name}", tensor, "u16")
    write("post-mask", mask, "u32")
    chunks, chunk_outputs = [], []
    next_state = state
    boundaries = sorted({0, min(1, rows), min(5, rows), min(9, rows), rows})
    for index, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        part = ple_post(hyper[:, begin:end], key[:, begin:end],
                        value[:, begin:end], *norms, conv, hidden_size=width,
                        hc_count=streams, state=next_state, mask=mask[:, begin:end])
        for name, tensor in (
            ("hyper", hyper[:, begin:end]), ("key", key[:, begin:end]),
            ("value", value[:, begin:end]), ("state_initial", next_state),
            ("expected_output", part.output), ("expected_state", part.conv_state),
        ):
            write(f"post-chunk{index}-{name}", tensor, "u16")
        write(f"post-chunk{index}-mask", mask[:, begin:end], "u32")
        chunks.append({"index": index, "begin": begin, "end": end,
                       "rows": end - begin})
        next_state = part.conv_state
        chunk_outputs.append(part.output)
    chunked = np.concatenate(chunk_outputs, axis=1)
    np.testing.assert_array_equal(chunked, full.output)
    np.testing.assert_array_equal(next_state, full.conv_state)
    write("post-expected_output_chunked", chunked, "u16")
    manifest = {
        "schema": "splash-flash-ple-cpu-fixtures-v1", "cpu_only": True,
        "gpu_commands": 0, "dense_table_materialized": False,
        "binary_order": "little-endian C row-major",
        "hash_geometry": {"lanes": 1, "rows": 8, "heads": 16,
                          "head_width": 160, "eos": EOS,
                          "table_rows": TABLE_ROWS},
        "post_geometry": geometry, "chunks": chunks,
        "post_expected_gate_semantics": "signed-root gate before sigmoid",
        "affine_reference_rounding": "F32 product+bias then BF16; GPU contraction oracle pending",
        "checkpoint_reads": checkpoint_reads,
        "post_chunk_outputs_equal_whole": True, "tensors": tensors,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--post-lanes", type=int, default=2)
    parser.add_argument("--post-rows", type=int, default=19)
    parser.add_argument("--post-width", type=int, default=32)
    args = parser.parse_args()
    manifest = write_binary_fixtures(args.output, checkpoint=args.checkpoint,
                                     post_lanes=args.post_lanes,
                                     post_rows=args.post_rows,
                                     post_width=args.post_width)
    print(json.dumps({"output": str(args.output),
                      "files": len(manifest["tensors"]),
                      "checkpoint_reads": manifest["checkpoint_reads"],
                      "post_chunk_outputs_equal_whole": True}))
