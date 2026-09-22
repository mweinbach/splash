#!/usr/bin/env python3
"""CPU-only, one-layer exact representation of certified Full512 I8 codes.

Only `pack` opens payloads. `inspect` reads bounded JSON and file metadata;
`--cpu-self-test` uses small synthetic arrays/files, never installed operands.
The CLI cannot produce a full-model sidecar or select a production route.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

import numpy as np

ALIGNMENT = 16384
GROUP = 64
EXPERTS = 512
LAYERS = 48
METADATA_LIMIT = 2 << 20
PROJECTIONS = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
SOURCE_IDENTITY = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
SOURCE_MANIFEST_SHA = "0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402"
STORE_MANIFEST_SHA = "ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1"
COEFFICIENT_CERT_SHA = "48ac7919aa82d245647e1bd262b193e8cf70b9dcee5729ec0d6f20787f1148d9"
CONVERSION_REPORT_SHA = "0aea438a13cf047b3348d22d0eb5076822258f5fa8fb2ca161d280d654112615"
CERTIFIED_CONVERTER_SHA = "616c7a279a88d245110ce27674a962f4597adf09bb7c23b54675c16a39529a35"
COEFFICIENT_BOUND = "abs(F64(code)*F64(stored_scale)-F64(source_BF16)) <=0.5*stored_scale +32*2^-24*row_absmax"
BOUND_SCOPE = "coefficient error only; no universal relative dot or whole-model bound"
STORE_SCHEMA = "splash-flash-int8-expert-store-v1"
SCHEMA = "splash-prefill-i8-lut-one-layer-v1"
CERT_SCHEMA = "splash-prefill-i8-lut-exact-byte-certificate-v1"
COEFFICIENT_POLICY = "source_q4_g64_f32_separate_multiply_add_then_bf16_rne_v1"
QUANTIZATION = "signed-symmetric-int8-rowwise-f32-scale"
ROUNDING = "F32 absmax/127; F32 division; nearest-even integer; clamp [-127,127]; zero row scale=1"
STORE_KEYS = {"schema", "source_identity_sha256", "source_manifest_sha256", "plan_sha256", "alignment", "target_layers", "selected_experts", "coefficient_policy", "quantization_format", "integer_rounding", "layers", "total_bytes", "planned_allocation_bytes"}
PLANE_KEYS = {"dtype", "shape", "offset", "length", "sha256"}


class PackError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PackError(message)


def integer(value, *, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum), "invalid integer metadata")
    return value


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def aligned(value):
    return (integer(value) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def nonfinite(value):
        raise PackError(f"non-finite JSON token: {value}")

    def finite_float(value):
        parsed = float(value)
        require(math.isfinite(parsed), f"non-finite JSON number: {value}")
        return parsed

    try:
        out = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)
    except (ValueError, UnicodeError) as error:
        raise PackError(f"invalid JSON: {error}") from error
    require(isinstance(out, dict), "metadata root must be an object")
    return out


def snapshot(path):
    details = Path(path).lstat()
    require(stat.S_ISREG(details.st_mode), f"not a regular file (symlinks refused): {path}")
    return (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns)


def open_readonly(path, *, expected_size=None, readonly=True):
    before = snapshot(path)
    details = Path(path).lstat()
    require(not readonly or not details.st_mode & 0o222, f"payload is writable: {path}")
    require(expected_size is None or before[2] == expected_size, f"payload size mismatch: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    current = os.fstat(fd)
    actual = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
    if actual != before:
        os.close(fd)
        raise PackError(f"file changed while opening: {path}")
    return fd, before


def read_metadata(path):
    fd, before = open_readonly(path, readonly=False)
    try:
        require(0 < before[2] <= METADATA_LIMIT, f"unbounded metadata: {path}")
        raw = os.pread(fd, before[2] + 1, 0)
        require(len(raw) == before[2] and snapshot(path) == before, f"metadata changed: {path}")
        return raw, strict_json(raw), before
    finally:
        os.close(fd)


def pread_exact(fd, length, offset):
    integer(length); integer(offset)
    chunks, cursor = [], 0
    while cursor < length:
        raw = os.pread(fd, min(8 << 20, length - cursor), offset + cursor)
        require(bool(raw), "truncated payload read")
        chunks.append(raw)
        cursor += len(raw)
    return b"".join(chunks)


def pwrite_exact(fd, raw, offset):
    integer(offset)
    view, cursor = memoryview(raw), 0
    while cursor < len(view):
        written = os.pwrite(fd, view[cursor:], offset + cursor)
        require(written > 0, "short output write")
        cursor += written


def range_sha(fd, offset, length):
    digest, cursor = hashlib.sha256(), 0
    while cursor < length:
        raw = os.pread(fd, min(8 << 20, length - cursor), offset + cursor)
        require(bool(raw), "truncated payload during hash verification")
        digest.update(raw)
        cursor += len(raw)
    return digest.hexdigest()


def check_shape(value, expected):
    require(isinstance(value, list) and len(value) == len(expected), "tensor rank mismatch")
    require(all(type(item) is int and item > 0 for item in value) and value == list(expected), "tensor shape mismatch")


def check_ids(ids, count):
    require(isinstance(ids, list) and len(ids) == count and all(type(x) is int for x in ids) and ids == list(range(count)), "rank order must be complete expert-ID order")


def plane(dtype, shape, offset):
    size = math.prod(shape) * (4 if dtype == "F32" else 1)
    return {"dtype": dtype, "shape": list(shape), "offset": aligned(offset), "length": size, "sha256": None}


def check_plane(item, expected):
    require(isinstance(item, dict) and set(item) == PLANE_KEYS, "unknown or missing plane metadata")
    require(item["dtype"] == expected["dtype"], "plane dtype mismatch")
    check_shape(item["shape"], expected["shape"])
    require(integer(item["offset"]) == expected["offset"] and integer(item["length"], minimum=1) == expected["length"], "plane extent/alignment mismatch")
    require(valid_sha(item["sha256"]), "invalid plane SHA256")


def layer_layout(layer, *, packed, count=EXPERTS, geometry=PROJECTIONS):
    integer(layer, maximum=LAYERS - 1); integer(count, minimum=1, maximum=EXPERTS)
    require(set(geometry) == set(PROJECTIONS), "all three projections are required")
    cursor, entries = 0, {}
    for name in PROJECTIONS:
        n, k = geometry[name]
        integer(n, minimum=1); integer(k, minimum=GROUP)
        require(k % GROUP == 0, "K must contain complete G64 groups")
        entry = {"source_prefix": f"language_model.model.layers.{layer}.mlp.switch_mlp.{name}", "dimensions": [count, n, k]}
        specs = (("ids", "U8", [count, n, k // 2]), ("lut", "I8", [count, n, k // GROUP, 16]), ("scales", "F32", [count, n])) if packed else (("codes", "I8", [count, n, k]), ("scales", "F32", [count, n]))
        for key, dtype, shape in specs:
            entry[key] = plane(dtype, shape, cursor)
            cursor = entry[key]["offset"] + entry[key]["length"]
        entries[name] = entry
    return {"layer_index": layer, "path": f"layer-{layer:02d}.bin", "bytes": aligned(cursor), "sha256": None, "projections": entries}


def check_layer(item, layer, *, packed=False, count=EXPERTS, geometry=PROJECTIONS):
    expected = layer_layout(layer, packed=packed, count=count, geometry=geometry)
    require(isinstance(item, dict) and set(item) == {"layer_index", "path", "bytes", "sha256", "projections"}, "unknown or missing layer metadata")
    require(type(item["layer_index"]) is int and item["layer_index"] == layer and item["path"] == expected["path"], "layer/path binding mismatch")
    require(integer(item["bytes"], minimum=1) == expected["bytes"] and valid_sha(item["sha256"]), "layer size/hash mismatch")
    require(isinstance(item["projections"], dict) and set(item["projections"]) == set(PROJECTIONS), "projection cardinality mismatch")
    for name, matrix in item["projections"].items():
        wanted = expected["projections"][name]
        require(isinstance(matrix, dict) and set(matrix) == set(wanted), "unknown or missing projection metadata")
        require(matrix["source_prefix"] == wanted["source_prefix"], "projection source binding mismatch")
        check_shape(matrix["dimensions"], wanted["dimensions"])
        for key in ("ids", "lut", "scales") if packed else ("codes", "scales"):
            check_plane(matrix[key], wanted[key])


def check_store(store):
    require(set(store) == STORE_KEYS, "unknown or missing Full512 manifest fields")
    require(store["schema"] == STORE_SCHEMA and store["source_identity_sha256"] == SOURCE_IDENTITY and store["source_manifest_sha256"] == SOURCE_MANIFEST_SHA, "Full512 source/store identity mismatch")
    require(store["coefficient_policy"] == COEFFICIENT_POLICY and store["quantization_format"] == QUANTIZATION and store["integer_rounding"] == ROUNDING, "Full512 arithmetic policy mismatch")
    require(type(store["alignment"]) is int and store["alignment"] == ALIGNMENT and type(store["target_layers"]) is int and store["target_layers"] == LAYERS and valid_sha(store["plan_sha256"]), "Full512 alignment/layer/plan metadata mismatch")
    require(isinstance(store["layers"], list) and isinstance(store["selected_experts"], list) and len(store["layers"]) == LAYERS and len(store["selected_experts"]) == LAYERS, "Full512 layer cardinality mismatch")
    for index in range(LAYERS):
        check_ids(store["selected_experts"][index], EXPERTS)
        check_layer(store["layers"][index], index)
    require(integer(store["total_bytes"]) == 121173442560 and integer(store["planned_allocation_bytes"]) == 121174228992, "Full512 byte ledger mismatch")


def check_certificate(cert, store_dir):
    require(cert.get("pass") is True and cert.get("gpu_work") is False and cert.get("certificate_completed_before_atomic_publication") is True, "certified Full512 publication proof is incomplete")
    require(cert.get("full_manifest_sha256") == STORE_MANIFEST_SHA and cert.get("output") == str(store_dir), "Full512 coefficient certificate binding mismatch")
    for key, value in {"coefficient_elements": 120795955200, "rows": 94371840, "top64_code_bytes_exact": 15099494400, "top64_scale_bytes_exact": 47185920, "total_bytes": 121173442560, "planned_allocation_bytes": 121174228992}.items():
        require(type(cert.get(key)) is int and cert[key] == value, f"incomplete Full512 coefficient certificate: {key}")
    require(cert.get("preserved_top64_payload_snapshots_unchanged") is True and cert.get("source_converter_sha256") == CERTIFIED_CONVERTER_SHA, "Full512 source certificate provenance mismatch")
    require(cert.get("coefficient_bound") == COEFFICIENT_BOUND and cert.get("bound_scope") == BOUND_SCOPE, "unsupported Full512 coefficient bound/scope")
    for key in ("maximum_absolute_error", "maximum_error_divided_by_bound", "error_square_sum", "reference_square_sum"):
        value = cert.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"missing/invalid coefficient audit metric: {key}")
    require(cert["maximum_error_divided_by_bound"] <= 1 and cert["reference_square_sum"] > 0 and integer(cert.get("zero_rows")) <= cert["rows"], "Full512 coefficient audit bound/count failed")
    require(cert.get("model_quality_qualified") is False, "Full512 certificate quality scope changed")


def relative_file(package, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "source shard path must remain inside package")
    path = package / relative
    require(path.resolve() == path and path.is_relative_to(package), "source shard path follows a symlink or escapes package")
    snapshot(path)
    return path


def check_source_tensor(item, *, package, records, shape, dtype):
    require(isinstance(item, dict), "source tensor is missing")
    require(item.get("dtype") == dtype, "original Q4 tensor dtype mismatch")
    check_shape(item.get("shape"), shape)
    size = math.prod(shape) * (4 if dtype == "U32" else 2)
    require(integer(item.get("offset")) % ALIGNMENT == 0 and integer(item.get("length"), minimum=1) == size, "original Q4 tensor length/alignment mismatch")
    path = relative_file(package, item.get("shard"))
    record = records.get(item["shard"])
    require(isinstance(record, dict) and valid_sha(record.get("sha256")), "source shard lacks a hash identity")
    details = path.lstat()
    # The published conversion report calls its O_RDONLY mappings "readonly
    # snapshots"; the original checkpoint files are 0644. Preserve those files
    # and require their exact hash-certified stat tuples rather than chmodding.
    require(integer(record.get("bytes"), minimum=1) == details.st_size and item["offset"] + size <= details.st_size, "original Q4 shard is truncated or differs from manifest size")
    return path


def prepare(source_dir, store_dir, layer, certificate=None, conversion_report=None):
    """Validate metadata/stat certificates; do not open any payload descriptor."""
    integer(layer, maximum=LAYERS - 1)
    source_dir, store_dir = Path(source_dir).absolute(), Path(store_dir).absolute()
    require(source_dir.resolve() == source_dir and store_dir.resolve() == store_dir and source_dir.is_dir() and store_dir.is_dir(), "source/store directories must be canonical without symlinks")
    certificate = Path(certificate) if certificate else store_dir / "coefficient-certificate.json"
    conversion_report = Path(conversion_report) if conversion_report else store_dir / "conversion-report.json"
    metadata_paths = [source_dir / "manifest.json", store_dir / "manifest.json", certificate, conversion_report]
    source_read, store_read, cert_read, report_read = [read_metadata(path) for path in metadata_paths]
    source_raw, source, _ = source_read
    store_raw, store, _ = store_read
    cert_raw, cert, _ = cert_read
    report_raw, report, _ = report_read
    require(sha(source_raw) == SOURCE_MANIFEST_SHA and sha(store_raw) == STORE_MANIFEST_SHA, "certified source/Full512 manifest bytes changed")
    require(sha(cert_raw) == COEFFICIENT_CERT_SHA and sha(report_raw) == CONVERSION_REPORT_SHA, "published Full512 certificate/conversion-report bytes changed")
    check_store(store); check_certificate(cert, store_dir)
    require(source.get("schema") == "splash-local-qwen4-affine-v1" and source.get("source_identity_sha256") == SOURCE_IDENTITY, "original Q4 source identity mismatch")
    require(report.get("source") == str(source_dir) and report.get("source_manifest_and_plan_sha_reverified_at_end") is True and report.get("source_plan_and_shard_snapshots_unchanged") is True and type(report.get("source_shards_verified_against_original_manifest_sha")) is int and report["source_shards_verified_against_original_manifest_sha"] == 14, "source hash/snapshot conversion report is incomplete")
    require(type(report.get("stored_coefficient_and_scale_bytes")) is int and report["stored_coefficient_and_scale_bytes"] == store["total_bytes"], "conversion report byte ledger mismatch")
    prior = report.get("source_readonly_snapshots")
    require(isinstance(prior, dict), "conversion report lacks certified source snapshots")
    require(isinstance(source.get("shards"), list) and isinstance(source.get("tensors"), dict) and isinstance(source.get("quantization"), dict), "source manifest lacks projection metadata")
    records = {}
    for row in source["shards"]:
        require(isinstance(row, dict) and isinstance(row.get("path"), str) and row["path"] not in records, "source shard records must be unique")
        records[row["path"]] = row
    q4, snapshots, bindings = {}, {path: read[2] for path, read in zip(metadata_paths, [source_read, store_read, cert_read, report_read])}, []
    require(prior.get(str(metadata_paths[0])) == list(snapshots[metadata_paths[0]]), "source manifest differs from certified conversion snapshot")
    for name, (n, k) in PROJECTIONS.items():
        prefix = store["layers"][layer]["projections"][name]["source_prefix"]
        quant = source["quantization"].get(prefix, {"bits": source["quantization"].get("bits"), "group_size": source["quantization"].get("group_size"), "mode": "affine"})
        require(isinstance(quant, dict) and type(quant.get("bits")) is int and quant["bits"] == 4 and type(quant.get("group_size")) is int and quant["group_size"] == GROUP and quant.get("mode") == "affine", "source projection must be affine Q4/G64")
        for suffix, dtype, shape in (("weight", "U32", [EXPERTS, n, k // 8]), ("scales", "BF16", [EXPERTS, n, k // GROUP]), ("biases", "BF16", [EXPERTS, n, k // GROUP])):
            descriptor = source["tensors"].get(f"{prefix}.{suffix}")
            path = check_source_tensor(descriptor, package=source_dir, records=records, shape=shape, dtype=dtype)
            current = snapshot(path)
            require(prior.get(str(path)) == list(current), f"source shard differs from its full-hash-certified snapshot: {path}")
            snapshots[path] = current
            if suffix == "weight":
                q4[name] = {"path": str(path), **copy.deepcopy(descriptor), "source_shard_sha256": records[descriptor["shard"]]["sha256"]}
                bindings.append({"projection": name, **q4[name], "source_snapshot": list(current)})
    entry = store["layers"][layer]
    input_path = store_dir / entry["path"]
    details = input_path.lstat()
    require(stat.S_ISREG(details.st_mode) and not details.st_mode & 0o222 and details.st_size == entry["bytes"], "bounded Full512 layer must be read-only regular and exact-size")
    snapshots[input_path] = snapshot(input_path)
    return {"source_dir": source_dir, "store_dir": store_dir, "layer": entry, "input_path": input_path, "q4": q4, "snapshots": snapshots, "bindings": bindings, "metadata_hashes": {"coefficient_certificate_sha256": sha(cert_raw), "conversion_report_sha256": sha(report_raw)}}


def validate_arrays(ids, codes):
    require(isinstance(ids, np.ndarray) and isinstance(codes, np.ndarray) and ids.dtype == np.dtype("u1") and codes.dtype == np.dtype("i1"), "packing requires U8 ID bytes and signed I8 saved codes")
    require(ids.ndim == 3 and codes.ndim == 3 and all(v > 0 for v in codes.shape), "packing requires nonempty rank-three tensors")
    b, n, k = codes.shape
    require(k % GROUP == 0 and ids.shape == (b, n, k // 2), "ID/code shape or G64 geometry mismatch")
    require(not np.any(codes == -128), "symmetric saved I8 codes cannot contain -128")
    return b, n, k


def derive_lut(ids, codes):
    """Derive a coherent Q4-ID→I8 function using byte-nibble expansion.

    Quantization collisions are legitimate. This is not an injective mapping;
    each observed Q4 ID must select exactly one code inside its G64 group.
    Unobserved IDs receive deterministic zero entries.
    """
    b, n, k = validate_arrays(ids, codes)
    by_group = ids.reshape(b, n, k // GROUP, GROUP // 2)
    expanded = np.empty((b, n, k // GROUP, GROUP), dtype=np.uint8)
    expanded[..., 0::2] = by_group & np.uint8(15)
    expanded[..., 1::2] = by_group >> np.uint8(4)
    saved = codes.reshape(expanded.shape)
    lut = np.zeros((b, n, k // GROUP, 16), dtype=np.int8)
    for code_id in range(16):
        mask = expanded == code_id
        present = np.any(mask, axis=-1)
        first = np.argmax(mask, axis=-1)
        value = np.take_along_axis(saved, first[..., None], axis=-1)[..., 0]
        require(not np.any(mask & (saved != value[..., None])), f"incoherent Q4-ID {code_id} maps to multiple saved I8 codes in a group")
        lut[..., code_id] = np.where(present, value, 0)
    return lut


def reconstruct_words(ids, lut, dimensions):
    """Independent decoder: LE U32 words/shifts rather than byte nibbles."""
    require(isinstance(dimensions, (list, tuple)) and len(dimensions) == 3 and all(type(x) is int and x > 0 for x in dimensions), "reconstruction requires positive rank-three dimensions")
    b, n, k = dimensions
    require(k % GROUP == 0 and isinstance(ids, np.ndarray) and ids.dtype == np.dtype("u1") and ids.shape == (b, n, k // 2), "reconstruction ID geometry/dtype mismatch")
    require(isinstance(lut, np.ndarray) and lut.dtype == np.dtype("i1") and lut.shape == (b, n, k // GROUP, 16), "reconstruction LUT geometry/dtype mismatch")
    words = np.ascontiguousarray(ids).view("<u4").reshape(b, n, k // 8)
    shifts = np.arange(8, dtype=np.uint32) * np.uint32(4)
    values = ((words[..., None] >> shifts) & np.uint32(15)).reshape(b, n, k // GROUP, GROUP)
    return np.take_along_axis(lut, values.astype(np.intp), axis=-1).reshape(b, n, k)


def check_scales(raw, b, n):
    require(len(raw) == b * n * 4, "saved F32 scale shape/length mismatch")
    values = np.frombuffer(raw, dtype="<f4")
    require(np.isfinite(values).all() and (values > np.float32(0)).all(), "saved row scales must remain finite and positive")


def readback(output_fd, input_fd, packed, original, batch_experts, *, count=EXPERTS, geometry=PROJECTIONS):
    checks = {}
    for name, (n, k) in geometry.items():
        wanted, source = packed["projections"][name], original["projections"][name]
        digests = {key: hashlib.sha256() for key in ("ids", "lut", "scales")}
        code_bytes, scale_bytes = 0, 0
        for begin in range(0, count, batch_experts):
            b = min(batch_experts, count - begin)
            raw_ids = pread_exact(output_fd, b * n * k // 2, wanted["ids"]["offset"] + begin * n * k // 2)
            raw_lut = pread_exact(output_fd, b * n * k // 4, wanted["lut"]["offset"] + begin * n * k // 4)
            raw_scales = pread_exact(output_fd, b * n * 4, wanted["scales"]["offset"] + begin * n * 4)
            ids = np.frombuffer(raw_ids, dtype=np.uint8).reshape(b, n, k // 2)
            lut = np.frombuffer(raw_lut, dtype=np.int8).reshape(b, n, k // GROUP, 16)
            decoded = reconstruct_words(ids, lut, (b, n, k))
            expected_codes = pread_exact(input_fd, b * n * k, source["codes"]["offset"] + begin * n * k)
            expected_scales = pread_exact(input_fd, b * n * 4, source["scales"]["offset"] + begin * n * 4)
            require(decoded.tobytes() == expected_codes, f"stored independent reconstruction mismatch: {name}/{begin}")
            require(raw_scales == expected_scales, f"stored late-scale byte mismatch: {name}/{begin}")
            for key, raw in (("ids", raw_ids), ("lut", raw_lut), ("scales", raw_scales)):
                digests[key].update(raw)
            code_bytes += len(expected_codes); scale_bytes += len(expected_scales)
        for key, digest in digests.items():
            require(digest.hexdigest() == wanted[key]["sha256"], f"stored {name}/{key} hash differs from written bytes")
        require(code_bytes == count * n * k and scale_bytes == count * n * 4, "independent reconstruction skipped a projection extent")
        checks[name] = {"coefficient_code_bytes": code_bytes, "scale_bytes": scale_bytes, "exact_code_bytes": True, "exact_scale_bytes": True}
    return checks


def write_json(path, item):
    raw = (json.dumps(item, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        require(stream.write(raw) == len(raw), "short metadata write")
        stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)
    return sha(raw)


def publish_exclusive(temporary, output):
    """Atomically publish without replacing even a concurrently created dir."""
    libc = ctypes.CDLL(None, use_errno=True)
    platform = os.uname().sysname
    if platform == "Darwin":
        # <sys/stdio.h>: RENAME_EXCL = 0x00000004.
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(temporary), os.fsencode(output), 0x4)
    elif platform == "Linux" and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(temporary), -100, os.fsencode(output), 1)
    else:
        raise PackError("platform lacks the required exclusive atomic rename")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def pack(prepared, output, *, batch_experts=4, emit=print):
    integer(batch_experts, minimum=1, maximum=8)
    output = Path(output).absolute()
    require(output.resolve() == output and not output.exists() and output.parent.is_dir(), "output must be a fresh canonical directory with an existing parent")
    for directory in (prepared["source_dir"], prepared["store_dir"]):
        require(not output.is_relative_to(directory) and not directory.is_relative_to(output), "output must be separate from source/store")
    original = prepared["layer"]
    layout = layer_layout(original["layer_index"], packed=True)
    require(shutil.disk_usage(output.parent).free >= layout["bytes"] + (64 << 20), "insufficient free disk for bounded packed layer")
    started = time.monotonic()
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    input_fd = output_fd = None
    try:
        input_fd, before = open_readonly(prepared["input_path"], expected_size=original["bytes"])
        require(before == prepared["snapshots"][prepared["input_path"]], "Full512 layer changed since preflight")
        output_path = temporary / layout["path"]
        output_fd = os.open(output_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        os.ftruncate(output_fd, layout["bytes"])
        q4_range_hashes = {}
        for name, (n, k) in PROJECTIONS.items():
            wanted, source = layout["projections"][name], original["projections"][name]
            q4 = prepared["q4"][name]
            q4_path = Path(q4["path"])
            q4_fd, before = open_readonly(q4_path, readonly=False)
            try:
                require(before == prepared["snapshots"][q4_path], "original Q4 shard changed since preflight")
                digests = {key: hashlib.sha256() for key in ("ids", "lut", "scales", "codes")}
                for begin in range(0, EXPERTS, batch_experts):
                    b = min(batch_experts, EXPERTS - begin)
                    raw_ids = pread_exact(q4_fd, b * n * k // 2, q4["offset"] + begin * n * k // 2)
                    raw_codes = pread_exact(input_fd, b * n * k, source["codes"]["offset"] + begin * n * k)
                    raw_scales = pread_exact(input_fd, b * n * 4, source["scales"]["offset"] + begin * n * 4)
                    check_scales(raw_scales, b, n)
                    ids = np.frombuffer(raw_ids, dtype=np.uint8).reshape(b, n, k // 2)
                    codes = np.frombuffer(raw_codes, dtype=np.int8).reshape(b, n, k)
                    lut = derive_lut(ids, codes)
                    raw_lut = lut.tobytes(order="C")
                    require(reconstruct_words(ids, lut, (b, n, k)).tobytes() == raw_codes, "independent pre-write reconstruction mismatch")
                    for key, raw, per_expert in (("ids", raw_ids, n * k // 2), ("lut", raw_lut, n * k // 4), ("scales", raw_scales, n * 4)):
                        pwrite_exact(output_fd, raw, wanted[key]["offset"] + begin * per_expert)
                        digests[key].update(raw)
                    digests["codes"].update(raw_codes)
                require(digests["codes"].hexdigest() == source["codes"]["sha256"] and digests["scales"].hexdigest() == source["scales"]["sha256"], f"bounded original I8 plane checksum mismatch: {name}")
                for key in ("ids", "lut", "scales"):
                    wanted[key]["sha256"] = digests[key].hexdigest()
                q4_range_hashes[name] = digests["ids"].hexdigest()
            finally:
                os.close(q4_fd)
            emit(json.dumps({"phase": "projection_packed", "layer": original["layer_index"], "projection": name, "experts": EXPERTS, "saved_i8_bytes": EXPERTS * n * k, "packed_id_lut_bytes": EXPERTS * n * k * 3 // 4}, sort_keys=True), flush=True)
        os.fsync(output_fd)
        checks = readback(output_fd, input_fd, layout, original, batch_experts)
        require(range_sha(input_fd, 0, original["bytes"]) == original["sha256"], "bounded original I8 layer checksum mismatch")
        layout["sha256"] = range_sha(output_fd, 0, layout["bytes"])
        require(os.fstat(output_fd).st_size == layout["bytes"], "packed layer file size mismatch")
        check_layer(layout, original["layer_index"], packed=True)
        for path, before in prepared["snapshots"].items():
            require(snapshot(path) == before, f"certified input changed during packing: {path}")
        os.close(output_fd); output_fd = None
        os.close(input_fd); input_fd = None
        output_path.chmod(0o444)
        code_bytes = sum(row["coefficient_code_bytes"] for row in checks.values())
        scale_bytes = sum(row["scale_bytes"] for row in checks.values())
        require(code_bytes == 2516582400 and scale_bytes == 7864320, "one-layer certificate cardinality mismatch")
        bindings = [{**row, "range_sha256": q4_range_hashes[row["projection"]]} for row in prepared["bindings"]]
        manifest = {"schema": SCHEMA, "alignment": ALIGNMENT, "group_size": GROUP, "layer_index": original["layer_index"], "rank_order": "expert_id", "selected_experts": list(range(EXPERTS)), "source_identity_sha256": SOURCE_IDENTITY, "source_manifest_sha256": SOURCE_MANIFEST_SHA, "source_store_manifest_sha256": STORE_MANIFEST_SHA, "path": layout["path"], "bytes": layout["bytes"], "sha256": layout["sha256"], "projections": layout["projections"], "source_i8_layer": {"layer_index": original["layer_index"], "path": str(prepared["input_path"]), "logical_size": original["bytes"], "sha256": original["sha256"], "projections": copy.deepcopy(original["projections"])}, "original_q4_id_inputs": bindings, "source_metadata_certificates": prepared["metadata_hashes"], "packing_policy": "original little-endian Q4 U32 bytes; low/even nibble first; observed ID selects coherent signed I8 code per G64; absent ID LUT entries zero", "arithmetic_policy": "all saved signed-I8 codes exact; original late F32 row-scale bytes retained; no requantization or activation/accumulation change", "packed_weight_bytes_per_g64": 48, "original_i8_weight_bytes_per_g64": 64}
        manifest_sha = write_json(temporary / "manifest.json", manifest)
        certificate = {"schema": CERT_SCHEMA, "pass": True, "gpu_work": False, "manifest_sha256": manifest_sha, "layer_index": original["layer_index"], "source_store_manifest_sha256": STORE_MANIFEST_SHA, "all_saved_i8_bytes_reconstructed_exactly": True, "all_saved_scale_bytes_copied_exactly": True, "stored_output_readback_verified": True, "source_and_store_snapshots_unchanged": True, "coefficient_code_bytes": code_bytes, "scale_bytes": scale_bytes, "projection_checks": checks, "bounded_original_i8_layer_sha256_verified": original["sha256"], "original_q4_id_bytes": code_bytes // 2, "packed_id_lut_bytes": code_bytes * 3 // 4, "packed_file_bytes": layout["bytes"], "source_shards_full_rehashed": False, "full_store_payload_rehashed": False, "source_shard_hash_policy": "prior conversion-report full-shard verification plus exact certified dev/ino/size/mtime/ctime snapshots with O_RDONLY mappings, rechecked before and after", "atomic_publication_after_certificate": True, "batch_experts": batch_experts, "elapsed_seconds": time.monotonic() - started}
        write_json(temporary / "certificate.json", certificate)
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        require(not output.exists(), "output appeared during packing")
        publish_exclusive(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return {"pass": True, "gpu_work": False, "output": str(output), "layer": original["layer_index"], "manifest_sha256": manifest_sha, "bytes": layout["bytes"], "coefficient_code_bytes": code_bytes, "scale_bytes": scale_bytes}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if input_fd is not None:
            os.close(input_fd)
        if output_fd is not None:
            os.close(output_fd)


def cpu_self_test():
    checks = []

    def check(value, name):
        require(value, f"CPU self-test failed: {name}")
        checks.append(name)

    def reject(action, name):
        try:
            action()
        except (PackError, OSError):
            checks.append(name)
        else:
            raise PackError(f"CPU negative self-test accepted: {name}")

    # Fixtures are independent of the converter: explicit nibble/lookup formula.
    qids = np.arange(2 * 3 * 128, dtype=np.uint16).reshape(2, 3, 128) % 16
    qids = qids.astype(np.uint8)
    ids = (qids[..., 0::2] | (qids[..., 1::2] << np.uint8(4))).copy()
    group_index = np.arange(12, dtype=np.int16).reshape(2, 3, 2, 1)
    table = (np.arange(16, dtype=np.int16)[None, None, None, :] * 7 - 60 + group_index).astype(np.int8)
    codes = np.take_along_axis(table, qids.reshape(2, 3, 2, 64).astype(np.intp), axis=-1).reshape(2, 3, 128)
    lut = derive_lut(ids, codes)
    check(np.array_equal(lut, table), "all sixteen IDs and rank-row-group ordering")
    check(np.array_equal(reconstruct_words(ids, lut, codes.shape), codes), "independent little-endian U32 reconstruction")
    check(ids[0, 0, :4].tolist() == [0x10, 0x32, 0x54, 0x76], "low-even nibble packing")
    collision = np.zeros_like(codes)
    check(np.all(derive_lut(ids, collision) == 0), "quantization collisions and zero groups are coherent")
    absent = np.full_like(ids, 0x33)
    absent_codes = np.full_like(codes, -127)
    absent_lut = derive_lut(absent, absent_codes)
    check(np.all(absent_lut[..., 3] == -127) and np.all(np.delete(absent_lut, 3, axis=-1) == 0), "absent IDs canonical zero and signed endpoint")
    incoherent = codes.copy(); incoherent[0, 0, 16] += np.int8(1)
    reject(lambda: derive_lut(ids, incoherent), "one ID mapping to two saved codes rejected")
    minus128 = codes.copy(); minus128[0, 0, 0] = -128
    reject(lambda: derive_lut(ids, minus128), "forbidden symmetric minus128 code rejected")
    reject(lambda: derive_lut(ids.astype(np.int8), codes), "ID signed dtype rejected")
    reject(lambda: derive_lut(ids, codes.astype(np.uint8)), "code unsigned dtype rejected")
    reject(lambda: derive_lut(ids[0], codes), "ID rank mismatch rejected")
    reject(lambda: derive_lut(ids[..., :-1], codes), "ID shape mismatch rejected")
    reject(lambda: derive_lut(ids[..., :31], codes[..., :62]), "partial G64 rejected")
    reject(lambda: reconstruct_words(ids, lut[..., :15], codes.shape), "LUT cardinality rejected")
    reject(lambda: reconstruct_words(ids, lut, (True, 3, 128)), "boolean dimension rejected")
    reject(lambda: check_ids([1, 0], 2), "unsorted rank IDs rejected")
    reject(lambda: check_ids([0, 0], 2), "duplicate rank IDs rejected")
    reject(lambda: check_ids([False, 1], 2), "boolean rank ID rejected")
    reject(lambda: strict_json(b'{"x":1,"x":2}'), "duplicate JSON field rejected")
    reject(lambda: strict_json(b'{"x":NaN}'), "nonfinite JSON rejected")
    reject(lambda: strict_json(b'{"x":1e400}'), "overflowed JSON exponent rejected")
    reject(lambda: strict_json(b'{"x":"\xff"}'), "invalid UTF8 rejected")
    reject(lambda: strict_json(b'[]'), "nonobject metadata rejected")
    geometry = {"gate_proj": (3, 128), "up_proj": (3, 128), "down_proj": (6, 64)}
    small = layer_layout(0, packed=True, count=2, geometry=geometry)
    small["sha256"] = "a" * 64
    for matrix in small["projections"].values():
        for key in ("ids", "lut", "scales"):
            matrix[key]["sha256"] = "b" * 64
    check_layer(small, 0, packed=True, count=2, geometry=geometry)
    checks.append("small complete three-projection canonical layout")
    bad = copy.deepcopy(small); del bad["projections"]["up_proj"]
    reject(lambda: check_layer(bad, 0, packed=True, count=2, geometry=geometry), "missing projection rejected")
    bad = copy.deepcopy(small); bad["projections"]["gate_proj"]["lut"]["offset"] += 1
    reject(lambda: check_layer(bad, 0, packed=True, count=2, geometry=geometry), "misaligned LUT extent rejected")
    bad = copy.deepcopy(small); bad["projections"]["down_proj"]["dimensions"][0] = True
    reject(lambda: check_layer(bad, 0, packed=True, count=2, geometry=geometry), "boolean tensor extent rejected")
    bad = copy.deepcopy(small); bad["projections"]["up_proj"]["ids"]["shape"].append(1)
    reject(lambda: check_layer(bad, 0, packed=True, count=2, geometry=geometry), "extra tensor rank rejected")
    bad = copy.deepcopy(small); bad["projections"]["gate_proj"]["scales"]["length"] -= 1
    reject(lambda: check_layer(bad, 0, packed=True, count=2, geometry=geometry), "truncated scale extent rejected")
    bad = copy.deepcopy(small); bad["projections"]["gate_proj"]["ids"]["sha256"] = "A" * 64
    reject(lambda: check_layer(bad, 0, packed=True, count=2, geometry=geometry), "uppercase digest rejected")
    reject(lambda: check_scales(np.array([0, 1], dtype="<f4").tobytes(), 1, 2), "zero row scale rejected")
    reject(lambda: check_scales(np.array([np.inf, 1], dtype="<f4").tobytes(), 1, 2), "nonfinite row scale rejected")
    synthetic_cert = {"pass": True, "gpu_work": False, "certificate_completed_before_atomic_publication": True, "full_manifest_sha256": STORE_MANIFEST_SHA, "output": "/synthetic-unused-store", "coefficient_elements": 120795955200, "rows": 94371840, "top64_code_bytes_exact": 15099494400, "top64_scale_bytes_exact": 47185920, "total_bytes": 121173442560, "planned_allocation_bytes": 121174228992, "preserved_top64_payload_snapshots_unchanged": True, "source_converter_sha256": CERTIFIED_CONVERTER_SHA, "coefficient_bound": COEFFICIENT_BOUND, "bound_scope": BOUND_SCOPE, "maximum_absolute_error": 0.1, "maximum_error_divided_by_bound": 0.9, "error_square_sum": 1.0, "reference_square_sum": 100.0, "zero_rows": 0, "model_quality_qualified": False}
    check_certificate(synthetic_cert, Path("/synthetic-unused-store"))
    checks.append("complete synthetic published coefficient certificate")
    bad_cert = copy.deepcopy(synthetic_cert); bad_cert["maximum_error_divided_by_bound"] = 2.0
    reject(lambda: check_certificate(bad_cert, Path("/synthetic-unused-store")), "failed coefficient bound rejected")
    bad_cert = copy.deepcopy(synthetic_cert); del bad_cert["maximum_absolute_error"]
    reject(lambda: check_certificate(bad_cert, Path("/synthetic-unused-store")), "missing coefficient audit rejected")
    bad_cert = copy.deepcopy(synthetic_cert); bad_cert["source_converter_sha256"] = "0" * 64
    reject(lambda: check_certificate(bad_cert, Path("/synthetic-unused-store")), "untrusted converter certificate rejected")
    bad_cert = copy.deepcopy(synthetic_cert); bad_cert["rows"] -= 1
    reject(lambda: check_certificate(bad_cert, Path("/synthetic-unused-store")), "incomplete coefficient cardinality rejected")
    with tempfile.TemporaryDirectory(prefix="splash-i8-lut-synthetic-") as temp:
        path = Path(temp) / "small.bin"
        raw = ids.tobytes() + lut.tobytes()
        with path.open("wb") as stream:
            stream.write(raw)
        path.chmod(0o444)
        fd, before = open_readonly(path, expected_size=len(raw))
        try:
            actual_ids = np.frombuffer(pread_exact(fd, ids.size, 0), dtype=np.uint8).reshape(ids.shape)
            actual_lut = np.frombuffer(pread_exact(fd, lut.size, ids.size), dtype=np.int8).reshape(lut.shape)
            check(np.array_equal(reconstruct_words(actual_ids, actual_lut, codes.shape), codes), "stored synthetic readback reconstruction")
            check(range_sha(fd, 0, len(raw)) == sha(raw) and snapshot(path) == before, "synthetic exact-file hash and snapshot")
            reject(lambda: pread_exact(fd, len(raw) + 1, 0), "truncated file rejected")
        finally:
            os.close(fd)
        path.chmod(0o644)
        reject(lambda: open_readonly(path), "writable payload rejected")
        alias = Path(temp) / "alias.bin"; alias.symlink_to(path)
        reject(lambda: open_readonly(alias), "symlink payload rejected")
        reject(lambda: relative_file(Path(temp), "../escape"), "escaping source path rejected")
        # Exercise the actual complete three-plane readback with independently
        # constructed synthetic LUT/codes, including every copied scale byte.
        original = layer_layout(0, packed=False, count=2, geometry=geometry)
        stored = layer_layout(0, packed=True, count=2, geometry=geometry)
        original_path, stored_path = Path(temp) / "original-i8.bin", Path(temp) / "stored-lut.bin"
        original_fd = os.open(original_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        stored_fd = os.open(stored_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.ftruncate(original_fd, original["bytes"])
            os.ftruncate(stored_fd, stored["bytes"])
            for name, (n, k) in geometry.items():
                qi = (np.arange(2 * n * k, dtype=np.uint16).reshape(2, n, k) % 16).astype(np.uint8)
                packed_ids = qi[..., 0::2] | (qi[..., 1::2] << np.uint8(4))
                expected_lut = np.broadcast_to((np.arange(16, dtype=np.int16) * 7 - 60).astype(np.int8), (2, n, k // GROUP, 16)).copy()
                expected_codes = np.take_along_axis(expected_lut, qi.reshape(2, n, k // GROUP, GROUP).astype(np.intp), axis=-1).reshape(2, n, k)
                expected_scales = (np.arange(2 * n, dtype=np.float32).reshape(2, n) + np.float32(1)) / np.float32(128)
                raw_items = {"ids": packed_ids.tobytes(), "lut": expected_lut.tobytes(), "codes": expected_codes.tobytes(), "scales": expected_scales.astype("<f4").tobytes()}
                for key in ("codes", "scales"):
                    descriptor = original["projections"][name][key]
                    pwrite_exact(original_fd, raw_items[key], descriptor["offset"])
                    descriptor["sha256"] = sha(raw_items[key])
                for key in ("ids", "lut", "scales"):
                    descriptor = stored["projections"][name][key]
                    pwrite_exact(stored_fd, raw_items[key], descriptor["offset"])
                    descriptor["sha256"] = sha(raw_items[key])
            observed = readback(stored_fd, original_fd, stored, original, 1, count=2, geometry=geometry)
            check(set(observed) == set(PROJECTIONS) and sum(row["coefficient_code_bytes"] for row in observed.values()) == 2304 and sum(row["scale_bytes"] for row in observed.values()) == 96, "complete synthetic three-projection disk readback and scale cardinalities")
            extent = stored["projections"]["gate_proj"]["lut"]["offset"]
            saved_byte = pread_exact(stored_fd, 1, extent)
            pwrite_exact(stored_fd, bytes([saved_byte[0] ^ 1]), extent)
            reject(lambda: readback(stored_fd, original_fd, stored, original, 1, count=2, geometry=geometry), "stored LUT code corruption rejected")
            pwrite_exact(stored_fd, saved_byte, extent)
            extent = stored["projections"]["down_proj"]["scales"]["offset"]
            saved_byte = pread_exact(stored_fd, 1, extent)
            pwrite_exact(stored_fd, bytes([saved_byte[0] ^ 1]), extent)
            reject(lambda: readback(stored_fd, original_fd, stored, original, 1, count=2, geometry=geometry), "stored scale byte corruption rejected")
        finally:
            os.close(original_fd); os.close(stored_fd)
        staged = Path(temp) / "stage"; staged.mkdir()
        (staged / "certificate.json").write_text('{"pass":true}\n')
        published = Path(temp) / "published"
        publish_exclusive(staged, published)
        check((published / "certificate.json").read_bytes() == b'{"pass":true}\n', "exclusive atomic synthetic publication")
        staged = Path(temp) / "stage2"; staged.mkdir()
        occupied = Path(temp) / "concurrent-empty-target"; occupied.mkdir()
        reject(lambda: publish_exclusive(staged, occupied), "concurrent empty destination cannot be replaced")
        check(staged.is_dir() and occupied.is_dir(), "exclusive publication preserves both directories on failure")
    full = layer_layout(0, packed=True)
    check(full["bytes"] == 1895301120 and sum(m["ids"]["length"] + m["lut"]["length"] for m in full["projections"].values()) == 2516582400 * 3 // 4, "production one-layer 48B versus64B accounting")
    return {"pass": True, "gpu_work": False, "installed_model_payload_bytes_read": 0, "checks": checks, "check_count": len(checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-self-test", action="store_true")
    commands = parser.add_subparsers(dest="command")
    for action in ("inspect", "pack"):
        command = commands.add_parser(action)
        command.add_argument("--source", "--package", dest="source", type=Path, required=True)
        command.add_argument("--store", type=Path, required=True)
        command.add_argument("--layer", type=int, required=True)
        command.add_argument("--certificate", type=Path)
        command.add_argument("--conversion-report", type=Path)
        if action == "pack":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--batch-experts", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_self_test:
        require(args.command is None, "self-test cannot be combined with a payload action")
        result = cpu_self_test()
    else:
        require(args.command is not None, "choose inspect, pack, or --cpu-self-test")
        prepared = prepare(args.source, args.store, args.layer, args.certificate, args.conversion_report)
        if args.command == "inspect":
            result = {"pass": True, "gpu_work": False, "model_payload_bytes_read": 0, "source": str(prepared["source_dir"]), "store": str(prepared["store_dir"]), "layer": args.layer, "original_i8_layer_bytes": prepared["layer"]["bytes"], "packed_layer_bytes": layer_layout(args.layer, packed=True)["bytes"], "rank_order": "expert_id", "experts": EXPERTS, "source_snapshot_certificate_checked": True}
        else:
            result = pack(prepared, args.output, batch_experts=args.batch_experts)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
