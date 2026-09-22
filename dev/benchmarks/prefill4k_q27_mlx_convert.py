#!/usr/bin/env python3
"""CPU-only native Q27 integer repack; full copying requires --run-root-memory.

The output is a declared native-coefficient model bridge, not recovery of all
original upstream parameter words. Quantized U32 codes/BF16 scales and biases
are lossless. Operative norm gains are retained. Native pre-exponentiated GDN
decay is preserved in a required sidecar; stock A_log slots are unused zeros.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
import prefill4k_q27_mlx_layout as layout

ROOT = layout.ROOT
DTYPE_BYTES = {"U32": 4, "F32": 4, "BF16": 2}
SOURCE_PAYLOAD_READ_BYTES = 0


def count_bytes(entry):
    return int(np.prod(entry["shape"], dtype=np.int64)) * DTYPE_BYTES[entry["dtype"]]


def read_exact(stream, offset, length):
    global SOURCE_PAYLOAD_READ_BYTES
    stream.seek(offset)
    data = stream.read(length)
    if len(data) != length:
        raise ValueError("native source truncated during bounded read")
    SOURCE_PAYLOAD_READ_BYTES += length
    return data


def payload_chunks(entry, package, chunk_bytes=16 << 20, native_decay=False):
    """Peak binary/array payload bounded by ~16 MiB, independent of model size."""
    if entry["native_role"] == "gdn-decay" and not native_decay:
        yield bytes(count_bytes(entry))
        return
    source = package / entry["source_file"]
    with source.open("rb") as stream:
        if "source_storage" not in entry:
            remaining, cursor = count_bytes(entry), entry["offset"]
            while remaining:
                size = min(remaining, chunk_bytes // 5)
                yield read_exact(stream, cursor, size)
                remaining -= size
                cursor += size
            return
        k = entry["source_k"]
        rows = entry["source_row_count"]
        begin = entry["source_row_begin"]
        width = entry["shape"][1]
        word_bytes = DTYPE_BYTES[entry["dtype"]]
        if entry["source_storage"] == "row-major":
            row_bytes = width * word_bytes
            per_chunk = max(1, chunk_bytes // 5 // row_bytes)
            for row in range(0, rows, per_chunk):
                actual = min(per_chunk, rows - row)
                yield read_exact(stream, entry["source_plane_offset"] + (begin + row) * row_bytes,
                                 actual * row_bytes)
            return
        groups = k // 64
        weight = entry["dtype"] == "U32"
        tile_bytes = 256 * width * word_bytes
        per_chunk = max(1, chunk_bytes // 5 // tile_bytes)
        first, end = begin // 256, (begin + rows + 255) // 256
        for tile in range(first, end, per_chunk):
            ntile = min(per_chunk, end - tile)
            raw = read_exact(stream, entry["source_plane_offset"] + tile * tile_bytes, ntile * tile_bytes)
            if weight:
                native = np.frombuffer(raw, "<u4").reshape(ntile, groups, 256, 8)
                logical = native.transpose(0, 2, 1, 3).reshape(ntile * 256, width)
                inverse = logical.reshape(ntile, 256, groups, 8).transpose(0, 2, 1, 3).tobytes()
            else:
                native = np.frombuffer(raw, "<u2").reshape(ntile, groups, 256)
                logical = native.transpose(0, 2, 1).reshape(ntile * 256, width)
                inverse = logical.reshape(ntile, 256, groups).transpose(0, 2, 1).tobytes()
            if inverse != raw:
                raise ValueError("integer inverse-permutation certificate failed")
            del inverse
            local_begin = max(begin, tile * 256) - tile * 256
            local_end = min(begin + rows, (tile + ntile) * 256) - tile * 256
            yield logical[local_begin:local_end].tobytes()


def write_safetensors(path, entries, package, native_decay=False):
    header = {"__metadata__": {"format": "mlx", "splash_native_coefficient_bridge": "v1"}}
    cursor = 0
    for entry in entries:
        size = count_bytes(entry)
        header[entry["destination_key"]] = {"dtype": entry["dtype"], "shape": entry["shape"],
                                            "data_offsets": [cursor, cursor + size]}
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    header_bytes = struct.pack("<Q", len(encoded)) + encoded
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    witnesses = []
    with path.open("x+b") as output:
        output.write(header_bytes)
        digest.update(header_bytes)
        for entry in entries:
            tensor_digest, written = hashlib.sha256(), 0
            for chunk in payload_chunks(entry, package, native_decay=native_decay):
                position = output.tell()
                output.write(chunk)
                output.flush()
                output.seek(position)
                if output.read(len(chunk)) != chunk:
                    raise ValueError("output chunk readback certificate failed")
                output.seek(position + len(chunk))
                tensor_digest.update(chunk)
                digest.update(chunk)
                written += len(chunk)
            if written != count_bytes(entry):
                raise ValueError("destination tensor extent differs from planned shape")
            witnesses.append({"key": entry["destination_key"], "bytes": written,
                              "sha256": tensor_digest.hexdigest(), "readback_exact": True,
                              "coefficients_requantized": False,
                              "source_semantics": "native_decay_exact" if native_decay else
                              "unused_A_log_placeholder" if entry["native_role"] == "gdn-decay" else
                              "source_native_operative_or_integer_planes"})
        if output.tell() != len(header_bytes) + cursor:
            raise ValueError("safetensors payload length mismatch")
    return {"path": str(path), "bytes": len(header_bytes) + cursor,
            "sha256": digest.hexdigest(), "tensor_witnesses": witnesses}


def source_snapshot(package, plan):
    names = [entry["path"] for entry in plan["source_files"]]
    names += ["manifest.json"]
    names += [str(path.relative_to(package)) for path in (package / "tokenizer").iterdir() if path.is_file()]
    return {name: (package / name).stat() for name in names}


def unchanged(package, before):
    for relative, old in before.items():
        now = (package / relative).stat()
        if (old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns) != (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns):
            raise ValueError("native source changed during adapter preparation")


def certify_samples(package, plan):
    selected = [entry for entry in plan["entries"] if
                (entry["source_file"] in ("target/layer-0.bin", "target/layer-3.bin", "target/head.bin", "target/embedding.bin"))
                and "source_storage" in entry]
    checked = []
    with tempfile.TemporaryDirectory(prefix="prefill4k_q27_repack_sample_") as temporary:
        for index, entry in enumerate(selected):
            sample = dict(entry)
            rows = min(4, entry["source_row_count"])
            sample["source_row_begin"] += entry["source_row_count"] - rows
            sample["source_row_count"] = rows
            sample["shape"] = [rows, entry["shape"][1]]
            witness = write_safetensors(Path(temporary) / f"sample-{index}.safetensors", [sample], package)
            # Independent scalar source coordinates versus the emitted row-major words.
            path = Path(witness["path"])
            with path.open("rb") as output:
                size = struct.unpack("<Q", output.read(8))[0]
                output.seek(8 + size)
                data = output.read()
            with (package / sample["source_file"]).open("rb") as source:
                for row in range(rows):
                    for column in range(sample["shape"][1]):
                        native_row = sample["source_row_begin"] + row
                        if sample["source_storage"] == "row-major":
                            address = native_row * sample["shape"][1] + column
                        elif sample["dtype"] == "U32":
                            address = layout.native_u32_word_index(native_row, column, sample["source_k"])
                        else:
                            address = layout.parameter_index(native_row, column, sample["source_k"])
                        unit = DTYPE_BYTES[sample["dtype"]]
                        expected = read_exact(source, sample["source_plane_offset"] + address * unit, unit)
                        actual = data[(row * sample["shape"][1] + column) * unit:
                                      (row * sample["shape"][1] + column + 1) * unit]
                        if expected != actual:
                            raise ValueError("independent scalar row-major certificate failed")
            checked.append({"key": entry["destination_key"], "rows_checked": rows,
                            "words_checked": rows * sample["shape"][1], "readback_exact": True,
                            "source_storage": sample["source_storage"]})
    return checked


def certify_native_parameters(package, plan):
    witnesses = []
    for entry in plan["entries"]:
        if "source_storage" in entry:
            continue
        if entry["native_role"] != "gdn-decay" and entry["source_file"] not in (
                "target/layer-0.bin", "target/layer-3.bin", "target/head.bin"):
            continue
        with (package / entry["source_file"]).open("rb") as source:
            if entry["native_role"] == "gdn-decay":
                raw = read_exact(source, entry["offset"], entry["bytes"])
                values = np.frombuffer(raw, "<f4")
                if not np.isfinite(values).all() or not (values < 0).all():
                    raise ValueError("native pre-exponentiated GDN decay must be finite negative F32")
            else:
                unit = DTYPE_BYTES[entry["dtype"]]
                length = count_bytes(entry)
                offsets = sorted({0, (length // unit // 2) * unit, length - unit})
                raw = b"".join(read_exact(source, entry["offset"] + offset, unit) for offset in offsets)
        witnesses.append({"key": entry["destination_key"], "native_role": entry["native_role"],
                          "source_dtype": entry["dtype"], "native_bytes_sampled": len(raw),
                          "raw_sample_sha256": hashlib.sha256(raw).hexdigest(),
                          "conversion": "native_decay_sidecar_exact_and_unused_A_log_placeholder" if
                          entry["native_role"] == "gdn-decay" else "native_operative_bytes_unchanged",
                          "stock_A_log_or_centered_norm_recovered": False})
    return witnesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=layout.PACKAGE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/prefill4k-q27-native-mlx-v1")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-root-memory", action="store_true", help="Root-only full target copy; no GPU; requires serialized memory/I/O slot")
    args = parser.parse_args()
    package, destination, report_path = (p.expanduser().resolve() for p in (args.package, args.output_dir, args.report))
    if not destination.is_relative_to(ROOT / "build") or destination == ROOT / "build":
        raise ValueError("private adapter output must be a new directory below workspace build")
    if report_path.exists() or report_path.is_relative_to(package):
        raise ValueError("choose a fresh report outside native source")
    plan = layout.make_plan(package)
    before = source_snapshot(package, plan)
    summary = {key: value for key, value in plan.items() if key not in ("entries", "native_sections")}
    summary.update({"schema": "splash-private-q27-native-coefficient-mlx-adapter-v1",
                    "quantized_code_scale_bias_bit_exact": True, "original_upstream_raw_parameters_recovered": False,
                    "native_operative_gains_preserved": True, "gdn_native_decay_bridge_mandatory": True,
                    "A_log_main_model_policy": "unused zero placeholder; never used by mandatory native-decay bridge",
                    "gdn_compute_policy": "mx.exp(native_a_scale.float32 * nn.softplus(a + dt_bias)); stock gated_delta state-update kernel unchanged",
                    "chunk_payload_budget_bytes": 16 << 20, "full_target_payload_bytes": sum(count_bytes(e) for e in plan["entries"]),
                    "model_payload_copy_byte_scope": "full destination model; temporary bounded sample copies reported separately",
                    "gpu_work": False, "full_conversion_executed": False,
                    "converter_sha256": layout.sha(Path(__file__)), "output_directory": str(destination)})
    if not args.run_root_memory:
        summary["sample_cpu_certificates"] = certify_samples(package, plan)
        summary["native_parameter_witnesses"] = certify_native_parameters(package, plan)
        summary["temporary_sample_tensor_bytes_written"] = sum(
            row["words_checked"] * (4 if row["key"].endswith(".weight") else 2)
            for row in summary["sample_cpu_certificates"])
    else:
        if destination.exists():
            raise FileExistsError("private full adapter directory must be fresh")
        destination.mkdir(parents=True)
        by_file = defaultdict(list)
        for entry in plan["entries"]:
            by_file[entry["source_file"]].append(entry)
        files, weight_map = [], {}
        for index, (source_file, entries) in enumerate(by_file.items(), 1):
            name = f"model-{index:05d}-of-{len(by_file):05d}.safetensors"
            files.append(write_safetensors(destination / name, entries, package))
            weight_map.update({entry["destination_key"]: name for entry in entries})
            unchanged(package, before)
            print(json.dumps({"converted_source_file": source_file, "shard": name}), flush=True)
        decay = [dict(entry, destination_key=entry["destination_key"].replace(".A_log", ".native_a_scale"))
                 for entry in plan["entries"] if entry["native_role"] == "gdn-decay"]
        files.append(write_safetensors(destination / "native-bridge/gdn-decay.safetensors", decay, package, native_decay=True))
        config = json.loads((package / "tokenizer/config.json").read_bytes())
        original_model_type = config["model_type"]
        config["model_type"] = "splash_native_qwen3_5_bridge"
        config["splash_native_coefficient_bridge"] = {"required": True, "version": 1, "original_model_type": original_model_type,
                                                      "native_decay": "native-bridge/gdn-decay.safetensors",
                                                      "original_raw_parameters_recovered": False, "A_log_placeholders_unused": True}
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (destination / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": summary["full_target_payload_bytes"]}, "weight_map": weight_map}, indent=2) + "\n")
        for source in (package / "tokenizer").iterdir():
            if source.is_file() and source.name != "config.json":
                shutil.copyfile(source, destination / source.name)
        placeholders = sum(count_bytes(entry) for entry in plan["entries"] if entry["native_role"] == "gdn-decay")
        native_sidecar = sum(count_bytes(entry) for entry in decay)
        summary.update(full_conversion_executed=True, output_files=files,
                       model_payload_copy_bytes=summary["full_target_payload_bytes"] - placeholders + native_sidecar,
                       model_payload_placeholder_bytes=placeholders,
                       source_payload_read_bytes=SOURCE_PAYLOAD_READ_BYTES,
                       safetensors_total_written_bytes=sum(record["bytes"] for record in files))
        (destination / "native-coefficient-manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    unchanged(package, before)
    summary["bounded_source_payload_read_bytes"] = SOURCE_PAYLOAD_READ_BYTES
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"valid": True, "full_conversion_executed": summary["full_conversion_executed"], "gpu_work": False,
                      "report": str(report_path), "sample_tensor_count": len(summary.get("sample_cpu_certificates", []))}))


if __name__ == "__main__":
    main()
