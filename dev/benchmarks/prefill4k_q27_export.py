#!/usr/bin/env python3
"""CPU-only, source-checked Qwen3.8-27B native affine-Q4 operand fixture.

ABI: little-endian <8s10I, 48 bytes. Magic Q27A0001, then N, K, group=64,
storageN=256, packed/scales/biases/F32/BF16 byte lengths, flags=0. Payloads
immediately follow in that order. Q4 payloads retain native tile256 storage;
F32 and BF16 coefficient payloads are tight, row-major [N,K].

F32 coefficients use separately rounded multiply and add. BF16 is rounded
once from those F32 coefficients. These are numerical alternates: native Q4
matmul applies affine parameters after group dots, so neither dense alternate
is a claim of native reduction, output, or whole-model parity.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = ROOT / "install/models/incoai/Qwen3.8-27B-Splash"
ALIGNMENT = 16384
HEADER = struct.Struct("<8s10I")
MAGIC = b"Q27A0001"
ROLES = ("gdn-input", "gdn-output", "mlp-gate", "mlp-up", "mlp-down")
PLANES = ("packed", "scales", "biases", "f32", "bf16")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def align(value):
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def snapshot(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def read_json(path):
    require(path.stat().st_size <= 4 << 20, f"Metadata too large: {path}")
    raw = path.read_bytes()
    return json.loads(raw), raw


def source_geometry():
    """Read this checkout's actual layout, and enforce the serialized contract."""
    paths = ("runtime/model/Qwen3_8.hpp", "runtime/model/Qwen3_8.cpp",
             "runtime/model/QwenTarget.hpp", "runtime/model/QwenTarget.cpp",
             "runtime/model/WeightStore.hpp", "runtime/model/WeightStore.cpp",
             "runtime/model/StateLayout.hpp",
             "runtime/metal/kernels/common/q4_mpp_tiles.h")
    source = {p: (ROOT / p).read_text() for p in paths}
    layout = source[paths[0]]
    fields = ("layers", "hiddenSize", "vocabularySize", "packedGdnWidth",
              "packedFullWidth", "convolutionDimension", "gdnKeyHeads",
              "gdnValueHeads", "gdnHeadDimension", "attentionWidth",
              "intermediateSize", "attentionQueryHeads", "attentionKvHeads",
              "attentionHeadDimension", "fullAttentionPeriod")
    geometry = {}
    for field in fields:
        found = re.findall(r"uint32_t\s+" + field + r"\s*=\s*(\d+)\s*;", layout)
        require(len(found) == 1, f"Layout declaration changed for {field}")
        geometry[field] = int(found[0])
    require('(layer + 1) % fullAttentionPeriod == 0' in layout,
            "Full-attention layer selection contract changed")
    require('layerMagic = "MDFL0006"' in layout, "Layer magic changed")
    taps = re.findall(r"kGdnConvolutionTaps\s*=\s*(\d+)\s*;", source[paths[6]])
    require(len(taps) == 1, "Convolution-tap declaration changed")
    geometry["convolutionTaps"] = int(taps[0])
    store_header = source[paths[4]]
    for token in ('kQ4GroupElements = 64', 'kQ4StorageN = 256',
                  'kWeightFileAlignment = 16 * 1024'):
        require(token in store_header, f"Storage contract changed: {token}")
    store = source[paths[5]]
    for token in ('uint64_t offset = 16;', 'uint64_t start = alignPacked(impl_->offset);',
                  'file.section(q4PackedBytes(outputSize, inputSize), label)',
                  'backend.view(packed, weightBytes, parameterBytes)',
                  'backend.view(packed, weightBytes + parameterBytes, parameterBytes)'):
        require(token in store, f"Native section contract changed: {token}")
    target_loader = source[paths[2]]
    ordered = ('layer.inputNorm = file.section', 'layer.mixer =',
               'layer.postAttentionNorm =', 'readFfn(file, layer);')
    positions = [target_loader.index(token) for token in ordered]
    require(positions == sorted(positions), "Target serialized order changed")
    mixer = source[paths[3]].split('QwenMixerWeights readQwenMixer(', 1)[1].split(
        'QwenTarget::QwenTarget(', 1)[0]
    ordered = ('"gdn-input"', '"gdn-convolution"', '"gdn-decay"',
               '"gdn-time-bias"', '"gdn-norm"', '"gdn-output"')
    require([mixer.index(token) for token in ordered] == sorted(
        mixer.index(token) for token in ordered), "GDN serialized order changed")
    ffn = source[paths[1]]
    require([ffn.index('"' + role + '"') for role in ROLES[2:]] == sorted(
        ffn.index('"' + role + '"') for role in ROLES[2:]), "MLP serialized order changed")
    tiles = source[paths[7]]
    require('ulong(tile) * quant_groups + quant_group' in tiles and
            'tile_offset + index[0]' in tiles,
            "Native tile256 shader parameter indexing changed")
    return geometry, {p: digest(text.encode()) for p, text in source.items()}


def layer_sections(g, layer):
    cursor = 16
    sections = []

    def section(role, length, n=None, k=None):
        nonlocal cursor
        cursor = align(cursor)
        record = {"role": role, "offset": cursor, "length": length}
        if n is not None:
            record.update(n=n, k=k, packed_bytes=n*k//2,
                          scale_bytes=n*k//32, bias_bytes=n*k//32)
        sections.append(record)
        cursor += length

    def projection(role, n, k):
        require(n > 0 and n % 256 == 0 and k > 0 and k % 64 == 0,
                "Projection is incompatible with tile256/G64")
        section(role, n*k*9//16, n, k)

    hidden = g["hiddenSize"]
    section("input-norm", hidden*2)
    full = (layer+1) % g["fullAttentionPeriod"] == 0
    if full:
        projection("attention-input", g["packedFullWidth"], hidden)
        section("query-norm", g["attentionHeadDimension"]*2)
        section("key-norm", g["attentionHeadDimension"]*2)
        projection("attention-output", hidden, g["attentionWidth"])
    else:
        projection("gdn-input", g["packedGdnWidth"], hidden)
        section("gdn-convolution", g["convolutionDimension"]*g["convolutionTaps"]*2)
        section("gdn-decay", g["gdnValueHeads"]*4)
        section("gdn-time-bias", g["gdnValueHeads"]*2)
        section("gdn-norm", g["gdnHeadDimension"]*2)
        projection("gdn-output", hidden, g["attentionWidth"])
    section("post-attention-norm", hidden*2)
    projection("mlp-gate", g["intermediateSize"], hidden)
    projection("mlp-up", g["intermediateSize"], hidden)
    projection("mlp-down", hidden, g["intermediateSize"])
    return sections, align(cursor)


def cache_sizing(g, artifacts):
    rows = {}
    for layer in range(g["layers"]):
        for s in layer_sections(g, layer)[0]:
            if "n" not in s:
                continue
            row = rows.setdefault(s["role"], {"n": s["n"], "k": s["k"], "count": 0})
            require((row["n"], row["k"]) == (s["n"], s["k"]), "Role geometry changed")
            row["count"] += 1
    for role, row in rows.items():
        count = row["n"]*row["k"]
        row.update(coefficient_elements_per_projection=count,
                   bf16_cache_bytes=row["count"]*align(count*2),
                   f32_cache_bytes=row["count"]*align(count*4),
                   source_q4_projection_bytes=row["count"]*count*9//16)
    body_bf16 = sum(row["bf16_cache_bytes"] for row in rows.values())
    body_f32 = sum(row["f32_cache_bytes"] for row in rows.values())
    vocab_elements = g["vocabularySize"]*g["hiddenSize"]
    return {"scope": "All actual target projection geometries; immutable coefficient planes only",
            "target_body_roles": rows,
            "target_body_bf16_cache_bytes": body_bf16,
            "target_body_f32_cache_bytes": body_f32,
            "logits_bf16_cache_bytes": align(vocab_elements*2),
            "logits_f32_cache_bytes": align(vocab_elements*4),
            "embedding_bf16_cache_bytes": align(vocab_elements*2),
            "embedding_f32_cache_bytes": align(vocab_elements*4),
            "target_body_and_logits_bf16_cache_bytes": body_bf16+align(vocab_elements*2),
            "target_body_and_logits_f32_cache_bytes": body_f32+align(vocab_elements*4),
            "manifest_declared_source_target_mapped_bytes": sum(
                a["size"] for p, a in artifacts.items() if p.startswith("target/")),
            "manifest_declared_all_source_weight_mapped_bytes": sum(
                a["size"] for p, a in artifacts.items() if p.endswith(".bin")),
            "source_retained": True,
            "excluded": "KV/state, activation/scratch, diagnostics, draft/vision converted caches and live admission"}


def inspect_package(package):
    package = package.resolve()
    manifest, raw = read_json(package / "manifest.json")
    require(manifest.get("model") == "Qwen3.8-27B" and manifest.get("schema_version") == 3,
            "Expected native Qwen3.8-27B schema3 package")
    fmt = manifest.get("format", {})
    expected = {"name": "splash-packed-q4", "q4_bits": 4, "q4_group_size": 64,
                "q4_storage_n": 256, "section_alignment_bytes": ALIGNMENT,
                "target_layer_magic": "MDFL0006"}
    require(all(fmt.get(k) == v for k, v in expected.items()), "Package format mismatch")
    g, sources = source_geometry()
    artifacts = {}
    for a in manifest["artifacts"]:
        p = a["path"]
        parts = PurePosixPath(p)
        require(not parts.is_absolute() and ".." not in parts.parts and
                p not in artifacts and type(a["size"]) is int and a["size"] > 0 and
                re.fullmatch(r"[a-f0-9]{64}", a["sha256"]), "Malformed artifact record")
        artifacts[p] = a
    config, config_raw = read_json(package / "tokenizer/config.json")
    require(digest(config_raw) == artifacts["tokenizer/config.json"]["sha256"],
            "Tokenizer config differs from declared source hash")
    text = config["text_config"]
    checks = {"layers": "num_hidden_layers", "hiddenSize": "hidden_size",
              "vocabularySize": "vocab_size", "intermediateSize": "intermediate_size",
              "attentionQueryHeads": "num_attention_heads", "attentionKvHeads": "num_key_value_heads",
              "attentionHeadDimension": "head_dim", "fullAttentionPeriod": "full_attention_interval",
              "gdnKeyHeads": "linear_num_key_heads", "gdnValueHeads": "linear_num_value_heads",
              "gdnHeadDimension": "linear_value_head_dim", "convolutionTaps": "linear_conv_kernel_dim"}
    require(all(g[k] == text[v] for k, v in checks.items()), "Source config/runtime layout mismatch")
    require(text["linear_key_head_dim"] == g["gdnHeadDimension"], "GDN key width mismatch")
    require(g["attentionWidth"] == g["attentionQueryHeads"]*g["attentionHeadDimension"],
            "Attention output geometry mismatch")
    require(g["convolutionDimension"] == (2*g["gdnKeyHeads"]+g["gdnValueHeads"])*g["gdnHeadDimension"],
            "GDN convolution geometry mismatch")
    require(config["quantization"] == {"group_size": 64, "bits": 4, "mode": "affine"},
            "Source quantization mismatch")
    expected_types = ["full_attention" if (i+1) % g["fullAttentionPeriod"] == 0 else
                      "linear_attention" for i in range(g["layers"])]
    require(text["layer_types"] == expected_types, "Source layer types mismatch")
    checked = []
    for layer in range(g["layers"]):
        relative = f"target/layer-{layer}.bin"
        path = package / relative
        _, expected_size = layer_sections(g, layer)
        require(path.stat().st_size == artifacts[relative]["size"] == expected_size,
                f"Layer size mismatch: {relative}")
        with path.open("rb") as stream:
            magic, index, kind = struct.unpack("<8sII", stream.read(16))
        require(magic == b"MDFL0006" and index == layer and
                kind == int((layer+1) % g["fullAttentionPeriod"] == 0),
                f"Layer header mismatch: {relative}")
        checked.append(relative)
    return {"package": str(package), "model": manifest["model"],
            "manifest_sha256": digest(raw),
            "artifact_set_sha256_declared": manifest["artifact_set_sha256"],
            "upstream": manifest["upstream"], "runtime_geometry": g,
            "source_contract_sha256": sources,
            "layer0_sections": layer_sections(g, 0)[0],
            "cache_memory_sizing": cache_sizing(g, artifacts),
            "headers_and_sizes_checked": checked,
            "full_model_payload_hashes_checked": False,
            "gpu_commands": 0}, artifacts


def reconstruct(packed, scales, biases, n, k):
    """Vectorized native tile256 unpack; caller supplies one output tile."""
    groups = k//64
    w = np.frombuffer(packed, np.uint8).reshape(n//256, groups, 256, 32)
    codes = np.empty((*w.shape[:-1], 64), np.uint8)
    codes[..., 0::2] = w & 15
    codes[..., 1::2] = w >> 4
    s = (np.frombuffer(scales, "<u2").astype(np.uint32) << 16).view(np.float32)
    b = (np.frombuffer(biases, "<u2").astype(np.uint32) << 16).view(np.float32)
    s, b = [a.reshape(n//256, groups, 256, 1) for a in (s, b)]
    require(np.isfinite(s).all() and np.isfinite(b).all(), "Nonfinite source affine parameters")
    product = np.multiply(codes.astype(np.float32), s, dtype=np.float32)
    values = np.add(product, b, dtype=np.float32).transpose(0, 2, 1, 3).reshape(n, k)
    require(np.isfinite(values).all(), "Nonfinite source coefficients")
    words = values.view(np.uint32)
    bf16 = ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype("<u2")
    require(np.all((bf16 & 0x7f80) != 0x7f80), "BF16 coefficient overflow")
    return np.ascontiguousarray(values, dtype="<f4"), np.ascontiguousarray(bf16)


# Independent scalar row/group/input traversal, memcpy bit casts and volatile
# F32 staging. Compiled CPU-only in a temporary directory; no Metal or MLX.
SCALAR_C = r'''
#include <stdint.h>
#include <stddef.h>
#include <string.h>
uint64_t q27_verify(uint32_t n, uint32_t k, const uint8_t *packed,
  const uint8_t *scales, const uint8_t *biases, const uint8_t *coeff,
  const uint8_t *bf16, uint64_t *bad) {
  uint64_t checked=0; uint32_t groups=k/64;
  for(uint32_t row=0;row<n;++row) for(uint32_t group=0;group<groups;++group) {
    uint64_t param=((uint64_t)(row/256)*groups+group)*256+row%256;
    uint16_t sw,bw; memcpy(&sw,scales+param*2,2); memcpy(&bw,biases+param*2,2);
    uint32_t sbits=(uint32_t)sw<<16, bbits=(uint32_t)bw<<16;
    float s,b; memcpy(&s,&sbits,4); memcpy(&b,&bbits,4);
    for(uint32_t input=0;input<64;++input) {
      uint8_t code=(packed[param*32+input/2]>>((input%2)*4))&15;
      volatile float product=(float)code*s;
      volatile float value=product+b;
      float copy=value; uint32_t expected,actual;
      memcpy(&expected,&copy,4);
      uint64_t index=(uint64_t)row*k+group*64+input;
      memcpy(&actual,coeff+index*4,4);
      if(actual!=expected) { *bad=index; return 1; }
      uint16_t expectedbf=(uint16_t)((expected+0x7fff+((expected>>16)&1))>>16),actualbf;
      memcpy(&actualbf,bf16+index*2,2);
      if(actualbf!=expectedbf) { *bad=index; return 2; }
      ++checked;
    }
  }
  *bad=checked; return 0;
}
'''


class ScalarOracle:
    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="prefill4k_q27_export_scalar_")
        tmp = Path(self.temporary.name)
        source, library = tmp/"scalar.c", tmp/"scalar.so"
        source.write_text(SCALAR_C)
        command = ["/usr/bin/cc", "-std=c11", "-O2", "-fno-fast-math", "-ffp-contract=off",
                   "-shared", "-fPIC", str(source), "-o", str(library)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        self.library = ctypes.CDLL(str(library))
        self.verify = self.library.q27_verify
        self.verify.argtypes = [ctypes.c_uint32, ctypes.c_uint32] + [ctypes.c_void_p]*5 + [ctypes.POINTER(ctypes.c_uint64)]
        self.verify.restype = ctypes.c_uint64
        return self

    def check(self, packed, scales, biases, f32, bf16, n, k):
        parts = [np.frombuffer(packed, np.uint8), np.frombuffer(scales, np.uint8),
                 np.frombuffer(biases, np.uint8), f32, bf16]
        bad = ctypes.c_uint64()
        result = self.verify(n, k, *(ctypes.c_void_p(a.ctypes.data) for a in parts), ctypes.byref(bad))
        require(result == 0, f"Independent scalar {'F32' if result == 1 else 'BF16'} mismatch at word {bad.value}")
        require(bad.value == n*k, "Incomplete scalar certificate")
        return bad.value

    def __exit__(self, *_):
        self.temporary.cleanup()


def parse_header(raw, total_bytes):
    require(len(raw) == HEADER.size, "Fixture header truncated")
    magic, n, k, group, storage, wb, sb, bb, fb, hb, flags = HEADER.unpack(raw)
    require(magic == MAGIC and n > 0 and n % 256 == 0 and k > 0 and k % 64 == 0 and
            group == 64 and storage == 256 and flags == 0, "Invalid fixture ABI/geometry")
    require((wb, sb, bb, fb, hb) == (n*k//2, n*k//32, n*k//32, n*k*4, n*k*2),
            "Fixture payload lengths mismatch")
    require(total_bytes == HEADER.size+wb+sb+bb+fb+hb, "Fixture truncated or trailing bytes")
    return n, k


def self_test():
    rng = np.random.default_rng(270038)
    n, k = 512, 128
    packed = rng.integers(0, 256, n*k//2, np.uint8).tobytes()
    # Signed scales, cancellation, zero, subnormals and BF16 boundaries.
    palette = np.array([0, 0x8000, 1, 0x8001, 0x3f00, 0xbf00, 0x3f81, 0xbf81,
                        0x3580, 0xb580, 0x3b81, 0xbb81], dtype="<u2")
    scales = rng.choice(palette, n*k//64).astype("<u2").tobytes()
    biases = rng.choice(palette, n*k//64).astype("<u2").tobytes()
    f32, bf16 = reconstruct(packed, scales, biases, n, k)
    with ScalarOracle() as oracle:
        words = oracle.check(packed, scales, biases, f32, bf16, n, k)
        corrupted = f32.copy()
        corrupted.view(np.uint32).flat[-1] ^= np.uint32(1)
        try:
            oracle.check(packed, scales, biases, corrupted, bf16, n, k)
        except ValueError:
            pass
        else:
            raise ValueError("Scalar oracle missed deliberate F32 corruption")
        corrupted_bf16 = bf16.copy()
        corrupted_bf16.flat[0] ^= np.uint16(1)
        try:
            oracle.check(packed, scales, biases, f32, corrupted_bf16, n, k)
        except ValueError:
            pass
        else:
            raise ValueError("Scalar oracle missed deliberate BF16 corruption")
    header = HEADER.pack(MAGIC, n, k, 64, 256, n*k//2, n*k//32, n*k//32, n*k*4, n*k*2, 0)
    size = HEADER.size+n*k*105//16
    require(parse_header(header, size) == (n, k), "Fixture parser self-test failed")
    rejected = 0
    cases = [(header, size-1), (header, size+1), (header[:-1], size)]
    for index, value in ((0, b"BADMAGIC"), (1, 255), (2, 127), (3, 32), (4, 128), (5, 1), (10, 1)):
        parts = list(HEADER.unpack(header))
        parts[index] = value
        cases.append((HEADER.pack(*parts), size))
    for raw, total in cases:
        try:
            parse_header(raw, total)
        except ValueError:
            rejected += 1
        else:
            raise ValueError("Fixture parser accepted malformed fixture")
    return {"passed": True, "scalar_f32_and_bf16_words_checked": words,
            "deliberate_coefficient_corruptions_rejected": 2,
            "malformed_fixture_cases_rejected": rejected, "gpu_commands": 0}


def export_fixture(args, report, artifacts):
    output = args.output.resolve()
    require(output.name.startswith("prefill4k_q27_export"), "Use a newly named prefill4k_q27_export* fixture")
    require(output.suffix == ".bin", "Fixture destination must end in .bin")
    require(not output.exists() and not output.with_suffix(".json").exists(), "Refusing existing fixture destination")
    package = Path(report["package"])
    require(package not in output.parents, "Fixture cannot be written into source package")
    section = next(s for s in report["layer0_sections"] if s["role"] == args.role)
    full_n, k = section["n"], section["k"]
    begin = args.output_begin
    n = full_n if args.output_count is None else args.output_count
    require(begin >= 0 and begin % 256 == 0 and n > 0 and n % 256 == 0 and
            begin+n <= full_n, "Output subset must cover complete native tile256 slabs")
    source = package / "target/layer-0.bin"
    before = snapshot(source)
    source_sha = sha_file(source)
    require(source_sha == artifacts["target/layer-0.bin"]["sha256"], "Original source layer SHA256 mismatch")
    lengths = (n*k//2, n*k//32, n*k//32, n*k*4, n*k*2)
    header = HEADER.pack(MAGIC, n, k, 64, 256, *lengths, 0)
    total = HEADER.size+sum(lengths)
    parse_header(header, total)
    fixture_offsets, cursor = {}, HEADER.size
    for plane, length in zip(PLANES, lengths):
        fixture_offsets[plane] = {"offset": cursor, "length": length}
        cursor += length
    source_offsets = {"packed": section["offset"]+begin*k//2,
                      "scales": section["offset"]+full_n*k//2+begin*k//32,
                      "biases": section["offset"]+full_n*k//2+full_n*k//32+begin*k//32}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name+".temporary")
    require(not temporary.exists(), "Refusing existing temporary fixture")
    hashes = {p: hashlib.sha256() for p in PLANES}
    checked = 0
    try:
        with source.open("rb") as src, temporary.open("xb+") as dst, ScalarOracle() as oracle:
            dst.write(header)
            dst.truncate(total)
            for tile in range(n//256):
                raw = {}
                for plane, bytes_per_tile in (("packed", 256*k//2), ("scales", 256*k//32), ("biases", 256*k//32)):
                    src.seek(source_offsets[plane]+tile*bytes_per_tile)
                    raw[plane] = src.read(bytes_per_tile)
                    require(len(raw[plane]) == bytes_per_tile, "Source tile truncated")
                f32, bf16 = reconstruct(raw["packed"], raw["scales"], raw["biases"], 256, k)
                checked += oracle.check(raw["packed"], raw["scales"], raw["biases"], f32, bf16, 256, k)
                raw.update(f32=f32.tobytes(), bf16=bf16.tobytes())
                for plane in PLANES:
                    dst.seek(fixture_offsets[plane]["offset"]+tile*len(raw[plane]))
                    dst.write(raw[plane])
                    hashes[plane].update(raw[plane])
            dst.flush()
            os.fsync(dst.fileno())
        require(snapshot(source) == before, "Source changed during export")
        require(sha_file(package / "manifest.json") == report["manifest_sha256"],
                "Source manifest changed during export")
        require(checked == n*k, "Incomplete all-word scalar certificate")
        require(temporary.stat().st_size == total, "Written fixture size mismatch")
        payload_sha = {}
        with temporary.open("rb") as stream:
            parse_header(stream.read(HEADER.size), total)
            for plane in PLANES:
                h = hashlib.sha256()
                remaining = fixture_offsets[plane]["length"]
                while remaining:
                    block = stream.read(min(8 << 20, remaining))
                    require(bool(block), "Written fixture payload truncated")
                    h.update(block)
                    remaining -= len(block)
                payload_sha[plane] = h.hexdigest()
                require(payload_sha[plane] == hashes[plane].hexdigest(), "Written payload differs from certified words")
        for plane in PLANES:
            fixture_offsets[plane]["sha256"] = payload_sha[plane]
        fixture_report = dict(report, mode="cpu_source_operand_export", fixture={
            "path": str(output), "magic": MAGIC.decode(), "header_bytes": HEADER.size,
            "header_struct": "<8s10I", "bytes": total, "sha256": sha_file(temporary),
            "role": args.role, "layer": 0, "source_n": full_n, "n": n, "k": k,
            "output_begin": begin, "output_count": n, "full_projection": begin == 0 and n == full_n,
            "group_size": 64, "storage_n": 256, "flags": 0, "planes": fixture_offsets,
            "source_artifact": dict(artifacts["target/layer-0.bin"], verified_sha256=source_sha),
            "source_projection_section": section,
            "source_planes": {p: {"offset": source_offsets[p], "length": lengths[i],
                                    "sha256": payload_sha[p]} for i, p in enumerate(PLANES[:3])},
            "scalar_certificate": {"f32_words_checked": checked, "bf16_words_checked": checked,
                                   "exact_word_mismatches": 0, "oracle_source_sha256": digest(SCALAR_C.encode()),
                                   "compiler_flags": ["-O2", "-fno-fast-math", "-ffp-contract=off"],
                                   "staging": "scalar integer nibble unpack; volatile F32 multiply then F32 add; BF16 RNE once"},
            "numerical_scope": "Dense coefficients are numerical alternates; native Q4 group epilogue/reduction and whole-model parity require separate qualification",
            "native_storage": "param=((row//256)*(K//64)+group)*256+row%256; byte=param*32+input//2; low nibble first",
            "coefficient_storage": "tight row-major [N,K]"})
        os.link(temporary, output)
        temporary.unlink()
        output.with_suffix(".json").write_text(json.dumps(fixture_report, indent=2, sort_keys=True)+"\n")
        return fixture_report
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--role", choices=ROLES, default="gdn-output")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-begin", type=int, default=0)
    parser.add_argument("--output-count", type=int)
    parser.add_argument("--check", action="store_true", help="Cheap metadata/header/parser/scalar self-test; no large model payload hashes or export")
    parser.add_argument("--report", type=Path, help="New prefill4k_q27_export*.json inspection report")
    args = parser.parse_args()
    require(args.check != bool(args.output), "Choose --check or --output")
    report, artifacts = inspect_package(args.package)
    if args.check:
        report["self_test"] = self_test()
        report["mode"] = "cpu_metadata_and_synthetic_check"
    else:
        report = export_fixture(args, report, artifacts)
        report["mode"] = "cpu_source_operand_export"
    if args.report:
        require(args.report.name.startswith("prefill4k_q27_export") and args.report.suffix == ".json" and
                not args.report.exists(), "Use a new prefill4k_q27_export*.json report")
        require(Path(report["package"]) not in args.report.resolve().parents,
                "Inspection report cannot be written into source package")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
