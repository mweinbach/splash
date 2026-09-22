#!/usr/bin/env python3
"""Pure-stdlib Q27 native-to-MLX layout plan; no conversion or GPU work.

Plans loaded mlx_lm.models.qwen3_5.Model keys. Quantized codes and BF16 affine
parameters are losslessly permutable. Native decay stores an operative F32
a_scale, not an A_log; recovering upstream A_log is deliberately NOT certified.
Native norms are operative gains and must not receive another +1 shift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "install/models/incoai/Qwen3.8-27B-Splash"
SITE = Path("/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages")
ALIGN = 16384


def require(value, message):
    if not value:
        raise ValueError(message)


def align(value):
    return (value + ALIGN - 1) & -ALIGN


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry():
    raw = (ROOT / "runtime/model/Qwen3_8.hpp").read_text()
    fields = ("layers", "hiddenSize", "vocabularySize", "packedGdnWidth",
              "packedFullWidth", "convolutionDimension", "gdnKeyHeads",
              "gdnValueHeads", "gdnHeadDimension", "attentionWidth",
              "intermediateSize", "attentionQueryHeads", "attentionKvHeads",
              "attentionHeadDimension", "fullAttentionPeriod")
    result = {}
    for field in fields:
        found = re.findall(r"uint32_t\s+" + field + r"\s*=\s*(\d+)\s*;", raw)
        require(len(found) == 1, f"Runtime declaration changed: {field}")
        result[field] = int(found[0])
    taps = re.findall(r"kGdnConvolutionTaps\s*=\s*(\d+)\s*;",
                      (ROOT / "runtime/model/StateLayout.hpp").read_text())
    require(len(taps) == 1, "Convolution layout changed")
    result["convolutionTaps"] = int(taps[0])
    require(result["layers"] == 64 and result["fullAttentionPeriod"] == 4 and
            result["hiddenSize"] == 5120 and result["gdnValueHeads"] == 48 and
            result["gdnKeyHeads"] == 16 and result["gdnHeadDimension"] == 128 and
            result["packedGdnWidth"] == 16640 and result["packedFullWidth"] == 14336,
            "This private plan supports the current native Q27 geometry only")
    require(result["convolutionDimension"] ==
            (2*result["gdnKeyHeads"]+result["gdnValueHeads"])*result["gdnHeadDimension"],
            "Native convolution width is inconsistent")
    require(result["attentionWidth"] == result["attentionQueryHeads"]*result["attentionHeadDimension"] and
            result["packedFullWidth"] ==
            (2*result["attentionQueryHeads"]+2*result["attentionKvHeads"])*result["attentionHeadDimension"],
            "Native full-attention packed/output widths are inconsistent")
    return result


def parameter_index(row, group, k):
    return ((row // 256) * (k // 64) + group) * 256 + row % 256


def native_u32_word_index(row, logical_word, k):
    return parameter_index(row, logical_word // 8, k) * 8 + logical_word % 8


def make_plan(package=PACKAGE):
    package = Path(package).resolve()
    manifest_path = package / "manifest.json"
    require(manifest_path.stat().st_size <= 4 << 20, "Manifest too large")
    manifest = json.loads(manifest_path.read_bytes())
    config_path = package / "tokenizer/config.json"
    config = json.loads(config_path.read_bytes())
    require(manifest["schema_version"] == 3 and manifest["model"] == "Qwen3.8-27B",
            "Expected native Q27 package")
    fmt = manifest["format"]
    require(fmt["q4_bits"] == 4 and fmt["q4_group_size"] == 64 and
            fmt["q4_storage_n"] == 256 and fmt["section_alignment_bytes"] == ALIGN,
            "Native quantization/alignment changed")
    artifacts = {entry["path"]: entry for entry in manifest["artifacts"]}
    require(sha(config_path) == artifacts["tokenizer/config.json"]["sha256"],
            "Source config checksum mismatch")
    g = geometry()
    text = config["text_config"]
    for key, field in (("num_hidden_layers", "layers"), ("hidden_size", "hiddenSize"),
                       ("intermediate_size", "intermediateSize"), ("vocab_size", "vocabularySize"),
                       ("num_attention_heads", "attentionQueryHeads"), ("num_key_value_heads", "attentionKvHeads"),
                       ("head_dim", "attentionHeadDimension"), ("full_attention_interval", "fullAttentionPeriod"),
                       ("linear_num_key_heads", "gdnKeyHeads"), ("linear_num_value_heads", "gdnValueHeads"),
                       ("linear_key_head_dim", "gdnHeadDimension"), ("linear_value_head_dim", "gdnHeadDimension"),
                       ("linear_conv_kernel_dim", "convolutionTaps")):
        require(text[key] == g[field], f"Config/runtime mismatch: {key}")
    require(not text["tie_word_embeddings"] and not text["attention_bias"], "Unsupported tied/bias layout")
    entries, sections, files = [], [], []
    current_file, cursor = None, 16

    def begin_file(relative, magic, first, second):
        nonlocal current_file, cursor
        current_file, cursor = relative, 16
        path = package / relative
        with path.open("rb") as stream:
            header = stream.read(16)
        require(header == struct.pack("<8sII", magic, first, second), f"Native header mismatch: {relative}")

    def section(role, size, n=None, k=None):
        nonlocal cursor
        cursor = align(cursor)
        record = {"source_file": current_file, "native_role": role,
                  "offset": cursor, "bytes": size}
        if n is not None:
            record.update(n=n, k=k)
        sections.append(record)
        cursor += size
        return record

    def finish_file():
        path = package / current_file
        size = align(cursor)
        require(path.stat().st_size == artifacts[current_file]["size"] == size,
                f"Native schema/extent mismatch: {current_file}")
        files.append(dict(artifacts[current_file], header_and_size_checked=True,
                          full_payload_sha256_recomputed=False))

    def plain(key, role, shape, dtype="BF16", conversion="copy_bytes_and_reshape", note=None):
        count = 1
        for dim in shape:
            count *= dim
        s = section(role, count * (4 if dtype == "F32" else 2))
        entry = dict(s, destination_key=key, shape=shape, dtype=dtype, conversion=conversion)
        if note:
            entry["note"] = note
        entries.append(entry)

    def emit_q4(prefix, s, begin, count, storage="tile256"):
        n, k = s["n"], s["k"]
        require(0 <= begin and begin + count <= n and k % 64 == 0, "Invalid logical row slice")
        lengths = (n*k//2, n*k//32, n*k//32)
        offsets = s.get("plane_offsets", (s["offset"], s["offset"]+lengths[0], s["offset"]+lengths[0]+lengths[1]))
        for suffix, dtype, shape, offset, length in zip(
                ("weight", "scales", "biases"), ("U32", "BF16", "BF16"),
                ([count, k//8], [count, k//64], [count, k//64]), offsets, lengths):
            entries.append({"destination_key": prefix+"."+suffix, "dtype": dtype, "shape": shape,
                            "source_file": s["source_file"], "native_role": s["native_role"],
                            "source_plane_offset": offset, "source_plane_bytes": length,
                            "source_storage": storage, "source_n": n, "source_k": k,
                            "source_row_begin": begin, "source_row_count": count,
                            "conversion": "integer_word_permutation_only" if storage == "tile256" else "copy_bytes",
                            "coefficients_requantized": False})

    def q4(role, n, k, prefix=None, slices=None):
        s = section(role, n*k*9//16, n, k)
        if prefix:
            emit_q4(prefix, s, 0, n)
        for destination, begin, count in slices or []:
            emit_q4(destination, s, begin, count)

    h, d, value_heads = g["hiddenSize"], g["gdnHeadDimension"], g["gdnValueHeads"]
    key_width = g["gdnKeyHeads"]*d
    value_width = value_heads*d
    conv_width = 2*key_width+value_width
    for layer in range(g["layers"]):
        full = (layer+1) % g["fullAttentionPeriod"] == 0
        prefix = f"language_model.model.layers.{layer}"
        begin_file(f"target/layer-{layer}.bin", b"MDFL0006", layer, int(full))
        plain(prefix+".input_layernorm.weight", "input-norm", [h], note="Operative gain; no +1 or inverse centering")
        if full:
            attn = prefix+".self_attn"
            qwidth = 2*g["attentionQueryHeads"]*g["attentionHeadDimension"]
            kvwidth = g["attentionKvHeads"]*g["attentionHeadDimension"]
            q4("attention-input", g["packedFullWidth"], h, slices=[
                (attn+".q_proj", 0, qwidth), (attn+".k_proj", qwidth, kvwidth),
                (attn+".v_proj", qwidth+kvwidth, kvwidth)])
            plain(attn+".q_norm.weight", "query-norm", [g["attentionHeadDimension"]], note="Operative gain; no +1")
            plain(attn+".k_norm.weight", "key-norm", [g["attentionHeadDimension"]], note="Operative gain; no +1")
            q4("attention-output", h, g["attentionWidth"], prefix=attn+".o_proj")
        else:
            attn = prefix+".linear_attn"
            q4("gdn-input", g["packedGdnWidth"], h, slices=[
                (attn+".in_proj_qkv", 0, conv_width),
                (attn+".in_proj_z", conv_width, value_width),
                (attn+".in_proj_b", conv_width+value_width, value_heads),
                (attn+".in_proj_a", conv_width+value_width+value_heads, value_heads)])
            plain(attn+".conv1d.weight", "gdn-convolution", [conv_width, g["convolutionTaps"], 1],
                  note="Native channel-major oldest-to-newest taps; already MLX-sanitized [C,taps,1]")
            plain(attn+".A_log", "gdn-decay", [value_heads], "F32",
                  conversion="NOT_BYTE_COPY_requires_original_A_log_or_independent_forward_exp_certificate",
                  note="Source F32 is a_scale used directly in exp(a_scale*softplus), semantically -exp(A_log); stock A_log words are not certified recoverable. Preserve raw scale in an explicitly labeled private adapter if needed.")
            plain(attn+".dt_bias", "gdn-time-bias", [value_heads])
            plain(attn+".norm.weight", "gdn-norm", [d], note="Operative gated RMSNorm gain; no +1")
            q4("gdn-output", h, value_width, prefix=attn+".out_proj")
        plain(prefix+".post_attention_layernorm.weight", "post-attention-norm", [h], note="Operative gain; no +1")
        q4("mlp-gate", g["intermediateSize"], h, prefix=prefix+".mlp.gate_proj")
        q4("mlp-up", g["intermediateSize"], h, prefix=prefix+".mlp.up_proj")
        q4("mlp-down", h, g["intermediateSize"], prefix=prefix+".mlp.down_proj")
        finish_file()
    begin_file("target/head.bin", b"MDFL0002", g["layers"], 2)
    plain("language_model.model.norm.weight", "final-norm", [h], note="Operative gain; no +1")
    q4("logits", g["vocabularySize"], h, prefix="language_model.lm_head")
    finish_file()
    begin_file("target/embedding.bin", b"MDFE0001", g["vocabularySize"], h)
    n, k = g["vocabularySize"], h
    w = section("embedding-weights", n*k//2)
    s = section("embedding-scales", n*k//32)
    b = section("embedding-biases", n*k//32)
    emit_q4("language_model.model.embed_tokens", dict(w, native_role="embedding", n=n, k=k,
                                                    plane_offsets=(w["offset"],s["offset"],b["offset"])),
            0, n, storage="row-major")
    finish_file()
    keys = [entry["destination_key"] for entry in entries]
    require(len(keys) == len(set(keys)) == 1847, "Destination key coverage changed")
    codepaths = [ROOT / p for p in ("runtime/model/Qwen3_8.hpp", "runtime/model/Qwen3_8.cpp",
                 "runtime/model/StateLayout.hpp", "runtime/model/QwenTarget.hpp",
                 "runtime/model/QwenTarget.cpp", "runtime/model/WeightStore.hpp", "runtime/model/WeightStore.cpp",
                 "runtime/metal/kernels/common/q4_mpp_tiles.h", "runtime/metal/kernels/common/gdn_primitives.h",
                 "runtime/metal/kernels/common/attention_qkv_prepare.h",
                 "runtime/metal/kernels/shared/normalization.metal", "runtime/metal/kernels/shared/embedding.metal")]
    codepaths += [SITE / p for p in ("mlx_lm/utils.py", "mlx_lm/models/qwen3_5.py",
                                   "mlx_lm/models/qwen3_next.py", "mlx_lm/models/gated_delta.py")]
    codepaths.append(Path(__file__).resolve())
    return {"schema": "splash-private-q27-native-mlx-layout-plan-v1", "package": str(package),
            "manifest_sha256": sha(manifest_path), "upstream": manifest["upstream"],
            "original_target_converter_sha256_declared": manifest["converter"]["target_sha256"],
            "original_target_converter_source_located": False,
            "runtime_geometry": g, "entries": entries, "native_sections": sections, "source_files": files,
            "destination_key_count": len(keys), "unresolved_A_log_vectors": 48,
            "all_upstream_source_words_recovered": False,
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            "source_code_sha256": {str(path): sha(path) for path in codepaths},
            "gdn_rows": {"qkv": [0, 10240], "z": [10240, 16384], "b": [16384, 16432],
                         "a": [16432, 16480], "discard_padding": [16480, 16640]},
            "attention_rows": {"q_proj": [0, 12288], "k_proj": [12288, 13312], "v_proj": [13312, 14336],
                               "q_proj_layout": "24 heads, each query256 then gate256; preserve this interleaving"},
            "rotary": {"stored_native_tensor": False, "rotary_pairs": 32, "rotary_dims": 64,
                       "head_dim": 256, "rope_theta": text["rope_parameters"]["rope_theta"],
                       "partial_rotary_factor": text["partial_rotary_factor"],
                       "configuration_only": True},
            "loader_sanitize": "Use loaded language_model.* keys, conv [C,4,1], no mtp.*. Current stock qwen3_5 sanitize then leaves operative gains unchanged.",
            "native_u32_index": "(((row//256)*(K//64)+word//8)*256+row%256)*8+word%8; little endian",
            "native_parameter_index": "((row//256)*(K//64)+group)*256+row%256",
            "gpu_commands": 0, "model_payload_copy_bytes": 0,
            "payload_hashes_recomputed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--plan", action="store_true", help="Print full tensor metadata plan; default prints bounded summary")
    args = parser.parse_args()
    plan = make_plan(args.package)
    if args.plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        summary = {k: plan[k] for k in ("schema", "package", "manifest_sha256", "destination_key_count",
                   "unresolved_A_log_vectors", "all_upstream_source_words_recovered", "gdn_rows", "attention_rows",
                   "loader_sanitize", "gpu_commands", "model_payload_copy_bytes")}
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
