#!/usr/bin/env python3
"""Source-only fixed-R4 F32 selection for the private Q4/I8 hybrid snapshot.

The transforms return text; they never mutate a source tree. The witness reads
JSON metadata and source text only. It does not open, map, hash, or load model or
operand payloads, instantiate Metal, or build an executable.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping


FLAG = "SPLASH_FLASH_HYBRID_Q4_I8_FIXED_R4"
CPP_PATH = "runtime/flash/FlashFloatDenseCache.cpp"
HEADER_PATH = "runtime/flash/FlashFloatDenseCache.hpp"
EXPECTED_COUNT = 118
EXPECTED_PAYLOAD_BYTES = 3_247_964_160
ALIGNMENT = 16_384

_CPP_ANCHOR = "namespace splash::flash {\nbool flashQSAOutF32N32Enabled() {"
_FLAG_FUNCTION = '''bool flashHybridQ4I8FixedR4Enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_FLASH_HYBRID_Q4_I8_FIXED_R4");
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("SPLASH_FLASH_HYBRID_Q4_I8_FIXED_R4 must be 0 or 1");
  }();
  return enabled;
}
'''
_ORIGINAL_DEFAULT = '''std::vector<std::string> FlashFloatDenseCache::defaultPrefixes(const FlashWeights &weights,
                                                            bool includeVocabularyHead) {
  return FlashDenseCache::defaultPrefixes(weights, includeVocabularyHead);
}
'''
_FIXED_R4_DEFAULT = '''std::vector<std::string> FlashFloatDenseCache::defaultPrefixes(const FlashWeights &weights,
                                                            bool includeVocabularyHead) {
  if (!flashHybridQ4I8FixedR4Enabled())
    return FlashDenseCache::defaultPrefixes(weights, includeVocabularyHead);
  if (includeVocabularyHead)
    throw std::invalid_argument("private fixed-R4 hybrid requires the original code head and no F32 vocabulary cache");
  for (const char *flag : {"SPLASH_FLASH_FLOAT_DENSE_CACHE", "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE",
                           "SPLASH_FLASH_QMV_F32"}) {
    const char *value = std::getenv(flag);
    if (!value || std::string_view(value) != "1")
      throw std::invalid_argument(std::string("private fixed-R4 hybrid requires ") + flag + "=1");
  }
  const auto &descriptor = weights.descriptor();
  if (descriptor.layers != 48 || descriptor.pleLayerIndices.size() != 1 ||
      descriptor.pleLayerIndices.front() != 1)
    throw std::invalid_argument("private fixed-R4 hybrid requires 48 layers and zero-based PLE layer 1 only");
  // Use the same bounded metadata selection for construction and admission.
  // This changes which saved F32 maps are retained, never their coefficients.
  const auto names = normalize(FlashDenseCache::defaultPrefixes(weights, false));
  std::vector<std::string> selected;
  uint64_t payloadBytes = 0;
  uint32_t hcUp = 0, qsaOutput = 0, generic = 0;
  for (const auto &name : names) {
    const auto role = policyRole(name);
    // Forward executes PLE projections on the descriptor's sole active layer.
    if (role.starts_with("ple.") &&
        !name.starts_with("language_model.model.layers.1.ple.")) continue;
    const auto &p = weights.projection(name);
    if (!flashFloatDenseSmallRowsPolicy(name, 4, p.outputSize, p.inputSize,
                                       p.bits, p.groupSize)) continue;
    if (p.experts != 1)
      throw std::invalid_argument("private fixed-R4 hybrid selected a non-dense projection");
    payloadBytes = plus(payloadBytes, rounded(product(product(p.outputSize, p.inputSize), 4)));
    if (role == "hc_up") ++hcUp;
    else if (role == "self_attn.o_proj") ++qsaOutput;
    else ++generic;
    selected.push_back(name);
  }
  if (selected.size() != 118 || hcUp != 97 || qsaOutput != 7 || generic != 14 ||
      payloadBytes != uint64_t{3247964160})
    throw std::invalid_argument("private fixed-R4 hybrid F32 selection requires 118 maps (97 HC-up, 7 QSA output, 14 generic) and 3247964160 payload bytes");
  return selected;
}
'''
_HEADER_ANCHOR = "namespace splash::flash {\n\nenum class FlashFloatDenseSmallRowsTile"
_HEADER_DECLARATION = '''// Private singleton/depth-3 hybrid identity; unset/0 is off, 1 is on.
// The strict flag is exposed so Worker can reject dependencies before Metal.
[[nodiscard]] bool flashHybridQ4I8FixedR4Enabled();

'''


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{label}: expected exactly one frozen source anchor")
    return text.replace(old, new, 1)


def transform_cpp(text: str) -> str:
    """Insert strict opt-in and fixed-R4 metadata pruning; retain other bytes."""
    if "flashHybridQ4I8FixedR4Enabled" in text or FLAG in text:
        raise ValueError("fixed-R4 overlay is already present")
    result = _replace_once(text, _CPP_ANCHOR,
                           "namespace splash::flash {\n" + _FLAG_FUNCTION +
                           "bool flashQSAOutF32N32Enabled() {", "strict flag")
    return _replace_once(result, _ORIGINAL_DEFAULT, _FIXED_R4_DEFAULT,
                         "F32 default selection")


def transform_header(text: str) -> str:
    """Expose only the strict identity flag to the private Worker overlay."""
    if "flashHybridQ4I8FixedR4Enabled" in text:
        raise ValueError("fixed-R4 declaration is already present")
    return _replace_once(text, _HEADER_ANCHOR,
                         "namespace splash::flash {\n\n" + _HEADER_DECLARATION +
                         "enum class FlashFloatDenseSmallRowsTile", "header declaration")


def transform_sources(sources: Mapping[str, str]) -> dict[str, str]:
    """Return a copy with only the two F32 cache source files transformed."""
    result = dict(sources)
    result[CPP_PATH] = transform_cpp(sources[CPP_PATH])
    result[HEADER_PATH] = transform_header(sources[HEADER_PATH])
    return result


def validate_source_transform(originals: Mapping[str, str],
                              transformed: Mapping[str, str]) -> dict:
    """Prove the existing F32 policy, coefficient cache and kernels are intact."""
    expected = transform_sources(originals)
    if dict(transformed) != expected:
        raise ValueError("unexpected source edits outside the two text transforms")
    cpp = transformed[CPP_PATH]
    recovered = _replace_once(cpp, _FIXED_R4_DEFAULT, _ORIGINAL_DEFAULT, "reverse selection")
    recovered = _replace_once(recovered, _FLAG_FUNCTION, "", "reverse strict flag")
    if recovered != originals[CPP_PATH]:
        raise ValueError("original F32 implementation bytes changed")
    header = _replace_once(transformed[HEADER_PATH], _HEADER_DECLARATION, "", "reverse declaration")
    if header != originals[HEADER_PATH]:
        raise ValueError("original F32 header bytes changed")
    return dict(only_files_changed=[CPP_PATH, HEADER_PATH],
                raw_f32_coefficients_precision_changed=False,
                coefficient_cache_and_dispatch_blocks_byte_identical=True,
                original_29_entry_policy_byte_identical=True,
                flag_off_calls_original_default_prefixes=True,
                strict_global_flag=FLAG)


def _default_dense_names(model: dict, config: dict) -> list[str]:
    """Mirror the existing default inventory using tensor presence metadata."""
    names = []

    def add(name: str) -> None:
        if name + ".weight" in model["tensors"]:
            names.append(name)

    for layer in range(config["num_hidden_layers"]):
        prefix = f"language_model.model.layers.{layer}"
        for kind in ("attn_hyper_connection", "mlp_hyper_connection"):
            add(prefix + "." + kind + ".input_mix_weight_down")
            add(prefix + "." + kind + ".input_mix_weight_up")
        layer_type = config["layer_types"][layer]
        if layer_type == "linear_attention":
            for kind in ("in_proj_qkv", "in_proj_z", "out_proj"):
                add(prefix + ".linear_attn." + kind)
        elif layer_type == "full_attention":
            for kind in ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"):
                add(prefix + ".self_attn." + kind)
        else:
            raise ValueError("unknown layer kind in default dense metadata")
        for kind in ("gate_proj", "up_proj", "down_proj"):
            add(prefix + ".mlp.shared_expert." + kind)
        add(prefix + ".ple.key_proj")
        add(prefix + ".ple.value_proj")
    add("language_model.model.hyper_connection_mixer.input_mix_weight_down")
    add("language_model.model.hyper_connection_mixer.input_mix_weight_up")
    return sorted(set(names))


def _policy_role(name: str) -> str:
    """Mirror policyRole's bounded prefix grammar without tensor data."""
    layer_match = re.fullmatch(r"language_model\.model\.layers\.([0-9]{1,2})\.(.+)", name)
    if layer_match is not None:
        if int(layer_match[1]) >= 48:
            return ""
        role = layer_match[2]
        if role in ("attn_hyper_connection.input_mix_weight_up",
                    "mlp_hyper_connection.input_mix_weight_up"):
            return "hc_up"
        return role
    if name == "language_model.model.hyper_connection_mixer.input_mix_weight_up":
        return "hc_up"
    return ""


def metadata_witness(root: Path, transformed_sources: Mapping[str, str],
                     forward_text: str) -> dict:
    """Calculate the exact guarded R4 set from saved/model JSON metadata only."""
    root = root.resolve()
    model_path = root / "install/local-models/Flash-Next-oQ4e-mtp-v1/manifest.json"
    saved_path = root / "install/local-models/Flash-Next-operands-v1/manifest.json"
    config_path = model_path.parent / "config.json"
    model = json.loads(model_path.read_text())
    saved = json.loads(saved_path.read_text())
    config = json.loads(config_path.read_text())["text_config"]
    if model["source_identity_sha256"] != saved["source_identity_sha256"]:
        raise ValueError("saved operands do not match source identity")
    if config["num_hidden_layers"] != 48 or config["ple_layer_ids"] != [2]:
        raise ValueError("metadata is not the fixed 48-layer/single-PLE model")
    cpp = transformed_sources[CPP_PATH]
    if _FIXED_R4_DEFAULT not in cpp or _FLAG_FUNCTION not in cpp:
        raise ValueError("source lacks the exact guarded selection branch")
    if _HEADER_DECLARATION not in transformed_sources[HEADER_PATH]:
        raise ValueError("source lacks the strict pre-backend identity declaration")
    policy_pattern = re.compile(
        r'\{"([^"\n]+)",\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),'
        r'\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\}')
    policy = {}
    for match in policy_pattern.finditer(cpp):
        role, *values = match.groups()
        outputs, inputs, bits, group, *tiles = map(int, values)
        key = (role, outputs, inputs, bits, group)
        if key in policy:
            raise ValueError("duplicate F32 policy geometry")
        policy[key] = tiles
    if len(policy) != 29:
        raise ValueError("the frozen F32 policy must contain 29 geometries")
    entries = {e["projection"]: e for e in saved["entries"] if e["format"] == "F32"}
    if len(entries) != 508 or "language_model.lm_head" in entries:
        raise ValueError("saved metadata differs from the original 508-map body inventory")
    default_names = _default_dense_names(model, config)
    if default_names != sorted(entries):
        raise ValueError("saved F32 body metadata differs from the existing default dense names")
    selected = []
    for name in default_names:
        entry = entries[name]
        role = _policy_role(name)
        if role.startswith("ple.") and not name.startswith("language_model.model.layers.1.ple."):
            continue
        p = entry["source"]
        geometry = (role, p["output_size"], p["input_size"], p["bits"], p["group_size"])
        tiles = policy.get(geometry)
        if tiles is None or tiles[0] < 0:
            continue
        if p["experts"] != 1:
            raise ValueError("selected metadata projection is not dense")
        q = model["quantization"].get(name, model["quantization"])
        if (q["bits"], q["group_size"]) != (p["bits"], p["group_size"]):
            raise ValueError("selected quantization metadata differs from source")
        t = model["tensors"]
        if (t[name + ".weight"]["shape"][-2] != p["output_size"] or
                t[name + ".scales"]["shape"][-1] * p["group_size"] != p["input_size"]):
            raise ValueError("selected tensor shape metadata differs from source")
        logical = p["output_size"] * p["input_size"] * 4
        allocated = (logical + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        if (entry["logical_bytes"] != logical or entry["allocated_bytes"] != allocated or
                entry["shape"] != [p["output_size"], p["input_size"]] or
                entry["operand_math"] != "original-affine-contractoff-f32-coefficients-row-major-v1"):
            raise ValueError("selected saved F32 metadata has inconsistent extent/precision")
        route = "hc_up" if role == "hc_up" else "qsa_output" if role == "self_attn.o_proj" else "generic_projection"
        selected.append(dict(projection=name, role=role, route=route, f32_tile=tiles[0],
                             allocated_bytes=allocated, source=dict(p)))
    routes = collections.Counter(e["route"] for e in selected)
    payload_bytes = sum(e["allocated_bytes"] for e in selected)
    if (len(selected) != EXPECTED_COUNT or payload_bytes != EXPECTED_PAYLOAD_BYTES or
            routes != dict(hc_up=97, qsa_output=7, generic_projection=14)):
        raise ValueError("metadata does not yield the exact guarded 118-map R4 selection")
    constructor_call = "const auto prefixes = FlashFloatDenseCache::defaultPrefixes(weights, !codeHead);"
    planner_call = 'FlashFloatDenseCache::defaultPrefixes(weights, !fusionEnabled("SPLASH_FLASH_INT8_HEAD"))'
    if constructor_call not in forward_text or planner_call not in forward_text:
        raise ValueError("Forward constructor and planner no longer share defaultPrefixes")
    constructor_use = "floatDenseCache = std::make_unique<FlashFloatDenseCache>(backend, weights, prefixes);"
    if constructor_use not in forward_text:
        raise ValueError("Forward constructor no longer passes the selected names to the F32 cache")
    selected_cache_loop = "for (const auto &name : names) {"
    selected_saved_map = "store->mapTensor(backend, flashOperandSpec(name, FlashOperandFormat::F32, p))"
    selected_expansion = 'if (!saved) graph.add("flash_float_dense_cache_expand"'
    if any(text not in cpp for text in (selected_cache_loop, selected_saved_map, selected_expansion)):
        raise ValueError("F32 cache constructor no longer maps or expands its selected names only")
    names = [e["projection"] for e in selected]
    return dict(schema="splash-private-hybrid-fixed-r4-source-metadata-witness-v1",
                gpu_execution=False, executable_built=False, model_loaded=False,
                model_payload_bytes_read=0, model_payload_hashed=False,
                source_identity_sha256=model["source_identity_sha256"],
                flag=FLAG, physical_rows=4, count=len(selected), payload_bytes=payload_bytes,
                planned_cache_bytes_including_diagnostics=payload_bytes + ALIGNMENT,
                original_default_f32_count=len(entries),
                pruned_count=len(entries) - len(selected),
                pruned_payload_bytes=sum(e["allocated_bytes"] for e in entries.values()) - payload_bytes,
                by_route={route: dict(count=routes[route], payload_bytes=sum(e["allocated_bytes"] for e in selected if e["route"] == route)) for route in sorted(routes)},
                selected_names_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest(),
                constructor_and_planner_share_default_prefixes=True,
                constructor_maps_or_expands_only_selected_prefixes=True,
                policy_geometry_count=len(policy),
                raw_f32_coefficient_precision_changed=False,
                source_witness=dict(selection_branch=_FIXED_R4_DEFAULT,
                                    constructor_selection_call=constructor_call,
                                    constructor_selected_names_use=constructor_use,
                                    constructor_selected_cache_loop=selected_cache_loop,
                                    constructor_selected_saved_map=selected_saved_map,
                                    constructor_selected_expansion=selected_expansion,
                                    planner_selection_call=planner_call),
                metadata_files_read=[str(model_path), str(saved_path), str(config_path)],
                selected=selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source", type=Path,
                        default=Path("build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1/source"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = args.source if args.source.is_absolute() else root / args.source
    originals = {name: (source / name).read_text() for name in (CPP_PATH, HEADER_PATH)}
    transformed = transform_sources(originals)
    result = metadata_witness(root, transformed, (source / "runtime/flash/FlashForward.cpp").read_text())
    result["source_transform_validation"] = validate_source_transform(originals, transformed)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("schema", "model_payload_bytes_read", "count", "payload_bytes", "pruned_count", "pruned_payload_bytes", "by_route")}))


if __name__ == "__main__":
    main()
