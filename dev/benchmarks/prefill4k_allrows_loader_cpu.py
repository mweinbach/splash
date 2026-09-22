#!/usr/bin/env python3
"""Bounded CPU-only checks for the private all-row target-expert loader.

This reads JSON metadata and source text only. It never opens model payloads,
imports an inference framework, creates a Metal device, or submits GPU work.
The optional standalone C++ helper check uses a temporary directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT = 16384
METADATA_LIMIT = 2 << 20
SOURCE_IDENTITY = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
SOURCE_MANIFEST_SHA256 = "0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402"
FULL512_MANIFEST_SHA256 = "ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1"
SOURCE_PAYLOAD_SIGNATURE = "675e220b8c60ea0a61e812125bf1acdb1c5e85f50891f5e74b97099761701ae4"
FULL512_PAYLOAD_SIGNATURE = "5eb8a67efcb9fd0085a629903fe36a388c50aa02e27e071b6673b5be54491ad7"
SOURCE_FINGERPRINT = "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0"
ROLES = ("gate_proj", "up_proj", "down_proj")
PLANES = ("weight", "scales", "biases")
TARGET_PATTERN = re.compile(
    r"language_model\.model\.layers\.(0|[1-9][0-9]*)\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)\Z"
)
PLE_PATTERN = re.compile(
    r"language_model\.model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\."
    r"shards\.(0|[1-9][0-9]*)\.(weight|scales|biases)\Z"
)


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def aligned(value: int) -> int:
    require(value >= 0, "negative page extent")
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def read_metadata(path: Path) -> tuple[bytes, dict]:
    require(path.is_file(), f"missing metadata: {path}")
    require(0 < path.stat().st_size <= METADATA_LIMIT, f"unbounded metadata: {path}")
    with path.open("rb") as stream:
        raw = stream.read(METADATA_LIMIT + 1)
    require(len(raw) <= METADATA_LIMIT, "metadata grew beyond bounded read")
    result = json.loads(raw)
    require(isinstance(result, dict), "metadata must be an object")
    return raw, result


def target_name(name: str) -> bool:
    match = TARGET_PATTERN.fullmatch(name)
    return bool(match and int(match[1]) < 48)


def ple_name(name: str) -> bool:
    match = PLE_PATTERN.fullmatch(name)
    return bool(match and int(match[1]) < 128)


def payload_signature(records: list[dict]) -> str:
    value = [(row["path"], row["bytes"], row["sha256"]) for row in records]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def load_overlay():
    path = ROOT / "dev/benchmarks/prefill4k_allrows_loader.py"
    require(path.is_file(), "private loader module is not available")
    # Importing this dev helper must not create bytecode outside this new test.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("prefill4k_allrows_loader_under_test", path)
    require(spec is not None and spec.loader is not None, "cannot import private loader helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.TARGET_SOURCE_IDENTITY == SOURCE_IDENTITY, "overlay source identity changed")
    require(module.TARGET_MANIFEST_SHA256 == SOURCE_MANIFEST_SHA256, "overlay source manifest lock changed")
    require(module.TARGET_FINGERPRINT == SOURCE_FINGERPRINT, "overlay architecture fingerprint changed")
    require(module.TARGET_FULL512_MANIFEST_SHA256 == FULL512_MANIFEST_SHA256,
            "overlay certified Full512 manifest lock changed")
    return module


def predicate_cases() -> list[tuple[str, bool]]:
    positives = [
        (f"language_model.model.layers.{layer}.mlp.switch_mlp.{role}.{plane}", True)
        for layer in range(48) for role in ROLES for plane in PLANES
    ]
    negatives: set[str] = set()
    for layer in ("", "00", "01", "047", "48", "49", "-1", "+0", "+1", "0.0", " 1", "1 ",
                  "18446744073709551616", "9999999999999999999999999999999999999999", "\u0661", "\uff11"):
        for role in ROLES:
            for plane in PLANES:
                negatives.add(f"language_model.model.layers.{layer}.mlp.switch_mlp.{role}.{plane}")
    for name, _ in positives:
        for suffix in (".extra", ".weight", ".", " ", "\n", "\r", "\x00"):
            negatives.add(name + suffix)
        negatives.add("x." + name)
        negatives.add(name.replace(".switch_mlp.", ".shared_expert."))
        negatives.add(name.replace(".switch_mlp.", ".shared_expert_gate."))
        negatives.add(name.replace("language_model.model.layers.", "mtp.layers."))
        negatives.add(name.replace("language_model.model.layers.", "language_model.model.mtp.layers."))
        negatives.add(name.replace(".switch_mlp.", ".switch_mlpx."))
        negatives.add(name.replace(".mlp.", ".mlp.."))
    for role in ("gate", "up", "down", "gate_projs", "Gate_proj", "", "router"):
        negatives.add(f"language_model.model.layers.0.mlp.switch_mlp.{role}.weight")
    for plane in ("bias", "scale", "weights", "Weight", "", "codes"):
        negatives.add(f"language_model.model.layers.0.mlp.switch_mlp.gate_proj.{plane}")
    require(len(positives) == 432, "predicate positive inventory changed")
    require(not any(target_name(name) for name in negatives), "test negative unexpectedly matches independent predicate")
    return positives + [(name, False) for name in sorted(negatives)]


def independent_source_plan(source: dict) -> dict:
    require(source["source_identity_sha256"] == SOURCE_IDENTITY, "source identity changed")
    require(source["schema"] == "splash-local-qwen4-affine-v1" and source["alignment"] == ALIGNMENT,
            "source schema/alignment changed")
    require(len(source["tensors"]) == 3748 and len(source["shards"]) == 21, "source inventory changed")
    require(payload_signature(source["shards"]) == SOURCE_PAYLOAD_SIGNATURE, "source payload signatures changed")
    require(source["quantization"]["bits"] == 4 and source["quantization"]["group_size"] == 64,
            "source Q4/G64 default changed")
    target = {name: record for name, record in source["tensors"].items() if target_name(name)}
    wanted = {name for name, accepted in predicate_cases() if accepted}
    require(set(target) == wanted, "target omission set must contain exactly the 432 checked names")
    for name, record in target.items():
        match = TARGET_PATTERN.fullmatch(name)
        require(match is not None, "target name parser disagreement")
        role, plane = match[2], match[3]
        n, k = (2560, 640) if role == "down_proj" else (640, 2560)
        packed = plane == "weight"
        expected_shape = [512, n, k // (8 if packed else 64)]
        require(record["shape"] == expected_shape, f"source expert geometry changed: {name}")
        require(record["dtype"] == ("U32" if packed else "BF16"), f"source expert dtype changed: {name}")
        expected_bytes = 512 * n * expected_shape[2] * (4 if packed else 2)
        require(record["length"] == expected_bytes, f"source expert byte geometry changed: {name}")
        prefix = name.rsplit(".", 1)[0]
        override = source["quantization"].get(prefix, {})
        require(override.get("bits", 4) == 4 and override.get("group_size", 64) == 64
                and override.get("mode", "affine") == "affine", f"expert Q4/G64 route changed: {prefix}")
    stats = {
        "target_tensor_count": 0, "target_logical_bytes": 0, "target_page_bytes": 0,
        "ple_tensor_count": 0, "ple_logical_bytes": 0, "ple_page_bytes": 0,
        "gpu_mapped_bytes": 0, "disk_only_payload_bytes": 0, "gpu_window_count": 0,
        "fully_disk_payload_count": 0, "original_payload_bytes": 0, "windows": {},
    }
    seen: set[str] = set()
    for shard in source["shards"]:
        shard_name = shard["path"]
        require(shard_name not in seen, "duplicate source shard")
        seen.add(shard_name)
        rows = sorted((r["offset"], r["offset"] + r["length"], name)
                      for name, r in source["tensors"].items() if r["shard"] == shard_name)
        require(rows and shard["bytes"] % ALIGNMENT == 0, "source shard has no aligned tensor rows")
        cursor = 0
        windows: list[list[int]] = []
        for begin, end, name in rows:
            require(begin == aligned(cursor) and begin < end <= shard["bytes"], "source packing is not canonical")
            padded_end = aligned(end)
            page_bytes = padded_end - begin
            is_target, is_ple = target_name(name), ple_name(name)
            require(not (is_target and is_ple), "target/PLE omission sets overlap")
            if is_target or is_ple:
                category = "target" if is_target else "ple"
                stats[category + "_tensor_count"] += 1
                stats[category + "_logical_bytes"] += end - begin
                stats[category + "_page_bytes"] += page_bytes
                stats["disk_only_payload_bytes"] += page_bytes
            else:
                stats["gpu_mapped_bytes"] += page_bytes
                if windows and windows[-1][1] == begin:
                    windows[-1][1] = padded_end
                else:
                    windows.append([begin, padded_end])
            cursor = end
        require(aligned(cursor) == shard["bytes"], "source tail geometry changed")
        stats["gpu_window_count"] += len(windows)
        stats["fully_disk_payload_count"] += not windows
        stats["original_payload_bytes"] += shard["bytes"]
        stats["windows"][shard_name] = windows
    require(seen == {row["shard"] for row in source["tensors"].values()}, "tensor names unknown source shard")
    expected = {
        "target_tensor_count": 432, "target_logical_bytes": 67947724800, "target_page_bytes": 67947724800,
        "ple_tensor_count": 384, "ple_logical_bytes": 32000153600, "ple_page_bytes": 32002539520,
        "gpu_mapped_bytes": 6370164736, "disk_only_payload_bytes": 99950264320,
        "gpu_window_count": 7, "fully_disk_payload_count": 19, "original_payload_bytes": 106320429056,
    }
    require({key: stats[key] for key in expected} == expected, "checked private omission/page-window geometry changed")
    require(stats["gpu_mapped_bytes"] + stats["disk_only_payload_bytes"] == stats["original_payload_bytes"],
            "CPU page plan does not partition source payloads")
    return stats


def check_full512_metadata(full_path: Path, source: dict) -> dict:
    raw, full = read_metadata(full_path)
    require(hashlib.sha256(raw).hexdigest() == FULL512_MANIFEST_SHA256, "certified Full512 metadata bytes changed")
    _, certificate = read_metadata(full_path.parent / "coefficient-certificate.json")
    require(certificate["pass"] is True and certificate["gpu_work"] is False
            and certificate["certificate_completed_before_atomic_publication"] is True,
            "Full512 coefficient certificate is incomplete")
    require(certificate["full_manifest_sha256"] == FULL512_MANIFEST_SHA256, "certificate/Full512 manifest identity changed")
    require(full["source_identity_sha256"] == source["source_identity_sha256"]
            and full["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256, "Full512/source metadata binding changed")
    require(full["target_layers"] == 48 and len(full["layers"]) == 48
            and full["selected_experts"] == [list(range(512)) for _ in range(48)], "Full512 inventory changed")
    require(full["total_bytes"] == 121173442560 and full["planned_allocation_bytes"] == 121174228992,
            "Full512 byte ledger changed")
    require(payload_signature(full["layers"]) == FULL512_PAYLOAD_SIGNATURE, "Full512 payload signatures changed")
    for layer_index, layer in enumerate(full["layers"]):
        require(layer["layer_index"] == layer_index and layer["bytes"] == 2524446720,
                "Full512 layer geometry changed")
        require(set(layer["projections"]) == set(ROLES), "Full512 projection inventory changed")
        extents = []
        for role, projection in layer["projections"].items():
            n, k = (2560, 640) if role == "down_proj" else (640, 2560)
            prefix = f"language_model.model.layers.{layer_index}.mlp.switch_mlp.{role}"
            require(projection["source_prefix"] == prefix and prefix + ".weight" in source["tensors"],
                    "Full512 projection source binding changed")
            require(projection["dimensions"] == [512, n, k], "Full512 source dimensions changed")
            for plane, shape, dtype, length in (
                ("codes", [512, n, k], "I8", 512 * n * k),
                ("scales", [512, n], "F32", 512 * n * 4),
            ):
                row = projection[plane]
                require(row["shape"] == shape and row["dtype"] == dtype and row["length"] == length,
                        "Full512 plane geometry changed")
                require(row["offset"] % ALIGNMENT == 0 and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                        "Full512 plane identity/alignment changed")
                extents.append((row["offset"], row["offset"] + row["length"]))
        cursor = 0
        for begin, end in sorted(extents):
            require(begin == aligned(cursor) and begin < end <= layer["bytes"], "Full512 packing is not canonical")
            cursor = end
        require(aligned(cursor) == layer["bytes"], "Full512 tail bytes changed")
    return {"manifest_sha256": FULL512_MANIFEST_SHA256, "payload_manifest_signature": FULL512_PAYLOAD_SIGNATURE,
            "certificate_metadata_pass": True, "payload_sha_reverification": False,
            "coefficient_model_quality_qualified": certificate["model_quality_qualified"]}


def inspect_without_payload_reads(overlay, source_path: Path, full_path: Path) -> tuple[dict, dict]:
    allowed = {source_path.resolve(), full_path.resolve()}
    original_read = Path.read_bytes

    def bounded_read(path: Path) -> bytes:
        require(path.resolve() in allowed, f"inspection attempted unexpected file read: {path}")
        require(path.stat().st_size <= METADATA_LIMIT, "inspection attempted unbounded metadata read")
        return original_read(path)

    with patch.object(Path, "read_bytes", bounded_read):
        return overlay.inspect_source_manifest(source_path), overlay.inspect_full512_manifest(full_path)


def compare_inspection(inspection: dict, independent: dict, source: dict) -> None:
    equivalent = {
        "target_tensor_count": independent["target_tensor_count"],
        "target_projection_count": 144,
        "target_logical_bytes": independent["target_logical_bytes"],
        "target_payload_bytes": independent["target_page_bytes"],
        "disk_tensor_count": independent["target_tensor_count"] + independent["ple_tensor_count"],
        "disk_projection_count": 272,
        "disk_logical_bytes": independent["target_logical_bytes"] + independent["ple_logical_bytes"],
        "disk_payload_bytes": independent["disk_only_payload_bytes"],
        "original_payload_bytes": independent["original_payload_bytes"],
        "gpu_mapped_bytes": independent["gpu_mapped_bytes"],
        "native_window_count": independent["gpu_window_count"],
        "fully_disk_payload_count": independent["fully_disk_payload_count"],
    }
    require({key: inspection[key] for key in equivalent} == equivalent,
            "overlay inspection and independent omission plan differ")
    windows = [{"path": shard["path"], "begin": begin, "end": end}
               for shard in source["shards"]
               for begin, end in independent["windows"][shard["path"]]]
    require(inspection["windows"] == windows, "overlay native windows differ from independent canonical page plan")
    signature = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target_records = [{"name": name, **record} for name, record in sorted(source["tensors"].items())
                      if target_name(name)]
    require(inspection["window_signature_sha256"] == signature(windows), "native window signature differs")
    require(inspection["skip_signature_sha256"] == signature(target_records), "target omission signature differs")


def check_source_transform(overlay) -> dict:
    relatives = ("runtime/flash/FlashWeights.hpp", "runtime/flash/FlashWeights.mm")
    transformed = {}
    for relative in relatives:
        source = (ROOT / relative).read_text()
        require(len(source.encode()) <= METADATA_LIMIT, "source check exceeded bounded text size")
        transformed[relative] = overlay.transform(relative, source)
        require(transformed[relative] != source, f"private loader transform did not patch {relative}")
    sentinel = "untouched source text\n"
    for relative in ("runtime/flash/FlashMTP.cpp", "runtime/flash/FlashForward.cpp",
                     "runtime/flash/FlashInt8ExpertStore.mm", "runtime/flash/FlashPLESSDStore.mm",
                     "runtime/metal/MetalBackend.mm"):
        require(overlay.transform(relative, sentinel) == sentinel,
                f"loader transform must modify only FlashWeights.hpp/.mm: {relative}")
    text = transformed["runtime/flash/FlashWeights.mm"]
    original = (ROOT / "runtime/flash/FlashWeights.mm").read_text()
    hash_guards = lambda value: re.findall(r"if \(verifyPayloadHashes[\s\S]*?fail\([^;]*;", value)
    require(hash_guards(original) == hash_guards(text) and len(hash_guards(text)) == 2,
            "private omission must preserve both existing verifyPayloadHashes conditions unchanged")
    loader = text[text.index("FlashWeights FlashWeights::load("):text.index("const FlashTensor &FlashWeights::tensor(")]
    preflight = loader.index("loadFlashInt8ExpertStoreMetadata(allRowsStorePath,")
    first_map = loader.index("Mapping::open(")
    first_wrap = loader.index("backend.wrapSharedMemory(")
    require(preflight < first_map and preflight < first_wrap, "certified-store preflight must precede all original mapping/wrapping")
    require(loader.index("manifestDigest !=") < preflight, "qualified source manifest lock must precede store preflight")
    require("if (!streamingEnabled)" in loader[:preflight]
            and 'flashAllRowsFull512TargetValue(std::getenv("SPLASH_FLASH_ALLROWS_FULL512_TARGET"))' in loader[:preflight],
            "private omission flags must fail closed before native buffers")
    for digest in (SOURCE_IDENTITY, SOURCE_MANIFEST_SHA256, FULL512_MANIFEST_SHA256, SOURCE_FINGERPRINT):
        require(digest in loader[:first_map], "private preflight lost a locked identity")
    require("layer.selectedIDs.size() != 512" in loader[:first_map]
            and "layer.selectedIDs[expert] != expert" in loader[:first_map],
            "private preflight lost canonical all512 coverage checks")
    require(text.count("if (impl.pleSSD.enabled && flashAllRowsTargetDiskOnlyTensor(name))") == 2,
            "both original tensor/projection branches must use the exact disk-only predicate")
    require("offset + length, flashAllRowsTargetDiskOnlyTensor(tensorName)" in loader,
            "native page window planner must omit PLE or exact target tensors")
    require("p.weights->buffer || p.scales->buffer || p.biases->buffer" in loader,
            "omitted original targets must retain checked nil native buffers")
    require("projectionMetadata(target + projection)" in loader
            and "const auto &value = result.projectionMetadata(prefix);" in loader,
            "original semantics must validate metadata without ordinary GPU lookup")
    require("impl.pleSSD.targetOriginalDiskTensorCount != 432" in loader
            and "impl.pleSSD.targetOriginalDiskProjectionCount != 144" in loader,
            "original target ownership inventory guards missing")
    require("impl.bases.push_back(std::move(base));" in loader
            and "backend.view(window->base, offset - window->begin, length)" in loader,
            "remaining original tensors lost stable native window ownership")
    require('impl.diskProjections.at(ple + "ngram_embedding.shards." + std::to_string(part))' in loader,
            "PLE SSD staging must remain restricted to original PLE parts")
    metadata_start = text.index("const FlashAffineProjection &FlashWeights::projectionMetadata(")
    metadata_end = text.index("bool FlashWeights::contains(", metadata_start)
    metadata = text[metadata_start:metadata_end]
    require("impl_->projections.find(prefix)" in metadata and "return found->second;" in metadata,
            "projectionMetadata must expose stable original projection entries")
    for forbidden in ("Mapping::open", "wrapSharedMemory", "backend.view", "reserve(", "make_unique", "mmap"):
        require(forbidden not in metadata, "projectionMetadata must not allocate/materialize native buffers")
    require(text.count("GPU fallback/materialization forbidden") == 2,
            "ordinary tensor/projection lookup must fail closed for disk-only original coefficients")
    return {"modified_relatives": list(relatives), "store_preflight_before_original_mapping": True,
            "nil_target_buffers_checked": True, "metadata_lookup_materializes_gpu": False,
            "ordinary_gpu_fallback_forbidden": True}


def check_fail_closed_metadata(overlay, source: dict, full_path: Path) -> list[str]:
    _, full = read_metadata(full_path)
    checks = []
    with tempfile.TemporaryDirectory(prefix="splash-allrows-metadata-cpu-") as directory:
        path = Path(directory) / "manifest.json"

        def rejected(record, inspect, label, digest_attribute=None):
            raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            require(len(raw) <= METADATA_LIMIT, "negative fixture exceeded metadata bound")
            path.write_bytes(raw)
            context = patch.object(overlay, digest_attribute, hashlib.sha256(raw).hexdigest()) if digest_attribute else patch.object(overlay, "ROOT", ROOT)
            with context:
                try:
                    inspect(path)
                except (ValueError, KeyError, TypeError):
                    checks.append(label)
                else:
                    raise AssertionError(f"private metadata inspection accepted {label}")

        # Byte-lock negatives intentionally keep the real immutable digest.
        for label, key, value in (
            ("source_identity_lock", "source_identity_sha256", "0" * 64),
            ("source_alignment_lock", "alignment", 8192),
            ("source_quantization_lock", "quantization", {"bits": 8, "group_size": 64}),
        ):
            changed = copy.deepcopy(source)
            changed[key] = value
            rejected(changed, overlay.inspect_source_manifest, label)
        # Controlled digest replacement exercises inner geometry guards while
        # remaining wholly CPU-only; production remains locked to exact bytes.
        tensor_name = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
        for label, key, value in (
            ("target_expert_geometry", "shape", [511, 640, 320]),
            ("target_packed_geometry", "shape", [512, 640, 321]),
            ("target_source_dtype", "dtype", "BF16"),
            ("target_source_length", "length", 419430399),
        ):
            changed = copy.deepcopy(source)
            changed["tensors"][tensor_name][key] = value
            rejected(changed, overlay.inspect_source_manifest, label, "TARGET_MANIFEST_SHA256")
        changed = copy.deepcopy(source)
        changed["tensors"][tensor_name.replace("layers.0.", "layers.00.")] = changed["tensors"].pop(tensor_name)
        rejected(changed, overlay.inspect_source_manifest, "target_exact_inventory", "TARGET_MANIFEST_SHA256")
        changed = copy.deepcopy(source)
        other = next(name for name in changed["tensors"] if not target_name(name) and not ple_name(name))
        changed["tensors"][other]["offset"] += ALIGNMENT
        rejected(changed, overlay.inspect_source_manifest, "canonical_partition", "TARGET_MANIFEST_SHA256")
        changed = copy.deepcopy(source)
        changed["shards"][0]["bytes"] += ALIGNMENT
        rejected(changed, overlay.inspect_source_manifest, "canonical_tail", "TARGET_MANIFEST_SHA256")
        for label, mutate in (
            ("full512_store_byte_lock", lambda store: store.update(total_bytes=1)),
            ("full512_inventory", lambda store: store["selected_experts"][0].pop()),
            ("full512_identity", lambda store: store.update(source_identity_sha256="0" * 64)),
            ("full512_layer_inventory", lambda store: store["layers"].pop()),
        ):
            changed = copy.deepcopy(full)
            mutate(changed)
            rejected(changed, overlay.inspect_full512_manifest, label,
                     None if label.endswith("byte_lock") else "TARGET_FULL512_MANIFEST_SHA256")
        path.write_text('{"schema":1,"schema":2}')
        try:
            overlay.inspect_source_manifest(path)
        except ValueError:
            checks.append("duplicate_metadata_key")
        else:
            raise AssertionError("private metadata inspection accepted duplicate key")
        with path.open("wb") as stream:
            stream.truncate((32 << 20) + 1)
        try:
            overlay.inspect_source_manifest(path)
        except ValueError:
            checks.append("metadata_bounded_size")
        else:
            raise AssertionError("private metadata inspection accepted oversized sparse metadata")
    return checks


def check_cpp_helpers(overlay, cases: list[tuple[str, bool]], source: dict) -> dict:
    compiler = shutil.which("clang++")
    require(compiler is not None, "clang++ unavailable for independent CPU helper qualification")
    cpp = r'''#include "flash/FlashPLESSDLayout.hpp"
#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
using namespace splash::flash;
'''
    cpp += overlay.CPP_HELPERS
    cpp += r'''
int main() {
  if (flashAllRowsFull512TargetValue(nullptr) || flashAllRowsFull512TargetValue("0") ||
      !flashAllRowsFull512TargetValue("1")) return 1;
  for (const char *bad : {"", "01", "true", "-1", "2", " 1", "1 "}) {
    bool denied = false;
    try { (void)flashAllRowsFull512TargetValue(bad); }
    catch (const std::invalid_argument &) { denied = true; }
    if (!denied) return 2;
  }
  std::string line;
  uint64_t count = 0;
  while (std::getline(std::cin, line)) {
    if (line.size() < 3 || (line.size() - 3) % 2 || line[2] != ':') return 3;
    std::string name;
    for (size_t i = 3; i < line.size(); i += 2) {
      const auto digit = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
        throw std::runtime_error("invalid case hex");
      };
      name.push_back(static_cast<char>((digit(line[i]) << 4) | digit(line[i + 1])));
    }
    if (flashAllRowsTargetSwitchTensor(name) != (line[0] == '1') ||
        flashAllRowsTargetDiskOnlyTensor(name) != (line[1] == '1')) {
      std::cerr << "CPU predicate mismatch in case " << count << '\n';
      return 4;
    }
    ++count;
  }
  const auto allDisk = flashPLESSDPlan(32768, {{0, 16384, true}, {16384, 32768, true}});
  if (!allDisk.windows.empty() || allDisk.diskOnlyBytes != 32768 || allDisk.diskTensorCount != 2) return 5;
  const auto split = flashPLESSDPlan(49152, {{0, 16384, false}, {16384, 32768, true}, {32768, 49152, false}});
  if (split.windows.size() != 2 || split.mappedBytes != 32768 || split.diskOnlyBytes != 16384) return 6;
  bool denied = false;
  try { (void)flashPLESSDPlan(49152, {{0, 16384, false}, {32768, 49152, false}}); }
  catch (const std::invalid_argument &) { denied = true; }
  if (!denied) return 7;
  std::cout << count << '\n';
}
'''
    all_cases = list(cases) + [(name, target_name(name)) for name in sorted(source["tensors"])]
    stdin = "".join(f"{int(expected)}{int(expected or ple_name(name))}:{name.encode().hex()}\n"
                    for name, expected in all_cases)
    require(len(stdin) <= (4 << 20), "standalone helper input exceeded bounded CPU fixture size")
    with tempfile.TemporaryDirectory(prefix="splash-allrows-helper-cpu-") as directory:
        path = Path(directory) / "helper.cpp"
        binary = Path(directory) / "helper-cpu"
        path.write_text(cpp)
        compile_result = subprocess.run([compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                                         "-I", str(ROOT / "runtime"), str(path), "-o", str(binary)],
                                        capture_output=True, text=True, timeout=60, check=False)
        require(compile_result.returncode == 0, "CPU helper compilation failed: " + compile_result.stderr)
        run_result = subprocess.run([str(binary)], input=stdin, capture_output=True, text=True,
                                    timeout=30, check=False)
        require(run_result.returncode == 0 and run_result.stdout.strip() == str(len(all_cases)),
                "CPU helper predicate/flag/layout failure: " + run_result.stderr)
    return {"compiled": True, "gpu_work": False, "predicate_cases": len(all_cases),
            "model_manifest_names_checked": len(source["tensors"]), "strict_flag_cases": 10,
            "canonical_layout_cases": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path,
                        default=ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1/manifest.json")
    parser.add_argument("--full512-manifest", type=Path,
                        default=ROOT / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1/manifest.json")
    parser.add_argument("--skip-compiler", action="store_true", help="run bounded metadata/static checks without compiling C++")
    args = parser.parse_args()
    raw, source = read_metadata(args.source_manifest)
    require(hashlib.sha256(raw).hexdigest() == SOURCE_MANIFEST_SHA256, "source manifest byte lock changed")
    overlay = load_overlay()
    cases = predicate_cases()
    for name, expected in cases:
        require(overlay.target_switch_tensor(name) is expected, f"private predicate mismatch: {name!r}")
    plan = independent_source_plan(source)
    full = check_full512_metadata(args.full512_manifest, source)
    inspection, store_inspection = inspect_without_payload_reads(overlay, args.source_manifest, args.full512_manifest)
    compare_inspection(inspection, plan, source)
    static_checks = check_source_transform(overlay)
    fail_closed_checks = check_fail_closed_metadata(overlay, source, args.full512_manifest)
    cpp_checks = ({"compiled": False, "skipped_by_request": True} if args.skip_compiler
                  else check_cpp_helpers(overlay, cases, source))
    result = {"pass": True, "gpu_work": False, "model_loaded": False, "payload_bytes_read": 0,
              "source_manifest_bytes_read": len(raw), "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
              "source_identity_sha256": SOURCE_IDENTITY, "source_payload_manifest_signature": SOURCE_PAYLOAD_SIGNATURE,
              "predicate_cases": len(cases), "predicate_positive_cases": 432,
              "predicate_negative_cases": len(cases) - 432, "independent_source_plan": plan,
              "full512": full, "overlay_inspection": inspection, "overlay_store_inspection": store_inspection,
              "static_ownership_checks": static_checks, "fail_closed_metadata_checks": fail_closed_checks,
              "cpp_helpers": cpp_checks}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
