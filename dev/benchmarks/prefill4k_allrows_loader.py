#!/usr/bin/env python3
"""Private all-row Full512 target loader transform; no model/GPU execution.

The checked original target switch coefficients remain on disk. Their existing
FlashTensor/FlashAffineProjection entries retain immutable shape/stride metadata
but no native buffer, and ordinary GPU lookup fails closed. Trained MTP and
every other original role keep the existing PLE-SSD native-window ownership.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT = 16384
TARGET_SOURCE_IDENTITY = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e"
TARGET_MANIFEST_SHA256 = "0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402"
TARGET_FINGERPRINT = "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0"
TARGET_FULL512_MANIFEST_SHA256 = "ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1"
TARGET_LOGICAL_BYTES = 67947724800
GPU_MAPPED_BYTES = 6370164736
DISK_PAYLOAD_BYTES = 99950264320
TARGET_NAME = re.compile(
    r"language_model\.model\.layers\.(?:[0-9]|[1-3][0-9]|4[0-7])"
    r"\.mlp\.switch_mlp\.(?:gate_proj|up_proj|down_proj)\.(?:weight|scales|biases)\Z"
)
PLE_NAME = re.compile(
    r"language_model\.model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\.shards\."
    r"(?:[0-9]|[1-9][0-9]|1[01][0-9]|12[0-7])\.(?:weight|scales|biases)\Z"
)

# Used verbatim by both the private Objective-C++ loader and its CPU-only test.
CPP_HELPERS = r'''// Private omission is exact target E512 switch storage only. In particular,
// trained mtp.layers.0 banks and shared experts never satisfy this predicate.
bool flashAllRowsTargetSwitchTensor(std::string_view name) {
  constexpr std::string_view prefix = "language_model.model.layers.";
  if (!name.starts_with(prefix)) return false;
  name.remove_prefix(prefix.size());
  const auto dot = name.find('.');
  if (dot == std::string_view::npos || dot == 0) return false;
  const auto layer = name.substr(0, dot);
  if (layer.size() > 1 && layer.front() == '0') return false;
  uint32_t index = 0;
  for (char digit : layer) {
    if (digit < '0' || digit > '9' || index > 47) return false;
    index = index * 10 + static_cast<uint32_t>(digit - '0');
  }
  if (index >= 48) return false;
  name.remove_prefix(dot);
  constexpr std::string_view switchPrefix = ".mlp.switch_mlp.";
  if (!name.starts_with(switchPrefix)) return false;
  name.remove_prefix(switchPrefix.size());
  const auto planeDot = name.find('.');
  if (planeDot == std::string_view::npos) return false;
  const auto plane = name.substr(0, planeDot);
  const auto suffix = name.substr(planeDot);
  return (plane == "gate_proj" || plane == "up_proj" || plane == "down_proj") &&
      (suffix == ".weight" || suffix == ".scales" || suffix == ".biases");
}

bool flashAllRowsTargetDiskOnlyTensor(std::string_view name) {
  return flashPLESSDTableTensor(name) || flashAllRowsTargetSwitchTensor(name);
}

bool flashAllRowsFull512TargetValue(const char *value) {
  if (!value || std::string_view(value) == "0") return false;
  if (std::string_view(value) == "1") return true;
  throw std::invalid_argument("PRIVATE SPLASH_FLASH_ALLROWS_FULL512_TARGET must be 0 or 1");
}

'''


def target_switch_tensor(name: str) -> bool:
    return isinstance(name, str) and TARGET_NAME.fullmatch(name) is not None


def ple_tensor(name: str) -> bool:
    return isinstance(name, str) and PLE_NAME.fullmatch(name) is not None


def replacement(text: str, before: str, after: str, expected: int = 1) -> str:
    actual = text.count(before)
    if actual != expected:
        raise RuntimeError(f"Private all-row loader source drift: expected {expected}, got {actual}: {before!r}")
    return text.replace(before, after)


def _bounded_json(path: Path, limit: int) -> tuple[bytes, dict]:
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Metadata file exceeds the bounded loader inspection size")
    raw = path.read_bytes()
    def unique_pairs(pairs):
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("Duplicate metadata key")
            record[key] = value
        return record
    record = json.loads(raw, object_pairs_hook=unique_pairs)
    if not isinstance(record, dict):
        raise ValueError("Metadata root must be an object")
    return raw, record


def inspect_source_manifest(path: Path) -> dict:
    """Inspect only the locked 1 MiB metadata; never open source payloads."""
    raw, source = _bounded_json(Path(path), 32 << 20)
    if (hashlib.sha256(raw).hexdigest() != TARGET_MANIFEST_SHA256
            or source.get("schema") != "splash-local-qwen4-affine-v1"
            or source.get("source_identity_sha256") != TARGET_SOURCE_IDENTITY
            or source.get("alignment") != ALIGNMENT):
        raise ValueError("All-row omission requires the exact qualified source and manifest")
    tensors, shards = source["tensors"], source["shards"]
    if len(tensors) != 3748 or len(shards) != 21:
        raise ValueError("Original inventory changed")
    selected = {name: record for name, record in tensors.items() if target_switch_tensor(name)}
    expected = {
        f"language_model.model.layers.{layer}.mlp.switch_mlp.{plane}.{suffix}"
        for layer in range(48) for plane in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "scales", "biases")
    }
    if set(selected) != expected:
        raise ValueError("Target omission inventory must contain exactly 432 checked tensors")
    for name, record in selected.items():
        down = ".down_proj." in name
        n, k = (2560, 640) if down else (640, 2560)
        weight = name.endswith(".weight")
        expected_shape = [512, n, k // (8 if weight else 64)]
        expected_bytes = 512 * n * k // (2 if weight else 32)
        if record["dtype"] != ("U32" if weight else "BF16") or record["shape"] != expected_shape or record["length"] != expected_bytes:
            raise ValueError("Original target Q4/G64 coefficient metadata changed")
    windows, disk_count, disk_logical, disk_payload, total_payload, fully_disk = [], 0, 0, 0, 0, 0
    for shard in shards:
        file_bytes, shard_name = shard["bytes"], shard["path"]
        total_payload += file_bytes
        cursor, shard_windows = 0, []
        records = sorted((record["offset"], record["offset"] + record["length"], name)
                         for name, record in tensors.items() if record["shard"] == shard_name)
        if not records or not file_bytes or file_bytes % ALIGNMENT:
            raise ValueError("Empty or unaligned original shard")
        for begin, end, name in records:
            padded_end = (end + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
            if begin != ((cursor + ALIGNMENT - 1) & ~(ALIGNMENT - 1)) or end <= begin or padded_end > file_bytes:
                raise ValueError("Original canonical tensor partition changed")
            if target_switch_tensor(name) or ple_tensor(name):
                disk_count += 1
                disk_logical += end - begin
                disk_payload += padded_end - begin
            elif shard_windows and shard_windows[-1]["end"] == begin:
                shard_windows[-1]["end"] = padded_end
            else:
                shard_windows.append({"path": shard_name, "begin": begin, "end": padded_end})
            cursor = end
        if ((cursor + ALIGNMENT - 1) & ~(ALIGNMENT - 1)) != file_bytes:
            raise ValueError("Original canonical tail changed")
        fully_disk += not shard_windows
        windows.extend(shard_windows)
    target_logical = sum(record["length"] for record in selected.values())
    target_payload = sum(((record["offset"] + record["length"] + ALIGNMENT - 1) & ~(ALIGNMENT - 1)) - record["offset"] for record in selected.values())
    gpu_bytes = sum(window["end"] - window["begin"] for window in windows)
    if (target_logical != TARGET_LOGICAL_BYTES or target_payload != TARGET_LOGICAL_BYTES
            or gpu_bytes != GPU_MAPPED_BYTES or disk_payload != DISK_PAYLOAD_BYTES
            or disk_count != 816 or len(windows) != 7 or fully_disk != 19
            or total_payload != 106320429056 or gpu_bytes + disk_payload != total_payload):
        raise ValueError("Private all-row window geometry changed")
    skip_records = [{"name": name, **record} for name, record in sorted(selected.items())]
    signature = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": "prefill4k-private-allrows-original-omission-v1", "gpu_work": False,
        "source_identity": TARGET_SOURCE_IDENTITY, "source_manifest_sha256": TARGET_MANIFEST_SHA256,
        "target_tensor_count": len(selected), "target_projection_count": len(selected) // 3,
        "target_logical_bytes": target_logical, "target_payload_bytes": target_payload,
        "disk_tensor_count": disk_count, "disk_projection_count": disk_count // 3,
        "disk_logical_bytes": disk_logical, "disk_payload_bytes": disk_payload,
        "original_payload_bytes": total_payload, "gpu_mapped_bytes": gpu_bytes,
        "native_window_count": len(windows), "fully_disk_payload_count": fully_disk,
        "window_signature_sha256": signature(windows), "skip_signature_sha256": signature(skip_records),
        "windows": windows,
    }


def inspect_full512_manifest(path: Path) -> dict:
    """Bounded certified-store metadata lock; no payload reads or hashing."""
    raw, store = _bounded_json(Path(path), 8 << 20)
    if (hashlib.sha256(raw).hexdigest() != TARGET_FULL512_MANIFEST_SHA256
            or store.get("source_identity_sha256") != TARGET_SOURCE_IDENTITY
            or store.get("source_manifest_sha256") != TARGET_MANIFEST_SHA256
            or store.get("total_bytes") != 121173442560
            or store.get("planned_allocation_bytes") != 121174228992
            or len(store.get("layers", [])) != 48
            or store.get("selected_experts") != [list(range(512)) for _ in range(48)]):
        raise ValueError("Original omission requires the exact certified all512 store metadata")
    return {"source_identity": TARGET_SOURCE_IDENTITY, "source_manifest_sha256": TARGET_MANIFEST_SHA256,
            "store_manifest_sha256": TARGET_FULL512_MANIFEST_SHA256,
            "target_layers": 48, "experts_per_layer": 512, "gpu_work": False}


def transform(relative: str, text: str) -> str:
    if relative == "runtime/flash/FlashWeights.hpp":
        text = replacement(text, "  bool enabled = false;\n  uint64_t originalPayloadBytes", """  bool enabled = false;
  // Private all-row target alternative: omitted original coefficients remain
  // checked disk metadata, never mapped GPU source operands.
  bool allRowsFull512Target = false;
  uint64_t targetOriginalDiskTensorCount = 0;
  uint64_t targetOriginalDiskProjectionCount = 0;
  uint64_t targetOriginalDiskLogicalBytes = 0;
  uint64_t targetOriginalDiskPayloadBytes = 0;
  std::string targetFull512StoreManifestSha256;
  uint64_t originalPayloadBytes""")
        text = replacement(text, "  [[nodiscard]] bool contains(std::string_view name) const noexcept;", """  // Stable original shape/format metadata only; omitted target entries have
  // nil buffers. This method never allocates, maps, or materializes a GPU view.
  [[nodiscard]] const FlashAffineProjection &projectionMetadata(std::string_view prefix) const;
  [[nodiscard]] bool contains(std::string_view name) const noexcept;""")
        return replacement(text, "// Checked original affine bytes retained only on disk in optional PLE SSD", "// Checked original affine bytes retained only on disk in private target or PLE SSD")
    if relative != "runtime/flash/FlashWeights.mm":
        return text
    text = replacement(text, '#include "flash/FlashPLESSDStore.hpp"', '#include "flash/FlashPLESSDStore.hpp"\n#include "flash/FlashInt8ExpertStoreMetadata.hpp"')
    text = replacement(text, "NSString *ns(std::string_view value) {", CPP_HELPERS + "NSString *ns(std::string_view value) {")
    text = replacement(text, '    const std::string sourceIdentity = stringValue(manifest[@"source_identity_sha256"]);\n    checkDigest(sourceIdentity);', '''    const std::string sourceIdentity = stringValue(manifest[@"source_identity_sha256"]);
    checkDigest(sourceIdentity);
    // Private compile only. Refuse omission before any native original buffer
    // creation unless source and the completed certified Full512 metadata match.
    if (!flashAllRowsFull512TargetValue(std::getenv("SPLASH_FLASH_ALLROWS_FULL512_TARGET")))
      fail("PRIVATE all-row target loader requires SPLASH_FLASH_ALLROWS_FULL512_TARGET=1");
    if (!streamingEnabled)
      fail("PRIVATE all-row target loader requires PLE SSD streaming=1");
    if (sourceIdentity != "''' + TARGET_SOURCE_IDENTITY + '''" ||
        manifestDigest != "''' + TARGET_MANIFEST_SHA256 + '''")
      fail("PRIVATE all-row target omission requires the exact checked source/manifest");
    const char *allRowsStorePath = std::getenv("SPLASH_FLASH_INT8_EXPERT_STORE");
    if (!allRowsStorePath || !*allRowsStorePath)
      fail("PRIVATE all-row target omission requires a completed Full512 store");
    const auto allRowsStore = loadFlashInt8ExpertStoreMetadata(allRowsStorePath,
        sourceIdentity, "''' + TARGET_FINGERPRINT + '''", NormConvention::OnePlusWeight);
    if (allRowsStore.identitySha256 != "''' + TARGET_FULL512_MANIFEST_SHA256 + '''" ||
        allRowsStore.sourceManifestSha256 != manifestDigest ||
        allRowsStore.totalBytes != 121173442560ULL || allRowsStore.plannedBytes != 121174228992ULL)
      fail("PRIVATE all-row target omission requires the exact certified Full512 metadata");
    for (const auto &layer : allRowsStore.layers) {
      if (layer.selectedIDs.size() != 512)
        fail("PRIVATE all-row target omission requires every expert in every target layer");
      for (uint32_t expert = 0; expert < 512; ++expert)
        if (layer.selectedIDs[expert] != expert)
          fail("PRIVATE Full512 target rank inventory must be canonical all512");
    }''')
    text = replacement(text, "    impl.pleSSD.enabled = streamingEnabled;", """    impl.pleSSD.enabled = streamingEnabled;
    impl.pleSSD.allRowsFull512Target = true;
    impl.pleSSD.targetFull512StoreManifestSha256 = allRowsStore.identitySha256;""")
    text = replacement(text, '        layoutRanges[shardName].push_back({offset, offset + length, flashPLESSDTableTensor(stringValue(key))});', '''        const std::string tensorName = stringValue(key);
        layoutRanges[shardName].push_back({offset, offset + length, flashAllRowsTargetDiskOnlyTensor(tensorName)});
        if (flashAllRowsTargetSwitchTensor(tensorName))
          impl.pleSSD.targetOriginalDiskPayloadBytes += aligned(offset + length) - offset;''')
    text = replacement(text, "impl.pleSSD.diskTensorCount != 384 || impl.pleSSD.nativeWindowCount != 28 ||\n          impl.pleSSD.fullyDiskPayloadCount != 6 || impl.pleSSD.originalPayloadBytes != 106320429056ULL ||\n          impl.pleSSD.gpuMappedBytes != 74317889536ULL || impl.pleSSD.diskOnlyPayloadBytes != 32002539520ULL", """impl.pleSSD.diskTensorCount != 816 || impl.pleSSD.nativeWindowCount != 7 ||
          impl.pleSSD.fullyDiskPayloadCount != 19 || impl.pleSSD.originalPayloadBytes != 106320429056ULL ||
          impl.pleSSD.gpuMappedBytes != 6370164736ULL || impl.pleSSD.diskOnlyPayloadBytes != 99950264320ULL ||
          impl.pleSSD.targetOriginalDiskPayloadBytes != 67947724800ULL""")
    text = replacement(text, 'fail("PLE SSD mode requires the checked original 128-part payload window geometry");', 'fail("PRIVATE all-row target mode requires the checked PLE-or-target disk-only payload windows");')
    text = replacement(text, "if (impl.pleSSD.enabled && flashPLESSDTableTensor(name))", "if (impl.pleSSD.enabled && flashAllRowsTargetDiskOnlyTensor(name))", expected=2)
    text = replacement(text, "        impl.pleSSD.diskOnlyLogicalBytes += tensor.logicalBytes;", """        impl.pleSSD.diskOnlyLogicalBytes += tensor.logicalBytes;
        if (flashAllRowsTargetSwitchTensor(name)) {
          ++impl.pleSSD.targetOriginalDiskTensorCount;
          impl.pleSSD.targetOriginalDiskLogicalBytes += tensor.logicalBytes;
        }""")
    text = replacement(text, "        impl.diskProjections.emplace(prefix, disk);", """        impl.diskProjections.emplace(prefix, disk);
        if (flashAllRowsTargetSwitchTensor(name))
          ++impl.pleSSD.targetOriginalDiskProjectionCount;""")
    text = replacement(text, "      const auto &value = result.projection(prefix);", "      const auto &value = result.projectionMetadata(prefix);")
    text = replacement(text, '      if (impl.pleSSD.diskProjectionCount != impl.descriptor.pleParts)\n        fail("PLE SSD disk projection inventory is incomplete");', '''      if (impl.pleSSD.diskProjectionCount != impl.descriptor.pleParts + 144 ||
          impl.pleSSD.targetOriginalDiskTensorCount != 432 ||
          impl.pleSSD.targetOriginalDiskProjectionCount != 144 ||
          impl.pleSSD.targetOriginalDiskLogicalBytes != 67947724800ULL)
        fail("PRIVATE disk-only PLE/target projection inventory is incomplete");
      // Exactly432 original target tensors keep nil GPU buffers and stable
      // affine metadata. No omitted target is admitted to the PLE SSD store.
      for (uint32_t layer = 0; layer < 48; ++layer) {
        const std::string target = "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp.";
        for (const auto *projection : {"gate_proj", "up_proj", "down_proj"}) {
          const auto &p = result.projectionMetadata(target + projection);
          if (!p.weights || !p.scales || !p.biases || p.weights->buffer || p.scales->buffer || p.biases->buffer)
            fail("PRIVATE omitted target projection unexpectedly owns a native GPU buffer");
        }
      }''')
    text = replacement(text, 'fail("PLE tensor is disk-only; use diskProjection: " + std::string(name));', 'fail("PRIVATE original tensor is disk-only; GPU fallback/materialization forbidden: " + std::string(name));')
    text = replacement(text, 'fail("PLE projection is disk-only; use diskProjection: " + std::string(prefix));', 'fail("PRIVATE original projection is disk-only; GPU fallback/materialization forbidden: " + std::string(prefix));')
    text = replacement(text, "bool FlashWeights::contains(std::string_view name) const noexcept {", """const FlashAffineProjection &FlashWeights::projectionMetadata(std::string_view prefix) const {
  if (!impl_) fail("weights are not loaded");
  const auto found = impl_->projections.find(prefix);
  if (found == impl_->projections.end()) fail("missing original projection metadata: " + std::string(prefix));
  return found->second;
}
bool FlashWeights::contains(std::string_view name) const noexcept {""")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1")
    parser.add_argument("--store", type=Path, default=ROOT / "build/prefill4k-fullcache-artifacts/int8-experts-all512-v1")
    args = parser.parse_args()
    report = inspect_source_manifest(args.package / "manifest.json")
    report["full512_metadata"] = inspect_full512_manifest(args.store / "manifest.json")
    # This entry point is inspection only. The owning generator calls transform
    # and writes private copies; this module cannot overwrite runtime sources.
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
