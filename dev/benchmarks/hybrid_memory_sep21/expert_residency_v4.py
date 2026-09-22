#!/usr/bin/env python3
"""Metadata census and sealed expert-only original-Q4 residency experiment.

No weight payload is read. New v4 reuses v3 numerical objects/kernels and adds
only a checked existing-owner getter and an opt-in startup composite lease.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FLAG = "SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT"
ALIGNMENT = 16384
GETTER = '''FlashOriginalTextResidencySelection FlashWeights::checkedHybridTargetExpertResidency() const {
  if (!impl_ || !impl_->pleSSD.enabled)
    fail("private hybrid expert residency requires checked PLE SSD owners");
  if (impl_->sourceIdentity != "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e" ||
      impl_->fingerprint != "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0" ||
      impl_->descriptor.layers != 48 || impl_->descriptor.experts != 512 ||
      impl_->descriptor.hiddenSize != 2560 || impl_->pleSSD.nativeWindowCount != 28 ||
      impl_->pleSSD.gpuMappedBytes != 74317889536ULL || impl_->pleSSD.diskTensorCount != 384)
    fail("private hybrid expert residency requires the exact qualified SSD/source geometry");
  // The loader already classifies entire native owners. Restrict that existing
  // selection to owners required by target Q4 expert views; legacy whole-base
  // text residency remains a separate, unchanged policy.
  const auto &eligible = impl_->originalTextResidency;
  if (eligible.buffers.size() != 27 || eligible.paths.size() != 27 ||
      eligible.mappedBytes != 71294320640ULL || eligible.excludedBaseCount != 1 ||
      eligible.excludedMappedBytes != 3023568896ULL || eligible.excludedPLEBaseCount != 1 ||
      eligible.excludedVisionBaseCount != 1 || eligible.excludedUnknownBaseCount != 0)
    fail("private hybrid expert residency owner classification changed");
  std::vector<bool> selected(eligible.buffers.size(), false);
  for (const auto &base : eligible.buffers)
    if (!base || base.storage() != metal::BufferStorage::Shared || !base.contents() || !base.sizeBytes())
      fail("private hybrid expert residency has an invalid immutable owner");
  uint64_t logicalBytes = 0, tensorViews = 0;
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const auto leading = "language_model.model.layers." + std::to_string(layer) + ".mlp.switch_mlp.";
    for (uint32_t role = 0; role < 3; ++role) {
      const auto &p = projection(leading + (role == 0 ? "gate_proj" : role == 1 ? "up_proj" : "down_proj"));
      const uint32_t n = role == 2 ? 2560 : 640, k = role == 2 ? 640 : 2560;
      if (p.experts != 512 || p.outputSize != n || p.inputSize != k || p.bits != 4 || p.groupSize != 64 ||
          !p.weights || !p.scales || !p.biases || p.weights->dtype != FlashDType::U32 ||
          p.scales->dtype != FlashDType::BF16 || p.biases->dtype != FlashDType::BF16 ||
          p.weightRowStrideBytes != k / 2 || p.weightExpertStrideBytes != uint64_t{n} * k / 2 ||
          p.parameterRowStrideBytes != k / 32 || p.parameterExpertStrideBytes != uint64_t{n} * k / 32)
        fail("private hybrid expert residency requires unchanged E512 Q4/G64 projections");
      const std::array<const FlashTensor *, 3> planes{p.weights, p.scales, p.biases};
      for (uint32_t plane = 0; plane < 3; ++plane) {
        const auto &tensor = *planes[plane];
        const auto &view = tensor.buffer;
        const uint64_t expected = uint64_t{512} * n * k / (plane == 0 ? 2 : 32);
        if (tensor.logicalBytes != expected || !view || view.storage() != metal::BufferStorage::Shared ||
            !view.contents() || view.sizeBytes() < expected)
          fail("private hybrid expert residency source view extent changed");
        const auto address = reinterpret_cast<uintptr_t>(view.contents());
        uint32_t matches = 0;
        size_t owner = 0;
        for (size_t index = 0; index < eligible.buffers.size(); ++index) {
          const auto &base = eligible.buffers[index];
          const auto begin = reinterpret_cast<uintptr_t>(base.contents());
          if (address < begin) continue;
          const uint64_t offset = address - begin;
          if (offset <= base.sizeBytes() && view.sizeBytes() <= base.sizeBytes() - offset) {
            ++matches; owner = index;
          }
        }
        if (matches != 1)
          fail("private hybrid expert residency view has missing, partial or ambiguous qualified ownership");
        selected[owner] = true;
        if (logicalBytes > UINT64_MAX - tensor.logicalBytes)
          fail("private hybrid expert residency logical extent overflows");
        logicalBytes += tensor.logicalBytes; ++tensorViews;
      }
    }
  }
  if (tensorViews != 432 || logicalBytes != 67947724800ULL)
    fail("private hybrid expert residency target plane census changed");
  FlashOriginalTextResidencySelection result;
  for (size_t index = 0; index < selected.size(); ++index) if (selected[index]) {
    const auto &base = eligible.buffers[index];
    if (result.mappedBytes > UINT64_MAX - base.sizeBytes())
      fail("private hybrid expert residency owner sum overflows");
    result.mappedBytes += base.sizeBytes();
    result.buffers.push_back(base); result.paths.push_back(eligible.paths[index]);
  }
  // The additional whole-owner bytes are trained MTP expert planes sharing
  // shard 21. No second mapping, payload scan or weight backing is created.
  if (result.buffers.size() != 25 || result.paths.size() != 25 || result.mappedBytes != 69363302400ULL)
    fail("private hybrid expert residency requires exact 25 existing expert owners");
  result.excludedBaseCount = 3;
  result.excludedMappedBytes = 4954587136ULL;
  result.excludedPLEBaseCount = 1; result.excludedVisionBaseCount = 1;
  return result;
}
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rounded(n: int) -> int:
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def census() -> dict:
    manifest_path = ROOT / "install/local-models/Flash-Next-oQ4e-mtp-v1/manifest.json"
    metadata = json.loads(manifest_path.read_text())
    tensors = metadata["tensors"]
    windows = []
    disk_views = 0
    disk_bytes = 0
    for shard in metadata["shards"]:
        rel = shard["path"]
        cursor = 0
        shard_windows = []
        entries = sorted(((name, t) for name, t in tensors.items() if t["shard"] == rel), key=lambda e: e[1]["offset"])
        for name, t in entries:
            begin, end = t["offset"], t["offset"] + t["length"]
            assert begin == rounded(cursor)
            padded_end = rounded(end)
            disk = name.startswith("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.")
            if disk:
                disk_views += 1; disk_bytes += padded_end - begin
            else:
                if shard_windows and shard_windows[-1]["end"] == begin:
                    shard_windows[-1]["end"] = padded_end
                else:
                    shard_windows.append(dict(shard=rel, begin=begin, end=padded_end))
            cursor = end
        assert rounded(cursor) == shard["bytes"]
        windows.extend(shard_windows)
    for w in windows:
        w["bytes"] = w["end"] - w["begin"]
        names = [name for name, t in tensors.items() if t["shard"] == w["shard"] and w["begin"] <= t["offset"] and t["offset"] + t["length"] <= w["end"]]
        target = [name for name in names if name.startswith("language_model.model.layers.") and ".mlp.switch_mlp." in name]
        w["target_expert_tensor_count"] = len(target)
        w["target_expert_logical_bytes"] = sum(tensors[name]["length"] for name in target)
        categories = set("Vision" if name.startswith("vision_tower.") or "vision" in name else "PLE" if ".ple." in name or "ngram" in name else "MTP" if name.startswith("mtp.") else "Text" if name.startswith("language_model.") else "Unknown" for name in names)
        w["categories"] = sorted(categories)
        w["eligible"] = bool(categories) and categories <= {"Text", "MTP"}
    selected = [w for w in windows if w["target_expert_tensor_count"]]
    assert len(windows) == 28 and sum(w["bytes"] for w in windows) == 74317889536
    assert disk_views == 384 and disk_bytes == 32002539520
    assert len(selected) == 25 and all(w["eligible"] for w in selected)
    mapped = sum(w["bytes"] for w in selected)
    logical = sum(w["target_expert_logical_bytes"] for w in selected)
    assert mapped == 69363302400 and logical == 67947724800
    assert sum(w["target_expert_tensor_count"] for w in selected) == 432
    report_path = ROOT / "build/release/flash/sep21-hybrid-q4-decode-i8-prefill-b1-v3.json"
    report = json.loads(report_path.read_text())
    initial = report["server_runs"][0]["initial_status"]
    host = initial["memory_governor"]
    headroom = host["host_available_bytes"] - host["host_reserve_bytes"]
    required = mapped + (2 << 30)
    saved = initial["saved_operands_residency"]
    return dict(schema="splash-hybrid-expert-only-original-q4-residency-metadata-census-v1",
                gpu_execution=False, model_payload_bytes_read=0, window_count=28,
                selected_owner_count=25, selected_owner_bytes=mapped,
                target_tensor_view_count=432, target_logical_bytes=logical,
                co_owned_trained_mtp_bytes=mapped-logical,
                saved_owner_count=saved["registered_base_allocation_count"],
                saved_owner_bytes=saved["registered_base_allocation_bytes"],
                composite_owner_count=saved["registered_base_allocation_count"]+25,
                composite_owner_bytes=saved["registered_base_allocation_bytes"]+mapped,
                existing_host_policy_margin_bytes=2 << 30,
                retained_initial_host_headroom_bytes=headroom, required_initial_host_headroom_bytes=required,
                retained_initial_policy_passes=headroom >= required,
                retained_initial_policy_spare_bytes=headroom-required,
                fresh_runtime_governor_snapshot_authoritative=True, new_weight_backing_bytes=0,
                selected_owners=selected,
                provenance=[dict(path=str(p), sha256=sha(p.read_bytes())) for p in (manifest_path, report_path)])


def replace(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"v4 source anchor changed: {before!r}")
    return text.replace(before, after, 1)


def worker(text: str) -> str:
    text = replace(text, "  std::string originalFailureReason;\n", """  std::string originalFailureReason;
  bool hybridExpertRequested = false, hybridExpertAdded = false;
  FlashOriginalTextResidencySelection hybridExpertSelection;
  engine::MemoryGovernorSnapshot hybridExpertHost{};
  std::string hybridExpertFailureReason;
""")
    text = replace(text, '      const bool savedResidencyRequested = environmentSwitch("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT");',
                   '      const bool hybridExpertResidencyRequested = environmentSwitch("SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT");\n      const bool savedResidencyRequested = environmentSwitch("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT");')
    text = replace(text, '      for (const char *name : {"SPLASH_FLASH_BLOCKED_MOE",',
                   '''      if (hybridExpertResidencyRequested && !savedResidencyRequested)
        throw std::invalid_argument("private hybrid Q4 expert residency requires saved composite operands");
      for (const char *name : {"SPLASH_FLASH_BLOCKED_MOE",''')
    text = replace(text, "      savedResidency.originalRequested = originalTextResidencyRequested;", """      savedResidency.originalRequested = originalTextResidencyRequested;
      savedResidency.hybridExpertRequested = hybridExpertResidencyRequested;""")
    add = '''        if (hybridExpertResidencyRequested) {
          savedResidency.hybridExpertSelection = weights.checkedHybridTargetExpertResidency();
          governor.setPressure(pressure.value());
          savedResidency.hybridExpertHost = governor.snapshot();
          const auto &host = savedResidency.hybridExpertHost;
          if (flashOriginalResidencyHostAllowed(host.hostMeasurementValid, host.growthAllowed,
              host.pressure == engine::MemoryPressure::Normal && host.systemPressure == engine::MemoryPressure::Normal,
              host.hostHeadroomBytes, savedResidency.hybridExpertSelection.mappedBytes)) {
            operands.insert(operands.end(), savedResidency.hybridExpertSelection.buffers.begin(),
                savedResidency.hybridExpertSelection.buffers.end());
            savedResidency.hybridExpertAdded = true;
          } else {
            savedResidency.hybridExpertFailureReason = "host reserve protected; existing expert owners require mapped bytes plus unchanged 2 GiB headroom";
          }
        }
'''
    text = replace(text, "        if (originalTextResidencyRequested) {", add + "        if (originalTextResidencyRequested) {")
    text = replace(text, '                savedResidency.originalAdded ? "Splash verified derived and pure-text original operands"',
                   '''                savedResidency.hybridExpertAdded ? "Splash private composite saved I8 and existing original Q4 expert owners" :
                savedResidency.originalAdded ? "Splash verified derived and pure-text original operands"''')
    text = replace(text, "        if (savedResidency.originalAdded && !savedResidencyLease)", """        if (savedResidency.hybridExpertAdded && savedResidencyLease &&
            (savedResidencyLease.bufferCount() != 748 || savedResidencyLease.byteCount() != 202252746752ULL))
          throw std::logic_error("private hybrid composite residency owner census changed");
        if (savedResidency.hybridExpertAdded && !savedResidencyLease)
          savedResidency.hybridExpertFailureReason = savedResidency.failureReason;
        if (savedResidency.originalAdded && !savedResidencyLease)""")
    text = replace(text, '      << R"(,"target_numerical_derivative_sha256":)"',
                   '      << R"(,"hybrid_q4_expert_residency_requested":)" << (savedResidency_.hybridExpertRequested ? "true" : "false")\n      << R"(,"target_numerical_derivative_sha256":)"')
    text = replace(text, '      << R"(,"scope":)" << json::quote(savedResidency_.originalAdded',
                   '''      << R"(,"scope":)" << json::quote(savedResidency_.hybridExpertAdded
          ? "saved I8/dense operands plus 25 existing SSD owners required by original Q4 target experts; co-owned trained MTP included; PLE/vision/unknown owners excluded" : savedResidency_.originalAdded''')
    status = '''      << R"(,"hybrid_q4_expert_residency":{"requested":)" << (savedResidency_.hybridExpertRequested ? "true" : "false")
      << R"(,"added_to_composite":)" << (savedResidency_.hybridExpertAdded ? "true" : "false")
      << R"(,"active":)" << (savedResidency_.hybridExpertAdded && savedResidencyLease_ && healthy && !transport_.stopping() ? "true" : "false")
      << R"(,"selected_owner_count":)" << savedResidency_.hybridExpertSelection.buffers.size()
      << R"(,"selected_owner_bytes":)" << savedResidency_.hybridExpertSelection.mappedBytes
      << R"(,"target_tensor_view_count":432,"target_logical_bytes":67947724800,"co_owned_trained_mtp_bytes":1415577600,"new_weight_backing_bytes":0)"
      << R"(,"registered_composite_owner_count":)" << savedResidencyLease_.bufferCount()
      << R"(,"registered_composite_owner_bytes":)" << savedResidencyLease_.byteCount()
      << R"(,"host_measurement_valid":)" << (savedResidency_.hybridExpertHost.hostMeasurementValid ? "true" : "false")
      << R"(,"host_available_bytes":)" << savedResidency_.hybridExpertHost.hostAvailableBytes
      << R"(,"host_reserve_bytes":)" << savedResidency_.hybridExpertHost.hostReserveBytes
      << R"(,"host_headroom_bytes":)" << savedResidency_.hybridExpertHost.hostHeadroomBytes
      << R"(,"required_host_headroom_bytes":)" << (savedResidency_.hybridExpertRequested ? 71510786048ULL : 0)
      << R"(,"failure_reason":)" << json::quote(savedResidency_.hybridExpertFailureReason)
      << R"(,"physical_pinning_verified":false,"performance_hypothesis":"reduce phase-dependent rewiring; unproven"})"
'''
    text = replace(text, '      << R"(,"original_text_residency":{"requested":)"', status + '      << R"(,"original_text_residency":{"requested":)"')
    return text


def prepare(base: Path, output: Path) -> dict:
    plan = census()
    assert plan["retained_initial_policy_passes"]
    if output.exists() or ROOT / "build" not in output.resolve().parents:
        raise ValueError("Choose a fresh private v4 snapshot")
    parent_bytes = (base / "overlay-manifest.json").read_bytes()
    parent = json.loads(parent_bytes)
    output.mkdir(parents=True)
    changed_paths = []
    records = []
    for e in parent["files"]:
        rel = e["path"]
        data = (base / "source" / rel).read_bytes()
        assert sha(data) == e["overlay_sha256"], rel
        text = data.decode()
        if rel == "runtime/flash/FlashWeights.hpp":
            text = replace(text, "  [[nodiscard]] FlashOriginalTextResidencySelection checkedOriginalTextResidency() const;",
                           "  [[nodiscard]] FlashOriginalTextResidencySelection checkedOriginalTextResidency() const;\n  // Private exact SSD target-expert existing-owner selection; no additional backing.\n  [[nodiscard]] FlashOriginalTextResidencySelection checkedHybridTargetExpertResidency() const;")
        elif rel == "runtime/flash/FlashWeights.mm":
            text = replace(text, "FlashOriginalTextResidencySelection FlashWeights::checkedOriginalTextResidency() const {", GETTER + "FlashOriginalTextResidencySelection FlashWeights::checkedOriginalTextResidency() const {")
        elif rel == "runtime/flash/FlashWorker.mm":
            text = worker(text)
        changed = text.encode()
        if changed != data:
            changed_paths.append(rel)
        destination = output / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(changed)
        records.append({**e, "v3_source_sha256": sha(data), "expert_residency_changed": changed != data, "overlay_sha256": sha(changed)})
    assert sorted(changed_paths) == ["runtime/flash/FlashWeights.hpp", "runtime/flash/FlashWeights.mm", "runtime/flash/FlashWorker.mm"]
    copied = []
    for e in parent["frozen_link_inputs"]:
        data = (base / e["private_path"]).read_bytes()
        assert sha(data) == e["sha256"]
        dest = output / e["private_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        copied.append(e)
    objects = []
    for path in sorted((base / "host").glob("*.o")):
        if path.name in {"FlashWorker.o", "FlashWeights.o"}:
            continue
        data = path.read_bytes(); dest = output / "host" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
        objects.append(dict(path=str(dest.relative_to(output)), sha256=sha(data), source=str(path)))
    binaries = []
    for name in ("splash.metallib", "policy-cpu", "prefill4k-attribution"):
        path = base / name; data = path.read_bytes(); dest = output / name
        dest.write_bytes(data); dest.chmod(path.stat().st_mode & 0o777)
        binaries.append(dict(path=name, source=str(path), sha256=sha(data)))
    (output / "link-inputs.mk").write_bytes((base / "link-inputs.mk").read_bytes())
    profile = json.loads((base / "environment.json").read_text()); profile[FLAG] = "1"
    (output / "environment.json").write_text(json.dumps(profile, indent=2) + "\n")
    (output / "expert-residency-metadata-census.json").write_text(json.dumps(plan, indent=2) + "\n")
    manifest = {**parent, "route": parent["route"] + "-expert-residency-composite-v4",
                "v3_source_parent": str(base), "v3_source_parent_manifest_sha256": sha(parent_bytes),
                "files": records, "frozen_link_inputs": copied,
                "hash_matched_reused_host_objects": objects, "hash_matched_reused_binaries": binaries,
                "expert_residency_changed_sources": changed_paths, "expert_residency_opt_in_flag": FLAG,
                "expert_residency_metadata_census": plan, "fixed_profile": profile,
                "numerical_policy_changed": False, "new_weight_backing_bytes": 0}
    (output / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    witness = json.loads((base / "cpu-source-witness.json").read_text())
    witness["expert_residency_composite"] = dict(metadata_plan=plan, changed_sources=changed_paths,
                                                original_loader_legacy_guard_unchanged=True,
                                                numerical_policy_changed=False, new_weight_backing_bytes=0)
    (output / "cpu-source-witness.json").write_text(json.dumps(witness, indent=2) + "\n")
    return dict(prepared=str(output), gpu_execution=False, model_payload_bytes_read=0,
                selected_owner_count=25, selected_owner_bytes=plan["selected_owner_bytes"],
                composite_owner_count=748, composite_owner_bytes=plan["composite_owner_bytes"],
                numerical_identity_sha256=parent["target_numerical_derivative_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/hybrid-q4-i8-fixed-r4-sep21-v3")
    parser.add_argument("--output", type=Path, default=ROOT / "build/hybrid-q4-i8-expert-residency-sep21-v4")
    parser.add_argument("--census-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(census() if args.census_only else prepare(args.base.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
