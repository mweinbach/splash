#!/usr/bin/env python3
"""Seal a CPU-only batch teacher cache-prefix snapshot over the batch SG8 tree."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[3]
FLAG = "SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY"
SEMANTICS = "mtp-batch-teacher-cache-only-compact-real-lanes-original-qsa-cache-prefix-no-attention-or-mlp-v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"Sealed batch teacher source anchor drift: {before[:100]}")
    return text.replace(before, after, 1)


def owner_header(text: str) -> str:
    anchor = "  [[nodiscard]] bool ownsState(const FlashMTPState &state) const noexcept;"
    return once(text, anchor, """  void batchAddTeacherQSACache(metal::CommandGraph &graph, const FlashQSAFastInputs &inputs,
      FlashQSAState &state, FlashQSAWorkspace &workspace,
      FlashQSAFastWorkspace &fastWorkspace, uint32_t begin, uint32_t rows);
""" + anchor)


def owner_source(text: str) -> str:
    anchor = "bool FlashMTPForward::ownsState(const FlashMTPState &state) const noexcept {"
    return once(text, anchor, """void FlashMTPForward::batchAddTeacherQSACache(metal::CommandGraph &graph,
    const FlashQSAFastInputs &inputs, FlashQSAState &state,
    FlashQSAWorkspace &workspace, FlashQSAFastWorkspace &fastWorkspace,
    uint32_t begin, uint32_t rows) {
  if (!impl_) throw std::logic_error("Flash MTP head is not initialized");
  addTeacherQSACache(graph, inputs, state, workspace, fastWorkspace, begin, rows,
      impl_->attentionPolicy);
}
""" + anchor)


def batch_header(text: str) -> str:
    text = once(text, "namespace splash::flash {", "namespace splash::flash {\n" +
        f'inline constexpr const char *kFlashBatchMTPTeacherCacheSemantics = "{SEMANTICS}";\n' +
        "[[nodiscard]] bool flashPrivateBatchMTPTeacherCacheOnlyEnabled();")
    text = once(text, "  [[nodiscard]] static uint64_t workspacePlannedBytes", """  // Independent compact teacher pairs persist only their original per-lane
  // QSA cache prefix. No head hidden/logits buffer is returned or published.
  [[nodiscard]] metal::CommandTiming primeTeacherCache(
      std::span<FlashMTPState *const> states,
      metal::MetalBuffer previousHiddenBF16,
      std::span<const uint32_t> compactNextTokens,
      std::span<const uint32_t> laneCounts);
  [[nodiscard]] static uint64_t workspacePlannedBytes""")
    return once(text, "  struct Impl;", """  [[nodiscard]] FlashBatchMTPResult forwardImpl(
      std::span<FlashMTPState *const> states,
      metal::MetalBuffer previousHiddenBF16,
      std::span<const uint32_t> compactNextTokens,
      std::span<const uint32_t> laneCounts, FlashMTPLogits logits,
      bool teacherCacheOnly);
  struct Impl;""")


def batch_source(text: str) -> str:
    text = once(text, "struct FlashBatchMTPForward::Impl final {", f"""bool flashPrivateBatchMTPTeacherCacheOnlyEnabled() {{
  if (!enabled("{FLAG}")) return false;
  if (!enabled("SPLASH_FLASH_MTP") || !enabled("SPLASH_FLASH_BATCH_PREFILL") ||
      !enabled("SPLASH_FLASH_BATCH_MTP_PREFILL"))
    throw std::invalid_argument("{FLAG}=1 requires SPLASH_FLASH_MTP=1, SPLASH_FLASH_BATCH_PREFILL=1 and SPLASH_FLASH_BATCH_MTP_PREFILL=1");
  return true;
}}

struct FlashBatchMTPForward::Impl final {{""")
    anchor = """FlashBatchMTPResult FlashBatchMTPForward::forward(std::span<FlashMTPState *const> states,
    metal::MetalBuffer previousHidden, std::span<const uint32_t> nextTokens,
    std::span<const uint32_t> laneCounts, FlashMTPLogits mode) {"""
    text = once(text, anchor, anchor + """
  return forwardImpl(states, std::move(previousHidden), nextTokens, laneCounts, mode, false);
}

metal::CommandTiming FlashBatchMTPForward::primeTeacherCache(
    std::span<FlashMTPState *const> states, metal::MetalBuffer previousHidden,
    std::span<const uint32_t> nextTokens, std::span<const uint32_t> laneCounts) {
  return forwardImpl(states, std::move(previousHidden), nextTokens, laneCounts,
      FlashMTPLogits::None, true).timing;
}

FlashBatchMTPResult FlashBatchMTPForward::forwardImpl(std::span<FlashMTPState *const> states,
    metal::MetalBuffer previousHidden, std::span<const uint32_t> nextTokens,
    std::span<const uint32_t> laneCounts, FlashMTPLogits mode, bool teacherCacheOnly) {""")
    anchor = """    impl_->owner.batchAddQSA(graph, inputs, state.qsa, impl_->qsaWorkspace,
        impl_->qsaFastWorkspace, static_cast<uint32_t>(state.length), count);
  }
  affine(graph, attention + ".o_proj", attentionOutput, branch);"""
    replacement = """    if (teacherCacheOnly)
      impl_->owner.batchAddTeacherQSACache(graph, inputs, state.qsa, impl_->qsaWorkspace,
          impl_->qsaFastWorkspace, static_cast<uint32_t>(state.length), count);
    else
      impl_->owner.batchAddQSA(graph, inputs, state.qsa, impl_->qsaWorkspace,
          impl_->qsaFastWorkspace, static_cast<uint32_t>(state.length), count);
  }
  if (teacherCacheOnly) {
    metal::CommandTiming timing;
    try {
      timing = impl_->backend.submitCommand(graph.dispatches());
      uint32_t status = 0;
      std::memcpy(&status, diag.contents(), sizeof(status));
      if (status) throw std::runtime_error("Flash batch teacher cache sticky diagnostics failed: " + std::to_string(status));
    } catch (...) {
      for (auto *state : states) state->impl_->poisoned = true;
      throw;
    }
    // Publish every lane only after the shared submission and diagnostics pass.
    for (uint32_t lane = 0; lane < states.size(); ++lane)
      states[lane]->impl_->length = lengths[lane];
    return {timing, {}, 0, static_cast<uint32_t>(states.size()), {},
        std::move(offsets), std::move(lengths), {}, 0};
  }
  affine(graph, attention + ".o_proj", attentionOutput, branch);"""
    return once(text, anchor, replacement)


def worker(text: str) -> str:
    text = once(text, '  const bool mtpTeacherCacheOnly_ = environmentSwitch("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY");',
        '  const bool mtpTeacherCacheOnly_ = environmentSwitch("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY");\n' +
        '  const bool batchMTPTeacherCacheOnly_ = flashPrivateBatchMTPTeacherCacheOnlyEnabled();')
    text = once(text, "  uint64_t teacherCachePrimeCalls_ = 0;", """  uint64_t teacherCachePrimeCalls_ = 0;
  uint64_t batchTeacherCachePrimeCalls_ = 0, batchTeacherCachePrimeLanes_ = 0;
  uint64_t batchTeacherCachePrimePairs_ = 0;""")
    text = once(text, "      if (grouped) {\n        const auto primed = batchPrimeHead_->forward(headStates,", """      if (grouped && batchMTPTeacherCacheOnly_) {
        timing = batchPrimeHead_->primeTeacherCache(headStates,
            backend_.view(batchPrimeInput_, 0, uint64_t{totalRows} * kHyper * 2),
            compactTokens, counts);
      } else if (grouped) {
        const auto primed = batchPrimeHead_->forward(headStates,""")
    text = once(text, "      mtpPrime_.add(totalRows, timing, host);\n      if (grouped) {", """      if (grouped && batchMTPTeacherCacheOnly_) {
        ++batchTeacherCachePrimeCalls_;
        batchTeacherCachePrimeLanes_ += selectedLanes.size();
        batchTeacherCachePrimePairs_ += totalRows;
      }
      mtpPrime_.add(totalRows, timing, host);
      if (grouped) {""")
    text = once(text, '      << R"(,"mtp_attention_route":)"', '''      << R"(,"mtp_batch_teacher_priming_route":)" << (batchPrimeHead_ ? json::quote(batchMTPTeacherCacheOnly_
          ? kFlashBatchMTPTeacherCacheSemantics : "mtp-batch-full-forward-none-logits-v1") : "null")
      << R"(,"mtp_attention_route":)"''')
    text = once(text, '      << R"(,"teacher_cache_only_scope":"sequential prompt priming; true grouped head priming retains full None forward")"', '''      << R"(,"teacher_cache_only_scope":"sequential prompt priming")"
      << R"(,"batch_teacher_cache_only_requested":)" << (batchMTPTeacherCacheOnly_ ? "true" : "false")
      << R"(,"batch_teacher_cache_only_priming_calls":)" << batchTeacherCachePrimeCalls_
      << R"(,"batch_teacher_cache_only_completed_lanes":)" << batchTeacherCachePrimeLanes_
      << R"(,"batch_teacher_cache_only_completed_real_pairs":)" << batchTeacherCachePrimePairs_
      << R"(,"batch_teacher_cache_only_scope":"true grouped prompt priming only; independent real-lane QSA cache prefix")"''')
    return once(text, "      validateBatchHeadPrimeFlags(batchMTPPrefillEnabled, mtpEnabled, batchPrefillEnabled);",
        "      validateBatchHeadPrimeFlags(batchMTPPrefillEnabled, mtpEnabled, batchPrefillEnabled);\n" +
        "      (void)flashPrivateBatchMTPTeacherCacheOnlyEnabled(); // Validate before backend/model allocation.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "build/prefill4k-batch-bulk-gathered-sep21-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "build/prefill4k-batch-teacher-gathered-sep21-v1")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    if ROOT / "build" not in output.parents or output == base or output.exists():
        raise ValueError("Choose a fresh isolated build output")
    raw = (base / "overlay-manifest.json").read_bytes()
    parent = json.loads(raw)
    if not parent.get("batch_bulk_composed") or not parent.get("gathered_mpp_composed"):
        raise ValueError("Sealed batch exact SG8/capped gathered MPP base required")
    transforms = {"runtime/flash/FlashMTP.hpp": owner_header,
        "runtime/flash/FlashMTP.cpp": owner_source,
        "runtime/flash/FlashBatchMTPForward.hpp": batch_header,
        "runtime/flash/FlashBatchMTPForward.cpp": batch_source,
        "runtime/flash/FlashWorker.mm": worker}
    records, files, changed = [], {}, []
    for record in parent["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        original = (base / "source" / relative).read_bytes()
        if sha(original) != record["overlay_sha256"]:
            raise ValueError(f"Sealed source drift: {relative}")
        function = transforms.get(str(relative))
        data = function(original.decode()).encode() if function else original
        files[relative] = data
        new_record = dict(record)
        new_record.update({"batch_teacher_input_sha256": sha(original),
            "batch_teacher_changed": data != original, "overlay_sha256": sha(data)})
        if data != original:
            new_record["patched"] = True
            changed.append(str(relative))
        records.append(new_record)
    if set(changed) != set(transforms):
        raise AssertionError("Unexpected changed path set")
    support = ("dev/benchmarks/prefill_batch_teacher_sep21/oracle.mm",
        "dev/benchmarks/prefill_batch_teacher_sep21/policy_cpu.cpp",
        "dev/benchmarks/prefill4k_attribution.mm")
    support_hashes = {}
    for relative in support:
        data = (ROOT / relative).read_bytes()
        files[Path(relative)] = data
        support_hashes[relative] = sha(data)
        records.append({"path": relative, "new_private_file": True,
            "snapshot_support_file": True, "batch_teacher_changed": False,
            "original_sha256": sha(data), "overlay_sha256": sha(data)})
    manifest = copy.deepcopy(parent)
    manifest.update({"route": "private-full512-cappedgathered-batch-exactsg8-and-teacher-cache-prefix-defaultoff-v1",
        "batch_teacher_composed": True, "batch_teacher_base_build": str(base),
        "batch_teacher_input_manifest_sha256": sha(raw), "batch_teacher_flag": FLAG,
        "batch_teacher_flag_default": False, "batch_teacher_semantics": SEMANTICS,
        "batch_teacher_extra_workspace_bytes": 0,
        "batch_teacher_oracle_version": 2,
        "batch_teacher_support_source_hashes": support_hashes,
        "batch_teacher_generator_sha256": sha(Path(__file__).read_bytes()),
        "normal_sources_modified": False, "gpu_executed": False,
        "payload_bytes_read": 0, "batch_teacher_model_quality_qualified": False, "files": records})
    audit = {"schema": "splash-batch-teacher-cache-cpu-source-v1",
        "source_files_hash_verified": len(parent["files"]), "changed_paths": changed,
        "support_source_files_snapshotted": list(support_hashes),
        "extra_workspace_bytes": 0, "workspace_and_state_planners_unchanged": True,
        "new_flag_validated_before_backend_creation": True,
        "singleton_teacher_cache_prefix_extractor_reused": True,
        "full_forward_none_last_all_contract_preserved": True,
        "grouped_initial_validation_preserved_before_state_or_scratch_mutation": True,
        "completion_counters_after_successful_submission_and_lane_metadata_validation": True,
        "trunk_and_batch_main_qsa_bytes_unchanged": all(not r["batch_teacher_changed"]
            for r in records if any(n in r["path"] for n in ("FlashForward", "FlashBatchForward",
                "FlashBatchPrefill", "FlashBatchVerify", "FlashDenseCache", "FlashInt8ExpertStore"))),
        "all_metal_sources_byte_unchanged": all(not r["batch_teacher_changed"] for r in records
            if r["path"].endswith(".metal")), "gpu_executed": False, "payload_bytes_read": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="batch-teacher-cache-", dir=output.parent) as temporary:
        staged = Path(temporary) / "output"
        for relative, data in files.items():
            destination = staged / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        (staged / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staged / "batch-teacher-cpu-source-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        staged.rename(output)
    print(json.dumps({"prepared": str(output), **audit}))


if __name__ == "__main__":
    main()
