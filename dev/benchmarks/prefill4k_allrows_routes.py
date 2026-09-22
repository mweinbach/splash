#!/usr/bin/env python3
"""Strict private all-row Full512 target executor transforms; no GPU work.

Compose after the wide/Full512 transforms. The accompanying loader and Store
transforms must provide metadata-only original expert records and independently
validated all-hit graph construction. This module never rewrites production.
Trained MTP executors are intentionally outside the transformed file set.
"""
from __future__ import annotations

import argparse
from pathlib import Path


FLAG = "SPLASH_FLASH_ALLROWS_FULL512_TARGET"
ROUTE = ";private-allrows-full512-target-m16-below256-v1"
MARKER = "// Private all-row Full512 target executor overlay v1."
RELATIVES = (
    "runtime/flash/FlashForward.hpp",
    "runtime/flash/FlashForward.cpp",
    "runtime/flash/FlashBatchForward.cpp",
    "runtime/flash/FlashBatchVerify.cpp",
    "runtime/flash/FlashBatchPrefill.cpp",
)


def replacement(text: str, before: str, after: str, count: int = 1) -> str:
    actual = text.count(before)
    if actual != count:
        raise RuntimeError(
            f"Private all-row route source drift: expected {count}, got {actual}: {before!r}"
        )
    return text.replace(before, after)


def _batch_transform(relative: str, text: str) -> str:
    verify = relative.endswith("FlashBatchVerify.cpp")
    extent = "maximumLanes * maximumRows" if verify else "maximumLanes"
    actual = "flattened" if verify else "lanes"
    text = replacement(
        text,
        '#include "flash/FlashMoE.hpp"',
        '#include "flash/FlashMoE.hpp"\n#include "flash/FlashMoEBlocked.hpp"\n'
        '#include "flash/FlashInt8ExpertStore.hpp"',
    )
    text = replacement(
        text,
        "  FlashQSAFastWorkspace qsaFast;",
        "  FlashQSAFastWorkspace qsaFast;\n"
        "  const bool allRowsInt8Target = trunk.allRowsInt8TargetEnabled();\n"
        "  const FlashInt8ExpertStore *int8ExpertStore = nullptr;\n"
        "  FlashMoEBlockedScratch blockedScratch;",
    )
    guard = f'''    if (fusionEnabled("{FLAG}") != allRowsInt8Target)
      throw std::invalid_argument("private all-row Full512 flag changed after source trunk construction");
    if (allRowsInt8Target) {{
      int8ExpertStore = trunk.batchInt8ExpertStore();
      if (!int8ExpertStore)
        throw std::invalid_argument("private all-row target requires the source Full512 Store");
      for (uint32_t layer = 0; layer < 48; ++layer) {{
        const auto ids = int8ExpertStore->selectedExpertIDs(layer);
        if (ids.size() != 512)
          throw std::invalid_argument("private all-row target requires all 512 experts in every layer");
        for (uint32_t expert = 0; expert < 512; ++expert)
          if (ids[expert] != expert)
            throw std::invalid_argument("private all-row target requires canonical complete expert IDs");
      }}
    }}
'''
    text = replacement(
        text,
        "    const uint64_t before = backend.memoryStats().allocatedBytes;",
        guard + "    const uint64_t before = backend.memoryStats().allocatedBytes;\n"
        "    if (allRowsInt8Target)\n"
        f"      blockedScratch = allocateMoEBlockedScratch(backend, {extent}, kSelections);",
    )
    baseline = f'''    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.gate_proj"),
        expertIDs, bf(Slot::ExpertGate, kSelections * 640), diag, {actual}, kSelections);
    addGatheredAffine(graph, mixed, impl_->weights.projection(mlp + ".switch_mlp.up_proj"),
        expertIDs, bf(Slot::ExpertUp, kSelections * 640), diag, {actual}, kSelections);
    addSiLUMultiply(graph, bf(Slot::ExpertGate, kSelections * 640), bf(Slot::ExpertUp, kSelections * 640),
        bf(Slot::Intermediate, kSelections * 640), diag, {actual}, 640, kSelections);
    addGatheredAffine(graph, bf(Slot::Intermediate, kSelections * 640),
        impl_->weights.projection(mlp + ".switch_mlp.down_proj"), expertIDs,
        bf(Slot::ExpertDown, kSelections * kWidth), diag, {actual}, kSelections, true);'''
    hit = f'''    if (impl_->allRowsInt8Target) {{
      constexpr auto tile = FlashMoEBlockedTile::M16N64;
      addMoEBlockedPack(graph, mixed, expertIDs, impl_->blockedScratch, diag, {actual}, tile, kSelections);
      impl_->int8ExpertStore->addGateUp(graph, layer, impl_->blockedScratch, diag, {actual}, tile, kSelections);
      impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, {actual}, tile, kSelections);
    }} else {{
{baseline}
    }}'''
    text = replacement(text, baseline, hit)
    text = replacement(
        text,
        "    addCombine(graph, bf(Slot::ExpertDown, kSelections * kWidth), expertIDs, routes,",
        "    addCombine(graph, impl_->allRowsInt8Target ? impl_->blockedScratch.scatteredDown :\n"
        "        bf(Slot::ExpertDown, kSelections * kWidth), expertIDs, routes,",
    )
    planned_marker = (
        "    total += FlashPLESSD::plannedBytes(maximumLanes, maximumRows);\n  return total;"
        if verify
        else "    total += FlashPLESSD::plannedBytes(maximumLanes, 1);\n  return total;"
    )
    text = replacement(
        text,
        planned_marker,
        planned_marker.replace(
            "  return total;",
            f'  if (fusionEnabled("{FLAG}"))\n'
            f"    total += flashMoEBlockedWorkspacePlannedBytes({extent}, kSelections, kAlignment);\n"
            "  return total;",
        ),
    )
    return text


def _transform(relative: str, text: str) -> str:
    """Transform only private copies, failing closed when source anchors drift."""
    if relative == "runtime/flash/FlashForward.hpp":
        return replacement(
            text,
            "  [[nodiscard]] const FlashInt8ExpertStore *batchInt8ExpertStore() const noexcept;",
            "  [[nodiscard]] const FlashInt8ExpertStore *batchInt8ExpertStore() const noexcept;\n"
            "  // Private numerical target derivative, frozen at construction.\n"
            "  [[nodiscard]] bool allRowsInt8TargetEnabled() const noexcept;",
        )
    if relative == "runtime/flash/FlashForward.cpp":
        text = replacement(
            text,
            '#include "flash/FlashInt8ExpertStore.hpp"',
            '#include "flash/FlashInt8ExpertStore.hpp"\n'
            '#include "flash/FlashInt8ExpertStoreMetadata.hpp"',
        )
        text = replacement(
            text,
            '  const bool blockMoE = fusionEnabled("SPLASH_FLASH_BLOCKED_MOE");',
            '  const bool blockMoE = fusionEnabled("SPLASH_FLASH_BLOCKED_MOE");\n'
            f'  const bool allRowsInt8Target = fusionEnabled("{FLAG}");',
        )
        guard = '''    // Bounded metadata preflight precedes every backend workspace/cache allocation.
    if (allRowsInt8Target) {
      if (!blockMoE)
        throw std::invalid_argument("private all-row Full512 target requires blocked MoE");
      const auto directory = int8ExpertDirectory();
      if (!directory)
        throw std::invalid_argument("private all-row Full512 target requires a saved INT8 Store");
      const auto metadata = loadFlashInt8ExpertStoreMetadata(*directory, weights.sourceIdentity(),
          weights.manifestFingerprint(), weights.normConvention());
      for (uint32_t layer = 0; layer < 48; ++layer) {
        const auto &ids = metadata.layers[layer].selectedIDs;
        if (ids.size() != 512)
          throw std::invalid_argument("private all-row target requires all 512 experts in every layer");
        for (uint32_t expert = 0; expert < 512; ++expert)
          if (ids[expert] != expert)
            throw std::invalid_argument("private all-row target requires canonical complete expert IDs");
      }
    }
'''
        text = replacement(
            text,
            "    const uint64_t before = backend.memoryStats().allocatedBytes;",
            guard + "    const uint64_t before = backend.memoryStats().allocatedBytes;",
        )
        text = replacement(
            text,
            "    if (blockMoE && maximumRows >= 256)\n"
            "      blockedScratch = allocateMoEBlockedScratch(backend, maximumRows, kSelections);",
            "    if (blockMoE && (allRowsInt8Target || maximumRows >= 256))\n"
            "      blockedScratch = allocateMoEBlockedScratch(backend, maximumRows, kSelections);",
        )
        text = replacement(
            text,
            "    const bool blocked = impl_->blockMoE && rows >= 256;",
            "    const bool blocked = impl_->blockMoE && (impl_->allRowsInt8Target || rows >= 256);",
        )
        text = replacement(
            text,
            "      const auto tile = flashMoEBlockedTile(rows, bool(impl_->expertCaches[layer]));",
            "      const auto tile = impl_->allRowsInt8Target && rows < 256\n"
            "          ? FlashMoEBlockedTile::M16N64 : flashMoEBlockedTile(rows, bool(impl_->expertCaches[layer]));",
        )
        text = replacement(
            text,
            "const FlashInt8ExpertStore *FlashForward::batchInt8ExpertStore() const noexcept {",
            "bool FlashForward::allRowsInt8TargetEnabled() const noexcept {\n"
            "  return impl_ && impl_->allRowsInt8Target;\n}\n\n"
            "const FlashInt8ExpertStore *FlashForward::batchInt8ExpertStore() const noexcept {",
        )
        text = replacement(
            text,
            "return std::string(flashAffineSemantics()) +",
            "return std::string(flashAffineSemantics()) +\n"
            f'      (impl_->allRowsInt8Target ? "{ROUTE}" : "") +',
        )
        # Existing Worker planning separately admits wide blocked arenas. Only
        # the newly allocated small singleton arena belongs in this subtotal.
        text = replacement(
            text,
            "  return total;\n}\n\nstd::string FlashForward::kernelRoutes() const {",
            f'  if (fusionEnabled("{FLAG}") && maximumRows < 256)\n'
            "    total += flashMoEBlockedWorkspacePlannedBytes(maximumRows, kSelections, kAlignment);\n"
            "  return total;\n}\n\nstd::string FlashForward::kernelRoutes() const {",
        )
        return text
    if relative in (
        "runtime/flash/FlashBatchForward.cpp",
        "runtime/flash/FlashBatchVerify.cpp",
    ):
        return _batch_transform(relative, text)
    if relative == "runtime/flash/FlashBatchPrefill.cpp":
        text = replacement(
            text,
            '  const bool blockMoE = enabled("SPLASH_FLASH_BLOCKED_MOE");',
            '  const bool blockMoE = enabled("SPLASH_FLASH_BLOCKED_MOE");\n'
            "  const bool allRowsInt8Target = trunk.allRowsInt8TargetEnabled();",
        )
        guard = f'''    if (enabled("{FLAG}") != allRowsInt8Target)
      throw std::invalid_argument("private all-row Full512 flag changed after source trunk construction");
    if (allRowsInt8Target && (!blockMoE || !trunk.batchInt8ExpertStore()))
      throw std::invalid_argument("private all-row prefill requires blocked source Full512 target");
'''
        text = replacement(
            text,
            "    const uint64_t before = backend.memoryStats().allocatedBytes;",
            guard + "    const uint64_t before = backend.memoryStats().allocatedBytes;",
        )
        text = replacement(
            text,
            "    if (blockMoE && maximumLanes * maximumRows >= 256)",
            "    if (blockMoE && (allRowsInt8Target || maximumLanes * maximumRows >= 256))",
        )
        text = replacement(
            text,
            "  if (lanes * rows >= 256) total += blockedPlanned(lanes * rows);",
            f'  if (lanes * rows >= 256 || enabled("{FLAG}"))\n'
            "    total += blockedPlanned(lanes * rows);",
        )
        text = replacement(
            text,
            "    const bool useBlocked = impl_->blockMoE && flat >= 256;",
            "    const bool useBlocked = impl_->blockMoE && (impl_->allRowsInt8Target || flat >= 256);",
        )
        return replacement(
            text,
            "      const auto tile = flashMoEBlockedTile(flat, impl_->trunk.batchExpertCache(layer) != nullptr);",
            "      const auto tile = impl_->allRowsInt8Target && flat < 256\n"
            "          ? FlashMoEBlockedTile::M16N64 : flashMoEBlockedTile(flat, impl_->trunk.batchExpertCache(layer) != nullptr);",
        )
    return text


def transform(relative: str, text: str) -> str:
    if relative not in RELATIVES:
        return text
    if MARKER in text:
        raise RuntimeError("Private all-row route overlay is already applied")
    return MARKER + "\n" + _transform(relative, text)


def cpu_self_test(root: Path) -> None:
    """Check integration anchors and fail-closed transforms without backend construction."""
    checks = 0
    for relative in RELATIVES:
        source = (root / relative).read_text()
        changed = transform(relative, source)
        assert changed != source
        assert FLAG in changed or relative.endswith(".hpp")
        try:
            transform(relative, changed)
        except RuntimeError:
            checks += 1
        else:
            raise AssertionError("double transformation must fail closed")
        checks += 1
    for relative in (
        "runtime/flash/FlashMTP.cpp",
        "runtime/flash/FlashBatchMTPForward.cpp",
        "runtime/flash/FlashWeights.mm",
        "runtime/flash/FlashInt8ExpertStore.mm",
    ):
        source = (root / relative).read_text()
        assert transform(relative, source) == source
        checks += 1
    print(f"private all-row executor CPU source checks passed: {checks}; GPU work: false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cpu-self-test", action="store_true", required=True)
    args = parser.parse_args()
    cpu_self_test(args.root.resolve())


if __name__ == "__main__":
    main()
