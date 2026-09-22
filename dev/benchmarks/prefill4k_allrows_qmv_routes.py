"""Private rows1..16 gathered-I8 target route composition."""
from __future__ import annotations


def replace(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise RuntimeError(f"Gathered QMV route source drift: {before!r}")
    return text.replace(before, after)


def transform(relative: str, text: str) -> str:
    if relative == "runtime/flash/FlashForward.cpp":
        marker = "    const bool blocked = impl_->blockMoE && (impl_->allRowsInt8Target || rows >= 256);\n    if (blocked) {"
        new = '''    const bool gatheredQMV = impl_->allRowsInt8Target && rows <= 16 &&
        impl_->int8ExpertStore && impl_->int8ExpertStore->gatheredQMVEnabled();
    const bool blocked = impl_->blockMoE && (impl_->allRowsInt8Target || rows >= 256);
    if (gatheredQMV) {
      impl_->int8ExpertStore->addGatheredQMVGateUp(graph, layer, mixed, ids,
          bf(Scratch::ExpertIntermediate, kSelections * 640), diag, rows, kSelections);
      impl_->int8ExpertStore->addGatheredQMVDown(graph, layer,
          bf(Scratch::ExpertIntermediate, kSelections * 640), ids,
          bf(Scratch::ExpertDown, kSelections * kWidth), diag, rows, kSelections);
    } else if (blocked) {'''
        text = replace(text, marker, new)
        return replace(text, "addCombine(graph, blocked ? impl_->blockedScratch.scatteredDown", "addCombine(graph, blocked && !gatheredQMV ? impl_->blockedScratch.scatteredDown")
    if relative in ["runtime/flash/FlashBatchForward.cpp", "runtime/flash/FlashBatchVerify.cpp"]:
        rows = "lanes" if relative.endswith("FlashBatchForward.cpp") else "flattened"
        marker = "    if (impl_->allRowsInt8Target) {\n      constexpr auto tile = FlashMoEBlockedTile::M16N64;"
        new = f'''    const bool gatheredQMV = impl_->allRowsInt8Target && {rows} <= 16 &&
        impl_->int8ExpertStore && impl_->int8ExpertStore->gatheredQMVEnabled();
    if (gatheredQMV) {{
      impl_->int8ExpertStore->addGatheredQMVGateUp(graph, layer, mixed, expertIDs,
          bf(Slot::Intermediate, kSelections * 640), diag, {rows}, kSelections);
      impl_->int8ExpertStore->addGatheredQMVDown(graph, layer,
          bf(Slot::Intermediate, kSelections * 640), expertIDs,
          bf(Slot::ExpertDown, kSelections * kWidth), diag, {rows}, kSelections);
    }} else if (impl_->allRowsInt8Target) {{
      constexpr auto tile = FlashMoEBlockedTile::M16N64;'''
        text = replace(text, marker, new)
        return replace(text, "addCombine(graph, impl_->allRowsInt8Target ? impl_->blockedScratch.scatteredDown", "addCombine(graph, impl_->allRowsInt8Target && !gatheredQMV ? impl_->blockedScratch.scatteredDown")
    if relative == "runtime/flash/FlashBatchPrefill.cpp":
        marker = "    const bool useBlocked = impl_->blockMoE && (impl_->allRowsInt8Target || flat >= 256);\n    if (useBlocked) {"
        new = '''    const auto *qmvStore = impl_->trunk.batchInt8ExpertStore();
    const bool gatheredQMV = impl_->allRowsInt8Target && flat <= 16 && qmvStore &&
        qmvStore->gatheredQMVEnabled();
    const bool useBlocked = impl_->blockMoE && (impl_->allRowsInt8Target || flat >= 256);
    if (gatheredQMV) {
      qmvStore->addGatheredQMVGateUp(graph, layer, mixed, expertIDs,
          bf(Slot::Intermediate, kSelections * 640), diag, flat, kSelections);
      qmvStore->addGatheredQMVDown(graph, layer,
          bf(Slot::Intermediate, kSelections * 640), expertIDs,
          bf(Slot::ExpertDown, kSelections * kWidth), diag, flat, kSelections);
    } else if (useBlocked) {'''
        text = replace(text, marker, new)
        return replace(text, "addCombine(graph, useBlocked ? impl_->blocked.scatteredDown", "addCombine(graph, useBlocked && !gatheredQMV ? impl_->blocked.scatteredDown")
    if relative == "runtime/flash/FlashWorker.mm":
        text = replace(text, '#include "flash/FlashInt8ExpertStore.hpp"', '#include "flash/FlashInt8ExpertStore.hpp"\n#include "flash/FlashGatheredI8QMV.hpp"')
        text = replace(text, "      const auto earlyFull = loadFlashInt8ExpertStoreMetadata", "      (void)gathered_i8_qmv::requested(); // Invalid flag rejects before backend creation.\n      const auto earlyFull = loadFlashInt8ExpertStoreMetadata")
        marker = '      << R"(,"large_row_full_inventory_graph_calls":)" << persistedExpertGraphs.large_row_full_inventory_graph_calls'
        fields = '''      << R"(,"gathered_qmv_counter_scope":"physical rows1..16; graph construction; distinct lane-strided F32 reduction")"
      << R"(,"gathered_qmv_gate_up_graph_calls":)" << persistedExpertGraphs.gathered_qmv_gate_up_graph_calls
      << R"(,"gathered_qmv_gate_up_graph_rows":)" << persistedExpertGraphs.gathered_qmv_gate_up_graph_rows
      << R"(,"gathered_qmv_down_graph_calls":)" << persistedExpertGraphs.gathered_qmv_down_graph_calls
      << R"(,"gathered_qmv_down_graph_rows":)" << persistedExpertGraphs.gathered_qmv_down_graph_rows'''
        return replace(text, marker, marker + "\n" + fields)
    return text
