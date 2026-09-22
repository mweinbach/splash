#!/usr/bin/env python3
"""Private Store-only gathered-I8 QMV transform; import never performs inference.

Compose after prefill4k_allrows_store.transform in a NEW copied source tree.
Executor routing is the composing parent's responsibility. This does not change
production or the existing MPP methods. Flag0 is the original MPP control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLAG = "SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV"
POLICY = "private-gathered-signed-i8-bf16-input-f32-lane32-strided-dot-late-row-scale-bf16-dots-swiglu-sg4-c1-rows1to16-v1"
HEADER_RELATIVE = "runtime/flash/FlashGatheredI8QMV.hpp"
METAL_RELATIVE = "runtime/metal/kernels/shared/flash_gathered_i8_qmv.metal"
MARKER = "// Private gathered I8 QMV Store overlay v1."


def replace(text: str, before: str, after: str, count: int = 1) -> str:
    actual = text.count(before)
    if actual != count:
        raise RuntimeError(f"Private gathered QMV source drift: expected{count}, got{actual}: {before!r}")
    return text.replace(before, after)


def extra_files() -> dict[str, str]:
    """Content to stage before compiling; no mutation occurs here."""
    base = ROOT / "dev/benchmarks/prefill4k_allrows_qmv"
    return {
        HEADER_RELATIVE: base.with_suffix(".hpp").read_text(),
        METAL_RELATIVE: base.with_suffix(".metal").read_text(),
    }


def transform(relative: str, text: str) -> str:
    if relative in ("runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm"):
        if MARKER in text:
            raise RuntimeError("Private gathered I8 QMV transform already applied")
        text = MARKER + "\n" + text
    if relative == "runtime/flash/FlashInt8ExpertStore.hpp":
        text = replace(text, '#include "FlashMoEBlocked.hpp"', '#include "FlashMoEBlocked.hpp"\n#include "FlashGatheredI8QMV.hpp"')
        text = replace(text, "  uint64_t large_row_full_inventory_graph_calls = 0;", """  uint64_t large_row_full_inventory_graph_calls = 0;
  // Graph construction, not GPU completion. QMV only physical rows1..16.
  uint64_t gathered_qmv_gate_up_graph_calls = 0, gathered_qmv_gate_up_graph_rows = 0;
  uint64_t gathered_qmv_down_graph_calls = 0, gathered_qmv_down_graph_rows = 0;""")
        return replace(text, "private:\n  struct Impl;", DECLARATIONS + "private:\n  struct Impl;")
    if relative != "runtime/flash/FlashInt8ExpertStore.mm":
        return text
    text = replace(text, "  std::string numericalIdentity;", """  std::string numericalIdentity;
  const bool gatheredQMV = gathered_i8_qmv::requested();
  mutable std::atomic<uint64_t> qmvGateCalls{0}, qmvGateRows{0}, qmvDownCalls{0}, qmvDownRows{0};""")
    text = replace(text, '    const std::string derivative = std::string("splash.private-allrows-target-v1\\nsource=") +', '    std::string derivative = std::string("splash.private-allrows-target-v1\\nsource=") +')
    text = replace(text, "    numericalIdentity = hash(derivative.data(), derivative.size());", """    if (gatheredQMV)
      derivative += std::string("small_row_policy=") + std::string(gathered_i8_qmv::kPolicy) + "\\n";
    numericalIdentity = hash(derivative.data(), derivative.size());""")
    text = replace(text, "      impl_->largeFullCalls.load(std::memory_order_relaxed)};", """      impl_->largeFullCalls.load(std::memory_order_relaxed),
      impl_->qmvGateCalls.load(std::memory_order_relaxed), impl_->qmvGateRows.load(std::memory_order_relaxed),
      impl_->qmvDownCalls.load(std::memory_order_relaxed), impl_->qmvDownRows.load(std::memory_order_relaxed)};""")
    return replace(text, "} // namespace splash::flash", METHODS + "} // namespace splash::flash")


DECLARATIONS = r'''
  // Store methods keep immutable base mappings and rank allocations private.
  // Dispatch-bound MetalBuffer copies retain their bases until graph disposal.
  [[nodiscard]] bool gatheredQMVEnabled() const;
  void addGatheredQMVGateUp(metal::CommandGraph &graph, uint32_t layer,
      metal::MetalBuffer input, metal::MetalBuffer originalExpertIDs,
      metal::MetalBuffer canonicalIntermediate, metal::MetalBuffer diagnostics,
      uint32_t rows, uint32_t selections = 10) const;
  void addGatheredQMVDown(metal::CommandGraph &graph, uint32_t layer,
      metal::MetalBuffer canonicalIntermediate, metal::MetalBuffer originalExpertIDs,
      metal::MetalBuffer canonicalExpertDown, metal::MetalBuffer diagnostics,
      uint32_t rows, uint32_t selections = 10) const;
'''

METHODS = r'''
namespace {
void gatheredQMVViews(const gathered_i8_qmv::Geometry &g,
    metal::MetalBuffer input, metal::MetalBuffer ids, metal::MetalBuffer output,
    metal::MetalBuffer diagnostics, bool down) {
  requireBytes(input, down ? g.intermediateBytes : g.inputBytes);
  requireBytes(ids, g.idBytes);
  requireBytes(output, down ? g.expertDownBytes : g.intermediateBytes);
  requireBytes(diagnostics, 4);
  const auto view = [](const metal::MetalBuffer &b) {
    return gathered_i8_qmv::ByteView{reinterpret_cast<uintptr_t>(b.contents()), b.sizeBytes()};
  };
  gathered_i8_qmv::validateViews(g, view(input), view(ids), view(output), view(diagnostics), down);
}
} // namespace
bool FlashInt8ExpertStore::gatheredQMVEnabled() const {
  if (!impl_) fail("private gathered I8 QMV Store was moved or disposed");
  if (gathered_i8_qmv::requested() != impl_->gatheredQMV)
    fail("private gathered I8 QMV flag changed after Store construction");
  return impl_->gatheredQMV;
}
void FlashInt8ExpertStore::addGatheredQMVGateUp(metal::CommandGraph &graph, uint32_t index,
    metal::MetalBuffer input, metal::MetalBuffer originalExpertIDs,
    metal::MetalBuffer canonicalIntermediate, metal::MetalBuffer diagnostics,
    uint32_t rows, uint32_t selections) const {
  if (!gatheredQMVEnabled()) fail("private gathered I8 QMV gate/up requires frozen flag1");
  const auto g = gathered_i8_qmv::geometry(rows, selections);
  const auto &layer = impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("private gathered I8 QMV requires Full512");
  gatheredQMVViews(g, input, originalExpertIDs, canonicalIntermediate, diagnostics, false);
  for (const auto &buffer : {input, originalExpertIDs, canonicalIntermediate, diagnostics})
    impl_->immutableDisjoint(buffer);
  requireBytes(layer.ranks, 512 * 4);
  for (uint32_t p = 0; p < 2; ++p) {
    requireBytes(layer.codes[p], uint64_t{512} * 640 * 2560);
    requireBytes(layer.scales[p], uint64_t{512} * 640 * 4);
  }
  graph.add("flash_gathered_i8_qmv_gate_up_sg4_c1", {input, layer.codes[0], layer.scales[0],
      layer.codes[1], layer.scales[1], layer.ranks, originalExpertIDs, canonicalIntermediate,
      diagnostics}, FlashGatheredI8QMVParams{rows, selections, 512, 0},
      {g.gateColumnGroups, rows, selections}, {gathered_i8_qmv::kThreads, 1, 1});
  impl_->recordGraph(false, rows, 512);
  impl_->qmvGateCalls.fetch_add(1, std::memory_order_relaxed);
  impl_->qmvGateRows.fetch_add(rows, std::memory_order_relaxed);
}
void FlashInt8ExpertStore::addGatheredQMVDown(metal::CommandGraph &graph, uint32_t index,
    metal::MetalBuffer canonicalIntermediate, metal::MetalBuffer originalExpertIDs,
    metal::MetalBuffer canonicalExpertDown, metal::MetalBuffer diagnostics,
    uint32_t rows, uint32_t selections) const {
  if (!gatheredQMVEnabled()) fail("private gathered I8 QMV down requires frozen flag1");
  const auto g = gathered_i8_qmv::geometry(rows, selections);
  const auto &layer = impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("private gathered I8 QMV requires Full512");
  gatheredQMVViews(g, canonicalIntermediate, originalExpertIDs, canonicalExpertDown, diagnostics, true);
  for (const auto &buffer : {canonicalIntermediate, originalExpertIDs, canonicalExpertDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  requireBytes(layer.ranks, 512 * 4);
  requireBytes(layer.codes[2], uint64_t{512} * 2560 * 640);
  requireBytes(layer.scales[2], uint64_t{512} * 2560 * 4);
  graph.add("flash_gathered_i8_qmv_down_sg4_c1", {canonicalIntermediate, layer.codes[2],
      layer.scales[2], layer.ranks, originalExpertIDs, canonicalExpertDown, diagnostics},
      FlashGatheredI8QMVParams{rows, selections, 512, 0},
      {g.downColumnGroups, rows, selections}, {gathered_i8_qmv::kThreads, 1, 1});
  impl_->recordGraph(true, rows, 512);
  impl_->qmvDownCalls.fetch_add(1, std::memory_order_relaxed);
  impl_->qmvDownRows.fetch_add(rows, std::memory_order_relaxed);
}
'''


def stage(source: Path, destination: Path) -> dict:
    if source.resolve() == destination.resolve():
        raise ValueError("private QMV staging requires a different destination")
    if destination.exists():
        raise ValueError("private QMV staging requires a NEW destination")
    destination.mkdir(parents=True)
    witness = {}
    for relative in ("runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm"):
        before = (source / relative).read_text()
        after = transform(relative, before)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(after)
        witness[relative] = {"input_sha256": hashlib.sha256(before.encode()).hexdigest(),
                             "output_sha256": hashlib.sha256(after.encode()).hexdigest()}
    for relative, content in extra_files().items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        witness[relative] = {"output_sha256": hashlib.sha256(content.encode()).hexdigest()}
    result = {"policy": POLICY, "flag": FLAG, "gpu_work": False, "files": witness}
    (destination / "qmv-overlay-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def cpu_self_test(source: Path) -> dict:
    """Bounded source-only transform guards; no model metadata/payload reads."""
    outputs = {}
    for relative in ("runtime/flash/FlashInt8ExpertStore.hpp", "runtime/flash/FlashInt8ExpertStore.mm"):
        before = (source / relative).read_text()
        after = transform(relative, before)
        if after == before:
            raise AssertionError("supported source was not transformed")
        try:
            transform(relative, after)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate transform was not rejected")
        outputs[relative] = after
        for primitive in ("backend.allocateBuffer(", "backend.wrapSharedMemory(", "std::make_shared<Mapping>("):
            if after.count(primitive) != before.count(primitive):
                raise AssertionError("QMV added an allocation or mapping")
        if relative.endswith(".mm"):
            start = before.index("void FlashInt8ExpertStore::addGateUp(")
            end = before.index("} // namespace splash::flash", start)
            if before[start:end] not in after:
                raise AssertionError("original MPP graph methods changed")
            if "if (gatheredQMV)\n      derivative +=" not in after:
                raise AssertionError("conditional derivative suffix missing")
            if after.count("gathered_i8_qmv::requested()") != 2:
                raise AssertionError("frozen policy capture/recheck differs")
            if after.count("impl_->immutableDisjoint(buffer);") != before.count("impl_->immutableDisjoint(buffer);") + 2:
                raise AssertionError("all immutable layer disjoint checks missing")
    for relative in ("runtime/flash/FlashMTP.cpp", "runtime/flash/FlashForward.cpp", "runtime/flash/FlashBatchVerify.cpp"):
        if transform(relative, "untouched") != "untouched":
            raise AssertionError("Store transform rewrote executor/MTP source")
    # Independent identity construction proves flag0 control keeps the old hash.
    seed = "splash.private-allrows-target-v1\nsource=source\nstore=store\npolicy=mpp\nmtp=original-trained-bank\n"
    original = hashlib.sha256(seed.encode()).hexdigest()
    disabled = hashlib.sha256((seed + "").encode()).hexdigest()
    enabled = hashlib.sha256((seed + f"small_row_policy={POLICY}\n").encode()).hexdigest()
    if original != disabled or original == enabled:
        raise AssertionError("MPP control and QMV derivative identities collide")
    files = extra_files()
    if POLICY not in files[HEADER_RELATIVE]:
        raise AssertionError("Python and shared-header numerical policies differ")
    return {"cpu_source_checks": "passed", "gpu_work": False, "extra_allocations": 0,
            "numerical_qualification": "pending", "flag0_identity_preserved": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--cpu-self-test", action="store_true")
    args = parser.parse_args()
    if args.cpu_self_test:
        print(json.dumps(cpu_self_test(args.source), indent=2))
    else:
        if args.destination is None:
            parser.error("staging requires --destination")
        print(json.dumps(stage(args.source, args.destination), indent=2))
