#!/usr/bin/env python3
"""Prepare private full-expert-cache runtime sources without changing production."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def replacement(text, before, after, count=1):
    actual = text.count(before)
    if actual != count:
        raise RuntimeError(f"Private fullcache source drift: expected{count}, got{actual}: {before!r}")
    return text.replace(before, after)


def transform(relative, text):
    if relative == "runtime/flash/FlashInt8ExpertStoreMetadata.mm":
        text = replacement(text, "inventoryCount != 32 && inventoryCount != 64 && inventoryCount != 128", "inventoryCount != 32 && inventoryCount != 64 && inventoryCount != 128 && inventoryCount != 256 && inventoryCount != 512")
        return text.replace("exactly 32, 64 or 128 IDs", "exactly 32, 64, 128, 256 or512 IDs")
    if relative == "runtime/flash/FlashInt8ExpertStoreMetadata.hpp":
        return text.replace("exactly 32, 64 or 128 IDs", "exactly 32, 64, 128, 256 or512 IDs")
    if relative == "runtime/flash/FlashInt8ExpertStore.hpp":
        text = replacement(text, "saved-selected-i8-row-f32scale-bf16activation-f32accum-whole-k-q4-misses-v1", "private-i8-row-f32scale-bf16activation-f32accum-whole-k-inventory32to512-full512-no-q4miss-dispatch-v1")
        text = replacement(text, "namespace splash::flash {", """namespace splash::flash {
// Private diagnostics: successful graph construction, not GPU completion.
struct FlashInt8ExpertStoreGraphCounters {
  uint64_t gate_up_graph_calls = 0, gate_up_graph_rows = 0;
  uint64_t down_graph_calls = 0, down_graph_rows = 0;
  uint64_t encoded_hit_dispatches = 0, encoded_miss_dispatches = 0;
  uint64_t full_inventory_graph_calls = 0;
};""")
        return replacement(text, "  void addGateUp(metal::CommandGraph &graph, uint32_t layer,", "  [[nodiscard]] FlashInt8ExpertStoreGraphCounters graphCounters() const noexcept;\n\n  void addGateUp(metal::CommandGraph &graph, uint32_t layer,")
    if relative == "runtime/flash/FlashInt8ExpertStore.mm":
        # Exactly512 sorted unique IDs in [0,511] prove every original ID is
        # covered. The immutable constructor builds every rank before exposure.
        # Avoid even scheduling no-op miss groups that bind raw Q4 allocations.
        text = replacement(text, '  graph.add(pipeline(miss.flags ? "gate_up_miss_direct"', '  if (miss.stored_experts <512) graph.add(pipeline(miss.flags ? "gate_up_miss_direct"')
        text = replacement(text, '  graph.add(pipeline(miss.flags ? "down_miss_direct"', '  if (miss.stored_experts <512) graph.add(pipeline(miss.flags ? "down_miss_direct"')
        text = replacement(text, "#include <algorithm>", "#include <algorithm>\n#include <atomic>")
        text = replacement(text, "  uint64_t allocated = 0;", """  uint64_t allocated = 0;
  mutable std::atomic<uint64_t> gateCalls{0}, gateRows{0}, downCalls{0}, downRows{0};
  mutable std::atomic<uint64_t> hitDispatches{0}, missDispatches{0}, fullCalls{0};
  void recordGraph(bool down, uint32_t rows, uint32_t inventory) const noexcept {
    (down ? downCalls : gateCalls).fetch_add(1, std::memory_order_relaxed);
    (down ? downRows : gateRows).fetch_add(rows, std::memory_order_relaxed);
    hitDispatches.fetch_add(1, std::memory_order_relaxed);
    if (inventory <512) missDispatches.fetch_add(1, std::memory_order_relaxed);
    else fullCalls.fetch_add(1, std::memory_order_relaxed);
  }""")
        text = replacement(text, "void FlashInt8ExpertStore::addGateUp(", """FlashInt8ExpertStoreGraphCounters FlashInt8ExpertStore::graphCounters() const noexcept {
  if (!impl_) return {};
  return {impl_->gateCalls.load(std::memory_order_relaxed), impl_->gateRows.load(std::memory_order_relaxed),
      impl_->downCalls.load(std::memory_order_relaxed), impl_->downRows.load(std::memory_order_relaxed),
      impl_->hitDispatches.load(std::memory_order_relaxed), impl_->missDispatches.load(std::memory_order_relaxed),
      impl_->fullCalls.load(std::memory_order_relaxed)};
}

void FlashInt8ExpertStore::addGateUp(""")
        text = replacement(text, "miss, {10, miss.blocked.job_capacity, 1}, {threads, 1, 1});\n}", "miss, {10, miss.blocked.job_capacity, 1}, {threads, 1, 1});\n  impl_->recordGraph(false, rows, miss.stored_experts);\n}")
        return replacement(text, "}, miss, {40, miss.blocked.job_capacity, 1}, {threads, 1, 1});\n}", "}, miss, {40, miss.blocked.job_capacity, 1}, {threads, 1, 1});\n  impl_->recordGraph(true, rows, miss.stored_experts);\n}")
    if relative == "runtime/flash/FlashWorker.mm":
        text = replacement(text, "  uint64_t persistedExpertCount = 0;", "  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};\n  uint64_t persistedExpertCount = 0;")
        return replacement(text, '      << R"(,"scope":"large-row target prefill; original Q4 misses, decode and trained MTP"})"', '''      << R"(,"scope":"large-row target prefill; original Q4 misses, decode and trained MTP","graph_counters":{"scope":"graph construction, not GPU completion","gate_up_graph_calls":)" << persistedExpertGraphs.gate_up_graph_calls
      << R"(,"gate_up_graph_rows":)" << persistedExpertGraphs.gate_up_graph_rows
      << R"(,"down_graph_calls":)" << persistedExpertGraphs.down_graph_calls
      << R"(,"down_graph_rows":)" << persistedExpertGraphs.down_graph_rows
      << R"(,"encoded_hit_dispatches":)" << persistedExpertGraphs.encoded_hit_dispatches
      << R"(,"encoded_miss_dispatches":)" << persistedExpertGraphs.encoded_miss_dispatches
      << R"(,"full_inventory_graph_calls":)" << persistedExpertGraphs.full_inventory_graph_calls
      << R"(}})"''')
    if relative == "runtime/metal/kernels/shared/flash_int8_expert_store.metal":
        return replacement(text, "p.stored_experts > 128", "p.stored_experts > 512", 2)
    if relative == "dev/benchmarks/flash_int8_expert_store_metadata_cpu.mm":
        text = replacement(text, "for (uint32_t inventory : {32, 64, 128})", "for (uint32_t inventory : {32, 64, 128, 256, 512})")
        text = replacement(text, "[ids addObject:@(layer + rank)]", "[ids addObject:@(counts[layer] ==512 ? rank : layer + rank)]")
        text = replacement(text, "expectedIDs.push_back(i + rank)", "expectedIDs.push_back(inventory ==512 ? rank : i + rank)")
        return text.replace("exactly 32, 64 or 128 IDs", "exactly 32, 64, 128, 256 or512 IDs")
    if relative == "dev/benchmarks/flash_int8_expert_store_oracle.mm":
        text = replacement(text, 'require(hot.size() >= 10 && cold.size() >= 10, "synthetic top10 patterns need at least10 stored and10 missing experts");', 'require(hot.size() >=10 && ((pattern !="miss-only" && pattern !="mixed") || cold.size() >=10), "synthetic pattern requires available stored/missing experts");')
        text = replacement(text, "for (uint32_t count : {10u, 32u, 128u})", "for (uint32_t count : {10u, 32u, 128u, 256u, 512u})")
        text = replacement(text, "      const auto ids = patternIDs(129, hot, pattern);", '      if (count ==512 && (std::string_view(pattern) =="miss-only" || std::string_view(pattern) =="mixed")) continue;\n      const auto ids = patternIDs(129, hot, pattern);')
        text = replacement(text, '          if (selectedPattern && std::string_view(selectedPattern) != pattern) continue;', '          if (store->selectedExpertIDs(layer).size() ==512 && (std::string_view(pattern) =="miss-only" || std::string_view(pattern) =="mixed")) continue;\n          if (selectedPattern && std::string_view(selectedPattern) != pattern) continue;')
        # Primitive exactness remains required only on genuinely unchanged
        # source-Q4 misses. INT8 hits use declared finite numerical guards.
        return text.replace("flash-production-selected-int8-expert-store-oracle-v1", "prefill4k-private-full512-int8-expert-numerical-oracle-v1")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("build/prefill4k-fullcache"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if root / "build" not in output.parents:
        raise ValueError("Private overlay must remain inside repository build directory")
    relatives = [str(path.relative_to(root)) for path in sorted((root / "runtime/flash").glob("*")) if path.is_file()]
    relatives += ["runtime/metal/kernels/shared/flash_int8_expert_store.metal", "dev/benchmarks/flash_int8_expert_store_metadata_cpu.mm", "dev/benchmarks/flash_int8_expert_store_oracle.mm", "dev/benchmarks/flash_expert_int8_bucket_reference.hpp"]
    manifest = {"schema": 1, "route": "private-target-int8-cache-inventory32to512-full512-no-q4miss-dispatch-v1", "normal_sources_modified": False, "arithmetic_change": True, "fullcache_planned_payload_and_ranks_bytes": 121174228992, "files": []}
    for relative in relatives:
        original = (root / relative).read_bytes()
        changed = transform(relative, original.decode()).encode()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(changed)
        manifest["files"].append({"path": relative, "patched": changed != original, "original_sha256": hashlib.sha256(original).hexdigest(), "overlay_sha256": hashlib.sha256(changed).hexdigest()})
    output.joinpath("overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared": str(output), "gpu_work": False, "files": len(relatives), "patched": sum(item["patched"] for item in manifest["files"])}))


if __name__ == "__main__":
    main()
