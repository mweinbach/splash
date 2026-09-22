#!/usr/bin/env python3
"""Private exact discarded-GDN-producer guard correction; no inference/payloads."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("frozen GDN ILP source anchor changed: " + before[:80])
    return text.replace(before, after, 1)


def transform(source):
    source = once(source, '#include "FlashGDNBatchILP.hpp"',
        '#include "FlashGDNBatchILP.hpp"\n#include "dev/benchmarks/gdn_chunk_sep21/worker_bridge.hpp"')
    return once(source,
        '    const std::array<const char *, 4> expected{"flash_gdn_fused_prepare", "flash_gdn_staged_v16_t16",',
        '''    // Phase 1 is validated but never appended: the original ILP
    // recurrence below remains separate-multiply/add regardless of FMA flag.
    // Match precisely the staged helper's frozen per-lane producer selector.
    const char *discardedRecurrence = gdn_prefill_fma_sep21::eligible(
        lane.rows, 1, gdn_prefill_fma_sep21::requested())
        ? gdn_prefill_fma_sep21::kPipeline : "flash_gdn_staged_v16_t16";
    const std::array<const char *, 4> expected{"flash_gdn_fused_prepare", discardedRecurrence,''')


def guard_probe(source):
    start = source.index('    if (validated[slot].dispatches().size() != 4)')
    end = source.index('\n  }\n  // Inactive bindings', start)
    guard = source[start:end]
    return '''// CPU only: exact private source guard body, not a Metal backend.
#include "flash/FlashGDNBatchILP.hpp"
#include "dev/benchmarks/gdn_chunk_sep21/worker_bridge.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashGDNBatchILP.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <stdexcept>
using namespace splash::flash;
using namespace splash::metal;
void fail(const char *text) { throw std::invalid_argument(text); }
struct GraphView {
  const CommandGraph &graph;
  auto dispatches() const { return graph.dispatches(); }
};
void validate(const CommandGraph &producer, uint32_t rows) {
  const std::array<GraphView, 1> validated{{{producer}}};
  const uint32_t slot = 0;
  const FlashGDNBatchILPLane lane{{}, rows};
''' + guard + '''
}
int main(int argc, char **argv) {
  if (argc != 2 || (std::string_view(argv[1]) != "0" && std::string_view(argv[1]) != "1"))
    throw std::invalid_argument("guard-cpu FMA0-or-1");
  setenv("SPLASH_FLASH_GDN_PREFILL_FMA_SEP21", argv[1], 1);
  setenv("SPLASH_FLASH_GDN_STAGED", "1", 1);
  const bool selected = gdn_prefill_fma_sep21::requested();
  uint32_t checks = 0;
  const auto graph = [&](const char *recurrence, int corrupted = -1) {
    CommandGraph producer;
    FlashGDNParams params{2048,1,16,48,128,128,4,1e-6f,
        flashGDNConvolutionLaneBytes(),flashGDNRecurrentLaneBytes()};
    const std::array<const char *,4> names{"flash_gdn_fused_prepare",recurrence,
        "flash_gdn_output","flash_gdn_convolution_carry"};
    for (uint32_t phase = 0; phase < 4; ++phase) {
      std::vector<MetalBuffer> buffers(phase == 0 ? 11 : phase == 1 ? 6 : phase == 2 ? 5 : 3);
      producer.add(corrupted == int(phase) ? "arbitrary_unqualified_pipeline" : names[phase],
          std::move(buffers),params,{1,1,1});
    }
    return producer;
  };
  const auto rejection = [&](auto operation) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::logic_error("actual producer guard accepted invalid graph");
    ++checks;
  };
  for (uint32_t rows : {1u,63u,64u,512u,2048u}) {
    const bool eligible = gdn_prefill_fma_sep21::eligible(rows,1,selected);
    const char *expected = eligible ? gdn_prefill_fma_sep21::kPipeline : "flash_gdn_staged_v16_t16";
    auto valid = graph(expected); validate(valid,rows); ++checks;
    for (int phase = 0; phase < 4; ++phase)
      rejection([&] { validate(graph(expected,phase),rows); });
    const char *opposite = eligible ? "flash_gdn_staged_v16_t16" : gdn_prefill_fma_sep21::kPipeline;
    rejection([&] { validate(graph(opposite),rows); });
  }
  const char *expected = selected ? gdn_prefill_fma_sep21::kPipeline : "flash_gdn_staged_v16_t16";
  auto valid = graph(expected);
  auto badABI = graph(expected);
  const_cast<ComputeDispatch &>(badABI.dispatches()[0]).bytes[0].sizeBytes -= 1;
  rejection([&] { validate(badABI,2048); });
  auto badBinding = graph(expected);
  const_cast<ComputeDispatch &>(badBinding.dispatches()[2]).buffers[0].index = 9;
  rejection([&] { validate(badBinding,2048); });
  auto badCount = graph(expected);
  FlashGDNParams params{};
  badCount.add("extra_unqualified_phase",{},params,{1,1,1});
  rejection([&] { validate(badCount,2048); });
  setenv("SPLASH_FLASH_GDN_PREFILL_FMA_SEP21",selected ? "0" : "1",1);
  if (gdn_prefill_fma_sep21::requested() != selected)
    throw std::logic_error("producer selector did not preserve frozen FMA policy");
  ++checks;
  std::cout << "{\\"valid\\":true,\\"checks\\":" << checks << ",\\"FMA_requested\\":"
      << (selected ? "true" : "false")
      << ",\\"discarded_FMA1_producer_allowed_only_by_exact_selector\\":true"
      << ",\\"arbitrary_pipeline_ABI_binding_count_rejected\\":true"
      << ",\\"metal_backend_constructions\\":0,\\"gpu_commands\\":0}\\n";
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-parent", type=Path, default=ROOT / "build/current-batch-native-clock-sep22-v4")
    parser.add_argument("--output", type=Path, default=ROOT / "build/current-batch-native-clock-sep22-v5")
    args = parser.parse_args()
    clock, output = args.clock_parent.resolve(), args.output.resolve()
    if output.exists() or ROOT / "build" not in output.parents:
        raise ValueError("choose a fresh private build output")
    seal = json.loads((clock / "compiled-cpu-seal.json").read_text())
    parent = Path(seal["parent"])
    relative = "runtime/flash/FlashGDNBatchILP.cpp"
    original = (parent / "source" / relative).read_text()
    modified = transform(original)
    shutil.copytree(clock / "source", output / "source")
    (output / "source" / relative).write_text(modified)
    machinery = output / "machinery"
    machinery.mkdir(exist_ok=True)
    (machinery / "guard_cpu.cpp").write_text(guard_probe(modified))
    remainder_start = original.index('  // Inactive bindings')
    if modified[modified.index('  // Inactive bindings'):] != original[remainder_start:]:
        raise ValueError("appended GDN ILP graph changed")
    witness = {"schema": "splash-batch-gdn-discarded-producer-guard-plan-v1",
        "clock_parent": str(clock), "qualified_math_parent": str(parent),
        "changed_parent_cpp": relative, "clock_worker_source_unchanged": True,
        "all_other_producer_ABI_binding_extent_alias_owner_guards_unchanged": True,
        "phase1_selector": "exact frozen gdn_prefill_fma_sep21::eligible(lane.rows,1,requested())",
        "phase1_is_validated_but_never_appended": True,
        "actual_appended_phases": [0, "original separate-multiply/add merged ILP recurrence", 2, 3],
        "merged_ILP_recurrence_FMA": False,
        "global_FMA_flag_still_controls_other_qualified_prefill_paths": True,
        "no_FMA_or_GDN_ILP_flag_disabled": True,
        "kernel_changes": [], "header_changes": [], "numerical_route_changes": [],
        "gpu_executed": False, "model_payload_bytes_read": 0,
        "root_failed_warmup_report_retained": True}
    (output / "source-plan.json").write_text(json.dumps(witness, indent=2) + "\n")
    print(json.dumps(witness))


if __name__ == "__main__":
    main()
