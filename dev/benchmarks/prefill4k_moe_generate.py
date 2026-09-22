"""Build private MoE variants and an exact current-profile comparison oracle.

This derives candidates from the current source with checked substitutions.
It never edits runtime files, a model, or the current build directory.
"""
from pathlib import Path
import argparse


def replace(text, old, new, count=1):
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"source contract drift: expected {count}, got {actual}: {old[:100]!r}")
    return text.replace(old, new)


def generate(destination):
    destination.mkdir(parents=True, exist_ok=True)
    source = Path("runtime/metal/kernels/shared/flash_int8_expert_store.metal").read_text()
    # Existing Direct-A packed buffers already expose globally initialized +63
    # guard rows. All matrix output/route accesses remain masked to valid_rows.
    static = replace(source, "int(valid_rows)", "int(M)", 2)
    static = static.replace("flash_int8_expert_store_", "prefill4k_moe_static_")
    (destination / "static.metal").write_text(static)

    sg8 = replace(source, "32, 4", "32, 8", 6)
    sg8 = sg8.replace("flash_int8_expert_store_", "prefill4k_moe_sg8_")
    (destination / "sg8.metal").write_text(sg8)

    # Keep the original K64 descriptor and every multiply-accumulate in exactly
    # the same order. Stage two adjacent K64 tiles together to halve barriers.
    direct = Path("runtime/metal/kernels/common/flash_moe_direct_a_common.h").read_text()
    direct = direct.replace("flash_direct_a_q4x8_", "prefill4k_moe_paired_q4x8_")
    direct = replace(direct, "constexpr ushort N = 64, BK = 64;", "constexpr ushort N = 64, BK = 128, MAC_K = 64;", 2)
    direct = replace(direct, "auto a0 = a.template slice<BK, M>(0, 0);", "auto a0 = a.template slice<MAC_K, M>(0, 0);", 2)
    direct = replace(direct, "auto g0 = g.template slice<BK, N>(0, 0);", "auto g0 = g.template slice<MAC_K, N>(0, 0);\n  auto g1 = g.template slice<MAC_K, N>(MAC_K, 0);")
    direct = replace(direct, "auto u0 = u.template slice<BK, N>(0, 0);", "auto u0 = u.template slice<MAC_K, N>(0, 0);\n  auto u1 = u.template slice<MAC_K, N>(MAC_K, 0);")
    direct = replace(direct, "auto b0 = b.template slice<BK, N>(0, 0);", "auto b0 = b.template slice<MAC_K, N>(0, 0);\n  auto b1 = b.template slice<MAC_K, N>(MAC_K, 0);")
    direct = replace(direct, "matmul2d_descriptor(M, N, BK,", "matmul2d_descriptor(M, N, MAC_K,", 2)
    direct = replace(direct, "chunk < 40", "chunk < 20")
    direct = replace(direct, "chunk < 10", "chunk < 5")
    direct = replace(direct, "auto achunk = a.template slice<BK, M>(korigin, 0);", "auto achunk = a.template slice<MAC_K, M>(korigin, 0);\n    auto achunk1 = a.template slice<MAC_K, M>(korigin + MAC_K, 0);", 2)
    direct = replace(direct, "operation.run(achunk, u0, up_acc);", "operation.run(achunk, u0, up_acc);\n    operation.run(achunk1, g1, gate_acc);\n    operation.run(achunk1, u1, up_acc);")
    direct = replace(direct, "operation.run(achunk, b0, acc);", "operation.run(achunk, b0, acc);\n    operation.run(achunk1, b1, acc);")
    (destination / "paired_common.h").write_text(direct)
    paired = source.replace('"metal/kernels/common/flash_moe_direct_a_common.h"', '"paired_common.h"')
    paired = paired.replace("flash_direct_a_q4x8_", "prefill4k_moe_paired_q4x8_")
    paired = paired.replace("flash_int8_expert_store_", "prefill4k_moe_paired_")
    # Both wrapper arrays must accommodate paired B staging; staged-A branches
    # remain unchanged and have their own helpers. Only DIRECT wrappers run here.
    paired = replace(paired, "g[64 * 64], u[64 * 64]", "g[64 * 128], u[64 * 128]")
    paired = replace(paired, "bstage[64 * 64]", "bstage[64 * 128]")
    (destination / "paired.metal").write_text(paired)

    expanded = Path("runtime/metal/kernels/common/flash_moe_direct_a_common.h").read_text()
    expanded = expanded.replace("flash_direct_a_q4x8_", "prefill4k_moe_expanded_q4x8_")
    # Leave the original coefficient helper available to the exact conversion,
    # and remove only the gate/down per-job staging loops. Every K64 operation
    # still visits the same BF16 coefficient and activation in ascending K order.
    expanded = replace(expanded, "device bfloat *input, device const uchar *gate_w,", "device bfloat *input, device bfloat *gate_w,")
    expanded = replace(expanded, "device const uchar *up_w, device const uchar *up_s,", "device bfloat *up_w, device const uchar *up_s,")
    expanded = replace(expanded, "device bfloat *input, device const uchar *weights,", "device bfloat *input, device bfloat *weights,")
    expanded = replace(expanded, "auto g = tensor(staged_gate, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});", "auto g = tensor(gate_w + (ulong(expert) * 640 + norigin) * 2560, dextents<int, 2>{2560, N}, array<int, 2>{1, 2560});")
    expanded = replace(expanded, "auto u = tensor(staged_up, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});", "auto u = tensor(up_w + (ulong(expert) * 640 + norigin) * 2560, dextents<int, 2>{2560, N}, array<int, 2>{1, 2560});")
    expanded = replace(expanded, "auto b = tensor(staged_b, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});", "auto b = tensor(weights + (ulong(expert) * 2560 + norigin) * 640, dextents<int, 2>{640, N}, array<int, 2>{1, 640});")
    for phase in ("gate", "down"):
        begin = expanded.index("    for (uint i = tid; i < uint(N) * BK / 8;")
        end = expanded.index("    threadgroup_barrier(mem_flags::mem_threadgroup);", begin)
        new = "    auto gchunk = g.template slice<BK, N>(korigin, 0);\n    auto uchunk = u.template slice<BK, N>(korigin, 0);\n" if phase == "gate" else "    auto bchunk = b.template slice<BK, N>(korigin, 0);\n"
        expanded = expanded[:begin] + new + expanded[end:]
    expanded = replace(expanded, "    threadgroup_barrier(mem_flags::mem_threadgroup);\n", "", 4)
    expanded = replace(expanded, "operation.run(achunk, g0, gate_acc);", "operation.run(achunk, gchunk, gate_acc);")
    expanded = replace(expanded, "operation.run(achunk, u0, up_acc);", "operation.run(achunk, uchunk, up_acc);")
    expanded = replace(expanded, "operation.run(achunk, b0, acc);", "operation.run(achunk, bchunk, acc);")
    expanded = replace(expanded, "  const constant FlashMoEFusedParams &p = params.affine;", "  (void)gate_s; (void)gate_b; (void)up_s; (void)up_b; (void)staged_gate; (void)staged_up;\n  const constant FlashMoEFusedParams &p = params.affine;")
    expanded = replace(expanded, "  const constant FlashMoEDownFusedParams &p = params.affine;", "  (void)scales; (void)biases; (void)staged_b;\n  const constant FlashMoEDownFusedParams &p = params.affine;")
    (destination / "expanded_common.h").write_text(expanded)
    expanded_store = source.replace('"metal/kernels/common/flash_moe_direct_a_common.h"', '"expanded_common.h"')
    expanded_store = expanded_store.replace("flash_direct_a_q4x8_", "prefill4k_moe_expanded_q4x8_")
    expanded_store = expanded_store.replace("flash_int8_expert_store_", "prefill4k_moe_expanded_")
    # Source Q4 misses still appear in staged fallback declarations; only the
    # private direct wrappers cast their replacement BF16 plane bindings.
    expanded_store = replace(expanded_store, "prefill4k_moe_expanded_q4x8_gate_up_tile<M, SG>(a, gw, gs, gb, uw, us, ub,", "prefill4k_moe_expanded_q4x8_gate_up_tile<M, SG>(a, reinterpret_cast<device bfloat *>(const_cast<device uchar *>(gw)), gs, gb, reinterpret_cast<device bfloat *>(const_cast<device uchar *>(uw)), us, ub,")
    expanded_store = replace(expanded_store, "prefill4k_moe_expanded_q4x8_down_tile<M, SG>(a, w, s, b,", "prefill4k_moe_expanded_q4x8_down_tile<M, SG>(a, reinterpret_cast<device bfloat *>(const_cast<device uchar *>(w)), s, b,")
    # Eliminate unused staging arrays in the DIRECT instantiations. Staged
    # candidate wrappers are deliberately not linked/routed by this oracle.
    expanded_store += r'''
#include "dev/benchmarks/prefill4k_moe_expanded.h"
kernel void prefill4k_moe_expand_q4(
    device const uchar *weights [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device const uchar *biases [[buffer(2)]], device const uint *ranks [[buffer(3)]],
    device const uint *offsets [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device uint *diag [[buffer(6)]], constant Prefill4KMoEExpandedParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
  if (p.experts != 512 || p.reserved || group.x || group.y >= 512 || group.z ||
      threads.x != 256 || threads.y != 1 || threads.z != 1 ||
      !((p.input_size == 2560 && p.output_size == 640) ||
        (p.input_size == 640 && p.output_size == 2560)) ||
      !prefill4k_moe_expanded_q4x8_strides(p.output_size, p.input_size,
          p.weight_row_stride_bytes, p.weight_expert_stride_bytes,
          p.parameter_row_stride_bytes, p.parameter_expert_stride_bytes)) {
    if (!tid) flash_mpp_error(diag, 2u); return;
  }
  const uint expert = group.y;
  if (ranks[expert] != UINT_MAX || offsets[expert] == offsets[expert + 1]) return;
  const ulong matrix = ulong(p.output_size) * p.input_size;
  // One group per expert avoids hundreds of thousands of no-op groups when
  // a store hit or inactive expert rejects conversion. Each live group walks
  // its contiguous coefficient plane with all256 lanes and eight codes/lane.
  for (ulong index = ulong(tid) * 8; index <matrix; index +=256 * 8) {
  const uint n = uint(index / p.input_size), k = uint(index % p.input_size);
  const ulong woffset = ulong(expert) * p.weight_expert_stride_bytes + ulong(n) * p.weight_row_stride_bytes + k / 2;
  const uint codes = *reinterpret_cast<device const uint *>(weights + woffset);
  const ulong poffset = ulong(expert) * p.parameter_expert_stride_bytes + ulong(n) * p.parameter_row_stride_bytes + ulong(k / 64) * 2;
  const bfloat scale = *reinterpret_cast<device const bfloat *>(scales + poffset);
  const bfloat bias = *reinterpret_cast<device const bfloat *>(biases + poffset);
  for (ushort j = 0; j < 8; ++j) {
    const float reconstructed = flash_mpp_dequantize_f32((codes >> (j * 4)) & 15u, scale, bias);
    bfloat value = bfloat(reconstructed);
    if (!flash_mpp_finite(reconstructed) || !flash_mpp_finite(value)) {
      flash_mpp_error(diag, 4u); value = bfloat(0.0f);
    }
    output[ulong(expert) * matrix + index + j] = value;
  }
  }
}
'''
    (destination / "expanded.metal").write_text(expanded_store)

    oracle = Path("dev/benchmarks/flash_int8_expert_store_oracle.mm").read_text()
    oracle = replace(oracle, '"flash_expert_int8_bucket_reference.hpp"', '"dev/benchmarks/flash_expert_int8_bucket_reference.hpp"')
    oracle = replace(oracle, '#include "flash/FlashMoE.hpp"', '#include "flash/FlashMoE.hpp"\n#include "dev/benchmarks/prefill4k_moe_expanded.h"')
    oracle = replace(oracle, 'Replay runCase(MetalBackend &backend, const FlashWeights &weights, const FlashInt8ExpertStore &store,', 'Replay runCase(MetalBackend &backend, const FlashWeights &weights, const FlashInt8ExpertStore &store, splash::engine::MemoryGovernor &governor,')
    oracle = replace(oracle, 'runCase(backend, weights, *store, layer, rows', 'runCase(backend, weights, *store, governor, layer, rows')
    oracle = replace(oracle, "\n  const auto jobs = ref::makeJobs(packed, m);", "\n  const uint32_t candidateM = envNumber(\"PREFILL4K_MOE_TILE\", m, 64);\n  require(candidateM == 16 || candidateM == 32 || candidateM == 64, \"candidate tile must be16/32/64\");\n  const auto jobs = ref::makeJobs(packed, m);\n  const auto candidateJobs = ref::makeJobs(packed, candidateM);")
    for name in ("gate", "up", "down"):
        oracle = replace(oracle, f'  const auto &{name} = weights.projection(prefix + ".{name}_proj");\n', "")
    oracle = replace(oracle, "  const auto prefix = prefixFor(layer);\n", "")
    oracle = replace(oracle, "  const auto tile = static_cast<FlashMoEBlockedTile>(m);\n", "")
    oracle = replace(oracle, "  for (uint32_t which = 0; which < 2; ++which) {\n    scratch[which]", "  for (uint32_t which = 0; which < 2; ++which) {\n    const auto tile = static_cast<FlashMoEBlockedTile>(which ? candidateM : m);\n    scratch[which]")
    begin = oracle.index("    if (!which) {\n      addMoEBlockedGateUp")
    end = oracle.index("    addCombine(graphs[which]", begin)
    oracle = oracle[:begin] + "    store.addGateUp(graphs[which], layer, scratch[which], diagnostic[which], rows, tile);\n    store.addDownScatter(graphs[which], layer, scratch[which], diagnostic[which], rows, tile);\n" + oracle[end:]
    point = "  const auto healthy = [&] {"
    transform = r'''  require(flashMoEDirectAEnabled(), "static/paired oracle requires Direct-A guards and sanitation");
  std::array<std::vector<ComputeDispatch>, 2> commands;
  for (uint32_t which = 0; which < 2; ++which)
    commands[which].assign(graphs[which].dispatches().begin(), graphs[which].dispatches().end());
  const char *rawVariant = std::getenv("PREFILL4K_MOE_VARIANT");
  const std::string variant = rawVariant ? rawVariant : "static";
  require(variant == "production" || variant == "static" || variant == "paired" || variant == "expanded" || variant == "sg8", "unknown candidate variant");
  require(variant != "sg8" || candidateM == 32, "SG8 private variant requires candidateM32");
  CommandGraph expansion;
  std::array<MetalBuffer, 3> expandedPlanes;
  uint64_t expandedScratchPlannedBytes = 0;
  if (variant == "expanded") {
    constexpr uint64_t logicalPlaneBytes = uint64_t(512) * 2560 * 640 * 2;
    expandedScratchPlannedBytes = 3 * ((logicalPlaneBytes + kGuardBytes + 16383) & ~uint64_t(16383));
    auto expansionAdmission = governor.tryReserve(expandedScratchPlannedBytes);
    require(bool(expansionAdmission), "MemoryGovernor refused expanded BF16 scratch before allocation");
    const auto immutable = store.immutableWeightBuffers();
    const auto ranks = immutable[layer * 2 + 1];
    uint32_t plane = 0;
    for (const char *role : {"gate_proj", "up_proj", "down_proj"}) {
      const auto &p = weights.projection(prefixFor(layer) + "." + role);
      const uint64_t bytes = uint64_t(512) * p.inputSize * p.outputSize * 2;
      expandedPlanes[plane] = guarded(backend, bytes, guards);
      const Prefill4KMoEExpandedParams params{p.inputSize, p.outputSize, 512, 0,
          p.weightRowStrideBytes, p.weightExpertStrideBytes, p.parameterRowStrideBytes, p.parameterExpertStrideBytes};
      expansion.add("prefill4k_moe_expand_q4", {p.weights->buffer, p.scales->buffer, p.biases->buffer,
          ranks, scratch[1].buckets.offsets, expandedPlanes[plane], diagnostic[1]}, params,
          {1, 512, 1}, {256, 1, 1});
      ++plane;
    }
    expansionAdmission->commit();
  }
  for (auto &dispatch : commands[1]) {
    const auto &name = dispatch.pipelineName;
    const bool hit = name.starts_with("flash_int8_expert_store_gate_up_m") && !name.starts_with("flash_int8_expert_store_gate_up_miss_");
    const bool downHit = name.starts_with("flash_int8_expert_store_down_scatter_");
    const bool miss = name.starts_with("flash_int8_expert_store_gate_up_miss_direct_") || name.starts_with("flash_int8_expert_store_down_miss_direct_");
    if ((variant == "static" && (hit || downHit)) || ((variant == "paired" || variant == "expanded") && miss) || (variant == "sg8" && (hit || downHit || miss)))
      dispatch.pipelineName.replace(0, std::strlen("flash_int8_expert_store_"), "prefill4k_moe_" + variant + "_");
    if (variant == "sg8" && (hit || downHit || miss)) dispatch.threadsPerThreadgroup.x = 256;
    if (variant == "expanded" && miss) {
      const bool gate = dispatch.pipelineName.find("gate_up") != std::string::npos;
      dispatch.buffers[1].buffer = expandedPlanes[gate ? 0 : 2];
      if (gate) dispatch.buffers[4].buffer = expandedPlanes[1];
    }
  }
  if (variant == "expanded") {
    auto position = std::find_if(commands[1].begin(), commands[1].end(), [](const ComputeDispatch &d) {
      return d.pipelineName.starts_with("flash_int8_expert_store_gate_up_m");
    });
    require(position != commands[1].end(), "no current hit producer before expansion insertion");
    commands[1].insert(position, expansion.dispatches().begin(), expansion.dispatches().end());
  }
'''
    oracle = replace(oracle, point, transform + point)
    oracle = replace(oracle, "scratch[1], diagnostic[1], rows, tile);", "scratch[1], diagnostic[1], rows, static_cast<FlashMoEBlockedTile>(candidateM));")
    oracle = oracle.replace("backend.submitCommand(graphs[0].dispatches())", "backend.submitCommand(commands[0])")
    oracle = oracle.replace("backend.submitCommand(graphs[1].dispatches())", "backend.submitCommand(commands[1])")
    oracle = oracle.replace("backend.submitCommand(graphs[which].dispatches())", "backend.submitCommand(commands[which])")
    oracle = replace(oracle, "checkBuckets(scratch[1], packed, jobs);", "checkBuckets(scratch[1], packed, candidateJobs);", 2)
    oracle = replace(oracle, "  if (misses == routes) require(!activation.mismatches", "  require(!activation.mismatches")
    oracle = replace(oracle, "\"all-miss chain must be bit-exact to current Q4x8/DirectA control\"", "\"candidate entire BF16 chain must be bit-exact to current saved INT8/Direct-A control\"")
    oracle = replace(oracle, "  (void)compare(output[0], output[1], uint64_t{rows} * 2560, maxL2, minCosine);", "  require(compare(output[0], output[1], uint64_t{rows} * 2560, maxL2, minCosine).mismatches == 0, \"timed output must remain BF16 exact\");")
    oracle = replace(oracle, "names(out, graphs[0].dispatches())", "names(out, commands[0])")
    oracle = replace(oracle, "names(out, graphs[1].dispatches())", "names(out, commands[1])")
    oracle = replace(oracle, "CommandGraph graph;\n  MetalBuffer output", "CommandGraph graph;\n  std::vector<ComputeDispatch> commands;\n  MetalBuffer output")
    oracle = replace(oracle, "CommandGraph graph;\n  std::vector<ComputeDispatch>", "CommandGraph graph, expansion;\n  std::vector<ComputeDispatch>")
    oracle = replace(oracle, "backend.submitCommand(graph.dispatches())", "backend.submitCommand(commands)")
    oracle = replace(oracle, "Replay replay; replay.graph = std::move(graphs[1]);", "Replay replay; replay.graph = std::move(graphs[1]); replay.expansion = std::move(expansion); replay.commands = std::move(commands[1]);")
    oracle = replace(oracle, '<< ",\\\"pattern\\\":" << splash::json::quote(rawInput', '<< ",\\\"candidate_tile_m\\\":" << candidateM << ",\\\"expanded_scratch_planned_bytes\\\":" << expandedScratchPlannedBytes << ",\\\"candidate_variant\\\":" << splash::json::quote(variant)\n      << ",\\\"pattern\\\":" << splash::json::quote(rawInput')
    oracle = replace(oracle, '  functions.push_back(direct ? "flash_moe_direct_a_prepare_down"', r'''  const uint32_t candidateM = envNumber("PREFILL4K_MOE_TILE", m, 64);
  const char *rawVariant = std::getenv("PREFILL4K_MOE_VARIANT");
  const std::string variant = rawVariant ? rawVariant : "static";
  const std::string candidateSuffix = "_m" + std::to_string(candidateM) + (candidateM == 64 ? "_n64_sg8" : "_n64");
  if (variant != "production") {
    const std::string prefix = "prefill4k_moe_" + variant + "_";
    for (const char *phase : (variant == "static" || variant == "sg8" ? std::array<const char *, 2>{"gate_up", "down_scatter"} : std::array<const char *, 2>{"gate_up_miss_direct", "down_miss_direct"}))
      functions.push_back(prefix + phase + candidateSuffix);
    if (variant == "sg8") {
      functions.push_back(prefix + "gate_up_miss_direct" + candidateSuffix);
      functions.push_back(prefix + "down_miss_direct" + candidateSuffix);
    }
  }
  if (variant == "expanded") functions.push_back("prefill4k_moe_expand_q4");
  functions.push_back(direct ? "flash_moe_direct_a_prepare_down"''')
    oracle = replace(oracle, 'const uint32_t requested = name.ends_with(suffix) ? (m == 64 ? 256 : 128) : 256;', 'const bool candidate = name.starts_with("prefill4k_moe_");\n    const uint32_t requested = name == "prefill4k_moe_expand_q4" ? 256 : candidate ? (candidateM == 64 || variant == "sg8" ? 256 : 128) : name.ends_with(suffix) ? (m == 64 ? 256 : 128) : 256;')
    oracle = oracle.replace("flash-production-selected-int8-expert-store-oracle-v1", "prefill4k-moe-current-store-exact-comparison-v1")
    oracle = oracle.replace('\\\"numerical_alternative\\\":true', '\\\"numerical_alternative\\\":false')
    (destination / "oracle.mm").write_text(oracle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    generate(parser.parse_args().destination)
