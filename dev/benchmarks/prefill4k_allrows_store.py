"""Private Full512 all-target-row store transformation; no inference on import."""
from __future__ import annotations

POLICY = "private-allrows-full512-signed-i8-f32-late-row-scale-bf16-dots-m16tight-small-m32m64-large-no-target-q4gpu-v1"


def replace(text: str, before: str, after: str, count: int = 1) -> str:
    if text.count(before) != count:
        raise RuntimeError(f"All-row store source drift: {before!r}")
    return text.replace(before, after)


def transform(relative: str, text: str) -> str:
    if relative == "runtime/flash/FlashInt8ExpertStore.hpp":
        text = replace(text, "private-i8-row-f32scale-bf16activation-f32accum-whole-k-inventory32to512-full512-no-q4miss-dispatch-v1", POLICY)
        text = replace(text, "  uint64_t full_inventory_graph_calls = 0;", """  uint64_t full_inventory_graph_calls = 0;
  // Physical rows>=256 only, excluding tiny prefill/decode/verifier graphs.
  uint64_t large_row_gate_up_graph_calls = 0, large_row_gate_up_graph_rows = 0;
  uint64_t large_row_down_graph_calls = 0, large_row_down_graph_rows = 0;
  uint64_t large_row_encoded_hit_dispatches = 0, large_row_encoded_miss_dispatches = 0;
  uint64_t large_row_full_inventory_graph_calls = 0;""")
        return replace(text, "  [[nodiscard]] const std::string &identitySha256() const;", "  [[nodiscard]] const std::string &identitySha256() const;\n  [[nodiscard]] const std::string &numericalIdentitySha256() const;")
    if relative == "runtime/metal/kernels/shared/flash_int8_expert_store.metal":
        text = replace(text, "if (active > p.job_capacity || offsets[512] > p.route_capacity)", "if (active > min(p.job_capacity, p.route_capacity) || offsets[512] > p.route_capacity)")
        return replace(text, "  if (rank == UINT_MAX) return false;", "  if (rank == UINT_MAX) { if (p.stored_experts == 512 && !tid) flash_mpp_error(diag, 1u); return false; }")
    if relative != "runtime/flash/FlashInt8ExpertStore.mm":
        return text
    text = replace(text, "weights.projection(prefix +", "weights.projectionMetadata(prefix +", count=2)
    start = text.index("bool gateProducer(")
    end = text.index("} // namespace", start)
    text = text[:start] + GEOMETRY + text[end:]
    text = replace(text, "    std::array<FlashTensor, 9> sourceTensors;\n    std::array<FlashAffineProjection, 3> source;\n", "")
    text = replace(text, "  uint64_t allocated = 0;", "  uint64_t allocated = 0;\n  std::string numericalIdentity;")
    text = replace(text, "  mutable std::atomic<uint64_t> hitDispatches{0}, missDispatches{0}, fullCalls{0};", """  mutable std::atomic<uint64_t> hitDispatches{0}, missDispatches{0}, fullCalls{0};
  mutable std::atomic<uint64_t> largeGateCalls{0}, largeGateRows{0}, largeDownCalls{0}, largeDownRows{0};
  mutable std::atomic<uint64_t> largeHits{0}, largeMisses{0}, largeFullCalls{0};""")
    text = replace(text, "    else fullCalls.fetch_add(1, std::memory_order_relaxed);", """    else fullCalls.fetch_add(1, std::memory_order_relaxed);
    if (rows >= 256) {
      (down ? largeDownCalls : largeGateCalls).fetch_add(1, std::memory_order_relaxed);
      (down ? largeDownRows : largeGateRows).fetch_add(rows, std::memory_order_relaxed);
      largeHits.fetch_add(1, std::memory_order_relaxed);
      if (inventory < 512) largeMisses.fetch_add(1, std::memory_order_relaxed);
      else largeFullCalls.fetch_add(1, std::memory_order_relaxed);
    }""")
    text = replace(text, "      impl_->fullCalls.load(std::memory_order_relaxed)};", """      impl_->fullCalls.load(std::memory_order_relaxed),
      impl_->largeGateCalls.load(std::memory_order_relaxed), impl_->largeGateRows.load(std::memory_order_relaxed),
      impl_->largeDownCalls.load(std::memory_order_relaxed), impl_->largeDownRows.load(std::memory_order_relaxed),
      impl_->largeHits.load(std::memory_order_relaxed), impl_->largeMisses.load(std::memory_order_relaxed),
      impl_->largeFullCalls.load(std::memory_order_relaxed)};""")
    text = replace(text, "    sourceGeometry(weights);", """    sourceGeometry(weights);
    for (const auto &entry : metadata.layers) {
      if (entry.selectedIDs.size() != 512) fail("all-row target requires Full512 inventory");
      for (uint32_t id = 0; id < 512; ++id)
        if (entry.selectedIDs[id] != id) fail("all-row target requires canonical complete expert IDs");
    }
    const std::string derivative = std::string("splash.private-allrows-target-v1\\nsource=") +
        weights.manifestFingerprint() + "\\nstore=" + metadata.identitySha256 +
        "\\npolicy=" + kFlashInt8ExpertStoreSemantics + "\\nmtp=original-trained-bank\\n";
    numericalIdentity = hash(derivative.data(), derivative.size());""")
    copy_start = text.index("        layer.source[plane] =")
    copy_end = text.index("      }", copy_start)
    text = text[:copy_start] + text[copy_end:]
    text = replace(text, "      const auto prefix = \"language_model.model.layers.\" + std::to_string(index) + \".mlp.switch_mlp\";\n", "")
    text = replace(text, "      for (const auto &operand : layer.sourceTensors) disjoint(output, operand.buffer);\n", "")
    text = replace(text, "const std::string &FlashInt8ExpertStore::identitySha256() const { return impl_->metadata.identitySha256; }", "const std::string &FlashInt8ExpertStore::identitySha256() const { return impl_->metadata.identitySha256; }\nconst std::string &FlashInt8ExpertStore::numericalIdentitySha256() const { return impl_->numericalIdentity; }")
    text = replace(text, "  return loadFlashInt8ExpertStoreMetadata(directory, weights.sourceIdentity(),\n      weights.manifestFingerprint(), weights.normConvention()).plannedBytes;", """  const auto metadata = loadFlashInt8ExpertStoreMetadata(directory, weights.sourceIdentity(),
      weights.manifestFingerprint(), weights.normConvention());
  for (const auto &entry : metadata.layers)
    if (entry.selectedIDs.size() != 512) fail("all-row target planning requires Full512 inventory");
  return metadata.plannedBytes;""")
    start = text.index("void FlashInt8ExpertStore::addGateUp(")
    end = text.index("} // namespace splash::flash", start)
    return text[:start] + METHODS + text[end:]


GEOMETRY = r'''
// Metadata-only host validation. No source-Q4 buffers/graphs/count readback.
void allRowsScratch(const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
                    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) {
  const uint32_t m = uint32_t(tile);
  if (!rows || rows > 8192 || !selections || selections > 10 ||
      (m != 16 && m != 32 && m != 64) || rows > s.buckets.rowCapacity ||
      selections > s.buckets.selectionCapacity || uint64_t{rows} * selections > s.buckets.routeCapacity ||
      !flashMoEDirectAEnabled()) fail("all-row Full512 unsupported scratch/Direct-A geometry");
  if (m == 64 && (rows < 1024 || flashMoEBlockedTile(4096, false) != FlashMoEBlockedTile::M64N64))
    fail("all-row Full512 M64 requires the enabled wide policy");
  const uint32_t routes = rows * selections, jobs = moEBucketJobCapacity(rows, selections, m);
  if (jobs > s.buckets.jobCapacity) fail("all-row Full512 insufficient declared matrix jobs");
  requireBytes(diagnostics, 4);
  requireBytes(s.buckets.counts, 512 * 4); requireBytes(s.buckets.offsets, 513 * 4);
  requireBytes(s.buckets.routeMap, uint64_t{routes} * 4);
  requireBytes(s.buckets.canonicalToPacked, uint64_t{routes} * 4);
  requireBytes(s.buckets.jobOffsets, 513 * 4); requireBytes(s.buckets.jobCount, 4);
  requireBytes(s.buckets.tileJobs, uint64_t{jobs} * 8);
  requireBytes(s.buckets.packedInputs, uint64_t{routes + 63} * 2560 * 2);
  requireBytes(s.packedActivated, uint64_t{routes + 63} * 640 * 2);
  requireBytes(s.scatteredDown, uint64_t{routes} * 2560 * 2);
  const std::array scratch{s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics};
  for (size_t i = 0; i < scratch.size(); ++i)
    for (size_t j = i + 1; j < scratch.size(); ++j) disjoint(scratch[i], scratch[j]);
}
FlashInt8ExpertStoreParams allRowsParams(uint32_t rows, uint32_t selections,
                                        FlashMoEBlockedTile tile) {
  return {rows, selections, rows * selections,
      moEBucketJobCapacity(rows, selections, uint32_t(tile)), uint32_t(tile), 512, 0, 0};
}
uint32_t allRowsLaunch(const FlashInt8ExpertStoreParams &p) {
  return p.rows < 256 ? std::min(p.route_capacity, p.job_capacity) : p.job_capacity;
}
'''

METHODS = r'''
void FlashInt8ExpertStore::addGateUp(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  const auto &layer = impl_->layer(index);
  allRowsScratch(s, diagnostics, rows, tile, selections);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("all-row target dispatch requires Full512");
  for (const auto &buffer : {s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  const auto p = allRowsParams(rows, selections, tile);
  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;
  graph.add(pipeline("gate_up", tile), {s.buckets.packedInputs, layer.codes[0], layer.scales[0],
      layer.codes[1], layer.scales[1], layer.ranks, s.buckets.offsets, s.buckets.tileJobs,
      s.buckets.jobCount, s.packedActivated, diagnostics}, p,
      {10, allRowsLaunch(p), 1}, {threads, 1, 1});
  impl_->recordGraph(false, rows, 512);
}
void FlashInt8ExpertStore::addDownScatter(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  const auto &layer = impl_->layer(index);
  allRowsScratch(s, diagnostics, rows, tile, selections);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("all-row target dispatch requires Full512");
  for (const auto &buffer : {s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  const uint32_t routes = rows * selections;
  const FlashMoEBlockedDownParams poison{{rows, selections, 640, 2560, 512, 0, 0, 0,
      320, uint64_t{2560} * 320, 20, uint64_t{2560} * 20},
      routes, moEBucketJobCapacity(rows, selections, uint32_t(tile)), uint32_t(tile), 0};
  graph.add("flash_moe_blocked_poison_excluded_routes",
      {s.buckets.canonicalToPacked, s.scatteredDown, diagnostics}, poison,
      {10, routes, 1}, {256, 1, 1});
  graph.add("flash_moe_direct_a_prepare_down",
      {s.packedActivated, s.buckets.offsets, s.packedActivated, diagnostics},
      FlashMoEDirectAPrepareParams{routes, 640, 63, 0}, {routes + 63, 1, 1}, {256, 1, 1});
  const auto p = allRowsParams(rows, selections, tile);
  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;
  graph.add(pipeline("down_scatter", tile), {s.packedActivated, layer.codes[2], layer.scales[2],
      layer.ranks, s.buckets.offsets, s.buckets.tileJobs, s.buckets.jobCount,
      s.buckets.routeMap, s.scatteredDown, diagnostics}, p,
      {40, allRowsLaunch(p), 1}, {threads, 1, 1});
  impl_->recordGraph(true, rows, 512);
}
'''
