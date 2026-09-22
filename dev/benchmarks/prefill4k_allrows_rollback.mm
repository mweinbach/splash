// CPU-build only by delegated agents. Root owns all model/GPU execution.
#define main splash_allrows_rollback_unused_wide_main
#include "prefill4k_wide_state.mm"
#undef main
#include "flash/FlashInt8ExpertStoreMetadata.hpp"

namespace splash::flash {
class FlashAllrowsRollbackOracle final {
public:
  using A = FlashDeepPrefixOracle;
  struct BufferPlane final { std::string label; A::Type type; metal::MetalBuffer buffer; uint64_t liveBytes; };
  static std::vector<BufferPlane> buffers(const FlashRequestState &request) {
    if (!request.impl_) throw std::logic_error("rollback state is absent");
    const auto &s = *request.impl_; std::vector<BufferPlane> out;
    const auto add = [&](std::string name, A::Type type, const metal::MetalBuffer &buffer, uint64_t bytes) {
      if (!buffer || !buffer.contents() || bytes > buffer.sizeBytes()) throw std::logic_error("invalid rollback plane");
      out.push_back({std::move(name), type, buffer, bytes});
    };
    for (uint32_t layer = 0; layer < 48; ++layer) {
      const std::string p = "layer." + std::to_string(layer) + ".";
      if (s.gdn[layer].recurrent) {
        add(p + "gdn.convolution", A::Type::BF16, s.gdn[layer].convolution, flashGDNConvolutionLaneBytes());
        add(p + "gdn.recurrent", A::Type::F32, s.gdn[layer].recurrent, flashGDNRecurrentLaneBytes());
      } else {
        const auto &q = s.qsa[layer];
        add(p + "qsa.keys", A::Type::BF16, q.keys, s.length * 1024);
        add(p + "qsa.values", A::Type::BF16, q.values, s.length * 1024);
        add(p + "qsa.raw_index", A::Type::BF16, q.rawIndexKeys, s.length * 256);
        add(p + "qsa.complete_pooled_keys", A::Type::BF16, q.pooledKeys, (s.length / 4) * 256);
        add(p + "qsa.positions", A::Type::I64, q.indexPositions, s.length * 8);
      }
    }
    add("ple.token_history", A::Type::I64, s.pleHistory, 16);
    add("ple.convolution", A::Type::BF16, s.pleConvolution, uint64_t{9} * 10240 * 2);
    if (out.size() != 134) throw std::logic_error("rollback plane inventory changed");
    return out;
  }
  static uint64_t disjoint(const FlashForward &trunk, std::span<const FlashRequestState *const> requests) {
    struct Range { uintptr_t begin, end; };
    std::vector<Range> ranges;
    for (size_t i = 0; i < requests.size(); ++i) {
      const auto &r = *requests[i];
      if (!r.impl_ || r.impl_->owner != trunk.impl_->owner) throw std::logic_error("rollback state is not shared-trunk-owned");
      for (size_t j = 0; j < i; ++j)
        if (r.impl_->identity == requests[j]->impl_->identity) throw std::logic_error("rollback request identities alias");
      for (const auto &p : buffers(r)) {
        const auto begin = reinterpret_cast<uintptr_t>(p.buffer.contents());
        if (p.buffer.sizeBytes() > UINTPTR_MAX - begin) throw std::logic_error("rollback pointer range overflows");
        ranges.push_back({begin, begin + p.buffer.sizeBytes()});
      }
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range &a, const Range &b) { return a.begin < b.begin; });
    for (size_t i = 1; i < ranges.size(); ++i)
      if (ranges[i].begin < ranges[i - 1].end) throw std::logic_error("persistent Shared allocation ranges alias");
    return ranges.size();
  }
  static A::Snapshot snapshot(const FlashForward &trunk, const FlashRequestState &request) {
    std::lock_guard lock(trunk.impl_->mutex); const auto &s = *request.impl_;
    if (s.owner != trunk.impl_->owner) throw std::logic_error("rollback snapshot owner differs");
    A::Snapshot out{s.length, s.capacity, s.poisoned, s.pendingVerification, {}, {}, {}};
    for (const auto &p : buffers(request)) {
      A::Plane plane{p.label, p.type, p.liveBytes, std::vector<uint8_t>(p.buffer.sizeBytes())};
      std::memcpy(plane.bytes.data(), p.buffer.contents(), plane.bytes.size()); out.state.push_back(std::move(plane));
    }
    for (uint32_t layer = 0; layer < 48; ++layer) {
      const std::string p = "layer." + std::to_string(layer) + ".";
      if (s.gdn[layer].recurrent) {
        out.geometry.emplace_back(p + "gdn.convolution_lane_stride", s.gdn[layer].convolutionLaneStrideBytes);
        out.geometry.emplace_back(p + "gdn.recurrent_lane_stride", s.gdn[layer].recurrentLaneStrideBytes);
      } else out.geometry.emplace_back(p + "qsa.capacity", s.qsa[layer].capacity);
    }
    return out;
  }
  static void restore(FlashForward &trunk, FlashRequestState &request, const A::Snapshot &source) {
    std::lock_guard lock(trunk.impl_->mutex);
    if (!request.impl_ || request.impl_->owner != trunk.impl_->owner || source.capacity != request.capacity() ||
        source.poisoned || source.pending || trunk.impl_->pendingRows)
      throw std::logic_error("CPU restore requires a healthy same-owner idle boundary");
    const auto destination = buffers(request);
    if (destination.size() != source.state.size()) throw std::logic_error("rollback clone inventory differs");
    for (size_t i = 0; i < destination.size(); ++i) {
      const auto &to = destination[i]; const auto &from = source.state[i];
      if (to.label != from.label || to.type != from.type || to.buffer.sizeBytes() != from.bytes.size())
        throw std::logic_error("rollback clone geometry differs");
      std::memcpy(to.buffer.contents(), from.bytes.data(), from.bytes.size());
    }
    // Keep the request's distinct identity and its shared trunk owner.
    request.impl_->length = source.length; request.impl_->poisoned = false; request.impl_->pendingVerification = false;
  }
  struct Outputs final {
    uint32_t rows = 0, logitRows = 0;
    std::vector<A::Plane> planes;
    std::vector<FlashGreedyGPURowResult> greedy{};
  };
  static Outputs outputs(const FlashForward &trunk, const FlashForwardResult &result, uint32_t rows) {
    std::lock_guard lock(trunk.impl_->mutex); Outputs out{rows, result.logitRows, {}};
    const auto add = [&](std::string name, const metal::MetalBuffer &buffer, uint64_t bytes) {
      if (!buffer || !buffer.contents() || bytes > buffer.sizeBytes()) throw std::logic_error("rollback output extent differs");
      A::Plane p{std::move(name), A::Type::BF16, bytes, std::vector<uint8_t>(bytes)};
      std::memcpy(p.bytes.data(), buffer.contents(), bytes); out.planes.push_back(std::move(p));
    };
    add("last.logits", result.logitsBF16, uint64_t{result.logitRows} * 248320 * 2);
    add("last.pre_mixer_hyper", result.hiddenBF16, uint64_t{rows} * 10240 * 2);
    add("last.post_mixer_normalized_hidden", trunk.impl_->bf(Scratch::Mixed, rows, 2560), uint64_t{rows} * 2560 * 2);
    if (result.greedyResultsU32) {
      if (!result.greedyResultsU32.contents() || result.greedyRows != result.logitRows ||
          result.greedyResultsU32.sizeBytes() < uint64_t{result.greedyRows} * sizeof(FlashGreedyGPURowResult))
        throw std::logic_error("borrowed greedy output contract differs");
      out.greedy.resize(result.greedyRows);
      std::memcpy(out.greedy.data(), result.greedyResultsU32.contents(), out.greedy.size() * sizeof(FlashGreedyGPURowResult));
    }
    return out; // all verify rows copied before commit or another request
  }
  static std::vector<A::Plane> selected(const Outputs &output, uint32_t row) {
    if (row >= output.rows || output.planes.size() != 3) throw std::logic_error("rollback selected output row differs");
    std::vector<A::Plane> out;
    const std::array<uint32_t, 3> widths{248320, 10240, 2560};
    for (size_t i = 0; i < 3; ++i) {
      const uint64_t bytes = uint64_t{widths[i]} * 2, offset = uint64_t{i == 0 && output.logitRows == 1 ? 0 : row} * bytes;
      const auto &source = output.planes[i];
      if (offset + bytes > source.bytes.size()) throw std::logic_error("rollback owned output selection is outside copy");
      A::Plane p{source.label, source.type, bytes, std::vector<uint8_t>(bytes)};
      std::memcpy(p.bytes.data(), source.bytes.data() + offset, bytes); out.push_back(std::move(p));
    }
    return out;
  }
};
} // namespace splash::flash

namespace splash::allrows_rollback_oracle {
using namespace flash;
namespace common = splash::wide_state_oracle;
using R = FlashAllrowsRollbackOracle;
using A = FlashDeepPrefixOracle;
constexpr const char *kDerivative = "2e858faa201554642a443d48d38b7302159fe1a811c028a895454cc8ce5c073d";
constexpr uint32_t kCapacity = 16384, kRows = 2048;
bool primaryExact(const common::Summary &s) {
  return !s.stateLiveDifferingPlanes && !s.outputDifferingPlanes && !s.metadataDifferences && !s.nonfinitePlanes;
}
void summaryJSON(std::ostream &out, const common::Summary &s) {
  out << "{\"checkpoints\":" << s.checkpoints << ",\"state_plane_comparisons\":" << s.statePlanes
      << ",\"output_plane_comparisons\":" << s.outputPlanes << ",\"allocated_bytes_compared\":" << s.allocatedBytes
      << ",\"live_bytes_compared\":" << s.liveBytes << ",\"live_differing_planes\":" << s.stateLiveDifferingPlanes
      << ",\"allocated_differing_planes\":" << s.statePhysicalDifferingPlanes << ",\"output_differing_planes\":" << s.outputDifferingPlanes
      << ",\"metadata_differences\":" << s.metadataDifferences << ",\"nonfinite_plane_comparisons\":" << s.nonfinitePlanes
      << ",\"live_state_and_output_exact_finite\":" << (primaryExact(s) ? "true" : "false")
      << ",\"all_allocated_state_and_output_exact_finite\":" << (s.exact() ? "true" : "false") << '}';
}
void terminalJSON(std::ostream &trace, uint32_t phase, const A::Snapshot &base, const A::Snapshot &trial) {
  trace << "{\"phase\":\"abort0_terminal_state_no_restore_claim\",\"base_phase\":" << phase
      << ",\"base_length\":" << base.length << ",\"terminal_length\":" << trial.length
      << ",\"poisoned\":" << (trial.poisoned ? "true" : "false") << ",\"pending\":" << (trial.pending ? "true" : "false")
      << ",\"planes\":[";
  common::require(base.state.size() == trial.state.size(), "terminal inventory differs");
  for (size_t i = 0; i < base.state.size(); ++i) {
    const auto &b = base.state[i], &t = trial.state[i]; if (i) trace << ',';
    trace << "{\"plane\":" << json::quote(t.label) << ",\"dtype\":" << json::quote(common::typeName(t.type))
        << ",\"terminal_live_bytes\":" << t.liveBytes << ",\"base_live_bytes\":" << b.liveBytes
        << ",\"terminal_allocated_sha256\":" << json::quote(common::digest(t.bytes.data(), t.bytes.size()))
        << ",\"base_allocated_sha256\":" << json::quote(common::digest(b.bytes.data(), b.bytes.size())) << ",\"allocated_difference_from_base\":";
    common::writeDifference(trace, t.type, common::compareRange(t.type, b.bytes.data(), t.bytes.data(), t.bytes.size()));
    trace << '}';
  }
  trace << "]}\n"; trace.flush();
}
void cpuSelfTest() {
  common::cpuSelfTest();
  const R::Outputs source{4, 4, {
    {"last.logits", A::Type::BF16, uint64_t{4} * 248320 * 2, std::vector<uint8_t>(uint64_t{4} * 248320 * 2)},
    {"last.pre_mixer_hyper", A::Type::BF16, uint64_t{4} * 10240 * 2, std::vector<uint8_t>(uint64_t{4} * 10240 * 2)},
    {"last.post_mixer_normalized_hidden", A::Type::BF16, uint64_t{4} * 2560 * 2, std::vector<uint8_t>(uint64_t{4} * 2560 * 2)}}};
  for (uint32_t phase = 0; phase < 4; ++phase) for (uint32_t retained : {0u, 1u, 3u, 4u}) {
    common::require(2048 + phase + 4 + 7 < kCapacity, "rollback CPU capacity differs");
    if (retained) {
      const auto selected = R::selected(source, retained - 1);
      common::require(selected.size() == 3 && selected[0].bytes.size() == 248320 * 2 && selected[1].bytes.size() == 10240 * 2,
                      "rollback CPU selected prefix output shape differs");
    }
  }
  common::Summary stale; stale.statePhysicalDifferingPlanes = 1;
  common::require(primaryExact(stale) && !stale.exact(), "rollback CPU stale tail must remain reported separately");
  stale.stateLiveDifferingPlanes = 1;
  common::require(!primaryExact(stale), "rollback CPU live differences cannot pass primary gate");
  std::cout << "{\"schema\":\"splash-private-allrows-rollback-cpu-v1\",\"passed\":true,\"gpu_executed\":false,\"model_loaded\":false,\"payload_bytes_read\":0,\"retained_cases\":[0,1,3,4],\"base_pool_phases\":[0,1,2,3],\"checks\":[\"16_case_capacity_matrix\",\"owned_selected_verify_rows\",\"stale_physical_tail_separate\",\"live_difference_reject\",\"abort0_has_no_restore_claim\"]}\n";
}
void earlyPolicy() {
  for (const char *flag : {"SPLASH_FLASH_ALLROWS_FULL512_TARGET", "SPLASH_FLASH_PLE_SSD_STREAMING", "SPLASH_FLASH_BLOCKED_MOE",
                          "SPLASH_FLASH_MOE_DIRECT_A", "SPLASH_FLASH_MOE_Q4X8", "SPLASH_FLASH_PREFILL_DENSE_TILES"})
    common::require(common::enabled(flag), "rollback requires explicit allrow Full512/current Dense1 policy");
  common::require(!common::enabled("SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE") && !common::enabled("SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT"),
                  "rollback allrow policy forbids original/idle residency");
  common::require(!std::getenv("SPLASH_FLASH_HOT_EXPERT_PLAN"), "rollback forbids duplicate expert cache policy");
  const char *directory = std::getenv("SPLASH_FLASH_INT8_EXPERT_STORE"); common::require(directory && *directory, "rollback Full512 path missing");
  const auto metadata = loadFlashInt8ExpertStoreMetadata(directory,
      "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
      "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0", NormConvention::OnePlusWeight);
  common::require(metadata.identitySha256 == "ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1", "rollback certified Full512 manifest differs");
  for (const auto &layer : metadata.layers) {
    common::require(layer.selectedIDs.size() == 512, "rollback requires complete Full512 inventory");
    for (uint32_t id = 0; id < 512; ++id) common::require(layer.selectedIDs[id] == id, "rollback expert ranks differ");
  }
}
int run(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << "usage: rollback-oracle METALLIB PACKAGE EXACT_CODE2048_TOKENS_JSON REPORT_JSON | --cpu-self-test\n"
      "One Full512 shared trunk/store;3 physically disjoint requests; base2048+phase0..3, verify4 retained0(abort)/1/3/4,7 identical appends.\n"
      "Primary branches have same verify4 shape and different discarded suffixes; C forward(rows=k) is a separate producer diagnostic.\n"
      "Exit2 retains exact/numeric primary differences; physical stale QSA tails do not imply live-prefix contamination.\n"
      "Compilation/help/CPU-self-test load no model, read no payload and submit no GPU work. Root alone runs GPU.\n"; return 0;
  }
  common::require(argc == 5, "rollback arguments invalid; run --help");
  common::require(!std::filesystem::exists(argv[4]) && !std::filesystem::exists(std::string(argv[4]) + ".checkpoints.jsonl"), "choose fresh rollback report");
  const auto tokens = common::loadTokens(argv[3]); common::require(tokens.size() == 2048, "rollback requires exact code2048 fixture");
  earlyPolicy(); // metadata only and fail closed before backend creation
  std::ofstream trace(std::string(argv[4]) + ".checkpoints.jsonl"); common::require(bool(trace), "cannot open rollback trace");
  trace << std::setprecision(std::numeric_limits<double>::max_digits10);
  metal::MetalBackend backend(argv[1]); const auto weights = FlashWeights::load(backend, argv[2]);
  const auto original = weights.pleSSDStorageStats();
  common::require(original.allRowsFull512Target && original.targetOriginalDiskTensorCount == 432 &&
                  original.gpuMappedBytes == 6370164736ULL, "rollback original target omission differs");
  const uint64_t physical = NSProcessInfo.processInfo.physicalMemory, reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
  engine::MemoryGovernor governor(backend, physical - reserve, reserve);
  const uint64_t snapshots = 6 * FlashForward::requestStateBytes(kCapacity) + (128ULL << 20);
  const uint64_t planned = common::forwardPlan(weights, kRows) - FlashForward::workspacePlannedBytes(kCapacity, kRows, 0) +
      FlashForward::workspacePlannedBytes(kCapacity, kRows, 4) +
      2 * FlashForward::requestStateBytes(kCapacity) + snapshots;
  auto reservation = governor.tryReserve(planned); common::require(bool(reservation), "rollback shared arena/state/snapshot admission denied");
  FlashForward trunk(backend, weights, kCapacity, kRows, 4);
  const auto *store = trunk.batchInt8ExpertStore();
  common::require(store && store->numericalIdentitySha256() == kDerivative && store->mappedBytes() == 121173442560ULL,
                  "rollback actual target derivative differs");
  common::require(trunk.workspaceBytes() + 3 * FlashForward::requestStateBytes(kCapacity) + snapshots <= planned, "rollback actual arena exceeds admission");
  auto a = trunk.createState(), b = trunk.createState(), c = trunk.createState();
  const std::array<const FlashRequestState *, 3> requests{&a, &b, &c};
  uint64_t independenceChecks = R::disjoint(trunk, requests);
  metal::ResidencyLease residency;
  if (common::enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
    const auto operands = trunk.cachedOperandsOnly();
    if (!operands.empty()) residency = backend.requestWeightResidency(operands, "rollback one shared Full512 saved operand store");
  }
  reservation->commit();
  const auto initialResult = trunk.forward(c, tokens, false, true);
  const auto initialOutputs = R::outputs(trunk, initialResult, 2048);
  auto initialBase = R::snapshot(trunk, c); initialBase.outputs = R::selected(initialOutputs, 2047);
  common::Summary primary, producerDiagnostic, guards, canceledReset;
  uint64_t cases = 0, abortCases = 0, singletonAppends = 0;
  const std::array<uint32_t, 7> future{tokens[31], 198, tokens[32], 248044, 198, 248046, tokens[33]};
  for (uint32_t phase = 0; phase < 4; ++phase) {
    R::restore(trunk, c, initialBase);
    for (uint32_t p = 0; p < phase; ++p) {
      const auto result = trunk.forward(c, std::span(&tokens[20 + p], 1), false, true);
      const auto owned = R::outputs(trunk, result, 1); (void)owned;
    }
    auto base = R::snapshot(trunk, c);
    for (const uint32_t retained : {0u, 1u, 3u, 4u}) {
      ++cases; R::restore(trunk, a, base); R::restore(trunk, b, base); R::restore(trunk, c, base);
      independenceChecks += R::disjoint(trunk, requests);
      std::array<uint32_t, 4> trialA{tokens[41], tokens[42], 248044, 248046}, trialB = trialA;
      if (retained < 4) {
        for (uint32_t row = retained; row < 4; ++row) trialB[row] = row % 2 ? 248044 : 248046;
        if (trialB[retained] == trialA[retained]) trialB[retained] = tokens[43 + retained];
      }
      const std::string label = "base_phase" + std::to_string(phase) + ".retain" + std::to_string(retained) + ".";
      const auto resultA = trunk.verify(a, trialA); const auto outputA = R::outputs(trunk, resultA, 4);
      trace << "{\"phase\":" << json::quote(label + "owned_verify_a_greedy") << ",\"rows\":[";
      for (size_t row = 0; row < outputA.greedy.size(); ++row) {
        if (row) trace << ',';
        const auto &g = outputA.greedy[row];
        trace << "{\"row\":" << row << ",\"token\":" << g.token << ",\"rank\":" << g.rank << ",\"errors\":" << g.errors << ",\"reserved\":" << g.reserved << '}';
      }
      trace << "]}\n";
      if (!retained) {
        ++abortCases;
        {
          const auto before = R::snapshot(trunk, a); bool zeroRejected = false;
          try { (void)trunk.commitVerify(a, 0); } catch (const std::invalid_argument &) { zeroRejected = true; }
          common::require(zeroRejected, "zero retain must reject rather than claim rollback");
          const auto after = R::snapshot(trunk, a);
          common::compareCheckpoint(trace, label + "commit0_rejected_unmutated", 0, before, after, guards);
        }
        {
          const auto before = R::snapshot(trunk, b); bool peerRejected = false;
          try { (void)trunk.forward(b, std::span(&future[0], 1), false, true); } catch (const std::logic_error &) { peerRejected = true; }
          common::require(peerRejected, "shared tape must block peer trunk use");
          const auto after = R::snapshot(trunk, b);
          common::compareCheckpoint(trace, label + "peer_blocked_unmutated", 0, before, after, guards);
        }
        trunk.abortVerify(a);
        const auto terminal = R::snapshot(trunk, a);
        common::require(terminal.poisoned && !terminal.pending && terminal.length == base.length + 4, "abort0 terminal semantics differ");
        terminalJSON(trace, phase, base, terminal);
        bool poisonRejected = false;
        try { (void)trunk.forward(a, std::span(&future[0], 1), false, true); } catch (const std::invalid_argument &) { poisonRejected = true; }
        common::require(poisonRejected, "aborted request must not resume");
        a = FlashRequestState(); a = trunk.createState(); R::restore(trunk, a, base);
        independenceChecks += R::disjoint(trunk, requests);
        const auto resetA = R::snapshot(trunk, a), untouchedB = R::snapshot(trunk, b);
        common::compareCheckpoint(trace, label + "fresh_healthy_reset_vs_untouched", 0, resetA, untouchedB, canceledReset);
      } else {
        (void)trunk.commitVerify(a, retained);
        auto stateA = R::snapshot(trunk, a); stateA.outputs = R::selected(outputA, retained - 1);
        const auto resultB = trunk.verify(b, trialB); const auto outputB = R::outputs(trunk, resultB, 4);
        (void)trunk.commitVerify(b, retained);
        auto stateB = R::snapshot(trunk, b); stateB.outputs = R::selected(outputB, retained - 1);
        common::require(stateA.length == base.length + retained && stateB.length == stateA.length, "retained logical prefix differs");
        common::compareCheckpoint(trace, label + "same_verify4_shape_discarded_suffix", retained, stateA, stateB, primary);
        const auto resultC = trunk.forward(c, std::span(trialA).first(retained), false, true);
        const auto outputC = R::outputs(trunk, resultC, retained);
        auto stateC = R::snapshot(trunk, c); stateC.outputs = R::selected(outputC, retained - 1);
        common::compareCheckpoint(trace, label + "forward_rowsk_producer_diagnostic", retained, stateC, stateA, producerDiagnostic);
      }
      for (uint32_t step = 0; step < future.size(); ++step) {
        const auto input = std::span(&future[step], 1);
        const auto ar = trunk.forward(a, input, false, true); const auto ao = R::outputs(trunk, ar, 1);
        auto sa = R::snapshot(trunk, a); sa.outputs = R::selected(ao, 0);
        const auto br = trunk.forward(b, input, false, true); const auto bo = R::outputs(trunk, br, 1);
        auto sb = R::snapshot(trunk, b); sb.outputs = R::selected(bo, 0);
        common::compareCheckpoint(trace, label + "same_singleton_future_append", step, sa, sb, retained ? primary : canceledReset);
        if (retained) {
          const auto cr = trunk.forward(c, input, false, true); const auto co = R::outputs(trunk, cr, 1);
          auto sc = R::snapshot(trunk, c); sc.outputs = R::selected(co, 0);
          common::compareCheckpoint(trace, label + "producer_reference_future_append", step, sc, sa, producerDiagnostic);
        }
        ++singletonAppends;
      }
      std::cout << "allrows rollback phase=" << phase << " retained=" << retained << " primary_live_exact_so_far=" << primaryExact(primary) << '\n' << std::flush;
    }
    // Destruction/cancellation discards a trial. The next healthy peer forward
    // releases the expired shared tape; it never promotes the destroyed state.
    R::restore(trunk, a, base); R::restore(trunk, b, base); R::restore(trunk, c, base);
    const std::array<uint32_t, 4> cancel{tokens[50], 248044, tokens[51], 248046};
    const auto trial = trunk.verify(a, cancel); const auto ownedCancel = R::outputs(trunk, trial, 4); (void)ownedCancel;
    a = FlashRequestState();
    for (uint32_t step = 0; step < future.size(); ++step) {
      const auto input = std::span(&future[step], 1);
      const auto br = trunk.forward(b, input, false, true); const auto bo = R::outputs(trunk, br, 1);
      auto sb = R::snapshot(trunk, b); sb.outputs = R::selected(bo, 0);
      const auto cr = trunk.forward(c, input, false, true); const auto co = R::outputs(trunk, cr, 1);
      auto sc = R::snapshot(trunk, c); sc.outputs = R::selected(co, 0);
      common::compareCheckpoint(trace, "destroy_cancel_phase" + std::to_string(phase) + ".healthy_peer_vs_untouched_singleton", step, sc, sb, canceledReset);
    }
    a = trunk.createState(); R::restore(trunk, a, base); independenceChecks += R::disjoint(trunk, requests);
  }
  std::ofstream report(argv[4]); common::require(bool(report), "cannot open rollback final report");
  const bool primaryPass = primaryExact(primary) && guards.exact() && canceledReset.exact();
  report << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "{\"schema\":\"splash-private-allrows-rollback-continuation-v1\",\"gpu_executed\":true,\"completed\":true"
      << ",\"one_shared_flash_forward_and_full512_store\":true,\"request_state_count\":3,\"persistent_planes_per_request\":134"
      << ",\"physically_disjoint_shared_range_checks\":" << independenceChecks << ",\"all_borrowed_verify_and_forward_outputs_copied_immediately\":true"
      << ",\"target_numerical_derivative_sha256\":" << json::quote(store->numericalIdentitySha256())
      << ",\"full512_manifest_sha256\":" << json::quote(store->identitySha256()) << ",\"full512_mapped_bytes\":" << store->mappedBytes()
      << ",\"original_target_gpu_omitted\":true,\"capacity\":" << kCapacity << ",\"maximum_rows\":" << kRows << ",\"verify_rows\":4"
      << ",\"base_pool_phases\":[0,1,2,3],\"retained_cases\":[0,1,3,4],\"case_count\":" << cases << ",\"abort0_case_count\":" << abortCases
      << ",\"identical_append_steps_per_case\":7,\"primary_singleton_append_checks\":" << singletonAppends
      << ",\"primary_same_shape_live_state_outputs_exact_finite\":" << (primaryPass ? "true" : "false")
      << ",\"physical_tail_policy\":\"Discarded QSA suffixes remain allocated and may differ; every allocated word is still reported separately. Only active-prefix equality is a primary contamination check.\""
      << ",\"abort0_policy\":\"Terminal poison, trial length retained, no rollback-to-base claim; subsequent checks use newly allocated healthy request restored from owned CPU base bytes.\""
      << ",\"producer_reference_policy\":\"C consumes retained tokens with forward(rows=k); R1/R3 versus R4 producer differences are diagnostic and do not prove rollback failure or a universal exact invariant.\""
      << ",\"primary_same_verify_shape_and_future_append\":"; summaryJSON(report, primary);
  report << ",\"unmutated_forward_rowsk_diagnostic\":"; summaryJSON(report, producerDiagnostic);
  report << ",\"invalid_and_pending_guards\":"; summaryJSON(report, guards);
  report << ",\"abort_fresh_reset_and_destroy_cancel_healthy_peer\":"; summaryJSON(report, canceledReset);
  report << ",\"planned_reservation_bytes\":" << planned << ",\"cpu_owned_snapshot_budget_bytes\":" << snapshots
      << ",\"actual_trunk_workspace_bytes\":" << trunk.workspaceBytes() << ",\"metallib_sha256\":" << json::quote(common::fileDigest(argv[1]))
      << ",\"source_identity_sha256\":" << json::quote(weights.sourceIdentity()) << ",\"kernel_routes\":" << json::quote(trunk.kernelRoutes())
      << ",\"tokens_u32le_sha256\":" << json::quote(common::digest(tokens.data(), tokens.size() * 4))
      << ",\"timing_scope\":\"No performance claim; state snapshots and CPU hashing interleave synchronous trunk/restore commands; no HTTP or MTP instantiated\""
      << ",\"environment\":"; common::writeEnvironment(report); report << "}\n"; common::require(bool(report), "rollback report write failed");
  std::cout << "allrows rollback completed; primary_live_exact=" << primaryPass << " report=" << argv[4] << '\n';
  return primaryPass ? 0 : 2;
}
} // namespace splash::allrows_rollback_oracle
int main(int argc, char **argv) {
  @autoreleasepool {
    try { return splash::allrows_rollback_oracle::run(argc, argv); }
    catch (const std::exception &error) { std::cerr << "allrows rollback oracle error: " << error.what() << '\n'; return 1; }
  }
}
