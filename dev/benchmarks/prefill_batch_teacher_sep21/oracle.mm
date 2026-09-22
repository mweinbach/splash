// Strict native qualification. Compilation and --help submit no GPU work.
// Build against the candidate runtime include root and candidate FlashBatchMTP
// object; root owns all model loading and GPU execution.
#define main splash_batch_teacher_attribution_unused_main
#include "../prefill4k_attribution.mm"
#undef main
#include "flash/FlashBatchMTPForward.hpp"
#include "flash/FlashMTPStateInternal.hpp"

namespace splash::flash {
struct FlashMTPTeacherPrimeOracleAccess final {
  static std::array<uintptr_t, 2> binding(const FlashMTPState &state) {
    return {reinterpret_cast<uintptr_t>(state.impl_.get()),
        state.impl_ ? reinterpret_cast<uintptr_t>(state.impl_->owner.get()) : 0};
  }
  static std::array<metal::MetalBuffer, 5> buffers(const FlashMTPState &state) {
    if (!state.impl_) throw std::invalid_argument("oracle state is uninitialized");
    const auto &q = state.impl_->qsa;
    return {q.keys, q.values, q.rawIndexKeys, q.pooledKeys, q.indexPositions};
  }
};
}

namespace {
constexpr uint32_t kCapacity = 8192, kRows = 128, kLanes = 4;
constexpr uint32_t kHyper = 10240, kVocabulary = 248320;
constexpr uint64_t kFeatureBytes = uint64_t{kHyper} * sizeof(uint16_t);
struct Counters final {
  uint64_t checks = 0, comparedBytes = 0, finiteWords = 0;
  uint64_t primeCalls = 0, proposalCalls = 0, guardCalls = 0;
  uint64_t compactPairs = 0, equalCountCalls = 0, unequalCountCalls = 0;
  double normalGPU = 0, primeGPU = 0, normalWall = 0, primeWall = 0;
};
struct Progress final {
  std::filesystem::path report;
  std::string phase = "arguments";
  bool backendInitialized = false, targetForwardCompleted = false;
  Counters counts;
  std::vector<std::string> completedCases;
  std::string sourceIdentity, metallibDigest, fixtureDigest, targetHiddenDigest;
};
void progressRecord(const Progress &progress, bool failed = false,
    const std::string &error = {}) {
  if (progress.report.empty()) return;
  const std::filesystem::path destination = progress.report.string() +
      (failed ? ".failure.json" : ".checkpoint.json");
  const std::filesystem::path temporary = destination.string() + ".writing";
  std::ofstream report(temporary, std::ios::trunc);
  require(bool(report), "cannot write batch teacher progress record");
  report << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "{\"schema\":\"splash-batch-teacher-cache-oracle-progress-v2\",\"valid\":false"
      << ",\"qualification_complete\":false,\"failed\":" << (failed ? "true" : "false")
      << ",\"phase\":" << json::quote(progress.phase) << ",\"error\":" << json::quote(error)
      << ",\"gpu_backend_initialized\":" << (progress.backendInitialized ? "true" : "false")
      << ",\"real_target_forward_completed\":" << (progress.targetForwardCompleted ? "true" : "false")
      << ",\"source_identity_sha256\":" << json::quote(progress.sourceIdentity)
      << ",\"metallib_sha256\":" << json::quote(progress.metallibDigest)
      << ",\"tokens_u32le_sha256\":" << json::quote(progress.fixtureDigest)
      << ",\"real_target_hidden_bf16_sha256\":" << json::quote(progress.targetHiddenDigest)
      << ",\"checks\":" << progress.counts.checks
      << ",\"compared_bytes\":" << progress.counts.comparedBytes
      << ",\"finite_bf16_words_checked\":" << progress.counts.finiteWords
      << ",\"completed_cache_prime_folds\":" << progress.counts.primeCalls
      << ",\"completed_compact_teacher_pairs\":" << progress.counts.compactPairs
      << ",\"completed_future_proposals\":" << progress.counts.proposalCalls
      << ",\"completed_host_guards\":" << progress.counts.guardCalls
      << ",\"equal_count_calls\":" << progress.counts.equalCountCalls
      << ",\"unequal_count_calls\":" << progress.counts.unequalCountCalls
      << ",\"normal_gpu_seconds\":" << progress.counts.normalGPU
      << ",\"prime_gpu_seconds\":" << progress.counts.primeGPU
      << ",\"normal_command_wall_seconds\":" << progress.counts.normalWall
      << ",\"prime_command_wall_seconds\":" << progress.counts.primeWall
      << ",\"completed_case_labels\":[";
  for (size_t index = 0; index < progress.completedCases.size(); ++index) {
    if (index) report << ',';
    report << json::quote(progress.completedCases[index]);
  }
  report << "]}\n";
  require(bool(report), "batch teacher progress record write failed");
  report.close(); require(bool(report), "batch teacher progress record close failed");
  std::filesystem::rename(temporary, destination);
}
void timingValid(const metal::CommandTiming &timing) {
  require(std::isfinite(timing.gpuSeconds) && timing.gpuSeconds > 0 &&
      std::isfinite(timing.wallSeconds) && timing.wallSeconds > 0 &&
      timing.gpuSeconds < timing.wallSeconds * 1.1,
      "batch teacher timing ABI is invalid");
}
void finiteBF16(const void *source, uint64_t bytes, Counters &counts) {
  require(source && bytes % sizeof(uint16_t) == 0, "batch teacher BF16 extent is invalid");
  const auto *words = static_cast<const uint16_t *>(source);
  for (uint64_t index = 0; index < bytes / sizeof(uint16_t); ++index)
    require(std::isfinite(std::bit_cast<float>(uint32_t{words[index]} << 16)),
        "batch teacher found nonfinite BF16/F32 value");
  counts.finiteWords += bytes / sizeof(uint16_t);
}
std::vector<uint16_t> copiedBF16(const metal::MetalBuffer &buffer,
    uint64_t expectedWords, Counters &counts) {
  require(buffer && buffer.contents() && buffer.sizeBytes() == expectedWords * 2,
      "batch teacher result does not have its exact real extent");
  finiteBF16(buffer.contents(), buffer.sizeBytes(), counts);
  const auto *words = static_cast<const uint16_t *>(buffer.contents());
  return {words, words + expectedWords};
}
void cacheEqual(const FlashMTPState &left, const FlashMTPState &right,
    Counters &counts) {
  require(left.logicalLength() == right.logicalLength() &&
      left.capacity() == right.capacity() && !left.poisoned() && !right.poisoned(),
      "batch teacher logical cache state differs");
  ++counts.checks;
  const auto a = FlashMTPTeacherPrimeOracleAccess::buffers(left);
  const auto b = FlashMTPTeacherPrimeOracleAccess::buffers(right);
  for (size_t plane = 0; plane < a.size(); ++plane) {
    require(a[plane] && b[plane] && a[plane].contents() && b[plane].contents() &&
        a[plane].sizeBytes() == b[plane].sizeBytes(), "batch teacher cache extent differs");
    if (std::memcmp(a[plane].contents(), b[plane].contents(), a[plane].sizeBytes()) != 0) {
      constexpr std::array<const char *, 5> names{"keys", "values", "rawIndexKeys", "pooledKeys", "indexPositions"};
      const auto *leftBytes = static_cast<const uint8_t *>(a[plane].contents());
      const auto *rightBytes = static_cast<const uint8_t *>(b[plane].contents());
      uint64_t first = 0;
      while (first < a[plane].sizeBytes() && leftBytes[first] == rightBytes[first]) ++first;
      throw std::runtime_error(std::string("batch teacher full QSA cache differs: plane=") + names[plane] +
          ", byte=" + std::to_string(first) + ", logical_length=" + std::to_string(left.logicalLength()));
    }
    ++counts.checks; counts.comparedBytes += a[plane].sizeBytes();
    // Persistent QSA planes 0..3 are BF16; plane 4 is signed I64 positions.
    // Check both operands: equality alone must never admit matching NaNs.
    if (plane != 4) {
      finiteBF16(a[plane].contents(), a[plane].sizeBytes(), counts);
      finiteBF16(b[plane].contents(), b[plane].sizeBytes(), counts);
    }
  }
}
struct Snapshot final {
  uint64_t length = 0;
  uint32_t capacity = 0;
  bool poisoned = false;
  std::array<uintptr_t, 2> binding{};
  std::array<metal::MetalBuffer, 5> buffers;
  std::array<std::vector<uint8_t>, 5> planes;
};
Snapshot snapshot(const FlashMTPState &state) {
  Snapshot result{state.logicalLength(), state.capacity(), state.poisoned(),
      FlashMTPTeacherPrimeOracleAccess::binding(state), {}, {}};
  const auto buffers = FlashMTPTeacherPrimeOracleAccess::buffers(state);
  result.buffers = buffers;
  for (size_t plane = 0; plane < buffers.size(); ++plane) {
    require(buffers[plane].contents(), "cannot snapshot batch teacher guard state");
    const auto *bytes = static_cast<const uint8_t *>(buffers[plane].contents());
    result.planes[plane].assign(bytes, bytes + buffers[plane].sizeBytes());
  }
  return result;
}
void unchanged(const FlashMTPState &state, const Snapshot &before, Counters &counts) {
  require(state.logicalLength() == before.length && state.capacity() == before.capacity &&
      state.poisoned() == before.poisoned &&
      FlashMTPTeacherPrimeOracleAccess::binding(state) == before.binding,
      "batch teacher host guard mutated state metadata or owner binding");
  const auto buffers = FlashMTPTeacherPrimeOracleAccess::buffers(state);
  for (size_t plane = 0; plane < buffers.size(); ++plane) {
    require(buffers[plane].contents() && buffers[plane].sizeBytes() == before.planes[plane].size() &&
        buffers[plane].sameView(before.buffers[plane]) &&
        std::memcmp(buffers[plane].contents(), before.planes[plane].data(), before.planes[plane].size()) == 0,
        "batch teacher host guard mutated persistent cache bytes");
    ++counts.checks; counts.comparedBytes += before.planes[plane].size();
  }
  ++counts.checks;
}
struct Group final {
  FlashMTPForward &normalOwner, &primeOwner;
  FlashBatchMTPForward &normalBatch, &primeBatch;
  std::array<FlashMTPState, kLanes> normal, prime;
  Group(FlashMTPForward &a, FlashMTPForward &b,
      FlashBatchMTPForward &ab, FlashBatchMTPForward &bb)
      : normalOwner(a), primeOwner(b), normalBatch(ab), primeBatch(bb) {
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      normal[lane] = a.createState(); prime[lane] = b.createState();
    }
  }
  void equal(Counters &counts) const {
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      try { cacheEqual(normal[lane], prime[lane], counts); }
      catch (const std::exception &error) {
        throw std::runtime_error("lane=" + std::to_string(lane) + ": " + error.what());
      }
    }
  }
  void truncate(std::span<const uint32_t> ids, std::span<const uint64_t> retained,
      Counters &counts) {
    require(ids.size() == retained.size(), "batch teacher rollback geometry is invalid");
    for (size_t lane = 0; lane < ids.size(); ++lane) {
      normalOwner.truncate(normal[ids[lane]], retained[lane]);
      primeOwner.truncate(prime[ids[lane]], retained[lane]);
    }
    equal(counts);
  }
};
std::vector<FlashMTPState *> states(std::array<FlashMTPState, kLanes> &source,
    std::span<const uint32_t> ids) {
  std::vector<FlashMTPState *> result;
  for (uint32_t id : ids) {
    require(id < kLanes, "batch teacher fixture lane is outside bounds");
    result.push_back(&source[id]);
  }
  return result;
}
std::vector<uint32_t> offsets(std::span<const uint32_t> counts) {
  std::vector<uint32_t> result{0};
  for (uint32_t count : counts) result.push_back(result.back() + count);
  return result;
}
void metadata(const FlashBatchMTPResult &result, std::span<const uint32_t> ids,
    std::span<const uint32_t> counts, FlashMTPLogits mode,
    const std::array<FlashMTPState, kLanes> &source, Counters &checked) {
  const auto spans = offsets(counts);
  const uint32_t logitRows = mode == FlashMTPLogits::None ? 0 :
      mode == FlashMTPLogits::Last ? uint32_t(ids.size()) : spans.back();
  require(result.lanes == ids.size() && result.laneOffsets == spans &&
      result.logicalLengths.size() == ids.size() && result.logitRows == logitRows &&
      result.hiddenBF16 && result.hiddenBF16.sizeBytes() == uint64_t{spans.back()} * kFeatureBytes,
      "batch teacher returned padded or incorrect real lane metadata");
  for (size_t lane = 0; lane < ids.size(); ++lane)
    require(result.logicalLengths[lane] == source[ids[lane]].logicalLength(),
        "batch teacher result logical offset differs");
  if (mode == FlashMTPLogits::None)
    require(!result.logitsBF16 && !result.greedyResultsU32 && result.greedyRows == 0,
        "batch teacher generic None vocabulary contract changed");
  else
    require(result.logitsBF16 && result.logitsBF16.sizeBytes() == uint64_t{logitRows} * kVocabulary * 2 &&
        result.greedyResultsU32 && result.greedyResultsU32.contents() && result.greedyRows == logitRows &&
        result.greedyResultsU32.sizeBytes() == uint64_t{logitRows} * sizeof(FlashGreedyGPURowResult),
        "batch teacher real proposal vocabulary/greedy contract differs");
  ++checked.checks;
}
template <class Operation> void guard(Group &group, Operation operation, Counters &counts) {
  std::array<Snapshot, kLanes> normalBefore, primeBefore;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    normalBefore[lane] = snapshot(group.normal[lane]);
    primeBefore[lane] = snapshot(group.prime[lane]);
  }
  bool rejected = false;
  try { operation(); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "batch teacher invalid call was not rejected before mutation");
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    unchanged(group.normal[lane], normalBefore[lane], counts);
    unchanged(group.prime[lane], primeBefore[lane], counts);
  }
  ++counts.guardCalls; ++counts.checks;
}
void fold(Group &group, metal::MetalBuffer features, std::span<const uint32_t> tokens,
    std::span<const uint32_t> ids, std::span<const uint32_t> counts, Counters &checked) {
  require(ids.size() == counts.size() && !ids.empty(), "batch teacher fold fixture has invalid spans");
  const auto spans = offsets(counts);
  require(features.sizeBytes() == uint64_t{spans.back()} * kFeatureBytes && tokens.size() == spans.back(),
      "batch teacher fixture is not exactly compact");
  std::vector<uint64_t> before;
  for (uint32_t id : ids) {
    require(group.normal[id].logicalLength() == group.prime[id].logicalLength(),
        "batch teacher fold begins with different offsets");
    before.push_back(group.normal[id].logicalLength());
  }
  auto aStates = states(group.normal, ids), bStates = states(group.prime, ids);
  const auto original = group.normalBatch.forward(aStates, features, tokens, counts, FlashMTPLogits::None);
  timingValid(original.timing); metadata(original, ids, counts, FlashMTPLogits::None, group.normal, checked);
  (void)copiedBF16(original.hiddenBF16, uint64_t{spans.back()} * kHyper, checked);
  const auto candidate = group.primeBatch.primeTeacherCache(bStates, features, tokens, counts);
  timingValid(candidate);
  for (size_t lane = 0; lane < ids.size(); ++lane)
    require(group.normal[ids[lane]].logicalLength() == before[lane] + counts[lane] &&
        group.prime[ids[lane]].logicalLength() == before[lane] + counts[lane],
        "batch teacher fold advanced a padded or incorrect pair count");
  group.equal(checked);
  ++checked.primeCalls; checked.compactPairs += spans.back();
  const bool equalCounts = std::all_of(counts.begin(), counts.end(),
      [&](uint32_t count) { return count == counts.front(); });
  if (equalCounts) ++checked.equalCountCalls; else ++checked.unequalCountCalls;
  checked.normalGPU += original.timing.gpuSeconds; checked.primeGPU += candidate.gpuSeconds;
  checked.normalWall += original.timing.wallSeconds; checked.primeWall += candidate.wallSeconds;
}
struct Proposal final {
  std::vector<uint16_t> hidden, logits;
  std::vector<uint32_t> greedy;
  std::vector<uint8_t> greedyRecords;
};
Proposal proposal(Group &group, metal::MetalBuffer features,
    std::span<const uint32_t> tokens, std::span<const uint32_t> ids,
    std::span<const uint32_t> counts, FlashMTPLogits mode, Counters &checked) {
  auto aStates = states(group.normal, ids), bStates = states(group.prime, ids);
  const auto original = group.normalBatch.forward(aStates, features, tokens, counts, mode);
  timingValid(original.timing); metadata(original, ids, counts, mode, group.normal, checked);
  Proposal expected;
  expected.hidden = copiedBF16(original.hiddenBF16, uint64_t{offsets(counts).back()} * kHyper, checked);
  if (mode != FlashMTPLogits::None) {
    expected.logits = copiedBF16(original.logitsBF16, uint64_t{original.logitRows} * kVocabulary, checked);
    const auto *results = static_cast<const FlashGreedyGPURowResult *>(original.greedyResultsU32.contents());
    for (uint32_t row = 0; row < original.greedyRows; ++row)
      expected.greedy.push_back(greedyGPUResultToken(results[row], kVocabulary));
    const auto *recordBytes = static_cast<const uint8_t *>(original.greedyResultsU32.contents());
    expected.greedyRecords.assign(recordBytes, recordBytes + original.greedyResultsU32.sizeBytes());
  }
  const auto actual = group.primeBatch.forward(bStates, features, tokens, counts, mode);
  timingValid(actual.timing); metadata(actual, ids, counts, mode, group.prime, checked);
  require(expected.hidden == copiedBF16(actual.hiddenBF16, expected.hidden.size(), checked),
      "batch teacher future premixer hidden differs bitwise");
  ++checked.checks; checked.comparedBytes += expected.hidden.size() * 2;
  if (mode != FlashMTPLogits::None) {
    require(expected.logits == copiedBF16(actual.logitsBF16, expected.logits.size(), checked),
        "batch teacher future vocabulary differs bitwise");
    const auto *results = static_cast<const FlashGreedyGPURowResult *>(actual.greedyResultsU32.contents());
    for (uint32_t row = 0; row < actual.greedyRows; ++row)
      require(expected.greedy[row] == greedyGPUResultToken(results[row], kVocabulary),
          "batch teacher future GPU greedy token differs");
    require(expected.greedyRecords.size() == actual.greedyResultsU32.sizeBytes() &&
        std::memcmp(expected.greedyRecords.data(), actual.greedyResultsU32.contents(), expected.greedyRecords.size()) == 0,
        "batch teacher future GPU greedy compact record differs bitwise");
    checked.checks += 3;
    checked.comparedBytes += expected.logits.size() * 2 + expected.greedyRecords.size();
  }
  group.equal(checked); ++checked.proposalCalls;
  return expected;
}
std::vector<uint32_t> realPairs(metal::MetalBuffer destination,
    const metal::MetalBuffer &targetFeatures, std::span<const uint32_t> fixture,
    Group &group, std::span<const uint32_t> ids, std::span<const uint32_t> counts,
    bool specialTokens) {
  const auto spans = offsets(counts);
  require(destination.contents() && destination.sizeBytes() >= uint64_t{spans.back()} * kFeatureBytes,
      "batch teacher compact fixture arena is too small");
  std::vector<uint32_t> next;
  for (size_t lane = 0; lane < ids.size(); ++lane) {
    for (uint32_t row = 0; row < counts[lane]; ++row) {
      const uint64_t position = group.normal[ids[lane]].logicalLength() + row;
      const uint32_t source = uint32_t((position + uint64_t{ids[lane]} * 131) % 2047);
      std::memcpy(static_cast<uint8_t *>(destination.contents()) + uint64_t{spans[lane] + row} * kFeatureBytes,
          static_cast<const uint8_t *>(targetFeatures.contents()) + uint64_t{source} * kFeatureBytes,
          kFeatureBytes);
      uint32_t token = fixture[source + 1];
      if (specialTokens && position % 127 == 0) token = position % 2 ? 248046 : 248044;
      next.push_back(token);
    }
  }
  return next;
}
}

int main(int argc, char **argv) {
  @autoreleasepool {
    Progress progress;
    auto &checked = progress.counts;
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "usage: prefill-batch-teacher-oracle METALLIB PACKAGE EXACT_2048_TOKENS_JSON REPORT_JSON\n"
            "Strict grouped None/cache-prime full-plane bytes, real compact counts, future hidden/logits/greedy, rollback, guards.\n";
        return 0;
      }
      require(argc == 5, "invalid batch teacher oracle arguments");
      require(!std::filesystem::exists(argv[4]), "choose a fresh batch teacher oracle report");
      for (const char *suffix : {".checkpoint.json", ".failure.json", ".writing"})
        require(!std::filesystem::exists(std::string(argv[4]) + suffix),
            "choose fresh batch teacher oracle report and progress paths");
      progress.report = argv[4];
      require(enabled("SPLASH_FLASH_GPU_GREEDY"),
          "strict batch teacher oracle requires SPLASH_FLASH_GPU_GREEDY=1");
      const auto fixture = loadTokens(argv[3]);
      require(fixture.size() == 2048, "batch teacher oracle requires true 2048-row target fixture");
      progress.fixtureDigest = digest(fixture.data(), fixture.size() * 4);
      progress.phase = "backend_creation";
      metal::MetalBackend backend(argv[1]);
      progress.backendInitialized = true;
      progress.metallibDigest = hexadecimal(backend.metallibSha256());
      progress.phase = "model_loading";
      const auto weights = FlashWeights::load(backend, argv[2]);
      progress.sourceIdentity = weights.sourceIdentity();
      progress.phase = "workspace_construction";
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      uint64_t planned = FlashForward::workspacePlannedBytes(kCapacity, 2048, 4) +
          FlashForward::requestStateBytes(kCapacity) + FlashForward::expertCachePlannedBytes(weights) +
          FlashForward::floatDenseCachePlannedBytes(weights) + FlashForward::int8HeadPlannedBytes(weights) +
          2 * FlashMTPForward::workspacePlannedBytes(kCapacity, kRows) +
          2 * FlashBatchMTPForward::workspacePlannedBytes(kCapacity, kLanes, kRows, true) +
          8 * FlashMTPForward::requestStateBytes(kCapacity) + (256ULL << 20);
      if (enabled("SPLASH_FLASH_DENSE_CACHE"))
        planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true)) +
            2 * FlashMTPForward::denseCachePlannedBytes(weights);
      if (enabled("SPLASH_FLASH_QSA_F32")) planned += 32ULL << 20;
      if (enabled("SPLASH_FLASH_BLOCKED_MOE")) planned += flashMoEBlockedWorkspacePlannedBytes(2048, 10);
      auto reservation = governor.tryReserve(planned);
      require(bool(reservation), "batch teacher oracle governor denied construction");
      FlashForward target(backend, weights, kCapacity, 2048, 4);
      FlashMTPForward normalOwner(backend, weights, kCapacity, kRows);
      FlashMTPForward primeOwner(backend, weights, kCapacity, kRows);
      FlashBatchMTPForward normalBatch(normalOwner, kLanes, kRows, target.cachedVocabulary());
      FlashBatchMTPForward primeBatch(primeOwner, kLanes, kRows, target.cachedVocabulary());
      require(std::string_view(normalBatch.attentionRouteSemantics()) == primeBatch.attentionRouteSemantics() &&
          std::string_view(normalBatch.projectionRouteSemantics()) == primeBatch.projectionRouteSemantics(),
          "batch teacher baseline/candidate route differs");
      require(normalBatch.workspaceBytes() <= FlashBatchMTPForward::workspacePlannedBytes(kCapacity, kLanes, kRows, true) &&
          primeBatch.workspaceBytes() <= FlashBatchMTPForward::workspacePlannedBytes(kCapacity, kLanes, kRows, true),
          "batch teacher arena exceeds batch admission estimate");
      require(target.workspaceBytes() + normalOwner.workspaceBytes() + primeOwner.workspaceBytes() +
          normalBatch.workspaceBytes() + primeBatch.workspaceBytes() + FlashForward::requestStateBytes(kCapacity) +
          8 * FlashMTPForward::requestStateBytes(kCapacity) <= planned,
          "batch teacher oracle arenas exceed reservation");
      auto targetState = target.createState();
      const auto targetFeatures = backend.allocateBuffer(uint64_t{2048} * kFeatureBytes,
          metal::BufferStorage::Shared, "batch-teacher-owned-real-target-features");
      const auto compact = backend.allocateBuffer(uint64_t{kLanes} * kRows * kFeatureBytes,
          metal::BufferStorage::Shared, "batch-teacher-compact-real-pair-features");
      const auto chained = backend.allocateBuffer(uint64_t{kLanes} * kFeatureBytes,
          metal::BufferStorage::Shared, "batch-teacher-owned-proposal-continuation");
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        auto operands = target.cachedOperandsOnly();
        for (auto *head : {&normalOwner, &primeOwner}) {
          const auto buffers = head->cachedOperandsOnly();
          operands.insert(operands.end(), buffers.begin(), buffers.end());
        }
        if (!operands.empty()) residency = backend.requestWeightResidency(operands, "batch teacher saved operands only");
      }
      reservation->commit();
      progress.phase = "real_target_forward";
      const auto main = target.forward(targetState, fixture, false, true);
      timingValid(main.timing);
      require(main.hiddenBF16 && main.hiddenBF16.contents() && main.hiddenBF16.sizeBytes() == targetFeatures.sizeBytes(),
          "batch teacher real target premixer features absent");
      std::memcpy(targetFeatures.contents(), main.hiddenBF16.contents(), targetFeatures.sizeBytes());
      require(main.greedyResultsU32 && main.greedyResultsU32.contents(), "batch teacher target GPU greedy absent");
      const uint32_t anchor = greedyGPUResultToken(*static_cast<const FlashGreedyGPURowResult *>(main.greedyResultsU32.contents()), kVocabulary);
      finiteBF16(targetFeatures.contents(), targetFeatures.sizeBytes(), checked);
      const std::string targetHiddenDigest = digest(targetFeatures.contents(), targetFeatures.sizeBytes());
      progress.targetForwardCompleted = true;
      progress.targetHiddenDigest = targetHiddenDigest;
      progress.phase = "real_target_complete";
      progressRecord(progress);
      std::vector<std::string> labels;
      const std::array<uint32_t, kLanes> all{0, 1, 2, 3};
      const auto foldReal = [&](Group &group, std::span<const uint32_t> ids,
          std::span<const uint32_t> counts, bool special = false) {
        const auto next = realPairs(compact, targetFeatures, fixture, group, ids, counts, special);
        fold(group, backend.view(compact, 0, uint64_t{next.size()} * kFeatureBytes), next, ids, counts, checked);
      };
      for (const std::string label : {"true2047-B4-128x15-tail127", "unaligned-B4-ragged-compact",
              "eos-and-special-token-pairs-B4", "sparse-context8191-B1"}) {
        progress.phase = "case:" + label;
        Group group(normalOwner, primeOwner, normalBatch, primeBatch);
        group.equal(checked);
        std::vector<uint32_t> active(all.begin(), all.end());
        if (label == "sparse-context8191-B1") {
          active = {0};
          for (uint32_t begin = 0; begin < 8191; begin += kRows)
            foldReal(group, active, std::array<uint32_t, 1>{std::min(kRows, 8191 - begin)});
        } else if (label == "unaligned-B4-ragged-compact") {
          // All rows are physically compact; there is no max(counts) padding.
          foldReal(group, all, std::array<uint32_t, 4>{127, 1, 128, 3});
          foldReal(group, std::array<uint32_t, 3>{3, 1, 0}, std::array<uint32_t, 3>{1, 127, 1});
          foldReal(group, std::array<uint32_t, 2>{2, 0}, std::array<uint32_t, 2>{1, 128});
          foldReal(group, all, std::array<uint32_t, 4>{1, 128, 127, 128});
          foldReal(group, all, std::array<uint32_t, 4>{128, 1, 3, 1});
        } else {
          for (uint32_t begin = 0; begin < 2047; begin += kRows) {
            const uint32_t rows = std::min(kRows, 2047 - begin);
            foldReal(group, all, std::array<uint32_t, 4>{rows, rows, rows, rows},
                label == "eos-and-special-token-pairs-B4");
          }
        }
        const auto lastTarget = backend.view(targetFeatures, uint64_t{2047} * kFeatureBytes, kFeatureBytes);
        const std::vector<uint32_t> ones(active.size(), 1);
        std::vector<uint32_t> next(active.size(), anchor);
        for (size_t lane = 0; lane < active.size(); ++lane)
          std::memcpy(static_cast<uint8_t *>(chained.contents()) + lane * kFeatureBytes, lastTarget.contents(), kFeatureBytes);
        std::vector<uint64_t> teacherLengths;
        for (uint32_t id : active) teacherLengths.push_back(group.normal[id].logicalLength());
        const uint32_t depths = label == "sparse-context8191-B1" ? 1 : 3;
        Proposal first;
        for (uint32_t depth = 0; depth < depths; ++depth) {
          auto result = proposal(group, backend.view(chained, 0, uint64_t{active.size()} * kFeatureBytes),
              next, active, ones, FlashMTPLogits::Last, checked);
          if (depth == 0) first = result;
          std::memcpy(chained.contents(), result.hidden.data(), result.hidden.size() * 2);
          next = std::move(result.greedy);
        }
        if (label == "sparse-context8191-B1") {
          // Lane 1 is healthy with room; lane 0 is full. The second lane must
          // reject the entire command before any first-lane cache mutation.
          auto invalidStates = states(group.prime, std::array<uint32_t, 2>{1, 0});
          guard(group, [&] { (void)primeBatch.primeTeacherCache(invalidStates,
              backend.view(compact, 0, 2 * kFeatureBytes), std::array<uint32_t, 2>{anchor, anchor},
              std::array<uint32_t, 2>{1, 1}); }, checked);
        } else {
          std::vector<uint64_t> keep;
          for (size_t lane = 0; lane < active.size(); ++lane) keep.push_back(teacherLengths[lane] + 1);
          group.truncate(active, keep, checked);
          std::memcpy(chained.contents(), first.hidden.data(), first.hidden.size() * 2);
          next.assign(active.size(), 248046);
          (void)proposal(group, backend.view(chained, 0, uint64_t{active.size()} * kFeatureBytes),
              next, active, ones, FlashMTPLogits::Last, checked);
          // Restore true teacher offsets, re-prime ragged committed spans,
          // then test all real proposal rows and their exact GPU greedy IDs.
          group.truncate(active, teacherLengths, checked);
          foldReal(group, all, std::array<uint32_t, 4>{4, 3, 2, 1}, true);
          const std::array<uint32_t, 4> proposalCounts{1, 2, 3, 4};
          const auto tokens = realPairs(compact, targetFeatures, fixture, group, all, proposalCounts, true);
          (void)proposal(group, backend.view(compact, 0, uint64_t{tokens.size()} * kFeatureBytes),
              tokens, all, proposalCounts, FlashMTPLogits::All, checked);
        }
        for (const uint64_t retained : {0ULL, 1ULL, 3ULL, 4ULL, 127ULL}) {
          group.truncate(active, std::vector<uint64_t>(active.size(), retained), checked);
          foldReal(group, active, std::vector<uint32_t>(active.size(), kRows), true);
        }
        const auto tokens = realPairs(compact, targetFeatures, fixture, group, active, ones, false);
        (void)proposal(group, backend.view(compact, 0, uint64_t{tokens.size()} * kFeatureBytes),
            tokens, active, ones, FlashMTPLogits::Last, checked);
        labels.push_back(label);
        progress.completedCases.push_back(label);
        progress.phase = "case_complete:" + label;
        progressRecord(progress);
      }
      progress.phase = "host_guards";
      {
        Group group(normalOwner, primeOwner, normalBatch, primeBatch);
        foldReal(group, all, std::array<uint32_t, 4>{3, 4, 1, 2});
        const auto valid = backend.view(targetFeatures, 0, 2 * kFeatureBytes);
        auto single = states(group.prime, std::array<uint32_t, 1>{0});
        auto pair = states(group.prime, std::array<uint32_t, 2>{0, 1});
        auto entire = states(group.prime, all);
        const std::array<uint32_t, 2> goodTokens{anchor, anchor}, goodCounts{1, 1};
        const auto reject = [&](auto operation) { guard(group, operation, checked); };
        reject([&] { (void)primeBatch.primeTeacherCache({}, valid, {}, {}); });
        reject([&] { (void)primeBatch.primeTeacherCache(entire, valid, goodTokens, goodCounts); });
        reject([&] { (void)primeBatch.primeTeacherCache(pair, valid, goodTokens,
            std::array<uint32_t, 2>{1, 0}); });
        reject([&] { (void)primeBatch.primeTeacherCache(pair, valid, goodTokens,
            std::array<uint32_t, 2>{1, kRows + 1}); });
        reject([&] { (void)primeBatch.primeTeacherCache(pair, valid,
            std::array<uint32_t, 2>{anchor, kVocabulary}, goodCounts); });
        reject([&] { (void)primeBatch.primeTeacherCache(pair, valid,
            std::span<const uint32_t>(goodTokens).first(1), goodCounts); });
        reject([&] { (void)primeBatch.primeTeacherCache(pair,
            backend.view(valid, 0, 2 * kFeatureBytes - 2), goodTokens, goodCounts); });
        reject([&] { (void)primeBatch.primeTeacherCache(pair, {}, goodTokens, goodCounts); });
        reject([&] { (void)primeBatch.primeTeacherCache(single, valid,
            std::array<uint32_t, 1>{anchor}, std::array<uint32_t, 1>{0}); });
        std::array<FlashMTPState *, 2> duplicate{&group.prime[0], &group.prime[0]};
        reject([&] { (void)primeBatch.primeTeacherCache(duplicate, valid, goodTokens, goodCounts); });
        std::array<FlashMTPState *, 2> foreign{&group.prime[0], &group.normal[1]};
        reject([&] { (void)primeBatch.primeTeacherCache(foreign, valid, goodTokens, goodCounts); });
        std::array<FlashMTPState *, 2> null{&group.prime[0], nullptr};
        reject([&] { (void)primeBatch.primeTeacherCache(null, valid, goodTokens, goodCounts); });
        FlashMTPState uninitialized;
        const auto uninitializedBinding = FlashMTPTeacherPrimeOracleAccess::binding(uninitialized);
        const auto uninitializedLength = uninitialized.logicalLength();
        const auto uninitializedCapacity = uninitialized.capacity();
        const auto uninitializedPoisoned = uninitialized.poisoned();
        require(uninitializedBinding == std::array<uintptr_t, 2>{0, 0} &&
            uninitializedLength == 0 && uninitializedCapacity == 0 && uninitializedPoisoned,
            "uninitialized MTP state getter contract differs before rejected call");
        std::array<FlashMTPState *, 2> unowned{&group.prime[0], &uninitialized};
        reject([&] { (void)primeBatch.primeTeacherCache(unowned, valid, goodTokens, goodCounts); });
        require(FlashMTPTeacherPrimeOracleAccess::binding(uninitialized) == uninitializedBinding &&
            uninitialized.logicalLength() == uninitializedLength &&
            uninitialized.capacity() == uninitializedCapacity &&
            uninitialized.poisoned() == uninitializedPoisoned,
            "batch teacher uninitialized foreign state was mutated");
        ++checked.checks;
        std::array<FlashMTPState *, 5> tooMany{&group.prime[0], &group.prime[1],
            &group.prime[2], &group.prime[3], &group.normal[0]};
        reject([&] { (void)primeBatch.primeTeacherCache(tooMany, valid,
            std::array<uint32_t, 5>{anchor, anchor, anchor, anchor, anchor},
            std::array<uint32_t, 5>{1, 1, 1, 1, 1}); });
        reject([&] { (void)primeBatch.forward(pair, valid, goodTokens, goodCounts,
            static_cast<FlashMTPLogits>(255)); });
        reject([&] { (void)primeBatch.forward(pair,
            backend.view(compact, 0, 17 * kFeatureBytes), std::vector<uint32_t>(17, anchor),
            std::array<uint32_t, 2>{9, 8}, FlashMTPLogits::All); });
        group.equal(checked);
      }
      progress.phase = "host_guards_complete";
      progressRecord(progress);
      progress.phase = "numeric_failure_poison_publication";
      {
        // A numeric GPU failure must poison every submitted lane without
        // publishing offsets, while an uninvolved state remains healthy.
        Group group(normalOwner, primeOwner, normalBatch, primeBatch);
        const std::array<uint32_t, 3> ids{3, 1, 0};
        const std::array<uint32_t, 3> rows{1, 2, 3};
        foldReal(group, ids, rows);
        const auto next = realPairs(compact, targetFeatures, fixture, group, ids, rows, false);
        static_cast<uint16_t *>(compact.contents())[uint64_t{next.size() - 1} * kHyper] = 0x7fc0;
        const auto badFeatures = backend.view(compact, 0, uint64_t{next.size()} * kFeatureBytes);
        auto aStates = states(group.normal, ids), bStates = states(group.prime, ids);
        std::vector<uint64_t> before;
        for (uint32_t id : ids) before.push_back(group.prime[id].logicalLength());
        bool originalRejected = false, candidateRejected = false;
        const auto unaffectedNormal = snapshot(group.normal[2]);
        const auto unaffectedPrime = snapshot(group.prime[2]);
        try { (void)normalBatch.forward(aStates, badFeatures, next, rows, FlashMTPLogits::None); }
        catch (const std::runtime_error &error) {
          require(std::string_view(error.what()).starts_with("Flash batch MTP sticky diagnostics failed: "),
              (std::string("numeric full-forward oracle received unexpected error: ") + error.what()).c_str());
          originalRejected = true;
        }
        try { (void)primeBatch.primeTeacherCache(bStates, badFeatures, next, rows); }
        catch (const std::runtime_error &error) {
          require(std::string_view(error.what()).starts_with("Flash batch teacher cache sticky diagnostics failed: "),
              (std::string("numeric cache-prime oracle received unexpected error: ") + error.what()).c_str());
          candidateRejected = true;
        }
        require(originalRejected && candidateRejected, "batch teacher numeric failure was not rejected");
        for (size_t lane = 0; lane < ids.size(); ++lane)
          require(group.normal[ids[lane]].poisoned() && group.prime[ids[lane]].poisoned() &&
              group.normal[ids[lane]].logicalLength() == before[lane] &&
              group.prime[ids[lane]].logicalLength() == before[lane],
              "batch teacher numeric failure changed offset publication or lane poison semantics");
        require(!group.normal[2].poisoned() && !group.prime[2].poisoned() &&
            group.normal[2].logicalLength() == 0 && group.prime[2].logicalLength() == 0,
            "batch teacher numeric failure poisoned an uninvolved lane");
        unchanged(group.normal[2], unaffectedNormal, checked);
        unchanged(group.prime[2], unaffectedPrime, checked);
        ++checked.checks;
        guard(group, [&] { (void)primeBatch.primeTeacherCache(bStates, badFeatures, next, rows); }, checked);
      }
      require(checked.equalCountCalls > 0 && checked.unequalCountCalls > 0 && checked.guardCalls >= 15 &&
          checked.proposalCalls > 0, "batch teacher required scenario coverage is absent");
      progress.phase = "all_checks_complete";
      progressRecord(progress);
      // A failed assertion never creates a success report. Use a temporary
      // sibling for complete write/close, then publish the verified JSON.
      const std::filesystem::path reportPath(argv[4]);
      const std::filesystem::path temporary = reportPath.string() + ".writing";
      require(!std::filesystem::exists(temporary), "batch teacher report temporary already exists");
      std::ofstream report(temporary);
      require(bool(report), "cannot write batch teacher oracle report");
      report << std::setprecision(std::numeric_limits<double>::max_digits10)
          << "{\"schema\":\"splash-batch-teacher-cache-prime-oracle-v2\",\"valid\":true,\"gpu_executed\":true"
          << ",\"batch_semantics\":" << json::quote(kFlashBatchMTPSemantics)
          << ",\"teacher_semantics\":" << json::quote(kFlashBatchMTPTeacherCacheSemantics)
          << ",\"singleton_cache_prefix_semantics\":" << json::quote(kFlashMTPTeacherCacheSemantics)
          << ",\"attention_route\":" << json::quote(primeBatch.attentionRouteSemantics())
          << ",\"projection_route\":" << json::quote(primeBatch.projectionRouteSemantics())
          << ",\"source_identity_sha256\":" << json::quote(weights.sourceIdentity())
          << ",\"metallib_sha256\":" << json::quote(hexadecimal(backend.metallibSha256()))
          << ",\"tokens_u32le_sha256\":" << json::quote(digest(fixture.data(), fixture.size() * 4))
          << ",\"real_target_hidden_bf16_sha256\":" << json::quote(targetHiddenDigest)
          << ",\"separate_baseline_candidate_owners_and_workspaces\":true"
          << ",\"full_five_qsa_cache_planes_bitwise_equal_after_every_fold\":true"
          << ",\"true_compact_counts_equal_and_unequal\":true,\"no_fake_padding_pairs\":true"
          << ",\"future_real_target_hidden_logits_greedy_bitwise_equal\":true"
          << ",\"generic_none_contract_preserved\":true,\"rollback_reprime_continuation_equal\":true"
          << ",\"bf16_to_f32_values_finite\":true,\"host_guards_preserve_full_cache_bytes\":true"
          << ",\"capacity_foreign_duplicate_shape_guards_passed\":true"
          << ",\"uninitialized_foreign_state_impl_owner_and_getters_unchanged\":true"
          << ",\"owned_host_guard_impl_owner_and_buffer_views_unchanged\":true"
          << ",\"uninvolved_numeric_failure_lane_full_cache_unchanged\":true"
          << ",\"numeric_failure_lane_poison_publication_semantics_preserved\":true"
          << ",\"capacity\":" << kCapacity << ",\"maximum_rows_per_lane\":" << kRows
          << ",\"maximum_lanes\":" << kLanes << ",\"case_count\":" << labels.size()
          << ",\"checks\":" << checked.checks << ",\"compared_bytes\":" << checked.comparedBytes
          << ",\"finite_bf16_words_checked\":" << checked.finiteWords
          << ",\"teacher_pair_calls\":" << checked.primeCalls << ",\"compact_teacher_pairs\":" << checked.compactPairs
          << ",\"equal_count_calls\":" << checked.equalCountCalls << ",\"unequal_count_calls\":" << checked.unequalCountCalls
          << ",\"future_proposal_calls\":" << checked.proposalCalls << ",\"guard_calls\":" << checked.guardCalls
          << ",\"normal_gpu_seconds\":" << checked.normalGPU << ",\"prime_gpu_seconds\":" << checked.primeGPU
          << ",\"normal_command_wall_seconds\":" << checked.normalWall << ",\"prime_command_wall_seconds\":" << checked.primeWall
          << ",\"timing_scope\":\"correctness oracle only; mixed independent offsets and full host cache comparisons; use matched service controls for speed\""
          << ",\"case_labels\":[";
      for (size_t index = 0; index < labels.size(); ++index) {
        if (index) report << ',';
        report << json::quote(labels[index]);
      }
      report << "]}\n";
      require(bool(report), "batch teacher oracle report write failed");
      report.close(); require(bool(report), "batch teacher oracle report close failed");
      std::filesystem::rename(temporary, reportPath);
      std::cout << "batch teacher cache oracle passed; report=" << argv[4] << '\n';
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "batch teacher cache oracle failed: " << error.what() << '\n';
      try { progressRecord(progress, true, error.what()); }
      catch (const std::exception &recordError) {
        std::cerr << "batch teacher failure record failed: " << recordError.what() << '\n';
      }
      return 1;
    }
  }
}
