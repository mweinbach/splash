// Root-exclusive cache/future-head qualification. --help submits no GPU work.
#define main splash_prefill4k_attribution_unused_main
#include "prefill4k_attribution.mm"
#undef main
#include "flash/FlashMTPStateInternal.hpp"

namespace splash::flash {
struct FlashMTPTeacherPrimeOracleAccess final {
  static std::array<metal::MetalBuffer, 5> buffers(const FlashMTPState &state) {
    if (!state.impl_) throw std::invalid_argument("oracle state is uninitialized");
    const auto &q = state.impl_->qsa;
    return {q.keys, q.values, q.rawIndexKeys, q.pooledKeys, q.indexPositions};
  }
};
}
namespace {
constexpr uint32_t kTeacherHyper = 10240, kTeacherVocabulary = 248320;
struct OracleCounters { uint64_t checks = 0, comparedBytes = 0, finiteBF16Words = 0, primeCalls = 0, proposalCalls = 0; };
void oracleEqual(const FlashMTPState &left, const FlashMTPState &right, OracleCounters &counts) {
  require(left.logicalLength() == right.logicalLength() && left.capacity() == right.capacity() &&
      !left.poisoned() && !right.poisoned(), "teacher oracle logical state differs");
  ++counts.checks;
  const auto a = FlashMTPTeacherPrimeOracleAccess::buffers(left), b = FlashMTPTeacherPrimeOracleAccess::buffers(right);
  for (size_t plane = 0; plane < a.size(); ++plane) {
    require(a[plane].contents() && b[plane].contents() && a[plane].sizeBytes() == b[plane].sizeBytes(), "teacher cache plane extent differs");
    require(std::memcmp(a[plane].contents(), b[plane].contents(), a[plane].sizeBytes()) == 0, "teacher cache bytes differ");
    ++counts.checks; counts.comparedBytes += a[plane].sizeBytes();
    if (plane != 4) {
      const auto *words = static_cast<const uint16_t *>(a[plane].contents());
      for (uint64_t index = 0; index < a[plane].sizeBytes() / 2; ++index)
        require(std::isfinite(std::bit_cast<float>(uint32_t{words[index]} << 16)), "nonfinite teacher cache BF16/F32 value");
      counts.finiteBF16Words += a[plane].sizeBytes() / 2;
    }
  }
}
void oracleTiming(const metal::CommandTiming &timing) {
  require(std::isfinite(timing.gpuSeconds) && timing.gpuSeconds > 0 &&
      std::isfinite(timing.wallSeconds) && timing.wallSeconds > 0 &&
      timing.gpuSeconds < timing.wallSeconds * 1.1, "invalid teacher timing ABI");
}
std::vector<uint16_t> oracleWords(const metal::MetalBuffer &buffer) {
  require(buffer && buffer.contents() && buffer.sizeBytes() % 2 == 0, "invalid teacher BF16 result");
  const auto *words = static_cast<const uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < buffer.sizeBytes() / 2; ++index)
    require(std::isfinite(std::bit_cast<float>(uint32_t{words[index]} << 16)), "nonfinite teacher result BF16/F32 value");
  return {words, words + buffer.sizeBytes() / 2};
}
struct OracleProposal { std::vector<uint16_t> hidden, logits; uint32_t greedy = 0; };
OracleProposal oracleProposal(FlashMTPForward &head, FlashMTPForward &candidateHead, FlashMTPState &left, FlashMTPState &right,
    metal::MetalBuffer hidden, std::span<const uint32_t> tokens, FlashMTPLogits mode, OracleCounters &counts) {
  const auto first = head.forward(left, hidden, tokens, mode);
  oracleTiming(first.timing);
  OracleProposal a; a.hidden = oracleWords(first.hiddenBF16);
  if (mode != FlashMTPLogits::None) {
    a.logits = oracleWords(first.logitsBF16);
    require(first.greedyResultsU32 && first.greedyResultsU32.contents(), "proposal GPU greedy is absent");
    a.greedy = greedyGPUResultToken(*static_cast<const FlashGreedyGPURowResult *>(first.greedyResultsU32.contents()), kTeacherVocabulary);
  }
  const auto second = candidateHead.forward(right, hidden, tokens, mode);
  oracleTiming(second.timing);
  require(a.hidden == oracleWords(second.hiddenBF16), "future teacher proposal hidden differs");
  ++counts.checks; counts.comparedBytes += a.hidden.size() * 2;
  if (mode != FlashMTPLogits::None) {
    require(a.logits == oracleWords(second.logitsBF16), "future teacher proposal vocabulary differs");
    require(second.greedyResultsU32 && second.greedyResultsU32.contents() && a.greedy ==
        greedyGPUResultToken(*static_cast<const FlashGreedyGPURowResult *>(second.greedyResultsU32.contents()), kTeacherVocabulary), "future teacher greedy differs");
    counts.checks += 2; counts.comparedBytes += a.logits.size() * 2;
  } else {
    require(!first.logitsBF16 && !second.logitsBF16 && first.logitRows == 0 && second.logitRows == 0,
        "generic None forward vocabulary contract changed");
    ++counts.checks;
  }
  oracleEqual(left, right, counts); ++counts.proposalCalls;
  return a;
}
}
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "usage: prefill4k-teacher-oracle METALLIB PACKAGE EXACT_2048_TOKENS_JSON REPORT_JSON\n"
            "Root-exclusive full QSA cache bytes, finite BF16/F32 values, future proposals, rollback, EOS, partial128 boundaries.\n";
        return 0;
      }
      require(argc == 5, "invalid teacher oracle arguments");
      require(!std::filesystem::exists(argv[4]), "choose a fresh teacher oracle report");
      const auto tokens = loadTokens(argv[3]); require(tokens.size() == 2048, "teacher oracle requires true 2048-row target fixture");
      metal::MetalBackend backend(argv[1]);
      const auto weights = FlashWeights::load(backend, argv[2]);
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      uint64_t planned = FlashForward::workspacePlannedBytes(8192, 2048, 4) + FlashForward::requestStateBytes(8192) +
          FlashForward::expertCachePlannedBytes(weights) + FlashForward::floatDenseCachePlannedBytes(weights) + FlashForward::int8HeadPlannedBytes(weights) +
          2 * FlashMTPForward::workspacePlannedBytes(8192, 128) + 4 * FlashMTPForward::requestStateBytes(8192) + (128ULL << 20);
      if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true)) + 2 * FlashMTPForward::denseCachePlannedBytes(weights);
      if (enabled("SPLASH_FLASH_QSA_F32")) planned += 16ULL << 20;
      if (enabled("SPLASH_FLASH_BLOCKED_MOE")) planned += flashMoEBlockedWorkspacePlannedBytes(2048, 10);
      auto reservation = governor.tryReserve(planned); require(bool(reservation), "teacher oracle governor denied construction");
      FlashForward target(backend, weights, 8192, 2048, 4);
      FlashMTPForward head(backend, weights, 8192, 128), candidateHead(backend, weights, 8192, 128);
      require(target.workspaceBytes() + head.workspaceBytes() + candidateHead.workspaceBytes() + FlashForward::requestStateBytes(8192) +
          4 * FlashMTPForward::requestStateBytes(8192) <= planned, "teacher oracle arenas exceed reservation");
      auto targetState = target.createState();
      auto targetFeatures = backend.allocateBuffer(uint64_t{2048} * kTeacherHyper * 2, metal::BufferStorage::Shared, "teacher-oracle-owned-real-target-features");
      auto input = backend.allocateBuffer(uint64_t{128} * kTeacherHyper * 2, metal::BufferStorage::Shared, "teacher-oracle-owned-real-pair-features");
      auto chained = backend.allocateBuffer(uint64_t{4} * kTeacherHyper * 2, metal::BufferStorage::Shared, "teacher-oracle-owned-chained-features");
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        auto buffers = target.cachedOperandsOnly(); const auto hb = head.cachedOperandsOnly(); buffers.insert(buffers.end(), hb.begin(), hb.end());
        const auto cb = candidateHead.cachedOperandsOnly(); buffers.insert(buffers.end(), cb.begin(), cb.end());
        if (!buffers.empty()) residency = backend.requestWeightResidency(buffers, "teacher oracle saved operands only");
      }
      reservation->commit();
      const auto main = target.forward(targetState, tokens, false, true); oracleTiming(main.timing);
      require(main.hiddenBF16 && main.hiddenBF16.contents(), "teacher oracle target features absent");
      std::memcpy(targetFeatures.contents(), main.hiddenBF16.contents(), targetFeatures.sizeBytes());
      require(main.greedyResultsU32 && main.greedyResultsU32.contents(), "teacher oracle target GPU greedy absent");
      const uint32_t anchor = greedyGPUResultToken(*static_cast<const FlashGreedyGPURowResult *>(main.greedyResultsU32.contents()), kTeacherVocabulary);
      OracleCounters counts;
      double normalGPU = 0, candidateGPU = 0, normalWall = 0, candidateWall = 0;
      uint64_t cases = 0;
      std::vector<std::string> labels;
      for (const auto &label : {std::string("true2047-128x15-tail127"), std::string("unaligned127-1-128-1"),
              std::string("eos-and-special-token-pairs"), std::string("sparse-context8191")}) {
        auto left = head.createState(), right = candidateHead.createState();
        const uint32_t total = label == "sparse-context8191" ? 8191 : label == "unaligned127-1-128-1" ? 257 : 2047;
        std::vector<uint32_t> chunks;
        if (label == "unaligned127-1-128-1") chunks = {127, 1, 128, 1};
        else for (uint32_t begin = 0; begin < total; begin += 128) chunks.push_back(std::min(128u, total - begin));
        uint32_t begin = 0;
        for (const uint32_t count : chunks) {
          std::vector<uint32_t> next;
          for (uint32_t row = 0; row < count; ++row) {
            const uint32_t source = (begin + row) % 2047;
            std::memcpy(static_cast<uint8_t *>(input.contents()) + uint64_t{row} * kTeacherHyper * 2,
                static_cast<const uint8_t *>(targetFeatures.contents()) + uint64_t{source} * kTeacherHyper * 2, uint64_t{kTeacherHyper} * 2);
            uint32_t nextToken = tokens[source + 1];
            if (label == "eos-and-special-token-pairs" && ((begin + row) % 127 == 0))
              nextToken = (begin + row) % 2 ? 248046 : 248044;
            next.push_back(nextToken);
          }
          const auto features = backend.view(input, 0, uint64_t{count} * kTeacherHyper * 2);
          const auto a = head.forward(left, features, next, FlashMTPLogits::None);
          require(a.hiddenBF16 && a.hiddenRows == count && !a.logitsBF16 && a.logitRows == 0, "baseline None forward hidden contract changed");
          oracleTiming(a.timing); normalGPU += a.timing.gpuSeconds; normalWall += a.timing.wallSeconds;
          const auto b = candidateHead.primeTeacherCache(right, features, next);
          oracleTiming(b); candidateGPU += b.gpuSeconds; candidateWall += b.wallSeconds;
          ++counts.primeCalls; oracleEqual(left, right, counts); begin += count;
        }
        require(begin == total && left.logicalLength() == total && right.logicalLength() == total, "teacher pair coverage differs");
        ++counts.checks;
        if (total + 132 < 8192) {
          const auto lastTarget = backend.view(targetFeatures, uint64_t{2047} * kTeacherHyper * 2, uint64_t{kTeacherHyper} * 2);
          uint32_t next = anchor; metal::MetalBuffer features = lastTarget;
          std::vector<uint16_t> firstHidden;
          for (uint32_t depth = 0; depth < 3; ++depth) {
            const auto proposal = oracleProposal(head, candidateHead, left, right, features, std::span(&next, 1), FlashMTPLogits::Last, counts);
            if (depth == 0) firstHidden = proposal.hidden;
            std::memcpy(chained.contents(), proposal.hidden.data(), uint64_t{kTeacherHyper} * 2);
            features = backend.view(chained, 0, uint64_t{kTeacherHyper} * 2); next = proposal.greedy;
          }
          head.truncate(left, total + 1); candidateHead.truncate(right, total + 1); oracleEqual(left, right, counts);
          std::memcpy(chained.contents(), firstHidden.data(), uint64_t{kTeacherHyper} * 2); next = 248046;
          (void)oracleProposal(head, candidateHead, left, right, backend.view(chained, 0, uint64_t{kTeacherHyper} * 2), std::span(&next, 1), FlashMTPLogits::Last, counts);
          const std::array<uint32_t, 4> committed{anchor, 248044, 248046, 198};
          for (uint32_t row = 0; row < 4; ++row)
            std::memcpy(static_cast<uint8_t *>(chained.contents()) + uint64_t{row} * kTeacherHyper * 2, lastTarget.contents(), uint64_t{kTeacherHyper} * 2);
          (void)oracleProposal(head, candidateHead, left, right, chained, committed, FlashMTPLogits::None, counts);
        } else {
          const auto feature = backend.view(targetFeatures, 0, uint64_t{kTeacherHyper} * 2);
          (void)oracleProposal(head, candidateHead, left, right, feature, std::span(&anchor, 1), FlashMTPLogits::Last, counts);
          bool rejected = false; try { (void)candidateHead.primeTeacherCache(right, feature, std::span(&anchor, 1)); }
          catch (const std::invalid_argument &) { rejected = true; }
          require(rejected && !right.poisoned() && right.logicalLength() == 8192, "capacity-bound teacher rejection changed state"); ++counts.checks;
        }
        for (const uint32_t retained : {0u, 1u, 3u, 4u, 127u}) {
          head.truncate(left, retained); candidateHead.truncate(right, retained); oracleEqual(left, right, counts);
          const auto features = backend.view(targetFeatures, 0, uint64_t{128} * kTeacherHyper * 2);
          const auto next = std::span<const uint32_t>(tokens).subspan(1, 128);
          const auto original = head.forward(left, features, next, FlashMTPLogits::None);
          oracleTiming(original.timing);
          const auto candidate = candidateHead.primeTeacherCache(right, features, next);
          oracleTiming(candidate); ++counts.primeCalls; oracleEqual(left, right, counts);
        }
        labels.push_back(label); ++cases;
      }
      auto guard = candidateHead.createState();
      const auto feature = backend.view(targetFeatures, 0, uint64_t{kTeacherHyper} * 2);
      for (const uint32_t test : {0u, 1u, 2u}) {
        bool rejected = false;
        try {
          const uint32_t bad = 248320;
          if (test == 0) (void)candidateHead.primeTeacherCache(guard, feature, {});
          else if (test == 1) (void)candidateHead.primeTeacherCache(guard, feature, std::span(&bad, 1));
          else (void)candidateHead.primeTeacherCache(guard, backend.view(feature, 0, 2), std::span(&anchor, 1));
        } catch (const std::invalid_argument &) { rejected = true; }
        require(rejected && guard.logicalLength() == 0 && !guard.poisoned(), "invalid teacher input mutated logical state"); ++counts.checks;
      }
      {
        auto poisonedLeft = head.createState(), poisonedRight = candidateHead.createState();
        std::memcpy(input.contents(), feature.contents(), uint64_t{kTeacherHyper} * 2);
        static_cast<uint16_t *>(input.contents())[0] = 0x7fc0;
        const auto invalidFeature = backend.view(input, 0, uint64_t{kTeacherHyper} * 2);
        bool originalRejected = false, candidateRejected = false;
        try { (void)head.forward(poisonedLeft, invalidFeature, std::span(&anchor, 1), FlashMTPLogits::None); }
        catch (const std::runtime_error &) { originalRejected = true; }
        try { (void)candidateHead.primeTeacherCache(poisonedRight, invalidFeature, std::span(&anchor, 1)); }
        catch (const std::runtime_error &) { candidateRejected = true; }
        require(originalRejected && candidateRejected && poisonedLeft.poisoned() && poisonedRight.poisoned() &&
            poisonedLeft.logicalLength() == 0 && poisonedRight.logicalLength() == 0,
            "teacher numeric failure did not retain poison/publication semantics");
        ++counts.checks;
      }
      std::ofstream report(argv[4]); require(bool(report), "cannot write teacher oracle report");
      report << std::setprecision(std::numeric_limits<double>::max_digits10)
          << "{\"schema\":\"splash-teacher-cache-prime-oracle-v1\",\"valid\":true,\"gpu_executed\":true"
          << ",\"semantics\":" << json::quote(kFlashMTPTeacherCacheSemantics)
          << ",\"source_identity_sha256\":" << json::quote(weights.sourceIdentity())
          << ",\"metallib_sha256\":" << json::quote(hexadecimal(backend.metallibSha256()))
          << ",\"tokens_u32le_sha256\":" << json::quote(digest(tokens.data(), tokens.size() * 4))
          << ",\"separate_baseline_candidate_workspaces\":true,\"full_cache_bytes_equal\":true,\"future_hidden_logits_greedy_equal\":true,\"generic_none_contract_preserved\":true"
          << ",\"all_persistent_head_planes_are_bf16_or_i64\":true,\"bf16_to_f32_values_finite\":true,\"rollback_equal\":true"
          << ",\"numeric_failure_poison_semantics_preserved\":true"
          << ",\"case_count\":" << cases << ",\"checks\":" << counts.checks << ",\"compared_bytes\":" << counts.comparedBytes
          << ",\"finite_bf16_words_checked\":" << counts.finiteBF16Words << ",\"teacher_pair_calls\":" << counts.primeCalls
          << ",\"future_proposal_or_fold_calls\":" << counts.proposalCalls
          << ",\"normal_gpu_seconds\":" << normalGPU << ",\"candidate_gpu_seconds\":" << candidateGPU
          << ",\"normal_command_wall_seconds\":" << normalWall << ",\"candidate_command_wall_seconds\":" << candidateWall
          << ",\"timing_scope\":\"oracle timings include different contexts and follow cache/state comparisons; use separate matched normal/HTTP controls for speed\""
          << ",\"case_labels\":[";
      for (size_t i = 0; i < labels.size(); ++i) { if (i) report << ','; report << json::quote(labels[i]); }
      report << "]}\n"; require(bool(report), "teacher oracle report write failed");
      std::cout << "teacher cache oracle passed; report=" << argv[4] << '\n'; return 0;
    } catch (const std::exception &error) { std::cerr << "teacher cache oracle failed: " << error.what() << '\n'; return 1; }
  }
}
