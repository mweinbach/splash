// Root-exclusive cache/future-head qualification. --help submits no GPU work.
#define main splash_prefill4k_attribution_unused_main
#include "dev/benchmarks/prefill4k_attribution.mm"
#undef main
#include "flash/FlashMTPStateInternal.hpp"
#include "dev/benchmarks/mtp_teacher_bulk_sep21/bulk.hpp"
#include "dev/benchmarks/dense_w8a8_sep21/worker_bridge.hpp"
#include "dev/benchmarks/mtp_teacher_bulk_sep21/policy.hpp"

namespace splash::flash {
struct FlashMTPTeacherPrimeOracleAccess final {
  static std::array<metal::MetalBuffer, 5> buffers(const FlashMTPState &state) {
    if (!state.impl_) throw std::invalid_argument("oracle state is uninitialized");
    const auto &q = state.impl_->qsa;
    return {q.keys, q.values, q.rawIndexKeys, q.pooledKeys, q.indexPositions};
  }
  static void bind(FlashMTPState &state,const std::array<metal::MetalBuffer,5> &buffers) {
    auto &q=state.impl_->qsa;
    q.keys=buffers[0];q.values=buffers[1];q.rawIndexKeys=buffers[2];
    q.pooledKeys=buffers[3];q.indexPositions=buffers[4];
  }
};
}
namespace {
std::filesystem::path progressPath;
std::vector<std::string> progressCases;
std::string progressPhase="arguments";
bool progressBackend=false;
uint64_t progressMemoryPeak=0,progressReserved=0;
void checkpoint(bool failed=false,const std::string &error={}) {
  if(progressPath.empty())return;
  const auto path=progressPath.string()+(failed?".failure.json":".checkpoint.json");
  const auto temporary=path+".writing";
  std::ofstream out(temporary,std::ios::trunc);
  if(!out)throw std::runtime_error("cannot write teacher progress");
  out<<"{\"schema\":\"teacher-bulk-progress-v1\",\"valid\":false,\"qualification_complete\":false,\"failed\":"
      <<(failed?"true":"false")<<",\"phase\":"<<json::quote(progressPhase)
      <<",\"error\":"<<json::quote(error)<<",\"backend_initialized\":"<<(progressBackend?"true":"false")
      <<",\"peak_allocated_bytes\":"<<progressMemoryPeak<<",\"reserved_growth_bytes\":"<<progressReserved
      <<",\"completed_cases\":[";
  for(size_t i=0;i<progressCases.size();++i){if(i)out<<',';out<<json::quote(progressCases[i]);}
  out<<"]}\n";out.close();std::filesystem::rename(temporary,path);
}
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
struct OracleProposal { std::vector<uint16_t> hidden, logits; std::vector<uint8_t> greedyRecords; uint32_t greedy = 0,greedyRows=0; };
OracleProposal oracleProposal(FlashMTPForward &head, FlashMTPForward &candidateHead, FlashMTPState &left, FlashMTPState &right,
    metal::MetalBuffer hidden, std::span<const uint32_t> tokens, FlashMTPLogits mode, OracleCounters &counts) {
  const auto first = head.forward(left, hidden, tokens, mode);
  oracleTiming(first.timing);
  OracleProposal a; a.hidden = oracleWords(first.hiddenBF16);
  if (mode != FlashMTPLogits::None) {
    a.logits = oracleWords(first.logitsBF16);
    require(first.greedyResultsU32 && first.greedyResultsU32.contents(), "proposal GPU greedy is absent");
    a.greedyRows=first.greedyRows;
    const auto *record=static_cast<const uint8_t *>(first.greedyResultsU32.contents());
    a.greedyRecords.assign(record,record+uint64_t{a.greedyRows}*sizeof(FlashGreedyGPURowResult));
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
    require(second.greedyRows==a.greedyRows && std::memcmp(second.greedyResultsU32.contents(),
        a.greedyRecords.data(),a.greedyRecords.size())==0,"future all greedy records differ");
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
        std::cout << "usage: teacher-bulk-oracle METALLIB PACKAGE EXACT_2048_TOKENS_JSON REPORT_JSON\n"
            "Root-exclusive full QSA cache bytes, finite BF16/F32 values, future proposals, rollback, EOS, partial128 boundaries.\n";
        return 0;
      }
      require(argc == 5, "invalid teacher oracle arguments");
      const bool bulkFlag=teacher_bulk_sep21::parse(std::getenv(teacher_bulk_sep21::flag));
      teacher_bulk_sep21::validate(bulkFlag,{enabled("SPLASH_FLASH_MTP"),enabled("SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY"),
          enabled("SPLASH_FLASH_DENSE_CACHE"),enabled("SPLASH_FLASH_MTP_QSA_F32"),enabled("SPLASH_FLASH_MTP_QSA_MPP")});
      require(!std::filesystem::exists(argv[4]), "choose a fresh teacher oracle report");
      progressPath=argv[4];checkpoint();
      const auto tokens = loadTokens(argv[3]); require(tokens.size() == 2048, "teacher oracle requires true 2048-row target fixture");
      metal::MetalBackend backend(argv[1]);
      progressBackend=true;progressPhase="model-and-admission";checkpoint();
      const auto weights = FlashWeights::load(backend, argv[2]);
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      engine::MemoryGovernor governor(backend, physical - reserve, reserve);
      uint64_t planned = FlashForward::workspacePlannedBytes(8192, 2048, 4) + FlashForward::requestStateBytes(8192) +
          FlashForward::expertCachePlannedBytes(weights) + FlashForward::floatDenseCachePlannedBytes(weights) + FlashForward::int8HeadPlannedBytes(weights) +
          2 * FlashMTPForward::workspacePlannedBytes(8192, 128) + 4 * FlashMTPForward::requestStateBytes(8192) + (256ULL << 20) + FlashMTPTeacherBulkForward::plannedBytes;
      if(dense_w8a8_sep21::requiresCache(2048))planned+=dense_w8a8_sep21::Cache::plannedBytes();
      if (enabled("SPLASH_FLASH_DENSE_CACHE")) planned += FlashDenseCache::plannedBytes(weights, FlashDenseCache::defaultPrefixes(weights, true)) + 2 * FlashMTPForward::denseCachePlannedBytes(weights);
      if (enabled("SPLASH_FLASH_QSA_F32")) planned += 16ULL << 20;
      if (enabled("SPLASH_FLASH_BLOCKED_MOE")) planned += flashMoEBlockedWorkspacePlannedBytes(2048, 10);
      const auto initialAllocation=backend.memoryStats().allocatedBytes;
      progressReserved=planned;
      auto reservation = governor.tryReserve(planned); require(bool(reservation), "teacher oracle governor denied construction");
      FlashForward target(backend, weights, 8192, 2048, 4);
      FlashMTPForward head(backend, weights, 8192, 128), candidateHead(backend, weights, 8192, 128);
      FlashMTPTeacherBulkForward bulk(candidateHead);
      require(bulk.workspaceBytes() <= FlashMTPTeacherBulkForward::plannedBytes,"bulk arena exceeds reserved plan");
      require(target.workspaceBytes() + head.workspaceBytes() + candidateHead.workspaceBytes() + FlashForward::requestStateBytes(8192) +
          4 * FlashMTPForward::requestStateBytes(8192) <= planned, "teacher oracle arenas exceed reservation");
      auto targetState = target.createState();
      auto targetFeatures = backend.allocateBuffer(uint64_t{2048} * kTeacherHyper * 2, metal::BufferStorage::Shared, "teacher-oracle-owned-real-target-features");
      auto input = backend.allocateBuffer(uint64_t{2048} * kTeacherHyper * 2, metal::BufferStorage::Shared, "teacher-oracle-owned-real-pair-features");
      auto chained = backend.allocateBuffer(uint64_t{4} * kTeacherHyper * 2, metal::BufferStorage::Shared, "teacher-oracle-owned-chained-features");
      metal::ResidencyLease residency;
      if (enabled("SPLASH_FLASH_SAVED_OPERANDS_RESIDENT")) {
        auto buffers = target.cachedOperandsOnly(); const auto hb = head.cachedOperandsOnly(); buffers.insert(buffers.end(), hb.begin(), hb.end());
        const auto cb = candidateHead.cachedOperandsOnly(); buffers.insert(buffers.end(), cb.begin(), cb.end());
        if (!buffers.empty()) residency = backend.requestWeightResidency(buffers, "teacher oracle saved operands only");
      }
      reservation->commit();
      const auto main = target.forward(targetState, tokens, false, true); oracleTiming(main.timing);
      progressPhase="real-target-completed";progressMemoryPeak=backend.memoryStats().peakAllocatedBytes;checkpoint();
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
        else {
          uint32_t remaining=total;
          while(remaining>=128) {
            const auto complete=std::min(2048u,remaining/128*128);
            chunks.push_back(complete);remaining-=complete;
          }
          if(remaining)chunks.push_back(remaining);
        }
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
          for(uint32_t offset=0;offset<count;offset+=128) {
            const auto size=std::min(128u,count-offset);
            const auto controlFeatures=backend.view(features,uint64_t{offset}*kTeacherHyper*2,uint64_t{size}*kTeacherHyper*2);
            const auto timing=head.primeTeacherCache(left,controlFeatures,std::span<const uint32_t>(next).subspan(offset,size));
            oracleTiming(timing);normalGPU+=timing.gpuSeconds;normalWall+=timing.wallSeconds;
          }
          const auto b=count>=128?bulk.primeTeacherCache(right,features,next)
              :candidateHead.primeTeacherCache(right,features,next);
          oracleTiming(b);candidateGPU+=b.gpuSeconds;candidateWall+=b.wallSeconds;
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
          (void)oracleProposal(head, candidateHead, left, right, chained, committed, FlashMTPLogits::All, counts);
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
          const auto original = head.primeTeacherCache(left, features, next);
          oracleTiming(original);
          const auto candidate = bulk.primeTeacherCache(right, features, next);
          oracleTiming(candidate); ++counts.primeCalls; oracleEqual(left, right, counts);
        }
        labels.push_back(label); ++cases;progressCases.push_back(label);
        progressPhase="cache-future-cases";progressMemoryPeak=backend.memoryStats().peakAllocatedBytes;checkpoint();
      }
      // Seed arbitrary chronology with the original128-row API, then compare
      // the new bulk append across partial compression-block boundaries.
      for(const uint32_t seedLength:{511u,512u,2047u}) {
        auto left=head.createState(),right=candidateHead.createState();
        for(uint32_t offset=0;offset<seedLength;offset+=128) {
          const auto count=std::min(128u,seedLength-offset);
          const auto features=backend.view(targetFeatures,uint64_t{offset}*kTeacherHyper*2,uint64_t{count}*kTeacherHyper*2);
          const auto next=std::span<const uint32_t>(tokens).subspan(offset+1,count);
          (void)head.primeTeacherCache(left,features,next);(void)candidateHead.primeTeacherCache(right,features,next);
        }
        const auto features=backend.view(targetFeatures,0,uint64_t{128}*kTeacherHyper*2);
        const auto next=std::span<const uint32_t>(tokens).subspan(1,128);
        (void)head.primeTeacherCache(left,features,next);(void)bulk.primeTeacherCache(right,features,next);
        oracleEqual(left,right,counts);
        const auto future=backend.view(targetFeatures,0,uint64_t{4}*kTeacherHyper*2);
        (void)oracleProposal(head,candidateHead,left,right,future,std::span<const uint32_t>(tokens).first(4),FlashMTPLogits::All,counts);
        progressCases.push_back("bulk-begin"+std::to_string(seedLength));checkpoint();
      }
      {
        auto left=head.createState(),right=candidateHead.createState();
        std::array<std::array<metal::MetalBuffer,5>,2> guardedOwners;
        for(uint32_t policy=0;policy<2;++policy) {
          auto &state=policy?right:left;const auto original=FlashMTPTeacherPrimeOracleAccess::buffers(state);
          std::array<metal::MetalBuffer,5> views;
          for(size_t i=0;i<5;++i) {
            const auto size=original[i].sizeBytes();
            auto owner=backend.allocateBuffer(size+32768,metal::BufferStorage::Shared,"teacher oracle guarded cache");
            std::memset(owner.contents(),0xa5,owner.sizeBytes());
            std::memcpy(static_cast<uint8_t *>(owner.contents())+16384,original[i].contents(),size);
            views[i]=backend.view(owner,16384,size);guardedOwners[policy][i]=owner;
          }
          FlashMTPTeacherPrimeOracleAccess::bind(state,views);
        }
        const auto arena=bulk.oracleWorkspaceBuffers();
        constexpr std::array<uint32_t,17> rowBytes{8,5120,5120,5120,20480,20480,20480,20480,
            640,20480,8,8,5120,24576,1024,1024,1280};
        for(const auto &buffer:arena)std::memset(buffer.contents(),0xa5,buffer.sizeBytes());
        const auto features=backend.view(targetFeatures,0,uint64_t{1920}*kTeacherHyper*2);
        const auto next=std::span<const uint32_t>(tokens).subspan(1,1920);
        for(uint32_t offset=0;offset<1920;offset+=128)(void)head.primeTeacherCache(left,
            backend.view(features,uint64_t{offset}*kTeacherHyper*2,uint64_t{128}*kTeacherHyper*2),next.subspan(offset,128));
        (void)bulk.primeTeacherCache(right,features,next);oracleEqual(left,right,counts);
        const auto allCanary=[&](const metal::MetalBuffer &buffer,uint64_t start,uint64_t length) {
          const auto *bytes=static_cast<const uint8_t *>(buffer.contents());
          for(uint64_t offset=0;offset<length;++offset)require(bytes[start+offset]==0xa5,"teacher physical guard damaged");
          ++counts.checks;
        };
        for(size_t i=0;i<arena.size();++i){
          const uint64_t start=i<17?uint64_t{1920}*rowBytes[i]:4;
          allCanary(arena[i],start,arena[i].sizeBytes()-start);
        }
        for(const auto &policy:guardedOwners)for(const auto &buffer:policy){
          allCanary(buffer,0,16384);allCanary(buffer,buffer.sizeBytes()-16384,16384);
        }
        progressCases.push_back("inactive18arena-tails-and10cache-owner-redzones");checkpoint();
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
      {
        auto guard=candidateHead.createState();auto foreign=head.createState();FlashMTPState absent;
        const auto features=backend.view(targetFeatures,0,uint64_t{128}*kTeacherHyper*2);
        const auto valid=std::span<const uint32_t>(tokens).subspan(1,128);
        const auto unchanged=[&](auto &&operation,FlashMTPState &state) {
          const auto length=state.logicalLength();const auto capacity=state.capacity();
          const bool poison=state.poisoned();
          std::array<std::vector<uint8_t>,5> before;
          std::array<metal::MetalBuffer,5> original{};
          if(capacity) {
            original=FlashMTPTeacherPrimeOracleAccess::buffers(state);
            for(size_t i=0;i<5;++i) {
              const auto *data=static_cast<const uint8_t *>(original[i].contents());
              before[i].assign(data,data+original[i].sizeBytes());
            }
          }
          bool rejected=false;try {operation();}catch(const std::invalid_argument&){rejected=true;}
          require(rejected && state.logicalLength()==length && state.capacity()==capacity && state.poisoned()==poison,"bulk host guard changed metadata");
          if(capacity) {
            const auto after=FlashMTPTeacherPrimeOracleAccess::buffers(state);
            for(size_t i=0;i<5;++i) require(after[i].sameView(original[i]) &&
                std::memcmp(after[i].contents(),before[i].data(),before[i].size())==0,"bulk host guard changed cache/view");
          }
          ++counts.checks;
        };
        unchanged([&]{(void)bulk.primeTeacherCache(absent,features,valid);},absent);
        unchanged([&]{(void)bulk.primeTeacherCache(foreign,features,valid);},foreign);
        unchanged([&]{(void)bulk.primeTeacherCache(guard,features,{});},guard);
        unchanged([&]{(void)bulk.primeTeacherCache(guard,features,valid.first(127));},guard);
        unchanged([&]{(void)bulk.primeTeacherCache(guard,backend.view(features,0,2),valid);},guard);
        std::vector<uint32_t> invalid(valid.begin(),valid.end());invalid.back()=248320;
        unchanged([&]{(void)bulk.primeTeacherCache(guard,features,invalid);},guard);
        const auto cache=FlashMTPTeacherPrimeOracleAccess::buffers(guard);
        unchanged([&]{(void)bulk.primeTeacherCache(guard,cache[0],valid);},guard);
        unchanged([&]{(void)bulk.primeTeacherCache(guard,bulk.oracleWorkspaceBuffers()[6],valid);},guard);
        const auto foreignCoefficients=weights.projection("mtp.fc_hidden").weights->buffer;
        unchanged([&]{(void)bulk.primeTeacherCache(guard,foreignCoefficients,valid);},guard);
        candidateHead.truncate(guard,0);
        auto poisoned=head.createState(),bulkPoisoned=candidateHead.createState();
        std::memcpy(input.contents(),features.contents(),uint64_t{128}*kTeacherHyper*2);
        static_cast<uint16_t *>(input.contents())[0]=0x7fc0;
        const auto invalidFeatures=backend.view(input,0,uint64_t{128}*kTeacherHyper*2);
        bool controlRejected=false,bulkRejected=false;
        try{(void)head.primeTeacherCache(poisoned,invalidFeatures,valid);}catch(const std::runtime_error&){controlRejected=true;}
        try{(void)bulk.primeTeacherCache(bulkPoisoned,invalidFeatures,valid);}catch(const std::runtime_error&error){
          bulkRejected=std::string(error.what()).find("teacher bulk sticky diagnostics failed:")==0;
        }
        require(controlRejected && bulkRejected && poisoned.poisoned() && bulkPoisoned.poisoned() &&
            poisoned.logicalLength()==0 && bulkPoisoned.logicalLength()==0,"bulk numeric failure publication differs");
        unchanged([&]{(void)bulk.primeTeacherCache(bulkPoisoned,features,valid);},bulkPoisoned);
      }
      {
        auto control=head.createState(),candidate=candidateHead.createState();
        const auto features=backend.view(targetFeatures,0,uint64_t{128}*kTeacherHyper*2);
        const auto next=std::span<const uint32_t>(tokens).subspan(1,128);
        const auto oldControl=FlashMTPTeacherPrimeOracleAccess::buffers(control);
        const auto oldCandidate=FlashMTPTeacherPrimeOracleAccess::buffers(candidate);
        std::array<std::vector<uint8_t>,5> beforeControl,beforeCandidate;
        for(size_t i=0;i<5;++i) {
          const auto *a=static_cast<const uint8_t *>(oldControl[i].contents());
          const auto *b=static_cast<const uint8_t *>(oldCandidate[i].contents());
          beforeControl[i].assign(a,a+oldControl[i].sizeBytes());beforeCandidate[i].assign(b,b+oldCandidate[i].sizeBytes());
        }
        backend.setCancellationProbe([]{return true;});
        bool controlRejected=false,candidateRejected=false;
        try{(void)head.primeTeacherCache(control,features,next);}catch(const metal::MetalBackendError&){controlRejected=true;}
        try{(void)bulk.primeTeacherCache(candidate,features,next);}catch(const metal::MetalBackendError&){candidateRejected=true;}
        backend.setCancellationProbe({});
        require(controlRejected && candidateRejected && control.poisoned() && candidate.poisoned() &&
            control.logicalLength()==0 && candidate.logicalLength()==0,"cancelled precommit publication differs");
        for(size_t i=0;i<5;++i)require(std::memcmp(oldControl[i].contents(),beforeControl[i].data(),beforeControl[i].size())==0 &&
            std::memcmp(oldCandidate[i].contents(),beforeCandidate[i].data(),beforeCandidate[i].size())==0,"precommit cancellation wrote cache");
        // Cancellation after completed publication retains the healthy1920
        // prefix, matching Worker safe-point semantics; no tail is submitted.
        auto published=candidateHead.createState();
        (void)bulk.primeTeacherCache(published,backend.view(targetFeatures,0,uint64_t{1920}*kTeacherHyper*2),
            std::span<const uint32_t>(tokens).subspan(1,1920));
        backend.setCancellationProbe([]{return true;});
        require(published.logicalLength()==1920 && !published.poisoned(),"post-publication cancellation changed state");
        backend.setCancellationProbe({});
        progressCases.push_back("precommit-cancel-and-healthy1920-completion");checkpoint();
      }
      // Whole2047-pair sequences. Only real target features/token views; no
      // CPU cache/input reads occur between start and end of either sequence.
      auto timedControl=head.createState(),timedCandidate=candidateHead.createState();
      struct SequenceTiming {double gpu=0,call=0;uint32_t commands=0,pairs=0;};
      const auto sequence=[&](bool useBulk) {
        auto &state=useBulk?timedCandidate:timedControl;
        auto &owner=useBulk?candidateHead:head;owner.truncate(state,0);
        SequenceTiming result;const auto began=Clock::now();
        for(uint32_t offset=0;offset<2047;) {
          const uint32_t count=useBulk && offset==0?1920:std::min(128u,2047-offset);
          const auto features=backend.view(targetFeatures,uint64_t{offset}*kTeacherHyper*2,uint64_t{count}*kTeacherHyper*2);
          const auto next=std::span<const uint32_t>(tokens).subspan(offset+1,count);
          const auto timing=useBulk && count>=128?bulk.primeTeacherCache(state,features,next):owner.primeTeacherCache(state,features,next);
          result.gpu+=timing.gpuSeconds;++result.commands;result.pairs+=count;offset+=count;
        }
        result.call=std::chrono::duration<double>(Clock::now()-began).count();return result;
      };
      std::array<double,2> warmGPU{};uint32_t warmPairs=0;
      while(warmPairs<10 || warmGPU[0]<.150 || warmGPU[1]<.150) {
        for(uint32_t k=0;k<2;++k) {const bool policy=(warmPairs+k)%2;const auto timing=sequence(policy);warmGPU[policy]+=timing.gpu;}
        ++warmPairs;require(warmPairs<=1000,"teacher sequence warmup did not progress");
      }
      std::array<double,2> measuredGPU{},measuredCalls{};std::array<uint32_t,2> measuredCommands{};
      constexpr uint32_t measuredPairs=20;
      for(uint32_t pair=0;pair<measuredPairs;++pair)for(uint32_t k=0;k<2;++k) {
        const bool policy=(pair+k)%2;const auto timing=sequence(policy);
        require(timing.commands==(policy?2u:16u) && timing.pairs==2047,"teacher timing sequence coverage differs");
        measuredGPU[policy]+=timing.gpu;measuredCalls[policy]+=timing.call;measuredCommands[policy]+=timing.commands;
      }
      oracleEqual(timedControl,timedCandidate,counts);
      const auto finalMemory=backend.memoryStats();
      require(finalMemory.peakAllocatedBytes>=initialAllocation &&
          finalMemory.peakAllocatedBytes-initialAllocation<=planned,"teacher arena/test peak exceeds governor reservation");
      progressMemoryPeak=finalMemory.peakAllocatedBytes;progressPhase="all-cases-and-timing-complete";checkpoint();
      std::ofstream report(argv[4]); require(bool(report), "cannot write teacher oracle report");
      report << std::setprecision(std::numeric_limits<double>::max_digits10)
          << "{\"schema\":\"splash-singleton-teacher-bulk2048-oracle-v1\",\"valid\":true,\"gpu_executed\":true"
          << ",\"semantics\":" << json::quote(kFlashMTPTeacherCacheSemantics)
          << ",\"source_identity_sha256\":" << json::quote(weights.sourceIdentity())
          << ",\"metallib_sha256\":" << json::quote(hexadecimal(backend.metallibSha256()))
          << ",\"tokens_u32le_sha256\":" << json::quote(digest(tokens.data(), tokens.size() * 4))
          << ",\"separate_baseline_candidate_workspaces\":true,\"full_cache_bytes_equal\":true,\"future_hidden_logits_greedy_equal\":true,\"generic_none_contract_preserved\":true"
          << ",\"all_persistent_head_planes_are_bf16_or_i64\":true,\"bf16_to_f32_values_finite\":true,\"rollback_equal\":true"
          << ",\"numeric_failure_poison_semantics_preserved\":true"
          << ",\"bulk_workspace_planned_bytes\":" << FlashMTPTeacherBulkForward::plannedBytes
          << ",\"bulk_workspace_actual_bytes\":" << bulk.workspaceBytes()
          << ",\"real_governor_reservation_before_bulk_allocation\":true"
          << ",\"reserved_growth_bytes\":" << planned << ",\"initial_allocated_bytes\":" << initialAllocation
          << ",\"final_allocated_bytes\":" << finalMemory.allocatedBytes << ",\"peak_allocated_bytes\":" << finalMemory.peakAllocatedBytes
          << ",\"peak_growth_bytes\":" << finalMemory.peakAllocatedBytes-initialAllocation
          << ",\"qualification_complete\":true,\"ten_cache_owner_redzones_preserved\":true,\"eighteen_arena_inactive1920_tails_preserved\":true,\"precommit_cancel_no_cache_write_poison_preserved\":true,\"completed1920_healthy_publication_preserved\":true,\"worker_postbulk_cancel_integration_tested\":false"
          << ",\"warm_pairs\":" << warmPairs << ",\"control_warm_gpu_seconds\":" << warmGPU[0]
          << ",\"candidate_warm_gpu_seconds\":" << warmGPU[1]
          << ",\"matched_whole_sequence_pairs\":" << measuredPairs
          << ",\"control_commands_per2047pairs\":16,\"candidate_commands_per2047pairs\":2"
          << ",\"control_whole_gpu_ms\":" << measuredGPU[0]*1000/measuredPairs
          << ",\"candidate_whole_gpu_ms\":" << measuredGPU[1]*1000/measuredPairs
          << ",\"control_whole_call_ms\":" << measuredCalls[0]*1000/measuredPairs
          << ",\"candidate_whole_call_ms\":" << measuredCalls[1]*1000/measuredPairs
          << ",\"control_measured_commands\":" << measuredCommands[0]
          << ",\"candidate_measured_commands\":" << measuredCommands[1]
          << ",\"actual_target_premixer_hidden_sha256\":" << json::quote(digest(targetFeatures.contents(),targetFeatures.sizeBytes()))
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
    } catch (const std::exception &error) {
      try{checkpoint(true,error.what());}catch(const std::exception &failure){std::cerr<<"failure checkpoint: "<<failure.what()<<'\n';}
      std::cerr << "teacher cache oracle failed: " << error.what() << '\n'; return 1;
    }
  }
}
