// Private full trained-head proposal benchmark; never changes a target graph.
// --cpu-self-test exits before Metal device creation or model/fixture loading.
#include "FlashMTPFloatCandidate.hpp"
#include "FlashBatchMTPFloatCandidate.hpp"
#include "flash/FlashBatchMTPForward.hpp"
#include "flash/FlashDenseCache.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace base = splash::flash;
namespace candidate = splash::flash::mtp_float_candidate;
namespace metal = splash::metal;
constexpr uint32_t kWidth = 10240, kVocabulary = 248320, kCapacity = 4096;
constexpr uint32_t kMaximumRows = 128, kSamples = 6;
constexpr double kProposalLimit = .01;
enum class Mode : uint8_t { None, Last, All };

void require(bool value, std::string_view message) {
  if (!value) throw std::runtime_error(std::string(message));
}
bool flagValue(std::string_view value) {
  if (value == "1") return true;
  if (value == "0") return false;
  throw std::invalid_argument("oracle flag must be canonical 0 or 1");
}
bool flag(const char *name, bool fallback = false) {
  const char *value = std::getenv(name);
  return value ? flagValue(value) : fallback;
}
uint32_t oracleDecimal(std::string_view text, uint32_t minimum, uint32_t maximum) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string_view::npos ||
      (text.size() > 1 && text.front() == '0'))
    throw std::invalid_argument("oracle row count must be canonical decimal");
  uint64_t result = 0;
  for (char character : text) {
    if (result > (uint64_t{maximum} - uint32_t(character - '0')) / 10)
      throw std::invalid_argument("oracle row count exceeds supported range");
    result = result * 10 + uint32_t(character - '0');
  }
  if (result < minimum || result > maximum)
    throw std::invalid_argument("oracle row count is outside supported range");
  return uint32_t(result);
}
uint16_t bf16(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  if ((word & 0x7f800000u) == 0x7f800000u)
    return uint16_t((word >> 16) | ((word & 0x7fffffu) ? 0x40u : 0u));
  return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
float number(uint16_t word) { return std::bit_cast<float>(uint32_t{word} << 16); }
const char *modeName(Mode mode) {
  return mode == Mode::None ? "None" : mode == Mode::Last ? "Last" : "All";
}
uint32_t logitRows(uint32_t lanes, uint32_t rows, Mode mode) {
  if (!lanes || lanes > 4 || !rows || rows > kMaximumRows ||
      (mode == Mode::All && lanes * rows > 16))
    throw std::invalid_argument("unsupported proposal geometry");
  return mode == Mode::None ? 0 : mode == Mode::Last ? lanes : lanes * rows;
}
base::FlashMTPLogits baseMode(Mode mode) { return static_cast<base::FlashMTPLogits>(mode); }
candidate::FlashMTPLogits candidateMode(Mode mode) {
  return static_cast<candidate::FlashMTPLogits>(mode);
}
std::vector<uint16_t> copy(const metal::MetalBuffer &buffer, uint64_t words) {
  if (!words) { require(!buffer, "None unexpectedly returned a borrowed output"); return {}; }
  require(buffer && buffer.contents() && buffer.storage() == metal::BufferStorage::Shared &&
          words <= UINT64_MAX / 2 && buffer.sizeBytes() >= words * 2,
          "invalid borrowed BF16 output");
  const auto *begin = static_cast<const uint16_t *>(buffer.contents());
  return {begin, begin + words};
}
struct Error {
  uint64_t elements = 0, mismatches = 0, nonfinite = 0;
  double squaredError = 0, squaredReference = 0, maximumAbsolute = 0;
  void add(std::span<const uint16_t> actual, std::span<const uint16_t> expected) {
    require(actual.size() == expected.size(), "proposal comparison extents differ");
    for (size_t index = 0; index < actual.size(); ++index) {
      ++elements; mismatches += actual[index] != expected[index];
      const double a = number(actual[index]), b = number(expected[index]);
      if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; continue; }
      const double difference = a - b;
      squaredError += difference * difference; squaredReference += b * b;
      maximumAbsolute = std::max(maximumAbsolute, std::abs(difference));
    }
  }
  double relative() const {
    return std::sqrt(squaredError / std::max(1e-30, squaredReference));
  }
  bool acceptable() const { return !nonfinite && relative() <= kProposalLimit; }
  void write(std::ostream &out) const {
    out << "{\"elements\":" << elements << ",\"bf16_mismatches\":" << mismatches
        << ",\"nonfinite\":" << nonfinite << ",\"max_abs\":" << maximumAbsolute
        << ",\"relative_l2\":" << relative() << '}';
  }
};
double median(std::vector<double> values) {
  require(!values.empty(), "empty timing sample");
  std::sort(values.begin(), values.end());
  return values.size() % 2 ? values[values.size()/2] :
      (values[values.size()/2-1] + values[values.size()/2]) * .5;
}
struct Times {
  std::vector<double> gpu, commandWall, forwardWall;
  void write(std::ostream &out) const {
    out << "{\"samples\":" << gpu.size() << ",\"median_gpu_seconds\":" << median(gpu)
        << ",\"median_command_wall_seconds\":" << median(commandWall)
        << ",\"median_forward_wall_seconds\":" << median(forwardWall)
        << ",\"gpu_seconds\":[";
    for (size_t i=0;i<gpu.size();++i) { if(i) out << ','; out << gpu[i]; }
    out << "],\"forward_wall_seconds\":[";
    for (size_t i=0;i<forwardWall.size();++i) { if(i) out << ','; out << forwardWall[i]; }
    out << "]}";
  }
};
struct Snapshot {
  std::vector<uint16_t> hidden, logits;
  std::vector<uint64_t> lengths;
  metal::CommandTiming timing;
  double forwardWall = 0;
  uint32_t logitsRows = 0;
};
void validTiming(const metal::CommandTiming &timing) {
  require(std::isfinite(timing.gpuSeconds) && timing.gpuSeconds > 1e-9 &&
          std::isfinite(timing.wallSeconds) && timing.wallSeconds > 1e-9,
          "forward timing is zero/nonfinite; use fresh objects matching current CommandTiming ABI");
}
void sample(Times &times, const Snapshot &result) {
  validTiming(result.timing);
  require(std::isfinite(result.forwardWall) && result.forwardWall > 1e-9,
          "forward wall timing is zero/nonfinite");
  times.gpu.push_back(result.timing.gpuSeconds);
  times.commandWall.push_back(result.timing.wallSeconds);
  times.forwardWall.push_back(result.forwardWall);
}
uint32_t argmax(std::span<const uint16_t> row) {
  require(!row.empty(), "empty vocabulary row");
  uint32_t best = 0;
  for (uint32_t column=0;column<row.size();++column) {
    require(std::isfinite(number(row[column])), "nonfinite proposal vocabulary");
    if (number(row[column]) > number(row[best])) best = column;
  }
  return best; // Lowest token ID wins exact ties.
}
struct Argmax {
  uint64_t compared = 0, matched = 0;
  void add(const Snapshot &actual, const Snapshot &expected) {
    require(actual.logitsRows == expected.logitsRows, "vocabulary row metadata differs");
    for (uint32_t row=0;row<actual.logitsRows;++row) {
      ++compared;
      matched += argmax(std::span<const uint16_t>(actual.logits).subspan(uint64_t{row}*kVocabulary,kVocabulary)) ==
          argmax(std::span<const uint16_t>(expected.logits).subspan(uint64_t{row}*kVocabulary,kVocabulary));
    }
  }
};
template<class F> void rejected(F operation) {
  bool caught = false;
  try { operation(); } catch (const std::invalid_argument &) { caught = true; }
  require(caught, "malformed proposal call was accepted");
}
std::vector<uint16_t> hiddenFixture() {
  const char *path = std::getenv("FLASH_MTP_FLOAT_CANDIDATE_HIDDEN_BF16");
  if (!path) return {};
  require(*path, "hidden fixture path cannot be empty");
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  require(bool(file), "cannot open captured BF16 hidden fixture");
  const auto bytes = file.tellg();
  require(bytes > 0 && uint64_t(bytes) % (uint64_t{kWidth}*2) == 0 &&
          uint64_t(bytes) <= (256ULL<<20), "hidden fixture must contain complete BF16[rows,10240]");
  std::vector<uint16_t> result(uint64_t(bytes)/2);
  file.seekg(0); file.read(reinterpret_cast<char *>(result.data()),std::streamsize(bytes));
  require(bool(file), "captured BF16 hidden fixture is short");
  for(auto word:result) require(std::isfinite(number(word)), "captured hidden fixture is nonfinite");
  return result;
}
std::vector<uint32_t> fill(const metal::MetalBuffer &buffer,uint32_t lanes,uint32_t rows,
                          std::span<const uint64_t> starts,
                          std::span<const uint16_t> fixture,uint32_t salt=0) {
  require(starts.size() >= lanes && buffer.contents() &&
          buffer.sizeBytes() >= uint64_t{lanes}*rows*kWidth*2, "fixture extent invalid");
  std::vector<uint32_t> tokens;
  auto *destination = static_cast<uint16_t *>(buffer.contents());
  for(uint32_t lane=0;lane<lanes;++lane) for(uint32_t row=0;row<rows;++row) {
    const uint64_t position=starts[lane]+row;
    tokens.push_back(uint32_t((lane*173+position*31+salt*17+7)%240000));
    for(uint32_t column=0;column<kWidth;++column) {
      const uint64_t index=(uint64_t{lane}*rows+row)*kWidth+column;
      if(!fixture.empty()) {
        const uint64_t sourceRow=(position+lane*131+salt)%(fixture.size()/kWidth);
        destination[index]=fixture[sourceRow*kWidth+column];
      } else {
        const float value=std::sin(float(column%997)*.041f+float(position)*.173f+
            float(lane+salt)*.3f)*(.2f+float((column/2560+lane)%4)*.12f);
        destination[index]=bf16(value);
      }
    }
  }
  return tokens;
}
void cpuSelfTest() {
  uint64_t checks=0;
  for(uint32_t lanes:{1u,4u}) for(uint32_t rows:{1u,4u,8u})
    for(Mode mode:{Mode::None,Mode::Last,Mode::All}) {
      if(mode==Mode::All&&lanes*rows>16) {
        rejected([&]{(void)logitRows(lanes,rows,mode);}); ++checks; continue;
      }
      require(logitRows(lanes,rows,mode)==(mode==Mode::None?0:mode==Mode::Last?lanes:lanes*rows),
              "CPU row-mode contract failed"); ++checks;
    }
  for(auto text:{"0","1"}) { (void)flagValue(text); ++checks; }
  for(auto text:{"","true","01"," 1","2"}) {
    rejected([&]{(void)flagValue(text);}); ++checks;
  }
  for(auto text:{"1","128","2048"}) { (void)oracleDecimal(text,1,2048); ++checks; }
  for(auto text:{"0","2049","-1","01","1x"}) {
    rejected([&]{(void)oracleDecimal(text,1,2048);}); ++checks;
  }
  Error signedZero; signedZero.add(std::array<uint16_t,1>{0x8000},std::array<uint16_t,1>{0});
  require(signedZero.mismatches==1&&signedZero.relative()==0, "word equality was replaced by numeric equality"); ++checks;
  require(argmax(std::array<uint16_t,3>{bf16(1),bf16(1),bf16(-1)})==0,
          "argmax tie rule failed"); ++checks;
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
            << ",\"gpu_commands\":0,\"model_loaded\":false}\n";
}
void writeReport(const char *path,const std::string &text) {
  std::ofstream file(path); require(bool(file), "cannot open proposal report");
  file << text << '\n'; require(bool(file), "cannot write proposal report");
}
} // namespace

int main(int argc,char **argv) {
  @autoreleasepool {
    std::vector<std::string> records;
    try {
      if(argc==2&&std::string_view(argv[1])=="--cpu-self-test") {cpuSelfTest();return 0;}
      if(argc!=4) throw std::invalid_argument(
          "usage: flash-mtp-float-candidate-oracle METALLIB PACKAGE REPORT_JSON | --cpu-self-test");
      require(flag("FLASH_MTP_FLOAT_CANDIDATE"),
              "root must explicitly set FLASH_MTP_FLOAT_CANDIDATE=1; oracle never mutates flags");
      const uint32_t primeRows=std::getenv("FLASH_MTP_FLOAT_CANDIDATE_PRIME_ROWS")?
          oracleDecimal(std::getenv("FLASH_MTP_FLOAT_CANDIDATE_PRIME_ROWS"),1,2048):128;
      const bool cachedVocabulary=flag("FLASH_MTP_FLOAT_CANDIDATE_CACHED_VOCAB");
      const auto fixture=hiddenFixture();
      metal::MetalBackend backend(argv[1]);
      const auto weights=base::FlashWeights::load(backend,argv[2]);
      base::FlashMTPForward original(backend,weights,kCapacity,kMaximumRows);
      candidate::FlashMTPForward proposed(backend,weights,kCapacity,kMaximumRows);
      const bool dense=flag("SPLASH_FLASH_DENSE_CACHE");
      const uint64_t originalPlan=base::FlashMTPForward::workspacePlannedBytes(kCapacity,kMaximumRows)+
          (dense?base::FlashMTPForward::denseCachePlannedBytes(weights):0);
      const uint64_t candidatePlan=candidate::FlashMTPForward::workspacePlannedBytes(kCapacity,kMaximumRows)+
          (dense?candidate::FlashMTPForward::denseCachePlannedBytes(weights):0)+
          candidate::FlashMTPForward::floatCachePlannedBytes(weights);
      require(original.workspaceBytes()<=originalPlan&&proposed.workspaceBytes()<=candidatePlan,
              "head workspace exceeds its independent plan");
      require(std::string_view(original.attentionRouteSemantics())==proposed.attentionRouteSemantics(),
              "baseline and candidate attention routes differ");
      std::unique_ptr<base::FlashDenseCache> vocabulary;
      if(cachedVocabulary) {
        const std::array<std::string,1> prefix{"language_model.lm_head"};
        vocabulary=std::make_unique<base::FlashDenseCache>(backend,weights,prefix);
      }
      const auto *vocabularyTensor=vocabulary?&vocabulary->tensor("language_model.lm_head"):nullptr;
      if(vocabularyTensor) {
        require(vocabularyTensor->buffer.storage()==metal::BufferStorage::Shared&&
                vocabularyTensor->buffer.contents()&&
                reinterpret_cast<uintptr_t>(vocabularyTensor->buffer.contents())%2==0,
                "shared vocabulary visibility or BF16 alignment invalid");
        (void)backend.view(vocabularyTensor->buffer,0,vocabularyTensor->logicalBytes);
      }
      base::FlashBatchMTPForward originalJoint(original,4,kMaximumRows,vocabularyTensor);
      candidate::FlashBatchMTPForward candidateJoint(proposed,4,kMaximumRows,vocabularyTensor);
      require(originalJoint.workspaceBytes()<=base::FlashBatchMTPForward::workspacePlannedBytes(
                  kCapacity,4,kMaximumRows,cachedVocabulary)&&
              candidateJoint.workspaceBytes()<=candidate::FlashBatchMTPForward::workspacePlannedBytes(
                  kCapacity,4,kMaximumRows,cachedVocabulary), "joint workspace exceeds independent plan");
      std::array<base::FlashMTPState,4> reference;
      std::array<candidate::FlashMTPState,4> actual;
      const auto input=backend.allocateBuffer(uint64_t{4}*kMaximumRows*kWidth*2,
          metal::BufferStorage::Shared,"private-mtp-proposal-fixed-bf16-input");
      std::array<uint64_t,4> prefixes{};
      for(uint32_t lane=0;lane<4;++lane) {
        reference[lane]=original.createState(); actual[lane]=proposed.createState();
        prefixes[lane]=primeRows-std::min(lane,primeRows-1);
        for(uint32_t begin=0;begin<prefixes[lane];) {
          const uint32_t count=uint32_t(std::min<uint64_t>(kMaximumRows,prefixes[lane]-begin));
          const std::array<uint64_t,1> start{begin};
          const auto tokens=fill(input,1,count,start,fixture,lane);
          const auto view=backend.view(input,0,uint64_t{count}*kWidth*2);
          const auto a=original.forward(reference[lane],view,tokens,base::FlashMTPLogits::None);
          const auto originalTiming=a.timing;validTiming(originalTiming);
          const auto first=copy(a.hiddenBF16,uint64_t{count}*kWidth);
          const auto b=proposed.forward(actual[lane],view,tokens,candidate::FlashMTPLogits::None);
          const auto candidateTiming=b.timing;validTiming(candidateTiming);
          const auto second=copy(b.hiddenBF16,uint64_t{count}*kWidth);
          Error error;error.add(second,first);require(error.acceptable(),"priming exceeds proposal-only bound");
          begin+=count;
        }
      }
      const auto rewind=[&](uint32_t lanes) {
        for(uint32_t lane=0;lane<lanes;++lane) {
          original.truncate(reference[lane],prefixes[lane]); proposed.truncate(actual[lane],prefixes[lane]);
          require(reference[lane].logicalLength()==prefixes[lane]&&
                  actual[lane].logicalLength()==prefixes[lane]&&!reference[lane].poisoned()&&!actual[lane].poisoned(),
                  "rollback changed ownership/health/independent offset");
        }
      };
      const auto run=[&](bool useCandidate,uint32_t lanes,uint32_t rows,Mode mode,
                         const std::vector<uint32_t> &tokens) -> Snapshot {
        Snapshot snapshot;
        const auto view=backend.view(input,0,uint64_t{lanes}*rows*kWidth*2);
        const auto before=std::chrono::steady_clock::now();
        if(lanes==1) {
          if(useCandidate) {
            const auto result=proposed.forward(actual[0],view,tokens,candidateMode(mode));
            snapshot.forwardWall=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
            require(result.hiddenRows==rows&&result.logitRows==logitRows(lanes,rows,mode),"candidate single result metadata");
            snapshot.timing=result.timing;snapshot.logitsRows=result.logitRows;
            snapshot.hidden=copy(result.hiddenBF16,uint64_t{rows}*kWidth);
            snapshot.logits=copy(result.logitsBF16,uint64_t{result.logitRows}*kVocabulary);
            snapshot.lengths={result.logicalLength};
          } else {
            const auto result=original.forward(reference[0],view,tokens,baseMode(mode));
            snapshot.forwardWall=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
            require(result.hiddenRows==rows&&result.logitRows==logitRows(lanes,rows,mode),"baseline single result metadata");
            snapshot.timing=result.timing;snapshot.logitsRows=result.logitRows;
            snapshot.hidden=copy(result.hiddenBF16,uint64_t{rows}*kWidth);
            snapshot.logits=copy(result.logitsBF16,uint64_t{result.logitRows}*kVocabulary);
            snapshot.lengths={result.logicalLength};
          }
        } else {
          std::vector<uint32_t> counts(lanes,rows),offsets{0};
          for(uint32_t lane=0;lane<lanes;++lane) offsets.push_back(offsets.back()+rows);
          if(useCandidate) {
            std::vector<candidate::FlashMTPState *> states;
            for(uint32_t lane=0;lane<lanes;++lane) states.push_back(&actual[lane]);
            const auto result=candidateJoint.forward(states,view,tokens,counts,candidateMode(mode));
            snapshot.forwardWall=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
            require(result.lanes==lanes&&result.laneOffsets==offsets&&
                    result.logitRows==logitRows(lanes,rows,mode),"candidate joint result metadata");
            snapshot.timing=result.timing;snapshot.logitsRows=result.logitRows;
            snapshot.hidden=copy(result.hiddenBF16,uint64_t{lanes}*rows*kWidth);
            snapshot.logits=copy(result.logitsBF16,uint64_t{result.logitRows}*kVocabulary);
            snapshot.lengths=result.logicalLengths;
          } else {
            std::vector<base::FlashMTPState *> states;
            for(uint32_t lane=0;lane<lanes;++lane) states.push_back(&reference[lane]);
            const auto result=originalJoint.forward(states,view,tokens,counts,baseMode(mode));
            snapshot.forwardWall=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
            require(result.lanes==lanes&&result.laneOffsets==offsets&&
                    result.logitRows==logitRows(lanes,rows,mode),"baseline joint result metadata");
            snapshot.timing=result.timing;snapshot.logitsRows=result.logitRows;
            snapshot.hidden=copy(result.hiddenBF16,uint64_t{lanes}*rows*kWidth);
            snapshot.logits=copy(result.logitsBF16,uint64_t{result.logitRows}*kVocabulary);
            snapshot.lengths=result.logicalLengths;
          }
        }
        require(snapshot.lengths.size()==lanes,"result logical length extent");
        validTiming(snapshot.timing);
        for(uint32_t lane=0;lane<lanes;++lane)
          require(snapshot.lengths[lane]==(useCandidate?actual[lane].logicalLength():reference[lane].logicalLength())&&
                  !(useCandidate?actual[lane].poisoned():reference[lane].poisoned()),"result/state offset or health");
        return snapshot;
      };
      uint64_t guardChecks=0;bool allPass=true;
      for(uint32_t lanes:{1u,4u}) for(uint32_t rows:{1u,4u,8u})
        for(Mode mode:{Mode::None,Mode::Last,Mode::All}) {
          if(mode==Mode::All&&lanes*rows>16) continue;
          rewind(lanes);
          const auto tokens=fill(input,lanes,rows,prefixes,fixture);
          const auto immutableInput=copy(backend.view(input,0,uint64_t{lanes}*rows*kWidth*2),
                                         uint64_t{lanes}*rows*kWidth);
          const auto warmBase=run(false,lanes,rows,mode,tokens);
          const auto warmCandidate=run(true,lanes,rows,mode,tokens);
          Times baselineTimes,candidateTimes;Error hiddenError,logitsError;Argmax matches;
          for(uint32_t repetition=0;repetition<kSamples;++repetition) {
            rewind(lanes);
            Snapshot a,b;
            if(repetition%2) {b=run(true,lanes,rows,mode,tokens);a=run(false,lanes,rows,mode,tokens);}
            else {a=run(false,lanes,rows,mode,tokens);b=run(true,lanes,rows,mode,tokens);}
            require(a.hidden==warmBase.hidden&&a.logits==warmBase.logits&&
                    b.hidden==warmCandidate.hidden&&b.logits==warmCandidate.logits,"same-route replay is not BF16-word deterministic");
            require(a.lengths==b.lengths,"baseline/candidate independent lengths differ");
            for(uint32_t lane=0;lane<lanes;++lane)
              require(a.lengths[lane]==prefixes[lane]+rows,"append advanced wrong logical length");
            require(copy(backend.view(input,0,uint64_t{lanes}*rows*kWidth*2),
                         uint64_t{lanes}*rows*kWidth)==immutableInput,"proposal mutated fixed BF16 input");
            sample(baselineTimes,a);sample(candidateTimes,b);
            hiddenError.add(b.hidden,a.hidden);logitsError.add(b.logits,a.logits);matches.add(b,a);
          }
          // Continue from the completed window, then roll back inside it and
          // overwrite stale QSA rows with a different real-token continuation.
          std::array<uint64_t,4> starts=prefixes;
          for(uint32_t lane=0;lane<lanes;++lane) starts[lane]+=rows;
          auto continuedTokens=fill(input,lanes,1,starts,fixture,11);
          const auto continuedBase=run(false,lanes,1,Mode::None,continuedTokens);
          const auto continuedCandidate=run(true,lanes,1,Mode::None,continuedTokens);
          Error continuedError;continuedError.add(continuedCandidate.hidden,continuedBase.hidden);
          for(uint32_t lane=0;lane<lanes;++lane) {
            require(continuedBase.lengths[lane]==starts[lane]+1&&
                    continuedCandidate.lengths[lane]==starts[lane]+1,"continued append length");++guardChecks;
            starts[lane]=prefixes[lane]+rows/2;
            original.truncate(reference[lane],starts[lane]);proposed.truncate(actual[lane],starts[lane]);
          }
          continuedTokens=fill(input,lanes,1,starts,fixture,19);
          const auto rolledBase=run(false,lanes,1,Mode::None,continuedTokens);
          const auto rolledCandidate=run(true,lanes,1,Mode::None,continuedTokens);
          Error rolledError;rolledError.add(rolledCandidate.hidden,rolledBase.hidden);
          for(uint32_t lane=0;lane<lanes;++lane) {
            require(rolledBase.lengths[lane]==starts[lane]+1&&
                    rolledCandidate.lengths[lane]==starts[lane]+1,"rollback overwrite append length");++guardChecks;
          }
          const bool pass=hiddenError.acceptable()&&logitsError.acceptable()&&
              continuedError.acceptable()&&rolledError.acceptable();
          allPass=allPass&&pass;
          std::ostringstream record;record<<std::setprecision(12);
          record<<"{\"lanes\":"<<lanes<<",\"rows_per_lane\":"<<rows<<",\"mode\":"
              <<splash::json::quote(modeName(mode))<<",\"proposal_bound_pass\":"<<(pass?"true":"false")
              <<",\"baseline\":";baselineTimes.write(record);record<<",\"candidate\":";candidateTimes.write(record);
          record<<",\"gpu_speedup\":"<<median(baselineTimes.gpu)/median(candidateTimes.gpu)
              <<",\"forward_wall_speedup\":"<<median(baselineTimes.forwardWall)/median(candidateTimes.forwardWall)
              <<",\"hidden_error\":";hiddenError.write(record);record<<",\"vocabulary_error\":";logitsError.write(record);
          record<<",\"continued_hidden_error\":";continuedError.write(record);
          record<<",\"rollback_overwrite_hidden_error\":";rolledError.write(record);
          record<<",\"paired_argmax_compared\":"<<matches.compared<<",\"paired_argmax_matched\":"<<matches.matched
              <<",\"input_unchanged\":true,\"same_route_word_deterministic\":true}";records.push_back(record.str());
          std::cerr<<"private MTP proposal C"<<lanes<<" R"<<rows<<" "<<modeName(mode)<<" bound="<<pass<<'\n';
        }
      rewind(4);
      const auto guardTokens=fill(input,4,8,prefixes,fixture);
      const auto referenceLengths=prefixes;
      rejected([&]{base::FlashMTPState empty;(void)original.forward(empty,input,std::array<uint32_t,1>{7});});++guardChecks;
      rejected([&]{candidate::FlashMTPState empty;(void)proposed.forward(empty,input,std::array<uint32_t,1>{7});});++guardChecks;
      rejected([&]{original.truncate(reference[0],prefixes[0]+1);});++guardChecks;
      rejected([&]{proposed.truncate(actual[0],prefixes[0]+1);});++guardChecks;
      std::array<base::FlashMTPState *,4> baseStates{&reference[0],&reference[1],&reference[2],&reference[3]};
      std::array<candidate::FlashMTPState *,4> candidateStates{&actual[0],&actual[1],&actual[2],&actual[3]};
      rejected([&]{(void)originalJoint.forward(baseStates,input,guardTokens,std::array<uint32_t,4>{8,8,8,8},base::FlashMTPLogits::All);});++guardChecks;
      rejected([&]{(void)candidateJoint.forward(candidateStates,input,guardTokens,std::array<uint32_t,4>{8,8,8,8},candidate::FlashMTPLogits::All);});++guardChecks;
      rejected([&]{(void)originalJoint.forward(std::array<base::FlashMTPState *,2>{&reference[0],&reference[0]},
          input,std::array<uint32_t,2>{7,8},std::array<uint32_t,2>{1,1});});++guardChecks;
      rejected([&]{(void)candidateJoint.forward(std::array<candidate::FlashMTPState *,2>{&actual[0],&actual[0]},
          input,std::array<uint32_t,2>{7,8},std::array<uint32_t,2>{1,1});});++guardChecks;
      for(uint32_t lane=0;lane<4;++lane)
        require(reference[lane].logicalLength()==referenceLengths[lane]&&
                actual[lane].logicalLength()==referenceLengths[lane]&&!reference[lane].poisoned()&&!actual[lane].poisoned(),
                "rejected call mutated healthy independent state");
      const char *kernel=std::getenv("FLASH_MTP_FLOAT_KERNEL");
      std::ostringstream report;report<<std::setprecision(12)<<"{\"pass\":"<<(allPass?"true":"false")
          <<",\"proposal_only\":true,\"end_to_end_correctness_qualified\":false,\"exact_equivalence_required\":false"
          <<",\"relative_l2_proposal_limit\":0.01,\"samples_per_route\":6,\"timing_order\":\"AB-BA-alternating\""
          <<",\"timing_scope\":\"forward call only; output copies and truncation excluded; GPU command timing separately reported\""
          <<",\"candidate_kernel_environment\":"<<splash::json::quote(kernel?kernel:"f32")
          <<",\"candidate_projection_semantics\":"<<splash::json::quote(proposed.floatProjectionSemantics())
          <<",\"attention_route\":"<<splash::json::quote(original.attentionRouteSemantics())
          <<",\"hidden_fixture\":"<<splash::json::quote(fixture.empty()?"deterministic-synthetic-bf16":"captured-bf16-file-real-rows-cycled")
          <<",\"initial_prefix_rows\":[";
      for(size_t lane=0;lane<prefixes.size();++lane) {if(lane)report<<',';report<<prefixes[lane];}
      report<<"],\"cached_joint_vocabulary\":"<<(cachedVocabulary?"true":"false")
          <<",\"baseline_workspace_bytes\":"<<original.workspaceBytes()<<",\"baseline_planned_bytes\":"<<originalPlan
          <<",\"candidate_workspace_bytes\":"<<proposed.workspaceBytes()<<",\"candidate_planned_bytes\":"<<candidatePlan
          <<",\"candidate_float_cache_planned_bytes\":"<<candidate::FlashMTPForward::floatCachePlannedBytes(weights)
          <<",\"guard_checks\":"<<guardChecks<<",\"source_identity\":"<<splash::json::quote(weights.sourceIdentity())
          <<",\"original_model_modified\":false,\"production_routes_modified\":false,\"cases\":[";
      for(size_t index=0;index<records.size();++index) {if(index)report<<',';report<<records[index];}
      report<<"]}";writeReport(argv[3],report.str());return allPass?0:2;
    } catch(const std::exception &error) {
      std::cerr<<"private MTP float proposal oracle: "<<error.what()<<'\n';
      if(argc==4) {
        std::ostringstream report;report<<"{\"pass\":false,\"proposal_only\":true,\"end_to_end_correctness_qualified\":false,"
            <<"\"error\":"<<splash::json::quote(error.what())<<",\"completed_cases\":[";
        for(size_t index=0;index<records.size();++index) {if(index)report<<',';report<<records[index];}
        report<<"]}";try{writeReport(argv[3],report.str());}catch(...){}
      }
      return 1;
    }
  }
}
