// Isolated native backend screen. --cpu-self-test submits no GPU commands.
// Operand dimensions match Flash projections; data is deterministic BF16.
#include "metal/MetalBackend.hpp"
#include "engine/Json.hpp"
#include "FlashFloatBoundaryAudit.hpp"
#include "ultra_splitk_params.h"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::metal;
using namespace splash::flash::benchmark;
constexpr uint64_t guardWords = 32;
constexpr uint16_t bf16Sentinel = 0x7fc1;
constexpr uint32_t f32Sentinel = 0x7fc12345;
constexpr uint32_t sticky = 0x40000000;
void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}
uint64_t randomWord(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
uint32_t tileExtent(uint32_t value, uint32_t tile) { return (value + tile - 1) / tile * tile; }
uint32_t positiveEnv(const char *name, uint32_t fallback, uint32_t maximum) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  size_t used = 0;
  require(*raw && *raw != '-', std::string("invalid ") + name);
  const auto value = std::stoul(raw, &used);
  require(used == std::strlen(raw) && value && value <= maximum, std::string("invalid ") + name);
  return uint32_t(value);
}
template<class Word> struct Guarded {
  MetalBuffer base, view;
  uint64_t words, allocatedBytes;
  Word sentinel;
  Guarded(MetalBackend &backend, uint64_t count, Word marker, const char *label)
      : words(count), sentinel(marker) {
    const auto before = backend.memoryStats().allocatedBytes;
    base = backend.allocateBuffer((words + 2 * guardWords) * sizeof(Word), BufferStorage::Shared, label);
    view = backend.view(base, guardWords * sizeof(Word), words * sizeof(Word));
    allocatedBytes = backend.memoryStats().allocatedBytes - before;
    clear();
  }
  void clear() { std::fill_n(static_cast<Word *>(base.contents()), words + 2 * guardWords, sentinel); }
  void check(bool finite = false) const {
    const auto *raw = static_cast<const Word *>(base.contents());
    for (uint64_t i = 0; i < guardWords; ++i)
      require(raw[i] == sentinel && raw[guardWords + words + i] == sentinel, "guard overwritten");
    if (finite)
      for (uint64_t i = 0; i < words; ++i) {
        if constexpr (sizeof(Word) == 2)
          require(std::isfinite(number(raw[guardWords + i])), "BF16 output unwritten or nonfinite");
        else
          require(std::isfinite(std::bit_cast<float>(raw[guardWords + i])), "FP32 scratch unwritten or nonfinite");
      }
  }
  Word *values() { return static_cast<Word *>(view.contents()); }
  const Word *values() const { return static_cast<const Word *>(view.contents()); }
};
struct Shape { uint32_t rows, k, n, m, tileN, simd; const char *label; };
struct SourceF32 {
  uint32_t n, k, bits, group;
  std::string prefix, identity, coefficientSHA;
  std::vector<uint32_t> coefficients;
  uint64_t lowBF16Bits = 0;
};
SourceF32 loadSourceF32(const char *path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "cannot open source F32 artifact");
  auto read = [&](void *data, size_t bytes) {
    input.read(static_cast<char *>(data), std::streamsize(bytes));
    require(size_t(input.gcount()) == bytes, "short source F32 artifact");
  };
  std::array<char,8> magic{}; std::array<uint32_t,8> fields{}; std::array<char,64> identity{};
  read(magic.data(),magic.size()); read(fields.data(),fields.size()*4); read(identity.data(),identity.size());
  require(std::string(magic.data(),8) == "ULTRAF32" && fields[0] && fields[0] <= 32768 &&
      fields[1] && fields[1] <= 32768 && !fields[7] && fields[6] && fields[6] <= 1024,
      "invalid F32 artifact header");
  const auto n=fields[0], k=fields[1], bits=fields[2], group=fields[3], ws=fields[4], ps=fields[5];
  require((bits==4||bits==5||bits==6||bits==8) && (group==32||group==64||group==128) &&
      k%group==0 && ws==uint64_t(k)*bits/8 && ps==uint64_t(k)/group*2, "invalid source affine strides");
  SourceF32 source{n,k,bits,group,std::string(fields[6],'\0'),std::string(identity.data(),64),{}, {},0};
  read(source.prefix.data(),source.prefix.size());
  std::vector<uint8_t> packed(uint64_t(n)*ws);
  std::vector<uint16_t> scales(uint64_t(n)*ps/2), biases(scales.size());
  source.coefficients.resize(uint64_t(n)*k);
  read(packed.data(),packed.size()); read(scales.data(),scales.size()*2);
  read(biases.data(),biases.size()*2); read(source.coefficients.data(),source.coefficients.size()*4);
  require(input.peek()==std::char_traits<char>::eof(), "trailing source F32 bytes");
  for(uint32_t row=0;row<n;++row) for(uint32_t column=0;column<k;++column) {
    const uint64_t bit=uint64_t(column)*bits, byte=uint64_t(row)*ws+bit/8;
    uint32_t word=packed[byte]; const auto shift=uint32_t(bit%8);
    if(shift+bits>8) word|=uint32_t(packed[byte+1])<<8;
    const auto code=(word>>shift)&((1u<<bits)-1);
    const auto parameter=uint64_t(row)*(ps/2)+column/group;
    // Independent scalar staging, compiled with -ffp-contract=off.
    const float product=float(code)*number(scales[parameter]);
    const float coefficient=product+number(biases[parameter]);
    const auto actual=source.coefficients[uint64_t(row)*k+column];
    require(std::isfinite(coefficient) && actual==std::bit_cast<uint32_t>(coefficient),
        "F32 coefficient source mismatch");
    source.lowBF16Bits += (actual&0xffffu)!=0;
  }
  std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(source.coefficients.data(),CC_LONG(source.coefficients.size()*4),digest.data());
  std::ostringstream hex;
  for(auto byte:digest) hex<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(byte);
  source.coefficientSHA=hex.str();
  return source;
}
struct Coefficients {
  std::optional<Guarded<uint16_t>> bf16Storage;
  std::optional<Guarded<uint32_t>> f32Storage;
  MetalBuffer view;
  uint64_t words;
  bool f32;
  Coefficients(MetalBackend &backend,uint64_t count,bool floatMode) : words(count),f32(floatMode) {
    if(f32) { f32Storage.emplace(backend,count,f32Sentinel,"source F32 immutable coefficients"); view=f32Storage->view; }
    else { bf16Storage.emplace(backend,count,bf16Sentinel,"BF16 immutable coefficients"); view=bf16Storage->view; }
  }
  void *data() { return view.contents(); }
  size_t bytes() const { return size_t(words)*(f32?4:2); }
  void set(uint64_t i,float value) {
    if(f32) static_cast<uint32_t *>(data())[i]=std::bit_cast<uint32_t>(value);
    else static_cast<uint16_t *>(data())[i]=bf16(value);
  }
  float numberAt(uint64_t i) const {
    return f32 ? std::bit_cast<float>(static_cast<const uint32_t *>(view.contents())[i]) :
        number(static_cast<const uint16_t *>(view.contents())[i]);
  }
  void check() const { if(f32)f32Storage->check();else bf16Storage->check(); }
};
struct Dot { uint32_t row, column; double sum, absoluteProducts; uint16_t rounded; };
std::vector<uint32_t> samples(uint32_t extent, uint32_t count) {
  std::vector<uint32_t> result;
  count = std::min(extent, count);
  for (uint32_t i = 0; i < count; ++i)
    result.push_back(count == 1 ? 0 : uint64_t(i) * (extent - 1) / (count - 1));
  return result;
}
Dot referenceCell(const Shape &shape, const uint16_t *input, const Coefficients &weight,
                  uint32_t row,uint32_t column) {
  double sum=0,absoluteProducts=0;
  for(uint32_t k=0;k<shape.k;++k) {
    const double product=double(number(input[uint64_t(row)*shape.k+k]))*
        weight.numberAt(uint64_t(column)*shape.k+k);
    sum+=product; absoluteProducts+=std::abs(product);
  }
  return {row,column,sum,absoluteProducts,bf16Double(sum)};
}
std::vector<Dot> reference(const Shape &shape, const uint16_t *input, const Coefficients &weight) {
  std::vector<Dot> result;
  const auto rows = samples(shape.rows, shape.rows <= 16 ? shape.rows : 9);
  const auto columns = samples(shape.n, shape.n <= 130 ? shape.n : 19);
  for (uint32_t row : rows) for (uint32_t column : columns) {
    result.push_back(referenceCell(shape,input,weight,row,column));
  }
  return result;
}
struct Error {
  uint64_t cells = 0, exactMismatch = 0, intervalViolation = 0, strictViolation = 0,
      cancellationDominated = 0, wrongSign = 0;
  uint32_t maxULP = 0;
  double maxAbsolute = 0, squaredError = 0, squaredReference = 0;
  void add(uint16_t actual, const Dot &dot, uint32_t terms) {
    ++cells;
    require(std::isfinite(number(actual)), "nonfinite numerical output");
    exactMismatch += actual != dot.rounded;
    maxULP = std::max(maxULP, bf16ULP(actual, dot.rounded));
    const double delta = double(number(actual)) - number(dot.rounded);
    maxAbsolute = std::max(maxAbsolute, std::abs(delta));
    squaredError += delta * delta;
    squaredReference += double(number(dot.rounded)) * number(dot.rounded);
    const double bound = f32DotBound(terms, dot.absoluteProducts);
    strictViolation += !cellRelation(actual,dot.rounded,dot.sum,bound).pass;
    cancellationDominated += std::abs(dot.sum)<=bound;
    wrongSign += dot.sum!=0 && number(actual)!=0 && std::signbit(dot.sum)!=std::signbit(number(actual));
    const auto low = orderedBF16(bf16Double(dot.sum - bound));
    const auto high = orderedBF16(bf16Double(dot.sum + bound));
    const auto order = orderedBF16(actual);
    intervalViolation += order < low || order > high;
  }
  double relativeL2() const { return std::sqrt(squaredError / std::max(1e-30, squaredReference)); }
  void write(std::ostream &out) const {
    out << "{\"sampled_cells\":" << cells << ",\"exact_bf16_mismatches\":" << exactMismatch
        << ",\"f32_bound_interval_violations\":" << intervalViolation << ",\"max_bf16_ulp\":"
        << maxULP << ",\"max_abs\":" << maxAbsolute << ",\"relative_l2\":" << relativeL2()
        << ",\"strict_one_ulp_cell_violations\":" << strictViolation
        << ",\"cancellation_dominated_cells\":" << cancellationDominated
        << ",\"wrong_sign_cells\":" << wrongSign << '}';
  }
};
double median(std::vector<double> values) {
  require(!values.empty(), "empty timing samples");
  std::sort(values.begin(), values.end());
  const auto count = values.size();
  return count % 2 ? values[count / 2] : (values[count / 2 - 1] + values[count / 2]) / 2;
}
struct Route {
  UltraSplitKParams params;
  Guarded<uint16_t> output;
  Guarded<uint32_t> partial;
  std::vector<ComputeDispatch> dispatches;
  std::vector<double> gpu, wall;
  Error numerical;
  Error changedCellsNumerical, changedCellsControlNumerical;
  uint64_t changedExactMidpoints=0;
  double changedMaxMidpointDistance=0;
  uint64_t controlMismatch = 0;
  Route(MetalBackend &backend, const Shape &s, uint32_t partition)
      : params{s.rows, s.k, s.n, tileExtent(s.rows, s.m), tileExtent(s.n, s.tileN), partition, s.m, s.tileN},
        output(backend, uint64_t(s.rows) * s.n, bf16Sentinel, "split-K BF16 output"),
        partial(backend, partition ? uint64_t(params.padded_rows) * params.padded_outputs *
            ((s.k + partition - 1) / partition) : 1, f32Sentinel, "split-K FP32 guarded scratch") {}
  void prepare(const Shape &shape, const MetalBuffer &input, const MetalBuffer &weight,
               const MetalBuffer &diag, uint32_t batch, const MetalBuffer &padding, bool f32) {
    const uint32_t partitions = params.partition ? (params.input_size + params.partition - 1) / params.partition : 1;
    const auto pipeline = std::string(f32?"ultra_splitk_f32_m":"ultra_splitk_m") + std::to_string(shape.m) + "_n" +
        std::to_string(shape.tileN) + "_s" + std::to_string(shape.simd);
    for (uint32_t i = 0; i < batch; ++i) {
      if(f32) {
        ComputeDispatch pad;
        pad.pipelineName="ultra_splitk_pad";
        pad.buffers={{0,input},{1,padding},{2,diag}};
        pad.bytes={{3,&params,sizeof(params)}};
        pad.threadgroups={(uint64_t(params.padded_rows)*params.input_size+255)/256,1,1};
        pad.threadsPerThreadgroup={256,1,1};
        dispatches.push_back(std::move(pad));
      }
      ComputeDispatch multiply;
      multiply.pipelineName = pipeline;
      multiply.buffers = {{0, f32?padding:input}, {1, weight}, {2, output.view}, {3, partial.view}, {4, diag}};
      multiply.bytes = {{5, &params, sizeof(params)}};
      multiply.threadgroups = {params.padded_outputs / shape.tileN, params.padded_rows / shape.m, partitions};
      multiply.threadsPerThreadgroup = {shape.simd * 32, 1, 1};
      dispatches.push_back(std::move(multiply));
      if (params.partition) {
        ComputeDispatch reduce;
        reduce.pipelineName = "ultra_splitk_reduce";
        reduce.buffers = {{0, partial.view}, {1, output.view}, {2, diag}};
        reduce.bytes = {{3, &params, sizeof(params)}};
        reduce.threadgroups = {(uint64_t(params.rows) * params.output_size + 255) / 256, 1, 1};
        reduce.threadsPerThreadgroup = {256, 1, 1};
        dispatches.push_back(std::move(reduce));
      }
    }
  }
};
void cpuSelfTest() {
  uint64_t checks = 0;
  for (uint32_t bits = 0; bits < 65536; ++bits)
    if (std::isfinite(number(uint16_t(bits)))) {
      require(bf16(number(uint16_t(bits))) == bits, "BF16 roundtrip failed"); ++checks;
    }
  require(bf16Double(1.00390625) == 0x3f80 && bf16Double(1.01171875) == 0x3f82,
          "independent ties failed"); checks += 2;
  Error exact; exact.add(bf16(3), {0, 0, 3, 3, bf16(3)}, 32);
  require(exact.exactMismatch == 0 && exact.intervalViolation == 0, "exact interval failed"); ++checks;
  Error bad; bad.add(bf16(4), {0, 0, 3, 3, bf16(3)}, 32);
  require(bad.intervalViolation == 1, "interval failed to reject wrong result"); ++checks;
  for(float residual : {1.0f/32768.0f,-1.0f/32768.0f}) {
    Error cancellation;
    cancellation.add(0,{0,0,double(residual),264241152.0,bf16(residual)},6176);
    require(cancellation.intervalViolation==0 && cancellation.strictViolation==1 &&
        cancellation.cancellationDominated==1,"cancellation classifier hid a lost residual"); ++checks;
  }
  require(tileExtent(3, 8) == 8 && tileExtent(130, 128) == 256, "tail rounding failed"); ++checks;
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks << ",\"gpu_commands\":0}\n";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if(argc==3 && std::string(argv[1])=="--source-self-test") {
        const auto source=loadSourceF32(argv[2]);
        std::cout<<"{\"pass\":true,\"coefficient_checks\":"<<source.coefficients.size()
            <<",\"coefficients_with_nonzero_low_bf16_bits\":"<<source.lowBF16Bits
            <<",\"coefficient_sha256\":"<<splash::json::quote(source.coefficientSHA)
            <<",\"gpu_commands\":0}\n"; return 0;
      }
      require(argc == 3, "usage: ultra-splitk-oracle METALLIB REPORT_JSON | --cpu-self-test");
      const uint32_t repeats = positiveEnv("ULTRA_SPLITK_REPEATS", 9, 30);
      const uint32_t batch = positiveEnv("ULTRA_SPLITK_BATCH", 4, 32);
      const char *filter = std::getenv("ULTRA_SPLITK_SHAPE");
      const char *sourcePath=std::getenv("ULTRA_SPLITK_F32_SOURCE");
      std::optional<SourceF32> source;
      if(sourcePath)source=loadSourceF32(sourcePath);
      std::vector<Shape> shapes{
          {512,6144,2560,64,128,8,"prefill512_m64"},
          {2048,6144,2560,64,128,8,"prefill2048_m64"},
          {512,6144,2560,16,128,4,"prefill512_m16"},
          {2048,6144,2560,16,128,4,"prefill2048_m16"},
          {4,6144,2560,8,128,4,"verifier4"},
          {16,6144,2560,16,128,4,"verifier16"},
          {3,6176,130,8,128,4,"guard_row_n_k_tail"}};
      if(source) {
        require(source->k==6144 && source->n==2560,"source F32 screen expects N2560 K6144");
        shapes={
          {4,6144,2560,16,64,4,"f32_active_qsa_verify4_m16n64"},
          {16,6144,2560,16,64,4,"f32_active_qsa_verify16_m16n64"},
          {4,6144,2560,8,64,4,"f32_explore_verify4_m8n64"},
          {4,6144,2560,8,128,4,"f32_explore_verify4_m8n128"},
          {3,6176,130,8,64,4,"f32_guard_row_n_k_tail"},
          {4,6176,64,8,64,4,"f32_partition_cancellation"}};
      }
      MetalBackend backend(argv[1]);
      auto diag = backend.allocateBuffer(sizeof(uint32_t), BufferStorage::Shared, "split-K sticky diagnostic");
      auto *status = static_cast<uint32_t *>(diag.contents());
      std::ostringstream report;
      report << std::setprecision(12) << "{\"pass\":true,\"pass_scope\":\"guards, source coefficients, absolute arithmetic interval; strict BF16 gate reported separately\",\"device\":"
          << splash::json::quote(backend.capabilities().deviceName)
          << ",\"gpu_core_count\":" << backend.capabilities().gpuCoreCount
          << ",\"operands\":"<<splash::json::quote(source?
              "BF16 input; source-exact original affine F32 coefficients; model not loaded":
              "synthetic deterministic BF16 at Flash projection shapes; no model loaded")
          << ",\"timing_scope\":\"native command includes FP32 scratch writes and reduction; source F32 mode also includes GPU zero-padding; warm resident operands\""
          << ",\"numerical_scope\":\"sampled and every changed cell independent double dots; conservative FP32 interval plus separate strict one-ULP BF16 cell gate; no generation parity claim\""
          << ",\"repeats\":" << repeats << ",\"batch\":" << batch;
      if(source)report<<",\"source_prefix\":"<<splash::json::quote(source->prefix)
          <<",\"source_identity_sha256\":"<<splash::json::quote(source->identity)
          <<",\"coefficient_sha256\":"<<splash::json::quote(source->coefficientSHA)
          <<",\"coefficient_bf16_low_bit_nonzero_count\":"<<source->lowBF16Bits
          <<",\"coefficient_bit_exact_source_checks\":"<<source->coefficients.size();
      report<<",\"records\":[";
      bool first = true;
      uint64_t shapeCount = 0;
      for (const Shape &shape : shapes) {
        if (filter && std::string(shape.label).find(filter) == std::string::npos) continue;
        ++shapeCount;
        const auto paddedRows = tileExtent(shape.rows, shape.m), paddedN = tileExtent(shape.n, shape.tileN);
        Guarded<uint16_t> input(backend, uint64_t(paddedRows) * shape.k, bf16Sentinel, "split-K padded input");
        Coefficients weight(backend,uint64_t(paddedN)*shape.k,bool(source));
        Guarded<uint16_t> padding(backend,uint64_t(paddedRows)*shape.k,bf16Sentinel,"split-K GPU padding workspace");
        std::fill_n(input.values(), input.words, uint16_t(0));
        std::memset(weight.data(),0,weight.bytes());
        for (uint64_t i = 0; i < uint64_t(shape.rows) * shape.k; ++i)
          input.values()[i] = bf16(float(int32_t(randomWord(i + 0xc0ffee) % 2047) - 1023) / 1024.0f);
        for (uint64_t i = 0; i < uint64_t(shape.n) * shape.k; ++i)
          weight.set(i,float(int32_t(randomWord(i + 0xbadcafe) % 2047) - 1023) / 8192.0f);
        if(source)for(uint32_t n=0;n<shape.n;++n)for(uint32_t k=0;k<shape.k;++k)
          weight.set(uint64_t(n)*shape.k+k,k<source->k?
              std::bit_cast<float>(source->coefficients[uint64_t(n)*source->k+k]):0.0f);
        const bool cancellation=std::string(shape.label).find("partition_cancellation")!=std::string::npos;
        if(cancellation) {
          // Cancellation crosses the K partition boundary; small positive and
          // negative residuals test the gate even when a global L2 looks fine.
          std::fill_n(input.values(),uint64_t(shape.rows)*shape.k,bf16(1.0f));
          for(uint32_t n=0;n<shape.n;++n)for(uint32_t k=0;k<shape.k;++k) {
            float value=k<2048?65536.0f:(k<4096?-65536.0f:0.0f);
            const float residual=(n%2?-1.0f:1.0f)/1048576.0f;
            if(n%3==0) { if(k>=6144)value=residual; }
            else {
              // Equal large positive/negative counts, but put the residual
              // into the first partition so FP32 partial rounding can lose it.
              if(k<32)value=residual;
              if(k>=4064 && k<4096)value=0.0f;
            }
            weight.set(uint64_t(n)*shape.k+k,value);
          }
        }
        const std::vector<uint16_t> inputCopy(input.values(), input.values() + input.words);
        std::vector<std::byte> weightCopy(weight.bytes());
        std::memcpy(weightCopy.data(),weight.data(),weightCopy.size());
        const auto golden = reference(shape,input.values(),weight);
        std::array<Route,3> routes{Route(backend,shape,0),Route(backend,shape,2048),Route(backend,shape,4096)};
        for (auto &route : routes) route.prepare(shape,input.view,weight.view,diag,batch,padding.view,bool(source));
        for (uint32_t warm = 0; warm < 2; ++warm) for (auto &route : routes) {
          *status = sticky;
          (void)backend.submitCommand(route.dispatches);
          require(*status == sticky, "warmup changed sticky diagnostic");
        }
        // Rotating order prevents always giving one route the first sample.
        for (uint32_t repeat = 0; repeat < repeats; ++repeat)
          for (uint32_t offset = 0; offset < routes.size(); ++offset) {
            auto &route = routes[(repeat + offset) % routes.size()];
            *status = sticky;
            const auto timing = backend.submitCommand(route.dispatches);
            require(*status == sticky && timing.gpuSeconds > 0 && timing.wallSeconds > 0,
                    "timing or diagnostic invalid");
            route.gpu.push_back(timing.gpuSeconds / batch);
            route.wall.push_back(timing.wallSeconds / batch);
          }
        input.check(); weight.check();
        if(source) {
          padding.check(true);
          const auto realWords=uint64_t(shape.rows)*shape.k;
          require(std::memcmp(padding.values(),input.values(),realWords*2)==0,
              "GPU padding changed finite BF16 input words");
          for(uint64_t word=realWords;word<padding.words;++word)
            require(padding.values()[word]==0,"GPU padding tail is not positive zero");
        }
        require(std::memcmp(inputCopy.data(), input.values(), input.words * 2) == 0 &&
                std::memcmp(weightCopy.data(),weight.data(),weightCopy.size()) == 0,
                "immutable input or weight changed");
        for (auto &route : routes) {
          route.output.check(true); route.partial.check(bool(route.params.partition));
          const uint32_t parts = route.params.partition ?
              (shape.k + route.params.partition - 1) / route.params.partition : 1;
          for (const auto &dot : golden)
            route.numerical.add(route.output.values()[uint64_t(dot.row) * shape.n + dot.column], dot, shape.k + parts);
          require(route.numerical.intervalViolation == 0 && (cancellation||route.numerical.relativeL2() < .006),
                  "output failed independent double BF16 reduction-bound oracle");
          for (uint64_t cell = 0; cell < route.output.words; ++cell) {
            const auto actual=route.output.values()[cell],control=routes[0].output.values()[cell];
            if(actual==control)continue;
            ++route.controlMismatch;
            const auto dot=referenceCell(shape,input.values(),weight,uint32_t(cell/shape.n),uint32_t(cell%shape.n));
            route.changedCellsNumerical.add(actual,dot,shape.k+parts);
            route.changedCellsControlNumerical.add(control,dot,shape.k+1);
            const double midpoint=(double(number(actual))+number(control))*.5;
            route.changedExactMidpoints+=dot.sum==midpoint;
            route.changedMaxMidpointDistance=std::max(route.changedMaxMidpointDistance,std::abs(dot.sum-midpoint));
          }
          require(route.changedCellsNumerical.intervalViolation==0 &&
              route.changedCellsControlNumerical.intervalViolation==0,"changed-cell absolute interval failed");
          if (!first) report << ',';
          first = false;
          const double gpu = median(route.gpu), controlGPU = median(routes[0].gpu);
          report << "{\"shape\":" << splash::json::quote(shape.label) << ",\"rows\":" << shape.rows
              << ",\"k\":" << shape.k << ",\"n\":" << shape.n << ",\"tile_m\":" << shape.m
              << ",\"tile_n\":" << shape.tileN << ",\"simdgroups\":" << shape.simd
              << ",\"coefficient_kind\":"<<splash::json::quote(cancellation?"controlled cross-partition cancellation F32":
                  (source?"source-exact original affine F32":"synthetic BF16"))
              << ",\"partition_k\":" << route.params.partition << ",\"partitions\":" << parts
              << ",\"last_partition_k\":" << (route.params.partition ? shape.k - (parts - 1) * route.params.partition : shape.k)
              << ",\"scratch_logical_bytes\":" << (route.params.partition ? route.partial.words * 4 : 0)
              << ",\"scratch_actual_allocated_bytes\":" << (route.params.partition ? route.partial.allocatedBytes : 0)
              << ",\"unused_binding_dummy_allocated_bytes\":" << (route.params.partition ? 0 : route.partial.allocatedBytes)
              << ",\"scratch_write_bytes\":" << (route.params.partition ? route.partial.words * 4 : 0)
              << ",\"scratch_read_bytes\":" << (route.params.partition ? uint64_t(shape.rows) * shape.n * parts * 4 : 0)
              << ",\"scratch_write_read_bytes\":" << (route.params.partition ?
                  (route.partial.words + uint64_t(shape.rows) * shape.n * parts) * 4 : 0)
              << ",\"median_gpu_ms\":" << gpu * 1000 << ",\"median_wall_ms\":" << median(route.wall) * 1000
              << ",\"gpu_speedup_vs_whole_k\":" << controlGPU / gpu
              << ",\"effective_tflops\":" << 2.0 * shape.rows * shape.k * shape.n / gpu / 1e12
              << ",\"full_output_bf16_differences_vs_whole_k\":" << route.controlMismatch
              << ",\"guard_and_immutable_checks_pass\":true,\"numerical\":";
          route.numerical.write(report);
          report<<",\"every_changed_cell_split_oracle\":";route.changedCellsNumerical.write(report);
          report<<",\"every_changed_cell_control_oracle\":";route.changedCellsControlNumerical.write(report);
          report<<",\"changed_cells_exact_midpoint_count\":"<<route.changedExactMidpoints
              <<",\"changed_cells_max_midpoint_distance\":"<<route.changedMaxMidpointDistance
              <<",\"strict_bf16_cell_gate_pass\":"<<
                  ((!route.numerical.strictViolation&&!route.changedCellsNumerical.strictViolation)?"true":"false")<<'}';
          std::cerr << shape.label << " partition=" << route.params.partition << " gpu_ms=" << gpu * 1000
              << " speedup=" << controlGPU / gpu << " mismatches=" << route.controlMismatch << '\n';
        }
      }
      require(shapeCount > 0, "shape filter matched no shapes");
      report << "]}\n";
      std::ofstream output(argv[2]); require(bool(output), "cannot open report");
      output << report.str(); output.close(); require(bool(output), "cannot write report");
      std::cout << "{\"pass\":true,\"shape_count\":" << shapeCount << ",\"report\":"
          << splash::json::quote(argv[2]) << ",\"model_loaded\":false}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "ultra-splitk-oracle: " << error.what() << '\n'; return 1;
    }
  }
}
