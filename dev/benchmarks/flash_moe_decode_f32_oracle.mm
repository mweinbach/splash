// Private original-F32-coefficient MPP decode-MoE accuracy/performance oracle.
// --cpu-self-test and compilation create no backend and submit no GPU work.
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashAffine.hpp"
#include "FlashMoEDecodeF32.hpp"
#include "flash/FlashMoE.hpp"
#include "engine/Json.hpp"
#include "metal/abi/FlashMoEBuckets.h"
#include "../tests/flash/FlashMoEBucketsReference.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace splash::flash;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::CommandTiming;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
namespace ref = splash::flash::bucket_reference;
constexpr uint32_t kSticky = 0x80000000u;
constexpr uint64_t kGuardBytes = 64;

void require(bool value, const std::string &reason) {
  if (!value) throw std::runtime_error(reason);
}
uint16_t bf16(float value) {
  const uint32_t word = std::bit_cast<uint32_t>(value);
  return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
float number(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
void cpuSelfTest() {
  const std::array<uint32_t, 2> words{0x76543210u, 0xfedcba98u};
  const std::array<uint16_t, 16> golden{0xbf00, 0xbe80, 0x0000, 0x3e80,
      0x3f00, 0x3f40, 0x3f80, 0x3fa0, 0x3fc0, 0x3fe0, 0x4000, 0x4010,
      0x4020, 0x4030, 0x4040, 0x4050};
  for (uint32_t k = 0; k < 16; ++k) {
    const uint32_t code = (words[k / 8] >> ((k % 8) * 4)) & 15u;
    require(bf16(float(code) * 0.25f - 0.5f) == golden[k],
        "handwritten Q4/BF16 coefficient golden differs");
    const uint16_t signedGolden = k == 2 ? 0 : uint16_t(golden[k] ^ 0x8000u);
    require(bf16(float(code) * -0.25f + 0.5f) == signedGolden,
        "signed scale golden differs");
  }
  require(moEBucketJobCapacity(512, 10, 16) == 831, "512-row job bound differs");
  require(moEBucketJobCapacity(2048, 10, 32) == 1151, "2048-row job bound differs");
}
uint32_t envNumber(const char *name, uint32_t fallback, uint32_t limit) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  require(*raw && *raw != '-', std::string("invalid ") + name);
  size_t consumed = 0;
  const unsigned long parsed = std::stoul(raw, &consumed);
  require(consumed == std::strlen(raw) && parsed > 0 && parsed <= limit,
      std::string("invalid ") + name);
  return uint32_t(parsed);
}

std::string pipelineMetadata(const char *path) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  require(device != nil, "Metal device unavailable");
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&error];
  require(library != nil, "combined candidate library unavailable");
  std::ostringstream out;
  out << "{\"device_max_threadgroup_memory_bytes\":" << device.maxThreadgroupMemoryLength
      << ",\"pipelines\":[";
  bool first = true;
  const std::array<const char *,8> names{"flash_moe_decode_f32_gate_up_m8_n64",
      "flash_moe_decode_f32_down_scatter_m8_n64","flash_moe_decode_f32_coefficient_sample",
      "flash_expert_qmv_contig_k16_sg4_c2","flash_expert_qmv_contig_k8_sg2_c4",
      "flash_moe_decode_bf16_gate_up_m8_n64","flash_moe_decode_bf16_coefficient_sample",
      "flash_moe_q4x8_gate_up_m8_n64"};
  for (const char *name : names) {
    id<MTLFunction> function=[library newFunctionWithName:[NSString stringWithUTF8String:name]];
    require(function!=nil,std::string("required function missing: ")+name);
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline!=nil,std::string("pipeline creation failed: ")+name);
    const uint32_t threads=std::string_view(name).ends_with("coefficient_sample") ? 256u :
        std::string_view(name)=="flash_expert_qmv_contig_k8_sg2_c4" ? 64u : 128u;
    require(pipeline.threadExecutionWidth==32 && pipeline.maxTotalThreadsPerThreadgroup>=threads &&
        pipeline.staticThreadgroupMemoryLength<=device.maxThreadgroupMemoryLength,
        std::string("pipeline limits exceeded: ")+name+", static="+
        std::to_string(pipeline.staticThreadgroupMemoryLength));
    if (!first) out<<','; first=false;
    out<<"{\"name\":"<<splash::json::quote(name)<<",\"execution_width\":"<<pipeline.threadExecutionWidth
       <<",\"maximum_threads\":"<<pipeline.maxTotalThreadsPerThreadgroup
       <<",\"static_threadgroup_memory_bytes\":"<<pipeline.staticThreadgroupMemoryLength<<'}';
  }
  out << "]}";
#if !__has_feature(objc_arc)
  [library release]; [device release];
#endif
  return out.str();
}

template<class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values,
                                     const char *label) {
  auto result = backend.allocateBuffer(values.size() * sizeof(T), BufferStorage::Shared, label);
  std::memcpy(result.contents(), values.data(), values.size() * sizeof(T));
  return result;
}
template<class T> std::vector<T> readFile(const char *path, uint64_t elements) {
  require(std::filesystem::file_size(path) == elements * sizeof(T),
      std::string("raw fixture file has wrong size: ") + path);
  std::vector<T> result(elements);
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(T));
  require(bool(in), std::string("raw fixture file read failed: ") + path);
  return result;
}
struct Guard final {
  MetalBuffer allocation;
  uint64_t logicalBytes;
  bool clean() const {
    const auto *bytes = static_cast<const uint8_t *>(allocation.contents());
    return std::all_of(bytes + logicalBytes, bytes + logicalBytes + kGuardBytes,
        [](uint8_t byte) { return byte == 0x5a; });
  }
};
MetalBuffer guarded(MetalBackend &backend, uint64_t bytes, std::vector<Guard> &guards) {
  auto buffer = backend.allocateBuffer(bytes + kGuardBytes, BufferStorage::Shared,
      "Q4x8 oracle guarded output");
  std::memset(buffer.contents(), 0xa5, bytes);
  std::memset(static_cast<uint8_t *>(buffer.contents()) + bytes, 0x5a, kGuardBytes);
  guards.push_back({buffer, bytes});
  return backend.view(buffer, 0, bytes);
}
void sourceAlignment(const FlashAffineProjection &p) {
  require(p.bits == 4 && p.groupSize == 64 && p.experts == 512 &&
      p.weightRowStrideBytes % 4 == 0 && p.weightExpertStrideBytes % 4 == 0 &&
      p.weights && p.weights->buffer.contents() &&
      reinterpret_cast<uintptr_t>(p.weights->buffer.contents()) % 4 == 0,
      "Q4x8 candidate requires aligned original Q4/G64 expert planes");
}
template<class T> void expected(const MetalBuffer &buffer, const std::vector<T> &values,
                               const char *label) {
  require(buffer.sizeBytes() >= values.size() * sizeof(T) &&
      std::memcmp(buffer.contents(), values.data(), values.size() * sizeof(T)) == 0,
      std::string("independent bucket comparison failed: ") + label);
}
void checkBuckets(const FlashMoEBlockedScratch &scratch, const ref::Packed &packed,
                  const ref::Jobs &jobs) {
  require(std::memcmp(scratch.buckets.counts.contents(), packed.counts.data(), 512 * 4) == 0,
      "independent counts differ");
  require(std::memcmp(scratch.buckets.offsets.contents(), packed.offsets.data(), 513 * 4) == 0,
      "independent offsets differ");
  expected(scratch.buckets.routeMap, packed.routeMap, "stable route map");
  expected(scratch.buckets.canonicalToPacked, packed.canonicalToPacked, "inverse route map");
  expected(scratch.buckets.packedInputs, packed.inputs, "BF16 bit-copy inputs");
  require(*static_cast<const uint32_t *>(scratch.buckets.jobCount.contents()) == jobs.count,
      "independent active job count differs");
  require(std::memcmp(scratch.buckets.jobOffsets.contents(), jobs.offsets.data(), 513 * 4) == 0,
      "independent job offsets differ");
  const auto *actualJobs = static_cast<const FlashMoEBucketJob *>(scratch.buckets.tileJobs.contents());
  for (uint32_t index = 0; index < jobs.entries.size(); ++index)
    require(actualJobs[index].expert == jobs.entries[index].expert &&
        actualJobs[index].row_begin == jobs.entries[index].rowBegin,
        "independent active/inactive job records differ at " + std::to_string(index));
}
void times(std::ostream &out, const std::vector<CommandTiming> &values, bool gpu) {
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << (gpu ? values[i].gpuSeconds : values[i].wallSeconds) * 1000;
  }
  out << ']';
}

struct Error final {
  uint64_t elements=0,mismatches=0,nonfinite=0;
  uint32_t maxULP=0;
  double maxAbs=0,squaredError=0,squaredReference=0;
  void add(uint16_t actual,uint16_t expected) {
    ++elements; mismatches+=actual!=expected;
    const float a=number(actual),b=number(expected);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; return; }
    const double delta=double(a)-b;
    maxAbs=std::max(maxAbs,std::abs(delta));
    squaredError+=delta*delta; squaredReference+=double(b)*b;
    const auto ordered=[](uint16_t word) { return word&0x8000 ? uint32_t(0x8000-(word&0x7fff)) : uint32_t(0x8000+word); };
    const auto x=ordered(actual),y=ordered(expected);
    maxULP=std::max(maxULP,x>y ? x-y : y-x);
  }
  double relativeL2() const { return std::sqrt(squaredError/std::max(1e-30,squaredReference)); }
  void write(std::ostream &out) const {
    out<<"{\"elements\":"<<elements<<",\"bf16_mismatches\":"<<mismatches
       <<",\"nonfinite\":"<<nonfinite<<",\"max_bf16_ulp\":"<<maxULP
       <<",\"max_abs\":"<<maxAbs<<",\"relative_l2\":"<<relativeL2()<<'}';
  }
};
Error errors(const MetalBuffer &a,const MetalBuffer &b,uint64_t words) {
  const auto *actual=static_cast<const uint16_t *>(a.contents());
  const auto *expected=static_cast<const uint16_t *>(b.contents());
  Error result;
  for (uint64_t i=0; i<words; ++i) result.add(actual[i],expected[i]);
  return result;
}
float coefficient(const FlashAffineProjection &p,uint32_t expert,uint32_t n,uint32_t k) {
  const auto *w=static_cast<const uint8_t *>(p.weights->buffer.contents())+
      uint64_t(expert)*p.weightExpertStrideBytes+uint64_t(n)*p.weightRowStrideBytes;
  const uint32_t code=(w[k/2]>>((k%2)*4))&15u;
  const uint64_t offset=uint64_t(expert)*p.parameterExpertStrideBytes+
      uint64_t(n)*p.parameterRowStrideBytes+uint64_t(k/64)*2;
  uint16_t sf,bi;
  std::memcpy(&sf,static_cast<const std::byte *>(p.scales->buffer.contents())+offset,2);
  std::memcpy(&bi,static_cast<const std::byte *>(p.biases->buffer.contents())+offset,2);
  volatile float product=float(code)*number(sf);
  return product+number(bi);
}
struct CoefficientAudit final {
  uint64_t samples=0,negativeScales=0,nonBF16=0;
  bool roundedOperand=false;
  void write(std::ostream &out) const {
    out<<"{\"samples\":"<<samples<<",\"operand_bit_mismatches\":0,\"bf16_operand_rounding\":"
       <<(roundedOperand ? "true" : "false")<<",\"negative_scale_samples\":"
       <<negativeScales<<",\"not_bf16_representable\":"<<nonBF16<<'}';
  }
};
CoefficientAudit auditCoefficients(MetalBackend &backend,
    const std::array<const FlashAffineProjection *,3> &projections,
    const std::vector<int64_t> &ids,MetalBuffer diagnostic,bool roundedOperand) {
  std::vector<uint32_t> experts;
  for (int64_t id : ids)
    if (std::find(experts.begin(),experts.end(),uint32_t(id))==experts.end()) experts.push_back(uint32_t(id));
  if (experts.size()>8) experts.resize(8);
  std::vector<Guard> guards;
  const auto output=guarded(backend,4096*4,guards);
  CoefficientAudit result; result.roundedOperand=roundedOperand;
  for (const auto *projection : projections) {
    const auto &p=*projection;
    for (uint32_t rank=0; rank<experts.size(); ++rank) {
      const uint32_t expert=experts[rank];
      const uint32_t nb=(rank*7%(p.outputSize/64))*64,kb=(rank*11%(p.inputSize/64))*64;
      CommandGraph graph;
      splash::flash::candidate::addMoEDecodeF32CoefficientSample(graph,p,expert,nb,kb,output,diagnostic,roundedOperand);
      (void)backend.submitCommand(graph.dispatches());
      const auto *values=static_cast<const float *>(output.contents());
      for (uint32_t index=0; index<4096; ++index) {
        const uint32_t n=nb+index/64,k=kb+index%64;
        const float original=coefficient(p,expert,n,k);
        const float golden=roundedOperand ? number(bf16(original)) : original;
        require(std::isfinite(values[index]) && std::bit_cast<uint32_t>(values[index])==
            std::bit_cast<uint32_t>(golden),"F32 expert coefficient bits differ");
        const uint64_t offset=uint64_t(expert)*p.parameterExpertStrideBytes+
            uint64_t(n)*p.parameterRowStrideBytes+uint64_t(k/64)*2;
        uint16_t sf;
        std::memcpy(&sf,static_cast<const std::byte *>(p.scales->buffer.contents())+offset,2);
        result.negativeScales+=number(sf)<0;
        result.nonBF16+=original!=number(bf16(original)); ++result.samples;
      }
      require(guards[0].clean(),"coefficient sampler output tail changed");
    }
  }
  require(result.negativeScales && result.nonBF16,"source coefficient audit did not cover signed/non-BF16 values");
  return result;
}

bool runCase(MetalBackend &backend,const FlashWeights &weights,const std::string &prefix,
    uint32_t rows,const std::string &pattern,uint32_t pairs,double limit,bool roundedOperand,std::ostream &out) {
  std::vector<uint16_t> hidden(uint64_t{rows}*2560);
  std::vector<int64_t> ids(uint64_t{rows}*10);
  for (uint64_t i=0; i<hidden.size(); ++i)
    hidden[i]=bf16(float(int((i*73+i/2560*17)%257)-128)/512.0f);
  for (uint32_t row=0; row<rows; ++row)
    for (uint32_t slot=0; slot<10; ++slot)
      ids[uint64_t{row}*10+slot]=pattern=="concentrated" ? slot : (row*73+slot*53)%512;
  const char *rawIDs=std::getenv("FLASH_MOE_DECODE_F32_IDS"),*rawInput=std::getenv("FLASH_MOE_DECODE_F32_INPUT");
  if (rawIDs) ids=readFile<int64_t>(rawIDs,ids.size());
  if (rawInput) { require(rawIDs,"raw hidden input requires matching IDs"); hidden=readFile<uint16_t>(rawInput,hidden.size()); }
  const auto packed=ref::pack(hidden,ids,rows,10,kSticky);
  require(packed.diagnostic==kSticky,"invalid/nonfinite original fixture");
  const auto jobs=ref::makeJobs(packed,8);
  const auto &gate=weights.projection(prefix+".gate_proj"),&up=weights.projection(prefix+".up_proj"),
      &down=weights.projection(prefix+".down_proj");
  sourceAlignment(gate); sourceAlignment(up); sourceAlignment(down);
  const auto input=upload(backend,hidden,"F32 decode hidden"),expertIDs=upload(backend,ids,"F32 decode expert IDs");
  const auto routes=upload(backend,std::vector<uint16_t>(ids.size(),bf16(0.1f)),"F32 decode route weights");
  const auto shared=upload(backend,std::vector<uint16_t>(uint64_t{rows}*2560,0),"F32 decode shared zeros");
  const auto sharedGate=upload(backend,std::vector<uint16_t>(rows,0),"F32 decode shared gate zeros");
  const auto diagnostic=upload(backend,std::vector<uint32_t>{kSticky},"F32 decode sticky diagnostic");
  std::vector<Guard> guards;
  std::array<MetalBuffer,2> g,u,activation,outputs,downs;
  auto scratch=allocateMoEBlockedScratch(backend,rows);
  scratch.packedActivated=guarded(backend,uint64_t{rows}*10*640*2,guards);
  scratch.scatteredDown=guarded(backend,uint64_t{rows}*10*2560*2,guards);
  for (uint32_t i=0; i<2; ++i) {
    g[i]=guarded(backend,uint64_t{rows}*10*640*2,guards);
    u[i]=guarded(backend,uint64_t{rows}*10*640*2,guards);
    outputs[i]=guarded(backend,uint64_t{rows}*2560*2,guards);
  }
  activation[0]=guarded(backend,uint64_t{rows}*10*640*2,guards);
  downs[0]=guarded(backend,uint64_t{rows}*10*2560*2,guards);
  activation[1]=scratch.packedActivated; downs[1]=scratch.scatteredDown;
  std::array<CommandGraph,2> graphs;
  addGatheredAffine(graphs[0],input,gate,expertIDs,g[0],diagnostic,rows,10);
  addGatheredAffine(graphs[0],input,up,expertIDs,u[0],diagnostic,rows,10);
  addSiLUMultiply(graphs[0],g[0],u[0],activation[0],diagnostic,rows,640,10);
  addGatheredAffine(graphs[0],activation[0],down,expertIDs,downs[0],diagnostic,rows,10,true);
  addCombine(graphs[0],downs[0],expertIDs,routes,shared,sharedGate,outputs[0],diagnostic,rows,2560,512,10);
  require(graphs[0].dispatches()[0].pipelineName=="flash_expert_qmv_contig_k16_sg4_c2" &&
      graphs[0].dispatches()[3].pipelineName=="flash_expert_qmv_contig_k8_sg2_c4",
      "baseline must use actual selected expert QMV profile");
  addMoEBlockedPack(graphs[1],input,expertIDs,scratch,diagnostic,rows,FlashMoEBlockedTile::M8N64);
  if (roundedOperand) {
    addMoEBlockedGateUp(graphs[1],gate,up,scratch,diagnostic,rows,FlashMoEBlockedTile::M8N64);
    addMoEBlockedDownScatter(graphs[1],down,scratch,diagnostic,rows,FlashMoEBlockedTile::M8N64);
  } else {
    splash::flash::candidate::addMoEDecodeF32GateUp(graphs[1],gate,up,scratch,g[1],u[1],diagnostic,rows);
    splash::flash::candidate::addMoEDecodeF32Down(graphs[1],down,scratch,diagnostic,rows);
  }
  addCombine(graphs[1],downs[1],expertIDs,routes,shared,sharedGate,outputs[1],diagnostic,rows,2560,512,10);
  (void)backend.submitCommand(graphs[0].dispatches());
  (void)backend.submitCommand(graphs[1].dispatches());
  checkBuckets(scratch,packed,jobs);
  if (roundedOperand) {
    // Untimed boundary taps use the stock BF16 producer body. Prove its
    // activation matches the timed stock graph before using gate/up taps.
    std::vector<uint16_t> stockActivation(uint64_t{rows}*10*640);
    std::memcpy(stockActivation.data(),scratch.packedActivated.contents(),stockActivation.size()*2);
    CommandGraph taps;
    splash::flash::candidate::addMoEDecodeBF16GateUpTaps(taps,gate,up,scratch,g[1],u[1],diagnostic,rows);
    (void)backend.submitCommand(taps.dispatches());
    require(std::memcmp(stockActivation.data(),scratch.packedActivated.contents(),stockActivation.size()*2)==0,
        "untimed BF16 gate/up tap body differs from timed stock activation");
  }
  const auto audit=auditCoefficients(backend,{&gate,&up,&down},ids,diagnostic,roundedOperand);
  const auto compareStages=[&] {
    std::array<Error,5> errorsByStage{errors(g[1],g[0],uint64_t{rows}*10*640),
        errors(u[1],u[0],uint64_t{rows}*10*640),{},errors(downs[1],downs[0],uint64_t{rows}*10*2560),
        errors(outputs[1],outputs[0],uint64_t{rows}*2560)};
    const auto *actual=static_cast<const uint16_t *>(activation[1].contents());
    const auto *golden=static_cast<const uint16_t *>(activation[0].contents());
    for (uint32_t packedRow=0; packedRow<rows*10; ++packedRow)
      for (uint32_t n=0; n<640; ++n)
        errorsByStage[2].add(actual[uint64_t{packedRow}*640+n],golden[uint64_t{packed.routeMap[packedRow]}*640+n]);
    return errorsByStage;
  };
  auto stageErrors=compareStages();
  const auto healthy=[&] {
    require(*static_cast<const uint32_t *>(diagnostic.contents())==kSticky,"sticky diagnostics changed");
    for (const auto &guard : guards) require(guard.clean(),"decode candidate output canary changed");
  };
  healthy();
  std::array<std::vector<CommandTiming>,2> timing;
  for (uint32_t pair=0; pair<pairs; ++pair)
    for (uint32_t order=0; order<2; ++order) {
      const uint32_t which=(pair+order)%2;
      timing[which].push_back(backend.submitCommand(graphs[which].dispatches())); healthy();
    }
  stageErrors=compareStages(); checkBuckets(scratch,packed,jobs);
  bool accurate=true;
  for (const auto &error : stageErrors) accurate &= !error.nonfinite && error.relativeL2()<=limit;
  out<<"{\"rows\":"<<rows<<",\"pattern\":"<<splash::json::quote(rawInput ? "captured-input-and-ids" :
      rawIDs ? "captured-ids-synthetic-hidden" : pattern)<<",\"active_experts\":"
      <<std::count_if(packed.counts.begin(),packed.counts.end(),[](uint32_t n) { return n!=0; })
      <<",\"active_jobs\":"<<jobs.count<<",\"matrix_launch_jobs\":"<<rows*10
      <<",\"row_utilization\":"<<double(rows*10)/double(jobs.count*8)
      <<",\"accuracy_pass\":"<<(accurate ? "true" : "false")<<",\"canaries_clean\":true,\"coefficient_audit\":";
  audit.write(out); out<<",\"stages\":{";
  const std::array<const char *,5> labels{"gate","up","activation","down","combine"};
  for (uint32_t i=0; i<5; ++i) { if (i) out<<','; out<<splash::json::quote(labels[i])<<':'; stageErrors[i].write(out); }
  out<<"},\"control_gpu_ms\":"; times(out,timing[0],true);
  out<<",\"candidate_gpu_ms\":"; times(out,timing[1],true);
  out<<",\"control_wall_ms\":"; times(out,timing[0],false);
  out<<",\"candidate_wall_ms\":"; times(out,timing[1],false); out<<'}'; out.flush();
  return accurate;
}
} // namespace

int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      cpuSelfTest();
      volatile float product=15.0f*0.00787353515625f;
      const float coefficientTrap=product-0.03125f;
      require(coefficientTrap!=number(bf16(coefficientTrap)),"CPU F32 coefficient trap rounds to BF16");
      if (argc==2 && std::string_view(argv[1])=="--cpu-self-test") {
        std::cout<<"{\"pass\":true,\"gpu_work\":false,\"checks\":[\"Q4_signed_coefficients\",\"F32_not_BF16_trap\",\"M8_job_bounds\"]}\n"; return 0;
      }
      if (argc==3 && std::string_view(argv[1])=="--pipeline-metadata") {
        std::cout<<pipelineMetadata(argv[2])<<'\n'; return 0;
      }
      require(argc==4,"usage: flash-moe-decode-f32-oracle COMBINED_METALLIB PACKAGE REPORT_JSON");
      require(setenv("SPLASH_FLASH_QMV_F32","1",1)==0 && setenv("SPLASH_FLASH_EXPERT_QMV","1",1)==0 &&
          setenv("SPLASH_FLASH_MOE_Q4X8","1",1)==0,"cannot force actual QMV baseline profile");
      const uint32_t pairs=envNumber("FLASH_MOE_DECODE_F32_PAIRS",6,64);
      const char *selectedPrefix=std::getenv("FLASH_MOE_DECODE_F32_PREFIX");
      const std::string prefix=selectedPrefix ? selectedPrefix : "language_model.model.layers.0.mlp.switch_mlp";
      const char *modeRaw=std::getenv("FLASH_MOE_DECODE_MODE");
      const std::string_view mode=modeRaw ? modeRaw : "f32";
      require(mode=="f32" || mode=="bf16","FLASH_MOE_DECODE_MODE must be f32 or bf16");
      const bool roundedOperand=mode=="bf16";
      double limit=1e-3;
      if (const char *raw=std::getenv("FLASH_MOE_DECODE_F32_LIMIT")) {
        size_t used=0; limit=std::stod(raw,&used);
        require(used==std::strlen(raw) && limit>0 && limit<=(roundedOperand ? .005 : .001) && std::isfinite(limit),"invalid accuracy limit");
      }
      const auto metadata=pipelineMetadata(argv[1]);
      MetalBackend backend(argv[1]); const auto weights=FlashWeights::load(backend,argv[2]);
      std::ofstream out(argv[3]); require(bool(out),"cannot create report");
      out<<std::setprecision(12)<<"{\"schema\":\"flash-moe-decode-f32-v1\",\"source_identity\":"
          <<splash::json::quote(weights.sourceIdentity())<<",\"manifest_identity\":"
          <<splash::json::quote(weights.manifestFingerprint())<<",\"prefix\":"<<splash::json::quote(prefix)
          <<",\"baseline_math\":"<<splash::json::quote(flashAffineSemantics())
          <<",\"candidate_math\":"<<splash::json::quote(roundedOperand ? kFlashMoEBlockedQ4x8Semantics : splash::flash::candidate::kMoEDecodeF32Semantics)
          <<",\"activation_quantization\":false,\"weight_conversion\":false,\"coefficient_weight_dtype\":"
          <<splash::json::quote(roundedOperand ? "BF16-rounded-original-F32" : "F32")
          <<",\"untimed_gate_up_taps\":"<<(roundedOperand ? "true" : "false")
          <<",\"relative_l2_limit\":"<<limit<<",\"pipeline_metadata\":"<<metadata
          <<",\"timing_scope\":\"complete selected MoE chain; candidate includes bucket packing/jobs; alternating matched warm commands\",\"pairs\":"
          <<pairs<<",\"cases\":[";
      const uint32_t selectedRows=std::getenv("FLASH_MOE_DECODE_F32_ROWS") ? envNumber("FLASH_MOE_DECODE_F32_ROWS",16,16) : 0;
      bool first=true,allAccurate=true;
      for (uint32_t rows : selectedRows ? std::vector<uint32_t>{selectedRows} : std::vector<uint32_t>{2,4,8,16}) {
        for (const char *pattern : {"spread","concentrated"}) {
          if (!first) out<<','; first=false;
          allAccurate &= runCase(backend,weights,prefix,rows,pattern,pairs,limit,roundedOperand,out);
          if (std::getenv("FLASH_MOE_DECODE_F32_IDS")) break;
        }
      }
      out<<"],\"pass\":"<<(allAccurate ? "true" : "false")<<"}\n";
      out.flush(); require(bool(out),"report write failed");
      require(allAccurate,"original F32 coefficient MPP route exceeded stage accuracy limit; valid failure report saved");
      std::cout<<"{\"pass\":true,\"report\":"<<splash::json::quote(argv[3])<<"}\n"; return 0;
    } catch (const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
  }
}
