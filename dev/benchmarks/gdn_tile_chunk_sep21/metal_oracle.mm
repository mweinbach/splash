// Standalone two-stage synthetic oracle: never links the model/runtime/MLX. The default
// build only compiles. Explicit --resources/--quality/--bench create Metal
// resources and are intended to be invoked by the serialized GPU coordinator.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "abi.hpp"
#include "frozen/cpu_oracle.hpp"
#include "wy_cpu.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>

namespace {
using gdn_chunk_cpu::Fixture;
using gdn_chunk_cpu::Trace;
using gdn_chunk_cpu::from_bf16;
using gdn_chunk_cpu::serial;
using gdn_chunk_cpu::to_bf16;
constexpr size_t K = 128, V = 128, Heads = 48, Mixed = 10240, Out = 6144;
constexpr size_t State = Heads * V * K, Guard = 256;
constexpr unsigned char Poison = 0xa5;
constexpr double RelativeTolerance = 1e-4, AbsoluteScaleTolerance = 5e-4;
constexpr double BF16RelativeTolerance = .003, BF16AbsoluteScaleTolerance = .004;
bool strictF64Quality = true;

void require(bool condition, const std::string &reason) {
    if (!condition) throw std::runtime_error(reason);
}
NSString *ns(const std::string &s) { return [NSString stringWithUTF8String:s.c_str()]; }
std::string str(NSString *s) { return s ? std::string(s.UTF8String) : std::string(); }
void emit(NSDictionary *record) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:record
        options:NSJSONWritingSortedKeys error:&error];
    require(data != nil, "JSON serialization: " + str(error.localizedDescription));
    std::cout.write(static_cast<const char *>(data.bytes), data.length);
    std::cout << '\n';
    std::cout.flush();
}
std::string digest(const void *data, size_t bytes) {
    CC_SHA256_CTX context{};
    CC_SHA256_Init(&context);
    auto *p = static_cast<const unsigned char *>(data);
    while (bytes) {
        const size_t count = std::min(bytes, size_t(1) << 30);
        CC_SHA256_Update(&context, p, CC_LONG(count));
        p += count; bytes -= count;
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> value{};
    CC_SHA256_Final(value.data(), &context);
    std::ostringstream out;
    for (auto byte : value) out << std::hex << std::setfill('0') << std::setw(2) << int(byte);
    return out.str();
}

struct Guarded {
    id<MTLBuffer> allocation;
    size_t bytes;
    std::string name, frozenHash;
    Guarded(id<MTLDevice> device, size_t size, std::string label)
        : bytes(size), name(std::move(label)) {
        require(bytes > 0, "zero-size buffer");
        allocation = [device newBufferWithLength:bytes + 2 * Guard
                                         options:MTLResourceStorageModeShared];
        require(allocation != nil, "allocation failed: " + name);
        allocation.label = ns(name);
        std::memset(allocation.contents, Poison, bytes + 2 * Guard);
    }
    void *data() const { return static_cast<unsigned char *>(allocation.contents) + Guard; }
    template<typename T> T *as() const { return static_cast<T *>(data()); }
    void canaries() const {
        const auto *p = static_cast<const unsigned char *>(allocation.contents);
        for (size_t i = 0; i < Guard; ++i) {
            const bool front = p[i] != Poison;
            const bool back = p[Guard + bytes + i] != Poison;
            if (!front && !back) continue;
            const size_t offset=front?i:Guard+bytes+i;
            emit(@{@"kind":@"canary_failure", @"buffer":ns(name),
                @"side":front?@"front":@"back", @"guard_offset_bytes":@(i),
                @"allocation_offset_bytes":@(offset), @"payload_bytes":@(bytes),
                @"expected_byte":@(Poison), @"actual_byte":@(p[offset]),
                @"guard_bytes":@(Guard), @"allocation_bytes":@(allocation.length)});
            require(false,"buffer canary changed: " + name);
        }
    }
    void freeze() { frozenHash = digest(data(), bytes); }
    void immutable() const {
        canaries();
        require(!frozenHash.empty() && digest(data(), bytes) == frozenHash,
                "immutable buffer hash changed: " + name);
    }
    void fillNaN() { std::fill_n(as<float>(), bytes / sizeof(float),
                                      std::numeric_limits<float>::quiet_NaN()); }
};

constexpr size_t Time=32,Tile=32,Tiles=4;
constexpr uint32_t NativeBit=256,ForcedBit=512;
constexpr const char *PrepareName="private_gdn_tile_prepare_t32_sg8";
constexpr const char *MainName="private_gdn_tile_local_t32_v32_sg8";
constexpr const char *AuditName="private_gdn_tile_local_audit_t32_v32_sg8";
constexpr const char *ProbeName="private_gdn_tile_local_probe_t32_v32_sg8";
constexpr const char *NativeName="private_gdn_tile_native_control";
constexpr const char *NativeAuditName="private_gdn_tile_native_audit_control";
size_t oldPreparedStride() {return 3*Time*K+Time*Time+Time;}
size_t preparedStride() {return oldPreparedStride()+2*Time;}
struct GPU {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::map<std::string,id<MTLComputePipelineState>> pipelines;
    explicit GPU(const std::string &path) {
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device");
        queue = [device newCommandQueue];
        require(queue != nil, "no command queue");
        NSError *error = nil;
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:ns(path)] error:&error];
        require(library != nil, "metallib: " + str(error.localizedDescription));
        emit(@{@"kind": @"device", @"name": device.name, @"metallib": ns(path),
               @"recommended_working_set_bytes": @(device.recommendedMaxWorkingSetSize)});
    }
    id<MTLComputePipelineState> pipeline(const std::string &name) {
        const auto found = pipelines.find(name);
        if (found != pipelines.end()) return found->second;
        id<MTLFunction> function = [library newFunctionWithName:ns(name)];
        require(function != nil, "missing shader: " + name);
        NSError *error = nil;
        id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
        require(p != nil, "pipeline " + name + ": " + str(error.localizedDescription));
        pipelines.emplace(name,p);
        return p;
    }
};

struct Inputs {
    unsigned lanes;
    size_t rows;
    Guarded mixed, decay, beta, params;
    Inputs(GPU &gpu, const Fixture &f, unsigned batch=1,uint32_t mode=0)
        : lanes(batch), rows(f.rows), mixed(gpu.device, batch * f.rows * Mixed * 2, "mixed BF16"),
          decay(gpu.device, batch * f.rows * Heads * 4, "decay F32"),
          beta(gpu.device, batch * f.rows * Heads * 2, "beta BF16"),
          params(gpu.device, sizeof(TileChunkParams), "parameters") {
        require(batch==1,"standalone component supports B1 only");
        f.validate();
        require(f.key_dim == K && f.value_dim == V, "fixture must be K128/V128");
        for (size_t b = 0; b < batch; ++b) for (size_t t = 0; t < f.rows; ++t) {
            auto *target = mixed.as<uint16_t>() + (b * f.rows + t) * Mixed;
            for (size_t h = 0; h < 16; ++h) {
                std::copy_n(f.q.data() + t * K, K, target + h * K);
                std::copy_n(f.k.data() + t * K, K, target + 2048 + h * K);
            }
            for (size_t h = 0; h < Heads; ++h) {
                std::copy_n(f.v.data() + t * V, V, target + 4096 + h * V);
                decay.as<float>()[(b * f.rows + t) * Heads + h] = f.alpha[t];
                beta.as<uint16_t>()[(b * f.rows + t) * Heads + h] = f.beta[t];
            }
        }
        FlashGDNParams gdn{unsigned(f.rows), batch, 16, 48, 128, 128, 4, 1e-6f,
                         uint64_t(3 * Mixed * 2), uint64_t(State * 4)};
        TileChunkParams p{gdn,mode,0};
        std::memcpy(params.data(), &p, sizeof(p));
        mixed.freeze(); decay.freeze(); beta.freeze(); params.freeze();
    }
    Inputs(GPU &gpu, const std::vector<Fixture> &heads, unsigned batch=1,uint32_t mode=0)
        : Inputs(gpu, heads.at(0), batch,mode) {
        require(heads.size() == Heads, "heterogeneous input head count");
        for (size_t h = 0; h < Heads; ++h) {
            const auto &f = heads[h];
            f.validate();
            require(f.rows == rows && f.key_dim == K && f.value_dim == V,
                    "heterogeneous input dimensions");
            require(f.q == heads[(h / 3) * 3].q && f.k == heads[(h / 3) * 3].k,
                    "Q/K must be shared within a value-head trio");
        }
        for (size_t b = 0; b < batch; ++b) for (size_t t = 0; t < rows; ++t) {
            auto *target = mixed.as<uint16_t>() + (b * rows + t) * Mixed;
            for (size_t keyHead = 0; keyHead < 16; ++keyHead) {
                const auto &f = heads[keyHead * 3];
                std::copy_n(f.q.data() + t * K, K, target + keyHead * K);
                std::copy_n(f.k.data() + t * K, K, target + 2048 + keyHead * K);
            }
            for (size_t h = 0; h < Heads; ++h) {
                const auto &f = heads[h];
                std::copy_n(f.v.data() + t * V, V, target + 4096 + h * V);
                decay.as<float>()[(b * rows + t) * Heads + h] = f.alpha[t];
                beta.as<uint16_t>()[(b * rows + t) * Heads + h] = f.beta[t];
            }
        }
        mixed.freeze(); decay.freeze(); beta.freeze();
    }
    void check() const { mixed.immutable(); decay.immutable(); beta.immutable(); params.immutable(); }
};

struct Error {
    double maxAbs = 0, relativeRMS = 0, referencePeak = 0;
    size_t count = 0, nonFinite = 0;
    bool pass() const {
        return !nonFinite && relativeRMS <= RelativeTolerance &&
               maxAbs <= AbsoluteScaleTolerance * std::max(1.0, referencePeak);
    }
    NSDictionary *json() const {
        return @{@"count": @(count), @"nonfinite": @(nonFinite), @"max_abs": @(maxAbs),
                 @"relative_rms": @(relativeRMS), @"reference_peak": @(referencePeak),
                 @"relative_rms_tolerance": @(RelativeTolerance),
                 @"max_abs_tolerance": @(AbsoluteScaleTolerance * std::max(1.0, referencePeak)),
                 @"pass": @(pass())};
    }
};
template<typename Read> Error errors(size_t count, const std::vector<double> &truth,
                                     size_t repeats, Read read) {
    require(count == truth.size() * repeats, "metric reference dimension");
    Error r; r.count = count;
    long double error2 = 0, reference2 = 0;
    for (size_t i = 0; i < count; ++i) {
        const double x = read(i), y = truth[i % truth.size()];
        if (!std::isfinite(x) || !std::isfinite(y)) { ++r.nonFinite; continue; }
        const double difference = std::abs(x - y);
        r.maxAbs = std::max(r.maxAbs, difference);
        r.referencePeak = std::max(r.referencePeak, std::abs(y));
        error2 += static_cast<long double>(difference) * difference;
        reference2 += static_cast<long double>(y) * y;
    }
    const long double ratio = reference2 > 0 ? error2 / reference2 : error2;
    r.relativeRMS = double(std::sqrt(ratio));
    require(std::isfinite(r.maxAbs) && std::isfinite(r.relativeRMS), "metric overflow");
    return r;
}

std::vector<Fixture> heterogeneousFixtures(size_t rows, unsigned phase) {
    const Fixture base = gdn_chunk_cpu::continuation_fixture(rows);
    std::vector<Fixture> heads;
    for (size_t h = 0; h < Heads; ++h) {
        Fixture f = base;
        f.name = "heterogeneous_head_" + std::to_string(h);
        const size_t keyHead = h / 3, keyShift = (keyHead * 7 + phase * 19) % K;
        for (size_t t = 0; t < rows; ++t) {
            for (size_t d = 0; d < K; ++d) {
                f.q[t*K+d] = to_bf16(from_bf16(base.q[t*K+(d+keyShift)%K]) *
                                     (0.96f + float(keyHead) * .005f));
                f.k[t*K+d] = to_bf16(from_bf16(base.k[t*K+(d+keyShift)%K]) *
                                     (1.02f - float(keyHead) * .003f));
            }
            for (size_t v = 0; v < V; ++v)
                f.v[t*V+v] = to_bf16(from_bf16(base.v[t*V+(v+h*5+phase*11)%V]) *
                                     (.8f + float(h) * .007f) + float(h+phase) * .0003f);
            f.alpha[t] = .80f + float(h) * .003f + float((t+phase)%7) * .0005f;
            f.beta[t] = to_bf16(.15f + float(h) * .011f + float((t+phase)%13) * .0003f);
        }
        for (size_t i = 0; i < V*K; ++i)
            f.initial_state[i] = base.initial_state[(i+h*131)%(V*K)] *
                (.7f + float(h) * .01f) + (float(h)-23.5f) * .0001f;
        f.validate();
        heads.push_back(std::move(f));
    }
    return heads;
}

struct HeadTruth { std::vector<double> state, output; };
HeadTruth heterogeneousTruth(const std::vector<Fixture> &heads,
                             const std::vector<double> *priorState = nullptr) {
    require(heads.size() == Heads, "all-head truth dimensions");
    require(!priorState || priorState->size() == State, "all-head prior truth dimensions");
    HeadTruth truth;
    truth.state.resize(State); truth.output.resize(heads.front().rows * Out);
    for (size_t h = 0; h < Heads; ++h) {
        std::vector<double> prior;
        if (priorState) prior.assign(priorState->begin()+h*V*K,priorState->begin()+(h+1)*V*K);
        const auto trace = serial<double>(heads[h],priorState ? &prior : nullptr);
        std::copy(trace.final_state.begin(),trace.final_state.end(),truth.state.begin()+h*V*K);
        for (size_t t = 0; t < heads[h].rows; ++t)
            std::copy_n(trace.out.data()+t*V,V,truth.output.data()+t*Out+h*V);
    }
    return truth;
}

std::vector<float> heterogeneousSeeds(const std::vector<Fixture> &heads) {
    std::vector<float> seed(State);
    for (size_t h = 0; h < Heads; ++h)
        std::copy(heads[h].initial_state.begin(),heads[h].initial_state.end(),seed.begin()+h*V*K);
    return seed;
}

bool bf16Pass(const Error &e) {
    return !e.nonFinite && e.relativeRMS <= BF16RelativeTolerance &&
        e.maxAbs <= BF16AbsoluteScaleTolerance * std::max(1.0,e.referencePeak);
}
NSDictionary *bf16Metrics(const Error &e) {
    return @{@"count":@(e.count), @"nonfinite":@(e.nonFinite), @"max_abs":@(e.maxAbs),
        @"relative_rms":@(e.relativeRMS), @"reference_peak":@(e.referencePeak),
        @"relative_rms_tolerance":@(BF16RelativeTolerance),
        @"max_abs_tolerance":@(BF16AbsoluteScaleTolerance * std::max(1.0,e.referencePeak)),
        @"pass":@(bf16Pass(e))};
}

Fixture guardProofFixture(unsigned kind,size_t rows=67) {
    Fixture f;
    f.name=std::array<const char *,4>{{"proof_dyadic_cancellation","proof_postzero_underflow_huge_seed",
        "proof_real_zero_segments","proof_safe_impulse"}}.at(kind);
    f.rows=rows; f.q.assign(rows*K,0); f.k.assign(rows*K,0); f.v.assign(rows*V,0);
    f.beta.assign(rows,to_bf16(0)); f.alpha.assign(rows,1); f.initial_state.assign(V*K,0);
    for (size_t t=0;t<rows;++t) f.q[t*K]=f.k[t*K]=to_bf16(1);
    f.beta[0]=to_bf16(kind==0?.5f:1.f);
    for (size_t v=0;v<V;++v) {
        f.v[v]=to_bf16(kind==0?1.125f:(kind==1?std::ldexp(1.f,100):1.f));
        if (kind==0 || kind==1) f.initial_state[v*K]=kind==0?1.f:std::ldexp(1.f,100);
    }
    if (kind==1 || kind==2) {
        f.alpha[0]=0; f.alpha[1]=std::ldexp(1.f,-100);
        f.alpha[2]=kind==1?std::ldexp(1.f,-100):0;
        if (kind==2) f.alpha[3]=std::ldexp(1.f,-100);
    }
    f.validate(); return f;
}

std::vector<Fixture> normBoundaryFixtures() {
    std::vector<Fixture> heads;
    for (size_t h=0;h<Heads;++h) {
        Fixture f=guardProofFixture(3);
        std::fill(f.k.begin(),f.k.end(),0);
        if ((h/3)%3==0) {
            f.name="norm_subnormal_square"; f.k[0]=to_bf16(std::ldexp(1.f,-64));
        } else if ((h/3)%3==1) {
            f.name="norm_overflow_saturation"; f.k[0]=to_bf16(std::ldexp(1.f,64));
            for (size_t v=0;v<V;++v) f.v[v]=to_bf16(std::ldexp(1.f,-64));
        } else {
            f.name="state_finite_saturation"; std::fill(f.q.begin(),f.q.end(),0);
            std::fill(f.beta.begin(),f.beta.end(),to_bf16(0)); std::fill(f.v.begin(),f.v.end(),0);
            f.initial_state[0]=std::numeric_limits<float>::max();
        }
        f.validate(); heads.push_back(std::move(f));
    }
    return heads;
}


struct Results {
    size_t rows,chunks; bool audit,fullSeeds;
    Guarded state,output,diagnostics,prepared,range,decisions,history,delta,preOutput,incoming,seed;
    Results(GPU &gpu,size_t tokens,bool auditing=false,bool probe=false)
        : rows(tokens),chunks((tokens+Time-1)/Time),audit(auditing),fullSeeds(probe),
          state(gpu.device,State*4,"state F32"),output(gpu.device,tokens*Out*2,"output BF16"),
          diagnostics(gpu.device,4,"diagnostics"),
          prepared(gpu.device,chunks*Heads*preparedStride()*4,"prepared coefficients and norms"),
          range(gpu.device,chunks*Heads*4,"per-chunk/head range"),
          decisions(gpu.device,chunks*Heads*Tiles*4,"per-chunk/head/tile decisions"),
          history(gpu.device,audit?tokens*V*K*4:4,"head0 state history"),
          delta(gpu.device,audit?tokens*V*4:4,"head0 delta"),
          preOutput(gpu.device,audit?tokens*V*4:4,"head0 pre-BF16 output"),
          incoming(gpu.device,audit?chunks*V*K*4:(probe?chunks*State*4:4),"hybrid incoming seed tape"),
          seed(gpu.device,State*4,"immutable GPU reset seed") {}
    void reset(const std::vector<float> &initial) {
        require(initial.size()==V*K || initial.size()==State,"seed dimensions");
        for (size_t h=0;h<Heads;++h) std::copy_n(initial.data()+(initial.size()==State?h*V*K:0),V*K,state.as<float>()+h*V*K);
        std::memcpy(seed.data(),state.data(),state.bytes); seed.freeze();
        std::memset(output.data(),Poison,output.bytes); *diagnostics.as<uint32_t>()=0;
        prepared.fillNaN(); history.fillNaN(); delta.fillNaN(); preOutput.fillNaN(); incoming.fillNaN();
        std::fill_n(range.as<uint32_t>(),range.bytes/4,0xffffffffu);
        std::fill_n(decisions.as<uint32_t>(),decisions.bytes/4,0xffffffffu);
    }
    void check(bool head0=false) const {
        for (const Guarded *g:{&state,&output,&diagnostics,&prepared,&range,&decisions,&history,&delta,&preOutput,&incoming,&seed}) g->canaries();
        seed.immutable();
        if (head0) {
            require(std::memcmp(state.as<float>()+V*K,seed.as<float>()+V*K,(State-V*K)*4)==0,"head0 dispatch changed other states");
            for (size_t t=0;t<rows;++t) {
                const auto *bytes=reinterpret_cast<const unsigned char *>(output.as<uint16_t>()+t*Out+V);
                for (size_t i=0;i<(Out-V)*2;++i) require(bytes[i]==Poison,"head0 dispatch changed other outputs");
            }
        }
    }
    std::vector<float> headCarry() const {return {state.as<float>(),state.as<float>()+V*K};}
    std::vector<float> fullCarry() const {return {state.as<float>(),state.as<float>()+State};}
};

enum class Run {Candidate,Audit,Probe,Native,NativeAudit,V6Math,OriginalNative};
struct Timing {double gpu,wall;};
Timing dispatch(GPU &gpu,Inputs &input,Results &r,Run run,bool gpuReset=false,uint32_t tile=0) {
    const bool component=run==Run::Candidate || run==Run::Audit || run==Run::Probe;
    const bool v6=run==Run::V6Math;
    const bool audit=run==Run::Audit || run==Run::NativeAudit || v6;
    const bool original=run==Run::OriginalNative;
    const unsigned threads=original?512:256;
    const char *name=original?"flash_gdn_staged_v16_t16":(run==Run::Candidate?MainName:(run==Run::Audit?AuditName:(run==Run::Probe?ProbeName:
        (run==Run::Native?NativeName:(run==Run::NativeAudit?NativeAuditName:"private_gdn_tile_v6_math_audit_control")))));
    auto pipeline=gpu.pipeline(name);
    auto prep=(component||v6)?gpu.pipeline(v6?"private_gdn_tile_v6_prepare_control":PrepareName):nil;
    for (id<MTLComputePipelineState> p:{pipeline,prep}) if (p) {
        require(p.maxTotalThreadsPerThreadgroup>=(p==pipeline?threads:256u) && p.threadExecutionWidth==32,"component/control thread geometry");
        require(p.staticThreadgroupMemoryLength<=gpu.device.maxThreadgroupMemoryLength,"component/control TG memory");
    }
    if (v6) {
        require(input.rows<=Time && tile<Tiles,"v6 safe control dimensions");
        std::fill_n(r.range.as<uint32_t>(),r.range.bytes/4,0u);
        auto *params=input.params.as<TileChunkParams>(); params->reserved=tile;
        input.params.freeze();
    }
    const auto start=std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command=[gpu.queue commandBuffer];
    if (gpuReset) {
        id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
        [blit copyFromBuffer:r.seed.allocation sourceOffset:Guard toBuffer:r.state.allocation destinationOffset:Guard size:r.state.bytes];
        [blit endEncoding];
    }
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if (prep) {
        [encoder setComputePipelineState:prep];
        const std::array<const Guarded *,7> b{{&input.mixed,&input.decay,&input.beta,&r.prepared,&r.range,&r.diagnostics,&input.params}};
        for (size_t i=0;i<b.size();++i) [encoder setBuffer:b[i]->allocation offset:Guard atIndex:i];
        [encoder dispatchThreadgroups:MTLSizeMake(Heads,r.chunks,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [encoder setComputePipelineState:pipeline];
    const std::array<const Guarded *,6> common{{&input.mixed,&input.decay,&input.beta,&r.state,&r.output,&r.diagnostics}};
    for (size_t i=0;i<common.size();++i) [encoder setBuffer:common[i]->allocation offset:Guard atIndex:i];
    if (component) {
        [encoder setBuffer:r.prepared.allocation offset:Guard atIndex:6];
        [encoder setBuffer:r.range.allocation offset:Guard atIndex:7];
        [encoder setBuffer:r.decisions.allocation offset:Guard atIndex:8];
        if (run==Run::Audit) {
            [encoder setBuffer:r.history.allocation offset:Guard atIndex:9];
            [encoder setBuffer:r.delta.allocation offset:Guard atIndex:10];
            [encoder setBuffer:r.preOutput.allocation offset:Guard atIndex:11];
            [encoder setBuffer:r.incoming.allocation offset:Guard atIndex:12];
            [encoder setBuffer:input.params.allocation offset:Guard atIndex:13];
        } else if (run==Run::Probe) {
            [encoder setBuffer:r.incoming.allocation offset:Guard atIndex:9];
            [encoder setBuffer:input.params.allocation offset:Guard atIndex:10];
        } else [encoder setBuffer:input.params.allocation offset:Guard atIndex:9];
    } else if (v6) {
        [encoder setBuffer:r.prepared.allocation offset:Guard atIndex:6];
        [encoder setBuffer:r.range.allocation offset:Guard atIndex:7];
        [encoder setBuffer:r.history.allocation offset:Guard atIndex:8];
        [encoder setBuffer:r.delta.allocation offset:Guard atIndex:9];
        [encoder setBuffer:r.preOutput.allocation offset:Guard atIndex:10];
        [encoder setBuffer:input.params.allocation offset:Guard atIndex:11];
    } else if (audit) {
        [encoder setBuffer:r.history.allocation offset:Guard atIndex:6];
        [encoder setBuffer:r.delta.allocation offset:Guard atIndex:7];
        [encoder setBuffer:r.preOutput.allocation offset:Guard atIndex:8];
        [encoder setBuffer:input.params.allocation offset:Guard atIndex:9];
    } else [encoder setBuffer:input.params.allocation offset:Guard atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(audit?1:Heads,original?8:(v6?1:(component?Tiles:16)),1)
           threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"command failed: "+str(command.error.localizedDescription));
    const Timing t{command.GPUEndTime-command.GPUStartTime,
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()};
    require(std::isfinite(t.gpu) && t.gpu>0 && t.gpu<600,"invalid GPU timing"); return t;
}

size_t differingBytes(const void *a,const void *b,size_t n) {
    const auto *x=static_cast<const unsigned char *>(a),*y=static_cast<const unsigned char *>(b);size_t count=0;
    for (size_t i=0;i<n;++i) count+=x[i]!=y[i];return count;
}
Fixture sliced(const Fixture &f,size_t begin,size_t count,const std::vector<float> &seed) {
    Fixture s=f;s.rows=count;s.initial_state=seed;
    s.q.assign(f.q.begin()+begin*K,f.q.begin()+(begin+count)*K);
    s.k.assign(f.k.begin()+begin*K,f.k.begin()+(begin+count)*K);
    s.v.assign(f.v.begin()+begin*V,f.v.begin()+(begin+count)*V);
    s.beta.assign(f.beta.begin()+begin,f.beta.begin()+begin+count);
    s.alpha.assign(f.alpha.begin()+begin,f.alpha.begin()+begin+count);s.validate();return s;
}
NSDictionary *decisionMetrics(const Results &r,size_t heads=Heads) {
    size_t native=0,safe=0,forced=0,range=0,cancel=0,nonfinite=0,norm=0;
    NSMutableArray *words=[NSMutableArray array];
    for (size_t c=0;c<r.chunks;++c) for (size_t h=0;h<heads;++h) for (size_t t=0;t<Tiles;++t) {
        const uint32_t word=r.decisions.as<uint32_t>()[(c*Heads+h)*Tiles+t];
        require(!(word&~0x30fu),"undefined decision bits or unwritten decision");
        native+=(word&NativeBit)!=0;safe+=(word&NativeBit)==0;forced+=(word&ForcedBit)!=0;
        range+=(word&1)!=0;cancel+=(word&2)!=0;nonfinite+=(word&4)!=0;norm+=(word&8)!=0;
        [words addObject:@{@"chunk":@(c),@"head":@(h),@"tile":@(t),@"word":@(word)}];
    }
    return @{@"chunk_tiles":@(r.chunks*heads*Tiles),@"native_tiles":@(native),@"safe_tiles":@(safe),
        @"forced_tiles":@(forced),@"range_tiles":@(range),@"cancellation_tiles":@(cancel),
        @"nonfinite_tiles":@(nonfinite),@"norm_tiles":@(norm),@"decisions":words};
}

struct LocalProof {size_t nativeTiles=0,safeTiles=0,state=0,out=0,history=0,delta=0,pre=0;bool pass() const {return !state&&!out&&!history&&!delta&&!pre;}};
LocalProof localAuditProof(GPU &gpu,const Fixture &f,Results &r) {
    LocalProof p; r.incoming.freeze();
    for (size_t c=0;c<r.chunks;++c) {
        const size_t begin=c*Time,count=std::min(Time,r.rows-begin);
        std::vector<float> seed(r.incoming.as<float>()+c*V*K,r.incoming.as<float>()+(c+1)*V*K);
        auto chunk=sliced(f,begin,count,seed);
        Inputs nativeInput(gpu,chunk);Results native(gpu,count,true);native.reset(seed);
        dispatch(gpu,nativeInput,native,Run::NativeAudit);nativeInput.check();native.check(true);
        const auto *next=c+1<r.chunks?r.incoming.as<float>()+(c+1)*V*K:r.state.as<float>();
        for (size_t tile=0;tile<Tiles;++tile) {
            const uint32_t word=r.decisions.as<uint32_t>()[(c*Heads)*Tiles+tile];
            const bool isNative=word&NativeBit;
            Results safe(gpu,count,true);
            Inputs safeInput(gpu,chunk);
            const Results *control=&native;
            if (!isNative) {
                require(r.range.as<uint32_t>()[c*Heads]==0,"safe tile uses uncomputed range coefficients");
                safe.reset(seed);dispatch(gpu,safeInput,safe,Run::V6Math,false,uint32_t(tile));
                safeInput.check();safe.check(true);control=&safe;++p.safeTiles;
            } else ++p.nativeTiles;
            p.state+=differingBytes(next+tile*Tile*K,control->state.as<float>()+tile*Tile*K,Tile*K*4);
            for (size_t token=0;token<count;++token) {
                p.out+=differingBytes(r.output.as<uint16_t>()+(begin+token)*Out+tile*Tile,
                    control->output.as<uint16_t>()+token*Out+tile*Tile,Tile*2);
                p.delta+=differingBytes(r.delta.as<float>()+(begin+token)*V+tile*Tile,
                    control->delta.as<float>()+token*V+tile*Tile,Tile*4);
                p.pre+=differingBytes(r.preOutput.as<float>()+(begin+token)*V+tile*Tile,
                    control->preOutput.as<float>()+token*V+tile*Tile,Tile*4);
                p.history+=differingBytes(r.history.as<float>()+((begin+token)*V+tile*Tile)*K,
                    control->history.as<float>()+(token*V+tile*Tile)*K,Tile*K*4);
            }
            require(!*control->diagnostics.as<uint32_t>(),"local reference diagnostics nonzero");
        }
    }
    r.incoming.immutable();return p;
}

NSDictionary *coefficientMetrics(const Fixture &f,const Results &r) {
    const auto truth=gdn_nax_cpu::prepare<double>(f,Time);const size_t old=truth.stride();
    const std::array<size_t,5> offsets{{0,Time*K,2*Time*K,3*Time*K,3*Time*K+Time*Time}};
    const std::array<size_t,5> lengths{{Time*K,Time*K,Time*K,Time*Time,Time}};
    NSMutableDictionary *d=[NSMutableDictionary dictionary];const std::array<const char *,5> names{{"W_f32","U_f32","end_key_f32","score_f32","prefix_f32"}};
    bool rawPass=true;size_t skipped=0,required=0;
    for (size_t c=0;c<r.chunks;++c) skipped+=r.range.as<uint32_t>()[c*Heads]!=0;
    for (size_t c=0;c<r.chunks;++c) {
        bool safe=false;for (size_t tile=0;tile<Tiles;++tile) safe=safe||!(r.decisions.as<uint32_t>()[c*Heads*Tiles+tile]&NativeBit);
        required+=safe;
    }
    for (size_t field=0;field<5;++field) {
        std::vector<double> ref;std::vector<size_t> locations;
        for (size_t c=0;c<r.chunks;++c) for (size_t i=0;i<lengths[field];++i) {
            ref.push_back(truth.storage[c*old+offsets[field]+i]);locations.push_back(c*Heads*preparedStride()+offsets[field]+i);
        }
        const Error e=errors(ref.size(),ref,1,[&](size_t i) {return double(r.prepared.as<float>()[locations[i]]);});
        rawPass=rawPass&&e.pass();d[ns(names[field])]=e.json();
    }
    d[@"raw_pass"]=@(rawPass);d[@"range_skipped_chunks"]=@(skipped);
    d[@"computed_chunks"]=@(r.chunks-skipped);d[@"required_chunks"]=@(required);
    d[@"required"]=@(required>0);d[@"unused_chunks"]=@(r.chunks-required);
    d[@"not_computed_reason"]=@"Range-selected chunks return before Gram/WU; defined placeholder zeros are reported against unchanged F64 transforms, but these transforms are unused by native execution.";
    return d;
}

struct AuditOutcome {bool policy,strict;std::vector<float> carry;};
AuditOutcome auditStage(GPU &gpu,const Fixture &f,const std::vector<float> &seed,const Trace<double> &truth,
                       uint32_t mode,const char *stage) {
    Inputs input(gpu,f,1,mode);Results r(gpu,f.rows,true);r.reset(seed);
    dispatch(gpu,input,r,Run::Audit);input.check();r.check(true);
    const auto proof=localAuditProof(gpu,f,r);
    const Error history=errors(truth.history.size(),truth.history,1,[&](size_t i) {return double(r.history.as<float>()[i]);});
    const Error delta=errors(truth.delta.size(),truth.delta,1,[&](size_t i) {return double(r.delta.as<float>()[i]);});
    const Error pre=errors(truth.out.size(),truth.out,1,[&](size_t i) {return double(r.preOutput.as<float>()[i]);});
    const Error state=errors(V*K,truth.final_state,1,[&](size_t i) {return double(r.state.as<float>()[i]);});
    const Error out=errors(truth.out.size(),truth.out,1,[&](size_t i) {return double(from_bf16(r.output.as<uint16_t>()[(i/V)*Out+i%V]));});
    const auto coefficients=coefficientMetrics(f,r);
    const bool strict=history.pass()&&delta.pass()&&pre.pass()&&state.pass()&&[coefficients[@"raw_pass"] boolValue]&&!*r.diagnostics.as<uint32_t>();
    const bool policy=proof.pass()&&!*r.diagnostics.as<uint32_t>();
    strictF64Quality=strictF64Quality&&strict;
    emit(@{@"kind":@"tile_chunk_quality",@"fixture":ns(f.name),@"stage":ns(stage),@"mode":@(mode),@"rows":@(f.rows),
        @"guarded_policy_pass":@(policy),@"strict_f64_qualification_pass":@(strict),
        @"history_f32":history.json(),@"delta_f32":delta.json(),@"pre_bf16_output_f32":pre.json(),@"carried_state_f32":state.json(),
        @"output_bf16_vs_f64":bf16Metrics(out),@"prepared_coefficients":coefficients,@"selection":decisionMetrics(r,1),
        @"native_chunk_tiles_checked":@(proof.nativeTiles),@"safe_v6_tiles_checked":@(proof.safeTiles),
        @"state_mismatch_bytes":@(proof.state),@"output_mismatch_bytes":@(proof.out),@"history_mismatch_bytes":@(proof.history),
        @"delta_mismatch_bytes":@(proof.delta),@"preoutput_mismatch_bytes":@(proof.pre),
        @"same_hybrid_seed":@YES,@"canaries_pass":@YES,@"immutable_sha256_pass":@YES,
        @"raw_f64_gates_relaxed":@NO,@"whole_history_accuracy_certificate":@NO});
    return {policy,strict,r.headCarry()};
}

Fixture mixedTileFixture() {
    Fixture f=guardProofFixture(0);f.name="mixed_native_safe_value_tiles";
    for (size_t v=Tile;v<V;++v) f.initial_state[v*K]=0;
    return f;
}

bool fullHeadProof(GPU &gpu,const std::vector<Fixture> &heads,uint32_t mode,const char *label) {
    Inputs input(gpu,heads,1,mode);Results r(gpu,heads.front().rows,false,true);r.reset(heterogeneousSeeds(heads));
    dispatch(gpu,input,r,Run::Probe);input.check();r.check();r.incoming.freeze();
    Results timed(gpu,heads.front().rows);timed.reset(heterogeneousSeeds(heads));
    dispatch(gpu,input,timed,Run::Candidate);timed.check();input.check();
    const size_t timedState=differingBytes(r.state.data(),timed.state.data(),r.state.bytes);
    const size_t timedOut=differingBytes(r.output.data(),timed.output.data(),r.output.bytes);
    const size_t timedDecisions=differingBytes(r.decisions.data(),timed.decisions.data(),r.decisions.bytes);
    size_t nativeTiles=0,safeTiles=0,stateBytes=0,outputBytes=0;
    for (size_t c=0;c<r.chunks;++c) {
        const size_t begin=c*Time,count=std::min(Time,r.rows-begin);
        std::vector<float> seed(r.incoming.as<float>()+c*State,r.incoming.as<float>()+(c+1)*State);
        std::vector<Fixture> chunks;
        for (size_t h=0;h<Heads;++h) {
            std::vector<float> local(seed.begin()+h*V*K,seed.begin()+(h+1)*V*K);
            chunks.push_back(sliced(heads[h],begin,count,local));
        }
        Inputs referenceInput(gpu,chunks);Results reference(gpu,count);reference.reset(seed);
        dispatch(gpu,referenceInput,reference,Run::Native);referenceInput.check();reference.check();
        const auto *next=c+1<r.chunks?r.incoming.as<float>()+(c+1)*State:r.state.as<float>();
        for (size_t h=0;h<Heads;++h) for (size_t tile=0;tile<Tiles;++tile) {
            const auto word=r.decisions.as<uint32_t>()[(c*Heads+h)*Tiles+tile];
            if (!(word&NativeBit)) {++safeTiles;continue;} ++nativeTiles;
            stateBytes+=differingBytes(next+(h*V+tile*Tile)*K,reference.state.as<float>()+(h*V+tile*Tile)*K,Tile*K*4);
            for (size_t token=0;token<count;++token)
                outputBytes+=differingBytes(r.output.as<uint16_t>()+(begin+token)*Out+h*V+tile*Tile,
                    reference.output.as<uint16_t>()+token*Out+h*V+tile*Tile,Tile*2);
        }
        require(!*reference.diagnostics.as<uint32_t>(),"full-head native chunk control diagnostics");
    }
    r.incoming.immutable();
    const auto truth=heterogeneousTruth(heads);
    const Error state=errors(State,truth.state,1,[&](size_t i) {return double(r.state.as<float>()[i]);});
    const Error out=errors(truth.output.size(),truth.output,1,[&](size_t i) {return double(from_bf16(r.output.as<uint16_t>()[i]));});
    const bool pass=!stateBytes&&!outputBytes&&!timedState&&!timedOut&&!timedDecisions&&
        !*r.diagnostics.as<uint32_t>()&&!*timed.diagnostics.as<uint32_t>();
    emit(@{@"kind":@"tile_chunk_full_head_proof",@"fixture":ns(label),@"mode":@(mode),@"rows":@(r.rows),
        @"pass":@(pass),@"native_tiles_checked":@(nativeTiles),@"safe_tiles_reported":@(safeTiles),
        @"native_state_mismatch_bytes":@(stateBytes),@"native_output_mismatch_bytes":@(outputBytes),
        @"actual_timed_pipeline_checked":@YES,@"timed_vs_probe_state_mismatch_bytes":@(timedState),
        @"timed_vs_probe_output_mismatch_bytes":@(timedOut),@"timed_vs_probe_decision_mismatch_bytes":@(timedDecisions),
        @"carried_state_f32":state.json(),@"output_bf16_vs_f64":bf16Metrics(out),
        @"selection":decisionMetrics(r),@"same_hybrid_seed":@YES,@"canaries_pass":@YES,@"immutable_sha256_pass":@YES,
        @"safe_full_head_gate_scope":@"Safe math is independently checked by compact audit; full-head probe checks native-selected tiles."});
    return pass;
}

bool quality(GPU &gpu,bool proofOnly=false) {
    bool policy=true;strictF64Quality=true;
    auto fixtures=proofOnly?std::vector<Fixture>{}:gdn_chunk_cpu::make_fixtures();
    fixtures.push_back(mixedTileFixture());
    for (unsigned k=0;k<3;++k) fixtures.push_back(guardProofFixture(k));
    // The existing fixture suite supplies zero decays, zero beta, range,
    // cancellation, padding and R67 tails. Tiny row boundaries are explicit.
    if (!proofOnly) for (size_t rows:{size_t(1),size_t(31),size_t(32),size_t(33)})
        fixtures.push_back(gdn_chunk_cpu::continuation_fixture(rows));
    emit(@{@"kind":@"tile_chunk_quality_registration",@"batches":@1,@"time":@32,@"values":@32,@"simdgroups":@8,
        @"f32_relative_rms_tolerance":@(RelativeTolerance),@"f32_max_abs_scale_tolerance":@(AbsoluteScaleTolerance),
        @"native_control":@"V8 Time16 from actual incoming hybrid chunk seed",@"safe_control":@"independent frozen v6 local math, selected safe tile only",
        @"not_computed_transforms":@"Range-selected chunks expose defined zeros and explicit not-computed/unused metadata; raw F64 metrics remain reported."});
    size_t stages=0;
    for (const auto &f:fixtures) for (uint32_t mode:{0u,1u,2u}) {
        const auto truth=serial<double>(f);
        auto first=auditStage(gpu,f,f.initial_state,truth,mode,"initial");policy=policy&&first.policy;++stages;
        auto continuation=gdn_chunk_cpu::continuation_fixture(19);
        const auto continuedTruth=serial<double>(continuation,&truth.final_state);
        auto second=auditStage(gpu,continuation,first.carry,continuedTruth,mode,"continued_hybrid_carry");
        policy=policy&&second.policy;++stages;
    }
    std::vector<Fixture> mixed;
    for (size_t h=0;h<Heads;++h) mixed.push_back(h%2?guardProofFixture(1):mixedTileFixture());
    // Both templates share the same Q/K, preserving each key-head trio.
    policy=fullHeadProof(gpu,heterogeneousFixtures(67,0),0,"heterogeneous_normal")&&policy;
    for (uint32_t mode:{0u,1u,2u}) policy=fullHeadProof(gpu,mixed,mode,"heterogeneous_mixed_native_safe")&&policy;
    policy=fullHeadProof(gpu,normBoundaryFixtures(),0,"norm_underflow_saturation")&&policy;
    emit(@{@"kind":@"tile_chunk_quality_summary",@"guarded_policy_pass":@(policy),
        @"strict_f64_qualification_pass":@(strictF64Quality),@"audit_stages":@(stages),@"fixtures":@(fixtures.size()),
        @"forced_modes_checked":@[@0,@1,@2],@"fullR_native_replay":@NO,@"raw_f64_gates_relaxed":@NO});
    return policy;
}

void resources(GPU &gpu) {
    for (const char *name:{PrepareName,MainName,AuditName,ProbeName,NativeName,NativeAuditName,
        "private_gdn_tile_v6_prepare_control","private_gdn_tile_v6_math_audit_control","flash_gdn_staged_v16_t16"}) {
        auto p=gpu.pipeline(name);const unsigned threads=std::string(name)=="flash_gdn_staged_v16_t16"?512:256;
        require(p.maxTotalThreadsPerThreadgroup>=threads&&p.threadExecutionWidth==32,"pipeline geometry");
        require(p.staticThreadgroupMemoryLength<=gpu.device.maxThreadgroupMemoryLength,"pipeline memory limit");
        emit(@{@"kind":@"resources",@"name":ns(name),@"max_threads":@(p.maxTotalThreadsPerThreadgroup),
            @"execution_width":@(p.threadExecutionWidth),@"threadgroup_bytes":@(p.staticThreadgroupMemoryLength)});
    }
}

double median(std::vector<double> data) {require(!data.empty(),"empty timings");std::sort(data.begin(),data.end());return data[data.size()/2];}
void bench(GPU &gpu,size_t rows) {
    require(rows>=64&&rows<=2048,"initial performance scope R64..2048");
    const auto fixture=gdn_chunk_cpu::continuation_fixture(rows);Inputs input(gpu,fixture);Results r(gpu,rows);r.reset(fixture.initial_state);
    auto run=[&](bool candidate) {return dispatch(gpu,input,r,candidate?Run::Candidate:Run::OriginalNative,true);};
    double warm=0;size_t commands=0;
    while (warm<.150||commands<8) {warm+=run(false).gpu;warm+=run(true).gpu;commands+=2;require(commands<4096,"bounded warming exhausted");}
    emit(@{@"kind":@"bench_warmup",@"gpu_ms":@(warm*1000),@"commands":@(commands),@"cpu_tensor_access":@NO});
    std::mt19937 rng(0x47444332u);std::vector<double> aTimes,bTimes,speedups,wall;
    for (size_t pair=0;pair<10;++pair) {
        const bool reverse=rng()&1;const std::array<bool,4> order{{reverse,!reverse,!reverse,reverse}};
        std::array<Timing,2> a{},b{};size_t ai=0,bi=0;
        for (size_t slot=0;slot<4;++slot) {
            const auto t=run(order[slot]);if (order[slot]) b[bi++]=t;else a[ai++]=t;
            emit(@{@"kind":@"bench_sample",@"pair":@(pair),@"slot":@(slot),@"order":reverse?@"BAAB":@"ABBA",
                @"run":order[slot]?@"component":@"native_v16_t16",@"rows":@(rows),@"gpu_ms":@(t.gpu*1000),@"wall_ms":@(t.wall*1000)});
        }
        const double at=(a[0].gpu+a[1].gpu)*.5,bt=(b[0].gpu+b[1].gpu)*.5;
        aTimes.push_back(at);bTimes.push_back(bt);speedups.push_back(at/bt);wall.push_back((a[0].wall+a[1].wall)/(b[0].wall+b[1].wall));
    }
    input.check();r.check();require(!*r.diagnostics.as<uint32_t>(),"bench diagnostics nonzero");
    emit(@{@"kind":@"bench_summary",@"rows":@(rows),@"matched_pairs":@10,@"native_gpu_ms_median":@(median(aTimes)*1000),
        @"component_gpu_ms_median":@(median(bTimes)*1000),@"paired_gpu_speedup_median":@(median(speedups)),
        @"paired_wall_speedup_median":@(median(wall)),@"selection":decisionMetrics(r),@"cpu_tensor_access_during_timing":@NO,
        @"scope":@"B1 isolated preparation+tile/chunk recurrence; no model throughput claim",
        @"native_performance_baseline":@"frozen flash_gdn_staged_v16_t16; 512 threads; grid48x8",
        @"preparation_included":@YES,@"native_fallback_included":@YES,@"gpu_seed_reset_included":@YES,
        @"fullR_replay":@NO,@"snapshot":@NO,@"canaries_pass":@YES,@"immutable_sha256_pass":@YES});
}

void cpuLayout() {
    size_t cases=0,groups=0;
    for (size_t rows:{size_t(1),size_t(15),size_t(16),size_t(17),size_t(31),size_t(32),size_t(33),size_t(63),size_t(64),size_t(65),size_t(67),size_t(2047),size_t(2048)}) {
        const size_t chunks=(rows+Time-1)/Time;
        for (size_t c=0;c<chunks;++c) for (size_t h=0;h<Heads;++h) {
            require(((c*Heads+h)+1)*preparedStride()<=chunks*Heads*preparedStride(),"prepared bound");
            require(c*Heads+h<chunks*Heads,"range bound");++groups;
            for (size_t tile=0;tile<Tiles;++tile) {
                require((c*Heads+h)*Tiles+tile<chunks*Heads*Tiles,"decision bound");
                require((h*V+tile*Tile)*K+Tile*K<=State,"state tile bound");
                require(c*V*K+tile*Tile*K+Tile*K<=chunks*V*K,"audit incoming tape bound");
                require((c*Heads+h)*V*K+tile*Tile*K+Tile*K<=chunks*State,"full incoming tape bound");
                ++groups;
            }
        }
        ++cases;
    }
    const size_t coefs=64*Heads*preparedStride()*4,range=64*Heads*4,decisions=64*Heads*Tiles*4;
    const auto round=[](size_t n) {return (n+16383)/16384*16384;};
    const size_t physical=round(coefs)+round(range)+round(decisions);
    require(sizeof(TileChunkParams)==56&&physical==164823040,"component ABI/arena contract");
    emit(@{@"kind":@"cpu_layout",@"pass":@YES,@"geometry_cases":@(cases),@"checked_groups":@(groups),
        @"batches":@1,@"time":@32,@"values":@32,@"simdgroups":@8,@"abi_bytes":@56,
        @"coefficient_bytes_2k":@(coefs),@"range_bytes_2k":@(range),@"decision_bytes_2k":@(decisions),
        @"planned_arena_physical_bytes_2k":@(physical),@"audit_incoming_seed_bytes_2k":@(64*V*K*4),
        @"full_probe_seed_bytes_2k":@(64*State*4),@"main_threadgroup_declared_bytes":@9332,
        @"timed_snapshot":@NO,@"fullR_replay":@NO,@"metal_device_created":@NO});
}
} // namespace

int main(int argc,char **argv) {
    @autoreleasepool {try {
        if (argc==1) {std::cout<<"Usage: tile-oracle --cpu-layout|--resources|--quality|--branch-proof|--bench [--library path] [--rows 64..2048]\n";return 0;}
        std::string mode,library=(std::filesystem::absolute(argv[0]).parent_path()/"gdn-tile.metallib").string();size_t rows=2048;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if (arg=="--cpu-layout"||arg=="--resources"||arg=="--quality"||arg=="--branch-proof"||arg=="--bench") {require(mode.empty(),"choose one mode");mode=arg;}
            else if (arg=="--library") {require(++i<argc,"library path missing");library=argv[i];}
            else if (arg=="--rows") {require(++i<argc,"rows missing");size_t used=0;rows=std::stoul(argv[i],&used);require(used==std::strlen(argv[i])&&rows>=1&&rows<=2048,"rows invalid");}
            else throw std::runtime_error("unknown argument: "+arg);
        }
        require(!mode.empty(),"mode missing");if (mode=="--cpu-layout") {cpuLayout();return 0;}
        GPU gpu(library);if (mode=="--resources") resources(gpu);
        else if (mode=="--quality"||mode=="--branch-proof") return quality(gpu,mode=="--branch-proof")?0:2;
        else bench(gpu,rows);return 0;
    } catch(const std::exception &e) {emit(@{@"kind":@"error",@"message":ns(e.what())});return 1;}}
}
