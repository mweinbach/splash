// Standalone two-stage synthetic oracle: never links the model/runtime/MLX. The default
// build only compiles. Explicit --resources/--quality/--bench create Metal
// resources and are intended to be invoked by the serialized GPU coordinator.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "metal/abi/FlashGDN.h"
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
        for (size_t i = 0; i < Guard; ++i)
            require(p[i] == Poison && p[Guard + bytes + i] == Poison,
                    "buffer canary changed: " + name);
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

struct Variant { const char *name; unsigned values, time, groups; };
constexpr std::array<Variant, 2> Variants{{
    {"private_gdn_wy_v32_t32_sg4", 32, 32, 4},
    {"private_gdn_wy_v32_t32_sg8", 32, 32, 8}}};
constexpr Variant Canonical{"flash_gdn_staged_v16_t16", 16, 16, 16};
constexpr const char *NativeAuditName = "private_gdn_canonical_audit_v16_t16";
std::string auditName(const Variant &v) {
    return "private_gdn_wy_audit_v" + std::to_string(v.values) + "_t" +
        std::to_string(v.time) + "_sg" + std::to_string(v.groups);
}
std::string preparationName(const Variant &v) {
    return "private_gdn_wy_prepare_t" + std::to_string(v.time) + "_sg" + std::to_string(v.groups);
}
size_t oldPreparedStride(size_t time) { return 3*time*K + time*time + time; }
size_t preparedStride(size_t time) { return oldPreparedStride(time)+2*time; }
std::string v3PreparationName(const Variant &v) {
    return "private_gdn_wy_v3_prepare_t"+std::to_string(v.time)+"_sg"+std::to_string(v.groups);
}
std::string v3ApplyName(const Variant &v,bool audit) {
    return std::string("private_gdn_wy_v3_")+(audit?"audit_v":"v")+
        std::to_string(v.values)+"_t"+std::to_string(v.time)+"_sg"+std::to_string(v.groups);
}

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
    Inputs(GPU &gpu, const Fixture &f, unsigned batch)
        : lanes(batch), rows(f.rows), mixed(gpu.device, batch * f.rows * Mixed * 2, "mixed BF16"),
          decay(gpu.device, batch * f.rows * Heads * 4, "decay F32"),
          beta(gpu.device, batch * f.rows * Heads * 2, "beta BF16"),
          params(gpu.device, sizeof(FlashGDNParams), "parameters") {
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
        FlashGDNParams p{unsigned(f.rows), batch, 16, 48, 128, 128, 4, 1e-6f,
                         uint64_t(3 * Mixed * 2), uint64_t(State * 4)};
        std::memcpy(params.data(), &p, sizeof(p));
        mixed.freeze(); decay.freeze(); beta.freeze(); params.freeze();
    }
    Inputs(GPU &gpu, const std::vector<Fixture> &heads, unsigned batch)
        : Inputs(gpu, heads.at(0), batch) {
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

struct Results {
    unsigned lanes;
    size_t rows;
    bool audit;
    unsigned preparedTime;
    size_t coefficientStride;
    Guarded state, output, diagnostics, history, delta, preOutput, coefficients, seed, flags, snapshot, normReference;
    Results(GPU &gpu, size_t tokens, unsigned batch, bool auditing, unsigned time = 0,bool v3Control=false)
        : lanes(batch), rows(tokens), audit(auditing),
          preparedTime(time),
          coefficientStride(v3Control?oldPreparedStride(time):preparedStride(time)),
          state(gpu.device, batch * State * 4, "recurrent F32"),
          output(gpu.device, batch * tokens * Out * 2, "output BF16"),
          diagnostics(gpu.device, 4, "diagnostics"),
          history(gpu.device, auditing ? batch * tokens * V * K * 4 : 4, "audit state history F32"),
          delta(gpu.device, auditing ? batch * tokens * V * 4 : 4, "audit delta F32"),
          preOutput(gpu.device, auditing ? batch * tokens * V * 4 : 4, "audit pre-BF16 output F32"),
          coefficients(gpu.device, time ? size_t(batch) * ((tokens + time - 1) / time) * Heads * coefficientStride * 4 : 4,
                       "prepared W/U/end-key/score/prefix and cached norms"),
          seed(gpu.device, batch * State * 4, "immutable GPU benchmark state seed F32"),
          flags(gpu.device, batch * Heads * 4, "sticky per-lane/head eligibility flags"),
          snapshot(gpu.device, batch * State * 4, "immutable incoming state snapshot F32"),
          normReference(gpu.device,v3Control && time ? size_t(batch)*((tokens+time-1)/time)*Heads*2*time*4:4,
                        "diagnostic v3 coefficient norm/flag reference") {}
    void reset(const std::vector<float> &seed) {
        require(seed.size() == V * K || seed.size() == State, "state seed dimension");
        for (size_t b = 0; b < lanes; ++b) for (size_t h = 0; h < Heads; ++h)
            std::copy_n(seed.data() + (seed.size() == State ? h * V * K : 0), V * K,
                        state.as<float>() + b * State + h * V * K);
        std::memset(output.data(), Poison, output.bytes);
        *diagnostics.as<uint32_t>() = 0;
        history.fillNaN(); delta.fillNaN(); preOutput.fillNaN(); coefficients.fillNaN();
        std::memset(flags.data(),0,flags.bytes); snapshot.fillNaN();
        normReference.fillNaN();
        std::memcpy(this->seed.data(), state.data(), state.bytes); this->seed.freeze();
    }
    void check(bool headZeroOnly, const std::vector<float> &seed) const {
        for (const Guarded *buffer : {&state, &output, &diagnostics, &history, &delta, &preOutput, &coefficients, &this->seed, &flags, &snapshot, &normReference})
            buffer->canaries();
        if (headZeroOnly) {
            for (size_t b = 0; b < lanes; ++b) {
                for (size_t h = 1; h < Heads; ++h)
                    require(std::memcmp(state.as<float>() + b * State + h * V * K,
                                        seed.data()+(seed.size()==State?h*V*K:0), V * K * 4) == 0,
                            "head-zero dispatch changed another recurrent head");
                for (size_t t = 0; t < rows; ++t) {
                    const auto *p = reinterpret_cast<const unsigned char *>(
                        output.as<uint16_t>() + (b * rows + t) * Out + V);
                    for (size_t i = 0; i < (Out - V) * 2; ++i)
                        require(p[i] == Poison, "head-zero dispatch changed another output head");
                }
            }
        }
    }
    std::vector<float> carried() const { return {state.as<float>(), state.as<float>() + V * K}; }
    std::vector<uint16_t> headOutputs() const {
        std::vector<uint16_t> r(lanes * rows * V);
        for (size_t b = 0; b < lanes; ++b) for (size_t t = 0; t < rows; ++t)
            std::copy_n(output.as<uint16_t>() + (b * rows + t) * Out, V,
                        r.data() + (b * rows + t) * V);
        return r;
    }
};

struct Timing { double gpu, wall; };
Timing dispatch(GPU &gpu, id<MTLComputePipelineState> pipeline, const Variant &variant,
                const Inputs &input, Results &result, bool headZeroOnly, bool audit,
                id<MTLComputePipelineState> preparation = nil, bool resetFromGPUSeed = false) {
    const unsigned threads = variant.groups * 32;
    require(pipeline.maxTotalThreadsPerThreadgroup >= threads, "pipeline thread limit");
    require(pipeline.staticThreadgroupMemoryLength <= gpu.device.maxThreadgroupMemoryLength,
            "pipeline threadgroup memory exceeds device limit");
    require(pipeline.threadExecutionWidth==32,"candidate/native SIMD width");
    const unsigned headLimit = headZeroOnly ? 1 : Heads;
    id<MTLComputePipelineState> snapshot = nil, restore = nil, replay = nil;
    if (preparation) {
        snapshot = gpu.pipeline("private_gdn_wy_snapshot");
        restore = gpu.pipeline("private_gdn_wy_restore");
        replay = gpu.pipeline(audit ? "private_gdn_wy_native_fallback_audit" : "private_gdn_wy_native_fallback");
        for (id<MTLComputePipelineState> phase : {snapshot,restore,replay}) {
            require(phase.maxTotalThreadsPerThreadgroup >= (phase == replay ? 512u : 256u),
                    "guard/replay pipeline thread limit");
            require(phase.staticThreadgroupMemoryLength <= gpu.device.maxThreadgroupMemoryLength,
                    "guard/replay threadgroup memory limit");
            require(phase.threadExecutionWidth == 32,"native SIMD reduction width");
        }
    }
    const auto start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [gpu.queue commandBuffer];
    command.label = ns(variant.name);
    if (resetFromGPUSeed) {
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        [blit copyFromBuffer:result.seed.allocation sourceOffset:Guard
                   toBuffer:result.state.allocation destinationOffset:Guard size:result.state.bytes];
        [blit endEncoding];
    }
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (preparation) {
        require(result.preparedTime == variant.time, "prepared coefficient geometry mismatch");
        require(preparation.maxTotalThreadsPerThreadgroup >= threads, "preparation pipeline thread limit");
        require(preparation.staticThreadgroupMemoryLength <= gpu.device.maxThreadgroupMemoryLength,
                "preparation threadgroup memory exceeds device limit");
        [encoder setComputePipelineState:snapshot];
        const std::array<const Guarded *,5> save{{&result.state,&result.snapshot,&result.flags,&result.diagnostics,&input.params}};
        for (size_t i=0;i<save.size();++i) [encoder setBuffer:save[i]->allocation offset:Guard atIndex:i];
        [encoder dispatchThreadgroups:MTLSizeMake(headLimit,64,input.lanes)
               threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:preparation];
        [encoder setBuffer:input.mixed.allocation offset:Guard atIndex:0];
        [encoder setBuffer:input.decay.allocation offset:Guard atIndex:1];
        [encoder setBuffer:input.beta.allocation offset:Guard atIndex:2];
        [encoder setBuffer:result.coefficients.allocation offset:Guard atIndex:3];
        [encoder setBuffer:result.flags.allocation offset:Guard atIndex:4];
        [encoder setBuffer:result.diagnostics.allocation offset:Guard atIndex:5];
        [encoder setBuffer:input.params.allocation offset:Guard atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(Heads, (input.rows + variant.time - 1) / variant.time, input.lanes)
               threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        // The state phase reads coefficients produced by the preceding
        // dispatch. Both phases remain inside the measured command buffer.
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [encoder setComputePipelineState:pipeline];
    const std::array<const Guarded *, 6> buffers{{&input.mixed, &input.decay, &input.beta,
        &result.state, &result.output, &result.diagnostics}};
    for (size_t i = 0; i < buffers.size(); ++i)
        [encoder setBuffer:buffers[i]->allocation offset:Guard atIndex:i];
    if (preparation) {
        [encoder setBuffer:result.coefficients.allocation offset:Guard atIndex:6];
        [encoder setBuffer:result.flags.allocation offset:Guard atIndex:7];
        [encoder setBuffer:input.params.allocation offset:Guard atIndex:audit ? 11 : 8];
        if (audit) {
            [encoder setBuffer:result.history.allocation offset:Guard atIndex:8];
            [encoder setBuffer:result.delta.allocation offset:Guard atIndex:9];
            [encoder setBuffer:result.preOutput.allocation offset:Guard atIndex:10];
        }
    } else {
        [encoder setBuffer:input.params.allocation offset:Guard atIndex:6];
        if (audit) {
            [encoder setBuffer:result.history.allocation offset:Guard atIndex:7];
            [encoder setBuffer:result.delta.allocation offset:Guard atIndex:8];
            [encoder setBuffer:result.preOutput.allocation offset:Guard atIndex:9];
        }
    }
    [encoder dispatchThreadgroups:MTLSizeMake(headLimit, V / variant.values, input.lanes)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    if (preparation) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:restore];
        const std::array<const Guarded *,5> restoreBuffers{{&result.snapshot,&result.state,&result.flags,&result.diagnostics,&input.params}};
        for (size_t i=0;i<restoreBuffers.size();++i)
            [encoder setBuffer:restoreBuffers[i]->allocation offset:Guard atIndex:i];
        [encoder dispatchThreadgroups:MTLSizeMake(headLimit,64,input.lanes)
               threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:replay];
        for (size_t i=0;i<buffers.size();++i) [encoder setBuffer:buffers[i]->allocation offset:Guard atIndex:i];
        [encoder setBuffer:result.flags.allocation offset:Guard atIndex:6];
        [encoder setBuffer:input.params.allocation offset:Guard atIndex:audit ? 10 : 7];
        if (audit) {
            [encoder setBuffer:result.history.allocation offset:Guard atIndex:7];
            [encoder setBuffer:result.delta.allocation offset:Guard atIndex:8];
            [encoder setBuffer:result.preOutput.allocation offset:Guard atIndex:9];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(headLimit,8,input.lanes)
               threadsPerThreadgroup:MTLSizeMake(512,1,1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "command failed: " + str(command.error.localizedDescription));
    Timing t{command.GPUEndTime - command.GPUStartTime,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};
    require(std::isfinite(t.gpu) && t.gpu > 0 && t.gpu < 600 &&
            std::isfinite(t.wall) && t.wall > 0 && t.wall < 600, "invalid command timing");
    return t;
}

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

struct CoefficientQuality {
    std::array<Error,5> fields;
    size_t padded = 0, nonzeroPadded = 0;
    bool pass() const {
        return std::all_of(fields.begin(),fields.end(),[](const Error &e) {return e.pass();}) && !nonzeroPadded;
    }
    NSDictionary *json() const {
        return @{@"W_f32":fields[0].json(), @"U_f32":fields[1].json(),
            @"end_key_f32":fields[2].json(), @"score_f32":fields[3].json(),
            @"prefix_f32":fields[4].json(), @"padded_slots":@(padded),
            @"nonzero_padded_slots":@(nonzeroPadded), @"pass":@(pass())};
    }
};
CoefficientQuality coefficientQuality(const std::vector<Fixture> &fixtures, const Results &r,
                                     bool onlyRequired = false, size_t headLimit = Heads) {
    require(fixtures.size() == 1 || fixtures.size() == Heads, "coefficient truth head count");
    std::vector<gdn_nax_cpu::Prepared<double>> truths;
    for (const auto &fixture : fixtures)
        truths.push_back(gdn_nax_cpu::prepare<double>(fixture,r.preparedTime));
    const auto &truth = truths.front();
    const size_t time=r.preparedTime, truthStride=truth.stride(),stride=r.coefficientStride;
    const std::array<size_t,5> offsets{{0,time*K,2*time*K,3*time*K,3*time*K+time*time}};
    const std::array<size_t,5> lengths{{time*K,time*K,time*K,time*time,time}};
    CoefficientQuality q;
    for (size_t field=0;field<5;++field) {
        std::vector<double> ref;
        std::vector<size_t> locations;
        for (size_t batch=0;batch<r.lanes;++batch) for (size_t chunk=0;chunk<truth.num_chunks;++chunk)
            for (size_t head=0;head<(onlyRequired?headLimit:Heads);++head) {
                if (onlyRequired && r.flags.as<uint32_t>()[batch*Heads+head]) continue;
            for (size_t i=0;i<lengths[field];++i) {
                ref.push_back(truths[fixtures.size()==1?0:head].storage[chunk*truthStride+offsets[field]+i]);
                const size_t location=((batch*truth.num_chunks+chunk)*Heads+head)*stride+offsets[field]+i;
                locations.push_back(location);
                const size_t token=field==4?i:(field==3?i/time:i/K);
                const size_t previous=field==3?i%time:0;
                const bool pad=token>=truth.active_rows(chunk) || (field==3 && previous>=truth.active_rows(chunk));
                if (!pad) continue;
                ++q.padded;
                q.nonzeroPadded+=r.coefficients.as<float>()[location]!=0.0f;
            }
            }
        if (!ref.empty()) q.fields[field]=errors(ref.size(),ref,1,[&](size_t i) {
            return double(r.coefficients.as<float>()[locations[i]]);
        });
    }
    return q;
}
CoefficientQuality coefficientQuality(const Fixture &f, const Results &r) {
    return coefficientQuality(std::vector<Fixture>{f},r);
}

template<typename Required,typename Read> Error requiredErrors(size_t count,
    const std::vector<double> &truth, Required required, Read read) {
    std::vector<double> reference;
    std::vector<size_t> locations;
    for (size_t i=0;i<count;++i) if (required(i)) {
        reference.push_back(truth[i%truth.size()]); locations.push_back(i);
    }
    if (reference.empty()) return {};
    return errors(reference.size(),reference,1,[&](size_t i) {return read(locations[i]);});
}

NSDictionary *flagMetrics(const Results &r,size_t headLimit) {
    NSMutableArray *reasons=[NSMutableArray array];
    size_t flagged=0,range=0,cancel=0,nonfinite=0,norm=0;
    for (size_t b=0;b<r.lanes;++b) for (size_t h=0;h<headLimit;++h) {
        const uint32_t value=r.flags.as<uint32_t>()[b*Heads+h];
        flagged+=value!=0; range+=(value&1)!=0; cancel+=(value&2)!=0;
        nonfinite+=(value&4)!=0; norm+=(value&8)!=0;
        [reasons addObject:@{@"lane":@(b),@"head":@(h),@"reason_bits":@(value)}];
    }
    return @{@"flagged_heads":@(flagged),@"unflagged_heads":@(r.lanes*headLimit-flagged),
        @"range_heads":@(range),@"cancellation_heads":@(cancel),@"nonfinite_heads":@(nonfinite),
        @"norm_range_heads":@(norm),@"reasons":reasons};
}

size_t differingBytes(const void *a,const void *b,size_t bytes) {
    const auto *x=static_cast<const unsigned char *>(a),*y=static_cast<const unsigned char *>(b);
    size_t count=0;
    for (size_t i=0;i<bytes;++i) count+=x[i]!=y[i];
    return count;
}
struct NativeEquivalence {
    size_t heads=0,state=0,output=0,history=0,delta=0,preOutput=0;
    bool pass() const {return !state && !output && !history && !delta && !preOutput;}
    NSDictionary *json() const {
        return @{@"flagged_heads_checked":@(heads),@"state_mismatch_bytes":@(state),
            @"output_mismatch_bytes":@(output),@"history_mismatch_bytes":@(history),
            @"delta_mismatch_bytes":@(delta),@"preoutput_mismatch_bytes":@(preOutput),
            @"byte_equal":@(pass()),@"same_incoming_f32_seed":@YES};
    }
};
NativeEquivalence nativeEquivalence(const Results &r,const Results &native,size_t headLimit,bool audit) {
    NativeEquivalence match;
    for (size_t b=0;b<r.lanes;++b) for (size_t h=0;h<headLimit;++h) {
        require(std::memcmp(r.snapshot.as<float>()+b*State+h*V*K,
                            r.seed.as<float>()+b*State+h*V*K,V*K*4)==0,"incoming snapshot changed");
        if (!r.flags.as<uint32_t>()[b*Heads+h]) continue;
        ++match.heads;
        match.state+=differingBytes(r.state.as<float>()+b*State+h*V*K,
                                   native.state.as<float>()+b*State+h*V*K,V*K*4);
        for (size_t t=0;t<r.rows;++t)
            match.output+=differingBytes(r.output.as<uint16_t>()+(b*r.rows+t)*Out+h*V,
                                        native.output.as<uint16_t>()+(b*r.rows+t)*Out+h*V,V*2);
        if (audit) {
            require(h==0 && r.audit && native.audit,"compact native audit scope");
            match.history+=differingBytes(r.history.as<float>()+b*r.rows*V*K,
                                         native.history.as<float>()+b*r.rows*V*K,r.rows*V*K*4);
            match.delta+=differingBytes(r.delta.as<float>()+b*r.rows*V,
                                       native.delta.as<float>()+b*r.rows*V,r.rows*V*4);
            match.preOutput+=differingBytes(r.preOutput.as<float>()+b*r.rows*V,
                                           native.preOutput.as<float>()+b*r.rows*V,r.rows*V*4);
        }
    }
    return match;
}

NSDictionary *transformMetrics(const CoefficientQuality &raw,const Results &r,size_t headLimit) {
    size_t required=0;
    for (size_t b=0;b<r.lanes;++b) for (size_t h=0;h<headLimit;++h)
        required+=r.flags.as<uint32_t>()[b*Heads+h]==0;
    NSMutableDictionary *metrics=[raw.json() mutableCopy];
    metrics[@"raw_pass"]=@(raw.pass()); metrics[@"required"]=@(required>0);
    metrics[@"required_heads"]=@(required);
    metrics[@"unused_heads"]=@(r.lanes*headLimit-required);
    metrics[@"unused_reason"]=@"Flagged heads use fullRows native replay; their F32 transforms are unused. Raw transform metrics and failed gates remain reported.";
    return metrics;
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

std::pair<bool,std::vector<float>> qualifyTimedHeads(GPU &gpu,
    const std::vector<Fixture> &heads, const HeadTruth &truth, const std::vector<float> &seed,
    const Variant &variant, id<MTLComputePipelineState> pipeline,
    id<MTLComputePipelineState> preparation, const char *stage, unsigned lanes,
    const std::vector<uint32_t> *expectedFlags = nullptr) {
    Inputs input(gpu,heads,lanes);
    Results result(gpu,heads.front().rows,lanes,false,variant.time);
    Results native(gpu,heads.front().rows,lanes,false);
    result.reset(seed);
    native.reset(seed);
    // A missed timed-kernel fragment must remain nonfinite and fail numerics.
    std::fill_n(result.output.as<uint16_t>(),result.output.bytes/2,uint16_t(0x7fc0));
    dispatch(gpu,pipeline,variant,input,result,false,false,preparation);
    dispatch(gpu,gpu.pipeline(Canonical.name),Canonical,input,native,false,false);
    input.check(); result.check(false,seed); native.check(false,seed); result.seed.immutable();
    const Error state = errors(lanes*State,truth.state,lanes,
        [&](size_t i) {return double(result.state.as<float>()[i]);});
    const Error output = errors(lanes*truth.output.size(),truth.output,lanes,
        [&](size_t i) {return double(from_bf16(result.output.as<uint16_t>()[i]));});
    size_t mismatchCPU = 0;
    for (size_t i = 0; i < lanes*truth.output.size(); ++i)
        mismatchCPU += result.output.as<uint16_t>()[i] != to_bf16(float(truth.output[i%truth.output.size()]));
    const auto coefficients = coefficientQuality(heads,result);
    const auto requiredCoefficients = coefficientQuality(heads,result,true);
    const auto match = nativeEquivalence(result,native,Heads,false);
    const auto stateRequired=[&](size_t i) {
        return result.flags.as<uint32_t>()[(i/State)*Heads+(i%State)/(V*K)]==0;
    };
    const auto outputRequired=[&](size_t i) {
        return result.flags.as<uint32_t>()[(i/truth.output.size())*Heads+(i%Out)/V]==0;
    };
    const Error requiredState=requiredErrors(lanes*State,truth.state,stateRequired,
        [&](size_t i) {return double(result.state.as<float>()[i]);});
    const Error requiredOutput=requiredErrors(lanes*truth.output.size(),truth.output,outputRequired,
        [&](size_t i) {return double(from_bf16(result.output.as<uint16_t>()[i]));});
    const Error nativeState=errors(lanes*State,truth.state,lanes,
        [&](size_t i) {return double(native.state.as<float>()[i]);});
    const uint32_t diagnostics = *result.diagnostics.as<uint32_t>();
    const bool strict=state.pass() && bf16Pass(output) && coefficients.pass() && nativeState.pass() &&
        !diagnostics && !*native.diagnostics.as<uint32_t>();
    bool selectorPass=true;
    if (expectedFlags) for (size_t b=0;b<lanes;++b) for (size_t h=0;h<Heads;++h) {
        const auto actual=result.flags.as<uint32_t>()[b*Heads+h],expected=expectedFlags->at(h);
        selectorPass=selectorPass && (expected ? (actual&expected)==expected : actual==0);
    }
    strictF64Quality=strictF64Quality && strict;
    const bool pass=selectorPass && requiredState.pass() && bf16Pass(requiredOutput) && requiredCoefficients.pass() &&
        match.pass() && !diagnostics && !*native.diagnostics.as<uint32_t>();
    emit(@{@"kind":@"quality_timed_all_heads", @"variant":ns(variant.name),
        @"stage":ns(stage), @"rows":@(input.rows), @"lanes":@(lanes), @"heads":@(Heads),
        @"qualification_pass":@(pass), @"required":@YES,
        @"strict_f64_qualification_pass":@(strict),
        @"selector_expectation_pass":@(selectorPass), @"selector_expectations_required":@(expectedFlags!=nullptr),
        @"qualification_policy":@"unflagged strict math; flagged fullRows frozen-native byte equivalence",
        @"carried_state_f32":state.json(), @"output_bf16_vs_f64":bf16Metrics(output),
        @"canonical_carried_state_f32":nativeState.json(),
        @"unflagged_carried_state_f32":requiredState.json(),
        @"unflagged_output_bf16_vs_f64":bf16Metrics(requiredOutput),
        @"bf16_mismatch_cpu_float_round_rate":@(double(mismatchCPU)/(lanes*truth.output.size())),
        @"prepared_coefficients":transformMetrics(coefficients,result,Heads),
        @"required_transform_metrics":requiredCoefficients.json(),
        @"native_replay_equivalence":match.json(), @"eligibility":flagMetrics(result,Heads),
        @"candidate_diagnostics":@(diagnostics),
        @"canaries_pass":@YES, @"immutable_sha256_pass":@YES,
        @"input_sha256":ns(input.mixed.frozenHash), @"timed_pipeline_checked":@YES,
        @"heterogeneous_key_heads":@16, @"heterogeneous_value_heads":@48,
        @"lane_fixture_policy":@"identical lanes; every lane checked",
        @"reference":@"independent F64 serial recurrence; prior F64 state for continuation"});
    return {pass,{result.state.as<float>(),result.state.as<float>()+State}};
}

bool rangeTrap(const std::string &name) {
    return name.find("range") != std::string::npos || name.find("trap") != std::string::npos;
}
struct Carries { std::vector<float> candidate, canonical; };
std::pair<bool, Carries> qualifyStage(GPU &gpu, const Fixture &f, const Variant &variant,
    id<MTLComputePipelineState> candidatePipeline, id<MTLComputePipelineState> canonicalPipeline,
    id<MTLComputePipelineState> preparation,
    const std::vector<float> &candidateSeed, const std::vector<float> &canonicalSeed,
    const Trace<double> &truth, const char *stage, bool ordinary, unsigned lanes) {
    Inputs input(gpu, f, lanes);
    Results candidate(gpu, f.rows, lanes, true, variant.time), canonical(gpu, f.rows, lanes, true);
    require(canonicalSeed.size()==candidateSeed.size(),"native carry dimensions");
    candidate.reset(candidateSeed); canonical.reset(candidateSeed);
    dispatch(gpu, canonicalPipeline, Canonical, input, canonical, true, true);
    dispatch(gpu, candidatePipeline, variant, input, candidate, true, true, preparation);
    input.check(); candidate.check(true, candidateSeed); canonical.check(true, candidateSeed);
    candidate.seed.immutable();
    const Error history = errors(lanes * f.rows * V * K, truth.history, lanes,
        [&](size_t i) { return double(candidate.history.as<float>()[i]); });
    const Error delta = errors(lanes * f.rows * V, truth.delta, lanes,
        [&](size_t i) { return double(candidate.delta.as<float>()[i]); });
    const Error preOutput = errors(lanes * f.rows * V, truth.out, lanes,
        [&](size_t i) { return double(candidate.preOutput.as<float>()[i]); });
    const Error state = errors(lanes * V * K, truth.final_state, lanes,
        [&](size_t i) { return double(candidate.state.as<float>()[(i / (V * K)) * State + i % (V * K)]); });
    const Error nativeState = errors(lanes * V * K, truth.final_state, lanes,
        [&](size_t i) { return double(canonical.state.as<float>()[(i / (V * K)) * State + i % (V * K)]); });
    const auto candidateBF16 = candidate.headOutputs(), nativeBF16 = canonical.headOutputs();
    const Error candidateOutput = errors(candidateBF16.size(), truth.out, lanes,
        [&](size_t i) { return double(from_bf16(candidateBF16[i])); });
    const Error nativeOutput = errors(nativeBF16.size(), truth.out, lanes,
        [&](size_t i) { return double(from_bf16(nativeBF16[i])); });
    size_t mismatchNative = 0, mismatchCPU = 0, mismatchAudit = 0;
    for (size_t i = 0; i < candidateBF16.size(); ++i) {
        mismatchNative += candidateBF16[i] != nativeBF16[i];
        mismatchCPU += candidateBF16[i] != to_bf16(float(truth.out[i % truth.out.size()]));
        mismatchAudit += candidateBF16[i] != to_bf16(candidate.preOutput.as<float>()[i]);
    }
    const uint32_t candidateDiagnostics = *candidate.diagnostics.as<uint32_t>();
    const uint32_t nativeDiagnostics = *canonical.diagnostics.as<uint32_t>();
    const auto coefficientMetrics = coefficientQuality(f, candidate);
    const auto requiredCoefficients = coefficientQuality(std::vector<Fixture>{f},candidate,true,1);
    const auto match = nativeEquivalence(candidate,canonical,1,true);
    const auto neededHistory=[&](size_t i) {return !candidate.flags.as<uint32_t>()[(i/truth.history.size())*Heads];};
    const auto neededDelta=[&](size_t i) {return !candidate.flags.as<uint32_t>()[(i/truth.delta.size())*Heads];};
    const auto neededOutput=[&](size_t i) {return !candidate.flags.as<uint32_t>()[(i/truth.out.size())*Heads];};
    const auto neededState=[&](size_t i) {return !candidate.flags.as<uint32_t>()[(i/truth.final_state.size())*Heads];};
    const Error requiredHistory=requiredErrors(lanes*truth.history.size(),truth.history,neededHistory,
        [&](size_t i) {return double(candidate.history.as<float>()[i]);});
    const Error requiredDelta=requiredErrors(lanes*truth.delta.size(),truth.delta,neededDelta,
        [&](size_t i) {return double(candidate.delta.as<float>()[i]);});
    const Error requiredOutput=requiredErrors(lanes*truth.out.size(),truth.out,neededOutput,
        [&](size_t i) {return double(candidate.preOutput.as<float>()[i]);});
    const Error requiredState=requiredErrors(lanes*truth.final_state.size(),truth.final_state,neededState,
        [&](size_t i) {return double(candidate.state.as<float>()[(i/(V*K))*State+i%(V*K)]);});
    const Error requiredNativeState=requiredErrors(lanes*truth.final_state.size(),truth.final_state,neededState,
        [&](size_t i) {return double(canonical.state.as<float>()[(i/(V*K))*State+i%(V*K)]);});
    const Error nativeHistory=errors(lanes*truth.history.size(),truth.history,lanes,
        [&](size_t i) {return double(canonical.history.as<float>()[i]);});
    const Error nativeDelta=errors(lanes*truth.delta.size(),truth.delta,lanes,
        [&](size_t i) {return double(canonical.delta.as<float>()[i]);});
    const Error nativePreOutput=errors(lanes*truth.out.size(),truth.out,lanes,
        [&](size_t i) {return double(canonical.preOutput.as<float>()[i]);});
    // BF16-vs-F64 errors/mismatches are reported, not judged with an F32
    // tolerance. The exposed F32 intermediates and carried states qualify.
    const bool strict = history.pass() && delta.pass() && preOutput.pass() && state.pass() &&
                      nativeState.pass() && coefficientMetrics.pass() &&
                      nativeHistory.pass() && nativeDelta.pass() && nativePreOutput.pass() &&
                      !candidateDiagnostics && !nativeDiagnostics && !mismatchAudit;
    const bool pass=requiredHistory.pass() && requiredDelta.pass() && requiredOutput.pass() &&
        requiredState.pass() && requiredNativeState.pass() && requiredCoefficients.pass() && match.pass() &&
        !candidateDiagnostics && !nativeDiagnostics && !mismatchAudit;
    strictF64Quality=strictF64Quality && strict;
    emit(@{@"kind": @"quality", @"fixture": ns(f.name), @"stage": ns(stage),
           @"variant": ns(variant.name), @"rows": @(f.rows), @"lanes": @(lanes),
           @"classification": ordinary ? @"ordinary" : @"range_trap",
           @"qualification_pass": @(pass), @"required":@YES,
           @"strict_f64_qualification_pass":@(strict), @"original_ordinary_classification":@(ordinary),
           @"qualification_policy":@"unflagged strict math; flagged fullRows frozen-native byte equivalence",
           @"history_f32": history.json(), @"delta_f32": delta.json(),
           @"pre_bf16_output_f32": preOutput.json(), @"carried_state_f32": state.json(),
           @"canonical_carried_state_f32": nativeState.json(),
           @"canonical_history_f32":nativeHistory.json(), @"canonical_delta_f32":nativeDelta.json(),
           @"canonical_pre_bf16_output_f32":nativePreOutput.json(),
           @"unflagged_history_f32":requiredHistory.json(), @"unflagged_delta_f32":requiredDelta.json(),
           @"unflagged_pre_bf16_output_f32":requiredOutput.json(), @"unflagged_carried_state_f32":requiredState.json(),
           @"unflagged_canonical_carried_state_f32":requiredNativeState.json(),
           @"prepared_coefficients":transformMetrics(coefficientMetrics,candidate,1),
           @"required_transform_metrics":requiredCoefficients.json(),
           @"native_replay_equivalence":match.json(), @"eligibility":flagMetrics(candidate,1),
           @"output_bf16_vs_f64": candidateOutput.json(),
           @"canonical_output_bf16_vs_f64": nativeOutput.json(),
           @"bf16_mismatch_canonical_rate": @(double(mismatchNative) / candidateBF16.size()),
           @"bf16_mismatch_cpu_rate": @(double(mismatchCPU) / candidateBF16.size()),
           @"bf16_mismatch_own_f32_rate": @(double(mismatchAudit) / candidateBF16.size()),
           @"candidate_diagnostics": @(candidateDiagnostics), @"canonical_diagnostics": @(nativeDiagnostics),
           @"canaries_pass": @YES, @"immutable_sha256_pass": @YES,
           @"input_sha256": ns(input.mixed.frozenHash)});
    return {pass, {candidate.carried(), canonical.carried()}};
}

void resources(GPU &gpu) {
    for (const auto &[name,threads] : std::array<std::pair<const char *,unsigned>,5>{{
        {"private_gdn_wy_snapshot",256},{"private_gdn_wy_restore",256},
        {"private_gdn_wy_native_fallback",512},{"private_gdn_wy_native_fallback_audit",512},
        {NativeAuditName,512}}}) {
        auto phase=gpu.pipeline(name);
        require(phase.maxTotalThreadsPerThreadgroup>=threads,"guard/native thread limit");
        require(phase.threadExecutionWidth==32,"guard/native SIMD width");
        require(phase.staticThreadgroupMemoryLength<=gpu.device.maxThreadgroupMemoryLength,"guard/native memory limit");
        emit(@{@"kind":@"resources",@"name":ns(name),@"threadgroup_bytes":@(phase.staticThreadgroupMemoryLength),
            @"max_threads":@(phase.maxTotalThreadsPerThreadgroup),@"execution_width":@(phase.threadExecutionWidth)});
    }
    for (const auto &variant : Variants) {
        const std::string name=preparationName(variant);
        auto pipeline=gpu.pipeline(name);
        require(pipeline.maxTotalThreadsPerThreadgroup>=variant.groups*32,"preparation thread limit");
        require(pipeline.staticThreadgroupMemoryLength<=gpu.device.maxThreadgroupMemoryLength,"preparation memory limit");
        emit(@{@"kind":@"resources", @"name":ns(name),
            @"threadgroup_bytes":@(pipeline.staticThreadgroupMemoryLength),
            @"max_threads":@(pipeline.maxTotalThreadsPerThreadgroup),
            @"execution_width":@(pipeline.threadExecutionWidth)});
    }
    for (const auto &variant : Variants) {
        for (bool audit : {false, true}) {
            const std::string name = audit ? auditName(variant) : variant.name;
            auto pipeline = gpu.pipeline(name);
            require(pipeline.maxTotalThreadsPerThreadgroup>=variant.groups*32,"apply thread limit");
            require(pipeline.staticThreadgroupMemoryLength<=gpu.device.maxThreadgroupMemoryLength,"apply memory limit");
            emit(@{@"kind": @"resources", @"name": ns(name),
                   @"threadgroup_bytes": @(pipeline.staticThreadgroupMemoryLength),
                   @"max_threads": @(pipeline.maxTotalThreadsPerThreadgroup),
                   @"execution_width": @(pipeline.threadExecutionWidth)});
        }
    }
    auto pipeline = gpu.pipeline(Canonical.name);
    require(pipeline.maxTotalThreadsPerThreadgroup>=Canonical.groups*32,"canonical thread limit");
    require(pipeline.staticThreadgroupMemoryLength<=gpu.device.maxThreadgroupMemoryLength,"canonical memory limit");
    emit(@{@"kind": @"resources", @"name": ns(Canonical.name),
           @"threadgroup_bytes": @(pipeline.staticThreadgroupMemoryLength),
           @"max_threads": @(pipeline.maxTotalThreadsPerThreadgroup),
           @"execution_width": @(pipeline.threadExecutionWidth)});
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

void dispatchNormReference(GPU &gpu,const Inputs &input,Results &control,const Variant &variant) {
    const std::string name="private_gdn_wy_norm_reference_t"+std::to_string(variant.time)+
        "_sg"+std::to_string(variant.groups);
    auto pipeline=gpu.pipeline(name);
    require(control.coefficientStride==oldPreparedStride(variant.time),"norm reference uses v3 coefficients");
    require(pipeline.maxTotalThreadsPerThreadgroup>=variant.groups*32,"norm reference thread limit");
    require(pipeline.threadExecutionWidth==32,"norm reference SIMD width");
    id<MTLCommandBuffer> command=[gpu.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    const std::array<const Guarded *,4> buffers{{&control.coefficients,&control.normReference,&control.diagnostics,&input.params}};
    for (size_t i=0;i<buffers.size();++i) [encoder setBuffer:buffers[i]->allocation offset:Guard atIndex:i];
    [encoder dispatchThreadgroups:MTLSizeMake(Heads,(input.rows+variant.time-1)/variant.time,input.lanes)
           threadsPerThreadgroup:MTLSizeMake(variant.groups*32,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"norm reference command failed: "+str(command.error.localizedDescription));
}

struct GuardComparison {
    bool equivalent=false,policy=false,strict=false;
    std::vector<float> v3Carry,v4Carry;
};

GuardComparison compareGuardStage(GPU &gpu,const std::vector<Fixture> &heads,
    const HeadTruth &truth,const std::vector<double> *priorTruth,
    const std::vector<float> &v3Seed,const std::vector<float> &v4Seed,
    const Variant &variant,bool audit,const char *fixture,const char *stage) {
    const unsigned lanes=unsigned(v3Seed.size()/State);
    require(lanes>0 && v3Seed.size()==v4Seed.size(),"equivalence seed dimensions");
    Inputs input(gpu,heads,lanes);
    Results control(gpu,input.rows,lanes,audit,variant.time,true),candidate(gpu,input.rows,lanes,audit,variant.time);
    // Fixtures may repeat by lane; each lane has its own carried-state tape.
    auto resetTape=[&](Results &result,const std::vector<float> &seed) {
        result.reset({seed.begin(),seed.begin()+State});
        std::memcpy(result.state.data(),seed.data(),seed.size()*4);
        std::memcpy(result.seed.data(),seed.data(),seed.size()*4); result.seed.freeze();
        if (!audit) std::fill_n(result.output.as<uint16_t>(),result.output.bytes/2,uint16_t(0x7fc0));
    };
    resetTape(control,v3Seed); resetTape(candidate,v4Seed);
    dispatch(gpu,gpu.pipeline(v3ApplyName(variant,audit)),variant,input,control,audit,audit,
             gpu.pipeline(v3PreparationName(variant)));
    dispatch(gpu,gpu.pipeline(audit?auditName(variant):variant.name),variant,input,candidate,audit,audit,
             gpu.pipeline(preparationName(variant)));
    dispatchNormReference(gpu,input,control,variant);
    input.check(); control.seed.immutable(); candidate.seed.immutable();
    control.check(audit,{v3Seed.begin(),v3Seed.begin()+State});
    candidate.check(audit,{v4Seed.begin(),v4Seed.begin()+State});
    const size_t records=size_t(lanes)*((input.rows+variant.time-1)/variant.time)*Heads;
    size_t coefficientBytes=0,normBytes=0,normFlagBytes=0;
    for (size_t record=0;record<records;++record) {
        const auto *old=control.coefficients.as<float>()+record*control.coefficientStride;
        const auto *fresh=candidate.coefficients.as<float>()+record*candidate.coefficientStride;
        coefficientBytes+=differingBytes(old,fresh,oldPreparedStride(variant.time)*4);
        const auto *reference=control.normReference.as<float>()+record*2*variant.time;
        normBytes+=differingBytes(fresh+oldPreparedStride(variant.time),reference,variant.time*4);
        normFlagBytes+=differingBytes(fresh+oldPreparedStride(variant.time)+variant.time,
                                    reference+variant.time,variant.time*4);
    }
    const size_t flags=differingBytes(control.flags.data(),candidate.flags.data(),candidate.flags.bytes);
    const size_t snapshots=differingBytes(control.snapshot.data(),candidate.snapshot.data(),candidate.snapshot.bytes);
    const size_t states=differingBytes(control.state.data(),candidate.state.data(),candidate.state.bytes);
    const size_t outputs=differingBytes(control.output.data(),candidate.output.data(),candidate.output.bytes);
    const size_t histories=audit?differingBytes(control.history.data(),candidate.history.data(),candidate.history.bytes):0;
    const size_t deltas=audit?differingBytes(control.delta.data(),candidate.delta.data(),candidate.delta.bytes):0;
    const size_t preoutputs=audit?differingBytes(control.preOutput.data(),candidate.preOutput.data(),candidate.preOutput.bytes):0;
    const size_t diagnostics=differingBytes(control.diagnostics.data(),candidate.diagnostics.data(),4);
    const bool equivalent=!flags && !snapshots && !states && !outputs && !histories && !deltas &&
        !preoutputs && !diagnostics && !coefficientBytes && !normBytes && !normFlagBytes;
    const size_t headLimit=audit?1:Heads;
    const auto rawCoefficients=coefficientQuality(heads,candidate);
    const auto requiredCoefficients=coefficientQuality(heads,candidate,true,headLimit);
    const auto neededState=[&](size_t i) {return !candidate.flags.as<uint32_t>()[(i/State)*Heads+(i%State)/(V*K)];};
    const auto neededOutput=[&](size_t i) {return !candidate.flags.as<uint32_t>()[(i/truth.output.size())*Heads+(i%Out)/V];};
    Error rawState,rawOutput,requiredState,requiredOutput,rawHistory,rawDelta,rawPreoutput;
    Error neededHistory,neededDelta,neededPreoutput;
    if (audit) {
        std::vector<double> prior;
        if (priorTruth) prior.assign(priorTruth->begin(),priorTruth->begin()+V*K);
        const auto trace=serial<double>(heads.front(),priorTruth?&prior:nullptr);
        rawState=errors(lanes*trace.final_state.size(),trace.final_state,lanes,
            [&](size_t i) {return double(candidate.state.as<float>()[(i/(V*K))*State+i%(V*K)]);});
        rawHistory=errors(lanes*trace.history.size(),trace.history,lanes,[&](size_t i) {return double(candidate.history.as<float>()[i]);});
        rawDelta=errors(lanes*trace.delta.size(),trace.delta,lanes,[&](size_t i) {return double(candidate.delta.as<float>()[i]);});
        rawPreoutput=errors(lanes*trace.out.size(),trace.out,lanes,[&](size_t i) {return double(candidate.preOutput.as<float>()[i]);});
        auto needed=[&](size_t i,size_t perLane) {return !candidate.flags.as<uint32_t>()[(i/perLane)*Heads];};
        requiredState=requiredErrors(lanes*trace.final_state.size(),trace.final_state,
            [&](size_t i) {return needed(i,trace.final_state.size());},
            [&](size_t i) {return double(candidate.state.as<float>()[(i/(V*K))*State+i%(V*K)]);});
        neededHistory=requiredErrors(lanes*trace.history.size(),trace.history,
            [&](size_t i) {return needed(i,trace.history.size());},[&](size_t i) {return double(candidate.history.as<float>()[i]);});
        neededDelta=requiredErrors(lanes*trace.delta.size(),trace.delta,
            [&](size_t i) {return needed(i,trace.delta.size());},[&](size_t i) {return double(candidate.delta.as<float>()[i]);});
        neededPreoutput=requiredErrors(lanes*trace.out.size(),trace.out,
            [&](size_t i) {return needed(i,trace.out.size());},[&](size_t i) {return double(candidate.preOutput.as<float>()[i]);});
        const auto bf16=candidate.headOutputs();
        rawOutput=errors(bf16.size(),trace.out,lanes,[&](size_t i) {return double(from_bf16(bf16[i]));});
    } else {
        rawState=errors(lanes*State,truth.state,lanes,[&](size_t i) {return double(candidate.state.as<float>()[i]);});
        rawOutput=errors(lanes*truth.output.size(),truth.output,lanes,[&](size_t i) {return double(from_bf16(candidate.output.as<uint16_t>()[i]));});
        requiredState=requiredErrors(lanes*State,truth.state,neededState,[&](size_t i) {return double(candidate.state.as<float>()[i]);});
        requiredOutput=requiredErrors(lanes*truth.output.size(),truth.output,neededOutput,
            [&](size_t i) {return double(from_bf16(candidate.output.as<uint16_t>()[i]));});
    }
    const bool strict=rawState.pass() && bf16Pass(rawOutput) && rawCoefficients.pass() &&
        (!audit || (rawHistory.pass() && rawDelta.pass() && rawPreoutput.pass())) &&
        !*candidate.diagnostics.as<uint32_t>() && !*control.diagnostics.as<uint32_t>();
    const bool policy=equivalent && requiredState.pass() && requiredCoefficients.pass() &&
        (!audit?bf16Pass(requiredOutput):(neededHistory.pass() && neededDelta.pass() && neededPreoutput.pass())) &&
        !*candidate.diagnostics.as<uint32_t>() && !*control.diagnostics.as<uint32_t>();
    emit(@{@"kind":@"guard_equivalence",@"fixture":ns(fixture),@"stage":ns(stage),
        @"variant":ns(variant.name),@"audit":@(audit),@"rows":@(input.rows),@"lanes":@(lanes),
        @"pass":@(equivalent),@"guarded_policy_pass":@(policy),@"strict_f64_qualification_pass":@(strict),
        @"flags_mismatch_bytes":@(flags),@"snapshot_mismatch_bytes":@(snapshots),@"state_mismatch_bytes":@(states),
        @"output_mismatch_bytes":@(outputs),@"history_mismatch_bytes":@(histories),@"delta_mismatch_bytes":@(deltas),
        @"preoutput_mismatch_bytes":@(preoutputs),@"diagnostics_mismatch_bytes":@(diagnostics),
        @"old_coefficients_mismatch_bytes":@(coefficientBytes),@"cached_norm_mismatch_bytes":@(normBytes),
        @"cached_norm_flag_mismatch_bytes":@(normFlagBytes),@"old_coefficient_records_checked":@(records),
        @"cached_norm_tokens_checked":@(records*variant.time),@"v3_eligibility":flagMetrics(control,headLimit),
        @"v4_eligibility":flagMetrics(candidate,headLimit),@"prepared_coefficients":transformMetrics(rawCoefficients,candidate,headLimit),
        @"required_transform_metrics":requiredCoefficients.json(),@"carried_state_f32":rawState.json(),
        @"output_bf16_vs_f64":bf16Metrics(rawOutput),@"history_f32":rawHistory.json(),@"delta_f32":rawDelta.json(),
        @"pre_bf16_output_f32":rawPreoutput.json(),@"canaries_pass":@YES,@"immutable_sha256_pass":@YES,
        @"reference":@"frozen v3 control; diagnostic norm reference uses exact v3 lane/reduction sequence"});
    return {equivalent,policy,strict,{control.state.as<float>(),control.state.as<float>()+lanes*State},
        {candidate.state.as<float>(),candidate.state.as<float>()+lanes*State}};
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

bool guardEquivalence(GPU &gpu,unsigned lanes) {
    struct Case {const char *name;std::vector<Fixture> heads;bool clearContinuation;};
    std::vector<Case> cases;
    cases.push_back({"heterogeneous_normal",heterogeneousFixtures(67,0),false});
    std::vector<Fixture> mixed;
    for (size_t h=0;h<Heads;++h) mixed.push_back(guardProofFixture(unsigned(h%4)));
    cases.push_back({"mixed_postzero_cancel_realzero_safe",std::move(mixed),false});
    cases.push_back({"norm_ftz_and_finite_saturation",normBoundaryFixtures(),true});
    for (const auto &f:gdn_chunk_cpu::make_fixtures()) if (f.name.find("cancellation")!=std::string::npos)
        cases.push_back({"original_cancellation_gate_retained",std::vector<Fixture>(Heads,f),false});
    bool pass=true,policy=true,strict=true;size_t stages=0;
    emit(@{@"kind":@"guard_equivalence_registration",@"cases":@(cases.size()),@"variants":@(Variants.size()),
        @"initial_rows":@67,@"continuation_rows":@19,@"full_heads_and_compact_audit":@YES,
        @"byte_equal_fields":@"flags, snapshot, state, BF16 output, audit history/delta/preoutput, old coefficient fields, cached norms and flags",
        @"threshold_inflation_domain_changed":@NO,@"norm_reference_cpu_reduction_assumed":@NO});
    for (const auto &variant:Variants) for (const auto &c:cases) for (bool audit:{false,true}) {
        const auto truth=heterogeneousTruth(c.heads);
        const auto singleSeed=heterogeneousSeeds(c.heads);
        std::vector<float> seed;
        for (unsigned lane=0;lane<lanes;++lane) seed.insert(seed.end(),singleSeed.begin(),singleSeed.end());
        auto first=compareGuardStage(gpu,c.heads,truth,nullptr,seed,seed,variant,audit,c.name,"initial");
        auto continuedHeads=heterogeneousFixtures(19,1);
        if (c.clearContinuation) for (auto &f:continuedHeads) f.alpha[0]=0;
        const auto continuedTruth=heterogeneousTruth(continuedHeads,&truth.state);
        auto continued=compareGuardStage(gpu,continuedHeads,continuedTruth,&truth.state,
            first.v3Carry,first.v4Carry,variant,audit,c.name,c.clearContinuation?"continuation_with_real_zero":"continuation");
        pass=pass && first.equivalent && continued.equivalent;
        policy=policy && first.policy && continued.policy;
        strict=strict && first.strict && continued.strict;stages+=2;
    }
    emit(@{@"kind":@"guard_equivalence_summary",@"pass":@(pass),@"guarded_policy_pass":@(policy),
        @"strict_f64_qualification_pass":@(strict),@"checked_stages":@(stages),
        @"norm_cache_equivalence_only":@YES,@"whole_history_accuracy_certificate":@NO});
    return pass && policy;
}

bool quality(GPU &gpu, unsigned lanes,bool proofOnly=false) {
    bool pass = true;
    strictF64Quality=true;
    emit(@{@"kind":@"quality_timed_all_heads_registration", @"rows":@67,
        @"continuation_rows":@19, @"heads":@48, @"key_heads":@16,
        @"f32_relative_rms_tolerance":@(RelativeTolerance),
        @"bf16_relative_rms_tolerance":@(BF16RelativeTolerance),
        @"bf16_max_abs_scale_tolerance":@(BF16AbsoluteScaleTolerance),
        @"bf16_max_abs_scale":@"max(1, F64 reference peak)",
        @"output_initialization":@"BF16 quiet NaN; unwritten fragments fail"});
    const auto initialHeads = heterogeneousFixtures(67,0);
    const auto continuedHeads = heterogeneousFixtures(19,1);
    const auto initialHeadTruth = proofOnly ? HeadTruth{} : heterogeneousTruth(initialHeads);
    const auto continuedHeadTruth = proofOnly ? HeadTruth{} : heterogeneousTruth(continuedHeads,&initialHeadTruth.state);
    const auto initialHeadSeeds = heterogeneousSeeds(initialHeads);
    auto native = gpu.pipeline(NativeAuditName);
    auto fixtures = proofOnly ? std::vector<Fixture>{} : gdn_chunk_cpu::make_fixtures();
    for (unsigned kind=0;kind<3;++kind) fixtures.push_back(guardProofFixture(kind));
    std::vector<Fixture> mixedHeads;
    std::vector<uint32_t> expectedFlags;
    for (size_t h=0;h<Heads;++h) {
        mixedHeads.push_back(guardProofFixture(unsigned(h%4)));
        expectedFlags.push_back(h%4==0?2u:(h%4==1?1u:0u));
    }
    const auto mixedTruth=heterogeneousTruth(mixedHeads);
    const auto mixedSeeds=heterogeneousSeeds(mixedHeads);
    emit(@{@"kind":@"native_replay_proof_registration",@"forced_by_actual_inputs":@YES,
        @"fixture_selector_hardcoding":@NO,@"mixed_head_pattern":@"cancel, range, real-zero, safe",
        @"required_comparison":@"flagged full-state/fullRows BF16 bytes; flagged audit history/delta/preoutput bytes",
        @"proof_mode_only":@(proofOnly)});
    for (const auto &variant : Variants) {
        auto candidate = gpu.pipeline(auditName(variant));
        auto preparation = gpu.pipeline(preparationName(variant));
        for (const auto &fixture : fixtures) {
            const bool ordinary = !rangeTrap(fixture.name);
            const auto truth = serial<double>(fixture);
            auto initial = qualifyStage(gpu, fixture, variant, candidate, native, preparation,
                fixture.initial_state, fixture.initial_state, truth, "initial", ordinary, lanes);
            pass = pass && initial.first;
            const auto continuation = gdn_chunk_cpu::continuation_fixture(19);
            const auto continuedTruth = serial<double>(continuation, &truth.final_state);
            auto continued = qualifyStage(gpu, continuation, variant, candidate, native, preparation,
                initial.second.candidate, initial.second.canonical, continuedTruth,
                ("continuation_after_" + fixture.name).c_str(), ordinary, lanes);
            pass = pass && continued.first;
        }
        auto timed = gpu.pipeline(variant.name);
        if (!proofOnly) {
            auto initial = qualifyTimedHeads(gpu,initialHeads,initialHeadTruth,initialHeadSeeds,
                variant,timed,preparation,"heterogeneous_tail_initial",lanes);
            auto continued = qualifyTimedHeads(gpu,continuedHeads,continuedHeadTruth,initial.second,
                variant,timed,preparation,"heterogeneous_carried_continuation",lanes);
            pass = pass && initial.first && continued.first;
        }
        auto mixed=qualifyTimedHeads(gpu,mixedHeads,mixedTruth,mixedSeeds,
            variant,timed,preparation,"mixed_guard_branch_proof",lanes,&expectedFlags);
        const auto mixedContinuation=heterogeneousTruth(continuedHeads,&mixedTruth.state);
        auto carried=qualifyTimedHeads(gpu,continuedHeads,mixedContinuation,mixed.second,
            variant,timed,preparation,"mixed_guard_carried_continuation",lanes);
        pass=pass && mixed.first && carried.first;
    }
    emit(@{@"kind": @"quality_summary", @"pass": @(pass),
           @"strict_f64_qualification_pass":@(strictF64Quality),
           @"guarded_native_or_strict_policy_pass":@(pass),
           @"fixtures": @(fixtures.size()), @"variants": @(Variants.size()),
           @"continuation_rows": @19, @"timed_all_head_stages":@((proofOnly?2:4)*Variants.size()),
           @"reference": @"BF16 inputs, independent F64 serial recurrence"});
    return pass;
}

double median(std::vector<double> data) {
    require(!data.empty(), "no timings");
    std::sort(data.begin(), data.end());
    return data[data.size() / 2];
}

void bench(GPU &gpu, unsigned lanes, unsigned rows) {
    // Fixture construction allocates inputs only: no rows*16384 CPU history.
    const Fixture fixture = gdn_chunk_cpu::continuation_fixture(rows);
    Inputs input(gpu, fixture, lanes);
    auto native = gpu.pipeline(Canonical.name);
    std::mt19937 random(0x47444e21u);
    for (const auto &variant : Variants) {
        Results result(gpu, rows, lanes, false, variant.time);
        auto candidate = gpu.pipeline(variant.name);
        auto preparation = gpu.pipeline(preparationName(variant));
        result.reset(fixture.initial_state);
        auto run = [&](bool useCandidate) {
            const Timing timing = dispatch(gpu, useCandidate ? candidate : native,
                useCandidate ? variant : Canonical, input, result, false, false,
                useCandidate ? preparation : nil, true);
            return timing;
        };
        double warmed=0;
        unsigned warmCommands=0;
        while (warmed<0.150 || warmCommands<8) {
            warmed+=run(false).gpu; warmed+=run(true).gpu; warmCommands+=2;
            require(warmCommands<4096,"GPU warming exceeded bounded command count");
        }
        emit(@{@"kind":@"bench_warmup", @"variant":ns(variant.name),
            @"gpu_ms":@(warmed*1000), @"commands":@(warmCommands),
            @"cpu_tensor_access_during_warmup":@NO});
        std::vector<double> nativeTimes, candidateTimes, pairedSpeedups, wallSpeedups;
        for (unsigned pair = 0; pair < 10; ++pair) {
            const bool reverse = (random() & 1) != 0;
            const std::array<bool, 4> order{{reverse, !reverse, !reverse, reverse}};
            std::array<Timing, 2> a{}, b{};
            unsigned ai = 0, bi = 0;
            for (unsigned slot = 0; slot < order.size(); ++slot) {
                const Timing t = run(order[slot]);
                if (order[slot]) b[bi++] = t; else a[ai++] = t;
                emit(@{@"kind": @"bench_sample", @"candidate": ns(variant.name),
                       @"pair": @(pair), @"slot": @(slot),
                       @"run": order[slot] ? @"candidate" : @"canonical",
                       @"order": reverse ? @"BAAB" : @"ABBA", @"rows": @(rows),
                       @"lanes": @(lanes), @"gpu_ms": @(t.gpu * 1000), @"wall_ms": @(t.wall * 1000)});
            }
            const double at = (a[0].gpu + a[1].gpu) * 0.5, bt = (b[0].gpu + b[1].gpu) * 0.5;
            nativeTimes.push_back(at); candidateTimes.push_back(bt); pairedSpeedups.push_back(at / bt);
            wallSpeedups.push_back((a[0].wall + a[1].wall) / (b[0].wall + b[1].wall));
        }
        input.check(); result.check(false,fixture.initial_state); result.seed.immutable();
        require(!*result.diagnostics.as<uint32_t>(),"benchmark diagnostics nonzero");
        emit(@{@"kind": @"bench_summary", @"variant": ns(variant.name),
               @"rows": @(rows), @"lanes": @(lanes), @"matched_pairs": @10,
               @"canonical_gpu_ms_median": @(median(nativeTimes) * 1000),
               @"candidate_gpu_ms_median": @(median(candidateTimes) * 1000),
               @"paired_gpu_speedup_median": @(median(pairedSpeedups)),
               @"paired_wall_speedup_median": @(median(wallSpeedups)),
               @"synthetic_rows_per_second": @(double(rows) * lanes / median(candidateTimes)),
               @"canaries_pass": @YES, @"immutable_sha256_pass": @YES,
               @"scope": @"F32 W/U/score/end-key preparation plus 48-head device-state recurrence; not model prefill throughput",
               @"preparation_included_in_timing": @YES,
               @"coefficients_bytes": @(result.coefficients.bytes),
               @"snapshot_bytes":@(result.snapshot.bytes),@"flags_bytes":@(result.flags.bytes),
               @"eligibility":flagMetrics(result,Heads),
               @"snapshot_prepare_wy_restore_native_replay_included":@YES,
               @"gpu_state_reset_copy_included":@YES, @"cpu_tensor_access_during_timing":@NO});
    }
}
} // namespace

// Analytical device-address hulls and every dispatched preparation/apply tile.
// This mode creates no Metal device, buffer, pipeline, or command queue.
void cpuLayout() {
    size_t cases=0, tiles=0;
    for (size_t rows : {size_t(1),size_t(15),size_t(16),size_t(17),size_t(31),
                       size_t(32),size_t(33),size_t(63),size_t(64),size_t(65),
                       size_t(2047),size_t(2048)}) {
        for (size_t lanes : {size_t(1),size_t(2),size_t(4),size_t(32)}) {
            for (const auto &variant : Variants) {
                const size_t time=variant.time, chunks=(rows+time-1)/time;
                const size_t stride=preparedStride(time);
                const size_t coefficients=lanes*chunks*Heads*stride;
                require(coefficients<=std::numeric_limits<uint64_t>::max()/4,"workspace address overflow");
                require((lanes*rows-1)*Mixed+10239==lanes*rows*Mixed-1,"mixed address hull");
                require((lanes*rows-1)*Heads+47==lanes*rows*Heads-1,"gate address hull");
                require((lanes*rows-1)*Out+6143==lanes*rows*Out-1,"output address hull");
                require(((lanes*rows-1)*V+127)*K+127==lanes*rows*V*K-1,"audit history address hull");
                for (size_t lane=0;lane<lanes;++lane) for (size_t chunk=0;chunk<chunks;++chunk)
                    for (size_t head=0;head<Heads;++head) {
                        const size_t base=((lane*chunks+chunk)*Heads+head)*stride;
                        require(base+stride<=coefficients,"prepare tile workspace bound");
                        const size_t begin=chunk*time, count=std::min(time,rows-begin);
                        require(count>0 && begin+count<=rows,"dynamic tensor source tail bound");
                        ++tiles;
                    }
                for (size_t lane=0;lane<lanes;++lane) for (size_t head=0;head<Heads;++head)
                    for (size_t tile=0;tile<V/variant.values;++tile) {
                        const size_t base=lane*State+(head*V+tile*variant.values)*K;
                        require(base+variant.values*K<=lanes*State,"apply state tile bound");
                        require(tile*variant.values+variant.values<=V,"value tile bound");
                        ++tiles;
                    }
                for (size_t lane=0;lane<lanes;++lane) for (size_t head=0;head<Heads;++head) {
                    require(lane*Heads+head<lanes*Heads,"flag head bound");
                    for (size_t tile=0;tile<64;++tile) {
                        require((lane*Heads+head)*V*K+tile*256+255<lanes*State,"compact snapshot tile bound");
                        require(lane*State+head*V*K+tile*256+255<lanes*State,"restore state tile bound");
                        tiles+=2;
                    }
                    for (size_t tile=0;tile<8;++tile) {
                        require(lane*State+(head*V+tile*16)*K+16*K<=lanes*State,"native replay tile bound");
                        ++tiles;
                    }
                }
                ++cases;
            }
        }
    }
    const size_t coefficientBytes=size_t(64)*Heads*preparedStride(32)*4;
    const size_t snapshotBytes=State*4,flagBytes=Heads*4;
    const auto arenaRound=[](size_t bytes) {return (bytes+16383)/16384*16384;};
    const size_t logicalArena=coefficientBytes+snapshotBytes+flagBytes;
    const size_t plannedArena=arenaRound(coefficientBytes)+arenaRound(snapshotBytes)+arenaRound(flagBytes);
    require(logicalArena==167903424 && plannedArena==167919616,"v4 singleton arena contract");
    emit(@{@"kind":@"cpu_layout", @"pass":@YES, @"geometry_cases":@(cases),
        @"checked_tiles":@(tiles), @"time":@32, @"values":@32,
        @"sg_counts":@[@4,@8], @"rows_bounds":@[@1,@2048], @"lanes_bounds":@[@1,@32],
        @"workspace_bytes_2k_singleton":@(coefficientBytes),
        @"snapshot_bytes_singleton":@(snapshotBytes),@"flags_bytes_singleton":@(flagBytes),
        @"arena_logical_bytes_2k_singleton":@(logicalArena),
        @"arena_planned_physical_bytes_2k_singleton":@(plannedArena),
        @"guarded_scratch_allocation_bytes_2k_singleton":@(logicalArena+6*Guard),
        @"prepare_threadgroup_declared_bytes":@(size_t(3*32*32+4*32)*4),
        @"cached_weight_norm_and_flags_bytes_2k_singleton":@(size_t(64)*Heads*2*32*4),
        @"apply_threadgroup_declared_bytes":@(size_t(2*32*32+32+32+2*8)*4+4),
        @"apply_threadgroup_declared_bytes_sg4":@(size_t(2*32*32+32+32+2*4)*4+4),
        @"apply_threadgroup_declared_bytes_sg8":@(size_t(2*32*32+32+32+2*8)*4+4),
        @"scope":@"address hulls and tile bounds; runtime MPP resources and canaries still require Root GPU", 
        @"metal_device_created":@NO});
}

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc == 1) {
                std::cout << "Usage: metal-oracle-wy --cpu-layout|--resources|--quality|--fallback-proof|--guard-equivalence|--bench [--library path] [--lanes 1..32] [--rows 1..2048]\n";
                return 0; // No Metal/device/pipeline creation on help/default.
            }
            std::string mode, library = (std::filesystem::absolute(argv[0]).parent_path() /
                                         "gdn-wy.metallib").string();
            unsigned lanes = 1, rows = 2048;
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--cpu-layout" || arg == "--resources" || arg == "--quality" || arg == "--fallback-proof" || arg == "--guard-equivalence" || arg == "--bench") {
                    require(mode.empty(), "choose one mode"); mode = arg;
                } else if (arg == "--library") {
                    require(++i < argc, "--library needs path"); library = argv[i];
                } else if (arg == "--lanes" || arg == "--rows") {
                    require(++i < argc, arg + " needs integer");
                    size_t used = 0; const unsigned long value = std::stoul(argv[i], &used);
                    require(used == std::strlen(argv[i]) && value > 0 &&
                            value <= (arg == "--lanes" ? 32 : 2048), "invalid " + arg);
                    (arg == "--lanes" ? lanes : rows) = unsigned(value);
                } else throw std::runtime_error("unknown argument: " + arg);
            }
            require(!mode.empty(), "missing mode");
            if (mode=="--cpu-layout") { cpuLayout(); return 0; }
            GPU gpu(library);
            if (mode == "--resources") resources(gpu);
            else if (mode == "--quality") return quality(gpu, lanes) ? 0 : 2;
            else if (mode == "--fallback-proof") return quality(gpu,lanes,true) ? 0 : 2;
            else if (mode == "--guard-equivalence") return guardEquivalence(gpu,lanes) ? 0 : 2;
            else bench(gpu, lanes, rows);
            return 0;
        } catch (const std::exception &e) {
            emit(@{@"kind": @"error", @"message": ns(e.what())});
            return 1;
        }
    }
}
