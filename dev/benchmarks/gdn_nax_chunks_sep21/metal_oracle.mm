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
    {"private_gdn_wy_v32_t16_sg4", 32, 16, 4},
    {"private_gdn_wy_v32_t16_sg8", 32, 16, 8}}};
constexpr Variant Canonical{"flash_gdn_staged_v16_t16", 16, 16, 16};
std::string auditName(const Variant &v) {
    return "private_gdn_wy_audit_v" + std::to_string(v.values) + "_t" +
        std::to_string(v.time) + "_sg" + std::to_string(v.groups);
}
std::string preparationName(const Variant &v) {
    return "private_gdn_wy_prepare_t" + std::to_string(v.time) + "_sg" + std::to_string(v.groups);
}
size_t preparedStride(size_t time) { return 3*time*K + time*time + time; }

struct GPU {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
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
        id<MTLFunction> function = [library newFunctionWithName:ns(name)];
        require(function != nil, "missing shader: " + name);
        NSError *error = nil;
        id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
        require(p != nil, "pipeline " + name + ": " + str(error.localizedDescription));
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
    Guarded state, output, diagnostics, history, delta, preOutput, coefficients, seed;
    Results(GPU &gpu, size_t tokens, unsigned batch, bool auditing, unsigned time = 0)
        : lanes(batch), rows(tokens), audit(auditing),
          preparedTime(time),
          state(gpu.device, batch * State * 4, "recurrent F32"),
          output(gpu.device, batch * tokens * Out * 2, "output BF16"),
          diagnostics(gpu.device, 4, "diagnostics"),
          history(gpu.device, auditing ? batch * tokens * V * K * 4 : 4, "audit state history F32"),
          delta(gpu.device, auditing ? batch * tokens * V * 4 : 4, "audit delta F32"),
          preOutput(gpu.device, auditing ? batch * tokens * V * 4 : 4, "audit pre-BF16 output F32"),
          coefficients(gpu.device, time ? size_t(batch) * ((tokens + time - 1) / time) * Heads * preparedStride(time) * 4 : 4,
                       "prepared W/U/end-key/score/prefix F32"),
          seed(gpu.device, batch * State * 4, "immutable GPU benchmark state seed F32") {}
    void reset(const std::vector<float> &seed) {
        require(seed.size() == V * K || seed.size() == State, "state seed dimension");
        for (size_t b = 0; b < lanes; ++b) for (size_t h = 0; h < Heads; ++h)
            std::copy_n(seed.data() + (seed.size() == State ? h * V * K : 0), V * K,
                        state.as<float>() + b * State + h * V * K);
        std::memset(output.data(), Poison, output.bytes);
        *diagnostics.as<uint32_t>() = 0;
        history.fillNaN(); delta.fillNaN(); preOutput.fillNaN(); coefficients.fillNaN();
        std::memcpy(this->seed.data(), state.data(), state.bytes); this->seed.freeze();
    }
    void check(bool headZeroOnly, const std::vector<float> &seed) const {
        for (const Guarded *buffer : {&state, &output, &diagnostics, &history, &delta, &preOutput, &coefficients, &this->seed})
            buffer->canaries();
        if (headZeroOnly) {
            for (size_t b = 0; b < lanes; ++b) {
                for (size_t h = 1; h < Heads; ++h)
                    require(std::memcmp(state.as<float>() + b * State + h * V * K,
                                        seed.data(), V * K * 4) == 0,
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
        [encoder setComputePipelineState:preparation];
        [encoder setBuffer:input.mixed.allocation offset:Guard atIndex:0];
        [encoder setBuffer:input.decay.allocation offset:Guard atIndex:1];
        [encoder setBuffer:input.beta.allocation offset:Guard atIndex:2];
        [encoder setBuffer:result.coefficients.allocation offset:Guard atIndex:3];
        [encoder setBuffer:result.diagnostics.allocation offset:Guard atIndex:4];
        [encoder setBuffer:input.params.allocation offset:Guard atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(Heads, (input.rows + variant.time - 1) / variant.time, input.lanes)
               threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        // The state phase reads coefficients produced by the preceding
        // dispatch. Both phases remain inside the measured command buffer.
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [encoder setComputePipelineState:pipeline];
    const std::array<const Guarded *, 7> buffers{{&input.mixed, &input.decay, &input.beta,
        &result.state, &result.output, &result.diagnostics, &input.params}};
    for (size_t i = 0; i < buffers.size(); ++i)
        [encoder setBuffer:buffers[i]->allocation offset:Guard atIndex:i];
    if (preparation) [encoder setBuffer:result.coefficients.allocation offset:Guard atIndex:10];
    if (audit) {
        [encoder setBuffer:result.history.allocation offset:Guard atIndex:7];
        [encoder setBuffer:result.delta.allocation offset:Guard atIndex:8];
        [encoder setBuffer:result.preOutput.allocation offset:Guard atIndex:9];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(headZeroOnly ? 1 : Heads, V / variant.values, input.lanes)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
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
CoefficientQuality coefficientQuality(const std::vector<Fixture> &fixtures, const Results &r) {
    require(fixtures.size() == 1 || fixtures.size() == Heads, "coefficient truth head count");
    std::vector<gdn_nax_cpu::Prepared<double>> truths;
    for (const auto &fixture : fixtures)
        truths.push_back(gdn_nax_cpu::prepare<double>(fixture,r.preparedTime));
    const auto &truth = truths.front();
    const size_t time=r.preparedTime, stride=truth.stride();
    const std::array<size_t,5> offsets{{0,time*K,2*time*K,3*time*K,3*time*K+time*time}};
    const std::array<size_t,5> lengths{{time*K,time*K,time*K,time*time,time}};
    CoefficientQuality q;
    for (size_t field=0;field<5;++field) {
        std::vector<double> ref(truth.num_chunks*Heads*lengths[field]);
        for (size_t chunk=0;chunk<truth.num_chunks;++chunk)
            for (size_t head=0;head<Heads;++head)
                std::copy_n(truths[fixtures.size()==1?0:head].storage.data()+chunk*stride+offsets[field],lengths[field],
                            ref.data()+(chunk*Heads+head)*lengths[field]);
        q.fields[field]=errors(r.lanes*ref.size(),ref,r.lanes,[&](size_t i) {
            const size_t item=i/lengths[field], within=i%lengths[field];
            return double(r.coefficients.as<float>()[item*stride+offsets[field]+within]);
        });
    }
    for (size_t batch=0;batch<r.lanes;++batch) for (size_t chunk=0;chunk<truth.num_chunks;++chunk)
        for (size_t head=0;head<Heads;++head) for (size_t field=0;field<5;++field)
            for (size_t i=0;i<lengths[field];++i) {
                const size_t token=field==4?i:(field==3?i/time:i/K);
                const size_t previous=field==3?i%time:0;
                const bool pad=token>=truth.active_rows(chunk) || (field==3 && previous>=truth.active_rows(chunk));
                if (!pad) continue;
                ++q.padded;
                q.nonzeroPadded+=r.coefficients.as<float>()[((batch*truth.num_chunks+chunk)*Heads+head)*stride+offsets[field]+i]!=0.0f;
            }
    return q;
}
CoefficientQuality coefficientQuality(const Fixture &f, const Results &r) {
    return coefficientQuality(std::vector<Fixture>{f},r);
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
    id<MTLComputePipelineState> preparation, const char *stage, unsigned lanes) {
    Inputs input(gpu,heads,lanes);
    Results result(gpu,heads.front().rows,lanes,false,variant.time);
    result.reset(seed);
    // A missed timed-kernel fragment must remain nonfinite and fail numerics.
    std::fill_n(result.output.as<uint16_t>(),result.output.bytes/2,uint16_t(0x7fc0));
    dispatch(gpu,pipeline,variant,input,result,false,false,preparation);
    input.check(); result.check(false,seed); result.seed.immutable();
    const Error state = errors(lanes*State,truth.state,lanes,
        [&](size_t i) {return double(result.state.as<float>()[i]);});
    const Error output = errors(lanes*truth.output.size(),truth.output,lanes,
        [&](size_t i) {return double(from_bf16(result.output.as<uint16_t>()[i]));});
    size_t mismatchCPU = 0;
    for (size_t i = 0; i < lanes*truth.output.size(); ++i)
        mismatchCPU += result.output.as<uint16_t>()[i] != to_bf16(float(truth.output[i%truth.output.size()]));
    const auto coefficients = coefficientQuality(heads,result);
    const uint32_t diagnostics = *result.diagnostics.as<uint32_t>();
    const bool pass = state.pass() && bf16Pass(output) && coefficients.pass() && !diagnostics;
    emit(@{@"kind":@"quality_timed_all_heads", @"variant":ns(variant.name),
        @"stage":ns(stage), @"rows":@(input.rows), @"lanes":@(lanes), @"heads":@(Heads),
        @"qualification_pass":@(pass), @"required":@YES,
        @"carried_state_f32":state.json(), @"output_bf16_vs_f64":bf16Metrics(output),
        @"bf16_mismatch_cpu_float_round_rate":@(double(mismatchCPU)/(lanes*truth.output.size())),
        @"prepared_coefficients":coefficients.json(), @"candidate_diagnostics":@(diagnostics),
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
    Results candidate(gpu, f.rows, lanes, true, variant.time), canonical(gpu, f.rows, lanes, false);
    candidate.reset(candidateSeed); canonical.reset(canonicalSeed);
    dispatch(gpu, canonicalPipeline, Canonical, input, canonical, true, false);
    dispatch(gpu, candidatePipeline, variant, input, candidate, true, true, preparation);
    input.check(); candidate.check(true, candidateSeed); canonical.check(true, canonicalSeed);
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
    // BF16-vs-F64 errors/mismatches are reported, not judged with an F32
    // tolerance. The exposed F32 intermediates and carried states qualify.
    const bool pass = history.pass() && delta.pass() && preOutput.pass() && state.pass() &&
                      nativeState.pass() && coefficientMetrics.pass() &&
                      !candidateDiagnostics && !nativeDiagnostics && !mismatchAudit;
    emit(@{@"kind": @"quality", @"fixture": ns(f.name), @"stage": ns(stage),
           @"variant": ns(variant.name), @"rows": @(f.rows), @"lanes": @(lanes),
           @"classification": ordinary ? @"ordinary" : @"range_trap",
           @"qualification_pass": @(pass), @"required": @(ordinary),
           @"history_f32": history.json(), @"delta_f32": delta.json(),
           @"pre_bf16_output_f32": preOutput.json(), @"carried_state_f32": state.json(),
           @"canonical_carried_state_f32": nativeState.json(),
           @"prepared_coefficients": coefficientMetrics.json(),
           @"output_bf16_vs_f64": candidateOutput.json(),
           @"canonical_output_bf16_vs_f64": nativeOutput.json(),
           @"bf16_mismatch_canonical_rate": @(double(mismatchNative) / candidateBF16.size()),
           @"bf16_mismatch_cpu_rate": @(double(mismatchCPU) / candidateBF16.size()),
           @"bf16_mismatch_own_f32_rate": @(double(mismatchAudit) / candidateBF16.size()),
           @"candidate_diagnostics": @(candidateDiagnostics), @"canonical_diagnostics": @(nativeDiagnostics),
           @"canaries_pass": @YES, @"immutable_sha256_pass": @YES,
           @"input_sha256": ns(input.mixed.frozenHash)});
    return {pass || !ordinary, {candidate.carried(), canonical.carried()}};
}

void resources(GPU &gpu) {
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

bool quality(GPU &gpu, unsigned lanes) {
    bool pass = true;
    emit(@{@"kind":@"quality_timed_all_heads_registration", @"rows":@67,
        @"continuation_rows":@19, @"heads":@48, @"key_heads":@16,
        @"f32_relative_rms_tolerance":@(RelativeTolerance),
        @"bf16_relative_rms_tolerance":@(BF16RelativeTolerance),
        @"bf16_max_abs_scale_tolerance":@(BF16AbsoluteScaleTolerance),
        @"bf16_max_abs_scale":@"max(1, F64 reference peak)",
        @"output_initialization":@"BF16 quiet NaN; unwritten fragments fail"});
    const auto initialHeads = heterogeneousFixtures(67,0);
    const auto continuedHeads = heterogeneousFixtures(19,1);
    const auto initialHeadTruth = heterogeneousTruth(initialHeads);
    const auto continuedHeadTruth = heterogeneousTruth(continuedHeads,&initialHeadTruth.state);
    const auto initialHeadSeeds = heterogeneousSeeds(initialHeads);
    auto native = gpu.pipeline(Canonical.name);
    const auto fixtures = gdn_chunk_cpu::make_fixtures();
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
        auto initial = qualifyTimedHeads(gpu,initialHeads,initialHeadTruth,initialHeadSeeds,
            variant,timed,preparation,"heterogeneous_tail_initial",lanes);
        auto continued = qualifyTimedHeads(gpu,continuedHeads,continuedHeadTruth,initial.second,
            variant,timed,preparation,"heterogeneous_carried_continuation",lanes);
        pass = pass && initial.first && continued.first;
    }
    emit(@{@"kind": @"quality_summary", @"pass": @(pass),
           @"fixtures": @(fixtures.size()), @"variants": @(Variants.size()),
           @"continuation_rows": @19, @"timed_all_head_stages":@4,
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
                ++cases;
            }
        }
    }
    emit(@{@"kind":@"cpu_layout", @"pass":@YES, @"geometry_cases":@(cases),
        @"checked_tiles":@(tiles), @"time":@16, @"values":@32,
        @"sg_counts":@[@4,@8], @"rows_bounds":@[@1,@2048], @"lanes_bounds":@[@1,@32],
        @"workspace_bytes_2k_singleton":@(size_t(128)*Heads*preparedStride(16)*4),
        @"prepare_threadgroup_declared_bytes":@(size_t(3*16*16+3*16)*4),
        @"apply_threadgroup_declared_bytes":@(size_t(2*32*16)*4),
        @"scope":@"address hulls and tile bounds; runtime MPP resources and canaries still require Root GPU", 
        @"metal_device_created":@NO});
}

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc == 1) {
                std::cout << "Usage: metal-oracle-wy --cpu-layout|--resources|--quality|--bench [--library path] [--lanes 1..32] [--rows 1..2048]\n";
                return 0; // No Metal/device/pipeline creation on help/default.
            }
            std::string mode, library = (std::filesystem::absolute(argv[0]).parent_path() /
                                         "gdn-wy.metallib").string();
            unsigned lanes = 1, rows = 2048;
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--cpu-layout" || arg == "--resources" || arg == "--quality" || arg == "--bench") {
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
            else bench(gpu, lanes, rows);
            return 0;
        } catch (const std::exception &e) {
            emit(@{@"kind": @"error", @"message": ns(e.what())});
            return 1;
        }
    }
}
