// Standalone two-stage synthetic oracle: never links the model/runtime/MLX. The default
// build only compiles. Explicit --resources/--quality/--bench create Metal
// resources and are intended to be invoked by the serialized GPU coordinator.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "metal/abi/FlashGDN.h"
#include "cpu_oracle.hpp"
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

struct Variant { const char *name; unsigned values, time; };
constexpr std::array<Variant, 5> Variants{{
    {"private_gdn_chunk_prepared_v16_t16", 16, 16},
    {"private_gdn_chunk_prepared_v16_t32", 16, 32},
    {"private_gdn_chunk_prepared_v32_t16", 32, 16},
    {"private_gdn_chunk_prepared_register_v32_t16", 32, 16},
    {"private_gdn_chunk_prepared_register_v32_t32", 32, 32}}};
constexpr Variant Canonical{"flash_gdn_staged_v16_t16", 16, 16};
std::string auditName(const Variant &variant) {
    return std::string(variant.name).find("register") != std::string::npos
        ? "private_gdn_chunk_prepared_register_audit_v" + std::to_string(variant.values) + "_t" + std::to_string(variant.time)
        : "private_gdn_chunk_prepared_audit_v" + std::to_string(variant.values) + "_t" + std::to_string(variant.time);
}

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
    void check() const { mixed.immutable(); decay.immutable(); beta.immutable(); params.immutable(); }
};

struct Results {
    unsigned lanes;
    size_t rows;
    bool audit;
    unsigned preparedTime;
    Guarded state, output, diagnostics, history, delta, preOutput, coefficients;
    Results(GPU &gpu, size_t tokens, unsigned batch, bool auditing, unsigned time = 0)
        : lanes(batch), rows(tokens), audit(auditing),
          preparedTime(time),
          state(gpu.device, batch * State * 4, "recurrent F32"),
          output(gpu.device, batch * tokens * Out * 2, "output BF16"),
          diagnostics(gpu.device, 4, "diagnostics"),
          history(gpu.device, auditing ? batch * tokens * V * K * 4 : 4, "audit state history F32"),
          delta(gpu.device, auditing ? batch * tokens * V * 4 : 4, "audit delta F32"),
          preOutput(gpu.device, auditing ? batch * tokens * V * 4 : 4, "audit pre-BF16 output F32"),
          coefficients(gpu.device, time ? size_t(batch) * ((tokens + time - 1) / time) * 16 * 2 * time * time * 4 : 4,
                       "prepared raw Gram/QK coefficients F32") {}
    void reset(const std::vector<float> &seed) {
        require(seed.size() == V * K, "state seed dimension");
        for (size_t b = 0; b < lanes; ++b) for (size_t h = 0; h < Heads; ++h)
            std::copy(seed.begin(), seed.end(), state.as<float>() + b * State + h * V * K);
        std::memset(output.data(), Poison, output.bytes);
        *diagnostics.as<uint32_t>() = 0;
        history.fillNaN(); delta.fillNaN(); preOutput.fillNaN(); coefficients.fillNaN();
    }
    void check(bool headZeroOnly, const std::vector<float> &seed) const {
        for (const Guarded *buffer : {&state, &output, &diagnostics, &history, &delta, &preOutput, &coefficients})
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
                id<MTLComputePipelineState> preparation = nil) {
    const unsigned threads = std::string(variant.name) == Canonical.name ? 512 : 128;
    require(pipeline.maxTotalThreadsPerThreadgroup >= threads, "pipeline thread limit");
    require(pipeline.staticThreadgroupMemoryLength <= gpu.device.maxThreadgroupMemoryLength,
            "pipeline threadgroup memory exceeds device limit");
    const auto start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command = [gpu.queue commandBuffer];
    command.label = ns(variant.name);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (preparation) {
        require(result.preparedTime == variant.time, "prepared coefficient geometry mismatch");
        require(preparation.maxTotalThreadsPerThreadgroup >= 128, "preparation pipeline thread limit");
        [encoder setComputePipelineState:preparation];
        [encoder setBuffer:input.mixed.allocation offset:Guard atIndex:0];
        [encoder setBuffer:result.coefficients.allocation offset:Guard atIndex:1];
        [encoder setBuffer:result.diagnostics.allocation offset:Guard atIndex:2];
        [encoder setBuffer:input.params.allocation offset:Guard atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(16, (input.rows + variant.time - 1) / variant.time, input.lanes)
               threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
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
    Error gram, qk;
    size_t padded = 0, nonzeroPadded = 0;
    bool pass() const { return gram.pass() && qk.pass() && !nonzeroPadded; }
    NSDictionary *json() const {
        return @{@"raw_gram_f32": gram.json(), @"raw_qk_f32": qk.json(),
                 @"padded_slots": @(padded), @"nonzero_padded_slots": @(nonzeroPadded),
                 @"pass": @(pass())};
    }
};
CoefficientQuality coefficientQuality(const Fixture &f, const Results &result) {
    const size_t time = result.preparedTime, matrix = time * time;
    require(time > 0, "missing prepared coefficient shape");
    const size_t chunks = (f.rows + time - 1) / time;
    std::vector<double> gram(chunks * 16 * matrix, 0), qk(chunks * 16 * matrix, 0);
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const size_t begin = chunk * time, count = std::min(time, f.rows - begin);
        for (size_t t = 0; t < count; ++t) for (size_t previous = 0; previous < count; ++previous) {
            double kk = 0, q_dot_k = 0;
            for (size_t k = 0; k < K; ++k) {
                const double earlier = from_bf16(f.k[(begin + previous) * K + k]);
                kk += double(from_bf16(f.k[(begin + t) * K + k])) * earlier;
                q_dot_k += double(from_bf16(f.q[(begin + t) * K + k])) * earlier;
            }
            for (size_t head = 0; head < 16; ++head) {
                const size_t index = (chunk * 16 + head) * matrix + t * time + previous;
                gram[index] = kk; qk[index] = q_dot_k;
            }
        }
    }
    const auto *coefficients = result.coefficients.as<float>();
    CoefficientQuality r;
    r.gram = errors(result.lanes * gram.size(), gram, result.lanes,
        [&](size_t i) { return double(coefficients[(i / matrix) * 2 * matrix + i % matrix]); });
    r.qk = errors(result.lanes * qk.size(), qk, result.lanes,
        [&](size_t i) { return double(coefficients[(i / matrix) * 2 * matrix + matrix + i % matrix]); });
    for (size_t batch = 0; batch < result.lanes; ++batch) for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const size_t count = std::min(time, f.rows - chunk * time);
        for (size_t head = 0; head < 16; ++head) for (size_t kind = 0; kind < 2; ++kind)
            for (size_t t = 0; t < time; ++t) for (size_t previous = 0; previous < time; ++previous) {
                if (t < count && previous < count) continue;
                const size_t i = (((batch * chunks + chunk) * 16 + head) * 2 + kind) * matrix + t * time + previous;
                ++r.padded; r.nonzeroPadded += coefficients[i] != 0.0f;
            }
    }
    return r;
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
    for (unsigned time : {16, 32}) {
        const std::string name = "private_gdn_chunk_prepare_t" + std::to_string(time);
        auto pipeline = gpu.pipeline(name);
        emit(@{@"kind": @"resources", @"name": ns(name),
               @"threadgroup_bytes": @(pipeline.staticThreadgroupMemoryLength),
               @"max_threads": @(pipeline.maxTotalThreadsPerThreadgroup),
               @"execution_width": @(pipeline.threadExecutionWidth)});
    }
    for (const auto &variant : Variants) {
        for (bool audit : {false, true}) {
            const std::string name = audit ? auditName(variant) : variant.name;
            auto pipeline = gpu.pipeline(name);
            emit(@{@"kind": @"resources", @"name": ns(name),
                   @"threadgroup_bytes": @(pipeline.staticThreadgroupMemoryLength),
                   @"max_threads": @(pipeline.maxTotalThreadsPerThreadgroup),
                   @"execution_width": @(pipeline.threadExecutionWidth)});
        }
    }
    auto pipeline = gpu.pipeline(Canonical.name);
    emit(@{@"kind": @"resources", @"name": ns(Canonical.name),
           @"threadgroup_bytes": @(pipeline.staticThreadgroupMemoryLength),
           @"max_threads": @(pipeline.maxTotalThreadsPerThreadgroup),
           @"execution_width": @(pipeline.threadExecutionWidth)});
}

bool quality(GPU &gpu, unsigned lanes) {
    bool pass = true;
    auto native = gpu.pipeline(Canonical.name);
    const auto fixtures = gdn_chunk_cpu::make_fixtures();
    for (const auto &variant : Variants) {
        auto candidate = gpu.pipeline(auditName(variant));
        auto preparation = gpu.pipeline("private_gdn_chunk_prepare_t" + std::to_string(variant.time));
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
    }
    emit(@{@"kind": @"quality_summary", @"pass": @(pass),
           @"fixtures": @(fixtures.size()), @"variants": @(Variants.size()),
           @"continuation_rows": @19, @"reference": @"BF16 inputs, independent F64 serial recurrence"});
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
        auto preparation = gpu.pipeline("private_gdn_chunk_prepare_t" + std::to_string(variant.time));
        auto run = [&](bool useCandidate) {
            result.reset(fixture.initial_state);
            const Timing timing = dispatch(gpu, useCandidate ? candidate : native,
                useCandidate ? variant : Canonical, input, result, false, false,
                useCandidate ? preparation : nil);
            result.check(false, fixture.initial_state);
            require(!*result.diagnostics.as<uint32_t>(), "benchmark diagnostics nonzero");
            return timing;
        };
        for (unsigned warm = 0; warm < 3; ++warm) { run(false); run(true); }
        std::vector<double> nativeTimes, candidateTimes, pairedSpeedups, wallSpeedups;
        for (unsigned pair = 0; pair < 9; ++pair) {
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
        input.check();
        emit(@{@"kind": @"bench_summary", @"variant": ns(variant.name),
               @"rows": @(rows), @"lanes": @(lanes), @"matched_pairs": @9,
               @"canonical_gpu_ms_median": @(median(nativeTimes) * 1000),
               @"candidate_gpu_ms_median": @(median(candidateTimes) * 1000),
               @"paired_gpu_speedup_median": @(median(pairedSpeedups)),
               @"paired_wall_speedup_median": @(median(wallSpeedups)),
               @"synthetic_rows_per_second": @(double(rows) * lanes / median(candidateTimes)),
               @"canaries_pass": @YES, @"immutable_sha256_pass": @YES,
               @"scope": @"shared raw Gram/QK preparation plus 48-head recurrence; not model prefill throughput",
               @"preparation_included_in_timing": @YES,
               @"coefficients_bytes": @(result.coefficients.bytes)});
    }
}
} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc == 1) {
                std::cout << "Usage: metal-oracle-prepared --resources|--quality|--bench [--library path] [--lanes 1..32] [--rows 1..2048]\n";
                return 0; // No Metal/device/pipeline creation on help/default.
            }
            std::string mode, library = (std::filesystem::absolute(argv[0]).parent_path() /
                                         "gdn-chunk-prepared.metallib").string();
            unsigned lanes = 1, rows = 2048;
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--resources" || arg == "--quality" || arg == "--bench") {
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
