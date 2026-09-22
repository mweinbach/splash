#pragma once

// Isolated opt-in policy for the staged prefill call site. This header performs
// no GPU work, allocates no GPU resources, and reads no model payloads.

#include <CommonCrypto/CommonDigest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::gdn_prefill_fma_sep21 {

inline constexpr const char* kFlag = "SPLASH_FLASH_GDN_PREFILL_FMA_SEP21";
inline constexpr const char* kStagedFlag = "SPLASH_FLASH_GDN_STAGED";
inline constexpr const char* kPipeline = "private_gdn_scalar_fma_v16_t32";
inline constexpr const char* kPolicy =
    "qwen4-gdn-prefill-explicit-fma-f32-simd32-v16-t32-r64to2048-sep21-v1";
inline constexpr const char* kKernelSourceSHA256 =
    "166381991f5444585e7db3634f41432322efdd66bf08a8c402d5fed49e58c699";
inline constexpr const char* kMarker =
    ";private-gdn-prefill-fma-v16-t32-rows64to2048-lanes1to32-sep21-v1";
inline constexpr uint32_t kValues = 16;
inline constexpr uint32_t kTime = 32;
inline constexpr size_t kGPUAddedAllocations = 0;

namespace detail {

inline bool parseFlag(const char* raw) {
    if (!raw || std::strcmp(raw, "0") == 0) return false;
    if (std::strcmp(raw, "1") == 0) return true;
    throw std::invalid_argument(std::string(kFlag) + " must be exactly 0 or 1 (missing means 0)");
}

inline bool parseRequested(const char* raw, const char* stagedRaw) {
    // The disabled path returns before reading or validating stagedRaw.
    if (!parseFlag(raw)) return false;
    if (!stagedRaw || std::strcmp(stagedRaw, "1") != 0)
        throw std::invalid_argument(std::string(kFlag) + "=1 requires " + kStagedFlag + "=1");
    return true;
}

inline bool requestedFromEnvironment() {
    // Preserve the disabled native path even when another option is malformed.
    if (!parseFlag(std::getenv(kFlag))) return false;
    return parseRequested("1", std::getenv(kStagedFlag));
}

inline std::string sha256(std::string_view bytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) throw std::runtime_error("SHA256 initialization failed");
    size_t position = 0;
    constexpr size_t maxChunk = std::numeric_limits<CC_LONG>::max();
    while (position < bytes.size()) {
        const size_t remaining = bytes.size() - position;
        const size_t count = remaining < maxChunk ? remaining : maxChunk;
        if (CC_SHA256_Update(&context, bytes.data() + position, static_cast<CC_LONG>(count)) != 1)
            throw std::runtime_error("SHA256 update failed");
        position += count;
    }
    if (CC_SHA256_Final(digest.data(), &context) != 1) throw std::runtime_error("SHA256 finalization failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        result[2 * i] = hex[digest[i] >> 4];
        result[2 * i + 1] = hex[digest[i] & 15];
    }
    return result;
}

} // namespace detail

// Caller must invoke before backend construction; a successful first decision
// is immutable for the process. C++ static initialization is thread-safe.
inline bool requested() {
    static const bool enabled = detail::requestedFromEnvironment();
    return enabled;
}

constexpr bool eligible(size_t rows, size_t lanes, bool enabled) noexcept {
    return enabled && rows >= 64 && rows <= 2048 && lanes >= 1 && lanes <= 32;
}

constexpr const char* marker(bool enabled) noexcept { return enabled ? kMarker : ""; }

struct Route {
    const char* pipeline;
    uint32_t values;
    uint32_t time;
    bool usesPrivatePrefill;
};

// A convenience transform; the native pipeline and selectors are returned
// verbatim outside the qualified prefill rectangle or when disabled.
constexpr Route route(const char* nativePipeline, uint32_t nativeValues, uint32_t nativeTime,
                      size_t rows, size_t lanes, bool enabled) noexcept {
    return eligible(rows, lanes, enabled)
        ? Route{kPipeline, kValues, kTime, true}
        : Route{nativePipeline, nativeValues, nativeTime, false};
}

// Optional replacement of the old stage marker, preserving it when disabled.
constexpr const char* nativeStageMarker(const char* nativeMarker, bool enabled) noexcept {
    return enabled ? marker(true) : nativeMarker;
}

inline std::string numericalIdentity(std::string_view base, bool enabled) {
    if (!enabled) return std::string(base);
    std::string sourceBound(base);
    sourceBound += '\n';
    sourceBound += kPolicy;
    sourceBound += '\n';
    sourceBound += kKernelSourceSHA256;
    return detail::sha256(sourceBound);
}

inline std::string operationNumericalIdentity(std::string_view base, bool enabled) {
    return numericalIdentity(base, enabled);
}

} // namespace splash::flash::gdn_prefill_fma_sep21
