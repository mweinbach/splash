#include "WeightStore.hpp"
#include "ModelFactory.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <utility>

#if defined(SPLASH_METAL41_EXPERIMENT)
#include "ExperimentalFP8.hpp"
#endif
#if defined(SPLASH_INT8_EXPERIMENT)
#include "ExperimentalINT8.hpp"
#include "PreconvertedINT8.hpp"
#endif
#if defined(SPLASH_METAL41_EXPERIMENT) || defined(SPLASH_INT8_EXPERIMENT)
#include "ModelDescriptor.hpp"

#include <atomic>
#include <bit>
#include <cmath>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <cstdlib>
#include <chrono>
#endif

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace splash::model {

uint64_t checkedWeightMultiply(uint64_t left, uint64_t right,
                         std::string_view description) {
    if (left && right > std::numeric_limits<uint64_t>::max() / left) {
        throw WeightStoreError(std::string(description) + " overflows");
    }
    return left * right;
}

namespace {

[[nodiscard]] uint64_t checkedWeightAdd(uint64_t left, uint64_t right,
                                        std::string_view description) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        throw WeightStoreError(std::string(description) + " overflows");
    }
    return left + right;
}

[[nodiscard]] uint64_t q4Elements(uint32_t outputSize, uint32_t inputSize) {
    if (!outputSize || !inputSize || inputSize % kQ4GroupElements) {
        throw WeightStoreError(
            "Q4 projection dimensions must be positive and input-aligned");
    }
    return checkedWeightMultiply(outputSize, inputSize, "Q4 element count");
}

[[nodiscard]] uint64_t q8PackedBytes(uint32_t outputSize, uint32_t inputSize) {
    uint64_t elements = q4Elements(outputSize, inputSize);
    return checkedWeightAdd(
        elements,
        checkedWeightMultiply(elements / 32, 2, "Q8 parameter byte count"),
        "Q8 packed byte count");
}

} // namespace

uint64_t q4PackedBytes(uint32_t outputSize, uint32_t inputSize) {
    uint64_t elements = q4Elements(outputSize, inputSize);
    return checkedWeightMultiply(elements / 16, 9, "Q4 packed byte count");
}

void validateQ4Layout(uint32_t outputSize, uint32_t inputSize) {
    static_cast<void>(q4Elements(outputSize, inputSize));
    if (outputSize % kQ4StorageN) {
        throw WeightStoreError(
            "Q4 output dimension is incompatible with StorageN=256");
    }
}

namespace {

uint64_t alignPacked(uint64_t value) {
    return checkedWeightAdd(value, kWeightFileAlignment - 1,
                            "packed file alignment") &
           ~(kWeightFileAlignment - 1);
}

uint32_t loadLittleEndian32(const uint8_t *bytes) {
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
        (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

std::string systemError(std::string_view operation,
                        const std::filesystem::path &path, int error) {
    return std::string(operation) + " " + path.string() + ": " +
        std::error_code(error, std::generic_category()).message();
}

class MappedRegion final {
public:
    static std::shared_ptr<MappedRegion> openReadOnly(
        const std::filesystem::path &path) {
        int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) {
            throw WeightStoreError(systemError("unable to open", path, errno));
        }

        struct stat status {};
        if (fstat(descriptor, &status) != 0) {
            int error = errno;
            close(descriptor);
            throw WeightStoreError(systemError("unable to stat", path, error));
        }
        if (!S_ISREG(status.st_mode) || status.st_size <= 0) {
            close(descriptor);
            throw WeightStoreError("packed file is not a non-empty regular file: " +
                                   path.string());
        }
        uint64_t bytes = static_cast<uint64_t>(status.st_size);
        if (bytes > std::numeric_limits<size_t>::max()) {
            close(descriptor);
            throw WeightStoreError("packed file is too large to map: " +
                                   path.string());
        }

        // Metal can materialize MAP_PRIVATE file mappings as anonymous dirty
        // pages on GPU use. Keep immutable weights file-backed and reclaimable.
        void *address = mmap(nullptr, static_cast<size_t>(bytes), PROT_READ,
                             MAP_SHARED, descriptor, 0);
        int mapError = errno;
        close(descriptor);
        if (address == MAP_FAILED) {
            throw WeightStoreError(
                systemError("unable to mmap", path, mapError));
        }
        return std::shared_ptr<MappedRegion>(
            new MappedRegion(address, bytes));
    }

    ~MappedRegion() {
        if (address_) {
            munmap(address_, static_cast<size_t>(bytes_));
        }
    }

    MappedRegion(const MappedRegion &) = delete;
    MappedRegion &operator=(const MappedRegion &) = delete;

    [[nodiscard]] void *address() const noexcept { return address_; }
    [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
    MappedRegion(void *address, uint64_t bytes)
        : address_(address), bytes_(bytes) {}

    void *address_ = nullptr;
    uint64_t bytes_ = 0;
};

} // namespace

struct WeightFile::Impl {
    metal::MetalBackend *backend = nullptr;
    std::shared_ptr<MappedRegion> mapping;
    metal::MetalBuffer base;
    WeightFileRecord record;
    uint64_t offset = 16;
    bool finished = false;
};

WeightFile::WeightFile(metal::MetalBackend &backend,
                       std::filesystem::path path,
                       std::string relativePath,
                       std::string_view expectedMagic,
                       uint32_t expectedLayer,
                       uint32_t expectedType)
    : impl_(std::make_unique<Impl>()) {
    if (expectedMagic.size() != 8) {
        throw WeightStoreError("packed file magic must contain eight bytes");
    }
    impl_->backend = &backend;
    impl_->mapping = MappedRegion::openReadOnly(path);
    if (impl_->mapping->bytes() < 16 ||
        impl_->mapping->bytes() % kWeightFileAlignment) {
        throw WeightStoreError(
            "packed file size is not 16 KiB-aligned: " + path.string());
    }
    const auto *header = static_cast<const uint8_t *>(
        impl_->mapping->address());
    uint32_t layer = loadLittleEndian32(header + 8);
    uint32_t type = loadLittleEndian32(header + 12);
    if (std::memcmp(header, expectedMagic.data(), 8) != 0 ||
        layer != expectedLayer || type != expectedType) {
        throw WeightStoreError("packed file header mismatch: " + path.string());
    }

    impl_->record = {
        std::move(relativePath), std::string(expectedMagic), layer, type,
        impl_->mapping->bytes(),
    };
    impl_->base = backend.wrapSharedMemory(
        impl_->mapping->address(), impl_->mapping->bytes(), impl_->mapping,
        impl_->record.relativePath);
}

WeightFile::~WeightFile() = default;

metal::MetalBuffer WeightFile::section(uint64_t bytes,
                                       std::string_view label) {
    if (impl_->finished) {
        throw WeightStoreError("cannot add a section after packed file finish");
    }
    if (!bytes) throw WeightStoreError("packed section must not be empty");
    uint64_t start = alignPacked(impl_->offset);
    uint64_t end = checkedWeightAdd(start, bytes, "packed section end");
    if (start % kWeightFileAlignment || end > impl_->mapping->bytes()) {
        throw WeightStoreError(
            "packed file is truncated at section " + std::string(label));
    }
    impl_->offset = end;
    return impl_->backend->view(impl_->base, start, bytes);
}

void WeightFile::finish() {
    if (impl_->finished) return;
    uint64_t consumed = alignPacked(impl_->offset);
    if (consumed != impl_->mapping->bytes()) {
        throw WeightStoreError(
            "packed file has unconsumed or missing bytes: " +
            impl_->record.relativePath);
    }
    impl_->finished = true;
}

const WeightFileRecord &WeightFile::record() const noexcept {
    return impl_->record;
}

#if defined(SPLASH_METAL41_EXPERIMENT) || defined(SPLASH_INT8_EXPERIMENT)
void WeightFile::recordConvertedFingerprint(std::string_view label,
                                           std::string_view fingerprint) {
    if (impl_->finished || label.empty() || fingerprint.size() != 64 ||
        fingerprint.find_first_not_of("0123456789abcdef") !=
            std::string_view::npos) {
        throw WeightStoreError("invalid converted projection fingerprint");
    }
    impl_->record.convertedFingerprints.push_back(
        std::to_string(label.size()) + ":" + std::string(label) + ":" +
        std::string(fingerprint));
}

namespace {

std::string convertedDigest(const void *bytes, uint64_t count) {
    if (!bytes || !count || count > std::numeric_limits<CC_LONG>::max())
        throw WeightStoreError("converted plane is empty or too large to hash");
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    if (!CC_SHA256(bytes, static_cast<CC_LONG>(count), digest.data()))
        throw WeightStoreError("unable to fingerprint converted plane");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2);
    for (unsigned char byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 15]);
    }
    return result;
}

#if defined(SPLASH_METAL41_EXPERIMENT)
std::shared_ptr<metal::MetalBuffer>
convertedHalfWorkspace(metal::MetalBackend &backend) {
    // Allocation happens during loading, never during graph encoding. The
    // weak registry cannot retain a model's workspace after its weights die.
    static std::mutex mutex;
    static std::unordered_map<metal::MetalBackend *,
                              std::weak_ptr<metal::MetalBuffer>> workspaces;
    std::lock_guard lock(mutex);
    for (auto entry = workspaces.begin(); entry != workspaces.end();) {
        if (entry->second.expired())
            entry = workspaces.erase(entry);
        else
            ++entry;
    }
    auto &cached = workspaces[&backend];
    if (auto workspace = cached.lock()) {
        try {
            // Backend addresses may be reused while a caller retains old
            // buffers. view validates the allocation's backend identity.
            static_cast<void>(backend.view(
                *workspace, 0, experimental_fp8::kHalfScratchBytes));
            return workspace;
        } catch (const metal::MetalBackendError &) {
            if (!backend.healthy())
                throw;
        }
    }
    auto workspace = std::make_shared<metal::MetalBuffer>(backend.allocateBuffer(
        experimental_fp8::kHalfScratchBytes, metal::BufferStorage::Shared,
        "experimental FP8 shared HALF input"));
    cached = workspace;
    return workspace;
}

std::shared_ptr<ops::ExperimentalFP8Projection>
convertAffineQ4(metal::MetalBackend &backend, const ops::Q4Projection &source,
                std::string_view label) {
    using namespace experimental_fp8;
    const uint32_t n = source.outputSize;
    const uint32_t k = source.inputSize;
    if (n > std::numeric_limits<int32_t>::max() ||
        k > std::numeric_limits<int32_t>::max())
        throw WeightStoreError("experimental FP8 dimensions exceed tensor ABI");
    static_cast<void>(convertedProjectionBytes(n, k));
    const uint32_t scaleRowStride = static_cast<uint32_t>(convertedScaleRowStride(k));
    const uint64_t dataBytes = checkedWeightMultiply(n, k, "FP8 data bytes");
    const uint64_t scaleBytes = checkedWeightMultiply(n, scaleRowStride,
                                                     "FP8 scale bytes");
    // The isolated experiment hashes each retained plane with CommonCrypto's
    // one-shot API. Its supported model projections are below this ceiling.
    for (const uint64_t bytes : {dataBytes, scaleBytes, kHalfScratchBytes}) {
        if (bytes > std::numeric_limits<size_t>::max() ||
            bytes > backend.capabilities().maxBufferLengthBytes ||
            bytes > std::numeric_limits<CC_LONG>::max())
            throw WeightStoreError("experimental FP8 plane exceeds allocation or hash limit");
    }
    const auto *packed = static_cast<const uint8_t *>(source.weights.contents());
    const auto *scales = static_cast<const uint16_t *>(source.scales.contents());
    const auto *biases = static_cast<const uint16_t *>(source.biases.contents());
    if (!packed || !scales || !biases || source.weights.sizeBytes() < dataBytes / 2 ||
        source.scales.sizeBytes() < dataBytes / 32 ||
        source.biases.sizeBytes() < dataBytes / 32)
        throw WeightStoreError("experimental FP8 source is not a complete shared Q4 projection");

    auto result = std::make_shared<ops::ExperimentalFP8Projection>();
    const std::string prefix = "experimental FP8 " + std::string(label);
    result->data = backend.allocateBuffer(dataBytes, metal::BufferStorage::Shared,
                                          prefix + " data");
    result->scales = backend.allocateBuffer(scaleBytes, metal::BufferStorage::Shared,
                                            prefix + " scales");
    result->diagnostics = backend.allocateBuffer(16, metal::BufferStorage::Shared,
                                                 prefix + " diagnostics");
    if (!result->diagnostics.contents())
        throw WeightStoreError("experimental FP8 diagnostics are not CPU accessible");
    std::memset(result->diagnostics.contents(), 0, 16);
    result->sharedHalfInput = convertedHalfWorkspace(backend);
    result->halfInput = *result->sharedHalfInput;
    result->scaleRowStride = scaleRowStride;
    auto *data = static_cast<uint8_t *>(result->data.contents());
    auto *convertedScales = static_cast<uint8_t *>(result->scales.contents());
    if (!data || !convertedScales)
        throw WeightStoreError("experimental FP8 allocations are not CPU accessible");
    // Each logical scale row is tight along K/32; zero blocks use unit scale.
    std::memset(convertedScales, 127, static_cast<size_t>(scaleBytes));

    std::atomic<uint32_t> nextRow{0};
    std::atomic<bool> failed{false};
    std::mutex failureMutex;
    std::exception_ptr failure;
    const uint32_t quantGroups = k / 64;
    const auto worker = [&] {
        try {
            while (!failed.load(std::memory_order_relaxed)) {
                const uint32_t firstRow = nextRow.fetch_add(8, std::memory_order_relaxed);
                if (firstRow >= n)
                    break;
                for (uint32_t row = firstRow; row < std::min(n, firstRow + 8); ++row) {
                    const uint64_t slab = uint64_t{row / kQ4StorageN} * quantGroups * kQ4StorageN;
                    const uint32_t column = row % kQ4StorageN;
                    for (uint32_t group = 0; group < quantGroups; ++group) {
                        const uint64_t parameter = slab + uint64_t{group} * kQ4StorageN + column;
                        const float scale = std::bit_cast<float>(uint32_t{scales[parameter]} << 16);
                        const float bias = std::bit_cast<float>(uint32_t{biases[parameter]} << 16);
                        if (!std::isfinite(scale) || !std::isfinite(bias))
                            throw WeightStoreError("experimental FP8 source contains nonfinite affine parameters");
                        std::array<float, 16> levels{};
                        for (uint32_t q = 0; q < levels.size(); ++q)
                            levels[q] = static_cast<float>(q) * scale + bias;
                        for (uint32_t half = 0; half < 2; ++half) {
                            std::array<uint8_t, kBlockElements> quantized{};
                            uint32_t used = 0;
                            uint8_t minimum = 15;
                            uint8_t maximum = 0;
                            for (uint32_t index = 0; index < kBlockElements; ++index) {
                                const uint32_t input = half * kBlockElements + index;
                                const uint8_t byte = packed[parameter * 32 + input / 2];
                                const uint8_t q = (byte >> ((input & 1) * 4)) & 15;
                                quantized[index] = q;
                                used |= 1U << q;
                                minimum = std::min(minimum, q);
                                maximum = std::max(maximum, q);
                            }
                            const float peak = std::max(std::fabs(levels[minimum]),
                                                        std::fabs(levels[maximum]));
                            const uint8_t scaleCode = selectUE8M0Scale(peak);
                            const int scaleExponent = static_cast<int>(scaleCode) - 127;
                            std::array<uint8_t, 16> encoded{};
                            for (uint32_t q = 0; q < encoded.size(); ++q) {
                                if (used & (1U << q))
                                    encoded[q] = encodeE4M3(std::ldexp(levels[q], -scaleExponent));
                            }
                            const uint64_t destination = uint64_t{row} * k + group * 64 +
                                                         half * kBlockElements;
                            for (uint32_t index = 0; index < kBlockElements; ++index)
                                data[destination + index] = encoded[quantized[index]];
                            convertedScales[uint64_t{row} * scaleRowStride + group * 2 + half] = scaleCode;
                        }
                    }
                }
            }
        } catch (...) {
            std::lock_guard lock(failureMutex);
            if (!failure)
                failure = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    };
    {
        std::vector<std::jthread> workers;
        workers.reserve(12);
        for (uint32_t index = 0; index < 12; ++index)
            workers.emplace_back(worker);
    }
    if (failure)
        std::rethrow_exception(failure);

    std::ostringstream canonical;
    canonical << "splash.affine-q4-to-e4m3-ue8m0-block32-half-input-v2\n"
              << "n=" << n << "\nk=" << k << "\nscale_row_stride=" << scaleRowStride
              << "\nrounding=nearest-even\nzero_scale=127\n"
              << "scale_layout=output-major-tight-k-over-32\n"
              << "diagnostic_abi=stride-guard-16-v1\n"
              << "data=" << convertedDigest(data, dataBytes) << '\n'
              << "scales=" << convertedDigest(convertedScales, scaleBytes) << '\n';
    const std::string identity = canonical.str();
    result->fingerprint = convertedDigest(identity.data(), identity.size());
    return result;
}
#endif

#if defined(SPLASH_INT8_EXPERIMENT)
ops::ExperimentalINT8Policy selectedINT8Policy() {
    static const auto policy = [] {
        const char *environment = std::getenv("SPLASH_INT8_POLICY");
        const std::string_view name = environment ? environment : "hybrid";
        if (name == "all") return ops::ExperimentalINT8Policy::All;
        if (name == "target") return ops::ExperimentalINT8Policy::Target;
        if (name == "hybrid") return ops::ExperimentalINT8Policy::Hybrid;
        if (name == "q4") return ops::ExperimentalINT8Policy::Q4;
        throw WeightStoreError("SPLASH_INT8_POLICY must be all, target, hybrid, or q4");
    }();
    return policy;
}

std::string_view int8PolicyName(ops::ExperimentalINT8Policy policy) {
    switch (policy) {
    case ops::ExperimentalINT8Policy::All: return "all";
    case ops::ExperimentalINT8Policy::Target: return "target";
    case ops::ExperimentalINT8Policy::Hybrid: return "hybrid";
    case ops::ExperimentalINT8Policy::Q4: return "q4";
    }
    throw WeightStoreError("invalid INT8 precision policy");
}

enum class INT8ArtifactMode { Off, Read, Write };
struct INT8ArtifactConfiguration final {
    std::filesystem::path directory;
    INT8ArtifactMode mode = INT8ArtifactMode::Off;
};

const INT8ArtifactConfiguration &int8ArtifactConfiguration() {
    static const auto configuration = [] {
        INT8ArtifactConfiguration value;
        const char *directory = std::getenv("SPLASH_PRECONVERTED_INT8_DIR");
        const char *mode = std::getenv("SPLASH_PRECONVERTED_INT8_MODE");
        const std::string_view name = mode ? mode : directory && *directory ? "read" : "off";
        if (name == "off") return value;
        if (name != "read" && name != "write")
            throw WeightStoreError("SPLASH_PRECONVERTED_INT8_MODE must be off, read, or write");
        if (!directory || !*directory)
            throw WeightStoreError("preconverted INT8 read/write mode requires SPLASH_PRECONVERTED_INT8_DIR");
        value.directory = std::filesystem::path(directory);
        value.mode = name == "read" ? INT8ArtifactMode::Read : INT8ArtifactMode::Write;
        return value;
    }();
    return configuration;
}

struct INT8PreconversionCounters final {
    std::mutex mutex;
    INT8PreconversionTelemetry telemetry;
};
INT8PreconversionCounters &int8PreconversionCounters() {
    static INT8PreconversionCounters counters;
    return counters;
}

void recordINT8Preconversion(bool loaded, uint64_t bytes, double seconds) {
    auto &counters = int8PreconversionCounters();
    std::lock_guard lock(counters.mutex);
    if (loaded) {
        ++counters.telemetry.preconvertedProjections;
        counters.telemetry.preconvertedPayloadBytes += bytes;
        counters.telemetry.artifactLoadSeconds += seconds;
    } else {
        ++counters.telemetry.convertedProjections;
        counters.telemetry.convertedPayloadBytes += bytes;
        counters.telemetry.conversionSeconds += seconds;
    }
}

ops::ExperimentalINT8Role int8ProjectionRole(std::string_view path,
                                           std::string_view label) {
    if (path == "target/head.bin" && label == "logits")
        return ops::ExperimentalINT8Role::SharedVocabulary;
    if (path.starts_with("target/"))
        return ops::ExperimentalINT8Role::TargetBody;
    if (path.starts_with("draft/"))
        return ops::ExperimentalINT8Role::DraftBody;
    return ops::ExperimentalINT8Role::Unknown;
}

ops::ExperimentalINT8Kind int8ProjectionKind(std::string_view label) {
    return label == "mlp-down" ? ops::ExperimentalINT8Kind::MLPDown :
                                 ops::ExperimentalINT8Kind::Generic;
}

std::shared_ptr<ops::ExperimentalINT8Workspace>
convertedINT8Workspace(metal::MetalBackend &backend) {
    static std::mutex mutex;
    static std::unordered_map<metal::MetalBackend *,
                              std::weak_ptr<ops::ExperimentalINT8Workspace>> workspaces;
    std::lock_guard lock(mutex);
    for (auto entry = workspaces.begin(); entry != workspaces.end();) {
        if (entry->second.expired())
            entry = workspaces.erase(entry);
        else
            ++entry;
    }
    auto &cached = workspaces[&backend];
    if (auto workspace = cached.lock()) {
        try {
            static_cast<void>(backend.view(workspace->activationCodes, 0,
                                           experimental_int8::kCodeScratchBytes));
            static_cast<void>(backend.view(workspace->activationScales, 0,
                                           experimental_int8::kScaleScratchBytes));
            static_cast<void>(backend.view(workspace->partialPeaks, 0,
                                           experimental_int8::kPartialPeakScratchBytes));
            static_cast<void>(backend.view(workspace->partialInvalid, 0,
                                           experimental_int8::kPartialInvalidScratchBytes));
            return workspace;
        } catch (const metal::MetalBackendError &) {
            if (!backend.healthy())
                throw;
        }
    }
    auto workspace = std::make_shared<ops::ExperimentalINT8Workspace>();
    workspace->activationCodes = backend.allocateBuffer(
        experimental_int8::kCodeScratchBytes, metal::BufferStorage::Shared,
        "experimental INT8 shared activation codes");
    workspace->activationScales = backend.allocateBuffer(
        experimental_int8::kScaleScratchBytes, metal::BufferStorage::Shared,
        "experimental INT8 shared activation row scales");
    workspace->partialPeaks = backend.allocateBuffer(
        experimental_int8::kPartialPeakScratchBytes, metal::BufferStorage::Shared,
        "experimental INT8 shared partial activation peaks");
    workspace->partialInvalid = backend.allocateBuffer(
        experimental_int8::kPartialInvalidScratchBytes, metal::BufferStorage::Shared,
        "experimental INT8 shared partial activation invalid flags");
    cached = workspace;
    return workspace;
}

std::shared_ptr<ops::ExperimentalINT8Projection>
convertAffineQ4INT8(metal::MetalBackend &backend, const ops::Q4Projection &source,
                   std::string_view label) {
    using namespace experimental_int8;
    const auto began = std::chrono::steady_clock::now();
    const uint32_t n = source.outputSize;
    const uint32_t k = source.inputSize;
    if (n > std::numeric_limits<int32_t>::max() ||
        k > std::numeric_limits<int32_t>::max())
        throw WeightStoreError("experimental INT8 dimensions exceed tensor ABI");
    static_cast<void>(convertedProjectionBytes(n, k));
    const uint64_t dataBytes = checkedWeightMultiply(n, k, "INT8 data bytes");
    const uint64_t scaleBytes = checkedWeightMultiply(n, sizeof(float), "INT8 row scale bytes");
    for (const uint64_t bytes : {dataBytes, scaleBytes, kCodeScratchBytes, kScaleScratchBytes,
                                 kPartialPeakScratchBytes, kPartialInvalidScratchBytes}) {
        if (bytes > std::numeric_limits<size_t>::max() ||
            bytes > backend.capabilities().maxBufferLengthBytes ||
            bytes > std::numeric_limits<CC_LONG>::max())
            throw WeightStoreError("experimental INT8 plane exceeds allocation or hash limit");
    }
    const auto *packed = static_cast<const uint8_t *>(source.weights.contents());
    const auto *scales = static_cast<const uint16_t *>(source.scales.contents());
    const auto *biases = static_cast<const uint16_t *>(source.biases.contents());
    if (!packed || !scales || !biases || source.weights.sizeBytes() < dataBytes / 2 ||
        source.scales.sizeBytes() < dataBytes / 32 || source.biases.sizeBytes() < dataBytes / 32)
        throw WeightStoreError("experimental INT8 source is not a complete shared Q4 projection");
    const auto &artifact = int8ArtifactConfiguration();
    preconverted_int8::Identity artifactIdentity{
        n, k, static_cast<uint32_t>(source.int8Policy),
        static_cast<uint32_t>(source.int8Role), static_cast<uint32_t>(source.int8Kind), {}};
    if (artifact.mode != INT8ArtifactMode::Off) {
        artifactIdentity.sourceSha256 = preconverted_int8::sourceDigest(
            {packed, static_cast<size_t>(dataBytes / 2)},
            {reinterpret_cast<const uint8_t *>(scales), static_cast<size_t>(dataBytes / 32)},
            {reinterpret_cast<const uint8_t *>(biases), static_cast<size_t>(dataBytes / 32)});
    }
    auto result = std::make_shared<ops::ExperimentalINT8Projection>();
    const std::string prefix = "experimental INT8 " + std::string(label);
    if (artifact.mode == INT8ArtifactMode::Read) {
        const auto metadata = preconverted_int8::loadMetadata(artifact.directory, artifactIdentity);
        auto mapping = MappedRegion::openReadOnly(metadata.payloadPath);
        if (mapping->bytes() != metadata.payloadBytes)
            throw WeightStoreError("preconverted INT8 payload size changed during loading");
        preconverted_int8::validatePayload(metadata,
            {static_cast<const uint8_t *>(mapping->address()), static_cast<size_t>(mapping->bytes())});
        auto base = backend.wrapSharedMemory(mapping->address(), mapping->bytes(), mapping,
                                            prefix + " preconverted payload");
        result->data = backend.view(base, metadata.dataOffset, metadata.dataBytes);
        result->scales = backend.view(base, metadata.scaleOffset, metadata.scaleBytes);
        result->fingerprint = metadata.projectionFingerprint;
    } else {
        result->data = backend.allocateBuffer(dataBytes, metal::BufferStorage::Shared,
                                              prefix + " data");
        result->scales = backend.allocateBuffer(scaleBytes, metal::BufferStorage::Shared,
                                                prefix + " row scales");
    }
    result->diagnostics = backend.allocateBuffer(16, metal::BufferStorage::Shared,
                                                 prefix + " diagnostics");
    result->sharedWorkspace = convertedINT8Workspace(backend);
    result->activationCodes = result->sharedWorkspace->activationCodes;
    result->activationScales = result->sharedWorkspace->activationScales;
    result->partialPeaks = result->sharedWorkspace->partialPeaks;
    result->partialInvalid = result->sharedWorkspace->partialInvalid;
    result->policy = source.int8Policy;
    result->role = source.int8Role;
    result->kind = source.int8Kind;
    auto *data = static_cast<int8_t *>(result->data.contents());
    auto *convertedScales = static_cast<float *>(result->scales.contents());
    if (!data || !convertedScales || !result->diagnostics.contents())
        throw WeightStoreError("experimental INT8 allocations are not CPU accessible");
    std::memset(result->diagnostics.contents(), 0, 16);

    if (artifact.mode == INT8ArtifactMode::Read) {
        recordINT8Preconversion(true, dataBytes + scaleBytes,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
        return result;
    }

    std::atomic<uint32_t> nextRow{0};
    std::atomic<bool> failed{false};
    std::mutex failureMutex;
    std::exception_ptr failure;
    const uint32_t quantGroups = k / 64;
    const auto worker = [&] {
        try {
            while (!failed.load(std::memory_order_relaxed)) {
                const uint32_t firstRow = nextRow.fetch_add(8, std::memory_order_relaxed);
                if (firstRow >= n)
                    break;
                for (uint32_t row = firstRow; row < std::min(n, firstRow + 8); ++row) {
                    const uint64_t slab = uint64_t{row / kQ4StorageN} * quantGroups * kQ4StorageN;
                    const uint32_t column = row % kQ4StorageN;
                    std::array<uint16_t, kMaximumInputWidth / 64> usedCodes{};
                    float peak = 0.0f;
                    for (uint32_t group = 0; group < quantGroups; ++group) {
                        const uint64_t parameter = slab + uint64_t{group} * kQ4StorageN + column;
                        const float scale = std::bit_cast<float>(uint32_t{scales[parameter]} << 16);
                        const float bias = std::bit_cast<float>(uint32_t{biases[parameter]} << 16);
                        if (!std::isfinite(scale) || !std::isfinite(bias))
                            throw WeightStoreError("experimental INT8 source has nonfinite affine parameters");
                        uint8_t minimum = 15;
                        uint8_t maximum = 0;
                        uint16_t used = 0;
                        for (uint32_t index = 0; index < 32; ++index) {
                            const uint8_t byte = packed[parameter * 32 + index];
                            const uint8_t low = byte & 15;
                            const uint8_t high = byte >> 4;
                            used |= static_cast<uint16_t>((1U << low) | (1U << high));
                            minimum = std::min(minimum, std::min(low, high));
                            maximum = std::max(maximum, std::max(low, high));
                        }
                        const float low = static_cast<float>(minimum) * scale + bias;
                        const float high = static_cast<float>(maximum) * scale + bias;
                        if (!std::isfinite(low) || !std::isfinite(high))
                            throw WeightStoreError("experimental INT8 source has nonfinite materialized weights");
                        peak = std::max(peak, std::max(std::fabs(low), std::fabs(high)));
                        usedCodes[group] = used;
                    }
                    const float rowScale = selectRowScale(peak);
                    convertedScales[row] = rowScale;
                    for (uint32_t group = 0; group < quantGroups; ++group) {
                        const uint64_t parameter = slab + uint64_t{group} * kQ4StorageN + column;
                        const float scale = std::bit_cast<float>(uint32_t{scales[parameter]} << 16);
                        const float bias = std::bit_cast<float>(uint32_t{biases[parameter]} << 16);
                        std::array<int8_t, 16> encoded{};
                        for (uint32_t q = 0; q < encoded.size(); ++q)
                            if (usedCodes[group] & (1U << q))
                                encoded[q] = quantize(static_cast<float>(q) * scale + bias, rowScale);
                        const uint64_t destination = uint64_t{row} * k + group * 64;
                        for (uint32_t index = 0; index < 32; ++index) {
                            const uint8_t byte = packed[parameter * 32 + index];
                            data[destination + index * 2] = encoded[byte & 15];
                            data[destination + index * 2 + 1] = encoded[byte >> 4];
                        }
                    }
                }
            }
        } catch (...) {
            std::lock_guard lock(failureMutex);
            if (!failure)
                failure = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    };
    {
        std::vector<std::jthread> workers;
        workers.reserve(12);
        for (uint32_t index = 0; index < 12; ++index)
            workers.emplace_back(worker);
    }
    if (failure)
        std::rethrow_exception(failure);
    std::ostringstream canonical;
    canonical << "splash.affine-q4-to-whole-row-int8-w8a8-profile-v2\n"
              << "policy=" << int8PolicyName(source.int8Policy)
              << "\nrole=" << static_cast<uint32_t>(source.int8Role)
              << "\nkind=" << static_cast<uint32_t>(source.int8Kind) << '\n'
              << "n=" << n << "\nk=" << k
              << "\nweight_scale=maxabs-whole-output-row-div127\n"
              << "activation_scale=maxabs-whole-input-row-div127\n"
              << "rounding=nearest-even\nclamp=-127,127\nzero_scale=1\n"
              << "layout=output-major-n-k\nphases=prefill-decode-replay\n"
              << "diagnostic_abi=sticky-nonfinite-shape-16-v1\n"
              << "data=" << convertedDigest(data, dataBytes) << '\n'
              << "scales=" << convertedDigest(convertedScales, scaleBytes) << '\n';
    const std::string identity = canonical.str();
    result->fingerprint = convertedDigest(identity.data(), identity.size());
    recordINT8Preconversion(false, dataBytes + scaleBytes,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
    if (artifact.mode == INT8ArtifactMode::Write) {
        static_cast<void>(preconverted_int8::writeAtomically(artifact.directory, artifactIdentity,
            {data, static_cast<size_t>(dataBytes)},
            {convertedScales, static_cast<size_t>(n)}, result->fingerprint));
    }
    return result;
}
#endif

} // namespace

#if defined(SPLASH_METAL41_EXPERIMENT)
uint64_t predictConvertedModelExtraBytes(const ModelDescriptor &descriptor) {
    uint64_t bytes = alignPacked(experimental_fp8::kHalfScratchBytes);
    const auto projection = [&](uint32_t n, uint32_t k) {
        static_cast<void>(experimental_fp8::convertedProjectionBytes(n, k));
        const uint64_t dataBytes = checkedWeightMultiply(n, k, "converted model data bytes");
        const uint64_t scaleBytes = checkedWeightMultiply(
            n, experimental_fp8::convertedScaleRowStride(k), "converted model scale bytes");
        bytes = checkedWeightAdd(bytes, alignPacked(dataBytes),
                                 "converted model data allocation");
        bytes = checkedWeightAdd(bytes, alignPacked(scaleBytes),
                                 "converted model scale allocation");
        // The 16-byte guard receives its own physical Shared allocation.
        bytes = checkedWeightAdd(bytes, kWeightFileAlignment,
                                 "converted projection diagnostic allocation");
    };
    std::visit([&](const auto &layout) {
        for (uint32_t layer = 0; layer < layout.layers; ++layer) {
            projection(layout.isFullAttentionLayer(layer) ? layout.packedFullWidth :
                                                           layout.packedGdnWidth,
                       layout.hiddenSize);
            projection(layout.hiddenSize, layout.attentionWidth);
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(layout)>, Qwen3_8Layout>) {
                projection(layout.intermediateSize, layout.hiddenSize);
                projection(layout.intermediateSize, layout.hiddenSize);
                projection(layout.hiddenSize, layout.intermediateSize);
            }
        }
        projection(layout.vocabularySize, layout.hiddenSize);
    }, descriptor.target);
    const auto &draft = descriptor.draft;
    for (uint32_t layer = 0; layer < draft.layers; ++layer) {
        projection(draft.dynamicSize, draft.hiddenSize);
        projection(draft.qkvSize, draft.hiddenSize);
        projection(draft.hiddenSize, draft.attentionSize);
        projection(draft.dynamicSize, draft.hiddenSize);
        projection(draft.intermediateSize, draft.hiddenSize);
        projection(draft.intermediateSize, draft.hiddenSize);
        projection(draft.hiddenSize, draft.intermediateSize);
    }
    projection(draft.hiddenSize, draft.targetHiddenSize);
    projection(draft.selectorRank, draft.hiddenSize);
    return bytes;
}
#endif

#if defined(SPLASH_INT8_EXPERIMENT)
INT8PreconversionTelemetry int8PreconversionTelemetry() {
    auto &counters = int8PreconversionCounters();
    std::lock_guard lock(counters.mutex);
    return counters.telemetry;
}

uint64_t predictINT8ModelExtraBytes(const ModelDescriptor &descriptor) {
    const auto policy = selectedINT8Policy();
    uint64_t bytes = 0;
    bool hasConversion = false;
    const auto projection = [&](uint32_t n, uint32_t k, ops::ExperimentalINT8Role role,
                                ops::ExperimentalINT8Kind kind = ops::ExperimentalINT8Kind::Generic) {
        if (!ops::experimentalINT8Eligible(policy, role, kind))
            return;
        hasConversion = true;
        static_cast<void>(experimental_int8::convertedProjectionBytes(n, k));
        const uint64_t dataBytes = checkedWeightMultiply(n, k, "INT8 model data bytes");
        const uint64_t scaleBytes = checkedWeightMultiply(n, sizeof(float), "INT8 model row scale bytes");
        bytes = checkedWeightAdd(bytes, alignPacked(dataBytes), "INT8 model data allocation");
        bytes = checkedWeightAdd(bytes, alignPacked(scaleBytes), "INT8 model scale allocation");
        bytes = checkedWeightAdd(bytes, kWeightFileAlignment, "INT8 diagnostic allocation");
    };
    std::visit([&](const auto &layout) {
        const auto role = ops::ExperimentalINT8Role::TargetBody;
        for (uint32_t layer = 0; layer < layout.layers; ++layer) {
            projection(layout.isFullAttentionLayer(layer) ? layout.packedFullWidth : layout.packedGdnWidth,
                       layout.hiddenSize, role);
            projection(layout.hiddenSize, layout.attentionWidth, role);
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(layout)>, Qwen3_8Layout>) {
                projection(layout.intermediateSize, layout.hiddenSize, role);
                projection(layout.intermediateSize, layout.hiddenSize, role);
                projection(layout.hiddenSize, layout.intermediateSize, role,
                           ops::ExperimentalINT8Kind::MLPDown);
            }
        }
        projection(layout.vocabularySize, layout.hiddenSize,
                   ops::ExperimentalINT8Role::SharedVocabulary);
    }, descriptor.target);
    const auto &draft = descriptor.draft;
    const auto role = ops::ExperimentalINT8Role::DraftBody;
    for (uint32_t layer = 0; layer < draft.layers; ++layer) {
        projection(draft.dynamicSize, draft.hiddenSize, role);
        projection(draft.qkvSize, draft.hiddenSize, role);
        projection(draft.hiddenSize, draft.attentionSize, role);
        projection(draft.dynamicSize, draft.hiddenSize, role);
        projection(draft.intermediateSize, draft.hiddenSize, role);
        projection(draft.intermediateSize, draft.hiddenSize, role);
        projection(draft.hiddenSize, draft.intermediateSize, role,
                   ops::ExperimentalINT8Kind::MLPDown);
    }
    projection(draft.hiddenSize, draft.targetHiddenSize, role);
    projection(draft.selectorRank, draft.hiddenSize, role);
    if (hasConversion) {
        bytes = checkedWeightAdd(bytes, alignPacked(experimental_int8::kCodeScratchBytes),
                                 "INT8 shared activation code allocation");
        bytes = checkedWeightAdd(bytes, alignPacked(experimental_int8::kScaleScratchBytes),
                                 "INT8 shared activation scale allocation");
        bytes = checkedWeightAdd(bytes, alignPacked(experimental_int8::kPartialPeakScratchBytes),
                                 "INT8 shared partial peak allocation");
        bytes = checkedWeightAdd(bytes, alignPacked(experimental_int8::kPartialInvalidScratchBytes),
                                 "INT8 shared partial invalid allocation");
    }
    return bytes;
}
#endif
#endif

ops::Q4Projection readQ4Projection(WeightFile &file,
                                   metal::MetalBackend &backend,
                                   uint32_t outputSize,
                                   uint32_t inputSize,
                                   std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const uint64_t weightBytes = elements / 2;
    const uint64_t parameterBytes = elements / 32;
    metal::MetalBuffer packed =
        file.section(q4PackedBytes(outputSize, inputSize), label);
    ops::Q4Projection result{
        backend.view(packed, 0, weightBytes),
        backend.view(packed, weightBytes, parameterBytes),
        backend.view(packed, weightBytes + parameterBytes, parameterBytes),
        outputSize,
        inputSize,
    };
#if defined(SPLASH_METAL41_EXPERIMENT)
    result.experimentalFP8 = convertAffineQ4(backend, result, label);
    file.recordConvertedFingerprint(label, result.experimentalFP8->fingerprint);
#endif
#if defined(SPLASH_INT8_EXPERIMENT)
    result.int8Policy = selectedINT8Policy();
    result.int8Role = int8ProjectionRole(file.record().relativePath, label);
    result.int8Kind = int8ProjectionKind(label);
    if (ops::experimentalINT8Eligible(result.int8Policy, result.int8Role, result.int8Kind)) {
        result.experimentalINT8 = convertAffineQ4INT8(backend, result, label);
        file.recordConvertedFingerprint(label, result.experimentalINT8->fingerprint);
    }
#endif
    return result;
}

ops::Q4Projection readQ4ProjectionComponents(WeightFile &file,
                                             uint32_t outputSize,
                                             uint32_t inputSize,
                                             std::string_view label) {
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const std::string prefix(label);
    return {
        file.section(elements / 2, prefix + "-weights"),
        file.section(elements / 32, prefix + "-scales"),
        file.section(elements / 32, prefix + "-biases"),
        outputSize,
        inputSize,
    };
}

ops::Q8Projection readQ8Projection(WeightFile &file,
                                   metal::MetalBackend &backend,
                                   uint32_t outputSize,
                                   uint32_t inputSize,
                                   std::string_view label) {
    validateQ4Layout(outputSize, inputSize);
    const uint64_t elements = q4Elements(outputSize, inputSize);
    const uint64_t parameterBytes = elements / 32;
    metal::MetalBuffer packed =
        file.section(q8PackedBytes(outputSize, inputSize), label);
    return {
        backend.view(packed, 0, elements),
        backend.view(packed, elements, parameterBytes),
        backend.view(packed, elements + parameterBytes, parameterBytes),
        outputSize,
        inputSize,
    };
}

ops::ExpertQ4Projection
readExpertQ4Projection(WeightFile &file, uint32_t experts,
                       uint32_t outputSize, uint32_t inputSize,
                       std::string_view label) {
    if (!experts)
        throw WeightStoreError("expert projection requires experts");
    validateQ4Layout(outputSize, inputSize);
    const uint64_t stride = q4PackedBytes(outputSize, inputSize);
    return {
        file.section(checkedWeightMultiply(experts, stride,
                                           "expert Q4 slab bytes"),
                     label),
        experts,
        outputSize,
        inputSize,
        stride,
    };
}

std::vector<metal::MetalBuffer>
immutableWeightBuffers(const ModelPackage &package) {
    std::vector<metal::MetalBuffer> buffers;
    const auto add = [&](const metal::MetalBuffer &buffer) {
        if (!buffer || buffer.storage() != metal::BufferStorage::Shared)
            throw WeightStoreError("model residency requires loaded shared weight buffers");
        buffers.push_back(buffer);
    };
    const auto converted = [&](const ops::Q4Projection &projection) {
#if defined(SPLASH_METAL41_EXPERIMENT)
        if (projection.experimentalFP8) {
            add(projection.experimentalFP8->data);
            add(projection.experimentalFP8->scales);
        }
#endif
#if defined(SPLASH_INT8_EXPERIMENT)
        if (projection.experimentalINT8) {
            add(projection.experimentalINT8->data);
            add(projection.experimentalINT8->scales);
        }
#endif
#if !defined(SPLASH_METAL41_EXPERIMENT) && !defined(SPLASH_INT8_EXPERIMENT)
        static_cast<void>(projection);
#endif
    };

    // Each supported target loader maps one file per layer, then head.bin and
    // embedding.bin. A norm/embedding view retains that complete allocation,
    // including every other packed section. Check the schema rather than
    // silently omitting files if a loader gains another backing allocation.
    std::visit([&](const auto &weights) {
        if (weights.layers.size() != weights.layout.layers ||
            weights.files.size() != weights.layers.size() + 2)
            throw WeightStoreError("target weight file schema is incomplete for residency");
        for (const auto &layer : weights.layers) {
            add(layer.inputNorm);
            std::visit([&](const auto &mixer) {
                converted(mixer.inputProjection);
                converted(mixer.outputProjection);
            }, layer.mixer);
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(layer)>,
                                         Qwen3_8LayerWeights>) {
                converted(layer.gateProjection);
                converted(layer.upProjection);
                converted(layer.downProjection);
            }
            // MoE Q8 routing weights and expert slabs stay in the same layer
            // file as inputNorm, so the mapped representative includes them.
        }
        add(weights.finalNorm);
        add(weights.tokenEmbedding.weights);
        converted(weights.logitsProjection);
        converted(weights.tokenEmbedding);
    }, package.target);

    const auto &draft = package.draft;
    if (draft.layers.size() != draft.layout.layers ||
        draft.files.size() != draft.layers.size() + 1)
        throw WeightStoreError("draft weight file schema is incomplete for residency");
    for (const auto &layer : draft.layers) {
        add(layer.inputNorm);
        converted(layer.attentionDynamic);
        converted(layer.qkvProjection);
        converted(layer.outputProjection);
        converted(layer.mlpDynamic);
        converted(layer.gateProjection);
        converted(layer.upProjection);
        converted(layer.downProjection);
    }
    add(draft.hiddenNorm);
    converted(draft.contextProjection);
    converted(draft.selectorProjection);

    // The vision loader puts all tower/merger tensors in vision/model.bin.
    if (package.vision.files.size() != 1 ||
        package.vision.tensors.blocks.size() != package.vision.tensors.layout.depth)
        throw WeightStoreError("vision weight file schema is incomplete for residency");
    add(package.vision.tensors.patchEmbedding.weight);
    return buffers;
}

std::string weightManifestFingerprint(
    std::span<const WeightFileRecord> records) {
    std::vector<WeightFileRecord> sorted(records.begin(), records.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const WeightFileRecord &left,
                 const WeightFileRecord &right) {
                  return left.relativePath < right.relativePath;
              });
    std::ostringstream canonical;
    canonical << "splash-packed-manifest-v1\n";
#if defined(SPLASH_INT8_EXPERIMENT)
    // This also salts the q4 profile, which has no converted planes. One
    // immutable process policy reaches both target and combined identities.
    canonical << "int8_precision_profile_v2=" << int8PolicyName(selectedINT8Policy()) << '\n';
    canonical << "selection=all|target-and-shared-head|target-mlp-down|q4\n";
#endif
    for (const WeightFileRecord &record : sorted) {
        canonical << record.relativePath << '\t' << record.declaredBytes
                  << '\t' << record.magic << '\t' << record.layer << '\t'
                  << record.type << '\n';
#if defined(SPLASH_METAL41_EXPERIMENT)
        for (const auto &converted : record.convertedFingerprints)
            canonical << "converted_fp8=" << converted.size() << ':' << converted << '\n';
#endif
#if defined(SPLASH_INT8_EXPERIMENT)
        for (const auto &converted : record.convertedFingerprints)
            canonical << "converted_int8=" << converted.size() << ':' << converted << '\n';
#endif
    }
    std::string value = canonical.str();
    if (value.size() > std::numeric_limits<CC_LONG>::max()) {
        throw WeightStoreError("manifest is too large to fingerprint");
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    if (!CC_SHA256(value.data(), static_cast<CC_LONG>(value.size()),
                   digest.data())) {
        throw WeightStoreError("unable to calculate manifest SHA-256");
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2);
    for (unsigned char byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
    }
    return result;
}

} // namespace splash::model
