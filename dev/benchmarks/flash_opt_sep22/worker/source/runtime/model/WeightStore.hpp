#pragma once

#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

inline constexpr uint32_t kQ4GroupElements = 64;
inline constexpr uint64_t kBFloat16Bytes = 2;

inline constexpr uint64_t kWeightFileAlignment = 16 * 1024;
inline constexpr uint32_t kQ4StorageN = 256;

class WeightStoreError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct WeightFileRecord final {
  std::string relativePath;
  std::string magic;
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t declaredBytes = 0;
#if defined(SPLASH_METAL41_EXPERIMENT) || defined(SPLASH_INT8_EXPERIMENT)
  std::vector<std::string> convertedFingerprints{};
#endif
};

// A read-only mmap with one no-copy Metal base buffer.  Sections are checked,
// aligned views that retain the mapping; no model loader owns raw mmap state.
class WeightFile final {
public:
  WeightFile(metal::MetalBackend &backend, std::filesystem::path path,
             std::string relativePath, std::string_view expectedMagic,
             uint32_t expectedLayer, uint32_t expectedType);
  ~WeightFile();

  WeightFile(const WeightFile &) = delete;
  WeightFile &operator=(const WeightFile &) = delete;

  [[nodiscard]] metal::MetalBuffer section(uint64_t bytes,
                                            std::string_view label = {});
  void finish();
  [[nodiscard]] const WeightFileRecord &record() const noexcept;
#if defined(SPLASH_METAL41_EXPERIMENT) || defined(SPLASH_INT8_EXPERIMENT)
  void recordConvertedFingerprint(std::string_view label,
                                  std::string_view fingerprint);
#endif

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t checkedWeightMultiply(uint64_t left, uint64_t right,
                                             std::string_view description);
[[nodiscard]] uint64_t q4PackedBytes(uint32_t outputSize,
                                     uint32_t inputSize);
void validateQ4Layout(uint32_t outputSize, uint32_t inputSize);

[[nodiscard]] ops::Q4Projection
readQ4Projection(WeightFile &file, metal::MetalBackend &backend,
                 uint32_t outputSize, uint32_t inputSize,
                 std::string_view label);

// Embedding weights, scales and biases are independently aligned sections
// so token gather can bind each table directly.
[[nodiscard]] ops::Q4Projection
readQ4ProjectionComponents(WeightFile &file, uint32_t outputSize,
                           uint32_t inputSize, std::string_view label);

[[nodiscard]] ops::Q8Projection
readQ8Projection(WeightFile &file, metal::MetalBackend &backend,
                 uint32_t outputSize, uint32_t inputSize,
                 std::string_view label);

[[nodiscard]] ops::ExpertQ4Projection
readExpertQ4Projection(WeightFile &file, uint32_t experts,
                       uint32_t outputSize, uint32_t inputSize,
                       std::string_view label);

[[nodiscard]] std::string
weightManifestFingerprint(std::span<const WeightFileRecord> records);

struct ModelPackage;
// Existing views representing every mapped file allocation, plus immutable
// converted coefficient data/scales. The backend must resolve whole allocation
// identity/size: a representative may cover only one tensor in its mapped file.
// Mutable conversion workspaces, diagnostics, KV and state are excluded.
[[nodiscard]] std::vector<metal::MetalBuffer>
immutableWeightBuffers(const ModelPackage &package);

#if defined(SPLASH_METAL41_EXPERIMENT) || defined(SPLASH_INT8_EXPERIMENT)
struct ModelDescriptor;
#endif
#if defined(SPLASH_METAL41_EXPERIMENT)
// Additional persistent conversion planes plus the one shared HALF workspace.
// Original mapped Q4 files remain resident and are accounted separately.
[[nodiscard]] uint64_t
predictConvertedModelExtraBytes(const ModelDescriptor &descriptor);
#endif
#if defined(SPLASH_INT8_EXPERIMENT)
struct INT8PreconversionTelemetry final {
  uint64_t convertedProjections = 0;
  uint64_t preconvertedProjections = 0;
  uint64_t convertedPayloadBytes = 0;
  uint64_t preconvertedPayloadBytes = 0;
  double conversionSeconds = 0.0;
  double artifactLoadSeconds = 0.0;
};
[[nodiscard]] INT8PreconversionTelemetry int8PreconversionTelemetry();
[[nodiscard]] uint64_t
predictINT8ModelExtraBytes(const ModelDescriptor &descriptor);
#endif

} // namespace splash::model
