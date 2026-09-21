#pragma once

#include "metal/MetalBackend.hpp"
#include <array>
#include <filesystem>
#include <memory>
#include <string>

namespace splash::flash::candidate {

inline constexpr const char *kExactExpertLUTLayout =
    "E,Nblock64,Kgroup64,Nlane64,Q16";
inline constexpr const char *kExactExpertLUTPolicy =
    "contract-off-F32-multiply-then-F32-add-once-rounded-BF16-RNE-v1";

// CPU-only file, SHA256, geometry and finite checks. No backend is constructed
// and no GPU command is submitted. Immutable mmap lifetime survives all views.
class ExpertLUTSidecar final {
public:
  static ExpertLUTSidecar load(const std::filesystem::path &directory,
                             const std::filesystem::path &sourcePackage,
                             const std::string &expectedPrefix);
  [[nodiscard]] std::array<metal::MetalBuffer, 3>
  buffers(metal::MetalBackend &backend) const;
  [[nodiscard]] const std::string &manifestSHA() const { return manifestSHA_; }
  [[nodiscard]] const std::string &sourceIdentity() const { return sourceIdentity_; }
  [[nodiscard]] uint64_t savedBytes() const { return 3 * 419430400ULL; }
private:
  struct Plane;
  std::array<std::shared_ptr<Plane>, 3> planes_;
  std::string manifestSHA_;
  std::string sourceIdentity_;
};

struct LUTProofParams {
  uint32_t outputSize, inputSize;
  uint64_t parameterRowStrideBytes, parameterExpertStrideBytes;
};
static_assert(sizeof(LUTProofParams) == 24);

} // namespace splash::flash::candidate
