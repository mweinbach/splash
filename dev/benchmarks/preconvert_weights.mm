// Prepare or audit local derivative weights with the actual production loader.
#include "engine/Json.hpp"
#include "model/ModelFactory.hpp"
#include "model/WeightStore.hpp"

#import <Foundation/Foundation.h>
#include <mach-o/dyld.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

#if !defined(SPLASH_INT8_EXPERIMENT)
#error "preconvert-weights requires the INT8/hybrid build"
#endif

namespace {

bool isWithin(const std::filesystem::path &path,
              const std::filesystem::path &directory) {
  auto component = path.begin();
  for (const auto &parent : directory) {
    if (component == path.end() || *component != parent)
      return false;
    ++component;
  }
  return true;
}

bool aliasesFile(const std::filesystem::path &path,
                 const std::filesystem::path &protectedFile) {
  return std::filesystem::exists(path) &&
         std::filesystem::exists(protectedFile) &&
         std::filesystem::equivalent(path, protectedFile);
}

bool aliasesTreeFile(const std::filesystem::path &path,
                     const std::filesystem::path &root) {
  std::vector<std::filesystem::path> pending{root};
  std::set<std::filesystem::path> visited;
  while (!pending.empty()) {
    const auto directory = std::filesystem::canonical(pending.back());
    pending.pop_back();
    if (!visited.insert(directory).second)
      continue;
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
      if (entry.is_directory())
        pending.push_back(entry.path());
      else if (entry.is_regular_file() && aliasesFile(path, entry.path()))
        return true;
    }
  }
  return false;
}

std::filesystem::path executablePath() {
  std::vector<char> path(1024);
  uint32_t bytes = static_cast<uint32_t>(path.size());
  if (_NSGetExecutablePath(path.data(), &bytes)) {
    path.resize(bytes);
    if (_NSGetExecutablePath(path.data(), &bytes))
      throw std::runtime_error("could not resolve the converter executable path");
  }
  return std::filesystem::canonical(path.data());
}

void validatePaths(const std::filesystem::path &model,
                   const std::filesystem::path &artifacts,
                   const std::filesystem::path &report,
                   const std::filesystem::path &metallib) {
  if (model.empty() || artifacts.empty() || report.empty() || metallib.empty())
    throw std::invalid_argument("model, artifact, report and metallib paths must be nonempty");
  const auto modelRoot = std::filesystem::canonical(model);
  if (!std::filesystem::is_directory(modelRoot))
    throw std::invalid_argument("original model root must be a directory");
  const auto artifactRoot =
      std::filesystem::weakly_canonical(std::filesystem::absolute(artifacts));
  if (std::filesystem::exists(artifactRoot) &&
      !std::filesystem::is_directory(artifactRoot))
    throw std::invalid_argument("artifact directory path must be a directory");
  const auto reportPath =
      std::filesystem::weakly_canonical(std::filesystem::absolute(report));
  // Check lexical locations too: a report entered beneath a HF snapshot can
  // resolve through a source-file symlink into the Hub's external blob store.
  const auto lexicalModel = std::filesystem::absolute(model).lexically_normal();
  const auto lexicalArtifacts = std::filesystem::absolute(artifacts).lexically_normal();
  const auto lexicalReport = std::filesystem::absolute(report).lexically_normal();
  if (isWithin(artifactRoot, modelRoot) || isWithin(lexicalArtifacts, lexicalModel) ||
      isWithin(lexicalArtifacts, modelRoot))
    throw std::invalid_argument("artifact directory must be outside the original model root");
  if (isWithin(reportPath, modelRoot) || isWithin(lexicalReport, lexicalModel) ||
      isWithin(lexicalReport, modelRoot))
    throw std::invalid_argument("report must be outside the original model root");
  if (isWithin(reportPath, artifactRoot) || isWithin(lexicalReport, lexicalArtifacts))
    throw std::invalid_argument("report must be outside the derivative artifact directory");
  if (aliasesFile(report, metallib) || aliasesFile(report, executablePath()))
    throw std::invalid_argument("report must not alias the metallib or converter executable");
  // Existing hard links or direct blob-store paths can alias source files even
  // when both report locations above are outside the package directory. Follow
  // component-directory symlinks, visiting each canonical directory only once.
  if (std::filesystem::exists(report)) {
    if (aliasesTreeFile(report, modelRoot))
      throw std::invalid_argument("report must not alias an original model file");
    if (std::filesystem::is_directory(artifactRoot) && aliasesTreeFile(report, artifactRoot))
      throw std::invalid_argument("report must not alias a derivative artifact file");
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 6)
        throw std::invalid_argument("usage: preconvert-weights METALLIB MODEL_ROOT ARTIFACT_DIR read|write|off REPORT_JSON");
      const std::string_view mode(argv[4]);
      if (mode != "read" && mode != "write" && mode != "off")
        throw std::invalid_argument("mode must be read, write, or off");
      validatePaths(argv[2], argv[3], argv[5], argv[1]);
      if (::setenv("SPLASH_PRECONVERTED_INT8_DIR", argv[3], 1) != 0 ||
          ::setenv("SPLASH_PRECONVERTED_INT8_MODE", argv[4], 1) != 0)
        throw std::runtime_error("could not set derivative artifact configuration");
      const auto began = std::chrono::steady_clock::now();
      splash::metal::MetalBackend backend(argv[1]);
      const auto package = splash::model::loadModelPackage(backend, argv[2]);
      const auto elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - began).count();
      const auto stats = splash::model::int8PreconversionTelemetry();
      if (mode == "read" && (stats.convertedProjections ||
          (splash::model::predictINT8ModelExtraBytes(package.descriptor) &&
           !stats.preconvertedProjections)))
        throw std::runtime_error("strict artifact load did not skip all weight conversion");
      std::ofstream out(argv[5]);
      if (!out) throw std::runtime_error("could not open report output");
      out << std::setprecision(17) << "{\"pass\":true,\"mode\":"
          << splash::json::quote(mode) << ",\"model\":"
          << splash::json::quote(std::filesystem::canonical(argv[2]).string())
          << ",\"artifact_directory\":" << splash::json::quote(std::filesystem::absolute(argv[3]).string())
          << ",\"package_fingerprint\":" << splash::json::quote(package.manifestFingerprintSha256)
          << ",\"total_loader_seconds\":" << elapsed
          << ",\"converted_projections\":" << stats.convertedProjections
          << ",\"preconverted_projections\":" << stats.preconvertedProjections
          << ",\"converted_payload_bytes\":" << stats.convertedPayloadBytes
          << ",\"preconverted_payload_bytes\":" << stats.preconvertedPayloadBytes
          << ",\"conversion_seconds\":" << stats.conversionSeconds
          << ",\"artifact_load_seconds\":" << stats.artifactLoadSeconds
          << ",\"allocated_bytes\":" << backend.memoryStats().allocatedBytes << "}\n";
      out.flush();
      if (!out) throw std::runtime_error("could not write the report output");
      out.close();
      if (!out) throw std::runtime_error("could not close the report output");
      std::cerr << "preconverted_weights mode=" << mode
          << " converted=" << stats.convertedProjections
          << " cached=" << stats.preconvertedProjections
          << " loader_seconds=" << elapsed << '\n';
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "preconvert-weights: " << error.what() << '\n';
      return 1;
    }
  }
}
