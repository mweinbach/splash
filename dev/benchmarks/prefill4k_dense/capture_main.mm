#include "capture.hpp"
#define main prefill4k_attribution_main
#include "../prefill4k_attribution.mm"
#undef main

namespace prefill4k_dense {
void writeCaptures() {
  const char *raw = std::getenv("PREFILL4K_DENSE_CAPTURE");
  if (!raw || !*raw) return;
  const auto directory = std::filesystem::absolute(raw);
  require(captures.size() == 10 && captured.size() == 10,"dense capture is missing measured roles");
  require(!std::filesystem::exists(directory),"dense capture destination must be fresh");
  std::filesystem::create_directories(directory);
  std::ofstream manifest(directory / "manifest.json");
  require(bool(manifest), "cannot write dense capture manifest");
  manifest << "{\"schema\":\"splash-prefill4k-dense-fixtures-v1\",\"actual_activations\":true,\"cases\":[";
  bool first = true;
  for (size_t i = 0; i < captures.size(); ++i) {
    const auto &capture = captures[i];
    const auto inputPath = directory / ("input-" + std::to_string(i) + ".bin");
    const auto weightPath = directory / ("weights-" + std::to_string(i) + ".bin");
    const auto outputPath = directory / ("output-" + std::to_string(i) + ".bin");
    const auto write = [](const std::filesystem::path &path, const void *data, uint64_t bytes) {
      std::ofstream file(path,std::ios::binary); require(bool(file), "cannot write dense capture");
      file.write(static_cast<const char *>(data),std::streamsize(bytes));
      require(bool(file), "dense capture write incomplete");
    };
    const uint64_t coefficientBytes = uint64_t(capture.inputs)*capture.outputs*2;
    write(inputPath,capture.input.contents(),capture.input.sizeBytes());
    write(weightPath,capture.weights.contents(),coefficientBytes);
    write(outputPath,capture.output.contents(),capture.output.sizeBytes());
    if (!first) manifest << ','; first = false;
    manifest << "{\"projection\":" << splash::json::quote(capture.projection)
        << ",\"rows\":" << capture.rows << ",\"input_size\":" << capture.inputs
        << ",\"output_size\":" << capture.outputs << ",\"weights_file\":"
        << splash::json::quote(weightPath.string()) << ",\"weights_sha256\":"
        << splash::json::quote(digest(capture.weights.contents(),coefficientBytes))
        << ",\"input_file\":" << splash::json::quote(inputPath.string())
        << ",\"input_sha256\":" << splash::json::quote(digest(capture.input.contents(),capture.input.sizeBytes()))
        << ",\"expected_file\":" << splash::json::quote(outputPath.string())
        << ",\"expected_sha256\":" << splash::json::quote(digest(capture.output.contents(),capture.output.sizeBytes()))
        << '}';
  }
  manifest << "]}\n"; require(bool(manifest), "dense capture manifest incomplete");
  std::cout << "dense actual activation captures=" << captures.size() << " path=" << directory << '\n';
}
} // namespace prefill4k_dense

int main(int argc,char **argv) {
  const int result = prefill4k_attribution_main(argc,argv);
  if (result) return result;
  try { prefill4k_dense::writeCaptures(); return 0; }
  catch (const std::exception &error) { std::cerr << "dense capture write failed: " << error.what() << '\n'; return 1; }
}
