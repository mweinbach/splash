#include "dev/benchmarks/current_batch_sep22/NativeLifecycleTrace.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using splash::flash::batch_clock_sep22::Trace;
int main(int argc, char **argv) {
  if (argc != 2) throw std::invalid_argument("trace-cpu FRESH_TRACE_PATH");
  constexpr const char *marker = "SPLASH_FLASH_NATIVE_LIFECYCLE_TIMESTAMPS_SEP22";
  constexpr const char *pathFlag = "SPLASH_FLASH_NATIVE_LIFECYCLE_TRACE_SEP22";
  unsetenv(marker); unsetenv(pathFlag);
  if (Trace::fromEnvironment()) throw std::logic_error("absent marker created trace");
  uint32_t checks = 1;
  setenv(marker, "0", 1);
  if (Trace::fromEnvironment()) throw std::logic_error("disabled marker created trace");
  ++checks;
  for (const char *invalid : {"", "2", "true", " 1"}) {
    setenv(marker, invalid, 1);
    bool rejected = false;
    try { (void)Trace::fromEnvironment(); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::logic_error("invalid timestamp marker accepted");
    ++checks;
  }
  setenv(marker, "1", 1);
  bool rejected = false;
  try { (void)Trace::fromEnvironment(); }
  catch (const std::invalid_argument &) { rejected = true; }
  if (!rejected) throw std::logic_error("missing native trace path accepted");
  ++checks;
  setenv(pathFlag, argv[1], 1);
  {
    auto trace = Trace::fromEnvironment();
    trace->setSourceIdentity(std::string(64, 'a'));
    trace->record("first_emission", 1, 1, 1, 1000000000, 1, 2048);
    trace->record("done", 1, 1, 1, 2000000000, 256, 2048, "length");
    if (std::filesystem::file_size(argv[1]) != 0)
      throw std::logic_error("metadata capture wrote file before serving stopped");
    ++checks;
    trace->finish();
  }
  std::ifstream traceFile(argv[1]);
  std::string line; uint32_t lines = 0;
  while (std::getline(traceFile, line)) {
    if (line.find("native-worker-std-steady-nanoseconds-v1") == std::string::npos ||
        line.find(std::string(64, 'a')) == std::string::npos)
      throw std::logic_error("native trace omitted clock/source provenance");
    ++lines;
  }
  if (lines != 3) throw std::logic_error("native trace record count/footer differs");
  ++checks;
  rejected = false;
  try { (void)Trace::fromEnvironment(); }
  catch (const std::invalid_argument &) { rejected = true; }
  if (!rejected) throw std::logic_error("existing native trace path accepted");
  ++checks;
  unsetenv(marker); unsetenv(pathFlag);
  std::cout << "{\"valid\":true,\"checks\":" << checks
      << ",\"gpu_executed\":false,\"model_payload_bytes_read\":0}\n";
}
