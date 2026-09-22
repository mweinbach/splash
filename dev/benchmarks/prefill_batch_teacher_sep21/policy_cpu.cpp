#include "flash/FlashBatchMTPForward.hpp"
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using splash::flash::flashPrivateBatchMTPTeacherCacheOnlyEnabled;
int main() {
  constexpr const char *flag = "SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY";
  constexpr std::array dependencies{"SPLASH_FLASH_MTP", "SPLASH_FLASH_BATCH_PREFILL",
      "SPLASH_FLASH_BATCH_MTP_PREFILL"};
  uint32_t checks = 0, refusals = 0;
  splash::flash::FlashMTPState uninitialized;
  if (uninitialized.logicalLength() != 0 || uninitialized.capacity() != 0 || !uninitialized.poisoned())
    throw std::logic_error("uninitialized MTP state getter contract differs");
  ++checks;
  for (const char *dependency : dependencies) unsetenv(dependency);
  unsetenv(flag);
  if (flashPrivateBatchMTPTeacherCacheOnlyEnabled()) throw std::logic_error("absent flag must be disabled");
  ++checks;
  setenv(flag, "0", 1);
  if (flashPrivateBatchMTPTeacherCacheOnlyEnabled()) throw std::logic_error("zero flag must be disabled");
  ++checks;
  for (uint32_t mask = 0; mask < 8; ++mask) {
    for (uint32_t index = 0; index < dependencies.size(); ++index)
      setenv(dependencies[index], mask & (1U << index) ? "1" : "0", 1);
    setenv(flag, "1", 1);
    bool rejected = false, result = false;
    try { result = flashPrivateBatchMTPTeacherCacheOnlyEnabled(); }
    catch (const std::invalid_argument &) { rejected = true; ++refusals; }
    if (mask == 7 ? (rejected || !result) : (!rejected || result))
      throw std::logic_error("batch teacher dependency guard differs");
    ++checks;
  }
  for (const char *invalid : {"", "2", "true", "false", " 1", "1 ", "-1"}) {
    setenv(flag, invalid, 1);
    bool rejected = false;
    try { (void)flashPrivateBatchMTPTeacherCacheOnlyEnabled(); }
    catch (const std::invalid_argument &) { rejected = true; ++refusals; }
    if (!rejected) throw std::logic_error("batch teacher flag accepted non-binary value");
    ++checks;
  }
  for (const char *dependency : dependencies) unsetenv(dependency);
  unsetenv(flag);
  std::cout << "{\"valid\":true,\"gpu_work\":false,\"model_loaded\":false,\"checks\":"
      << checks << ",\"strict_refusals\":" << refusals << "}\n";
}
