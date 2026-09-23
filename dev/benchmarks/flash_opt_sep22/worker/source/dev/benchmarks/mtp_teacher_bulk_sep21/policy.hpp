#pragma once
#include <stdexcept>
#include <string_view>
namespace splash::flash::teacher_bulk_sep21 {
inline constexpr const char *flag="SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21";
[[nodiscard]] inline bool parse(const char *value) {
  if(!value || std::string_view(value)=="0")return false;
  if(std::string_view(value)=="1")return true;
  throw std::invalid_argument("SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21 must be 0 or 1");
}
struct Dependencies {bool mtp,teacherCacheOnly,denseCache,headF32,headMPP;};
inline void validate(bool requested,Dependencies dependencies) {
  if(requested && (!dependencies.mtp || !dependencies.teacherCacheOnly ||
      !dependencies.denseCache || !dependencies.headF32 || !dependencies.headMPP))
    throw std::invalid_argument("singleton teacher bulk requires original MTP/cache-only/dense/F32/MPP routes");
}
}
