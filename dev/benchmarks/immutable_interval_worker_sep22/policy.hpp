#pragma once
#include "source_identity.hpp"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::flash::immutable_interval_worker_sep22 {
inline constexpr const char*flag="SPLASH_FLASH_IMMUTABLE_INTERVAL_INDEX_SEP22";
inline constexpr const char*scope="CPU-only immutable96 alias lookup; positive bounded disjoint accepts indexed; rejected/odd/nonindexable queries retain literal original callback; FP/graph/public-guards unchanged";
inline bool parse(const char*value){
  if(!value||std::string_view(value)=="0")return false;
  if(std::string_view(value)=="1")return true;
  throw std::invalid_argument(std::string(flag)+" must be exactly0 or1");
}
inline bool requested(){
  const bool now=parse(std::getenv(flag));static const bool frozen=now;
  if(now!=frozen)throw std::logic_error("immutable interval index flag changed after freezing");
  return frozen;
}
inline void startup(){
  if(!requested())return;
  for(const char*name:{"SPLASH_FLASH_ALLROWS_FULL512_TARGET","SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22",
      "SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22"}){
    const char*value=std::getenv(name);
    if(!value||std::string_view(value)!="1")throw std::invalid_argument(std::string("immutable interval index requires ")+name+"=1");
  }
}
// External inline namespace objects: one cumulative process counter set,
// never reset by a Store constructor/move/destructor and never TU-static.
inline std::atomic<uint64_t> finalizedTables{0},finalizedSpans{0},indexableTables{0};
inline std::atomic<uint64_t> indexedAccepts{0},originalCallbacks{0};
} // namespace splash::flash::immutable_interval_worker_sep22
