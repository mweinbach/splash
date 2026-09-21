#pragma once
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace splash::metal::private_partitioned_residency {
inline constexpr std::size_t kMaximumSets = 32;
struct Group {
  std::vector<std::size_t> indices;
  uint64_t bytes = 0;
};
struct Plan {
  uint64_t capBytes = 0;
  uint64_t totalBytes = 0;
  std::vector<Group> groups;
};
inline uint64_t parseCap(const char *value) {
  if (!value) return 0;
  const std::string_view text(value);
  if (text.empty() || (text.size() > 1 && text.front() == '0'))
    throw std::invalid_argument("SPLASH_FLASH_PRIVATE_RESIDENCY_SET_CAP_BYTES must be a canonical unsigned integer");
  uint64_t result = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("SPLASH_FLASH_PRIVATE_RESIDENCY_SET_CAP_BYTES must be a canonical unsigned integer");
  return result;
}
inline Plan plan(const std::vector<uint64_t>& bytes, uint64_t cap) {
  Plan result;
  result.capBytes = cap;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const uint64_t size = bytes[i];
    if (!size) throw std::invalid_argument("private residency plan contains an empty allocation");
    if (size > std::numeric_limits<uint64_t>::max() - result.totalBytes)
      throw std::invalid_argument("private residency total overflows");
    if (cap && size > cap)
      throw std::invalid_argument("private residency cap is smaller than a native allocation");
    result.totalBytes += size;
    std::size_t selected = result.groups.size();
    for (std::size_t j = 0; j < result.groups.size(); ++j) {
      if (!cap || size <= cap - result.groups[j].bytes) { selected = j; break; }
    }
    if (selected == result.groups.size()) {
      if (result.groups.size() == kMaximumSets)
        throw std::invalid_argument("private residency plan exceeds the Metal32-set queue limit");
      result.groups.emplace_back();
    }
    auto& group = result.groups[selected];
    group.indices.push_back(i);
    group.bytes += size;
  }
  return result;
}
} // namespace splash::metal::private_partitioned_residency
