#pragma once
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace splash::metal::private_boundary_memory {
inline bool parseSwitch(const char *value) {
    if (!value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument(
        "SPLASH_FLASH_PRIVATE_SKIP_COMMAND_MEMORY_QUERIES must be 0 or 1");
}
enum class Boundary : size_t { PreCommit, PostCommit, Scheduled, Completed, Count };
struct Counters {
    std::array<std::atomic<uint64_t>, static_cast<size_t>(Boundary::Count)> queried{};
    std::array<std::atomic<uint64_t>, static_cast<size_t>(Boundary::Count)> skipped{};
    bool query(bool skip, Boundary boundary) noexcept {
        auto &counter = skip ? skipped[static_cast<size_t>(boundary)]
                             : queried[static_cast<size_t>(boundary)];
        counter.fetch_add(1, std::memory_order_relaxed);
        return !skip;
    }
};
} // namespace splash::metal::private_boundary_memory
