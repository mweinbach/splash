// CPU metadata/planning and Mach host-memory sampling only. No MetalBackend.
#include "flash/FlashForward.hpp"
#include "flash/FlashMTP.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "flash/FlashBatchForward.hpp"
#include "flash/FlashBatchPrefill.hpp"
#include "flash/FlashBatchVerify.hpp"
#include "flash/FlashBatchMTPForward.hpp"
#include "engine/MemoryGovernor.hpp"
#import <Foundation/Foundation.h>
#include <algorithm>
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 3) throw std::invalid_argument("usage: memory-cpu CAPACITY ROWS");
      const auto positive = [](const char *raw) {
        const std::string_view value(raw); uint32_t result = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !result)
          throw std::invalid_argument("memory planner integer must be positive decimal");
        return result;
      };
      const uint32_t capacity = positive(argv[1]), rows = positive(argv[2]);
      const uint64_t physical = NSProcessInfo.processInfo.physicalMemory;
      const uint64_t reserve = std::max<uint64_t>(16ULL << 30, physical / 10);
      const auto available = splash::engine::queryHostAvailableMemory();
      const auto rowPlan = splash::flash::FlashForward::workspacePlannedBytes(capacity, rows, 4);
      const auto blockedPlan = rows >= 256
          ? splash::flash::flashMoEBlockedWorkspacePlannedBytes(rows, 10) : 0;
      const auto targetState = splash::flash::FlashForward::requestStateBytes(capacity);
      const auto headState = splash::flash::FlashMTPForward::requestStateBytes(capacity);
      constexpr uint64_t depth3FoldHidden = uint64_t{4} * 10240 * 2;
      std::cout << "{\"gpu_work\":false,\"model_loaded\":false,\"capacity\":" << capacity
          << ",\"rows\":" << rows << ",\"physical_bytes\":" << physical
          << ",\"host_reserve_bytes\":" << reserve << ",\"engine_limit_bytes\":" << physical - reserve
          << ",\"host_measurement_valid\":" << (available ? "true" : "false")
          << ",\"host_available_bytes\":" << available.value_or(0)
          << ",\"host_headroom_bytes\":" << (available && *available > reserve ? *available - reserve : 0)
          << ",\"host_warning_margin_bytes\":" << splash::engine::kHostWarningMarginBytes
          << ",\"trunk_row_workspace_planned_bytes\":" << rowPlan
          << ",\"blocked_moe_workspace_planned_bytes\":" << blockedPlan
          << ",\"target_request_state_bytes\":" << targetState
          << ",\"mtp_request_state_bytes\":" << headState
          << ",\"fixed_depth3_mtp_committed_hidden_bytes\":" << depth3FoldHidden
          << ",\"request_state_bytes_per_lane\":" << targetState + headState + depth3FoldHidden
          << ",\"sequential_mtp_workspace_planned_bytes\":"
          << splash::flash::FlashMTPForward::workspacePlannedBytes(capacity, 128)
          << ",\"batch_decode_workspace_planned_bytes\":"
          << splash::flash::FlashBatchForward::workspacePlannedBytes(capacity, 4)
          << ",\"batch_prefill_workspace_planned_bytes\":"
          << splash::flash::FlashBatchPrefill::workspacePlannedBytes(capacity, 4, 2048)
          << ",\"batch_teacher_prime_workspace_planned_bytes\":"
          << splash::flash::FlashBatchMTPForward::workspacePlannedBytes(capacity, 4, 128, false)
          << ",\"joint_verifier_workspace_planned_bytes\":"
          << splash::flash::FlashBatchVerify::workspacePlannedBytes(capacity, 4, 4)
          << ",\"joint_head_workspace_planned_bytes_shared_vocabulary\":"
          << splash::flash::FlashBatchMTPForward::workspacePlannedBytes(capacity, 4, 4, true)
          << ",\"row_plan_scope\":\"actual compiled private planner; excludes immutable dense/F32/expert payload caches\"}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n'; return 1;
    }
  }
}
