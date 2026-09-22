#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <random>
#include <vector>

// CPU-only screening of the unchanged worker percentile algorithm and the
// governor's host-memory RPC sequence. This is not a service throughput test.
namespace {
using Clock = std::chrono::steady_clock;
volatile double checksum = 0;
double percentile(const std::deque<double> &window, double fraction) {
  if (window.empty()) return 0;
  std::vector<double> sorted(window.begin(), window.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted[static_cast<size_t>((sorted.size() - 1) * fraction)];
}
struct Quantiles { double p50 = 0, p95 = 0; };
Quantiles together(const std::deque<double> &window) {
  if (window.empty()) return {};
  std::vector<double> sorted(window.begin(), window.end());
  std::sort(sorted.begin(), sorted.end());
  return {sorted[static_cast<size_t>((sorted.size() - 1) * .5)],
          sorted[static_cast<size_t>((sorted.size() - 1) * .95)]};
}
template <class Work> double medianMicros(Work work, uint32_t repeats) {
  std::vector<double> samples;
  for (uint32_t sample = 0; sample != 7; ++sample) {
    const auto began = Clock::now();
    for (uint32_t repeat = 0; repeat != repeats; ++repeat) work();
    samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - began).count() / repeats);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}
uint64_t memoryRPCs() {
  mach_port_t host = mach_host_self();
  vm_size_t pageSize = 0;
  vm_statistics64_data_t statistics{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  kern_return_t pageResult = host_page_size(host, &pageSize);
  kern_return_t statisticsResult = pageResult == KERN_SUCCESS
      ? host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&statistics), &count)
      : pageResult;
  mach_port_deallocate(mach_task_self(), host);
  return statisticsResult == KERN_SUCCESS ? uint64_t{statistics.active_count} * pageSize : 0;
}
} // namespace

int main() {
  std::mt19937_64 random(0x5A7A5);
  std::uniform_real_distribution<double> values(0, 10000);
  std::cout << std::setprecision(12)
            << "{\"cpu_only\":true,\"gpu_work\":false,\"scope\":\"isolated percentile and host memory RPC cost; not service speed\",\"cases\":[";
  bool first = true;
  for (uint32_t count : {0, 83, 1024, 4096}) {
    std::deque<double> ttft, itl;
    for (uint32_t index = 0; index != count; ++index) {
      ttft.push_back(values(random)); itl.push_back(values(random));
    }
    const auto beforeTTFT = together(ttft), beforeITL = together(itl);
    if (beforeTTFT.p50 != percentile(ttft, .5) || beforeTTFT.p95 != percentile(ttft, .95) ||
        beforeITL.p50 != percentile(itl, .5) || beforeITL.p95 != percentile(itl, .95)) return 2;
    const double baseline = medianMicros([&] {
      checksum = percentile(ttft, .5) + percentile(ttft, .95) +
                 percentile(itl, .5) + percentile(itl, .95);
    }, 1000);
    const double oneSortPerWindow = medianMicros([&] {
      const auto a = together(ttft), b = together(itl);
      checksum = a.p50 + a.p95 + b.p50 + b.p95;
    }, 1000);
    const double cacheHit = medianMicros([&] {
      checksum = beforeTTFT.p50 + beforeTTFT.p95 + beforeITL.p50 + beforeITL.p95;
    }, 1000);
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"samples_per_window\":" << count << ",\"four_sorts_us\":" << baseline
              << ",\"two_sorts_us\":" << oneSortPerWindow << ",\"cache_hit_us\":" << cacheHit << '}';
  }
  const double rpc = medianMicros([&] { checksum = double(memoryRPCs()); }, 1000);
  std::cout << "],\"host_memory_rpc_sequence_us\":" << rpc << ",\"quantiles_exact\":true}\n";
}
