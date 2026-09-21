// Root-run primitive qualification. --cpu-self-test creates no Metal backend.
#include "flash/FlashBatchPrefill.hpp"
#include "metal/abi/FlashForward.h"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
using namespace splash::flash;
using namespace splash::metal;
using Clock = std::chrono::steady_clock;
constexpr uint64_t kGuardBytes = 128;
constexpr uint8_t kGuardValue = 0xa7;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template <class F> void rejected(F &&function, uint64_t &checks) {
  bool threw = false;
  try { function(); }
  catch (const std::invalid_argument &) { threw = true; }
  require(threw, "production copy metadata accepted invalid input");
  ++checks;
}
uint64_t cpuSelfTest() {
  uint64_t checks = 0;
  for (uint32_t lanes = 1; lanes <= 4; ++lanes)
    for (const uint32_t rows : {1u, 2u, 512u, 2048u}) {
      require(flashBatchPrefillHiddenCopyBytes(lanes, rows) ==
          uint64_t{lanes} * rows * 10240 * 2,
          "production hidden feature byte extent disagrees with shape");
      ++checks;
    }
  require(flashBatchPrefillHiddenCopyBytes(4, 2048) == 167772160,
      "maximum hidden features must occupy exactly 160 MiB");
  ++checks;
  for (const auto geometry : std::array<std::array<uint32_t, 2>, 5>{
           {{0, 1}, {5, 1}, {1, 0}, {1, 2049}, {UINT32_MAX, UINT32_MAX}}})
    rejected([&] { (void)flashBatchPrefillHiddenCopyBytes(geometry[0], geometry[1]); }, checks);
  const auto *aligned = reinterpret_cast<const void *>(uintptr_t{0x1000});
  for (const uint64_t copied : {uint64_t{4}, uint64_t{1028}, uint64_t{167772160}}) {
    validateFlashBatchPrefillCopyRange(aligned, copied, copied);
    validateFlashBatchPrefillCopyRange(aligned, copied + 16, copied);
    checks += 2;
  }
  rejected([&] { validateFlashBatchPrefillCopyRange(nullptr, 4, 4); }, checks);
  rejected([&] { validateFlashBatchPrefillCopyRange(aligned, 4, 0); }, checks);
  rejected([&] { validateFlashBatchPrefillCopyRange(aligned, 4, 2); }, checks);
  rejected([&] { validateFlashBatchPrefillCopyRange(aligned, 1028, 1026); }, checks);
  rejected([&] { validateFlashBatchPrefillCopyRange(aligned, 4, 8); }, checks);
  rejected([&] { validateFlashBatchPrefillCopyRange(
      reinterpret_cast<const void *>(uintptr_t{0x1002}), 8, 8); }, checks);
  const auto *high = reinterpret_cast<const void *>(
      std::numeric_limits<uintptr_t>::max() - uintptr_t{3});
  rejected([&] { validateFlashBatchPrefillCopyRange(high, 4, 4); }, checks);
  rejected([&] { validateFlashBatchPrefillCopyRange(aligned, UINT64_MAX, 4); }, checks);
  auto address = [](uintptr_t raw) { return reinterpret_cast<const void *>(raw); };
  for (const auto sample : std::array<std::array<uintptr_t, 5>, 9>{
           {{0x1000, 16, 0x1000, 16, 1}, {0x1000, 16, 0x1008, 8, 1},
            {0x1008, 8, 0x1000, 16, 1}, {0x1000, 16, 0x1010, 16, 0},
            {0x1010, 16, 0x1000, 16, 0}, {0x1000, 16, 0x1004, 4, 1},
            {0x1004, 4, 0x1000, 16, 1}, {0x1000, 0, 0x1000, 16, 0},
            {0, 16, 0x1000, 16, 0}}}) {
    require(flashBatchPrefillCopyRangesOverlap(address(sample[0]), sample[1],
        address(sample[2]), sample[3]) == bool(sample[4]),
        "production copy overlap predicate failed");
    ++checks;
  }
  rejected([&] { (void)flashBatchPrefillCopyRangesOverlap(high, 8, aligned, 4); }, checks);
  rejected([&] { (void)flashBatchPrefillCopyRangesOverlap(aligned, 4, high, 8); }, checks);
  std::cout << "GPU prefill copy metadata CPU self-test PASS; checks=" << checks
            << "; no Metal backend created" << std::endl;
  return checks;
}

std::string sha256(const void *data, uint64_t bytes) {
  CC_SHA256_CTX state{};
  CC_SHA256_Init(&state);
  auto *next = static_cast<const uint8_t *>(data);
  while (bytes) {
    const auto count = CC_LONG(std::min<uint64_t>(bytes, UINT32_MAX));
    CC_SHA256_Update(&state, next, count);
    next += count;
    bytes -= count;
  }
  std::array<uint8_t, 32> digest{};
  CC_SHA256_Final(digest.data(), &state);
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (const auto value : digest) {
    result += digits[value >> 4];
    result += digits[value & 15];
  }
  return result;
}
NSString *ns(const std::string &value) {
  return [[NSString alloc] initWithBytes:value.data() length:value.size()
                               encoding:NSUTF8StringEncoding];
}
double seconds(Clock::time_point began) {
  return std::chrono::duration<double>(Clock::now() - began).count();
}
double median(std::vector<double> values) {
  require(!values.empty(), "empty copy timing cohort");
  std::sort(values.begin(), values.end());
  return values.size() & 1 ? values[values.size() / 2] :
      (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
}
uint32_t environmentCount(const char *name, uint32_t fallback, bool allowZero) {
  const char *raw = std::getenv(name);
  if (!raw) return fallback;
  require(*raw != 0, "empty copy repeat count");
  uint32_t result = 0;
  for (const char *next = raw; *next; ++next) {
    require(*next >= '0' && *next <= '9', "copy repeat count is not decimal");
    result = result * 10 + uint32_t(*next - '0');
    require(result <= 256, "copy repeat count exceeds 256");
  }
  require(allowZero || result != 0, "copy repeat count is zero");
  return result;
}
void fillPayload(MetalBuffer &buffer) {
  auto *words = static_cast<uint32_t *>(buffer.contents());
  require(words && buffer.sizeBytes() % 4 == 0, "unaligned copy test payload");
  constexpr std::array<uint32_t, 12> traps{0x00000000, 0x80008000, 0x7f807f80,
      0xff80ff80, 0x7fc17fff, 0x00010001, 0x80018001, 0x7f7f7f7f,
      0xff7fff7f, 0x3f803f80, 0x00800080, 0xffffffff};
  for (uint64_t word = 0; word < buffer.sizeBytes() / 4; ++word) {
    const auto mixed = uint32_t(word) * 0x9e3779b9u ^ uint32_t(word >> 16) ^ 0x5a1961efu;
    words[word] = word < traps.size() ? traps[word] : mixed;
  }
}
struct Guarded final {
  MetalBuffer base, view;
  uint64_t offset, bytes;
  Guarded(MetalBackend &backend, uint64_t count, uint64_t prefix = kGuardBytes)
      : offset(prefix), bytes(count) {
    require(prefix >= 4 && prefix % 4 == 0, "invalid guarded copy view offset");
    base = backend.allocateBuffer(offset + bytes + kGuardBytes, BufferStorage::Shared);
    std::memset(base.contents(), kGuardValue, base.sizeBytes());
    view = backend.view(base, offset, bytes);
  }
  void check() const {
    auto *raw = static_cast<const uint8_t *>(base.contents());
    for (uint64_t index = 0; index < offset; ++index)
      require(raw[index] == kGuardValue, "copy changed destination prefix canary");
    for (uint64_t index = offset + bytes; index < base.sizeBytes(); ++index)
      require(raw[index] == kGuardValue, "copy changed destination suffix canary");
  }
};
void addCopy(CommandGraph &graph, const MetalBuffer &input, const MetalBuffer &output) {
  validateFlashBatchPrefillCopyRange(input.contents(), input.sizeBytes(), input.sizeBytes());
  validateFlashBatchPrefillCopyRange(output.contents(), output.sizeBytes(), input.sizeBytes());
  require(!flashBatchPrefillCopyRangesOverlap(input.contents(), input.sizeBytes(),
      output.contents(), input.sizeBytes()), "copy oracle must use independent allocations");
  const uint64_t words = input.sizeBytes() / 4;
  graph.add("flash_forward_copy_words", {input, output}, FlashForwardCopyParams{words},
      {(words + 255) / 256, 1, 1}, {256, 1, 1});
}
NSDictionary *runCase(MetalBackend &backend, uint64_t bytes, const std::string &label,
    uint32_t repeats, uint32_t warmup, uint64_t sourceOffset, uint64_t outputOffset) {
  Guarded immutable(backend, bytes, sourceOffset), producer(backend, bytes, 64),
      gpuDestination(backend, bytes, outputOffset), cpuDestination(backend, bytes, 260);
  fillPayload(immutable.view);
  std::memset(producer.view.contents(), 0x5c, bytes);
  const auto sourceDigest = sha256(immutable.view.contents(), bytes);
  CommandGraph sameCommand, producerOnly;
  addCopy(sameCommand, immutable.view, producer.view);
  addCopy(sameCommand, producer.view, gpuDestination.view);
  addCopy(producerOnly, immutable.view, producer.view);
  std::vector<double> gpuSeconds, gpuWall, cpuCopySeconds, baselineGpuSeconds, baselineWall;
  for (uint32_t sample = 0; sample < warmup + repeats; ++sample) {
    CommandTiming candidate{}, baseline{};
    double hostCopy = 0;
    auto gpu = [&] {
      std::memset(producer.view.contents(), 0x5c, bytes);
      std::memset(gpuDestination.view.contents(), 0xf3, bytes);
      candidate = backend.submitCommand(sameCommand.dispatches());
    };
    auto cpu = [&] {
      std::memset(producer.view.contents(), 0x5c, bytes);
      std::memset(cpuDestination.view.contents(), 0xf3, bytes);
      baseline = backend.submitCommand(producerOnly.dispatches());
      const auto began = Clock::now();
      std::memcpy(cpuDestination.view.contents(), producer.view.contents(), bytes);
      hostCopy = seconds(began);
    };
    // The two routes are alternated; no model or second GPU workload runs here.
    if (sample & 1) { cpu(); gpu(); } else { gpu(); cpu(); }
    require(std::memcmp(gpuDestination.view.contents(), immutable.view.contents(), bytes) == 0,
        "same-command producer and GPU destination differ bitwise");
    require(std::memcmp(cpuDestination.view.contents(), immutable.view.contents(), bytes) == 0,
        "producer and CPU destination differ bitwise");
    immutable.check(); producer.check(); gpuDestination.check(); cpuDestination.check();
    if (sample >= warmup) {
      gpuSeconds.push_back(candidate.gpuSeconds);
      gpuWall.push_back(candidate.wallSeconds);
      cpuCopySeconds.push_back(hostCopy);
      baselineGpuSeconds.push_back(baseline.gpuSeconds);
      baselineWall.push_back(baseline.wallSeconds + hostCopy);
    }
  }
  require(sha256(immutable.view.contents(), bytes) == sourceDigest,
      "GPU copy changed the immutable source");
  const double candidateMedian = median(gpuSeconds), hostMedian = median(cpuCopySeconds);
  std::cout << label << " PASS; bytes=" << bytes
            << "; same-command GPU ms=" << candidateMedian * 1000
            << "; host memcpy ms=" << hostMedian * 1000 << std::endl;
  return @{ @"label": ns(label), @"bytes": @(bytes),
      @"source_offset": @(sourceOffset), @"destination_offset": @(outputOffset),
      @"dispatches_per_candidate_command": @2, @"repeats": @(repeats),
      @"warmup": @(warmup), @"bit_exact": @YES, @"canaries_intact": @YES,
      @"source_unchanged": @YES, @"source_sha256": ns(sourceDigest),
      @"same_command_gpu_median_seconds": @(candidateMedian),
      @"same_command_wall_median_seconds": @(median(gpuWall)),
      @"producer_only_gpu_median_seconds": @(median(baselineGpuSeconds)),
      @"host_memcpy_median_seconds": @(hostMedian),
      @"producer_then_host_copy_wall_median_seconds": @(median(baselineWall)),
      @"timing_scope": @"synthetic producer plus independent destination; no whole-model speed claim" };
}

NSDictionary *ticketLifetime(MetalBackend &backend) {
  constexpr uint64_t bytes = 1028;
  const long pageValue = sysconf(_SC_PAGESIZE);
  require(pageValue > 0, "unable to determine host page alignment");
  const auto page = uint64_t(pageValue);
  const uint64_t allocationBytes = ((bytes + kGuardBytes * 2 + page - 1) / page) * page;
  void *mapping = nullptr;
  require(posix_memalign(&mapping, page, allocationBytes) == 0 && mapping,
      "unable to allocate asynchronous destination mapping");
  auto backing = std::shared_ptr<void>(mapping, [](void *memory) { std::free(memory); });
  std::weak_ptr<void> observer = backing;
  std::memset(mapping, kGuardValue, allocationBytes);
  Guarded immutable(backend, bytes, 64), producer(backend, bytes, 68);
  fillPayload(immutable.view);
  const auto sourceDigest = sha256(immutable.view.contents(), bytes);
  auto base = backend.wrapSharedMemory(mapping, allocationBytes, backing,
      "oracle ticket-only hidden destination");
  auto destination = backend.view(base, kGuardBytes, bytes);
  CommandGraph graph;
  addCopy(graph, immutable.view, producer.view);
  addCopy(graph, producer.view, destination);
  auto ticket = backend.submitCommandAsync(graph.dispatches());
  // Drop the original destination, its view, graph bindings, and the original
  // backing owner. The unconsumed command ticket now owns the destination.
  graph = CommandGraph{};
  destination = MetalBuffer{};
  base = MetalBuffer{};
  backing.reset();
  const bool readyWhenHandlesDropped = ticket.ready();
  require(!observer.expired(), "pending ticket released its destination mapping");
  const auto began = Clock::now();
  while (!ticket.ready()) {
    require(seconds(began) < 120, "asynchronous copy did not complete within 120 seconds");
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  require(!observer.expired(), "completed but unconsumed ticket released its destination");
  // Acquire only the completed host backing, without adding a MetalBuffer
  // owner. GPU completion makes the Shared bytes safe to inspect.
  auto completedBacking = observer.lock();
  require(bool(completedBacking), "ticket-only destination mapping vanished");
  const auto *result = static_cast<const uint8_t *>(completedBacking.get());
  require(std::memcmp(result + kGuardBytes, immutable.view.contents(), bytes) == 0,
      "ticket-only destination copy differs bitwise");
  for (uint64_t index = 0; index < allocationBytes; ++index)
    if (index < kGuardBytes || index >= kGuardBytes + bytes)
      require(result[index] == kGuardValue, "ticket-only destination canary changed");
  const auto timing = ticket.wait();
  immutable.check(); producer.check();
  require(sha256(immutable.view.contents(), bytes) == sourceDigest,
      "ticket-only destination test changed source");
  std::cout << "asynchronous destination lifetime PASS; original handles dropped before ticket consumption"
            << std::endl;
  return @{ @"bytes": @(bytes), @"original_destination_handles_dropped": @YES,
      @"ticket_ready_when_handles_dropped": @(readyWhenHandlesDropped),
      @"ticket_retained_destination_until_consumed": @YES, @"bit_exact": @YES,
      @"canaries_intact": @YES, @"source_unchanged": @YES,
      @"gpu_seconds": @(timing.gpuSeconds), @"wall_seconds": @(timing.wallSeconds) };
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") {
        (void)cpuSelfTest();
        return 0;
      }
      require(argc == 3,
          "usage: flash-gpu-prefill-copy-oracle <private-metallib> <report.json>");
      const auto checks = cpuSelfTest();
      const auto repeats = environmentCount("FLASH_GPU_PREFILL_COPY_REPEATS", 8, false);
      const auto warmup = environmentCount("FLASH_GPU_PREFILL_COPY_WARMUP", 2, true);
      MetalBackend backend(argv[1]);
      NSMutableArray *cases = [NSMutableArray array];
      [cases addObject:runCase(backend, 4, "one-word", repeats, warmup, 4, 12)];
      [cases addObject:runCase(backend, 1028, "partial-final-threadgroup", repeats, warmup, 68, 132)];
      [cases addObject:runCase(backend, flashBatchPrefillHiddenCopyBytes(1, 1),
          "B1-R1-true-BF16-hidden", repeats, warmup, 128, 256)];
      [cases addObject:runCase(backend, flashBatchPrefillHiddenCopyBytes(4, 512),
          "B4-R512-true-BF16-hidden", repeats, warmup, 64, 260)];
      [cases addObject:runCase(backend, flashBatchPrefillHiddenCopyBytes(4, 2048),
          "B4-R2048-true-BF16-hidden", repeats, warmup, 128, 256)];
      const auto lifetime = ticketLifetime(backend);
      const auto &digest = backend.metallibSha256();
      static constexpr char digits[] = "0123456789abcdef";
      std::string librarySHA;
      for (const auto value : digest) {
        librarySHA += digits[value >> 4]; librarySHA += digits[value & 15];
      }
      NSDictionary *report = @{ @"schema": @"splash-flash-gpu-prefill-copy-primitive-v1",
          @"pass": @YES, @"cpu_guard_checks": @(checks),
          @"pipeline": @"flash_forward_copy_words", @"device": ns(backend.capabilities().deviceName),
          @"private_metallib_sha256": ns(librarySHA), @"cases": cases,
          @"destination_lifetime": lifetime, @"submissions": @(backend.submissionCount()),
          @"maximum_feature_bytes": @(flashBatchPrefillHiddenCopyBytes(4, 2048)) };
      NSError *error = nil;
      NSData *json = [NSJSONSerialization dataWithJSONObject:report
          options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
      require(json != nil && error == nil, "unable to serialize copy qualification report");
      require([json writeToFile:ns(argv[2]) options:NSDataWritingAtomic error:&error] && !error,
          "unable to write copy qualification report");
      std::cout << "GPU prefill copy primitive qualification PASS; report=" << argv[2] << std::endl;
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "GPU prefill copy oracle FAIL: " << error.what() << std::endl;
      return 1;
    }
  }
}
