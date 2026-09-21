// Compilation and --cpu-self-test submit no GPU work. Root owns GPU execution.
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashGreedy.hpp"
#include "flash/FlashMTPWindow.hpp"
#include "engine/Json.hpp"

#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::flash;
using namespace splash::metal;
using Clock = std::chrono::steady_clock;
constexpr uint32_t kVocabulary = 248320;
constexpr uint64_t kGuard = 128;
uint32_t sink = 0;
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
uint32_t rank(uint16_t bits) {
  return !(bits & 0x7fff) ? 0x8000 : (bits & 0x8000)
      ? uint16_t(~bits) : uint16_t(bits ^ 0x8000);
}
bool finite(uint16_t bits) { return (bits & 0x7f80) != 0x7f80; }
float number(uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); }
uint32_t scalarGreedy(std::span<const uint16_t> values) {
  require(!values.empty(), "scalar greedy empty row");
  float best = -INFINITY;
  uint32_t token = 0;
  for (uint32_t id = 0; id < values.size(); ++id) {
    const float value = number(values[id]);
    require(std::isfinite(value), "scalar greedy nonfinite row");
    if (value > best) { best = value; token = id; }
  }
  return token;
}
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto count = values.size();
  return count & 1 ? values[count / 2] :
      (values[count / 2 - 1] + values[count / 2]) / 2;
}
void cpuSelfTest() {
  uint64_t checks = 0;
  std::vector<uint16_t> ordered;
  for (uint32_t word = 0; word <= 0xffff; ++word) {
    const auto bits = uint16_t(word);
    const std::array<uint16_t, 3> sample{0x8000, bits, 0};
    if (!finite(bits)) {
      bool rejected = false;
      try { (void)flashGreedyToken(sample); }
      catch (const std::runtime_error &) { rejected = true; }
      require(rejected, "CPU greedy accepted nonfinite encoding");
      ++checks;
      continue;
    }
    ordered.push_back(bits);
    require((rank(bits) > rank(0)) == (number(bits) > 0), "BF16 rank versus zero");
    require(flashGreedyToken(sample) == scalarGreedy(sample), "CPU greedy signed-zero encoding");
    checks += 2;
  }
  std::sort(ordered.begin(), ordered.end(), [](auto a, auto b) { return rank(a) < rank(b); });
  for (size_t i = 1; i < ordered.size(); ++i) {
    require(number(ordered[i - 1]) <= number(ordered[i]), "BF16 rank is not monotonic");
    const std::array pair{ordered[i - 1], ordered[i]};
    require(flashGreedyToken(pair) == scalarGreedy(pair), "CPU greedy adjacent finite values");
    checks += 2;
  }
  const std::array<uint32_t, 2> stops{248044, 248046};
  for (uint32_t rows = 1; rows <= 16; ++rows)
    for (uint32_t matches = 0; matches < rows; ++matches)
      for (uint32_t quota = 1; quota <= rows + 1; ++quota) {
        std::vector<uint32_t> inputs(rows), predictions(rows);
        std::iota(inputs.begin(), inputs.end(), 100);
        for (uint32_t row = 0; row < rows; ++row)
          predictions[row] = row < matches ? inputs[row + 1] : 1000 + row;
        const auto accepted = flashMTPAcceptGreedyPrefix(inputs, predictions, quota, stops);
        require(accepted && accepted->matchedDrafts == matches &&
            accepted->retainedRows == std::min(matches + 1, quota), "CPU prefix quota/mismatch");
        ++checks;
      }
  for (uint32_t rows = 1; rows <= 16; ++rows)
    for (uint32_t vocabulary : {1u, 2048u, 2049u, 248320u}) {
      const uint64_t partialBytes = uint64_t{rows} * ((vocabulary + 2047) / 2048) * 16;
      const uint64_t resultBytes = uint64_t{rows} * 16;
      const auto planned = greedyGPUWorkspacePlannedBytes(rows, vocabulary);
      require(planned >= partialBytes + resultBytes && planned % 16384 == 0 &&
          planned == ((partialBytes + 16383) & ~uint64_t{16383}) +
                     ((resultBytes + 16383) & ~uint64_t{16383}),
          "GPU greedy admission plan does not cover rounded independent allocations");
      ++checks;
    }
  for (uint32_t word = 0; word <= 0xffff; ++word)
    if (finite(uint16_t(word))) {
      require(greedyGPUResultToken({0, rank(uint16_t(word)), 0, 0}, 1) == 0,
          "GPU greedy compact reader rejected finite BF16 rank");
      ++checks;
    }
  for (const auto record : {FlashGreedyGPURowResult{1, 0xbf80, 0, 0},
       FlashGreedyGPURowResult{0, 0, 0, 0}, FlashGreedyGPURowResult{0, 0xffff, 0, 0},
       FlashGreedyGPURowResult{0, 0x7fff, 0, 0}, FlashGreedyGPURowResult{0, 0xbf80, 2, 0},
       FlashGreedyGPURowResult{0, 0xbf80, 0, 1}}) {
    bool rejected = false;
    try { (void)greedyGPUResultToken(record, 1); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "GPU greedy compact reader accepted malformed record");
    ++checks;
  }
  bool preservedNonfiniteMessage = false;
  try { (void)greedyGPUResultToken({UINT32_MAX, 0, 1, 0}, 1); }
  catch (const std::runtime_error &error) {
    preservedNonfiniteMessage = std::string(error.what()) == "non-finite Flash vocabulary logit";
  }
  require(preservedNonfiniteMessage, "GPU greedy compact reader changed nonfinite error message");
  ++checks;
  std::cout << "{\"pass\":true,\"cpu_checks\":" << checks
      << ",\"all_bf16_encodings\":65536,\"gpu_commands\":0}\n";
}

void cpuBenchmark() {
  std::cout << std::setprecision(12) << "{\"gpu_commands\":0,\"vocabulary\":"
      << kVocabulary << ",\"winner_position\":\"last\",\"benchmarks\":[";
  bool first = true;
  for (uint32_t rows : {1u, 2u, 4u, 8u, 16u}) {
    std::vector<uint16_t> values(uint64_t{rows} * kVocabulary);
    for (uint64_t id = 0; id < values.size(); ++id)
      values[id] = uint16_t(0xbf80 + (id * 13u % 0x3ff0));
    for (uint32_t row = 0; row < rows; ++row)
      values[uint64_t{row} * kVocabulary + kVocabulary - 1] = 0x7f7f;
    std::vector<double> samples;
    for (uint32_t sample = 0; sample < 15; ++sample) {
      const auto began = Clock::now();
      for (uint32_t repeat = 0; repeat < 32; ++repeat) {
        asm volatile("" : : "r"(values.data()) : "memory");
        for (uint32_t row = 0; row < rows; ++row)
          sink ^= flashGreedyToken({values.data() + uint64_t{row} * kVocabulary, kVocabulary});
      }
      samples.push_back(std::chrono::duration<double>(Clock::now() - began).count() / 32);
    }
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"rows\":" << rows << ",\"cpu_neon_us\":" << median(samples) * 1e6 << '}';
  }
  std::cout << "]}\n";
}

struct Guarded final {
  MetalBuffer base, view;
  uint64_t bytes;
  Guarded(MetalBackend &backend, uint64_t size) : bytes(size) {
    base = backend.allocateBuffer(size + 2 * kGuard, BufferStorage::Shared, "greedy GPU oracle guarded");
    std::memset(base.contents(), 0xa5, size + 2 * kGuard);
    view = backend.view(base, kGuard, size);
  }
  template <typename T> T *data() { return static_cast<T *>(view.contents()); }
  void check() const {
    const auto *words = static_cast<const uint8_t *>(base.contents());
    for (uint64_t byte = 0; byte < kGuard; ++byte)
      require(words[byte] == 0xa5 && words[kGuard + bytes + byte] == 0xa5,
              "greedy GPU oracle buffer canary changed");
  }
};

uint64_t qualifyArgmax(MetalBackend &backend) {
  uint64_t checks = 0;
  const std::array<uint32_t, 19> extents{
      1, 7, 31, 32, 33, 255, 256, 257, 2047, 2048, 2049,
      4095, 4096, 4097, 65536, 131072, 248319, 248320, 12345};
  for (uint32_t rows : {1u, 2u, 4u, 8u, 16u})
    for (uint32_t vocabulary : extents) {
      const uint32_t stride = vocabulary + 7;
      Guarded logits(backend, uint64_t{rows} * stride * 2);
      Guarded results(backend, uint64_t{rows} * sizeof(FlashGreedyGPURowResult));
      auto workspace = allocateGreedyGPUWorkspace(backend, rows, vocabulary, BufferStorage::Shared);
      Guarded partials(backend, workspace.partials.sizeBytes());
      workspace.partials = partials.view;
      auto *values = logits.data<uint16_t>();
      for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t id = 0; id < stride; ++id)
          values[uint64_t{row} * stride + id] = id >= vocabulary ? 0x7fc1
              : uint16_t(0x8001 + (id * 37u + row * 17u) % 0x7f7e);
        const uint32_t first = (row * 2048u + 31u) % vocabulary;
        const uint16_t maximum = row & 2 ? uint16_t{0x7f7f} :
            row & 1 ? uint16_t{0x8000} : uint16_t{0x0000};
        values[uint64_t{row} * stride + first] = maximum;
        values[uint64_t{row} * stride + vocabulary - 1] = row & 2 ? maximum :
            row & 1 ? uint16_t{0x0000} : uint16_t{0x8000};
      }
      const std::vector<uint16_t> before(values, values + uint64_t{rows} * stride);
      CommandGraph graph;
      addGreedyGPU(graph, logits.view, workspace, results.view, rows, vocabulary, stride);
      (void)backend.submitCommand(graph.dispatches());
      const auto *found = results.data<FlashGreedyGPURowResult>();
      for (uint32_t row = 0; row < rows; ++row) {
        const auto source = std::span(values + uint64_t{row} * stride, vocabulary);
        require(found[row].token == scalarGreedy(source) &&
            found[row].token == flashGreedyToken(source) && !found[row].errors &&
            found[row].rank == rank(source[found[row].token]) && !found[row].reserved,
            "GPU BF16 argmax differs from independent scalar/NEON reference");
        ++checks;
      }
      require(std::equal(before.begin(), before.end(), values), "GPU argmax changed source/padding");
      logits.check(); results.check(); partials.check();
    }
  // Every BF16 NaN/Inf payload/sign is checked at SIMD/TG/shard/tail boundaries.
  constexpr uint32_t rows = 16;
  const std::array<uint32_t, 8> positions{0, 7, 31, 255, 2047, 2048, 4096, kVocabulary - 1};
  auto workspace = allocateGreedyGPUWorkspace(backend, rows, kVocabulary);
  Guarded logits(backend, uint64_t{rows} * kVocabulary * 2);
  Guarded results(backend, rows * sizeof(FlashGreedyGPURowResult));
  auto *values = logits.data<uint16_t>();
  std::vector<uint16_t> invalid;
  for (uint32_t word = 0; word <= 0xffff; ++word)
    if (!finite(uint16_t(word))) invalid.push_back(uint16_t(word));
  for (uint32_t position : positions)
    for (size_t offset = 0; offset < invalid.size(); offset += rows) {
      std::fill(values, values + uint64_t{rows} * kVocabulary, uint16_t(0));
      for (uint32_t row = 0; row < rows; ++row)
        values[uint64_t{row} * kVocabulary + position] = invalid[offset + row];
      CommandGraph graph;
      addGreedyGPU(graph, logits.view, workspace, results.view, rows, kVocabulary);
      (void)backend.submitCommand(graph.dispatches());
      for (uint32_t row = 0; row < rows; ++row)
        require(results.data<FlashGreedyGPURowResult>()[row].errors == kFlashGreedyGPUErrorNonfinite &&
            results.data<FlashGreedyGPURowResult>()[row].token == UINT32_MAX,
            "GPU argmax accepted nonfinite payload/sign");
      checks += rows;
    }
  logits.check(); results.check();
  return checks;
}

struct PrefixCase final {
  uint32_t rows;
  std::vector<uint32_t> inputs, predictions;
  uint32_t quota;
};

uint64_t qualifyPrefix(MetalBackend &backend) {
  std::vector<PrefixCase> cases;
  for (uint32_t rows = 1; rows <= 16; ++rows) {
    for (uint32_t matches = 0; matches < rows; ++matches)
      for (uint32_t quota = 1; quota <= rows + 1; ++quota)
        for (uint32_t correction : {1000u, 248044u, 248046u}) {
          PrefixCase item{rows, std::vector<uint32_t>(rows), std::vector<uint32_t>(rows), quota};
          std::iota(item.inputs.begin(), item.inputs.end(), 100);
          for (uint32_t row = 0; row < rows; ++row)
            item.predictions[row] = row < matches ? item.inputs[row + 1] : 1000 + row;
          item.predictions[matches] = correction;
          cases.push_back(std::move(item));
        }
    for (uint32_t stopPosition = 0; stopPosition < rows; ++stopPosition)
      for (uint32_t quota = 1; quota <= rows + 1; ++quota)
        for (uint32_t stop : {248044u, 248046u}) {
          PrefixCase item{rows, std::vector<uint32_t>(rows), std::vector<uint32_t>(rows), quota};
          std::iota(item.inputs.begin(), item.inputs.end(), 100);
          if (stopPosition + 1 < rows) item.inputs[stopPosition + 1] = stop;
          for (uint32_t row = 0; row < rows; ++row)
            item.predictions[row] = row + 1 < rows ? item.inputs[row + 1] : 1000;
          if (stopPosition + 1 == rows) item.predictions[stopPosition] = stop;
          cases.push_back(std::move(item));
        }
  }
  // Prefix math is independently tested with synthetic valid producer partials.
  // Multiple tiny dispatches share a command, avoiding one completion per case.
  const uint32_t partitions = (kVocabulary + 2047) / 2048;
  constexpr size_t batchSize = 64;
  const std::array<uint32_t, 2> stops{248044, 248046};
  for (size_t offset = 0; offset < cases.size(); offset += batchSize) {
    const size_t count = std::min(batchSize, cases.size() - offset);
    const uint64_t partialStride = uint64_t{16} * partitions * sizeof(FlashGreedyGPURowResult);
    Guarded partials(backend, count * partialStride);
    Guarded inputs(backend, count * 16 * 4);
    Guarded quotas(backend, count * 4);
    Guarded results(backend, count * sizeof(FlashGreedyGPUPrefixResult));
    std::memset(partials.view.contents(), 0, partials.bytes);
    CommandGraph graph;
    for (size_t index = 0; index < count; ++index) {
      const auto &item = cases[offset + index];
      const auto pv = backend.view(partials.view, index * partialStride, partialStride);
      const auto iv = backend.view(inputs.view, index * 16 * 4, 16 * 4);
      const auto qv = backend.view(quotas.view, index * 4, 4);
      const auto rv = backend.view(results.view, index * sizeof(FlashGreedyGPUPrefixResult),
          sizeof(FlashGreedyGPUPrefixResult));
      auto *records = static_cast<FlashGreedyGPURowResult *>(pv.contents());
      for (uint32_t row = 0; row < item.rows; ++row)
        records[uint64_t{row} * partitions + (row * 37u % partitions)] =
            {item.predictions[row], 0xbf80, 0, 0};
      std::memcpy(iv.contents(), item.inputs.data(), item.rows * 4);
      *static_cast<uint32_t *>(qv.contents()) = item.quota;
      const FlashGreedyGPUParams p{item.rows, kVocabulary, kVocabulary, partitions,
          1, item.rows, 1, 0};
      graph.add("flash_greedy_gpu_prefix", {pv, iv, qv, rv}, p, {1, 1, 1}, {256, 1, 1});
    }
    (void)backend.submitCommand(graph.dispatches());
    for (size_t index = 0; index < count; ++index) {
      const auto &item = cases[offset + index];
      const auto expected = flashMTPAcceptGreedyPrefix(item.inputs, item.predictions, item.quota, stops);
      const auto &found = results.data<FlashGreedyGPUPrefixResult>()[index];
      require(expected && !found.errors && found.matched_drafts == expected->matchedDrafts &&
          found.retained_rows == expected->retainedRows &&
          found.finish == uint32_t(expected->finish), "GPU prefix counts/finish differ from CPU reference");
      for (uint32_t row = 0; row < 16; ++row) {
        require(found.output[row] == (row < expected->retainedRows ? expected->output[row] : UINT32_MAX),
            "GPU prefix output/unused tail differs from CPU reference");
        require(found.predictions[row] == (row < item.rows ? item.predictions[row] : UINT32_MAX),
            "GPU prefix predictions/unused tail differ from CPU reference");
      }
    }
    partials.check(); inputs.check(); quotas.check(); results.check();
  }
  // End-to-end real four-lane windows, including an inactive lane and errors in
  // a suffix the correction/quota would otherwise make semantically unused.
  constexpr uint32_t lanes = 4, rows = 4;
  auto workspace = allocateGreedyGPUWorkspace(backend, lanes * rows, kVocabulary);
  Guarded logits(backend, uint64_t{lanes * rows} * kVocabulary * 2);
  Guarded inputs(backend, lanes * rows * 4), quotas(backend, lanes * 4);
  Guarded results(backend, lanes * sizeof(FlashGreedyGPUPrefixResult));
  for (uint32_t variant = 0; variant < 4; ++variant) {
    auto *values = logits.data<uint16_t>();
    std::fill(values, values + uint64_t{lanes * rows} * kVocabulary, uint16_t{0xbf80});
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      quotas.data<uint32_t>()[lane] = lane + 1;
      for (uint32_t row = 0; row < rows; ++row) {
        inputs.data<uint32_t>()[lane * rows + row] = 100 + lane * 10 + row;
        const auto token = 101 + lane * 10 + row;
        values[uint64_t{lane * rows + row} * kVocabulary + token] = 0x3f80;
      }
    }
    if (variant == 1) values[uint64_t{rows - 1} * kVocabulary + kVocabulary - 1] = 0x7fc1;
    if (variant == 2) inputs.data<uint32_t>()[0] = kVocabulary;
    if (variant == 3) quotas.data<uint32_t>()[0] = 0;
    CommandGraph graph;
    addGreedyGPUPrefix(graph, logits.view, workspace, inputs.view, quotas.view,
        results.view, lanes, rows, kVocabulary, 0xb);
    (void)backend.submitCommand(graph.dispatches());
    const auto *found = results.data<FlashGreedyGPUPrefixResult>();
    require(!found[2].retained_rows && !found[2].errors && !found[2].matched_drafts && !found[2].finish,
        "GPU inactive lane did not retain zero");
    if (variant) require(!found[0].retained_rows && found[0].errors == (1u << (variant - 1)),
        "GPU invalid real prefix did not fail closed");
    else require(found[0].retained_rows == 1 && found[0].matched_drafts == 3 &&
        found[0].finish == kFlashGreedyGPUFinishLength && !found[0].errors,
        "GPU full prefix lost pre-quota matches");
    require(found[1].retained_rows == 2 && found[3].retained_rows == 4 &&
        !found[1].errors && !found[3].errors, "GPU bad lane affected healthy peers");
  }
  logits.check(); inputs.check(); quotas.check(); results.check();
  return cases.size() + 4;
}

uint64_t qualifyHost(MetalBackend &backend) {
  auto workspace = allocateGreedyGPUWorkspace(backend, 16, kVocabulary, BufferStorage::Shared);
  auto logits = backend.allocateBuffer(uint64_t{16} * kVocabulary * 2);
  auto results = backend.allocateBuffer(16 * sizeof(FlashGreedyGPURowResult));
  auto inputs = backend.allocateBuffer(16 * 4), quotas = backend.allocateBuffer(4 * 4);
  auto prefixResults = backend.allocateBuffer(4 * sizeof(FlashGreedyGPUPrefixResult));
  uint64_t checks = 0;
  auto rejected = [&](auto callback) {
    bool failed = false;
    CommandGraph graph;
    try { callback(graph); } catch (const std::invalid_argument &) { failed = true; }
    require(failed && graph.dispatches().empty(), "GPU greedy host guard mutated graph or accepted invalid inputs");
    ++checks;
  };
  for (uint32_t rows : {0u, 17u}) rejected([&](auto &g) {
    addGreedyGPU(g, logits, workspace, results, rows, kVocabulary);
  });
  for (uint32_t vocab : {0u, kVocabulary + 1}) rejected([&](auto &g) {
    addGreedyGPU(g, logits, workspace, results, 16, vocab);
  });
  rejected([&](auto &g) { addGreedyGPU(g, logits, workspace, results, 16, kVocabulary, kVocabulary - 1); });
  rejected([&](auto &g) { addGreedyGPU(g, logits, workspace, logits, 16, kVocabulary); });
  rejected([&](auto &g) { addGreedyGPU(g, logits, workspace, results, 16, kVocabulary, kVocabulary + 1); });
  auto malformed = workspace; malformed.partitionCapacity++;
  rejected([&](auto &g) { addGreedyGPU(g, logits, malformed, results, 16, kVocabulary); });
  malformed = workspace; malformed.rowCapacity = 8;
  rejected([&](auto &g) { addGreedyGPU(g, logits, malformed, results, 16, kVocabulary); });
  for (auto geometry : {std::array{0u, 4u, 0u}, std::array{5u, 4u, 0u},
                        std::array{4u, 5u, 0u}, std::array{4u, 4u, 16u}})
    rejected([&](auto &g) { addGreedyGPUPrefix(g, logits, workspace, inputs, quotas,
        prefixResults, geometry[0], geometry[1], kVocabulary, geometry[2]); });
  rejected([&](auto &g) { addGreedyGPUPrefix(g, logits, workspace, inputs, inputs,
      prefixResults, 4, 4, kVocabulary, 15); });
  return checks;
}

void benchmark(MetalBackend &backend, std::ostream &report) {
  bool first = true;
  for (uint32_t rows : {1u, 2u, 4u, 8u, 16u}) {
    auto workspace = allocateGreedyGPUWorkspace(backend, rows, kVocabulary);
    auto logits = backend.allocateBuffer(uint64_t{rows} * kVocabulary * 2);
    auto results = backend.allocateBuffer(rows * sizeof(FlashGreedyGPURowResult));
    auto *values = static_cast<uint16_t *>(logits.contents());
    for (uint64_t id = 0; id < uint64_t{rows} * kVocabulary; ++id)
      values[id] = uint16_t(0xbf80 + (id * 13u % 0x3ff0));
    for (uint32_t row = 0; row < rows; ++row)
      values[uint64_t{row} * kVocabulary + kVocabulary - 1] = 0x7f7f;
    CommandGraph graph;
    addGreedyGPU(graph, logits, workspace, results, rows, kVocabulary);
    for (uint32_t warmup = 0; warmup < 4; ++warmup)
      (void)backend.submitCommand(graph.dispatches());
    std::vector<double> cpuTimes, gpuTimes, wallTimes, repeatedGPUTimes, repeatedWallTimes;
    for (uint32_t sample = 0; sample < 15; ++sample) {
      const auto began = Clock::now();
      constexpr uint32_t cpuRepeats = 32;
      for (uint32_t repeat = 0; repeat < cpuRepeats; ++repeat) {
        asm volatile("" : : "r"(values) : "memory");
        for (uint32_t row = 0; row < rows; ++row)
          sink ^= flashGreedyToken({values + uint64_t{row} * kVocabulary, kVocabulary});
      }
      cpuTimes.push_back(std::chrono::duration<double>(Clock::now() - began).count() / cpuRepeats);
      const auto timing = backend.submitCommand(graph.dispatches());
      for (uint32_t row = 0; row < rows; ++row)
        require(static_cast<FlashGreedyGPURowResult *>(results.contents())[row].token == kVocabulary - 1 &&
            !static_cast<FlashGreedyGPURowResult *>(results.contents())[row].errors,
            "GPU benchmark selected wrong positive maximum");
      gpuTimes.push_back(timing.gpuSeconds); wallTimes.push_back(timing.wallSeconds);
      CommandGraph repeated;
      constexpr uint32_t gpuRepeats = 16;
      for (uint32_t repeat = 0; repeat < gpuRepeats; ++repeat)
        addGreedyGPU(repeated, logits, workspace, results, rows, kVocabulary);
      const auto repeatedTiming = backend.submitCommand(repeated.dispatches());
      repeatedGPUTimes.push_back(repeatedTiming.gpuSeconds / gpuRepeats);
      repeatedWallTimes.push_back(repeatedTiming.wallSeconds / gpuRepeats);
    }
    if (!first) report << ',';
    first = false;
    report << "{\"rows\":" << rows << ",\"vocabulary\":" << kVocabulary
        << ",\"cpu_neon_us\":" << median(cpuTimes) * 1e6
        << ",\"gpu_command_us\":" << median(gpuTimes) * 1e6
        << ",\"command_wall_us\":" << median(wallTimes) * 1e6
        << ",\"same_command_amortized_gpu_us\":" << median(repeatedGPUTimes) * 1e6
        << ",\"same_command_amortized_wall_us\":" << median(repeatedWallTimes) * 1e6
        << ",\"dispatches\":2,\"host_readback_bytes\":" << rows * sizeof(FlashGreedyGPURowResult) << '}';
  }
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if (argc == 2 && std::string(argv[1]) == "--cpu-benchmark") { cpuBenchmark(); return 0; }
      if (argc != 3) throw std::invalid_argument("usage: flash-greedy-gpu-oracle METALLIB REPORT_JSON | --cpu-self-test | --cpu-benchmark");
      MetalBackend backend(argv[1]);
      const auto hostChecks = qualifyHost(backend);
      const bool benchOnly = std::getenv("FLASH_GREEDY_GPU_BENCH_ONLY") &&
          std::string(std::getenv("FLASH_GREEDY_GPU_BENCH_ONLY")) == "1";
      const auto argmaxChecks = benchOnly ? 0 : qualifyArgmax(backend);
      const auto prefixChecks = benchOnly ? 0 : qualifyPrefix(backend);
      std::ofstream report(argv[2]);
      report << std::setprecision(12) << "{\"pass\":true,\"numerical_policy\":"
          << splash::json::quote(kFlashGreedyGPUNumericalPolicy)
          << ",\"host_guard_checks\":" << hostChecks << ",\"argmax_checks\":" << argmaxChecks
          << ",\"prefix_cases\":" << prefixChecks << ",\"benchmark_only\":" << (benchOnly ? "true" : "false")
          << ",\"qualification\":\"all finite encodings host proof; GPU negative logits, signed zero, poisoned strides/tails, every NaN/Inf payload at SIMD/TG/shard/tail boundaries; all 1..16 prefix lengths, mismatch positions, quotas, both EOS IDs, inactive lane and per-lane errors\",\"benchmarks\":[";
      benchmark(backend, report);
      report << "]}\n";
      require(bool(report), "could not write greedy GPU report");
      std::cout << "{\"pass\":true,\"host_guard_checks\":" << hostChecks
          << ",\"argmax_checks\":" << argmaxChecks << ",\"prefix_cases\":" << prefixChecks
          << ",\"report\":" << splash::json::quote(argv[2]) << "}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
