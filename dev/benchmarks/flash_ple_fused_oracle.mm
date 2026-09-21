// Root-run exact primitive qualification. --cpu-self-test submits no Metal work.
#include "flash/FlashPLEFused.hpp"
#include "metal/abi/FlashForward.h"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::flash;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::CommandTiming;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
constexpr uint32_t kSticky = 0x40000000;
constexpr uint8_t kGuard = 0xa7;

void require(bool yes, const std::string &message) {
  if (!yes) throw std::runtime_error(message);
}
uint16_t bf16(float value) {
  const uint32_t raw = std::bit_cast<uint32_t>(value);
  if ((raw & 0x7f800000u) == 0x7f800000u)
    return uint16_t((raw >> 16) | ((raw & 0x7fffffu) ? 0x40u : 0u));
  return uint16_t((raw + 0x7fffu + ((raw >> 16) & 1u)) >> 16);
}
template <class T> MetalBuffer buffer(MetalBackend &backend,
                                      const std::vector<T> &values) {
  auto output = backend.allocateBuffer(values.size() * sizeof(T));
  std::memcpy(output.contents(), values.data(), values.size() * sizeof(T));
  return output;
}
template <class T> FlashTensor tensor(MetalBackend &backend,
    const std::vector<T> &values, FlashDType dtype) {
  return {buffer(backend, values), dtype, {values.size()}, values.size() * sizeof(T)};
}
struct Guarded final {
  MetalBuffer base;
  MetalBuffer view;
  uint64_t bytes;
  Guarded(MetalBackend &backend, uint64_t length) : bytes(length) {
    base = backend.allocateBuffer(length + 128);
    std::memset(base.contents(), kGuard, base.sizeBytes());
    view = backend.view(base, 64, length);
  }
  void verify() const {
    const auto *p = static_cast<const uint8_t *>(base.contents());
    for (uint64_t i = 0; i < 64; ++i)
      require(p[i] == kGuard && p[64 + bytes + i] == kGuard,
                "PLE fused buffer canary changed");
  }
};
std::string hash(const void *bytes, uint64_t count) {
  CC_SHA256_CTX state{};
  CC_SHA256_Init(&state);
  auto *next = static_cast<const uint8_t *>(bytes);
  while (count) {
    const auto n = CC_LONG(std::min<uint64_t>(count, UINT32_MAX));
    CC_SHA256_Update(&state, next, n);
    next += n;
    count -= n;
  }
  std::array<uint8_t, 32> digest{};
  CC_SHA256_Final(digest.data(), &state);
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (auto byte : digest) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}
FlashPLEWeights synthetic(MetalBackend &backend, bool padded) {
  FlashPLEWeights w;
  w.multipliers = tensor(backend,
      std::vector<int64_t>{INT64_MAX, -17, INT64_MIN + 3}, FlashDType::I64);
  std::vector<int64_t> sizes(16, 17), offsets(16);
  for (uint32_t h = 0; h < 16; ++h) offsets[h] = h * 17;
  w.headVocabularySizes = tensor(backend, sizes, FlashDType::I64);
  w.headOffsets = tensor(backend, offsets, FlashDType::I64);
  w.sharedScale = tensor(backend,
      std::vector<uint16_t>{bf16(0.00019931793212890625f)}, FlashDType::BF16);
  const uint64_t strideW = padded ? 96 : 80, strideS = padded ? 16 : 10;
  for (uint32_t shard = 0; shard < 128; ++shard) {
    std::vector<uint8_t> packed(4 * strideW, 0x96);
    std::vector<uint16_t> scales(4 * strideS / 2, 0x42aa), biases(scales.size(), 0xc055);
    for (uint32_t row = 0; row < 4; ++row) {
      for (uint32_t channel = 0; channel < 160; channel += 2) {
        const uint32_t a = (shard * 4 + row + channel * 7) % 16;
        const uint32_t b = (shard * 4 + row + (channel + 1) * 7) % 16;
        packed[row * strideW + channel / 2] = uint8_t(a | (b << 4));
      }
      for (uint32_t g = 0; g < 5; ++g) {
        scales[row * strideS / 2 + g] = bf16(
            (shard % 2 ? -1.0f : 1.0f) * float(g + 1) / 32);
        biases[row * strideS / 2 + g] = bf16(float(int((shard + row) % 17) - 8) / 16);
      }
    }
    // Distinct nonzero source offsets exercise argument encoder view offsets.
    auto addView = [&](const void *data, uint64_t bytes) {
      auto base = backend.allocateBuffer(bytes + 128);
      std::memset(base.contents(), kGuard, base.sizeBytes());
      std::memcpy(static_cast<uint8_t *>(base.contents()) + 64, data, bytes);
      return backend.view(base, 64, bytes);
    };
    w.shards.push_back({addView(packed.data(), packed.size()),
        addView(scales.data(), scales.size() * 2),
        addView(biases.data(), biases.size() * 2), 4, strideW, strideS});
  }
  return w;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
uint32_t repeats() {
  const char *raw = std::getenv("FLASH_PLE_FUSED_REPEATS");
  if (!raw) return 16;
  const auto value = std::stoul(raw);
  require(value && value <= 256, "invalid PLE fused repeat count");
  return uint32_t(value);
}

NSDictionary *run(MetalBackend &backend, const FlashPLEWeights &w,
    const FlashPLEFused &fused, uint32_t lanes, uint32_t rows,
    std::string label, int invalid = 0, bool gatherOnly = false) {
  FlashPLEGeometry geometry;
  geometry.lanes = lanes;
  geometry.rows = rows;
  std::vector<int64_t> tokens(uint64_t(lanes) * rows), history(uint64_t(lanes) * 2);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    history[lane * 2] = lane ? lane * 23 + 7 : geometry.eosToken;
    history[lane * 2 + 1] = lane ? lane * 13 + 5 : geometry.eosToken;
    for (uint32_t row = 0; row < rows; ++row)
      tokens[uint64_t(lane) * rows + row] = row % 7 == 3 ? geometry.eosToken :
          int64_t((uint64_t(row) * 1543 + lane * 9137 + 19) % geometry.vocabularySize);
  }
  if (invalid == 1) tokens[rows / 2] = -1;
  if (invalid == 2) history[0] = -1; // Masked by the initial previous EOS.
  if (invalid == 3) history[1] = -1;
  auto tokenBuffer = buffer(backend, tokens);
  auto controlHistory = buffer(backend, history), fusedHistory = buffer(backend, history);
  const uint64_t selections = uint64_t(lanes) * rows * 16;
  Guarded controlIDs(backend, selections * 8), fusedIDs(backend, selections * 8);
  Guarded controlOutput(backend, selections * 160 * 2), fusedOutput(backend, selections * 160 * 2);
  auto controlDiag = buffer(backend, std::vector<uint32_t>{kSticky});
  auto fusedDiag = buffer(backend, std::vector<uint32_t>{kSticky});
  CommandGraph control, candidate;
  if (gatherOnly) {
    auto *ids = static_cast<int64_t *>(controlIDs.view.contents());
    for (uint64_t i = 0; i < selections; ++i)
      ids[i] = int64_t((i / 2 % 128) * w.shards.front().rows +
          (i % 2 ? w.shards.front().rows - 1 : 0));
    if (invalid) { ids[0] = -1; ids[1] = int64_t(w.tableRows()); }
    std::memcpy(fusedIDs.view.contents(), ids, selections * 8);
    addPLEGather(control, w, controlIDs.view, controlOutput.view, controlDiag, geometry);
    require(fused.addGather(candidate, fusedIDs.view, fusedOutput.view, fusedDiag, geometry),
              "fused direct gather unavailable");
  } else {
    addPLENgramIDs(control, w, tokenBuffer, controlHistory, controlIDs.view, controlDiag, geometry);
    addPLEGather(control, w, controlIDs.view, controlOutput.view, controlDiag, geometry);
    require(fused.addHashGather(candidate, tokenBuffer, fusedHistory, fusedIDs.view,
        fusedOutput.view, fusedDiag, geometry), "fused hash gather unavailable");
  }
  auto reset = [&] {
    std::memcpy(controlHistory.contents(), history.data(), history.size() * 8);
    std::memcpy(fusedHistory.contents(), history.data(), history.size() * 8);
    *static_cast<uint32_t *>(controlDiag.contents()) = kSticky;
    *static_cast<uint32_t *>(fusedDiag.contents()) = kSticky;
  };
  reset();
  static_cast<void>(backend.submitCommand(control.dispatches()));
  static_cast<void>(backend.submitCommand(candidate.dispatches()));
  require(std::memcmp(controlIDs.view.contents(), fusedIDs.view.contents(), selections * 8) == 0,
              "PLE fused IDs differ: " + label);
  require(std::memcmp(controlOutput.view.contents(), fusedOutput.view.contents(),
      selections * 160 * 2) == 0, "PLE fused BF16 gather differs: " + label);
  require(std::memcmp(controlHistory.contents(), fusedHistory.contents(), history.size() * 8) == 0,
              "PLE fused history differs: " + label);
  require(std::memcmp(controlDiag.contents(), fusedDiag.contents(), 4) == 0,
              "PLE fused diagnostics differ: " + label);
  require(std::memcmp(tokenBuffer.contents(), tokens.data(), tokens.size() * 8) == 0,
              "PLE fused input token bytes changed");
  if (!invalid && !gatherOnly) {
    auto expectedHistory = history;
    const auto expected = computePLENgramIDs(tokens, expectedHistory,
        {static_cast<const int64_t *>(w.multipliers.buffer.contents()), 3},
        {static_cast<const int64_t *>(w.headVocabularySizes.buffer.contents()), 16},
        {static_cast<const int64_t *>(w.headOffsets.buffer.contents()), 16},
        geometry, w.tableRows());
    require(std::memcmp(expected.data(), fusedIDs.view.contents(), selections * 8) == 0 &&
        std::memcmp(expectedHistory.data(), fusedHistory.contents(), history.size() * 8) == 0,
            "PLE fused independent CPU hash mismatch");
  }
  for (const auto *guard : {&controlIDs, &fusedIDs, &controlOutput, &fusedOutput}) guard->verify();
  std::vector<double> controlGPU, fusedGPU, controlWall, fusedWall;
  if (!invalid) {
    const auto count = repeats();
    for (uint32_t sample = 0; sample < count + 3; ++sample) {
      reset();
      CommandTiming a, b;
      // Alternate order to avoid systematic residency/thermal bias.
      if (sample % 2) {
        b = backend.submitCommand(candidate.dispatches());
        a = backend.submitCommand(control.dispatches());
      } else {
        a = backend.submitCommand(control.dispatches());
        b = backend.submitCommand(candidate.dispatches());
      }
      if (sample >= 3) {
        controlGPU.push_back(a.gpuSeconds); fusedGPU.push_back(b.gpuSeconds);
        controlWall.push_back(a.wallSeconds); fusedWall.push_back(b.wallSeconds);
      }
    }
  }
  std::cout << label << " B=" << lanes << " R=" << rows << " exact PASS";
  if (!controlGPU.empty()) std::cout << " controlGPU=" << median(controlGPU) * 1e6
      << "us fusedGPU=" << median(fusedGPU) * 1e6 << "us";
  std::cout << std::endl;
  return @{@"label": @(label.c_str()), @"lanes": @(lanes), @"rows": @(rows),
      @"invalid_case": @(invalid), @"gather_only": @(gatherOnly),
      @"exact": @YES, @"bf16_elements": @(selections * 160),
      @"output_sha256": @(hash(fusedOutput.view.contents(), fusedOutput.bytes).c_str()),
      @"ids_sha256": @(hash(fusedIDs.view.contents(), fusedIDs.bytes).c_str()),
      @"diagnostics": @(*static_cast<const uint32_t *>(fusedDiag.contents())),
      @"control_dispatches": @(control.dispatches().size()),
      @"candidate_dispatches": @(candidate.dispatches().size()),
      @"control_gpu_seconds": @(controlGPU.empty() ? 0 : median(controlGPU)),
      @"candidate_gpu_seconds": @(fusedGPU.empty() ? 0 : median(fusedGPU)),
      @"control_wall_seconds": @(controlWall.empty() ? 0 : median(controlWall)),
      @"candidate_wall_seconds": @(fusedWall.empty() ? 0 : median(fusedWall))};
}

void negativeGuards(MetalBackend &backend, const FlashPLEWeights &w,
                    const char *metallib) {
  auto scalar = backend.allocateBuffer(64);
  auto expect = [&](auto fn) {
    bool rejected = false;
    try { fn(); } catch (const splash::metal::MetalBackendError &) { rejected = true; }
    require(rejected, "argument-buffer invalid binding was accepted");
  };
  expect([&] { static_cast<void>(backend.makeReadOnlyArgumentBuffer(
      "flash_ple_hash_gather128", 0, std::vector<splash::metal::BufferBinding>{})); });
  expect([&] { static_cast<void>(backend.makeReadOnlyArgumentBuffer(
      "flash_ple_hash_gather128", 0,
      std::vector<splash::metal::BufferBinding>{{0, scalar}, {0, scalar}})); });
  expect([&] { static_cast<void>(backend.makeReadOnlyArgumentBuffer(
      "flash_ple_hash_gather128", 0,
      std::vector<splash::metal::BufferBinding>{{1, scalar}})); });
  expect([&] { static_cast<void>(backend.makeReadOnlyArgumentBuffer(
      "flash_ple_hash_gather128", 0,
      std::vector<splash::metal::BufferBinding>{{0, {}}})); });
  expect([&] { static_cast<void>(backend.readOnlyArgumentBufferByteCount("missing_function", 0)); });
  expect([&] { static_cast<void>(backend.readOnlyArgumentBufferByteCount("flash_ple_hash_gather128", 31)); });
  // Complete reflected pointer layout is mandatory; no null trailing slots.
  expect([&] { static_cast<void>(backend.makeReadOnlyArgumentBuffer(
      "flash_ple_hash_gather128", 0,
      std::vector<splash::metal::BufferBinding>{{0, scalar}})); });
  MetalBackend other(metallib);
  auto foreign = other.allocateBuffer(64);
  expect([&] { static_cast<void>(backend.makeReadOnlyArgumentBuffer(
      "flash_ple_hash_gather128", 0,
      std::vector<splash::metal::BufferBinding>{{0, foreign}})); });
  auto bad = w;
  bad.shards.pop_back();
  bool rejected = false;
  try { FlashPLEFused fail(backend, bad); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "fused table accepted 127 shards");
}

void ticketOwnership(MetalBackend &backend) {
  auto w = synthetic(backend, false);
  auto fused = std::make_unique<FlashPLEFused>(backend, w);
  FlashPLEGeometry geometry;
  geometry.rows = 16;
  std::vector<int64_t> selections(256);
  for (uint32_t i = 0; i < selections.size(); ++i)
    selections[i] = int64_t((i / 2) * 4 + (i % 2 ? 3 : 0));
  auto ids = buffer(backend, selections);
  auto expected = backend.allocateBuffer(256 * 160 * 2);
  auto actual = backend.allocateBuffer(expected.sizeBytes());
  auto diagnostics = buffer(backend, std::vector<uint32_t>{kSticky});
  CommandGraph control;
  addPLEGather(control, w, ids, expected, diagnostics, geometry);
  static_cast<void>(backend.submitCommand(control.dispatches()));
  CommandGraph candidate;
  require(fused->addGather(candidate, ids, actual, diagnostics, geometry),
            "ticket ownership candidate unavailable");
  auto ticket = backend.submitCommandAsync(candidate.dispatches());
  bool safePointRejected = false;
  try { FlashPLEFused fail(backend, w); }
  catch (const splash::metal::MetalBackendError &) { safePointRejected = true; }
  require(safePointRejected, "argument-buffer creation accepted outstanding ticket");
  // Only the pending ticket and Metal command now own the pointer buffer and
  // its source allocations. Temporary views and the operator have vanished.
  control = CommandGraph{};
  candidate = CommandGraph{};
  fused.reset();
  w = FlashPLEWeights{};
  static_cast<void>(ticket.wait());
  require(std::memcmp(actual.contents(), expected.contents(), actual.sizeBytes()) == 0,
            "indirect source allocations did not survive pending ticket");
  std::cout << "PLE fused asynchronous indirect-source lifetime PASS" << std::endl;
}

FlashPLEWeights mixedPrivate(MetalBackend &backend, const FlashPLEWeights &shared) {
  auto result = shared;
  const auto &source = shared.shards[0].weights;
  auto base = backend.allocateBuffer(source.sizeBytes() + 128, BufferStorage::Private);
  auto destination = backend.view(base, 64, source.sizeBytes());
  CommandGraph copy;
  copy.add("flash_forward_copy_words", {source, destination},
      FlashForwardCopyParams{source.sizeBytes() / 4},
      {(source.sizeBytes() / 4 + 255) / 256, 1, 1}, {256, 1, 1});
  static_cast<void>(backend.submitCommand(copy.dispatches()));
  result.shards[0].weights = destination;
  return result;
}

void cpuSelfTest() {
  FlashPLEGeometry geometry;
  const std::vector<int64_t> tokens{19}, multipliers{INT64_MAX, -17, INT64_MIN + 3};
  std::vector<int64_t> history{geometry.eosToken, geometry.eosToken}, sizes(16, 17), offsets(16);
  for (uint32_t h = 0; h < 16; ++h) offsets[h] = h * 17;
  const auto ids = computePLENgramIDs(tokens, history, multipliers, sizes, offsets, geometry, 512);
  require(ids.size() == 16 && history == std::vector<int64_t>{geometry.eosToken, 19},
              "CPU PLE reference extent/history failed");
  for (uint32_t h = 0; h < 16; ++h)
    require(ids[h] >= offsets[h] && ids[h] < offsets[h] + sizes[h], "CPU PLE modulo range failed");
  std::cout << "PLE fused oracle CPU self-test PASS; no Metal backend created" << std::endl;
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc == 2 && std::string(argv[1]) == "--cpu-self-test") { cpuSelfTest(); return 0; }
      if (argc == 3 && std::string(argv[1]) == "--abi-self-test") {
        MetalBackend backend(argv[2]);
        if (!backend.supportsArgumentBuffersTier2()) return 77;
        auto tiny = synthetic(backend, false);
        FlashPLEFused fused(backend, tiny);
        negativeGuards(backend, tiny, argv[2]);
        require(fused.allocatedBytes() <= kFlashPLEFusedMaximumArgumentBytes,
            "PLE fused ABI admission bound failed");
        require(backend.submissionCount() == 0, "ABI self-test submitted GPU commands");
        std::cout << "PLE fused reflected ABI/negative bindings PASS; argument bytes="
            << fused.allocatedBytes() << "; zero GPU submissions" << std::endl;
        return 0;
      }
      require(argc == 3 || argc == 4,
          "usage: flash-ple-fused-oracle <metallib> <report.json> [local-package]");
      MetalBackend backend(argv[1]);
      if (!backend.supportsArgumentBuffersTier2()) {
        std::cout << "SKIP: Metal Tier 2 argument buffers unavailable; qualified control retained" << std::endl;
        return 77;
      }
      auto tiny = synthetic(backend, false);
      FlashPLEFused fused(backend, tiny);
      negativeGuards(backend, tiny, argv[1]);
      ticketOwnership(backend);
      NSMutableArray *cases = [NSMutableArray array];
      for (uint32_t rows : {1u, 2u, 4u, 8u, 16u, 128u, 512u, 2048u})
        [cases addObject:run(backend, tiny, fused, 1, rows, "synthetic")];
      for (uint32_t rows : {1u, 4u, 128u, 512u})
        [cases addObject:run(backend, tiny, fused, 4, rows, "synthetic-batch")];
      [cases addObject:run(backend, tiny, fused, 1, 16, "all-shard-boundaries", 0, true)];
      [cases addObject:run(backend, tiny, fused, 1, 16, "invalid-gather-IDs", 1, true)];
      for (int invalid : {1, 2, 3})
        [cases addObject:run(backend, tiny, fused, 2, 8, "invalid-hash-input", invalid)];
      auto padded = synthetic(backend, true);
      FlashPLEFused paddedFused(backend, padded);
      [cases addObject:run(backend, padded, paddedFused, 1, 16, "padded-source-strides", 0, true)];
      [cases addObject:run(backend, padded, paddedFused, 4, 8, "padded-hash")];
      auto privateWeights = mixedPrivate(backend, tiny);
      FlashPLEFused privateFused(backend, privateWeights);
      [cases addObject:run(backend, privateWeights, privateFused, 1, 16,
          "mixed-private-shared-source-views", 0, true)];
      auto badParams = tiny;
      auto badSizes = std::vector<int64_t>(16, 17); badSizes[5] = 0;
      badParams.headVocabularySizes = tensor(backend, badSizes, FlashDType::I64);
      FlashPLEFused badFused(backend, badParams);
      [cases addObject:run(backend, badParams, badFused, 1, 4, "invalid-head-size", 4)];
      auto badOffsets = std::vector<int64_t>(16, 0); badOffsets[9] = INT64_MAX;
      badParams.headVocabularySizes = tiny.headVocabularySizes;
      badParams.headOffsets = tensor(backend, badOffsets, FlashDType::I64);
      FlashPLEFused badOffsetFused(backend, badParams);
      [cases addObject:run(backend, badParams, badOffsetFused, 2, 4, "invalid-head-offset", 4)];
      std::string sourceIdentity = "synthetic-original-Q4-G32";
      if (argc == 4) {
        auto source = FlashWeights::load(backend, argv[3]);
        sourceIdentity = source.sourceIdentity();
        auto original = FlashPLEWeights::fromWeights(source);
        FlashPLEFused originalFused(backend, original);
        for (uint32_t rows : {1u, 8u, 128u, 512u, 2048u})
          [cases addObject:run(backend, original, originalFused, 1, rows, "original-checkpoint")];
        [cases addObject:run(backend, original, originalFused, 4, 512, "original-checkpoint-batch")];
        [cases addObject:run(backend, original, originalFused, 1, 16, "original-all-shard-boundaries", 0, true)];
      }
      const auto &sha = backend.metallibSha256();
      static constexpr char digits[] = "0123456789abcdef";
      std::string libHash;
      for (auto byte : sha) { libHash += digits[byte >> 4]; libHash += digits[byte & 15]; }
      NSDictionary *report = @{@"schema": @"splash-flash-ple-fused-qualification-v1",
          @"exact": @YES, @"device": @(backend.capabilities().deviceName.c_str()),
          @"source_identity": @(sourceIdentity.c_str()), @"metallib_sha256": @(libHash.c_str()),
          @"argument_buffer_bytes": @(fused.allocatedBytes()),
          @"negative_binding_guards": @YES, @"async_indirect_lifetime": @YES,
          @"shader_validation": @(std::getenv("MTL_SHADER_VALIDATION") ?: "unset"),
          @"repeats": @(repeats()), @"cases": cases};
      NSError *error = nil;
      NSData *json = [NSJSONSerialization dataWithJSONObject:report
          options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
      require(json != nil, "cannot encode PLE fused report");
      require([json writeToFile:@(argv[2]) atomically:YES], "cannot write PLE fused report");
      std::cout << "PLE fused all cases PASS: " << cases.count << std::endl;
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "PLE fused oracle FAIL: " << error.what() << std::endl;
      return 1;
    }
  }
}
