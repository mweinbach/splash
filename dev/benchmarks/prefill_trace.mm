// Metadata-only prefill attribution. Baselines use the ordinary executor;
// counter traces explicitly report any encoder/barrier perturbation.
#include "engine/Engine.hpp"
#include "engine/Json.hpp"
#include "engine/MemoryGovernor.hpp"
#include "metal/ProfilingJson.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include "ops/Q8PageStorage.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {
using Clock = std::chrono::steady_clock;

uint32_t count(std::string_view text, std::string_view option, bool zero = false) {
  uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      (!zero && !value))
    throw std::invalid_argument(std::string(option) + " requires an integer");
  return value;
}

std::vector<uint32_t> prompt(uint32_t tokens) {
  constexpr std::array<uint32_t, 3> user{248045, 846, 198};
  constexpr std::array<uint32_t, 17> sentence{
      7734, 264, 11346, 11, 7072, 12, 26829, 8627, 883,
      279, 3712, 314, 279, 12386, 19825, 13, 220};
  constexpr std::array<uint32_t, 9> assistant{
      248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
  if (tokens <= user.size() + assistant.size())
    throw std::invalid_argument("prompt is too short for a chat request");
  std::vector<uint32_t> result(user.begin(), user.end());
  for (uint32_t index = 0; index < tokens - user.size() - assistant.size(); ++index)
    result.push_back(sentence[index % sentence.size()]);
  result.insert(result.end(), assistant.begin(), assistant.end());
  return result;
}

struct Chunk final {
  uint32_t begin = 0, end = 0, totalRows = 0, draftRows = 0, snapshots = 0;
  uint64_t snapshotBytes = 0;
  double runtimeSeconds = 0.0, snapshotSeconds = 0.0;
};

struct Run final {
  std::string label;
  uint32_t sample = 0;
  double fullSeconds = 0.0, gpuSeconds = 0.0, legacyWallSeconds = 0.0;
  uint64_t targetRows = 0, draftRows = 0, submissions = 0, droppedPhases = 0;
  std::vector<Chunk> chunks;
  std::vector<model::ModelPhaseProfile> phases;
  std::vector<metal::CommandDispatchProfile> commands;
};

void validateProfiles(const Run &value, uint32_t width, bool cpuProfile,
                      metal::CommandDispatchProfilingMode mode) {
  if (value.submissions != value.chunks.size())
    throw std::runtime_error("prefill did not submit one command per chunk");
  if (value.droppedPhases)
    throw std::runtime_error("model CPU phase profiles were dropped");
  if (!cpuProfile && !value.phases.empty())
    throw std::runtime_error("disabled model CPU profiling retained spans");
  if (mode == metal::CommandDispatchProfilingMode::Off) {
    if (!value.commands.empty())
      throw std::runtime_error("disabled backend profiling retained commands");
  } else if (value.commands.size() != value.submissions) {
    throw std::runtime_error("backend command profile collection is incomplete");
  }
  if (!cpuProfile)
    return;
  if (value.phases.size() != value.chunks.size() * 3)
    throw std::runtime_error("model CPU phase profile collection is incomplete");
  constexpr std::array stages{model::ModelPhaseStage::GraphBuild,
                              model::ModelPhaseStage::TicketWait,
                              model::ModelPhaseStage::CompletionHost};
  uint64_t previousSequence = 0;
  for (size_t index = 0; index < value.chunks.size(); ++index) {
    const Chunk &chunk = value.chunks[index];
    const auto &first = value.phases[index * stages.size()];
    if (!first.commandSequence || first.commandSequence <= previousSequence)
      throw std::runtime_error("model CPU phase command sequence is not ordered");
    previousSequence = first.commandSequence;
    double previousEnd = 0.0;
    for (size_t stage = 0; stage < stages.size(); ++stage) {
      const auto &phase = value.phases[index * stages.size() + stage];
      if (phase.commandSequence != first.commandSequence ||
          phase.kind != WorkKind::Prefill || phase.stage != stages[stage] ||
          phase.lanes != width || phase.rows != chunk.totalRows ||
          phase.draftContextRows != chunk.draftRows || !phase.dispatches ||
          phase.dispatches != first.dispatches)
        throw std::runtime_error("model CPU phase metadata does not match the chunk");
      if (!std::isfinite(phase.beganSteadySeconds) ||
          !std::isfinite(phase.endedSteadySeconds) ||
          !std::isfinite(phase.wallSeconds) ||
          phase.beganSteadySeconds < previousEnd ||
          phase.endedSteadySeconds < phase.beganSteadySeconds ||
          phase.wallSeconds < 0.0)
        throw std::runtime_error("model CPU phase timestamps are invalid");
      previousEnd = phase.endedSteadySeconds;
      for (uint32_t lane = 0; lane < width; ++lane)
        if (phase.logicalBegin[lane] != chunk.begin ||
            phase.logicalEnd[lane] != chunk.end)
          throw std::runtime_error("model CPU phase logical range changed");
    }
    if (mode != metal::CommandDispatchProfilingMode::Off &&
        value.commands[index].sequence != first.commandSequence)
      throw std::runtime_error("model and backend command profile sequences differ");
  }
}

Run run(metal::MetalBackend &backend, model::Runtime &runtime,
        std::span<const uint32_t> input, uint32_t width, uint32_t pagesPerLane,
        uint32_t checkpoint, std::string label, uint32_t sample,
        bool cpuProfile, metal::CommandDispatchProfilingMode mode) {
  backend.setCommandDispatchProfiling(mode);
  runtime.setModelPhaseProfiling(cpuProfile);
  static_cast<void>(backend.takeCommandDispatchProfiles());
  static_cast<void>(runtime.takeModelPhaseProfiles());
  const uint64_t droppedBefore = runtime.modelPhaseProfilesDropped();
  std::vector<uint32_t> boundaries;
  const uint32_t replayBoundary = (input.size() - 1) / kv::kPageTokens * kv::kPageTokens;
  if (checkpoint)
    for (uint64_t boundary = checkpoint; boundary < replayBoundary; boundary += checkpoint)
      boundaries.push_back(static_cast<uint32_t>(boundary));
  if (replayBoundary)
    boundaries.push_back(replayBoundary);
  const DraftContextPlan draftPlan =
      planDraftContext(0, input.size(), std::nullopt, boundaries);
  std::array<std::vector<uint32_t>, model::ExecutionLimits::maximumBatchWidth> lanePages;
  for (uint32_t lane = 0; lane < width; ++lane) {
    EngineRequest request;
    request.id = lane + 1;
    request.prompt.assign(input.begin(), input.end());
    request.maxNewTokens = 16;
    runtime.beginColdRequest(request.modelView(), lane);
    runtime.setDraftContextPlan(request.id, draftPlan);
    for (uint32_t page = 0; page < pagesPerLane; ++page)
      lanePages[lane].push_back(lane * pagesPerLane + page);
  }
  // Request initialization is outside both prefill timing scopes.
  static_cast<void>(backend.takeCommandDispatchProfiles());
  const auto before = runtime.telemetry();
  const uint64_t submissionsBefore = backend.submissionCount();
  Run result;
  result.label = std::move(label);
  result.sample = sample;
  const auto began = Clock::now();
  uint32_t offset = 0;
  size_t boundaryIndex = 0;
  std::vector<std::shared_ptr<const CompositeState>> snapshots(width);
  while (offset < input.size()) {
    uint32_t end = std::min<uint32_t>(input.size(), offset +
        model::ExecutionLimits::prefillTokenBudget / width);
    if (boundaryIndex < boundaries.size())
      end = std::min(end, boundaries[boundaryIndex]);
    BatchPlan plan{WorkKind::Prefill, BatchCohort::Greedy, {}, DecodeStage::Regular};
    std::vector<ModelBatchItem> items;
    for (uint32_t lane = 0; lane < width; ++lane) {
      plan.items.push_back({lane + 1, end - offset, offset});
      items.push_back({lane + 1, lane, offset, offset, end - offset, lanePages[lane]});
      items.back().inputTokens = input.subspan(offset, end - offset);
    }
    Chunk chunk{offset, end, width * (end - offset)};
    const auto chunkBegin = Clock::now();
    const auto results = runtime.prefill(plan, items);
    chunk.runtimeSeconds = std::chrono::duration<double>(Clock::now() - chunkBegin).count();
    if (results.size() != width || std::any_of(results.begin(), results.end(),
        [&](const auto &step) { return step.consumedPromptTokens != end - offset; }))
      throw std::runtime_error("prefill consumed the wrong row count");
    for (uint32_t lane = 0; lane < width; ++lane)
      if (results[lane].requestId != lane + 1)
        throw std::runtime_error("prefill result lane order changed");
    for (const auto &capture : draftCaptureSpansForDispatch(draftPlan, offset, end))
      chunk.draftRows += width * (capture.absoluteEnd - capture.absoluteBegin);
    if (boundaryIndex < boundaries.size() && end == boundaries[boundaryIndex]) {
      const auto snapshotBegin = Clock::now();
      for (uint32_t lane = 0; lane < width; ++lane) {
        snapshots[lane].reset();
        snapshots[lane] = runtime.snapshot(lane + 1);
        if (!snapshots[lane])
          throw std::runtime_error("could not materialize the prefill state");
        ++chunk.snapshots;
        chunk.snapshotBytes += snapshots[lane]->bytes();
      }
      chunk.snapshotSeconds = std::chrono::duration<double>(Clock::now() - snapshotBegin).count();
      ++boundaryIndex;
    }
    // The backend retains only eight profiles. Drain each finished command so
    // long/B4 runs preserve the earlier chunks as well as the latest ones.
    if (mode != metal::CommandDispatchProfilingMode::Off) {
      auto profiles = backend.takeCommandDispatchProfiles();
      for (auto &profile : profiles)
        result.commands.push_back(std::move(profile));
    }
    result.chunks.push_back(chunk);
    offset = end;
  }
  result.fullSeconds = std::chrono::duration<double>(Clock::now() - began).count();
  const auto after = runtime.telemetry();
  result.gpuSeconds = after.totalPrefillGpuSeconds - before.totalPrefillGpuSeconds;
  result.legacyWallSeconds = after.totalPrefillWallSeconds - before.totalPrefillWallSeconds;
  result.targetRows = after.targetPrefillRows - before.targetPrefillRows;
  result.draftRows = after.draftContextRowsActive + after.draftContextRowsMaterialization -
      before.draftContextRowsActive - before.draftContextRowsMaterialization;
  result.submissions = backend.submissionCount() - submissionsBefore;
  result.phases = runtime.takeModelPhaseProfiles();
  auto remainingCommands = backend.takeCommandDispatchProfiles();
  for (auto &profile : remainingCommands)
    result.commands.push_back(std::move(profile));
  result.droppedPhases = runtime.modelPhaseProfilesDropped() - droppedBefore;
  if (result.targetRows != input.size() * width)
    throw std::runtime_error("target prefill work changed");
  if (result.draftRows != draftPlan.draftContextRows() * width ||
      boundaryIndex != boundaries.size())
    throw std::runtime_error("draft materialization work changed");
  validateProfiles(result, width, cpuProfile, mode);
  for (uint32_t lane = 0; lane < width; ++lane)
    runtime.end(lane + 1);
  backend.setCommandDispatchProfiling(metal::CommandDispatchProfilingMode::Off);
  runtime.setModelPhaseProfiling(false);
  return result;
}

void write(std::ostream &out, const Run &value) {
  out << "{\"label\":" << json::quote(value.label) << ",\"sample\":" << value.sample
      << ",\"full_seconds\":" << value.fullSeconds << ",\"prefill_gpu_seconds\":" << value.gpuSeconds
      << ",\"legacy_prefill_wall_seconds\":" << value.legacyWallSeconds
      << ",\"target_rows\":" << value.targetRows << ",\"draft_rows\":" << value.draftRows
      << ",\"submissions\":" << value.submissions << ",\"dropped_phases\":" << value.droppedPhases
      << ",\"chunks\":[";
  for (size_t index = 0; index < value.chunks.size(); ++index) {
    if (index) out << ',';
    const auto &chunk = value.chunks[index];
    out << "{\"begin\":" << chunk.begin << ",\"end\":" << chunk.end
        << ",\"rows\":" << chunk.totalRows << ",\"draft_rows\":" << chunk.draftRows
        << ",\"snapshots\":" << chunk.snapshots << ",\"snapshot_bytes\":" << chunk.snapshotBytes
        << ",\"runtime_seconds\":" << chunk.runtimeSeconds
        << ",\"snapshot_seconds\":" << chunk.snapshotSeconds << '}';
  }
  out << "],\"phases\":[";
  for (size_t index = 0; index < value.phases.size(); ++index) {
    if (index) out << ',';
    profiling::writeJson(out, value.phases[index]);
  }
  out << "],\"commands\":[";
  for (size_t index = 0; index < value.commands.size(); ++index) {
    if (index) out << ',';
    profiling::writeJson(out, value.commands[index]);
  }
  out << "]}";
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 3)
        throw std::invalid_argument("usage: prefill-trace METALLIB MODEL_ROOT --output FILE "
            "[--prompt-tokens N] [--width 1..4] [--samples N] [--warmups N] "
            "[--checkpoint-tokens N] [--gpu-profile off|command|dispatch|stage]");
      uint32_t tokens = 256, width = 1, samples = 2, warmups = 1;
      uint32_t checkpoint = EngineConfig{}.prefillCheckpointTokens;
      std::filesystem::path output;
      auto mode = metal::CommandDispatchProfilingMode::Command;
      for (int index = 3; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("option requires a value");
        const std::string_view option(argv[index]), value(argv[index + 1]);
        if (option == "--output") output = value;
        else if (option == "--prompt-tokens") tokens = count(value, option);
        else if (option == "--width") width = count(value, option);
        else if (option == "--samples") samples = count(value, option);
        else if (option == "--warmups") warmups = count(value, option, true);
        else if (option == "--checkpoint-tokens") checkpoint = count(value, option, true);
        else if (option == "--gpu-profile") {
          if (value == "off") mode = metal::CommandDispatchProfilingMode::Off;
          else if (value == "command") mode = metal::CommandDispatchProfilingMode::Command;
          else if (value == "dispatch") mode = metal::CommandDispatchProfilingMode::DispatchBoundary;
          else if (value == "stage") mode = metal::CommandDispatchProfilingMode::StagePerDispatch;
          else throw std::invalid_argument("invalid GPU profiling mode");
        } else throw std::invalid_argument("unknown option");
      }
      if (output.empty() || width > model::ExecutionLimits::maximumBatchWidth || tokens > 32768 ||
          (checkpoint && (checkpoint < model::ExecutionLimits::draftContextTokens || checkpoint % kv::kPageTokens)))
        throw std::invalid_argument("invalid output, width, prompt length, or checkpoint interval");
      metal::MetalBackend backend(argv[1]);
      auto package = model::loadModelPackage(backend, argv[2]);
      ops::ExecutionPlans operators(backend.capabilities());
      const auto memory = model::plannedRuntimeMemory(backend.capabilities(), package, operators);
      const uint32_t pagesPerLane = (tokens + 256) / kv::kPageTokens + 2;
      const uint32_t mappingPages = package.targetKvLayout().sparseMappingBatchPages();
      const uint32_t pageCount = (pagesPerLane * width + mappingPages - 1) / mappingPages * mappingPages;
      MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
      kv::Q8PageStorage pages(backend, governor.allocationAdmission(), package.targetKvLayout(), pageCount);
      for (uint32_t page = 0; page < pageCount; ++page)
        if (!pages.ensureResident(page)) throw std::runtime_error("could not back the KV pages");
      model::QwenStateStorage states(backend, governor.allocationAdmission(), package.stateLayout());
      model::Runtime runtime({backend, governor.allocationAdmission(), package, pages, states, operators,
          std::max<uint32_t>(16384, tokens + 256), memory.pipelineReserveBytes, memory.runtimeOverheadReserveBytes});
      const auto input = prompt(tokens);
      for (uint32_t index = 0; index < warmups; ++index)
        static_cast<void>(run(backend, runtime, input, width, pagesPerLane, checkpoint,
            "warmup", index, false, metal::CommandDispatchProfilingMode::Off));
      std::ofstream out(output);
      if (!out) throw std::runtime_error("could not open trace output");
      out << std::setprecision(17) << "{\"schema_version\":1,\"device\":"
          << json::quote(backend.capabilities().deviceName) << ",\"prompt_tokens\":" << tokens
          << ",\"width\":" << width << ",\"checkpoint_tokens\":" << checkpoint
          << ",\"gpu_profile_mode\":" << json::quote(metal::commandDispatchProfilingModeName(mode))
          << ",\"capability\":";
      profiling::writeJson(out, backend.commandDispatchProfilingCapability());
      out << ",\"scope\":\"Direct cold runtime with Page32 materialization boundaries and uniform lane chunks. "
          "One rolling snapshot copy per lane is included; Engine scheduling/cache bookkeeping and HTTP are excluded. "
          "Engine may reuse shared cache states instead of making these per-lane copies. Ticket waits overlap GPU execution. "
          "Enabled traces include metadata collection overhead; counter modes report encoder/barrier perturbation. "
          "Baseline runs use the ordinary uninstrumented executor.\",\"runs\":[";
      bool first = true;
      for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t pass = 0; pass < 3; ++pass) {
          auto value = run(backend, runtime, input, width, pagesPerLane, checkpoint,
              pass == 1 ? "trace" : pass == 0 ? "baseline_before" : "baseline_after",
              sample, pass == 1, pass == 1 ? mode : metal::CommandDispatchProfilingMode::Off);
          if (!first) out << ',';
          first = false;
          write(out, value);
          out.flush();
          std::cerr << value.label << " sample=" << sample << " gpu_ms=" << value.gpuSeconds * 1000
              << " runtime_and_state_ms=" << value.fullSeconds * 1000 << '\n';
        }
      }
      out << "]}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "prefill-trace: " << error.what() << '\n';
      return 1;
    }
  }
}
