"""Private small-row policy and CPU graph-counter phase attribution."""
from __future__ import annotations


def replace(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise RuntimeError(f"Private phase attribution source drift: {before!r}")
    return text.replace(before, after)


def transform(relative: str, text: str) -> str:
    if relative == "runtime/flash/FlashForward.cpp":
        marker = '      (impl_->allRowsInt8Target ? ";private-allrows-full512-target-m16-below256-v1" : "") +'
        return replace(text, marker, marker + '''
      (impl_->int8ExpertStore && impl_->int8ExpertStore->gatheredQMVEnabled()
          ? std::string(";") + std::string(gathered_i8_qmv::numericalPolicy(impl_->int8ExpertStore->gatheredQMVColumns()))
          : "") +''')
    if relative != "runtime/flash/FlashWorker.mm":
        return text
    marker = '      << R"(,"target_all_rows_full512":true,"original_target_gpu_omitted":true)"'
    text = replace(text, marker, marker + '''
      << R"(,"target_prefill_expert_policy":"unchanged signed-I8 M16/M32/M64 for physical rows17..8192")"
      << R"(,"target_small_row_expert_scope":"physical rows1..16; decode/verifier and tiny prefill/tails")"
      << R"(,"target_gathered_qmv_enabled":)" << (persistedExperts && persistedExperts->gatheredQMVEnabled() ? "true" : "false")
      << R"(,"target_gathered_qmv_columns":)" << (persistedExperts ? persistedExperts->gatheredQMVColumns() : 1)
      << R"(,"target_small_row_numerical_policy":)" << json::quote(persistedExperts && persistedExperts->gatheredQMVEnabled()
          ? std::string(gathered_i8_qmv::numericalPolicy(persistedExperts->gatheredQMVColumns()))
          : "existing MPP M16 signed-I8 late row scale")''')
    text = replace(text, '        words_((weights.descriptor().vocabularySize + 31) / 32) {}', '''        words_((weights.descriptor().vocabularySize + 31) / 32) {
    if (const auto *store = forward_.batchInt8ExpertStore()) lastExpertCounters_ = store->graphCounters();
  }''')
    marker = "  void traceRequestCommand(const char *phase, const char *role, RequestCookie cookie,"
    text = replace(text, marker, METHODS + marker)
    text = replace(text, "    noteUserGPUCommand(timing);\n    if (!requestCommandTrace_) return;\n    const std::array<FlashRequestTraceLane, 1>", "    noteUserGPUCommand(timing);\n    attributeExpertPhase(phase, role, 1, rows);\n    if (!requestCommandTrace_) return;\n    const std::array<FlashRequestTraceLane, 1>")
    text = replace(text, "    noteUserGPUCommand(timing);\n    if (!requestCommandTrace_) return;\n    std::array<FlashRequestTraceLane, kConcurrent>", """    noteUserGPUCommand(timing);
    uint32_t sourceRows = 0;
    if (!width || width > kConcurrent) throw std::logic_error("invalid expert-phase width");
    for (uint32_t lane = 0; lane < width; ++lane) sourceRows += rowsAt(lane);
    attributeExpertPhase(phase, role, width, sourceRows);
    if (!requestCommandTrace_) return;
    std::array<FlashRequestTraceLane, kConcurrent>""")
    text = replace(text, "  Transport &transport_;", """  struct ExpertPhase {
    uint64_t commands = 0, sourceRows = 0, gateCalls = 0, gateRows = 0, downCalls = 0, downRows = 0;
    uint64_t qmvGateCalls = 0, qmvGateRows = 0, qmvDownCalls = 0, qmvDownRows = 0;
    std::array<uint64_t, 4> widths{};
    std::array<uint64_t, 3> rowBuckets{};
  };
  std::array<ExpertPhase, 3> expertPhases_{};
  FlashInt8ExpertStoreGraphCounters lastExpertCounters_{};
  Transport &transport_;""")
    marker = '      << R"(,"scheduler":{"mode":)"'
    text = replace(text, marker, '      << R"(,"expert_kernel_phase_attribution":)" << expertPhaseJSON()\n' + marker)
    return text


METHODS = r'''
  void attributeExpertPhase(const char *phase, const char *role, uint32_t width, uint32_t rows) {
    if (std::string_view(role) != "target_trunk") return;
    const auto *store = forward_.batchInt8ExpertStore();
    if (!store) return;
    const uint32_t index = std::string_view(phase) == "prefill" ? 0 :
        std::string_view(phase) == "autoregressive_decode" ? 1 :
        std::string_view(phase) == "target_verify" ? 2 : 3;
    if (index == 3 || !rows || !width || width > 4)
      throw std::logic_error("unclassified private target expert phase");
    const auto now = store->graphCounters();
    const auto delta = [](uint64_t value, uint64_t previous) {
      if (value < previous) throw std::logic_error("private expert phase counter regressed");
      return value - previous;
    };
    auto &out = expertPhases_[index]; ++out.commands; out.sourceRows += rows;
    ++out.widths[width - 1]; ++out.rowBuckets[rows <= 16 ? 0 : rows < 256 ? 1 : 2];
    out.gateCalls += delta(now.gate_up_graph_calls, lastExpertCounters_.gate_up_graph_calls);
    out.gateRows += delta(now.gate_up_graph_rows, lastExpertCounters_.gate_up_graph_rows);
    out.downCalls += delta(now.down_graph_calls, lastExpertCounters_.down_graph_calls);
    out.downRows += delta(now.down_graph_rows, lastExpertCounters_.down_graph_rows);
    out.qmvGateCalls += delta(now.gathered_qmv_gate_up_graph_calls, lastExpertCounters_.gathered_qmv_gate_up_graph_calls);
    out.qmvGateRows += delta(now.gathered_qmv_gate_up_graph_rows, lastExpertCounters_.gathered_qmv_gate_up_graph_rows);
    out.qmvDownCalls += delta(now.gathered_qmv_down_graph_calls, lastExpertCounters_.gathered_qmv_down_graph_calls);
    out.qmvDownRows += delta(now.gathered_qmv_down_graph_rows, lastExpertCounters_.gathered_qmv_down_graph_rows);
    lastExpertCounters_ = now;
  }
  std::string expertPhaseJSON() const {
    std::ostringstream out;
    out << "{\"scope\":\"CPU Store graph-counter deltas after successful serialized target calls; not GPU timestamps or CPU readback; startup excluded\"";
    constexpr std::array<const char *, 3> names{"prefill", "autoregressive_decode", "target_verify"};
    for (uint32_t i = 0; i < names.size(); ++i) {
      const auto &p = expertPhases_[i];
      out << ',' << json::quote(names[i]) << ":{\"commands\":" << p.commands
          << ",\"source_rows\":" << p.sourceRows << ",\"gate_graph_calls\":" << p.gateCalls
          << ",\"gate_graph_rows\":" << p.gateRows << ",\"down_graph_calls\":" << p.downCalls
          << ",\"down_graph_rows\":" << p.downRows << ",\"qmv_gate_graph_calls\":" << p.qmvGateCalls
          << ",\"qmv_gate_graph_rows\":" << p.qmvGateRows << ",\"qmv_down_graph_calls\":" << p.qmvDownCalls
          << ",\"qmv_down_graph_rows\":" << p.qmvDownRows << ",\"by_width\":[";
      for (uint32_t j = 0; j < p.widths.size(); ++j) { if (j) out << ','; out << p.widths[j]; }
      out << "],\"commands_by_source_row_bucket_1to16_17to255_256plus\":[";
      for (uint32_t j = 0; j < p.rowBuckets.size(); ++j) { if (j) out << ','; out << p.rowBuckets[j]; }
      out << "]}";
    }
    out << '}'; return out.str();
  }
'''
