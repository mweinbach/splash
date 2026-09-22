"""Two private TU modifications only; no public header/Core/shader/math change."""
PRIVATE="dev/benchmarks/immutable_interval_worker_sep22"
def once(text,old,new):
    if text.count(old)!=1:raise ValueError("immutable guard anchor drift:"+old[:90])
    return text.replace(old,new)
def transform(path,text):
    if path not in ["runtime/flash/FlashInt8ExpertStore.mm","runtime/flash/FlashWorker.mm"]:return text
    text='#include "'+PRIVATE+'/policy.hpp"\n'+text
    if path.endswith("FlashInt8ExpertStore.mm"):
        text='#include "'+PRIVATE+'/index.hpp"\n#include <optional>\n'+text
        text=once(text,"  std::array<Layer, 48> layers;",
            "  std::array<Layer, 48> layers;\n  const bool immutableIndexEnabled = immutable_interval_worker_sep22::requested();\n  std::optional<immutable_interval_index_sep22::Index96<>> immutableIndex;")
        text=once(text,"    allocated = after - before;","""    allocated = after - before;
    if (immutableIndexEnabled) {
      std::array<immutable_interval_index_sep22::Span<>,96> spans{};
      for (uint32_t index=0; index<48; ++index) {
        const auto &layer=layers[index];
        spans[index*2]={reinterpret_cast<uintptr_t>(layer.base.contents()),layer.base.sizeBytes()};
        spans[index*2+1]={reinterpret_cast<uintptr_t>(layer.ranks.contents()),layer.ranks.sizeBytes()};
      }
      immutableIndex.emplace(spans);
      if (immutableIndex->indexable())
        immutable_interval_worker_sep22::indexableTables.fetch_add(1,std::memory_order_relaxed);
      immutable_interval_worker_sep22::finalizedSpans.fetch_add(96,std::memory_order_relaxed);
      immutable_interval_worker_sep22::finalizedTables.fetch_add(1,std::memory_order_relaxed);
    }""")
        text=once(text,"""  void immutableDisjoint(const metal::MetalBuffer &output) const {
    for (const auto &layer : layers) {
      disjoint(output, layer.base); disjoint(output, layer.ranks);
    }
  }""","""  void immutableDisjoint(const metal::MetalBuffer &output) const {
    if (immutable_interval_worker_sep22::requested()!=immutableIndexEnabled)
      throw std::logic_error("immutable interval index Store frozen policy changed");
    if (immutableIndexEnabled && immutableIndex &&
        immutableIndex->lookup({reinterpret_cast<uintptr_t>(output.contents()),output.sizeBytes()}) ==
            immutable_interval_index_sep22::Lookup::ProvenDisjoint) {
      immutable_interval_worker_sep22::indexedAccepts.fetch_add(1,std::memory_order_relaxed);
      return;
    }
    // Count immediately BEFORE the original callback, including throws.
    immutable_interval_worker_sep22::originalCallbacks.fetch_add(1,std::memory_order_relaxed);
    for (const auto &layer : layers) {
      disjoint(output, layer.base); disjoint(output, layer.ranks);
    }
  }""")
    else:
        text=once(text,"      (void)adaptive_expert_tail_sg2k128_sep21::requested(); // Freeze tail control before paths/metadata/backend.",
            "      immutable_interval_worker_sep22::startup(); // Freeze strict index before paths/model/backend.\n      (void)adaptive_expert_tail_sg2k128_sep21::requested(); // Freeze tail control before paths/metadata/backend.")
        text=once(text,'      << R"(,"gdn_verification_storage":{"lazy_enabled":)"',"""      << R"(,"immutable_interval_index_guard":{"schema":"CPU-immutable96-exact-positive-index-original-fallback-v1","requested":)"
      << (immutable_interval_worker_sep22::requested()?"true":"false")
      << R"(,"source_policy_sha256":)" << json::quote(immutable_interval_worker_sep22::sourcePolicySHA)
      << R"(,"scope":)" << json::quote(immutable_interval_worker_sep22::scope)
      << R"(,"GPU_allocation_bytes_added":0,"FP_graph_public_guards_unchanged":true,"counter_scope":"process-cumulative CPU metadata lookups; not GPU completion","finalized_tables":)"
      << immutable_interval_worker_sep22::finalizedTables.load(std::memory_order_relaxed)
      << R"(,"finalized_spans":)" << immutable_interval_worker_sep22::finalizedSpans.load(std::memory_order_relaxed)
      << R"(,"indexable_tables":)" << immutable_interval_worker_sep22::indexableTables.load(std::memory_order_relaxed)
      << R"(,"indexed_accepts":)" << immutable_interval_worker_sep22::indexedAccepts.load(std::memory_order_relaxed)
      << R"(,"original_callbacks":)" << immutable_interval_worker_sep22::originalCallbacks.load(std::memory_order_relaxed) << '}'
      << R"(,"gdn_verification_storage":{"lazy_enabled":)" """.rstrip())
    return text
