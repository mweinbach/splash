#pragma once
// Actual shipping preflight: the union is the original Pack preflight. Every
// callback is the unchanged original native guard, also compiled verbatim in
// the CPU legacy-vs-bundled decision harness. No values or tensors are read.
#include <cstdint>
#include <initializer_list>

namespace splash::flash::compact_r4_preflight_sep22 {
template<class Scratch,class Buffer,class CheckScratch,class RequireBytes,
         class Disjoint,class ImmutableDisjoint>
void validateComplete(const Scratch &s,Buffer input,Buffer ids,Buffer ranks,
    Buffer diagnostics,uint32_t rows,uint32_t selections,CheckScratch checkScratch,
    RequireBytes requireBytes,Disjoint disjoint,ImmutableDisjoint immutableDisjoint) {
  checkScratch(s,diagnostics,rows,selections);
  requireBytes(input,uint64_t{rows}*2560*2);
  requireBytes(ids,uint64_t{rows}*selections*8);
  requireBytes(ranks,uint64_t{512}*4);
  disjoint(input,ids);
  for(const auto &b:{s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,
      s.buckets.canonicalToPacked,s.buckets.packedInputs,s.buckets.jobOffsets,
      s.buckets.jobCount,s.buckets.tileJobs,s.packedActivated,s.scatteredDown,diagnostics}) {
    disjoint(input,b);disjoint(ids,b);immutableDisjoint(b);
  }
  immutableDisjoint(input);immutableDisjoint(ids);
}
} // namespace splash::flash::compact_r4_preflight_sep22
