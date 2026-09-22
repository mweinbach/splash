#pragma once
// Metadata-only exact immutable range index. The owning Store must keep the
// original source allocation/view set fixed for this index's lifetime.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace splash::flash::immutable_interval_index_sep22 {
template<class Address=uintptr_t> struct Span {
  Address begin=0;
  uint64_t length=0;
};
enum class Lookup : uint8_t {ProvenDisjoint,NeedsOriginal};

template<class Address=uintptr_t> class Index96 final {
  static_assert(std::is_unsigned_v<Address> && sizeof(Address)<=sizeof(uint64_t));
 public:
  using Interval=Span<Address>;
  explicit Index96(const std::array<Interval,96>&source) noexcept {
    struct End {Address begin,end;};
    std::array<End,96> ordered{};
    constexpr Address maximum=std::numeric_limits<Address>::max();
    for(size_t i=0;i<source.size();++i) {
      const auto &s=source[i];
      // Compare the complete uint64 length BEFORE narrowing or adding.
      if(!s.begin || !s.length || s.length>uint64_t(maximum-s.begin))return;
      ordered[i]={s.begin,Address(s.begin+Address(s.length))};
    }
    std::sort(ordered.begin(),ordered.end(),[](const End&a,const End&b) noexcept {
      return a.begin<b.begin;
    });
    Address end=0;
    for(size_t i=0;i<ordered.size();++i) {
      starts_[i]=ordered[i].begin;end=std::max(end,ordered[i].end);maximumEnds_[i]=end;
    }
    indexable_=true;
  }
  Index96(const Index96&)=default;
  Index96(Index96&&)=default;
  Index96&operator=(const Index96&)=delete;
  Index96&operator=(Index96&&)=delete;
  bool indexable()const noexcept{return indexable_;}
  Lookup lookup(Interval query)const noexcept {
    constexpr Address maximum=std::numeric_limits<Address>::max();
    if(!indexable_ || !query.begin || !query.length ||
        query.length>uint64_t(maximum-query.begin))return Lookup::NeedsOriginal;
    const Address end=Address(query.begin+Address(query.length));
    const auto stop=std::lower_bound(starts_.begin(),starts_.end(),end);
    if(stop!=starts_.begin() && maximumEnds_[size_t(stop-starts_.begin()-1)]>query.begin)
      return Lookup::NeedsOriginal;
    return Lookup::ProvenDisjoint;
  }
  // All rejected/odd/nonindexable cases execute the ORIGINAL callback. This
  // preserves exact error category/text and original first-failing interval.
  // The fast path can only accept a proved positive bounded disjoint query.
  template<class Original> bool validate(Interval query,Original&&original)const {
    if(lookup(query)==Lookup::ProvenDisjoint)return true;
    std::forward<Original>(original)();return false;
  }
 private:
  std::array<Address,96> starts_{},maximumEnds_{};
  bool indexable_=false;
};
} // namespace splash::flash::immutable_interval_index_sep22
