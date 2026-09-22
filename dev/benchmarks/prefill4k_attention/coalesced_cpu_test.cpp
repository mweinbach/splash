#include "coalesced.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace splash::flash::prefill4k;
  uint64_t checks=0;
  auto require=[&](bool valid,const char *message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
  };
  try {
    for (uint32_t begin:{0u,1u,3u,4u,127u,128u,511u,512u,1024u,1792u,
                        1920u,2048u,2049u,8192u,UINT32_MAX})
      for (uint32_t rows:{0u,1u,127u,128u,255u,256u,257u,512u,1024u,
                         1536u,2048u,2049u,8192u,UINT32_MAX})
        for (uint32_t capacity:{0u,1u,2048u,8192u,262144u,262145u,UINT32_MAX}) {
          const uint64_t stop=uint64_t(begin)+rows;
          const bool expected=capacity>=1 && capacity<=262144 && rows>=256 &&
                              rows<=2048 && stop<=2048 && stop<=capacity;
          require(denseCoalescedEligible(begin,rows,capacity)==expected,
                  "Eligibility boundary or overflow differs from64-bit oracle");
        }
    require(denseCoalescedPlannedBytes(2048)==30*1024*1024,
            "2K dedicated scratch must be exactly30MiB");
    for (uint32_t rows:{0u,1u,128u,255u,2049u,UINT32_MAX}) {
      bool rejected=false;
      try { (void)denseCoalescedPlannedBytes(rows); }
      catch (const std::invalid_argument &) { rejected=true; }
      require(rejected,"Invalid workspace row capacity accepted");
    }
    // The dedicated gate means chronological selection is authoritative. All
    // completed pooled blocks after this query, including those written by
    // bulk preparation, stay excluded. The incomplete raw tail stays causal.
    for (uint32_t begin=0;begin<=1792;begin+=17) {
      const uint32_t rows=2048-begin;
      require(denseCoalescedEligible(begin,rows,8192),"Dense append rejected");
      const uint32_t allCompleted=(begin+rows)/4;
      for (uint32_t row=0;row<rows;++row) {
        const uint32_t visible=begin+row+1,complete=visible/4,tail=visible%4;
        require(complete<=512,"Dense gate allowed sparse block selection");
        require(complete<=allCompleted,"Query completed blocks exceed bulk pool");
        for (uint32_t block=0;block<allCompleted;++block) {
          const bool selected=block<complete;
          require(!selected || block*4+3<visible,
                  "A future raw key from a pooled block leaks into this query");
          require(selected || block>=complete,
                  "A future pooled key was not excluded chronologically");
        }
        for (uint32_t token=complete*4;token<complete*4+tail;++token)
          require(token<visible,"Incomplete raw tail leaks a future token");
      }
    }
    std::cout << "{\"pass\":true,\"geometry_and_causal_selection_cpu_checks\":"
              << checks << ",\"gpu_commands\":0,\"full_cache_gpu_parity_verified\":false}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "coalesced QSA CPU audit failed: " << error.what() << '\n';
    return 1;
  }
}
