#include "Policy.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
using namespace splash::metal::private_partitioned_residency;
int main() {
  uint64_t checks = 0;
  const auto require = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("private partition check failed"); };
  const auto rejects = [&](auto&& fn) { bool bad=false; try { fn(); } catch(const std::invalid_argument&) { bad=true; } require(bad); };
  require(parseCap(nullptr)==0); require(parseCap("0")==0); require(parseCap("8589934592")==8589934592ULL);
  require(parseCap("18446744073709551615")==UINT64_MAX);
  for (const char* bad : {"", "00", "01", "+1", "-1", "1 ", " 1", "1x", "18446744073709551616"}) rejects([&]{(void)parseCap(bad);});
  require(plan({},0).groups.empty());
  require(plan({1,2,3},0).groups.size()==1);
  require(plan({1,2,3},0).totalBytes==6);
  require(plan({6,6,4,4},10).groups.size()==2);
  require(plan({6,6,4,4},10).groups[0].bytes==10);
  rejects([&]{(void)plan({0},0);});
  rejects([&]{(void)plan({11},10);});
  rejects([&]{(void)plan({UINT64_MAX,1},0);});
  require(plan(std::vector<uint64_t>(32,10),10).groups.size()==32);
  rejects([&]{(void)plan(std::vector<uint64_t>(33,10),10);});
  std::mt19937_64 random(0x981bc2);
  for (uint64_t trial=0; trial<2000; ++trial) {
    const uint64_t cap=64+(random()%256);
    std::vector<uint64_t> source;
    for (uint64_t i=0,n=1+(random()%31);i<n;++i) source.push_back(1+(random()%cap));
    const auto value=plan(source,cap);
    require(value.groups.size()<=32);
    require(value.totalBytes==std::accumulate(source.begin(),source.end(),uint64_t{0}));
    std::vector<unsigned> visits(source.size());
    for (const auto& group:value.groups) {
      uint64_t sum=0;
      require(!group.indices.empty());
      for (auto index:group.indices) { require(index<source.size()); ++visits[index]; sum+=source[index]; }
      require(sum==group.bytes); require(sum<=cap);
    }
    for(auto count:visits) require(count==1);
    const auto repeat=plan(source,cap);
    require(repeat.groups.size()==value.groups.size());
    for(size_t i=0;i<value.groups.size();++i) require(value.groups[i].indices==repeat.groups[i].indices);
  }
  std::cout << "{\"valid\":true,\"gpu_work\":false,\"checks\":"<<checks<<"}\n";
}
