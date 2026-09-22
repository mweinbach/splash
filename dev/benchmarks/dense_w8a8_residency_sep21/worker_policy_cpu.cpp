#include "worker_bridge.hpp"
#include <iostream>
using namespace splash::flash::dense_w8a8_residency_sep21;
int main(int argc,char **argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--freeze0" || mode == "--freeze1") {
      const bool value = mode == "--freeze1"; setenv(kFlag,value ? "1" : "0",1);
      if (requested() != value) throw std::runtime_error("residency freeze first value differs");
      setenv(kFlag,value ? "0" : "1",1); if (requested() != value) throw std::runtime_error("residency policy changed after freeze");
      std::cout << "residency selector frozen\n"; return 0;
    }
    if (mode == "--invalid") {
      setenv(kFlag,"bad",1); bool rejected = false; try { (void)requested(); } catch (const std::invalid_argument &) { rejected = true; }
      if (!rejected) throw std::runtime_error("invalid residency selector accepted"); std::cout << "invalid residency selector rejected before backend\n"; return 0;
    }
    uint64_t checks = 0; const auto require = [&](bool value) { if (!value) throw std::runtime_error("residency policy proof failed"); ++checks; };
    require(!parse(nullptr) && !parse("0") && parse("1"));
    for (const char *v : {"","true"," 1","01","-1","2"}) { bool rejected = false; try { (void)parse(v); } catch (const std::invalid_argument &) { rejected = true; } require(rejected); }
    uint64_t sourceBytes = 0;
    for (const auto &prefix : splash::flash::dense_w8a8_sep21::selectedPrefixes()) {
      const auto g = splash::flash::dense_w8a8_sep21::geometry(prefix); const auto bytes = uint64_t(g.k)*g.n*2;
      require(bytes%splash::flash::dense_w8a8_sep21::kAllocationAlignment == 0); sourceBytes += bytes;
    }
    require(sourceBytes == kBF16SourceBytes);
    for (bool cache : {false,true}) for (bool w8 : {false,true}) for (bool prune : {false,true}) {
      const auto p = plan(cache,w8,prune);
      require(p.omittedBF16Owners == (cache && w8 && prune ? 84u : 0u));
      require(p.omittedBF16Bytes == (cache && w8 && prune ? kBF16SourceBytes : 0));
      require(p.omittedI8Owners == (cache && !w8 && prune ? 168u : 0u));
      require(includeDerived(cache,w8,prune) == (cache && !(prune && !w8)));
      require(expectedHybridOwners(cache,w8,prune) == 748+p.retainedI8Owners-p.omittedBF16Owners);
      require(expectedHybridBytes(cache,w8,prune) == 202252746752ULL+p.retainedI8Bytes-p.omittedBF16Bytes);
    }
    require(expectedHybridOwners(true,true,true) == 832 && expectedHybridBytes(true,true,true) == 200368848896ULL);
    require(expectedHybridOwners(true,false,true) == 748 && expectedHybridBytes(true,false,true) == 202252746752ULL);
    require(expectedHybridOwners(true,true,false) == 916 && expectedHybridBytes(true,true,false) == 204143722496ULL);
    std::cout << "{\"persistent_residency_cpu_policy\":\"passed\",\"checks\":" << checks << ",\"omitted_source_count\":84,\"omitted_source_bytes\":" << kBF16SourceBytes << ",\"gpu_work\":false,\"payload_reads\":false}\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
