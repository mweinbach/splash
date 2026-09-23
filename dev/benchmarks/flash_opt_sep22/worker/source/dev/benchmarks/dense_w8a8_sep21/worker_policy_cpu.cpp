#include "worker_bridge.hpp"
#include <iostream>
#include <vector>
using namespace splash::flash::dense_w8a8_sep21;
int main(int argc,char **argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--freeze0" || mode == "--freeze1") {
      const bool want = mode == "--freeze1"; setenv(kFlag,want ? "1" : "0",1);
      if (requested() != want) throw std::runtime_error("selector freeze initial mismatch");
      setenv(kFlag,want ? "0" : "1",1); if (requested() != want) throw std::runtime_error("selector changed after freeze");
      std::cout << "dense W8A8 selector freeze passed\n"; return 0;
    }
    if (mode == "--invalid") {
      setenv(kFlag,"bad",1); bool rejected = false; try { (void)requested(); } catch (const std::invalid_argument &) { rejected = true; }
      if (!rejected) throw std::runtime_error("invalid selector accepted"); std::cout << "dense W8A8 invalid selector rejected before backend\n"; return 0;
    }
    uint64_t checks = 0; const auto require = [&](bool value) { if (!value) throw std::runtime_error("dense W8A8 policy failed"); ++checks; };
    require(!parse(nullptr) && !parse("0") && parse("1"));
    for (const char *v : {"","true"," 1","01","2","-1"}) { bool rejected = false; try { (void)parse(v); } catch (const std::invalid_argument &) { rejected = true; } require(rejected); }
    require(selectedPrefixes().size() == 84); require(kImmutableBufferCount == 168);
    const auto source = std::string(kCacheMarker) + std::string(64,'a');
    const auto left = std::string(selectionMarker(false)) + source, right = std::string(selectionMarker(true)) + source;
    require(numericalIdentity(std::string(64,'b'),left) == numericalIdentity(std::string(64,'b'),right));
    require(numericalIdentity(std::string(64,'b'),left) != numericalIdentity(std::string(64,'c'),left));
    const std::string hybridFMA0 = "1a00dd45649f14de4ad48bafa32ff67f1207aaf27b201de01bc6641a210134e2";
    const std::string hybridFMA1 = "c9fc21162d0b27d39e9291e1846af3f0a77f524e17798b7c50163af8d98c86f8";
    require(numericalIdentity(hybridFMA0,left) != numericalIdentity(hybridFMA1,left));
    require(numericalIdentity(hybridFMA0,left) == numericalIdentity(hybridFMA0,right));
    require(numericalIdentity(hybridFMA1,left) == numericalIdentity(hybridFMA1,right));
    require(Cache::plannedBytes() > 0 && Workspace::plannedBytes() > 0);
    for (const auto &prefix : selectedPrefixes()) {
      const auto g = geometry(prefix); require(bool(g));
      for (uint32_t rows : {1u,2u,4u,8u,16u,32u,128u,2047u,2048u,2049u,4096u,8192u})
        for (bool verify : {false,true}) for (bool main : {false,true}) for (bool selected : {false,true})
          require(mainProjectionEligible(prefix,rows,g.n,g.k,verify,main,selected) == (rows == 2048 && !verify && main && selected));
      require(!projectionPlan(prefix,2048,g.n,g.k,true)); require(!projectionPlan(prefix,2048,g.n,g.k,false,8));
      require(!projectionPlan(prefix,2048,g.n+64,g.k,false)); require(!projectionPlan(prefix,2048,g.n,g.k+32,false));
    }
    for (const char *bad : {"language_model.lm_head","language_model.model.layers.0.linear_attn.out_proj","language_model.model.layers.3.self_attn.o_proj","language_model.model.layers.3.self_attn.k_proj","language_model.model.layers.0.mlp.shared_expert.down_proj","language_model.model.layers.0.mlp.gate","language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up","language_model.mtp.layers.0.linear_attn.in_proj_qkv","language_model.model.layers.00.linear_attn.in_proj_qkv"}) require(!geometry(bad));
    std::cout << "{\"dense_w8a8_worker_cpu_policy\":\"passed\",\"checks\":" << checks << ",\"cache_planned_bytes\":" << Cache::plannedBytes() << ",\"workspace_planned_bytes\":" << Workspace::plannedBytes() << ",\"gpu_work\":false,\"payload_reads\":false}\n";
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
