#include "flash/FlashPrefillDenseTiles.hpp"
#include "metal/abi/FlashDenseCache.h"
#include <array>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace splash::flash;
static_assert(sizeof(FlashDenseCacheParams) == 32);
static_assert(offsetof(FlashDenseCacheParams,reserved) == 28);
namespace {
void require(bool value,const char *message) { if (!value) throw std::runtime_error(message); }
}
int main() {
  try {
    uint64_t checks = 0;
    require(!parseFlashPrefillDenseTilesFlag(nullptr),"unset flag enabled candidate"); ++checks;
    require(!parseFlashPrefillDenseTilesFlag("0"),"off flag enabled candidate"); ++checks;
    require(parseFlashPrefillDenseTilesFlag("1"),"on flag disabled candidate"); ++checks;
    for (const char *bad : {"", "true", "false", "-1", "2", "01", "1 "}) {
      bool rejected = false;
      try { (void)parseFlashPrefillDenseTilesFlag(bad); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected,"invalid flag accepted"); ++checks;
    }
    struct Expected { uint32_t inputs,outputs,groups,mode; };
    const std::array<Expected,7> selections{{
        {10240,320,8,0}, {2560,10240,4,3}, {2560,6144,4,3}, {6144,2560,4,3},
        {2560,12288,4,4}, {2560,512,8,2}, {2560,640,8,4}}};
    for (const auto expected : selections) {
      const auto plan = flashPrefillDenseTilePolicy(2048,expected.outputs,expected.inputs);
      require(plan && plan.tileRows == 128 && plan.tileOutputs == 64 &&
          plan.simdGroups == expected.groups && static_cast<uint32_t>(plan.traversal) == expected.mode,
          "measured shape selected the wrong tile or traversal"); ++checks;
      for (uint32_t rows : {0u,1u,16u,128u,512u,1024u,2047u,2049u,4096u,8192u}) {
        require(!flashPrefillDenseTilePolicy(rows,expected.outputs,expected.inputs),
                "policy extrapolated to an unqualified row count"); ++checks;
      }
      for (int delta : {-1,1}) {
        require(!flashPrefillDenseTilePolicy(2048,uint32_t(int(expected.outputs)+delta),expected.inputs),
                "policy extrapolated to unqualified columns"); ++checks;
        require(!flashPrefillDenseTilePolicy(2048,expected.outputs,uint32_t(int(expected.inputs)+delta)),
                "policy extrapolated to unqualified K"); ++checks;
      }
    }
    require(!flashPrefillDenseTilePolicy(2048,10240,320),"HCup selected a slower candidate"); ++checks;
    require(!flashPrefillDenseTilePolicy(2048,2560,640),"shared down selected a slower candidate"); ++checks;
    require(!flashPrefillDenseTilePolicy(2048,2560,2560),"PLEvalue selected insignificant candidate"); ++checks;
    const auto equal = [](FlashPrefillDenseTilePlan a,FlashPrefillDenseTilePlan b) {
      return a.tileRows == b.tileRows && a.tileOutputs == b.tileOutputs &&
          a.simdGroups == b.simdGroups && a.traversal == b.traversal;
    };
    require(equal(flashPrefillDenseTilePolicy("language_model.model.hyper_connection_mixer.input_mix_weight_down",
        2048,320,10240),flashPrefillDenseTilePolicy(2048,320,10240)),"main HCdown route changed"); ++checks;
    for (uint32_t layer = 0; layer < 48; ++layer) {
      const std::string prefix = "language_model.model.layers."+std::to_string(layer)+".";
      struct Role { const char *name;uint32_t outputs,inputs; };
      std::vector<Role> roles{{"attn_hyper_connection.input_mix_weight_down",320,10240},
          {"mlp_hyper_connection.input_mix_weight_down",320,10240}};
      if (layer%4 != 3) {
        roles.push_back({"linear_attn.in_proj_qkv",10240,2560});
        roles.push_back({"linear_attn.in_proj_z",6144,2560});
        roles.push_back({"linear_attn.out_proj",2560,6144});
      } else {
        roles.push_back({"self_attn.q_proj",12288,2560});
        roles.push_back({"self_attn.k_proj",512,2560});
        roles.push_back({"self_attn.v_proj",512,2560});
        roles.push_back({"self_attn.indexer.index_qk_proj",640,2560});
        roles.push_back({"self_attn.o_proj",2560,6144});
      }
      if (layer == 1) roles.push_back({"ple.key_proj",10240,2560});
      for (const auto role : roles) {
        const auto plan = flashPrefillDenseTilePolicy(prefix+role.name,2048,role.outputs,role.inputs);
        require(plan && equal(plan,flashPrefillDenseTilePolicy(2048,role.outputs,role.inputs)),
                "qualified role no longer selects its proven shape plan"); ++checks;
        require(!flashPrefillDenseTilePolicy(prefix+role.name,2048,role.outputs,role.inputs+32),
                "qualified role accepted unqualified dimensions"); ++checks;
      }
      require(!flashPrefillDenseTilePolicy(prefix+"mlp.shared_expert.gate_proj",2048,640,2560) &&
          !flashPrefillDenseTilePolicy(prefix+"mlp.shared_expert.up_proj",2048,640,2560),
          "shared gate/up unexpectedly selected tiled route"); ++checks;
    }
    for (const char *prefix : {"mtp.layers.0.self_attn.q_proj",
        "language_model.model.layers.48.self_attn.q_proj",
        "language_model.model.layers.03.self_attn.q_proj",
        "language_model.model.layers.-1.self_attn.q_proj",
        "language_model.model.layers.x.self_attn.q_proj",
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.3.self_attn.unknown",
        "language_model.model.layers.3.self_attn.q_proj.extra",
        "other.layers.3.self_attn.q_proj"}) {
      require(!flashPrefillDenseTilePolicy(prefix,2048,12288,2560),"unknown role/prefix selected tiled route"); ++checks;
    }
    require(!flashPrefillDenseTilePolicy("language_model.model.layers.2.ple.key_proj",2048,10240,2560),
            "unqualified PLE layer selected tiled route"); ++checks;
    for (uint32_t mode = 0; mode <= 4; ++mode) {
      for (uint32_t rows : {1u,3u,16u,31u}) for (uint32_t columns : {1u,5u,8u,10u,40u,80u,192u}) {
        const auto grid = flashDenseTraversalGrid(rows,columns,static_cast<FlashDenseTraversal>(mode));
        std::vector<uint32_t> visits(uint64_t(rows)*columns);
        for (uint32_t y = 0; y < grid.y; ++y) for (uint32_t x = 0; x < grid.x; ++x) {
          uint32_t r = y,c = x;
          if (mode == 1) { r = x;c = y; }
          else if (mode >= 2) {
            const uint32_t log = mode-1;
            r = (y<<log)+(x&((1u<<log)-1));c = x>>log;
          }
          if (r < rows && c < columns) ++visits[uint64_t(r)*columns+c];
        }
        for (uint32_t v : visits) require(v == 1,"prefill traversal lost/duplicated a tile"); ++checks;
      }
    }
    std::cout << "{\"prefill_dense_policy\":\"passed\",\"checks\":" << checks << "}\n"; return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
