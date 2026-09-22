#include "flash/FlashDecodeBF16DensePolicy.hpp"
#include <iostream>

int main() {
  using namespace splash::flash::bf16_decode;
  const auto check = [](bool value) { if (!value) throw std::runtime_error("BF16 selector CPU check failed"); };
  unsetenv("SPLASH_FLASH_DECODE_BF16_DENSE"); unsetenv("SPLASH_FLASH_DECODE_BF16_DENSE_SCOPE"); unsetenv("SPLASH_FLASH_DECODE_BF16_DENSE_ROLES");
  check(!requested().enabled && workspaceBytes(requested()) == 0);
  setenv("SPLASH_FLASH_DECODE_BF16_DENSE", "1", 1);
  setenv("SPLASH_FLASH_DENSE_CACHE", "1", 1); setenv("SPLASH_FLASH_FLOAT_DENSE_CACHE", "1", 1); setenv("SPLASH_FLASH_DENSE_SMALL_ROWS", "0", 1);
  const auto policy = requested(); validateDependencies(policy); validateFrozen(policy); check(workspaceBytes(policy) == 1048576);
  for (uint32_t layer = 0; layer < 48; ++layer) for (uint32_t rows : {1u,2u,4u,8u,16u}) {
    const std::string prefix = "language_model.model.layers." + std::to_string(layer);
    check(geometry(policy, prefix + ".linear_attn.in_proj_qkv", rows, 10240, 2560));
    check(geometry(policy, prefix + ".self_attn.o_proj", rows, 2560, 6144));
    check(geometry(policy, prefix + ".mlp.shared_expert.down_proj", rows, 2560, 640));
    check(geometry(policy, prefix + ".ple.value_proj", rows, 2560, 2560));
    check(geometry(policy, prefix + ".attn_hyper_connection.input_mix_weight_up", rows, 10240, 320) == (rows == 1 || rows == 2));
    check(!geometry(policy, prefix + ".mlp.switch_mlp.down_proj", rows, 2560, 640));
  }
  for (uint32_t rows : {0u,3u,5u,15u,17u,2048u}) check(!geometry(policy,"language_model.model.layers.0.self_attn.o_proj", rows,2560,6144));
  for (const char *prefix : {"language_model.lm_head","mtp.layers.0.self_attn.o_proj","language_model.model.layers.48.self_attn.o_proj",
                           "language_model.model.layers.-1.self_attn.o_proj","language_model.model.layers.x.self_attn.o_proj"})
    check(!geometry(policy,prefix,4,2560,6144));
  check(!geometry(policy,"language_model.model.layers.3.self_attn.o_proj",4,2561,6144));
  check(geometry(policy,"language_model.model.hyper_connection_mixer.input_mix_weight_up",1,10240,320));
  check(!geometry(policy,"language_model.model.hyper_connection_mixer.input_mix_weight_up",4,10240,320));
  setenv("SPLASH_FLASH_DECODE_BF16_DENSE_ROLES","attention",1); const auto attention = requested();
  check(geometry(attention,"language_model.model.layers.3.self_attn.o_proj",4,2560,6144));
  check(!geometry(attention,"language_model.model.layers.3.mlp.shared_expert.down_proj",4,2560,640));
  check(identity(attention) != identity(policy));
  bool frozenRejected = false; try { validateFrozen(policy); } catch (const std::logic_error &) { frozenRejected = true; } check(frozenRejected);
  setenv("SPLASH_FLASH_DECODE_BF16_DENSE_SCOPE","existing_f32_route",1); check(requested().scope == Scope::ExistingF32Route);
  setenv("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE","0",1);
  const auto notSelective = requested(); setenv("SPLASH_FLASH_FLOAT_DENSE_SELECTIVE","1",1); const auto selective = requested();
  check(identity(notSelective) != identity(selective));
  setenv("SPLASH_FLASH_DECODE_BF16_DENSE_SCOPE","all_cached",1); setenv("SPLASH_FLASH_DECODE_BF16_DENSE_ROLES","hc_up_small",1);
  check(geometry(requested(),"language_model.model.layers.2.mlp_hyper_connection.input_mix_weight_up",2,10240,320));
  check(!geometry(requested(),"language_model.model.layers.2.mlp_hyper_connection.input_mix_weight_up",4,10240,320));
  check(!geometry(requested(),"language_model.model.layers.2.self_attn.o_proj",2,2560,6144));
  setenv("SPLASH_FLASH_FLOAT_DENSE_CACHE","0",1);
  bool depsRejected = false; try { validateDependencies(requested()); } catch (const std::invalid_argument &) { depsRejected = true; } check(depsRejected);
  for (const char *invalid : {"","2","01"," 1","1 ","-1"}) {
    setenv("SPLASH_FLASH_DECODE_BF16_DENSE",invalid,1);
    bool rejected = false; try { (void)requested(); } catch (const std::invalid_argument &) { rejected = true; } check(rejected);
  }
  std::cout << "{\"bf16_selector_cpu_pass\":true,\"gpu_work\":false,\"payload_bytes_read\":0}\n";
}
