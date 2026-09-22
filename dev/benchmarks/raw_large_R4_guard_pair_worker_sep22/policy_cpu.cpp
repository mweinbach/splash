// Pure host contracts with synthetic metadata/addresses; no backend or tensors.
#define RAW_LARGE_R4_POLICY_CPU_ONLY 1
#include "policy.hpp"
#include <iostream>
#include <map>
namespace r = splash::flash::raw_large_r4_guard_pair_sep22;
namespace {
uint64_t checks = 0;
void check(bool value) { if (!value) throw std::runtime_error("raw large R4 CPU contract failed"); ++checks; }
template <class F> void refused(F f) { bool bad = false; try { f(); } catch (const std::exception &) { bad = true; } check(bad); }
FlashAffineParams params(uint32_t bits, uint32_t group, uint32_t k, uint32_t n) {
  const uint64_t code = uint64_t(k) * bits / 8, parameter = uint64_t(k / group) * 2;
  return {4, 1, k, n, 1, bits, group, 0, code, uint64_t(n) * code, parameter, uint64_t(n) * parameter};
}
r::Observation source(std::string prefix, uint32_t bits, uint32_t group, uint32_t k, uint32_t n, uint32_t index, bool tile = false) {
  r::Observation o; o.prefix = std::move(prefix); o.projectionIdentity = 64 * uint64_t(index + 1); o.params = params(bits, group, k, n);
  const uintptr_t base = 0x1000000000ULL + uint64_t(index) * 0x100000000ULL;
  o.original[0] = {r::DType::U32, 2, {n, uint64_t(k) * bits / 32}, uint64_t(n) * o.params.weight_row_stride_bytes,
      {base, uint64_t(n) * o.params.weight_row_stride_bytes}};
  for (size_t i : {1u, 2u}) o.original[i] = {r::DType::BF16, 2, {n, k / group}, uint64_t(n) * o.params.parameter_row_stride_bytes,
      {base + uint64_t(i) * 0x20000000ULL, uint64_t(n) * o.params.parameter_row_stride_bytes}};
  o.cacheMember = true; o.selectedF32Tile = tile;
  o.cached = {r::DType::F32, 2, {n, k}, uint64_t(n) * k * 4, {base + 0x80000000ULL, uint64_t(n) * k * 4}};
  o.legacyQ4 = r::role(o.prefix) == r::Role::GDNQKV && bits == 4 && group == 64;
  return o;
}
std::vector<r::Observation> fixture() {
  std::vector<r::Observation> roles; uint32_t gdn = 0, qsa = 0;
  const auto add = [&](const std::string &prefix, uint32_t bits, uint32_t group, uint32_t k, uint32_t n, bool tile = false) {
    roles.push_back(source(prefix, bits, group, k, n, uint32_t(roles.size()), tile));
  };
  for (uint32_t i = 0; i < 48; ++i) {
    const auto prefix = "language_model.model.layers." + std::to_string(i) + '.';
    if (i % 4 != 3) {
      add(prefix + "linear_attn.in_proj_qkv", gdn < 26 ? 4 : (gdn < 30 ? 5 : 6), 64, 2560, 10240, gdn >= 30);
      add(prefix + "linear_attn.in_proj_z", gdn < 26 ? 5 : 6, gdn < 26 ? 128 : 64, 2560, 6144);
      add(prefix + "linear_attn.out_proj", 5, 128, 6144, 2560); ++gdn;
    } else {
      add(prefix + "self_attn.q_proj", qsa < 5 ? 4 : 5, 64, 2560, 12288, qsa >= 5);
      add(prefix + "self_attn.o_proj", qsa < 5 ? 4 : 5, 64, 6144, 2560, qsa >= 5); ++qsa;
    }
  }
  add("language_model.model.layers.1.ple.key_proj", 4, 64, 2560, 10240); return roles;
}
r::Descriptor descriptor(const r::Entry &e) {
  r::Descriptor d; d.groups = {e.params.output_size / 8, 4, 1}; d.threads = {64, 1, 1};
  d.buffers = 7; d.payloads = 1; d.paramsIndex = 7; d.paramsBytes = 64; d.paramsPresent = true; d.params = e.params;
  for (uint32_t i = 0; i < 7; ++i) d.indices[i] = i;
  return d;
}
std::array<r::Span, 7> spans(const r::Entry &e) {
  const r::Span input{0x100000000ULL, uint64_t(4) * e.params.input_size * 2};
  return {input, e.original[0], e.original[1], e.original[2], input,
      r::Span{0x200000000ULL, uint64_t(4) * e.params.output_size * 2}, r::Span{0x300000000ULL, 4}};
}
void contracts() {
  for (const char *value : {"", "2", "true", "01", " 1", "-1", "1\n"}) refused([&] { (void)r::parse(value); });
  check(r::parse(nullptr) == r::FlagState::Missing); check(r::parse("0") == r::FlagState::Disabled); check(r::parse("1") == r::FlagState::Enabled);
  for (const char *value : {static_cast<const char *>(nullptr), "0", "1"}) {
    const r::FrozenFlag flag(value); check(flag.check(value) == (value && std::string_view(value) == "1"));
    for (const char *changed : {static_cast<const char *>(nullptr), "0", "1"})
      if (r::parse(changed) != r::parse(value)) refused([&] { (void)flag.check(changed); });
  }
  std::map<std::string, std::string> env; for (auto name : r::kDependencies) env[name] = "1";
  const auto get = [&](const char *name) -> const char * { const auto p = env.find(name); return p == env.end() ? nullptr : p->second.c_str(); };
  r::dependencies(true, get); ++checks;
  for (const char *name : r::kDependencies) {
    for (const char *wrong : {"0", "true", "01"}) { env[name] = wrong; refused([&] { r::dependencies(true, get); }); }
    env.erase(name); refused([&] { r::dependencies(true, get); }); env[name] = "1";
  }
  env["SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21"] = "1"; refused([&] { r::dependencies(true, get); });
  env.clear(); r::dependencies(false, get); ++checks;
  for (uint32_t layer = 0; layer < 64; ++layer) {
    const auto p = "language_model.model.layers." + std::to_string(layer) + '.';
    check((r::role(p + "linear_attn.in_proj_qkv") == r::Role::GDNQKV) == (layer < 48 && layer % 4 != 3));
    check((r::role(p + "self_attn.q_proj") == r::Role::QSAQ) == (layer < 48 && layer % 4 == 3));
  }
  for (const char *p : {"language_model.model.layers.01.linear_attn.in_proj_qkv", "language_model.model.layers.0.ple.key_proj",
      "language_model.model.layers.1.ple.value_proj", "language_model.model.layers.1.linear_attn.in_proj_a", "language_model.lm_head", "mtp.fc_hidden"}) check(r::role(p) == r::Role::None);
  // Exercise both natural-shape stride minima and the exact unsigned upper
  // bound, independent of the immutable compact-source descriptor token.
  for (const auto &s : r::kSignatures) {
    auto p = params(s.bits, s.group, s.k, s.n); check(r::parametersValid(p));
    auto bad = p; --bad.weight_row_stride_bytes; check(!r::parametersValid(bad));
    bad = p; bad.parameter_row_stride_bytes -= 2; check(!r::parametersValid(bad));
    bad = p; ++bad.parameter_row_stride_bytes; check(!r::parametersValid(bad));
    const uint64_t code = uint64_t(s.k) * s.bits / 8, parameter = uint64_t(s.k / s.group) * 2;
    p.weight_row_stride_bytes = (UINT64_MAX - code) / (s.n - 1); check(r::parametersValid(p));
    bad = p; ++bad.weight_row_stride_bytes; check(!r::parametersValid(bad));
    p = params(s.bits, s.group, s.k, s.n);
    p.parameter_row_stride_bytes = ((UINT64_MAX - parameter) / (s.n - 1)) & ~uint64_t(1); check(r::parametersValid(p));
    bad = p; bad.parameter_row_stride_bytes += 2; check(!r::parametersValid(bad));
  }
  auto roles = fixture(); const auto proof = r::inventory(r::kModel, r::kManifest, roles);
  check(proof.potential == 133 && proof.raw == 113 && proof.legacy == 26 && proof.entries.size() == 87 && proof.f32 == 20);
  for (size_t i = 0; i < 7; ++i) check(proof.shapes[i] == r::kSignatures[i].expected && proof.newShapes[i] == r::kSignatures[i].expectedNew);
  for (const auto &o : roles) for (uint32_t rows = 0; rows <= 17; ++rows) for (bool verify : {false, true}) for (bool singleton : {false, true}) {
    const bool expected = rows == 4 && verify && singleton && !o.selectedF32Tile && !o.legacyQ4;
    check(bool(r::select(proof, o.projectionIdentity, rows, verify, singleton)) == expected);
  }
  check(!r::select(proof, UINTPTR_MAX, 4, true, true));
  refused([&] { (void)r::inventory("wrong", r::kManifest, roles); }); refused([&] { (void)r::inventory(r::kModel, "wrong", roles); });
  for (uint32_t change = 0; change < 15; ++change) {
    auto bad = roles;
    switch (change) {
    case 0: bad.pop_back(); break; case 1: bad[1] = bad[0]; break;
    case 2: bad[1].projectionIdentity = bad[0].projectionIdentity; break;
    case 3: bad[0].cacheMember = false; break; case 4: bad[0].selectedF32Tile = true; break;
    case 5: bad[0].params.experts = 2; break; case 6: bad[0].params.flags = 1; break;
    case 7: bad[0].params.weight_row_stride_bytes += 4; break;
    case 8: bad[0].original[0].dtype = r::DType::F32; break;
    case 9: --bad[0].original[1].logicalBytes; break;
    case 10: bad[0].original[2].view.address += 2; break;
    case 11: --bad[0].cached.shape[1]; break;
    case 12: bad[0].legacyQ4 = false; break;
    case 13: bad[0].params.output_size = 10239; break;
    case 14: bad[0].prefix = "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up"; break;
    }
    refused([&] { (void)r::inventory(r::kModel, r::kManifest, bad); });
  }
  std::array<bool, 7> checked{};
  for (const auto &e : proof.entries) {
    if (checked[e.shape]) continue; checked[e.shape] = true;
    const auto pipeline = r::originalPipeline(e.params); auto d = descriptor(e); d.pipeline = pipeline; const auto v = spans(e);
    r::descriptor(d, e, v); ++checks;
    check(r::candidatePipeline(e.params).starts_with("raw_large_R4_guard_pair_sep22_timed_"));
    for (uint32_t change = 0; change < 22; ++change) {
      auto bad = d; auto views = v;
      switch (change) {
      case 0: bad.pipeline = "wrong"; break; case 1: bad.groups[1] = 2; break;
      case 2: bad.groups[2] = 2; break; case 3: bad.threads[0] = 32; break;
      case 4: bad.buffers = 6; break; case 5: bad.payloads = 2; break;
      case 6: bad.paramsIndex = 8; break; case 7: bad.paramsBytes = 63; break;
      case 8: bad.paramsPresent = false; break; case 9: bad.indices[3] = 4; break;
      case 10: bad.params.flags = 1; break; case 11: bad.params.rows = 5; break;
      case 12: views[5] = views[1]; break; case 13: views[6] = views[5]; break;
      case 14: views[4].address += 8; break; case 15: views[2].address += 16384; break;
      case 16: --views[0].bytes; break; case 17: --views[5].bytes; break;
      case 18: views[6].address += 2; break;
      case 19: views[5] = {UINTPTR_MAX - 1, 8}; break;
      case 20: bad.params.weight_row_stride_bytes = UINT64_MAX; break;
      case 21: bad.params.parameter_row_stride_bytes = UINT64_MAX - 1; break;
      }
      refused([&] { r::descriptor(bad, e, views); });
    }
  }
  for (bool covered : checked) check(covered);
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--dependency") { r::validateDependencies(); }
    else if (argc == 2 && std::string_view(argv[1]) == "--lifetime") {
      const char *before = std::getenv(r::kFlag); const bool first = r::requested();
      setenv(r::kFlag, first ? "0" : "1", 1); refused([] { (void)r::requested(); });
      (void)before;
    } else { check(argc == 1); contracts(); }
    std::cout << "{\"pass\":true,\"checks\":" << checks << ",\"GPU_work\":false,\"payload_reads\":0,\"inventory_scope\":\"synthetic metadata and fake addresses; no model or graph execution\"}\n";
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
