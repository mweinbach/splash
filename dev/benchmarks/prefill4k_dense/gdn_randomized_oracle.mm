// Same compiled math/fixture helpers; this driver changes only timing order.
#define main original_gdn_pair_entrypoint
#include "gdn_actual_oracle.mm"
#undef main
#include <random>

namespace {
std::vector<std::byte> snapshotBytes(const Guarded &b) {
  const auto *begin=static_cast<const std::byte *>(b.view.contents());return {begin,begin+b.bytes};
}
struct Expected {
  std::vector<std::byte> state,recurrence,output;
  explicit Expected(const Fixture &f):state(snapshotBytes(f.state)),recurrence(snapshotBytes(f.recurrence)),output(snapshotBytes(f.output)){}
  void check(const Fixture &f,bool finalOutput) const {
    require(std::memcmp(f.state.view.contents(),state.data(),state.size())==0,"randomized GDN F32 state differs");
    require(std::memcmp(f.recurrence.view.contents(),recurrence.data(),recurrence.size())==0,"randomized GDN BF16 recurrence differs");
    if (finalOutput) require(std::memcmp(f.output.view.contents(),output.data(),output.size())==0,"randomized GDN final BF16 output differs");
    f.verify();
  }
};
constexpr std::array<Variant,5> allKinds{{{"flash_gdn_staged_v16_t16",512},
    {"private_gdn_delayq_v16_t16",512},{"private_gdn_delayq_v16_t32",512},
    {"private_gdn_ilp_v16_t16_s8",256},{"private_gdn_ilp_v16_t32_s8",256}}};
}

int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if (argc==2 && std::string(argv[1])=="--cpu-self-test") {cpuTest();return 0;}
      require(argc==4 || (argc==5 && std::string(argv[4])=="--recurrence-only"),
          "usage: randomized-oracle METALLIB MANIFEST REPORT [--recurrence-only]");
      const bool complete=argc==4;
      const auto source=manifest(argv[2]);const uint64_t rows=source.rows;
      std::map<std::string,std::vector<std::byte>> files;
      for (const auto &[name,bytes] : std::array<std::pair<const char *,uint64_t>,9>{{
          {"mixed",rows*10240*2},{"decay",rows*48*4},{"beta",rows*48*2},{"z",rows*6144*2},{"norm",128*2},
          {"initial_state",48*128*128*4},{"expected_state",48*128*128*4},
          {"expected_recurrence",rows*6144*2},{"expected_output",rows*6144*2}}}) files[name]=read(source,name,bytes);
      require(std::any_of(files["expected_state"].begin(),files["expected_state"].end(),[](std::byte b){return b!=std::byte{};}),
          "nonzero carried state required");
      MetalBackend backend(argv[1]);
      // Every route uses these identical native buffers and addresses.
      Fixture fixture(backend,source,files);
      std::vector<CommandGraph> commands;
      for (const auto kind : allKinds) commands.push_back(graph(fixture,source,kind.name,kind.threads,complete));
      auto fullBaseline=graph(fixture,source,allKinds[0].name,allKinds[0].threads,true);
      constexpr uint64_t seed=0x47534e3252474241ULL;
      std::mt19937_64 random(seed);
      std::ostringstream report;
      report<<std::setprecision(12)<<"{\"schema\":\"splash-actual-gdn-randomized-v1\",\"pass\":true,"
          <<"\"timing_size_bytes\":"<<sizeof(CommandTiming)<<",\"same_input_state_output_addresses\":true,"
          <<"\"random_seed\":"<<seed<<",\"warmup_cycles_each_route\":6,\"matched_cycles\":9,"
          <<"\"scope\":"<<splash::json::quote(complete?"recurrence plus identical output normalization/gating":"recurrence only")
          <<",\"error_limit\":\"zero bytes for complete state/recurrence/output and all canaries\",\"cases\":[";
      bool first=true;
      for (bool carried : {false,true}) {
        const auto &initial=files.at(carried?"expected_state":"initial_state");
        fixture.reset(initial);valid(backend.submitCommand(fullBaseline.dispatches()));fixture.verify();
        const Expected expected(fixture);
        if (!carried) require(expected.state==files.at("expected_state") && expected.recurrence==files.at("expected_recurrence") &&
            expected.output==files.at("expected_output"),"randomized baseline does not match actual model capture");
        std::array<std::vector<double>,5> gpu,wall;
        std::vector<std::array<uint32_t,5>> orders;
        for (uint32_t cycle=0;cycle<15;++cycle) {
          std::array<uint32_t,5> order{0,1,2,3,4};std::shuffle(order.begin(),order.end(),random);
          if (cycle>=6) orders.push_back(order);
          for (uint32_t kind : order) {
            fixture.reset(initial);
            const auto timing=backend.submitCommand(commands[kind].dispatches());valid(timing);expected.check(fixture,complete);
            if (cycle>=6) {gpu[kind].push_back(timing.gpuSeconds*1000);wall[kind].push_back(timing.wallSeconds*1000);}
          }
        }
        fixture.immutable();
        if (!first) report<<',';first=false;
        report<<"{\"carried_after_actual_2k\":"<<(carried?"true":"false")<<",\"all_bytes_exact\":true,\"orders\":[";
        for (size_t i=0;i<orders.size();++i) {
          if (i) report<<',';report<<'[';
          for (uint32_t j=0;j<5;++j) {if (j) report<<',';report<<orders[i][j];}report<<']';
        }
        report<<"],\"variants\":[";
        for (uint32_t kind=0;kind<5;++kind) {
          std::vector<double> ratios;uint32_t positive=0;
          for (uint32_t pair=0;pair<9;++pair) {const double r=gpu[0][pair]/gpu[kind][pair];ratios.push_back(r);positive+=r>1;}
          if (kind) report<<',';
          report<<"{\"name\":"<<splash::json::quote(allKinds[kind].name)<<",\"threads\":"<<allKinds[kind].threads
              <<",\"median_gpu_ms\":"<<median(gpu[kind])<<",\"median_wall_ms\":"<<median(wall[kind])
              <<",\"median_paired_speedup\":"<<median(ratios)<<",\"positive_pairs\":"<<positive<<",\"gpu_ms\":[";
          for (uint32_t i=0;i<9;++i) {if (i) report<<',';report<<gpu[kind][i];}
          report<<"],\"paired_speedups\":[";
          for (uint32_t i=0;i<9;++i) {if (i) report<<',';report<<ratios[i];}report<<"]}";
          std::cerr<<allKinds[kind].name<<" carried="<<carried<<" medianGPUms="<<median(gpu[kind])
              <<" paired="<<median(ratios)<<" positive="<<positive<<"/9\n";
        }
        report<<"]}";
      }
      report<<"]}\n";std::ofstream out(argv[3]);require(bool(out),"cannot write randomized GDN report");out<<report.str();return 0;
    } catch (const std::exception &e) {std::cerr<<"randomized actual GDN oracle failed: "<<e.what()<<'\n';return 1;}
  }
}
