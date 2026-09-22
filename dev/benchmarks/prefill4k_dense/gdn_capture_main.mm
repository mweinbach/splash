#include "gdn_capture.hpp"
#define main original_attribution_main
#include "../prefill4k_attribution.mm"
#undef main

namespace private_gdn_capture {
void write() {
  const char *raw=std::getenv("PREFILL4K_GDN_CAPTURE");if (!raw || !*raw) return;
  require(snapshot.has_value(),"actual GDN snapshot absent");
  const auto directory=std::filesystem::absolute(raw);
  require(!std::filesystem::exists(directory),"GDN capture directory must be fresh");
  std::filesystem::create_directories(directory);const auto &s=*snapshot;
  std::ofstream manifest(directory/"manifest.json");require(bool(manifest),"cannot write GDN manifest");
  manifest << "{\"schema\":\"splash-actual-gdn-layer-v1\",\"layer\":0,\"rows\":"<<s.rows
      <<",\"lanes\":1,\"norm_epsilon\":"<<std::setprecision(12)<<s.params.norm_epsilon<<",\"files\":{";
  bool first=true;
  for (const auto &[name,buffer] : std::array<std::pair<const char *,splash::metal::MetalBuffer>,9>{{
      {"mixed",s.mixed},{"decay",s.decay},{"beta",s.beta},{"z",s.z},{"norm",s.norm},
      {"initial_state",s.initialState},{"expected_state",s.expectedState},
      {"expected_recurrence",s.expectedRecurrence},{"expected_output",s.expectedOutput}}}) {
    const auto path=directory/(std::string(name)+".bin");
    std::ofstream out(path,std::ios::binary);require(bool(out),"cannot write GDN snapshot");
    out.write(static_cast<const char *>(buffer.contents()),std::streamsize(buffer.sizeBytes()));
    require(bool(out),"GDN snapshot write incomplete");if (!first) manifest<<',';first=false;
    manifest<<splash::json::quote(name)<<":{\"path\":"<<splash::json::quote(path.string())
        <<",\"bytes\":"<<buffer.sizeBytes()<<",\"sha256\":"
        <<splash::json::quote(digest(buffer.contents(),buffer.sizeBytes()))<<'}';
  }
  manifest<<"}}\n";require(bool(manifest),"GDN manifest incomplete");
  std::cout<<"actual GDN layer-zero snapshot="<<directory<<'\n';
}
}
int main(int argc,char **argv) {
  const int result=original_attribution_main(argc,argv);if (result) return result;
  try {private_gdn_capture::write();return 0;}
  catch (const std::exception &e) {std::cerr<<"GDN snapshot write failed: "<<e.what()<<'\n';return 1;}
}
