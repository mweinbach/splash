#include "capture.hpp"
#include <tuple>
#define main prefill4k_attribution_main
#include "capture_attribution.mm"
#undef main

namespace prefill4k_attention {
void writeCaptures() {
  const char *raw=std::getenv("PREFILL4K_ATTENTION_CAPTURE");
  if (!raw || !*raw) return;
  const auto directory=std::filesystem::absolute(raw);
  require(captures.size()==3 && captured==std::set<uint32_t>{3,27,47},"QSA actual capture missing expected layers");
  require(!std::filesystem::exists(directory),"Choose a fresh QSA capture directory");
  std::filesystem::create_directories(directory);
  const auto write=[](const std::filesystem::path &path,const void *data,uint64_t bytes) {
    std::ofstream file(path,std::ios::binary);require(bool(file),"Cannot write QSA capture array");
    file.write(static_cast<const char *>(data),std::streamsize(bytes));
    require(bool(file),"QSA capture array write incomplete");
  };
  for (const auto &value:captures) {
    const auto folder=directory/("layer"+std::to_string(value.layer));
    std::filesystem::create_directory(folder);
    write(folder/"queries.bin",value.queries.contents(),value.queries.sizeBytes());
    write(folder/"keys.bin",value.keys.contents(),value.keys.sizeBytes());
    write(folder/"values.bin",value.values.contents(),value.values.sizeBytes());
    std::vector<uint16_t> gates(uint64_t(value.rows)*24*256);
    const auto *projection=static_cast<const uint16_t *>(value.rawProjection.contents());
    for (uint32_t row=0;row<value.rows;++row)
      for (uint32_t head=0;head<24;++head)
        std::copy_n(projection+(uint64_t(row)*24+head)*512+256,256,
                    gates.data()+(uint64_t(row)*24+head)*256);
    write(folder/"gates.bin",gates.data(),gates.size()*2);
    std::ofstream manifest(folder/"manifest.json");require(bool(manifest),"Cannot write QSA capture manifest");
    manifest << "{\"query_offset\":" << value.queryOffset << ",\"metadata\":{\"schema\":\"splash-private-actual-prepared-qsa-v1\","
        "\"actual_activations\":true,\"layer\":" << value.layer << ",\"queries_are_gpu_normalized_and_rotated\":true,"
        "\"scope\":" << splash::json::quote(value.rows == 2048 && value.queryOffset == 0
            ? "all2048queriesofuncached2Kprefill" : "last128queriesofuncached2Kprefill")
        << "},\"arrays\":{";
    bool first=true;
    for (const auto &[name,file,buffer] :
         {std::tuple{"queries_bf16","queries.bin",value.queries},std::tuple{"keys_bf16","keys.bin",value.keys},
          std::tuple{"values_bf16","values.bin",value.values}}) {
      if (!first) manifest << ',';first=false;
      manifest << splash::json::quote(name) << ":{\"file\":" << splash::json::quote(file)
          << ",\"dtype\":\"<u2\",\"shape\":[" << (std::string(name)=="queries_bf16"?value.rows:value.prefixRows)
          << ',' << (std::string(name)=="queries_bf16"?24:2) << ",256],\"sha256\":"
          << splash::json::quote(digest(buffer.contents(),buffer.sizeBytes())) << '}';
    }
    manifest << ",\"gates_bf16\":{\"file\":\"gates.bin\",\"dtype\":\"<u2\",\"shape\":["
        << value.rows << ",24,256],\"sha256\":" << splash::json::quote(digest(gates.data(),gates.size()*2)) << "}}}\n";
    require(bool(manifest),"QSA capture manifest write incomplete");
  }
  std::cout << "actual prepared QSA captures=" << captures.size() << " path=" << directory << '\n';
}
} // namespace prefill4k_attention
int main(int argc,char **argv) {
  const int result=prefill4k_attribution_main(argc,argv);
  if (result) return result;
  try { prefill4k_attention::writeCaptures();return 0; }
  catch (const std::exception &error) { std::cerr << "QSA capture write failed: " << error.what() << '\n';return 1; }
}
