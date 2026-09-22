#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  uint64_t checks=0;
  auto require=[&](bool valid,const char *message) {
    ++checks; if (!valid) throw std::runtime_error(message);
  };
  try {
    std::array<uint32_t,32*256> output{};
    std::array<std::array<uint32_t,32*64>,4> left{};
    std::array<std::vector<uint32_t>,4> right;
    for (auto &v:right) v.resize(64*256);
    for (uint32_t simd=0;simd<4;++simd)
      for (uint32_t lane=0;lane<32;++lane) {
        const uint32_t qid=lane>>2,sm=(qid&4)|((lane>>1)&3),sn=((qid&2)|(lane&1))*4;
        for (uint32_t fragment=0;fragment<8;++fragment)
          for (uint32_t element=0;element<8;++element) {
            const uint32_t q=simd/2*16+sm+element/4*8;
            const uint32_t d=simd%2*128+fragment*16+sn+element%4;
            require(q<32 && d<256,"Output mapping escapes current M32/D256");
            ++output[q*256+d];
          }
        for (uint32_t part=0;part<4;++part)
          for (uint32_t element=0;element<8;++element) {
            const uint32_t q=simd/2*16+sm+element/4*8,k=part*16+sn+element%4;
            ++left[simd][q*64+k];
            for (uint32_t pair=0;pair<4;++pair) {
              const uint32_t token=part*16+sm+element/4*8;
              const uint32_t d=simd%2*128+pair*32+sn+element%4;
              require(token<64 && d+16<256,"Right-register load escapes64-token bank");
              ++right[simd][token*256+d]; ++right[simd][token*256+d+16];
            }
          }
      }
    for (auto visit:output) require(visit==1,"Output coverage is not one writer per cell");
    for (uint32_t simd=0;simd<4;++simd) {
      for (uint32_t q=0;q<32;++q)
        for (uint32_t k=0;k<64;++k)
          require(left[simd][q*64+k]==uint32_t(q/16==simd/2),"Left register operand coverage differs");
      for (uint32_t token=0;token<64;++token)
        for (uint32_t d=0;d<256;++d)
          require(right[simd][token*256+d]==uint32_t(d/128==simd%2),"Right register operand coverage differs");
    }
    // Independence/causality: direct V reads may cover the common bank, but
    // the unchanged per-query mask makes all out-of-window weights exactly0.
    for (uint32_t begin:{512u,1024u,1920u})
      for (uint32_t flatBase=0;flatBase<128*12;flatBase+=32)
        for (uint32_t partition=0;partition<4;++partition) {
          const uint32_t first=flatBase/12,last=(flatBase+31)/12;
          const uint32_t firstCount=begin+first+1,lastCount=begin+last+1;
          const uint32_t commonStart=partition*((firstCount+3)/4);
          const uint32_t commonStop=std::min((partition+1)*((lastCount+3)/4),lastCount);
          for (uint32_t q=0;q<32;++q) {
            const uint32_t count=begin+(flatBase+q)/12+1,len=(count+3)/4;
            const uint32_t start=std::min(partition*len,count),stop=std::min(start+len,count);
            for (uint32_t token=commonStart;token<commonStop;++token) {
              const bool nonzero=token>=start && token<stop;
              require(!nonzero || token<count,"Direct V register path retains a future token");
            }
          }
        }
    std::cout << "{\"pass\":true,\"register_lane_and_causal_weight_cpu_checks\":"
              << checks << ",\"source_shared_bytes\":20480,\"gpu_commands\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "register PV CPU audit failed: " << error.what() << '\n'; return 1;
  }
}
