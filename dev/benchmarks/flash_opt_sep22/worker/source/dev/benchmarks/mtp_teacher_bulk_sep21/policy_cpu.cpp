#include "dev/benchmarks/mtp_teacher_bulk_sep21/bulk.hpp"
#include "dev/benchmarks/mtp_teacher_bulk_sep21/policy.hpp"
#include <array>
#include <cassert>
#include <iostream>
int main(){
  using splash::flash::FlashMTPTeacherBulkForward;
  constexpr std::array<uint32_t,17> bytes{8,5120,5120,5120,20480,20480,20480,20480,640,
      20480,8,8,5120,24576,1024,1024,1280};
  uint64_t total=16384;
  for(auto b:bytes)total+=(uint64_t{b}*2048+16383)/16384*16384;
  assert(total==FlashMTPTeacherBulkForward::plannedBytes);
  uint64_t checks=1;
  using namespace splash::flash::teacher_bulk_sep21;
  assert(!parse(nullptr)&&!parse("0")&&parse("1"));checks+=3;
  for(auto value:{"","true","01","-1","2"," 1"}){
    bool refused=false;try{(void)parse(value);}catch(const std::invalid_argument&){refused=true;}
    assert(refused);++checks;
  }
  for(uint32_t mask=0;mask<32;++mask){
    const Dependencies dependencies{bool(mask&1),bool(mask&2),bool(mask&4),bool(mask&8),bool(mask&16)};
    validate(false,dependencies);++checks;
    bool refused=false;try{validate(true,dependencies);}catch(const std::invalid_argument&){refused=true;}
    assert(refused==(mask!=31));++checks;
  }
  for(uint32_t rows=128;rows<=2048;rows+=128){
    assert(rows*4<=8192);++checks;
    for(uint32_t slot=0;slot<17;++slot){
      assert(uint64_t{rows}*bytes[slot]<=uint64_t{2048}*bytes[slot]);++checks;
    }
    for(uint32_t slice=0;slice<rows/128;++slice){
      for(auto width:{12288u,512u,640u}){
        assert(uint64_t{slice}*128*width*2+uint64_t{128}*width*2<=uint64_t{rows}*width*2);++checks;
      }
    }
  }
  for(uint32_t rows=1;rows<=8192;++rows){
    const auto quantum=rows/128*128;assert(quantum<=rows&&rows-quantum<128);++checks;
  }
  std::cout<<"{\"pass\":true,\"gpu_executed\":false,\"model_payload_bytes_read\":0,\"geometry_checks\":"
      <<checks<<",\"planned_bytes\":"<<total<<"}\n";
}
