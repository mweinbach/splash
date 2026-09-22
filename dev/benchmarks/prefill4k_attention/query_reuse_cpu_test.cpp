#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main() {
  constexpr uint32_t M=32,Threads=128,Prefix=2*M;
  union Bank { uint16_t query[M*256]; float alpha[M]; };
  static_assert(sizeof(Bank)+64*64*sizeof(uint16_t)+M*64*sizeof(float)==32768);
  uint64_t checks=0;
  auto require=[&](bool valid,const char *message) {
    ++checks; if (!valid) throw std::runtime_error(message);
  };
  try {
    for (uint32_t rows:{32u,33u,63u,64u,127u,128u})
      for (uint32_t kv=0;kv<2;++kv)
        for (uint32_t group=0;group<(rows*12+M-1)/M;++group) {
          std::array<uint16_t,M*256> source{};
          for (uint32_t i=0;i<source.size();++i)
            source[i]=uint16_t((i*181+group*59+kv*31)^(i>>3));
          std::array<uint32_t,M*256> visits{};
          std::array<uint16_t,Threads> retained{};
          Bank bank{};
          for (uint32_t tid=0;tid<Threads;++tid)
            for (uint32_t i=tid;i<M*256;i+=Threads) {
              const uint32_t flat=group*M+i/256;
              bank.query[i]=flat<rows*12?source[i]:0;
              source[i]=bank.query[i];
              ++visits[i];
              if (i==tid && tid<Prefix) retained[tid]=bank.query[i];
            }
          for (auto visit:visits) require(visit==1,"Query bank load ownership differs");
          for (uint32_t tokenBank=0;tokenBank<8;++tokenBank) {
            for (uint32_t tid=0;tid<Prefix;++tid) bank.query[tid]=retained[tid];
            for (uint32_t chunk=0;chunk<4;++chunk)
              for (uint32_t head=0;head<M;++head)
                for (uint32_t dim=0;dim<64;++dim) {
                  const uint32_t address=head*256+chunk*64+dim;
                  require(bank.query[address]==source[address],
                          "Strided QK operand differs after alpha prefix restoration");
                }
            for (uint32_t h=0;h<M;++h) bank.alpha[h]=float((h+tokenBank)%17)*.0625f;
            for (uint32_t i=Prefix;i<M*256;++i)
              require(bank.query[i]==source[i],"Alpha overwrote query data outside retained prefix");
          }
        }
    std::cout << "{\"pass\":true,\"query_bank_alias_and_strided_operand_cpu_checks\":"
              << checks << ",\"shared_memory_bytes_source\":32768,\"gpu_commands\":0}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "query-reuse CPU audit failed: " << error.what() << '\n'; return 1;
  }
}
