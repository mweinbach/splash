// Root-only GPU oracle. --cpu-self-test creates no Metal device.
#include "metal/MetalBackend.hpp"
#include "metal/CommandGraph.hpp"
#include "IndirectDispatchPolicy.hpp"

#include <array>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace splash::metal;
struct Params { uint32_t count, slotWords, x, y, z, reserved; };
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class F> void reject(F function, const char *message) {
  bool rejected = false;
  try { function(); } catch (const MetalBackendError &) { rejected = true; }
  require(rejected, message);
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--cpu-self-test") {
      require(private_indirect::validArgumentsRange(12,0), "exact12 extent rejected");
      require(!private_indirect::validArgumentsRange(12,4), "short triplet accepted");
      require(!private_indirect::validArgumentsRange(16,1), "unaligned triplet accepted");
      require(private_indirect::validThreads(32,1,1), "valid threads rejected");
      require(!private_indirect::validThreads(0,1,1), "zero threads accepted");
      require(!IndirectDispatchSource{}, "default source is registered");
      std::cout << "{\"valid\":true,\"gpu_commands\":0,\"checks\":6}\n"; return 0;
    }
    require(argc == 2, "usage: private-indirect-oracle METALLIB | --cpu-self-test");
    MetalBackend backend(argv[1]);
    MetalBackend foreign(argv[1]);
    auto input = backend.allocateBuffer(64*4,BufferStorage::Shared,"primitive immutable input");
    auto control = backend.allocateBuffer(4,BufferStorage::Shared,"primitive GPU guard input");
    auto arguments = backend.allocateBuffer(96,BufferStorage::Private,"GPU-owned indirect triplets");
    const auto protectedInput = std::array<MetalBuffer,1>{input};
    const uint64_t before = backend.memoryStats().allocatedBytes;
    auto source = backend.registerIndirectDispatchSource(arguments,protectedInput);
    require(backend.memoryStats().allocatedBytes == before, "registration charged backing twice");
    reject([&]{ (void)backend.registerIndirectDispatchSource(input,protectedInput); }, "immutable alias accepted");
    reject([&]{ (void)backend.registerIndirectDispatchSource(backend.view(input,4,12),protectedInput); }, "partial immutable alias accepted");
    reject([&]{ (void)foreign.registerIndirectDispatchSource(arguments,{}); }, "foreign argument source accepted");
    reject([&]{ (void)backend.registerIndirectDispatchSource(backend.view(input,1,12),{}); }, "unaligned argument view accepted");
    reject([&]{ (void)backend.registerIndirectDispatchSource(backend.view(input,0,8),{}); }, "short argument source accepted");
    auto *values = static_cast<uint32_t *>(input.contents());
    for (uint32_t i=0;i<64;++i) values[i]=i*1664525u+1013904223u;
    uint64_t commands=0, scenarios=0;
    for (const auto groups : std::array<DispatchSize,4>{{{2,1,1},{0,0,0},{2,0,1},{2,1,0}}}) {
      auto base = backend.allocateBuffer((64+16)*4,BufferStorage::Shared,"guarded copy destination");
      auto output = backend.view(base,8*4,64*4);
      auto *guarded = static_cast<uint32_t *>(base.contents());
      std::fill(guarded,guarded+80,0xdeadbeefu);
      *static_cast<uint32_t *>(control.contents())=1;
      const Params params{64,3,uint32_t(groups.x),uint32_t(groups.y),uint32_t(groups.z),0};
      CommandGraph graph;
      graph.add("private_indirect_emit",{arguments,control},params,{1,1,1},{32,1,1});
      graph.add("private_indirect_copy",{input,output},params,{2,1,1},{32,1,1});
      std::vector<ComputeDispatch> list(graph.dispatches().begin(),graph.dispatches().end());
      list[1].indirectGroups=ComputeDispatch::IndirectGroups{source,12};
      reject([&]{auto bad=list[1];bad.indirectGroups->offsetBytes=1;(void)backend.submit(bad);},"unaligned dispatch offset accepted");
      reject([&]{auto bad=list[1];bad.indirectGroups->offsetBytes=88;(void)backend.submit(bad);},"short dispatch extent accepted");
      reject([&]{auto bad=list[1];bad.indirectGroups->source={};(void)backend.submit(bad);},"unregistered source accepted");
      reject([&]{(void)foreign.submit(list[1]);},"foreign dispatch source accepted");
      const auto timing=backend.submitCommand(list); ++commands;
      require(timing.wallSeconds>=0 && timing.gpuSeconds>=0,"normal command timing invalid");
      const bool active=groups.x && groups.y && groups.z;
      for(uint32_t i=0;i<80;++i)
        require(guarded[i]==(active && i>=8 && i<72 ? values[i-8] : 0xdeadbeefu),"zero dispatch wrote or copy guard changed");
      ++scenarios;
    }
    // The command ticket must retain dimensions after all caller-owned graph,
    // source-token and argument-buffer references are destroyed.
    auto lifetimeOutput=backend.allocateBuffer(64*4,BufferStorage::Shared,"ticket retained destination");
    {
      CommandGraph producer;
      const Params params{64,3,2,1,1,0};
      producer.add("private_indirect_emit",{arguments,control},params,{1,1,1},{32,1,1});
      (void)backend.submitCommand(producer.dispatches()); ++commands;
    }
    CommandTicket ticket;
    {
      CommandGraph graph;
      const Params params{64,3,2,1,1,0};
      graph.add("private_indirect_copy",{input,lifetimeOutput},params,{2,1,1},{32,1,1});
      std::vector<ComputeDispatch> list(graph.dispatches().begin(),graph.dispatches().end());
      list[0].indirectGroups=ComputeDispatch::IndirectGroups{source,12};
      ticket=backend.submitCommandAsync(list);
    }
    source={}; arguments={};
    (void)ticket.wait(); ++commands;
    require(std::memcmp(lifetimeOutput.contents(),input.contents(),64*4)==0,"ticket lost indirect argument ownership");
    std::cout<<"{\"valid\":true,\"gpu_commands\":"<<commands<<",\"scenarios\":"<<scenarios
      <<",\"zero_dispatch_no_writes\":true,\"guarded_copy_exact\":true,\"ticket_retention\":true,\"dimension_host_readbacks\":0}\n";
    return 0;
  } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
