// Prefill routed-expert GEMM benchmark: mk_moe_prefill_*_pipe vs mk_pf2_*.
//   ./pf2_bench [tokens] [zipf_s]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return uint16_t((u + ((u >> 16) & 1) + 0x7fff) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char **argv) {
  @autoreleasepool {
    const uint32_t tokens = argc > 1 ? atoi(argv[1]) : 2048;
    const double zipf = argc > 2 ? atof(argv[2]) : 0.0;
    const uint32_t E = 512, SEL = 10, routes = tokens * SEL, LAYERS = 2;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"pf2.metallib"] error:&err];
    if (!lib) { printf("lib %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    // Expert popularity: weight ~ 1/(rank+1)^s, each token picks 10 distinct experts.
    std::vector<double> weight(E);
    std::vector<uint32_t> perm(E);
    for (uint32_t e = 0; e < E; ++e) perm[e] = e;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (uint32_t e = 0; e < E; ++e) weight[perm[e]] = 1.0 / std::pow(double(e + 1), zipf);
    std::discrete_distribution<uint32_t> pick(weight.begin(), weight.end());
    std::vector<uint32_t> counts(E, 0);
    for (uint32_t t = 0; t < tokens; ++t) {
      std::vector<uint32_t> chosen;
      while (chosen.size() < SEL) {
        const uint32_t e = pick(rng);
        if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) chosen.push_back(e);
      }
      for (uint32_t e : chosen) counts[e]++;
    }
    std::vector<uint32_t> offsets(513, 0);
    for (uint32_t e = 0; e < E; ++e) offsets[e + 1] = offsets[e] + counts[e];
    std::vector<FlashMoEBucketJob> jobs;
    uint32_t used = 0;
    for (uint32_t e = 0; e < E; ++e)
      for (uint32_t r = 0; r < counts[e]; r += 64) { jobs.push_back({e, offsets[e] + r}); used += std::min(64u, counts[e] - r); }
    const uint32_t jobCapacity = (routes + 63) / 64 + 511;
    const uint32_t jobCount = uint32_t(jobs.size());
    uint32_t maxc = 0; for (auto c : counts) maxc = std::max(maxc, c);
    printf("tokens %u zipf %.2f: %u jobs, mean rows/job %.1f, max bucket %u, 64-row tile utilization %.1f%%\n",
           tokens, zipf, jobCount, double(routes) / jobCount, maxc, 100.0 * routes / (64.0 * jobCount));
    jobs.resize(jobCapacity, FlashMoEBucketJob{UINT32_MAX, 0});
    auto buf = [&](const void *src, uint64_t bytes) {
      id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      if (src) memcpy(b.contents, src, bytes); else memset(b.contents, 0, bytes);
      return b;
    };
    id<MTLBuffer> offB = buf(offsets.data(), offsets.size() * 4), jobB = buf(jobs.data(), jobs.size() * 8);
    id<MTLBuffer> cntB = buf(&jobCount, 4), diag = buf(nullptr, 64);
    std::vector<uint32_t> map(routes);
    for (uint32_t i = 0; i < routes; ++i) map[i] = i;
    std::shuffle(map.begin(), map.end(), rng);
    id<MTLBuffer> mapB = buf(map.data(), routes * 4);
    struct Q4 { id<MTLBuffer> w, s, b; };
    auto makeQ4 = [&](uint32_t N, uint32_t K) {
      const uint64_t wb = uint64_t(E) * N * K / 2, pc = uint64_t(E) * N * (K / 64);
      Q4 m{buf(nullptr, wb), buf(nullptr, pc * 2), buf(nullptr, pc * 2)};
      uint32_t *wp = (uint32_t *)m.w.contents;
      for (uint64_t i = 0; i < wb / 4; ++i) wp[i] = rng();
      uint16_t *sp = (uint16_t *)m.s.contents, *bp = (uint16_t *)m.b.contents;
      for (uint64_t i = 0; i < pc; ++i) { sp[i] = f2bf(0.002f * (1.f + 0.2f * std::fabs(nd(rng)))); bp[i] = f2bf(-0.01f + 0.002f * nd(rng)); }
      return m;
    };
    std::vector<Q4> G, U, D;
    for (uint32_t l = 0; l < LAYERS; ++l) { G.push_back(makeQ4(640, 2560)); U.push_back(makeQ4(640, 2560)); D.push_back(makeQ4(2560, 640)); }
    id<MTLBuffer> xin = buf(nullptr, uint64_t(routes) * 2560 * 2), iin = buf(nullptr, uint64_t(routes) * 640 * 2);
    for (uint64_t i = 0; i < uint64_t(routes) * 2560; ++i) ((uint16_t *)xin.contents)[i] = f2bf(nd(rng) * 0.5f);
    for (uint64_t i = 0; i < uint64_t(routes) * 640; ++i) ((uint16_t *)iin.contents)[i] = f2bf(nd(rng) * 0.1f);
    id<MTLBuffer> outA = buf(nullptr, uint64_t(routes) * 2560 * 2), outB = buf(nullptr, uint64_t(routes) * 2560 * 2);
    FlashMoEBlockedGateParams gp{};
    gp.affine = {tokens, SEL, 2560, 640, E, 0, 0, 0, 1280, 640ull * 1280, 80, 640ull * 80, 1280, 640ull * 1280, 80, 640ull * 80};
    gp.route_capacity = routes; gp.job_capacity = jobCapacity; gp.tile_rows = 64;
    FlashMoEBlockedDownParams dp{};
    dp.affine = {tokens, SEL, 640, 2560, E, 0, 0, 0, 320, 2560ull * 320, 20, 2560ull * 20};
    dp.route_capacity = routes; dp.job_capacity = jobCapacity; dp.tile_rows = 64;
    auto run = [&](NSString *name, bool gate, uint32_t grid, id<MTLBuffer> out, int reps, bool timed) {
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
      if (!pso) { printf("pso %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
      double best = 1e9;
      for (int t = 0; t < (timed ? 5 : 1); ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        for (int rep = 0; rep < reps; ++rep) {
          const uint32_t l = timed ? rep % LAYERS : LAYERS - 1;
          if (gate) {
            [e setBuffer:xin offset:0 atIndex:0];
            [e setBuffer:G[l].w offset:0 atIndex:1]; [e setBuffer:G[l].s offset:0 atIndex:2]; [e setBuffer:G[l].b offset:0 atIndex:3];
            [e setBuffer:U[l].w offset:0 atIndex:4]; [e setBuffer:U[l].s offset:0 atIndex:5]; [e setBuffer:U[l].b offset:0 atIndex:6];
            [e setBuffer:offB offset:0 atIndex:7]; [e setBuffer:jobB offset:0 atIndex:8]; [e setBuffer:cntB offset:0 atIndex:9];
            [e setBuffer:out offset:0 atIndex:10]; [e setBuffer:diag offset:0 atIndex:11]; [e setBytes:&gp length:sizeof gp atIndex:12];
          } else {
            [e setBuffer:iin offset:0 atIndex:0];
            [e setBuffer:D[l].w offset:0 atIndex:1]; [e setBuffer:D[l].s offset:0 atIndex:2]; [e setBuffer:D[l].b offset:0 atIndex:3];
            [e setBuffer:offB offset:0 atIndex:4]; [e setBuffer:jobB offset:0 atIndex:5]; [e setBuffer:cntB offset:0 atIndex:6];
            [e setBuffer:mapB offset:0 atIndex:7]; [e setBuffer:out offset:0 atIndex:8]; [e setBuffer:diag offset:0 atIndex:9];
            [e setBytes:&dp length:sizeof dp atIndex:10];
          }
          [e dispatchThreadgroups:MTLSizeMake(grid, jobCapacity, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / reps);
      }
      return best;
    };
    // Warm up.
    for (int i = 0; i < 6; ++i) run(@"mk_moe_prefill_gate_up_pipe", true, 10, outA, 8, false);
    const double guFlop = double(routes) * 2 * 640 * 2560 * 2, dnFlop = double(routes) * 2560 * 640 * 2;
    struct K { NSString *name; bool gate; uint32_t grid; };
    const K kernels[] = {{@"mk_moe_prefill_gate_up_pipe", true, 10}, {@"mk_pf2_gate_up", true, 10},
                         {@"mk_pf2s_gate_up", true, 10},
                         {@"mk_moe_prefill_down_pipe", false, 20}, {@"mk_pf2_down", false, 20},
                         {@"mk_pf2s_down", false, 20}};
    for (const K &k : kernels) {
      const double t = run(k.name, k.gate, k.grid, outA, 8, true);
      printf("  %-30s %8.1f us  %6.1f TFLOPS\n", k.name.UTF8String, t * 1e6, (k.gate ? guFlop : dnFlop) / t / 1e12);
    }
    // Correctness: new vs old on the last layer.
    for (int gate = 1; gate >= 0; --gate) {
      memset(outA.contents, 0, outA.length); memset(outB.contents, 0, outB.length);
      run(gate ? @"mk_moe_prefill_gate_up_pipe" : @"mk_moe_prefill_down_pipe", gate, gate ? 10 : 20, outA, 1, false);
      run(gate ? @"mk_pf2s_gate_up" : @"mk_pf2s_down", gate, gate ? 10 : 20, outB, 1, false);
      const uint64_t n = uint64_t(routes) * (gate ? 640 : 2560);
      double maxd = 0, maxv = 0;
      for (uint64_t i = 0; i < n; ++i) {
        const double a = bf2f(((uint16_t *)outA.contents)[i]), b = bf2f(((uint16_t *)outB.contents)[i]);
        maxd = std::max(maxd, std::fabs(a - b)); maxv = std::max(maxv, std::fabs(a));
      }
      printf("  %s new vs old: max |diff| %.3e (max |value| %.3e)\n", gate ? "gate_up" : "down", maxd, maxv);
    }
  }
  return 0;
}
