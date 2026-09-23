// Compare opt_moe_q4_{gate_up,down_scatter} with flash_moe_direct_a_* on a
// synthetic bucket layout. Usage: moe_test <metallib> <M>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal/abi/FlashMoEBlocked.h"
#include "metal/abi/FlashMoEBuckets.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return uint16_t((u + ((u >> 16) & 1) + 0x7fff) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char **argv) {
  @autoreleasepool {
    const uint32_t M = atoi(argv[2]);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    std::mt19937 rng(11);
    // Buckets: (expert, rows)
    const bool bench = argc > 3;
    std::vector<std::pair<uint32_t, uint32_t>> buckets{{3, 40}, {100, 70}, {257, 1}, {511, 33}, {7, 64}};
    uint32_t rowsN = 24, sel = 10;
    if (bench) {
      rowsN = 2048; buckets.clear();
      std::vector<uint32_t> c(512, 0);
      for (uint32_t r = 0; r < rowsN * sel; ++r) c[rng() % 512]++;
      for (uint32_t e = 0; e < 512; ++e) buckets.push_back({e, c[e]});
    }
    const uint32_t routes = rowsN * sel;
    uint32_t used = 0; for (auto &b : buckets) used += b.second;
    if (used > routes) { printf("too many\n"); return 1; }
    std::vector<uint32_t> offsets(513, 0), counts(512, 0);
    for (auto &b : buckets) counts[b.first] = b.second;
    for (uint32_t e = 0; e < 512; ++e) offsets[e + 1] = offsets[e] + counts[e];
    std::vector<FlashMoEBucketJob> jobs;
    for (uint32_t e = 0; e < 512; ++e)
      for (uint32_t r = offsets[e]; r < offsets[e + 1]; r += M) jobs.push_back({e, r});
    const uint32_t jobCapacity = (routes + M - 1) / M + 511;
    auto buf = [&](size_t bytes) { return [dev newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared]; };
    const uint64_t guard = 63;
    id<MTLBuffer> packed = buf((routes + guard) * 2560 * 2), act0 = buf((routes + guard) * 640 * 2), act1 = buf((routes + guard) * 640 * 2);
    id<MTLBuffer> off = buf(513 * 4), jb = buf(jobCapacity * 8), jc = buf(4), rmap = buf(routes * 4), diag = buf(16);
    memcpy(off.contents, offsets.data(), 513 * 4); memcpy(jb.contents, jobs.data(), jobs.size() * 8);
    *(uint32_t *)jc.contents = uint32_t(jobs.size());
    { auto *p = (uint16_t *)packed.contents; std::uniform_real_distribution<float> d(-1, 1);
      for (uint64_t i = 0; i < (routes + guard) * 2560; ++i) p[i] = i < uint64_t(used) * 2560 ? f2bf(bench ? 0.5f : d(rng)) : 0; }
    { auto *m = (uint32_t *)rmap.contents; for (uint32_t i = 0; i < routes; ++i) m[i] = (i * 37) % routes; }
    const uint64_t guRow = 1280, guExp = 640 * guRow, guP = 40 * 2, guPExp = 640 * guP;
    const uint64_t dnRow = 320, dnExp = 2560 * dnRow, dnP = 10 * 2, dnPExp = 2560 * dnP;
    id<MTLBuffer> gw = buf(512 * guExp), gs = buf(512 * guPExp), gbb = buf(512 * guPExp);
    id<MTLBuffer> uw = buf(512 * guExp), us = buf(512 * guPExp), ubb = buf(512 * guPExp);
    id<MTLBuffer> dw = buf(512 * dnExp), ds = buf(512 * dnPExp), dbb = buf(512 * dnPExp);
    auto fillExpert = [&](id<MTLBuffer> w, id<MTLBuffer> s, id<MTLBuffer> b, uint64_t we, uint64_t pe, uint32_t e) {
      auto *pw = (uint8_t *)w.contents + e * we; for (uint64_t i = 0; i < we; ++i) pw[i] = bench ? uint8_t(i * 131 + e) : uint8_t(rng());
      std::uniform_real_distribution<float> sd(0.001f, 0.02f), bd(-0.1f, 0.1f);
      auto *ps = (uint16_t *)((uint8_t *)s.contents + e * pe), *pb = (uint16_t *)((uint8_t *)b.contents + e * pe);
      for (uint64_t i = 0; i < pe / 2; ++i) { ps[i] = f2bf(sd(rng)); pb[i] = f2bf(bd(rng)); } };
    for (auto &bk : buckets) {
      fillExpert(gw, gs, gbb, guExp, guPExp, bk.first); fillExpert(uw, us, ubb, guExp, guPExp, bk.first);
      fillExpert(dw, ds, dbb, dnExp, dnPExp, bk.first);
    }
    FlashMoEBlockedGateParams gp{{rowsN, sel, 2560, 640, 512, 0, 0, 0, guRow, guExp, guP, guPExp, guRow, guExp, guP, guPExp},
                                 routes, jobCapacity, M, 0};
    FlashMoEBlockedDownParams dp{{rowsN, sel, 640, 2560, 512, 0, 0, 0, dnRow, dnExp, dnP, dnPExp}, routes, jobCapacity, M, 0};
    id<MTLCommandQueue> q = [dev newCommandQueue];
    auto pso = [&](NSString *n) { id<MTLFunction> f = [L newFunctionWithName:n]; if (!f) { printf("missing %s\n", n.UTF8String); exit(1); } return [dev newComputePipelineStateWithFunction:f error:&err]; };
    NSString *suffix = M == 64 ? @"_m64_n64_sg8" : [NSString stringWithFormat:@"_m%u_n64", M];
    const NSUInteger threads = getenv("THREADS") ? atoi(getenv("THREADS")) : (M == 64 ? 256 : 128);
    id<MTLBuffer> down0 = buf(routes * 2560 * 2), down1 = buf(routes * 2560 * 2);
    id<MTLBuffer> sums0 = buf((routes + guard) * 40 * 4), sums1 = buf((routes + guard) * 10 * 4);
    auto run = [&](NSString *gate, NSString *down, id<MTLBuffer> act, id<MTLBuffer> out) {
      const bool direct = [gate hasPrefix:@"opt_moe_q4d"];
      memset(act.contents, 0, act.length); memset(out.contents, 0, out.length);
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      if (direct) {
        [e setComputePipelineState:pso(@"opt_moe_group_sums")];
        uint32_t shape[2] = {uint32_t(routes + guard), 2560};
        [e setBuffer:packed offset:0 atIndex:0]; [e setBuffer:sums0 offset:0 atIndex:1];
        [e setBytes:shape length:8 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(routes + guard, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }
      [e setComputePipelineState:pso([gate stringByAppendingString:suffix])];
      id<MTLBuffer> g[] = {packed, gw, gs, gbb, uw, us, ubb, off, jb, jc, act, direct ? sums0 : diag};
      for (int i = 0; i < 12; ++i) [e setBuffer:g[i] offset:0 atIndex:i];
      [e setBytes:&gp length:sizeof gp atIndex:12];
      const NSUInteger tgThreads = threads;
      [e dispatchThreadgroups:MTLSizeMake(10, jobCapacity, 1) threadsPerThreadgroup:MTLSizeMake(tgThreads, 1, 1)];
      if (direct) {
        [e setComputePipelineState:pso(@"opt_moe_group_sums")];
        uint32_t shape[2] = {uint32_t(routes + guard), 640};
        [e setBuffer:act offset:0 atIndex:0]; [e setBuffer:sums1 offset:0 atIndex:1];
        [e setBytes:shape length:8 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(routes + guard, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      }
      [e setComputePipelineState:pso([down stringByAppendingString:suffix])];
      id<MTLBuffer> d[] = {act, dw, ds, dbb, off, jb, jc, rmap, out, direct ? sums1 : diag};
      for (int i = 0; i < 10; ++i) [e setBuffer:d[i] offset:0 atIndex:i];
      [e setBytes:&dp length:sizeof dp atIndex:10];
      [e dispatchThreadgroups:MTLSizeMake(40, jobCapacity, 1) threadsPerThreadgroup:MTLSizeMake(tgThreads, 1, 1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      return cb.GPUEndTime - cb.GPUStartTime;
    };
    if (bench) {
      NSString *g = @(argc > 4 ? argv[4] : "opt_moe_q4_gate_up"), *d = @(argc > 5 ? argv[5] : "opt_moe_q4_down_scatter");
      double best = 1e9;
      for (int rep = 0; rep < 4; ++rep) best = std::min(best, run(g, d, act1, down1));
      if (argc > 7) {
        run(@(argv[6]), @(argv[7]), act0, down0);
        run(g, d, act1, down1);
        const uint16_t *pa = (const uint16_t *)act0.contents, *pb = (const uint16_t *)act1.contents;
        uint64_t diff = 0; double maxd = 0;
        for (uint64_t i = 0; i < uint64_t(used) * 640; ++i) { diff += pa[i] != pb[i]; maxd = std::max(maxd, (double)std::fabs(bf2f(pa[i]) - bf2f(pb[i]))); }
        const uint16_t *da = (const uint16_t *)down0.contents, *db = (const uint16_t *)down1.contents;
        uint64_t ddiff = 0; double dmax = 0;
        for (uint64_t i = 0; i < uint64_t(routes) * 2560; ++i) { ddiff += da[i] != db[i]; dmax = std::max(dmax, (double)std::fabs(bf2f(da[i]) - bf2f(db[i]))); }
        printf("compare act differ=%llu max=%.4g  down differ=%llu max=%.4g\n", diff, maxd, ddiff, dmax);
      }
      printf("M=%u %s+%s %.3f ms (jobs %zu)\n", M, g.UTF8String, d.UTF8String, best * 1e3, jobs.size());
      return 0;
    }
    run(@"flash_moe_direct_a_gate_up", @"flash_moe_direct_a_down_scatter", act0, down0);
    run(getenv("DIRECT") ? @"opt_moe_q4d_gate_up" : @"opt_moe_q4_gate_up",
        getenv("DIRECT") ? @"opt_moe_q4d_down_scatter" : @"opt_moe_q4_down_scatter", act1, down1);
    auto cmp = [&](const char *name, id<MTLBuffer> a, id<MTLBuffer> b, uint64_t n) {
      const uint16_t *pa = (const uint16_t *)a.contents, *pb = (const uint16_t *)b.contents;
      double maxd = 0, maxa = 0; uint64_t diff = 0, nz = 0;
      for (uint64_t i = 0; i < n; ++i) { const double x = bf2f(pa[i]), y = bf2f(pb[i]);
        maxd = std::max(maxd, std::fabs(x - y)); maxa = std::max(maxa, std::fabs(x)); diff += pa[i] != pb[i]; nz += pa[i] != 0; }
      printf("%-10s n=%llu nonzero=%llu differ=%llu maxdiff=%.4g maxabs=%.4g\n", name, n, nz, diff, maxd, maxa); };
    cmp("activated", act0, act1, uint64_t(used) * 640);
    cmp("down", down0, down1, uint64_t(routes) * 2560);
    printf("diag %u\n", *(uint32_t *)diag.contents);
  }
  return 0;
}
