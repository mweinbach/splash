// Gathered-expert GEMV prototype benchmark: U unique experts of a decode window.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Plan { uint32_t rows, unique, pad0, pad1; uint32_t experts[80]; uint32_t row_mask[80]; };
struct GUParams { uint32_t K, N, rows, pad; uint64_t w_row_stride, w_expert_stride, p_row_stride, p_expert_stride; };

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t r = ((u >> 16) & 1) + 0x7fff; return uint16_t((u + r) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char **argv) {
  @autoreleasepool {
    const uint32_t R = argc > 1 ? atoi(argv[1]) : 5;
    const uint32_t U = argc > 2 ? atoi(argv[2]) : 45;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"moe_mpp.metallib"] error:&err];
    if (!lib) { printf("lib %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    std::mt19937 rng(7);
    const uint32_t E = 512;
    // gate/up: [E, 640, 2560] Q4 G64; down: [E, 2560, 640]
    auto makeQ4 = [&](uint32_t N, uint32_t K, __strong id<MTLBuffer> &w, __strong id<MTLBuffer> &s, __strong id<MTLBuffer> &b) {
      const uint64_t wbytes = uint64_t(E) * N * K / 2, pcount = uint64_t(E) * N * (K / 64);
      w = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
      s = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      b = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      uint32_t *wp = (uint32_t *)w.contents;
      for (uint64_t i = 0; i < wbytes / 4; ++i) wp[i] = rng();
      uint16_t *sp = (uint16_t *)s.contents, *bp = (uint16_t *)b.contents;
      std::normal_distribution<float> nd(0.f, 1.f);
      for (uint64_t i = 0; i < pcount; ++i) { sp[i] = f2bf(0.004f + 0.001f * std::fabs(nd(rng))); bp[i] = f2bf(-0.03f + 0.003f * nd(rng)); }
    };
    id<MTLBuffer> gw, gs, gb, uw, us, ub, dw, ds, db;
    makeQ4(640, 2560, gw, gs, gb);
    makeQ4(640, 2560, uw, us, ub);
    makeQ4(2560, 640, dw, ds, db);
    auto transposeS = [&](id<MTLBuffer> s, uint32_t N, uint32_t K) {
      const uint32_t ng = K / 64;
      id<MTLBuffer> t = [dev newBufferWithLength:uint64_t(E) * N * ng * 2 options:MTLResourceStorageModeShared];
      const uint16_t *sp = (const uint16_t *)s.contents; uint16_t *tp = (uint16_t *)t.contents;
      for (uint64_t e = 0; e < E; ++e) for (uint32_t n = 0; n < N; ++n) for (uint32_t g = 0; g < ng; ++g)
        tp[(e * ng + g) * N + n] = sp[(e * N + n) * ng + g];
      return t;
    };
    id<MTLBuffer> gst = transposeS(gs, 640, 2560), ust = transposeS(us, 640, 2560), dst = transposeS(ds, 2560, 640);
    std::normal_distribution<float> nd(0.f, 1.f);
    id<MTLBuffer> x = [dev newBufferWithLength:8 * 2560 * 2 options:MTLResourceStorageModeShared];
    for (uint32_t i = 0; i < 8 * 2560; ++i) ((uint16_t *)x.contents)[i] = f2bf(nd(rng) * 0.5f);
    id<MTLBuffer> xi = [dev newBufferWithLength:80 * 16 * 640 * 2 options:MTLResourceStorageModeShared];
    for (uint32_t i = 0; i < 80 * 16 * 640; ++i) ((uint16_t *)xi.contents)[i] = f2bf(nd(rng) * 0.1f);
    const int NPLAN = 16;
    std::vector<id<MTLBuffer>> plans;
    for (int k = 0; k < NPLAN; ++k) {
      Plan p{}; p.rows = R; p.unique = U;
      std::vector<uint32_t> ids(E); for (uint32_t i = 0; i < E; ++i) ids[i] = i;
      std::shuffle(ids.begin(), ids.end(), rng);
      std::sort(ids.begin(), ids.begin() + U);
      for (uint32_t u = 0; u < U; ++u) { p.experts[u] = ids[u]; p.row_mask[u] = 1u << (rng() % R); }
      id<MTLBuffer> pb = [dev newBufferWithBytes:&p length:sizeof(p) options:MTLResourceStorageModeShared];
      plans.push_back(pb);
    }
    id<MTLBuffer> out = [dev newBufferWithLength:80 * 16 * 2560 * 2 options:MTLResourceStorageModeShared];
    for (NSString *name in [lib.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
      std::string n = name.UTF8String;
      int M = 0, NT = 0, SK = 0, MODE = 0, EPI = 0;
      sscanf(n.c_str(), "mpp_gathered_m%d_nt%d_sk%d_mode%d_epi%d", &M, &NT, &SK, &MODE, &EPI);
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
      if (!pso) { printf("pso %s: %s\n", n.c_str(), err.localizedDescription.UTF8String); continue; }
      const uint32_t K = MODE ? 2560 : 640, N = MODE ? 640 : 2560;
      GUParams gp{K, N, R, 0, K / 2, uint64_t(N) * K / 2, K / 64, uint64_t(N) * (K / 64)};
      double best = 1e9;
      for (int t = 0; t < 6; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        const int reps = 16;
        for (int rep = 0; rep < reps; ++rep) {
          [e setBuffer:MODE ? x : xi offset:0 atIndex:0];
          [e setBuffer:MODE ? gw : dw offset:0 atIndex:1]; [e setBuffer:MODE ? gs : ds offset:0 atIndex:2]; [e setBuffer:MODE ? gb : db offset:0 atIndex:3];
          [e setBuffer:MODE ? uw : dw offset:0 atIndex:4]; [e setBuffer:MODE ? us : ds offset:0 atIndex:5]; [e setBuffer:MODE ? ub : db offset:0 atIndex:6];
          [e setBuffer:plans[(t * reps + rep) % NPLAN] offset:0 atIndex:7];
          [e setBuffer:out offset:0 atIndex:8];
          [e setBytes:&gp length:sizeof(gp) atIndex:9];
          [e setBuffer:MODE ? gst : dst offset:0 atIndex:10]; [e setBuffer:MODE ? ust : dst offset:0 atIndex:11];
          [e dispatchThreadgroups:MTLSizeMake(N / NT, U, 1) threadsPerThreadgroup:MTLSizeMake(SK * 32, 1, 1)];
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (t) best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / reps);
      }
      // correctness on the last plan used ((5*16+15) % NPLAN)
      const Plan *pl = (const Plan *)plans[(5 * 16 + 15) % NPLAN].contents;
      double maxerr = 0, maxref = 0;
      for (int trial = 0; trial < 64; ++trial) {
        const uint32_t u = rng() % U, col = rng() % N;
        uint32_t m = 0; while (!(pl->row_mask[u] & (1u << m))) ++m;
        const uint32_t ex = pl->experts[u];
        auto dot = [&](id<MTLBuffer> w, id<MTLBuffer> s, id<MTLBuffer> b) {
          const uint8_t *wp = (const uint8_t *)w.contents + (uint64_t(ex) * N + col) * (K / 2);
          const uint16_t *sp = (const uint16_t *)s.contents + (uint64_t(ex) * N + col) * (K / 64);
          const uint16_t *bp = (const uint16_t *)b.contents + (uint64_t(ex) * N + col) * (K / 64);
          const uint16_t *xp = MODE ? (const uint16_t *)x.contents + m * 2560 : (const uint16_t *)xi.contents + (uint64_t(u) * M + m) * 640;
          double acc = 0;
          for (uint32_t k = 0; k < K; ++k) {
            const uint32_t code = (wp[k / 2] >> ((k & 1) * 4)) & 15;
            acc += double(bf2f(xp[k])) * (double(bf2f(sp[k / 64])) * code + double(bf2f(bp[k / 64])));
          }
          return acc;
        };
        double ref;
        if (MODE) {
          const float g = bf2f(f2bf(float(dot(gw, gs, gb)))), up = bf2f(f2bf(float(dot(uw, us, ub))));
          const float sig = 1.f / (1.f + std::exp(-g));
          ref = double(g * sig * up);
        } else {
          ref = dot(dw, ds, db);
        }
        const double got = bf2f(((const uint16_t *)out.contents)[(uint64_t(u) * M + m) * N + col]);
        maxerr = std::max(maxerr, std::fabs(got - ref)); maxref = std::max(maxref, std::fabs(ref));
      }
      const double bytes = double(U) * N * K * (0.5 + 4.0 / 64) * (MODE ? 2 : 1);
      printf("%-36s R=%u U=%u %8.2f us %7.1f GB/s  relerr %.2e\n", n.c_str(), R, U, best * 1e6, bytes / best / 1e9, maxerr / std::max(maxref, 1e-9));
    }
  }
  return 0;
}
