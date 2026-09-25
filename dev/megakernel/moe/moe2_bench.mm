// Decode-window routed-expert benchmark: worker mk_moe_gate_up/down vs mk2 kernels.
//   ./moe2_bench R U
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t r = ((u >> 16) & 1) + 0x7fff; return uint16_t((u + r) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }

struct MkAffine { uint64_t wRow, wExpert, pRow, pExpert; uint32_t bits, group, pad0, pad1; };
struct MkMoEParams { uint32_t rows, pad0, pad1, pad2; MkAffine gate, up, down, sharedDown; };
struct MkMoEPlan { uint32_t rows, unique, pad0, pad1; uint32_t experts[80]; uint32_t routeOf[80][8]; uint32_t routeExpert[80]; float routeWeight[80]; };
struct Mk2Plan { uint32_t rows, unique, pad0, pad1; uint32_t experts[80]; };
struct Mk2Params { uint32_t rows, pad0, pad1, pad2; uint64_t codeExpertStride, paramExpertStride; };

static void tileExpert(const uint8_t *src, uint8_t *dst, uint32_t N, uint32_t K) {
  const uint32_t tiles = K / 64, rowBytes = K / 2;
  memset(dst, 0, uint64_t(N) * K / 2);
  for (uint32_t nt = 0; nt < N / 32; ++nt)
    for (uint32_t kt = 0; kt < tiles; ++kt) {
      uint8_t *t = dst + (uint64_t(nt) * tiles + kt) * 1024;
      for (uint32_t L = 0; L < 32; ++L) {
        const uint32_t kL = 4 * (L & 1) + 8 * ((L >> 3) & 1), nL = ((L >> 1) & 3) + 4 * ((L >> 4) & 1);
        for (uint32_t e = 0; e < 64; ++e) {
          const uint32_t k = kt * 64 + kL + 16 * (e >> 4) + (e & 3), n = nt * 32 + nL + 8 * ((e >> 2) & 3);
          const uint32_t c = (src[uint64_t(n) * rowBytes + k / 2] >> ((k & 1) * 4)) & 15;
          t[L * 32 + 4 * (e / 8) + (e & 3)] |= uint8_t(c << (4 * ((e / 4) & 1)));
        }
      }
    }
}

int main(int argc, char **argv) {
  @autoreleasepool {
    const uint32_t R = argc > 1 ? atoi(argv[1]) : 5, U = argc > 2 ? atoi(argv[2]) : 35;
    const char *filter = argc > 3 ? argv[3] : "";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"moe2.metallib"] error:&err];
    if (!lib) { printf("lib %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    const uint32_t E = 512, LAYERS = 2;
    struct Mat { id<MTLBuffer> w, s, b, tiled, params; uint32_t N, K; };
    auto make = [&](uint32_t N, uint32_t K) {
      Mat m; m.N = N; m.K = K;
      const uint64_t wbytes = uint64_t(E) * N * K / 2, pcount = uint64_t(E) * N * (K / 64);
      m.w = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
      m.s = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      m.b = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      m.tiled = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
      m.params = [dev newBufferWithLength:pcount * 4 options:MTLResourceStorageModeShared];
      uint32_t *wp = (uint32_t *)m.w.contents;
      for (uint64_t i = 0; i < wbytes / 4; ++i) wp[i] = rng();
      uint16_t *sp = (uint16_t *)m.s.contents, *bp = (uint16_t *)m.b.contents, *pp = (uint16_t *)m.params.contents;
      for (uint64_t i = 0; i < pcount; ++i) { sp[i] = f2bf(0.002f * (1.f + 0.2f * std::fabs(nd(rng)))); bp[i] = f2bf(-0.01f + 0.002f * nd(rng)); }
      const uint32_t ng = K / 64;
      dispatch_apply(E, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t e) {
        tileExpert((const uint8_t *)m.w.contents + e * N * K / 2, (uint8_t *)m.tiled.contents + e * N * K / 2, N, K);
        for (uint32_t n = 0; n < N; ++n)
          for (uint32_t g = 0; g < ng; ++g) {
            pp[e * 2 * N * ng + uint64_t(g) * N + n] = sp[(e * N + n) * ng + g];
            pp[e * 2 * N * ng + uint64_t(ng + g) * N + n] = bp[(e * N + n) * ng + g];
          }
      });
      return m;
    };
    std::vector<Mat> G, Up, D;
    for (uint32_t l = 0; l < LAYERS; ++l) { G.push_back(make(640, 2560)); Up.push_back(make(640, 2560)); D.push_back(make(2560, 640)); }
    id<MTLBuffer> x = [dev newBufferWithLength:16 * 2560 * 2 options:MTLResourceStorageModeShared];
    memset(x.contents, 0, 16 * 2560 * 2);
    for (uint32_t i = 0; i < R * 2560; ++i) ((uint16_t *)x.contents)[i] = f2bf(nd(rng) * 0.5f);
    id<MTLBuffer> xi16 = [dev newBufferWithLength:80 * 16 * 640 * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> xi8 = [dev newBufferWithLength:80 * 8 * 640 * 2 options:MTLResourceStorageModeShared];
    memset(xi16.contents, 0, xi16.length);
    for (uint32_t u = 0; u < 80; ++u)
      for (uint32_t m = 0; m < 8; ++m)
        for (uint32_t k = 0; k < 640; ++k) {
          const uint16_t v = m < R ? f2bf(nd(rng) * 0.1f) : 0;
          ((uint16_t *)xi16.contents)[(u * 16 + m) * 640 + k] = v;
          ((uint16_t *)xi8.contents)[(u * 8 + m) * 640 + k] = v;
        }
    id<MTLBuffer> out = [dev newBufferWithLength:80 * 16 * 2560 * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> sharedInter = [dev newBufferWithLength:16 * 640 * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> sharedDown = [dev newBufferWithLength:16 * 2560 * 2 options:MTLResourceStorageModeShared];
    const int NPLAN = 16;
    std::vector<id<MTLBuffer>> plans, plans2;
    std::vector<std::vector<uint32_t>> planIds;
    for (int k = 0; k < NPLAN; ++k) {
      std::vector<uint32_t> ids(E);
      for (uint32_t i = 0; i < E; ++i) ids[i] = i;
      std::shuffle(ids.begin(), ids.end(), rng);
      std::sort(ids.begin(), ids.begin() + U);
      MkMoEPlan p{}; p.rows = R; p.unique = U;
      Mk2Plan p2{}; p2.rows = R; p2.unique = U;
      for (uint32_t u = 0; u < U; ++u) {
        p.experts[u] = p2.experts[u] = ids[u];
        for (uint32_t m = 0; m < 8; ++m) p.routeOf[u][m] = m < R ? u * 8 + m : ~0u;
      }
      plans.push_back([dev newBufferWithBytes:&p length:sizeof p options:MTLResourceStorageModeShared]);
      plans2.push_back([dev newBufferWithBytes:&p2 length:sizeof p2 options:MTLResourceStorageModeShared]);
      planIds.push_back(std::vector<uint32_t>(ids.begin(), ids.begin() + U));
    }
    auto affine = [](uint32_t N, uint32_t K) {
      return MkAffine{K / 2, uint64_t(N) * K / 2, K / 64, uint64_t(N) * (K / 64), 4, 64, 0, 0};
    };
    MkMoEParams mp{R, 0, 0, 0, affine(640, 2560), affine(640, 2560), affine(2560, 640), affine(2560, 640)};
    // Warm the GPU clock.
    {
      id<MTLComputePipelineState> warm = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mk_moe_gate_up"] error:&err];
      const auto start = [NSDate date];
      while (-[start timeIntervalSinceNow] < 0.6) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:warm];
        for (int rep = 0; rep < 16; ++rep) {
          const Mat &g = G[rep % LAYERS], &u = Up[rep % LAYERS];
          [e setBuffer:x offset:0 atIndex:0];
          [e setBuffer:g.w offset:0 atIndex:1]; [e setBuffer:g.s offset:0 atIndex:2]; [e setBuffer:g.b offset:0 atIndex:3];
          [e setBuffer:u.w offset:0 atIndex:4]; [e setBuffer:u.s offset:0 atIndex:5]; [e setBuffer:u.b offset:0 atIndex:6];
          [e setBuffer:out offset:0 atIndex:7]; [e setBuffer:plans[rep % NPLAN] offset:0 atIndex:8];
          [e setBytes:&mp length:sizeof mp atIndex:9];
          [e dispatchThreadgroups:MTLSizeMake(20, U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      }
    }
    const double guBytes = double(U) * 2 * 640 * 2560 * (0.5 + 4.0 / 64), dnBytes = double(U) * 2560 * 640 * (0.5 + 4.0 / 64);
    for (NSString *fn in [lib.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
      const std::string name = fn.UTF8String;
      if (*filter && name.find(filter) == std::string::npos) continue;
      const bool base = name == "mk_moe_gate_up" || name == "mk_moe_down";
      const bool mk2 = name.rfind("mk2_", 0) == 0;
      if (!base && !mk2) continue;
      const bool gateUp = name.find("gate_up") != std::string::npos;
      uint32_t layout = 0, sk = 0;
      uint32_t depth = 1;
      if (mk2) sscanf(name.c_str() + name.find("_l") + 2, "%u_sk%u_d%u", &layout, &sk, &depth);
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:fn] error:&err];
      if (!pso) { printf("pso %s: %s\n", name.c_str(), err.localizedDescription.UTF8String); continue; }
      memset(out.contents, 0, out.length);
      double best = 1e9;
      const int reps = 32;
      for (int t = 0; t < 6; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [e setComputePipelineState:pso];
        for (int rep = 0; rep < reps; ++rep) {
          const Mat &g = G[rep % LAYERS], &u = Up[rep % LAYERS], &d = D[rep % LAYERS];
          const int pi = (t * reps + rep) % NPLAN;
          if (rep) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
          if (base && gateUp) {
            [e setBuffer:x offset:0 atIndex:0];
            [e setBuffer:g.w offset:0 atIndex:1]; [e setBuffer:g.s offset:0 atIndex:2]; [e setBuffer:g.b offset:0 atIndex:3];
            [e setBuffer:u.w offset:0 atIndex:4]; [e setBuffer:u.s offset:0 atIndex:5]; [e setBuffer:u.b offset:0 atIndex:6];
            [e setBuffer:out offset:0 atIndex:7]; [e setBuffer:plans[pi] offset:0 atIndex:8];
            [e setBytes:&mp length:sizeof mp atIndex:9];
            [e dispatchThreadgroups:MTLSizeMake(20, U, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          } else if (base) {
            [e setBuffer:xi16 offset:0 atIndex:0]; [e setBuffer:sharedInter offset:0 atIndex:1];
            [e setBuffer:d.w offset:0 atIndex:2]; [e setBuffer:d.s offset:0 atIndex:3]; [e setBuffer:d.b offset:0 atIndex:4];
            [e setBuffer:d.w offset:0 atIndex:5]; [e setBuffer:d.s offset:0 atIndex:6]; [e setBuffer:d.b offset:0 atIndex:7];
            [e setBuffer:out offset:0 atIndex:8]; [e setBuffer:sharedDown offset:0 atIndex:9];
            [e setBuffer:plans[pi] offset:0 atIndex:10]; [e setBytes:&mp length:sizeof mp atIndex:11];
            [e dispatchThreadgroups:MTLSizeMake(80, U, 1) threadsPerThreadgroup:MTLSizeMake(160, 1, 1)];
          } else if (gateUp) {
            const Mk2Params p2{R, 0, 0, 0, uint64_t(640) * 2560 / 2, uint64_t(640) * 40 * 2};
            [e setBuffer:x offset:0 atIndex:0];
            [e setBuffer:(layout ? g.w : g.tiled) offset:0 atIndex:1]; [e setBuffer:(layout ? u.w : u.tiled) offset:0 atIndex:2];
            [e setBuffer:g.params offset:0 atIndex:3]; [e setBuffer:u.params offset:0 atIndex:4];
            [e setBuffer:out offset:0 atIndex:5]; [e setBuffer:plans2[pi] offset:0 atIndex:6];
            [e setBytes:&p2 length:sizeof p2 atIndex:7];
            [e dispatchThreadgroups:MTLSizeMake(20, U, 1) threadsPerThreadgroup:MTLSizeMake(32 * sk, 1, 1)];
          } else {
            const Mk2Params p2{R, 0, 0, 0, uint64_t(2560) * 640 / 2, uint64_t(2560) * 10 * 2};
            [e setBuffer:xi8 offset:0 atIndex:0];
            [e setBuffer:(layout ? d.w : d.tiled) offset:0 atIndex:1]; [e setBuffer:d.params offset:0 atIndex:2];
            [e setBuffer:out offset:0 atIndex:3]; [e setBuffer:plans2[pi] offset:0 atIndex:4];
            [e setBytes:&p2 length:sizeof p2 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(80, U, 1) threadsPerThreadgroup:MTLSizeMake(32 * sk, 1, 1)];
          }
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (t) best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / reps);
      }
      // Reference check for mk2 kernels (last dispatch: layer (reps-1)%LAYERS, plan (5*reps+reps-1)%NPLAN).
      double relerr = -1;
      if (mk2) {
        const int li = (reps - 1) % LAYERS, pi = (5 * reps + reps - 1) % NPLAN;
        double maxerr = 0, maxref = 0;
        for (int trial = 0; trial < 96; ++trial) {
          const uint32_t u = rng() % U, m = rng() % R, N = gateUp ? 640 : 2560, K = gateUp ? 2560 : 640;
          const uint32_t n = rng() % N, ex = planIds[pi][u];
          auto dot = [&](const Mat &mat, const uint16_t *xp) {
            const uint8_t *wp = (const uint8_t *)mat.w.contents + (uint64_t(ex) * N + n) * (K / 2);
            const uint16_t *sp = (const uint16_t *)mat.s.contents + (uint64_t(ex) * N + n) * (K / 64);
            const uint16_t *bp = (const uint16_t *)mat.b.contents + (uint64_t(ex) * N + n) * (K / 64);
            double acc = 0;
            for (uint32_t k = 0; k < K; ++k)
              acc += double(bf2f(xp[k])) * (double(bf2f(sp[k / 64])) * ((wp[k / 2] >> ((k & 1) * 4)) & 15) + double(bf2f(bp[k / 64])));
            return acc;
          };
          double ref;
          if (gateUp) {
            const uint16_t *xp = (const uint16_t *)x.contents + m * 2560;
            const float gv = bf2f(f2bf(float(dot(G[li], xp)))), uv = bf2f(f2bf(float(dot(Up[li], xp))));
            ref = double(gv / (1.f + std::exp(-gv)) * uv);
          } else {
            ref = dot(D[li], (const uint16_t *)xi8.contents + (u * 8 + m) * 640);
          }
          const double got = bf2f(((const uint16_t *)out.contents)[(uint64_t(u) * 8 + m) * N + n]);
          maxerr = std::max(maxerr, std::fabs(got - ref)); maxref = std::max(maxref, std::fabs(ref));
        }
        relerr = maxerr / std::max(maxref, 1e-12);
      }
      const double bytes = gateUp ? guBytes : dnBytes;
      printf("  %-24s R=%u U=%u %8.2f us %7.1f GB/s  relerr %.1e\n", name.c_str(), R, U, best * 1e6, bytes / best / 1e9, relerr);
    }
  }
  return 0;
}
