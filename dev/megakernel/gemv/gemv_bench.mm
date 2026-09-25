// Few-row quantized GEMV benchmark on cold weights (dependent dispatches).
//   ./gemv_bench BITS G N K R [pipeline-substring]
// Compares mk_qmv (planar 5/6-bit), opt_qmv and opt_mppq (MLX layout).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Params { uint32_t K, N, rows, xStride, yStride, row0; uint64_t wRowStride, pRowStride; };
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t r = ((u >> 16) & 1) + 0x7fff; return uint16_t((u + r) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }

// MLX bitstream code k of a row.
static uint32_t mlxCode(const uint8_t *row, uint32_t k, uint32_t bits) {
  const uint64_t bit = uint64_t(k) * bits, byte = bit >> 3;
  uint32_t v = row[byte] | (uint32_t(row[byte + 1]) << 8);
  return (v >> (bit & 7)) & ((1u << bits) - 1u);
}

// Planar repack of one row (5/6-bit); identity for 4/8.
static void planarRow(const uint8_t *src, uint8_t *dst, uint32_t K, uint32_t bits, uint32_t G) {
  const uint32_t gbytes = G * bits / 8;
  for (uint32_t g = 0; g < K / G; ++g) {
    uint8_t *grp = dst + g * gbytes;
    memset(grp, 0, gbytes);
    for (uint32_t j = 0; j < G; ++j) {
      const uint32_t c = mlxCode(src, g * G + j, bits);
      grp[j / 2] |= uint8_t((c & 15u) << (4 * (j & 1)));
      if (bits == 5) grp[G / 2 + j / 8] |= uint8_t(((c >> 4) & 1u) << (j % 8));
      else grp[G / 2 + j / 4] |= uint8_t(((c >> 4) & 3u) << (2 * (j % 4)));
    }
  }
}

// Lane-major 32x64 tiles for mk_mpt (see mk_mpt.metal).
static void tileRepack(const uint8_t *src, uint8_t *dst, uint32_t N, uint32_t K, uint32_t bits) {
  const uint64_t rowBytes = uint64_t(K) * bits / 8;
  const uint32_t tiles = K / 64, tileBytes = 256 * bits;
  memset(dst, 0, uint64_t(N / 32) * tiles * tileBytes);
  for (uint32_t nt = 0; nt < N / 32; ++nt)
    for (uint32_t kt = 0; kt < tiles; ++kt) {
      uint8_t *t = dst + (uint64_t(nt) * tiles + kt) * tileBytes;
      for (uint32_t L = 0; L < 32; ++L) {
        const uint32_t kL = 4 * (L & 1) + 8 * ((L >> 3) & 1), nL = ((L >> 1) & 3) + 4 * ((L >> 4) & 1);
        for (uint32_t e = 0; e < 64; ++e) {
          const uint32_t j = e & 3, nb = (e >> 2) & 3, kb = e >> 4;
          const uint32_t k = kt * 64 + kL + 16 * kb + j, n = nt * 32 + nL + 8 * nb;
          const uint32_t c = mlxCode(src + uint64_t(n) * rowBytes, k, bits);
          if (bits == 8) { t[L * 64 + e] = uint8_t(c); continue; }
          const uint32_t u = e / 8, h = (e / 4) & 1, b = e & 3;
          t[L * 32 + 4 * u + b] |= uint8_t((c & 15u) << (4 * h));
          if (bits == 5) t[1024 + L * 8 + e / 8] |= uint8_t(((c >> 4) & 1u) << (e % 8));
          if (bits == 6) t[1024 + L * 16 + e / 4] |= uint8_t(((c >> 4) & 3u) << (2 * (e % 4)));
        }
      }
    }
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 6) { printf("usage: gemv_bench BITS G N K R [filter]\n"); return 1; }
    const uint32_t bits = atoi(argv[1]), G = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), R = atoi(argv[5]);
    const char *filter = argc > 6 ? argv[6] : "";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    NSString *libPath = [[[NSString stringWithUTF8String:argv[0]] stringByDeletingLastPathComponent]
        stringByAppendingPathComponent:@"gemv.metallib"];
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&err];
    if (!lib) { printf("lib %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    const uint64_t rowBytes = uint64_t(K) * bits / 8, wbytes = rowBytes * N;
    const uint64_t pcount = uint64_t(N) * (K / G);
    const uint64_t layerBytes = wbytes + pcount * 4;
    const int L = int(std::max<uint64_t>(8, (768ull << 20) / layerBytes + 1));
    std::vector<id<MTLBuffer>> W, WP, S, B, ST, BT, WT;
    std::vector<uint8_t> planar(rowBytes);
    for (int l = 0; l < L; ++l) {
      id<MTLBuffer> w = [dev newBufferWithLength:wbytes + 64 options:MTLResourceStorageModeShared];
      uint32_t *wp = (uint32_t *)w.contents;
      for (uint64_t i = 0; i < (wbytes + 64) / 4; ++i) wp[i] = rng();
      id<MTLBuffer> wpl = w;
      if (bits == 5 || bits == 6) {
        wpl = [dev newBufferWithLength:wbytes + 64 options:MTLResourceStorageModeShared];
        for (uint32_t n = 0; n < N; ++n)
          planarRow((const uint8_t *)w.contents + n * rowBytes, (uint8_t *)wpl.contents + n * rowBytes, K, bits, G);
      }
      id<MTLBuffer> s = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> b = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      uint16_t *sp = (uint16_t *)s.contents, *bp = (uint16_t *)b.contents;
      const float scale = 0.02f / float(1u << bits);
      for (uint64_t i = 0; i < pcount; ++i) { sp[i] = f2bf(scale * (1.f + 0.2f * std::fabs(nd(rng)))); bp[i] = f2bf(-0.01f + 0.002f * nd(rng)); }
      id<MTLBuffer> st = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> bt = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      const uint32_t ng = K / G;
      for (uint32_t n = 0; n < N; ++n)
        for (uint32_t g = 0; g < ng; ++g) {
          ((uint16_t *)st.contents)[uint64_t(g) * N + n] = sp[uint64_t(n) * ng + g];
          ((uint16_t *)bt.contents)[uint64_t(g) * N + n] = bp[uint64_t(n) * ng + g];
        }
      id<MTLBuffer> wt = w;
      if (N % 32 == 0 && K % 64 == 0) {
        wt = [dev newBufferWithLength:wbytes + 64 options:MTLResourceStorageModeShared];
        tileRepack((const uint8_t *)w.contents, (uint8_t *)wt.contents, N, K, bits);
      }
      WT.push_back(wt);
      W.push_back(w); WP.push_back(wpl); S.push_back(s); B.push_back(b); ST.push_back(st); BT.push_back(bt);
    }
    id<MTLBuffer> x = [dev newBufferWithLength:16 * K * 2 options:MTLResourceStorageModeShared];
    memset(x.contents, 0, 16 * K * 2);
    for (uint32_t i = 0; i < R * K; ++i) ((uint16_t *)x.contents)[i] = f2bf(nd(rng));
    id<MTLBuffer> y = [dev newBufferWithLength:16 * N * 2 options:MTLResourceStorageModeShared];
    // Reference for the last layer.
    std::vector<double> ref(uint64_t(R) * N);
    {
      const uint8_t *wp = (const uint8_t *)W[L - 1].contents;
      const uint16_t *sp = (const uint16_t *)S[L - 1].contents, *bp = (const uint16_t *)B[L - 1].contents;
      const uint16_t *xp = (const uint16_t *)x.contents;
      for (uint32_t n = 0; n < N; ++n)
        for (uint32_t m = 0; m < R; ++m) {
          double acc = 0;
          for (uint32_t k = 0; k < K; ++k) {
            const uint64_t pi = uint64_t(n) * (K / G) + k / G;
            acc += double(bf2f(xp[m * K + k])) * (double(bf2f(sp[pi])) * mlxCode(wp + n * rowBytes, k, bits) + double(bf2f(bp[pi])));
          }
          ref[uint64_t(m) * N + n] = acc;
        }
    }
    printf("bits %u G %u N %u K %u R %u: %.2f MB/layer, %d layers\n", bits, G, N, K, R, layerBytes / 1e6, L);
    const std::string mkPrefix = "mk_qmv_b" + std::to_string(bits) + "_g" + std::to_string(G) + "_r";
    const std::string optPrefix = "opt_qmv_b" + std::to_string(bits) + "_g" + std::to_string(G) + "_r";
    const std::string mppPrefix = "opt_mppq_b" + std::to_string(bits) + "_g" + std::to_string(G) + "_";
    const std::string mpvPrefix = "mk_mpv_b" + std::to_string(bits) + "_g" + std::to_string(G) + "_";
    const std::string mptPrefix = "mk_mpt_b" + std::to_string(bits) + "_g" + std::to_string(G) + "_";
    struct Result { std::string name; double us, gbs, err; };
    {
      // Bring the GPU to its sustained clock before timing anything.
      id<MTLComputePipelineState> warm = nil;
      for (NSString *fn in lib.functionNames) {
        if ([fn hasPrefix:@"opt_qmv_b4_g64_r1_n4_sn1_sk4_v8"]) {
          warm = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:fn] error:&err];
          break;
        }
      }
      if (warm) {
        const Params wp{K, N, 1, K, N, 0, rowBytes, (K / G) * 2};
        const auto start = [NSDate date];
        while (-[start timeIntervalSinceNow] < 0.6) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:warm];
          [e setBuffer:x offset:0 atIndex:0];
          [e setBuffer:y offset:0 atIndex:4];
          [e setBytes:&wp length:sizeof wp atIndex:5];
          for (int l = 0; l < L; ++l) {
            [e setBuffer:W[l] offset:0 atIndex:1];
            [e setBuffer:S[l] offset:0 atIndex:2];
            [e setBuffer:B[l] offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake((N + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          }
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        }
      }
    }
    std::vector<Result> results;
    for (NSString *fn in [lib.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
      const std::string name = fn.UTF8String;
      if (*filter && name.find(filter) == std::string::npos) continue;
      uint32_t groups = 0, threads = 0, cap = 0, rn = 0, sn = 0, sk = 0, v = 0, nt = 0;
      bool planarLayout = false, transposed = false, tiled = false;
      if (name.rfind(mkPrefix, 0) == 0 &&
          sscanf(name.c_str() + mkPrefix.size(), "%u_n%u_sn%u_sk%u_c%u", &cap, &rn, &sn, &sk, &v) == 5) {
        if (cap != R) continue;
        groups = (N + rn * sn - 1) / (rn * sn); threads = 32 * sn * sk; planarLayout = true;
      } else if (name.rfind(optPrefix, 0) == 0 &&
                 sscanf(name.c_str() + optPrefix.size(), "%u_n%u_sn%u_sk%u_v%u", &cap, &rn, &sn, &sk, &v) == 5) {
        if (cap != R) continue;
        groups = (N + rn * sn - 1) / (rn * sn); threads = 32 * sn * sk;
      } else if (name.rfind(mpvPrefix, 0) == 0 && sscanf(name.c_str() + mpvPrefix.size(), "sk%u_v%u", &sk, &v) == 2) {
        if (N % 32 || R > 8) continue;
        groups = N / 32; threads = 32 * sk; transposed = true;
      } else if (name.rfind(mptPrefix, 0) == 0 && sscanf(name.c_str() + mptPrefix.size(), "sk%u_x%u", &sk, &v) == 2) {
        if (N % 32 || K % 64 || R > 8) continue;
        groups = N / 32; threads = 32 * sk; transposed = true; tiled = true;
      } else if (name.rfind(mppPrefix, 0) == 0 && sscanf(name.c_str() + mppPrefix.size(), "nt%u_sk%u", &nt, &sk) == 2) {
        if (N % nt || R < 2) continue;
        groups = N / nt; threads = 32 * sk;
      } else {
        continue;
      }
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:fn] error:&err];
      if (!pso) { printf("pso %s: %s\n", name.c_str(), err.localizedDescription.UTF8String); continue; }
      if (threads > pso.maxTotalThreadsPerThreadgroup) continue;
      const Params p{K, N, R, K, N, 0, rowBytes, (K / G) * 2};
      memset(y.contents, 0, 16 * N * 2);
      double best = 1e9;
      for (int t = 0; t < 8; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [e setComputePipelineState:pso];
        [e setBuffer:x offset:0 atIndex:0];
        [e setBuffer:y offset:0 atIndex:4];
        [e setBytes:&p length:sizeof p atIndex:5];
        for (int l = 0; l < L; ++l) {
          [e setBuffer:(tiled ? WT[l] : planarLayout ? WP[l] : W[l]) offset:0 atIndex:1];
          [e setBuffer:(transposed ? ST[l] : S[l]) offset:0 atIndex:2];
          [e setBuffer:(transposed ? BT[l] : B[l]) offset:0 atIndex:3];
          if (l) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (t) best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / L);
      }
      double maxerr = 0, maxref = 0;
      const uint16_t *yp = (const uint16_t *)y.contents;
      for (uint32_t m = 0; m < R; ++m)
        for (uint32_t n = 0; n < N; ++n) {
          const double r = ref[uint64_t(m) * N + n];
          maxerr = std::max(maxerr, std::fabs(double(bf2f(yp[uint64_t(m) * N + n])) - r));
          maxref = std::max(maxref, std::fabs(r));
        }
      results.push_back({name, best * 1e6, layerBytes / best / 1e9, maxerr / std::max(maxref, 1e-12)});
    }
    std::sort(results.begin(), results.end(), [](const Result &a, const Result &b) { return a.us < b.us; });
    for (const auto &r : results)
      printf("  %-40s %8.2f us %7.1f GB/s  relerr %.1e%s\n", r.name.c_str(), r.us, r.gbs, r.err, r.err > 2e-2 ? "  WRONG" : "");
  }
  return 0;
}
