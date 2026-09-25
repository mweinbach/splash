// Hyper-connection block benchmark: opt_hc_down + opt_hc_up_mix vs mk_hc_down + mk_hc_up.
//   ./hc_bench BITS R
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t r = ((u >> 16) & 1) + 0x7fff; return uint16_t((u + r) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }
static uint32_t mlxCode(const uint8_t *row, uint32_t k, uint32_t bits) {
  const uint64_t bit = uint64_t(k) * bits, byte = bit >> 3;
  uint32_t v = row[byte] | (uint32_t(row[byte + 1]) << 8);
  return (v >> (bit & 7)) & ((1u << bits) - 1u);
}

struct Mat { std::vector<uint8_t> w; std::vector<uint16_t> s, b; uint32_t N, K, bits; };
static Mat makeMat(std::mt19937 &rng, uint32_t N, uint32_t K, uint32_t bits) {
  std::normal_distribution<float> nd(0.f, 1.f);
  Mat m{std::vector<uint8_t>(uint64_t(N) * K * bits / 8 + 8), std::vector<uint16_t>(uint64_t(N) * (K / 64)),
        std::vector<uint16_t>(uint64_t(N) * (K / 64)), N, K, bits};
  for (auto &v : m.w) v = uint8_t(rng());
  const float scale = 0.05f / float(1u << bits);
  for (auto &v : m.s) v = f2bf(scale * (1.f + 0.2f * std::fabs(nd(rng))));
  for (auto &v : m.b) v = f2bf(-scale * float(1u << bits) / 2 + 0.002f * nd(rng));
  return m;
}
// Tiles (lane-major) + transposed params [g][NP] scales then biases, optional column map.
static void tile(const Mat &m, std::vector<uint8_t> &tiles, std::vector<uint16_t> &params, const std::vector<uint32_t> &map) {
  const uint32_t NP = uint32_t(map.size()), K = m.K, bits = m.bits, tilesK = K / 64, tb = 256 * bits, groups = K / 64;
  const uint64_t rowBytes = uint64_t(K) * bits / 8;
  tiles.assign(uint64_t(NP) * K * bits / 8, 0);
  params.assign(uint64_t(2) * NP * groups, 0);
  for (uint32_t nt = 0; nt < NP / 32; ++nt)
    for (uint32_t kt = 0; kt < tilesK; ++kt) {
      uint8_t *t = tiles.data() + (uint64_t(nt) * tilesK + kt) * tb;
      for (uint32_t L = 0; L < 32; ++L) {
        const uint32_t kL = 4 * (L & 1) + 8 * ((L >> 3) & 1), nL = ((L >> 1) & 3) + 4 * ((L >> 4) & 1);
        for (uint32_t e = 0; e < 64; ++e) {
          const uint32_t k = kt * 64 + kL + 16 * (e >> 4) + (e & 3), col = nt * 32 + nL + 8 * ((e >> 2) & 3);
          const uint32_t n = map[col];
          const uint32_t c = n < m.N ? mlxCode(m.w.data() + uint64_t(n) * rowBytes, k, bits) : 0;
          if (bits == 8) { t[L * 64 + e] = uint8_t(c); continue; }
          t[L * 32 + 4 * (e / 8) + (e & 3)] |= uint8_t((c & 15u) << (4 * ((e / 4) & 1)));
          if (bits == 5) t[1024 + L * 8 + e / 8] |= uint8_t(((c >> 4) & 1u) << (e % 8));
          if (bits == 6) t[1024 + L * 16 + e / 4] |= uint8_t(((c >> 4) & 3u) << (2 * (e % 4)));
        }
      }
    }
  for (uint32_t col = 0; col < NP; ++col) {
    const uint32_t n = map[col];
    if (n >= m.N) continue;
    for (uint32_t g = 0; g < groups; ++g) {
      params[uint64_t(g) * NP + col] = m.s[uint64_t(n) * groups + g];
      params[uint64_t(groups + g) * NP + col] = m.b[uint64_t(n) * groups + g];
    }
  }
}

struct Seg { uint32_t bits, group, paddedN, pad0; uint64_t tileOffset, paramOffset, biasOffset, pad1; };
struct DownP { uint32_t rows, splits, blocks, pad0; Seg down, inject; };
struct UpP { uint32_t rows, splits, hasInjection, pad0; Seg up; };
struct OptHCParams { uint32_t rows, hasInjection, injBits, injGroup; uint64_t downWRow, downPRow, injWRow, injPRow, upWRow, upPRow; uint32_t row0, reserved; };

int main(int argc, char **argv) {
  @autoreleasepool {
    const uint32_t bits = argc > 1 ? atoi(argv[1]) : 5, R = argc > 2 ? atoi(argv[2]) : 5;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"hc.metallib"] error:&err];
    if (!lib) { printf("lib %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto buf = [&](const void *p, uint64_t n) {
      id<MTLBuffer> b = [dev newBufferWithLength:std::max<uint64_t>(n, 16) options:MTLResourceStorageModeShared];
      if (p) memcpy(b.contents, p, n); else memset(b.contents, 0, b.length);
      return b;
    };
    const int L = 48;  // distinct HC blocks (cold weights)
    struct Block { id<MTLBuffer> dw, ds, db, iw, is, ib, uw, us, ub, dt, dp, ut, up; uint64_t injTileOffset, injParamOffset; };
    std::vector<Block> blocks;
    Mat keepDown, keepInj, keepUp;
    for (int l = 0; l < L; ++l) {
      Mat down = makeMat(rng, 320, 10240, bits), inj = makeMat(rng, 4, 10240, bits), up = makeMat(rng, 10240, 320, bits);
      std::vector<uint32_t> dmap(320), imap(32), umap(10240);
      for (uint32_t i = 0; i < 320; ++i) dmap[i] = i;
      for (uint32_t i = 0; i < 32; ++i) imap[i] = i < 4 ? i : ~0u;
      for (uint32_t c = 0; c < 10240; ++c) umap[c] = (c % 4) * 2560 + 8 * (c / 32) + (c % 32) / 4;
      std::vector<uint8_t> dt, it, ut; std::vector<uint16_t> dp, ip, up2;
      tile(down, dt, dp, dmap); tile(inj, it, ip, imap); tile(up, ut, up2, umap);
      Block b;
      b.injTileOffset = dt.size(); b.injParamOffset = dp.size();
      std::vector<uint8_t> gt(dt); gt.insert(gt.end(), it.begin(), it.end());
      std::vector<uint16_t> gp(dp); gp.insert(gp.end(), ip.begin(), ip.end());
      b.dw = buf(down.w.data(), down.w.size()); b.ds = buf(down.s.data(), down.s.size() * 2); b.db = buf(down.b.data(), down.b.size() * 2);
      b.iw = buf(inj.w.data(), inj.w.size()); b.is = buf(inj.s.data(), inj.s.size() * 2); b.ib = buf(inj.b.data(), inj.b.size() * 2);
      b.uw = buf(up.w.data(), up.w.size()); b.us = buf(up.s.data(), up.s.size() * 2); b.ub = buf(up.b.data(), up.b.size() * 2);
      b.dt = buf(gt.data(), gt.size()); b.dp = buf(gp.data(), gp.size() * 2);
      b.ut = buf(ut.data(), ut.size()); b.up = buf(up2.data(), up2.size() * 2);
      blocks.push_back(b);
    }
    std::vector<uint16_t> normv(uint64_t(8) * 10240, 0);
    for (uint32_t i = 0; i < R * 10240; ++i) normv[i] = f2bf(nd(rng));
    id<MTLBuffer> norm = buf(normv.data(), normv.size() * 2);
    id<MTLBuffer> act = buf(nullptr, 8 * 320 * 2), gatesA = buf(nullptr, 8 * 4 * 2), gatesB = buf(nullptr, 8 * 4 * 2);
    id<MTLBuffer> mixedA = buf(nullptr, 8 * 2560 * 2), mixedB = buf(nullptr, 8 * 2560 * 2), partial = buf(nullptr, 4 * 8 * 352 * 4);
    auto pso = [&](NSString *n) {
      id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&err];
      if (!p) { printf("pso %s: %s\n", n.UTF8String, err.localizedDescription.UTF8String); exit(1); }
      return p;
    };
    NSString *capName = [NSString stringWithFormat:@"%u", R];
    id<MTLComputePipelineState> od = pso([NSString stringWithFormat:@"opt_hc_down_b%u_g64_r%@", bits, capName]);
    id<MTLComputePipelineState> ou = pso([NSString stringWithFormat:@"opt_hc_up_mix_b%u_g64_r%@", bits, capName]);
    id<MTLComputePipelineState> md = pso(@"mk_hc_down_sk8"), mu = pso(@"mk_hc_up");
    const int only = argc > 3 ? atoi(argv[3]) : 0;  // 1: down only, 2: up only
    const uint64_t rowD = 10240 * bits / 8, rowU = 320 * bits / 8;
    const OptHCParams op{R, 1, bits, 64, rowD, 160 * 2, rowD, 160 * 2, rowU, 5 * 2, 0, 0};
    auto encodeOpt = [&](id<MTLComputeCommandEncoder> e, const Block &b) {
      [e setComputePipelineState:od];
      [e setBuffer:norm offset:0 atIndex:0];
      [e setBuffer:b.dw offset:0 atIndex:1]; [e setBuffer:b.ds offset:0 atIndex:2]; [e setBuffer:b.db offset:0 atIndex:3];
      [e setBuffer:b.iw offset:0 atIndex:4]; [e setBuffer:b.is offset:0 atIndex:5]; [e setBuffer:b.ib offset:0 atIndex:6];
      [e setBuffer:act offset:0 atIndex:7]; [e setBuffer:gatesA offset:0 atIndex:8]; [e setBytes:&op length:sizeof op atIndex:9];
      if (only != 2) [e dispatchThreadgroups:MTLSizeMake(161, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
      if (only == 1) return;
      [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [e setComputePipelineState:ou];
      [e setBuffer:norm offset:0 atIndex:0]; [e setBuffer:act offset:0 atIndex:1];
      [e setBuffer:b.uw offset:0 atIndex:2]; [e setBuffer:b.us offset:0 atIndex:3]; [e setBuffer:b.ub offset:0 atIndex:4];
      [e setBuffer:mixedA offset:0 atIndex:5]; [e setBytes:&op length:sizeof op atIndex:6];
      [e dispatchThreadgroups:MTLSizeMake(640, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };
    auto encodeMk = [&](id<MTLComputeCommandEncoder> e, const Block &b) {
      const DownP dp{R, 4, 11, 0, {bits, 64, 320, 0, 0, 0, 160ull * 320, 0},
                     {bits, 64, 32, 0, b.injTileOffset, b.injParamOffset, 160ull * 32, 0}};
      [e setComputePipelineState:md];
      [e setBuffer:norm offset:0 atIndex:0]; [e setBuffer:b.dt offset:0 atIndex:1]; [e setBuffer:b.dp offset:0 atIndex:2];
      [e setBuffer:partial offset:0 atIndex:3]; [e setBytes:&dp length:sizeof dp atIndex:4];
      if (only != 2) [e dispatchThreadgroups:MTLSizeMake(11, 4, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      if (only == 1) return;
      [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const UpP up{R, 4, 1, 0, {bits, 64, 10240, 0, 0, 0, 5ull * 10240, 0}};
      [e setComputePipelineState:mu];
      [e setBuffer:norm offset:0 atIndex:0]; [e setBuffer:partial offset:0 atIndex:1];
      [e setBuffer:b.ut offset:0 atIndex:2]; [e setBuffer:b.up offset:0 atIndex:3];
      [e setBuffer:mixedB offset:0 atIndex:4]; [e setBuffer:gatesB offset:0 atIndex:5]; [e setBytes:&up length:sizeof up atIndex:6];
      [e dispatchThreadgroups:MTLSizeMake(80, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    };
    auto time = [&](bool mk) {
      double best = 1e9;
      for (int t = 0; t < 8; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (int l = 0; l < L; ++l) {
          if (l) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
          if (mk) encodeMk(e, blocks[l]); else encodeOpt(e, blocks[l]);
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (t) best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / L);
      }
      return best;
    };
    for (int w = 0; w < 30; ++w) time(false);
    const double to = time(false), tm = time(true);
    printf("bits %u R %u: opt_hc down+up %.2f us, mk_hc down+up %.2f us\n", bits, R, to * 1e6, tm * 1e6);
    double maxd = 0, maxv = 0, gd = 0;
    for (uint32_t i = 0; i < R * 2560; ++i) {
      const double a = bf2f(((uint16_t *)mixedA.contents)[i]), b = bf2f(((uint16_t *)mixedB.contents)[i]);
      maxd = std::max(maxd, std::fabs(a - b)); maxv = std::max(maxv, std::fabs(a));
    }
    for (uint32_t i = 0; i < R * 4; ++i)
      gd = std::max(gd, std::fabs(double(bf2f(((uint16_t *)gatesA.contents)[i])) - bf2f(((uint16_t *)gatesB.contents)[i])));
    printf("  mixed max |diff| %.3e (max %.3e), gates max |diff| %.3e\n", maxd, maxv, gd);
  }
  return 0;
}
