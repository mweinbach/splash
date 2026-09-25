#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

static id<MTLComputePipelineState> pso(id<MTLDevice> dev, id<MTLLibrary> lib, const char *name) {
  NSError *err = nil;
  id<MTLFunction> f = [lib newFunctionWithName:@(name)];
  if (!f) { fprintf(stderr, "missing %s\n", name); exit(1); }
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
  if (!p) { fprintf(stderr, "pso %s: %s\n", name, err.localizedDescription.UTF8String); exit(1); }
  return p;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"probe.metallib"] error:&err];
    if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    const char *which = argc > 1 ? argv[1] : "all";
    printf("device %s maxThreadsPerTG %lu recommendedWorkingSet %.1f GB\n", dev.name.UTF8String,
           (unsigned long)dev.maxThreadsPerThreadgroup.width, dev.recommendedMaxWorkingSetSize / 1e9);

    if (!strcmp(which, "all") || !strcmp(which, "bw")) {
      const uint64_t bytes = 4ull << 30;
      id<MTLBuffer> src = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
      id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b fillBuffer:src range:NSMakeRange(0, bytes) value:3];
        [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
      }
      for (const char *k : {"bw_read", "bw_read_chunk"}) {
        id<MTLComputePipelineState> p = pso(dev, lib, k);
        for (uint64_t sz : {16ull << 20, 64ull << 20, 256ull << 20, 1ull << 30, 4ull << 30}) {
          for (uint groups : {80u, 160u, 320u, 640u, 1280u, 2560u}) {
            for (uint tsz : {256u, 1024u}) {
              uint count = uint(sz / 16);
              double best = 1e9;
              for (int t = 0; t < 5; ++t) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p];
                // Rotate offsets so small sizes do not sit in cache across trials.
                uint64_t off = sz < (1ull << 30) ? (uint64_t(t) * sz) % (bytes - sz) : 0;
                [e setBuffer:src offset:off atIndex:0];
                [e setBuffer:out offset:0 atIndex:1];
                [e setBytes:&count length:4 atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tsz, 1, 1)];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                if (t) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
              }
              printf("%-14s size %6.0f MB groups %5u tsz %4u: %8.1f GB/s (%.1f us)\n", k, sz / 1e6, groups, tsz,
                     sz / best / 1e9, best * 1e6);
            }
          }
        }
      }
    }

    if (!strcmp(which, "storage")) {
      const uint64_t size = 4ull << 30;
      const char *path = "/Users/mweinbach/Projects/splash/install/local-models/Flash-Next-oQ4e-mtp-v1/weights/model-00010-of-00021.bin";
      int fd = open(path, O_RDONLY);
      void *base = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
      close(fd);
      volatile uint64_t sink = 0;
      for (uint64_t o = 0; o < size; o += 16384) sink += ((const uint8_t *)base)[o];
      id<MTLBuffer> bufs[4];
      bufs[0] = [dev newBufferWithLength:size options:MTLResourceStorageModePrivate];
      bufs[1] = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
      memset(bufs[1].contents, 1, size);
      bufs[2] = [dev newBufferWithBytesNoCopy:base length:size options:MTLResourceStorageModeShared deallocator:nil];
      bufs[3] = [dev newBufferWithLength:size options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
      memset(bufs[3].contents, 1, size);
      {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b fillBuffer:bufs[0] range:NSMakeRange(0, size) value:3];
        [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
      }
      const char *names[4] = {"private", "shared", "mmap", "shared-untracked"};
      id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      id<MTLComputePipelineState> p = pso(dev, lib, "bw_read_chunk");
      for (int round = 0; round < 2; ++round)
        for (int k = 0; k < 4; ++k) {
          for (uint64_t sz : {2ull << 20, 17ull << 20}) {
            double best = 1e9;
            uint64_t cursor = 0;
            const int n = 48;
            for (int t = 0; t < 3; ++t) {
              id<MTLCommandBuffer> cb = [q commandBuffer];
              id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
              [e setComputePipelineState:p];
              uint count = uint(sz / 16);
              [e setBytes:&count length:4 atIndex:2];
              [e setBuffer:out offset:0 atIndex:1];
              for (int i = 0; i < n; ++i) {
                cursor = (cursor + sz + 16384) % (size - sz);
                cursor &= ~uint64_t(16383);
                [e setBuffer:bufs[k] offset:cursor atIndex:0];
                if (i) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [e dispatchThreadgroups:MTLSizeMake(640, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
              }
              [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
              best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
            }
            printf("round %d %-17s %4.0f MB x48 serialized: %7.2f us/dispatch %6.1f GB/s\n", round, names[k], sz / 1e6,
                   best * 1e6 / n, sz * n / best / 1e9);
          }
        }
      (void)sink;
    }

    if (!strcmp(which, "mmap")) {
      // Cold reads from a no-copy buffer over an mmap'd package shard vs a Metal buffer.
      const char *path = argc > 2 ? argv[2] : "/Users/mweinbach/Projects/splash/install/local-models/Flash-Next-oQ4e-mtp-v1/weights/model-00009-of-00021.bin";
      int fd = open(path, O_RDONLY);
      const off_t size = lseek(fd, 0, SEEK_END);
      void *base = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
      close(fd);
      id<MTLBuffer> file = [dev newBufferWithBytesNoCopy:base length:size options:MTLResourceStorageModeShared deallocator:nil];
      // Touch every page on the CPU so the file is resident in the page cache.
      volatile uint64_t sink = 0;
      for (off_t o = 0; o < size; o += 16384) sink += ((const uint8_t *)base)[o];
      id<MTLBuffer> metal = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
      memcpy(metal.contents, base, size);
      id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      id<MTLComputePipelineState> p = pso(dev, lib, "bw_read_chunk");
      for (int which2 = 0; which2 < 2; ++which2) {
        id<MTLBuffer> src = which2 ? metal : file;
        for (uint64_t sz : {2ull << 20, 16ull << 20, 64ull << 20}) {
          for (int rep = 0; rep < 2; ++rep) {
            double best = 1e9;
            uint64_t cursor = 0;
            const int n = 48;
            for (int t = 0; t < 3; ++t) {
              id<MTLCommandBuffer> cb = [q commandBuffer];
              id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
              [e setComputePipelineState:p];
              uint count = uint(sz / 16);
              [e setBytes:&count length:4 atIndex:2];
              [e setBuffer:out offset:0 atIndex:1];
              for (int i = 0; i < n; ++i) {
                cursor = (cursor + sz + 16384) % (uint64_t(size) - sz);
                cursor &= ~uint64_t(16383);
                [e setBuffer:src offset:cursor atIndex:0];
                if (i) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [e dispatchThreadgroups:MTLSizeMake(640, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
              }
              [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
              best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
            }
            printf("%-6s %4.0f MB x48 serialized: %7.2f us/dispatch %6.1f GB/s\n", which2 ? "metal" : "mmap", sz / 1e6,
                   best * 1e6 / n, sz * n / best / 1e9);
          }
        }
      }
      (void)sink;
    }

    if (!strcmp(which, "all") || !strcmp(which, "cold")) {
      // 48 dependent-free dispatches, each reading a distinct cold region of `sz`.
      const uint64_t bytes = 8ull << 30;
      id<MTLBuffer> src = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
      id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b fillBuffer:src range:NSMakeRange(0, bytes) value:3];
        [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
      }
      id<MTLComputePipelineState> p = pso(dev, lib, "bw_read_chunk");
      uint64_t cursor = 0;
      for (uint64_t sz : {2ull << 20, 8ull << 20, 16ull << 20, 32ull << 20, 128ull << 20}) {
        for (uint groups : {160u, 640u, 1280u}) {
          for (int barrier = 0; barrier < 2; ++barrier) {
            double best = 1e9;
            const int n = 48;
            for (int t = 0; t < 3; ++t) {
              id<MTLCommandBuffer> cb = [q commandBuffer];
              id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
              [e setComputePipelineState:p];
              uint count = uint(sz / 16);
              [e setBytes:&count length:4 atIndex:2];
              [e setBuffer:out offset:0 atIndex:1];
              for (int i = 0; i < n; ++i) {
                cursor = (cursor + sz + 4096) % (bytes - sz);
                cursor &= ~uint64_t(255);
                [e setBuffer:src offset:cursor atIndex:0];
                if (barrier && i) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
              }
              [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
              best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
            }
            printf("cold %4.0f MB x48 groups %4u %-10s: %7.2f us/dispatch  %6.1f GB/s\n", sz / 1e6, groups,
                   barrier ? "serialized" : "overlap", best * 1e6 / n, sz * n / best / 1e9);
          }
        }
      }
    }

    if (!strcmp(which, "all") || !strcmp(which, "bcast")) {
      id<MTLBuffer> src = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModePrivate];
      id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      id<MTLComputePipelineState> p = pso(dev, lib, "bw_bcast");
      for (uint kb : {16u, 32u, 64u, 100u, 200u}) {
        for (uint groups : {40u, 80u, 160u, 320u}) {
          uint count = kb * 1024 / 16;
          double best = 1e9;
          for (int t = 0; t < 5; ++t) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p];
            [e setBuffer:src offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e setBytes:&count length:4 atIndex:2];
            for (int rep = 0; rep < 20; ++rep)
              [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (t) best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / 20);
          }
          printf("bcast %4u KB x %3u groups: %.2f us  aggregate %.0f GB/s\n", kb, groups, best * 1e6,
                 double(kb) * 1024 * groups / best / 1e9);
        }
      }
    }

    if (!strcmp(which, "all") || !strcmp(which, "disp")) {
      id<MTLComputePipelineState> p = pso(dev, lib, "chain_step");
      const uint n = 10240;
      id<MTLBuffer> a = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> b = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
      for (uint groups : {1u, 40u, 160u}) {
        for (int mode = 0; mode < 2; ++mode) {
          double best = 1e9;
          const int N = 1000;
          for (int t = 0; t < 4; ++t) {
            memset(a.contents, 0, n * 4);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:mode == 0 ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent];
            [e setComputePipelineState:p];
            [e setBytes:&n length:4 atIndex:2];
            for (int i = 0; i < N; ++i) {
              [e setBuffer:(i & 1) ? b : a offset:0 atIndex:0];
              [e setBuffer:(i & 1) ? a : b offset:0 atIndex:1];
              if (mode == 1 && i) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
              uint tpg = std::max(1u, (n + groups - 1) / groups);
              tpg = std::min(1024u, (tpg + 31) / 32 * 32);
              [e dispatchThreadgroups:MTLSizeMake((n + tpg - 1) / tpg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            }
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (t) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
          }
          const uint *r = (const uint *)((N & 1) ? b.contents : a.contents);
          bool ok = true;
          for (uint i = 0; i < n; ++i) ok &= r[i] == (uint)N;
          printf("dispatch chain groups~%3u %-18s %.2f us/dispatch %s\n", groups, mode ? "concurrent+barrier" : "serial",
                 best * 1e6 / N, ok ? "OK" : "WRONG");
        }
      }
    }

    if (!strcmp(which, "all") || !strcmp(which, "bar2")) {
      const uint n = 10240;
      id<MTLBuffer> buf = [dev newBufferWithLength:2 * n * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> ctr = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      id<MTLBuffer> status = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      for (int kind = 0; kind < 4; ++kind) {
        id<MTLComputePipelineState> p = pso(dev, lib, kind == 3 ? "persistent_chain4" : kind == 2 ? "persistent_chain3" : "persistent_chain2");
        uint allFence = kind == 1;
        for (uint groups : {40u, 80u, 160u}) {
          for (uint tsz : {256u, 1024u}) {
            const uint phases = 2000;
            double best = 1e9;
            bool ok = true;
            uint failed = 0;
            for (int t = 0; t < 4; ++t) {
              memset(buf.contents, 0, 2 * n * 4);
              memset(ctr.contents, 0, 64);
              memset(status.contents, 0, 64);
              id<MTLCommandBuffer> cb = [q commandBuffer];
              id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
              [e setComputePipelineState:p];
              [e setBuffer:buf offset:0 atIndex:0];
              [e setBuffer:ctr offset:0 atIndex:1];
              [e setBytes:&n length:4 atIndex:2];
              [e setBytes:&phases length:4 atIndex:3];
              [e setBuffer:status offset:0 atIndex:4];
              [e setBytes:&allFence length:4 atIndex:5];
              [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tsz, 1, 1)];
              [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
              failed += ((uint *)status.contents)[0];
              if (t) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
              const uint *r = (const uint *)buf.contents + (phases & 1) * n;
              for (uint i = 0; i < n; ++i) ok &= r[i] == phases;
            }
            printf("release-barrier %-12s groups %3u tsz %4u: %.2f us/barrier %s%s\n",
                   kind == 0 ? "tid0-fence" : kind == 1 ? "all-fence" : kind == 2 ? "atomic-data" : "coherent", groups, tsz,
                   best * 1e6 / phases, ok ? "OK" : "WRONG", failed ? " TIMEOUT" : "");
          }
        }
      }
    }

    if (!strcmp(which, "all") || !strcmp(which, "bar")) {
      id<MTLComputePipelineState> pc = pso(dev, lib, "persistent_chain");
      id<MTLComputePipelineState> pb = pso(dev, lib, "persistent_barrier");
      printf("persistent pso maxThreads %lu execWidth %lu\n", (unsigned long)pc.maxTotalThreadsPerThreadgroup,
             (unsigned long)pc.threadExecutionWidth);
      const uint n = 10240;
      id<MTLBuffer> buf = [dev newBufferWithLength:2 * n * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> ctr = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      id<MTLBuffer> status = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      for (int kind = 0; kind < 2; ++kind) {
        for (uint groups : {80u, 160u, 240u, 320u}) {
          for (uint tsz : {256u, 512u, 1024u}) {
            if (groups * tsz > 80u * 2048u) continue;
            const uint phases = 2000;
            double best = 1e9;
            bool ok = true;
            uint failed = 0;
            for (int t = 0; t < 4; ++t) {
              memset(buf.contents, 0, 2 * n * 4);
              memset(ctr.contents, 0, 64);
              memset(status.contents, 0, 64);
              id<MTLCommandBuffer> cb = [q commandBuffer];
              id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
              [e setComputePipelineState:kind ? pb : pc];
              [e setBuffer:buf offset:0 atIndex:0];
              [e setBuffer:ctr offset:0 atIndex:1];
              [e setBytes:&n length:4 atIndex:2];
              [e setBytes:&phases length:4 atIndex:3];
              [e setBuffer:status offset:0 atIndex:4];
              [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tsz, 1, 1)];
              [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
              failed += ((uint *)status.contents)[0];
              if (t) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
              if (!kind) {
                const uint *r = (const uint *)buf.contents + (phases & 1) * n;
                for (uint i = 0; i < n; ++i) ok &= r[i] == phases;
              }
            }
            printf("persistent %-8s groups %3u tsz %4u: %.2f us/barrier %s%s\n", kind ? "barrier" : "chain", groups, tsz,
                   best * 1e6 / phases, kind ? "" : (ok ? "OK" : "WRONG"), failed ? " TIMEOUT" : "");
          }
        }
      }
    }
  }
  return 0;
}
