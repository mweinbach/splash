# Private minimal Metal resource probe

This probe varies declared native buffer count and bytes while reading only one
uint32 per buffer. It loads only the original immutable `FlashWeights`, with no
target, trained head, saved operands, speculative scratch, request state, or
residency registration. `tiny-only` and `anonymous4g` skip even the model loader.
It makes no production or launcher changes.

The backend reflects separate readonly argument-buffer layouts for 1, 4, and
21 sources. Each submitted argument buffer retains exactly its supplied source
allocations. The unchanged backend declares their complete native buffers using
`useResources(...Read)`. A four-byte shader read therefore still declares an
entire 5GB native source resource. The probe never binds unused original shards.

Every pipeline, source, output, graph, argument buffer, and CPU expected word is
prepared before the idle samples. Kernels execute one 32-thread group and write
only one word per source. Leading and trailing guards and every expected source
word must match. There are no weight scans, conversions, or shader matrix math.

Build and CPU qualification:

```sh
make -j5 -f dev/benchmarks/idle_minimal_resource_probe/Makefile all
build/idle-minimal-resource-probe-v1/oracle --cpu-self-test
```

Root runs all GPU jobs serially with model servers stopped. Choose fresh report
paths. The package argument is ignored when the selected mode skips the loader.

```sh
build/idle-minimal-resource-probe-v1/oracle \
  build/idle-minimal-resource-probe-v1/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/idle-root-original21-v1.json \
  --mode original21 --idle 9 --repeats 2 --tiny-wake
```

Use `tiny-only`, then `model-tiny`, then `original1`, `original4`, and
`original21`. `all` runs tiny and all three original geometries in one process;
default intervals are 1, 3, 6, and 9 seconds. Largest original shards are selected
for the 1/4 cases; the report records every original index and native size.
`--tiny-wake` adds a second idle trial followed by a tiny command immediately
before the large-resource command. `anonymous4g` allocates one owned 4GiB Shared
buffer but writes only its first word; this does not assume the driver physically
commits all 4GiB. `--hold-seconds 15` allows an Instruments attachment after the
probe prints `READY pid=...` and before GPU warmup. Selectors are bounded to at
most five intervals of 0..9 seconds, three repeats, and a 45-second startup hold.

Timing uses normal Command profiling without timestamp sampling, added encoder
boundaries, or per-dispatch replay. The existing backend's memory sampling stays
in place. Full command metadata is retained in both the report and accompanying
`.commands.jsonl` file. Shader execution uses `GPUStartTime`/`GPUEndTime`;
`kernelStartTime`/`kernelEndTime` concern the OS kernel and must not be described
as shader execution. The existing Mach-to-steady bridge converts the hardware
GPU clock before subtraction against host commit timestamps. Actual previous GPU
completion-to-current-commit duration is reported alongside requested sleep.
Host preparation, encoding, memory queries, callbacks, and OS-kernel timing
remain separate and should not be summed as nonoverlapping overhead.

A fast tiny-only and model-loaded tiny command followed by a slow original21
command indicates that global GPU power wake alone is insufficient. A fast tiny
wake followed by a slow original21 command is an even stronger control. A slow
tiny wake followed by a fast original21 command needs driver events to separate
global wake from shared driver initialization. Resource count and declared bytes
are confounded in these initial cases, so report both and use follow-up controls
before attributing a proportional size effect. Driver trace `WireMemory` spans
provide additional evidence but no causal proof by themselves.

The CPU self-test passed 199 checks. All five host objects were compiled freshly
with the current 200-byte `CommandTiming` ABI and Metal 4.1. The frozen build
manifest records source/object/artifact SHA256 values. The implementation agent
has executed zero GPU commands; the Root GPU reports are separate qualification.

## Root measurements from September 20

Root executed `v10-resource-tiny-only.json` and
`v10-resource-model-scaling.json`. The first completed ten exact guarded samples;
the second completed 25. The CPU summary is
`build/release/flash/v10-resource-root-cause-cpu-summary-v2.json`.

| Declared source geometry | Native bytes | After 3s idle | After 3s idle then tiny wake | Immediate same-source followup |
| --- | ---: | ---: | ---: | ---: |
| Tiny with model loaded, unbound | 16,384 | 9.115ms | — | 0.073ms |
| Original 1 base | 5,204,606,976 | 59.303ms | 65.403ms | 0.106–0.125ms |
| Original 4 bases | 20,775,960,576 | 175.299ms | 167.149ms | 0.107–0.112ms |
| Original 21 bases | 106,320,429,056 | 848.343ms | 854.810ms | 0.110–0.117ms |

Values are host commit-end to actual hardware GPU-start. Each geometry has one
idle trial and one tiny-wake trial in this screen, so these are observed values,
not a broad repeatability confidence interval. The tiny commands preceding the
1/4/21 cases waited 10.149/9.292/9.028ms respectively; the following large command
still waited even though actual GPU idle since tiny completion was only about
0.2ms. Tiny-only without any model mapping waited 9.341ms after 3s and 7.103ms
after 9s. Merely loading the original model did not turn tiny dispatches into a
large stall.

Large-resource shader work took 6.6–76.4 microseconds across this screen. The OS
kernel began within 7–33 microseconds of host commit, and its measured interval
covered 59.261/175.196/848.276ms for the direct idle cases. Application preparation,
encoding, and commit were collectively under 0.21ms in those samples; individual
device-memory queries took at most 0.9 microseconds. The delay therefore occurred
below application encoding and before hardware GPU execution, and global GPU
wake alone is insufficient to explain it. Resource bytes and count remain
confounded. A matching minimal-probe driver trace is still required to connect
the scaling to specific memory preparation events. These results do not establish
a production HTTP fix or quantify full-model prefill latency.
