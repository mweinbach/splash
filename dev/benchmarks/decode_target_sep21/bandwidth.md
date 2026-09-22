This standalone native oracle measures explicit memory-payload throughput using
the current core `MetalBackend` and fresh ABI200 objects. It does not load model
files. Root runs GPU work only after model unload, under its exclusive GPU slot.

Build and CPU self-test:

```sh
make -f dev/benchmarks/decode_target_sep21/Makefile -j4 all
build/sep21-memory-payload/bandwidth --cpu-self-test
```

The self-test passed197,714 coverage/checksum checks. An independent agent also
extracted those routines and ran them with no Metal/model APIs linked. Metal
and host compile with warnings as errors. Source-only preparation performed no
GPU/device work. The build provenance manifest lives at
`build/sep21-memory-payload/build-manifest.json`.

Root-only run, with a fresh output path:

```sh
build/sep21-memory-payload/bandwidth --gpu \
  build/sep21-memory-payload/bandwidth.metallib \
  build/release/flash/sep21-sustained-memory-payload-v1.json
```

Default tracked allocation is24 GiB plus917,632 bytes of guarded per-CTA result
storage, below32 GiB. There are two immutable8 GiB Shared sources with different
deterministic indexed uint32 patterns and one8 GiB Shared copy destination.
Read8 GiB alternates sources; read16 GiB alternates the order of both sources.
Copy reads8 GiB and writes8 GiB, alternating source into the separate destination.
`--buffer-gib 4` optionally selects4/8 GiB read scopes and4 GiB copy arrays.

Initialization runs on GPU outside timing and touches every source page. Each
case then has three complete GPU warmups and seven measured samples. Destination
copy warmups touch every output page. This excludes first pipeline creation,
source faults and initial output commitment from warmed samples. All seven
GPU and synchronous command-wall durations are retained separately.

Each CTA covers65,536 uint4 vectors with256 threads. Read results contain exact
64-bit sums for all four vector components, actual vectors counted, distinct
first/last word fingerprints and a per-command stamp; CPU validates every CTA
outside timing. Counts aggregate with uint64. Exact64 sums use an independent
floor-sum formula for the modulo32 indexed pattern. A modulo32 checksum alone
would repeat across these chunks and is deliberately insufficient.

Copy timing contains uint4 loads/stores and tiny actual-count/stamp records.
After every copy command, a separate untimed GPU scan compares **every copied
component** against the indexed source pattern, computes all CTA checksums,
counts and fingerprints, and checks canaries. Both source buffers receive
whole-word pattern validation before and after the experiment, establishing
unchanged contents. Shader guards fail closed unless SIMD width is32 and the
threadgroup contains256 threads.

Reports distinguish full data bytes read/written from read-fingerprint scalar
loads and56-byte result-record writes. Primary read throughput isdata bytes/GPU
seconds. Copy throughput counts read+write bytes; it must not be interpreted as
one-way read bandwidth. Hashes/CPU scans/validation dispatches lie outside all
timed commands. The exact64 reduction adds a small instruction cost to reads.

These tests establish a **measured sustained effective payload rate** for large
resident arrays, addressing the vocabulary primitive's cache/reread uncertainty.
They do not collect physical DRAM transaction counters, assign buffers to dies,
or prove that affine/MPP decode kernels can sustain the same rate.

After Root's measurement, refresh the roofline using the lower of the two read
medians. The measured copy rate is retained separately:

```sh
.venv/bin/python dev/benchmarks/decode_target_sep21/roofline.py \
  --bandwidth-report build/release/flash/sep21-sustained-memory-payload-v1.json \
  --out dev/benchmarks/decode_target_sep21/roofline-measured-sep21.json
```

Only a completed, guard/source-validated report below32 GiB is accepted. The
result remains a traffic scenario; measured kernel/acceptance targets govern
attainable generation expectations.
