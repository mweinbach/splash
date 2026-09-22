This exact scheduling candidate merges adjacent GDN A/B projection launches.
The existing source has two independent K2560→N48 QMV projections using the
same BF16 mixed input. Each source uses Q5/G128 or Q6/G64. Both new planes call
the unchanged original helper with original indexing, FP32 prescaling/XSUM,
group qdot accumulation, SIMD sum and final BF16 cast. No numerical split,
coefficient demotion, cache allocation or activation change is introduced.

The new grid is `{6,rows,2}`, with64threads. PlaneZ selects A/B source metadata,
weights and output; original helperZ is0. Both descriptors validate before any
plane reads/writes. Source ownership stays with existing immutable buffers.
The host first validates each original `addAffine` graph, then checks every
output/diagnostic/tap against input, all source views and other writable views,
before appending one pair dispatch. Eligible full target calls replace72
projection dispatches with36paired dispatches.

Root's actual-input component proof passed all six layer0/1/21 ×R1/R4 cases,
including Q6/Q6, Q5/Q5 and mixed Q6/Q5 formats. Full rawF32/BF16 words,
14guard/trap cases, poisoned replay and immutable inputs/sources passed.
Its warm matched component speedups were1.129–1.452×. That report is
`build/release/flash/sep21-gdn-ab-merge-actual-input-r1-r4-v1.json`.
This is component evidence; complete worker/model behavior remains pending.

CPU component preparation:

```sh
.venv/bin/python dev/benchmarks/gdn_ab_merge_sep21/prepare.py \
  --build build/gdn-ab-merge-sep21-oracle-v2
```

Only Root invokes GPU mode. Keep existing private allrows/SSD/certified store
flags and `SPLASH_FLASH_QMV_F32=1`:

```sh
FLASH_GDN_AB_LAYERS=0,1,21 FLASH_GDN_AB_ROWS=1,4 FLASH_GDN_AB_PAIRS=10 \
build/gdn-ab-merge-sep21-oracle-v2/oracle --gpu \
  build/gdn-ab-merge-sep21-oracle-v2/splash.metallib PACKAGE NEW_REPORT
```

Probe instrumentation adds only full rawF32 stores plus pointer plumbing;
its journal restores original byte/token-identical math bodies. Exact F32/BF16
gates precede at least150msGPU warming per route and even AB/BA timing, with no
CPU model-buffer access until all timing finishes. Both inputs/weights and all
guarded outputs/diagnostics remain checked. CPU compilation/tests touch no GPU
or model payloads.

The full worker uses strict `SPLASH_FLASH_GDN_AB_MERGE_SEP21=0/1`, default0 and
frozen before paths/backend. It selects only main Forward ordinary/verify
physical rows1/4 with qualified source tuples; other rows/formats retain the
original routes. Batch policy is unchanged. No public header or cache plan
changes, new GPU bytes or target numerical derivative change occurs. Static
flag/semantic status is in identity; mutable graph counts are top-level in
`gdn_ab_merge_route_counters`, preserving full benchmark identity checks.

```sh
.venv/bin/python dev/benchmarks/gdn_ab_merge_sep21/worker_prepare.py \
  --base build/prefill-qsa-twopass-sep21-worker-v1 \
  --build build/gdn-ab-merge-qsa-sep21-worker-v1
make -f dev/benchmarks/gdn_ab_merge_sep21/worker.mk \
  BUILD=build/gdn-ab-merge-qsa-sep21-worker-v1 -j4 cpu
.venv/bin/python dev/benchmarks/gdn_ab_merge_sep21/worker_witness.py \
  build/gdn-ab-merge-qsa-sep21-worker-v1
```

The earlier sealed HC-v3 fallback is
`build/gdn-ab-merge-hc-v3-sep21-worker-v1`. Root compares same-worker flag0/1
generation, full state/continuation, lifecycle, coding outputs and22-case
service semantics before promotion. Prefill4K and whole-model speed are not
inferred from this small component gain.
