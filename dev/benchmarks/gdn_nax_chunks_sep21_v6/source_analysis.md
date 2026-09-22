# Guarded v3 source cost audit for the v4 W-norm cache

CPU-only source audit. No kernel, oracle, sealed v3 worker, model payload, or GPU
state was changed. Root supplied v3 SG8 timing 1.651875 ms versus unguarded v2
1.180650 ms, with zero fallback heads. The difference is 0.471225 ms (+39.9123%).
These timing values were reported by Root; this audit does not independently
measure them or allocate the difference to phases. Every timing/occupancy/cache
explanation below is a **HYPOTHESIS**, not a measured attribution.

Counts use R=2048, B=1, H=48, T=32, V=32, G=8, K=128, full chunks, non-audit
entrypoints. Define P=H*ceil(R/T)=3072 head/chunks and A=P*(128/V)=12,288 apply
group/chunks. Apply has 192 threadgroups, each processing 64 chunks. Unless
otherwise stated, all apply groups pass their initial skip and execute all chunks.
Counts describe source operations/API calls, not compiler instructions, DRAM
transactions, hidden MPP scratch, or an occupancy result.

## Work present even when no head falls back

| Phase | Threadgroups | Threads/group | Thread invocations | Work at zero fallback |
| --- | ---: | ---: | ---: | --- |
| snapshot/flag init | 3072 | 256 | 786,432 | Read and save every recurrent uint word; 48 flag stores |
| preparation | 3072 | 256 | 786,432 | Full W/U/E/score/prefix preparation plus range selector |
| guarded apply | 192 | 256 | 49,152 | Four value tiles/head; 64 sequential chunks each |
| restore | 3072 | 256 | 786,432 | Every thread reads its head's atomic flag; no state copy |
| native replay | 384 | 512 | 196,608 | Every thread reads its head's atomic flag and returns |
| **total** | **9792** | mixed | **2,605,056** | Five compute dispatches |

Unguarded v2 has preparation+apply only: 3264 threadgroups, 835,584 thread
invocations, two dispatches. Guarding adds three dispatches and 1,769,472 thread
invocations even when all flags stay zero. Source anchors:
`v3/metal_oracle.mm:294-357`, `v3/snapshot.metal:18-25`,
`v3/native_fallback.metal:114-123` (paths are under `dev/benchmarks/`).

## Selector arithmetic and reductions

State norms already run once/value row/chunk. W norms run once/token/chunk in
each of the four apply value tiles. Moving W norms to preparation saves three
of those four copies; it does not change S norms or the delta selector.

| Source-level selector count | v3 | Once/head/chunk cached W norm |
| --- | ---: | ---: |
| S norms | A*V = 393,216 | 393,216 |
| W norms | A*T = 393,216 | P*T = 98,304 |
| Norm square multiplies, and local `sum +=` additions, each | 100,663,296 | 62,914,560 |
| Norm square roots, and inflate multiplies, each | 786,432 | 491,520 |
| Delta conditioning entries | A*V*T = 12,582,912 | unchanged |
| Cross-SIMD leader aggregation additions | 2*G*A = 196,608 | unchanged |
| **Selector multiply subtotal** | **139,198,464** | **101,154,816** |
| **Selector addition subtotal** | **138,608,640** | **100,859,904** |
| Selector SIMD reduction collectives | 983,040 | 688,128 |
| Lane invocations of these collectives | 31,457,280 | 22,020,096 |

Each 128-key norm executes 128 square multiplies and 128 source additions, one
`simd_sum` collective (32 lane calls), and one lane-zero sqrt/inflation multiply.
Each delta conditioning entry adds one `abs`, three multiplies and three
additions; existing `u-sw` subtraction is excluded. Delta condition `abs` count
is 12,582,912. The threshold `0.095f*sumC` adds 0..12,288 data-dependent
multiplies. SIMD reduction lowering is compiler-dependent and is not counted as
an assumed 31 scalar additions here.

W caching removes 294,912 norms: 37,748,736 square multiplies and additions each,
294,912 sqrt/inflation pairs and 294,912 collectives. It reduces norm count 37.5%
and the complete listed selector multiply/add subtotal about 27.3%. Two cached
scalar slots/token do not imply two norms/token. Preserve the original four-key
lane accumulation, SIMD reduction, sqrt, inflation, raw-word eligibility tests,
and sticky reason merge semantics. Anchors: `v3/candidate.metal:227-255`
(norms), `:269-295` (conditioning and serial cross-SIMD leader aggregation).

Cache correctness point: if preparation rereads the device W plane to compute
the cached norms, the cooperative W stores need device-memory visibility before
that cross-lane read. V3's existing post-W barrier at `candidate.metal:149` only
names `mem_threadgroup`; include `mem_device` for the new device reread unless a
separately proven register/shared path avoids it. Preserve identical norm input
words and scalar accumulation order before making timing comparisons.

Preparation's range selector separately executes T(T+1)/2=528 prefix scan
iterations/head/chunk, 1,622,016 globally. `product *= alpha` belongs to the
existing transform. The added `segment *= abs(alpha)` contributes 0..1,622,016
extra multiplies and `abs` calls; the upper bound applies if every alpha is
nonzero. Actual zero resets the segment. Anchor: `v3/candidate.metal:72-94`.

For scale, unchanged nominal matrix multiply work is:

| Phase | MPP calls | Multiply terms from declared dimensions | Nominal 2*m*n*k FLOPs |
| --- | ---: | ---: | ---: |
| preparation: KK, W, U, QK | 4*P = 12,288 | 1,610,612,736 | 3,221,225,472 |
| apply: SW, SQ, score*D, D*E | 4*A = 49,152 | 5,234,491,392 | 10,468,982,784 |

The scalar triangular inverse also remains unchanged: P*T*T*(T-1)/2 =
48,758,784 multiply/subtract pairs. MPP descriptors and implementation lowering
may execute padded/internal work; the dimension counts are not instruction
counts. Anchors: `v3/candidate.metal:103-162,124-129,209-217,258-352`.

## Synchronization and atomic source counts

Preparation has seven explicit threadgroup barriers/head/chunk, as in v2:
7*P=21,504. Apply has one initial skip barrier/group plus five/chunk:
192+5*A=61,632. V2 has three/chunk: 3*A=36,864. The guard adds **24,768 explicit
threadgroup barrier instances**. Total v3 preparation+apply is 83,136 versus
58,368 in v2. These count explicit source barriers only; internal MPP
synchronization is excluded. W caching still needs the S-norm-to-conditioning
barrier and does not automatically remove either extra per-chunk guard barrier.
Anchors: `v3/candidate.metal:71,114,122,132,135,149,152,201,257,283,310,347,363`.

Standalone oracle encodes four buffer barriers: after snapshot, preparation,
apply and restore (`v3/metal_oracle.mm:299,312,339,346`). V2 has one after
preparation. The sealed worker adapter has five, including after native replay;
its next output stage consumes recurrence rows. Do not equate oracle and worker
barrier counts or use either count as a duration.

At zero fallback, exact source atomic operations are:

| Operation | Count |
| --- | ---: |
| snapshot flag stores | 48 |
| apply initial flag loads | 192 |
| restore flag loads | 48*64*256 = 786,432 |
| replay flag loads | 48*8*512 = 196,608 |
| **flag loads total** | **983,232** |
| conditional guard ORs | **0** when all final sticky flags are zero |

The restore/replay loads are issued per thread in the source, not once/group.
Their 3,932,928 issued bytes hit only a 192-byte flag plane; actual compiler
coalescing, atomic serialization, and cache traffic are unknown. Snapshot adds
6,291,456 issued state/snapshot bytes plus 192 flag-store bytes even when replay
does no work. Anchors: `v3/snapshot.metal:22-25`,
`v3/native_fallback.metal:119`, `v3/candidate.metal:200`.

There are five prep and ten apply conditional flag-OR sites. Conservative source
upper bounds are prep P*[2*T+3*T*(T+1)/2]=5,062,656 and admitted apply
A*[2*(V+T)*(128+32)+2*V*T+1+V*T+128*V]=213,921,792. Those are separate bounds,
not an attainable sum: prep rejection skips apply; global norm and cancellation
OR branches are mutually exclusive. Invalid-geometry diagnostic ORs are excluded
from the valid benchmark. Zero-fallback timing cannot be blamed on guard OR
contention, but still includes guard tests and unconditional flag loads.

## Storage and issued scalar traffic

| Logical device storage | v3 R2048/B1 | Planned v4 with 2*T extra slots/head/chunk |
| --- | ---: | ---: |
| coefficients | 163,971,072 B | 164,757,504 B |
| immutable incoming snapshot | 3,145,728 B | unchanged |
| head flags | 192 B | unchanged |
| **logical added arena** | **167,116,992 B** | **167,903,424 B** |
| **16 KiB rounded physical arena** | **167,133,184 B** | **167,919,616 B** |

The cache adds P*(2*T)*4 = **786,432 B**, 0.75 MiB. This is a geometry-based
storage calculation from Root's stated v4 plan, not a measured allocation. It
requires matching layout/planner/extent proofs before any worker update.

V3 preparation zeros the full 163,971,072-byte coefficient plane, then writes
W/U/E (150,994,944 B), lower-inclusive score (6,488,064 B), and prefix
(393,216 B): **321,847,296 issued store bytes** in total. Scalar selector norm
reads add **201,326,592 issued bytes of S** and **201,326,592 of W**. Cached W
computes its norm from 50,331,648 issued W bytes, saving **150,994,944 B** of
repeated scalar W reads before counting cache-slot stores/reads. These are
issued source bytes and overlap data already read by MPP; they are not DRAM
bandwidth measurements. Zeroing and transform stores are common to v2/v3.

Declared threadgroup scratch is preparation 12,800 B (v2 12,672) and SG8 apply
8516 B (v2 8192). Resource/occupancy changes must be checked with actual pipeline
resource records; declaration size alone does not establish occupancy.

## Likely contributions and a bounded attribution follow-up

**HYPOTHESIS:** repeated W scalar norm work is a credible cost because it adds
50.3 million scalar squares/additions and 393,216 sqrt/reduction pairs inside the
long-lived apply group, rereading 192 MiB of W. Caching saves 75% of that W work
while changing only 0.75 MiB of persistent storage. Remaining S norms and the
12.6 million delta checks mean this optimization cannot be assumed to recover
the entire 0.471225 ms gap.

**HYPOTHESIS:** the two extra per-chunk barriers and leader-only cross-SIMD
aggregation create waiting/serialization beyond their small scalar addition
count. **HYPOTHESIS:** snapshot/restore/replay launch work, their unconditional
atomic loads and three extra encoder barriers contribute a fixed tax at zero
fallback. **HYPOTHESIS:** extra live selector tensors/scalars or declaration
thresholds reduce occupancy or cause spills. None has a measured phase share.

If restore/replay prove material in phase samples, a separate adapter experiment
could load their stable per-head flag once/SIMD group and broadcast it. At the
source level, restore loads would become 24,576 and replay loads 6144, with 192
apply loads unchanged: 30,912 instead of 983,232. Existing inter-phase barriers
make the flags stable throughout these two read-only phases. This need not add a
threadgroup barrier or touch the literal native recurrence, but the compiler may
already coalesce uniform atomic reads, so the benefit is a **HYPOTHESIS** and
requires native replay proof again. Do not implement it by broadcasting a flag
while an apply phase can still OR the same head.

A useful independent diagnostic should sample snapshot, prep, apply, restore and
replay boundaries inside one command buffer, retain exact production barriers,
and report both phase timestamps and the full command duration. Existing
`metal::CommandDispatchProfilingMode::DispatchBoundary` and profiling JSON APIs
provide the pattern; `StagePerDispatch` changes encoder boundaries and should be
a separately labeled diagnostic. Counter timestamp barriers also instrument the
execution, so compare a separate uninstrumented ABBA/BAAB baseline with exactly
the same seeds and inputs. Do not subtract totals from unrelated warmed commands
or use `GPUEndTime-GPUStartTime` as a per-kernel duration.

No attribution tool was added during this bounded source audit: SDK agent owns
the live equivalence oracle, and v4 source is still being prepared. A separate
tool can reuse the frozen harness/phase bindings after that source settles,
without modifying v3 or the sealed worker. No new whole worker is proposed until
v4 guard equivalence and performance pass.
