The rejected v8 joint vocabulary route multiplied BF16 activations by raw Q8
codes, then applied group scale and bias to the partial dot. The active route
rounds each reconstructed coefficient to BF16 before multiplying. The private
v9 primitives preserve that coefficient boundary and keep the strict full-lane
vocabulary relative-L2 limit at 0.0001.

The first shader reconstructs an original unsigned code with the same helper
and BF16 cast used by `flash_dense_cache_expand_bf16`. It decodes K64 or K128
slices into threadgroup BF16, then uses BF16×BF16 MPP with F32
`multiply_accumulate`. Each original coefficient remains read only. The full
635,699,200-coefficient audit compares actual reconstructed words against the
existing BF16 cache; this runs outside timed projections.

Root's first actual GPU screen passed the complete coefficient audit and all
five full four-lane vocabulary outputs were exactly equal to the cached
control. Accuracy was fixed, but the fastest N64K64/s4 projection took about
2.098 ms against 1.582 ms for the control. Wider or K128 tiles were slower. This
route is private and does not change defaults.

The independently generated exact BF16 dictionary provides a second decoder.
There are only 41,151 distinct original BF16 scale/bias pairs among 9,932,800
groups. A 21,069,312-byte BF16[pair,256] table and 19,865,600-byte U16[N,40]
index table select exactly the same coefficient as the qualified cache while
reusing the original 635,699,200 code bytes. The independent CPU converter
checks all 10,534,656 dictionary entries against exact F64 affine arithmetic
and nearest-even BF16 rounding. Every selected actual coefficient is still
audited on the GPU against the cache. Dictionary storage is 676,634,112 bytes
including existing codes, versus 1,271,398,400 bytes for expanded BF16 weights;
this is a storage result, not a throughput claim.

Root's first lookup screen was exact but slower, about 4.287 ms against 1.623
ms. A validated lookup specialization removes invariant per-coefficient pair
and finiteness checks after the mandatory full audit. N32K64/s4 and N64K64/s8
remained exact but took about 2.052 and 2.082 ms against controls near 1.6 ms.
The table introduces a dependent coefficient lookup for every code, so less
weight storage alone does not establish faster execution.

The next bounded screen fills MPP's BF16 right-input cooperative tensor
directly in registers. Apple's installed `MPPTensorOpsMatMul2dImpl.h` requires
a single SIMD group for cooperative inputs. N32K64/s1 and N64K32/s1 therefore
avoid the threadgroup weight arena and forty or eighty staging barriers.
Each valid fragment element uses the public multidimensional coordinate,
the original coefficient helper and BF16 cast, then F32 accumulation. This is
compiled and CPU-qualified. Root's actual GPU screen now passes the complete
coefficient audit and every four-lane full-vocabulary BF16 word for captured
real, random, cancellation and sparse inputs. N32K64/s1 took 0.992 ms against
1.578 ms for the cached control, a 37.2% reduction. The N64K64/s1 and
N64K32/s1 alternatives lose and remain private. Shader-validation correctness
also passes; its debug timings are not performance evidence.

Rebuild a fresh register artifact without GPU execution:

```sh
.venv/bin/python -B dev/benchmarks/build_flash_joint_q8_exact_bf16_v9.py \
  --mode register --build build/flash-joint-q8-exact-bf16-v9-register-fresh
```

Root's serial GPU invocation (omit `--run` for a read-only invocation preview):

```sh
.venv/bin/python -B dev/benchmarks/flash_joint_q8_exact_bf16_v9_run.py \
  --build build/flash-joint-q8-exact-bf16-v9-register-repro \
  --report build/release/flash/exact-register-v9-fresh.json \
  --input build/release/flash/trained-head-real-inputs-v8/head-length2048-rows1.bf16 \
  --variants 1,2 --patterns 1 --pairs 4 --run
```

The captured input has one real normalized BF16[2560] head row. This bounded
primitive repeats that exact row in four lanes; synthetic screens supply four
different rows. It is not a real four-request head or service qualification.
Root's private full-head screen now passes at contexts 128 and 2048 with four
distinct real last-target features, teacher-primed head QSA, four future causal
head proposals and truncate-overwrite at retained suffix lengths 0–3. Every
head premixer, vocabulary and QSA plane is byte exact; greedy records match.
Full-head GPU medians improve from 3.257 to 2.563 ms at context128 and from
3.534 to 2.919 ms at context2048. Separate rows2/3 primitive screens including
nonfinite inputs preserve expected sticky diagnostics and greedy records.
The completed reports are `build/release/flash/v9-exact-q8-register-performance.json`,
`v9-exact-q8-register-patterns.json`, `v9-exact-q8-full-head-proof.json`,
`v9-exact-q8-register-rows2.json` and `v9-exact-q8-register-rows3.json`.

The production module `FlashBF16Q8Head` is default off under
`SPLASH_FLASH_MTP_Q8_BF16_REGISTER`. Its branch is only the joint head's cached
`Last` vocabulary projection at 2–4 lanes. The module reconstructs original
Q8 codes directly into the BF16 cooperative right-input fragment, reuses the
existing padding arena, and allocates no weights or scratch. Source identity,
dtype/shape, extents, stride overflow, Shared alignment and every writable
overlap are checked. Only successfully completed projections increment its
public command/row counters. Independent CPU/source review passes, and seven
strict-switch modes cover 1,799 valid and 630 invalid geometry cases. Root's
full service quality and matched HTTP qualification remain required before
default promotion.

All artifacts use the fresh 200-byte command-timing ABI from the v8 snapshot.
The primitive checks guarded outputs, sticky diagnostics, and exact GPU greedy
records. Build and invocation manifests freeze source, objects, binaries,
metallibs, input and dictionary hashes. The original checkpoint, oMLX
installation, local default and GitHub state remain unchanged.
