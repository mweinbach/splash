# Exact lazy GDN copy fusion

This private candidate projects verification QKV directly into each lazy record's owned retained tape and saves old convolution history inside the ordered carry dispatch. It removes72copy dispatches in a singleton four-row verifier. Joint verifiers remove `(lanes+1)*36` copies. It adds no GPU allocation or math/precision change. Prefill, ordinary AR, trained MTP and R1 lazy bypass graphs remain unchanged.

The recurrent F32 initial snapshot remains in the original persistent kernel, before the first recurrence update. History stays unchanged throughout persistent verification. The new carry saves all three old rows for its unique `(lane,channel)` before the first history overwrite, then executes the original ascending carry loop. The verify-to-carry dispatch boundary stays present.

Only `b.qkv.sameView(record.rawQKVDestination(actualRows,actualLanes))` may alias its owned tape. Offset, short, oversized, wrong-geometry and foreign-record views are rejected. Pending records cannot supply another destination. Other work/state/weight/tape pairs remain disjoint. Joint validation skips precisely the work0/ownedRawQKV pair and still checks every other pair before packing.

The option `SPLASH_FLASH_GDN_LAZY_COPY_FUSION_SEP21` is strictly0/1 and captured per constructor. Existing objects do not change when the environment changes. Worker startup requires existing lazy rollback and fused GDN flags. Flag0 preserves prior graphs, route identity and numerical derivative. Flag1 adds an exact scheduling route marker; expert derivative identity remains unchanged.

Prepared worker: `build/lazy-copy-fusion-sep21-v2/{splash-flash,splash.metallib}`. Its CPU self-test passed;245sources and115copied object/AIR inputs are sealed in `manifest.json`. v1 failed only offline Metal compilation of an enum atomic operand and is preserved. Usev2. No worker GPU correctness or throughput is implied by compilation.

Prepared final layer qualifier: `build/lazy-copy-fusion-sep21-layer-oracle-v2/{oracle,splash.metallib}`. It passed11501CPU checks with zero backend constructions or GPU commands;245sources and50linked inputs are sealed. Root may run after unloading other GPU/model work:

```sh
build/lazy-copy-fusion-sep21-layer-oracle-v2/oracle \
  build/lazy-copy-fusion-sep21-layer-oracle-v2/splash.metallib \
  build/release/flash/sep21-lazy-copy-fusion-layer-v2.json
```

The default strict matrix tests R1/2/3/4/8/16; lanes1/2/3/4 with wider rows restricted to singleton; cold/warm/extreme coefficients; every kept prefix and mixed lanes; full acceptance, terminal lanes and abort; three changed carried sequences and ordinary future continuation; all six tape arenas, including inactive capacity/guards; padded BF16/F32 live state, immutable inputs, ownership/lifecycle rejection and64numeric boundary cases. Zero differing bytes is the gate. Continuation counters describe helper invocations, including all-terminal cases that submit no command. This layer test explicitly excludes whole-model logits/QSA cache/performance qualification.

A sequential whole-target oracle is CPU-ready at `build/lazy-copy-fusion-sep21-whole-oracle-v1/{oracle,splash.metallib}`. It seals246sources and50linked inputs. It compares every request cache plane, logits, hidden/compact greedy records and future continuation between flag0/1 before any model promotion. Its CPU preparation and Root GPU result are independent milestones. It covers B1 kept1/2/3/4 and four changed future inputs; B4 and trained MTP remain outside its scope.

```sh
# Apply the current complete pointwise/gathered Full512 profile/environment.
build/lazy-copy-fusion-sep21-whole-oracle-v1/oracle --gpu \
  build/lazy-copy-fusion-sep21-whole-oracle-v1/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/prefill4k-fixture/code2048.tokens.json \
  build/release/flash/sep21-lazy-copy-fusion-whole-v1.json
```

The whole oracle requires32GiB free disk headroom for a bounded control spill and uses1MiB streaming comparison scratch. It creates control/candidate backends sequentially. Source/object/library/tokenizer/canonical-token identity is checked. The custom copy-fusion flag is captured and deliberately flipped internally to verify constructor immutability.

An actual joint-GDN-wrapper qualifier is CPU-ready at `build/lazy-copy-fusion-sep21-batch-oracle-v1/{oracle,splash.metallib}`. It seals246sources/50inputs and passed63CPU shape/planner checks without a backend. It exercises actual native request pack/scatter, precise alias rejection before any pack dispatch, mixed partial/full/terminal commits, independent eager prefixes, full retained arenas/guards and ordinary R1 continuation for rows2/3/4 and lanes2/3/4. It loads no model and does not establish joint whole-target QSA/PLE/logit parity.

```sh
build/lazy-copy-fusion-sep21-batch-oracle-v1/oracle \
  build/lazy-copy-fusion-sep21-batch-oracle-v1/splash.metallib \
  build/release/flash/sep21-lazy-copy-fusion-batch-v1.json
```

The saved stage diagnostic gives1.17ms as a copy-elimination ceiling. Stage profiling changes encoder boundaries, and the fused history stores still cost time. It is not an unsampled speedup prediction.
