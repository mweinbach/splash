The private candidate adds explicit `FlashBatchMTPForward::primeTeacherCache`
over the sealed batch SG8 QSA and wide cached-dense candidate. It retains the
same compact real input fusion, attention hyper connection, and Q/K/V/index
projections as full grouped `None` forward. Each independent lane then uses the
singleton head's authoritative QSA prepare/pooling cache-writing prefix. The
remaining attention and head MLP work is omitted. Full `None`, `Last`, and `All`
forward retain their original graph and borrowed result contract.

`SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY=1` selects this operation only for
true grouped prompt priming. It is strictly `0` or `1`, defaults to off, and
requires `SPLASH_FLASH_MTP=1`, `SPLASH_FLASH_BATCH_PREFILL=1`, and
`SPLASH_FLASH_BATCH_MTP_PREFILL=1`. Its syntax and dependencies are checked
before model/config loading or Metal backend creation. Singleton and lone
tail priming continue to use the existing singleton teacher flag.

Prepare and compile without model loading or GPU execution:

```sh
.venv/bin/python dev/benchmarks/prefill_batch_teacher_sep21/prepare.py --output build/prefill4k-batch-teacher-gathered-sep21-v2
.venv/bin/python dev/benchmarks/prefill_batch_teacher_sep21/build.py --build build/prefill4k-batch-teacher-gathered-sep21-v2
build/prefill4k-batch-teacher-gathered-sep21-v2/policy-cpu
build/prefill4k-batch-teacher-gathered-sep21-v2/splash-flash --cpu-self-test
```

The current private source tree is
`build/prefill4k-batch-teacher-gathered-sep21-v2`. The generator verifies every
sealed parent source hash before transformation and modifies only the two
head API/header pairs and Worker. Every Metal source and all trunk, batch main
QSA, verifier, and state/workspace planner bytes remain unchanged. The host
builder recompiles the three affected executors, snapshots other byte-identical
parent objects and four core objects, and copies the exact parent metallib.
`host-build-audit.json` records the source, object, binary, and library hashes.

CPU verification passed 18 actual parser/dependency/getter cases, including 14
refusals, the normal Worker suite, a serve-path invalid-flag refusal before
config/backend creation, and 12 identical planner values for flag off/on and
the sealed parent. The optimization adds zero workspace or state bytes.

Status exposes `mtp_batch_teacher_priming_route` and the completed grouped
cache-prime command, lane, and real-pair counters. They advance after submission,
diagnostics, and every lane's healthy expected length have passed. Existing
timing and cohort accounting remains in use.

The linked `batch-teacher-oracle` uses a real target 2048-token fixture, compares
all five persistent QSA cache planes bitwise after every compact fold, requires
finite BF16/F32 values, and compares future `Last`/`All` head hidden, vocabulary,
and exact greedy records. Cases include true B4 2047-pair priming, sparse B1
8191-pair priming, B1..4 ragged boundaries, EOS pairs, rollback/reprime,
pre-mutation capacity/foreign/duplicate/shape guards, and lane poison semantics.
It requires four arguments: metallib, installed model package, exact 2048-token
fixture JSON, and a fresh report JSON. `--help` constructs no backend.

The root v1 GPU attempt reached a late host-guard assertion after all four
cache/future/rollback cases. That assertion incorrectly expected an uninitialized
state to report `poisoned() == false`; the existing getter returns true when
`impl_` is absent. The v1 failure reconstruction records this limit explicitly,
and its original oracle source and binary are preserved. V2 leaves every runtime
source, ownership guard, Worker binary, and metallib byte-identical. It compares
the rejected uninitialized state's exact before/after impl, owner, and public
getter values, requires all owned-state impl/owner/buffer views to remain
unchanged on host guards, and checks the uninvolved numeric-failure lane's full
cache. The oracle, attribution dependency, and CPU getter policy are sealed into
the v2 source manifest.

V2 writes exact counters and completed case labels to
`REPORT_JSON.checkpoint.json` after each major case. Exceptions write
`REPORT_JSON.failure.json`; neither is an overall success report. The sidecar
writer was exercised before backend/model creation in CPU validation.

The native oracle and service qualification must be run by the root GPU owner.
Use fresh report filenames. Compare teacher flag `0` and `1` in the same binary
with the batch-main `SPLASH_FLASH_BATCH_QSA_BULK_PREFILL` setting held fixed to
isolate this optimization; compare both new flags off against the original
combined cap4 build separately. No GPU execution has been performed during
candidate preparation or compilation.
