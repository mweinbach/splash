This private experiment retains original packed-Q4 target expert operands for
standard decode, every singleton verification, and prefill calls below 256 rows.
Main nonverification calls at 256 or more rows use the certified Full512-I8 store.
The current four-row selective F32 dense/HC-up and Q8 vocabulary policies remain
unchanged. Large prefill retains the qualified role-aware 2K BF16 dense selector,
exact SG8 QSA, teacher cache-only priming, and exact pointwise kernels.

The fixed four-row F32 selection has 118 maps and 3,247,964,160 payload bytes:
97 HC-up maps, seven QSA output maps (five N32 eligible), and 14 other projection
maps. Removing the other 390 maps saves 11,143,741,440 bytes. Constructor and
admission planner select the same names. The dense manifest stays intact; no
coefficient is converted, changed, or regenerated. Existing original Q4 bases,
BF16 maps, F32 diagnostics/padding, Full512 ranks, and trained MTP caches remain
charged. Copied tensors and Metal views retain existing owners without creating
coefficient copies.

The historical CPU ledger estimates 215.03 GB for one eligible request and
216.90 GB for four requests before the additional SG8 bulk workspace. This
experiment disables 5.378 GB of true batch arenas and adds 234.36 MB of exact bulk
QSA workspace, giving a source-derived planner estimate of 209.89 GB for one
eligible request or 211.75 GB for four cooperatively scheduled requests.
Root's compiled CPU planner remains authoritative. Admission remains governed by fresh
native, device, host-headroom, and pressure checks. Pipeline/CPU overhead and
file-backed reclaim accounting are not an admission proof.

The experiment requires `SPLASH_FLASH_HYBRID_Q4_I8_FIXED_R4=1`, a 2K singleton
arena, context at most 16K, and fixed depth 3. All true batch flags must be zero;
`SPLASH_FLASH_GDN_BATCH_ILP` must also be zero because its native flag parser
requires batch prefill. Individual requests can cooperate through the original scheduler. Larger joint
MTP cohorts need the separate R8/R16 F32 inventories and are unsupported here.
The stable numerical identity explicitly binds the phase policy and selection.
The I8 prefill state means there is no universal original-Q4 model parity claim.
Prefix caching stays disabled.

Prepare source and metadata only:

```sh
.venv/bin/python dev/benchmarks/hybrid_memory_sep21/plan.py \
  --out build/release/flash/sep21-hybrid-r4-f32-pruning-cpu-plan-v1.json
.venv/bin/python dev/benchmarks/hybrid_memory_sep21/prepare.py \
  --base build/prefill4k-wide-fullcache \
  --output build/hybrid-q4-i8-fixed-r4-sep21-v2
```

Root can compile and run CPU-only checks:

```sh
make -f dev/benchmarks/hybrid_memory_sep21/worker.mk -j 8 all
make -f dev/benchmarks/hybrid_memory_sep21/worker.mk cpu-self-test
.venv/bin/python dev/benchmarks/hybrid_memory_sep21/startup_guard.py \
  --build build/hybrid-q4-i8-fixed-r4-sep21-v2 \
  --out build/release/flash/sep21-hybrid-startup-profile-guard.json
```

`environment.json` records the fixed benchmark flags. Root must add the existing
certified `SPLASH_FLASH_OPERAND_STORE` and `SPLASH_FLASH_INT8_EXPERT_STORE` paths
for runtime qualification. Standard decode additionally sets MTP and teacher
cache-only flags to zero. The startup preflight rejects incompatible dependencies
before backend creation. `SPLASH_FLASH_PRIVATE_ADMISSION_REPORT` supports a fresh
path for the existing post-original-weight/pre-trunk governor diagnostic.

Preparation writes `overlay-manifest.json`, `cpu-source-witness.json`, and frozen
link inputs. It reads source files, object/AIR artifacts, and JSON metadata; it
does not read any model payload, instantiate Metal, or run a GPU command.

`startup_guard.py` runs the actual compiled worker against a temporary package
containing configuration metadata only and an empty operand-store directory.
The valid profile must reach the deliberate missing Full512 manifest refusal;
this happens before backend creation. Negative cases exercise incompatible
profile flags. A profile failure that precedes the deliberate refusal is a failed
guard, even when `--cpu-self-test` passed. The original v2 compiled worker also
hard-requires the now-disabled batch-ILP flag in its private dependency list;
The repaired, root-ready worker is
`build/hybrid-q4-i8-fixed-r4-sep21-v3/splash-flash`. It removes the unused
batch-ILP requirement and rejects the competing BF16 small-row cache before
backend creation. Its eight compiled startup guard cases and complete CPU
self-test passed. Original phase/F32 numerical identity, all numerical sources,
kernel library, and nine non-Worker host objects are unchanged from v2.

The bounded repair reuses the already compiled v2 closure:

```sh
.venv/bin/python dev/benchmarks/hybrid_memory_sep21/repair_v3.py
make -f dev/benchmarks/hybrid_memory_sep21/repair_worker.mk -j 2 all
make -f dev/benchmarks/hybrid_memory_sep21/repair_worker.mk cpu-self-test
.venv/bin/python dev/benchmarks/hybrid_memory_sep21/startup_guard.py \
  --build build/hybrid-q4-i8-fixed-r4-sep21-v3 \
  --out build/release/flash/sep21-hybrid-startup-profile-guard-v3.json
```

The repair helper requires a fresh output and leaves the prior v2 source and
compiled artifacts intact. Clear ambient `SPLASH_FLASH_` values before applying
`environment.json`; in particular, `HOT_EXPERT_PLAN` must be entirely unset.

The v3 report admitted 209.24 GB of backing and reached 71.19 tok/s steady
decode. Its GPU prefill time stayed approximately 611 ms while command-boundary
delays changed substantially. Initial host availability dropped by 73.13 GB
after the first wave, and later requests retried admission while host headroom
was near the 1 GiB warning margin. The saved-only residency lease excluded raw
Q4 expert owners. Driver rewiring is a hypothesis; this report did not collect
physical wired-page or DRAM counters. Residency alone cannot make unchanged
611 ms GPU work exceed 4K prefill.

The separate opt-in v4 worker is
`build/hybrid-q4-i8-expert-residency-sep21-v4/splash-flash`.
`SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT=1` selects exactly 25 existing SSD owners
required by the 432 target-Q4 tensor views. The complete owners occupy
69,363,302,400 bytes, including 1,415,577,600 bytes of co-owned trained MTP expert
weights. All owners were already classified as pure Text or Text|MTP by the
loader. PLE, vision, and unknown owners remain excluded, and legacy whole-base
text residency still cannot be enabled.

The composite contains 748 existing owners and 202,252,746,752 bytes. No weight
backing is added. The unchanged host policy requires the selected original-owner
bytes plus 2 GiB of fresh host headroom; it passed the retained v3 initial sample
by 2.43 GB. A fresh governor snapshot decides whether v4 adds the owners. A denied
addition leaves the saved-only lease and records a failure reason. Status includes
the actual owner census, fresh host sample, and whether the composite is active.
The numerical phase/F32 identity and kernel library remain unchanged.

```sh
.venv/bin/python dev/benchmarks/hybrid_memory_sep21/expert_residency_v4.py
make -f dev/benchmarks/hybrid_memory_sep21/residency_worker.mk -j 2 all
make -f dev/benchmarks/hybrid_memory_sep21/residency_worker.mk cpu-self-test
.venv/bin/python dev/benchmarks/hybrid_memory_sep21/startup_guard.py \
  --build build/hybrid-q4-i8-expert-residency-sep21-v4 \
  --out build/release/flash/sep21-hybrid-expert-residency-startup-guard-v4.json
```

V4 compilation, all 49 CPU checks, pointwise policy modes, and 12 compiled
startup guards passed. Root must establish runtime admission, composite activation,
physical behavior, and performance in its serialized model comparison.
