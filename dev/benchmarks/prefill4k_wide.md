# Private singleton prefill rows 4096 / 8192

This experiment keeps normal sources and builds unchanged. The private overlay
widens the singleton trunk arena to 8192 physical rows, using the existing checked
scratch formulas and the existing MemoryGovernor reservation before construction.
It does not change arithmetic, trained coefficients, context limits, QSA causal
ordering, MTP draft depth, or request-state ownership.

Prepare and compile without GPU work:

```sh
.venv/bin/python dev/benchmarks/prefill4k_wide_overlay.py
make -f Makefile -f dev/benchmarks/prefill4k_wide.mk prefill4k-wide -j 8
build/prefill4k-wide/splash-flash --cpu-self-test
```

Refresh the overlay and rebuild after other production candidates change. All
Flash host objects are private and include the copied headers; this avoids a
mixed maximum-row constant across translation units. The private metallib reuses
the original AIRs except the three GDN shader files, whose parameter validity
guards are widened to match the host guard. Q/K width constants of 2048 remain
unchanged. `overlay-manifest.json` records original and overlay SHA-256 digests.

The original blocking points are the Worker allowed-row parser, Forward
constructor guard and workspace guard, GDN host row constant, PLE SSD host staging
guard, and the three GDN shader parameter guards. The private route adds
`private-singleton-prefill-arena-max8192-causal-qsa128-v1` to kernel-routes status.
The independent batch GDN ILP candidate retains its 2048-row lane limit.
Batch-prefill stays at at most 2048 rows per lane / 8192 total. Optional standalone
fused MoE primitives still cap at 2048 and are not used by FlashForward's existing
blocked-MoE path. Blocked MoE, buckets, shared experts, and affine MPP already
support 8192 physical rows. Existing M64 selection begins at 4096 physical rows.

The target vocabulary scratch remains capped at 128 rows and prefill requests
only Last logits. QSA workspaces and ordered causal QSA calls remain capped at
128 rows. Verification remains capped at 16 rows. The teacher head continues
priming exact adjacent hidden/next-token pairs in 128-row pieces, including all
8191 real pairs of an 8192-token prompt. Each final anchor feature is copied
before any peer trunk work can reuse the target scratch.

CPU self-test checks valid 4096/8192 parser geometry, invalid 4095/8193/16384
settings, true adjacent teacher-pair slicing at all wide boundaries, monotonically
increasing scratch admission, planning rejection above 8192, GDN host guard
positive/reject cases before any buffer work, and singleton PLE staging extents.
Source transforms require exact guard matches and abort when source changes.

GPU qualification remains pending and is serialized by root. Use the current
profile flags and the frozen `code2048`, `code4096`, `code8192` fixtures from
`build/release/flash/prefill4k-fixture`. Run every larger prompt in capacity 16384
for continuation. Compare 2048/4096/8192 trunk arenas against the same frozen
baseline, including MTP teacher priming in end-to-end prefill time. Track:

- `/status` `maximum_prefill_rows`, `workspace_bytes`, `prefill_batches`,
  `prefill_rows`, `mtp_prompt_priming_rows`, active requests and in-flight state.
- Delta prefill rows equal actual prompt tokens; trunk-command count drops from
  four to one for 8192 tokens when using an 8192-row arena. Teacher-pair totals
  remain exactly prompt tokens minus one.
- Logit/hidden diagnostics and 32–128 token greedy continuation versus the
  2048-row baseline; compare tokens exactly and inspect logits/BF16 differences
  if the existing M32-to-M64 MoE policy changes rounding.
- Partial wide prompt windows, final 1/2/3-row tails, stop/EOS and output budgets,
  cancellation/deadline during a wide command and after teacher priming,
  overlapping request IDs, re-admission, scheduler idle, and allocation refusal
  with a deliberately insufficient native budget.
- PLE-history, convolution-state and GDN recurrence continuation after wide
  windows; same prompt consumed as chunks of 2048 versus one wide window, then
  several identical one-token continuations. No output or hidden scratch is
  retained as a borrowed view across a subsequent trunk call.

Only promote a measured benefit after actual numerical and service lifecycle
qualification. A private compiled artifact and CPU self-test do not establish
inference correctness or performance.

## Combined wide + Full512 layer

The combined layer imports the existing Full512 agent's strict source transforms
after the wide transforms. It copies every Flash host/header file once, including
the normal teacher-cache API and projected dense tile selector. All copied hosts
compile against the same header set. The extra INT8-store shader replaces the
original AIR; the three GDN shaders retain the wide parameter guards.

```sh
.venv/bin/python dev/benchmarks/prefill4k_wide_overlay.py \
  --fullcache --output build/prefill4k-wide-fullcache
make -f Makefile -f dev/benchmarks/prefill4k_wide.mk \
  BUILD=build/flash-next SPLASH_PRECISION=hybrid \
  PREFILL4K_WIDE_BUILD=build/prefill4k-wide-fullcache \
  PREFILL4K_WIDE_FULLCACHE=1 prefill4k-wide -j 2
build/prefill4k-wide-fullcache/splash-flash --cpu-self-test
build/prefill4k-wide-fullcache/metadata-cpu
build/prefill4k-wide-fullcache/expert-oracle --cpu-self-test
.venv/bin/python dev/benchmarks/prefill4k_wide_preflight.py \
  --output build/release/flash/CHOOSE-FRESH-combined-memory-preflight.json
```

The memory helper performs Mach host sampling and static planner calls without
constructing a MetalBackend or loading a model. Current capacity16384/rows8192/
depth3/full512 plans require 148,614,299,648 trunk bytes. The complete static plan
with four eligible request states is 231,217,774,592 bytes, leaving
16,172,341,658 bytes inside the 247,390,116,250 engine budget before a 2GiB
experimental margin. This sum includes original mappings, saved BF16/F32 caches,
the full expert store and ranks, sequential/joint/batch workspaces, and owned
feature staging. Residency references never charge their backing twice.

The separate historical four-request estimate is 231,178,960,896 bytes. Its
transient allowance comes from a measured singleton workload and is not a proven
peak bound. The preflight rejects a candidate that fails either the complete
engine plan with margin or its live necessary largest-stage host check. An
unloaded host sample is not actual post-weight startup admission. Disk
`minimum_free_bytes` does not satisfy a RAM requirement.

Set `SPLASH_FLASH_PRIVATE_ADMISSION_REPORT` to a fresh path during root's actual
startup. Before the trunk/cache constructor, the private Worker refreshes pressure
and writes original allocation bytes, exact planned trunk bytes, current governor
reservations, host reserve/headroom, and engine/host stage fit booleans. Its
subsequent `tryReserve` refreshes those measurements again and rejects any stage
that exceeds engine budget, host reserve plus1GiB warning margin, or Normal
pressure. Later state/head/batch reservations retain the same admission rules.
Use `SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=0` for combined warmed comparisons;
the qualified Top64 maintenance union does not describe Full512.

Full512 changes arithmetic and has its own coefficient/quality gates. Qualify
wide state continuation separately against Top64 first, then qualify Full512
quality, long-prefill cancellation/deadline, admission failure, concurrency and
unload/reload using the combined source. Exact 64-token generation hashes alone
do not prove persistent GDN/QSA/PLE state equality.
