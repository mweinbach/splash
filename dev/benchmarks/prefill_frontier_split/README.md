# Private idle-prefill resource frontier diagnostic

This private `FlashForward.cpp` snapshot keeps the public Forward ABI, full
graph construction, original numerical stages, and every resource binding
unchanged. `SPLASH_FLASH_PRIVATE_PREFILL_FRONTIER_SPLIT=1` changes execution only
for a nonverification call with exactly 128 rows and logical begin zero. The
first two dispatches must be the original embedding and HC-expand operations;
the clone validates those names before submitting two synchronous graph
subspans. All other rows, decode/continuation positions, and verification calls
retain one original command. The flag is absent/off by default.

The first command binds one original native base (5,204,606,976 bytes) rather
than the full original-base frontier (106,320,429,056 bytes). The second command
contains every remaining dispatch. Both complete before diagnostics are read
or logical state length is published. Errors poison the request as in the
original route. GPU/wall/host command timings are summed, while the oracle
also measures complete call wall time including the boundary between submits.
No overlap is promised: the existing backend permits one outstanding ticket.

The dedicated oracle loads the normal 2048-row trunk arena, 16-row prefix
storage, 128-row trained-head arena, and normal saved-derived residency union.
It excludes the Worker's extra batch arenas and HTTP, so this is a driver
attribution test, not an HTTP qualification or production throughput score.
The head is constructed only to match normal operand/cache residency; no head
inference executes. The four prefills are whole/split immediately, then split
and whole after separate nine-second idle intervals. `FLASH_FRONTIER_FIRST`
can reverse the first two calls; `FLASH_FRONTIER_IDLE_SECONDS=0..15` controls
the latter intervals. Each call uses a fresh trunk state and executes two
continuation tokens afterward. Full BF16 vocabulary logits and both subsequent
full-logit rows must match exactly across all four states.

Profiles are drained immediately after target prefill, before logit copying,
hashing, or continuation work. `REPORT.commands.jsonl` names the segments
`whole_graph`, `embedding_expand`, and `remaining_layers`; it preserves raw
command and driver kernel timestamps plus Mach/steady clock bridges, and
reports actual cross-clock commit-to-GPU delays. The final report compares
total call time and sums segment delays. A faster first segment with no lower
total latency is redistribution, not a speedup.

Fresh artifacts:

```sh
make -f dev/benchmarks/prefill_frontier_split/Makefile \
  BUILD=build/flash-private-prefill-frontier-split-v1 -j8 \
  build/flash-private-prefill-frontier-split-v1/prefill-frontier-split-oracle
```

Root alone may run the GPU oracle with normal v7 environment/derived-store
paths already applied, after stopping its owned serving process:

```sh
build/flash-private-prefill-frontier-split-v1/prefill-frontier-split-oracle \
  build/flash-private-prefill-frontier-split-v1/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/flash-v6-verifier-attribution/fixture/ctx128-width1-lane0.tokens.json \
  build/release/flash/FRESH-frontier-report.json
```

The fresh host objects/library compiled and `--help` ran without Metal/device
execution. The private route/flag/ABI policy passed 41,011 assertions under
AddressSanitizer and UndefinedBehaviorSanitizer. Frozen artifact hashes and
CPU scope are in
`build/release/flash/private-prefill-frontier-split-cpu-qualification.json`.
GPU exactness, continuation, idle delay, and total latency remain Root's gates.
Production source, model payloads, and local default profile are unchanged.
