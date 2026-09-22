# Isolated prefill FMA worker

This experiment freezes `build/moe-pointwise-sep21-worker-v1`, adds the Root
qualified explicit-FMA scalar GDN kernel, and selects its V16/T32 entry only
inside `addGDNStagedPrefill` for physical rows 64 through 2048 and lanes 1
through 32. It adds no GPU buffers. Decode, verification capture, lazy rollback,
replay, MTP and their existing source/object/AIR implementations remain intact.

The opt-in is `SPLASH_FLASH_GDN_PREFILL_FMA_SEP21=1`; it requires
`SPLASH_FLASH_GDN_STAGED=1`. Missing or `0` selects the native graphs. Other
values and missing dependencies reject before paths, metadata or backend
creation. The successful first policy decision is immutable for the process.

The enabled route changes F32 rounding through explicit FMA and retains BF16
source activations, F32 state and existing BF16 output boundaries. It has its
own route marker and numerical policy. Worker status replaces
`target_numerical_derivative_sha256` with SHA256 of the base numerical identity,
policy and qualified kernel source SHA, separated by newlines. The original
identity remains in `target_base_numerical_derivative_sha256`. Flag 0 preserves
the original target identity exactly.

Build and source/CPU checks:

```
.venv/bin/python dev/benchmarks/gdn_chunk_sep21/worker_overlay.py
make -f dev/benchmarks/gdn_chunk_sep21/worker.mk -j4
make -f dev/benchmarks/gdn_chunk_sep21/worker.mk cpu-self-test
.venv/bin/python dev/benchmarks/gdn_chunk_sep21/worker_witness.py
```

Artifacts are in `build/gdn-prefill-fma-sep21-worker-v1`: `splash-flash`,
`splash.metallib`, `prefill4k-attribution`, `policy-cpu`, `overlay-manifest.json`
and `source-witness.json`. All 243 source files and 116 inherited link inputs
are sealed; the build includes no live runtime-header dependencies.

The CPU policy test passed 557,600 route cases and 1,254,725 checks, including
all original staged tile selectors, bounds, malformed flags, dependencies,
process freeze and numerical-identity behavior. ASan/UBSan passed. The worker's
existing CPU self-test also passed with `gpu_work=false`. Real worker startup
with an invalid new flag or an unsatisfied dependency rejected before resolving
a deliberately missing model directory.

The source witness passed all 41 checks, including exact native flag-0 graph
selection/order, protected forward/batch/verification/replay ranges, 18 protected
module source chains, 177 runtime/ABI headers and inherited object/AIR hashes.

Root measured the isolated V16/T32 recurrence at roughly 1.708 ms against
1.890 ms native for 2048 synthetic rows (1.106x). The F32 state/output/history,
continuation and canary gates passed. The original delta-only relative guard
fails on a cancellation fixture with exactly the same measured error in native
canonical and FMA arithmetic; the old evidence and threshold remain unchanged.

These are primitive/source results. Full-model logits and carried-state
behavior, decoded sequences, MTP acceptance and matched end-to-end prefill and
decode measurements remain Root's required gates. No model payloads or GPU
resources were read or created by the integration agents.
