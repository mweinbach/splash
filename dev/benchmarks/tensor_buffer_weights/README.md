# Private per-tensor original buffers

This experiment changes only native buffer granularity. Production loader and
worker files, original payload bytes, manifests, norm audit, source identity,
and effective model fingerprint are untouched. GPU performance is unmeasured.

Private runtime: `build/flash-private-tensor-weights-v1/splash-flash` with its
adjacent metallib. Set `SPLASH_FLASH_PRIVATE_TENSOR_BUFFERS=1` to wrap each
original tensor's existing readonly mmap address plus aligned manifest offset,
with a native length rounded to 16 KiB. Tensor views then use offset zero and
their exact logical length. Mode `0` or absent retains whole-shard buffers.

In per-tensor mode the loader creates no whole-shard native allocation. All
3,748 native windows retain the same 21 Mapping owners. Canonical packing
validation proves every rounded window fits before the next tensor or shard
end. The windows partition 106,320,429,056 bytes exactly, including 26,292,326
padding bytes, with no overlap, gap, or second ledger charge. The first variant
maps every tensor: total backing is unchanged rather than reduced.

`immutableWeightBuffers()` returns all rounded windows so writable alias guards
still cover padding. PLE Tier2 pointers remain native-buffer plus zero view
offset; source allocations are retained and declared Read by existing backend
argument-buffer handling. PLE's indirect resource list grows from roughly eight
shared payload bases to 384 table planes, a potential additional cost to test.
Source/norm/fingerprint semantics remain identical despite storage changes.

`SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT=1` is mutually excluded because that
experiment assumes 13 selected whole-shard bases. Its selection construction is
skipped in per-tensor mode. Verified saved-derived residency remains compatible.

Potential future lazy-map savings are 901,332,992 rounded vision bytes and
2,482,044,928 original dense body coefficient bytes (3,157,491,712 including
the raw vocabulary projection). This variant elides none. Current singleton/raw
routes still use original dense coefficients, so omitting them without another
qualified route would break inference. All original vision tensors remain
present for identical loader inventory and accounting.

`/status.private_original_weight_storage` reports mode, native/mapping counts,
logical and padded bytes, padding, and unchanged numerical bytes. Actual driver
allocatedSize remains governed by the normal backend ledger. No physical
pinning guarantee is asserted.

The private build uses a Clang VFS header overlay so both prefixed and sibling
quoted includes see the same private FlashWeights definition. A simple `-I`
overlay alone causes duplicate production/private class definitions.

```sh
make -f dev/benchmarks/tensor_buffer_weights/Makefile \
  BUILD=build/flash-private-tensor-weights-v1 -j8 flash-next
.venv/bin/python dev/benchmarks/tensor_buffer_weights/audit_layout.py \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/private-tensor-weights-layout-audit.json
```

The fresh worker's 44 CPU checks pass and the manifest audit covers every tensor
window. Root must still validate readonly GPU access, full model quality, PLE
behavior, trace driver wiring, and matched throughput. Smaller native resources
may reduce the observed whole-shard WireMemory delay; this remains a hypothesis.
