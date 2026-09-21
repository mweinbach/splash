# Private exact original bytes in owned Shared buffers

This experiment creates the original model's 21 whole-shard Metal buffers with
`allocateBuffer(..., Shared)`, copies the original aligned payloads once, and
compares every byte against the readonly source mapping. It preserves the
3,748 tensor offsets, packed Q4/Q5/Q6/Q8 bytes, BF16 scales and biases, source
identity, norm audit and effective model fingerprint. It changes no weight
arithmetic and creates no disk payloads.

Production files, public headers, class layout, launcher profile and checkpoint
files are untouched. The private worker and loader were snapshotted from current
production. No VFS or ABI header overlay is needed.

Set `SPLASH_FLASH_PRIVATE_OWNED_ORIGINAL=1` for the copy. Absent or `0` retains
the original readonly mmap route. Other values fail before model loading. The
private worker constructs `MemoryGovernor` before loading and supplies its
existing startup admission callback outside FlashWeights' public interface.
Each shard reserves destination bytes plus the worst-case one temporary source
mapping, copies and checks it, then releases that mapping before committing the
reservation. The source mapping is never registered as a Metal buffer in owned
mode. Padding validation uses owned Shared contents after copying, and stored
Shard mapping pointers are null in this mode.

Final original native requested storage is 106,320,429,056 bytes, unchanged.
The largest temporary source is 5,204,606,976 bytes. Copy admission therefore
reserves at most 10,409,213,952 bytes at once, alongside prior owned destinations.
Only one source mapping is live. Normal backend `allocatedSize` accounting
charges the owned destination once; transient source headroom is protected by
the reservation but is never added to steady native allocation accounting.
Owned backing is not a guarantee of permanent physical pinning or reduced
idle latency; Root must measure the actual driver and HTTP behavior.

The private `/status.private_original_weight_storage` reports copy and full-byte
comparison timing, optional payload SHA timing, full load timing, admission
counts, opened/released/retained source mappings and unchanged format markers.
Payload SHA verification retains the original optional API behavior; manifest,
small-file and source-record identities are always checked. The full memcmp is
always performed in owned mode.

```sh
make -f dev/benchmarks/owned_original_weights/Makefile \
  BUILD=build/flash-private-owned-original-v1 -j8 flash-next
build/flash-private-owned-original-v1/splash-flash --cpu-self-test
.venv/bin/python -B dev/benchmarks/owned_original_weights/audit_layout.py \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/private-owned-original-layout-audit.json
```

Root alone should load or execute the model, using this private worker and its
adjacent `splash.metallib`, the same effective v7 flags, plus the owned-storage
switch. Test startup status, exact greedy outputs and continuations, held-out
quality/lifecycle checks, cold and post-idle command timestamps and matched
throughput before considering a local default. Measure copy time and increased
startup pressure as costs even if later idle latency improves.
