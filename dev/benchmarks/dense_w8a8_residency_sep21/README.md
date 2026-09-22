Private persistent residency selection over the sealed dense W8A8 hybrid worker.
Preparation, compilation and CPU witnessing never create a backend/cache, read
model or captured operands, or run a GPU command. The original sealed workers
and their archived machinery remain unchanged.

```
python3 dev/benchmarks/dense_w8a8_residency_sep21/worker_overlay.py
make -f build/dense-w8a8-residency-prune-sep21-worker-v1/machinery/worker.mk \
  BUILD=build/dense-w8a8-residency-prune-sep21-worker-v1 -j4 all cpu-self-test
python3 build/dense-w8a8-residency-prune-sep21-worker-v1/machinery/worker_witness.py \
  --output build/dense-w8a8-residency-prune-sep21-worker-v1/cpu-witness-v1.json
```

Use fresh output names. `--base` optionally selects the sealed pure dense v3
worker; `--base-make` must name that worker's archived `machinery/worker.mk`.
The freezer authenticates parent source/input/artifact seals, retains the full
effective object/AIR closure except Forward and Worker, and reuses the old
cache object. It copies the parent metallib byte for byte. Make contains no
Metal compilation or metallib link command.

`SPLASH_FLASH_DENSE_W8A8_RESIDENCY_PRUNE_SEP21` accepts exactly `0` or `1`, defaults
to `0`, and freezes before paths, metadata or backend access. Flag 0 retains
the parent's persistent owner set. Flag 1 with dense W8A8 active omits exactly
84 original BF16 coefficient owners from that persistent lease; flag 1 with
dense W8A8 disabled omits the 168 unused I8/scale coefficient owners instead.
All backing owners, getters, fallback bindings, numerical identity, resource
plans and MemoryGovernor reservations remain intact.

For the hybrid parent at maximum rows ≥2048:

| Prune | Dense W8A8 | Persistent owners | Persistent bytes |
|---|---|---:|---:|
| 0 | 0 or 1 | 916 | 204,143,722,496 |
| 1 | 0 | 748 | 202,252,746,752 |
| 1 | 1 | 832 | 200,368,848,896 |

The omitted BF16 coefficient bytes are **3,774,873,600**, computed from all
84 exact `[N,K]` BF16 shapes; each full source extent is already 16-KiB aligned.
Filtering requires exact metadata and unique full-view saved-owner membership.
The original 25 Q4 expert owners and their host headroom guard stay unchanged.
Workspace buffers are absent from persistent coefficient selection.

[contract.md](contract.md) records the audited ordinary Metal command/binding
contract and primary Apple URLs. Dispatch-time transient residency and strong
backing retention support fallback execution when an owner is outside the
persistent set. This is a supported inference, independent of idle pinning,
reclaim timing or performance. Serialized root GPU qualification remains pending.

The archived witness compares the entire Forward source outside its selector
state and persistent getter, preserves the exact numerical identity expression,
compares resource admission and lifetime code, checks every reused AIR/object,
verifies metallib byte equality, exercises strict flags for both dense states,
and runs inherited dense/cache and main-worker CPU suites.
