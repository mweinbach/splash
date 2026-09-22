This private one-layer primitive stores each original signed I8 integer code
exactly in BF16 and screens four producer variants against corresponding
original I8 controls. The cache contains **unscaled integer codes**. It never
incorporates the F32 row scales; those source views and post-dot scale
operations stay unchanged, along with BF16 projection, SwiGLU and output
scatter boundaries.

The variants are:

| Variant | Candidate suffix | Native job rows | SIMD groups | Reduction |
| --- | --- | --- | --- | --- |
| 1 | `m32_n64_sg4` | 32 | 4 | Whole K, F32 multiply |
| 2 | `m32_n64_sg2` | 32 | 2 | Whole K, F32 multiply |
| 3 | `m64_n64_sg8` | 64 | 8 | Whole K, F32 multiply |
| 4 | `m32_n64_k128_sg2` | 32 | 2 | Fixed K128, F32 accumulation |

Both gate/up and down/scatter use names beginning with
`prefill_moe_sep21_bf16_codes_`. Each candidate retains identical original
native M32 or M64 buckets, jobs, parameters, preparation and combine
dispatches to its I8 control. It adds no hit-only job lists or prefix
dispatches. Synthetic fixtures retain the frozen oracle's BF16 hidden
values, original I64 routes and unequal route weights.

The GPU oracle maps only one certified Full512 I8 layer of 2,524,446,720
bytes and a 16KiB rank allocation. Three BF16 code planes contain
5,033,164,800 logical bytes. Each receives a 64-byte tail guard and rounds
to a 16KiB allocation, totaling 5,033,213,952 allocated cache bytes. The
memory governor admits this cache independently before allocating it, in
addition to the original one-layer plan and 1GiB scratch reservation. The
bounded total is 8,631,418,880 bytes, protecting at least 16GiB or 10% of
host RAM. It excludes the full model and production 48-layer store
constructor. Physical rows are restricted to 1024–2048.

Strict C++ and Metal 4.1 compilation and CPU checks read no model payloads
and create no Metal device:

```sh
make -f dev/benchmarks/prefill_moe_sep21/bf16_codes/Makefile -j4 cpu-self-test
```

The build generates `candidate.metal` from `generate_kernel.py` and the
parent `memory.metal` source, then links its AIR with the original parent
AIR inventory. It requires no private memory, register or gathered AIRs.
`all` only builds; `cpu-self-test` also runs `--cpu-self-test`.

Prepare all four variants without executing GPU work:

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/bf16_codes/run.py \
  --rows 2048 --pairs 4 --pattern spread-all --strict \
  --report build/prefill-moe-sep21-bf16-codes/spread-all-new.json
```

Add `--run` only in the exclusive root GPU window. `--variant 1` through
`--variant 4` selects one candidate; omission screens all four. Patterns
are `hit-concentrated`, `hit-spread` and `spread-all`. Raw BF16 hidden/I64
IDs are accepted together with `--input` and `--ids`. Synthetic inputs use
the inherited divisor 512 by default; `--normalized` sets
`PREFILL_MOE_BF16_CODES_NORMALIZED=1` and uses divisor 74. Normalization
cannot be combined with raw input.

Conversion uses a 256-word lookup of exact signed-I8-to-BF16 values. GPU
execution independently verifies every cached word before and after
timing. Conversion, verification and payload checks remain outside
complete-chain timings. Every candidate reports complete BF16 equality
and errors for activated, scattered-down and combined output before
timing, against relative L2 ≤0.001 and cosine ≥0.999999. Strict mode
rejects any complete BF16 difference; rejected candidates retain a
failure report with no timing samples. Numerical alternatives may be
timed after passing the guards when strict mode is omitted.

This is primitive qualification, not model-quality or full-worker
qualification. A complete-chain speedup greater than 25% is required
before proceeding to a full-worker Hot64 code-cache experiment of roughly
30GB. The runner records code, binary/metallib and manifest hashes plus
the exact command and sanitized controls. Preparing an invocation reads
no input, IDs or model payloads.
