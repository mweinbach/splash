This private oracle screens three right-only M32N64 gate/up and down/scatter
candidates against the original native M32N64 signed-I8 control over one
shared, certified Full512 layer. Control and candidates use identical original
M32 buckets, jobs, parameters, route fixtures, preparation and combine
dispatches. The candidate producer pipelines use one SIMD group per job and
sequential N32 column halves. No extra hit lists, prefix dispatches or gathered
shaders are added.

The candidate keeps unchanged BF16 A in a device tensor with bounded logical
K and valid-row extents. It manually fills cooperative BF16 B registers from
the unchanged signed-I8 payload; this conversion is exact. MPP accumulates
into a zero-initialized F32 destination, then uses the original F32 post-dot
row scales and every original BF16 SwiGLU and output boundary. Since only B
is cooperative, the register/register K32 restriction does not apply: these
are actual fixed K64/K128/K256 MPP reduction operations.

| Variant | Descriptor K | Gate/up blocks (K2560) | Down blocks (K640) |
| --- | ---: | ---: | ---: |
| 1 | 64 | 40 | 10 |
| 2 | 128 | 20 | 5 |
| 3 | 256 | 10 | 3 |

K256 down uses a final K128 device-A tensor extent and masks cooperative B to
zero for indices k >= 640. The fixed K256 descriptor observes A's remaining
logical K extent. Every candidate must pass complete-output checks; successful
compilation alone does not establish numerical equality or performance.

The native adapter retains the frozen store oracle's BF16 hidden fixture,
original I64 route patterns, unequal route weights, independent bucket
reference and complete-output comparisons. It uses the existing one-layer
loader rather than the production 48-layer store constructor or full Q4
model. Full512 has no Q4 misses. Only one 2,524,446,720-byte payload and its
rank map are mapped. Admission reserves another 1GiB for both guarded scratch
sets, fixtures and transient guarded replacements, protecting at least 16GiB
or 10% of host RAM. Physical rows are restricted to 1024–2048.

Build and run the CPU-only checks:

```sh
make -f dev/benchmarks/prefill_moe_sep21/right_only/Makefile -j4 cpu-self-test
```

C++ and Metal 4.1 compilation are strict. CPU checks read no model payloads and
create no Metal device. The native AIR inventory is the same original parent
inventory as the one-layer baseline, with its signed-I8 control AIR. The only
private AIR is `build/prefill-moe-sep21/right_only.air`, compiled from
`dev/benchmarks/prefill_moe_sep21/right_only.metal`; private memory, both-input
register and gathered AIRs are excluded. `all` only builds; `cpu-self-test`
also runs `--cpu-self-test`.

Prepare an invocation for all three variants without executing GPU work:

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/right_only/run.py \
  --rows 2048 --pairs 4 --pattern spread-all --strict \
  --report build/prefill-moe-sep21-right-only/spread-all-new.json
```

Add `--run` only in the exclusive root GPU window. `--variant 1`, `2` or `3`
sets `PREFILL_MOE_SEP21_VARIANT` to screen one candidate; omission screens all
three. The default is Full512, a single layer, rows 2048 and `spread-all`.
Patterns are `hit-concentrated`, `hit-spread` and `spread-all`. Raw BF16 hidden
and I64 IDs are accepted together with `--input` and `--ids`. Synthetic hidden
inputs use the inherited divisor 512 by default; `--normalized` sets
`PREFILL_MOE_SEP21_NORMALIZED=1` and uses divisor 74. Normalization cannot be
combined with raw input.

Before timing, each variant reports equality of every activation,
scattered-down and combined BF16 element, and full-output errors against
relative L2 <= 0.001 and cosine >= 0.999999. Guard-passing numerical
alternatives may be timed unless `--strict` requires complete BF16 equality.
Rejected candidates remain in the report with their failure and no timing
samples. Warmed complete-chain GPU and wall timings rotate candidates and
alternate control/candidate order. Synthetic equality and error checks do
not establish model-quality qualification.

The runner records code, binary/metallib and manifest hashes, exact command,
sanitized controls, operand storage and actual descriptor shapes. Preparing
an invocation reads no input, IDs or model payloads. GPU execution alone maps
the admitted single layer; it never constructs or scans all 48 payloads.
