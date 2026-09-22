This private W8A8 screen quantizes BF16 activations per row while retaining
the original signed I8 coefficient bytes and F32 coefficient row scales.
Both variants use native M32N64 jobs and whole dynamic-K reduction, with
SG4 (`m32_n64_sg4`) or SG2 (`m32_n64_sg2`). Their gate/up and down/scatter
names begin with `prefill_moe_sep21_w8a8_`; independent producer audits
use an `_audit` suffix. Original buckets, jobs, parameters, preparation,
scatter and combine behavior stay unchanged. There are no extra hit lists
or converted coefficient caches.

Each GPU quantizer finds a row's maximum absolute BF16 value, writes
signed I8 values using round-to-nearest-even and clamps to [-127,127],
with F32 scale `maxabs/127`. Zero rows use scale 1. Nonfinite inputs
sanitize to zero and set diagnostic bit 4. Gate quantization uses
`prefill_moe_sep21_w8a8_quantize_gate_t256`; down quantization uses
`prefill_moe_sep21_w8a8_quantize_down_t128`. Both GPU quantization
commands are included in complete-chain timings.

Synthetic hidden fixtures use true per-row RMS normalization with final
BF16 rounding by default. The report records the measured input RMS
minimum and maximum. Raw BF16 hidden/I64 route fixtures are accepted
together with `--input` and `--ids`; the runner reads neither payload.
Unequal route weights and independent native bucket references remain.

Before timing, the oracle independently checks every quantized I8 value
and F32 scale on the CPU for all active rows. Mandatory samples check
exact raw I32 dots, scaled F32 dots, and an F64 quantization envelope
against original BF16 activations, original I8 coefficients and original
F32 coefficient row scales. Scaled F32 and raw I32 audits cover gate,
up and canonical down outputs. Audit kernels and CPU verification are
excluded from complete-chain timings.

The preregistered guards require relative L2 ≤0.05 and cosine ≥0.9985
for the complete activated, scattered-down and combined BF16 outputs.
Full BF16 equality is also reported. `--strict` requires complete BF16
equality before timing; rejected candidates retain their failure report
with no timing samples. This experiment is always labeled a numerical
alternative. Semantic quality, model quality and MTP acceptance remain
unqualified.

The oracle maps one certified 2,524,446,720-byte Full512 I8 layer plus its
16KiB rank buffer and keeps the original 1GiB scratch reservation.
Quantized gate/down activations and their F32 row scales have
`rows*10+63` rows, with K2560/K640 respectively. F32 scaled-output and
I32 raw-dot audits have `rows*10` rows and N640/N640/N2560 for gate,
up and down. The memory governor separately admits every guarded
quantization and audit allocation before construction. Exact new byte
counts come from `cache.hpp` and are recorded by the runtime oracle for
the selected row extent; the runner derives the same ten allocations
with a 64-byte guard and 16KiB rounding for its invocation witness.
It protects at least 16GiB or 10% of host RAM,
excludes the full model and production 48-layer constructor, and bounds
physical rows to 1024–2048.

Strict C++ and Metal 4.1 compilation and CPU self-tests read no model
payloads and create no Metal device:

```sh
make -f dev/benchmarks/prefill_moe_sep21/w8a8/Makefile -j4 cpu-self-test
```

The build generates `candidate.metal` from its positional shader
generator and the parent `memory.metal` template, then links candidate
AIR with the original parent AIR inventory. It requires no private
memory, register or gathered AIRs. `all` only builds; `cpu-self-test`
also runs `--cpu-self-test`.

Prepare both variants without executing GPU work:

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/w8a8/run.py \
  --rows 2048 --pairs 4 --pattern spread-all \
  --report build/prefill-moe-sep21-w8a8/spread-all-new.json
```

Add `--run` only in the exclusive root GPU window. `--variant 1` or
`--variant 2` selects one candidate; omission screens both. Patterns are
`hit-concentrated`, `hit-spread` and `spread-all`. The runner records
source, binary/metallib and manifest hashes, exact controls and command,
and copies measured RMS statistics from GPU report metadata when run.
Preparation reads no input, IDs or model payloads.

## Sealed whole-worker selector

`build/prefill-moe-sep21-w8a8-pointwise-worker-v1` combines the qualified
pointwise snapshot with this activation-quantized main-prefill producer.
Use `SPLASH_FLASH_PREFILL_MOE_W8A8=2` for SG2, `1` for SG4, and unset/`0`
for the original graph and numerical identity. Main, nonverification,
canonical Full512 R2048/M32 calls select W8A8. Other row shapes, decoding,
verification and the trained MTP bank retain their existing producers.
The selector is a numerical experiment requiring actual model semantics and
MTP acceptance checks before profile promotion.

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/w8a8/worker_overlay.py
make -f dev/benchmarks/prefill_moe_sep21/w8a8/worker.mk -j3 all cpu-self-test
```

The snapshot copies and verifies all 240 pointwise source seals, freezes
117 link inputs, and adds three private files. It links its own Store,
Forward and Worker objects with the sealed inherited objects; it does not
link by wildcard from a partial host directory. The original BF16-A Store
methods and shared primitive W8A8 arithmetic/quantizers are independently
checked unchanged. New worker producer names have contiguous binding ABI:
gate activation scale11/parameters12, down activation scale10/parameters11.
These binding-only wrappers use the same audited math.

Only four governed quantization buffers are added. Planned/logical bytes
are 65,945,600/65,901,944 at maxRows2048, and 263,045,120/263,001,464 at
maxRows8192. `FlashForward::workspacePlannedBytes` includes the matching
maximum-row allocation before construction. No weight cache or normal-service
audit buffer is introduced. Both GPU quantizers remain in the timed graph.

Active arithmetic is bound into kernel routes and the target numerical
derivative. Runtime `prefill_w8a8_route_counters` live outside identity,
report graph construction rather than GPU completion, and expose actual
workspace admission/allocation and logical rows. The sealed CPU witness is
`cpu-sealed-witness-v1.json` in the build directory. Source compilation,
full worker/helper CPU checks and seal verification passed; GPU/model
qualification belongs to Root.
