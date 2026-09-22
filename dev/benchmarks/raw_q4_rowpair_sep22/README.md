# One raw Q4 two-row projection component

The source/proof plan is [decode_strategy_refresh_sep22.md](../decode_strategy_refresh_sep22.md).
Root approved CPU preparation only. There is no worker route, cache, new default,
GPU qualification or present performance result.

The initial domain is one original layer1 GDN QKV projection: physical R4,
K2560/N10240/Q4/G64/dense flags0. The paired implementation keeps the original
F32XSUM `load_vector[16]`, separate row qdot accumulators, parenthesized Q4 sums,
chunk `result+=`, SIMD sum and BF16 boundary. It shares integer packed words/masks
and exact BF16 scale/bias conversion only. Old5120 CTAs become2560,64threads each.
There is no weight cache, sidecar, extra normal workspace or extra timed dispatch.
The older primary trace gives this first domain only1.352415ms; a2x component
gain would be around1.5% of the current cycle. Wider raw projection coverage is
a separate proposal, not authorized or measured.

CPU build into a fresh directory:

```sh
.venv/bin/python -B dev/benchmarks/raw_q4_rowpair_sep22/prepare.py \
  --build build/NEW_RAW_Q4_ROWPAIR_COMPONENT_DIRECTORY
```

The builder copies only source/code/Core4 objects and the authenticated actual
native QMV AIR. It compiles with Metal4.1/O3 and the native default fast policy,
retaining original safe source pragmas. The primitive library contains the eight
immutable original QMV entries and three private entries. It is explicitly a
subset primitive library, not a replacement Worker library. The baseline timed
entry is the actual original shipping Q4 kernel from the unchanged AIR, not a
newly recomputed control. `LLVM-prefix.json` and retained `.ll` files expose all
kernel/helper floating instructions and attributes for independent review.
The nonempty transitive LLVM result is an ordered opcode/intrinsic/attribute
census after SSA operand erasure; it does not prove the dependency tree or GPU
register equivalence. Source-tap journals byte-restore original/control and
candidate timed bodies, and actual GPU equality remains mandatory.
F32 tap stores occur after production BF16/finite diagnostics/output stores.

`--cpu-self-test` and `--help` exit before all metadata/payload/device code. The
CPU tests cover ABI, complete dispatch geometry, extent overflow, writable
aliases, partial/overlarge grids, every output owner and all four integer masks.
They do not claim actual Metal alias/error or register proof.

Root-only GPU invocation after independent review and exclusive GPU scheduling,
with a fresh report path:

```sh
build/raw-q4-rowpair-sep22-component-v5/oracle --gpu \
  build/raw-q4-rowpair-sep22-component-v5/component.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/NEW_RAW_Q4_ROWPAIR_REPORT.json
```

This opens only layer1's three selected native tensor spans,14,745,600 bytes,
and labels the default deterministic BF16 input synthetic. Add one optional
`CURRENT_INPUT_METADATA_JSON` argument only for a Root-generated current R4
projection-input capture. Its schema is
`splash-raw-q4-rowpair-current-register-input-v1`, with exact canonical
`projection`, `rows:4`, `input_size:2560`, original `source_identity_sha256`,
`file`, `payload_sha256` and explicit `scope`. An older prefill fixture does not
become current MTP evidence by changing its metadata.

GPU timing requires full raw-F32/BF16 equality, actual shipping/probe parity,
per-row L2<=1e-4/cosine>=.999999, exact zero-reference behavior, sticky/canary and
immutable source checks. The unchanged `FlashFloatBoundaryAudit::f32DotBound`
is retained as sampled F64 **diagnostics**, with its failures explicit. A generic
coefficient-dot bound is not an original factored-QMV universal certificate and
does not override the mandatory equality/quality gates. No newly widened bound
or BF16 midpoint exception is introduced.

The64MiB governor reservation includes all selected native spans, BF16 inputs,
production/tapped outputs, raw-F32 taps, diagnostics and guards. Actual native
allocation delta and zero-buffer teardown are required. Each complete4-row
shipping/timed-candidate projection warms for>=150ms GPU, then18 balanced AB/BA
pairs retain every GPU/wall/position sample. No CPU buffer read/hash/diagnostic or
canary scan occurs between warmup and the last measured position. No Root run
is authorized by the mere existence of a CPU executable.

The frozen CPU artifact is currently v5. Its303142 CPU checks pass with no
device, metadata or payload access; its shader library is byte-identical to v4.
V5 fixes post-timing verification by snapshotting actual last measured BF16
outputs before resetting/replaying probes. Original/control opcode censuses are
117/117 and candidate/probe50/50, with dependency/register equivalence expressly
unproved. V4's command is superseded. V5's `run-root-synthetic.sh` and command
receipt remain unexecuted pending exclusive GPU release. Root independent CPU
source/artifact review passed; `READY-CPU-reviewed.json` and
`root-synthetic-command-reviewed-v2.json` bind the review and Root pins. The
approved scope is bounded synthetic GPU only. No actual current input/state or service qualification
is claimed. This live README update is documentation only; the v5 frozen code
snapshot and its manifest remain immutable.
