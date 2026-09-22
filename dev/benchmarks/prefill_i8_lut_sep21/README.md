# Exact saved-I8 LUT screen

This isolated component stores each original Q4/G64 group's 32 bytes of nibble
IDs plus a 16-entry signed-I8 LUT. Every reconstructed coefficient must equal
the certified saved I8 byte; original late F32 row-scale views remain bound.
The three compressed coefficient planes use 25% fewer bytes. This is a
one-layer primitive screen; `model_quality_qualified` remains false.

Packing and origin certificates are described in [PACKING.md](PACKING.md).
Preparation, SDK probes, compilation and CPU self-tests read source and bounded
JSON metadata only. Actual packing and `run.py --run` belong to the exclusive
root GPU experiment window.

```sh
.venv/bin/python dev/benchmarks/prefill_i8_lut_sep21/verify_sdk.py
make -f dev/benchmarks/prefill_i8_lut_sep21/component.mk -j4 all
make -f dev/benchmarks/prefill_i8_lut_sep21/component.mk cpu-self-test
```

The SDK rejects cooperative input/register B tensors spanning SG2 or SG4.
The eight runnable candidates use a cooperative **threadgroup** B tile, with
device BF16 A, M32/N64, SG2 or SG4, K128 or K256, and staged BF16 or I8 B.
The shader includes the whole-K SG4 and current-best SG2/K128 controls, plus
an uncompressed same-descriptor/staging/K-loop control for each candidate.
`sdk-compile.json` records the rejected register topology and the compiled
threadgroup topology.

The equality policy was fixed before GPU execution. Compression must preserve
raw F32 dots, scaled F32 values, scaled BF16 boundaries and the complete BF16
chain against its uncompressed same-descriptor control. Every candidate's
complete BF16 chain must also equal the current best SG2/K128 chain. SG2/K128
candidates must additionally equal the current best's raw/scaled arithmetic.
Whole-K SG4 comparisons remain visible as a contrast; its raw agreement is
not an eligibility requirement across different descriptors.

The default fixture contains 2,048 rows normalized by each row's actual RMS
before BF16 rounding. It screens spread-all (40 routes/expert) and concentrated
top-10 routing. Paired raw BF16 inputs and I64 route IDs can replace it with
`--input` and `--ids`.

```sh
.venv/bin/python dev/benchmarks/prefill_i8_lut_sep21/run.py \
  --pack-dir build/prefill-i8-lut-sep21/packed-layer-00 \
  --pairs 4 --pattern both \
  --report build/prefill-i8-lut-sep21/fresh-component-report.json
```

Without `--run`, this writes an invocation witness and executes no GPU work.
Adding `--run` executes the root-owned bounded experiment. Select one of the
eight variants with `--variant 1` through `--variant 8`; `--pattern spread-all`
or `--pattern hit-concentrated` selects one route pattern.
Pair counts must be even, from 2 through 32, to balance first/second ordering.

Each eligible candidate is timed as a complete expert chain, gate/up alone,
and down alone consuming the same original control's activated BF16 boundary.
Balanced pairs compare it with whole-SG4, current-best SG2/K128, and the matched
uncompressed staging control. Every arm accrues at least 150 ms of GPU warm
work immediately before its timing scope. No CPU diagnostic or output reads,
payload scans, hashes or writes occur between warm and timed submissions.
All audits, reconstructions, canary checks and immutable hashes stay outside
timing. Rejected candidates retain complete comparisons and receive no timing.

Untimed GPU negative cases cover IDs -1/512/duplicate, ranks UINTMAX/512,
expert/job bounds, job-count bounds, terminal offsets and canonical route
bounds. The report requires sticky error bits, unchanged poisoned producer
outputs and clean canaries. Compilation and CPU tests do not establish those
GPU results.

The loader bypasses production Full512 metadata validation while separately
binding the original chosen-layer descriptors to the pinned store manifest.
It maps one original I8 layer and one packed layer, retaining readonly mapping
lifetimes in every graph. It reserves those mappings, one canonical rank map,
5 GiB for bounded scratch/audits, and a host reserve of max(16 GiB, 10%). It
certifies every reconstructed source code and unchanged original F32 scales
before and after all timing scopes.
