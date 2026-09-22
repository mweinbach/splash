# Optional ordinary R1 vector I8 decode worker

The isolated worker at
`build/gemv-decode-r1-pointwise-sg2tail-sep21-worker-v1` composes over the sealed
pure I8 pointwise/SG2-K128 tail/FMA worker. The strict frozen switch is
`SPLASH_FLASH_GEMV_DECODE_R1_SEP21=0|1`; missing is disabled. Enabled startup
requires the existing gathered MPP switch enabled. All inherited launch
requirements remain unchanged.

Only the ordinary autoregressive singleton call uses the new `forwardDecode`
wrapper. Existing `forward()` calls preserve all prompt windows, including
singleton prompts and one-row tails, plus MTP seed calls. `verify()` and every
batch executor remain unchanged. The wrapper requires one token after a
nonempty prompt and passes an explicit ordinary-decode bit to the existing
trunk/state implementation. The selector requires that bit, original gathered
eligibility, physical rows1, nonverification and the frozen vector flag.

The new Store methods reuse the original canonical hidden, I64 IDs,
intermediate and down buffers, Full512 coefficient/scale/rank views and
immutable/disjointness guards. Gate/up dispatch is `{160,1,10}`, down is
`{640,1,10}`, both128 threads. Original gather, blocked, prefill SG2/tail,
combine layout, MTP sources and workspace/request/verification planners are
preserved. No GPU allocation, coefficient cache or execution scratch is added.
Separate counters identify graph construction rather than GPU completion.

Shipping Metal source is an extraction of the sealed v1b vector source: all
arithmetic helpers and the two SIMD32/O4 timed producer invocations remain
identical; only the ABI name/include and geometry restriction to physical R1
change. L16 producers and all untimed tap/scalar entrypoints are absent from
the shipping AIR. The numerical-alternative marker binds the original sealed
kernel, FTZ/RTZ certificate, preregistration, proof document, shipping extraction,
ABI, bridge and ordinary-decode policy schema. Disabled numerical identity is
the original base identity.

Root's synthetic R1 one-layer report qualifies the numerical alternative
component only. The mixed R4 alternative failed its frozen stage gate and is
excluded. This worker makes no full-model quality or performance claim.
Root must complete matched standard decode measurements and the frozen22-case
semantic suite before considering promotion.

```sh
python3 dev/benchmarks/gemv_decode_r1_worker_sep21/worker_prepare.py
make -f dev/benchmarks/gemv_decode_r1_worker_sep21/worker.mk -j4 all
python3 dev/benchmarks/gemv_decode_r1_worker_sep21/worker_witness.py \
  --output build/gemv-decode-r1-pointwise-sg2tail-sep21-worker-v1/NEW-witness.json
```

The subtree completed warnings-as-errors host/Metal compilation, independent
strict-policy CPU tests over rows1..8192, original worker CPU checks, source
closure and fail-before-path flag tests. It performed no GPU work, model loads
or model payload reads and modified no production sources.
