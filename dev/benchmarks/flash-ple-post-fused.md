PLE post-projection fusion candidate
===================================

`FlashPLEPostFused` combines seven original ordered dispatches into three.
The embedding hash/gather and affine key/value projections precede this
operator and are independent of this candidate.

`addPLEPostProjectAndInject` selects this candidate when the process starts
with `SPLASH_FLASH_PLE_POST_FUSED=1`. Missing/0 retains the existing separate
post/injection graph; other values reject. The route is frozen after its first
read, and `flashPLEPostRouteSemantics()` reports that actual choice. This
entry point retains the existing post arguments and injects into the exact
hyper-input view; it does not allocate additional model or request storage.

| Original operations | Candidate dispatch | Numerical boundary |
| --- | --- | --- |
| Key RMS, query RMS, dot gate, gated-value RMS | `flash_ple_post_norm_gate_fused` | Separate F32 RMS sums; BF16 normalized operands, dot local folds, SIMD levels, nonlinear gate arrays, gated values, convolution normalization |
| Depthwise convolution, SiLU, PLE residual, hyper injection | `flash_ple_post_convolution_inject_fused` | F32 four-tap accumulation; BF16 convolution, sigmoid, activation, PLE residual, injection |
| Nine-row convolution carry | Existing `flash_ple_update_convolution_state` | Exact BF16 copies in chronological order |

Each first-dispatch threadgroup owns one token and hyper stream. Its key and
query RMS sums follow the existing `FlashHC` four-contiguous-column traversal
and two SIMD levels, with separate sums that share barriers. The actual source
width 2560 uses 640 threads in both RMS and dot gate. For other widths, the
launch uses the larger of the source RMS/gate thread counts and limits each
reduction to its original active partition. Widths at most 64 retain the
source's sequential BF16 dot fold.

Gamma may independently be BF16 or F32 in each of the three norm tensors.
`OnePlusWeight` forms F32 `1+raw_weight`; `DirectGamma` uses F32 `raw_weight`.
Each normalized value rounds to BF16 before it participates in the dot gate.
The standalone gate sigmoid uses precise exp and the convolution SiLU sigmoid
retains the original fast-exp helper and BF16 promotions.

The first group writes every gated BF16 value before a device-memory barrier.
It then reads its own completed row using the source RMS traversal and writes
`normalizedConvolution`. Later-row convolution cannot be fused into that
group without cross-group synchronization, so it remains a later dispatch.
`normalizedConvolution` stays materialized for exact retained-prefix state
restoration, including speculative windows of sixteen rows.

The convolution dispatch only reads the previous nine-row state. It performs
the original four taps at dilation three in the same order, then the same BF16
SiLU/add/inject boundaries. Injection can overwrite the exact input hyper
view because each thread reads only its own hyper element and earlier query
normalization is complete. The ordered existing state update prevents parallel
rows from racing over history.

Qualification compares the candidate to the existing native operator bit for
bit, including all four scratch arrays, PLE/injected output, masked/nonzero
history, retained prefixes, continuation, diagnostics, and guards. A native
exact comparison establishes preservation of the previously qualified route;
it does not establish a new independent model-level accuracy result or HTTP
speedup. GPU qualification and timing belong to the Root runner only.

Root qualification completed exact preservation for 28 checkpoint
gamma/convolution cases at width 2560, B1..4 and R1..2048, including in-place
injection and retained prefixes. The uninstrumented paired whole-operator
timings typically improved 1.2–1.35x; the slowest source case was 1.17x.
`build/release/flash/ple-post-fused-full-source.json` retains the measurements.
Another 144 generic-width/row/lane cases passed with Metal Shader Validation
in `build/release/flash/ple-post-fused-generic-shader-validation.json`; those
instrumented timings establish correctness rather than deployment speed.

The CPU-only `flash_ple_post_profile_test.cpp` passed eleven isolated literal,
freeze, route-tag and empty-graph rejection cases without creating a backend
or issuing GPU commands.

The isolated executable takes `METALLIB REPORT_JSON [--package PACKAGE]`.
`--cpu-self-test` runs 72,843 host checks without creating a Metal backend or
submitting GPU commands. A first GPU screen is:

```sh
FLASH_PLE_POST_WIDTHS=2560 FLASH_PLE_POST_ROWS=1,4,8,16 \
FLASH_PLE_POST_REPEATS=5 FLASH_PLE_POST_WARMUP=2 \
build/flash-ple-post-fused/flash-ple-post-fused-oracle \
build/flash-ple-post-fused/splash.metallib \
build/release/flash/ple-post-fused-small.json
```

Leaving `FLASH_PLE_POST_ROWS` unset covers 1/4/8/16/128/512/2048 rows per lane.
Leaving `FLASH_PLE_POST_WIDTHS` unset also covers reduction-partition/tail
traps at widths 32/65/128/513/768/1025/2563/4097 with four real rows. The default
compact sweep varies all four lane counts, gamma conventions/dtypes, history,
and masking; `FLASH_PLE_POST_FULL=1` expands the selected product. Prefix and
in-place injection qualifications default on. Masked NaN/infinite projected
values separately prove that the numeric diagnostic remains sticky while the
masked output is finite. Timings alternate the paired baseline/candidate order
and exclude state reset and host validation.
