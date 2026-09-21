# Independent CPU audit of the next v5 optimization round

This audit reads completed local reports; it performs no inference, GPU work,
GitHub writes, or default changes. [Decision data](../../build/release/flash/v5-next-round-independent-cpu-summary.json)
binds raw report hashes, source metadata checks, and qualification scope.
[Exact-prefill follow-up](../../build/release/flash/v5-next-exact-prefill-cpu-analysis.json)
records the subsequently completed full-model run.

Matched HTTP medians are complete-wave output tokens/s, with greedy
reasoning-off requests, zero cached tokens, exact 128/2K prompts, all 128 output
tokens per request, warmup and two measured samples on M5 Ultra/256 GiB.

| Policy | 128 C1 | 128 C4 | 2K C1 | 2K C4 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| This-round v5 baseline | 99.098 | 151.578 | 64.618 | 80.041 | Retain |
| Batch window 1,024 | 103.529 | 159.124 | 66.342 | 78.541 | Disabled; long C4−1.875% |
| Original-text residency | 101.002 | 152.998 | 64.411 | 77.853 | Disabled; long C4−2.733% |
| Exact shared/output prefill | 100.931 | 154.465 | 65.651 | 78.537 | Opt-in; long C4−1.879% |

All three candidates passed 22 bounded quality/protocol cases, and all 21
performance requests—including warmup—have exact text, prompt hash, usage,
finish and validation equality against this-round baseline. The window and
original-text candidates also retain 22/22 exact quality-case text equality.
These finite checks do not establish general model quality. Their performance
does not justify new defaults on this completed evidence.

The B1 stage trace has complete classification across 48 model layers and 2,993
timed dispatches: MoE 46.80%, dense 22.43%, QSA 14.64%, GDN 9.29%, HC 3.37%, and
shared experts 2.02% of the sampled command GPU duration. Stage profiling
changes encoder boundaries; these are descriptive fractions, not throughput.

| Primitive candidate | Evidence | Practical interpretation |
| --- | --- | --- |
| Shared expert M32N128 fusion | Both R2,048/8,192 cases exact for dots, activation and complete chain;4→2 dispatches;1.570×/1.315× isolated GPU speedup | Shared family is only 2.02% of B1 sampled GPU time. R2K latency reduction implies approximately 0.73% illustrative prefill potential, not a full-model gain. |
| Dense output M64N128 |24 source-operand cases byte-exact; GDN-output speedups1.524×/1.189×/1.109× at512/2,048/8,192 rows | Role gating matters: wide input/QKV and HC shapes often lose. N2,560/K6,144 outputs account for 48 calls and 4.38% of B1 sampled GPU duration, implying approximately 0.70% illustrative prefill potential atR2K. |
| QMV-only proposal head |0/17 synthetic cases pass unchanged 1% bound; one real 2K/128-output AR pairing passes,119/119 drafts accepted | Keep experimental; target parity is bounded and proposal acceptance/HTTP performance remains separate. |
| F32 proposal control |Also0/17 synthetic-bound passes; same 288/288 repeated argmax matches as QMV | Synthetic failure does not uniquely identify QMV. Reconcile reference/precision contract without weakening the bound to promote a default. |
| GDN V32/T32/SG8 ILP layout |32/32 warm-state cases byte-exact; B4 speedups1.145×/1.193× at512/2,048 rows; B1 slightly slower | Continue B4-specific qualification. Cold state, extreme gates, tails, Shader Validation and full-model evidence are absent from this screen. |

The shared and dense output estimates combine to only approximately1.43% of
the B1 sampled prefill GPU duration. They multiply real descriptive family
fractions by synthetic primitive reductions; they are prioritization examples,
not predictions or measured HTTP improvements. Larger opportunities remain
in MoE and QSA, whose sampled shares are much larger.

Both proposal screens retain finite, unchanged inputs, 90 lifecycle guards and
288/288 repeated argmax matches, but QMV hidden relative L2 is 1.50–5.36% and
F32 control 1.39–5.36%. The F32 control also reaches 1.086% continuation and
3.594% rollback-overwrite error. Exact repeated argmax alone does not satisfy
the stated hidden/state qualification. Real-prompt QMV parity is an offline
paired probe, not matched HTTP throughput.

Fresh GDN ABI provenance records `sizeof(CommandTiming)=200`, rebuilt private
runtime objects, executable/metallib/source hashes and finite valid timings.
All eight geometries fit device thread/TGM limits, and all 32 cases preserve
intermediates, BF16 output, F32 state, history and padding exactly. This is
legitimate primitive evidence; it does not supply the missing cold/extreme or
whole-service qualification.

In the actual exact-prefill service, true target-trunk GPU time grows
9.025→9.226seconds, while command wall grows10.776→10.864seconds. The
commit-to-scheduled callback interval falls1.739→1.628seconds, but the
200 ms GPU increase offsets that 111 ms interval decrease. Head-prime and decode
GPU timing also increase in otherwise unchanged phases. No unique cause can
be established from these two samples. Inclusive prefill already includes
head priming; subtract it exactly once. Named host intervals overlap, and a
scheduled callback measures CPU handler arrival rather than actual GPU start.

The original config, tokenizer, tokenizer config and safetensors index still
match their aligned copies byte-for-byte. The aligned manifest remains SHA
`0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402`,
with a matching digest file, and reports retain original source/layout
identities. This CPU audit did not rescan complete multi-GB checkpoint payloads
while root GPU measurements were running. Private proposal reports explicitly
record no original-model or production-route modifications.
