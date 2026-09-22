# Physical I8 RHS tile transpose, bounded component only

The current R4 expert producer source presents B as `tensor{K,64}` with
strides `{1,K}`. Its tile base is `(rank*Width + tile*64)*K`. A plain
`[rank,Ntile,64,K]` repack would reproduce the existing byte order.

This experiment instead stores `[rank,Ntile,K,64]`: within the unchanged tile
base, byte `nLocal*K+k` moves to `k*64+nLocal`. The corrected tensor is
`{64,K}` with strides `{1,64}` and transpose-right false. Every logical signed
I8 coefficient remains identical. F32 scales
still use `rank*Width + tile*64+nLocal`.

Keep M16/N64/dynamic-K/multiply/SG4, adjusting transpose-right to match the
changed physical tensor axes (old `{K,64}`/`{1,K}` uses transpose-right true),
device-A and malformed-input TG-A paths, raw/scaled F32 values, BF16 casts,
sigmoid/Silu sequence, diagnostics, masking and rank rules. Change no K loop,
type, LUT, scalar dot association or tile geometry. Opaque MPP can lower a
different stride differently; logical equality is not proof of F32 bit parity.

The old diagnostic measured approximately 7.159 ms gate/up plus 2.999 ms down
across 48 layers. Routing/combine add approximately 1.767 ms and are separate.
The current integer compactor already reduces native setup to six dispatches;
that closed optimization is not part of this experiment.

R4 useful arithmetic is 18.874368 GFLOP per 48-layer pass. Unique I8 code and
F32 scale source payload is `236,666,880 * U` bytes per pass, where mean expert
union U is in [10,40]. Descriptor output coverage is 1/16 for the gathered
single-row operation and `2.5/U` for bucket-native M16 jobs. These are logical
coverage ratios; inactive arithmetic and hardware utilization were not measured.

A 20% throughput increase at the current 46.11 ms cycle requires 7.685 ms less
latency. Even using the older 10.158 ms expert budget as a ceiling, producers
must reach 2.473 ms. The archived 1,173.09 GB/s resident-read payload proxy
would require U at most about 12.26 and near that proxy rate. Archived U=24.5156
is only a sensitivity, not current routing evidence. No 20% gain is promised.

The hypothesis is improved coefficient transaction locality or MPP lowering
from contiguous N bytes within a K slice. It remains an unmeasured hypothesis.

First qualification uses one Root-read original expert: three code banks total
4,915,200 bytes and three F32 scale banks total 15,360 bytes (4,930,560 bytes).
No whole-layer 2.524 GB mapping or whole-store repack is required. The fixture
has a 512-entry in-bounds rank table and ten distinct IDs per row mapping rank0.
It is a repeated-one-expert synthetic routing primitive, not real Full512 top10.
Production Full512 host validators remain unchanged and are not reused with a
short bank. Timing this hot small fixture cannot certify whole-model bandwidth.

Before timing require a CPU bijection and offset proof, exact original AIR
compiler policy, no private weak-symbol collisions, original untapped control
versus reference taps, candidate untapped versus candidate taps, full raw and
scaled F32 bit parity, BF16 gate/up/activation/down parity, immutable old/new
codes and scales, sticky diagnostics, canaries, nonfinite/TG fallback and rank
guards. The host rejects an omitted valid grid; a shader cannot detect omitted
groups. Warm each shipping chain for at least 150 ms GPU and use 18 position
balanced shipping-only samples with no buffer readback between warm and timing.

Only Root may read coefficient/input payloads or run the GPU component. Agents
prepare and audit source/program artifacts only. Whole-worker integration is
outside this bounded authorization.
