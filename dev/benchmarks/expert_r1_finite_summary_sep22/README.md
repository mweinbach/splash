This private R1 component tests GPU finite summaries ahead of the unchanged
native I8 expert projections. The original two-dispatch control remains the
authenticated gathered MPP AIR. The candidate executes four fixed dispatches:
hidden summary, gate/up, activation summary, and down. M16N64, SG4/128 producer
threads, dynamic whole K, rank and duplicate validation, original tensor views,
late F32 scales, and BF16 projection/SwiGLU boundaries remain unchanged.

Each summary uses one SIMD32 threadgroup. Lane 0 resets the entire 64-byte
packet, a device barrier orders the reset, and the SIMD scans one hidden row
or ten activation routes using the original BF16 exponent-bit predicate.
Down summary skips an invalid route's activation read. Summary never adds
diagnostic bit 4. A valid finite flag skips the repeated producer scan and
retains the original device A pointer. Nonfinite or invalid packet metadata
invokes the literal original threadgroup scan/sanitize path, preserving its
rank-dependent diagnostic behavior.

The packet contains sixteen U32 words. Words 0–9 hold finite flags, with
0 meaning finite and 1 meaning nonfinite; inactive gate words 1–9 are zero.
Words 10–15 are magic 0x46535231, nonzero epoch, role 1/2, original rows 1,
K 2560/640, and reserved zero. Every candidate submission receives a fresh
inline epoch shared by its four dispatches, with wrap rejected. Consumers
read flags only. No CPU buffer read or write occurs during warmup or timing.
The fixed producer graph, immutable source views, complete packet overwrite,
and matching epoch establish freshness. This does not authenticate an
arbitrarily forged flag value carrying current metadata.

Original source counts are 100×2560 gate/up and 400×640 down BF16 scan words,
totaling 512,000 words and 1,000 producer scan barriers per R1 layer. Summaries
read 2,560+10×640=8,960 words. Finite consumers skip the original scan barriers.
These source counts do not measure the isolated scan budget or predict a
whole-model improvement; two extra dispatches may outweigh the saving.

Root alone opens the ten selected original expert coefficient ranges,
totaling 49,305,600 bytes, and executes GPU work. One 64-byte flag owner is
charged as 16 KiB within the normal 128 MiB fixture reservation. No Full512
payload is mapped or target model constructed. The synthetic normalized R1
input is a bounded primitive fixture, not current causal model input.

Before timing, Root must pass exact raw/scaled F32 and BF16 projection,
SwiGLU, own-chain, diagnostics, canaries, and immutable-operand checks; verify
the packet overwrite, stale positive/negative, role/domain/reserved/topology,
invalid-rank, nonfinite, and healthy replay cases; and measure actual current
and peak allocation plus zero-owner return and backend teardown. The complete
original two-dispatch and candidate four-dispatch chains each receive at least
150 ms of GPU warmup and eighteen balanced sample pairs. Final taps bind to
the last timed shipping output.

This CPU component build does not grant worker integration, standard-model
timing, or promotion. Those require Root's primitive result, current actual
input/state/output proof, and separate authorization. The axis-permutation
variant and uncertified old R1 vector alternative remain closed.
