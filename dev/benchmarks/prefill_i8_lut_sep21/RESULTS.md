# Staged I8 LUT branch closed

Root qualified the original-Q4-ID plus 16-entry signed-I8 LUT representation on
the bounded layer-0 pack. Every saved I8 code and original late F32 scale byte
was reconstructed/copied exactly. The packed layer is 1,895,301,120 bytes;
coefficient payload is 48 instead of 64 bytes per G64 group. No full store was
created.

V2's six native malformed-ID cases passed. At 2048 true-RMS spread rows, SG2K128
I8 and BF16 shared-tile candidates both passed raw F32 dots, scaled F32, scaled
BF16 boundaries, complete BF16 activation/scatter/combine parity and malformed
guards. Representation/primitive fidelity is distinct from model qualification.

| Complete chain | Current best SG2K128 | Matched uncompressed shared tile | Compressed LUT shared tile |
| --- | ---: | ---: | ---: |
| I8 B | 5.38 ms | 10.2 ms | 15.3 ms |
| BF16 B | 5.397 ms | 13.931 ms | 22.204 ms |

I8 gate/up measured 2.94/6.63/10.26 ms, and down 1.37/2.55/4.00 ms, in the same
current-best/matched/LUT order. Reports are
`build/release/flash/sep21-i8-lut-r2k-v2.json` and
`build/release/flash/sep21-i8-lut-bf16-stage-r2k-v2.json`.

Source identifies concrete extra work without establishing a hardware cause:

- K128 gate/up has 80 explicit threadgroup barriers (four per block); down has 10.
  Gate/up sequentially overwrite and reuse one B tile. MPP may synchronize
  internally as well.
- Both staged paths add device-to-threadgroup writes followed by MPP B reads.
  Current best uses device operands directly. Complete M32 control tiles have
  static A/B extents, while staged A always has dynamic extents; this comparison
  does not isolate address space alone.
- The compressed path has eight LUT-index and store expressions per aligned
  eight-code chunk. Exact generated instruction counts are unknown.
- BF16 doubles shared B from 8 to 16 KiB at K128, adds exact I8→BF16 casts, and
  doubles staged element width. Bandwidth/occupancy effects remain hypotheses.

The SDK supports BF16×I8→F32 and BF16×BF16→F32. There is no evidence from these
headers that I8 threadgroup operands cause unsupported arithmetic or a software
fallback. Shader/register topology, cache activity and limiter counters were
not collected.

Already compiled K256 variants halve gate barriers to 40 and reduce down to 6;
SG4 halves staging chunks per thread. Those changes do not demonstrate recovery
from the measured regression. Root closed this staging topology after both
dtype tests. No additional GPU screen, 90-GB sidecar or whole-model worker is
warranted from these results.
