The existing capped worker can screen actual batch verification without a new
binary. `FlashBatchVerify::verifyBatch` accepts1..4 lanes and1..4 real rows per
lane, then uses `flattened=lanes*rows` for every dense and target expert stage.
The current maxcap4 therefore selects four-row singleton verification but
leaves B2×4 and B4×4 verification on the bucket route. Maxcap8 selects B2×4;
maxcap16 selects B4×4. Singleton depth3 still submits at most4 physical rows.

The direct gathered MPP producer already computes one canonical row/expert
slot in each CTA. Splitting a16-row frame into per-lane4-row host dispatches
does not change that work or add weight reuse; it adds command dispatches.
Gathered gate/up plus down submit500 CTAs per physical row, so8000 at16 rows.
The bucket control also encodes the capped160-route job grid, but only its
nonempty expert jobs execute the full matrix operations. One existing actual
singleton16-row fixture has median49.5 unique experts per layer, range25..76,
from160 assignments. Direct gather therefore performs about3.23× as many
active expert-tile bodies at its median layer. This fixture does not represent
B4×4 diversity. Its source is
`build/release/flash/v6-verify-ctx2048-singleton-R16-stage.json.routes.jsonl`.

The synthetic16-row repeated-route component was slower with gather, whereas
rows1/2/4 components were faster and exact. Root should first run the existing
one-layer oracle at16 rows with spread/permuted patterns, then same-binary
actual B4×4 generation with maxcap16. This report does not establish a16-row
batch gain or authorize promotion.

A possible adaptive route avoids CPU readback by submitting a bounded GPU
predicate kernel over at most160 I64 IDs. It counts distinct experts and
writes one selector. Both expert routes then inspect the same selector before
performing work and emit the same canonical down plane for the unchanged
combine. Full bucket preprocessing must also be guarded; performing it before
choosing gather would erase much of the candidate's saving. The current
backend encodes host-owned group dimensions, so this approach still launches
both fixed CTA grids and inactive bodies return early. Avoiding those grids
requires the separate private indirect-dispatch backend ABI. A predicate also
needs admitted, retained scratch and a policy identity. No such arithmetic or
backend changes were implemented in this round.

The remaining dense opportunity is repeated input padding. The existing
original-F32 small-row helper adds `flash_float_dense_small_rows_pad` before
every projection, even when consecutive Q/K/V/index or QKV/Z projections use
the same input and compatible row/K/padding geometry. Reusing prepared A is
potentially exact because the pad kernel copies BF16 bits and writes positive
zero padding. A safe API needs one explicit preparation followed by several
projection consumers, with every current host view guard preserved. Broad
graph-level memoization is unsafe: HC-up and INT8 vocabulary helpers reuse and
overwrite the same arena, and the graph has no buffer read/write annotations.
At16 rows some M16 projections need no extra rows, but bypassing their copy
changes tensorA placement/alignment and still requires exact GPU parity.
The real four-row verifier attribution oracle should establish whether pad
dispatch cost is material before implementing this route.
