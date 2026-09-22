This isolated screen tests scheduling locality on the installed Flash Next
checkpoint's layer 0 MoE Q4/G64 expert projection planes. It uses the current
production arithmetic helpers unchanged, with output-fast, row-fast,
selection-fast, route-fast, row swizzles of 2/4/8, and selection swizzles of 2/4.
It does not assign work or memory to a GPU die. V2 also tests expert-sorted
route jobs generated on the GPU, followed by projection/scatter in either
output-fast or route-fast order. A single GPU prelude computes stable ranks
by expert ID with the canonical route as the tie-break, costing O(routes²)
comparisons at a maximum of 160 verifier routes. The prelude and job buffer
access are included before every measured sorted projection.

Build and CPU-only mapping coverage check:

```sh
make -f dev/benchmarks/ultra_quantized_locality/Makefile -j4 all
build/ultra-quantized-locality/oracle --cpu-self-test
```

Run under the root agent's exclusive GPU slot. The current profile uses the
contiguous explicit F32 coefficient kernels and ten selected experts:

```sh
build/ultra-quantized-locality/oracle --families contig --rows 1,4,16 --selections 10 --samples 9 --repeat 8 --out build/release/ultra-quantized-locality-contig-v2.json
```

Focused GPU-sorted job comparison, including a shifted-slot router pattern
that retains seven of ten experts and changes the remaining experts per row:

```sh
build/ultra-quantized-locality/oracle --families contig --rows 1,4,16 --selections 10 --patterns shared,shifted,disjoint --sorted-only 1 --samples 11 --repeat 1 --out build/release/ultra-quantized-locality-sorted-v2.json
```

The default metallib path is the sibling of the executable, allowing the root
to freeze complete executable/library pairs before further candidate builds.

The original affine control can separately test one and ten selected experts:

```sh
build/ultra-quantized-locality/oracle --families affine --rows 1,4,16 --selections 1,10 --samples 9 --repeat 8 --out build/release/ultra-quantized-locality-affine-v2.json
```

Only the requested gate, up, and down tensor ranges are mapped from the derived
model package, with `PROT_READ`, page alignment, bounds and layout checks. No
full model is constructed. Router patterns cover shared IDs across rows and
different IDs across rows, plus the shifted-slot overlap pattern. Only selected expert source bytes are faulted in or
hashed. All candidate outputs must match every production BF16 word and every
per-route argmax for random and cancellation inputs. Guard regions, diagnostics,
producer bytes and selected source bytes are checked. GPU-generated route jobs
must match an independent CPU stable sort over every canonical route, including
expert ties. The independent FP64
numerical check samples at most eight routes and sixteen output columns; it is
not an end-to-end model correctness claim.

Timing samples rotate the candidate order and record both GPU and wall time.
Each command repeats a projection to reduce host timing noise, so these are
warm projection timings. The repetitions retain the same source experts and
should be followed by full-request measurements before adopting a traversal.

The root's frozen V1 contig screen at nine samples and eight repeats passed
full BF16 and argmax equivalence across all 18 cases. At four verifier rows,
the best remapping improved gate/up projections by at most 0.75%; down had no
measured improvement. The largest gain at sixteen disjoint rows was about 3%
for gate/up. These results do not justify a production traversal change.
Evidence: `build/release/ultra-quantized-locality-contig-v1-frozen.json`.

The root's V2 sorted-job screen at eleven samples and one repeat passed every
BF16 word, per-route argmax, and independently sorted route ownership across
all 27 cases. Four-row shifted-slot timings were 37.12µs for the original gate
projection versus 44.13µs with GPU sorting, 66.87µs versus 79.21µs for up, and
81.50µs versus 95.46µs for down. Sorting/scatter added about 17–19% to these
projections. Shared and disjoint patterns also slowed. This candidate remains
experimental and does not justify a production change.
Evidence: `build/release/ultra-quantized-locality-sorted-v2.json`.

The follow-up with eight repeats also passed all 27 cases and found no sorted
variant faster than the original. At four shifted rows, gate was 40.31µs
original versus 48.63µs sorted, up 25.16µs versus 30.36µs, and down 38.32µs
versus 45.55µs. The GPU prelude and additional job accesses remain a net cost
even with warmed expert weights. Each V2 run checked 19,353,600 BF16 values.
Evidence: `build/release/ultra-quantized-locality-sorted-v2-warm.json`.

If revisiting expert locality, an explicit weight-reuse tile that processes
several same-expert routes may be a stronger lever than only reordering work.
That would require a new arithmetic/occupancy oracle and full-request timing;
it is not implemented in this experiment.
