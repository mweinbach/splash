Current decode is limited mainly by GPU execution. With the current GPU costs
held fixed, removing every other cost would raise standard singleton35.82 to
38.67 tok/s, qualified R5 singleton61.54 to68.19, and qualified batch4 151.00
to166.17. Faster GPU kernels are required to approach larger traffic scenarios.

The context-matched original singleton decoded68.434 tok/s with80 cycles and
175 accepted drafts:255/80=3.1875 useful tokens/cycle. The best acceptance-
preserving FMA-only diagnostic decoded70.857 with79 cycles/176 drafts:
255/79=3.227848. Its3528.617 prefill rate remains below4K and its new setting
has not completed original22 qualification. FMA+HC measured70.765/3535.487.

All rates below are aggregate tok/s. “Unique” assumes optimistic source-operand
sharing; “stream” repeats historical raw-row/expert loads while retaining
explicit MPP row-tile reuse. Both use1173.0877 GB/s of **effective resident-array
payload**, not measured physical DRAM traffic or an attainable decode peak.

| Configuration | Useful tokens per lane/cycle | Unique scenario | Streaming reference | Observed decode |
| --- | ---: | ---: | ---: | ---: |
| Standard B1 |1|197.72|197.72|35.821, new setting unqualified|
| Standard B2 |1|275.04|209.65|No current matched qualified score substituted|
| Standard B4 |1|309.71|213.09|No current matched qualified score substituted|
| MTP3 B1 original control |3.187500|257.22|152.90|68.434|
| MTP3 B1 FMA-only diagnostic |3.227848|260.48|154.81|70.857|
| Qualified MTP3 B2 |3.227848|296.26|182.20|104.420|
| Qualified MTP3 B4 |3.109756|369.49|191.83|151.000|

Current B4 has82 native cohort graphs and1020 emitted post-first tokens, or
1020/(4*82)=3.109756 useful tokens per lane/cycle. The older78-cycle/708-draft
numerator overstated the current scenarios. B2 has79 graphs/510 tokens.
Current B4 prefill4094.042 has all three trials above4K; B2 prefill3619.698 is
below4K. Both current batch original22 comparisons have no new regressions.

The qualified4K singleton is a separate **MTP4/R5** configuration:61.544 decode,
4006.925 median prefill, all three prefill trials above4K, and original22 no new
regressions. Its79 cycles/176 drafts also yield3.227848 useful tokens/cycle, but
the four-row/depth3 denominator is not applied to its five-row verifier.
Current standard B1 prefill4043.737 is above4K in all trials; its new performance
setting remains unqualified. Its completed context-matched older control decoded
35.4897 with3233.602 prefill:current decode improved0.9345%, and prefill rose
to4043.737 with all three trials above4K. The significant decode goal remains
unmet. [Supplemental scalar evidence](/Users/mweinbach/Projects/splash/build/release/flash/sep22-standard-B1-matched-decode-roof-supplement-v1.json)
resolves the pending-control text in the preserved v4 report.

The exact inherited operand model uses2.614257 GB of target dense operands,
0.675430 GB vocabulary,2.366669 GB for ten selected I8 experts across48 layers,
and0.276824/0.503316 GB standard/MTP state floors per lane. Verifier F32 unique
operands are3.248/9.068/12.098 GB at R4/R8/R16. R16 explicitly requests22.345
GB across its M tiles. Expert unions assume independent uniform lanes and one
older temporal routing diagnostic; current physical transactions are unmeasured.

Compute evidence also rules out treating the traffic scenario as a hardware
guarantee. One real R5 stage diagnostic measured23.593 useful linear-equivalent
GFLOP of expert projections in9.274 ms (2.544 TF/s),113 large raw projections
in5.614 ms (3.885 TF/s), and13.998 padded F32-cache GFLOP in4.904 ms (2.855 TF/s).
These rates include kernel costs and omit unpack/scale/nonlinearity operations
from the FLOP numerator. The instrumented stage is not canonical throughput.

A favorable synthetic R16/u10 expert chain attained13.797 useful-equivalent
TF/s; mixed R16 and R8/u40 attained6.391 and2.623. Those shape/route measurements
do not establish a silicon peak. Scalar R5 supplies at most five valid rows to
an M16 expert tile:even complete expert sharing gives at most31.25% valid rows.

Holding measured acceptance and all other native costs fixed, a twofold target-
verifier improvement conditionally predicts99.24/167.96/241.38 tok/s for qualified
R5/B2/B4. Those gains require actual kernel improvements; changes in traffic/cache
reuse may also be necessary. No calibrated dtype/shape/job-occupancy compute peak
or physical DRAM counter measurement proves that these rates are attainable.

Full-precision calculations, source hashes and scope distinctions are in
[the new report](/Users/mweinbach/Projects/splash/build/release/flash/sep22-context-matched-decode-roofs-and-current-observations-v4.json).
[The CPU-only script](/Users/mweinbach/Projects/splash/dev/benchmarks/sep22_decode_roofs_context_refresh.py)
preserves every earlier report. Native common graphs support at most four lanes.
