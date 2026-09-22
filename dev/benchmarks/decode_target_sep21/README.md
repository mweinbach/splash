This CPU-only calculation separates three things: Apple specifications, an
effective warm operand-payload rate from a measured primitive, and generation
targets derived from measured command phases. It never opens model payloads or
executes GPU work. The JSON retains metadata/report/source hashes.

The original packed-Q4 ideal below is distinct from the current privateFull512
I8 numerical derivative. Full512 codes+lateF32 row scales occupy121.1734 GB,
and10 selected experts across48 layers occupy2.3667 GB (78.34% more than Q4's
1.3271 GB). Full512 compact singleton active weights total5.6564 GB. With the
state floor, its standard specification traffic bound is202 tok/s, and the844
GB/s primitive-proxy scenario is142 tok/s.

Accounting additionally for the current4/8/16-row F32 target cache operands,
Full512 MTP prefix3.1875 produces a primitive-proxy traffic scenario of185 /211
/273 aggregate tok/s at batch1 /2 /4. Full acceptance yields232 /264 /342. The
state-inclusive logical cycle footprints are14.54 /25.56 /39.49 GB. These are
optimistic footprint scenarios, not attainable promises: the844 GB/s rate was
measured only as Q8 vocabulary payload/time, and small-row I8 compute,
dispatches, cache rereads and route preparation can prevent reaching it.
Native batch8/16 remain unsupported as enlarged graphs and are modeled as
serial4-lane groups with the batch4 aggregate bound.

For a concrete measured-kernel target, the originalFull512 coding service ran
60.55 decode tok/s,52.643 ms/cycle and3.1875 committed tokens/cycle. Layer0's
repeated4-row synthetic I8 MoE chain took0.5094 ms with old buckets versus
0.2715 ms with direct gathered MPP, preserving every old-MPP stage/output.
If all48 layers reproduced that gain, cycle time would fall11.420 ms and
decode would reach77.32 tok/s. This is a conditional extrapolation requiring
whole-worker evidence, which can include different expert overlap and late
layer cache behavior. It is substantially more grounded than setting a target
at an optimistic memory roof.

The direct gathered chain's measured linear-equivalent rates were0.493 TFLOP/s
at1 row,0.885 at2 rows,1.448 at4 rows, and2.134 at16 rows. These include warm
three-projection expert chains and count multiply+add as two operations;
opaque I8 conversion/MPP work is not counted as a separate FLOP. The16-row
direct variant was slower than old bucket MPP (0.7372 versus0.5289 ms).
These are actual family timings, not a GPU tensor-compute peak. Synthetic
normalized inputs and exact old-MPP parity do not qualify original-model
accuracy; both retain strictF64 failures. The JSON contains every source report
and measured median.

The JSON also models scalar depths1..15 with perfect acceptance. Full512's
existing-cache proxy scenarios are194 /245 /232 /272 /363 tok/s at depth1
/2 /3 /7 /15; specificationdepth15 is515 tok/s. The lowerdepth2 footprint
avoids4-row F32 operand expansion, but raw projections can have a higher
compute cost. Deep folds remain ordered chunks of at most8 true target pairs;
only the final fold produces vocabulary logits. The model optimistically
assumes within-fold trained-head expert sharing. These are unqualified
scenarios: olddepth15/depth3 outputs differed, accepted prefixes can shorten,
and peers cap depth3. Any actual depth change must first establish output/state
parity, acceptance and whole-worker speed.

Run with the repository Python environment:

```sh
.venv/bin/python dev/benchmarks/decode_target_sep21/roofline.py \
  --out dev/benchmarks/decode_target_sep21/roofline-sep21.json
```

`--payload-gbps` changes the scenario throughput; it is not a hardware
measurement. `--temporal-unique` overrides unique experts in four consecutive
verifier rows, from10 (complete sharing) to40 (disjoint). Its default24.515625
comes from192 four-row windows across48 layers of **one retained16-row
speculative diagnostic case**. It is calibration evidence, not a representative
depth3 route benchmark. The average adjacent-row expert overlap in that case
was25.90%.

Apple specifies1.2 TB/s for the80-core M5 Ultra on its
[Mac Studio specifications page](https://www.apple.com/mac-studio/specs/),
checked September21. The retained Q8 vocabulary primitive consumed675,430,400
logical bytes in roughly0.8 ms:844.288 GB/s **effective operand-payload rate**.
Neither this arithmetic nor the existing locality tests measured sustained
physical DRAM bandwidth. Source:
`dev/benchmarks/flash-head-q8-r1-v8.md`.

From the exact model manifest, original target packed operands comprise2.6143
GB of dense/other layer weights,0.6754 GB of vocabulary weights, and1.3271 GB
for10/512 experts in48 layers:4.6168 GB for one token. The trained one-layer
MTP head is0.7708 GB including its vocabulary projection and10 selected
experts. The32 GB PLE table is row gathered; the entire table is not read for
every token. The vision tower and complete token embedding table are also
excluded from text generation traffic.

The weight-only singleton standard bound is260 tok/s at specification, or183
tok/s at the measured primitive's payload proxy. Including a recurrent
read/write and selected2K QSA-cache floor gives245 and173 tok/s respectively.
Both omit scratch, dispatch, host, SSD-read, transaction, and arithmetic costs.

For a batchB, independent lanes with uniform expert frequency have expected
expert unionU(B,s)=512*(1-(1-s/512)^B), where s is10 for ordinary decode or the
calibrated24.515625 for a four-row verifier. Let D=2.6143 GB,V=0.6754 GB,
E=67.9477/512 GB per expert across target layers, and H(B)=0.0677+V+
(1.4156/512)*U(B,10) GB for the trained head. The traffic models are:

- Standard cycle bytes: D+V+E*U(B,10)+B*S_AR.
- Depth3 MTP cycle bytes: D+V+E*U(B,24.515625)+3*H(B)+B*S_MTP.
- Standard aggregate tok/s: B*bandwidth/cycle_bytes.
- MTP aggregate tok/s: B*committed_prefix*bandwidth/cycle_bytes.

The3 head calls cover a committed target-feature fold and two subsequent draft
calls. The state floors are S_AR=0.2768 GB and S_MTP=0.5033 GB per sequence;
MTP includes an initial recurrent-state snapshot for exact lazy rollback.
Numerical changes, removal of target feature folding, altered rollback, and
reduced vocabulary are outside these calculations.

With compact operands and the calibrated expert union, theoretical specification
roofs and primitive-proxy scenarios are below. These are aggregate rates; they
are not demonstrated attainable speeds. MTP numerator3 matches the retained
coding acceptance, while4 assumes all proposals commit. Different content gives
different prefixes: counting averaged3.984 and prose2.442.

| Batch | Standard spec / proxy | MTP prefix3 spec / proxy | MTP prefix4 spec / proxy |
| --- | ---: | ---: | ---: |
|1|245 /173|385 /271|513 /361|
|2|371 /261|552 /388|736 /518|
|4|503 /354|721 /508|962 /677|
|8|622 /438|894 /629|1192 /838|
|16|735 /517|1115 /784|1486 /1046|

The native executors cap4 lanes. Thus batch8/16 above require one enlarged
graph with actual weight reuse; independent4-lane groups have the batch4
aggregate ceiling in this model. They do not multiply aggregate throughput by
the number of groups. Physical caches may retain some operands across separate
groups, but no such guaranteed reuse is assumed here.

Existing small-row target routes expand some original coefficients to F32.
Their unique logical operand footprint increases2.629 GB at4 verifier rows,
7.546 GB at8, and10.044 GB at16. This materially lowers even the optimistic
traffic scenario. Accounting for these existing operand footprints gives
primitive-proxy coding bounds211 /246 /338 tok/s at batch1 /2 /4 and
full-acceptance bounds282 /328 /450. These remain ceilings: shader rereads and
instruction/occupancy/dispatch costs are not included. The JSON includes this
route-specific bound and its source-policy inventory.

Useful nearer targets come from the retained warmed depth3 coding command
trace, which measured32.391 ms of target verification,5.784 ms of head/restore
GPU work, and1.379 ms of command boundaries per cycle. Verification is84.85%
of GPU decode time. Three committed tokens/cycle yields:

| Required verifier improvement | Coding prediction | Full prefix4 prediction |
| --- | ---: | ---: |
|1.5x|104 tok/s|139 tok/s|
|2x|128 tok/s|171 tok/s|
|3x|167 tok/s|223 tok/s|
|4x|197 tok/s|262 tok/s|

These preserve measured non-verifier costs and acceptance. They are conditional
engineering targets, not an achieved speed or proof of attainability. The
untraced service medians were72.70 coding,97.14 counting, and59.08 prose tok/s.
Task acceptance must be measured directly as committed tokens per cycle;
an aggregate per-proposal acceptance ratio alone is insufficient to derive a
prefix distribution. In particular, increasing depth or subtracting every head
call cannot establish an exact or useful generation speedup.

No supported absolute compute ceiling is available from these reports or
Apple's public BF16 specifications. The target's quantized linear projections
represent at least13.21 GFLOP/token when counting multiply+add as two operations
(including vocabulary), before unquantized router, recurrence, attention,
normalization, and coefficient formation. A measured family-specific compute
roof must additionally constrain the traffic bounds.
