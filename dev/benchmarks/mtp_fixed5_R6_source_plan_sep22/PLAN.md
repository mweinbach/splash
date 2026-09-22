Fixed depth 5 is a modest next opportunity, not a route to the hardware ceiling by itself. Current fixed4 produces 255 outputs in 79 cycles at 61.544 tok/s, equivalent to 52.447557ms per cycle. Holding the first four prefix outcomes unchanged, an unknown fifth-position match probability q5 adds 25/79×q5 expected outputs. The extra full-cycle cost must stay below 5.141917×q5 ms. Illustrative q5=0.5 tolerates 2.570959ms; q5=0.75 tolerates 3.856438ms. The unobserved q5=1 limit allows at most 9.8039% gain at unchanged cost.

The extra head chain consumes part of that allowance. Current aggregate Head timing includes committed folds and chains: 7.07674ms GPU and 8.180686ms host per cycle across approximately 3.962 commands. Dividing gives average-command proxies 1.786142ms GPU and 2.064774ms host, not measured marginal chain costs. Current verifier timing is 39.841529ms GPU/42.934147ms host. A whole-stage linear extra-row scenario would add 7.968306ms GPU and fail the threshold, but it is not a forecast: native M16 work, expert union and cache reuse do not necessarily scale linearly.

R6 keeps the same selective F32 column as R5 because both are below 8 rows. It incurs no additional loss of the R4-only row-pair or HC padding optimizations; those are already excluded in R5. Raw QMV, HCDown, routing, target rows and restoration can still add cost. No R6 acceptance or current-input cost is measured.

| R6 geometry | Value |
|---|---:|
| Routes | 60 |
| Native/M8 backing jobs | 515/519 |
| Untouched backing jobs | 4 |
| Pack/prepare groups | 123 |
| Planner threadgroup bytes | 2592 |
| Native GU/down launch Y | 60 |

A future R6 integer kernel can preserve the qualified algorithm, 32-byte parameters, 8-byte jobs and all original M16 floating consumers. Current R5 admission explicitly requires depth4/five-row tape, so setting depth 5 alone fails. A fresh composite would need depth 5/six-row admission while retaining the literal R5 emitter for actual H4 windows. Keep R5 counters at 48×H4 and new R6 counters at 48×H5. Retained folds≤6 remain within the raw fold limit 8; true joint batches remain capped at 3 drafts.

No build, adapter edit or experiment is authorized by this plan. A credible source budget should precede any R6 component preparation. Fresh composite state, task and throughput verification would remain necessary.
