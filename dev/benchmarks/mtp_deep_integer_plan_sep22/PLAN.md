Start with a fixed R5 integer planner and depth 4. The current worker already supports explicit singleton depths 1–15, but a full depth 4 or depth 6 proposal uses R5 or R7 and reaches the ordinary ten-stage blocked MoE path. Gathered MPP remains capped at 4, and the qualified compact selector accepts only R4.

The useful change replaces five integer setup dispatches with one parallel planner. Keep the original direct-A pack descriptor and call the unchanged Store gate/up and down methods. That changes ten dispatches to six per layer and removes 192 dispatches and 49,344 metadata threadgroups per 48-layer verifier. These counts establish scope; they do not establish a latency gain.

Build the original pack graph locally to run its existing metadata and alias checks. Authenticate all six setup descriptors before changing the caller graph, then append the new planner and the literal original pack descriptor. Graph construction performs no GPU submission, allocation or operand read. This approach preserves the Store methods, public headers and current R4/C1 guards.

| Actual rows | R5 | R7 |
|---|---:|---:|
| Routes | 50 | 70 |
| Native job capacity | 515 | 516 |
| Minimum M8 job backing | 518 | 520 |
| Pack/prepare groups | 113 | 133 |
| Gate/up groups | 10×50 | 10×70 |
| Down groups | 40×50 | 40×70 |
| Planner threadgroup bytes | 2512 | 2672 |

The native floating kernels retain M16/N64/SG4, 128 threads, the same tensors, coefficients, whole-K multiplication, scales, BF16 casts and SwiGLU. The integer planner must preserve stable expert/canonical-route order, every map/count/offset/job, duplicate and invalid-ID diagnostics, rank behavior and the original active job extent. Larger backing tails remain untouched. Invalid geometry must reset fixed bounded ranges without reading inputs or trusting malformed parameters as write sizes.

R5 and R7 keep the current R4 column of the selective F32 policy because both are below 8 rows. They lose the R4-only raw Q4 row pairing and 97 HC padding savings. The new planner restores the 192-dispatch MoE setup advantage. It initially keeps the original public host validations; there is no need to widen the collapsed R4 preflight bundle.

Depth4 adds one head chain and depth 6 adds three, unless a proposed EOS ends drafting. Retained prefixes of at most 5 or 7 fit one raw-coefficient head fold, whose maximum chunk is 8. Quota/EOS can produce smaller actual windows; preserve their original fallback, including the current fast R4 branch. Concurrent singleton peers and true joint batches remain capped at 3 drafts, so this proposal addresses B1.

Current fixed3 emits 2.428571 retained outputs per cycle at an equivalent 45.627562ms. Holding its first three prefix outcomes identical, let q4–q6 be the unknown later conditional match probabilities. Depth4 wins only if its additional cycle cost is below 5.725812×q4 ms. Depth6 wins only below 5.725812×(q4+q4q5+q4q5q6) ms. Illustrative continuation probabilities of 0.5 give total break-even costs 48.490468ms and 50.637647ms. These are sensitivities, not acceptance estimates: deeper shapes can change earlier logits and actual costs.

Before timing, compare fresh old/new R5 metadata, native FP32/BF16 outputs, diagnostics and guarded tails. A whole candidate then needs larger tape/head ownership admission, retained-prefix restoration and lifecycle proof, the unchanged original22 tasks, and canonical uncached2K/256 measurements. The current task helper requires cap3; any future depth 4/6 adapter must explicitly authenticate the intended cap difference while preserving every unrelated gate and grader. No helper, build or inference change is authorized by this plan. Closed adaptive results remain closed.
