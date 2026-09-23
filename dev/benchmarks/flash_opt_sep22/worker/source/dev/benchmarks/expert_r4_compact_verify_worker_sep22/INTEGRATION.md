# Private singleton R4 verifier integration

Parent: `build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5`.
Candidate: `build/compact-native-r4-verify-teacher-sep22-worker-v1b`.
Flag: `SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22`, missing or exact0 disables,
exact1 enables. Malformed values and any change from the frozen optional
environment state reject. Enabling additionally requires Full512, gathered MPP
with frozen maximum rows exactly4, original blocked MoE, direct-A and Q4X8.
These dependencies validate before paths, metadata, model or backend creation.

Only `FlashForward::forwardImpl` with `verification=true` and physical rows4
selects this route. Its six ordered stages are the GPU-qualified parallel
integer planner, original BF16 pack, original M16/N64/SG4 gate/up, original
excluded-route poison, original down preparation and original M16 down scatter.
Combine reads the original canonical blocked scatter. All original Float
Store methods are unmodified; no new coefficient conversion, quantization,
cache, global scratch or GPU allocation is added. The integer planner is
literally the qualified component-v2 source and uses2432B threadgroup storage.

All prefill, singleton ordinary AR, MTP head/seed, batch forward, batch verify,
teacher, shared expert, QSA/GDN, state and rollback source remains unchanged
except the explicitly reviewed optional selector and its identity/status
plumbing in four host sources. An enabled profile creates a source-bound
execution derivative. Disabled identity marker and derivative seed are exactly
inherited. Status records requested and constructed-store enabled separately,
honest six-versus-two stage counts, graph-construction counters and
`full_model_quality_qualified=false`; none claims GPU completion or model quality.

All50 non-core host translation units rebuild from the sealed source tree.
This includes the six actual consumers of the new public Store bridge header:
singleton Forward, Worker, Store, BatchPrefill, BatchForward and BatchVerify.
The four existing core objects retain their authenticated parent bytes. The
complete original shader AIR closure is unchanged, with only the new qualified
integer planner AIR added. CPU closure witnesses reject any live workspace
header dependency. A final receipt authenticates source, object, library,
policy tests, source witness and independent review.

The six-pattern component receipt confirms native-control exactness and zero
current-gather raw/scaled/BF16/compiled-activation/down mismatches for the tested
normal inputs, plus all pre-timing metadata, malformed, guard and diagnostic
gates. Original native malformed-rank handling is intentionally reported as a
pre-existing difference from gathered handling: native skips/sets bit1 while
gathered may poison/sets bit5. Full512 store admission requires complete valid
rank metadata; a synthetic fault does not establish universal gathered parity.
The rejected software-dot N16 route is absent from this worker and remains rejected.

Private semantic-strict tooling preserves every frozen22 prompt specification,
body, grader, common execution policy and inherited coverage gate. Added status
and ownership checks bind the planner marker to its generated source identity.
For isolated successful main requests, compact graph calls per stage equal48
times the delta of `mtp.completed_cycles_by_proposed_depth[3]`, and rows equal
four times calls. Shortened proposals, standard/constrained requests and
prefill/AR/head/batch calls do not falsely require a compact hit. Existing
persisted encoded-hit projection counters retain their original two-projection
meaning; they do not become six-stage completion counters.

Root alone launches model/GPU qualification. Before promotion, test actual
normalized decode inputs and stage quality; unchanged frozen22 semantic
results; proposal acceptance; future lifetime, truncated prefixes, EOS/budget,
cancel/deadline and rollback recovery; then matched full-output-budget decode
and prefill timings. Synthetic oldmixed gather-to-compact GPU speedup1.0744
and other tested overlap patterns do not predict a full-model improvement.
U40 regressed1.16%; there is no host-read unique-count gate.

For a Root run, keep the exact qualified W5 environment and replace only the
worker/library path with this candidate and set the new flag to1. Flag0 provides
the inherited route in the same freshly built candidate binary. The final
private semantic CLI receipt supplies a sealed-source command with the current
target derivative obtained by Root; this agent did not read or hash any model
or capture payload and did not invoke GPU work.
