# One validated R4 native chain; CPU proposal and safety proof

Parent is the frozen compact-only worker
`build/compact-native-r4-verify-teacher-sep22-worker-v1b`. Its actual whole-trunk
state proof passed under Root, and its strict frozen22 comparison preserves
all generation texts and20/22 outcomes. Its measured overall decode gain is
zero and prefill3928.9 is below4K; this proposal is not a promotion claim.

The current six-stage chain invokes35 immutable-disjoint scans per expert
layer: Pack13, native gate11 and native down11. Every scan compares the view
with both base and ranks of all48 stored expert layers. It repeats the11-view
scratch pairwise55 comparisons in all three APIs. The original Pack preflight
already covers the union of every gate/down guard: geometry and all11 extents,
all scratch pairs, H/IDs/ranks extents, H versus IDs, H/IDs versus all11 scratch
views, complete inventory and all13 views versus all96 immutable spans.

Use that SAME complete guard union once, with thirteen unique logical bundle
roles (H, IDs, diagnostic and ten scratch views). Do not deduplicate distinct
roles merely because they happen to alias: such aliases must still reject.
Run the existing `allRowsScratch`, `requireBytes`, `disjoint` and
`immutableDisjoint` implementations through the portable preflight routine.
Keep all original public native gate/down APIs and their full validations
literal and unchanged for other callers. Alignment is not a new requirement:
the original native guards admit addressable sufficient Shared views and
disjoint byte intervals; preserve those decisions, including disjoint slices
of the same allocation. Do not read IDs, ranks or activation values on CPU.

After checking the frozen feature flags, layer index and complete512 inventory,
first capture strong copies of all buffer views and the selected source-layer
descriptors. Validate those captured views. Construct a private stack token
binding the Store Impl epoch, selected layer and this exact CommandGraph. Only
the owning chain method can construct or use it. The unchecked append consumes
the token's validated copies, never the caller's mutable scratch after admission.
The token is not returned, cached, shared across calls, retained across mutation
or usable in a second graph. The graph then retains its normal strong copies.
Store source-layer fields are private and immutable after construction; the
normal executor mutex/lifetime and backend allocation-owner validation remain.
No new public token/unchecked entry point or mutable source escape is added.

Append exactly the original planner/pack dispatches and the original public
native producer suffixes after their validation sections, including original
poison and in-place prepare-down bindings. Compare extracted suffix text and
all pipeline/buffer/parameter/grid/thread descriptors in the CPU source witness.
Keep graph counters honest and identical. No role math, shader, workspace,
coefficient cache, sidecar, new GPU allocation or tensor representation changes.
The original metallib is copied/authenticated, with no shader rebuild.

This changes35 immutable scans to13 and188 local byte-range comparisons to78
per layer, preserving every original rejection condition. Immutable range
checks drop3360 to1248 (saving2112); total range comparisons drop3548 to1326
(saving2222). Across48 layers and105 cycles that is11,198,880 fewer host range
comparisons. It does not assume these comparisons explain a measured duration.

Source/metadata timing shows candidate-minus-parent medians: verifier GPU
-109.89ms, backend command wall -105.34ms, total forward host +12.44ms. The
forward-host-minus-backend-wall residual grows120.99ms. Graph construction and
these guards occur before backend submission; backend preparation/encoding
grow only6.21/4.07ms. Ticket blocking wait decreases109.78ms, so increased
callback/external waiting is not established. Concurrent CPU compilers are
also not a proven cause. A matched future Root run determines actual benefit.

Use a separate default-off strict optional flag for the CPU preflight profile.
Flag0 preserves the parent's validated three-wrapper chain; flag1 requires
original compact enabled and all its qualified dependencies. Select only the
same singleton physicalR4 verification branch. Bind the new source identity
to guard policy/status and add only provenance/ownership/graph-construction
coverage to the parent's frozen semantic runner. Every original22 body,
grader, execution policy and teacher/prefill coverage gate remains unchanged.
Rebuild all50 non-core host TUs because the Store public interface changes;
freeze the exact original Core4 and metallib. Flag0 versus1 math is identical.

Before Root timing, independent review must prove the full guard union and
private token lifetime. A CPU decision harness compiles VERBATIM frozen old
guard bodies/guard sections over metadata-only fake buffer adapters, then
compares the actual shipping portable new routine using those same guard
callbacks. This is not a reimplementation of the new decision logic. Test
actual accept/reject parity across fixed/nonfixed geometry, flags, layer/index
inventory, scratch capacities, every role's sufficient/short/null/storage
metadata, each pair's exact/partial/touching/disjoint aliases, all48 immutable
base/rank intersections, invalid addresses and distinct shared-allocation
slices. Preserve all legacy host rejection and GPU diagnostic behavior. Root
alone runs meaningful actual old-versus-bundled alias/parameter/diagnostic and
full-arena/state/future/rollback/semantic/acceptance checks before performance.

Build only after the preflight guard-coverage and private-token source proof is
reviewed. No CPU implementation-mirror tests, GPU/model payload access,
spawning or production edits are authorized for this preparation.
