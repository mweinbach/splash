# Real BatchVerify max16 state qualification

This is a Root-run correctness oracle for the default-off integer-only
R8/R16 compact planner in `compact-native-batch-verify-clock-sep22-worker-v1c`.
It is preparation for qualification, not a throughput or qualification result.

Both oracle roles use one cloned, sealed candidate closure. Control sets the
compact switch to 0 and retains the original native planner; candidate sets it
to 1. Original floating AIRs, worker methods, target numerical flags, model,
2,048 actual input tokens and capacities are identical. New clone friendships
expose readonly state/arena inventory. Diagnostic guarded views retain the
original buffer extents and add exactly admitted redzones. No production files
or floating kernels are edited.

The actual model campaign constructs one target trunk, one persistent
maximum-four-lane/four-row BatchVerify and four independently owned request
states. All four receive the complete canonical 2K prefill. It executes fresh
R8/R16, R16→R8→R16 using the same max16 scratch owners, and active4→3→2→1→4
with real mixed commits and fresh ownership for terminal lanes. Retains 1/2/3/4
preserve healthy prefixes; retain0 uses both live-null and destroyed-null
terminal lanes. Wrapper moves keep the same implementation/identity. A
temporarily moved source object tests the actual source guard without creating
any second model graph or coefficient cache.

Every completed/rejected operation compares all 531 jobs, 160 route maps and
inverses, 223 packed/prepared rows, canonical down, counts, offsets and job
counts, including original inactive ranges and buffer tails. Every non-tape batch scratch byte receives the same finite nonzero
seed before the campaign; original A5 lazy tape initialization remains intact.
The seed exposes extra inactive writes; later R8 commands retain prior R16 data.
The process checks ten scratch-owner redzones, all request redzones, all lazy
tape canaries, owner/identity/extent stability and the exact allocation ledger.
Every result compares full BF16 logits, hidden streams and all GPU greedy
records. Future ordinary greedy steps and 4/8-row appends replace truncated
provisional tails. Actual rows1..3 with active4/2 lanes retain the old route.

The real API has no standalone truncate or deadline callback. Mixed
`commitBatch` is its real logical truncation. `abortBatch` is tested after a
pending wrapper is destroyed and a survivor moves, including idempotence,
poisoned-state rejection and healthy next-cohort recovery. Actual worker
cancel/deadline/reuse checks remain separate Root runtime work. Direct
foreign-trunk rejection remains unchanged-source evidence; this oracle does
not construct a second simultaneous model to claim that runtime proof. It
does test actual wrong-cohort sibling, pending, duplicate, reordered, excessive
retain, expired/nonzero and null/nonzero rejections. Trunk `commitVerify(0)` is
never used; terminal semantics belong to `commitBatch(0)` only.

Large checkpoints are replay partitions. Each process runs the entire
deterministic campaign; selected one or two checkpoints additionally compare
all 134 physical planes of each still-live request and every owned BatchVerify
scratch/tape buffer, including all 216 physical lazy rollback arenas. Clone-only readonly metadata accessors return the actual original native
owner charge and opaque allocation identity. Each distinct owner is charged
once; their actual sum must equal original BatchVerify workspace plus measured
guard delta. Raw or rounded view lengths never substitute native charge. Serialized bytes cover complete
legal MetalBuffer lengths; allocator-only inaccessible page slack is not
represented as readable tensor data. Every frame preflights exact extent and
the aggregate spill is hard limited to 4 GiB before writing. Root runs export,
then comparison in a separate process, confirms backend destruction, and may
delete that completed spill before proceeding. Comparisons cannot use two
simultaneous models or replace byte parity with digest-only parity.

The reservation retains original workspace/cache/Store/head/BF16/blocked
categories, the separate 1,890,975,744-byte W8 coefficient category, four state
categories and 2,195,456 bytes of guard backing per state, max16 batch workspace,
a source-derived maximum 281,436 bytes of scratch guard backing and diagnostic
margin. After construction exposes original native charges, the exact guard
plan is computed and actual guard delta must equal it. The original Private
greedy partial buffer remains Private; a separately admitted 32,768-byte Shared
diagnostic backing exposes exactly its original 31,232-byte readable range.
The existing flash_forward_copy_words kernel initializes Private data and
reads it back at full checkpoints; each async copy ticket is consumed with
wait() before CPU access. Debug copies are counted separately from actual model
verify/commit calls. The mirror is never included in original workspace twice. A wrong-cohort
sibling uses its own governor reservation. No admission or ledger guard is
bypassed. Prefill and proof capacity are 4096; there is no trained MTP head,
worker residency lease, token-rate or model-quality claim.

CPU-only preparation:

```sh
.venv/bin/python -B dev/benchmarks/batch_verify_exact_sep22/build.py \
  --worker build/compact-native-batch-verify-clock-sep22-worker-v1c \
  --build build/batchverify-exact-compact-sep22-v13
.venv/bin/python -B dev/benchmarks/batch_verify_exact_sep22/audit.py \
  --build build/batchverify-exact-compact-sep22-v13 \
  --output build/batchverify-exact-compact-sep22-v13/cpu-source-audit.json
```

Root independently pins each new executable, obtains independent CPU review,
then prepares commands with `commands.py`. Use the existing nativeclock-v5
`root-3-command.txt` as environment metadata. Suggested first selected
checkpoint: `fresh-r16.pending`. CPU_READY lists all 18 checkpoint names. Each
must appear once in the final partition receipt audit before declaring the
whole campaign qualified; one successful partition is not whole-state
completion.
