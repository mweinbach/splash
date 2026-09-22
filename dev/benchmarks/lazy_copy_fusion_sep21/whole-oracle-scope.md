# Sequential whole-target lazy copy oracle

`whole_oracle.mm` passed Clang syntax checking with
`-Wall -Wextra -Werror` against `build/lazy-copy-fusion-sep21-v2/source`.
Root owns GPU execution; this source preparation read no model payload and
executed no GPU commands.

The CLI is:

```
whole-oracle --cpu-only
whole-oracle --gpu FROZEN_METALLIB MODEL_PACKAGE CODE2048_TOKENS_JSON FRESH_REPORT_JSON
```

The CPU branch runs before any package, tokenizer, prompt or provenance reads.
It checks byte comparison, finite-value gates, checkpoint serialization,
metadata and inactive-tail mismatch rejection, truncated-file rejection,
capacity admission formulas, changed future-input selection and bound preflight.
It does not construct a Metal backend.

The GPU qualifier performs a complete control pass with constructor selection0,
then destroys its request/result/operand handles, residency lease, Forward,
weights and backend before constructing selection1. The existing lazy GDN
policy remains1 throughout both passes. It changes the custom environment after
construction to prove selection is frozen on each existing object. All other
kernel-route and coefficient identities must match exactly.

Each B1 keep1/2/3/4 case starts from a fresh canonical2048-token prefix, verifies
the fixed four incoming tokens71093,12305,198,464, commits the retained prefix,
then consumes four deterministic changed inputs derived from control greedy
predictions. The candidate receives those same inputs. Every checkpoint requires
zero differing bytes for all134 physical request planes, including inactive
QSA suffixes and padding, plus length/capacity/pending/poison/owner predicates.
Borrowed logits, pre-mixer hidden rows and entire16-byte greedy records are
spilled before any subsequent trunk call. Floating checkpoint values must be
finite and every real compact greedy record must pass its native validator.

Additional cases cover terminal abort, destruction of a pending trial, healthy
peer continuation, moved-from/default invalid-state guards, same-owner peer
commit rejection, peer blocking during a live tape and physical range
disjointness. Raw owner/identity addresses are checked for stability within a
pass and never compared between different backends.

The control spill is preregistered at32GiB; exact serialized and cumulative
sizes are checked before opening each checkpoint. Candidate comparisons use
at most1MiB of streaming comparison scratch and do not write a second full set.
The oracle preserves atomic checkpoint/progress publication and rejects prior
output artifacts rather than replacing them. Partial progress/failure reports
cannot be mistaken for a complete qualification.

Frozen source/object entries are SHA-verified before and after execution.
The supplied library must match the closure header before any backend is
constructed, and both backends must report that same loaded digest. Executable,
canonical token JSON/U32LE bytes and tokenizer/config files are hashed and
rechecked. Full512 inventory, numerical store identity, original GPU omission
ledger and governor reservation are checked at construction.

B4 joint verification, true foreign-owner requests, broader contexts, trained
MTP, semantic quality and service performance are explicitly unexercised.
Source compilation or the CPU self-test does not establish GPU equality.
