The bounded host uses the frozen original online bulk attention helper and
current parent headers from `rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2`.
It does not use the two-pass attention arena, probability policy or tolerances.

The original prepared prefix runs untimed for two independent inputs/states.
The host verifies both prepared query/index-query/selection sheets and all five
physical cache planes. Synthetic projected rows have true RMS1 before BF16
rounding and OnePlusWeight zero norms. Real mode is Root-only and requires the
pinned old B4 lane0 layer3 projected capture, including all projections, norms
and captured original output. The original prefix derives all five cache planes
from those exact inputs. Real capacity stays16384; synthetic capacity is4096.
Future projections are synthetic in both modes and do not establish a model
trajectory or canonical B1 equivalence.

The performance control is the immutable original shipping AIR. Its early,
temporal and reducer dispatches are3328/128,4608/256 and2048x24/256. Candidate
adds the ushort-only pack64x8x2/256 before the same three logical dispatches.
The pack changes only V addresses to `(kv*256+d)*2048+token`. Queries, keys,
original values, selection, projection and cache state remain immutable during
attention. Both producer plans retain Params64 and physical partition stride4;
rows0..127 consume only partition0, later rows consume4 partitions. Unwritten
first-window partitions are poisoned identically and compared as sentinel bytes.

Exact qualification compares full F32 statistics/numerators and gated BF16
outputs, including inactive sentinel sheets. Untimed native/candidate reducer
taps append F32 quotient and rounded ungated BF16 slots6/7 after original
Params slot5. One live division/cast supplies both taps and the original gate;
no debug re-reduction or alternate arithmetic is used. One shared tap arena is
reused sequentially with host snapshots. Tap gated outputs must equal the last
shipping outputs before/after timing. Active normal fields must be finite.
Malformed-input bit parity is separate from numerical qualification.

Before any fixture allocation, the host enumerates every16KiB-page-rounded
owner plus its leading/trailing canaries. Two independent bulk workspaces,
states and main projected inputs/outputs remain resident. One ordinary128-row
workspace and one original128x32 online-MPP scratch pair are reused sequentially
for untimed prefix and future calls. One F32/BF16 quotient arena is reused;
one2,097,152-byte logical packed-V owner is charged with its canaries. Temporary
future inputs are included at their maximum128-row size. The total must fit a
1GiB normal-governor reservation. Current and actual peak owned/device deltas,
sparse absence, valid host measurement, growth and zero denials are checked.

Host rejects invalid eligibility, parameters, extents, aliases and complete-grid
topology before submission with a specific reason/count check. GPU negative
tests construct valid bounded descriptors first, then inject reserved parameters
or invalid thread/grid descriptors and submit through the normal backend. Pack
negative tests must retain full output sentinel bytes and source immutability.
Producer malformed selection/NaN/Inf diagnostics and full output bytes must
match the original source policy. Recovery restores source bits, recomputes pack,
and repeats complete exact normal qualification before timing.

Future appends1,3,7,128 use the original ordinary helper on each independent
state with shared untimed scratch. Both outputs and all five physical cache
planes must match after each append; earlier keys/values/raw-index/positions and
completed pooled blocks remain unchanged. Perturbing future projections must
not change the prepared2K attention outputs/partials or the earlier cache prefix.

Only original3 versus pack+3 is timed. Each receives at least150ms measured GPU
warm work, then18 pairs with alternating balanced positions. No CPU buffer
reads/writes, resets, guards, snapshots or allocator reads occur between warm
start and final timed submit. Future appends, prefix and all taps stay untimed.
All fixtures/graphs release to zero owned bytes before backend.stop(); backend
destruction precedes success publication.

CLI is `oracle METALLIB FRESH_REPORT --synthetic` or
`oracle METALLIB FRESH_REPORT --actual MANIFEST`. Root must set
`QSA_ONLINE_PACKED_V_ROOT_GPU=1`. `--cpu-self-test` precedes every backend,
capture-metadata and payload path. Compilation waits for source/ABI review GO.
