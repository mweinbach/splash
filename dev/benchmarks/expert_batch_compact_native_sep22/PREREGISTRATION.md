# Fixed R8/R16 compact integer planner component

Root authorized CPU build only. Root alone invokes GPU and reads model payloads.
This distinct private component compiles separately with rows8 or rows16. It
replaces five native integer setup stages with one CTA256 planner, keeping
all five original pack/float/poison/prepare/down dispatches and bindings.
No new graph allocation, coefficient cache, sidecar, activation quantization,
producer arithmetic, input dtype, matrix tile or rank filtering is permitted.

R8 routes80/native capacity516/original M8 backing521/padded143/TG2752 bytes;
R16 routes160/native capacity521/backing531/padded223/TG3392 bytes. Compare all
512 counts,513 offsets and job offsets, every map/inverse, jobCount and each
native job sentinel. Extra5/10 backing records must remain untouched. Duplicate
buckets produce5/10 STEP16 jobs. Stable order is expert then original route.
Original same-row duplicate and invalid I64-ID sticky behavior remains exact.

The original two-level integer SIMD prefix/barrier ordering is retained. The
second scan overwrites groupTotals first, then a uniform CTA barrier finishes
all first-prefix readers before groupPrefixes is reused. Every barrier is
reached uniformly by one fixed full CTA. Malformed params/grid/partial CTA clear
live metadata via one writer before unchanged consumers inspect it.

Pre-timing gates: exact original native metadata, all padded inputs, raw/scaled
F32 and BF16 gate/up/down probes, compiled SwiGLU, raw/prepared activation,
canonical full down and diagnostics. Require original global AND per-route
relativeL2<=1e-4 and cosine>=.999999, with exact zero-reference norm. Report
sampled F64 sign/cancellation/strict/exceptional certificates without masking
strict-sensitive statuses. Check poisoned replay, aliases, invalid/large IDs,
all-invalid/nonfinite hidden rows, >16 duplicate jobs, malformed rank skip,
canaries and source immutability. Preserve the qualified R4 oldmixed rows0..3
exactly (route34/id431), including normalized input construction. Required
cases R8 U10/20/40/80, R16 U10/20/40/80/160, oldmixed/permuted both.

Gather cap4 is unsupported for R8/R16 and is never executed or claimed timed.
Native malformed-rank skip/sticky1 differs from cap4 gathered NaN/sticky5;
execute original-vs-compact native parity only and document that distinction.

All gates precede >=150ms GPU warm time for EACH variant and balanced18 default
old-native10 vs compact6 shipping-only samples; no CPU operand/diagnostic/model
read in either loop. The conditional ~6ms/48-layer setup-saving target derives
from R4 measurements and is not an R8/R16 performance claim. A primitive pass
never qualifies whole-model batch lifetime/rollback/acceptance/semantics.
