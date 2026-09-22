# Current teacher bulk per-role profile

This is an observation-only private component over the already qualified
`build/mtp-teacher-bulk-sep21-v4`. Root authorized its CPU build on September22.
There is no elision, shader change, new graph API, Worker change or model read
by this agent. Root alone runs the GPU executable.

## Graph, setup and timing

- Reuse the frozen v4 host objects, Core objects, original metallib and source
  snapshot. A new harness translation unit is the only compiled change. Its
  actual target capture follows v4's real2048-token QSA premixer path, then
  makes the same one-time owned40MiB feature copy. No synthetic input is used.
- Keep the original target, two trained128-row head arenas,18-buffer
  `310181888`-byte bulk arena, original fixture buffers and existing conservative
  governor reservation. Counter allocations are diagnostic and must fit the
  reserved setup allowance; report actual allocation/peak growth and the
  profile mode's instrumentation flags. Never bypass admission.
- Every measured sequence truncates its own healthy state to0, submits the
  **original** bulk1920 graph with fifteen chronological128 prepare/pool
  prefixes, then submits the **original**127 tail at logical1920. It publishes
  exactly2047 pairs in two commands. Truncation stays outside the timed interval,
  consistently with the original component's sequence timing.
- Warm the ordinary sequence until at least150ms accumulated GPU execution.
  Warm the selected profile mode separately. Alternate complete ordinary and
  profiled sequences in a balanced even number of AB/BA pairs. Report every
  sample/position and both aggregate GPU/API durations. Normal durations are
  the only ordinary performance baseline; timestamped durations are diagnostic.
- Use existing `CommandDispatchProfilingMode`: prefer supported
  DispatchBoundary, otherwise explicitly choose supported StagePerDispatch.
  A requested unsupported mode is recorded as unsupported, never silently
  replaced by legacy one-dispatch-per-command replay. No phase-subset replay is
  needed or authorized for this first profile.
- Drain the completed profile immediately after each complete two-command
  sequence, retaining the two records in host memory. The backend's8-record
  retention cannot lose a command. Require zero dropped profiles, complete
  dispatch metadata and valid timestamps before deriving any role duration.
  Buffer reads, copies, comparisons, hashes and file writes do not occur
  between measured sequences. Original token writes and4-byte sticky diagnostic
  reads remain part of the original API call.

## Role contracts

The bulk q projection is one original cached BF16 M32N128 dispatch. Validate
the entire role from its pipeline, grid, threads and bindings: A is
BF16[1920,2560], B is BF16[12288,2560], output is BF16[1920,12288], diagnostic
extent4bytes. Its grid is96 output tiles by60 row tiles, with128 threads.

The raw injection role is the original Q5/G64 scalar projection, validated as
BF16[1920,10240] input, original4-output weights/scales/biases, BF16[1920,4]
output,4-byte diagnostics, grid[1,1920,1],256threads. Use the projection metadata
to validate original packed and parameter extents, rather than assuming that
all parameters have one storage type or row stride.

Validate the q role precedes the15 chronological prepare dispatches and the
injection role precedes the HC mix. Report the tail as a separate command and
do not infer its split cached/scalar q projection from the1920 geometry.

For each of the15 original `flash_qsa_fast_prepare` dispatches, validate
BF16[128,12288] q, BF16[128,512] K/V, BF16[128,640] index bindings and
grid[128,30,1]. The complete prepare time includes24 query heads,2 key heads,
4 index-query heads, V/raw-index copies and position/finite diagnostics.
The sum is an **upper bound** on discarded query-normalization/RoPE cost;
q-only cost is not isolated. A24/30 fraction or query-only subtraction would
change geometry/execution and is prohibited as an unchanged-graph measurement.

Also report the retained15 pool costs, q/injection/mix/prepare subtotal,
unclassified dispatch time, whole command time, sum of timed dispatches and
the residual. Timestamp barriers or encoder splits can serialize previously
overlapping work, so their sum is not a guaranteed elision saving.

## Acceptance and follow-up

The report is a profile, not a quality or optimization qualification. It must
state that current graph math is unchanged, expose unsupported/failure status,
and avoid claiming a present speedup from historical128-window measurements.
If timestamps are unavailable, close this first attempt with ordinary timing
and explicit unavailable role costs; any split-command experiment requires a
separate reviewed plan and labels its overhead/context changes.

Only a substantial current measured q/injection budget warrants continued
finite-domain elision proof. All5cache planes/future output/cancel/recovery
gates in `cacheonly_elision_plan_sep22.md` still apply before any elision is
built. A few-ms teacher budget does not explain the observed target GPU/wall
gap of7–16ms.
