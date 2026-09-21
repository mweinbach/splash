# Native prefill tracing

Tracing is an optional diagnostic tool. The normal server enables no profiling
and writes no trace unless `SPLASH_PREFILL_TRACE_FILE` is nonempty. An inherited
GPU-mode variable alone has no effect.

To trace a separate local server from this checkout:

```sh
SPLASH_PREFILL_TRACE_FILE="$PWD/build/prefill-local.jsonl" \
SPLASH_PREFILL_TRACE_GPU_MODE=command \
splash serve --model incoai/Qwen3.8-27B-Splash --port 8011 --no-webui
```

To surround direct-runtime traces with uninstrumented controls:

```sh
make benchmark-prefill-trace MODEL=incoai/Qwen3.8-27B-Splash \
  BUILD=build/profile-local PREFILL_TRACE_ARGS='--prompt-tokens 256 --gpu-profile stage'
.venv/bin/python dev/benchmarks/analyze_prefill_trace.py \
  build/profile-local/prefill-trace.json
```

The analyzer also accepts native JSONL files. Direct-runtime packing and state
copies are useful for kernel attribution; native HTTP traces expose the actual
Scheduler chunk boundaries and Engine cache publication.

Set a fresh local output path for each private run:

```
export SPLASH_PREFILL_TRACE_FILE=/absolute/path/prefill-trace.jsonl
export SPLASH_PREFILL_TRACE_GPU_MODE=command
```

A file alone defaults to `command`. Explicit modes are:

| Mode | Collection |
|---|---|
| `off` | CPU model-phase spans and existing fused batch GPU/wall telemetry |
| `command` | CPU spans, host command timestamps, ordinary GPU command interval and dispatch metadata; no timestamp counter sampling |
| `dispatch` | Exact requested counter sampling at dispatch boundaries, with sampling barriers |
| `stage` | Exact requested counter sampling through separate encoder stages |

Unsupported modes produce an explicit `unsupported` status/profile and retain
normal execution; another profiling mode is never chosen as a fallback. Invalid
mode values disable optional tracing and report `invalid_mode`. Initialization,
file and collection failures report status without turning inference into an
engine failure. Trace JSON never goes to stdout, which carries the binary native
protocol. Only regular local files are accepted.

The sink starts after bootstrap warmup and clears startup profiles. The native
host observer drains completed command profiles after EVERY completed batch,
discarding decode records. This avoids stale decode records overflowing the
backend's eight-profile queue. Only completed prefill records are appended.
Backend profiling remains global while enabled, so explicit profiling also
perturbs decode; use tracing for attribution and separate uninstrumented runs
for performance conclusions.

Each `prefill_completed` JSONL event joins model `command_sequence` to command
`sequence`. It contains graph construction, ticket waiting and Runtime completion
host spans; host submit/encode/commit/scheduled/completed/ready timestamps;
ordinary fused telemetry; capability/status; dispatch names, geometry, resource
byte counts and counter timestamps when valid. It contains no prompts, token IDs,
inline parameter bytes, buffer contents or weight data. Logical row ranges and
aggregate row counts are metadata.
Binding byte counts describe exact bound buffer views, not estimated memory
traffic or the number of bytes a kernel actually reads.

`completion_gap_seconds` measures from the latest Runtime `completion_host` end
to the native observer after Engine cache publication. It includes host work and
possible state-copy waits; it is not an exclusive CPU-compute span. Ticket wait
overlaps command/GPU timing, so these spans must not be added together.

`device_memory_samples` reports `pre_commit`, `post_commit`, `scheduled` and
`completed` property-sampling spans using absolute steady-clock seconds. Each
has begin/end/duration/valid. Pending samples preserve their positive begin but
emit null end/duration; an unavailable sample is not a zero-duration observation.
Counter calibration uses Metal's CPU-nanosecond clock, kept distinct from the
host steady-clock timestamps and raw GPU ticks.

`clock_bridges` preserves precommit and completion bridges between Mach absolute
seconds and the host steady clock, including raw ticks, timebase, offset,
uncertainty and validity. Ordinary raw kernel start/end values have their own
validity flag. Use the bridge before comparing clock axes; do not treat raw
GPU/Mach and steady seconds as interchangeable.

Output appends without truncating existing data, capped at 64MiB per file and
8MiB per event. File limits disable tracing with a status record where space is
available. Use one fresh file per native process. Serializer limits 4096 dispatch
records and 32 binding records per dispatch, retaining true backend counts and
explicit truncation fields. Backend metadata limits are also preserved.

Shared C++ serializer: `runtime/metal/ProfilingJson.hpp`, namespace
`splash::profiling`, overloaded `writeJson(std::ostream&, value)` for capability,
command profile, dispatch timestamp, model phase and device-memory sampling.
It preserves caller formatting, emits decimal/classic JSON despite hex/showbase
or custom locale, preserves 64-bit integer precision, and emits null for nonfinite
numbers. Native sink/entry wiring: `runtime/engine/PrefillTrace.hpp`,
`NativeRuntime.hpp/.cpp`, and `runtime/main.mm`.

CPU validation covers delayed completion, both prefill/decode callbacks,
throwing-observer isolation, environment defaults/exact modes, JSON formatting,
escaping, integer precision, nulls, vector bounds and pending sampling spans.
Native-loop tests use an immediate/held CPU fake model; no GPU work is required.
The serializer test can emit fixtures with `--emit-json` for strict external
JSON parsing. Running-server binaries and `build/splash` are not replaced by
these tests.

The separate `SPLASH_MODEL_WEIGHT_RESIDENCY=1` experiment retains immutable
model-file backings and converted weight data/scales in one Metal residency set.
It excludes activations, diagnostics, sparse KV and state caches. It changes
resource residency without changing weights or precision. Absence leaves it off;
startup stderr reports unique allocation count and existing ledger bytes. The
lease persists until safe backend shutdown and outstanding ticket consumption.
