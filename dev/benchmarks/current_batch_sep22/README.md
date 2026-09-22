The baseline commands are in
`build/current-batch-teacher-sep22-v1/root-3-command.txt` and
`root-standard-command.txt`. Root runs them serially. Each command starts and
fully unloads a fresh worker for its MTP mode, measures B4 followed by B2 with
one warmup and three measured trials, and uses identical canonical coding2048
prompts, 256 output tokens per lane, temperature zero, no cached tokens, and
16384 context. The exact qualified parent flags remain in use, including
Full512 target and gathered physical-row cap **4**. Singleton teacher bulk and
QA pause are explicitly zero. No semantic suite runs in either batch worker.
The existing frozen semantic plan and strict audit files remain unchanged.

The qualified parent's actual grouped teacher priming still calls full compact
`FlashBatchMTPForward::forward(None)`. Its singleton bulk optimization executes
only in singleton `tick()`. The cumulative general teacher-cache API counter
also includes lone peer and batch fallback calls, so it cannot be equated to
singleton-bulk completed-command counts after B2/B4 activity. Leaving bulk off
keeps all inactive bulk counters zero and permits an honest baseline without
changing the runtime or weakening the singleton-only quality audit.

Current singleton dense W8A8 and main QSA two-pass guards require singleton
2048-row nonverification input. Batch projections call the original owner
projection method with `singletonMain=false`; true batch main QSA uses its own
executor. The 4K singleton prefill result therefore does not establish 4K batch
prefill. Measure the actual prefill-width and decode/verifier-width deltas.
Supported native lane width remains four; MTP B4 may verify sixteen physical
token rows, which is still four lanes. No native B8 or B16 graph is claimed.

After each root report, metadata-only audit can verify actual emitted-token
counts and requested native graph usage:

```sh
.venv/bin/python -B dev/benchmarks/current_batch_sep22/audit_report.py --report REPORT_JSON --output FRESH_AUDIT_JSON
```

The audit's native command rate uses summed **actual post-first emitted tokens**
and native decode host time. It never uses prepared verifier prediction counts
as its numerator. The client's common first-content to last-Done aggregate is
explicitly labeled as a client metric. Existing `DoneEvent` supplies exact
request-relative native intervals but no shared absolute native clock, so the
baseline audit leaves exact common-span native aggregate unavailable.

Optional private native clocks are source-prepared at
`build/current-batch-native-clock-sep22-v1`. Only Worker changes: an optional
fresh-path metadata sink logs the original `Clock::now()` timestamp at first
token emission and Done, with instance, request, generation, actual emitted
tokens, and finish reason. No kernel, wire protocol, numerical route, state,
or allocation ledger changes are needed. Strict frontend source compilation
passed without backend creation, payload reads, or hashing. Root must compile
and link this Worker in place of the parent Worker, retain all other authenticated
parent objects and the exact metallib, and seal the resulting private binary
before GPU use. The optional flag is a fresh absolute path:
`SPLASH_FLASH_NATIVE_LIFECYCLE_TRACE_SEP22=/absolute/fresh.jsonl`.

For that separate instrumented run, pass `--native-lifecycle-trace TRACE_JSONL`
to `audit_report.py`. The audit requires exactly the wave's native request
count, complete first/Done event pairs, no overlap across idle wave boundaries,
actual emitted counts matching request metrics, and individual intervals
matching authoritative Done micros. Aggregate native decode is then summed
post-first emitted tokens divided by the common earliest native first emission
to latest native Done span. File-writing overhead is part of that instrumented
measurement; both modes need the same sink policy.

B1 semantic qualification remains separate. These performance commands make
no B2/B4 semantic-plan qualification claim. The older B4 MTP result belongs to
its earlier build and is not evidence for this parent.
