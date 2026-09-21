# Request-labelled command diagnostics

Enable only for a local diagnostic run with a fresh absolute path:

```sh
SPLASH_FLASH_REQUEST_COMMAND_TRACE=/tmp/splash-request-command-trace-fresh.jsonl \
  ./splash serve --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --local-package install/local-models/Flash-Next-oQ4e-mtp-v1 --port 8011
```

The sink is opened exclusively with0600/CLOEXEC before loading the model. Existing files, symlinks, relative/empty paths and nonexistent parent directories fail startup. The flag is optional: absent means no file, no backend profiling configuration, no clocks/profiles/serialization on command hooks. Normal routes/defaults remain independent.

When enabled, the Worker selects the existing Command metadata mode after startup constructors. It does not split encoders, insert timestamp barriers or add GPU synchronization. Metadata collection is diagnostic overhead. A profile is drained after every target/head forward, verification and prefix-restore call, before a control drain can destroy a labelled request. Startup conversion commands are not labelled as requests.

Fixed phase names cover target prefill, prompt head priming, committed head folds, draft head chains, target verification, prefix restoration and AR decode. Records contain Worker instance ID, request ID/generation, true per-lane row counts, role, profile status/truncation/drop counts, absolute hardware GPU start/end, host submission/commit/callback boundaries and both Mach/steady clock bridges. They omit request text, tokens, pipeline/profile-reason strings and paths. Labels identify the command's participating requests, not wire-arrival causality.

`backend_submit_entry_to_gpu_start_seconds` measures command submission entry to actual hardware GPU start after a valid clock conversion. Preparation/encoding occur within that interval. Commit-begin/end delays use the same hardware start; negative commit-end delay is retained because GPU work may start before commit() returns. `hardware_gpu_end_to_completed_callback_seconds` distinguishes callback delivery from hardware completion.

Hardware timestamps and cross-clock validity are separate. Conversion requires two valid finite bridges whose offset drift fits their summed uncertainty plus5µs; consistent host/hardware boundaries are also required. Invalid/missing comparisons serialize `null`, with a fixed reason. All floating-point values use max_digits10 serialization. Scheduled callback timestamps are explicitly not hardware scheduling timestamps.

A full accepted prefix can resolve without submitting a restore graph. The trace records `resolved_without_gpu_submit`; it does not invent a GPU timestamp. Normal status exposes enabled/record/missing/unexpected counts through `request_command_trace`, without the sink path.

CPU checks:

```sh
make -f dev/tests/engine/flash_request_command_trace_test.mk test
```

The Foundation-only suite checks independently parsed JSON and clocks at large uptimes, negative commit overlap, unavailable/invalid clocks, cookie/ragged-row identity, metadata privacy, exclusive file creation and descriptor lifetime. It creates no Metal backend/model/device and submits no GPU work. Actual ON/OFF HTTP traces belong to the root coordinator's serialized qualification.
