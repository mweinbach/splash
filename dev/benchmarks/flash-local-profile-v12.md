# Local Flash-Next profile v12

The accepted M5 Ultra / 256 GiB local profile enables the public exact QSA bulk
prefill route and its SG8 schedule:

- `SPLASH_FLASH_QSA_BULK_PREFILL=1`
- `SPLASH_FLASH_QSA_BULK_PREFILL_SG8=1`

These are the only two additions to accepted V11:42 becomes44 static flags.
Draft depth remains3, sequential teacher cache priming and authenticated dense
tiles remain enabled, both prefill windows remain2,048 rows, and source/hardware
and optional Top64 saved-store qualification pins are unchanged. Larger expert
inventories, all-row Full512, gathered MPP/C2, wide prefill, NAX and split-K are
not selected by V12.

Bulk QSA runs only on successful fresh main-target appends beginning at0 with
exactly2,048 physical rows, sufficient capacity, and no verification. It
preserves the existing BF16 output/F32 intermediate arithmetic and cache
semantics. Later windows, different sizes, verification, trained MTP and true
grouped prefill use their previous routes. Completed-work counters report
main-target prefill calls/tokens, layer calls and SG8 layer calls; grouped
kernel identity may inherit source-route tags, so tags alone do not prove
grouped bulk execution.

The actual public kernel completed four2K/256 coding responses with identical
original golden output hashes and speculative counts. Three warmed trials
measured **2,539.4 prefill tok/s**. The historical control was2,424.5 tok/s,
an observed **+4.74%**, but requested idle maintenance while the public run
explicitly disabled it. This is not a strictly matched maintenance-policy
performance comparison. The earlier private bulk run with matching historical
maintenance policy measured2,530.7 tok/s. These results do not establish4K
prefill tok/s.

All22 frozen semantic texts matched their preceding baseline exactly:
19/22 tasks passed, with the same two arithmetic errors and prohibited `.get`
Python instruction violation, zero new regressions. Completed bulk coverage
was22 calls/45,056 tokens/264 QSA layers/264 SG8 layers. The separate eight
HTTP protocol/lifecycle cases passed. Four public coding responses retained
256-token golden hashes and identical acceptance. Same-public-binary OFF/ON
startup allocation differed by **223.5 MiB**; OFF selectors absent incurred
no optional bulk arena. Memory admission/ledger and fresh healthy idle checks
passed, and both qualification servers were unloaded.

The full proof and policy caveat are in
`build/release/flash/prefill4k-qsa-public-whole-model-qualification-v1.json`.
It records public runtime/source/artifact identities, completed counters,
allocation and cleanup separately from task success and observed timings.

Explicit opt-outs remain authoritative:

```sh
SPLASH_FLASH_QSA_BULK_PREFILL=0 \
./splash serve --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --local-package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --port 8011 --max-context 8192 --no-webui
```

Disabling bulk suppresses its implied SG8 default. Setting SG8=0 alone retains
the public SG4 bulk schedule. Any explicit F32/MPP/ROW_TILES=0 suppresses both
implied bulk selectors. The SG8 merger lists all prerequisites directly
because merging does not recurse through adjusted defaults. Explicit child
values—including contradictions or invalid values—survive for eager native
validation before model/backend construction. Neither selector depends on
batching, MTP or dense cache. Native selectors still default off when the
qualified profile is absent.

Only exact V12/source/architecture/hardware metadata selects these defaults.
Historical V11 remains42/depth3; accepted V10 remains40/depth3/N32; its original
placement snapshot remains39/depth15. V5–V9 retain30/34/36/37/38 flags/depth15 and
their original RAM gates. No historical helper inherits either new selector.
Normal runtime rebuild and default-launch proof will be recorded separately
by the root coordinator.
